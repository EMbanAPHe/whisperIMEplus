package com.whisperonnx.asr;

import android.Manifest;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.media.AudioAttributes;
import android.media.AudioFocusRequest;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Build;
import android.util.Log;

import androidx.core.app.ActivityCompat;
import androidx.preference.PreferenceManager;

import com.konovalov.vad.webrtc.Vad;
import com.konovalov.vad.webrtc.VadWebRTC;
import com.konovalov.vad.webrtc.config.FrameSize;
import com.konovalov.vad.webrtc.config.Mode;
import com.konovalov.vad.webrtc.config.SampleRate;
import com.whisperonnx.R;

import java.io.ByteArrayOutputStream;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Two-level recording lifecycle:
 *
 *   STREAM level  — controlled by openStream() / closeStream()
 *     AudioRecord is open and reading at 30 ms/frame.  Tied to keyboard
 *     visibility: openStream() from onStartInputView, closeStream() from
 *     onFinishInputView.  When the keyboard is hidden the mic is released.
 *
 *   SESSION level — controlled by start() / stop()
 *     VAD + segmentation runs on top of the already-open stream.
 *     Between sessions the read loop keeps draining the hardware buffer into
 *     the pre-roll ring buffer so it is already populated when the next
 *     session starts — no warmup gap, no clipped onset consonants.
 *
 * Session generation counter:
 *     Each start() call increments mSessionGen.  The read loop tracks the
 *     last gen it committed to.  A change in gen (rapid stop→start) triggers
 *     a clean flush + state reset before the new session begins, preventing
 *     partial-segment data from bleeding across sessions.
 *
 * stop() is fire-and-forget:
 *     It sets mSessionActive=false and returns immediately.  The read loop
 *     handles flushing asynchronously on the next frame (~30 ms).  This
 *     eliminates the wait/notify race that caused 3-second freezes.
 */
public class Recorder {

    public interface RecorderListener {
        void onUpdateReceived(String message);
    }

    private static final String TAG = "Recorder";

    public static final String MSG_RECORDING       = "Recording...";
    public static final String MSG_RECORDING_DONE  = "Recording done...!";
    public static final String MSG_RECORDING_ERROR = "Recording error...";
    public static final String MSG_VAD_TIMEOUT     = "VAD timeout — no speech detected";

    // Audio constants
    private static final int SAMPLE_RATE      = 16_000;
    private static final int BYTES_PER_SAMPLE = 2;
    private static final int VAD_FRAME_SIZE   = 480;        // 30 ms at 16 kHz
    private static final int VAD_FRAME_BYTES  = VAD_FRAME_SIZE * BYTES_PER_SAMPLE;
    private static final int PRE_ROLL_FRAMES  = 17;         // ~500 ms
    private static final int TAIL_OVERLAP_FRAMES = 10;      // ~300 ms
    private static final int MAX_SEGMENT_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * 20; // 20s hard cap
    private static final int MIN_SEGMENT_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE / 10;
    private static final int TIMEOUT_FRAMES   = (SAMPLE_RATE / VAD_FRAME_SIZE) * 30;

    private final Context      mContext;
    private final AudioManager mAudioManager;
    private AudioFocusRequest  mAudioFocusRequest = null;

    private volatile RecorderListener mListener;

    // ── Stream control (AudioRecord open/close) ────────────────────────────────
    private final ReentrantLock  streamLock      = new ReentrantLock();
    private final Condition      streamSignal    = streamLock.newCondition();
    private volatile boolean     streamRequested = false;   // openStream() called
    private volatile boolean     streamShouldClose = false; // closeStream() called
    private volatile AudioRecord mAudioRecord    = null;    // for shutdown() / closeStream()
    private final AtomicBoolean  mStreamOpen     = new AtomicBoolean(false);

    // ── Session control (VAD on/off within running stream) ─────────────────────
    private final AtomicBoolean  mSessionActive  = new AtomicBoolean(false);
    /**
     * Incremented by each start() call.  The read loop compares its local copy
     * to detect a new session request even when stop() and start() arrive faster
     * than one frame, ensuring stale session state is always flushed cleanly.
     */
    private final AtomicInteger  mSessionGen     = new AtomicInteger(0);

    // ── VAD config (written by initVad() on main thread, read by worker) ───────
    private volatile boolean  useVAD               = false;
    private volatile int      vadAmplitudeThreshold = 300;
    private volatile long     vadSuppressedUntil   = 0L;
    /** When false, audio focus is not requested so background audio keeps playing. */
    private volatile boolean  muteAudioOnRecord    = true;
    private VadWebRTC vad = null;  // created by initVad(), only read/closed by worker

    private final SharedPreferences sp;
    private final Thread workerThread;

    public Recorder(Context context) {
        mContext      = context;
        mAudioManager = (AudioManager) context.getSystemService(Context.AUDIO_SERVICE);
        sp            = PreferenceManager.getDefaultSharedPreferences(context);
        workerThread  = new Thread(this::recordLoop, "RecorderWorker");
        workerThread.setDaemon(true);
        workerThread.start();
    }

    /** Call when the preference changes. Thread-safe. */
    public void setMuteAudioOnRecord(boolean mute) {
        muteAudioOnRecord = mute;
    }

    public void setListener(RecorderListener listener) {
        mListener = listener;
    }

    // ── Public API ─────────────────────────────────────────────────────────────

    /**
     * Open the AudioRecord stream. Call from onStartInputView.
     * The stream starts reading into the pre-roll buffer immediately so that
     * the first start() call incurs no warmup delay.
     * No-op if stream is already open.
     */
    public void openStream() {
        if (mStreamOpen.get()) return;
        streamLock.lock();
        try {
            streamRequested   = true;
            streamShouldClose = false;
            streamSignal.signal();
        } finally {
            streamLock.unlock();
        }
    }

    /**
     * Close the AudioRecord stream. Call from onFinishInputView.
     * Stops the current session (if any) and releases the mic.
     * Safe to call even if no stream is open.
     */
    public void closeStream() {
        mSessionActive.set(false);
        streamShouldClose = true;
        // Unblock ar.read() so the loop can exit promptly
        AudioRecord ar = mAudioRecord;
        if (ar != null) {
            try { ar.stop(); } catch (Exception ignored) {}
        }
    }

    /**
     * Begin a recording session.
     * If the stream is open (AudioRecord running), the session begins on the
     * next frame (~30 ms) with the pre-roll buffer already populated.
     * If the stream is not yet open, also opens it (first use).
     */
    public void start() {
        mSessionGen.incrementAndGet();   // new gen — triggers clean state reset in loop
        mSessionActive.set(true);
        if (!mStreamOpen.get()) openStream();
    }

    /**
     * End the current session. Fire-and-forget — returns immediately.
     * The read loop flushes any partial segment on the next frame (~30 ms).
     * AudioRecord keeps running so the next start() has no warmup gap.
     */
    public void stop() {
        mSessionActive.set(false);
    }

    /**
     * Suppress the VAD amplitude gate for durationMs ms.
     * Called by the IME during spacebar haptics.
     */
    public void suppressVadTrigger(int durationMs) {
        vadSuppressedUntil = android.os.SystemClock.uptimeMillis() + durationMs;
    }

    /**
     * Initialise the WebRTC VAD with current preferences.
     * Must be called before start() when VAD mode is desired.
     * Safe to call on the main thread — the VAD object is only used by the worker.
     */
    public void initVad() {
        int silenceMs = sp.getInt("silenceDurationMs", 1000);
        int modeIdx   = sp.getInt("vadMode", 2);
        Mode vadMode  = modeIdx == 0 ? Mode.NORMAL
                      : modeIdx == 1 ? Mode.AGGRESSIVE
                      : Mode.VERY_AGGRESSIVE;
        // Close previous VAD instance if any (clean replacement)
        if (vad != null) { try { vad.close(); } catch (Exception ignored) {} vad = null; }
        vad = Vad.builder()
                .setSampleRate(SampleRate.SAMPLE_RATE_16K)
                .setFrameSize(FrameSize.FRAME_SIZE_480)
                .setMode(vadMode)
                .setSilenceDurationMs(silenceMs)
                .setSpeechDurationMs(50)
                .build();
        vadAmplitudeThreshold = sp.getInt("vadAmplitudeThreshold", 300);
        useVAD = true;
        Log.d(TAG, "VAD init: silenceMs=" + silenceMs + " mode=" + vadMode
                + " ampThreshold=" + vadAmplitudeThreshold);
    }

    /** True if a VAD session is currently active. */
    public boolean isSessionActive() {
        return mSessionActive.get();
    }

    /** True if the AudioRecord stream is open (keyboard is showing). */
    public boolean isInProgress() {
        return mSessionActive.get() || mStreamOpen.get();
    }

    /** Permanently shut down — call from onDestroy(). */
    public void shutdown() {
        mSessionActive.set(false);
        closeStream();             // releases AudioRecord
        workerThread.interrupt();  // wake from streamSignal.await() if still idle
    }

    private void sendUpdate(String message) {
        RecorderListener l = mListener;
        if (l != null) l.onUpdateReceived(message);
    }

    // ── Worker thread ──────────────────────────────────────────────────────────

    private void recordLoop() {
        while (!Thread.currentThread().isInterrupted()) {

            // ── Wait for openStream() ──────────────────────────────────────────
            // Capture shouldOpen inside the lock, release via finally, THEN
            // decide whether to skip — avoids double-unlock (which caused
            // IllegalMonitorStateException on the RecorderWorker thread).
            boolean shouldOpen;
            streamLock.lock();
            try {
                while (!streamRequested && !Thread.currentThread().isInterrupted()) {
                    try { streamSignal.await(); }
                    catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        return;
                    }
                }
                if (Thread.currentThread().isInterrupted()) return;
                streamRequested   = false;
                // If closeStream() was already called while we waited, don't open.
                shouldOpen        = !streamShouldClose;
                streamShouldClose = false;
            } finally {
                // Defensive guard: unlock only if we actually hold the lock.
                // Under certain Android Condition.await() / interrupt races the
                // lock might not be held here, which caused IllegalMonitorStateException
                // crashes in earlier builds. isHeldByCurrentThread() is thread-safe
                // and prevents the crash without masking any real double-unlock.
                if (streamLock.isHeldByCurrentThread()) {
                    streamLock.unlock();
                }
            }
            if (!shouldOpen) continue; // skip AudioRecord open safely — lock already released

            // ── Permission check ───────────────────────────────────────────────
            if (ActivityCompat.checkSelfPermission(mContext, Manifest.permission.RECORD_AUDIO)
                    != PackageManager.PERMISSION_GRANTED) {
                sendUpdate(mContext.getString(R.string.need_record_audio_permission));
                continue;
            }

            // ── Open AudioRecord ───────────────────────────────────────────────
            requestAudioFocus();
            int bufSize = Math.max(
                    AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO,
                            AudioFormat.ENCODING_PCM_16BIT),
                    VAD_FRAME_BYTES * 4);

            AudioRecord ar = new AudioRecord.Builder()
                    .setAudioSource(MediaRecorder.AudioSource.VOICE_RECOGNITION)
                    .setAudioFormat(new AudioFormat.Builder()
                            .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .setSampleRate(SAMPLE_RATE).build())
                    .setBufferSizeInBytes(bufSize)
                    .build();

            if (ar.getState() != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord failed to init");
                ar.release();
                abandonAudioFocus();
                sendUpdate(MSG_RECORDING_ERROR);
                continue;
            }

            ar.startRecording();
            mAudioRecord = ar;
            mStreamOpen.set(true);
            Log.d(TAG, "AudioRecord opened");

            // ── Persistent state (across sessions, reset on closeStream) ───────
            byte[][] preRoll    = new byte[PRE_ROLL_FRAMES][VAD_FRAME_BYTES];
            int      prHead     = 0;    // next write slot (oldest entry)
            int      prCount    = 0;    // valid frames in buffer

            // ── Per-session state ──────────────────────────────────────────────
            ByteArrayOutputStream segment = new ByteArrayOutputStream(96 * 1024);
            boolean inSpeech      = false;
            boolean hadSpeech     = false;
            int     silentFrames  = 0;
            // sustainedSpeechFrames: counts consecutive frames classified as speech.
            // Used for upward noise floor adaptation — if the environment is consistently
            // loud but stationary (e.g. fan, AC, traffic), we nudge adaptFloor upward so
            // the gate eventually rises above the noise. Real speech has high variance;
            // stationary background noise has low variance relative to the floor.
            int     sustainedSpeechFrames = 0;
            // Noise floor estimate — bootstrap from first BOOTSTRAP_FRAMES of audio
            // so the adaptive gate has an immediate estimate rather than starting at 0.
            // Updated during silent frames (slow EMA) and nudged upward during sustained
            // stationary "speech" frames (low variance = probable background noise).
            float   adaptFloor    = 0f;
            // Noise variance estimate — speech has much higher intra-frame variance
            // than stationary background noise. Used as a secondary discriminator.
            float   adaptVariance = 0f;
            int     bootstrapFrames = 0;
            final int BOOTSTRAP_FRAMES = 20; // ~600ms of silence to seed the estimates
            // After this many consecutive speech frames with low variance, raise the floor.
            // 100 frames × 30ms = 3 seconds of sustained stationary noise triggers adaptation.
            final int SUSTAINED_NOISE_FRAMES = 100;
            byte[]  tailOverlap   = null;

            // Generation tracking — detects rapid stop→start sequences
            int loopGen = 0;  // gen the read loop is currently committed to (0 = idle)

            byte[] frame = new byte[VAD_FRAME_BYTES];

            // ── Continuous read loop ───────────────────────────────────────────
            try {
                while (!streamShouldClose && !Thread.currentThread().isInterrupted()) {
                    int n = ar.read(frame, 0, VAD_FRAME_BYTES);
                    if (n <= 0) break; // AudioRecord stopped externally (closeStream)

                    boolean active = mSessionActive.get();
                    int     curGen = mSessionGen.get();

                    // ── Session ended or new session started ───────────────────
                    // Detects both normal stop() and rapid stop→start (gen change).
                    if (loopGen != 0 && (!active || curGen != loopGen)) {
                        // Flush any partial segment before resetting state
                        if (inSpeech && segment.size() > MIN_SEGMENT_BYTES) {
                            tailOverlap = flushSegment(segment, 0); // no tail on manual stop
                        }
                        inSpeech = false; hadSpeech = false;
                        silentFrames = 0; adaptFloor = 0f; segment.reset();
                        if (useVAD) {
                            useVAD = false;
                            if (vad != null) { vad.close(); vad = null; }
                        }
                        loopGen = 0; // back to idle
                    }

                    // ── New session beginning ──────────────────────────────────
                    if (active && loopGen == 0) {
                        loopGen  = curGen;
                        inSpeech = false; hadSpeech = false;
                        silentFrames = 0;
                        // Reset noise estimates for new session so we re-bootstrap
                        // quickly in the new environment rather than carrying over
                        // a stale estimate from a previous session in a different room.
                        adaptFloor = 0f; adaptVariance = 0f; bootstrapFrames = 0;
                        segment.reset();
                        // Pre-roll buffer already populated from idle frames
                        Log.d(TAG, "Session gen=" + loopGen + " pre-roll=" + prCount + " frames");
                    }

                    // ── Update pre-roll (always, when not mid-speech) ──────────
                    // In idle mode: keeps buffer fresh so next session has no warmup gap.
                    // In active mode: maintained for the onset of the NEXT speech event.
                    if (!inSpeech) {
                        System.arraycopy(frame, 0, preRoll[prHead], 0, VAD_FRAME_BYTES);
                        prHead = (prHead + 1) % PRE_ROLL_FRAMES;
                        if (prCount < PRE_ROLL_FRAMES) prCount++;
                    }

                    if (loopGen == 0) continue; // idle — pre-roll only, skip VAD

                    // ── VAD + segmentation ─────────────────────────────────────
                    if (useVAD) {
                        // ── Speech gate: SNR + variance + WebRTC VAD ────────────────
                        // Threshold = max(user floor, adaptive floor × snr_factor).
                        // snr_factor: 1.5 when adaptFloor has converged, ensuring the
                        // gate tracks the environment rather than using an absolute level.
                        // When adaptFloor is still at 0 (no silence yet), fall back to
                        // vadAmplitudeThreshold to avoid passing everything through.
                        float snrFactor = 1.5f;
                        int   threshold;
                        if (adaptFloor > 50f) {
                            // Noise floor converged — use SNR gate; ignore fixed threshold
                            // unless it is explicitly set higher (user override).
                            threshold = (int) Math.max(vadAmplitudeThreshold,
                                    adaptFloor * snrFactor);
                        } else {
                            // No estimate yet — use fixed threshold as sole gate.
                            threshold = vadAmplitudeThreshold;
                        }
                        boolean suppressed = !inSpeech
                                && android.os.SystemClock.uptimeMillis() < vadSuppressedUntil;
                        float frameRms = rmsOf(frame, n);
                        float frameVar = varianceOf(frame, n, frameRms);
                        // Primary gate: amplitude SNR.
                        // Secondary gate: variance SNR — catches soft consonants whose
                        // RMS is below the amplitude threshold but whose spectral structure
                        // is speech-like (high variance relative to noise floor).
                        boolean highVariance = adaptVariance > 10f
                                && frameVar > adaptVariance * 3.5f;
                        boolean loud    = inSpeech
                                || (!suppressed && (frameRms >= threshold || highVariance));
                        boolean isSpeech = loud && vad.isSpeech(frame);

                        if (isSpeech) {
                            if (!inSpeech) {
                                // Prepend tail overlap from previous segment, then pre-roll
                                if (tailOverlap != null) {
                                    segment.write(tailOverlap, 0, tailOverlap.length);
                                    tailOverlap = null;
                                }
                                // Pre-roll: oldest→newest starting at prHead
                                for (int i = 0; i < prCount; i++) {
                                    segment.write(preRoll[(prHead + i) % PRE_ROLL_FRAMES],
                                            0, VAD_FRAME_BYTES);
                                }
                                inSpeech  = true;
                                hadSpeech = true;
                                silentFrames = 0;
                                sustainedSpeechFrames = 0;
                                sendUpdate(MSG_RECORDING);
                            }
                            segment.write(frame, 0, n);
                            silentFrames = 0;
                            sustainedSpeechFrames++;

                            // ── Upward noise floor adaptation for sustained stationary noise ──
                            // If we have been "in speech" for many frames but the frame variance
                            // is low (noise is stationary, not speech-like), the environment has
                            // probably changed to a louder background.  Nudge adaptFloor upward
                            // slowly so the SNR gate eventually rises above the noise.
                            // Guard: only adapt if variance suggests non-speech (< 2× floor variance).
                            // This prevents real long speech from being misclassified as noise.
                            if (sustainedSpeechFrames > SUSTAINED_NOISE_FRAMES
                                    && adaptVariance > 5f
                                    && frameVar < adaptVariance * 2.0f) {
                                // Raise floor gently toward current RMS × 0.85.
                                // ×0.85 leaves a 15% margin so real speech onset still triggers.
                                float targetFloor = frameRms * 0.85f;
                                if (targetFloor > adaptFloor) {
                                    adaptFloor = adaptFloor * 0.98f + targetFloor * 0.02f;
                                }
                            }
                            if (segment.size() >= MAX_SEGMENT_BYTES) {
                                tailOverlap = flushSegment(segment, TAIL_OVERLAP_FRAMES);
                                inSpeech    = false;
                                sustainedSpeechFrames = 0;
                            }
                        } else {
                            if (inSpeech) {
                                segment.write(frame, 0, n); // trailing silence frame
                                inSpeech    = false;
                                sustainedSpeechFrames = 0;
                                tailOverlap = flushSegment(segment, TAIL_OVERLAP_FRAMES);
                                silentFrames = 0;
                            } else {
                                silentFrames++;
                                frameRms = rmsOf(frame, n);
                                frameVar = varianceOf(frame, n, frameRms);
                                if (bootstrapFrames < BOOTSTRAP_FRAMES) {
                                    // Fast bootstrap: use first N silent frames to get
                                    // an immediate noise estimate before EMA converges.
                                    bootstrapFrames++;
                                    float alpha = 1f / bootstrapFrames;
                                    adaptFloor    = adaptFloor    * (1f - alpha) + frameRms * alpha;
                                    adaptVariance = adaptVariance * (1f - alpha) + frameVar * alpha;
                                } else {
                                    // Slow EMA: track gradual environment changes.
                                    adaptFloor    = adaptFloor    * 0.97f + frameRms * 0.03f;
                                    adaptVariance = adaptVariance * 0.97f + frameVar * 0.03f;
                                }
                                if (!hadSpeech && silentFrames >= TIMEOUT_FRAMES) {
                                    sendUpdate(MSG_VAD_TIMEOUT);
                                    silentFrames = 0;
                                }
                            }
                        }
                    } else {
                        // Non-VAD: accumulate everything
                        if (!inSpeech) { sendUpdate(MSG_RECORDING); inSpeech = true; hadSpeech = true; }
                        segment.write(frame, 0, n);
                        if (segment.size() >= MAX_SEGMENT_BYTES)
                            tailOverlap = flushSegment(segment, TAIL_OVERLAP_FRAMES);
                    }
                }
            } catch (Exception e) {
                Log.e(TAG, "Read loop error", e);
                sendUpdate(MSG_RECORDING_ERROR);
            } finally {
                mStreamOpen.set(false);
                mSessionActive.set(false);
                try { ar.stop(); } catch (Exception ignored) {}
                ar.release();
                mAudioRecord = null;
                if (vad != null) { try { vad.close(); } catch (Exception ignored) {} vad = null; }
                useVAD = false;
                abandonAudioFocus();
                Log.d(TAG, "AudioRecord released");
            }
        }
    }

    /**
     * Flush the accumulated segment to RecordBuffer and send MSG_RECORDING_DONE.
     * Returns the last tailFrames frames as the tail overlap for the next segment.
     * AudioRecord keeps running — this is NOT a stop event.
     */
    private byte[] flushSegment(ByteArrayOutputStream segment, int tailFrames) {
        byte[] audio = segment.toByteArray();
        segment.reset();

        if (audio.length < MIN_SEGMENT_BYTES) {
            Log.d(TAG, "Segment too short (" + audio.length + " bytes) — discarding");
            sendUpdate(MSG_RECORDING_ERROR);
            return null;
        }

        byte[] tail = null;
        if (tailFrames > 0) {
            int tailBytes = Math.min(tailFrames * VAD_FRAME_BYTES, audio.length);
            tail = new byte[tailBytes];
            System.arraycopy(audio, audio.length - tailBytes, tail, 0, tailBytes);
        }

        RecordBuffer.setOutputBuffer(audio);
        sendUpdate(MSG_RECORDING_DONE);
        return tail;
    }

    private void requestAudioFocus() {
        if (!muteAudioOnRecord) return; // user prefers background audio to continue
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            AudioFocusRequest req = new AudioFocusRequest
                    .Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_EXCLUSIVE)
                    .setAudioAttributes(new AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build())
                    .setAcceptsDelayedFocusGain(false)
                    .setOnAudioFocusChangeListener(c -> {})
                    .build();
            mAudioManager.requestAudioFocus(req);
            mAudioFocusRequest = req;
        } else {
            //noinspection deprecation
            mAudioManager.requestAudioFocus(null, AudioManager.STREAM_VOICE_CALL,
                    AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_EXCLUSIVE);
        }
    }

    private void abandonAudioFocus() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            if (mAudioFocusRequest != null) {
                mAudioManager.abandonAudioFocusRequest(mAudioFocusRequest);
                mAudioFocusRequest = null;
            }
        } else {
            //noinspection deprecation
            mAudioManager.abandonAudioFocus(null);
        }
    }

    private static int rmsOf(byte[] pcm, int len) {
        if (len < 2) return 0;
        long sum = 0;
        int s = len / 2;
        for (int i = 0; i < len - 1; i += 2) {
            short v = (short) ((pcm[i + 1] << 8) | (pcm[i] & 0xFF));
            sum += (long) v * v;
        }
        return (int) Math.sqrt((double) sum / s);
    }

    /**
     * Compute the variance of sample amplitudes within a PCM frame.
     * Speech frames have high variance (rich harmonic content); stationary
     * background noise has low variance (spectrally uniform). Passing the
     * already-computed RMS avoids redundant work.
     */
    private static float varianceOf(byte[] pcm, int len, float mean) {
        if (len < 2) return 0f;
        double sumSq = 0;
        int s = 0;
        for (int i = 0; i < len - 1; i += 2) {
            short v = (short) ((pcm[i + 1] << 8) | (pcm[i] & 0xFF));
            float diff = Math.abs(v) - mean;
            sumSq += diff * diff;
            s++;
        }
        return s > 0 ? (float)(sumSq / s) : 0f;
    }
}
