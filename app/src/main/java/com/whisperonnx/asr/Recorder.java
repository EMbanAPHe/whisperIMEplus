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
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Continuous-recording VAD recorder.
 *
 * AudioRecord runs for the entire session lifetime — it is opened once in
 * start() and closed once in stop(). It does NOT stop and restart between
 * sentences. This eliminates:
 *
 *   • The inter-sentence gap where AudioRecord is being reinitialised and
 *     the first few frames of the next sentence are silently dropped.
 *
 *   • The mInProgress race where MSG_RECORDING_DONE is sent and the IME
 *     calls start() before the worker thread's finally block clears the flag.
 *
 *   • Repeated AudioRecord allocation / AudioFocus round-trips.
 *
 * Architecture:
 *   One permanent worker thread (recordLoop) blocks until start() is called.
 *   When started, it opens AudioRecord and enters a continuous read loop.
 *   VAD classifies each 30ms frame.  When a speech segment ends, the segment
 *   (+ pre-roll + post-roll) is extracted from internal buffers and signalled
 *   to the IME via MSG_RECORDING_DONE.  AudioRecord keeps running for the
 *   next segment.  MSG_RECORDING signals the IME to show the RECORDING state.
 *
 *   stop() signals the read loop to exit and waits (with timeout) for it to
 *   finish the current frame before closing AudioRecord.
 *
 * Pre-roll buffer:
 *   The last PRE_ROLL_FRAMES frames before speech detection are prepended to
 *   each segment.  This captures the onset of the first phoneme, which VAD
 *   would otherwise miss during its confirmation window.
 */
public class Recorder {

    public interface RecorderListener {
        void onUpdateReceived(String message);
    }

    private static final String TAG = "Recorder";
    public static final String MSG_RECORDING      = "Recording...";
    public static final String MSG_RECORDING_DONE = "Recording done...!";
    public static final String MSG_RECORDING_ERROR = "Recording error...";
    public static final String MSG_VAD_TIMEOUT    = "VAD timeout — no speech detected";

    private static final long STOP_TIMEOUT_MS = 3_000;

    // Audio constants
    private static final int SAMPLE_RATE     = 16_000;
    private static final int BYTES_PER_SAMPLE = 2;
    private static final int VAD_FRAME_SIZE  = 480;          // 30ms at 16kHz
    private static final int VAD_FRAME_BYTES = VAD_FRAME_SIZE * BYTES_PER_SAMPLE;

    // Pre-roll: frames of audio before speech onset to prepend to each segment.
    // 500ms = ~16 frames catches the consonant onset VAD misses during confirmation.
    private static final int PRE_ROLL_FRAMES = 17; // ~500ms

    // Maximum segment length: 30 seconds
    private static final int MAX_SEGMENT_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * 30;

    // Minimum segment length (100ms) — anything shorter is likely a false trigger
    private static final int MIN_SEGMENT_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE / 10;

    // Tail overlap: append this many frames from the END of each segment to the
    // START of the next. Prevents boundary words from being cut off when a phoneme
    // falls exactly at the moment VAD transitions from speech to silence.
    // 300ms = ~10 frames at 16kHz/30ms-per-frame
    private static final int TAIL_OVERLAP_FRAMES = 10;

    private final Context mContext;
    private final AudioManager mAudioManager;
    private AudioFocusRequest mAudioFocusRequest = null;

    private RecorderListener mListener;
    private final Lock lock = new ReentrantLock();
    private final Condition hasTask = lock.newCondition();
    private final Object stopLock = new Object();

    // mInProgress: true while AudioRecord is open and the read loop is running.
    // Set true before opening AudioRecord; cleared when the loop exits.
    private final AtomicBoolean mInProgress = new AtomicBoolean(false);

    // shouldStartRecording: set by start(), cleared when recording begins.
    // Only accessed under lock.
    private boolean shouldStartRecording = false;

    private volatile boolean useVAD = false;
    private volatile int     vadAmplitudeThreshold = 300;
    private volatile long    vadSuppressedUntil    = 0L;
    private VadWebRTC vad = null;

    /**
     * Tail overlap: last TAIL_OVERLAP_FRAMES frames from the most recently flushed
     * segment. Prepended to the next speech segment to avoid boundary word loss.
     */
    private byte[] tailOverlapBuffer = null;

    /**
     * Adaptive noise floor: running average of silence-frame RMS values, used to
     * dynamically raise the amplitude gate when the environment is noisier than
     * the user's static threshold setting. Prevents false triggers in variable
     * noise environments without requiring manual re-adjustment of the slider.
     */
    private float adaptiveNoiseFloor = 0f;
    private static final float NOISE_FLOOR_ALPHA = 0.05f; // EMA coefficient

    private SharedPreferences sp;
    private final Thread workerThread;

    public Recorder(Context context) {
        this.mContext     = context;
        this.mAudioManager = (AudioManager) context.getSystemService(Context.AUDIO_SERVICE);
        sp = PreferenceManager.getDefaultSharedPreferences(mContext);
        workerThread = new Thread(this::recordLoop, "RecorderWorker");
        workerThread.setDaemon(true);
        workerThread.start();
    }

    public void setListener(RecorderListener listener) {
        this.mListener = listener;
    }

    /** Begin a recording session. AudioRecord is opened and runs until stop(). */
    public void start() {
        if (!mInProgress.compareAndSet(false, true)) {
            Log.d(TAG, "Already recording");
            return;
        }
        lock.lock();
        try {
            shouldStartRecording = true;
            hasTask.signal();
        } finally {
            lock.unlock();
        }
    }

    /**
     * Suppress the VAD amplitude gate for durationMs ms.
     * Used by the IME to blank the gate during spacebar haptics.
     */
    public void suppressVadTrigger(int durationMs) {
        vadSuppressedUntil = android.os.SystemClock.uptimeMillis() + durationMs;
    }

    public void initVad() {
        int silenceDurationMs = sp.getInt("silenceDurationMs", 1000);
        int modeIndex         = sp.getInt("vadMode", 2);
        Mode vadMode = modeIndex == 0 ? Mode.NORMAL
                     : modeIndex == 1 ? Mode.AGGRESSIVE
                     : Mode.VERY_AGGRESSIVE;
        vad = Vad.builder()
                .setSampleRate(SampleRate.SAMPLE_RATE_16K)
                .setFrameSize(FrameSize.FRAME_SIZE_480)
                .setMode(vadMode)
                .setSilenceDurationMs(silenceDurationMs)
                .setSpeechDurationMs(50)
                .build();
        vadAmplitudeThreshold = sp.getInt("vadAmplitudeThreshold", 300);
        useVAD = true;
        Log.d(TAG, "VAD initialized: silenceMs=" + silenceDurationMs
                + " mode=" + vadMode + " ampThreshold=" + vadAmplitudeThreshold);
    }

    /**
     * Stop the recording session. Signals the read loop to exit and waits
     * (with timeout) for it to finish. AudioRecord is closed inside the loop.
     */
    public void stop() {
        Log.d(TAG, "stop() called");
        mInProgress.set(false);
        synchronized (stopLock) {
            try {
                stopLock.wait(STOP_TIMEOUT_MS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    /** Permanently shuts down the worker thread. Call from onDestroy(). */
    public void shutdown() {
        workerThread.interrupt();
    }

    public boolean isInProgress() {
        return mInProgress.get();
    }

    private void sendUpdate(String message) {
        if (mListener != null) mListener.onUpdateReceived(message);
    }

    // ── Worker thread ────────────────────────────────────────────────────────

    private void recordLoop() {
        while (true) {
            lock.lock();
            try {
                while (!shouldStartRecording) hasTask.await();
                shouldStartRecording = false;
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            } finally {
                lock.unlock();
            }

            // Re-assert mInProgress=true immediately before starting the session.
            //
            // Race this prevents:
            //   1. Old session runs, mInProgress=true
            //   2. stop() → mInProgress.set(false) → waits on stopLock
            //   3. Old session exits its while loop, cleans up
            //   4. start() → compareAndSet(false,true) → succeeds, shouldStartRecording=true
            //   5. Old session's recordLoop finally → mInProgress.set(false) ← OVERWRITES step 4
            //   6. Worker loops, finds shouldStartRecording=true, calls recordSession()
            //   7. recordSession()'s while(mInProgress.get()) → false → exits immediately
            //   → New recording captures nothing
            //
            // Setting true here (after consuming shouldStartRecording, before calling
            // recordSession) ensures the new session's while loop sees the correct state
            // regardless of what the previous finally did.
            mInProgress.set(true);

            try {
                recordSession();
            } catch (Exception e) {
                Log.e(TAG, "Recording session error", e);
                sendUpdate(MSG_RECORDING_ERROR);
            } finally {
                mInProgress.set(false);
                notifyStopWaiters();
            }
        }
    }

    private void recordSession() {
        if (ActivityCompat.checkSelfPermission(mContext, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            sendUpdate(mContext.getString(R.string.need_record_audio_permission));
            notifyStopWaiters();
            return;
        }

        // ── Audio focus ───────────────────────────────────────────────────────
        requestAudioFocus();

        // ── AudioRecord setup ─────────────────────────────────────────────────
        int bufferSize = Math.max(
                AudioRecord.getMinBufferSize(SAMPLE_RATE,
                        AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT),
                VAD_FRAME_BYTES * 4);

        AudioRecord audioRecord = new AudioRecord.Builder()
                .setAudioSource(MediaRecorder.AudioSource.VOICE_RECOGNITION)
                .setAudioFormat(new AudioFormat.Builder()
                        .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setSampleRate(SAMPLE_RATE)
                        .build())
                .setBufferSizeInBytes(bufferSize)
                .build();

        if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord failed to init");
            audioRecord.release();
            abandonAudioFocus();
            sendUpdate(MSG_RECORDING_ERROR);
            notifyStopWaiters();
            return;
        }

        audioRecord.startRecording();
        Log.d(TAG, "AudioRecord started — continuous session");

        // ── Pre-roll ring buffer ──────────────────────────────────────────────
        // Circular buffer: last PRE_ROLL_FRAMES frames before speech onset.
        byte[][] preRoll = new byte[PRE_ROLL_FRAMES][VAD_FRAME_BYTES];
        int preRollHead  = 0;   // next write slot (oldest slot to overwrite)
        int preRollCount = 0;   // how many valid frames are in the buffer

        // ── Per-segment state ─────────────────────────────────────────────────
        byte[] frame       = new byte[VAD_FRAME_BYTES];
        ByteArrayOutputStream segment  = new ByteArrayOutputStream(SAMPLE_RATE * BYTES_PER_SAMPLE * 3);
        boolean inSpeech   = false;
        boolean hadSpeech  = false;      // any speech this session
        int     silentFrames = 0;        // silence frames after speech (for timeout)
        final int TIMEOUT_FRAMES = (SAMPLE_RATE / VAD_FRAME_SIZE) * 30; // 30s of frames

        while (mInProgress.get()) {
            int bytesRead = audioRecord.read(frame, 0, VAD_FRAME_BYTES);
            if (bytesRead <= 0) {
                Log.w(TAG, "AudioRecord read error: " + bytesRead);
                break;
            }

            if (useVAD) {
                // Amplitude gate — only pre-speech
                boolean vadSuppressed = !inSpeech
                        && (android.os.SystemClock.uptimeMillis() < vadSuppressedUntil);
                // Use whichever is higher: user's static threshold or adaptive noise floor.
                // The adaptive floor rises gradually in noisy environments, preventing
                // false triggers when ambient noise exceeds the user's slider setting.
                int effectiveThreshold = (int) Math.max(vadAmplitudeThreshold, adaptiveNoiseFloor * 1.5f);
                boolean loudEnough = inSpeech
                        || (!vadSuppressed && rmsOf(frame, bytesRead) >= effectiveThreshold);
                boolean isSpeech = loudEnough && vad.isSpeech(frame);

                if (isSpeech) {
                    if (!inSpeech) {
                        // Speech onset — prepend tail overlap from previous segment (if any),
                        // then the pre-roll buffer. This ensures:
                        //   • Boundary words from the previous segment are repeated for context
                        //   • Consonant onset of the current segment is not clipped by VAD delay
                        if (tailOverlapBuffer != null) {
                            segment.write(tailOverlapBuffer, 0, tailOverlapBuffer.length);
                            tailOverlapBuffer = null;
                        }
                        if (preRollCount > 0) {
                            int start = preRollCount < PRE_ROLL_FRAMES
                                    ? 0
                                    : (preRollHead % PRE_ROLL_FRAMES);
                            for (int i = 0; i < preRollCount; i++) {
                                segment.write(preRoll[(start + i) % PRE_ROLL_FRAMES], 0, VAD_FRAME_BYTES);
                            }
                        }
                        inSpeech = true;
                        hadSpeech = true;
                        silentFrames = 0;
                        // Reset adaptive noise floor at speech onset — the energy spike
                        // from recording would skew the floor calculation.
                        adaptiveNoiseFloor = 0f;
                        sendUpdate(MSG_RECORDING);
                        Log.d(TAG, "Speech onset (pre-roll=" + preRollCount + " tail=" + (tailOverlapBuffer != null ? "yes" : "no") + ")");
                    }
                    segment.write(frame, 0, bytesRead);
                    silentFrames = 0;

                    // Hard limit: force-end if segment is getting too long
                    if (segment.size() >= MAX_SEGMENT_BYTES) {
                        Log.d(TAG, "Max segment length reached — flushing");
                        flushSegment(segment, TAIL_OVERLAP_FRAMES);
                        inSpeech = false;
                        silentFrames = 0;
                    }
                } else {
                    if (inSpeech) {
                        // Silence detected after speech — write trailing silence for context,
                        // then flush the completed segment.
                        segment.write(frame, 0, bytesRead);
                        inSpeech = false;
                        flushSegment(segment, TAIL_OVERLAP_FRAMES);
                        silentFrames = 0;
                    } else {
                        silentFrames++;
                        // Update pre-roll ring buffer (also stores silence → speech transition)
                        System.arraycopy(frame, 0, preRoll[preRollHead % PRE_ROLL_FRAMES], 0, VAD_FRAME_BYTES);
                        preRollHead = (preRollHead + 1) % PRE_ROLL_FRAMES;
                        if (preRollCount < PRE_ROLL_FRAMES) preRollCount++;

                        // Adaptive noise floor: update running average of silence energy.
                        // This tracks the current ambient noise level so the effective
                        // threshold rises automatically in noisier environments.
                        int frameRms = rmsOf(frame, bytesRead);
                        adaptiveNoiseFloor = adaptiveNoiseFloor * (1f - NOISE_FLOOR_ALPHA)
                                + frameRms * NOISE_FLOOR_ALPHA;

                        // 30s without ANY speech: fire timeout
                        if (!hadSpeech && silentFrames >= TIMEOUT_FRAMES) {
                            Log.d(TAG, "VAD timeout — no speech in 30s");
                            sendUpdate(MSG_VAD_TIMEOUT);
                            silentFrames = 0;
                        }
                    }
                }
            } else {
                // Non-VAD mode: accumulate everything, no segmentation
                if (!inSpeech) {
                    sendUpdate(MSG_RECORDING);
                    inSpeech = true;
                    hadSpeech = true;
                }
                segment.write(frame, 0, bytesRead);
                if (segment.size() >= MAX_SEGMENT_BYTES) {
                    flushSegment(segment, TAIL_OVERLAP_FRAMES);
                }
            }
        }

        // ── Session ended (stop() was called) ────────────────────────────────
        // Flush any partial segment
        if (inSpeech && segment.size() > MIN_SEGMENT_BYTES) {
            Log.d(TAG, "Flushing partial segment on stop: " + segment.size() + " bytes");
            flushSegment(segment, 0); // no tail overlap needed at session end
        }
        // Clear tail overlap buffer — won't be used after session ends
        tailOverlapBuffer = null;
        adaptiveNoiseFloor = 0f;

        if (useVAD) {
            useVAD = false;
            vad.close();
            vad = null;
        }
        audioRecord.stop();
        audioRecord.release();
        abandonAudioFocus();
        Log.d(TAG, "AudioRecord released — session ended");
    }

    /**
     * Write the accumulated segment to RecordBuffer and signal MSG_RECORDING_DONE.
     * Saves the last tailFrames frames as tail overlap for the next segment.
     * AudioRecord continues running — this is NOT a stop/start cycle.
     *
     * @param segment    the accumulated audio for this speech segment
     * @param tailFrames how many frames from the end to save as tail overlap (0 = none)
     */
    private void flushSegment(ByteArrayOutputStream segment, int tailFrames) {
        byte[] audio = segment.toByteArray();
        segment.reset();

        if (audio.length < MIN_SEGMENT_BYTES) {
            Log.d(TAG, "Segment too short (" + audio.length + " bytes) — discarding");
            sendUpdate(MSG_RECORDING_ERROR);
            return;
        }

        // Save tail overlap for the next segment
        if (tailFrames > 0) {
            int tailBytes = Math.min(tailFrames * VAD_FRAME_BYTES, audio.length);
            tailOverlapBuffer = new byte[tailBytes];
            System.arraycopy(audio, audio.length - tailBytes, tailOverlapBuffer, 0, tailBytes);
        }

        RecordBuffer.setOutputBuffer(audio);
        // MSG_RECORDING_DONE signals the IME to process this segment.
        // AudioRecord continues running for the next sentence.
        sendUpdate(MSG_RECORDING_DONE);
    }

    private void requestAudioFocus() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            AudioFocusRequest req = new AudioFocusRequest.Builder(
                    AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_EXCLUSIVE)
                    .setAudioAttributes(new AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build())
                    .setAcceptsDelayedFocusGain(false)
                    .setOnAudioFocusChangeListener(change -> {})
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

    private void notifyStopWaiters() {
        synchronized (stopLock) {
            stopLock.notifyAll();
        }
    }

    private static int rmsOf(byte[] pcm16, int length) {
        if (length < 2) return 0;
        long sumSq = 0;
        int samples = length / 2;
        for (int i = 0; i < length - 1; i += 2) {
            short s = (short) ((pcm16[i + 1] << 8) | (pcm16[i] & 0xFF));
            sumSq += (long) s * s;
        }
        return (int) Math.sqrt((double) sumSq / samples);
    }
}
