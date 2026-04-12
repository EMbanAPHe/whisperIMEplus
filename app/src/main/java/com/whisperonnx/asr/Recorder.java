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
 * AudioRecord opens ONCE and runs for the entire IME service lifetime — it never
 * closes between recording sessions. This eliminates the 100–300ms gap where
 * AudioRecord is being re-initialised between sessions, during which any speech
 * is lost before a single frame is read.
 *
 * Architecture:
 *   The worker thread has two phases:
 *
 *   Phase 1 — Initial wait (IDLE, no AudioRecord):
 *     Blocks in hasTask.await() waiting for the first start() call.
 *     No audio hardware is active yet.
 *
 *   Phase 2 — Continuous read (STREAM, AudioRecord always open):
 *     AudioRecord opens once and reads frames at the hardware cadence (~30ms/frame).
 *     The read loop runs two sub-modes controlled by mSessionActive:
 *
 *       IDLE sub-mode  (mSessionActive=false):
 *         Reads frames and updates the pre-roll ring buffer only.
 *         Drains the hardware buffer so audio does not back up.
 *         When start() is called, mSessionActive→true and the session begins
 *         with 500ms of pre-roll already populated.
 *
 *       ACTIVE sub-mode (mSessionActive=true):
 *         Full VAD + amplitude gate + segment extraction.
 *         Segments are flushed via MSG_RECORDING_DONE; AudioRecord keeps running.
 *
 *   stop() transition (ACTIVE → IDLE):
 *     Sets mSessionActive=false. The read loop detects the transition on the next
 *     frame (~30ms), flushes any partial segment, and calls notifyStopWaiters()
 *     so stop() can return. AudioRecord keeps running.
 *
 *   start() transition (IDLE → ACTIVE):
 *     Sets mSessionActive=true. The read loop detects the transition on the next
 *     frame. Pre-roll buffer is already populated — no warmup gap.
 *
 *   shutdown():
 *     Sets shutdownRequested=true and calls audioRecord.stop() to unblock
 *     the pending read(). The read loop exits cleanly.
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

    private static final long STOP_TIMEOUT_MS  = 3_000;

    // Audio constants
    private static final int SAMPLE_RATE      = 16_000;
    private static final int BYTES_PER_SAMPLE = 2;
    private static final int VAD_FRAME_SIZE   = 480;           // 30ms at 16kHz
    private static final int VAD_FRAME_BYTES  = VAD_FRAME_SIZE * BYTES_PER_SAMPLE;

    // Pre-roll: frames prepended at speech onset to capture consonant attack.
    // 500ms = ~17 frames — persists across sessions so it's always populated.
    private static final int PRE_ROLL_FRAMES  = 17;

    // Tail overlap: frames from end of each segment prepended to the next.
    private static final int TAIL_OVERLAP_FRAMES = 10;         // ~300ms

    // Segment limits
    private static final int MAX_SEGMENT_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * 30;
    private static final int MIN_SEGMENT_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE / 10;

    private final Context      mContext;
    private final AudioManager mAudioManager;
    private AudioFocusRequest  mAudioFocusRequest = null;

    private RecorderListener mListener;
    private final Object     stopLock = new Object();

    // ── Session control ───────────────────────────────────────────────────────
    /**
     * True while the worker thread's VAD session is active.
     * Set true by start(), false by stop().
     * The read loop checks this each frame (~30ms cadence) to switch modes.
     *
     * Separate from mStreamOpen so that "stream is running" can be true while
     * "session is active" is false (between-session idle phase).
     */
    private final AtomicBoolean mSessionActive   = new AtomicBoolean(false);

    /**
     * True once AudioRecord is open and the continuous read loop is running.
     * Used for isInProgress() so the IME can tell the stream is alive.
     */
    private final AtomicBoolean mStreamOpen      = new AtomicBoolean(false);

    // ── Initial start gate ────────────────────────────────────────────────────
    // Used ONCE to wake the thread from its initial await before AudioRecord opens.
    private final Lock      startLock         = new ReentrantLock();
    private final Condition startCondition    = startLock.newCondition();
    private volatile boolean firstStartSignalled = false;

    // ── VAD state (set by initVad(), read by worker thread) ──────────────────
    private volatile boolean useVAD               = false;
    private volatile int     vadAmplitudeThreshold = 300;
    private volatile long    vadSuppressedUntil   = 0L;
    private VadWebRTC vad = null;

    private SharedPreferences sp;

    // ── AudioRecord reference (for shutdown()) ────────────────────────────────
    private volatile AudioRecord mAudioRecord = null;
    private volatile boolean     shutdownRequested = false;

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

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Begin a VAD recording session.
     * If AudioRecord is already running (between-session idle), the session begins
     * on the next frame (~30ms) with the pre-roll buffer already populated.
     * If this is the very first call, opens AudioRecord first (once per service life).
     */
    public void start() {
        mSessionActive.set(true);
        // Signal the initial await on the very first call
        if (!mStreamOpen.get() && !firstStartSignalled) {
            startLock.lock();
            try {
                firstStartSignalled = true;
                startCondition.signal();
            } finally {
                startLock.unlock();
            }
        }
    }

    /**
     * Suppress the VAD amplitude gate for durationMs ms.
     * Called by the IME during spacebar haptics to prevent vibration triggering.
     */
    public void suppressVadTrigger(int durationMs) {
        vadSuppressedUntil = android.os.SystemClock.uptimeMillis() + durationMs;
    }

    /**
     * Initialise the WebRTC VAD with current preferences.
     * Must be called before start() when VAD mode is desired.
     */
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
     * End the current recording session. Sets mSessionActive=false and waits
     * (up to STOP_TIMEOUT_MS) for the read loop to flush any partial segment.
     * AudioRecord keeps running — the next start() incurs no warmup delay.
     */
    public void stop() {
        if (!mSessionActive.get()) return; // already idle
        mSessionActive.set(false);
        synchronized (stopLock) {
            try {
                stopLock.wait(STOP_TIMEOUT_MS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    /**
     * Permanently shuts down the recorder.
     * Stops and releases AudioRecord, interrupts the worker thread.
     * Call from the IME's onDestroy(). The Recorder cannot be used after this.
     */
    public void shutdown() {
        shutdownRequested = true;
        mSessionActive.set(false);
        // Unblock the initial await (in case start() was never called)
        startLock.lock();
        try {
            firstStartSignalled = true;
            startCondition.signalAll();
        } finally {
            startLock.unlock();
        }
        // Unblock audioRecord.read() so the thread can exit
        AudioRecord ar = mAudioRecord;
        if (ar != null) {
            try { ar.stop(); } catch (Exception ignored) {}
        }
        workerThread.interrupt();
    }

    /**
     * True while the VAD session is active (speech is being captured/segmented).
     * Also returns true while AudioRecord is open (stream is running) so the IME
     * can use this to check whether the recorder needs to be stopped on exit.
     */
    public boolean isInProgress() {
        return mSessionActive.get() || mStreamOpen.get();
    }

    /**
     * True only while a VAD session is actively running (between start() and stop()).
     * Unlike isInProgress(), returns false when AudioRecord is open but idle
     * (between sessions). Use this in the IME's defer-start logic so that
     * "stream open but no active session" does not trigger an unnecessary defer.
     */
    public boolean isSessionActive() {
        return mSessionActive.get();
    }

    private void sendUpdate(String message) {
        if (mListener != null) mListener.onUpdateReceived(message);
    }

    // ── Worker thread ─────────────────────────────────────────────────────────

    private void recordLoop() {
        // ── Phase 1: initial await ────────────────────────────────────────────
        // Block until the first start() call. No audio hardware active yet.
        startLock.lock();
        try {
            while (!firstStartSignalled) {
                try { startCondition.await(); }
                catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        } finally {
            startLock.unlock();
        }

        if (shutdownRequested || Thread.currentThread().isInterrupted()) return;

        // Check microphone permission
        if (ActivityCompat.checkSelfPermission(mContext, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            sendUpdate(mContext.getString(R.string.need_record_audio_permission));
            return;
        }

        // ── Phase 2: open AudioRecord (once for service lifetime) ─────────────
        requestAudioFocus();

        int bufferSize = Math.max(
                AudioRecord.getMinBufferSize(SAMPLE_RATE,
                        AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT),
                VAD_FRAME_BYTES * 4);

        AudioRecord ar = new AudioRecord.Builder()
                .setAudioSource(MediaRecorder.AudioSource.VOICE_RECOGNITION)
                .setAudioFormat(new AudioFormat.Builder()
                        .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setSampleRate(SAMPLE_RATE)
                        .build())
                .setBufferSizeInBytes(bufferSize)
                .build();

        if (ar.getState() != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord failed to init");
            ar.release();
            abandonAudioFocus();
            sendUpdate(MSG_RECORDING_ERROR);
            return;
        }

        ar.startRecording();
        mAudioRecord = ar;
        mStreamOpen.set(true);
        Log.d(TAG, "AudioRecord opened — continuous stream running");

        // ── State that PERSISTS across sessions ───────────────────────────────
        // Pre-roll ring buffer: populated in idle mode so it's ready the instant
        // a new session starts — no warmup gap, no clipped consonants.
        byte[][] preRoll = new byte[PRE_ROLL_FRAMES][VAD_FRAME_BYTES];
        int preRollHead  = 0;
        int preRollCount = 0;
        byte[] tailOverlapBuffer = null;

        // ── Per-session state (reset on each IDLE→ACTIVE transition) ─────────
        ByteArrayOutputStream segment = new ByteArrayOutputStream(96 * 1024);
        boolean inSpeech     = false;
        boolean hadSpeech    = false;
        int     silentFrames = 0;
        float   adaptiveFloor = 0f;
        final int TIMEOUT_FRAMES = (SAMPLE_RATE / VAD_FRAME_SIZE) * 30; // 30s

        // Transition tracking — detect edge changes to mSessionActive
        boolean wasActive = mSessionActive.get(); // true if start() already called

        byte[] frame = new byte[VAD_FRAME_BYTES];

        // ── Phase 3: continuous read loop ─────────────────────────────────────
        try {
            while (!shutdownRequested && !Thread.currentThread().isInterrupted()) {
                int bytesRead = ar.read(frame, 0, VAD_FRAME_BYTES);
                if (bytesRead <= 0) {
                    Log.w(TAG, "AudioRecord read returned " + bytesRead + " — exiting");
                    break;
                }

                boolean nowActive = mSessionActive.get();

                // ── Transition: ACTIVE → IDLE ─────────────────────────────────
                // stop() was called. Flush any partial segment and notify stop().
                if (wasActive && !nowActive) {
                    wasActive = false;
                    if (inSpeech && segment.size() > MIN_SEGMENT_BYTES) {
                        Log.d(TAG, "Session ended mid-speech — flushing partial segment");
                        flushSegment(segment, 0);
                    }
                    // Reset per-session state
                    inSpeech = false; hadSpeech = false;
                    silentFrames = 0; adaptiveFloor = 0f;
                    segment.reset();
                    if (useVAD) {
                        useVAD = false;
                        if (vad != null) { vad.close(); vad = null; }
                    }
                    // Wake stop() — audio has been committed
                    synchronized (stopLock) { stopLock.notifyAll(); }
                }

                // ── Transition: IDLE → ACTIVE ─────────────────────────────────
                // start() was called. Pre-roll buffer is already populated.
                if (!wasActive && nowActive) {
                    wasActive = true;
                    inSpeech = false; hadSpeech = false;
                    silentFrames = 0; adaptiveFloor = 0f;
                    segment.reset();
                    Log.d(TAG, "Session started — pre-roll has " + preRollCount + " frames ready");
                }

                // ── Always update pre-roll (both modes) ───────────────────────
                // In IDLE mode: keeps the buffer fresh so the next session has
                // no warmup gap. In ACTIVE mode: pre-roll is used at speech onset.
                if (!inSpeech) {
                    // Only update pre-roll when not mid-speech (onset detection only)
                    System.arraycopy(frame, 0, preRoll[preRollHead % PRE_ROLL_FRAMES], 0, VAD_FRAME_BYTES);
                    preRollHead = (preRollHead + 1) % PRE_ROLL_FRAMES;
                    if (preRollCount < PRE_ROLL_FRAMES) preRollCount++;
                }

                // ── Skip VAD processing when idle ─────────────────────────────
                if (!nowActive) continue;

                // ── VAD + segmentation (ACTIVE mode only) ─────────────────────
                if (useVAD) {
                    // Amplitude gate — only pre-speech
                    int effectiveThreshold = (int) Math.max(vadAmplitudeThreshold, adaptiveFloor * 1.5f);
                    boolean vadSuppressed  = !inSpeech
                            && (android.os.SystemClock.uptimeMillis() < vadSuppressedUntil);
                    boolean loudEnough = inSpeech
                            || (!vadSuppressed && rmsOf(frame, bytesRead) >= effectiveThreshold);
                    boolean isSpeech   = loudEnough && vad.isSpeech(frame);

                    if (isSpeech) {
                        if (!inSpeech) {
                            // Speech onset — prepend tail overlap then pre-roll
                            if (tailOverlapBuffer != null) {
                                segment.write(tailOverlapBuffer, 0, tailOverlapBuffer.length);
                                tailOverlapBuffer = null;
                            }
                            if (preRollCount > 0) {
                                int start = preRollCount < PRE_ROLL_FRAMES
                                        ? 0 : (preRollHead % PRE_ROLL_FRAMES);
                                for (int i = 0; i < preRollCount; i++) {
                                    segment.write(preRoll[(start + i) % PRE_ROLL_FRAMES],
                                            0, VAD_FRAME_BYTES);
                                }
                            }
                            inSpeech  = true;
                            hadSpeech = true;
                            silentFrames = 0;
                            adaptiveFloor = 0f;
                            sendUpdate(MSG_RECORDING);
                        }
                        segment.write(frame, 0, bytesRead);
                        silentFrames = 0;

                        if (segment.size() >= MAX_SEGMENT_BYTES) {
                            tailOverlapBuffer = flushSegment(segment, TAIL_OVERLAP_FRAMES);
                            inSpeech = false;
                            silentFrames = 0;
                        }
                    } else {
                        if (inSpeech) {
                            segment.write(frame, 0, bytesRead);
                            inSpeech = false;
                            tailOverlapBuffer = flushSegment(segment, TAIL_OVERLAP_FRAMES);
                            silentFrames = 0;
                        } else {
                            silentFrames++;
                            // Adaptive noise floor
                            int frameRms = rmsOf(frame, bytesRead);
                            adaptiveFloor = adaptiveFloor * (1f - 0.05f) + frameRms * 0.05f;

                            if (!hadSpeech && silentFrames >= TIMEOUT_FRAMES) {
                                Log.d(TAG, "VAD timeout — no speech in 30s");
                                sendUpdate(MSG_VAD_TIMEOUT);
                                silentFrames = 0;
                            }
                        }
                    }
                } else {
                    // Non-VAD mode: accumulate everything
                    if (!inSpeech) {
                        sendUpdate(MSG_RECORDING);
                        inSpeech  = true;
                        hadSpeech = true;
                    }
                    segment.write(frame, 0, bytesRead);
                    if (segment.size() >= MAX_SEGMENT_BYTES) {
                        tailOverlapBuffer = flushSegment(segment, TAIL_OVERLAP_FRAMES);
                    }
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "Error in continuous read loop", e);
            sendUpdate(MSG_RECORDING_ERROR);
        } finally {
            mStreamOpen.set(false);
            mSessionActive.set(false);
            try { ar.stop(); } catch (Exception ignored) {}
            ar.release();
            mAudioRecord = null;
            if (vad != null) { vad.close(); vad = null; }
            abandonAudioFocus();
            synchronized (stopLock) { stopLock.notifyAll(); }
            Log.d(TAG, "AudioRecord released — stream closed");
        }
    }

    /**
     * Flush the accumulated segment to RecordBuffer and signal MSG_RECORDING_DONE.
     * Returns the tail overlap bytes (last tailFrames frames) for the next segment,
     * or null if tailFrames is 0.
     * AudioRecord keeps running — this is not a stop event.
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
