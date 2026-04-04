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
                boolean loudEnough = inSpeech
                        || (!vadSuppressed && rmsOf(frame, bytesRead) >= vadAmplitudeThreshold);
                boolean isSpeech = loudEnough && vad.isSpeech(frame);

                if (isSpeech) {
                    if (!inSpeech) {
                        // Speech onset — prepend pre-roll to capture consonant onset
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
                        sendUpdate(MSG_RECORDING);
                        Log.d(TAG, "Speech onset (with " + preRollCount + " pre-roll frames)");
                    }
                    segment.write(frame, 0, bytesRead);
                    silentFrames = 0;

                    // Hard limit: force-end if segment is getting too long
                    if (segment.size() >= MAX_SEGMENT_BYTES) {
                        Log.d(TAG, "Max segment length reached — flushing");
                        flushSegment(segment);
                        inSpeech = false;
                        silentFrames = 0;
                    }
                } else {
                    if (inSpeech) {
                        // Still in the silence extension period — write trailing silence
                        segment.write(frame, 0, bytesRead);
                        silentFrames++;
                        // VAD's silenceDurationMs controls when it transitions out of SPEECH.
                        // Once isSpeech returns false, that already accounts for silence duration.
                        // Flush immediately.
                        inSpeech = false;
                        flushSegment(segment);
                        silentFrames = 0;
                    } else {
                        silentFrames++;
                        // Update pre-roll ring buffer with silence frames too —
                        // pre-roll includes the transition period into speech
                        System.arraycopy(frame, 0, preRoll[preRollHead % PRE_ROLL_FRAMES], 0, VAD_FRAME_BYTES);
                        preRollHead = (preRollHead + 1) % PRE_ROLL_FRAMES;
                        if (preRollCount < PRE_ROLL_FRAMES) preRollCount++;

                        // 30s without ANY speech: fire timeout
                        if (!hadSpeech && silentFrames >= TIMEOUT_FRAMES) {
                            Log.d(TAG, "VAD timeout — no speech in 30s");
                            sendUpdate(MSG_VAD_TIMEOUT);
                            // Reset for next 30s window — don't stop AudioRecord
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
                    flushSegment(segment);
                }
            }
        }

        // ── Session ended (stop() was called) ────────────────────────────────
        // Flush any partial segment
        if (inSpeech && segment.size() > MIN_SEGMENT_BYTES) {
            Log.d(TAG, "Flushing partial segment on stop: " + segment.size() + " bytes");
            flushSegment(segment);
        }

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
     * Resets the segment stream for the next utterance.
     * AudioRecord continues running — this is NOT a stop/start cycle.
     */
    private void flushSegment(ByteArrayOutputStream segment) {
        byte[] audio = segment.toByteArray();
        segment.reset();

        if (audio.length < MIN_SEGMENT_BYTES) {
            Log.d(TAG, "Segment too short (" + audio.length + " bytes) — discarding");
            sendUpdate(MSG_RECORDING_ERROR);
            return;
        }

        RecordBuffer.setOutputBuffer(audio);
        // IMPORTANT: MSG_RECORDING_DONE is sent AFTER the buffer is written and
        // BEFORE AudioRecord is stopped — the IME must NOT call stop() in response
        // to this message. It only needs to process the audio. AudioRecord continues.
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
