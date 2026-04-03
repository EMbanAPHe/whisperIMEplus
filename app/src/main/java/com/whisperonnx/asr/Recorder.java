package com.whisperonnx.asr;

import android.Manifest;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.MediaRecorder;
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

public class Recorder {

    public interface RecorderListener {
        void onUpdateReceived(String message);
    }

    private static final String TAG = "Recorder";
    public static final String MSG_RECORDING = "Recording...";
    public static final String MSG_RECORDING_DONE = "Recording done...!";
    public static final String MSG_RECORDING_ERROR = "Recording error...";
    /**
     * Fired when VAD mode times out (30 s) without detecting any speech.
     * No audio is sent to RecordBuffer.  The IME restarts VAD silently in
     * streaming mode, or goes idle in single-take mode.
     */
    public static final String MSG_VAD_TIMEOUT = "VAD timeout — no speech detected";

    /**
     * Maximum time (ms) that stop() will wait for the recording thread to finish
     * draining audio and calling notify().  Under normal operation the thread
     * finishes within one audio-buffer read (~30 ms). 3 000 ms is a large safety
     * margin that covers even worst-case audio-system latency and eliminates the
     * permanent deadlock that could occur if the permission-revocation path or any
     * future early-return skips the notify() call.
     */
    private static final long STOP_TIMEOUT_MS = 3_000;

    private final Context mContext;
    private final AtomicBoolean mInProgress = new AtomicBoolean(false);

    private RecorderListener mListener;
    private final Lock lock = new ReentrantLock();
    private final Condition hasTask = lock.newCondition();
    private final Object fileSavedLock = new Object();

    private boolean shouldStartRecording = false; // accessed only under lock — no volatile needed
    private volatile boolean useVAD = false;  // volatile: written on main, read on worker
    private volatile int vadAmplitudeThreshold = 500; // RMS below this = silence pre-gate
    /**
     * When set to a future uptime timestamp, the VAD amplitude gate is raised
     * very high until that time. Call suppressVadTrigger() from the IME to blank
     * the gate during spacebar swipe haptics, which couple through the chassis
     * into the microphone and would otherwise trigger recording.
     */
    private volatile long vadSuppressedUntil = 0L;
    private VadWebRTC vad = null;
    private static final int VAD_FRAME_SIZE = 480;
    private SharedPreferences sp;

    private final Thread workerThread;

    public Recorder(Context context) {
        this.mContext = context;
        sp = PreferenceManager.getDefaultSharedPreferences(mContext);
        workerThread = new Thread(this::recordLoop, "RecorderWorker");
        workerThread.start();
    }

    public void setListener(RecorderListener listener) {
        this.mListener = listener;
    }

    public void start() {
        if (!mInProgress.compareAndSet(false, true)) {
            Log.d(TAG, "Recording is already in progress...");
            return;
        }
        lock.lock();
        try {
            Log.d(TAG, "Recording starts now");
            shouldStartRecording = true;
            hasTask.signal();
        } finally {
            lock.unlock();
        }
    }

    /**
     * Suppresses the VAD amplitude gate for the given duration (ms) starting now.
     * Safe to call from any thread — field is volatile.
     * Typical use: call with 250ms each time a cursor-movement haptic fires during
     * a spacebar swipe, so the vibration picked up by the mic doesn't start recording.
     */
    public void suppressVadTrigger(int durationMs) {
        vadSuppressedUntil = android.os.SystemClock.uptimeMillis() + durationMs;
    }

    public void initVad() {
        int silenceDurationMs = sp.getInt("silenceDurationMs", 1000);
        // speechDurationMs is fixed at 50ms (1-2 frames) — configuring it via
        // settings caused word clipping even at minimum values, because the VAD
        // must hear that many ms of speech before confirming. 50ms is imperceptible.
        int modeIndex = sp.getInt("vadMode", 2); // 0=Normal 1=Aggressive 2=VeryAggressive
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
        // Amplitude gate: RMS threshold applied only BEFORE speech is detected.
        // Blocks haptic feedback vibrations, button taps, and quiet ambient noise
        // from triggering recording. Does NOT affect mid-speech frames.
        vadAmplitudeThreshold = sp.getInt("vadAmplitudeThreshold", 800);
        useVAD = true;
        Log.d(TAG, "VAD initialized: silenceMs=" + silenceDurationMs
                + " mode=" + vadMode + " ampThreshold=" + vadAmplitudeThreshold);
    }

    /**
     * Signals the recording thread to stop and waits for it to finish.
     *
     * Uses a timed wait (STOP_TIMEOUT_MS) instead of an indefinite wait() so
     * that this method always returns even if the recording thread exits via an
     * early-return path that does not call notify() (e.g. permission revoked
     * mid-session).  Under normal operation the thread calls notify() well within
     * one audio buffer read (~30 ms), so the timeout is never reached.
     */
    public void stop() {
        Log.d(TAG, "Recording stopped");
        mInProgress.set(false);

        synchronized (fileSavedLock) {
            try {
                fileSavedLock.wait(STOP_TIMEOUT_MS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    /**
     * Permanently shuts down the Recorder.  Interrupts the worker thread so it
     * exits its while(true) loop cleanly.  Call this from the IME's onDestroy()
     * after stop() has been called.  The Recorder cannot be used after shutdown().
     */
    public void shutdown() {
        workerThread.interrupt();
    }

    public boolean isInProgress() {
        return mInProgress.get();
    }

    private void sendUpdate(String message) {
        if (mListener != null)
            mListener.onUpdateReceived(message);
    }

    private void recordLoop() {
        while (true) {
            lock.lock();
            try {
                while (!shouldStartRecording) {
                    hasTask.await();
                }
                shouldStartRecording = false;
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            } finally {
                lock.unlock();
            }

            try {
                recordAudio();
            } catch (Exception e) {
                Log.e(TAG, "Recording error in recordLoop", e);
                sendUpdate(MSG_RECORDING_ERROR);  // must match constant, not e.getMessage()
                notifyStopWaiter();               // unblock any stop() waiter
            } finally {
                mInProgress.set(false);
            }
        }
    }

    private void recordAudio() {
        if (ActivityCompat.checkSelfPermission(mContext, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            Log.d(TAG, "AudioRecord permission is not granted");
            sendUpdate(mContext.getString(R.string.need_record_audio_permission));
            // Notify stop() so it doesn't wait indefinitely.
            // (This path does not reach the normal notify() at the end of the method.)
            notifyStopWaiter();
            return;
        }

        int channels = 1;
        int bytesPerSample = 2;
        int sampleRateInHz = 16000;
        int channelConfig = AudioFormat.CHANNEL_IN_MONO;
        int audioFormat = AudioFormat.ENCODING_PCM_16BIT;
        int audioSource = MediaRecorder.AudioSource.VOICE_RECOGNITION;

        int bufferSize = AudioRecord.getMinBufferSize(sampleRateInHz, channelConfig, audioFormat);
        if (bufferSize < VAD_FRAME_SIZE * 2) bufferSize = VAD_FRAME_SIZE * 2;

        AudioManager audioManager = (AudioManager) mContext.getSystemService(Context.AUDIO_SERVICE);
        // Only start Bluetooth SCO if a headset is connected — starting it
        // unconditionally wakes the Bluetooth stack and causes minor battery drain
        // and occasional audio routing glitches on devices with no BT headset.
        boolean useBluetooth = audioManager.isBluetoothScoAvailableOffCall()
                && audioManager.isBluetoothScoOn();
        if (useBluetooth) {
            audioManager.startBluetoothSco();
            audioManager.setBluetoothScoOn(true);
        }

        AudioRecord.Builder builder = new AudioRecord.Builder()
                .setAudioSource(audioSource)
                .setAudioFormat(new AudioFormat.Builder()
                        .setChannelMask(channelConfig)
                        .setEncoding(audioFormat)
                        .setSampleRate(sampleRateInHz)
                        .build())
                .setBufferSizeInBytes(bufferSize);

        AudioRecord audioRecord = builder.build();
        if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord failed to initialise — state: " + audioRecord.getState());
            audioRecord.release();
            if (useBluetooth) {
                audioManager.stopBluetoothSco();
                audioManager.setBluetoothScoOn(false);
            }
            sendUpdate(MSG_RECORDING_ERROR);
            notifyStopWaiter();
            return;
        }
        audioRecord.startRecording();

        int bytesForThirtySeconds = sampleRateInHz * bytesPerSample * channels * 30;

        ByteArrayOutputStream outputBuffer = new ByteArrayOutputStream();

        byte[] audioData = new byte[bufferSize];
        int totalBytesRead = 0;

        boolean isSpeech;
        boolean isRecording = false;
        boolean hadSpeech   = false;   // true once VAD detects speech at least once
        // Fixed-size ring buffer for VAD — avoids the full outputBuffer.toByteArray()
        // copy that previously happened on every iteration (creating a growing heap
        // allocation every ~30ms, causing GC pressure over long recordings).
        byte[] vadWindow = new byte[VAD_FRAME_SIZE * 2];

        while (mInProgress.get() && totalBytesRead < bytesForThirtySeconds) {
            int bytesRead = audioRecord.read(audioData, 0, VAD_FRAME_SIZE * 2);
            if (bytesRead <= 0) {
                Log.d(TAG, "AudioRecord error, bytes read: " + bytesRead);
                break;
            }
            totalBytesRead += bytesRead;

            if (useVAD) {
                // Update the fixed VAD window with the latest frame
                int srcLen = Math.min(bytesRead, VAD_FRAME_SIZE * 2);
                System.arraycopy(audioData, bytesRead - srcLen, vadWindow,
                        VAD_FRAME_SIZE * 2 - srcLen, srcLen);

                // Pre-gate: only apply amplitude threshold BEFORE speech starts.
                // Once recording is active (isRecording=true) let VAD handle silence
                // detection normally — gating mid-speech drops fricatives ('s','f','h').
                //
                // vadSuppressedUntil: temporarily raises the effective threshold to
                // Integer.MAX_VALUE during spacebar cursor-swipe haptics. CLOCK_TICK
                // vibrations couple through the phone chassis into the mic at ~200-500
                // RMS — within range of the normal threshold. Suppression window blanks
                // the gate for ~250 ms per cursor step so haptics never start recording.
                boolean vadSuppressed = !isRecording
                        && (android.os.SystemClock.uptimeMillis() < vadSuppressedUntil);
                boolean loudEnough = isRecording
                        || (!vadSuppressed && rmsOf(audioData, bytesRead) >= vadAmplitudeThreshold);
                isSpeech = loudEnough && vad.isSpeech(vadWindow);
                if (isSpeech) {
                    if (!isRecording) {
                        if (Log.isLoggable(TAG, Log.DEBUG)) Log.d(TAG, "VAD Speech detected: recording starts");
                        sendUpdate(MSG_RECORDING);
                        hadSpeech = true;
                    }
                    isRecording = true;
                    // Only accumulate output AFTER speech detected — no point
                    // buffering silence that Whisper will never process.
                    outputBuffer.write(audioData, 0, bytesRead);
                } else {
                    if (isRecording) {
                        // Write the final trailing silence frame so Whisper
                        // sees a clean end to the speech segment.
                        outputBuffer.write(audioData, 0, bytesRead);
                        isRecording = false;
                        mInProgress.set(false);
                    }
                }
            } else {
                // Non-VAD mode: accumulate everything
                outputBuffer.write(audioData, 0, bytesRead);
                if (!isRecording) sendUpdate(MSG_RECORDING);
                isRecording = true;
                hadSpeech   = true;
            }
        }
        if (Log.isLoggable(TAG, Log.DEBUG)) Log.d(TAG, "Total bytes recorded: " + totalBytesRead + ", hadSpeech: " + hadSpeech);

        if (useVAD) {
            useVAD = false;
            vad.close();
            vad = null;
            Log.d(TAG, "Closing VAD");
        }
        audioRecord.stop();
        audioRecord.release();
        if (useBluetooth) {
            audioManager.stopBluetoothSco();
            audioManager.setBluetoothScoOn(false);
        }

        if (!hadSpeech) {
            // The 30-second VAD window elapsed without detecting any speech.
            // Don't write silence to RecordBuffer — that would cause Whisper to
            // process empty audio and output "(und)" on every timeout.
            // Fire MSG_VAD_TIMEOUT so the IME can restart VAD silently.
            Log.d(TAG, "VAD timeout — no speech detected, skipping Whisper");
            sendUpdate(MSG_VAD_TIMEOUT);
        } else {
            RecordBuffer.setOutputBuffer(outputBuffer.toByteArray());
            // 3200 bytes = 100ms at 16kHz/16-bit — Whisper handles short audio fine
            if (totalBytesRead > 3200) {
                sendUpdate(MSG_RECORDING_DONE);
            } else {
                sendUpdate(MSG_RECORDING_ERROR);
            }
        }

        notifyStopWaiter();
    }

    /**
     * Returns the RMS (root mean square) amplitude of a PCM-16 audio frame.
     * Values range 0–32767. Typical speech is 1000–8000; haptic vibrations
     * picked up by the mic are typically 100–800; silence is 0–200.
     */
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

    /** Wakes any thread blocked in stop()'s timed wait. */
    private void notifyStopWaiter() {
        synchronized (fileSavedLock) {
            fileSavedLock.notify();
        }
    }
}
