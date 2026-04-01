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
    public static final String ACTION_STOP = "Stop";
    public static final String ACTION_RECORD = "Record";
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

    private volatile boolean shouldStartRecording = false;
    private volatile boolean useVAD = false;  // volatile: written on main, read on worker
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

    public void initVad() {
        int silenceDurationMs = sp.getInt("silenceDurationMs", 800);
        vad = Vad.builder()
                .setSampleRate(SampleRate.SAMPLE_RATE_16K)
                .setFrameSize(FrameSize.FRAME_SIZE_480)
                .setMode(Mode.VERY_AGGRESSIVE)
                .setSilenceDurationMs(silenceDurationMs)
                .setSpeechDurationMs(200)
                .build();
        useVAD = true;
        Log.d(TAG, "VAD initialized");
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
                Log.e(TAG, "Recording error...", e);
                sendUpdate(e.getMessage());
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
            if (bytesRead > 0) {
                outputBuffer.write(audioData, 0, bytesRead);
                totalBytesRead += bytesRead;
            } else {
                Log.d(TAG, "AudioRecord error, bytes read: " + bytesRead);
                break;
            }

            if (useVAD) {
                // Copy the most recent VAD_FRAME_SIZE*2 bytes directly from audioData
                // into vadWindow — no full outputBuffer.toByteArray() copy needed.
                int srcLen = Math.min(bytesRead, VAD_FRAME_SIZE * 2);
                System.arraycopy(audioData, bytesRead - srcLen, vadWindow,
                        VAD_FRAME_SIZE * 2 - srcLen, srcLen);

                isSpeech = vad.isSpeech(vadWindow);
                if (isSpeech) {
                    if (!isRecording) {
                        Log.d(TAG, "VAD Speech detected: recording starts");
                        sendUpdate(MSG_RECORDING);
                        hadSpeech = true;
                    }
                    isRecording = true;
                } else {
                    if (isRecording) {
                        isRecording = false;
                        mInProgress.set(false);
                    }
                }
            } else {
                if (!isRecording) sendUpdate(MSG_RECORDING);
                isRecording = true;
                hadSpeech   = true;
            }
        }
        Log.d(TAG, "Total bytes recorded: " + totalBytesRead + ", hadSpeech: " + hadSpeech);

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
            if (totalBytesRead > 6400) {
                sendUpdate(MSG_RECORDING_DONE);
            } else {
                sendUpdate(MSG_RECORDING_ERROR);
            }
        }

        notifyStopWaiter();
    }

    /** Wakes any thread blocked in stop()'s timed wait. */
    private void notifyStopWaiter() {
        synchronized (fileSavedLock) {
            fileSavedLock.notify();
        }
    }
}
