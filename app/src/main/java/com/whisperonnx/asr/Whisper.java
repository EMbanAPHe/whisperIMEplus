package com.whisperonnx.asr;

import static android.content.Intent.FLAG_ACTIVITY_NEW_TASK;

import android.content.Context;
import android.content.Intent;
import android.util.Log;

import com.whisperonnx.SetupActivity;
import com.whisperonnx.voice_translation.neural_networks.NeuralNetworkApi;
import com.whisperonnx.voice_translation.neural_networks.voice.Recognizer;
import com.whisperonnx.voice_translation.neural_networks.voice.RecognizerListener;

import java.io.File;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class Whisper {

    public interface WhisperListener {
        void onUpdateReceived(String message);
        void onResultReceived(WhisperResult result);
    }

    private static final String TAG = "Whisper";
    public static final String MSG_PROCESSING_DONE = "Processing done...!"; // used in RecognizerListener log

    private final AtomicBoolean mInProgress = new AtomicBoolean(false);

    private Recognizer.Action mAction;
    private String mLangCode = "";
    private WhisperListener mUpdateListener;

    private final Lock taskLock = new ReentrantLock();
    private final Condition hasTask = taskLock.newCondition();
    private volatile boolean taskAvailable = false;
    private Recognizer recognizer = null;
    private Context mContext;
    private long startTime;

    /**
     * Reference to the processRecordBufferLoop thread so unloadModel() can
     * interrupt it cleanly, preventing it from remaining permanently blocked
     * on hasTask.await() after the recognizer has been destroyed.
     */
    private Thread processingThread = null;

    public Whisper(Context context) {
        mContext = context;

        // Check if model is installed
        File sdcardDataFolder = mContext.getExternalFilesDir(null);

        if (sdcardDataFolder != null && !sdcardDataFolder.exists()
                && !sdcardDataFolder.mkdirs()) {
            Log.e(TAG, "Failed to make directory: " + sdcardDataFolder);
            return;
        }

        // Guard against null from getExternalFilesDir (e.g. external storage unavailable)
        if (sdcardDataFolder == null) {
            Log.e(TAG, "External files directory is null — cannot check model");
            return;
        }

        File[] files = sdcardDataFolder.listFiles();

        // listFiles() returns null if the path is not a directory or an I/O error occurs
        int modelFileCount = 0;
        if (files != null) {
            for (File file : files) {
                if (file.isFile()) modelFileCount++;
            }
        }

        if (modelFileCount != 6) {
            // Model not installed — launch setup
            Intent intent = new Intent(mContext, SetupActivity.class);
            intent.addFlags(FLAG_ACTIVITY_NEW_TASK);
            mContext.startActivity(intent);
        }
        // processingThread is started at the end of loadModel() once the
        // recognizer is initialised, not here. Starting it in the constructor
        // before loadModel() is called would allow start() to signal the thread
        // before recognizer exists, causing a NullPointerException.
    }

    public void setListener(WhisperListener listener) {
        this.mUpdateListener = listener;
    }

    public void loadModel() {
        recognizer = new Recognizer(mContext, false, new NeuralNetworkApi.InitListener() {
            @Override
            public void onInitializationFinished() {
                Log.d(TAG, "Recognizer initialized");
            }

            @Override
            public void onError(int[] reasons, long value) {
                Log.d(TAG, "Recognizer init error");
            }
        });

        recognizer.addCallback(new RecognizerListener() {
            @Override
            public void onSpeechRecognizedResult(String text, String languageCode,
                    double confidenceScore, boolean isFinal) {
                Log.d(TAG, languageCode + " " + text);
                WhisperResult whisperResult = new WhisperResult(text, languageCode, mAction);
                sendResult(whisperResult);
                long timeTaken = android.os.SystemClock.elapsedRealtime() - startTime;
                Log.d(TAG, "Time Taken for transcription: " + timeTaken + "ms");
                // MSG_PROCESSING_DONE not sent — onUpdateReceived is a no-op in the IME listener
            }

            @Override
            public void onError(int[] reasons, long value) {
                Log.d(TAG, "ERROR during recognition");
            }
        });

        // Start the processing thread here, after recognizer is fully initialised,
        // so processRecordBuffer() can never be called on a null recognizer.
        processingThread = new Thread(this::processRecordBufferLoop, "WhisperInference");
        processingThread.start();
    }

    /**
     * Destroys the ONNX recognizer and interrupts the processing thread.
     *
     * Calling interrupt() on processingThread causes hasTask.await() to throw
     * InterruptedException, which processRecordBufferLoop catches and uses to
     * exit its loop cleanly.  Without this, the thread would remain permanently
     * blocked after recognizer.destroy() is called, leaking a thread until the
     * process is killed.
     */
    public void unloadModel() {
        if (processingThread != null) {
            processingThread.interrupt();
            // Join with a generous timeout so we don't destroy the recognizer
            // while native ONNX inference is still accessing it. The interrupt
            // will unblock hasTask.await() immediately if the thread is idle;
            // if inference is running, we wait for it to complete naturally
            // (recognize() cannot be cancelled mid-run without native support).
            try {
                processingThread.join(5000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            if (processingThread.isAlive()) {
                Log.w(TAG, "Processing thread did not stop within 5s — proceeding with destroy");
            }
            processingThread = null;
        }
        if (recognizer != null) {
            recognizer.destroy();
            recognizer = null;
        }
    }

    public void setAction(Recognizer.Action action) {
        this.mAction = action;
    }

    public void setLanguage(String language) {
        this.mLangCode = language;
    }

    public void start() {
        if (!mInProgress.compareAndSet(false, true)) {
            Log.d(TAG, "Execution is already in progress...");
            return;
        }
        taskLock.lock();
        try {
            taskAvailable = true;
            hasTask.signal();
        } finally {
            taskLock.unlock();
        }
    }

    public void stop() {
        mInProgress.set(false);
    }

    public boolean isInProgress() {
        return mInProgress.get();
    }

    private void processRecordBufferLoop() {
        while (!Thread.currentThread().isInterrupted()) {
            taskLock.lock();
            try {
                while (!taskAvailable) {
                    hasTask.await(); // throws InterruptedException when interrupted
                }
                processRecordBuffer();
                taskAvailable = false;
            } catch (InterruptedException e) {
                // Restore interrupt flag and exit loop cleanly
                Thread.currentThread().interrupt();
            } finally {
                taskLock.unlock();
            }
        }
        Log.d(TAG, "processRecordBufferLoop exiting cleanly");
    }

    private void processRecordBuffer() {
        // Capture a local reference to guard against unloadModel() nulling the field
        // on another thread while inference is running (defensive; lifecycle should
        // prevent this, but safe to be explicit).
        final Recognizer r = recognizer;
        try {
            if (r != null && RecordBuffer.getOutputBuffer() != null) {
                startTime = android.os.SystemClock.elapsedRealtime(); // monotonic — unaffected by clock changes
                // MSG_PROCESSING is not used by the listener (onUpdateReceived is a no-op)
                // so we skip that call entirely.
                r.recognize(RecordBuffer.getSamples(), 1, mLangCode, mAction);
            } else {
                sendUpdate("Engine not initialized or audio buffer is empty");
            }
        } catch (Exception e) {
            Log.e(TAG, "Error during transcription", e);
            sendUpdate("Transcription failed: " + e.getMessage());
        } finally {
            mInProgress.set(false);
        }
    }

    private void sendUpdate(String message) {
        if (mUpdateListener != null) {
            mUpdateListener.onUpdateReceived(message);
        }
    }

    private void sendResult(WhisperResult whisperResult) {
        if (mUpdateListener != null) {
            mUpdateListener.onResultReceived(whisperResult);
        }
    }
}
