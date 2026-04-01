package com.whisperonnx;

import static com.whisperonnx.voice_translation.neural_networks.voice.Recognizer.ACTION_TRANSCRIBE;
import static com.whisperonnx.voice_translation.neural_networks.voice.Recognizer.ACTION_TRANSLATE;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.inputmethodservice.InputMethodService;
import android.os.Handler;
import android.view.HapticFeedbackConstants;
import android.os.Looper;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.ExtractedText;
import android.view.inputmethod.ExtractedTextRequest;
import android.widget.ImageButton;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.core.content.ContextCompat;
import androidx.preference.PreferenceManager;

import com.github.houbb.opencc4j.util.ZhConverterUtil;
import com.whisperonnx.asr.RecordBuffer;
import com.whisperonnx.asr.Recorder;
import com.whisperonnx.asr.Whisper;
import com.whisperonnx.asr.WhisperResult;
import com.whisperonnx.utils.HapticFeedback;

/**
 * Whisper+ IME keyboard panel.
 *
 * ── Architecture notes ─────────────────────────────────────────────────────
 *
 * Recorder lifecycle
 *   Created once in onCreate() and reused for the service lifetime.
 *   Recorder's worker thread (while-true loop) runs continuously but blocks
 *   on a Condition lock when idle — negligible CPU/battery cost at rest.
 *   Creating it once avoids the thread-leak that occurred when it was
 *   recreated in onCreateInputView() on every keyboard inflation.
 *
 * Whisper lifecycle
 *   Created once in onCreate() and loaded on a background thread so the
 *   ONNX initialisation (1-3 s) does not block the main thread and the
 *   model is ready before the user first opens the keyboard.
 *   The listener is re-set in onCreateInputView() via setWhisperListener()
 *   so result callbacks always reference the current view's fields.
 *   unloadModel() is only called in onDestroy(), keeping the model hot
 *   across multiple keyboard invocations.
 *
 * Threading contract
 *   RecorderListener callbacks  → Recorder worker thread  → UI via handler
 *   WhisperListener callbacks   → Whisper inference thread → UI via handler
 *   InputConnection.commitText()  safe from any thread (Android docs)
 *   Recorder.stop() blocks for ≤ one audio frame (~30 ms) under normal use.
 *   It is called synchronously from the recording thread's callbacks and
 *   asynchronously (new Thread) everywhere else to avoid stalling the UI.
 *
 * Mic button — three visual states
 *   IDLE       rounded_button_tonal       dark grey    not listening
 *   LISTENING  rounded_button_listening   dim purple   VAD running, no speech yet
 *   RECORDING  rounded_button_background  full purple  speech detected, capturing
 *
 * Haptic feedback
 *   All buttons use performHapticFeedback(KEYBOARD_TAP, FLAG_IGNORE_VIEW_SETTING)
 *   which respects the global haptic-feedback system setting (same as
 *   FlorisBoard) but ignores the per-view flag so it works inside an IME
 *   where views are not directly attached to an Activity window.
 *   The existing HapticFeedback.vibrate() is retained for the end-of-
 *   recording pulse (a distinct, slightly heavier feel).
 *
 * Keyboard height
 *   Sized to match FlorisBoard's default "normal" height: 0.25 × display
 *   height (portrait) / 0.37 × display height (landscape).  The calculation
 *   uses dm.heightPixels (full physical pixels) so density is accounted for.
 */
public class WhisperInputMethodService extends InputMethodService {

    private static final String TAG = "WhisperIME";

    // ── Widgets — assigned in onCreateInputView, null before first inflation ─
    private ImageButton  btnRecord;
    private ImageButton  btnKeyboard;
    private ImageButton  btnSettings;
    private ImageButton  btnSend;
    private ImageButton  btnUndo;
    private ImageButton  btnRedo;
    private ImageButton  btnDel;
    private ImageButton  btnEnter;
    private View         btnCancel;   // TextView in layout, declared as View
    private View         btnClear;    // TextView in layout, declared as View
    private View         btnSpace;    // TextView in layout, declared as View
    private TextView     tvStatus;
    private ProgressBar  processingBar;
    private LinearLayout layoutKeyboardContent;

    // ── Static: survives onCreateInputView re-inflations ──────────────────────
    private static boolean translate = false;

    // ── Session state — volatile for cross-thread visibility ──────────────────

    /** True from startRecordingSession() until the session fully ends. */
    private volatile boolean isListening     = false;
    /** True only while VAD has confirmed speech (between MSG_RECORDING and DONE). */
    private volatile boolean isRecording     = false;
    /** Set before stopping the recorder to suppress post-stop transcription/restart. */
    private volatile boolean cancelRequested = false;
    /** True while continuous VAD-restart loop is desired. */
    private volatile boolean continuousMode  = false;
    /** Set to true after loadModel() completes on its background thread. */
    private volatile boolean modelReady      = false;

    /**
     * Pipelined audio segments waiting to be transcribed.
     * Enqueued by the recorder listener, consumed by processNextQueued().
     * Accessed only on the main thread (via handler.post), so no synchronisation needed.
     */
    private final java.util.ArrayDeque<byte[]> audioQueue = new java.util.ArrayDeque<>();

    // ── Preferences ───────────────────────────────────────────────────────────
    private boolean prefAutoStart;
    private boolean prefAutoStop;
    private boolean prefAutoSwitch;
    private boolean prefAutoSend;

    // Long-press repeat runnables — stored as fields so onFinishInputView can cancel them
    private Runnable delInitial;
    private Runnable delFast;
    private Recorder          mRecorder;
    private Whisper           mWhisper;
    private SharedPreferences sp;
    private final Handler     handler = new Handler(Looper.getMainLooper());
    private Context           mContext;
    /**
     * Single-threaded executor for Recorder.stop() calls.
     * Prevents unbounded thread creation under rapid cancel/re-open cycles.
     * Daemon thread so it does not block JVM shutdown.
     */
    private final java.util.concurrent.ExecutorService stopExecutor =
            java.util.concurrent.Executors.newSingleThreadExecutor(r -> {
                Thread t = new Thread(r, "RecorderStopper");
                t.setDaemon(true);
                return t;
            });

    // ═══════════════════════════════════════════════════════════════════════════
    // Service lifecycle  — Recorder and Whisper are created/destroyed here only
    // ═══════════════════════════════════════════════════════════════════════════

    @Override
    public void onCreate() {
        mContext = this;
        super.onCreate();

        // ── Recorder: one instance for the service lifetime ────────────────────
        // The Recorder constructor starts a permanent worker thread (while-true
        // loop blocked on a Condition). Creating it here prevents the thread-leak
        // that occurred when new Recorder() was called in onCreateInputView().
        mRecorder = new Recorder(this);

        // ── Whisper: load on a background thread to avoid main-thread blocking ─
        mWhisper = new Whisper(this);
        final Whisper whisperRef = mWhisper; // local capture — immune to onDestroy nulling mWhisper
        new Thread(() -> {
            whisperRef.loadModel();
            handler.post(() -> {
                // Confirm mWhisper still points to the same instance (not destroyed)
                if (mWhisper != whisperRef) return;
                modelReady = true;
                Log.d(TAG, "Model ready");
                if (isInputViewShown() && prefAutoStart && !isListening) {
                    startRecordingSession(prefAutoStop);
                }
                if (tvStatus != null) tvStatus.setVisibility(View.GONE);
            });
        }, "WhisperModelLoader").start();

        // Read preferences early so prefAutoStart is valid in the callback above
        sp = PreferenceManager.getDefaultSharedPreferences(this);
        loadPrefs();
    }

    @Override
    public void onDestroy() {
        // Cancel all pending handler callbacks first, before anything else.
        // This prevents post-destroy Runnables (from Whisper/Recorder callbacks
        // still in flight) from running against null or stale widget references.
        handler.removeCallbacksAndMessages(null);

        // Shut down the RecorderStopper executor — no new tasks after this
        stopExecutor.shutdownNow();

        cancelRequested = true;
        continuousMode  = false;
        audioQueue.clear();
        if (mRecorder != null && mRecorder.isInProgress()) mRecorder.stop();
        if (mRecorder != null) mRecorder.shutdown();

        // Stop Whisper inference before releasing the model. mWhisper.stop() sets
        // mInProgress=false so the processRecordBufferLoop thread stops waiting for
        // a new task and the recognizer.destroy() call below happens cleanly.
        // This also prevents the processRecordBufferLoop thread from being left
        // permanently blocked on hasTask.await() after recognizer.destroy().
        if (mWhisper != null) {
            mWhisper.stop();       // signal inference to stop if running
            mWhisper.unloadModel(); // destroy the ONNX recognizer
            mWhisper = null;
        }

        super.onDestroy();
    }

    @Override
    public void onStartInput(EditorInfo attribute, boolean restarting) {
        // For TYPE_NULL fields (e.g. terminal, password managers) just stop
        // recording — keeping the model loaded saves reload time on the next field.
        if (attribute.inputType == EditorInfo.TYPE_NULL) {
            exitListeningMode(true);
        }
    }

    @Override
    public void onStartInputView(EditorInfo attribute, boolean restarting) {
        loadPrefs();
        setWhisperListener(); // re-bind listener to current view fields

        if (!modelReady) {
            // Model is still loading — show a status message and wait.
            // The onCreate background thread will call startRecordingSession()
            // via handler.post() when the model becomes ready.
            if (tvStatus != null) {
                tvStatus.setText(getString(R.string.loading_model));
                tvStatus.setVisibility(View.VISIBLE);
            }
            return;
        }

        if (prefAutoStart && !isListening) {
            startRecordingSession(prefAutoStop);
        }
    }

    @Override
    public void onFinishInputView(boolean finishingInput) {
        // Keyboard is hiding — stop the session asynchronously so we never
        // block the main thread waiting for the recorder to drain.
        exitListeningMode(true);

        // Cancel any pending long-press repeat Runnables
        if (delInitial != null) { handler.removeCallbacks(delInitial); delInitial = null; }
        if (delFast    != null) { handler.removeCallbacks(delFast);    delFast    = null; }

        // Null out all widget references so that any handler.post() callbacks
        // that fire before the next onCreateInputView() don't operate on a
        // detached view hierarchy. setMicState, setCancelEnabled, and
        // resetProgressUi all guard against null, so this is safe.
        btnRecord             = null;
        btnKeyboard           = null;
        btnSettings           = null;
        btnSend               = null;
        btnUndo               = null;
        btnRedo               = null;
        btnDel                = null;
        btnEnter              = null;
        btnCancel             = null;
        btnClear              = null;
        btnSpace              = null;
        tvStatus              = null;
        processingBar         = null;
        layoutKeyboardContent = null;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // IME view
    // ═══════════════════════════════════════════════════════════════════════════

    @Override
    @SuppressLint("ClickableViewAccessibility")
    public View onCreateInputView() {
        sp = PreferenceManager.getDefaultSharedPreferences(this);
        loadPrefs();

        View view = getLayoutInflater().inflate(R.layout.voice_service, null);

        // ── Bind all widgets ───────────────────────────────────────────────────
        btnRecord             = view.findViewById(R.id.btnRecord);
        btnKeyboard           = view.findViewById(R.id.btnKeyboard);
        btnSettings           = view.findViewById(R.id.btnSettings);
        btnSend               = view.findViewById(R.id.btnSend);
        btnUndo               = view.findViewById(R.id.btnUndo);
        btnRedo               = view.findViewById(R.id.btnRedo);
        btnDel                = view.findViewById(R.id.btnDel);
        btnEnter              = view.findViewById(R.id.btnEnter);
        btnCancel             = view.findViewById(R.id.btnCancel);
        btnClear              = view.findViewById(R.id.btnClear);
        btnSpace              = view.findViewById(R.id.btnSpace);
        processingBar         = view.findViewById(R.id.processing_bar);
        tvStatus              = view.findViewById(R.id.tv_status);
        layoutKeyboardContent = view.findViewById(R.id.layout_keyboard_content);

        // Size the keyboard panel to match FlorisBoard's default height
        setKeyboardHeight();

        // Re-attach the Whisper listener so callbacks reference the new views
        setWhisperListener();

        // Set the Recorder listener — the lambda reads current field values at
        // callback time, so re-setting here is not strictly required, but it
        // guarantees the listener is always attached after any future changes.
        setRecorderListener();

        checkRecordPermission();

        // ── Restore button visual state after orientation change / re-inflation ─
        // The new view starts in idle state; if a session is active we must
        // restore the correct colours so the UI matches the actual state.
        restoreButtonState();

        // ────────────────────────────────────────────────────────────────────────
        // Button listeners
        // All buttons call tap() for immediate KEYBOARD_TAP haptic feedback.
        // ────────────────────────────────────────────────────────────────────────

        // ── Mic — toggle listening mode ────────────────────────────────────────
        btnRecord.setOnClickListener(v -> {
            tap(v);
            if (isListening) {
                // Tapped while active → exit listening mode entirely
                exitListeningMode(true);
            } else {
                if (!checkRecordPermission()) return;
                if (!modelReady) {
                    if (tvStatus != null) {
                        tvStatus.setText(getString(R.string.loading_model));
                        tvStatus.setVisibility(View.VISIBLE);
                    }
                    return;
                }
                startRecordingSession(prefAutoStop);
            }
        });

        // ── Cancel — hard stop in all modes ───────────────────────────────────
        // Stops the recorder, cancels any in-progress transcription, discards
        // the current audio buffer and goes idle immediately.
        btnCancel.setOnClickListener(v -> {
            tap(v);
            if (mWhisper != null) stopTranscription();
            exitListeningMode(true);
        });

        // ── Settings ──────────────────────────────────────────────────────────
        btnSettings.setOnClickListener(v -> {
            tap(v);
            Intent i = new Intent(this, SettingsActivity.class);
            i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(i);
        });

        // ── Switch keyboard ────────────────────────────────────────────────────
        btnKeyboard.setOnClickListener(v -> {
            tap(v);
            if (mWhisper != null) stopTranscription();
            exitListeningMode(false); // sync ok here — not yet hiding the view
            switchToPreviousInputMethod();
        });

        // ── Delete unselected text ─────────────────────────────────────────────
        btnClear.setOnClickListener(v -> {
            tap(v);
            if (getCurrentInputConnection() == null) return;
            ExtractedText et = getCurrentInputConnection()
                    .getExtractedText(new ExtractedTextRequest(), 0);
            if (et != null && et.text != null) {
                int len = et.text.length();
                CharSequence before = getCurrentInputConnection()
                        .getTextBeforeCursor(len, 0);
                CharSequence after = getCurrentInputConnection()
                        .getTextAfterCursor(len, 0);
                if (before != null && after != null) {
                    getCurrentInputConnection()
                            .deleteSurroundingText(before.length(), after.length());
                }
            }
        });

        // ── Undo / Redo ────────────────────────────────────────────────────────
        btnUndo.setOnClickListener(v -> { tap(v); sendCtrlKey(KeyEvent.KEYCODE_Z, false); });
        btnRedo.setOnClickListener(v -> { tap(v); sendCtrlKey(KeyEvent.KEYCODE_Z, true);  });

        // ── Backspace with long-press repeat ───────────────────────────────────
        // Uses instance-level delInitial / delFast so onFinishInputView() can
        // cancel them even if the view is torn down during a long-press.
        btnDel.setOnTouchListener((v, event) -> {
            switch (event.getAction()) {
                case MotionEvent.ACTION_DOWN:
                    tap(v);
                    doDelete();
                    delInitial = () -> {
                        doDelete();
                        delFast = new Runnable() {
                            @Override public void run() {
                                doDelete();
                                handler.postDelayed(this, 50);
                            }
                        };
                        handler.postDelayed(delFast, 50);
                    };
                    handler.postDelayed(delInitial, 400);
                    break;
                case MotionEvent.ACTION_UP:
                case MotionEvent.ACTION_CANCEL:
                    if (delInitial != null) { handler.removeCallbacks(delInitial); delInitial = null; }
                    if (delFast    != null) { handler.removeCallbacks(delFast);    delFast    = null; }
                    break;
                default: break;
            }
            return true;
        });

        // ── Space bar ─────────────────────────────────────────────────────────
        btnSpace.setOnClickListener(v -> {
            tap(v);
            if (getCurrentInputConnection() != null)
                getCurrentInputConnection().commitText(" ", 1);
        });

        // ── Send (IME action) ──────────────────────────────────────────────────
        btnSend.setOnClickListener(v -> {
            tap(v);
            if (getCurrentInputConnection() == null) return;
            EditorInfo ei = getCurrentInputEditorInfo();
            if (ei != null && ei.actionId != 0
                    && ei.actionId != EditorInfo.IME_ACTION_NONE
                    && ei.actionId != EditorInfo.IME_ACTION_UNSPECIFIED) {
                getCurrentInputConnection().performEditorAction(ei.actionId);
            } else {
                getCurrentInputConnection()
                        .performEditorAction(EditorInfo.IME_ACTION_SEND);
            }
        });

        // ── Enter / newline ────────────────────────────────────────────────────
        btnEnter.setOnClickListener(v -> {
            tap(v);
            if (getCurrentInputConnection() == null) return;
            getCurrentInputConnection().sendKeyEvent(
                    new KeyEvent(KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_ENTER));
            getCurrentInputConnection().sendKeyEvent(
                    new KeyEvent(KeyEvent.ACTION_UP, KeyEvent.KEYCODE_ENTER));
        });

        return view;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Recorder listener  (can be re-set safely on view re-inflation)
    // ═══════════════════════════════════════════════════════════════════════════

    private void setRecorderListener() {
        mRecorder.setListener(message -> {
            // Called on Recorder worker thread — all UI via handler.post()

            if (message.equals(Recorder.MSG_RECORDING)) {
                // VAD confirmed speech — upgrade to RECORDING (bright) state
                isRecording = true;
                handler.post(() ->
                        setMicState(MicState.RECORDING));

            } else if (message.equals(Recorder.MSG_RECORDING_DONE)) {
                // Speech segment captured.
                //
                // IMPORTANT: we are on the Recorder worker thread. recordLoop()'s
                // finally { mInProgress.set(false) } has NOT yet run — it executes
                // after recordAudio() returns, which is after MSG_RECORDING_DONE fires.
                // Therefore we must NOT call mRecorder.start() here directly, or
                // recordLoop()'s finally block would reset mInProgress=false and
                // break the next recording. We POST everything to the handler so it
                // runs on the main thread AFTER the finally block has completed.
                isRecording = false;
                HapticFeedback.vibrate(mContext);

                if (!cancelRequested) {
                    // Snapshot the audio bytes NOW, before any new recording can
                    // overwrite RecordBuffer. RecordBuffer.getOutputBuffer() is
                    // synchronized so this is thread-safe.
                    final byte[] capturedAudio = RecordBuffer.getOutputBuffer();

                    handler.post(() -> {
                        // By the time this runs on the main thread:
                        //   • recordLoop()'s finally block has completed
                        //   • mRecorder.mInProgress is false
                        //   • It is safe to call mRecorder.start()
                        if (cancelRequested) return; // check again on main thread

                        // Enqueue this segment for transcription
                        audioQueue.offer(capturedAudio);

                        // Start transcription of this segment immediately
                        processNextQueued();

                        if (continuousMode) {
                            // Pipeline: restart recording for the NEXT segment while
                            // Whisper processes the current one.
                            // initVad() must be called before start() each time.
                            mRecorder.initVad();
                            mRecorder.start();
                            // Mic shows LISTENING — waiting for next speech
                            setMicState(MicState.LISTENING);
                        } else {
                            // Single-take: show LISTENING (dim) during processing
                            setMicState(MicState.LISTENING);
                        }
                    });
                } else {
                    // Cancelled — go fully idle.
                    // NOTE: do NOT reset cancelRequested here. Whisper may still be
                    // processing a previously buffered segment on its inference thread.
                    // The guard in onResultReceived reads cancelRequested; clearing it
                    // here would allow that result to be committed after cancel.
                    // cancelRequested is only cleared in startRecordingSession() when
                    // the user explicitly starts a new session.
                    isListening    = false;
                    continuousMode = false;
                    handler.post(() -> {
                        audioQueue.clear();
                        setMicState(MicState.IDLE);
                        setCancelEnabled(false);
                        resetProgressUi();
                    });
                }

            } else if (message.equals(Recorder.MSG_RECORDING_ERROR)) {
                isRecording    = false;
                isListening    = false;
                // Do NOT reset cancelRequested here — see MSG_RECORDING_DONE comment.
                continuousMode = false;
                HapticFeedback.vibrate(mContext);
                handler.post(() -> {
                    setMicState(MicState.IDLE);
                    setCancelEnabled(false);
                    if (tvStatus != null) {
                        tvStatus.setText(getString(R.string.error_no_input));
                        tvStatus.setVisibility(View.VISIBLE);
                    }
                    resetProgressUi();
                });
            }
        });
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Whisper listener  (re-set on view re-inflation to keep references fresh)
    // ═══════════════════════════════════════════════════════════════════════════

    private void setWhisperListener() {
        if (mWhisper == null) return;
        mWhisper.setListener(new Whisper.WhisperListener() {
            @Override public void onUpdateReceived(String message) { /* unused */ }

            @Override
            public void onResultReceived(WhisperResult whisperResult) {
                // Runs on Whisper inference thread.
                //
                // ── Cancel guard — check FIRST, before committing any text ────────
                // cancelRequested is volatile. Reading it here on the inference thread
                // is guaranteed to see the value written by the main thread when the
                // user pressed Cancel, even though ONNX inference cannot be interrupted
                // mid-run. This ensures no text is ever committed after Cancel.
                if (cancelRequested) {
                    handler.post(() -> {
                        audioQueue.clear();
                        isListening = false;
                        setMicState(MicState.IDLE);
                        setCancelEnabled(false);
                        resetProgressUi();
                    });
                    return;
                }

                handler.post(() -> resetProgressUi());

                // Chinese script conversion
                String result = whisperResult.getResult();
                if ("zh".equals(whisperResult.getLanguage())) {
                    boolean simple = sp != null && sp.getBoolean("simpleChinese", false);
                    result = simple ? ZhConverterUtil.toSimple(result)
                                    : ZhConverterUtil.toTraditional(result);
                }

                final String trimmed = result.trim();
                if (!trimmed.isEmpty() && getCurrentInputConnection() != null) {
                    // commitText is thread-safe per Android docs.
                    // Trailing space separates consecutive utterances.
                    getCurrentInputConnection().commitText(trimmed + " ", 1);
                }

                // Auto-send
                if (prefAutoSend && getCurrentInputConnection() != null) {
                    EditorInfo ei = getCurrentInputEditorInfo();
                    if (ei != null && ei.actionId != 0
                            && ei.actionId != EditorInfo.IME_ACTION_NONE) {
                        getCurrentInputConnection().performEditorAction(ei.actionId);
                    } else {
                        getCurrentInputConnection()
                                .performEditorAction(EditorInfo.IME_ACTION_SEND);
                    }
                }

                // Auto-switch (single-take sessions only)
                if (prefAutoSwitch && !continuousMode) {
                    isListening = false;
                    handler.post(() -> {
                        audioQueue.clear();
                        setMicState(MicState.IDLE);
                        setCancelEnabled(false);
                        switchToPreviousInputMethod();
                    });
                    return;
                }

                // In continuous mode, recording is already running in parallel
                // (restarted in MSG_RECORDING_DONE handler before transcription began).
                // All we do here is process the next queued segment if one arrived
                // while we were busy, then update the UI.
                handler.post(() -> {
                    if (cancelRequested) {
                        // Session was cancelled while we were transcribing
                        audioQueue.clear();
                        isListening = false;
                        setMicState(MicState.IDLE);
                        setCancelEnabled(false);
                        return;
                    }
                    if (!continuousMode) {
                        // Single-take: session ends after this result
                        isListening = false;
                        setMicState(MicState.IDLE);
                        setCancelEnabled(false);
                    } else {
                        // Drain the queue — process the next buffered segment
                        // if one arrived while we were busy with this one.
                        processNextQueued();
                        // If nothing queued, mic is already running (LISTENING or
                        // RECORDING), so we just leave the state as-is.
                    }
                });
            }
        });
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Session helpers
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Enter listening mode.
     * @param useVad  true  → VAD auto-stops on silence, then restarts after
     *                        each transcription (continuous/streaming mode).
     *                false → user taps mic again to stop.
     */
    private void startRecordingSession(boolean useVad) {
        if (!checkRecordPermission()) return;
        cancelRequested         = false;
        isListening             = true;
        continuousMode          = useVad;
        handler.post(() -> {
            setMicState(MicState.LISTENING);
            setCancelEnabled(true);
            if (tvStatus != null) tvStatus.setVisibility(View.GONE);
            resetProgressUi();
        });
        if (useVad) mRecorder.initVad();
        mRecorder.start();
    }

    /**
     * Exit listening mode fully — stop the recorder, clear all session flags,
     * reset the mic button to IDLE.
     *
     * @param async  true  → Recorder.stop() runs on a background thread.
     *                        The isInProgress() check is done INSIDE the thread,
     *                        not before starting it, eliminating the TOCTOU race
     *                        where recording could end naturally between the check
     *                        and the wait() call inside stop(). If recording has
     *                        already ended, isInProgress() returns false and we
     *                        skip stop() entirely — no deadlock.
     *               false → synchronous. Only used from btnKeyboard where we
     *                        know recording is not in progress (session was
     *                        already stopped before this call).
     */
    private void exitListeningMode(boolean async) {
        cancelRequested = true;
        continuousMode  = false;
        isListening     = false;
        isRecording     = false;

        if (mRecorder != null) {
            if (async) {
                final Recorder r = mRecorder;
                stopExecutor.execute(() -> {
                    if (r.isInProgress()) r.stop();
                });
            } else {
                if (mRecorder.isInProgress()) mRecorder.stop();
            }
        }

        // Update UI immediately — cancelRequested suppresses any pending
        // MSG_RECORDING_DONE from triggering transcription.
        handler.post(() -> {
            audioQueue.clear();
            setMicState(MicState.IDLE);
            setCancelEnabled(false);
            resetProgressUi();
        });
    }

    /**
     * Writes the given audio bytes to RecordBuffer and signals Whisper to
     * begin inference.  Must be called on the main thread (it's always called
     * from handler.post or processNextQueued, both of which run on main).
     */
    private void startTranscription(byte[] audioBytes) {
        if (mWhisper == null || audioBytes == null) return;
        if (processingBar != null) {
            processingBar.setProgress(0);
            processingBar.setIndeterminate(true);
        }
        // Write the snapshot to RecordBuffer before calling mWhisper.start().
        // RecordBuffer.setOutputBuffer() is synchronized, safe from main thread.
        RecordBuffer.setOutputBuffer(audioBytes);
        mWhisper.setAction(translate ? ACTION_TRANSLATE : ACTION_TRANSCRIBE);
        mWhisper.setLanguage(sp != null ? sp.getString("language", "auto") : "auto");
        Log.d(TAG, "startTranscription: " + audioBytes.length + " bytes, translate=" + translate);
        mWhisper.start();
    }

    /**
     * Dequeues the next audio segment and transcribes it.
     * If the queue is empty, does nothing (recording is still running and will
     * enqueue the next segment when VAD fires).
     * Must be called on the main thread.
     */
    private void processNextQueued() {
        if (mWhisper != null && mWhisper.isInProgress()) {
            // Whisper is still busy — onResultReceived will call us again when done
            return;
        }
        byte[] next = audioQueue.poll();
        if (next != null) {
            startTranscription(next);
        }
    }

    private void stopTranscription() {
        audioQueue.clear(); // discard any buffered segments
        handler.post(() -> resetProgressUi());
        if (mWhisper != null) mWhisper.stop();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Mic button visual state
    // ═══════════════════════════════════════════════════════════════════════════

    private enum MicState { IDLE, LISTENING, RECORDING }

    /** Must be called on the main thread. */
    private void setMicState(MicState state) {
        if (btnRecord == null) return;
        switch (state) {
            case IDLE:
                btnRecord.setBackgroundResource(R.drawable.rounded_button_tonal);
                break;
            case LISTENING:
                btnRecord.setBackgroundResource(R.drawable.rounded_button_listening);
                break;
            case RECORDING:
                btnRecord.setBackgroundResource(R.drawable.rounded_button_background);
                break;
        }
    }

    /**
     * Restores button colours after onCreateInputView() re-inflates the view
     * (e.g. after an orientation change while recording was active).
     * Must be called on the main thread after all widgets are bound.
     */
    private void restoreButtonState() {
        if (isListening) {
            setCancelEnabled(true);
            setMicState(isRecording ? MicState.RECORDING : MicState.LISTENING);
        } else {
            setCancelEnabled(false);
            setMicState(MicState.IDLE);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // UI helpers — all must be called on the main thread unless noted
    // ═══════════════════════════════════════════════════════════════════════════

    private void setCancelEnabled(boolean enabled) {
        if (btnCancel == null) return;
        btnCancel.setEnabled(enabled);
        btnCancel.setAlpha(enabled ? 1.0f : 0.4f);
    }

    private void resetProgressUi() {
        if (processingBar != null) {
            processingBar.setIndeterminate(false);
            processingBar.setProgress(0);
        }
        if (tvStatus != null) tvStatus.setVisibility(View.GONE);
    }

    /**
     * Haptic feedback on button tap.
     * Uses FLAG_IGNORE_VIEW_SETTING so it fires in the IME window even without
     * a focused Activity, while still honouring the global haptic setting.
     * This matches FlorisBoard's approach.
     */
    private void tap(View v) {
        v.performHapticFeedback(
                HapticFeedbackConstants.KEYBOARD_TAP,
                HapticFeedbackConstants.FLAG_IGNORE_VIEW_SETTING);
    }

    /**
     * Sizes the keyboard content area.
     * Target: ~match FlorisBoard's default "normal" height.
     * Portrait  0.25 × display height  (previously 0.30, then 0.26)
     * Landscape 0.37 × display height  (previously 0.41)
     * The overhead (progress bar 4dp + margins 9dp + padding 8dp ≈ 21dp) is
     * outside this area, so the total keyboard height is slightly more than
     * the content fraction.
     */
    private void setKeyboardHeight() {
        if (layoutKeyboardContent == null) return;
        DisplayMetrics dm = getResources().getDisplayMetrics();
        boolean landscape = getResources().getConfiguration().orientation
                == Configuration.ORIENTATION_LANDSCAPE;
        int targetPx = (int) (dm.heightPixels * (landscape ? 0.38f : 0.29f));
        ViewGroup.LayoutParams lp = layoutKeyboardContent.getLayoutParams();
        lp.height = targetPx;
        layoutKeyboardContent.setLayoutParams(lp);
    }

    private boolean checkRecordPermission() {
        boolean ok = ContextCompat.checkSelfPermission(
                this, android.Manifest.permission.RECORD_AUDIO)
                == PackageManager.PERMISSION_GRANTED;
        if (!ok && tvStatus != null) {
            tvStatus.setText(getString(R.string.need_record_audio_permission));
            tvStatus.setVisibility(View.VISIBLE);
        }
        return ok;
    }

    private void doDelete() {
        if (getCurrentInputConnection() == null) return;
        CharSequence sel = getCurrentInputConnection().getSelectedText(0);
        if (sel != null && sel.length() > 0) {
            getCurrentInputConnection().commitText("", 1);
        } else {
            getCurrentInputConnection().deleteSurroundingText(1, 0);
        }
    }

    private void sendCtrlKey(int keyCode, boolean withShift) {
        if (getCurrentInputConnection() == null) return;
        int meta = KeyEvent.META_CTRL_ON | (withShift ? KeyEvent.META_SHIFT_ON : 0);
        long t = System.currentTimeMillis();
        getCurrentInputConnection().sendKeyEvent(
                new KeyEvent(t, t, KeyEvent.ACTION_DOWN, keyCode, 0, meta));
        getCurrentInputConnection().sendKeyEvent(
                new KeyEvent(t, t, KeyEvent.ACTION_UP, keyCode, 0, 0));
    }

    /** True after the first loadPrefs() call has written all default values. */
    private boolean prefsInitialised = false;

    /**
     * Reads preferences and writes defaults the first time so keys are
     * present in SharedPreferences before any Settings widget reads them.
     * Default-writing is guarded by prefsInitialised so the disk write only
     * happens once per service lifetime regardless of how many times loadPrefs
     * is called.
     */
    private void loadPrefs() {
        if (sp == null) sp = PreferenceManager.getDefaultSharedPreferences(this);

        if (!prefsInitialised) {
            SharedPreferences.Editor ed = null;
            if (!sp.contains(SettingsActivity.KEY_LAUNCH_LISTENING)) {
                if (ed == null) ed = sp.edit();
                ed.putBoolean(SettingsActivity.KEY_LAUNCH_LISTENING, true);
            }
            if (!sp.contains(SettingsActivity.KEY_AUTO_START)) {
                if (ed == null) ed = sp.edit();
                ed.putBoolean(SettingsActivity.KEY_AUTO_START, true);
            }
            if (!sp.contains(SettingsActivity.KEY_AUTO_STOP)) {
                if (ed == null) ed = sp.edit();
                ed.putBoolean(SettingsActivity.KEY_AUTO_STOP, true);
            }
            if (!sp.contains(SettingsActivity.KEY_AUTO_SWITCH)) {
                if (ed == null) ed = sp.edit();
                ed.putBoolean(SettingsActivity.KEY_AUTO_SWITCH, false);
            }
            if (!sp.contains(SettingsActivity.KEY_AUTO_SEND)) {
                if (ed == null) ed = sp.edit();
                ed.putBoolean(SettingsActivity.KEY_AUTO_SEND, false);
            }
            if (ed != null) ed.apply();
            prefsInitialised = true;
        }

        prefAutoStart  = sp.getBoolean(SettingsActivity.KEY_LAUNCH_LISTENING, true);
        prefAutoStop   = sp.getBoolean(SettingsActivity.KEY_AUTO_STOP,        true);
        prefAutoSwitch = sp.getBoolean(SettingsActivity.KEY_AUTO_SWITCH,      false);
        prefAutoSend   = sp.getBoolean(SettingsActivity.KEY_AUTO_SEND,        false);
    }
}
