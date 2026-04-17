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
import android.widget.PopupWindow;
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
 *   Recorder.stop() is fire-and-forget (returns instantly).
 *   AudioRecord stays open between sessions within one keyboard show.
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

    private static final String TAG            = "WhisperIME";
    private static final String PREF_LAST_CLOSE_TIME = "imeLastCloseTimeMs";
    private static final long   IDLE_TIMEOUT_MS      = 10L * 60 * 1000; // 10 minutes

    // ── Widgets — assigned in onCreateInputView, null before first inflation ─
    private ImageButton  btnRecord;
    private ImageButton  btnKeyboard;
    private ImageButton  btnSend;
    private ImageButton  btnUndo;
    private ImageButton  btnRedo;
    private ImageButton  btnDel;
    private ImageButton  btnEnter;
    private View         btnCancel;    // TextView in layout, declared as View
    private View         btnClear;     // Delete Unselected — TextView
    private View         btnSpace;     // TextView
    private View         btnDiscard;   // Discard Audio — TextView
    private ImageButton  btnSelectAll; // Select All — icon button
    private ImageButton  btnCut;       // Cut — icon button
    private ImageButton  btnCopy;      // Copy — icon button
    private ImageButton  btnPaste;     // Paste — icon button
    private View         btnFullStop;  // Full Stop — TextView (long press = punctuation popup)
    private ImageButton  btnForwardDel; // Forward Delete — icon button
    private TextView     tvStatus;
    private ProgressBar  processingBar;
    private ViewGroup    layoutKeyboardContent;

    // translate: not static — each service instance owns its own state.
    // volatile so startTranscription() (main thread) sees any future writes from other paths.
    private volatile boolean translate = false;

    // ── Session state — volatile for cross-thread visibility ──────────────────

    /** True from startRecordingSession() until the session fully ends. */
    private volatile boolean isListening     = false;
    /** True only while VAD has confirmed speech (between MSG_RECORDING and DONE). */
    private volatile boolean isRecording     = false;
    /** Set before stopping the recorder to suppress post-stop transcription/restart. */
    private volatile boolean cancelRequested = false;
    /**
     * Set by discardAudioAndContinue() to suppress the next Whisper result without
     * cancelling the whole session. Cleared when the next transcription starts.
     */
    private volatile boolean discardRequested = false;
    /** True while continuous VAD-restart loop is desired. */
    private volatile boolean continuousMode  = false;
    /** Set to true after loadModel() completes on its background thread. */
    private volatile boolean modelReady      = false;

    /**
     * Session lifecycle counters — each is incremented when the corresponding
     * operation starts, and callbacks capture the value AT START TIME, not at
     * fire time.  This means a callback can never act on behalf of a session
     * that was superseded, even if it fires after a new session has begun.
     *
     * sessionGen       — master counter; incremented by startRecordingSession()
     *                    and exitListeningMode().  Used for UI-only callbacks
     *                    (exitListeningMode's handler.post) and RecorderStopper.
     *
     * activeRecordingSessionId — set to the new sessionGen value each time
     *                    mRecorder.start() is called.  The recorder listener
     *                    captures this at callback-fire time, so it always
     *                    reflects the session that STARTED the recording it
     *                    is reporting on.
     *
     * activeWhisperSessionId   — set to sessionGen each time mWhisper.start()
     *                    is called (inside startTranscription).  The Whisper
     *                    listener captures this so stale inference results from
     *                    a closed or cancelled session are never committed.
     */
    private volatile int sessionGen                 = 0;
    private volatile int activeRecordingSessionId   = 0;
    private volatile int activeWhisperSessionId     = 0;

    /**
     * When startRecordingSession() is called but the recorder is still running
     * from a prior session, we cannot call mRecorder.start() immediately —
     * it would fail silently. Instead we park the intended session ID here.
     * The RecorderStopper thread picks this up after the old recording stops
     * and starts the new recording from the background thread's handler.post.
     *
     * Only written on the main thread. Read on main thread (handler.post) and
     * the RecorderStopper executor thread (executor is single-threaded, and the
     * read is inside handler.post, so effectively main-thread only).
     */
    private volatile int     pendingStartSessionId = 0;
    private volatile boolean pendingStartVad       = false;

    /**
     * Pipelined audio segments waiting to be transcribed.
     * Enqueued by the recorder listener, consumed by processNextQueued().
     * Accessed only on the main thread (via handler.post), so no synchronisation needed.
     */
    private final java.util.ArrayDeque<byte[]> audioQueue = new java.util.ArrayDeque<>();

    /**
     * Ring buffer of the last N texts committed via commitText().
     * Used by undoLastTranscription() to remove exactly what was last added.
     * Capacity capped at MAX_UNDO_HISTORY to bound memory use.
     */
    private static final int MAX_UNDO_HISTORY = 10;
    private final java.util.ArrayDeque<String> transcriptionHistory = new java.util.ArrayDeque<>();

    // ── Preferences ───────────────────────────────────────────────────────────
    private boolean prefAutoStart;
    private boolean prefAutoStop;
    private boolean prefAutoSwitch;
    private boolean prefAutoSend;
    /** If false, suppress the end-of-recording haptic pulse in streaming mode. */
    private boolean prefHapticRecording;
    /**
     * If false, do not auto-capitalise the first letter of each transcription segment.
     * Whisper's Recognizer always capitalises — we undo it here when this is false.
     * Useful when dictating mid-sentence fragments or keywords without sentence structure.
     */
    private boolean prefAutoCapitalise;
    /**
     * When true, the next onStartInputView auto-start is suppressed.
     * Set by: Cancel button, mic button (stop), keyboard close mid-session.
     * Cleared by: mic button tap (deliberate record), or when a different editor
     * is focused (different packageName or fieldId than the last cancel).
     * This prevents the pattern: cancel → send → keyboard re-shown → auto-records.
     */
    private boolean suppressNextAutoStart = false;
    /** Package name + field ID at the time suppressNextAutoStart was set. */
    private String  suppressedForEditor   = "";
    /** Uptime when suppression was set — expires after SUPPRESS_TIMEOUT_MS. */
    private long    suppressSetAt         = 0L;
    /** Max time (ms) suppression is active. After this, auto-start resumes normally. */
    private static final long SUPPRESS_TIMEOUT_MS = 8_000L;
    /**
     * Set when the user taps the mic button to START while Whisper is still
     * processing the last segment. The new session is deferred until
     * onResultReceived commits the result, then auto-starts.
     */
    private volatile boolean startAfterWhisper = false;
    /**
     * Language detected in the first "auto" segment of the current session.
     * Subsequent segments use this language instead of re-running detection,
     * saving ~15% of inference time and preventing mid-session language flipping.
     * Reset to "" when the user's language setting changes or a new session starts.
     */
    private String sessionLanguage = "";

    // ── v12 new buttons ─────────────────────────────────────────────
    private View         btnBreakAudio;   // Break Audio - cuts segment, keeps listening
    private ImageButton  btnCapsToggle;   // Capitalisation cycle: auto / upper / lower
    private ImageButton  btnMenuSwap;     // Swap to/from secondary layout
    private LinearLayout layoutPrimary;   // Primary layout container
    private LinearLayout layoutSecondary; // Secondary layout container (audio paused)

    /**
     * Capitalisation override applied to each Whisper result:
     *   0 = auto  (honours prefAutoCapitalise - default)
     *   1 = force upper  (always capitalise first letter)
     *   2 = force lower  (always lowercase first letter)
     */
    private int capsOverride = 0;

    /** True while the secondary layout is showing. */
    private boolean isSecondaryLayout = false;

    /**
     * Whether audio was active when the secondary layout was opened,
     * so we can resume it when the user returns to the primary layout.
     */
    private boolean wasListeningBeforeSecondary = false;

    // ── v12 modifier-key state ────────────────────────────────────────────────
    /** Shift is toggled on — next D-pad direction extends selection. */
    private boolean shiftActive = false;
    /** Ctrl modifier active — sent with the next key event. */
    private boolean ctrlActive  = false;
    /** Alt modifier active — sent with the next key event. */
    private boolean altActive   = false;
    /** Refs to modifier buttons so we can update their visual state. */
    private View btnSecShift;
    private View btnSecCtrl;
    private View btnSecAlt;
    /** Ref to D-pad area for reset on layout switch. */
    private View btnDpad;

    // Long-press repeat runnables - stored as fields so onFinishInputView can cancel them
    private Runnable delInitial;
    private Runnable delFast;
    private Runnable fwdDelInitial;
    private Runnable fwdDelFast;
    private Runnable enterInitial;
    private Runnable enterFast;
    private Recorder          mRecorder;
    private Whisper           mWhisper;
    private SharedPreferences sp;
    private final Handler     handler = new Handler(Looper.getMainLooper());
    /** Active punctuation popup reference. */
    private PopupWindow punctuationPopup = null;
    /** Uptime (ms) when the punctuation popup was last dismissed.
     * Used to suppress the click that dismissed it from immediately reopening it.
     * The dismiss fires during ACTION_DOWN; the click fires during ACTION_UP.
     * If the click arrives within 400ms of dismissal, it is ignored.
     */
    private long punctuationDismissedAt = 0L;
    /** True while the punctuation popup is intentionally showing. */
    private boolean punctuationVisible = false;
    /** Active keyboard options popup — null when dismissed. */
    private PopupWindow keyboardPopup    = null;
    /**
     * Single-threaded executor for Recorder.stop() calls.
     * Prevents unbounded thread creation under rapid cancel/re-open cycles.
     * Daemon thread so it does not block JVM shutdown.
     */
    /** Reused transparent background for popup windows — avoids per-open allocation. */
    private static final android.graphics.drawable.ColorDrawable TRANSPARENT_DRAWABLE =
            new android.graphics.drawable.ColorDrawable(android.graphics.Color.TRANSPARENT);

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
        super.onCreate();

        // ── Recorder: one instance for the service lifetime ────────────────────
        // The Recorder constructor starts a permanent worker thread (while-true
        // loop blocked on a Condition). Creating it here prevents the thread-leak
        // that occurred when new Recorder() was called in onCreateInputView().
        mRecorder = new Recorder(this);

        // ── Whisper: load on a background thread to avoid main-thread blocking ─
        mWhisper = new Whisper(this);
        final Whisper whisperRef = mWhisper; // local capture — immune to onDestroy nulling mWhisper
        Thread modelLoader = new Thread(() -> {
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
        }, "WhisperModelLoader");
        modelLoader.setDaemon(true);
        modelLoader.start();

        // Apply defaults once, then read prefs
        sp = PreferenceManager.getDefaultSharedPreferences(this);
        applyDefaultPrefs();
        loadPrefs();
    }

    @Override
    public void onDestroy() {
        // Cancel all pending handler callbacks first, before anything else.
        // This prevents post-destroy Runnables (from Whisper/Recorder callbacks
        // still in flight) from running against null or stale widget references.
        handler.removeCallbacksAndMessages(null);

        cancelRequested = true;
        continuousMode  = false;
        audioQueue.clear();
        // stop() and closeStream() are now instant (fire-and-forget) — safe on main thread.
        if (mRecorder != null) {
            mRecorder.stop();
            mRecorder.closeStream();
            mRecorder.shutdown();
        }

        // Stop Whisper inference before releasing the model.
        if (mWhisper != null) {
            mWhisper.stop();
            mWhisper.unloadModel();
            mWhisper = null;
        }

        super.onDestroy();
        stopExecutor.shutdownNow();
    }

    @Override
    public void onStartInput(EditorInfo attribute, boolean restarting) {
        int type = attribute.inputType;
        int cls  = type & android.text.InputType.TYPE_MASK_CLASS;
        int var  = type & android.text.InputType.TYPE_MASK_VARIATION;

        // Stop recording for TYPE_NULL (terminal, password manager) and for all
        // password variations — recording into a password field is a security risk.
        boolean isPassword =
            (cls == android.text.InputType.TYPE_CLASS_TEXT && (
                var == android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD ||
                var == android.text.InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD ||
                var == android.text.InputType.TYPE_TEXT_VARIATION_WEB_PASSWORD)) ||
            (cls == android.text.InputType.TYPE_CLASS_NUMBER &&
                var == android.text.InputType.TYPE_NUMBER_VARIATION_PASSWORD);

        if (type == EditorInfo.TYPE_NULL || isPassword) {
            exitListeningMode(true);
        }
    }

    @Override
    public void onStartInputView(EditorInfo attribute, boolean restarting) {
        loadPrefs();
        setWhisperListener(); // re-bind listener so callbacks reference current widget refs

        // Reset visual state unconditionally on every keyboard show.
        // onCreateInputView() is not always called on reshow — Android reuses the
        // existing view. exitListeningMode() already cleared isListening/isRecording
        // so restoreButtonState() correctly shows IDLE here, and the new session
        // (startRecordingSession below) will update it to LISTENING immediately after.
        restoreButtonState();
        resetProgressUi();

        // Idle timeout: if the keyboard has been closed for > 10 minutes and the
        // user enabled "Return to previous keyboard after idle", switch back now.
        if (sp != null && sp.getBoolean("imeReturnAfterIdle", false)) {
            long lastClose = sp.getLong(PREF_LAST_CLOSE_TIME, 0L);
            if (lastClose > 0 && android.os.SystemClock.elapsedRealtime() - lastClose > IDLE_TIMEOUT_MS) {
                sp.edit().putLong(PREF_LAST_CLOSE_TIME, 0L).apply(); // reset so it only fires once
                Log.d(TAG, "Idle timeout reached — switching to previous input method");
                switchToPreviousInputMethod();
                return; // don't show our keyboard
            }
        }

        // Warm up AudioRecord now so the first start() has no initialization gap.
        // openStream() opens the mic hardware immediately; pre-roll starts filling.
        // closeStream() is called from onFinishInputView when the keyboard hides.
        if (mRecorder != null) mRecorder.openStream();

        if (!modelReady) {
            if (tvStatus != null) {
                tvStatus.setText(getString(R.string.loading_model));
                tvStatus.setVisibility(View.VISIBLE);
            }
            return;
        }

        if (prefAutoStart && !isListening) {
            String editorKey = currentEditorKey(attribute);
            long suppressAge = android.os.SystemClock.uptimeMillis() - suppressSetAt;
            if (suppressNextAutoStart
                    && suppressedForEditor.equals(editorKey)
                    && suppressAge < SUPPRESS_TIMEOUT_MS) {
                // Same editor, within the suppression window after an explicit cancel/stop.
                // User deliberately stopped — don't auto-start. They can tap mic when ready.
                suppressNextAutoStart = false;
                if (Log.isLoggable(TAG, Log.DEBUG))
                    Log.d(TAG, "Auto-start suppressed (age=" + suppressAge + "ms)");
            } else {
                // Different editor, or suppression expired, or never set — auto-start.
                suppressNextAutoStart = false;
                startRecordingSession(prefAutoStop);
            }
        }
    }

    @Override
    public void onFinishInputView(boolean finishingInput) {
        // Record close timestamp for idle-timeout detection
        if (sp != null) {
            sp.edit().putLong(PREF_LAST_CLOSE_TIME,
                    android.os.SystemClock.elapsedRealtime()).apply();
        }

        // Stop transcription, end session, and release the mic.
        if (mWhisper != null) stopTranscription();
        exitListeningMode(true);
        if (mRecorder != null) mRecorder.closeStream(); // release mic when keyboard hides

        // Cancel any pending long-press repeat Runnables.
        if (delInitial    != null) { handler.removeCallbacks(delInitial);    delInitial    = null; }
        if (delFast       != null) { handler.removeCallbacks(delFast);       delFast       = null; }
        if (fwdDelInitial != null) { handler.removeCallbacks(fwdDelInitial); fwdDelInitial = null; }
        if (fwdDelFast    != null) { handler.removeCallbacks(fwdDelFast);    fwdDelFast    = null; }
        if (enterInitial  != null) { handler.removeCallbacks(enterInitial);  enterInitial  = null; }
        if (enterFast     != null) { handler.removeCallbacks(enterFast);     enterFast     = null; }
        // Dismiss any open popups when keyboard hides
        if (punctuationPopup != null) { punctuationPopup.dismiss(); punctuationPopup = null; punctuationVisible = false; }
        if (keyboardPopup    != null) { keyboardPopup.dismiss();    keyboardPopup    = null; }

        // Synchronously reset the visual state NOW, while widget references are
        // still valid. Android does NOT always call onCreateInputView() when the
        // keyboard is reshown — it reuses the existing view. If we null the widget
        // refs here (as we did previously), all subsequent UI updates in
        // onStartInputView / startRecordingSession are silent no-ops, leaving the
        // view frozen showing the old recording/processing state.
        //
        // Session guards in all callbacks already prevent stale updates, so there
        // is no need to null the refs for safety.
        setMicState(MicState.IDLE);
        setCancelEnabled(false);
        resetProgressUi();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // IME view
    // ═══════════════════════════════════════════════════════════════════════════

    @Override
    @SuppressLint("ClickableViewAccessibility")
    public View onCreateInputView() {
        sp = PreferenceManager.getDefaultSharedPreferences(this);
        // loadPrefs() is called in onStartInputView which always follows — skip here

        View view = getLayoutInflater().inflate(R.layout.voice_service, null);

        // ── Bind all widgets ───────────────────────────────────────────────────
        btnRecord             = view.findViewById(R.id.btnRecord);
        btnKeyboard           = view.findViewById(R.id.btnKeyboard);
        btnSend               = view.findViewById(R.id.btnSend);
        btnUndo               = view.findViewById(R.id.btnUndo);
        btnRedo               = view.findViewById(R.id.btnRedo);
        btnDel                = view.findViewById(R.id.btnDel);
        btnEnter              = view.findViewById(R.id.btnEnter);
        btnCancel             = view.findViewById(R.id.btnCancel);
        btnClear              = view.findViewById(R.id.btnClear);
        btnSpace              = view.findViewById(R.id.btnSpace);
        btnDiscard            = view.findViewById(R.id.btnDiscard);
        btnSelectAll          = view.findViewById(R.id.btnSelectAll);
        btnCut                = view.findViewById(R.id.btnCut);
        btnCopy               = view.findViewById(R.id.btnCopy);
        btnPaste              = view.findViewById(R.id.btnPaste);
        btnFullStop           = view.findViewById(R.id.btnFullStop);
        btnForwardDel         = view.findViewById(R.id.btnForwardDel);
        processingBar         = view.findViewById(R.id.processing_bar);
        tvStatus              = view.findViewById(R.id.tv_status);
        layoutKeyboardContent = view.findViewById(R.id.layout_keyboard_content);

        // v12 new buttons and layout containers
        btnBreakAudio   = view.findViewById(R.id.btnBreakAudio);
        btnCapsToggle   = view.findViewById(R.id.btnCapsToggle);
        btnMenuSwap     = view.findViewById(R.id.btnMenuSwap);
        layoutPrimary   = view.findViewById(R.id.layout_primary);
        layoutSecondary = view.findViewById(R.id.layout_secondary);

        // Restore caps icon in case of orientation change/re-inflation
        updateCapsIcon();
        // v12 modifier + D-pad bindings
        btnSecShift = view.findViewById(R.id.btnSecShift);
        btnSecCtrl  = view.findViewById(R.id.btnSecCtrl);
        btnSecAlt   = view.findViewById(R.id.btnSecAlt);
        btnDpad     = view.findViewById(R.id.btnDpad);
        // Restore modifier visual state after re-inflation
        updateModifierUI();

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
                // Stop recording and the continuous loop, but let any in-progress
                // transcription finish and commit. If you want to DISCARD the current
                // audio use the Cancel button instead.
                stopListeningGracefully();
            } else {
                // Deliberate tap — clear any pending suppression
                suppressNextAutoStart = false;
                if (!checkRecordPermission()) return;
                if (!modelReady) {
                    if (tvStatus != null) {
                        tvStatus.setText(getString(R.string.loading_model));
                        tvStatus.setVisibility(View.VISIBLE);
                    }
                    return;
                }
                // If the recognizer is still producing a result from the last session,
                // wait for it to finish before starting a new one. isResultPending()
                // stays true until onSpeechRecognizedResult fires — unlike isInProgress()
                // which clears almost immediately because Recognizer spawns its own thread.
                if (mWhisper != null && mWhisper.isResultPending()) {
                    // Result still incoming — defer the new session until it commits.
                    // Show LISTENING state so the user knows the tap was registered.
                    startAfterWhisper = true;
                    setMicState(MicState.LISTENING);
                    setCancelEnabled(true);
                    if (Log.isLoggable(TAG, Log.DEBUG))
                        Log.d(TAG, "Mic tapped while result pending — deferring start");
                } else {
                    startRecordingSession(prefAutoStop);
                }
            }
        });

        // ── Cancel — hard stop: discard everything, go idle ───────────────────
        btnCancel.setOnClickListener(v -> {
            tap(v);
            // Set suppression BEFORE exitListeningMode so currentEditorKey() still works
            suppressNextAutoStart = true;
            suppressedForEditor   = currentEditorKey();
            suppressSetAt         = android.os.SystemClock.uptimeMillis();
            if (mWhisper != null) stopTranscription();
            exitListeningMode(true);
        });

        // ── Discard Audio — discard current audio but stay listening ──────────
        btnDiscard.setOnClickListener(v -> {
            tap(v);
            discardAudioAndContinue();
        });

        // ── Keyboard — tap: switch keyboard  |  long-press: options popup ─────
        // If popup is showing when tapped, the tap closes it AND switches keyboard.
        btnKeyboard.setOnClickListener(v -> {
            tap(v);
            if (keyboardPopup != null && keyboardPopup.isShowing()) {
                keyboardPopup.dismiss(); // dismiss first — listener will null the field
            }
            if (mWhisper != null) stopTranscription();
            exitListeningMode(false);
            switchToPreviousInputMethod();
        });
        btnKeyboard.setOnLongClickListener(v -> {
            showKeyboardOptionsPopup(v);
            return true;
        });

        // ── Select All ────────────────────────────────────────────────────────
        btnSelectAll.setOnClickListener(v -> {
            tap(v);
            if (getCurrentInputConnection() != null)
                sendCtrlKey(KeyEvent.KEYCODE_A, false);
        });

        // ── Delete Unselected ─────────────────────────────────────────────────
        // Keeps only the selected text; deletes everything else.
        // Uses getTextBeforeCursor/getTextAfterCursor — the Android contract
        // guarantees these return text OUTSIDE the selection, so deleteSurrounding
        // Text(before.length(), after.length()) can never touch the selected region.
        btnClear.setOnClickListener(v -> {
            tap(v);
            if (getCurrentInputConnection() == null) return;
            CharSequence before = getCurrentInputConnection()
                    .getTextBeforeCursor(10_000, 0);
            CharSequence after  = getCurrentInputConnection()
                    .getTextAfterCursor(10_000, 0);
            if (before == null) before = "";
            if (after  == null) after  = "";
            getCurrentInputConnection()
                    .deleteSurroundingText(before.length(), after.length());
        });

        // ── Cut | Copy | Paste ────────────────────────────────────────────────
        btnCut.setOnClickListener(v   -> { tap(v); sendCtrlKey(KeyEvent.KEYCODE_X, false); });
        btnCopy.setOnClickListener(v  -> { tap(v); sendCtrlKey(KeyEvent.KEYCODE_C, false); });
        btnPaste.setOnClickListener(v -> { tap(v); sendCtrlKey(KeyEvent.KEYCODE_V, false); });

        // ── Undo / Redo ────────────────────────────────────────────────────────
        // Short press: standard Ctrl+Z. Long press: remove last transcription.
        btnUndo.setOnClickListener(v -> { tap(v); sendCtrlKey(KeyEvent.KEYCODE_Z, false); });
        btnUndo.setOnLongClickListener(v -> {
            tap(v);
            undoLastTranscription();
            return true;
        });
        btnRedo.setOnClickListener(v -> { tap(v); sendCtrlKey(KeyEvent.KEYCODE_Z, true);  });

        // ── Break Audio ────────────────────────────────────────────────
        // Stops the current audio segment so Whisper transcribes it immediately,
        // then restarts listening without any gap — like a manual VAD trigger.
        // Enabled only while a session is active (same enable pattern as btnDiscard).
        if (btnBreakAudio != null) {
            btnBreakAudio.setOnClickListener(v -> {
                tap(v);
                breakAudioAndContinue();
            });
        }

        // ── Caps Toggle ────────────────────────────────────────────────
        // Cycles: auto (hollow shift) -> force upper (solid shift) -> force lower (down shift)
        if (btnCapsToggle != null) {
            btnCapsToggle.setOnClickListener(v -> {
                tap(v);
                capsOverride = (capsOverride + 1) % 3;
                updateCapsIcon();
            });
        }

        // ── Menu / Layout Swap ──────────────────────────────────────────
        if (btnMenuSwap != null) {
            btnMenuSwap.setOnClickListener(v -> {
                tap(v);
                toggleSecondaryLayout();
            });
        }

        // ── Secondary layout buttons ───────────────────────────────────
        // KB button in secondary (same as primary keyboard button)
        View btnKbSec = view.findViewById(R.id.btnKeyboardSec);
        if (btnKbSec != null) btnKbSec.setOnClickListener(v -> {
            tap(v);
            switchToPreviousInputMethod();
        });
        // Streaming toggle in secondary layout
        // Language display (tap to cycle through common languages)
        // Secondary layout: menu swap button (left col bottom)
        View btnSecMe = view.findViewById(R.id.btnSecMenu);
        if (btnSecMe != null) btnSecMe.setOnClickListener(v -> { tap(v); toggleSecondaryLayout(); });
        // Secondary layout: right col text-edit buttons
        View btnScCut = view.findViewById(R.id.btnSecCut);
        View btnScCpy = view.findViewById(R.id.btnSecCopy);
        View btnScPst = view.findViewById(R.id.btnSecPaste);
        View btnScDel = view.findViewById(R.id.btnSecDel);
        View btnScEnt = view.findViewById(R.id.btnSecEnter);
        if (btnScCut != null) btnScCut.setOnClickListener(v ->  { tap(v); sendCtrlKey(KeyEvent.KEYCODE_X, false); });
        if (btnScCpy != null) btnScCpy.setOnClickListener(v ->  { tap(v); sendCtrlKey(KeyEvent.KEYCODE_C, false); });
        if (btnScPst != null) btnScPst.setOnClickListener(v ->  { tap(v); sendCtrlKey(KeyEvent.KEYCODE_V, false); });
        if (btnScDel != null) btnScDel.setOnTouchListener((v, ev) -> {
            // Backspace with long-press repeat (reuses doDelete helper)
            switch (ev.getAction()) {
                case android.view.MotionEvent.ACTION_DOWN:
                    tap(v); doDelete();
                    Runnable secDelFast = new Runnable() {
                        @Override public void run() { doDelete(); handler.postDelayed(this, 50); }
                    };
                    handler.postDelayed(secDelFast, 400);
                    v.setTag(secDelFast);
                    break;
                case android.view.MotionEvent.ACTION_UP:
                case android.view.MotionEvent.ACTION_CANCEL:
                    Object tag = v.getTag();
                    if (tag instanceof Runnable) { handler.removeCallbacks((Runnable) tag); v.setTag(null); }
                    break;
            }
            return true;
        });
        if (btnScEnt != null) btnScEnt.setOnClickListener(v -> {
            tap(v);
            android.view.inputmethod.InputConnection ic = getCurrentInputConnection();
            if (ic == null) return;
            long t = android.os.SystemClock.uptimeMillis();
            ic.sendKeyEvent(new KeyEvent(t, t, KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_ENTER, 0));
            ic.sendKeyEvent(new KeyEvent(t, t, KeyEvent.ACTION_UP,   KeyEvent.KEYCODE_ENTER, 0));
        });

        // ── Modifier keys (Shift / Ctrl / Alt) ──────────────────────────────
        if (btnSecShift != null) btnSecShift.setOnClickListener(v -> {
            tap(v); shiftActive = !shiftActive; updateModifierUI();
        });
        if (btnSecCtrl != null) btnSecCtrl.setOnClickListener(v -> {
            tap(v); ctrlActive = !ctrlActive; updateModifierUI();
        });
        if (btnSecAlt != null) btnSecAlt.setOnClickListener(v -> {
            tap(v); altActive = !altActive; updateModifierUI();
        });

        // ── D-pad — drag from center + tap zones at edges ──────────────────
        if (btnDpad != null) {
            final android.util.DisplayMetrics dmDpad = getResources().getDisplayMetrics();
            final float H_SENS = 8f * dmDpad.density;   // px per char, horizontal drag
            final float V_SENS = 28f * dmDpad.density;  // px per line, vertical drag
            final float DRAG_THRESHOLD = 8f * dmDpad.density;
            final float[] dpadDown    = {0f, 0f};
            final int[]   dpadCurSt   = {-1};   // cursor pos at touch-down
            final int[]   dpadSelAnch = {-1};   // selection anchor for shift+drag
            final int[]   dpadTxtLen  = {0};
            final int[]   dpadLastH   = {-1};   // last cursor pos (haptic gate)
            final int[]   dpadLastV   = {0};    // last vertical step count
            final boolean[] dpadDrag  = {false};
            final boolean[] dpadMoved = {false};

            btnDpad.setOnTouchListener((v, ev) -> {
                float x = ev.getX(), y = ev.getY();
                float vW = v.getWidth(), vH = v.getHeight();
                switch (ev.getAction()) {
                    case MotionEvent.ACTION_DOWN: {
                        dpadDown[0] = x; dpadDown[1] = y;
                        dpadDrag[0] = false; dpadMoved[0] = false;
                        dpadLastV[0] = 0;
                        android.view.inputmethod.InputConnection ic = getCurrentInputConnection();
                        if (ic != null) {
                            android.view.inputmethod.ExtractedText et =
                                    ic.getExtractedText(new android.view.inputmethod.ExtractedTextRequest(), 0);
                            if (et != null && et.text != null) {
                                dpadCurSt[0]   = et.selectionStart;
                                dpadSelAnch[0] = et.selectionStart;
                                dpadTxtLen[0]  = et.text.length();
                                dpadLastH[0]   = et.selectionStart;
                            } else { dpadCurSt[0] = -1; }
                        }
                        break;
                    }
                    case MotionEvent.ACTION_MOVE: {
                        float dx = x - dpadDown[0], dy = y - dpadDown[1];
                        float dist = (float) Math.sqrt(dx * dx + dy * dy);
                        if (!dpadDrag[0] && dist > DRAG_THRESHOLD) {
                            dpadDrag[0] = true; dpadMoved[0] = true;
                        }
                        if (!dpadDrag[0]) break;
                        android.view.inputmethod.InputConnection ic = getCurrentInputConnection();
                        // ── Horizontal axis: absolute setSelection (no direction lock) ──
                        // Always computed from total dx so direction changes freely.
                        if (dpadCurSt[0] >= 0 && ic != null) {
                            int charDelta = (int)(dx / H_SENS);
                            int target = Math.max(0, Math.min(dpadTxtLen[0], dpadCurSt[0] + charDelta));
                            // Shift held: keep selection anchor, move active end.
                            // No shift: collapse selection to cursor position.
                            if (shiftActive) ic.setSelection(dpadSelAnch[0], target);
                            else             ic.setSelection(target, target);
                            if (target != dpadLastH[0]) {
                                v.performHapticFeedback(HapticFeedbackConstants.CLOCK_TICK,
                                        HapticFeedbackConstants.FLAG_IGNORE_VIEW_SETTING);
                                if (mRecorder != null) mRecorder.suppressVadTrigger(250);
                                dpadLastH[0] = target;
                            }
                        }
                        // ── Vertical axis: incremental DPAD_UP/DOWN (no direction lock) ──
                        // Computed from total dy so reversing direction works naturally.
                        {
                            int steps = (int)(dy / V_SENS);
                            int diff  = steps - dpadLastV[0];
                            if (diff != 0 && ic != null) {
                                int kc = diff > 0 ? KeyEvent.KEYCODE_DPAD_DOWN : KeyEvent.KEYCODE_DPAD_UP;
                                int meta = 0;
                                if (shiftActive) meta |= KeyEvent.META_SHIFT_ON;
                                if (ctrlActive)  meta |= KeyEvent.META_CTRL_ON;
                                for (int i = 0; i < Math.abs(diff); i++) {
                                    long t = android.os.SystemClock.uptimeMillis();
                                    ic.sendKeyEvent(new KeyEvent(t, t, KeyEvent.ACTION_DOWN, kc, 0, meta));
                                    ic.sendKeyEvent(new KeyEvent(t, t, KeyEvent.ACTION_UP,   kc, 0, meta));
                                }
                                dpadLastV[0] = steps;
                                v.performHapticFeedback(HapticFeedbackConstants.CLOCK_TICK,
                                        HapticFeedbackConstants.FLAG_IGNORE_VIEW_SETTING);
                            }
                        }
                        break;
                    }
                    case MotionEvent.ACTION_UP: {
                        if (!dpadMoved[0]) {
                            // Tap — determine zone from position within view
                            float relX = vW > 0 ? x / vW : 0.5f;
                            float relY = vH > 0 ? y / vH : 0.5f;
                            int kc = 0;
                            if      (relY < 0.28f) kc = KeyEvent.KEYCODE_DPAD_UP;
                            else if (relY > 0.72f) kc = KeyEvent.KEYCODE_DPAD_DOWN;
                            else if (relX < 0.28f) kc = KeyEvent.KEYCODE_DPAD_LEFT;
                            else if (relX > 0.72f) kc = KeyEvent.KEYCODE_DPAD_RIGHT;
                            if (kc != 0) {
                                tap(v);
                                android.view.inputmethod.InputConnection ic = getCurrentInputConnection();
                                if (ic != null) {
                                    int meta = 0;
                                    if (shiftActive) meta |= KeyEvent.META_SHIFT_ON;
                                    if (ctrlActive)  meta |= KeyEvent.META_CTRL_ON;
                                    if (altActive)   meta |= KeyEvent.META_ALT_ON;
                                    long t = android.os.SystemClock.uptimeMillis();
                                    ic.sendKeyEvent(new KeyEvent(t, t, KeyEvent.ACTION_DOWN, kc, 0, meta));
                                    ic.sendKeyEvent(new KeyEvent(t, t, KeyEvent.ACTION_UP,   kc, 0, meta));
                                }
                            }
                        }
                        break;
                    }
                }
                return true;
            });
        }

        // Secondary layout — Tab and Esc buttons
        View btnSecTab = view.findViewById(R.id.btnSecTab);
        View btnSecEsc = view.findViewById(R.id.btnSecEsc);
        View btnSecHome = view.findViewById(R.id.btnSecHome);
        View btnSecEnd  = view.findViewById(R.id.btnSecEnd);
        View btnSecPgUp = view.findViewById(R.id.btnSecPgUp);
        View btnSecPgDn = view.findViewById(R.id.btnSecPgDn);
        if (btnSecTab  != null) btnSecTab.setOnClickListener(v -> {
            tap(v);
            android.view.inputmethod.InputConnection ic = getCurrentInputConnection();
            if (ic != null) {
                long t = android.os.SystemClock.uptimeMillis();
                ic.sendKeyEvent(new KeyEvent(t, t, KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_TAB, 0));
                ic.sendKeyEvent(new KeyEvent(t, t, KeyEvent.ACTION_UP,   KeyEvent.KEYCODE_TAB, 0));
            }
        });
        if (btnSecEsc  != null) btnSecEsc.setOnClickListener(v -> {
            tap(v);
            android.view.inputmethod.InputConnection ic = getCurrentInputConnection();
            if (ic != null) {
                long t = android.os.SystemClock.uptimeMillis();
                ic.sendKeyEvent(new KeyEvent(t, t, KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_ESCAPE, 0));
                ic.sendKeyEvent(new KeyEvent(t, t, KeyEvent.ACTION_UP,   KeyEvent.KEYCODE_ESCAPE, 0));
            }
        });
        if (btnSecHome != null) btnSecHome.setOnClickListener(v -> {
            tap(v);
            android.view.inputmethod.InputConnection ic = getCurrentInputConnection();
            if (ic != null) {
                int meta = shiftActive ? KeyEvent.META_SHIFT_ON : 0;
                long t = android.os.SystemClock.uptimeMillis();
                ic.sendKeyEvent(new KeyEvent(t, t, KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_MOVE_HOME, 0, meta));
                ic.sendKeyEvent(new KeyEvent(t, t, KeyEvent.ACTION_UP,   KeyEvent.KEYCODE_MOVE_HOME, 0, meta));
            }
        });
        if (btnSecEnd  != null) btnSecEnd.setOnClickListener(v -> {
            tap(v);
            android.view.inputmethod.InputConnection ic = getCurrentInputConnection();
            if (ic != null) {
                int meta = shiftActive ? KeyEvent.META_SHIFT_ON : 0;
                long t = android.os.SystemClock.uptimeMillis();
                ic.sendKeyEvent(new KeyEvent(t, t, KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_MOVE_END, 0, meta));
                ic.sendKeyEvent(new KeyEvent(t, t, KeyEvent.ACTION_UP,   KeyEvent.KEYCODE_MOVE_END, 0, meta));
            }
        });
        if (btnSecPgUp != null) btnSecPgUp.setOnClickListener(v -> {
            tap(v);
            android.view.inputmethod.InputConnection ic = getCurrentInputConnection();
            if (ic != null) {
                int meta = shiftActive ? KeyEvent.META_SHIFT_ON : 0;
                long t = android.os.SystemClock.uptimeMillis();
                ic.sendKeyEvent(new KeyEvent(t, t, KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_PAGE_UP, 0, meta));
                ic.sendKeyEvent(new KeyEvent(t, t, KeyEvent.ACTION_UP,   KeyEvent.KEYCODE_PAGE_UP, 0, meta));
            }
        });
        if (btnSecPgDn != null) btnSecPgDn.setOnClickListener(v -> {
            tap(v);
            android.view.inputmethod.InputConnection ic = getCurrentInputConnection();
            if (ic != null) {
                int meta = shiftActive ? KeyEvent.META_SHIFT_ON : 0;
                long t = android.os.SystemClock.uptimeMillis();
                ic.sendKeyEvent(new KeyEvent(t, t, KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_PAGE_DOWN, 0, meta));
                ic.sendKeyEvent(new KeyEvent(t, t, KeyEvent.ACTION_UP,   KeyEvent.KEYCODE_PAGE_DOWN, 0, meta));
            }
        });

        // ── Full Stop / Punctuation toggle — tap toggles popup open/closed ──────
        // Does NOT insert a period. Press again to close popup.
        // punctuationVisible guards against the race where the popup's
        // outside-touch dismiss fires before this click event, causing the
        // click to reopen the popup that was just closed.
        btnFullStop.setOnClickListener(v -> {
            tap(v);
            if (punctuationVisible) {
                // Popup is explicitly open — user tapped the button to close it
                if (punctuationPopup != null) punctuationPopup.dismiss();
            } else if (android.os.SystemClock.uptimeMillis() - punctuationDismissedAt > 400) {
                // Only open if the popup was not *just* dismissed.
                // When the popup is open and the user taps the button, the popup's
                // outside-touch handler fires during ACTION_DOWN (sets dismiss time,
                // sets punctuationVisible=false), then the click fires during ACTION_UP.
                // The 400ms window ensures that click does not reopen the popup.
                showPunctuationPopup(v);
            }
        });

        // ── Forward Delete with long-press repeat (uses own fwdDel runnables) ──
        btnForwardDel.setOnTouchListener((v, event) -> {
            switch (event.getAction()) {
                case MotionEvent.ACTION_DOWN:
                    tap(v);
                    doForwardDelete();
                    fwdDelInitial = () -> {
                        doForwardDelete();
                        fwdDelFast = new Runnable() {
                            @Override public void run() {
                                doForwardDelete();
                                handler.postDelayed(this, 50);
                            }
                        };
                        handler.postDelayed(fwdDelFast, 50);
                    };
                    handler.postDelayed(fwdDelInitial, 400);
                    break;
                case MotionEvent.ACTION_UP:
                case MotionEvent.ACTION_CANCEL:
                    if (fwdDelInitial != null) { handler.removeCallbacks(fwdDelInitial); fwdDelInitial = null; }
                    if (fwdDelFast    != null) { handler.removeCallbacks(fwdDelFast);    fwdDelFast    = null; }
                    break;
                default: break;
            }
            return true;
        });

        // ── Backspace with long-press repeat ───────────────────────────────────
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

        // ── Space bar — tap inserts space; drag left/right moves cursor ─────────
        // Uses the FlorisBoard absolute-position approach: on ACTION_DOWN we snapshot
        // the cursor position and text length via getExtractedText(); on every
        // ACTION_MOVE we compute targetPos = snapshotCursor + (totalPixelDelta / sensitivity),
        // clamp to [0, textLength], and call setSelection() once.  This is immune to
        // fast swipes because there is no incremental accumulation — the boundary clamp
        // is always applied to the full gesture delta, not to each tiny step.
        // No KEYCODE_DPAD events are sent, so the "focus escapes the field" bug cannot
        // occur regardless of speed.
        final float SWIPE_SENSITIVITY_PX = 8f * getResources().getDisplayMetrics().density;
        final float[] spaceDownX       = {0f};
        final int[]   spaceCursorStart = {-1};
        final int[]   spaceTextLen     = {0};
        final int[]   spaceLastPos     = {-1}; // last char position, for haptic gating
        final boolean[] spaceMoved     = {false};
        btnSpace.setOnTouchListener((v, event) -> {
            switch (event.getAction()) {
                case MotionEvent.ACTION_DOWN: {
                    spaceDownX[0]   = event.getX();
                    spaceMoved[0]   = false;
                    android.view.inputmethod.InputConnection ic = getCurrentInputConnection();
                    if (ic != null) {
                        ExtractedText et = ic.getExtractedText(new ExtractedTextRequest(), 0);
                        if (et != null && et.text != null) {
                            spaceCursorStart[0] = et.selectionStart;
                            spaceTextLen[0]     = et.text.length();
                            spaceLastPos[0]     = et.selectionStart;
                        } else {
                            spaceCursorStart[0] = -1; // IC unavailable — fall back gracefully
                        }
                    } else {
                        spaceCursorStart[0] = -1;
                    }
                    break;
                }
                case MotionEvent.ACTION_MOVE: {
                    if (spaceCursorStart[0] < 0) break; // no snapshot — skip
                    float totalDeltaPx = event.getX() - spaceDownX[0];
                    int charDelta = (int)(totalDeltaPx / SWIPE_SENSITIVITY_PX);
                    int targetPos = spaceCursorStart[0] + charDelta;
                    targetPos = Math.max(0, Math.min(spaceTextLen[0], targetPos));
                    // Move cursor atomically — no DPAD events, no focus-escape risk.
                    android.view.inputmethod.InputConnection ic = getCurrentInputConnection();
                    if (ic != null) {
                        ic.setSelection(targetPos, targetPos);
                    }
                    // Haptic fires once per character boundary crossed (not per pixel event).
                    if (targetPos != spaceLastPos[0]) {
                        spaceMoved[0] = true;
                        v.performHapticFeedback(HapticFeedbackConstants.CLOCK_TICK,
                                HapticFeedbackConstants.FLAG_IGNORE_VIEW_SETTING);
                        // Suppress VAD for 250 ms — CLOCK_TICK vibrations couple into mic.
                        if (mRecorder != null) mRecorder.suppressVadTrigger(250);
                        spaceLastPos[0] = targetPos;
                    }
                    break;
                }
                case MotionEvent.ACTION_UP:
                    if (!spaceMoved[0]) {
                        // No swipe occurred — treat as a tap
                        tap(v);
                        if (getCurrentInputConnection() != null)
                            getCurrentInputConnection().commitText(" ", 1);
                    }
                    break;
                default: break;
            }
            return true;
        });

        // ── Send (IME action) ──────────────────────────────────────────────────
        btnSend.setOnClickListener(v -> {
            tap(v);
            android.view.inputmethod.InputConnection ic = getCurrentInputConnection();
            if (ic == null) return;
            EditorInfo ei = getCurrentInputEditorInfo();
            int actionId = ei != null ? ei.actionId : 0;
            if (actionId != 0
                    && actionId != EditorInfo.IME_ACTION_NONE
                    && actionId != EditorInfo.IME_ACTION_UNSPECIFIED) {
                // App declared a specific action (Search, Done, Go, etc.) — use it.
                ic.performEditorAction(actionId);
            } else {
                // No specific action declared (Signal, many chat apps).
                // These apps handle KEYCODE_ENTER for send — performEditorAction()
                // with any action code is ignored and causes the keyboard to hide.
                long t = android.os.SystemClock.uptimeMillis();
                ic.sendKeyEvent(new KeyEvent(t, t, KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_ENTER, 0));
                ic.sendKeyEvent(new KeyEvent(t, t, KeyEvent.ACTION_UP,   KeyEvent.KEYCODE_ENTER, 0));
            }
        });

        // ── Enter with long-press repeat ───────────────────────────────────────
        btnEnter.setOnTouchListener((v, event) -> {
            switch (event.getAction()) {
                case MotionEvent.ACTION_DOWN:
                    tap(v);
                    doEnter();
                    enterInitial = () -> {
                        doEnter();
                        enterFast = new Runnable() {
                            @Override public void run() {
                                doEnter();
                                handler.postDelayed(this, 50);
                            }
                        };
                        handler.postDelayed(enterFast, 50);
                    };
                    handler.postDelayed(enterInitial, 400);
                    break;
                case MotionEvent.ACTION_UP:
                case MotionEvent.ACTION_CANCEL:
                    if (enterInitial != null) { handler.removeCallbacks(enterInitial); enterInitial = null; }
                    if (enterFast    != null) { handler.removeCallbacks(enterFast);    enterFast    = null; }
                    break;
                default: break;
            }
            return true;
        });

        return view;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Recorder listener  (can be re-set safely on view re-inflation)
    // ═══════════════════════════════════════════════════════════════════════════

    private void setRecorderListener() {
        mRecorder.setListener(message -> {
            // Called on Recorder worker thread — all UI via handler.post().
            // AudioRecord now runs CONTINUOUSLY for the session — MSG_RECORDING_DONE
            // means a speech segment was extracted and buffered, NOT that AudioRecord
            // stopped. Do NOT restart the recorder in response to MSG_RECORDING_DONE.
            final int sid = activeRecordingSessionId;

            if (message.equals(Recorder.MSG_RECORDING)) {
                // VAD confirmed speech onset — show RECORDING state
                isRecording = true;
                handler.post(() -> {
                    if (sessionGen != sid) return;
                    setMicState(MicState.RECORDING);
                });

            } else if (message.equals(Recorder.MSG_RECORDING_DONE)) {
                // A speech segment has been extracted and written to RecordBuffer.
                // AudioRecord is still running — return to LISTENING state and
                // queue the segment for transcription.
                isRecording = false;
                if (prefHapticRecording) HapticFeedback.vibrate(this);

                if (!cancelRequested) {
                    final byte[] capturedAudio = RecordBuffer.getOutputBuffer();
                    handler.post(() -> {
                        if (sessionGen != sid || cancelRequested) return;
                        // Queue and transcribe — AudioRecord keeps running, no restart needed
                        audioQueue.offer(capturedAudio);
                        processNextQueued();
                        // Return to LISTENING so the user knows we're ready for the next sentence
                        setMicState(MicState.LISTENING);
                    });
                } else {
                    isListening    = false;
                    continuousMode = false;
                    handler.post(() -> {
                        if (sessionGen != sid) return;
                        audioQueue.clear();
                        setMicState(MicState.IDLE);
                        setCancelEnabled(false);
                        resetProgressUi();
                    });
                }

            } else if (message.equals(Recorder.MSG_RECORDING_ERROR)) {
                isRecording    = false;
                isListening    = false;
                continuousMode = false;
                if (prefHapticRecording) HapticFeedback.vibrate(this);
                handler.post(() -> {
                    if (sessionGen != sid) return;
                    setMicState(MicState.IDLE);
                    setCancelEnabled(false);
                    if (tvStatus != null) {
                        tvStatus.setText(getString(R.string.error_no_input));
                        tvStatus.setVisibility(View.VISIBLE);
                    }
                    resetProgressUi();
                });

            } else if (message.equals(Recorder.MSG_VAD_TIMEOUT)) {
                // 30 seconds of silence with no speech. AudioRecord is still running.
                // In single-take (non-continuous) mode: go idle.
                // In streaming mode: Recorder already reset its silence counter internally,
                // no restart needed — just stay in LISTENING state.
                isRecording = false;
                final boolean wasStreaming = continuousMode;
                handler.post(() -> {
                    if (sessionGen != sid) return;
                    if (!wasStreaming || !isListening || cancelRequested) {
                        isListening    = false;
                        continuousMode = false;
                        setMicState(MicState.IDLE);
                        setCancelEnabled(false);
                        resetProgressUi();
                    }
                    // else: streaming mode — stay LISTENING, AudioRecord continues
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
                // Capture activeWhisperSessionId — stamped at the moment
                // mWhisper.start() was called for THIS inference job, not at
                // callback-fire time. This means:
                //   • If cancel was pressed then record was tapped again
                //     (startRecordingSession increments sessionGen, resets
                //     cancelRequested=false, but does NOT change
                //     activeWhisperSessionId until the next transcription starts),
                //     sid != sessionGen and we discard the old result.
                //   • If the keyboard was closed mid-inference, same protection.
                final int sid = activeWhisperSessionId;

                // Primary guard: session mismatch — this result is stale, discard it.
                if (sessionGen != sid) {
                    // The old inference is now done so mWhisper.isInProgress() just
                    // became false.  Post to the main thread to drain any audio the
                    // new session queued while Whisper was busy with this old job.
                    // Do NOT touch isListening, mic state, or cancelRequested here —
                    // the active session owns those fields.
                    // Only drain the queue if Whisper is not already running —
                    // the new session may have already called processNextQueued() and
                    // started Whisper. Calling it again would be a no-op (isInProgress
                    // guard inside) but it's cleaner to check here.
                    handler.post(() -> {
                        if (mWhisper != null && !mWhisper.isInProgress() && !mWhisper.isResultPending()) {
                            processNextQueued();
                        }
                        // Consume any pending deferred session start that was queued
                        // while we were processing this (now stale) segment.
                        if (startAfterWhisper && !cancelRequested) {
                            startAfterWhisper = false;
                            startRecordingSession(prefAutoStop);
                        }
                    });
                    return;
                }

                // Secondary guard: discard (keep session alive, just drop this result)
                if (discardRequested) {
                    discardRequested = false;
                    handler.post(() -> resetProgressUi());
                    return;
                }

                // Tertiary guard: cancelled
                if (cancelRequested) {
                    handler.post(() -> {
                        if (activeWhisperSessionId != sid) return;
                        audioQueue.clear();
                        isListening = false;
                        setMicState(MicState.IDLE);
                        setCancelEnabled(false);
                        resetProgressUi();
                    });
                    return;
                }

                handler.post(() -> {
                    if (activeWhisperSessionId == sid) resetProgressUi();
                });

                // Chinese script conversion
                String result = whisperResult.getResult();
                if ("zh".equals(whisperResult.getLanguage())) {
                    boolean simple = sp != null && sp.getBoolean("simpleChinese", false);
                    result = simple ? ZhConverterUtil.toSimple(result)
                                    : ZhConverterUtil.toTraditional(result);
                }

                // Lock session language after first auto-detection.
                // whisperResult.getLanguage() contains the language code Whisper
                // identified from the audio tokens (not the user's "auto" setting).
                // Storing it lets subsequent segments skip language detection (~15%
                // of inference cost) and prevents mid-session language flipping.
                final String detectedLang = whisperResult.getLanguage();
                if (!detectedLang.isEmpty() && !detectedLang.equals("??")
                        && !detectedLang.equals("auto") && sessionLanguage.isEmpty()) {
                    sessionLanguage = detectedLang;
                    Log.d(TAG, "Session language locked to: " + sessionLanguage);
                }

                // Undo Recognizer's forced capitalisation if user prefers raw case.
                // Recognizer.correctText() always uppercases the first character.
                // Apply capitalisation override (v12 caps toggle).
                // capsOverride 0 = auto: honour prefAutoCapitalise (Whisper always
                //   capitalises first char, so lower it if the pref says not to).
                // capsOverride 1 = force upper: ensure first char is uppercase.
                // capsOverride 2 = force lower: always lowercase the first char.
                // applyCapitalization handles every sentence in the result block,
                // not just the first character — Whisper may return multiple sentences.
                result = applyCapitalization(result);

                // Commit text — guarded by session check on this thread
                final String trimmed = result.trim();
                // Whisper annotates non-speech audio with parenthetical/bracketed
                // descriptions: (air whooshing), [blank audio], (upbeat music), etc.
                // These are model artefacts — discard any result that is entirely
                // one or more parenthetical/bracketed tokens with no real words.
                if (!trimmed.isEmpty() && !isNonSpeechAnnotation(trimmed)
                        && getCurrentInputConnection() != null) {
                    String committed = trimmed + " ";
                    getCurrentInputConnection().commitText(committed, 1);
                    // Record for word-level undo (long-press Undo button)
                    handler.post(() -> {
                        if (transcriptionHistory.size() >= MAX_UNDO_HISTORY)
                            transcriptionHistory.pollFirst();
                        transcriptionHistory.addLast(committed);
                    });
                }

                // Auto-send
                if (prefAutoSend && getCurrentInputConnection() != null) {
                    EditorInfo ei = getCurrentInputEditorInfo();
                    int autoActionId = ei != null ? ei.actionId : 0;
                    if (autoActionId != 0
                            && autoActionId != EditorInfo.IME_ACTION_NONE
                            && autoActionId != EditorInfo.IME_ACTION_UNSPECIFIED) {
                        getCurrentInputConnection().performEditorAction(autoActionId);
                    } else {
                        long t = android.os.SystemClock.uptimeMillis();
                        getCurrentInputConnection().sendKeyEvent(
                                new KeyEvent(t, t, KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_ENTER, 0));
                        getCurrentInputConnection().sendKeyEvent(
                                new KeyEvent(t, t, KeyEvent.ACTION_UP,   KeyEvent.KEYCODE_ENTER, 0));
                    }
                }

                // Auto-switch (single-take sessions only)
                if (prefAutoSwitch && !continuousMode) {
                    isListening = false;
                    handler.post(() -> {
                        if (activeWhisperSessionId != sid) return;
                        audioQueue.clear();
                        setMicState(MicState.IDLE);
                        setCancelEnabled(false);
                        switchToPreviousInputMethod();
                    });
                    return;
                }

                // Post-result state update on main thread
                handler.post(() -> {
                    if (activeWhisperSessionId != sid || cancelRequested) {
                        audioQueue.clear();
                        isListening = false;
                        setMicState(MicState.IDLE);
                        setCancelEnabled(false);
                        return;
                    }
                    if (!continuousMode) {
                        // Drain any remaining queued segments (e.g. pipelined audio
                        // that arrived while we were transcribing this one) before
                        // going IDLE. processNextQueued() starts Whisper if queued;
                        // it will return here when done and reach the else branch.
                        processNextQueued();
                        if (!mWhisper.isInProgress()) {
                            // Queue was empty — nothing started.
                            if (startAfterWhisper) {
                                // User tapped mic while we were processing — start now.
                                startAfterWhisper = false;
                                startRecordingSession(prefAutoStop);
                            } else {
                                isListening = false;
                                setMicState(MicState.IDLE);
                                setCancelEnabled(false);
                            }
                        }
                        // else: Whisper started on next queued segment; when it
                        // finishes onResultReceived fires again, hits this same
                        // branch, drains again until the queue is empty.
                    } else {
                        // Continuous mode — drain next queued segment.
                        processNextQueued();
                        // If the user tapped the mic while we were processing,
                        // startAfterWhisper is set. In continuous mode the session
                        // is already running so just clear the flag — the mic is
                        // still active and will pick up the next sentence naturally.
                        if (startAfterWhisper) {
                            startAfterWhisper = false;
                            // In continuous mode the recorder is still running, so we
                            // don't need to start a new session — just clear the flag.
                        }
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
        cancelRequested = false;
        isListening     = true;
        continuousMode  = useVad;
        final int sid   = ++sessionGen;

        handler.post(() -> {
            if (sessionGen != sid) return;
            setMicState(MicState.LISTENING);
            setCancelEnabled(true);
            if (tvStatus != null) tvStatus.setVisibility(View.GONE);
            resetProgressUi();
        });

        // start() is fire-and-forget — safe to call even if a session is active.
        // The Recorder's generation counter handles rapid stop→start by flushing
        // the previous session's state cleanly before the new one begins.
        pendingStartSessionId = 0;
        activeRecordingSessionId = sid;
        if (useVad) mRecorder.initVad();
        mRecorder.start();
    }

    /**
     * Soft stop — stops recording and the continuous loop but does NOT set
     * cancelRequested.  Any transcription currently in progress finishes
     * normally and its result is committed to the text field.  Goes IDLE
     * once the queue is fully drained.
     *
     * Used by the record button when tapped while active.
     * The cancel button uses exitListeningMode() which discards everything.
     */
    // ═══════════════════════════════════════════════════════════════════════════
    // New action methods
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Discard current audio and keep listening.
     * Like cancel, but stays in the session — the session ID does not change and
     * cancelRequested is never set, so a new recording starts seamlessly.
     */
    private void discardAudioAndContinue() {
        if (!isListening) return;
        audioQueue.clear();
        discardRequested = true;    // suppress the in-flight Whisper result
        if (mWhisper != null) mWhisper.stop();
        resetProgressUi();

        // stop() is instant — stop current session and immediately start a new one.
        // The generation counter in Recorder handles the flush of the discarded audio.
        if (mRecorder.isSessionActive()) mRecorder.stop();
        activeRecordingSessionId = sessionGen;
        if (continuousMode) mRecorder.initVad();
        mRecorder.start();
        setMicState(MicState.LISTENING);
    }

    /**
     * Show the keyboard long-press options popup above the anchor view.
     * Three options: [Toggle Streaming] [⌨ Keyboard] [⚙ Settings]
     */
    private void showKeyboardOptionsPopup(View anchor) {
        if (keyboardPopup != null && keyboardPopup.isShowing()) {
            keyboardPopup.dismiss();
            return;
        }
        android.view.LayoutInflater inflater = android.view.LayoutInflater.from(this);
        android.view.View popupView = inflater.inflate(R.layout.popup_keyboard_options, null);

        android.widget.TextView tvStreaming = popupView.findViewById(R.id.popup_streaming);
        tvStreaming.setText("Streaming\n" + (prefAutoStop ? "[ON]" : "[OFF]"));

        // focusable=false so that tapping buttons behind/around the popup still works.
        // The keyboard button press will dismiss the popup via dismiss() call AND
        // execute the switch. setBackgroundDrawable is needed for outside-touch to work.
        PopupWindow popup = new PopupWindow(popupView,
                android.view.ViewGroup.LayoutParams.WRAP_CONTENT,
                android.view.ViewGroup.LayoutParams.WRAP_CONTENT, false);
        popup.setElevation(8f * getResources().getDisplayMetrics().density);
        popup.setOutsideTouchable(true);
        popup.setBackgroundDrawable(TRANSPARENT_DRAWABLE);
        popup.setOnDismissListener(() -> keyboardPopup = null);
        keyboardPopup = popup;

        // Center popup over anchor
        popupView.measure(android.view.View.MeasureSpec.UNSPECIFIED,
                          android.view.View.MeasureSpec.UNSPECIFIED);
        int popupW = popupView.getMeasuredWidth();
        int popupH = popupView.getMeasuredHeight();
        int xOff = (anchor.getWidth() - popupW) / 2;
        int yOff = -anchor.getHeight() - popupH - 8;
        popup.showAsDropDown(anchor, xOff, yOff);

        tvStreaming.setOnClickListener(v -> {
            tap(v);
            popup.dismiss();
            prefAutoStop = !prefAutoStop;
            sp.edit().putBoolean(SettingsActivity.KEY_AUTO_STOP, prefAutoStop).apply();
            if (isListening) continuousMode = prefAutoStop;
            if (Log.isLoggable(TAG, Log.DEBUG)) Log.d(TAG, "Streaming toggled to: " + prefAutoStop);
        });

        popupView.findViewById(R.id.popup_settings).setOnClickListener(v -> {
            tap(v);
            popup.dismiss();
            Intent i = new Intent(this, SettingsActivity.class);
            i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(i);
        });
    }

    private void showPunctuationPopup(View anchor) {
        android.view.LayoutInflater inflater = android.view.LayoutInflater.from(this);
        android.view.View popupView = inflater.inflate(R.layout.popup_punctuation, null);

        // Non-focusable so touches go through to buttons underneath.
        // Background required for outside-touch dismissal to work.
        PopupWindow popup = new PopupWindow(popupView,
                android.view.ViewGroup.LayoutParams.WRAP_CONTENT,
                android.view.ViewGroup.LayoutParams.WRAP_CONTENT, false);
        popup.setElevation(8f * getResources().getDisplayMetrics().density);
        popup.setOutsideTouchable(true);
        popup.setBackgroundDrawable(TRANSPARENT_DRAWABLE);
        popup.setOnDismissListener(() -> {
            punctuationPopup = null;
            punctuationVisible = false;
            punctuationDismissedAt = android.os.SystemClock.uptimeMillis();
        });
        punctuationPopup = popup;
        punctuationVisible = true;

        // Center popup horizontally over the anchor button
        popupView.measure(android.view.View.MeasureSpec.UNSPECIFIED,
                          android.view.View.MeasureSpec.UNSPECIFIED);
        int popupW = popupView.getMeasuredWidth();
        int popupH = popupView.getMeasuredHeight();
        int xOff = (anchor.getWidth() - popupW) / 2;
        int yOff = -anchor.getHeight() - popupH - 8;
        popup.showAsDropDown(anchor, xOff, yOff);

        // Row 1: common punctuation — does NOT dismiss on press
        int[] ids1   = {R.id.punct_comma, R.id.punct_period, R.id.punct_exclaim,
                         R.id.punct_question, R.id.punct_dash,
                         R.id.punct_lparen,   R.id.punct_rparen};
        String[] chs1 = {",", ".", "!", "?", "-", "(", ")"};

        // Row 2: symbols
        int[] ids2   = {R.id.punct_semicolon, R.id.punct_colon, R.id.punct_quote,
                         R.id.punct_apostrophe, R.id.punct_at,
                         R.id.punct_amp,        R.id.punct_slash};
        String[] chs2 = {";", ":", String.valueOf((char)34), "'", "@", "&", "/"};

        for (int i = 0; i < ids1.length; i++) {
            final String ch = chs1[i];
            android.view.View btn = popupView.findViewById(ids1[i]);
            if (btn != null) btn.setOnClickListener(v -> {
                tap(v);
                if (getCurrentInputConnection() != null)
                    getCurrentInputConnection().commitText(ch, 1);
            });
        }
        for (int i = 0; i < ids2.length; i++) {
            final String ch = chs2[i];
            android.view.View btn = popupView.findViewById(ids2[i]);
            if (btn != null) btn.setOnClickListener(v -> {
                tap(v);
                if (getCurrentInputConnection() != null)
                    getCurrentInputConnection().commitText(ch, 1);
            });
        }
    }

    private void doEnter() {
        if (getCurrentInputConnection() == null) return;
        long t = android.os.SystemClock.uptimeMillis();
        getCurrentInputConnection().sendKeyEvent(
                new KeyEvent(t, t, KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_ENTER, 0));
        getCurrentInputConnection().sendKeyEvent(
                new KeyEvent(t, t, KeyEvent.ACTION_UP,   KeyEvent.KEYCODE_ENTER, 0));
    }

    private void doForwardDelete() {
        if (getCurrentInputConnection() == null) return;
        CharSequence sel = getCurrentInputConnection().getSelectedText(0);
        if (sel != null && sel.length() > 0) {
            getCurrentInputConnection().commitText("", 1);
        } else {
            getCurrentInputConnection().deleteSurroundingText(0, 1);
        }
    }

    /**
     * Removes the most recent transcription result from the text field.
     * Uses the transcriptionHistory ring buffer to know exactly how many
     * characters were committed, so it deletes precisely that text without
     * touching anything the user typed manually or from a different session.
     * Called by long-pressing the Undo button.
     */
    private void undoLastTranscription() {
        if (transcriptionHistory.isEmpty()) {
            // Nothing to undo — brief status flash
            if (tvStatus != null) {
                tvStatus.setText("Nothing to undo");
                tvStatus.setVisibility(android.view.View.VISIBLE);
                handler.postDelayed(() -> {
                    if (tvStatus != null) tvStatus.setVisibility(android.view.View.GONE);
                }, 1500);
            }
            return;
        }
        String last = transcriptionHistory.pollLast();
        if (getCurrentInputConnection() == null) return;
        // Verify the text before the cursor actually ends with what we committed
        // before deleting — guards against the user having moved the cursor or
        // edited manually since the transcription was committed.
        CharSequence before = getCurrentInputConnection()
                .getTextBeforeCursor(last.length(), 0);
        if (before != null && before.toString().equals(last)) {
            getCurrentInputConnection().deleteSurroundingText(last.length(), 0);
        } else {
            // Cursor has moved or content changed — fall back to Ctrl+Z
            sendCtrlKey(KeyEvent.KEYCODE_Z, false);
        }
    }

    // ── v12 new methods ────────────────────────────────────────────────────────

    /**
     * Break Audio: stop the current recording segment so Whisper transcribes it
     * immediately, then restart listening without any perceptible gap.
     * Unlike discardAudioAndContinue(), the captured audio IS committed — this
     * is a manual VAD trigger, not a discard. Only callable while a session is active.
     */
    // ── v12 capitalisation helpers ────────────────────────────────────────────

    /**
     * Apply capsOverride (or prefAutoCapitalise) to an entire Whisper result.
     * Whisper may return multiple sentences in one block (e.g. "Hello. How are you?").
     * All sentence-starting capitals must be handled, not just the very first character.
     */
    private String applyCapitalization(String text) {
        if (text == null || text.isEmpty()) return text;
        if (capsOverride == 1) {
            // Force uppercase: capitalise first char and every sentence start.
            // Whisper normally does this already, but handle any edge cases.
            return uppercaseSentenceStarts(text);
        } else if (capsOverride == 2) {
            // Force lowercase: strip all sentence-start capitals Whisper inserted.
            // This is the only mode that touches mid-block sentence starts.
            return lowercaseSentenceStarts(text);
        } else {
            // Auto mode: only touch the very first character of the result,
            // exactly matching the original pre-v13 behaviour.
            // Whisper handles mid-block sentence capitalisation itself — we must
            // not strip those or multi-sentence blocks lose their sentence breaks.
            if (!prefAutoCapitalise && text.length() > 0
                    && Character.isUpperCase(text.charAt(0))) {
                return Character.toLowerCase(text.charAt(0)) + text.substring(1);
            }
            return text;
        }
    }

    /**
     * Capitalise the first letter of every sentence in {@code text}.
     * Used by force-upper (capsOverride=1).
     */
    private String uppercaseSentenceStarts(String text) {
        if (text == null || text.isEmpty()) return text;
        StringBuilder sb = new StringBuilder(text);
        if (Character.isLowerCase(sb.charAt(0))) {
            sb.setCharAt(0, Character.toUpperCase(sb.charAt(0)));
        }
        for (int i = 1; i < sb.length() - 1; i++) {
            char punct = sb.charAt(i - 1);
            char space = sb.charAt(i);
            char next  = sb.charAt(i + 1);
            if ((punct == '.' || punct == '!' || punct == '?')
                    && space == ' ' && Character.isLowerCase(next)) {
                sb.setCharAt(i + 1, Character.toUpperCase(next));
            }
        }
        return sb.toString();
    }

    /**
     * Lowercase the first letter of every sentence in {@code text}.
     * Targets the first character and any uppercase letter that immediately follows
     * a sentence-ending punctuation mark ({@code .}, {@code !}, {@code ?}) and a space.
     */
    private String lowercaseSentenceStarts(String text) {
        if (text == null || text.isEmpty()) return text;
        StringBuilder sb = new StringBuilder(text);
        // First character of the whole result
        if (Character.isUpperCase(sb.charAt(0))) {
            sb.setCharAt(0, Character.toLowerCase(sb.charAt(0)));
        }
        // Subsequent sentence starts: look for ". X", "! X", "? X" patterns
        for (int i = 1; i < sb.length() - 1; i++) {
            char punct = sb.charAt(i - 1);
            char space = sb.charAt(i);
            char next  = sb.charAt(i + 1);
            if ((punct == '.' || punct == '!' || punct == '?')
                    && space == ' ' && Character.isUpperCase(next)) {
                sb.setCharAt(i + 1, Character.toLowerCase(next));
            }
        }
        return sb.toString();
    }

    // ── v12 modifier UI ────────────────────────────────────────────────────────

    /**
     * Highlight active modifier buttons (Shift / Ctrl / Alt) so the user can see
     * which modifiers are currently toggled on.  Active = full opacity, inactive = dim.
     */
    private void updateModifierUI() {
        if (btnSecShift != null) btnSecShift.setAlpha(shiftActive ? 1.0f : 0.55f);
        if (btnSecCtrl  != null) btnSecCtrl.setAlpha(ctrlActive  ? 1.0f : 0.55f);
        if (btnSecAlt   != null) btnSecAlt.setAlpha(altActive    ? 1.0f : 0.55f);
    }

    private void breakAudioAndContinue() {
        if (!isListening || mRecorder == null) return;
        // Stop the current chunk. MSG_RECORDING_DONE will fire and route the
        // audio to Whisper. discardRequested is NOT set, so it commits normally.
        if (mRecorder.isSessionActive()) mRecorder.stop();
        // Immediately restart to keep listening.
        activeRecordingSessionId = sessionGen;
        if (continuousMode) mRecorder.initVad();
        mRecorder.start();
        setMicState(MicState.LISTENING);
    }

    /**
     * Update the caps-toggle button icon to reflect the current capsOverride state.
     * Safe to call at any time; no-ops if the button reference is null.
     */
    private void updateCapsIcon() {
        if (btnCapsToggle == null) return;
        switch (capsOverride) {
            case 1:  // Force uppercase first letter
                btnCapsToggle.setImageResource(R.drawable.ic_caps_upper_36dp);
                break;
            case 2:  // Force lowercase first letter
                btnCapsToggle.setImageResource(R.drawable.ic_caps_lower_36dp);
                break;
            default: // Auto — honours prefAutoCapitalise setting
                btnCapsToggle.setImageResource(R.drawable.ic_caps_auto_36dp);
                break;
        }
    }

    /**
     * Toggle between the primary (recording) and secondary (extended controls) layouts.
     *
     * Switching TO secondary:
     *   - Records whether audio was active so we can resume on return.
     *   - Stops recording/transcription cleanly via exitListeningMode().
     *   - Hides the primary layout, shows the secondary layout.
     *   - Refreshes the streaming/language labels.
     *
     * Switching BACK to primary:
     *   - Hides the secondary layout, shows the primary layout.
     *   - If audio was active before the switch, resumes the recording session.
     */
    private void toggleSecondaryLayout() {
        isSecondaryLayout = !isSecondaryLayout;
        if (layoutPrimary == null || layoutSecondary == null) return;

        if (isSecondaryLayout) {
            // Switching to secondary — snapshot state and pause audio
            wasListeningBeforeSecondary = isListening;
            if (isListening) exitListeningMode(true);
            layoutPrimary.setVisibility(View.GONE);
            layoutSecondary.setVisibility(View.VISIBLE);
            updateSecondaryLabels();
        } else {
            // Switching back to primary — restore audio if it was running
            layoutSecondary.setVisibility(View.GONE);
            layoutPrimary.setVisibility(View.VISIBLE);
            if (wasListeningBeforeSecondary && !isListening) {
                suppressNextAutoStart = false;  // allow restart
                startRecordingSession(prefAutoStop);
            }
        }
    }

    /**
     * Refresh the dynamic labels in the secondary layout (streaming mode, language).
     * Called whenever the secondary layout is shown or a value changes.
     */
    private void updateSecondaryLabels() {
        // Labels removed in v13 — method retained for future use.
    }

    private void stopListeningGracefully() {
        continuousMode        = false;   // stop the continuous loop
        isListening           = false;   // prevent auto-restart
        pendingStartSessionId = 0;       // cancel any deferred recorder start
        // cancelRequested stays false — Whisper result will be committed normally.
        // sessionGen is NOT incremented — existing session IDs remain valid so
        // all in-flight callbacks (MSG_RECORDING_DONE, onResultReceived) pass
        // their session guards and process normally.

        // stop() is fire-and-forget — call directly on main thread (returns instantly)
        if (mRecorder != null && mRecorder.isSessionActive()) mRecorder.stop();

        transcriptionHistory.clear(); // undo history doesn't survive a soft stop
        sessionLanguage = "";         // reset language lock so next session detects fresh
        suppressNextAutoStart = true;
        suppressedForEditor   = currentEditorKey();
        suppressSetAt         = android.os.SystemClock.uptimeMillis();

        // If Whisper is processing or audio is queued, stay in LISTENING (dim)
        // state so the user sees the result is still pending.
        // onResultReceived will drain the queue and go IDLE when done.
        if ((mWhisper != null && (mWhisper.isInProgress() || mWhisper.isResultPending()))
                || !audioQueue.isEmpty()) {
            // Result is still incoming — stay in LISTENING state so user knows
            // the committed text is still on its way.
            setMicState(MicState.LISTENING);
            // Cancel button remains enabled so the user can still discard
        } else {
            setMicState(MicState.IDLE);
            setCancelEnabled(false);
            resetProgressUi();
        }
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
        cancelRequested       = true;
        continuousMode        = false;
        isListening           = false;
        isRecording           = false;
        pendingStartSessionId = 0;   // cancel any deferred start
        startAfterWhisper     = false; // cancel any deferred new session
        transcriptionHistory.clear(); // undo history is session-scoped
        sessionLanguage       = "";   // reset language lock for next session
        // suppressNextAutoStart is intentionally NOT set here.
        // exitListeningMode is called from onFinishInputView (keyboard close / app switch)
        // as well as from explicit Cancel actions. Setting it here would block auto-start
        // when returning to the same field after switching apps.
        // Suppression is set explicitly by the Cancel button and stop button handlers.
        final int sid         = ++sessionGen;

        if (mRecorder != null) {
            // stop() is instant — call directly regardless of async flag.
            if (mRecorder != null && mRecorder.isSessionActive()) mRecorder.stop();
        }

        handler.post(() -> {
            if (sessionGen != sid) return;
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
        RecordBuffer.setOutputBuffer(audioBytes);
        mWhisper.setAction(translate ? ACTION_TRANSLATE : ACTION_TRANSCRIBE);
        // Use the locked session language if we've already detected it this session.
        // Falls back to the user's setting (or "auto") for the first segment.
        String userLang = sp != null ? sp.getString("language", "auto") : "auto";
        String effectiveLang = (!sessionLanguage.isEmpty() && userLang.equals("auto"))
                ? sessionLanguage : userLang;
        mWhisper.setLanguage(effectiveLang);
        discardRequested = false;             // clear any pending discard
        activeWhisperSessionId = sessionGen;  // stamp BEFORE mWhisper.start()
        if (Log.isLoggable(TAG, Log.DEBUG)) Log.d(TAG, "startTranscription: " + audioBytes.length + " bytes, session=" + activeWhisperSessionId);
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
        audioQueue.clear();     // discard any buffered segments
        resetProgressUi();      // called on main thread — no need to post
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
        if (btnCancel != null) {
            btnCancel.setEnabled(enabled);
            btnCancel.setAlpha(enabled ? 1.0f : 0.4f);
        }
        if (btnDiscard != null) {
            btnDiscard.setEnabled(enabled);
            btnDiscard.setAlpha(enabled ? 1.0f : 0.4f);
        }
        // Break Audio follows the same enable/disable pattern as Discard Audio:
        // it is only meaningful while a session is active.
        if (btnBreakAudio != null) {
            btnBreakAudio.setEnabled(enabled);
            btnBreakAudio.setAlpha(enabled ? 1.0f : 0.4f);
        }
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
        long t = android.os.SystemClock.uptimeMillis();  // KeyEvent requires uptimeMillis
        getCurrentInputConnection().sendKeyEvent(
                new KeyEvent(t, t, KeyEvent.ACTION_DOWN, keyCode, 0, meta));
        getCurrentInputConnection().sendKeyEvent(
                new KeyEvent(t, t, KeyEvent.ACTION_UP, keyCode, 0, 0));
    }

    /**
     * Returns true if the Whisper result is entirely a non-speech annotation.
     * Whisper outputs these when it hears non-vocal audio: (air whooshing),
     * [blank audio], (upbeat music), (car engine revving), etc.
     * A result is considered non-speech if every "token" (parenthetical or
     * bracketed group) is an annotation, with no normal words between them.
     */
    private static boolean isNonSpeechAnnotation(String text) {
        // Strip all (…) and […] groups. If nothing remains (after trimming),
        // the entire result is annotations.
        String stripped = text
                .replaceAll("\\([^)]*\\)", "")   // remove (...)
                .replaceAll("\\[[^\\]]*\\]", "") // remove [...]
                .trim();
        return stripped.isEmpty();
    }

    /**
     * Returns a stable key identifying the currently active editor.
     * Used to distinguish "same field after send" from "new field focused".
     * Combines the app package name with the field ID — two different text
     * fields in the same app produce different keys.
     */
    private String currentEditorKey() {
        EditorInfo ei = getCurrentInputEditorInfo();
        if (ei == null) return "";
        return (ei.packageName != null ? ei.packageName : "") + "/" + ei.fieldId;
    }

    /** Overload accepting an EditorInfo directly (used in onStartInputView). */
    private String currentEditorKey(EditorInfo attribute) {
        if (attribute == null) return "";
        return (attribute.packageName != null ? attribute.packageName : "") + "/" + attribute.fieldId;
    }

    /**
     * Reads current preferences. Defaults are set once in onCreate() via
     * applyDefaultPrefs() — this method just reads.
     */
    private void loadPrefs() {
        if (sp == null) sp = PreferenceManager.getDefaultSharedPreferences(this);
        prefAutoStart       = sp.getBoolean(SettingsActivity.KEY_LAUNCH_LISTENING, true);
        prefAutoStop        = sp.getBoolean(SettingsActivity.KEY_AUTO_STOP,        true);
        prefAutoSwitch      = sp.getBoolean(SettingsActivity.KEY_AUTO_SWITCH,      false);
        prefAutoSend        = sp.getBoolean(SettingsActivity.KEY_AUTO_SEND,        false);
        prefHapticRecording  = sp.getBoolean(SettingsActivity.KEY_HAPTIC_RECORDING, true);
        prefAutoCapitalise   = sp.getBoolean(SettingsActivity.KEY_AUTO_CAPITALISE,   true);
    }

    /** Writes preference defaults once on first install. Called only from onCreate(). */
    private void applyDefaultPrefs() {
        if (sp == null) sp = PreferenceManager.getDefaultSharedPreferences(this);
        SharedPreferences.Editor ed = null;
        for (String[] kv : new String[][]{
            {SettingsActivity.KEY_LAUNCH_LISTENING, "true"},
            {SettingsActivity.KEY_AUTO_START,       "true"},
            {SettingsActivity.KEY_AUTO_STOP,        "true"},
            {SettingsActivity.KEY_AUTO_SWITCH,      "false"},
            {SettingsActivity.KEY_AUTO_SEND,        "false"},
        }) {
            if (!sp.contains(kv[0])) {
                if (ed == null) ed = sp.edit();
                ed.putBoolean(kv[0], Boolean.parseBoolean(kv[1]));
            }
        }
        if (ed != null) ed.apply();
    }
}
