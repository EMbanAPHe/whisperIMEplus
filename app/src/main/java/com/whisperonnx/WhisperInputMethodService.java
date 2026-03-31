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
import com.whisperonnx.asr.Recorder;
import com.whisperonnx.asr.Whisper;
import com.whisperonnx.asr.WhisperResult;
import com.whisperonnx.utils.HapticFeedback;

/**
 * Whisper+ IME keyboard panel.
 *
 * Mic button — three visual states
 * ──────────────────────────────────
 *   IDLE       rounded_button_tonal      dark grey    not in listening mode
 *   LISTENING  rounded_button_listening  mid-purple   VAD running, waiting for speech
 *   RECORDING  rounded_button_background full-purple  speech detected, capturing audio
 *
 * State machine
 * ─────────────
 *   isListening  tracks whether we have called mRecorder.start() and are
 *                waiting for (or actively capturing) audio.  Set true in
 *                startRecordingSession(), cleared by Cancel, keyboard close,
 *                and MSG_RECORDING_DONE/ERROR callbacks.
 *
 *   isRecording  true only while VAD has confirmed speech is present
 *                (MSG_RECORDING callback received).
 *
 *   continuousMode  true while auto-restart is desired after each result.
 *
 * Mic tap behaviour
 * ─────────────────
 *   If isListening → stop (exit listening mode, set continuousMode=false)
 *   If not         → enter listening mode
 *
 * Cancel button
 * ─────────────
 *   Enabled as soon as isListening becomes true.
 *   Exits listening mode and stops any in-progress transcription.
 */
public class WhisperInputMethodService extends InputMethodService {

    private static final String TAG = "WhisperIME";

    // ── Widgets ───────────────────────────────────────────────────────────────
    private ImageButton  btnRecord;
    private ImageButton  btnKeyboard;
    private ImageButton  btnSettings;
    private ImageButton  btnSend;
    private ImageButton  btnUndo;
    private ImageButton  btnRedo;
    private ImageButton  btnDel;
    private ImageButton  btnEnter;
    private View         btnCancel;   // TextView in layout
    private View         btnClear;    // TextView in layout
    private View         btnSpace;    // TextView in layout
    private TextView     tvStatus;
    private ProgressBar  processingBar;
    private LinearLayout layoutKeyboardContent;

    // ── Static state (survives view re-inflation) ──────────────────────────────
    private static boolean translate = false;

    // ── Session state (volatile — read from recorder/whisper threads) ──────────
    /**
     * True from the moment startRecordingSession() is called until the session
     * ends (Cancel, keyboard close, MSG_RECORDING_DONE/ERROR without continuous
     * restart).  Controls the LISTENING visual state and Cancel availability.
     */
    private volatile boolean isListening    = false;

    /**
     * True only while the VAD has detected actual speech in the audio stream
     * (MSG_RECORDING received, MSG_RECORDING_DONE not yet received).
     * Controls the RECORDING (bright) visual state.
     */
    private volatile boolean isRecording    = false;

    /**
     * Set before calling Recorder.stop() via Cancel or mic tap-to-stop, so
     * the RecorderListener knows NOT to start transcription or restart.
     */
    private volatile boolean cancelRequested = false;

    /**
     * True while we want auto-restart after each transcription result.
     */
    private volatile boolean continuousMode  = false;

    // ── Preferences ───────────────────────────────────────────────────────────
    private boolean prefAutoStart;
    private boolean prefAutoStop;
    private boolean prefAutoSwitch;
    private boolean prefAutoSend;

    // ── Core ──────────────────────────────────────────────────────────────────
    private Recorder          mRecorder = null;
    private Whisper           mWhisper  = null;
    private SharedPreferences sp        = null;
    private final Handler     handler   = new Handler(Looper.getMainLooper());
    private Context           mContext;

    // ═══════════════════════════════════════════════════════════════════════════
    // Service lifecycle
    // ═══════════════════════════════════════════════════════════════════════════

    @Override
    public void onCreate() {
        mContext = this;
        super.onCreate();
    }

    @Override
    public void onDestroy() {
        exitListeningMode(false);
        deinitModel();
        super.onDestroy();
    }

    @Override
    public void onStartInput(EditorInfo attribute, boolean restarting) {
        if (attribute.inputType == EditorInfo.TYPE_NULL) {
            exitListeningMode(false);
            deinitModel();
        }
    }

    @Override
    public void onStartInputView(EditorInfo attribute, boolean restarting) {
        if (mWhisper == null) initModel();
        loadPrefs();
        if (prefAutoStart && !isListening) {
            startRecordingSession(prefAutoStop);
        }
    }

    @Override
    public void onFinishInputView(boolean finishingInput) {
        exitListeningMode(false);
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

        setKeyboardHeight();
        checkRecordPermission();

        // ── Recorder ──────────────────────────────────────────────────────────
        mRecorder = new Recorder(this);
        mRecorder.setListener(message -> {
            // Called on Recorder worker thread — all UI via handler.post()

            if (message.equals(Recorder.MSG_RECORDING)) {
                // VAD confirmed speech — upgrade to RECORDING (bright) colour
                isRecording = true;
                handler.post(() ->
                        btnRecord.setBackgroundResource(R.drawable.rounded_button_background));

            } else if (message.equals(Recorder.MSG_RECORDING_DONE)) {
                // Speech segment finished — downgrade back to LISTENING colour
                // while transcription runs (or to IDLE if cancel was requested)
                isRecording = false;
                HapticFeedback.vibrate(mContext);

                if (!cancelRequested) {
                    // Keep LISTENING colour during transcription
                    handler.post(() ->
                            btnRecord.setBackgroundResource(R.drawable.rounded_button_listening));
                    startTranscription();
                } else {
                    // User cancelled — go fully IDLE
                    isListening    = false;
                    cancelRequested = false;
                    continuousMode  = false;
                    handler.post(() -> {
                        setMicIdle();
                        setCancelEnabled(false);
                        resetProgressUi();
                    });
                }

            } else if (message.equals(Recorder.MSG_RECORDING_ERROR)) {
                isRecording     = false;
                isListening     = false;
                cancelRequested = false;
                continuousMode  = false;
                HapticFeedback.vibrate(mContext);
                handler.post(() -> {
                    setMicIdle();
                    setCancelEnabled(false);
                    tvStatus.setText(getString(R.string.error_no_input));
                    tvStatus.setVisibility(View.VISIBLE);
                    processingBar.setProgress(0);
                    processingBar.setIndeterminate(false);
                });
            }
        });

        // ── Mic — toggle listening mode on/off ────────────────────────────────
        btnRecord.setOnClickListener(v -> {
            if (isListening) {
                // Already listening — user wants to stop the whole session
                exitListeningMode(true);   // async stop on background thread
            } else {
                if (!checkRecordPermission()) return;
                startRecordingSession(prefAutoStop);
            }
        });

        // ── Cancel — exits listening mode completely ───────────────────────────
        btnCancel.setOnClickListener(v -> {
            if (mWhisper != null) stopTranscription();
            exitListeningMode(true);
        });

        // ── Settings ──────────────────────────────────────────────────────────
        btnSettings.setOnClickListener(v -> {
            Intent i = new Intent(this, SettingsActivity.class);
            i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(i);
        });

        // ── Switch keyboard ────────────────────────────────────────────────────
        btnKeyboard.setOnClickListener(v -> {
            if (mWhisper != null) stopTranscription();
            exitListeningMode(false);
            switchToPreviousInputMethod();
        });

        // ── Clear all unselected text ──────────────────────────────────────────
        btnClear.setOnClickListener(v -> {
            if (getCurrentInputConnection() == null) return;
            ExtractedText et = getCurrentInputConnection()
                    .getExtractedText(new ExtractedTextRequest(), 0);
            if (et != null && et.text != null) {
                int len = et.text.length();
                CharSequence before = getCurrentInputConnection().getTextBeforeCursor(len, 0);
                CharSequence after  = getCurrentInputConnection().getTextAfterCursor(len, 0);
                if (before != null && after != null) {
                    getCurrentInputConnection()
                            .deleteSurroundingText(before.length(), after.length());
                }
            }
        });

        // ── Undo / Redo ────────────────────────────────────────────────────────
        btnUndo.setOnClickListener(v -> sendCtrlKey(KeyEvent.KEYCODE_Z, false));
        btnRedo.setOnClickListener(v -> sendCtrlKey(KeyEvent.KEYCODE_Z, true));

        // ── Backspace with long-press repeat ───────────────────────────────────
        btnDel.setOnTouchListener(new View.OnTouchListener() {
            private Runnable initial, fast;
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                switch (event.getAction()) {
                    case MotionEvent.ACTION_DOWN:
                        doDelete();
                        initial = () -> {
                            doDelete();
                            fast = new Runnable() {
                                @Override public void run() {
                                    doDelete();
                                    handler.postDelayed(this, 50);
                                }
                            };
                            handler.postDelayed(fast, 50);
                        };
                        handler.postDelayed(initial, 400);
                        break;
                    case MotionEvent.ACTION_UP:
                    case MotionEvent.ACTION_CANCEL:
                        if (initial != null) handler.removeCallbacks(initial);
                        if (fast    != null) handler.removeCallbacks(fast);
                        break;
                    default: break;
                }
                return true;
            }
        });

        // ── Space ─────────────────────────────────────────────────────────────
        btnSpace.setOnClickListener(v -> {
            if (getCurrentInputConnection() != null)
                getCurrentInputConnection().commitText(" ", 1);
        });

        // ── Send (IME action) ──────────────────────────────────────────────────
        btnSend.setOnClickListener(v -> {
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
            if (getCurrentInputConnection() == null) return;
            getCurrentInputConnection().sendKeyEvent(
                    new KeyEvent(KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_ENTER));
            getCurrentInputConnection().sendKeyEvent(
                    new KeyEvent(KeyEvent.ACTION_UP, KeyEvent.KEYCODE_ENTER));
        });

        return view;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Session helpers
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Enters listening mode.
     *
     * @param useVad  true = VAD auto-stops on silence; restart after each
     *                transcription result (continuous/streaming).
     *                false = user taps mic again to stop.
     */
    private void startRecordingSession(boolean useVad) {
        if (!checkRecordPermission()) return;
        cancelRequested = false;
        isListening     = true;
        continuousMode  = useVad;
        handler.post(() -> {
            // LISTENING state — mid-purple
            btnRecord.setBackgroundResource(R.drawable.rounded_button_listening);
            setCancelEnabled(true);
            tvStatus.setVisibility(View.GONE);
            processingBar.setIndeterminate(false);
            processingBar.setProgress(0);
        });
        if (useVad) mRecorder.initVad();
        mRecorder.start();
    }

    /**
     * Exits listening mode fully — stops the recorder (optionally async),
     * clears all session flags, and resets the mic to IDLE.
     *
     * @param async  If true, Recorder.stop() runs on a background thread so
     *               the main thread is never blocked (used by Cancel & mic tap).
     *               If false, called from lifecycle callbacks where blocking
     *               briefly is acceptable.
     */
    private void exitListeningMode(boolean async) {
        cancelRequested = true;
        continuousMode  = false;
        isListening     = false;
        isRecording     = false;

        if (mRecorder != null && mRecorder.isInProgress()) {
            if (async) {
                new Thread(() -> mRecorder.stop()).start();
            } else {
                mRecorder.stop();
            }
        }

        handler.post(() -> {
            setMicIdle();
            setCancelEnabled(false);
            resetProgressUi();
        });
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Model
    // ═══════════════════════════════════════════════════════════════════════════

    private void initModel() {
        mWhisper = new Whisper(this);
        mWhisper.loadModel();
        mWhisper.setListener(new Whisper.WhisperListener() {
            @Override public void onUpdateReceived(String message) { /* unused */ }

            @Override
            public void onResultReceived(WhisperResult whisperResult) {
                // Runs on Whisper inference thread.
                handler.post(() -> {
                    processingBar.setIndeterminate(false);
                    processingBar.setProgress(0);
                    tvStatus.setVisibility(View.GONE);
                });

                // Chinese script conversion
                String result = whisperResult.getResult();
                if ("zh".equals(whisperResult.getLanguage())) {
                    boolean simple = sp != null && sp.getBoolean("simpleChinese", false);
                    result = simple ? ZhConverterUtil.toSimple(result)
                                    : ZhConverterUtil.toTraditional(result);
                }

                final String trimmed = result.trim();
                if (!trimmed.isEmpty() && getCurrentInputConnection() != null) {
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

                // Auto-switch (only for non-continuous sessions)
                if (prefAutoSwitch && !continuousMode) {
                    isListening = false;
                    handler.post(() -> {
                        setMicIdle();
                        setCancelEnabled(false);
                        switchToPreviousInputMethod();
                    });
                    return;
                }

                // ── Continuous streaming: restart for next utterance ────────────
                if (continuousMode && !cancelRequested) {
                    // Post to main thread; Recorder's mInProgress is false by now.
                    handler.post(() -> startRecordingSession(true));
                } else {
                    // Session ended cleanly after a single (non-continuous) take
                    isListening = false;
                    handler.post(() -> {
                        setMicIdle();
                        setCancelEnabled(false);
                    });
                }
            }
        });
    }

    private void startTranscription() {
        handler.post(() -> {
            processingBar.setProgress(0);
            processingBar.setIndeterminate(true);
        });
        if (mWhisper != null) {
            mWhisper.setAction(translate ? ACTION_TRANSLATE : ACTION_TRANSCRIBE);
            mWhisper.setLanguage(sp != null ? sp.getString("language", "auto") : "auto");
            Log.d(TAG, "startTranscription translate=" + translate);
            mWhisper.start();
        }
    }

    private void stopTranscription() {
        handler.post(() -> {
            processingBar.setIndeterminate(false);
            processingBar.setProgress(0);
        });
        if (mWhisper != null) mWhisper.stop();
    }

    private void deinitModel() {
        if (mWhisper != null) { mWhisper.unloadModel(); mWhisper = null; }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // UI helpers — all must be called on the main thread
    // ═══════════════════════════════════════════════════════════════════════════

    /** Mic → IDLE (dark grey tonal). */
    private void setMicIdle() {
        if (btnRecord != null)
            btnRecord.setBackgroundResource(R.drawable.rounded_button_tonal);
    }

    private void setCancelEnabled(boolean enabled) {
        if (btnCancel != null) {
            btnCancel.setEnabled(enabled);
            btnCancel.setAlpha(enabled ? 1.0f : 0.4f);
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
     * Sizes the content area.
     * v3 was 0.33 portrait / 0.45 landscape.
     * −10 % → 0.297 → 0.30 portrait / 0.405 → 0.41 landscape.
     */
    private void setKeyboardHeight() {
        DisplayMetrics dm = getResources().getDisplayMetrics();
        boolean landscape = getResources().getConfiguration().orientation
                == Configuration.ORIENTATION_LANDSCAPE;
        int targetPx = (int) (dm.heightPixels * (landscape ? 0.41f : 0.30f));
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

    private void loadPrefs() {
        if (sp == null) sp = PreferenceManager.getDefaultSharedPreferences(this);
        SharedPreferences.Editor ed = null;
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

        prefAutoStart  = sp.getBoolean(SettingsActivity.KEY_AUTO_START,  true);
        prefAutoStop   = sp.getBoolean(SettingsActivity.KEY_AUTO_STOP,   true);
        prefAutoSwitch = sp.getBoolean(SettingsActivity.KEY_AUTO_SWITCH, false);
        prefAutoSend   = sp.getBoolean(SettingsActivity.KEY_AUTO_SEND,   false);
    }
}
