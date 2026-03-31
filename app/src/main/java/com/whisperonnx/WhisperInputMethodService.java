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

public class WhisperInputMethodService extends InputMethodService {

    private static final String TAG = "WhisperIME";

    // ── Widgets ──────────────────────────────────────────────────────────────
    private ImageButton btnRecord;
    private ImageButton btnKeyboard;
    private ImageButton btnSettings;
    private ImageButton btnTranslate;
    private ImageButton btnCancel;
    private ImageButton btnClear;
    private ImageButton btnUndo;
    private ImageButton btnRedo;
    private ImageButton btnDel;
    private ImageButton btnSend;
    private ImageButton btnEnter;
    private TextView    tvStatus;
    private ProgressBar processingBar;
    private LinearLayout layoutKeyboardContent;

    // ── Static state (survives view re-inflation) ─────────────────────────────
    private static boolean translate = false;

    // ── Volatile state (read across threads) ──────────────────────────────────
    private volatile boolean isRecording     = false;
    private volatile boolean cancelRequested = false;
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
    private final Handler handler       = new Handler(Looper.getMainLooper());
    private Context mContext;

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
        continuousMode = false;
        deinitModel();
        if (mRecorder != null && mRecorder.isInProgress()) mRecorder.stop();
        super.onDestroy();
    }

    @Override
    public void onStartInput(EditorInfo attribute, boolean restarting) {
        if (attribute.inputType == EditorInfo.TYPE_NULL) {
            continuousMode = false;
            deinitModel();
            if (mRecorder != null && mRecorder.isInProgress()) mRecorder.stop();
        }
    }

    @Override
    public void onStartInputView(EditorInfo attribute, boolean restarting) {
        if (mWhisper == null) initModel();
        loadPrefs();
        updateTranslateIcon();
        if (prefAutoStart && !isRecording) {
            startRecordingSession(prefAutoStop);
        }
    }

    @Override
    public void onFinishInputView(boolean finishingInput) {
        cancelRequested = true;
        continuousMode  = false;
        if (mRecorder != null && mRecorder.isInProgress()) {
            new Thread(() -> mRecorder.stop()).start();
        }
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
        btnTranslate          = view.findViewById(R.id.btnTranslate);
        btnCancel             = view.findViewById(R.id.btnCancel);
        btnClear              = view.findViewById(R.id.btnClear);
        btnUndo               = view.findViewById(R.id.btnUndo);
        btnRedo               = view.findViewById(R.id.btnRedo);
        btnDel                = view.findViewById(R.id.btnDel);
        btnSend               = view.findViewById(R.id.btnSend);
        btnEnter              = view.findViewById(R.id.btnEnter);
        processingBar         = view.findViewById(R.id.processing_bar);
        tvStatus              = view.findViewById(R.id.tv_status);
        layoutKeyboardContent = view.findViewById(R.id.layout_keyboard_content);

        setKeyboardHeight();
        updateTranslateIcon();
        checkRecordPermission();

        // ── Recorder ──────────────────────────────────────────────────────────
        mRecorder = new Recorder(this);
        mRecorder.setListener(message -> {
            // Called on Recorder worker thread — UI updates via handler.post()
            if (message.equals(Recorder.MSG_RECORDING)) {
                isRecording = true;
                handler.post(() -> {
                    btnRecord.setBackgroundResource(
                            R.drawable.rounded_button_background_pressed);
                    btnCancel.setEnabled(true);
                    btnCancel.setAlpha(1.0f);
                    processingBar.setIndeterminate(false);
                    processingBar.setProgress(100);
                });

            } else if (message.equals(Recorder.MSG_RECORDING_DONE)) {
                isRecording = false;
                HapticFeedback.vibrate(mContext);
                handler.post(() ->
                        btnRecord.setBackgroundResource(R.drawable.rounded_button_background));
                if (!cancelRequested) {
                    startTranscription();
                } else {
                    cancelRequested = false;
                    continuousMode  = false;
                    handler.post(() -> {
                        resetProgressUi();
                        btnCancel.setEnabled(false);
                        btnCancel.setAlpha(0.4f);
                    });
                }

            } else if (message.equals(Recorder.MSG_RECORDING_ERROR)) {
                isRecording    = false;
                cancelRequested = false;
                continuousMode  = false;
                HapticFeedback.vibrate(mContext);
                handler.post(() -> {
                    btnRecord.setBackgroundResource(R.drawable.rounded_button_background);
                    btnCancel.setEnabled(false);
                    btnCancel.setAlpha(0.4f);
                    tvStatus.setText(getString(R.string.error_no_input));
                    tvStatus.setVisibility(View.VISIBLE);
                    processingBar.setProgress(0);
                    processingBar.setIndeterminate(false);
                });
            }
        });

        // ── Mic — tap to toggle ────────────────────────────────────────────────
        btnRecord.setOnClickListener(v -> {
            if (isRecording) {
                continuousMode = false;          // end loop after this recording
                if (mRecorder != null && mRecorder.isInProgress()) mRecorder.stop();
            } else {
                if (!checkRecordPermission()) return;
                startRecordingSession(prefAutoStop);
            }
        });

        // ── Cancel ────────────────────────────────────────────────────────────
        btnCancel.setOnClickListener(v -> {
            cancelRequested = true;
            continuousMode  = false;
            if (mWhisper != null) stopTranscription();
            if (mRecorder != null && mRecorder.isInProgress()) {
                new Thread(() -> mRecorder.stop()).start();
            }
            isRecording = false;
            handler.post(() -> {
                btnRecord.setBackgroundResource(R.drawable.rounded_button_background);
                btnCancel.setEnabled(false);
                btnCancel.setAlpha(0.4f);
                resetProgressUi();
            });
        });

        // ── Settings ──────────────────────────────────────────────────────────
        btnSettings.setOnClickListener(v -> {
            Intent i = new Intent(this, SettingsActivity.class);
            i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(i);
        });

        // ── Translate toggle ───────────────────────────────────────────────────
        btnTranslate.setOnClickListener(v -> {
            translate = !translate;
            updateTranslateIcon();
        });

        // ── Switch keyboard ────────────────────────────────────────────────────
        btnKeyboard.setOnClickListener(v -> {
            cancelRequested = true;
            continuousMode  = false;
            if (mWhisper != null) stopTranscription();
            switchToPreviousInputMethod();
        });

        // ── Clear unselected text ──────────────────────────────────────────────
        btnClear.setOnClickListener(v -> {
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
        btnUndo.setOnClickListener(v -> sendCtrlKey(KeyEvent.KEYCODE_Z, false));
        btnRedo.setOnClickListener(v -> sendCtrlKey(KeyEvent.KEYCODE_Z, true));

        // ── Backspace with repeat ──────────────────────────────────────────────
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

        // ── Send ──────────────────────────────────────────────────────────────
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
    // Recording
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Starts a recording session.
     * @param useVad  true = VAD auto-stops on silence; recording restarts
     *                after each transcription result (streaming).
     *                false = user must tap mic again to stop.
     */
    private void startRecordingSession(boolean useVad) {
        if (!checkRecordPermission()) return;
        cancelRequested = false;
        continuousMode  = useVad;
        handler.post(() -> {
            tvStatus.setVisibility(View.GONE);
            processingBar.setIndeterminate(false);
            processingBar.setProgress(0);
        });
        if (useVad) mRecorder.initVad();
        mRecorder.start();
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

                String result = whisperResult.getResult();
                if ("zh".equals(whisperResult.getLanguage())) {
                    boolean simple = sp != null && sp.getBoolean("simpleChinese", false);
                    result = simple ? ZhConverterUtil.toSimple(result)
                                    : ZhConverterUtil.toTraditional(result);
                }

                final String trimmed = result.trim();
                if (!trimmed.isEmpty() && getCurrentInputConnection() != null) {
                    // Safe from non-UI thread; trailing space separates utterances.
                    getCurrentInputConnection().commitText(trimmed + " ", 1);
                }

                // Auto-send
                if (prefAutoSend && getCurrentInputConnection() != null) {
                    EditorInfo ei = getCurrentInputEditorInfo();
                    if (ei != null && ei.actionId != 0
                            && ei.actionId != EditorInfo.IME_ACTION_NONE) {
                        getCurrentInputConnection().performEditorAction(ei.actionId);
                    } else {
                        getCurrentInputConnection().performEditorAction(
                                EditorInfo.IME_ACTION_SEND);
                    }
                }

                // Auto-switch back (only in non-continuous sessions)
                if (prefAutoSwitch && !continuousMode) {
                    handler.post(() -> switchToPreviousInputMethod());
                    return;
                }

                // Continuous streaming — restart recording for next utterance
                if (continuousMode && !cancelRequested) {
                    handler.post(() -> startRecordingSession(true));
                } else {
                    handler.post(() -> {
                        btnCancel.setEnabled(false);
                        btnCancel.setAlpha(0.4f);
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
    // UI helpers
    // ═══════════════════════════════════════════════════════════════════════════

    /** Sizes the keyboard panel to match the typical system keyboard height. */
    private void setKeyboardHeight() {
        DisplayMetrics dm = getResources().getDisplayMetrics();
        boolean landscape = getResources().getConfiguration().orientation
                == Configuration.ORIENTATION_LANDSCAPE;
        int targetPx = (int) (dm.heightPixels * (landscape ? 0.60f : 0.44f));
        ViewGroup.LayoutParams lp = layoutKeyboardContent.getLayoutParams();
        lp.height = targetPx;
        layoutKeyboardContent.setLayoutParams(lp);
    }

    private void resetProgressUi() {
        processingBar.setIndeterminate(false);
        processingBar.setProgress(0);
        tvStatus.setVisibility(View.GONE);
    }

    private void updateTranslateIcon() {
        if (btnTranslate != null) {
            btnTranslate.setImageResource(translate
                    ? R.drawable.ic_english_on_36dp
                    : R.drawable.ic_english_off_36dp);
        }
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
        prefAutoStart  = sp.getBoolean(SettingsActivity.KEY_AUTO_START,  true);
        prefAutoStop   = sp.getBoolean(SettingsActivity.KEY_AUTO_STOP,   true);
        prefAutoSwitch = sp.getBoolean(SettingsActivity.KEY_AUTO_SWITCH, false);
        prefAutoSend   = sp.getBoolean(SettingsActivity.KEY_AUTO_SEND,   false);
    }
}
