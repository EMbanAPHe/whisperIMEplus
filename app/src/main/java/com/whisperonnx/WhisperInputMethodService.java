package com.whisperonnx;

import static com.whisperonnx.voice_translation.neural_networks.voice.Recognizer.ACTION_TRANSCRIBE;
import static com.whisperonnx.voice_translation.neural_networks.voice.Recognizer.ACTION_TRANSLATE;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.inputmethodservice.InputMethodService;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.ExtractedText;
import android.view.inputmethod.ExtractedTextRequest;
import android.widget.ImageButton;
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
 * Transcribro-style IME for Whisper+.
 *
 * Layout changes from the original:
 *  - Full-panel layout matching Transcribro's two-column design.
 *  - Left column: Settings + Keyboard (stacked), Cancel, large Mic.
 *  - Right column: Clear-unselected, Undo/Redo, Translate/Backspace, Send/Return.
 *  - Auto-mode removed (was rarely used and complicated the UI).
 *  - Lang1 / Lang2 buttons removed (use Settings → language preference instead).
 *
 * Threading notes:
 *  - Recorder.RecorderListener callbacks fire on the Recorder's worker thread.
 *    Every UI update inside the listener uses handler.post().
 *  - Whisper.WhisperListener.onResultReceived fires on Whisper's thread.
 *    InputConnection.commitText() is safe to call from any thread per Android docs,
 *    so we call it directly (same pattern as the original code).
 *  - Button click / touch handlers all run on the main thread.
 *  - Recorder.stop() is a blocking call (waits for the recording thread to finish).
 *    For the Cancel button we run it on a background thread to avoid a brief UI freeze.
 *    For the mic toggle button we keep it on the main thread (matches original, blocks
 *    for at most one audio buffer read ≈ 30 ms which is imperceptible).
 */
public class WhisperInputMethodService extends InputMethodService {

    private static final String TAG = "WhisperInputMethodService";

    // ── UI widgets ──────────────────────────────────────────────────────────────
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

    // ── State ────────────────────────────────────────────────────────────────────
    /**
     * Kept static so translate mode survives keyboard re-inflations within the same
     * app session, matching the behaviour of the original code.
     */
    private static boolean translate = false;

    /**
     * Tracks whether recording is active. Written on both the recording thread
     * (via RecorderListener) and the main thread (button clicks); declared volatile
     * so visibility is guaranteed across threads.
     */
    private volatile boolean isRecording = false;

    /**
     * Set by the Cancel button before calling Recorder.stop(). The RecorderListener
     * checks this flag to decide whether to start transcription after recording ends.
     * Volatile because it is written on the main thread and read on the recording thread.
     */
    private volatile boolean cancelRequested = false;

    // ── Core components ──────────────────────────────────────────────────────────
    private Recorder         mRecorder = null;
    private Whisper          mWhisper  = null;
    private SharedPreferences sp       = null;
    private final Handler handler      = new Handler(Looper.getMainLooper());
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
        deinitModel();
        // stop() is blocking; on destroy the system allows a brief pause.
        if (mRecorder != null && mRecorder.isInProgress()) {
            mRecorder.stop();
        }
        super.onDestroy();
    }

    @Override
    public void onStartInput(EditorInfo attribute, boolean restarting) {
        // When the target field has TYPE_NULL (e.g. password managers, terminal apps)
        // we release the model to save memory.
        if (attribute.inputType == EditorInfo.TYPE_NULL) {
            Log.d(TAG, "TYPE_NULL field — releasing model");
            deinitModel();
            if (mRecorder != null && mRecorder.isInProgress()) {
                mRecorder.stop();
            }
        }
    }

    @Override
    public void onStartInputView(EditorInfo attribute, boolean restarting) {
        if (mWhisper == null) initModel();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // IME view creation
    // ═══════════════════════════════════════════════════════════════════════════

    @Override
    @SuppressLint("ClickableViewAccessibility")
    public View onCreateInputView() {
        sp = PreferenceManager.getDefaultSharedPreferences(this);

        View view = getLayoutInflater().inflate(R.layout.voice_service, null);

        // Bind all widgets
        btnRecord    = view.findViewById(R.id.btnRecord);
        btnKeyboard  = view.findViewById(R.id.btnKeyboard);
        btnSettings  = view.findViewById(R.id.btnSettings);
        btnTranslate = view.findViewById(R.id.btnTranslate);
        btnCancel    = view.findViewById(R.id.btnCancel);
        btnClear     = view.findViewById(R.id.btnClear);
        btnUndo      = view.findViewById(R.id.btnUndo);
        btnRedo      = view.findViewById(R.id.btnRedo);
        btnDel       = view.findViewById(R.id.btnDel);
        btnSend      = view.findViewById(R.id.btnSend);
        btnEnter     = view.findViewById(R.id.btnEnter);
        processingBar = view.findViewById(R.id.processing_bar);
        tvStatus     = view.findViewById(R.id.tv_status);

        // Restore translate icon to match the current static state
        updateTranslateIcon();

        checkRecordPermission();

        // ── Recorder setup ──────────────────────────────────────────────────────
        mRecorder = new Recorder(this);
        mRecorder.setListener(message -> {
            // ⚠ This callback runs on the Recorder's worker thread — use handler.post() for UI.
            if (message.equals(Recorder.MSG_RECORDING)) {
                isRecording = true;
                handler.post(() -> {
                    btnRecord.setBackgroundResource(R.drawable.rounded_button_background_pressed);
                    btnCancel.setEnabled(true);
                    btnCancel.setAlpha(1.0f);
                });

            } else if (message.equals(Recorder.MSG_RECORDING_DONE)) {
                isRecording = false;
                HapticFeedback.vibrate(mContext);
                handler.post(() -> {
                    btnRecord.setBackgroundResource(R.drawable.rounded_button_background);
                    btnCancel.setEnabled(false);
                    btnCancel.setAlpha(0.4f);
                });
                if (!cancelRequested) {
                    startTranscription();   // safe to call from worker thread
                } else {
                    cancelRequested = false;
                    handler.post(() -> {
                        processingBar.setProgress(0);
                        processingBar.setIndeterminate(false);
                    });
                }

            } else if (message.equals(Recorder.MSG_RECORDING_ERROR)) {
                isRecording = false;
                HapticFeedback.vibrate(mContext);
                handler.post(() -> {
                    btnRecord.setBackgroundResource(R.drawable.rounded_button_background);
                    btnCancel.setEnabled(false);
                    btnCancel.setAlpha(0.4f);
                    tvStatus.setText(getString(R.string.error_no_input));
                    tvStatus.setVisibility(View.VISIBLE);
                    processingBar.setProgress(0);
                });
            }
        });

        // ── Mic button — tap to toggle ──────────────────────────────────────────
        //
        // When recording is active a tap calls Recorder.stop() which blocks briefly
        // (~one audio-buffer read ≈ 30 ms) before returning.  This matches the original
        // code's ACTION_UP handler and is imperceptible to the user.
        btnRecord.setOnClickListener(v -> {
            if (isRecording) {
                if (mRecorder != null && mRecorder.isInProgress()) {
                    mRecorder.stop(); // brief block on main thread, same as original
                }
            } else {
                if (!checkRecordPermission()) return;
                tvStatus.setVisibility(View.GONE);
                processingBar.setProgress(0);
                mRecorder.start();
            }
        });

        // ── Cancel — stops recording/transcription without committing text ───────
        //
        // We run Recorder.stop() on a background thread here because the user may
        // press Cancel while a long recording is still in progress; we don't want
        // to block the main thread while waiting for the audio buffer to drain.
        btnCancel.setOnClickListener(v -> {
            cancelRequested = true;
            // Stop any in-progress transcription first (fast, non-blocking)
            if (mWhisper != null) stopTranscription();
            // Stop recorder in background to avoid blocking the main thread
            if (mRecorder != null && mRecorder.isInProgress()) {
                new Thread(() -> mRecorder.stop()).start();
            }
            // Update UI immediately on main thread
            isRecording = false;
            btnRecord.setBackgroundResource(R.drawable.rounded_button_background);
            btnCancel.setEnabled(false);
            btnCancel.setAlpha(0.4f);
            processingBar.setProgress(0);
            processingBar.setIndeterminate(false);
            tvStatus.setVisibility(View.GONE);
        });

        // ── Settings — opens the existing SettingsActivity ───────────────────────
        btnSettings.setOnClickListener(v -> {
            Intent intent = new Intent(this, SettingsActivity.class);
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(intent);
        });

        // ── Translate toggle (keeps the static boolean in sync) ──────────────────
        btnTranslate.setOnClickListener(v -> {
            translate = !translate;
            updateTranslateIcon();
        });

        // ── Keyboard switch ──────────────────────────────────────────────────────
        btnKeyboard.setOnClickListener(v -> {
            if (mWhisper != null) stopTranscription();
            switchToPreviousInputMethod();
        });

        // ── Clear unselected — mirrors Transcribro's behaviour ───────────────────
        //
        // Deletes all text that is NOT currently selected (i.e. everything before
        // and after the cursor / selection).  Selected text is preserved so the
        // user can immediately replace it by speaking.
        btnClear.setOnClickListener(v -> {
            if (getCurrentInputConnection() == null) return;
            ExtractedText et = getCurrentInputConnection()
                    .getExtractedText(new ExtractedTextRequest(), 0);
            if (et != null && et.text != null) {
                int totalLen = et.text.length();
                CharSequence before = getCurrentInputConnection()
                        .getTextBeforeCursor(totalLen, 0);
                CharSequence after  = getCurrentInputConnection()
                        .getTextAfterCursor(totalLen, 0);
                if (before != null && after != null) {
                    getCurrentInputConnection()
                            .deleteSurroundingText(before.length(), after.length());
                }
            }
        });

        // ── Undo (Ctrl+Z) ────────────────────────────────────────────────────────
        btnUndo.setOnClickListener(v -> sendCtrlKey(KeyEvent.KEYCODE_Z, false));

        // ── Redo (Ctrl+Shift+Z) ──────────────────────────────────────────────────
        btnRedo.setOnClickListener(v -> sendCtrlKey(KeyEvent.KEYCODE_Z, true));

        // ── Backspace with initial delay then fast repeat ────────────────────────
        btnDel.setOnTouchListener(new View.OnTouchListener() {
            private Runnable initialRepeat;
            private Runnable fastRepeat;

            @Override
            public boolean onTouch(View v, MotionEvent event) {
                switch (event.getAction()) {
                    case MotionEvent.ACTION_DOWN:
                        doDelete();
                        initialRepeat = () -> {
                            doDelete();
                            fastRepeat = new Runnable() {
                                @Override
                                public void run() {
                                    doDelete();
                                    handler.postDelayed(this, 50);
                                }
                            };
                            handler.postDelayed(fastRepeat, 50);
                        };
                        handler.postDelayed(initialRepeat, 400);
                        break;
                    case MotionEvent.ACTION_UP:
                    case MotionEvent.ACTION_CANCEL:
                        if (initialRepeat != null) handler.removeCallbacks(initialRepeat);
                        if (fastRepeat   != null) handler.removeCallbacks(fastRepeat);
                        break;
                    default:
                        break;
                }
                return true;
            }
        });

        // ── Send — triggers the field's IME action (Send, Go, Search, Next…) ─────
        btnSend.setOnClickListener(v -> {
            if (getCurrentInputConnection() == null) return;
            EditorInfo ei = getCurrentInputEditorInfo();
            if (ei != null
                    && ei.actionId != 0
                    && ei.actionId != EditorInfo.IME_ACTION_NONE
                    && ei.actionId != EditorInfo.IME_ACTION_UNSPECIFIED) {
                getCurrentInputConnection().performEditorAction(ei.actionId);
            } else {
                // Fallback: send the generic SEND action (works in most messaging apps)
                getCurrentInputConnection().performEditorAction(EditorInfo.IME_ACTION_SEND);
            }
        });

        // ── Return / newline ─────────────────────────────────────────────────────
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
    // Helper methods
    // ═══════════════════════════════════════════════════════════════════════════

    /** Deletes the current selection, or the character before the cursor if nothing is selected. */
    private void doDelete() {
        if (getCurrentInputConnection() == null) return;
        CharSequence sel = getCurrentInputConnection().getSelectedText(0);
        if (sel != null && sel.length() > 0) {
            // Delete the selection by committing an empty string
            getCurrentInputConnection().commitText("", 1);
        } else {
            getCurrentInputConnection().deleteSurroundingText(1, 0);
        }
    }

    /** Sends a Ctrl (optionally +Shift) key event for undo/redo. */
    private void sendCtrlKey(int keyCode, boolean withShift) {
        if (getCurrentInputConnection() == null) return;
        int downMeta = KeyEvent.META_CTRL_ON | (withShift ? KeyEvent.META_SHIFT_ON : 0);
        long now = System.currentTimeMillis();
        getCurrentInputConnection().sendKeyEvent(
                new KeyEvent(now, now, KeyEvent.ACTION_DOWN, keyCode, 0, downMeta));
        getCurrentInputConnection().sendKeyEvent(
                new KeyEvent(now, now, KeyEvent.ACTION_UP,   keyCode, 0, 0));
    }

    /** Updates the translate button icon to reflect the current static translate flag. */
    private void updateTranslateIcon() {
        if (btnTranslate != null) {
            btnTranslate.setImageResource(
                    translate ? R.drawable.ic_english_on_36dp : R.drawable.ic_english_off_36dp);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Model and recording
    // ═══════════════════════════════════════════════════════════════════════════

    private void initModel() {
        mWhisper = new Whisper(this);
        mWhisper.loadModel();
        mWhisper.setListener(new Whisper.WhisperListener() {

            @Override
            public void onUpdateReceived(String message) {
                // Not used in this implementation.
            }

            @Override
            public void onResultReceived(WhisperResult whisperResult) {
                // ⚠ Runs on Whisper's inference thread.
                // UI updates go through handler; commitText() is thread-safe per Android docs.
                handler.post(() -> {
                    processingBar.setIndeterminate(false);
                    tvStatus.setVisibility(View.GONE);
                });

                String result = whisperResult.getResult();

                // Chinese script conversion (same as original)
                if ("zh".equals(whisperResult.getLanguage())) {
                    boolean simpleChinese = sp.getBoolean("simpleChinese", false);
                    result = simpleChinese
                            ? ZhConverterUtil.toSimple(result)
                            : ZhConverterUtil.toTraditional(result);
                }

                final String trimmed = result.trim();
                if (!trimmed.isEmpty() && getCurrentInputConnection() != null) {
                    // Append a trailing space so consecutive dictation doesn't merge words.
                    // InputConnection.commitText() is safe to call from a non-UI thread.
                    getCurrentInputConnection().commitText(trimmed + " ", 1);
                }
            }
        });
    }

    /**
     * Kicks off Whisper inference on the audio buffered by the Recorder.
     * Safe to call from the Recorder worker thread (Whisper runs its own thread internally).
     */
    private void startTranscription() {
        handler.post(() -> {
            processingBar.setProgress(0);
            processingBar.setIndeterminate(true);
        });
        if (mWhisper != null) {
            mWhisper.setAction(translate ? ACTION_TRANSLATE : ACTION_TRANSCRIBE);
            String langCode = sp.getString("language", "auto");
            Log.d(TAG, "startTranscription: lang=" + langCode + " translate=" + translate);
            mWhisper.setLanguage(langCode);
            mWhisper.start();
        }
    }

    private void stopTranscription() {
        handler.post(() -> processingBar.setIndeterminate(false));
        if (mWhisper != null) mWhisper.stop();
    }

    /**
     * Returns true if RECORD_AUDIO permission is granted.
     * If not, shows an error message.  Call only from the main thread.
     */
    private boolean checkRecordPermission() {
        int status = ContextCompat.checkSelfPermission(
                this, android.Manifest.permission.RECORD_AUDIO);
        if (status != PackageManager.PERMISSION_GRANTED) {
            tvStatus.setText(getString(R.string.need_record_audio_permission));
            tvStatus.setVisibility(View.VISIBLE);
        }
        return status == PackageManager.PERMISSION_GRANTED;
    }

    private void deinitModel() {
        if (mWhisper != null) {
            mWhisper.unloadModel();
            mWhisper = null;
        }
    }
}
