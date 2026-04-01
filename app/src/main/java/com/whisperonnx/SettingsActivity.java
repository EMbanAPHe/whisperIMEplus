package com.whisperonnx;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.util.Pair;
import android.view.MenuItem;
import android.view.View;
import android.widget.AdapterView;
import android.widget.CheckBox;
import android.widget.ImageButton;
import android.widget.Spinner;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.SwitchCompat;
import androidx.core.content.ContextCompat;
import androidx.preference.PreferenceManager;

import com.google.android.material.slider.RangeSlider;
import com.whisperonnx.utils.LanguagePairAdapter;
import com.whisperonnx.utils.ThemeUtils;

import java.util.ArrayList;
import java.util.List;

public class SettingsActivity extends AppCompatActivity {
    private static final String TAG = "SettingsActivity";

    // ── Preference keys ────────────────────────────────────────────────────────
    // KEY_LAUNCH_LISTENING — new, separate: open keyboard in listening mode
    public static final String KEY_LAUNCH_LISTENING = "imeLaunchListening";
    // KEY_AUTO_START kept for backward compat; not used by IME any more.
    public static final String KEY_AUTO_START  = "imeAutoStart";
    public static final String KEY_AUTO_STOP   = "imeAutoStop";
    public static final String KEY_AUTO_SWITCH = "imeAutoSwitch";
    public static final String KEY_AUTO_SEND   = "imeAutoSend";

    private SharedPreferences sp = null;
    private Spinner spinnerLanguage;
    private Spinner spinnerLanguage1IME;
    private Spinner spinnerLanguage2IME;
    private CheckBox modeSimpleChinese;
    private CheckBox modeSimpleChineseIME;
    private RangeSlider minSilence;
    private int langSelected;

    @SuppressLint("ClickableViewAccessibility")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_settings);
        ThemeUtils.setStatusBarAppearance(this);
        ActionBar actionBar = getSupportActionBar();
        if (actionBar != null) actionBar.setDisplayHomeAsUpEnabled(true);

        sp = PreferenceManager.getDefaultSharedPreferences(this);

        // ── Language slot buttons (existing behaviour, unchanged) ──────────────
        if (!sp.contains("langSelected")) {
            String langCodeIME = sp.getString("language", "auto");
            sp.edit()
                    .putInt("langSelected", 1)
                    .putString("language1", langCodeIME)
                    .putString("language2", "auto")
                    .apply();
        }

        ImageButton btnLang1 = findViewById(R.id.btnLang1);
        ImageButton btnLang2 = findViewById(R.id.btnLang2);
        langSelected = sp.getInt("langSelected", 1);
        updateLangButtons(btnLang1, btnLang2);

        btnLang1.setOnClickListener(v -> {
            String lang = sp.getString("language1", "auto");
            sp.edit().putInt("langSelected", 1).putString("language", lang).apply();
            langSelected = 1;
            updateLangButtons(btnLang1, btnLang2);
        });
        btnLang2.setOnClickListener(v -> {
            String lang = sp.getString("language2", "auto");
            sp.edit().putInt("langSelected", 2).putString("language", lang).apply();
            langSelected = 2;
            updateLangButtons(btnLang1, btnLang2);
        });

        // ── IME language spinners (existing behaviour, unchanged) ───────────────
        spinnerLanguage1IME = findViewById(R.id.spnrLanguage1_ime);
        spinnerLanguage2IME = findViewById(R.id.spnrLanguage2_ime);
        List<Pair<String, String>> languagePairs = LanguagePairAdapter.getLanguagePairs(this);

        LanguagePairAdapter adapter1 = new LanguagePairAdapter(
                this, android.R.layout.simple_spinner_item, languagePairs);
        LanguagePairAdapter adapter2 = new LanguagePairAdapter(
                this, android.R.layout.simple_spinner_item, languagePairs);
        adapter1.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        adapter2.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerLanguage1IME.setAdapter(adapter1);
        spinnerLanguage2IME.setAdapter(adapter2);
        spinnerLanguage1IME.setSelection(adapter1.getIndexByCode(sp.getString("language1", "auto")));
        spinnerLanguage2IME.setSelection(adapter2.getIndexByCode(sp.getString("language2", "auto")));

        spinnerLanguage1IME.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(AdapterView<?> p, View v, int i, long id) {
                sp.edit().putString("language1", languagePairs.get(i).first).apply();
                if (langSelected == 1)
                    sp.edit().putString("language", languagePairs.get(i).first).apply();
            }
            @Override public void onNothingSelected(AdapterView<?> p) {}
        });
        spinnerLanguage2IME.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(AdapterView<?> p, View v, int i, long id) {
                sp.edit().putString("language2", languagePairs.get(i).first).apply();
                if (langSelected == 2)
                    sp.edit().putString("language", languagePairs.get(i).first).apply();
            }
            @Override public void onNothingSelected(AdapterView<?> p) {}
        });

        modeSimpleChineseIME = findViewById(R.id.mode_simple_chinese_ime);
        modeSimpleChineseIME.setChecked(sp.getBoolean("simpleChinese", false));
        modeSimpleChineseIME.setOnCheckedChangeListener((btn, checked) ->
                sp.edit().putBoolean("simpleChinese", checked).apply());

        // ── Voice keyboard switches ────────────────────────────────────────────
        // Launch in listening mode — new, separate toggle
        wireSwitchCompat(R.id.switch_launch_listening, KEY_LAUNCH_LISTENING, true);
        // Streaming / auto-stop
        wireSwitchCompat(R.id.switch_auto_stop,   KEY_AUTO_STOP,   true);
        wireSwitchCompat(R.id.switch_auto_switch, KEY_AUTO_SWITCH, false);
        wireSwitchCompat(R.id.switch_auto_send,   KEY_AUTO_SEND,   false);

        // ── Recognition-service language spinner (existing, unchanged) ──────────
        spinnerLanguage = findViewById(R.id.spnrLanguage);
        LanguagePairAdapter adapterRec = new LanguagePairAdapter(
                this, android.R.layout.simple_spinner_item, languagePairs);
        adapterRec.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerLanguage.setAdapter(adapterRec);
        spinnerLanguage.setSelection(
                adapterRec.getIndexByCode(sp.getString("recognitionServiceLanguage", "auto")));
        spinnerLanguage.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(AdapterView<?> p, View v, int i, long id) {
                sp.edit().putString("recognitionServiceLanguage", languagePairs.get(i).first).apply();
            }
            @Override public void onNothingSelected(AdapterView<?> p) {}
        });

        modeSimpleChinese = findViewById(R.id.mode_simple_chinese);
        modeSimpleChinese.setChecked(sp.getBoolean("RecognitionServiceSimpleChinese", false));
        modeSimpleChinese.setOnCheckedChangeListener((btn, checked) ->
                sp.edit().putBoolean("RecognitionServiceSimpleChinese", checked).apply());

        // ── Min-silence slider (existing, unchanged) ───────────────────────────
        minSilence = findViewById(R.id.settings_min_silence);
        minSilence.setValues((float) sp.getInt("silenceDurationMs", 800));
        minSilence.addOnChangeListener((@NonNull RangeSlider slider, float value, boolean fromUser) ->
                sp.edit().putInt("silenceDurationMs", (int) value).apply());

        checkPermissions();
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    private void wireSwitchCompat(int viewId, String prefKey, boolean defaultValue) {
        SwitchCompat sw = findViewById(viewId);
        if (sw == null) return;
        sw.setChecked(sp.getBoolean(prefKey, defaultValue));
        sw.setOnCheckedChangeListener((btn, checked) ->
                sp.edit().putBoolean(prefKey, checked).apply());
    }

    private void updateLangButtons(ImageButton b1, ImageButton b2) {
        b1.setImageResource(langSelected == 1
                ? R.drawable.ic_counter_1_on_36dp : R.drawable.ic_counter_1_off_36dp);
        b2.setImageResource(langSelected == 2
                ? R.drawable.ic_counter_2_on_36dp : R.drawable.ic_counter_2_off_36dp);
    }

    private void checkPermissions() {
        List<String> perms = new ArrayList<>();
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            perms.add(Manifest.permission.RECORD_AUDIO);
            Toast.makeText(this, getString(R.string.need_record_audio_permission),
                    Toast.LENGTH_SHORT).show();
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU
                && ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
            perms.add(Manifest.permission.POST_NOTIFICATIONS);
        }
        if (!perms.isEmpty())
            requestPermissions(perms.toArray(new String[0]), 0);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        Log.d(TAG, "Record permission " +
                (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED
                        ? "granted" : "not granted"));
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item.getItemId() == android.R.id.home) { finish(); return true; }
        return super.onOptionsItemSelected(item);
    }
}
