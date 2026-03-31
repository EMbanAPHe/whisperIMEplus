import android.content.Context
import android.content.SharedPreferences

class PreferenceHelper(context: Context) {

    private val PREFS_NAME               = "com.k2fsa.sherpa.onnx.tts.engine"
    private val SPEED_KEY                = "speed"
    private val SID_KEY                  = "speaker_id"
    private val USE_SYSTEM_RATE_PITCH_KEY = "use_system_rate_pitch"
    private val NUM_THREADS_KEY          = "num_threads"
    private val PROVIDER_KEY             = "provider"
    private val SILENCE_SCALE_KEY        = "silence_scale"

    private val sharedPreferences: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    // ── Speed ────────────────────────────────────────────────────────────────

    fun setSpeed(value: Float) =
        sharedPreferences.edit().putFloat(SPEED_KEY, value).apply()

    fun getSpeed(): Float =
        sharedPreferences.getFloat(SPEED_KEY, 1.0f)

    // ── Speaker ID ───────────────────────────────────────────────────────────

    fun setSid(value: Int) =
        sharedPreferences.edit().putInt(SID_KEY, value).apply()

    fun getSid(): Int =
        sharedPreferences.getInt(SID_KEY, 0)

    // ── System speed/pitch pass-through ──────────────────────────────────────

    fun setUseSystemRatePitch(value: Boolean) =
        sharedPreferences.edit().putBoolean(USE_SYSTEM_RATE_PITCH_KEY, value).apply()

    fun getUseSystemRatePitch(): Boolean =
        sharedPreferences.getBoolean(USE_SYSTEM_RATE_PITCH_KEY, false)

    // ── Thread count ─────────────────────────────────────────────────────────
    // Default 4 matches getOfflineTtsConfig's automatic choice for Kokoro/Kitten.
    // 1 or 2 threads often outperform 4 on big.LITTLE devices because ONNX
    // won't schedule work onto slow efficiency cores.
    // Valid values: 1, 2, 4

    fun setNumThreads(value: Int) =
        sharedPreferences.edit().putInt(NUM_THREADS_KEY, value).apply()

    fun getNumThreads(): Int =
        sharedPreferences.getInt(NUM_THREADS_KEY, 4)

    // ── Execution provider ───────────────────────────────────────────────────
    // "cpu"     — plain ONNX CPU (safe, always works)
    // "xnnpack" — Google XNNPack (NEON vectorised CPU, ~10-30% faster, no risk)
    // "nnapi"   — Android NNAPI (may route to NPU/GPU; requires android-27 build;
    //             falls back to CPU automatically if unsupported)

    fun setProvider(value: String) =
        sharedPreferences.edit().putString(PROVIDER_KEY, value).apply()

    fun getProvider(): String =
        sharedPreferences.getString(PROVIDER_KEY, "cpu") ?: "cpu"

    // ── Silence scale ────────────────────────────────────────────────────────
    // Controls how much of Kokoro's natural inter-clause silence is kept.
    // Applied at every clause boundary created by the text pre-splitter.
    // Range: 0.05 (very tight) → 0.3 (relaxed).  Default 0.2 = sherpa default.
    // Lower values reduce audible gaps at comma/semicolon splits but can make
    // boundary transitions sound slightly clipped.

    fun setSilenceScale(value: Float) =
        sharedPreferences.edit().putFloat(SILENCE_SCALE_KEY, value).apply()

    fun getSilenceScale(): Float =
        sharedPreferences.getFloat(SILENCE_SCALE_KEY, 0.2f)
}
