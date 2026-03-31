package com.k2fsa.sherpa.onnx.tts.engine

import PreferenceHelper
import android.content.Context
import android.content.res.AssetManager
import android.util.Log
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.getOfflineTtsConfig
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

const val MIN_TTS_SPEED = 0.2f
const val MAX_TTS_SPEED = 3.0f
object TtsEngine {
    private const val TAG = "TtsEngine"

    // The OfflineTts instance. TtsService captures a local reference at the
    // start of each synthesis call, so setting this to null mid-synthesis is
    // safe — the ongoing call completes with the old instance, and the next
    // call will pick up the freshly initialised one.
    @Volatile var tts: OfflineTts? = null

    var lang:  String? = null
    var lang2: String? = null

    val speedState:              MutableState<Float>   = mutableFloatStateOf(1.0F)
    val speakerIdState:          MutableState<Int>     = mutableIntStateOf(0)
    val useSystemRatePitchState: MutableState<Boolean> = mutableStateOf(false)

    var speed: Float
        get() = speedState.value
        set(value) { speedState.value = value }

    var speakerId: Int
        get() = speakerIdState.value
        set(value) { speakerIdState.value = value }

    var useSystemRatePitch: Boolean
        get() = useSystemRatePitchState.value
        set(value) { useSystemRatePitchState.value = value }

    private var modelDir:          String? = null
    private var modelName:         String? = null
    private var acousticModelName: String? = null
    private var vocoder:           String? = null
    private var voices:            String? = null
    private var ruleFsts:          String? = null
    private var ruleFars:          String? = null
    private var lexicon:           String? = null
    private var dataDir:           String? = null
    private var assets:            AssetManager? = null
    private var isKitten           = false

    // Tracks whether the asset data directory has already been extracted so we
    // don't re-copy on every reinitialise.
    private var dataDirExtracted   = false
    private var extractedDataDir:  String? = null  // absolute path after extraction

    init {
        val bcModelDir  = BuildConfig.TTSENGINE_MODEL_DIR.trim()
        val bcModelName = BuildConfig.TTSENGINE_MODEL_NAME.trim()
        val bcVoices    = BuildConfig.TTSENGINE_VOICES.trim()
        val bcDataDir   = BuildConfig.TTSENGINE_DATA_DIR.trim()
        val bcLang      = BuildConfig.TTSENGINE_LANG.trim()
        val bcIsKitten  = BuildConfig.TTSENGINE_IS_KITTEN

        if (bcModelDir.isNotEmpty()) {
            modelDir  = bcModelDir
            modelName = bcModelName.ifEmpty { null }
            voices    = bcVoices.ifEmpty { null }
            dataDir   = bcDataDir.ifEmpty { null }
            lang      = bcLang.ifEmpty { "eng" }
            isKitten  = bcIsKitten
            Log.i(TAG, "Model from BuildConfig: dir=$modelDir name=$modelName voices=$voices")
        } else {
            modelName = null; acousticModelName = null; vocoder = null
            voices = null; modelDir = null; ruleFsts = null; ruleFars = null
            lexicon = null; dataDir = null; lang = null; lang2 = null

            // ── Uncomment ONE for local dev builds ──────────────────────────
            // kokoro-en-v0_19
            // modelDir  = "kokoro-en-v0_19"
            // modelName = "model.onnx"
            // voices    = "voices.bin"
            // dataDir   = "kokoro-en-v0_19/espeak-ng-data"
            // lang      = "eng"

            // kokoro-int8-en-v0_19
            // modelDir  = "kokoro-int8-en-v0_19"
            // modelName = "model.int8.onnx"
            // voices    = "voices.bin"
            // dataDir   = "kokoro-int8-en-v0_19/espeak-ng-data"
            // lang      = "eng"

            // kitten-nano-en-v0_2-fp16
            // modelDir  = "kitten-nano-en-v0_2-fp16"
            // modelName = "model.fp16.onnx"
            // voices    = "voices.bin"
            // dataDir   = "kitten-nano-en-v0_2-fp16/espeak-ng-data"
            // lang      = "eng"
            // isKitten  = true
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /** Create the TTS engine if it hasn't been created yet. */
    fun createTts(context: Context) {
        if (tts == null) {
            Log.i(TAG, "createTts: initialising")
            initTts(context)
        }
    }

    /**
     * Destroy and recreate the TTS engine with updated settings from prefs.
     * Call this after the user changes provider or thread count.
     * Must be called from a background thread — involves file I/O.
     */
    fun reinitialize(context: Context) {
        Log.i(TAG, "reinitialize: destroying old TTS instance")
        tts?.release()
        tts = null
        // Note: dataDirExtracted stays true — assets were already extracted to disk,
        // re-copying is unnecessary. initTts() reuses extractedDataDir directly.
        initTts(context)
    }

    // ── Private ───────────────────────────────────────────────────────────────

    private fun initTts(context: Context) {
        assets = context.assets

        // Extract asset data dir once; on reinitialise we reuse the existing path.
        val resolvedDataDir: String? = if (dataDir != null) {
            if (!dataDirExtracted) {
                val newDir = copyDataDir(context, dataDir!!)
                extractedDataDir = "$newDir/$dataDir"
                dataDirExtracted = true
            }
            extractedDataDir
        } else null

        val prefs = PreferenceHelper(context)
        speed              = prefs.getSpeed()
        speakerId          = prefs.getSid()
        useSystemRatePitch = prefs.getUseSystemRatePitch()

        val numThreads   = prefs.getNumThreads()
        val provider     = prefs.getProvider()
        val silenceScale = prefs.getSilenceScale()

        Log.i(TAG, "initTts: threads=$numThreads provider=$provider")

        // getOfflineTtsConfig() hard-codes provider="cpu" and sets its own thread
        // default.  Because OfflineTtsModelConfig uses var fields we can simply
        // override them on the returned config before constructing OfflineTts.
        val config = getOfflineTtsConfig(
            modelDir          = modelDir!!,
            modelName         = modelName         ?: "",
            acousticModelName = acousticModelName ?: "",
            vocoder           = vocoder           ?: "",
            voices            = voices            ?: "",
            lexicon           = lexicon           ?: "",
            dataDir           = resolvedDataDir   ?: "",
            dictDir           = "",
            ruleFsts          = ruleFsts          ?: "",
            ruleFars          = ruleFars          ?: "",
            numThreads        = numThreads,
            isKitten          = isKitten,
        )

        // Override fields that getOfflineTtsConfig hardcodes.
        config.model.provider = provider
        config.silenceScale   = silenceScale

        tts = OfflineTts(assetManager = assets, config = config)
        Log.i(TAG, "TTS ready: sampleRate=${tts!!.sampleRate()} " +
                "speakers=${tts!!.numSpeakers()} " +
                "threads=${config.model.numThreads} provider=${config.model.provider} " +
                "silenceScale=${config.silenceScale}")
    }

    private fun copyDataDir(context: Context, dataDir: String): String {
        Log.i(TAG, "Copying data dir: $dataDir")
        copyAssets(context, dataDir)
        return context.getExternalFilesDir(null)!!.absolutePath
    }

    private fun copyAssets(context: Context, path: String) {
        try {
            val list = context.assets.list(path)!!
            if (list.isEmpty()) {
                copyFile(context, path)
            } else {
                File("${context.getExternalFilesDir(null)}/$path").mkdirs()
                for (asset in list) {
                    copyAssets(context, if (path.isEmpty()) asset else "$path/$asset")
                }
            }
        } catch (ex: IOException) {
            Log.e(TAG, "Failed to copy $path: $ex")
        }
    }

    private fun copyFile(context: Context, filename: String) {
        try {
            val dest = "${context.getExternalFilesDir(null)}/$filename"
            context.assets.open(filename).use { ins ->
                FileOutputStream(dest).use { out ->
                    val buf = ByteArray(8192)
                    var n: Int
                    while (ins.read(buf).also { n = it } != -1) out.write(buf, 0, n)
                }
            }
        } catch (ex: Exception) {
            Log.e(TAG, "Failed to copy $filename: $ex")
        }
    }

}
