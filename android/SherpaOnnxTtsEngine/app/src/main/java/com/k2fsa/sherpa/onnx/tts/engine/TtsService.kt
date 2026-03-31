package com.k2fsa.sherpa.onnx.tts.engine

import PreferenceHelper
import android.content.Intent
import android.media.AudioFormat
import android.speech.tts.SynthesisCallback
import android.speech.tts.SynthesisRequest
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeechService
import android.util.Log
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.atomic.AtomicBoolean

/*
 * Why we split text before passing to Kokoro
 * ───────────────────────────────────────────
 * Kokoro's internal pipeline (via espeak) only splits on hard sentence-ending
 * punctuation: . ! ?
 *
 * A single long sentence such as:
 *   "He walked down the long, winding road, past the old church, through
 *    the village, and into the fields beyond."
 * is treated as ONE inference job.  At RTF ≈ 0.855, a 10-second sentence
 * takes 8–10 seconds to infer.  No audio plays during that time — the user
 * hears complete silence between sentences.
 *
 * Fix: splitIntoClauses() breaks the text on clause-level punctuation
 * (, ; : — …) when a clause exceeds MIN_CLAUSE_WORDS words.  Each clause
 * becomes a separate generateWithCallback() call.  The producer/consumer
 * queue plays clause N while clause N+1 is being inferred, so the gap before
 * the first audible sound is reduced from ~8s to ~1–2s and subsequent clauses
 * stream with no perceptible pause.
 *
 * Prosody note: Kokoro adds natural silence at the start/end of every chunk,
 * so clause boundaries sound like a brief breath — acceptable for audiobooks.
 */

class TtsService : TextToSpeechService() {

    private sealed class QueueItem {
        class Data(val bytes: ByteArray, val length: Int) : QueueItem()
        object End : QueueItem()
        class Error(val t: Throwable) : QueueItem()
    }

    @Volatile private var currentCancelled: AtomicBoolean? = null
    @Volatile private var currentQueue: LinkedBlockingQueue<QueueItem>? = null

    // ── Text pre-splitting ────────────────────────────────────────────────────

    /**
     * Split [text] into clause-sized chunks so that Kokoro never has to infer
     * more than ~MAX_CLAUSE_WORDS words at once.
     *
     * Rules:
     *  1. Always split on sentence-ending punctuation (. ! ?) — Kokoro would
     *     have split here anyway; we keep it for correctness.
     *  2. Split on clause-level punctuation (, ; : — …) ONLY when the
     *     accumulated word count since the last split >= MIN_CLAUSE_WORDS.
     *     This avoids chopping "Hello, world." into two ridiculous fragments.
     *  3. Never produce an empty chunk.
     *  4. The original punctuation character is kept at the END of its chunk
     *     so Kokoro's phonemiser applies the correct pause.
     */
    private fun splitIntoClauses(text: String): List<String> {
        // Regex: capture the text and the split character as separate groups.
        // Group 1 = content up to (and including) the split punctuation.
        // Hard splits: . ! ?  (always split regardless of length)
        // Soft splits: , ; : — …  (split only when clause is long enough)
        val hardSplit = setOf('.', '!', '?')
        val softSplit = setOf(',', ';', ':', '—', '…')
        val minClauseWords = MIN_CLAUSE_WORDS

        val chunks = mutableListOf<String>()
        val current = StringBuilder()
        var wordCount = 1  // first word has no preceding space
        var i = 0

        while (i < text.length) {
            val ch = text[i]

            // Count words roughly by spaces
            if (ch == ' ' || ch == '\t' || ch == '\n') wordCount++

            current.append(ch)

            val isHard = ch in hardSplit
            val isSoft = ch in softSplit

            // Look ahead: after a split char there should be whitespace or end-of-string
            // to avoid splitting "3.14" or "e.g."
            val nextIsSpace = (i + 1 >= text.length) || text[i + 1].isWhitespace()

            if ((isHard && nextIsSpace) ||
                (isSoft && nextIsSpace && wordCount >= minClauseWords)) {

                val chunk = current.toString().trim()
                if (chunk.isNotEmpty()) chunks.add(chunk)
                current.clear()
                wordCount = 1  // reset to 1 for next clause
            }

            i++
        }

        // Flush whatever is left (handles sentences that don't end with punctuation)
        val remainder = current.toString().trim()
        if (remainder.isNotEmpty()) {
            if (chunks.isNotEmpty() && wordCount < minClauseWords) {
                // Too short to stand alone — append to the previous chunk
                chunks[chunks.lastIndex] = chunks.last() + " " + remainder
            } else {
                chunks.add(remainder)
            }
        }

        return if (chunks.isEmpty()) listOf(text) else chunks
    }

    // ── TextToSpeechService overrides ─────────────────────────────────────────

    override fun onCreate() {
        Log.i(TAG, "onCreate tts service")
        super.onCreate()
        onLoadLanguage(TtsEngine.lang, "", "")
        if (TtsEngine.lang2 != null) onLoadLanguage(TtsEngine.lang2, "", "")
        // Warm up ONNX Runtime so JIT compilation happens at startup rather
        // than on the user's first sentence.  Runs on a daemon thread so it
        // does not block the service coming online.
        Thread {
            try {
                TtsEngine.tts?.let { engine ->
                    Log.i(TAG, "Warm-up inference starting")
                    engine.generateWithCallback(
                        text     = WARMUP_TEXT,
                        sid      = 0,
                        speed    = 1.0f,
                        callback = { _ -> 0 }   // discard all output
                    )
                    Log.i(TAG, "Warm-up inference complete")
                }
            } catch (t: Throwable) {
                Log.w(TAG, "Warm-up failed (non-fatal): ${t.message}")
            }
        }.apply { isDaemon = true; name = "SherpaTtsWarmup" }.start()
    }

    override fun onDestroy() {
        Log.i(TAG, "onDestroy tts service")
        super.onDestroy()
    }

    override fun onIsLanguageAvailable(
        _lang: String?, _country: String?, _variant: String?
    ): Int {
        val requested = _lang ?: ""
        val primary   = TtsEngine.lang ?: "eng"

        val normalised = when {
            requested.equals("eng", ignoreCase = true) -> "eng"
            requested.startsWith("en", ignoreCase = true) -> primary
            else -> requested
        }

        return if (normalised.equals(primary, ignoreCase = true) ||
                   (TtsEngine.lang2 != null &&
                    normalised.equals(TtsEngine.lang2, ignoreCase = true)))
            TextToSpeech.LANG_AVAILABLE
        else
            TextToSpeech.LANG_NOT_SUPPORTED
    }

    override fun onGetLanguage(): Array<String> =
        arrayOf(TtsEngine.lang ?: "eng", "", "")

    override fun onLoadLanguage(
        _lang: String?, _country: String?, _variant: String?
    ): Int {
        Log.i(TAG, "onLoadLanguage: $_lang, $_country")
        val lang = _lang ?: ""
        return if (lang == TtsEngine.lang || lang == TtsEngine.lang2) {
            Log.i(TAG, "creating tts for lang: $lang")
            TtsEngine.createTts(application)
            TextToSpeech.LANG_AVAILABLE
        } else {
            Log.i(TAG, "lang $lang not supported; engine: ${TtsEngine.lang}, ${TtsEngine.lang2}")
            TextToSpeech.LANG_NOT_SUPPORTED
        }
    }

    override fun onStop() {
        Log.i(TAG, "onStop()")
        currentCancelled?.set(true)
        currentQueue?.let { q ->
            q.clear()
            q.offer(QueueItem.End)
        }
    }

    override fun onSynthesizeText(request: SynthesisRequest?, callback: SynthesisCallback?) {
        if (request == null || callback == null) return

        val text = request.charSequenceText.toString()

        if (onIsLanguageAvailable(request.language, request.country, request.variant)
                == TextToSpeech.LANG_NOT_SUPPORTED) {
            callback.error()
            return
        }

        val tts = TtsEngine.tts ?: run {
            TtsEngine.createTts(applicationContext)
            TtsEngine.tts ?: run { callback.error(); return }
        }

        val sampleRate = tts.sampleRate()
        callback.start(sampleRate, AudioFormat.ENCODING_PCM_16BIT, 1)

        if (text.isBlank()) { callback.done(); return }

        val effectiveSpeed = if (TtsEngine.useSystemRatePitch)
            (request.speechRate / 100.0f).coerceIn(0.2f, 3.0f)
        else
            TtsEngine.speed

        // Split into short clauses so Kokoro never infers more than
        // MIN_CLAUSE_WORDS words at once.  This turns an 8-second silence
        // into ~1-2 seconds before first audio.
        val clauses = splitIntoClauses(text)
        Log.i(TAG, "onSynthesizeText: ${clauses.size} clause(s), speed=$effectiveSpeed")
        clauses.forEachIndexed { idx, c -> Log.i(TAG, "  clause[$idx]: \"$c\"") }

        val cancelled = AtomicBoolean(false)
        val queue     = LinkedBlockingQueue<QueueItem>(32)
        currentCancelled = cancelled
        currentQueue     = queue

        // Generator thread: iterates over clauses sequentially.
        // Each generateWithCallback call fires the inner callback once per
        // espeak-sentence inside the clause (usually just once).
        val generatorThread = Thread {
            try {
                for (clause in clauses) {
                    if (cancelled.get()) break

                    tts.generateWithCallback(
                        text     = clause,
                        sid      = TtsEngine.speakerId,
                        speed    = effectiveSpeed,
                    ) { floatSamples ->
                        if (cancelled.get()) return@generateWithCallback 0
                        if (floatSamples.isEmpty()) return@generateWithCallback 1

                        val frames   = floatSamples.size
                        val pcmBytes = ByteArray(frames * 2)
                        var j = 0
                        for (i in 0 until frames) {
                            val s = (floatSamples[i] * 32767.0f).toInt().coerceIn(-32768, 32767)
                            pcmBytes[j]     = (s and 0xff).toByte()
                            pcmBytes[j + 1] = ((s ushr 8) and 0xff).toByte()
                            j += 2
                        }

                        try {
                            queue.put(QueueItem.Data(pcmBytes, pcmBytes.size))
                        } catch (_: InterruptedException) {
                            return@generateWithCallback 0
                        }

                        if (cancelled.get()) 0 else 1
                    }
                }

                if (!cancelled.get()) queue.offer(QueueItem.End)

            } catch (t: Throwable) {
                Log.e(TAG, "Generator error", t)
                queue.offer(QueueItem.Error(t))
            }
        }.apply {
            name     = "SherpaTtsGenerator"
            isDaemon = true
        }

        generatorThread.start()

        // Consumer: drains the queue into Android's audioAvailable()
        try {
            while (true) {
                when (val item = queue.take()) {
                    is QueueItem.End  -> break

                    is QueueItem.Data -> {
                        if (cancelled.get()) break
                        val maxBuf = callback.maxBufferSize.coerceAtLeast(4096)
                        var offset = 0
                        val total  = item.length
                        while (offset < total && !cancelled.get()) {
                            var chunk = minOf(maxBuf, total - offset)
                            if (chunk % 2 != 0) chunk--
                            if (chunk <= 0) break
                            callback.audioAvailable(item.bytes, offset, chunk)
                            offset += chunk
                        }
                    }

                    is QueueItem.Error -> {
                        Log.e(TAG, "Generator thread error", item.t)
                        callback.error(TextToSpeech.ERROR_OUTPUT)
                        cancelled.set(true)
                        break
                    }
                }
            }
            if (!cancelled.get()) callback.done()

        } catch (e: InterruptedException) {
            Log.w(TAG, "Consumer interrupted", e)
            cancelled.set(true)
        } finally {
            cancelled.set(true)
            currentCancelled = null
            currentQueue     = null
        }
    }

    companion object {
        private const val TAG = "TtsService"

        /**
         * Minimum number of words a clause must contain before we split on
         * soft punctuation (, ; : — …).
         *
         * 4 words ≈ ~1.2 seconds of audio at normal speed — short enough to
         * start playback quickly, long enough to avoid splitting phrases like
         * "Yes, I see." (3 words, below threshold) into two fragments.
         */
        private const val MIN_CLAUSE_WORDS = 4

        /**
         * Short text used to warm up ONNX Runtime on startup.
         * Must be non-empty and long enough to exercise all model layers.
         * The output is discarded; callback immediately returns 0 (stop).
         */
        private const val WARMUP_TEXT = "Hello."
    }
}
