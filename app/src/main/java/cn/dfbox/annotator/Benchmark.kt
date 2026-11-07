package cn.dfbox.annotator

import android.content.res.AssetManager
import android.graphics.Bitmap

object Benchmark {

    data class Row(
        val backend: Backend,
        val firstRunMs: Long,
        val avgMs: Double,
        val p50Ms: Double,
        val p95Ms: Double,
        val fps: Double,
        val failure: String? = null
    )

    /**
     * 对每个后端各跑若干轮纯推理，返回耗时统计。
     * 使用同一张图、同一组阈值，保证可比。
     */
    fun run(
        assets: AssetManager,
        bitmap: Bitmap,
        conf: Float,
        iou: Float,
        warmup: Int = 3,
        runs: Int = 12
    ): List<Row> {
        val rows = ArrayList<Row>(Backend.values().size)
        for (backend in Backend.values()) {
            var detector: Detector? = null
            try {
                detector = TfliteDetector(assets, backend)
                var firstRun = 0L
                for (i in 0 until warmup) {
                    val result = detector.detect(bitmap, conf, iou)
                    if (i == 0) firstRun = result.inferenceMs
                }
                val samples = LongArray(runs)
                for (i in 0 until runs) {
                    samples[i] = detector.detect(bitmap, conf, iou).inferenceMs
                }
                samples.sort()
                val avg = samples.average()
                rows.add(
                    Row(
                        backend = backend,
                        firstRunMs = firstRun,
                        avgMs = avg,
                        p50Ms = samples[samples.size / 2].toDouble(),
                        p95Ms = samples[(samples.size * 95 / 100).coerceAtMost(samples.size - 1)].toDouble(),
                        fps = if (avg > 0) 1000.0 / avg else 0.0
                    )
                )
            } catch (t: Throwable) {
                rows.add(
                    Row(
                        backend = backend,
                        firstRunMs = -1,
                        avgMs = 0.0,
                        p50Ms = 0.0,
                        p95Ms = 0.0,
                        fps = 0.0,
                        failure = "${t.javaClass.simpleName}: ${t.message ?: ""}"
                    )
                )
            } finally {
                runCatching { detector?.close() }
            }
        }
        return rows
    }
}