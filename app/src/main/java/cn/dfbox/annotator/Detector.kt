package cn.dfbox.annotator

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/** 单个检测框，坐标已还原到原图像素坐标 */
data class Detection(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val score: Float,
    val cls: Int
)

data class InferenceResult(
    val detections: List<Detection>,
    val preprocessMs: Long,
    val inferenceMs: Long,
    val postprocessMs: Long,
    val backend: String,
    val inputLayout: String
)

interface Detector {
    val backend: String
    fun detect(src: Bitmap, confThreshold: Float, iouThreshold: Float = 0.45f): InferenceResult
    fun close()
}

enum class DetectorKind(val label: String) {
    ONNX("ONNX"),
    TFLITE("TFLite")
}

/** letterbox 结果：640×640 输入图 + 还原到原图所需的缩放/填充参数 */
class Letterboxed(
    val bitmap: Bitmap,
    val scaleX: Float,
    val scaleY: Float,
    val padX: Float,
    val padY: Float
)

object Preprocess {

    const val INPUT_SIZE = 640
    private const val PAD_GRAY = 114

    fun letterbox(src: Bitmap, size: Int = INPUT_SIZE): Letterboxed {
        val w = src.width
        val h = src.height
        val scale = min(size.toFloat() / w, size.toFloat() / h)
        val nw = (w * scale + 0.5f).toInt().coerceIn(1, size)
        val nh = (h * scale + 0.5f).toInt().coerceIn(1, size)
        val padX = (size - nw) / 2f
        val padY = (size - nh) / 2f

        val out = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.drawColor(Color.rgb(PAD_GRAY, PAD_GRAY, PAD_GRAY))
        val paint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)
        val matrix = Matrix()
        matrix.setScale(nw.toFloat() / w, nh.toFloat() / h)
        matrix.postTranslate(padX, padY)
        canvas.drawBitmap(src, matrix, paint)

        return Letterboxed(out, nw.toFloat() / w, nh.toFloat() / h, padX, padY)
    }

    /** NHWC 排布，RGB 通道，/255 归一化 */
    fun fillNhwc(bitmap: Bitmap, dst: FloatArray, size: Int = INPUT_SIZE) {
        val px = IntArray(size * size)
        bitmap.getPixels(px, 0, size, 0, 0, size, size)
        var i = 0
        for (p in px) {
            dst[i++] = ((p shr 16) and 0xFF) / 255f
            dst[i++] = ((p shr 8) and 0xFF) / 255f
            dst[i++] = (p and 0xFF) / 255f
        }
    }

    /** NCHW 排布，RGB 通道，/255 归一化 */
    fun fillNchw(bitmap: Bitmap, dst: FloatArray, size: Int = INPUT_SIZE) {
        val px = IntArray(size * size)
        bitmap.getPixels(px, 0, size, 0, 0, size, size)
        val plane = size * size
        for (idx in px.indices) {
            val p = px[idx]
            dst[idx] = ((p shr 16) and 0xFF) / 255f
            dst[plane + idx] = ((p shr 8) and 0xFF) / 255f
            dst[2 * plane + idx] = (p and 0xFF) / 255f
        }
    }
}

object YoloPostProcess {

    /**
     * 解析 [1, 4+nc, anchors] 的 YOLO 输出。
     * 坐标单位自动判定：ONNX 导出通常为像素（0..imgsz），TFLite 导出为归一化（0..1）。
     */
    fun decode(
        out: FloatArray,
        channels: Int,
        anchors: Int,
        inputSize: Int,
        confThreshold: Float,
        iouThreshold: Float,
        lb: Letterboxed,
        imageWidth: Int,
        imageHeight: Int
    ): List<Detection> {
        val numClasses = channels - 4
        require(numClasses >= 1) { "输出通道数异常：$channels" }

        var maxAbsBox = 0f
        for (c in 0 until 4) {
            val base = c * anchors
            for (a in 0 until anchors) {
                val v = abs(out[base + a])
                if (v > maxAbsBox) maxAbsBox = v
            }
        }
        val unit = if (maxAbsBox <= 2f) inputSize.toFloat() else 1f

        val candidates = ArrayList<Detection>(64)
        for (a in 0 until anchors) {
            var bestScore = 0f
            var bestClass = -1
            for (c in 0 until numClasses) {
                val s = out[(4 + c) * anchors + a]
                if (s > bestScore) {
                    bestScore = s
                    bestClass = c
                }
            }
            if (bestClass < 0 || bestScore < confThreshold) continue

            val cx = out[a] * unit
            val cy = out[anchors + a] * unit
            val bw = out[2 * anchors + a] * unit
            val bh = out[3 * anchors + a] * unit

            var x1 = (cx - bw / 2f - lb.padX) / lb.scaleX
            var y1 = (cy - bh / 2f - lb.padY) / lb.scaleY
            var x2 = (cx + bw / 2f - lb.padX) / lb.scaleX
            var y2 = (cy + bh / 2f - lb.padY) / lb.scaleY

            x1 = x1.coerceIn(0f, imageWidth.toFloat())
            y1 = y1.coerceIn(0f, imageHeight.toFloat())
            x2 = x2.coerceIn(0f, imageWidth.toFloat())
            y2 = y2.coerceIn(0f, imageHeight.toFloat())

            if (x2 - x1 < 1f || y2 - y1 < 1f) continue
            candidates.add(Detection(x1, y1, x2, y2, bestScore, bestClass))
        }
        return nms(candidates, iouThreshold)
    }

    private fun iou(a: Detection, b: Detection): Float {
        val ix1 = max(a.x1, b.x1)
        val iy1 = max(a.y1, b.y1)
        val ix2 = min(a.x2, b.x2)
        val iy2 = min(a.y2, b.y2)
        val iw = ix2 - ix1
        val ih = iy2 - iy1
        if (iw <= 0f || ih <= 0f) return 0f
        val inter = iw * ih
        val union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
        return if (union <= 0f) 0f else inter / union
    }

    private fun nms(candidates: List<Detection>, iouThreshold: Float): List<Detection> {
        val sorted = candidates.sortedByDescending { it.score }
        val removed = BooleanArray(sorted.size)
        val kept = ArrayList<Detection>(sorted.size)
        for (i in sorted.indices) {
            if (removed[i]) continue
            val a = sorted[i]
            kept.add(a)
            for (j in i + 1 until sorted.size) {
                if (removed[j]) continue
                val b = sorted[j]
                if (a.cls == b.cls && iou(a, b) > iouThreshold) removed[j] = true
            }
        }
        return kept
    }
}

object Detectors {

    val LABELS = arrayOf("enemy", "teammate")

    const val ONNX_ASSET = "yolo11n_dfv15.onnx"
    const val TFLITE_ASSET = "yolo11n_dfv15.tflite"

    fun create(assets: AssetManager, kind: DetectorKind): Detector = when (kind) {
        DetectorKind.ONNX -> OrtDetector(assets, ONNX_ASSET)
        DetectorKind.TFLITE -> TfliteDetector(assets, TFLITE_ASSET)
    }
}
