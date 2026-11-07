package cn.dfbox.annotator

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.os.SystemClock
import java.nio.FloatBuffer

class OrtDetector(assets: AssetManager, assetName: String) : Detector {

    override val backend: String = "ONNX Runtime"

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private val nhwc: Boolean
    private val inputSize: Int
    private val channels: Int
    private val anchors: Int
    private val transposed: Boolean

    init {
        val modelBytes = assets.open(assetName).use { it.readBytes() }
        val options = OrtSession.SessionOptions()
        options.setIntraOpNumThreads(threads())
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        session = env.createSession(modelBytes, options)

        val input = session.inputInfo.entries.first()
        inputName = input.key
        val inShape = (input.value.info as TensorInfo).shape.map { it.toInt() }
        require(inShape.size == 4) { "输入维度异常：${inShape.joinToString()} " }
        nhwc = inShape[3] == 3
        inputSize = if (nhwc) inShape[1] else inShape[2]

        val outShape = (session.outputInfo.values.first().info as TensorInfo).shape.map { it.toInt() }
        require(outShape.size == 3) { "输出维度异常：${outShape.joinToString()} " }
        if (outShape[1] <= outShape[2]) {
            channels = outShape[1]
            anchors = outShape[2]
            transposed = false
        } else {
            channels = outShape[2]
            anchors = outShape[1]
            transposed = true
        }
    }

    override fun detect(src: Bitmap, confThreshold: Float, iouThreshold: Float): InferenceResult {
        val t0 = SystemClock.elapsedRealtime()
        val lb = Preprocess.letterbox(src, inputSize)
        val data = FloatArray(inputSize * inputSize * 3)
        if (nhwc) {
            Preprocess.fillNhwc(lb.bitmap, data, inputSize)
        } else {
            Preprocess.fillNchw(lb.bitmap, data, inputSize)
        }
        val layout = if (nhwc) {
            "NHWC ${inputSize}×${inputSize}×3"
        } else {
            "NCHW 3×${inputSize}×${inputSize}"
        }
        lb.bitmap.recycle()
        val t1 = SystemClock.elapsedRealtime()

        val shape = if (nhwc) {
            longArrayOf(1L, inputSize.toLong(), inputSize.toLong(), 3L)
        } else {
            longArrayOf(1L, 3L, inputSize.toLong(), inputSize.toLong())
        }
        var raw = FloatArray(0)
        OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape).use { input ->
            session.run(mapOf(inputName to input)).use { result ->
                val tensor = result.get(0) as OnnxTensor
                val buffer = tensor.floatBuffer
                raw = FloatArray(buffer.remaining())
                buffer.get(raw)
            }
        }
        val t2 = SystemClock.elapsedRealtime()

        val channelMajor = if (transposed) toChannelMajor(raw) else raw
        val detections = YoloPostProcess.decode(
            channelMajor, channels, anchors, inputSize,
            confThreshold, iouThreshold, lb, src.width, src.height
        )
        val t3 = SystemClock.elapsedRealtime()

        return InferenceResult(
            detections = detections,
            preprocessMs = t1 - t0,
            inferenceMs = t2 - t1,
            postprocessMs = t3 - t2,
            backend = backend,
            inputLayout = "$layout · float32 /255"
        )
    }

    override fun close() {
        session.close()
    }

    /** [1, anchors, channels] -> [1, channels, anchors] */
    private fun toChannelMajor(raw: FloatArray): FloatArray {
        val out = FloatArray(channels * anchors)
        for (a in 0 until anchors) {
            val srcBase = a * channels
            for (c in 0 until channels) {
                out[c * anchors + a] = raw[srcBase + c]
            }
        }
        return out
    }

    private fun threads(): Int = Runtime.getRuntime().availableProcessors().coerceIn(2, 4)
}
