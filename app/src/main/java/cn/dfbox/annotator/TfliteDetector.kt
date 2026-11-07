package cn.dfbox.annotator

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class TfliteDetector(assets: AssetManager, assetName: String) : Detector {

    override val backend: String = "TFLite"

    private val interpreter: Interpreter
    private val nhwc: Boolean
    private val inputSize: Int
    private val channels: Int
    private val anchors: Int
    private val transposed: Boolean
    private val outBuffer: ByteBuffer

    init {
        val model = loadModel(assets, assetName)
        val options = Interpreter.Options()
        options.setNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(2, 4))
        interpreter = Interpreter(model, options)

        val inTensor = interpreter.getInputTensor(0)
        require(inTensor.dataType() == DataType.FLOAT32) {
            "仅支持 float32 输入，当前为 ${inTensor.dataType()}"
        }
        val inShape = inTensor.shape()
        require(inShape.size == 4) { "输入维度异常：${inShape.joinToString()} " }
        nhwc = inShape[3] == 3
        inputSize = if (nhwc) inShape[1] else inShape[2]

        val outShape = interpreter.getOutputTensor(0).shape()
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
        outBuffer = ByteBuffer.allocateDirect(channels * anchors * 4).order(ByteOrder.nativeOrder())
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

        val inBuffer = ByteBuffer.allocateDirect(data.size * 4).order(ByteOrder.nativeOrder())
        inBuffer.asFloatBuffer().put(data)
        inBuffer.rewind()
        outBuffer.rewind()
        interpreter.run(inBuffer, outBuffer)
        outBuffer.rewind()
        val raw = FloatArray(channels * anchors)
        outBuffer.asFloatBuffer().get(raw)
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
        interpreter.close()
    }

    private fun loadModel(assets: AssetManager, assetName: String): ByteBuffer {
        val bytes = assets.open(assetName).use { it.readBytes() }
        val buffer = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
        buffer.put(bytes)
        buffer.rewind()
        return buffer
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
}
