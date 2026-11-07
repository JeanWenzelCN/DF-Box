package cn.dfbox.annotator

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.nio.ByteBuffer

/**
 * LiteRT 后端。NNAPI 优先：创建 delegate 失败或设备不支持时自动回落 XNNPACK CPU。
 */
class TfliteDetector(assets: AssetManager, backend: Backend) : Detector {

    private val modelBuffer: ByteBuffer = ModelAssets.load(assets)
    private val nnApiDelegate: NnApiDelegate?
    private val interpreter: Interpreter
    private val delegated: Boolean

    private val nhwc: Boolean
    private val inputSize: Int
    private val channels: Int
    private val anchors: Int
    private val transposed: Boolean
    private val inBuffer: ByteBuffer
    private val outBuffer: ByteBuffer

    override val backendLabel: String
        get() = if (delegated) "NNAPI" else "CPU"

    init {
        var delegate: NnApiDelegate? = null
        var created: Interpreter? = null

        if (backend == Backend.NNAPI) {
            try {
                val options = NnApiDelegate.Options().setUseNnapiCpu(false)
                delegate = NnApiDelegate(options)
                val interpreterOptions = Interpreter.Options()
                interpreterOptions.setNumThreads(threads())
                interpreterOptions.addDelegate(delegate)
                created = Interpreter(modelBuffer, interpreterOptions)
            } catch (t: Throwable) {
                runCatching { delegate?.close() }
                delegate = null
                created = null
            }
        }

        nnApiDelegate = delegate
        delegated = delegate != null
        interpreter = created ?: Interpreter(
            modelBuffer,
            Interpreter.Options().setNumThreads(threads())
        )

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

        inBuffer = ByteBuffer.allocateDirect(inputSize * inputSize * 3 * 4)
        outBuffer = ByteBuffer.allocateDirect(channels * anchors * 4)
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

        inBuffer.rewind()
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
            backendLabel = backendLabel,
            inputLayout = "$layout · float32 /255"
        )
    }

    override fun close() {
        runCatching { interpreter.close() }
        runCatching { nnApiDelegate?.close() }
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