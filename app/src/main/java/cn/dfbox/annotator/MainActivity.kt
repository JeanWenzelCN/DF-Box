package cn.dfbox.annotator

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.ImageView
import android.widget.SeekBar
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.util.Locale
import kotlin.math.max

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var overlay: BoxOverlay
    private lateinit var hintView: TextView
    private lateinit var statusView: TextView
    private lateinit var saveButton: Button
    private lateinit var modelSpinner: Spinner
    private lateinit var confSeek: SeekBar
    private lateinit var confValue: TextView

    private var source: Bitmap? = null
    private var detections: List<Detection> = emptyList()
    private var inferJob: Job? = null

    private var detector: Detector? = null
    private var detectorKind: DetectorKind = DetectorKind.ONNX

    private val pickImage =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            if (uri != null) loadImage(uri)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        overlay = findViewById(R.id.overlay)
        hintView = findViewById(R.id.hintText)
        statusView = findViewById(R.id.statusText)
        saveButton = findViewById(R.id.btnSave)
        modelSpinner = findViewById(R.id.spinnerModel)
        confSeek = findViewById(R.id.seekConf)
        confValue = findViewById(R.id.textConf)

        findViewById<Button>(R.id.btnPick).setOnClickListener {
            pickImage.launch("image/*")
        }
        saveButton.setOnClickListener { saveAnnotated() }

        modelSpinner.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            DetectorKind.values().map { it.label }
        )
        modelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: AdapterView<*>?,
                view: View?,
                position: Int,
                id: Long
            ) {
                val kind = DetectorKind.values()[position]
                if (kind != detectorKind) {
                    detectorKind = kind
                    detector?.close()
                    detector = null
                    if (source != null) runInference()
                }
            }

            override fun onNothingSelected(parent: AdapterView<*>?) = Unit
        }

        confSeek.max = MAX_PERCENT - MIN_PERCENT
        confSeek.progress = 20
        confValue.text = formatConf(threshold())
        confSeek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                confValue.text = formatConf(threshold())
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit

            override fun onStopTrackingTouch(seekBar: SeekBar?) {
                if (source != null) runInference()
            }
        })

        statusView.text = getString(R.string.status_idle)
    }

    private fun threshold(): Float = (confSeek.progress + MIN_PERCENT) / 100f

    private fun formatConf(value: Float): String = String.format(Locale.US, "%.2f", value)

    private fun loadImage(uri: Uri) {
        lifecycleScope.launch {
            val bitmap = withContext(Dispatchers.IO) { decodeBitmap(uri, MAX_SIDE) }
            if (bitmap == null) {
                toast(getString(R.string.toast_decode_failed))
                return@launch
            }
            source = bitmap
            detections = emptyList()
            imageView.setImageBitmap(bitmap)
            overlay.setResult(bitmap.width, bitmap.height, emptyList())
            hintView.visibility = View.GONE
            saveButton.isEnabled = false
            runInference()
        }
    }

    private fun runInference() {
        val bitmap = source ?: return
        val conf = threshold()
        val kind = detectorKind
        inferJob?.cancel()
        statusView.text = getString(R.string.status_running)
        inferJob = lifecycleScope.launch {
            try {
                val engine = withContext(Dispatchers.Default) { obtainDetector(kind) }
                val result = withContext(Dispatchers.Default) { engine.detect(bitmap, conf) }
                detections = result.detections
                overlay.setResult(bitmap.width, bitmap.height, result.detections)
                saveButton.isEnabled = result.detections.isNotEmpty()
                statusView.text = buildStatus(result)
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                statusView.text = getString(
                    R.string.status_failed,
                    t.javaClass.simpleName,
                    t.message ?: ""
                )
            }
        }
    }

    private fun obtainDetector(kind: DetectorKind): Detector {
        val existing = detector
        if (existing != null && detectorKind == kind) return existing
        existing?.close()
        val created = Detectors.create(assets, kind)
        detector = created
        return created
    }

    private fun buildStatus(result: InferenceResult): String {
        val counts = result.detections.groupingBy { it.cls }.eachCount().toSortedMap()
        val detail = counts.entries.joinToString(" · ") { (cls, count) ->
            val name = Detectors.LABELS.getOrElse(cls) { "cls$cls" }
            "$name × $count"
        }
        val builder = StringBuilder()
        builder.append(result.backend)
            .append(" · ")
            .append(result.detections.size)
            .append(" 个目标")
        if (detail.isNotEmpty()) builder.append("（").append(detail).append("）")
        builder.append('\n')
            .append("预处理 ").append(result.preprocessMs).append(" ms")
            .append(" · 推理 ").append(result.inferenceMs).append(" ms")
            .append(" · 后处理 ").append(result.postprocessMs).append(" ms")
        builder.append('\n').append("输入 ").append(result.inputLayout)
        return builder.toString()
    }

    private fun saveAnnotated() {
        val bitmap = source ?: return
        val boxes = detections
        lifecycleScope.launch {
            val path = withContext(Dispatchers.IO) {
                val out = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
                val canvas = Canvas(out)
                canvas.drawBitmap(bitmap, 0f, 0f, null)
                val stroke = max(2f, bitmap.width / 320f)
                BoxRenderer.draw(
                    canvas = canvas,
                    boxes = boxes,
                    sx = 1f,
                    sy = 1f,
                    offsetX = 0f,
                    offsetY = 0f,
                    strokeWidth = stroke,
                    textSize = stroke * 5.5f,
                    labels = Detectors.LABELS
                )
                val dir = File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), "DFBox")
                dir.mkdirs()
                val file = File(dir, "annotated_" + System.currentTimeMillis() + ".jpg")
                FileOutputStream(file).use { out.compress(Bitmap.CompressFormat.JPEG, 92, it) }
                out.recycle()
                file.absolutePath
            }
            toast(getString(R.string.toast_saved, path))
            statusView.append("\n" + getString(R.string.status_saved, path))
        }
    }

    private fun decodeBitmap(uri: Uri, maxSide: Int): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxSide) sample *= 2
        val options = BitmapFactory.Options().apply { inSampleSize = sample }
        return contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, options)
        }
    }

    private fun toast(text: String) {
        Toast.makeText(this, text, Toast.LENGTH_LONG).show()
    }

    override fun onDestroy() {
        inferJob?.cancel()
        detector?.close()
        detector = null
        super.onDestroy()
    }

    private companion object {
        const val MAX_SIDE = 2048
        const val MIN_PERCENT = 5
        const val MAX_PERCENT = 95
    }
}