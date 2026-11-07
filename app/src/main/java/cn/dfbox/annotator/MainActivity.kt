package cn.dfbox.annotator

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.slider.Slider
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

    private lateinit var toolbar: MaterialToolbar
    private lateinit var imageView: ImageView
    private lateinit var overlay: BoxOverlay
    private lateinit var hintView: TextView
    private lateinit var statusView: TextView
    private lateinit var confSlider: Slider
    private lateinit var confValue: TextView
    private lateinit var saveButton: Button

    private lateinit var prefs: Prefs

    private var original: Bitmap? = null
    private var source: Bitmap? = null
    private var detections: List<Detection> = emptyList()
    private var lastResult: InferenceResult? = null
    private var inferJob: Job? = null

    private var detector: TfliteDetector? = null
    private var detectorBackend: Backend? = null

    private val pickImage =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            if (uri != null) loadImage(uri)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        prefs = Prefs(this)

        toolbar = findViewById(R.id.toolbar)
        imageView = findViewById(R.id.imageView)
        overlay = findViewById(R.id.overlay)
        hintView = findViewById(R.id.hintText)
        statusView = findViewById(R.id.statusText)
        confSlider = findViewById(R.id.sliderConf)
        confValue = findViewById(R.id.textConf)
        saveButton = findViewById(R.id.btnSave)

        toolbar.setOnMenuItemClickListener { item ->
            when (item.itemId) {
                R.id.action_settings -> {
                    settingsDialog()
                    true
                }
                R.id.action_bench -> {
                    benchmarkDialog()
                    true
                }
                else -> false
            }
        }

        findViewById<Button>(R.id.btnPick).setOnClickListener { pickImage.launch("image/*") }
        saveButton.setOnClickListener { saveAnnotated() }

        confSlider.valueFrom = 0.05f
        confSlider.valueTo = 0.95f
        confSlider.stepSize = 0.05f
        confSlider.value = prefs.conf.coerceIn(0.05f, 0.95f)
        confValue.text = format(prefs.conf)
        confSlider.addOnChangeListener { _, value, fromUser ->
            if (fromUser) confValue.text = format(value)
        }
        confSlider.addOnSliderTouchListener(object : Slider.OnSliderTouchListener {
            override fun onStartTrackingTouch(slider: Slider) = Unit
            override fun onStopTrackingTouch(slider: Slider) {
                prefs.conf = slider.value
                if (source != null) predict()
            }
        })

        updateIdleStatus()
        if (!prefs.onboarded) showOnboarding()
    }

    // ---------- 推理 ----------

    private fun predict() {
        val bitmap = source ?: return
        val conf = prefs.conf
        val iou = prefs.iou
        inferJob?.cancel()
        statusView.text = getString(R.string.status_running)
        inferJob = lifecycleScope.launch {
            try {
                val engine = withContext(Dispatchers.Default) { obtainDetector() }
                val result = withContext(Dispatchers.Default) { engine.detect(bitmap, conf, iou) }
                lastResult = result
                detections = result.detections
                overlay.setResult(bitmap.width, bitmap.height, result.detections)
                saveButton.isEnabled = result.detections.isNotEmpty()
                statusView.text = statusText(result, bitmap)
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

    private fun obtainDetector(): TfliteDetector {
        val backend = prefs.backend
        val existing = detector
        if (existing != null && detectorBackend == backend) return existing
        existing?.close()
        val created = TfliteDetector(assets, backend)
        detector = created
        detectorBackend = backend
        return created
    }

    private fun invalidateDetector() {
        inferJob?.cancel()
        detector?.close()
        detector = null
        detectorBackend = null
    }

    private fun statusText(result: InferenceResult, bitmap: Bitmap): String {
        val counts = result.detections.groupingBy { it.cls }.eachCount().toSortedMap()
        val detail = counts.entries.joinToString(" · ") { (cls, count) ->
            val name = ModelAssets.LABELS.getOrElse(cls) { "cls$cls" }
            "$name × $count"
        }
        return buildString {
            append(result.backendLabel).append(" · ").append(result.detections.size).append(" 个目标")
            if (detail.isNotEmpty()) append("（").append(detail).append("）")
            append('\n').append("预处理 ").append(result.preprocessMs).append(" ms")
            append(" · 推理 ").append(result.inferenceMs).append(" ms")
            append(" · 后处理 ").append(result.postprocessMs).append(" ms")
            append('\n').append("输入 ").append(bitmap.width).append('×').append(bitmap.height)
            append(" → ").append(result.inputLayout)
        }
    }

    private fun updateIdleStatus() {
        statusView.text = getString(
            R.string.status_idle,
            prefs.backend.label,
            prefs.procSize
        )
    }

    // ---------- 图片 ----------

    private fun loadImage(uri: Uri) {
        lifecycleScope.launch {
            val decoded = withContext(Dispatchers.IO) { decodeBitmap(uri) }
            if (decoded == null) {
                toast(getString(R.string.toast_decode_failed))
                return@launch
            }
            original = decoded
            val scaled = scaleTo(decoded, prefs.procSize)
            source = scaled
            detections = emptyList()
            imageView.setImageBitmap(scaled)
            overlay.setResult(scaled.width, scaled.height, emptyList())
            hintView.visibility = View.GONE
            saveButton.isEnabled = false
            predict()
        }
    }

    private fun decodeBitmap(uri: Uri): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / (sample * 2) >= MAX_SIDE) sample *= 2
        val options = BitmapFactory.Options().apply { inSampleSize = sample }
        return contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, options)
        }
    }

    private fun scaleTo(bitmap: Bitmap, limit: Int): Bitmap {
        val maxSide = max(bitmap.width, bitmap.height)
        if (maxSide <= limit) return bitmap
        val scale = limit.toFloat() / maxSide
        val w = (bitmap.width * scale).toInt().coerceAtLeast(1)
        val h = (bitmap.height * scale).toInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(bitmap, w, h, true)
    }

    // ---------- 设置 / 测速 / 引导 ----------

    private fun showOnboarding() {
        MaterialAlertDialogBuilder(this)
            .setTitle(R.string.onboard_title)
            .setMessage(R.string.onboard_message)
            .setPositiveButton(R.string.onboard_go) { _, _ ->
                prefs.onboarded = true
                settingsDialog()
            }
            .setNegativeButton(R.string.onboard_later) { _, _ ->
                prefs.onboarded = true
            }
            .show()
    }

    private fun settingsDialog() {
        val density = resources.displayMetrics.density
        fun dp(value: Int) = (value * density).toInt()

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(24), dp(4), dp(24), 0)
        }

        root.addView(sectionLabel(getString(R.string.label_backend)))
        val group = RadioGroup(this)
        Backend.values().forEach { backend ->
            val radio = RadioButton(this)
            radio.id = View.generateViewId()
            radio.tag = backend
            radio.text = "${backend.label} — ${backend.detail}"
            radio.textSize = 13f
            radio.isChecked = prefs.backend == backend
            group.addView(radio)
        }
        root.addView(group)

        root.addView(sectionLabel(getString(R.string.label_resolution)))
        val sizes = intArrayOf(320, 480, 640, 960)
        val spinner = Spinner(this)
        spinner.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            sizes.map { "$it px" }
        )
        val current = sizes.indexOf(prefs.procSize)
        spinner.setSelection(if (current < 0) 2 else current)
        root.addView(spinner)

        val confLabel = sectionLabel("${getString(R.string.label_conf)} ${format(prefs.conf)}")
        root.addView(confLabel)
        val dialogSlider = Slider(this).apply {
            valueFrom = 0.05f
            valueTo = 0.95f
            stepSize = 0.05f
            value = prefs.conf.coerceIn(0.05f, 0.95f)
        }
        dialogSlider.addOnChangeListener { _, value, _ ->
            confLabel.text = "${getString(R.string.label_conf)} ${format(value)}"
        }
        root.addView(dialogSlider)

        MaterialAlertDialogBuilder(this)
            .setTitle(R.string.settings_title)
            .setView(root)
            .setPositiveButton(R.string.apply) { _, _ ->
                (group.findViewById<RadioButton>(group.checkedRadioButtonId)?.tag as? Backend)
                    ?.let { prefs.backend = it }
                prefs.procSize = sizes[spinner.selectedItemPosition]
                prefs.conf = dialogSlider.value
                confSlider.value = prefs.conf
                confValue.text = format(prefs.conf)
                invalidateDetector()

                val base = original
                if (base != null) {
                    val rescaled = scaleTo(base, prefs.procSize)
                    source = rescaled
                    imageView.setImageBitmap(rescaled)
                    overlay.setResult(rescaled.width, rescaled.height, emptyList())
                    saveButton.isEnabled = false
                    predict()
                } else {
                    updateIdleStatus()
                }
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun benchmarkDialog() {
        val bitmap = source ?: SampleImage.create()
        statusView.text = getString(R.string.status_bench)
        lifecycleScope.launch {
            val rows = withContext(Dispatchers.Default) {
                Benchmark.run(assets, bitmap, prefs.conf, prefs.iou)
            }
            statusView.text = lastResult?.let { statusText(it, source ?: bitmap) }
                ?: getString(R.string.status_idle, prefs.backend.label, prefs.procSize)

            val message = buildString {
                rows.forEach { row ->
                    if (row.failure != null) {
                        append(row.backend.label).append("：失败（").append(row.failure).append("）\n")
                    } else {
                        append(row.backend.label).append('\n')
                        append("  avg ").append(String.format(Locale.US, "%.1f", row.avgMs)).append(" ms")
                        append(" · p50 ").append(String.format(Locale.US, "%.1f", row.p50Ms)).append(" ms")
                        append(" · p95 ").append(String.format(Locale.US, "%.1f", row.p95Ms)).append(" ms")
                        append(" · ").append(String.format(Locale.US, "%.1f", row.fps)).append(" fps\n")
                        append("  首次 ").append(row.firstRunMs).append(" ms\n")
                    }
                }
            }
            val fastest = rows.filter { it.failure == null }.minByOrNull { it.avgMs }

            MaterialAlertDialogBuilder(this@MainActivity)
                .setTitle(getString(R.string.bench_title))
                .setMessage(message.trim())
                .setPositiveButton(getString(R.string.bench_apply)) { _, _ ->
                    fastest?.let { row ->
                        prefs.backend = row.backend
                        invalidateDetector()
                        if (source != null) predict() else updateIdleStatus()
                    }
                }
                .setNegativeButton(R.string.close, null)
                .show()
        }
    }

    // ---------- 保存 ----------

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
                    labels = ModelAssets.LABELS
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

    // ---------- 杂项 ----------

    private fun sectionLabel(text: String): TextView = TextView(this).apply {
        this.text = text
        textSize = 13f
        setPadding(0, (12 * resources.displayMetrics.density).toInt(), 0, (2 * resources.displayMetrics.density).toInt())
    }

    private fun format(value: Float): String = String.format(Locale.US, "%.2f", value)

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
    }
}