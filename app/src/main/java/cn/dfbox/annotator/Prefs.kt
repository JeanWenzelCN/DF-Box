package cn.dfbox.annotator

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint

class Prefs(context: Context) {

    private val sp = context.getSharedPreferences("dfbox_prefs", Context.MODE_PRIVATE)

    var backend: Backend
        get() = runCatching { Backend.valueOf(sp.getString(KEY_BACKEND, Backend.NNAPI.name)!!) }
            .getOrDefault(Backend.NNAPI)
        set(value) = sp.edit().putString(KEY_BACKEND, value.name).apply()

    /** 送进模型前的处理分辨率上限（长边），模型输入固定 640 */
    var procSize: Int
        get() = sp.getInt(KEY_PROC_SIZE, 640)
        set(value) = sp.edit().putInt(KEY_PROC_SIZE, value).apply()

    var conf: Float
        get() = sp.getFloat(KEY_CONF, 0.25f)
        set(value) = sp.edit().putFloat(KEY_CONF, value).apply()

    var iou: Float
        get() = sp.getFloat(KEY_IOU, 0.45f)
        set(value) = sp.edit().putFloat(KEY_IOU, value).apply()

    var onboarded: Boolean
        get() = sp.getBoolean(KEY_ONBOARDED, false)
        set(value) = sp.edit().putBoolean(KEY_ONBOARDED, value).apply()

    private companion object {
        const val KEY_BACKEND = "backend"
        const val KEY_PROC_SIZE = "proc_size"
        const val KEY_CONF = "conf"
        const val KEY_IOU = "iou"
        const val KEY_ONBOARDED = "onboarded"
    }
}

enum class Backend(val label: String, val detail: String) {
    NNAPI("NNAPI（NPU 优先）", "走系统 NNAPI，能下沉到 DSP/NPU 就用；未命中算子回落 CPU"),
    CPU("CPU（XNNPACK）", "纯 CPU 多线程，行为最稳定，方便做基线对比")
}

/** 测速用的固定测试图，没有选中图片时使用 */
object SampleImage {

    fun create(size: Int = 640): Bitmap {
        val bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        canvas.drawColor(Color.rgb(24, 28, 33))
        val paint = Paint(Paint.ANTI_ALIAS_FLAG)
        paint.color = Color.rgb(70, 120, 90)
        canvas.drawRect(size * 0.10f, size * 0.60f, size * 0.90f, size * 0.85f, paint)
        paint.color = Color.rgb(200, 110, 70)
        canvas.drawRect(size * 0.25f, size * 0.30f, size * 0.40f, size * 0.62f, paint)
        canvas.drawRect(size * 0.60f, size * 0.32f, size * 0.74f, size * 0.63f, paint)
        paint.color = Color.rgb(220, 220, 220)
        paint.textSize = size * 0.05f
        canvas.drawText("DF-Box bench", size * 0.34f, size * 0.18f, paint)
        return bitmap
    }
}