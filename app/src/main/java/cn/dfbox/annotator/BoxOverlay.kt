package cn.dfbox.annotator

import android.content.Context
import android.graphics.Canvas
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

/** 覆盖在 ImageView 上的标注层，按 fitCenter 的显示区域换算坐标 */
class BoxOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var boxes: List<Detection> = emptyList()
    private var bitmapWidth = 0
    private var bitmapHeight = 0

    fun setResult(width: Int, height: Int, detections: List<Detection>) {
        bitmapWidth = width
        bitmapHeight = height
        boxes = detections
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (bitmapWidth <= 0 || bitmapHeight <= 0 || boxes.isEmpty()) return

        val scale = min(width.toFloat() / bitmapWidth, height.toFloat() / bitmapHeight)
        val offsetX = (width - bitmapWidth * scale) / 2f
        val offsetY = (height - bitmapHeight * scale) / 2f
        BoxRenderer.draw(
            canvas = canvas,
            boxes = boxes,
            sx = scale,
            sy = scale,
            offsetX = offsetX,
            offsetY = offsetY,
            strokeWidth = dp(2f),
            textSize = dp(11f),
            labels = Detectors.LABELS
        )
    }

    private fun dp(value: Float): Float = value * resources.displayMetrics.density
}
