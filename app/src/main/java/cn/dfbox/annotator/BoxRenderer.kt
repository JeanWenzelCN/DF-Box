package cn.dfbox.annotator

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import java.util.Locale

object BoxRenderer {

    private val CLASS_COLORS = intArrayOf(
        0xFFFF453A.toInt(), // enemy 红
        0xFF30D158.toInt(), // teammate 绿
        0xFF64D2FF.toInt(),
        0xFFFFD60A.toInt()
    )

    fun colorOf(cls: Int): Int {
        val index = ((cls % CLASS_COLORS.size) + CLASS_COLORS.size) % CLASS_COLORS.size
        return CLASS_COLORS[index]
    }

    fun draw(
        canvas: Canvas,
        boxes: List<Detection>,
        sx: Float,
        sy: Float,
        offsetX: Float,
        offsetY: Float,
        strokeWidth: Float,
        textSize: Float,
        labels: Array<String>
    ) {
        if (boxes.isEmpty()) return

        val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            this.strokeWidth = strokeWidth
        }
        val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            this.textSize = textSize
            typeface = Typeface.DEFAULT_BOLD
            color = Color.WHITE
        }
        val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }

        for (b in boxes) {
            val color = colorOf(b.cls)
            boxPaint.color = color
            fillPaint.color = color

            val left = offsetX + b.x1 * sx
            val top = offsetY + b.y1 * sy
            val right = offsetX + b.x2 * sx
            val bottom = offsetY + b.y2 * sy
            canvas.drawRect(left, top, right, bottom, boxPaint)

            val label = (labels.getOrNull(b.cls) ?: "cls${b.cls}") +
                String.format(Locale.US, " %.2f", b.score)
            val textWidth = textPaint.measureText(label)
            val labelHeight = textSize * 1.3f
            val padH = textSize * 0.35f
            val labelTop = if (top - labelHeight >= 0f) top - labelHeight else top
            canvas.drawRect(left, labelTop, left + textWidth + padH * 2f, labelTop + labelHeight, fillPaint)
            canvas.drawText(label, left + padH, labelTop + labelHeight - textSize * 0.3f, textPaint)
        }
    }
}
