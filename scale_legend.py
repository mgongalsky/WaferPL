"""Compact legends showing the actual raw-value to color mapping."""
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import QWidget
from wafer_data import render_pixels


class ScaleLegend(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []
        self.locked = False
        self.setFixedWidth(126)

    def set_scales(self, panel, locked):
        self.locked = locked
        self.rows = [(mode, panel.styles.get(mode), mode in panel.layers)
                     for mode in sorted(panel.selected_modes)]
        self.setMinimumHeight(30 + 94 * len(self.rows))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        painter.setPen(self.palette().windowText().color())
        painter.drawText(5, 18, 'Общая шкала' if self.locked else 'Шкала RAW')
        for index, (mode, style, present) in enumerate(self.rows):
            y = 34 + index * 94
            painter.setPen(self.palette().windowText().color())
            name = painter.fontMetrics().elidedText(mode, Qt.TextElideMode.ElideRight, 116)
            painter.drawText(5, y, name)
            if style is None:
                painter.drawText(5, y + 20, 'Нет шкалы')
                continue
            values = np.linspace(style.low, style.high, 112).reshape(1, -1)
            gray = render_pixels(values, style.low, style.high, style.gamma, style.invert)
            rgb = np.rint(gray[..., None] * (np.array(style.color) / 255)).astype(np.uint8)
            image = QImage(rgb.data, 112, 1, rgb.strides[0], QImage.Format.Format_RGB888).copy()
            painter.drawImage(5, y + 7, image.scaled(112, 14))
            painter.drawText(5, y + 37, f'{style.low:.5g} … {style.high:.5g}')
            painter.drawText(5, y + 53, f'γ={style.gamma:.3g}' + (' · инв.' if style.invert else ''))
            if not present:
                painter.drawText(5, y + 69, 'Нет канала')
        painter.end()
