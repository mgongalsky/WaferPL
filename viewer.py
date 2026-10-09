"""Run viewer.py in the project's Python environment to browse data/."""
from pathlib import Path
import sys

import numpy as np
from PyQt6.QtCore import QPointF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QGraphicsScene,
    QGraphicsView, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPlainTextEdit, QPushButton, QSplitter, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)

from wafer_data import (capture_metadata, discover, display_limits,
                        image_coordinates, pixel_value, read_frame, render_pixels)


class ImageView(QGraphicsView):
    pixel_hovered = pyqtSignal(float, float)
    pointer_left = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.item = None
        self.zoom_mode = 'Вписать'
        self.pan_position = None
        self.setMouseTracking(True)
        self.setBackgroundBrush(QColor('#20242b'))
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        # Stable viewport dimensions avoid fit/scrollbar resize feedback.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

    def clear_image(self):
        self.scene().clear()
        self.item = None
        self.setSceneRect(0, 0, 0, 0)
        self.resetTransform()

    def set_pixels(self, pixels):
        pixels = np.ascontiguousarray(pixels)
        height, width = pixels.shape[:2]
        format_ = QImage.Format.Format_Grayscale8 if pixels.ndim == 2 else QImage.Format.Format_RGB888
        # Own the Qt buffer independently of the temporary NumPy display array.
        image = QImage(pixels.data, width, height, pixels.strides[0], format_).copy()
        pixmap = QPixmap.fromImage(image)
        if pixmap.isNull():
            raise ValueError('Не удалось создать изображение Qt')
        if self.item is None:
            self.item = self.scene().addPixmap(pixmap)
            self.item.setTransformationMode(Qt.TransformationMode.FastTransformation)
        else:
            self.item.setPixmap(pixmap)
        self.setSceneRect(0, 0, width, height)
        self.apply_zoom()

    def set_zoom(self, mode):
        self.zoom_mode = mode
        self.apply_zoom()

    def apply_zoom(self):
        if self.item is None:
            return
        if self.zoom_mode == 'Вписать':
            rect = self.sceneRect()
            viewport_size = getattr(self, 'fit_viewport_size', None)
            width, height = viewport_size or (self.viewport().width(), self.viewport().height())
            scale = min(max(1, width - 2) / rect.width(), max(1, height - 2) / rect.height())
        else:
            scale = float(self.zoom_mode.rstrip('%')) / 100
        self.resetTransform()
        self.scale(scale, scale)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.zoom_mode == 'Вписать':
            self.apply_zoom()

    def report_position(self, position):
        point = self.mapToScene(position)
        self.pixel_hovered.emit(point.x(), point.y())

    def mouseMoveEvent(self, event):
        if self.pan_position is not None:
            delta = event.position().toPoint() - self.pan_position
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self.pan_position = event.position().toPoint()
        self.report_position(event.position().toPoint())
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.pan_position = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        self.report_position(event.position().toPoint())
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.pan_position = None
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self.pointer_left.emit()
        super().leaveEvent(event)


class Viewer(QMainWindow):
    def __init__(self, folder: Path):
        super().__init__()
        self.setWindowTitle('WaferPL — TIFF viewer')
        self.resize(1400, 900)
        self.setMinimumSize(950, 620)
        self.folder = Path(folder)
        self.frame = None
        self.capture = None
        self.build_ui()
        self.scan_timer = QTimer(self)
        self.scan_timer.setSingleShot(True)
        self.scan_timer.timeout.connect(self.scan)
        self.scan_timer.start(0)

    @staticmethod
    def add_button(layout, text, callback):
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        top = QHBoxLayout()
        layout.addLayout(top)
        self.add_button(top, 'Открыть папку…', self.choose_folder)
        self.add_button(top, 'Обновить', self.scan)
        self.heading = QLabel('WaferPL')
        self.heading.setWordWrap(True)
        top.addWidget(self.heading, 1)
        split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(split, 1)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.currentItemChanged.connect(self.select)
        split.addWidget(self.tree)
        right = QWidget()
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([380, 1000])
        panel = QVBoxLayout(right)
        controls = QHBoxLayout()
        panel.addLayout(controls)
        self.low = QLineEdit('0')
        self.high = QLineEdit('65535')
        self.gamma = QLineEdit('1')
        for label, entry in (('Мин.', self.low), ('Макс.', self.high), ('Гамма', self.gamma)):
            controls.addWidget(QLabel(label))
            entry.setMaximumWidth(100)
            controls.addWidget(entry)
            entry.returnPressed.connect(self.draw)
        self.add_button(controls, 'Применить', self.draw)
        self.invert = QCheckBox('Инверсия')
        self.invert.clicked.connect(self.draw)
        controls.addWidget(self.invert)
        controls.addStretch()
        tools = QHBoxLayout()
        panel.addLayout(tools)
        self.add_button(tools, 'Авто 1–99,5%', lambda: self.set_limits(True))
        self.add_button(tools, 'Весь диапазон', lambda: self.set_limits(False))
        tools.addWidget(QLabel('Масштаб'))
        self.zoom = QComboBox()
        self.zoom.addItems(['Вписать', '25%', '50%', '100%', '200%'])
        tools.addWidget(self.zoom)
        tools.addWidget(QLabel('Страница'))
        self.pages = QComboBox()
        tools.addWidget(self.pages)
        tools.addStretch()
        self.view = ImageView()
        panel.addWidget(self.view, 1)
        self.zoom.currentTextChanged.connect(self.view.set_zoom)
        self.pages.activated.connect(self.load)
        self.view.pixel_hovered.connect(self.inspect_pixel)
        self.view.pointer_left.connect(lambda: self.statusBar().showMessage('Наведите курсор на изображение для чтения пикселя'))
        self.metadata = QPlainTextEdit()
        self.metadata.setReadOnly(True)
        self.metadata.setMaximumHeight(190)
        panel.addWidget(self.metadata)
        self.statusBar().showMessage('Выберите TIFF')

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Открыть папку', str(self.folder))
        if folder:
            self.folder = Path(folder)
            self.scan()

    def clear_image(self):
        self.frame = None
        self.view.clear_image()
        self.pages.clear()
        self.metadata.clear()
        self.heading.setText('WaferPL')
        self.statusBar().showMessage('Выберите TIFF')

    def scan(self):
        self.scan_timer.stop()
        self.clear_image()
        self.capture = None
        self.tree.clear()
        try:
            if not self.folder.is_dir():
                raise ValueError(f'Папка не найдена: {self.folder}')
            captures = discover(self.folder)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, 'Чтение папки', str(error))
            return
        groups = {}
        initial = None
        preferred = False
        for capture in captures:
            key = (capture.path.parent, capture.sample)
            if key not in groups:
                relative = capture.path.parent.relative_to(self.folder)
                groups[key] = QTreeWidgetItem(self.tree, [f'{capture.sample} | {relative}'])
                groups[key].setExpanded(True)
            kind = 'RAW16' if capture.raw else 'TIFF экспорт'
            item = QTreeWidgetItem(groups[key], [f'[{kind}] {capture.path.name}'])
            item.setData(0, Qt.ItemDataRole.UserRole, capture)
            item.setToolTip(0, str(capture.path))
            is_preferred = capture.raw and '261009_Mono16' in str(capture.path.parent)
            if initial is None or (is_preferred and not preferred):
                initial = item
                preferred = is_preferred
        self.statusBar().showMessage(f'TIFF: {len(captures)}. RAW16 — для измерений; экспорт — для просмотра.')
        if initial is not None:
            self.tree.setCurrentItem(initial)
            self.tree.scrollToItem(initial)
        else:
            self.metadata.setPlainText('В выбранной папке и её подпапках TIFF не найдены.')

    def select(self, current, previous=None):
        if current is None:
            return
        capture = current.data(0, Qt.ItemDataRole.UserRole)
        if capture is None:
            return
        self.capture = capture
        self.load(0)

    def load(self, page_index=0):
        if self.capture is None:
            return
        self.clear_image()
        try:
            self.frame = read_frame(self.capture.path, page_index)
            self.pages.addItems([str(i + 1) for i in range(self.frame.page_count)])
            self.pages.setCurrentIndex(page_index)
            self.invert.setChecked(self.frame.white_is_zero)
            self.gamma.setText('1')
            pixels = self.frame.pixels
            height, width = pixels.shape[:2]
            self.heading.setText(f'{self.capture.sample} — {self.capture.mode} — {width}×{height}, {pixels.dtype}')
            kind = 'RAW16: значения TIFF' if self.capture.raw else 'ЭКСПОРТ: значения TIFF, не исходный сигнал прибора'
            details = [str(self.capture.path), kind,
                       f'Образец: {self.capture.sample} ({self.capture.sample_source})',
                       f'Режим из имени файла: {self.capture.mode}',
                       'Контраст, гамма и инверсия меняют только отображение.',
                       'SCN: параметры протокола; числовые коды не преобразованы в длины волн.',
                       'exposure_time приведён как в SCN; единица в XML не указана.']
            details.extend(f'{key}: {value}' for key, value in self.frame.tags.items())
            details.extend(f'{key}: {value}' for key, value in capture_metadata(self.capture).items())
            self.metadata.setPlainText('\n'.join(details))
            self.set_limits(True)
            self.view.horizontalScrollBar().setValue(0)
            self.view.verticalScrollBar().setValue(0)
        except Exception as error:
            self.clear_image()
            self.statusBar().showMessage('Не удалось открыть выбранный TIFF')
            QMessageBox.warning(self, 'Чтение TIFF', f'{self.capture.path.name}\n{error}')

    def set_limits(self, auto):
        if self.frame is None:
            return
        try:
            low, high = display_limits(self.frame.pixels, auto)
            self.low.setText(f'{low:.8g}')
            self.high.setText(f'{high:.8g}')
            self.draw()
        except ValueError as error:
            self.statusBar().showMessage(str(error))

    def draw(self):
        if self.frame is None:
            return
        try:
            low, high, gamma = (float(entry.text().replace(',', '.')) for entry in (self.low, self.high, self.gamma))
            rendered = render_pixels(self.frame.pixels, low, high, gamma, self.invert.isChecked())
            self.view.set_pixels(rendered)
        except (ValueError, MemoryError) as error:
            self.statusBar().showMessage(f'Не удалось обновить отображение: {error}')
            return
        self.statusBar().showMessage(f'Шкала {low:g}…{high:g}; гамма {gamma:g}. Наведите курсор для чтения исходного пикселя.')

    def inspect_pixel(self, scene_x, scene_y):
        if self.frame is None:
            return
        pixels = self.frame.pixels
        size = (pixels.shape[1], pixels.shape[0])
        # Scene coordinates are source pixels, regardless of zoom, scroll or DPI.
        coordinates = image_coordinates(scene_x, scene_y, size, size)
        if coordinates is None:
            self.statusBar().showMessage('За пределами изображения')
            return
        x, y = coordinates
        value = pixel_value(pixels, x, y)
        label = 'Значение RAW16' if self.capture.raw else 'Значение экспорта TIFF'
        self.statusBar().showMessage(f'x={x}, y={y} (с нуля) | {label}: {value} | {pixels.dtype} | страница {self.pages.currentText()}')


def main():
    from comparison_viewer import ComparisonViewer

    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'data'
    app = QApplication(sys.argv)
    window = ComparisonViewer(folder)
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
