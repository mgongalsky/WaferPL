"""Interactive wafer circles and cancellable registration."""
import math

from PyQt6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPen
from PyQt6.QtWidgets import (QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout,
                             QLabel, QPushButton, QVBoxLayout, QWidget)
from viewer import ImageView
from wafer_data import display_limits, render_pixels
from registration import Circle, Registration, refine, preview


class CircleView(ImageView):
    circle_changed = pyqtSignal(object)
    orientation_changed = pyqtSignal(float)
    drawing_finished = pyqtSignal()

    def __init__(self, pixels, circle, parent=None):
        super().__init__(parent)
        self.circle = circle
        self.orientation = 0.0
        self.draw_mode = False
        self.drag = None
        self.setMinimumSize(160, 160)
        self.set_pixels(pixels)
        self.fit_circle()

    def fit_circle(self):
        rect = self.item.boundingRect()
        c = self.circle
        rect = rect.united(QRectF(c.x-c.radius, c.y-c.radius, 2*c.radius, 2*c.radius))
        margin = max(rect.width(), rect.height()) * 0.18
        self.setSceneRect(rect.adjusted(-margin, -margin, margin, margin))
        self.apply_zoom()

    def handle_positions(self):
        c = self.circle
        centre = QPointF(c.x+0.5, c.y+0.5)
        theta = math.radians(self.orientation)
        scale = max(abs(self.transform().m11()), 1e-9)
        return {
            'move': centre,
            'radius': centre + QPointF(-math.sin(theta), math.cos(theta))*c.radius,
            'rotate': centre + QPointF(math.cos(theta), math.sin(theta))*(c.radius+26/scale),
        }

    def hit_test(self, position):
        if self.draw_mode:
            return 'draw'
        for kind, point in self.handle_positions().items():
            delta = self.viewportTransform().map(point) - QPointF(position)
            if math.hypot(delta.x(), delta.y()) <= 11:
                return kind
        point = self.mapToScene(position) - QPointF(self.circle.x+0.5, self.circle.y+0.5)
        distance = math.hypot(point.x(), point.y())
        scale = max(abs(self.transform().m11()), 1e-9)
        if abs(distance-self.circle.radius)*scale <= 7:
            return 'radius'
        return 'move' if distance < self.circle.radius else None

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        if not hasattr(self, 'circle'):
            return
        painter.save()
        handles = self.handle_positions()
        pen = QPen(QColor('#ffcc33'), 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(handles['move'], self.circle.radius, self.circle.radius)
        painter.drawLine(handles['move'], handles['rotate'])
        # Handles and labels stay the same screen size at every zoom level.
        points = {kind: self.viewportTransform().map(point) for kind, point in handles.items()}
        painter.resetTransform()
        for kind, point in points.items():
            painter.setBrush(QColor('#20242b'))
            if kind == 'rotate':
                painter.setPen(QPen(QColor('#55ddff'), 2))
                painter.drawEllipse(point, 8, 8)
                label = f'↻ {self.orientation:.1f}°'
            else:
                painter.setPen(QPen(QColor('#ffcc33'), 2))
                painter.drawRect(QRectF(point.x()-6, point.y()-6, 12, 12))
                label = 'Центр' if kind == 'move' else 'Радиус'
            painter.drawText(point + QPointF(12, -10), label)
        painter.restore()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            kind = self.hit_test(event.position().toPoint())
            if kind is not None:
                point = self.mapToScene(event.position().toPoint()) - QPointF(0.5, 0.5)
                self.drag = (kind, point, self.circle)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drag is not None:
            point = self.mapToScene(event.position().toPoint()) - QPointF(0.5, 0.5)
            kind, start, original = self.drag
            if kind == 'rotate':
                dx, dy = point.x()-original.x, point.y()-original.y
                if math.hypot(dx, dy) > 1e-9:
                    self.orientation = math.degrees(math.atan2(dy, dx))
                    self.orientation_changed.emit(self.orientation)
            else:
                if kind == 'draw':
                    self.circle = Circle(start.x(), start.y(), max(1.0, math.hypot(point.x()-start.x(), point.y()-start.y())))
                elif kind == 'radius':
                    self.circle = Circle(original.x, original.y, max(1.0, math.hypot(point.x()-original.x, point.y()-original.y)))
                else:
                    self.circle = Circle(original.x+point.x()-start.x(), original.y+point.y()-start.y(), original.radius)
                self.circle_changed.emit(self.circle)
            self.viewport().update()
            event.accept()
            return
        if self.pan_position is None:
            kind = self.hit_test(event.position().toPoint())
            cursors = {'move': Qt.CursorShape.SizeAllCursor, 'radius': Qt.CursorShape.SizeFDiagCursor,
                       'rotate': Qt.CursorShape.PointingHandCursor, 'draw': Qt.CursorShape.CrossCursor}
            self.setCursor(cursors.get(kind, Qt.CursorShape.ArrowCursor))
            tips = {'move': 'Перетащите центр или внутреннюю область круга',
                    'radius': 'Потяните край: радиус изменится на обеих панелях',
                    'rotate': 'Перетащите голубую ручку вокруг центра для поворота',
                    'draw': 'Протяните от центра к краю нового круга'}
            self.setToolTip(tips.get(kind, 'Средняя кнопка мыши — перемещение вида'))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drag is not None:
            kind = self.drag[0]
            self.mouseMoveEvent(event)
            self.drag = None
            if kind == 'draw':
                self.draw_mode = False
                self.drawing_finished.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class CircleEditor(QWidget):
    changed = pyqtSignal()

    def __init__(self, frame, title, circle, radius_editable=True):
        super().__init__()
        layout = QVBoxLayout(self)
        label = QLabel(title)
        label.setWordWrap(True)
        layout.addWidget(label)
        low, high = display_limits(frame.pixels)
        self.view = CircleView(render_pixels(frame.pixels, low, high, invert=frame.white_is_zero), circle)
        layout.addWidget(self.view, 1)
        form = QFormLayout()
        layout.addLayout(form)
        self.fields = []
        for name, value in (('Центр X, px', circle.x), ('Центр Y, px', circle.y), ('Общий радиус, px', circle.radius)):
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(1 if 'радиус' in name else -1e9, 1e9)
            spin.setValue(value)
            spin.setKeyboardTracking(False)
            form.addRow(name, spin)
            self.fields.append(spin)
            spin.valueChanged.connect(self.from_fields)
        self.angle = QDoubleSpinBox()
        self.angle.setRange(-180, 180)
        self.angle.setDecimals(2)
        self.angle.setSuffix('°')
        self.angle.setKeyboardTracking(False)
        self.angle.setToolTip('Направление метки на этой пластине; положительный угол — по часовой стрелке')
        form.addRow('Поворот ↻', self.angle)
        self.angle.valueChanged.connect(self.from_angle)
        self.view.circle_changed.connect(self.from_view)
        self.view.orientation_changed.connect(self.from_rotation)
        row = QHBoxLayout()
        layout.addLayout(row)
        self.draw_button = QPushButton('Новый круг: центр → край')
        self.draw_button.setCheckable(True)
        self.draw_button.toggled.connect(self.set_draw_mode)
        self.view.drawing_finished.connect(lambda: self.draw_button.setChecked(False))
        row.addWidget(self.draw_button)
        fit = QPushButton('Вписать круг')
        fit.clicked.connect(self.view.fit_circle)
        row.addWidget(fit)

    def set_draw_mode(self, enabled):
        self.view.draw_mode = enabled
        self.view.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def from_fields(self):
        self.set_circle(Circle(*(field.value() for field in self.fields)))
        self.changed.emit()

    def from_view(self, circle):
        self.set_circle(circle)
        self.changed.emit()

    def from_angle(self, value):
        self.view.orientation = value
        self.view.viewport().update()
        self.changed.emit()

    def from_rotation(self, value):
        self.angle.blockSignals(True)
        self.angle.setValue(value)
        self.angle.blockSignals(False)
        self.changed.emit()

    def set_circle(self, circle):
        self.view.circle = circle
        for field, value in zip(self.fields, (circle.x, circle.y, circle.radius)):
            field.blockSignals(True)
            field.setValue(value)
            field.blockSignals(False)
        self.view.viewport().update()


class RegistrationWorker(QThread):
    completed = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, images, transform, parent=None):
        super().__init__(parent)
        self.images = images
        self.transform = transform

    def run(self):
        try:
            t = self.transform
            result = refine(*self.images, t.left, t.right, t.angle, self.isInterruptionRequested)
            if not self.isInterruptionRequested():
                self.completed.emit(result, None)
        except InterruptedError:
            self.failed.emit('Поиск поворота отменён')
        except Exception as error:
            self.failed.emit(f'Поворот не изменён: {error}')


class RegistrationDialog(QDialog):
    def __init__(self, left_frame, right_frame, titles, parent=None):
        super().__init__(parent)
        from PyQt6.QtWidgets import QComboBox, QSlider
        self.setWindowTitle('WaferPL — совмещение пластин')
        self.resize(1350, 900)
        self.worker = None
        self.closing = False
        self.result = None
        self.preview_layers = None
        self.images = tuple(65535 - f.pixels if f.white_is_zero else f.pixels for f in (left_frame, right_frame))
        layout = QVBoxLayout(self)
        help_text = QLabel(
            'Обведите границу пластины кругом на каждом снимке. Центр — перемещение; край — общий радиус; '
            'голубая ручка ↻ — направление на общую деталь. Снимки могут отличаться после обработки. '
            'Поиск уточняет только поворот (±15°), не сдвигает круги и не меняет размер.')
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        self.editors = QWidget()
        row = QHBoxLayout(self.editors)
        h, w = left_frame.pixels.shape
        radius = max(1.0, min(h, w) * 0.45)
        self.left = CircleEditor(left_frame, titles[0], Circle((w - 1) / 2, (h - 1) / 2, radius))
        h, w = right_frame.pixels.shape
        self.right = CircleEditor(right_frame, titles[1], Circle((w - 1) / 2, (h - 1) / 2, radius))
        row.addWidget(self.left)
        row.addWidget(self.right)
        layout.addWidget(self.editors, 2)
        actions = QHBoxLayout()
        layout.addLayout(actions)
        self.manual = QPushButton('Показать совмещение')
        self.manual.setToolTip('Обновить результат по заданным кругам и углам без поиска')
        self.manual.clicked.connect(self.show_manual)
        actions.addWidget(self.manual)
        self.align = QPushButton('Уточнить поворот по деталям')
        self.align.setToolTip(
            'Найти согласованный поворот по нескольким участкам. При неоднозначности угол останется прежним.')
        self.align.clicked.connect(self.start)
        actions.addWidget(self.align)
        self.cancel = QPushButton('Отменить поиск')
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self.cancel_search)
        actions.addWidget(self.cancel)
        display_row = QHBoxLayout()
        layout.addLayout(display_row)
        self.display_mode = QComboBox()
        self.display_mode.addItem('Смешивание снимков', 'blend')
        self.display_mode.addItem('Красный / голубой', 'color')
        display_row.addWidget(self.display_mode)
        display_row.addWidget(QLabel('Левый снимок'))
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(50)
        self.opacity.setToolTip('Вклад правого снимка: 0% — только левый, 100% — только правый')
        display_row.addWidget(self.opacity, 1)
        display_row.addWidget(QLabel('Правый снимок'))
        self.opacity_label = QLabel('50%')
        display_row.addWidget(self.opacity_label)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        legend = QLabel(
            'Смешивание помогает сравнить изменения после обработки. В цветном режиме: левый — красный, правый — голубой. '
            'Разный цвет внутри пластины не обязательно означает ошибку совмещения. Яркости нормированы отдельно; RAW не меняется.')
        legend.setWordWrap(True)
        layout.addWidget(legend)
        self.output = ImageView()
        self.output.setMinimumSize(160, 160)
        layout.addWidget(self.output, 1)
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.update_preview)
        self.left.changed.connect(lambda: self.sync_circle(self.left, self.right))
        self.right.changed.connect(lambda: self.sync_circle(self.right, self.left))
        self.display_mode.currentIndexChanged.connect(self.render_layers)
        self.opacity.valueChanged.connect(self.render_layers)
        self.preview_timer.start(0)

    def sync_circle(self, source, target):
        c = target.view.circle
        target.set_circle(Circle(c.x, c.y, source.view.circle.radius))
        self.invalidate()

    def invalidate(self):
        self.result = None
        self.status.setText('Обновление совмещения по кругам…')
        if self.worker is None and not self.closing and not self.preview_timer.isActive():
            self.preview_timer.start(80)

    def initial_transform(self):
        angle = (self.right.view.orientation - self.left.view.orientation + 180) % 360 - 180
        return Registration(self.left.view.circle, self.right.view.circle, angle)

    def render_layers(self):
        from registration_preview import compose_layers
        mode = self.display_mode.currentData()
        self.opacity.setEnabled(True)
        self.opacity.setToolTip(
            'Баланс снимков: 0% — только левый, 50% — оба, 100% — только правый. Работает в сером и цветном режимах.')
        self.opacity_label.setText(f'{self.opacity.value()}%')
        if self.preview_layers is not None:
            self.output.set_pixels(compose_layers(self.preview_layers, mode, self.opacity.value() / 100))

    def build_preview(self, limit):
        from registration_preview import aligned_layers
        try:
            self.preview_layers = aligned_layers(self.images, self.result or self.initial_transform(), limit)
            self.render_layers()
            return True
        except (ValueError, MemoryError) as error:
            self.preview_layers = None
            self.output.clear_image()
            self.status.setText(str(error))
            return False

    def update_preview(self):
        if self.worker is not None or self.closing:
            return
        if self.build_preview(450) and self.result is None:
            self.status.setText(f'Совмещение по кругам · разница углов: {self.initial_transform().angle:.2f}°. '
                                'Центры и размер задаются кругами; детали могут различаться.')

    def show_manual(self):
        if self.worker is not None or self.closing:
            return
        self.preview_timer.stop()
        if self.build_preview(900):
            transform = self.result or self.initial_transform()
            self.status.setText(f'Совмещение показано · разница углов: {transform.angle:.2f}°. Поиск не запускался.')

    def set_busy(self, busy):
        for widget in (self.editors, self.manual, self.align):
            widget.setEnabled(not busy)
        self.cancel.setEnabled(busy)

    def start(self):
        if self.worker is not None:
            return
        self.preview_timer.stop()
        self.build_preview(450)
        self.set_busy(True)
        self.status.setText('Ищем общий поворот по деталям. Центры кругов и радиус зафиксированы…')
        self.worker = RegistrationWorker(self.images, self.initial_transform(), self)
        self.worker.completed.connect(self.show_result)
        self.worker.failed.connect(self.status.setText)
        self.worker.finished.connect(self.search_finished)
        self.worker.start()

    def show_result(self, result, image):
        if self.closing or self.worker is None or self.worker.isInterruptionRequested():
            return
        self.result = result
        # Reflect the accepted orientation in both the numeric control and handle.
        orientation = (self.left.view.orientation + result.angle + 180) % 360 - 180
        self.right.angle.blockSignals(True)
        self.right.angle.setValue(orientation)
        self.right.angle.blockSignals(False)
        self.right.view.orientation = orientation
        self.right.view.viewport().update()
        if self.build_preview(900):
            self.status.setText(f'Поворот уточнён: {result.angle:.2f}°. Центры и радиус сохранены. '
                                f'Сходство согласованных деталей: {result.score:.2f}. Проверьте результат визуально.')

    def cancel_search(self):
        if self.worker is not None:
            self.worker.requestInterruption()
            self.status.setText('Отмена поиска…')

    def search_finished(self):
        cancelled = self.worker.isInterruptionRequested()
        self.worker.deleteLater()
        self.worker = None
        self.set_busy(False)
        if cancelled:
            self.status.setText('Поиск отменён. Положение и угол сохранены.')
        if self.closing:
            super().reject()

    def reject(self):
        self.preview_timer.stop()
        self.closing = True
        if self.worker is not None:
            self.cancel_search()
        else:
            super().reject()

    def closeEvent(self, event):
        if self.worker is not None:
            self.reject()
            event.ignore()
        else:
            self.preview_timer.stop()
            self.closing = True
            super().closeEvent(event)
