"""Background feature matching and inspectable correspondence display."""
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QPen
from PyQt6.QtWidgets import QDialog, QLabel, QVBoxLayout
from feature_registration import refine_features
from viewer import ImageView
from wafer_data import display_limits, render_pixels


class FeatureWorker(QThread):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, images, transform, parent=None):
        super().__init__(parent)
        self.images = images
        self.transform = transform

    def run(self):
        try:
            t = self.transform
            result = refine_features(*self.images, t.left, t.right, t.angle, self.isInterruptionRequested)
            if not self.isInterruptionRequested():
                self.completed.emit(result)
        except InterruptedError:
            self.failed.emit('Поиск соответствий отменён')
        except Exception as error:
            self.failed.emit(f'Совмещение не изменено: {error}')


def show_correspondences(parent, images, result):
    dialog = QDialog(parent)
    dialog.setWindowTitle('Подтверждённые соответствия SIFT / RANSAC')
    dialog.resize(1200, 750)
    layout = QVBoxLayout(dialog)
    label = QLabel(f'Подтверждено {len(result.errors)} из {result.matches} пар; '
                   f'среднеквадратичная ошибка {result.rms:.2f} px правого TIFF. '
                   'Номера обозначают пары деталей; линии соединяют их на исходных снимках. '
                   'Проверьте, что это действительно одни и те же детали.')
    label.setWordWrap(True)
    layout.addWidget(label)
    rendered = []
    factors = []
    for image in images:
        h, w = image.shape
        stride = max(1, int(np.ceil(max(h, w)/800)))
        small = image[::stride, ::stride]
        low, high = display_limits(small)
        rendered.append(render_pixels(small, low, high))
        factors.append(stride)
    gap = 40
    offset = rendered[0].shape[1] + gap
    height = max(image.shape[0] for image in rendered)
    canvas = np.zeros((height, offset + rendered[1].shape[1]), dtype=np.uint8)
    canvas[:rendered[0].shape[0], :rendered[0].shape[1]] = rendered[0]
    canvas[:rendered[1].shape[0], offset:] = rendered[1]
    view = ImageView()
    view.set_pixels(canvas)
    layout.addWidget(view, 1)
    scene = view.scene()
    # Bound visual clutter while preserving all matches in the numeric summary.
    indices = np.linspace(0, len(result.errors)-1, min(60, len(result.errors)), dtype=int)
    for index in indices:
        a = result.left_points[index]/factors[0] + 0.5
        b = result.right_points[index]/factors[1] + [offset+0.5, 0.5]
        color = QColor.fromHsv((int(index)*47) % 360, 200, 255)
        pen = QPen(color, 1)
        pen.setCosmetic(True)
        scene.addLine(float(a[0]), float(a[1]), float(b[0]), float(b[1]), pen)
        for point in (a, b):
            scene.addEllipse(float(point[0]-3), float(point[1]-3), 6, 6, pen)
            text = scene.addSimpleText(str(index+1))
            text.setBrush(color)
            text.setPos(float(point[0]+4), float(point[1]+2))
    dialog.exec()
    dialog.deleteLater()
