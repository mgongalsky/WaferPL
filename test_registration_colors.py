import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import unittest
import numpy as np
from PyQt6.QtWidgets import QApplication
from registration_workspace import RegistrationWorkspace
from registration_preview import compose_layers
from wafer_data import Frame


class RegistrationColorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_window_starts_in_color_and_respects_explicit_gray_choice(self):
        pixels = np.arange(4096, dtype=np.uint16).reshape(64, 64)
        frame = Frame(pixels, {}, False, 1)
        window = RegistrationWorkspace(frame, frame, ('Левый', 'Правый'))
        try:
            window.preview_timer.stop()
            self.assertEqual(window.display_mode.currentData(), 'color')
            a = np.array([[200, 0, 100]], dtype=np.uint8)
            b = np.array([[0, 200, 100]], dtype=np.uint8)
            valid = np.ones(a.shape, dtype=bool)
            window.preview_layers = a, b, valid, valid
            window.render_layers()
            image = window.output.item.pixmap().toImage()
            rgb = lambda x: image.pixelColor(x, 0).getRgb()[:3]
            self.assertEqual(rgb(0), (200, 0, 0))
            self.assertEqual(rgb(1), (0, 200, 200))
            self.assertEqual(rgb(2), (100, 100, 100))
            window.display_mode.setCurrentIndex(window.display_mode.findData('blend'))
            window.update_color_buttons()
            self.assertEqual(window.display_mode.currentData(), 'blend')
            self.assertTrue(window.opacity.isEnabled())
        finally:
            window.close()
            window.deleteLater()
            self.app.processEvents()

    def test_color_array_is_not_modified_by_rendering(self):
        palette = np.array([[255., 0., 0.], [0., 255., 255.]])
        original = palette.copy()
        image = np.full((2, 2), 100, dtype=np.uint8)
        valid = np.ones(image.shape, dtype=bool)
        layers = image, image, valid, valid
        first = compose_layers(layers, 'color', colors=palette)
        second = compose_layers(layers, 'color', colors=palette)
        np.testing.assert_array_equal(palette, original)
        np.testing.assert_array_equal(first, second)


if __name__ == '__main__':
    unittest.main()
