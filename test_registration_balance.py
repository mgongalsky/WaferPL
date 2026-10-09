import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import unittest
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from registration_preview import compose_layers
from registration_workspace import RegistrationWorkspace
from wafer_data import Frame


class BalanceTests(unittest.TestCase):
    def test_color_endpoints_midpoint_and_missing_coverage(self):
        a = np.array([[120, 120, 120]], dtype=np.uint8)
        b = np.array([[200, 200, 200]], dtype=np.uint8)
        va = np.array([[True, True, False]])
        vb = np.array([[True, False, True]])
        layers = a, b, va, vb
        np.testing.assert_array_equal(compose_layers(layers, 'color', 0),
                                      [[[120, 0, 0], [120, 0, 0], [0, 0, 0]]])
        np.testing.assert_array_equal(compose_layers(layers, 'color', 1),
                                      [[[0, 200, 200], [0, 0, 0], [0, 200, 200]]])
        np.testing.assert_array_equal(compose_layers(layers, 'color', 0.5),
                                      [[[120, 200, 200], [120, 0, 0], [0, 200, 200]]])
        np.testing.assert_array_equal(compose_layers(layers, 'color', 0.25)[0, 0], [120, 100, 100])
        np.testing.assert_array_equal(compose_layers(layers, 'color', 0.75)[0, 0], [60, 200, 200])
        colors = ((0, 255, 0), (255, 0, 255))
        np.testing.assert_array_equal(compose_layers(layers, 'color', 0, colors)[0, 0], [0, 120, 0])
        np.testing.assert_array_equal(compose_layers(layers, 'color', 1, colors)[0, 0], [200, 0, 200])

    def test_workspace_slider_responds_and_preserves_alignment(self):
        app = QApplication.instance() or QApplication([])
        pixels = np.arange(4096, dtype=np.uint16).reshape(64, 64)
        frame = Frame(pixels, {}, False, 1)
        window = RegistrationWorkspace(frame, frame, ('Левый', 'Правый'))
        try:
            window.show()
            app.processEvents()
            window.show_manual()
            transform = window.initial_transform()
            window.result = transform
            self.assertEqual(window.display_mode.currentData(), 'color')
            self.assertTrue(window.opacity.isEnabled())
            window.opacity.setFocus()
            QTest.keyClick(window.opacity, Qt.Key.Key_Home)
            first = window.output.item.pixmap().toImage().pixelColor(32, 32)
            self.assertEqual(window.opacity.value(), 0)
            self.assertGreater(first.red(), 0)
            self.assertEqual((first.green(), first.blue()), (0, 0))
            QTest.keyClick(window.opacity, Qt.Key.Key_End)
            last = window.output.item.pixmap().toImage().pixelColor(32, 32)
            self.assertEqual(window.opacity.value(), 100)
            self.assertEqual(last.red(), 0)
            self.assertGreater(last.green(), 0)
            self.assertEqual(last.green(), last.blue())
            self.assertIs(window.result, transform)
            self.assertEqual(window.initial_transform(), transform)
        finally:
            window.close()
            window.deleteLater()
            app.processEvents()


if __name__ == '__main__':
    unittest.main()
