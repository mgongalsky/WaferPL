import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tifffile
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from viewer import Viewer


class ViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.source = np.zeros((700, 1000), dtype=np.uint16)
        self.source[300, 400] = 60000
        self.path = self.folder / 'capture(Flamingo).raw16.tif'
        with tifffile.TiffWriter(self.path) as writer:
            writer.write(self.source, photometric='miniswhite')
            writer.write(np.full((10, 12), 12345, dtype=np.uint16))
        self.warning = patch('viewer.QMessageBox.warning').start()
        self.addCleanup(patch.stopall)
        self.window = Viewer(self.folder)
        self.addCleanup(self.close_window)
        self.window.show()
        self.app.processEvents()
        self.assertIsNotNone(self.window.frame)
        self.warning.assert_not_called()

    def close_window(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_mouse_measurement_after_zoom_and_scroll(self):
        window = self.window
        window.zoom.setCurrentText('200%')
        self.app.processEvents()
        window.view.centerOn(QPointF(400.5, 300.5))
        self.app.processEvents()
        self.assertGreater(window.view.horizontalScrollBar().value(), 0)
        point = window.view.mapFromScene(QPointF(400.5, 300.5))
        QTest.mouseMove(window.view.viewport(), point)
        QTest.mouseClick(window.view.viewport(), Qt.MouseButton.LeftButton, pos=point)
        self.app.processEvents()
        status = window.statusBar().currentMessage()
        self.assertIn('x=400, y=300', status)
        self.assertIn('60000', status)
        window.inspect_pixel(-0.1, 0)
        self.assertEqual(window.statusBar().currentMessage(), 'За пределами изображения')

    def test_display_changes_preserve_data_and_invalid_input_keeps_image(self):
        window = self.window
        self.assertTrue(window.invert.isChecked())
        window.low.setText('100')
        window.high.setText('65000')
        window.gamma.setText('2')
        window.draw()
        np.testing.assert_array_equal(window.frame.pixels, self.source)
        self.assertFalse(window.frame.pixels.flags.writeable)
        previous = window.view.item.pixmap().cacheKey()
        window.high.setText('bad input')
        window.draw()
        self.assertEqual(window.view.item.pixmap().cacheKey(), previous)
        self.assertIn('Не удалось', window.statusBar().currentMessage())

    def test_page_switch_rgb_and_empty_folder(self):
        window = self.window
        self.assertEqual(window.pages.count(), 2)
        window.pages.setCurrentIndex(1)
        window.pages.activated.emit(1)
        self.assertEqual(window.frame.pixels.shape, (10, 12))
        self.assertEqual(window.pages.currentText(), '2')
        self.assertFalse(window.invert.isChecked())
        window.inspect_pixel(0, 0)
        self.assertIn('12345', window.statusBar().currentMessage())
        rgb_path = self.folder / 'rgb.tif'
        tifffile.imwrite(rgb_path, np.array([[[10, 20, 30]]], dtype=np.uint8), photometric='rgb')
        window.scan()
        group = window.tree.topLevelItem(0)
        for i in range(group.childCount()):
            item = group.child(i)
            if item.data(0, Qt.ItemDataRole.UserRole).path == rgb_path:
                window.tree.setCurrentItem(item)
                break
        self.assertEqual(window.frame.pixels.shape, (1, 1, 3))
        window.low.setText('0')
        window.high.setText('255')
        window.draw()
        color = window.view.item.pixmap().toImage().pixelColor(0, 0)
        self.assertEqual((color.red(), color.green(), color.blue()), (10, 20, 30))
        empty = self.folder / 'empty'
        empty.mkdir()
        window.folder = empty
        window.scan()
        self.assertIsNone(window.frame)
        self.assertIsNone(window.view.item)
        self.assertEqual(window.pages.count(), 0)
        self.warning.assert_not_called()


if __name__ == '__main__':
    unittest.main()
