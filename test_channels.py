import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from pathlib import Path
import tempfile
import unittest

import numpy as np
import tifffile
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from channel_viewer import ChannelStyle, ChannelViewer, compose_overlay
from wafer_data import Frame


class OverlayTests(unittest.TestCase):
    def test_additive_colors_resizing_and_raw_preservation(self):
        red = np.array([[0, 65535], [65535, 0]], dtype=np.uint16)
        green = np.full((1, 1), 32768, dtype=np.uint16)
        original = red.copy()
        result = compose_overlay([
            (Frame(red, {}, False, 1), ChannelStyle(0, 65535, 1, False, (255, 0, 0))),
            (Frame(green, {}, False, 1), ChannelStyle(0, 65535, 1, False, (0, 255, 0))),
        ])
        np.testing.assert_array_equal(result[0, 0], [0, 128, 0])
        np.testing.assert_array_equal(result[0, 1], [255, 128, 0])
        np.testing.assert_array_equal(red, original)
        style = ChannelStyle(0, 65535, 1, False, (255, 255, 255))
        with self.assertRaises(ValueError):
            compose_overlay([(Frame(red, {}, False, 1), style),
                             (Frame(np.zeros((2, 3), dtype=np.uint16), {}, False, 1), style)])
        bright = Frame(np.full((1, 1), 65535, dtype=np.uint16), {}, False, 1)
        np.testing.assert_array_equal(compose_overlay([(bright, style), (bright, style)]), [[[255, 255, 255]]])


class ChannelViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        for sample in ('A', 'B', 'C'):
            folder = root / sample
            folder.mkdir()
            with tifffile.TiffWriter(folder / 'one(Red).raw16.tif') as writer:
                writer.write(np.full((4, 6), 10000, dtype=np.uint16))
                writer.write(np.full((4, 6), 12345, dtype=np.uint16))
            if sample != 'B':
                tifffile.imwrite(folder / 'two(Green).raw16.tif', np.full((2, 3), 30000, dtype=np.uint16))
            tifffile.imwrite(folder / 'one(Red).tif', np.zeros((4, 6, 3), dtype=np.uint8), photometric='rgb')
        self.window = ChannelViewer(root)
        self.addCleanup(self.close_window)
        self.window.show()
        self.window.activateWindow()
        self.app.processEvents()

    def close_window(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_raw_only_and_overlay_survives_missing_channel(self):
        w = self.window
        self.assertEqual(len(w.sample_keys), 3)
        self.assertTrue(all(c.raw for sample in w.samples.values() for captures in sample.values() for c in captures))
        w.selected_modes = {'Red', 'Green'}
        w.refresh_sample()
        w.set_channel_color('Red', (255, 0, 0))
        w.set_channel_color('Green', (0, 255, 0))
        w.inspect_pixel(5.8, 3.8)
        self.assertIn('Green: x=2, y=1, RAW16=30000', w.statusBar().currentMessage())
        self.assertIn('Red: x=5, y=3, RAW16=10000', w.statusBar().currentMessage())
        w.view.setFocus()
        QTest.keyClick(w.view, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
        self.app.processEvents()
        self.assertEqual(w.sample_key[1], 'B')
        self.assertEqual(w.selected_modes, {'Red', 'Green'})
        self.assertEqual(set(w.layers), {'Red'})
        self.assertIn('Green: отсутствует', w.notice.text())
        w.move_sample(1)
        self.assertEqual(set(w.layers), {'Red', 'Green'})
        self.assertEqual(w.styles['Green'].color, (0, 255, 0))
        w.move_sample(1)
        self.assertEqual(w.sample_key[1], 'C')

    def test_channel_cycle_pages_and_scale_persistence(self):
        w = self.window
        w.active_mode = 'Red'
        w.only_active()
        w.low.setText('0')
        w.high.setText('65535')
        w.gamma.setText('2')
        w.draw()
        w.pages.setCurrentIndex(1)
        w.pages.activated.emit(1)
        w.inspect_pixel(0, 0)
        self.assertIn('12345', w.statusBar().currentMessage())
        w.move_sample(1)
        self.assertEqual(w.selected_modes, {'Red'})
        self.assertEqual(w.styles['Red'].high, 65535)
        self.assertEqual(w.styles['Red'].gamma, 2)
        self.assertFalse(w.frame.pixels.flags.writeable)
        w.move_sample(1)
        w.cycle_channel(1)
        self.assertEqual(w.selected_modes, {'Green'})
        w.channel_items['Green'].setCheckState(0, Qt.CheckState.Unchecked)
        self.assertIsNone(w.view.item)
        w.move_sample(-1)
        self.assertEqual(w.selected_modes, set())


if __name__ == '__main__':
    unittest.main()
