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
from comparison_viewer import ComparisonViewer
from normalization import detail_settings
from channel_viewer import ChannelStyle, compose_overlay
from wafer_data import Frame, render_pixels


class NormalizationTests(unittest.TestCase):
    def test_exposure_changes_preserve_details_and_raw(self):
        source = np.arange(100, 1100, dtype=np.uint16).reshape(20, 50)
        bright = source * 10
        for invert in (False, True):
            results = []
            for pixels in (source, bright):
                original = pixels.copy()
                low, high, gamma = detail_settings(pixels, invert)
                display = render_pixels(pixels, low, high, gamma, invert)
                self.assertLess(np.mean(display == 255), 0.02)
                self.assertGreater(np.percentile(display, 90) - np.percentile(display, 10), 100)
                self.assertAlmostEqual(float(np.median(display)) / 255, 0.45, delta=0.02)
                np.testing.assert_array_equal(pixels, original)
                results.append(display)
            np.testing.assert_array_equal(*results)

    def test_constant_and_detector_rails(self):
        for value in (0, 42, 65535):
            low, high, gamma = detail_settings(np.full((5, 5), value, dtype=np.uint16))
            self.assertGreater(high, low)
            self.assertTrue(np.isfinite(gamma))
        pixels = np.concatenate([np.zeros(10000), np.arange(100, 1100), [65535]]).astype(np.uint16)
        low, high, gamma = detail_settings(pixels)
        self.assertGreater(low, 0)
        self.assertLess(high, 1100)

    def test_overlay_does_not_flatten_overlapping_highlights(self):
        pixels = np.array([[40000, 50000, 60000]], dtype=np.uint16)
        frame = Frame(pixels, {}, False, 1)
        style = ChannelStyle(0, 65535, 1, False, (255, 255, 255))
        image = compose_overlay([(frame, style), (frame, style)])
        self.assertTrue(np.all(np.diff(image[0, :, 0].astype(int)) > 0))
        self.assertLess(int(image.max()), 255)


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        for name, factor in [('A', 1), ('B', 10)]:
            folder = root / name
            folder.mkdir()
            for mode in ('Red', 'Green'):
                pixels = (np.arange(100, 1100, dtype=np.uint16).reshape(20, 50) * factor)
                tifffile.imwrite(folder / f'capture({mode}).raw16.tif', pixels)
        self.window = ComparisonViewer(root)
        self.addCleanup(self.close_window)
        self.window.show()
        self.window.activateWindow()
        self.app.processEvents()

    def close_window(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_auto_per_sample_and_manual_lock(self):
        panel = self.window.left
        mode = panel.active_mode
        initial_high = panel.styles[mode].high
        panel.move_sample(1)
        self.assertAlmostEqual(panel.styles[mode].high, initial_high * 10)
        self.assertEqual(panel.sample_selector.currentIndex(), 1)
        panel.low.setText('0')
        panel.high.setText('12000')
        panel.gamma.setText('1')
        panel.draw()
        panel.move_sample(-1)
        self.assertEqual(panel.styles[mode].high, 12000)
        self.assertFalse(panel.auto_checkbox.isChecked())
        panel.set_limits(True)
        self.assertAlmostEqual(panel.styles[mode].high, initial_high)
        self.assertTrue(panel.auto_checkbox.isChecked())
        index = panel.channel_selector.findData('Red')
        panel.channel_selector.setCurrentIndex(index)
        panel.channel_selector.activated.emit(index)
        self.assertEqual(panel.selected_modes, {'Red'})

    def test_independent_panels_focus_and_toggle(self):
        window = self.window
        window.compare.setChecked(True)
        self.app.processEvents()
        left, right = window.left, window.right
        self.assertIsNot(left.styles, right.styles)
        right.view.setFocus()
        self.app.processEvents()
        QTest.keyClick(right.view, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
        self.app.processEvents()
        self.assertEqual(right.sample_key[1], 'B')
        self.assertEqual(left.sample_key[1], 'A')
        right.set_channel_color(right.active_mode, (1, 2, 3))
        self.assertNotEqual(left.colors[right.active_mode], (1, 2, 3))
        self.assertEqual(left.sample_selector.currentIndex(), 0)
        self.assertEqual(right.sample_selector.currentIndex(), 1)
        window.compare.setChecked(False)
        self.assertTrue(right.isHidden())
        window.compare.setChecked(True)
        self.assertIs(window.right, right)
        self.assertEqual(right.sample_key[1], 'B')
        self.assertIsNotNone(left.view.item)
        self.assertIsNotNone(right.view.item)

    def test_landscape_layout_and_collapsible_sidebars(self):
        window = self.window
        window.resize(1600, 900)
        window.compare.setChecked(True)
        self.app.processEvents()
        window.equalize_views()
        self.assertEqual(window.dockWidgetArea(window.tree_dock), Qt.DockWidgetArea.LeftDockWidgetArea)
        self.assertEqual(window.dockWidgetArea(window.controls_dock), Qt.DockWidgetArea.RightDockWidgetArea)
        self.assertTrue(window.tree.isVisible())
        for panel in (window.left, window.right):
            self.assertGreater(panel.view.viewport().height(), panel.height() * 0.80)
            self.assertTrue(panel.sidebar.isHidden())
        self.assertLessEqual(abs(window.left.view.viewport().width() - window.right.view.viewport().width()), 1)
        before = window.left.view.viewport().width()
        window.controls_dock.hide()
        self.app.processEvents()
        window.equalize_views()
        self.assertGreater(window.left.view.viewport().width(), before + 50)
        self.assertLessEqual(abs(window.left.view.viewport().width() - window.right.view.viewport().width()), 1)
        window.controls_dock.show()
        window.activate_panel(window.right)
        window.right.sidebar_tabs.setCurrentIndex(2)
        self.app.processEvents()
        self.assertTrue(window.right.metadata.isVisible())
        self.assertFalse(window.left.metadata.isVisible())
        self.assertEqual(window.navigation.indexOf(window.sample_next),
                         window.navigation.indexOf(window.sample_previous) + 1)
        self.assertEqual(window.navigation.indexOf(window.channel_next),
                         window.navigation.indexOf(window.channel_previous) + 1)

    def test_shared_tree_controls_only_clicked_panel(self):
        window = self.window
        window.compare.setChecked(True)
        self.app.processEvents()
        left, right = window.left, window.right
        QTest.mouseClick(right.view.viewport(), Qt.MouseButton.LeftButton)
        self.app.processEvents()
        self.assertIs(window.active, right)
        key_b = next(key for key in right.sample_keys if key[1] == 'B')
        window.tree.setCurrentItem(window.tree_items[key_b])
        self.assertEqual(right.sample_key, key_b)
        self.assertEqual(left.sample_key[1], 'A')
        self.assertIs(window.controls.currentWidget(), right.sidebar_tabs)
        QTest.mouseClick(left.view.viewport(), Qt.MouseButton.LeftButton)
        self.app.processEvents()
        self.assertIs(window.active, left)
        self.assertEqual(window.tree.currentItem().data(0, Qt.ItemDataRole.UserRole), left.sample_key)
        self.assertIs(window.controls.currentWidget(), left.sidebar_tabs)
        self.assertEqual(right.sample_key, key_b)
        window.sample_next.click()
        self.assertEqual(left.sample_key, key_b)
        window.compare.setChecked(False)
        self.assertIs(window.active, left)

    def test_fit_matches_frame_size_at_different_raw_resolutions(self):
        window = self.window
        folder = window.folder / 'B'
        tifffile.imwrite(folder / 'capture(Green).raw16.tif', np.full((10, 25), 1000, dtype=np.uint16))
        window.compare.setChecked(True)
        window.right.move_sample(1)
        for panel in (window.left, window.right):
            panel.active_mode = 'Green'
            panel.only_active()
        for width, height in ((1600, 900), (1300, 750)):
            window.resize(width, height)
            self.app.processEvents()
            window.equalize_views()
            rectangles = [p.view.transform().mapRect(p.view.sceneRect()) for p in (window.left, window.right)]
            self.assertAlmostEqual(rectangles[0].width(), rectangles[1].width(), delta=1)
            self.assertAlmostEqual(rectangles[0].height(), rectangles[1].height(), delta=1)
        self.assertEqual(window.left.frame.pixels.shape, (20, 50))
        self.assertEqual(window.right.frame.pixels.shape, (10, 25))

    def test_scale_lock_preserves_mapping_across_samples_and_edits(self):
        from dataclasses import replace
        window = self.window
        window.compare.setChecked(True)
        window.right.move_sample(1)
        window.activate_panel(window.left)
        mode = window.left.active_mode
        dark_high = window.left.styles[mode].high
        window.scale_lock.setChecked(True)
        left, right = window.left, window.right
        reference = replace(left.styles[mode])
        self.assertGreater(reference.high, dark_high * 9)
        self.assertEqual(right.styles[mode], reference)
        self.assertFalse(right.auto_checkbox.isChecked())
        self.assertTrue(left.scale_legend.isVisible())
        self.assertTrue(right.scale_legend.isVisible())
        before = right.frame.pixels.copy()
        right.move_sample(-1)
        right.move_sample(1)
        self.assertEqual(right.styles[mode], reference)
        window.activate_panel(right)
        right.low.setText('0')
        right.high.setText('12000')
        right.gamma.setText('1')
        right.invert.setChecked(False)
        right.draw()
        right.set_channel_color(mode, (255, 0, 0))
        self.assertEqual(left.styles[mode], right.styles[mode])
        self.assertEqual(left.styles[mode].high, 12000)
        np.testing.assert_array_equal(right.frame.pixels, before)
        left_image = left.view.item.pixmap().toImage()
        right_image = right.view.item.pixmap().toImage()
        self.assertGreater(right_image.pixelColor(0, 0).red(), left_image.pixelColor(0, 0).red())
        self.assertEqual(left_image.pixelColor(0, 18), right_image.pixelColor(0, 0))
        right.high.setText('bad')
        right.draw()
        self.assertEqual(left.styles[mode].high, 12000)
        window.scale_lock.setChecked(False)
        right.high.setText('24000')
        right.draw()
        self.assertEqual(left.styles[mode].high, 12000)
        self.assertEqual(right.styles[mode].high, 24000)

    def test_locked_auto_is_one_shot_and_new_channels_share_scale(self):
        window = self.window
        window.compare.setChecked(True)
        window.right.move_sample(1)
        window.activate_panel(window.right)
        window.scale_lock.setChecked(True)
        mode = window.right.active_mode
        window.right.set_limits(True)
        shared = window.right.styles[mode]
        self.assertEqual(window.left.styles[mode], shared)
        window.right.move_sample(-1)
        self.assertEqual(window.right.styles[mode], shared)
        for panel in (window.left, window.right):
            panel.active_mode = 'Red'
            panel.only_active()
        self.assertEqual(window.left.styles['Red'], window.right.styles['Red'])
        window.compare.setChecked(False)
        window.left.high.setText('15000')
        window.left.draw()
        window.compare.setChecked(True)
        self.assertTrue(window.scale_lock.isChecked())
        self.assertEqual(window.left.styles['Red'], window.right.styles['Red'])

    def test_locked_channel_navigation_and_unlocked_independence(self):
        window = self.window
        window.compare.setChecked(True)
        left, right = window.left, window.right
        right.move_sample(1)
        right.active_mode = 'Red'
        right.only_active()
        window.activate_panel(right)
        samples = (left.sample_key, right.sample_key)
        window.scale_lock.setChecked(True)
        self.assertEqual(left.selected_modes, {'Red'})
        self.assertEqual(right.selected_modes, {'Red'})
        window.channel_next.click()
        self.assertEqual(left.selected_modes, {'Green'})
        self.assertEqual(right.selected_modes, {'Green'})
        self.assertEqual(left.active_mode, right.active_mode)
        right.view.setFocus()
        self.app.processEvents()
        QTest.keyClick(right.view, Qt.Key.Key_Left, Qt.KeyboardModifier.AltModifier)
        self.assertEqual(left.selected_modes, {'Red'})
        self.assertEqual(right.selected_modes, {'Red'})
        index = window.channel_selector.findData('Green')
        window.channel_selector.setCurrentIndex(index)
        window.channel_selector.activated.emit(index)
        self.assertEqual(left.selected_modes, {'Green'})
        self.assertEqual(right.selected_modes, {'Green'})
        self.assertEqual((left.sample_key, right.sample_key), samples)
        self.assertEqual(left.styles['Green'], right.styles['Green'])
        window.scale_lock.setChecked(False)
        window.channel_next.click()
        self.assertEqual(left.selected_modes, {'Green'})
        self.assertEqual(right.selected_modes, {'Red'})

    def test_locked_overlay_and_hidden_panel_follow_selection(self):
        window = self.window
        window.scale_lock.setChecked(True)
        left, right = window.left, window.right
        left.active_mode = 'Green'
        left.only_active()
        right.channel_items['Red'].setCheckState(0, Qt.CheckState.Checked)
        self.assertEqual(left.selected_modes, {'Green', 'Red'})
        self.assertEqual(right.selected_modes, {'Green', 'Red'})
        self.assertIsNot(left.selected_modes, right.selected_modes)
        left.channels.setCurrentItem(left.channel_items['Red'])
        self.assertEqual(right.active_mode, 'Red')
        self.assertEqual(right.selected_modes, {'Green', 'Red'})
        for mode in ('Green', 'Red'):
            left.channel_items[mode].setCheckState(0, Qt.CheckState.Unchecked)
        self.assertEqual(right.selected_modes, set())
        self.assertIsNone(left.view.item)
        self.assertIsNone(right.view.item)
        window.compare.setChecked(False)
        left.active_mode = 'Red'
        left.only_active()
        window.compare.setChecked(True)
        self.assertEqual(right.selected_modes, {'Red'})
        self.assertEqual(right.active_mode, 'Red')

    def test_locked_missing_channel_does_not_fall_back(self):
        window = self.window
        window.compare.setChecked(True)
        left, right = window.left, window.right
        right.move_sample(1)
        # Simulate a sample without this acquisition, without deleting fixture files.
        right.samples[right.sample_key].pop('Red')
        right.cache.clear()
        window.activate_panel(left)
        window.scale_lock.setChecked(True)
        left.active_mode = 'Red'
        left.only_active()
        self.assertEqual(right.selected_modes, {'Red'})
        self.assertEqual(right.active_mode, 'Red')
        self.assertEqual(right.layers, {})
        self.assertIsNone(right.view.item)
        self.assertIn('Red: отсутствует', right.notice.text())
        right.move_sample(-1)
        self.assertEqual(set(right.layers), {'Red'})
        self.assertEqual(left.styles['Red'], right.styles['Red'])

    def test_joint_scale_is_independent_of_active_side_and_resolution(self):
        from dataclasses import replace
        window = self.window
        bright = np.arange(1000, 11000, 40, dtype=np.uint16).reshape(10, 25)
        tifffile.imwrite(window.folder / 'B' / 'capture(Green).raw16.tif', bright)
        window.compare.setChecked(True)
        left, right = window.left, window.right
        right.move_sample(1)
        for panel in (left, right):
            panel.active_mode = 'Green'
            panel.only_active()
        originals = [p.frame.pixels.copy() for p in (left, right)]
        limits = [detail_settings(p.frame.pixels) for p in (left, right)]
        window.activate_panel(left)
        window.scale_lock.setChecked(True)
        reference = replace(left.styles['Green'])
        self.assertAlmostEqual(reference.low, min(s[0] for s in limits))
        self.assertAlmostEqual(reference.high, max(s[1] for s in limits))
        self.assertEqual(reference, right.styles['Green'])
        display = render_pixels(bright, reference.low, reference.high, reference.gamma, reference.invert)
        self.assertLess(np.mean(display == 255), 0.02)
        window.scale_lock.setChecked(False)
        window.activate_panel(right)
        window.scale_lock.setChecked(True)
        self.assertEqual(reference, left.styles['Green'])
        window.activate_panel(left)
        left.high.setText('500')
        left.draw()
        left.set_limits(True)
        self.assertEqual(reference, left.styles['Green'])
        self.assertEqual(reference, right.styles['Green'])
        for panel, original in zip((left, right), originals):
            np.testing.assert_array_equal(panel.frame.pixels, original)
        right.samples[right.sample_key].pop('Green')
        left.set_limits(True)
        self.assertAlmostEqual(left.styles['Green'].high, limits[0][1])
        self.assertEqual(left.styles['Green'], right.styles['Green'])

    def test_hover_reads_inactive_panel_without_clicking(self):
        from PyQt6.QtCore import QPointF, QEvent
        window = self.window
        window.compare.setChecked(True)
        left, right = window.left, window.right
        right.move_sample(1)
        self.app.processEvents()
        window.equalize_views()
        for active, hovered in ((left, right), (right, left)):
            window.activate_panel(active)
            selected = window.tree.currentItem()
            controls = window.controls.currentWidget()
            point = hovered.view.mapFromScene(QPointF(10.5, 5.5))
            QTest.mouseMove(hovered.view.viewport(), point)
            self.app.processEvents()
            mapped = hovered.view.mapToScene(point)
            x, y = int(mapped.x()), int(mapped.y())
            mode = next(iter(hovered.layers))
            expected = int(hovered.layers[mode].pixels[y, x])
            message = window.statusBar().currentMessage()
            self.assertIn(hovered.accessibleName(), message)
            self.assertIn(f'RAW16={expected}', message)
            self.assertIs(window.active, active)
            self.assertIs(window.tree.currentItem(), selected)
            self.assertIs(window.controls.currentWidget(), controls)
            window.statusBar().clearMessage()
            hovered.view.pixel_hovered.emit(mapped.x(), mapped.y())
            self.assertIn(f'RAW16={expected}', window.statusBar().currentMessage())
            hovered.view.pixel_hovered.emit(-1, -1)
            self.assertIn('За пределами изображения', window.statusBar().currentMessage())
            QApplication.sendEvent(hovered.view.viewport(), QEvent(QEvent.Type.Leave))
            self.assertNotIn('RAW16=', window.statusBar().currentMessage())


if __name__ == '__main__':
    unittest.main()
