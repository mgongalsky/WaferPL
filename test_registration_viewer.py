import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import math
import unittest
from unittest.mock import Mock
import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtTest import QTest, QSignalSpy
from PyQt6.QtWidgets import QApplication
from wafer_data import Frame
from registration_viewer import RegistrationDialog
from test_registration import texture


class RegistrationViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        y, x = np.mgrid[:64, :64]
        self.pixels = texture(x, y).astype(np.uint16)
        self.pixels.setflags(write=False)
        frame = Frame(self.pixels, {}, False, 1)
        self.dialog = RegistrationDialog(frame, frame, ('Левый', 'Правый'))
        self.dialog.show()
        self.app.processEvents()
        self.addCleanup(self.close_dialog)

    def close_dialog(self):
        if self.dialog.worker is not None:
            self.dialog.worker.requestInterruption()
            self.dialog.worker.wait(10000)
            self.app.processEvents()
        self.dialog.close()
        self.dialog.deleteLater()
        self.app.processEvents()

    def drag(self, view, start, end):
        start = view.mapFromScene(start)
        end = view.mapFromScene(end)
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(view.viewport(), end)
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)

    def test_shared_radius_from_both_panels_and_outside_centres(self):
        dialog = self.dialog
        self.assertFalse(hasattr(dialog, 'ratio'))
        dialog.left.fields[0].setValue(-20)
        dialog.left.fields[2].setValue(80)
        self.assertEqual(dialog.right.view.circle.radius, 80)
        dialog.right.fields[2].setValue(65)
        self.assertEqual(dialog.left.view.circle.radius, 65)
        self.assertEqual(dialog.left.view.circle.x, -20)
        dialog.right.fields[1].setValue(-15)
        self.assertEqual(dialog.right.view.circle.y, -15)
        self.assertTrue(dialog.right.fields[2].isEnabled())
        self.assertEqual(dialog.initial_transform().left.radius, dialog.initial_transform().right.radius)

    def test_mouse_draw_returns_to_handles_and_move_preserves_radius(self):
        dialog = self.dialog
        view = dialog.left.view
        view.set_zoom('200%')
        dialog.left.draw_button.setChecked(True)
        self.drag(view, QPointF(10.5, 10.5), QPointF(30.5, 10.5))
        self.assertAlmostEqual(view.circle.radius, 20, delta=0.6)
        self.assertEqual(view.circle.radius, dialog.right.view.circle.radius)
        self.assertFalse(dialog.left.draw_button.isChecked())
        self.assertFalse(view.draw_mode)
        before = view.circle
        centre = view.handle_positions()['move']
        self.drag(view, centre, centre + QPointF(8, 4))
        self.assertAlmostEqual(view.circle.x - before.x, 8, delta=0.6)
        self.assertAlmostEqual(view.circle.y - before.y, 4, delta=0.6)
        self.assertEqual(view.circle.radius, before.radius)

    def test_radius_handles_work_on_both_panels_at_different_zooms(self):
        dialog = self.dialog
        for editor, other, zoom in ((dialog.left, dialog.right, '200%'),
                                    (dialog.right, dialog.left, '100%')):
            view = editor.view
            view.set_zoom(zoom)
            before = view.circle
            handle = view.handle_positions()['radius']
            self.assertEqual(view.hit_test(view.mapFromScene(handle)), 'radius')
            self.drag(view, handle, handle + QPointF(0, 8))
            self.assertAlmostEqual(view.circle.radius, before.radius + 8, delta=0.6)
            self.assertEqual(view.circle.radius, other.view.circle.radius)
            self.assertEqual((view.circle.x, view.circle.y), (before.x, before.y))

    def test_rotation_handles_and_relative_angle(self):
        dialog = self.dialog
        for editor, degrees in ((dialog.left, -35), (dialog.right, 55)):
            view = editor.view
            view.set_zoom('200%')
            before = view.circle
            handles = view.handle_positions()
            self.assertEqual(view.hit_test(view.mapFromScene(handles['rotate'])), 'rotate')
            length = before.radius + 13
            theta = math.radians(degrees)
            end = handles['move'] + QPointF(math.cos(theta) * length, math.sin(theta) * length)
            self.drag(view, handles['rotate'], end)
            self.assertAlmostEqual(view.orientation, degrees, delta=1)
            self.assertAlmostEqual(editor.angle.value(), view.orientation, delta=0.01)
            self.assertEqual(view.circle, before)
        self.assertAlmostEqual(dialog.initial_transform().angle, 90, delta=2)
        dialog.left.angle.setValue(170)
        dialog.right.angle.setValue(-170)
        self.assertEqual(dialog.initial_transform().angle, 20)
        self.assertEqual(dialog.left.view.orientation, 170)
        self.assertEqual(dialog.right.view.orientation, -170)

    def test_live_preview_updates_and_invalidates_old_result(self):
        dialog = self.dialog
        dialog.show_manual()
        previous = dialog.output.item.pixmap().cacheKey()
        dialog.result = object()
        dialog.right.angle.setValue(30)
        self.assertIsNone(dialog.result)
        self.assertTrue(dialog.preview_timer.isActive())
        spy = QSignalSpy(dialog.preview_timer.timeout)
        self.assertTrue(spy.wait(3000))
        self.assertIsNotNone(dialog.output.item)
        self.assertNotEqual(dialog.output.item.pixmap().cacheKey(), previous)
        self.assertIn('30.00', dialog.status.text())

    def test_background_search_preserves_raw_and_stops_pending_preview(self):
        from unittest.mock import patch
        from registration import Registration
        original = self.pixels.copy()
        dialog = self.dialog
        dialog.left.angle.setValue(15)
        circles = (dialog.left.view.circle, dialog.right.view.circle)
        expected = Registration(*circles, angle=8.5, score=0.95, overlap=0.9)
        dialog.invalidate()
        self.assertTrue(dialog.preview_timer.isActive())
        with patch('registration_viewer.refine', return_value=expected):
            dialog.start()
            self.assertFalse(dialog.preview_timer.isActive())
            self.assertFalse(dialog.align.isEnabled())
            self.assertFalse(dialog.editors.isEnabled())
            spy = QSignalSpy(dialog.worker.finished)
            if not spy:
                self.assertTrue(spy.wait(15000))
            self.app.processEvents()
        self.assertIsNone(dialog.worker)
        self.assertTrue(dialog.align.isEnabled())
        self.assertIs(dialog.result, expected)
        self.assertEqual(dialog.right.angle.value(), 23.5)
        self.assertEqual(dialog.initial_transform().angle, 8.5)
        self.assertEqual((dialog.left.view.circle, dialog.right.view.circle), circles)
        self.assertIsNotNone(dialog.output.item)
        status = dialog.status.text()
        QTest.qWait(150)
        self.assertEqual(dialog.status.text(), status)
        self.assertIn('Поворот уточнён', status)
        dialog.opacity.setValue(80)
        dialog.display_mode.setCurrentIndex(1)
        dialog.show_manual()
        self.assertIs(dialog.result, expected)
        np.testing.assert_array_equal(self.pixels, original)

    def test_reject_requests_cancellation_before_closing(self):
        dialog = self.dialog
        worker = Mock()
        worker.isInterruptionRequested.return_value = True
        dialog.worker = worker
        dialog.reject()
        worker.requestInterruption.assert_called_once()
        self.assertTrue(dialog.closing)
        self.assertFalse(dialog.preview_timer.isActive())
        self.assertTrue(dialog.isVisible())
        dialog.search_finished()
        self.assertIsNone(dialog.worker)
        self.assertFalse(dialog.isVisible())

    def test_failed_rotation_keeps_manual_geometry_and_preview(self):
        from unittest.mock import patch
        dialog = self.dialog
        dialog.right.angle.setValue(12)
        dialog.show_manual()
        before = dialog.initial_transform()
        with patch('registration_viewer.refine', side_effect=ValueError('Нет общих деталей')):
            dialog.start()
            spy = QSignalSpy(dialog.worker.finished)
            if not spy:
                self.assertTrue(spy.wait(15000))
            self.app.processEvents()
        self.assertEqual(dialog.initial_transform(), before)
        self.assertIsNone(dialog.result)
        self.assertIsNotNone(dialog.output.item)
        self.assertIn('Поворот не изменён', dialog.status.text())
        self.assertTrue(dialog.align.isEnabled())

    def test_blend_controls_preserve_geometry(self):
        from registration_preview import compose_layers
        dialog = self.dialog
        dialog.show_manual()
        before = dialog.initial_transform()
        for mode in ('blend', 'color'):
            dialog.display_mode.setCurrentIndex(dialog.display_mode.findData(mode))
            self.assertTrue(dialog.opacity.isEnabled())
            for opacity in (0, 25, 50, 75, 100):
                dialog.opacity.setValue(opacity)
                expected = compose_layers(dialog.preview_layers, mode, opacity / 100)
                actual = dialog.output.item.pixmap().toImage().pixelColor(32, 32)
                self.assertEqual(actual.getRgb()[:3], tuple(expected[32, 32]))
                self.assertEqual(dialog.initial_transform(), before)


if __name__ == '__main__':
    unittest.main()
