import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PyQt6.QtTest import QSignalSpy
from PyQt6.QtWidgets import QApplication
from registration import Circle, Registration
from feature_registration import FeatureResult, refine_features
from registration_workspace import RegistrationWorkspace
from wafer_data import Frame


class FeatureImageTests(unittest.TestCase):
    def test_changed_spot_width_brightness_and_region(self):
        rng = np.random.default_rng(392)
        y, x = np.mgrid[:512, :512]
        circle = Circle(255, 255, 235)
        centres = rng.uniform(90, 422, (110, 2))
        widths = rng.uniform(2.3, 6.0, len(centres))
        amplitudes = rng.uniform(3000, 13000, len(centres))
        angle, dx, dy = 6.5, 4.2, -3.1
        c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        u, v = x-circle.x-dx, y-circle.y-dy
        sx = circle.x+c*u+s*v
        sy = circle.y-s*u+c*v
        a = np.full(x.shape, 2000.0)
        b = np.full(x.shape, 3500.0)
        for i, ((cx, cy), width, amplitude) in enumerate(zip(centres, widths, amplitudes)):
            a += amplitude*np.exp(-((x-cx)**2+(y-cy)**2)/(2*width**2))
            # Modify one quadrant completely, and change widths/exposure elsewhere.
            if cx > 255 and cy < 255:
                continue
            modified_width = width*(1.12 if i % 2 else 0.92)
            b += amplitude*(1.3 if i % 3 else 0.85)*np.exp(
                -((sx-cx)**2+(sy-cy)**2)/(2*modified_width**2))
        b += 11000*np.exp(-((x-350)**2+(y-150)**2)/180)
        a, b = [np.clip(image, 0, 65535).astype(np.uint16) for image in (a, b)]
        originals = a.copy(), b.copy()
        a.setflags(write=False)
        b.setflags(write=False)
        result = refine_features(a, b, circle, circle, angle=2)
        self.assertAlmostEqual(result.transform.angle, angle, delta=0.5)
        self.assertAlmostEqual(result.transform.dx, dx, delta=1)
        self.assertAlmostEqual(result.transform.dy, dy, delta=1)
        self.assertGreaterEqual(len(result.errors), 8)
        self.assertLess(result.rms, 1.7)
        for image, original in zip((a, b), originals):
            np.testing.assert_array_equal(image, original)


class FeatureWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        pixels = np.arange(4096, dtype=np.uint16).reshape(64, 64)
        frame = Frame(pixels, {}, False, 1)
        sources = []
        for name in ('left', 'right'):
            path = self.root / (name+'.tif')
            path.write_bytes(b'fixture')
            stat = path.stat()
            sources.append({'path': str(path.resolve()), 'page': 0, 'shape': [64, 64],
                            'dtype': 'uint16', 'white_is_zero': False,
                            'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
        self.window = RegistrationWorkspace(frame, frame, ('Левый', 'Правый'), sources=sources)
        self.window.show()
        self.app.processEvents()
        self.addCleanup(self.close_window)

    def close_window(self):
        window = self.window
        if window.worker is not None:
            window.worker.requestInterruption()
            window.worker.wait(10000)
            self.app.processEvents()
        window.close()
        window.deleteLater()
        self.app.processEvents()

    def wait_worker(self):
        spy = QSignalSpy(self.window.worker.finished)
        if not spy:
            self.assertTrue(spy.wait(10000))
        self.app.processEvents()
        self.assertIsNone(self.window.worker)

    def test_worker_applies_shift_and_saved_mapping_matches_evidence(self):
        window = self.window
        initial = window.initial_transform()
        transform = Registration(initial.left, initial.right, 7.25, 2.2, -1.1, score=0.8)
        points = np.array([[10., 10.], [45., 15.], [20., 48.], [40., 40.]])
        bx, by = transform.coordinates(points[:, 0], points[:, 1])
        evidence = FeatureResult(transform, points, np.column_stack((bx, by)), np.zeros(4), 5)
        with patch('feature_registration_ui.refine_features', return_value=evidence):
            window.start()
            self.assertFalse(window.editors.isEnabled())
            self.assertFalse(window.save_button.isEnabled())
            self.wait_worker()
        self.assertTrue(window.matches_button.isEnabled())
        self.assertTrue(window.save_button.isEnabled())
        self.assertAlmostEqual(window.right.view.circle.x, initial.right.x+2.2)
        self.assertAlmostEqual(window.right.view.circle.y, initial.right.y-1.1)
        self.assertEqual(window.right.view.circle.radius, initial.right.radius)
        self.assertIn('RMS', window.status.text())
        actual_x, actual_y = window.result.coordinates(points[:, 0], points[:, 1])
        np.testing.assert_allclose(actual_x, bx)
        np.testing.assert_allclose(actual_y, by)
        expected = window.session_document()
        path = self.root / 'alignment.json'
        window.save_parameters(path)
        with patch('feature_registration_ui.show_correspondences') as show:
            window.show_matches()
            show.assert_called_once()
        window.right.fields[0].setValue(0)
        self.assertFalse(window.matches_button.isEnabled())
        window.load_parameters(path)
        self.assertEqual(window.session_document(), expected)
        self.assertFalse(window.matches_button.isEnabled())
        self.assertIsNotNone(window.output.item)

    def test_failed_search_preserves_previous_alignment(self):
        window = self.window
        window.right.angle.setValue(12)
        window.right.fields[0].setValue(34)
        window.show_manual()
        before = window.session_document()
        with patch('feature_registration_ui.refine_features', side_effect=ValueError('Нет общих точек')):
            window.start()
            self.wait_worker()
        self.assertEqual(window.session_document(), before)
        self.assertIn('Совмещение не изменено', window.status.text())
        self.assertTrue(window.align.isEnabled())
        self.assertIsNotNone(window.output.item)


if __name__ == '__main__':
    unittest.main()
