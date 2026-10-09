import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication

from registration import Registration
from registration_preview import compose_layers
from registration_session import save_session, validate_session
from registration_workspace import RegistrationWorkspace
from wafer_data import Frame


class RegistrationWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        y, x = np.mgrid[:80, :80]
        self.pixels = (1000 + x * 200 + y * 100).astype(np.uint16)
        self.original = self.pixels.copy()
        self.pixels.setflags(write=False)
        self.sources = []
        for name in ('left', 'right'):
            path = self.root / (name + '.raw16.tif')
            path.write_bytes(name.encode('ascii'))
            stat = path.stat()
            self.sources.append({
                'path': str(path.resolve()), 'page': 0,
                'shape': [80, 80], 'dtype': 'uint16', 'white_is_zero': False,
                'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                'sample': name, 'channel': 'Red',
            })
        frame = Frame(self.pixels, {}, False, 1)
        self.window = RegistrationWorkspace(
            frame, frame, ('Левый', 'Правый'), sources=self.sources)
        self.window.show()
        self.app.processEvents()
        self.window.show_manual()
        self.addCleanup(self.close_window)

    def close_window(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_result_occupies_right_side_and_grows_with_window(self):
        window = self.window
        window.resize(1500, 900)
        self.app.processEvents()
        splitter = window.workspace_splitter
        self.assertEqual(splitter.count(), 2)
        self.assertGreater(splitter.widget(1).width(), splitter.widget(0).width())
        self.assertGreater(window.output.height(), window.height() * 0.65)
        first = window.output.height()
        window.resize(1500, 1100)
        self.app.processEvents()
        self.assertGreater(window.output.height(), first + 100)
        self.assertGreater(window.output.maximumHeight(), 1100)

    def test_color_selection_changes_display_without_geometry_changes(self):
        window = self.window
        before = window.initial_transform()
        with patch('registration_workspace.QColorDialog.getColor', return_value=QColor(0, 255, 0)):
            window.choose_color(0)
        with patch('registration_workspace.QColorDialog.getColor', return_value=QColor(255, 0, 255)):
            window.choose_color(1)
        self.assertEqual(window.colors, [(0, 255, 0), (255, 0, 255)])
        self.assertEqual(window.display_mode.currentData(), 'color')
        expected = compose_layers(window.preview_layers, 'color', colors=window.colors)
        actual = window.output.item.pixmap().toImage().pixelColor(40, 40)
        self.assertEqual((actual.red(), actual.green(), actual.blue()), tuple(expected[40, 40]))
        with patch('registration_workspace.QColorDialog.getColor', return_value=QColor()):
            window.choose_color(0)
        self.assertEqual(window.colors[0], (0, 255, 0))
        self.assertEqual(window.initial_transform(), before)
        np.testing.assert_array_equal(self.pixels, self.original)

    def test_manual_and_refined_sessions_round_trip(self):
        window = self.window
        for refined in (False, True):
            with self.subTest(refined=refined):
                window.left.fields[0].setValue(-12)
                window.left.fields[2].setValue(55)
                window.right.fields[1].setValue(28)
                window.left.angle.setValue(170)
                window.right.angle.setValue(-175)
                initial = window.initial_transform()
                window.result = Registration(
                    initial.left, initial.right, initial.angle, score=0.91, overlap=0.8
                ) if refined else None
                window.colors = [(20, 220, 80), (180, 30, 240)]
                window.display_mode.setCurrentIndex(1)
                window.opacity.setValue(73)
                expected = window.session_document()
                path = self.root / ('refined.json' if refined else 'manual.json')
                window.save_parameters(path)
                stored = json.loads(path.read_text(encoding='utf-8'))
                self.assertEqual(stored, expected)
                matrix = np.asarray(stored['matrix_left_to_right'])
                transform = window.result or window.initial_transform()
                point = np.array([21.5, 33.5, 1.0])
                np.testing.assert_allclose((matrix @ point)[:2], transform.coordinates(*point[:2]))
                window.right.angle.setValue(0)
                window.left.fields[2].setValue(25)
                window.colors = [(255, 0, 0), (0, 255, 255)]
                window.opacity.setValue(10)
                window.load_parameters(path)
                self.assertEqual(window.session_document(), expected)
                self.assertEqual(window.result is not None, refined)
                self.assertFalse(window.preview_timer.isActive())
                self.assertIsNotNone(window.output.item)
        np.testing.assert_array_equal(self.pixels, self.original)

    def test_wrong_sources_and_invalid_parameters_do_not_change_session(self):
        window = self.window
        original = window.session_document()
        mutations = (
            lambda d: d['sources'][0].update(page=1),
            lambda d: d['sources'][1].update(mtime_ns=d['sources'][1]['mtime_ns'] + 1),
            lambda d: d['transform']['left'].update(radius=-2),
            lambda d: d['transform'].update(dx=4),
            lambda d: d['display'].update(colors=[[256, 0, 0], [0, 255, 255]]),
            lambda d: d.update(version=999),
        )
        path = self.root / 'invalid.json'
        for mutate in mutations:
            document = copy.deepcopy(original)
            mutate(document)
            path.write_text(json.dumps(document), encoding='utf-8')
            with self.assertRaises(ValueError):
                window.load_parameters(path)
            self.assertEqual(window.session_document(), original)
        path.write_text('{broken', encoding='utf-8')
        with self.assertRaises(ValueError):
            window.load_parameters(path)
        self.assertEqual(window.session_document(), original)

    def test_atomic_save_preserves_previous_file_on_failure(self):
        path = self.root / 'session.json'
        document = self.window.session_document()
        save_session(path, document)
        before = path.read_bytes()
        files_before = set(self.root.iterdir())
        with patch('registration_session.os.replace', side_effect=OSError('write failed')):
            with self.assertRaises(OSError):
                save_session(path, document)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(set(self.root.iterdir()), files_before)
        source = Path(self.sources[0]['path'])
        original = source.read_bytes()
        with self.assertRaises(ValueError):
            self.window.save_parameters(source)
        self.assertEqual(source.read_bytes(), original)

    def test_matrix_and_orientation_inconsistency_are_rejected(self):
        original = self.window.session_document()
        bad_matrix = copy.deepcopy(original)
        bad_matrix['matrix_left_to_right'][0][2] += 1
        bad_orientation = copy.deepcopy(original)
        bad_orientation['orientations'][1] += 10
        for document in (bad_matrix, bad_orientation):
            with self.assertRaises(ValueError):
                validate_session(document, self.sources)


if __name__ == '__main__':
    unittest.main()
