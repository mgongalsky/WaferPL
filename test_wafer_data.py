from pathlib import Path
import tempfile
import unittest

import numpy as np
import tifffile

from wafer_data import (discover, read_frame, render_pixels, pixel_value,
                        display_limits, image_coordinates, scn_metadata)


class MeasurementTests(unittest.TestCase):
    def test_16bit_measurements_survive_display_changes(self):
        source = np.array([[0, 256, 65535], [17, 4096, 32768]], dtype=np.uint16)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'signal.tif'
            tifffile.imwrite(path, source, photometric='miniswhite')
            frame = read_frame(path)
            self.assertTrue(frame.white_is_zero)
            for gamma in (0.5, 1, 2):
                render_pixels(frame.pixels, 100, 10000, gamma, True)
                np.testing.assert_array_equal(frame.pixels, source)
            self.assertEqual(pixel_value(frame.pixels, 2, 0), 65535)
            self.assertFalse(frame.pixels.flags.writeable)
            with self.assertRaises(IndexError):
                pixel_value(frame.pixels, -1, 0)

    def test_display_endpoints_and_invalid_limits(self):
        source = np.array([[0, 32768, 65535]], dtype=np.uint16)
        np.testing.assert_array_equal(render_pixels(source, 0, 65535), [[0, 128, 255]])
        np.testing.assert_array_equal(render_pixels(source, 0, 65535, invert=True), [[255, 127, 0]])
        for low, high, gamma in ((0, 0, 1), (1, 0, 1), (0, 1, 0), (0, float('nan'), 1)):
            with self.assertRaises(ValueError):
                render_pixels(source, low, high, gamma)
        low, high = display_limits(np.full((2, 2), 42, dtype=np.uint16))
        self.assertGreater(high, low)

    def test_zoom_coordinates_and_outside_image(self):
        self.assertEqual(image_coordinates(50, 25, (100, 50), (2750, 2200)), (1375, 1100))
        self.assertEqual(image_coordinates(99.9, 49.9, (100, 50), (2750, 2200)), (2747, 2195))
        self.assertIsNone(image_coordinates(-0.1, 0, (100, 50), (2750, 2200)))
        self.assertIsNone(image_coordinates(100, 0, (100, 50), (2750, 2200)))

    def test_rgb_and_multiple_pages(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'pages.tif'
            rgb = np.array([[[1, 2, 3], [250, 251, 252]]], dtype=np.uint8)
            with tifffile.TiffWriter(path) as writer:
                writer.write(rgb, photometric='rgb')
                writer.write(np.array([[60000]], dtype=np.uint16))
            self.assertEqual(pixel_value(read_frame(path).pixels, 1, 0), [250, 251, 252])
            frame = read_frame(path, 1)
            self.assertEqual(frame.page_count, 2)
            self.assertEqual(pixel_value(frame.pixels, 0, 0), 60000)

    def test_sample_labels_require_matching_capture_name(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'MAX 12h00(Flamingo).raw16.tif').touch()
            (root / 'MAX 12h00(Flamingo).tif').touch()
            (root / 'Mono16MO_MAX 12h00(Flamingo).jpg').touch()
            captures = discover(root)
            self.assertEqual(len(captures), 2)
            self.assertTrue(captures[0].raw)
            self.assertEqual([c.sample for c in captures], ['Mono16MO', 'Mono16MO'])
            self.assertEqual(captures[0].mode, 'Flamingo')

    def test_scn_metadata_amid_binary_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'capture.scn'
            path.write_bytes(b'\x00\xffbinary\r\n<root><protocol_settings><application>'
                             b'<application_name>Cy7</application_name></application>'
                             b'<exposure_time>30</exposure_time></protocol_settings></root>\x00')
            metadata = scn_metadata(path)
            self.assertEqual(metadata['application_name'], 'Cy7')
            self.assertEqual(metadata['exposure_time'], '30')


class LocalDataTests(unittest.TestCase):
    def test_all_project_tiffs_decode_without_losing_bit_depth(self):
        root = Path(__file__).resolve().parent / 'data'
        captures = discover(root)
        if not captures:
            self.skipTest('Local instrument data is not present')
        for capture in captures:
            with self.subTest(file=capture.path.name):
                frame = read_frame(capture.path)
                self.assertGreater(frame.pixels.size, 0)
                if capture.raw:
                    self.assertEqual(frame.pixels.dtype, np.dtype('uint16'))
                    self.assertEqual(frame.pixels.ndim, 2)
                value = pixel_value(frame.pixels, 0, 0)
                self.assertEqual(value, frame.pixels[0, 0].tolist())


if __name__ == '__main__':
    unittest.main()
