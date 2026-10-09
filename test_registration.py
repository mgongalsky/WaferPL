import unittest
import numpy as np
from registration import Circle, Registration, refine, preview, sample
from registration_preview import compose_layers


def texture(x, y):
    result = 3000 + 500*np.sin(x/7.3) + 700*np.cos(y/9.1) + 400*np.sin((x+y)/5.7)
    for cx, cy, width, amplitude in ((25, 34, 7, 9000), (82, 45, 12, 13000),
                                      (55, 91, 9, 17000), (99, 103, 6, 8000)):
        result = result + amplitude*np.exp(-((x-cx)**2+(y-cy)**2)/(2*width**2))
    return result


def wafer_pattern(x, y, circle, angle=0, modified=False):
    u, v = (x-circle.x)/circle.radius, (y-circle.y)/circle.radius
    radius = np.hypot(u, v)
    theta = np.arctan2(v, u)-np.deg2rad(angle)
    details = (np.sin(17*theta+9*radius) + 0.7*np.cos(29*theta-13*radius)
               + 0.5*np.sin(11*theta+21*radius))
    if modified:
        # Replace a quarter of the wafer, rather than merely changing exposure.
        changed = (np.mod(theta, 2*np.pi) < np.pi/2)
        details = np.where(changed, 2*np.sin(41*theta-6*radius), details)
    return 18000 + 3000*details + 2000*radius


class RegistrationTests(unittest.TestCase):
    def test_rotation_with_modified_regions_keeps_circle_geometry(self):
        y, x = np.mgrid[:192, :192]
        left, right = Circle(94, 95, 86), Circle(99, 90, 86)
        a = wafer_pattern(x, y, left).astype(np.uint16)
        b = (wafer_pattern(x, y, right, 8.5, True)*1.4+500).astype(np.uint16)
        originals = (a.copy(), b.copy())
        a.setflags(write=False)
        b.setflags(write=False)
        result = refine(a, b, left, right, angle=4)
        self.assertAlmostEqual(result.angle, 8.5, delta=0.6)
        self.assertEqual((result.dx, result.dy), (0, 0))
        self.assertEqual((result.left, result.right), (left, right))
        self.assertGreater(result.score, 0.7)
        image = preview(a, b, result, limit=64)
        self.assertLessEqual(max(image.shape[:2]), 64)
        for actual, original in zip((a, b), originals):
            np.testing.assert_array_equal(actual, original)

    def test_cropped_circle_with_centre_outside_frame(self):
        y, x = np.mgrid[:160, :160]
        circle = Circle(-10, 78, 145)
        source = wafer_pattern(x, y, circle)
        result = refine(source, source*2+300, circle, circle)
        self.assertAlmostEqual(result.angle, 0, delta=0.5)
        self.assertEqual((result.dx, result.dy), (0, 0))
        self.assertGreater(result.score, 0.99)

    def test_radial_and_repeated_patterns_are_not_orientation_evidence(self):
        y, x = np.mgrid[:192, :192]
        circle = Circle(95, 95, 86)
        radius = np.hypot(x-95, y-95)
        theta = np.arctan2(y-95, x-95)
        for source in (np.ones(x.shape), 10000+radius*100,
                       10000+2000*np.sin(36*theta)):
            with self.subTest(pattern=float(np.std(source))):
                with self.assertRaises(ValueError):
                    refine(source, source, circle, circle)

    def test_missing_overlap_invalid_circle_and_cancellation(self):
        y, x = np.mgrid[:96, :96]
        circle = Circle(48, 48, 43)
        image = wafer_pattern(x, y, circle)
        with self.assertRaises(ValueError):
            refine(image, image, Circle(-200, -200, 10), circle)
        with self.assertRaises(ValueError):
            refine(image, image, circle, Circle(1000, 1000, 43))
        for radius in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                Circle(0, 0, radius)
        with self.assertRaises(InterruptedError):
            refine(image, image, circle, circle, cancelled=lambda: True)

    def test_sampling_does_not_wrap_and_transform_direction(self):
        image = np.arange(9).reshape(3, 3)
        values, valid = sample(image, np.array([-1., 0.5, 2., 3.]), np.array([0., 0.5, 2., 0.]))
        np.testing.assert_array_equal(valid, [False, True, True, False])
        np.testing.assert_allclose(values, [0, 2, 8, 0])
        transform = Registration(Circle(10, 20, 5), Circle(30, 40, 10), 90, 2, -3)
        x, y = transform.coordinates(15, 20)
        self.assertAlmostEqual(x, 32)
        self.assertAlmostEqual(y, 47)

    def test_blending_endpoints_and_missing_coverage(self):
        a = np.array([[20, 60, 90]], dtype=np.uint8)
        b = np.array([[100, 180, 220]], dtype=np.uint8)
        va = np.array([[True, True, False]])
        vb = np.array([[True, False, True]])
        layers = a, b, va, vb
        np.testing.assert_array_equal(compose_layers(layers, alpha=0)[..., 0], [[20, 60, 0]])
        np.testing.assert_array_equal(compose_layers(layers, alpha=1)[..., 0], [[100, 0, 220]])
        np.testing.assert_array_equal(compose_layers(layers, alpha=0.5)[..., 0], [[60, 60, 220]])
        color = compose_layers(layers, mode='color')
        np.testing.assert_array_equal(color[0, 0], [20, 100, 100])


if __name__ == '__main__':
    unittest.main()
