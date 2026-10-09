import unittest
import numpy as np
from registration import Circle, Registration
from feature_registration import fit_correspondences, extract_features


class FeatureRegistrationTests(unittest.TestCase):
    def test_rigid_consensus_rejects_outliers_without_changing_scale(self):
        rng = np.random.default_rng(17)
        circle = Circle(250, 250, 230)
        a = rng.uniform(100, 400, (60, 2))
        expected = Registration(circle, circle, 8.5, 5.2, -3.8)
        bx, by = expected.coordinates(a[:, 0], a[:, 1])
        b = np.column_stack((bx, by)) + rng.normal(0, 0.15, a.shape)
        b[40:] = rng.uniform(70, 430, (20, 2))
        result = fit_correspondences(a, b, circle, circle, angle=4)
        self.assertAlmostEqual(result.transform.angle, 8.5, delta=0.15)
        self.assertAlmostEqual(result.transform.dx, 5.2, delta=0.2)
        self.assertAlmostEqual(result.transform.dy, -3.8, delta=0.2)
        self.assertGreaterEqual(len(result.errors), 39)
        self.assertLess(result.rms, 0.5)
        self.assertEqual(result.transform.left, circle)
        self.assertEqual(result.transform.right, circle)

    def test_cluster_and_unrelated_points_are_rejected(self):
        rng = np.random.default_rng(91)
        circle = Circle(250, 250, 230)
        cluster = rng.normal(250, 2, (30, 2))
        with self.assertRaises(ValueError):
            fit_correspondences(cluster, cluster, circle, circle)
        a, b = rng.uniform(80, 420, (2, 40, 2))
        with self.assertRaises(ValueError):
            fit_correspondences(a, b, circle, circle)

    def test_cancellation(self):
        circle = Circle(32, 32, 30)
        with self.assertRaises(InterruptedError):
            extract_features(np.ones((64, 64)), circle, lambda: True)

    def test_sift_extracts_distinct_points_without_changing_raw(self):
        rng = np.random.default_rng(27)
        y, x = np.mgrid[:384, :384]
        values = np.full(x.shape, 3000.0)
        for cx, cy, radius, intensity in zip(rng.uniform(65, 320, 60), rng.uniform(65, 320, 60),
                                           rng.uniform(2, 7, 60), rng.uniform(3000, 22000, 60)):
            values += intensity*np.exp(-((x-cx)**2+(y-cy)**2)/(2*radius**2))
        image = np.clip(values, 0, 65535).astype(np.uint16)
        original = image.copy()
        image.setflags(write=False)
        points, descriptors = extract_features(image, Circle(192, 192, 180))
        self.assertGreaterEqual(len(points), 8)
        self.assertEqual(descriptors.shape, (len(points), 128))
        np.testing.assert_array_equal(image, original)


if __name__ == '__main__':
    unittest.main()
