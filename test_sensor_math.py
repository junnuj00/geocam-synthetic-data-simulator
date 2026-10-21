"""Run without Blender: python -m unittest test_sensor_math.py"""
import math
import tempfile
import unittest
from pathlib import Path

from sensor_math import add_depth_noise, intrinsics, project_camera_point, save_npy_f32


class TestSensorMath(unittest.TestCase):
    def test_pinhole_center_and_shift(self):
        K = intrinsics(640, 480, 35, 36)
        self.assertEqual(project_camera_point(0, 0, 10, K), (320, 240))
        u, v = project_camera_point(1, 2, 10, K)
        self.assertAlmostEqual(u, 320 + (640 * 35 / 36) / 10)
        self.assertAlmostEqual(v, 240 - 2 * (640 * 35 / 36) / 10)
        with self.assertRaises(ValueError):
            project_camera_point(0, 0, -1, K)

    def test_noise_seed_and_invalid(self):
        raw = [1.0, 3.0, float("inf"), 0.0, 20.0]
        a = add_depth_noise(raw, seed=11, dropout_probability=0)
        b = add_depth_noise(raw, seed=11, dropout_probability=0)
        self.assertEqual(a[:2], b[:2])
        self.assertTrue(math.isnan(a[2]))
        self.assertTrue(math.isnan(a[3]))
        self.assertEqual(raw[0], 1.0)

    def test_npy_pixel_top_down(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "depth.npy"
            # Source is bottom-up: first row bottom [3,4]; second row top [1,2]
            save_npy_f32(p, [3., 4., 1., 2.], width=2, height=2)
            import numpy as np
            observed = np.load(p)
            self.assertEqual(observed.shape, (2, 2))
            self.assertEqual(observed.dtype.name, "float32")
            self.assertEqual(observed.tolist(), [[1., 2.], [3., 4.]])


if __name__ == "__main__":
    unittest.main()
