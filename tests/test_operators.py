import unittest

import numpy as np

from threshold_w2sg.operators import soft_threshold


class SoftThresholdTests(unittest.TestCase):
    def test_composition_identity(self):
        values = np.linspace(-4.0, 4.0, 1001)
        first = 0.37
        second = 0.81
        composed = soft_threshold(soft_threshold(values, first), second)
        direct = soft_threshold(values, first + second)
        np.testing.assert_allclose(composed, direct, rtol=0.0, atol=2.0e-15)

    def test_nonnegative_threshold_required(self):
        with self.assertRaises(ValueError):
            soft_threshold(np.array([1.0]), -0.1)


if __name__ == "__main__":
    unittest.main()

