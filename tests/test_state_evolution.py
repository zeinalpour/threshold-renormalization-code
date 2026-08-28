import unittest

import numpy as np
from scipy.special import ndtr

from threshold_w2sg.operators import soft_threshold
from threshold_w2sg.state_evolution import (
    gaussian_soft_threshold_moments,
    solve_state_evolution,
    stage_one_moments,
)


class StateEvolutionTests(unittest.TestCase):
    def test_stage_one_active_probability_matches_gaussian_tail(self):
        p = 96
        gamma = 0.75
        sigma = 0.31
        threshold = 0.047
        result = stage_one_moments(p, gamma, sigma, threshold, 32)
        indices = np.arange(1, p + 1, dtype=np.float64)
        total_scale = np.sqrt(p * indices ** (-2.0 * gamma) + sigma**2)
        expected = float(np.mean(2.0 * ndtr(-threshold / total_scale)))
        self.assertAlmostEqual(result.active_fraction, expected, places=14)
        self.assertGreater(result.active_fraction, 0.0)
        self.assertLess(result.active_fraction, 1.0)

    def test_conditional_threshold_moments_match_monte_carlo(self):
        means = np.array([-1.0, 0.2, 1.4])
        sigma = 0.7
        threshold = 0.45
        expected_mean, expected_second, expected_active = (
            gaussian_soft_threshold_moments(means, sigma, threshold)
        )
        rng = np.random.default_rng(1701)
        samples = means + sigma * rng.standard_normal((600_000, means.size))
        outputs = soft_threshold(samples, threshold)
        np.testing.assert_allclose(outputs.mean(axis=0), expected_mean, atol=3.0e-3)
        np.testing.assert_allclose(
            np.mean(outputs**2, axis=0), expected_second, atol=6.0e-3
        )
        np.testing.assert_allclose(
            np.mean(np.abs(samples) > threshold, axis=0),
            expected_active,
            atol=2.0e-3,
        )

    def test_calibration_and_renormalization_bound(self):
        p = 48
        n = round(p**1.25)
        m = round(p**1.60)
        lambda1 = 0.05 * np.sqrt(2.0 * n / p)
        lambda2 = 0.20 * np.sqrt(2.0 * m / p)
        result = solve_state_evolution(
            p=p,
            n=n,
            m=m,
            gamma=0.75,
            delta=0.5,
            lambda1=lambda1,
            lambda2=lambda2,
            quadrature_nodes=32,
            tolerance=1.0e-10,
            max_iterations=3000,
            damping=0.6,
        )
        self.assertGreater(result.risk1, 0.0)
        self.assertGreater(result.risk2, 0.0)
        self.assertGreater(result.active1, 0.0)
        self.assertLess(result.active1, 1.0)
        self.assertGreater(result.active2, 0.0)
        self.assertLess(result.active2, 1.0)
        self.assertLess(result.composition_error, 2.0e-14)
        self.assertLessEqual(result.perturbation_mse, result.sigma2**2 * 1.000000001)
        residuals = [
            result.stage1_noise_residual,
            result.stage1_calibration_residual,
            result.stage2_noise_residual,
            result.stage2_calibration_residual,
        ]
        self.assertLess(max(residuals), 1.0e-8)

    def test_state_evolution_is_stable_across_high_quadrature_orders(self):
        p = 96
        n = round(p**1.25)
        m = round(p**1.60)
        lambda1 = 0.05 * np.sqrt(2.0 * n / p)
        lambda2 = 0.20 * np.sqrt(2.0 * m / p)
        common = dict(
            p=p,
            n=n,
            m=m,
            gamma=0.75,
            delta=0.5,
            lambda1=lambda1,
            lambda2=lambda2,
            tolerance=1.0e-11,
            max_iterations=4000,
            damping=0.6,
        )
        lower = solve_state_evolution(quadrature_nodes=256, **common)
        higher = solve_state_evolution(quadrature_nodes=512, **common)
        self.assertAlmostEqual(lower.risk1, higher.risk1, places=14)
        np.testing.assert_allclose(lower.risk2, higher.risk2, rtol=2.0e-4)
        self.assertEqual(
            np.sign(lower.risk2 - lower.risk1),
            np.sign(higher.risk2 - higher.risk1),
        )
        self.assertLess(abs(lower.active2 - higher.active2), 2.0e-3)


if __name__ == "__main__":
    unittest.main()
