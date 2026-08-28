import unittest

import numpy as np

from threshold_w2sg.lasso import solve_lasso_gram
from threshold_w2sg.operators import kkt_residual, soft_threshold


def reference_coordinate_descent(gram, score, penalty, tolerance=1.0e-12):
    coef = np.zeros_like(score)
    gradient = -score.copy()
    for _ in range(100_000):
        largest = 0.0
        for j in range(score.size):
            partial = score[j] - (gram[j] @ coef - gram[j, j] * coef[j])
            updated = soft_threshold(np.array([partial]), penalty)[0] / gram[j, j]
            change = updated - coef[j]
            if change != 0.0:
                coef[j] = updated
                largest = max(largest, abs(change))
        if largest <= tolerance:
            break
    return coef


class LassoSolverTests(unittest.TestCase):
    def test_matches_reference_solution(self):
        rng = np.random.default_rng(107)
        design = rng.standard_normal((80, 12))
        gram = design.T @ design / 12.0
        score = rng.standard_normal(12)
        penalty = 0.23
        expected = reference_coordinate_descent(gram, score, penalty)
        actual = solve_lasso_gram(
            gram,
            score,
            penalty,
            tolerance=1.0e-10,
            max_iterations=20_000,
        )
        self.assertTrue(actual.converged)
        self.assertLessEqual(actual.kkt_residual, 1.0e-10)
        np.testing.assert_allclose(actual.coef, expected, rtol=2.0e-8, atol=2.0e-9)
        self.assertLessEqual(
            kkt_residual(gram, score, actual.coef, penalty), 1.0e-10
        )


if __name__ == "__main__":
    unittest.main()

