import unittest

import numpy as np

from threshold_w2sg.wishart import sample_gaussian_gram, sample_stage1_score


class WishartTests(unittest.TestCase):
    def test_factorization_and_mean(self):
        rng = np.random.default_rng(31415)
        p = 4
        n = 20
        draws = []
        for _ in range(2000):
            gram, factor = sample_gaussian_gram(n, p, rng)
            np.testing.assert_allclose(gram, factor @ factor.T, atol=1.0e-12)
            draws.append(gram)
        empirical = np.mean(draws, axis=0)
        np.testing.assert_allclose(
            np.diag(empirical), np.full(p, n / p), rtol=0.025, atol=0.0
        )
        off_diagonal = empirical - np.diag(np.diag(empirical))
        self.assertLess(np.max(np.abs(off_diagonal)), 0.08)

    def test_conditional_stage1_score_is_deterministic_by_seed(self):
        beta = np.array([1.0, -2.0, 0.5])
        rng1 = np.random.default_rng(77)
        gram1, factor1 = sample_gaussian_gram(12, 3, rng1)
        score1 = sample_stage1_score(gram1, factor1, beta, 0.4, rng1)
        rng2 = np.random.default_rng(77)
        gram2, factor2 = sample_gaussian_gram(12, 3, rng2)
        score2 = sample_stage1_score(gram2, factor2, beta, 0.4, rng2)
        np.testing.assert_array_equal(gram1, gram2)
        np.testing.assert_array_equal(score1, score2)


if __name__ == "__main__":
    unittest.main()

