"""Exact Gaussian-Wishart sufficient-statistics simulation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def sample_power_law_signal(
    p: int,
    gamma: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw independent ``N(0, p*i^(-2 gamma))`` coordinates."""
    indices = np.arange(1, p + 1, dtype=np.float64)
    scales = np.sqrt(p) * indices ** (-gamma)
    return scales * rng.standard_normal(p)


def sample_gaussian_gram(
    sample_size: int,
    p: int,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    """Sample ``X.T @ X / p`` exactly using Bartlett decomposition.

    Returns the Gram matrix and a factor ``factor`` satisfying
    ``gram = factor @ factor.T``. The method requires ``sample_size >= p``,
    which holds in every configured oversampled experiment.
    """
    if sample_size < p:
        raise ValueError("Bartlett sampler requires sample_size >= p")
    bartlett = np.zeros((p, p), dtype=np.float64)
    lower = np.tril_indices(p, k=-1)
    bartlett[lower] = rng.standard_normal(lower[0].size)
    degrees = sample_size - np.arange(p, dtype=np.int64)
    bartlett[np.diag_indices(p)] = np.sqrt(rng.chisquare(degrees))
    factor = bartlett / np.sqrt(float(p))
    gram = factor @ factor.T
    return np.ascontiguousarray(gram), np.ascontiguousarray(factor)


def sample_stage1_score(
    gram: FloatArray,
    gram_factor: FloatArray,
    beta_star: FloatArray,
    delta: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Sample the exact Stage-I score conditional on the Gram matrix."""
    projected_noise = gram_factor @ rng.standard_normal(beta_star.size)
    return gram @ beta_star + np.sqrt(delta) * projected_noise

