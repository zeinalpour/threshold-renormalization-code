"""Deterministic Gaussian quadrature utilities."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.special import roots_hermitenorm


FloatArray = NDArray[np.float64]


@lru_cache(maxsize=None)
def standard_normal_rule(nodes: int) -> tuple[FloatArray, FloatArray]:
    """Return nodes and weights for expectations under ``N(0,1)``."""
    if nodes < 8:
        raise ValueError("at least eight Gauss-Hermite nodes are required")
    # ``roots_hermitenorm`` works directly with the exp(-x^2/2) weight and is
    # numerically stable at the higher orders used by the publication profile.
    raw_nodes, raw_weights = roots_hermitenorm(nodes)
    return (
        raw_nodes.astype(np.float64),
        (raw_weights / np.sqrt(2.0 * np.pi)).astype(np.float64),
    )
