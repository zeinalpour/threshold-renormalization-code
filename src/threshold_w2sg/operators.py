"""Basic convex-analysis operators and diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def soft_threshold(x: FloatArray, threshold: float) -> FloatArray:
    """Apply scalar soft thresholding elementwise."""
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")
    values = np.asarray(x, dtype=np.float64)
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def lasso_objective(
    gram: FloatArray,
    score: FloatArray,
    coef: FloatArray,
    penalty: float,
) -> float:
    """Return the Gram-form LASSO objective, up to a constant."""
    return float(
        0.5 * coef @ (gram @ coef) - score @ coef + penalty * np.abs(coef).sum()
    )


def kkt_residual(
    gram: FloatArray,
    score: FloatArray,
    coef: FloatArray,
    penalty: float,
    zero_tolerance: float = 1.0e-12,
) -> float:
    """Infinity-norm violation of the LASSO KKT conditions."""
    gradient = gram @ coef - score
    active = np.abs(coef) > zero_tolerance
    active_residual = 0.0
    inactive_residual = 0.0
    if np.any(active):
        active_residual = float(
            np.max(np.abs(gradient[active] + penalty * np.sign(coef[active])))
        )
    if np.any(~active):
        inactive_residual = float(
            np.max(np.maximum(np.abs(gradient[~active]) - penalty, 0.0))
        )
    return max(active_residual, inactive_residual)

