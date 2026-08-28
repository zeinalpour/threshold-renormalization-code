"""Deterministic accelerated proximal solver for dense Gram-form LASSO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .operators import kkt_residual, lasso_objective, soft_threshold


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LassoResult:
    coef: FloatArray
    converged: bool
    iterations: int
    kkt_residual: float
    objective: float
    lipschitz_constant: float
    restart_attempt: int


def _spectral_radius(
    gram: FloatArray,
    iterations: int,
) -> float:
    """Deterministically estimate the largest eigenvalue by power iteration."""
    p = gram.shape[0]
    vector = np.linspace(1.0, 2.0, p, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    for _ in range(iterations):
        updated = gram @ vector
        norm = np.linalg.norm(updated)
        if not np.isfinite(norm) or norm == 0.0:
            raise FloatingPointError("invalid Gram matrix in power iteration")
        vector = updated / norm
    return float(vector @ (gram @ vector))


def _run_fista(
    gram: FloatArray,
    score: FloatArray,
    penalty: float,
    lipschitz: float,
    tolerance: float,
    max_iterations: int,
    check_every: int,
    initial: FloatArray | None,
) -> tuple[FloatArray, int, float]:
    diagonal = np.diag(gram)
    if np.any(diagonal <= 0.0):
        raise ValueError("Gram matrix must have a strictly positive diagonal")
    if initial is None:
        current = soft_threshold(score, penalty) / diagonal
    else:
        current = np.asarray(initial, dtype=np.float64).copy()
    extrapolated = current.copy()
    momentum = 1.0
    residual = np.inf

    for iteration in range(1, max_iterations + 1):
        gradient = gram @ extrapolated - score
        updated = soft_threshold(
            extrapolated - gradient / lipschitz,
            penalty / lipschitz,
        )

        if iteration == 1 or iteration % check_every == 0:
            residual = kkt_residual(gram, score, updated, penalty)
            if residual <= tolerance:
                return updated, iteration, residual

        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
        next_extrapolated = updated + (
            (momentum - 1.0) / next_momentum
        ) * (updated - current)

        # Gradient-based adaptive restart prevents oscillation in strongly
        # convex, well-conditioned oversampled problems.
        if float((next_extrapolated - updated) @ (updated - current)) > 0.0:
            next_extrapolated = updated.copy()
            next_momentum = 1.0

        current = updated
        extrapolated = next_extrapolated
        momentum = next_momentum

    residual = kkt_residual(gram, score, current, penalty)
    return current, max_iterations, residual


def solve_lasso_gram(
    gram: FloatArray,
    score: FloatArray,
    penalty: float,
    *,
    tolerance: float = 1.0e-8,
    max_iterations: int = 20_000,
    check_every: int = 10,
    power_iterations: int = 100,
    spectral_safety: float = 1.05,
    initial: FloatArray | None = None,
) -> LassoResult:
    """Solve ``.5*b'Gb-q'b+penalty*|b|_1`` and verify its KKT residual.

    The power estimate is conservatively inflated. If an unlucky estimate is
    still too small to attain the requested KKT tolerance, the solver restarts
    with a doubled step denominator. No unconverged fit is reported as valid.
    """
    matrix = np.ascontiguousarray(gram, dtype=np.float64)
    vector = np.ascontiguousarray(score, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("gram must be square")
    if vector.shape != (matrix.shape[0],):
        raise ValueError("score has incompatible shape")
    if penalty < 0.0:
        raise ValueError("penalty must be nonnegative")
    if not np.allclose(matrix, matrix.T, rtol=1.0e-11, atol=1.0e-11):
        raise ValueError("gram must be symmetric")

    base_lipschitz = _spectral_radius(matrix, power_iterations) * spectral_safety
    last: tuple[FloatArray, int, float] | None = None
    for attempt in range(5):
        lipschitz = base_lipschitz * (2.0**attempt)
        last = _run_fista(
            matrix,
            vector,
            penalty,
            lipschitz,
            tolerance,
            max_iterations,
            check_every,
            initial,
        )
        coef, iterations, residual = last
        if np.all(np.isfinite(coef)) and residual <= tolerance:
            return LassoResult(
                coef=coef,
                converged=True,
                iterations=iterations,
                kkt_residual=residual,
                objective=lasso_objective(matrix, vector, coef, penalty),
                lipschitz_constant=lipschitz,
                restart_attempt=attempt,
            )

    assert last is not None
    coef, iterations, residual = last
    return LassoResult(
        coef=coef,
        converged=False,
        iterations=iterations,
        kkt_residual=float(residual),
        objective=lasso_objective(matrix, vector, coef, penalty),
        lipschitz_constant=base_lipschitz * 16.0,
        restart_attempt=4,
    )

