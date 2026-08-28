"""Finite-p two-stage scalar state evolution and LASSO calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import ndtr

from .operators import soft_threshold
from .quadrature import standard_normal_rule


@dataclass(frozen=True)
class StageOneMoments:
    risk: float
    active_fraction: float


@dataclass(frozen=True)
class StageTwoMoments:
    teacher_error: float
    risk: float
    active_fraction: float
    renormalized_risk: float
    perturbation_mse: float
    perturbation_bound_ratio: float
    composition_error: float


@dataclass(frozen=True)
class StateEvolutionResult:
    p: int
    n: int
    m: int
    lambda1: float
    lambda2: float
    sigma1: float
    sigma2: float
    b1: float
    b2: float
    risk1: float
    risk2: float
    teacher_error2: float
    active1: float
    active2: float
    renormalized_risk: float
    perturbation_mse: float
    perturbation_bound_ratio: float
    composition_error: float
    stage1_iterations: int
    stage2_iterations: int
    stage1_noise_residual: float
    stage1_calibration_residual: float
    stage2_noise_residual: float
    stage2_calibration_residual: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def gaussian_soft_threshold_moments(
    mean: np.ndarray | float,
    sigma: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``E[eta(Y)]``, ``E[eta(Y)^2]``, and ``P(|Y|>b)``.

    Here ``Y`` is Gaussian with the supplied mean and scalar standard
    deviation.  Evaluating the active probability analytically is essential:
    applying Gauss--Hermite quadrature directly to its discontinuous indicator
    can miss the interval around zero and incorrectly return probability one.
    """
    location = np.asarray(mean, dtype=np.float64)
    if threshold < 0.0:
        raise ValueError("threshold must be nonnegative")
    if sigma < 0.0:
        raise ValueError("sigma must be nonnegative")
    if sigma == 0.0:
        output = soft_threshold(location, threshold)
        active = (np.abs(location) > threshold).astype(np.float64)
        return output, output**2, active

    upper_z = (threshold - location) / sigma
    lower_z = (-threshold - location) / sigma
    upper_probability = ndtr(-upper_z)
    lower_probability = ndtr(lower_z)
    normalizer = np.sqrt(2.0 * np.pi)
    upper_density = np.exp(-0.5 * upper_z**2) / normalizer
    lower_density = np.exp(-0.5 * lower_z**2) / normalizer

    upper_shift = location - threshold
    lower_shift = location + threshold
    first_moment = (
        upper_shift * upper_probability
        + sigma * upper_density
        + lower_shift * lower_probability
        - sigma * lower_density
    )
    second_moment = (
        upper_shift**2 * upper_probability
        + 2.0 * upper_shift * sigma * upper_density
        + sigma**2 * (upper_probability + upper_z * upper_density)
        + lower_shift**2 * lower_probability
        - 2.0 * lower_shift * sigma * lower_density
        + sigma**2 * (lower_probability - lower_z * lower_density)
    )
    active_probability = upper_probability + lower_probability
    return first_moment, second_moment, active_probability


def _signal_variances(p: int, gamma: float) -> np.ndarray:
    indices = np.arange(1, p + 1, dtype=np.float64)
    return p * indices ** (-2.0 * gamma)


def stage_one_moments(
    p: int,
    gamma: float,
    sigma: float,
    threshold: float,
    quadrature_nodes: int,
) -> StageOneMoments:
    # These moments have closed forms.  In particular, the active probability
    # must not be approximated by quadrature of a discontinuous indicator.
    _ = quadrature_nodes
    signal_var = _signal_variances(p, gamma)
    total_var = signal_var + sigma**2
    total_scale = np.sqrt(total_var)
    standardized_threshold = threshold / total_scale
    upper_probability = ndtr(-standardized_threshold)
    normal_density = (
        np.exp(-0.5 * standardized_threshold**2) / np.sqrt(2.0 * np.pi)
    )
    output_second_moment = 2.0 * (
        (total_var + threshold**2) * upper_probability
        - threshold * total_scale * normal_density
    )
    # E[eta(U;b) beta*] = 2 Var(beta*) P(U>b) for the joint Gaussian pair.
    risk_by_coordinate = (
        output_second_moment
        - 4.0 * signal_var * upper_probability
        + signal_var
    )
    risk_by_coordinate = np.maximum(risk_by_coordinate, 0.0)
    active_by_coordinate = 2.0 * upper_probability
    return StageOneMoments(
        risk=float(np.mean(risk_by_coordinate)),
        active_fraction=float(np.mean(active_by_coordinate)),
    )


def stage_two_moments(
    p: int,
    gamma: float,
    sigma1: float,
    b1: float,
    sigma2: float,
    b2: float,
    quadrature_nodes: int,
    chunk_size: int = 128,
) -> StageTwoMoments:
    nodes, weights = standard_normal_rule(quadrature_nodes)
    signal_var = _signal_variances(p, gamma)
    total_var = signal_var + sigma1**2
    conditional_slope = signal_var / total_var
    conditional_var = signal_var * sigma1**2 / total_var

    totals = {
        "teacher": 0.0,
        "risk": 0.0,
        "active": 0.0,
        "renorm_risk": 0.0,
        "perturbation": 0.0,
    }
    max_composition_error = 0.0

    for start in range(0, p, chunk_size):
        stop = min(start + chunk_size, p)
        u = np.sqrt(total_var[start:stop])[:, None] * nodes[None, :]
        teacher = soft_threshold(u, b1)
        comparison = soft_threshold(teacher, b2)
        direct_comparison = soft_threshold(u, b1 + b2)
        max_composition_error = max(
            max_composition_error,
            float(np.max(np.abs(comparison - direct_comparison))),
        )
        student_mean, student_second, active_probability = (
            gaussian_soft_threshold_moments(teacher, sigma2, b2)
        )
        target_mean = conditional_slope[start:stop, None] * u

        teacher_error = (
            student_second - 2.0 * teacher * student_mean + teacher**2
        )
        risk_error = (
            student_second
            - 2.0 * target_mean * student_mean
            + target_mean**2
            + conditional_var[start:stop, None]
        )
        perturbation = (
            student_second
            - 2.0 * comparison * student_mean
            + comparison**2
        )
        renorm_error = (
            (comparison - target_mean) ** 2
            + conditional_var[start:stop, None]
        )

        totals["teacher"] += float(
            np.sum(np.maximum(teacher_error, 0.0) * weights[None, :])
        )
        totals["risk"] += float(
            np.sum(np.maximum(risk_error, 0.0) * weights[None, :])
        )
        totals["active"] += float(
            np.sum(active_probability * weights[None, :])
        )
        totals["perturbation"] += float(
            np.sum(np.maximum(perturbation, 0.0) * weights[None, :])
        )
        totals["renorm_risk"] += float(
            np.sum(np.maximum(renorm_error, 0.0) * weights[None, :])
        )

    teacher_error = totals["teacher"] / p
    risk = totals["risk"] / p
    active_fraction = totals["active"] / p
    perturbation_mse = totals["perturbation"] / p
    renormalized_risk = totals["renorm_risk"] / p
    ratio = perturbation_mse / sigma2**2 if sigma2 > 0 else 0.0
    return StageTwoMoments(
        teacher_error=teacher_error,
        risk=risk,
        active_fraction=active_fraction,
        renormalized_risk=renormalized_risk,
        perturbation_mse=perturbation_mse,
        perturbation_bound_ratio=ratio,
        composition_error=max_composition_error,
    )


def _relative_change(old: float, new: float) -> float:
    return abs(new - old) / max(1.0, abs(old), abs(new))


def solve_state_evolution(
    *,
    p: int,
    n: int,
    m: int,
    gamma: float,
    delta: float,
    lambda1: float,
    lambda2: float,
    quadrature_nodes: int,
    tolerance: float,
    max_iterations: int,
    damping: float,
) -> StateEvolutionResult:
    tau1_sq = p / n
    tau2_sq = p / m
    rho1 = n / p
    rho2 = m / p
    sigma1 = np.sqrt(delta * tau1_sq)
    b1 = lambda1 / rho1

    stage1 = stage_one_moments(p, gamma, sigma1, b1, quadrature_nodes)
    for iteration1 in range(1, max_iterations + 1):
        target_sigma = np.sqrt(tau1_sq * (delta + stage1.risk))
        target_b = lambda1 / (rho1 - stage1.active_fraction)
        next_sigma = (1.0 - damping) * sigma1 + damping * target_sigma
        next_b = (1.0 - damping) * b1 + damping * target_b
        change = max(_relative_change(sigma1, next_sigma), _relative_change(b1, next_b))
        sigma1, b1 = next_sigma, next_b
        stage1 = stage_one_moments(p, gamma, sigma1, b1, quadrature_nodes)
        if change <= tolerance:
            break
    else:
        raise RuntimeError("Stage-I state evolution did not converge")

    sigma2 = max(b1 * np.sqrt(tau2_sq), np.finfo(float).eps)
    b2 = lambda2 / rho2
    stage2 = stage_two_moments(
        p, gamma, sigma1, b1, sigma2, b2, quadrature_nodes
    )
    for iteration2 in range(1, max_iterations + 1):
        target_sigma = np.sqrt(tau2_sq * stage2.teacher_error)
        target_b = lambda2 / (rho2 - stage2.active_fraction)
        next_sigma = (1.0 - damping) * sigma2 + damping * target_sigma
        next_b = (1.0 - damping) * b2 + damping * target_b
        change = max(_relative_change(sigma2, next_sigma), _relative_change(b2, next_b))
        sigma2, b2 = next_sigma, next_b
        stage2 = stage_two_moments(
            p, gamma, sigma1, b1, sigma2, b2, quadrature_nodes
        )
        if change <= tolerance:
            break
    else:
        raise RuntimeError("Stage-II state evolution did not converge")

    for name, active_fraction, threshold in (
        ("Stage I", stage1.active_fraction, b1),
        ("Stage II", stage2.active_fraction, b2),
    ):
        if not 0.0 <= active_fraction <= 1.0:
            raise RuntimeError(f"{name} active fraction is not a probability")
        if threshold > 0.0 and not active_fraction < 1.0:
            raise RuntimeError(
                f"{name} active fraction must be below one for a positive threshold"
            )

    stage1_noise_residual = abs(sigma1**2 - tau1_sq * (delta + stage1.risk))
    stage1_calibration_residual = abs(
        lambda1 - b1 * (rho1 - stage1.active_fraction)
    )
    stage2_noise_residual = abs(sigma2**2 - tau2_sq * stage2.teacher_error)
    stage2_calibration_residual = abs(
        lambda2 - b2 * (rho2 - stage2.active_fraction)
    )
    residual_limit = max(100.0 * tolerance, 1.0e-9)
    if max(
        stage1_noise_residual,
        stage1_calibration_residual,
        stage2_noise_residual,
        stage2_calibration_residual,
    ) > residual_limit:
        raise RuntimeError("state-evolution fixed point failed residual validation")
    if stage2.perturbation_mse > sigma2**2 * (1.0 + 1.0e-9):
        raise RuntimeError("threshold-renormalization Lipschitz bound violated")

    return StateEvolutionResult(
        p=p,
        n=n,
        m=m,
        lambda1=lambda1,
        lambda2=lambda2,
        sigma1=sigma1,
        sigma2=sigma2,
        b1=b1,
        b2=b2,
        risk1=stage1.risk,
        risk2=stage2.risk,
        teacher_error2=stage2.teacher_error,
        active1=stage1.active_fraction,
        active2=stage2.active_fraction,
        renormalized_risk=stage2.renormalized_risk,
        perturbation_mse=stage2.perturbation_mse,
        perturbation_bound_ratio=stage2.perturbation_bound_ratio,
        composition_error=stage2.composition_error,
        stage1_iterations=iteration1,
        stage2_iterations=iteration2,
        stage1_noise_residual=stage1_noise_residual,
        stage1_calibration_residual=stage1_calibration_residual,
        stage2_noise_residual=stage2_noise_residual,
        stage2_calibration_residual=stage2_calibration_residual,
    )
