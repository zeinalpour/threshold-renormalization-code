"""One exact finite-sample two-stage LASSO realization."""

from __future__ import annotations

import time
from typing import Any
import resource

import numpy as np
from threadpoolctl import threadpool_limits

from .config import CaseConfig, ExperimentConfig
from .lasso import solve_lasso_gram
from .wishart import (
    sample_gaussian_gram,
    sample_power_law_signal,
    sample_stage1_score,
)


def _rng(base_seed: int, case_index: int, p: int, replicate: int, stream: int):
    sequence = np.random.SeedSequence(
        [base_seed, case_index, p, replicate, stream]
    )
    return np.random.default_rng(sequence)


def run_replicate(
    config: ExperimentConfig,
    case: CaseConfig,
    case_index: int,
    p: int,
    replicate: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    n = round(p**case.alpha1)
    m = round(p**case.alpha2)
    lambda1 = config.model.nu1 * np.sqrt(2.0 * n / p)
    lambda2 = config.model.nu2 * np.sqrt(2.0 * m / p)
    signal_rng = _rng(config.base_seed, case_index, p, replicate, 0)
    stage1_rng = _rng(config.base_seed, case_index, p, replicate, 1)
    stage2_rng = _rng(config.base_seed, case_index, p, replicate, 2)

    with threadpool_limits(limits=1):
        beta_star = sample_power_law_signal(
            p, config.model.gamma, signal_rng
        )
        gram1, factor1 = sample_gaussian_gram(n, p, stage1_rng)
        score1 = sample_stage1_score(
            gram1,
            factor1,
            beta_star,
            config.model.delta,
            stage1_rng,
        )
        result1 = solve_lasso_gram(
            gram1,
            score1,
            lambda1,
            tolerance=config.solver.kkt_tolerance,
            max_iterations=config.solver.max_iterations,
            check_every=config.solver.check_every,
            power_iterations=config.solver.power_iterations,
            spectral_safety=config.solver.spectral_safety,
        )
        if not result1.converged:
            raise RuntimeError(
                f"Stage-I LASSO failed: case={case.id}, p={p}, rep={replicate}, "
                f"KKT={result1.kkt_residual:.3e}"
            )

        del gram1, factor1, score1
        gram2, _ = sample_gaussian_gram(m, p, stage2_rng)
        score2 = gram2 @ result1.coef
        result2 = solve_lasso_gram(
            gram2,
            score2,
            lambda2,
            tolerance=config.solver.kkt_tolerance,
            max_iterations=config.solver.max_iterations,
            check_every=config.solver.check_every,
            power_iterations=config.solver.power_iterations,
            spectral_safety=config.solver.spectral_safety,
        )
        if not result2.converged:
            raise RuntimeError(
                f"Stage-II LASSO failed: case={case.id}, p={p}, rep={replicate}, "
                f"KKT={result2.kkt_residual:.3e}"
            )

        risk1 = float(np.mean((result1.coef - beta_star) ** 2))
        risk2 = float(np.mean((result2.coef - beta_star) ** 2))
        teacher_error = float(np.mean((result2.coef - result1.coef) ** 2))

    return {
        "case_id": case.id,
        "case_label": case.label,
        "case_index": case_index,
        "alpha1": case.alpha1,
        "alpha2": case.alpha2,
        "p": p,
        "n": n,
        "m": m,
        "replicate": replicate,
        "seed_descriptor": (
            f"[{config.base_seed},{case_index},{p},{replicate},stream]"
        ),
        "lambda1": lambda1,
        "lambda2": lambda2,
        "risk1": risk1,
        "risk2": risk2,
        "risk_difference": risk2 - risk1,
        "relative_improvement": (risk1 - risk2) / risk1,
        "teacher_error2": teacher_error,
        "active1": float(np.mean(np.abs(result1.coef) > 1.0e-12)),
        "active2": float(np.mean(np.abs(result2.coef) > 1.0e-12)),
        "stage1_iterations": result1.iterations,
        "stage2_iterations": result2.iterations,
        "stage1_kkt": result1.kkt_residual,
        "stage2_kkt": result2.kkt_residual,
        "stage1_solver_restart": result1.restart_attempt,
        "stage2_solver_restart": result2.restart_attempt,
        "worker_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024.0,
        "elapsed_seconds": time.perf_counter() - started,
    }
