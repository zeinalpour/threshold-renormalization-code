"""End-to-end experiment orchestration."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import generate_all_artifacts
from .config import CaseConfig, ExperimentConfig
from .simulation import run_replicate
from .state_evolution import solve_state_evolution
from .statistics import accuracy_by_dimension, summarize_cells
from .system_info import collect_system_info


STATE_EVOLUTION_METHOD = (
    "analytic Stage-I Gaussian moments and active probabilities; analytic "
    "conditional Stage-II truncated-normal moments; Gauss-Hermite quadrature "
    "only over the Stage-I scalar input"
)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def _atomic_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _worker(payload):
    config, case, case_index, p, replicate = payload
    return run_replicate(config, case, case_index, p, replicate)


def _state_evolution_rows(config: ExperimentConfig) -> pd.DataFrame:
    rows = []
    total = len(config.cases) * len(config.p_values)
    completed = 0
    for case_index, case in enumerate(config.cases):
        for p in config.p_values:
            n = round(p**case.alpha1)
            m = round(p**case.alpha2)
            lambda1 = config.model.nu1 * np.sqrt(2.0 * n / p)
            lambda2 = config.model.nu2 * np.sqrt(2.0 * m / p)
            result = solve_state_evolution(
                p=p,
                n=n,
                m=m,
                gamma=config.model.gamma,
                delta=config.model.delta,
                lambda1=lambda1,
                lambda2=lambda2,
                quadrature_nodes=config.state_evolution.quadrature_nodes,
                tolerance=config.state_evolution.tolerance,
                max_iterations=config.state_evolution.max_iterations,
                damping=config.state_evolution.damping,
            )
            row = {
                "case_id": case.id,
                "case_label": case.label,
                "case_index": case_index,
                "alpha1": case.alpha1,
                "alpha2": case.alpha2,
                **result.to_dict(),
            }
            rows.append(row)
            completed += 1
            print(
                f"state evolution {completed}/{total}: {case.id}, p={p}",
                flush=True,
            )
    return pd.DataFrame(rows).sort_values(["case_index", "p"]).reset_index(drop=True)


def _payloads(config: ExperimentConfig):
    for case_index, case in enumerate(config.cases):
        for p in config.p_values:
            for replicate in range(config.replicates):
                yield config, case, case_index, p, replicate


def _config_hash(config: ExperimentConfig) -> str:
    payload = {
        "profile_name": config.profile_name,
        "base_seed": config.base_seed,
        "p_values": config.p_values,
        "replicates": config.replicates,
        "bootstrap_resamples": config.bootstrap_resamples,
        "model": asdict(config.model),
        "cases": [asdict(case) for case in config.cases],
        "solver": asdict(config.solver),
        "state_evolution": asdict(config.state_evolution),
        "state_evolution_method": STATE_EVOLUTION_METHOD,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    output = config.output_dir.resolve()
    raw_dir = output / "raw"
    summary_dir = output / "summary"
    started = time.perf_counter()
    system = collect_system_info()

    print(f"profile: {config.profile_name}", flush=True)
    print(f"output: {output}", flush=True)
    print("solving deterministic state evolution", flush=True)
    state = _state_evolution_rows(config)
    _atomic_csv(state, raw_dir / "state_evolution.csv")

    total = len(config.cases) * len(config.p_values) * config.replicates
    print(f"running {total} independent two-stage fits", flush=True)
    records: list[dict[str, Any]] = []
    report_every = max(1, min(100, total // 20))
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        for completed, record in enumerate(
            executor.map(_worker, _payloads(config), chunksize=2), start=1
        ):
            records.append(record)
            if completed % report_every == 0 or completed == total:
                elapsed = time.perf_counter() - started
                print(
                    f"empirical fits {completed}/{total} "
                    f"({elapsed:.1f}s elapsed)",
                    flush=True,
                )

    raw = pd.DataFrame(records).sort_values(
        ["case_index", "p", "replicate"]
    ).reset_index(drop=True)
    if len(raw) != total:
        raise RuntimeError("incomplete empirical result set")
    if raw.duplicated(["case_id", "p", "replicate"]).any():
        raise RuntimeError("duplicate empirical result keys")
    _atomic_csv(raw, raw_dir / "empirical_replicates.csv")

    summary = summarize_cells(
        raw,
        state,
        base_seed=config.base_seed,
        bootstrap_resamples=config.bootstrap_resamples,
    )
    accuracy = accuracy_by_dimension(summary)
    largest = summary[summary["p"] == summary["p"].max()].copy()
    _atomic_csv(summary, summary_dir / "cell_summary.csv")
    _atomic_csv(accuracy, summary_dir / "accuracy_by_dimension.csv")
    _atomic_csv(largest, summary_dir / "phase_at_largest_p.csv")

    wall_seconds = time.perf_counter() - started
    residual_columns = [
        "stage1_noise_residual",
        "stage1_calibration_residual",
        "stage2_noise_residual",
        "stage2_calibration_residual",
    ]
    metadata = {
        "profile_name": config.profile_name,
        "config_path": f"configs/{config.source_path.name}",
        "config_sha256": _config_hash(config),
        "base_seed": config.base_seed,
        "seed_rule": "SeedSequence([base_seed, case_index, p, replicate, stream])",
        "bootstrap_seed_rule": (
            "SeedSequence([base_seed, 90000, case_index, p, metric_code])"
        ),
        "two_stage_fits": total,
        "individual_lasso_fits": 2 * total,
        "state_evolution_cells": len(state),
        "replicates_per_cell": config.replicates,
        "bootstrap_resamples": config.bootstrap_resamples,
        "workers": config.workers,
        "p_values": list(config.p_values),
        "model": asdict(config.model),
        "cases": [asdict(case) for case in config.cases],
        "solver": asdict(config.solver),
        "state_evolution": asdict(config.state_evolution),
        "state_evolution_method": STATE_EVOLUTION_METHOD,
        "wall_seconds": wall_seconds,
        "aggregate_fit_seconds": float(raw["elapsed_seconds"].sum()),
        "max_worker_peak_rss_mib": float(raw["worker_peak_rss_mib"].max()),
        "max_kkt_residual": float(
            max(raw["stage1_kkt"].max(), raw["stage2_kkt"].max())
        ),
        "max_state_evolution_residual": float(
            state[residual_columns].to_numpy().max()
        ),
        "max_composition_error": float(state["composition_error"].max()),
        "max_perturbation_bound_ratio": float(
            state["perturbation_bound_ratio"].max()
        ),
        "system": system,
    }
    _atomic_json(metadata, output / "run_metadata.json")
    generate_all_artifacts(
        summary=summary,
        accuracy=accuracy,
        state=state,
        metadata=metadata,
        output_root=output,
        gamma=config.model.gamma,
    )
    print(f"completed in {wall_seconds:.1f}s", flush=True)
    return metadata


def recompute_state_evolution(config: ExperimentConfig) -> dict[str, Any]:
    """Replace state evolution and derived artifacts while retaining raw fits."""
    output = config.output_dir.resolve()
    raw_path = output / "raw" / "empirical_replicates.csv"
    metadata_path = output / "run_metadata.json"
    if not raw_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "saved empirical repetitions and metadata are required"
        )
    raw = pd.read_csv(raw_path)
    expected_cells = {
        (case.id, p): config.replicates
        for case in config.cases
        for p in config.p_values
    }
    actual_cells = raw.groupby(["case_id", "p"]).size().to_dict()
    if actual_cells != expected_cells:
        raise RuntimeError("saved empirical repetitions do not match the config")

    started = time.perf_counter()
    state = _state_evolution_rows(config)
    _atomic_csv(state, output / "raw" / "state_evolution.csv")
    summary = summarize_cells(
        raw,
        state,
        base_seed=config.base_seed,
        bootstrap_resamples=config.bootstrap_resamples,
    )
    accuracy = accuracy_by_dimension(summary)
    summary_dir = output / "summary"
    _atomic_csv(summary, summary_dir / "cell_summary.csv")
    _atomic_csv(accuracy, summary_dir / "accuracy_by_dimension.csv")
    _atomic_csv(
        summary[summary["p"] == summary["p"].max()],
        summary_dir / "phase_at_largest_p.csv",
    )

    residual_columns = [
        "stage1_noise_residual",
        "stage1_calibration_residual",
        "stage2_noise_residual",
        "stage2_calibration_residual",
    ]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "config_sha256": _config_hash(config),
            "state_evolution": asdict(config.state_evolution),
            "state_evolution_method": STATE_EVOLUTION_METHOD,
            "state_evolution_cells": len(state),
            "state_evolution_recompute_seconds": time.perf_counter() - started,
            "max_state_evolution_residual": float(
                state[residual_columns].to_numpy().max()
            ),
            "max_composition_error": float(state["composition_error"].max()),
            "max_perturbation_bound_ratio": float(
                state["perturbation_bound_ratio"].max()
            ),
        }
    )
    _atomic_json(metadata, metadata_path)
    generate_all_artifacts(
        summary=summary,
        accuracy=accuracy,
        state=state,
        metadata=metadata,
        output_root=output,
        gamma=config.model.gamma,
    )
    return metadata


def regenerate_artifacts(config: ExperimentConfig) -> dict[str, Any]:
    """Regenerate summaries, intervals, figures, and LaTeX from saved raw data."""
    output = config.output_dir.resolve()
    raw = pd.read_csv(output / "raw" / "empirical_replicates.csv")
    state = pd.read_csv(output / "raw" / "state_evolution.csv")
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata["config_sha256"] != _config_hash(config):
        raise RuntimeError("configuration does not match the saved raw results")
    summary = summarize_cells(
        raw,
        state,
        base_seed=config.base_seed,
        bootstrap_resamples=config.bootstrap_resamples,
    )
    accuracy = accuracy_by_dimension(summary)
    summary_dir = output / "summary"
    _atomic_csv(summary, summary_dir / "cell_summary.csv")
    _atomic_csv(accuracy, summary_dir / "accuracy_by_dimension.csv")
    _atomic_csv(
        summary[summary["p"] == summary["p"].max()],
        summary_dir / "phase_at_largest_p.csv",
    )
    generate_all_artifacts(
        summary=summary,
        accuracy=accuracy,
        state=state,
        metadata=metadata,
        output_root=output,
        gamma=config.model.gamma,
    )
    return metadata
