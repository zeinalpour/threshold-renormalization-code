"""Paired Monte Carlo summaries and deterministic bootstrap intervals."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def bootstrap_mean_interval(
    values: Iterable[float],
    *,
    resamples: int,
    seed_words: list[int],
    confidence: float = 0.95,
    chunk_size: int = 1000,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of independent replicates."""
    sample = np.asarray(list(values), dtype=np.float64)
    if sample.ndim != 1 or sample.size < 2:
        raise ValueError("bootstrap requires at least two one-dimensional values")
    rng = np.random.default_rng(np.random.SeedSequence(seed_words))
    means = np.empty(resamples, dtype=np.float64)
    written = 0
    while written < resamples:
        count = min(chunk_size, resamples - written)
        indices = rng.integers(0, sample.size, size=(count, sample.size))
        means[written : written + count] = sample[indices].mean(axis=1)
        written += count
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


def summarize_cells(
    raw: pd.DataFrame,
    state_evolution: pd.DataFrame,
    *,
    base_seed: int,
    bootstrap_resamples: int,
) -> pd.DataFrame:
    se_indexed = state_evolution.set_index(["case_id", "p"])
    rows: list[dict[str, float | int | str | bool]] = []
    metric_codes = {
        "risk1": 1,
        "risk2": 2,
        "risk_difference": 3,
        "relative_improvement": 4,
    }
    for (case_id, p), group in raw.groupby(["case_id", "p"], sort=True):
        first = group.iloc[0]
        case_index = int(first["case_index"])
        row: dict[str, float | int | str | bool] = {
            "case_id": case_id,
            "case_label": str(first["case_label"]),
            "case_index": case_index,
            "alpha1": float(first["alpha1"]),
            "alpha2": float(first["alpha2"]),
            "p": int(p),
            "n": int(first["n"]),
            "m": int(first["m"]),
            "replicates": int(len(group)),
        }
        for metric, metric_code in metric_codes.items():
            values = group[metric].to_numpy(dtype=np.float64)
            low, high = bootstrap_mean_interval(
                values,
                resamples=bootstrap_resamples,
                seed_words=[base_seed, 90_000, case_index, int(p), metric_code],
            )
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1))
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high

        se = se_indexed.loc[(case_id, p)]
        row["risk1_se"] = float(se["risk1"])
        row["risk2_se"] = float(se["risk2"])
        row["risk_difference_se"] = float(se["risk2"] - se["risk1"])
        row["risk1_relative_error"] = abs(
            float(row["risk1_mean"]) - float(row["risk1_se"])
        ) / float(row["risk1_se"])
        row["risk2_relative_error"] = abs(
            float(row["risk2_mean"]) - float(row["risk2_se"])
        ) / float(row["risk2_se"])
        row["empirical_improves"] = float(row["risk_difference_mean"]) < 0.0
        row["se_improves"] = float(row["risk_difference_se"]) < 0.0
        row["ci_excludes_zero"] = (
            float(row["risk_difference_ci_high"]) < 0.0
            or float(row["risk_difference_ci_low"]) > 0.0
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["case_index", "p"]).reset_index(drop=True)


def accuracy_by_dimension(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p, group in summary.groupby("p", sort=True):
        rows.append(
            {
                "p": int(p),
                "stage1_median_relative_error": float(
                    group["risk1_relative_error"].median()
                ),
                "stage1_max_relative_error": float(
                    group["risk1_relative_error"].max()
                ),
                "stage2_median_relative_error": float(
                    group["risk2_relative_error"].median()
                ),
                "stage2_max_relative_error": float(
                    group["risk2_relative_error"].max()
                ),
            }
        )
    return pd.DataFrame(rows)

