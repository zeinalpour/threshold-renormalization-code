"""Validated YAML configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CaseConfig:
    id: str
    label: str
    alpha1: float
    alpha2: float


@dataclass(frozen=True)
class ModelConfig:
    gamma: float
    delta: float
    nu1: float
    nu2: float


@dataclass(frozen=True)
class SolverConfig:
    kkt_tolerance: float
    max_iterations: int
    check_every: int
    power_iterations: int
    spectral_safety: float


@dataclass(frozen=True)
class StateEvolutionConfig:
    quadrature_nodes: int
    tolerance: float
    max_iterations: int
    damping: float


@dataclass(frozen=True)
class ExperimentConfig:
    profile_name: str
    output_dir: Path
    base_seed: int
    p_values: tuple[int, ...]
    replicates: int
    bootstrap_resamples: int
    workers: int
    model: ModelConfig
    cases: tuple[CaseConfig, ...]
    solver: SolverConfig
    state_evolution: StateEvolutionConfig
    source_path: Path


def _positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path).resolve()
    data: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8"))
    model = ModelConfig(**data["model"])
    solver = SolverConfig(**data["solver"])
    state = StateEvolutionConfig(**data["state_evolution"])
    cases = tuple(CaseConfig(**case) for case in data["cases"])
    config = ExperimentConfig(
        profile_name=str(data["profile_name"]),
        output_dir=Path(data["output_dir"]),
        base_seed=int(data["base_seed"]),
        p_values=tuple(int(p) for p in data["p_values"]),
        replicates=int(data["replicates"]),
        bootstrap_resamples=int(data["bootstrap_resamples"]),
        workers=int(data["workers"]),
        model=model,
        cases=cases,
        solver=solver,
        state_evolution=state,
        source_path=source,
    )
    _validate(config)
    return config


def _validate(config: ExperimentConfig) -> None:
    if len(set(config.p_values)) != len(config.p_values):
        raise ValueError("p_values must be unique")
    if len({case.id for case in config.cases}) != len(config.cases):
        raise ValueError("case identifiers must be unique")
    if any(p < 8 for p in config.p_values):
        raise ValueError("all dimensions must be at least eight")
    if any(case.alpha1 <= 1 or case.alpha2 <= 1 for case in config.cases):
        raise ValueError("all sample-size exponents must exceed one")
    if config.model.gamma <= 0.5:
        raise ValueError("gamma must exceed one half")
    _positive("delta", config.model.delta)
    _positive("nu1", config.model.nu1)
    _positive("nu2", config.model.nu2)
    _positive("kkt_tolerance", config.solver.kkt_tolerance)
    _positive("state-evolution tolerance", config.state_evolution.tolerance)
    if not 0 < config.state_evolution.damping <= 1:
        raise ValueError("state-evolution damping must lie in (0,1]")
    if config.replicates < 2:
        raise ValueError("at least two replicates are required")
    if config.bootstrap_resamples < 100:
        raise ValueError("at least 100 bootstrap resamples are required")
    if config.workers < 1:
        raise ValueError("workers must be positive")

