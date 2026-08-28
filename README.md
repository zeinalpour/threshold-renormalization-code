# Threshold Renormalization: Reproduction Package

This anonymous package reproduces the synthetic experiments for
"Threshold Renormalization Explains Weak-to-Strong Generalization in Diagonal
Linear Networks." It implements the finite-sample two-stage LASSO model, its
finite-\(p\) scalar state evolution, paired uncertainty estimates, and every
table and figure used in the numerical-validation appendix.

## What is reproduced

1. Empirical population risk versus finite-\(p\) state-evolution risk.
2. Matched tests on both sides of the weak-to-strong phase boundaries.
3. The Stage-II threshold-renormalization discrepancy and its Lipschitz bound.

All data are synthetic. No download, accelerator, or proprietary solver is
required.

## Environment

The verified environment is Python 3.12 on Linux. Create an isolated
environment and install the exact dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e .
```

The code deliberately uses only public NumPy/SciPy APIs. Tests use Python's
standard-library `unittest` runner.

## Fast verification

The smoke profile runs a tiny end-to-end experiment and regenerates all output
types:

```bash
python -m unittest discover -s tests -v
python -m threshold_w2sg.cli run --config configs/smoke.yaml
```

Expected outputs appear under `results/smoke/`.

## Publication run

The paper configuration uses six off-critical exponent pairs, four dimensions,
and 300 independent paired Monte Carlo repetitions per cell:

```bash
python -m threshold_w2sg.cli run --config configs/publication.yaml
```

This command creates:

- `raw/empirical_replicates.csv`: one row per independent two-stage fit;
- `raw/state_evolution.csv`: deterministic finite-\(p\) predictions;
- `summary/cell_summary.csv`: means, confidence intervals, and discrepancies;
- `summary/accuracy_by_dimension.csv`;
- `summary/phase_at_largest_p.csv`;
- `figures/*.pdf` and `figures/*.png`;
- `latex/*.tex`: generated tables and the complete numerical appendix;
- `run_metadata.json`: seeds, versions, hardware, timing, and convergence data.

Figures, tables, summaries, and bootstrap intervals can be regenerated from
the saved raw results without rerunning the LASSO fits:

```bash
python -m threshold_w2sg.cli render --config configs/publication.yaml
```

If the state-evolution implementation or quadrature configuration changes,
recompute only the deterministic state evolution and all derived artifacts
while retaining the registered empirical repetitions:

```bash
python -m threshold_w2sg.cli recompute-se --config configs/publication.yaml
```

## Reproducibility contract

- The master seed is recorded in the YAML configuration.
- Every Monte Carlo stream is a deterministic NumPy `SeedSequence` indexed by
  case, dimension, repetition, and stream role.
- Bootstrap intervals are pointwise paired percentile intervals for the mean,
  using 10,000 resamples and separately registered deterministic streams.
- Each LASSO fit is accepted only if its KKT residual satisfies the configured
  tolerance.
- State-evolution fixed points are accepted only if both noise and calibration
  residuals satisfy the configured tolerance.
- Failed convergence terminates the run; results are never silently retained.
- Raw results are written before aggregation, so every reported number is
  auditable.

## Runtime profiles

`configs/smoke.yaml` is for installation checks only. Its numerical output must
not be cited. `configs/publication.yaml` is the sole source of manuscript
numbers. Worker count affects runtime but not random streams or result order.

## Numerical conventions

The finite-sample objectives are

\[
\frac12\left\|y-X\beta/\sqrt p\right\|_2^2+\lambda\|\beta\|_1.
\]

Gaussian-Wishart Gram matrices are sampled exactly with Bartlett's
decomposition. Stage-I score noise is sampled conditionally on the Gram matrix,
which preserves the exact joint sufficient-statistics law. Population risk is
evaluated exactly as \(p^{-1}\|\widehat\beta-\beta^\ast\|_2^2\); no finite test
set is used.

Stage-I Gaussian moments and active probabilities are evaluated in closed
form. Stage-II soft-threshold moments and active probabilities are evaluated
analytically conditional on the Stage-I scalar input; high-order Gauss-Hermite
quadrature is used only for the remaining smooth outer expectation. This avoids
quadrature of discontinuous activity indicators.

## License and limitations

The implementation is released under the MIT License. It validates a stylized
Gaussian, power-law model up to the dimensions in the configuration. It does
not prove oversampled state evolution and does not establish the same mechanism
for nonlinear networks or real data.
