# Threshold Renormalization: Reproduction Package

Official code and reproduction package for **“Threshold Renormalization Explains Weak-to-Strong Generalization in Diagonal Linear Networks.”**

**Paper:** [OpenReview](https://openreview.net/forum?id=MoQnYY6zCh)  
**Venue:** AXIOM 2026 — NeurIPS Workshop  
**Authors:** Mohammad Zeinalpour, Amir Najafi

This repository reproduces the synthetic experiments and numerical validation results reported in the paper. It implements the finite-sample two-stage LASSO model, finite-\(p\) scalar state evolution, paired uncertainty estimates, and the numerical appendix's tables and figures.

## What is reproduced

1. Empirical population risk versus finite-\(p\) state-evolution risk.
2. Matched tests on both sides of the weak-to-strong phase boundaries.
3. The Stage-II threshold-renormalization discrepancy and its Lipschitz bound.

All experiments use synthetic data. No external dataset, accelerator, or proprietary solver is required.

## Environment

The verified environment is Python 3.12 on Linux.

Create an isolated environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e .
```

The implementation uses public NumPy/SciPy APIs. Tests use Python's standard-library `unittest` runner.

## Quick verification

Run the tests and the smoke experiment:

```bash
python -m unittest discover -s tests -v
python -m threshold_w2sg.cli run --config configs/smoke.yaml
```

Expected outputs are written to `results/smoke/`.

**Note:** The smoke profile is intended only to verify installation and end-to-end execution. Its numerical output must not be used as manuscript results.

## Reproducing the publication results

The publication configuration uses six off-critical exponent pairs, four dimensions, and 300 independent paired Monte Carlo repetitions per cell.

Run the full experiment:

```bash
python -m threshold_w2sg.cli run --config configs/publication.yaml
```

The run generates the following artifacts:

| Output | Description |
|---|---|
| `raw/empirical_replicates.csv` | Results from individual two-stage fits |
| `raw/state_evolution.csv` | Deterministic finite-\(p\) predictions |
| `summary/cell_summary.csv` | Means, confidence intervals, and discrepancies |
| `summary/accuracy_by_dimension.csv` | Accuracy summaries by dimension |
| `summary/phase_at_largest_p.csv` | Results at the largest tested dimension |
| `figures/*.pdf` and `figures/*.png` | Generated figures |
| `latex/*.tex` | Generated tables and numerical appendix |
| `run_metadata.json` | Seeds, software versions, hardware, timings, and convergence information |

### Regenerating figures and tables

Regenerate the figures, tables, summaries, and bootstrap intervals from the saved raw results without rerunning the LASSO fits:

```bash
python -m threshold_w2sg.cli render --config configs/publication.yaml
```

### Recomputing state evolution

If the state-evolution implementation or quadrature configuration changes, recompute the deterministic state evolution and derived artifacts while retaining the registered empirical repetitions:

```bash
python -m threshold_w2sg.cli recompute-se --config configs/publication.yaml
```

## Reproducibility contract

The implementation follows these reproducibility guarantees:

- The master seed is recorded in the YAML configuration.
- Each Monte Carlo stream is a deterministic NumPy `SeedSequence`, indexed by case, dimension, repetition, and stream role.
- Bootstrap intervals are pointwise paired percentile intervals for the mean, using 10,000 resamples and separately registered deterministic streams.
- A LASSO fit is accepted only when its KKT residual satisfies the configured tolerance.
- State-evolution fixed points are accepted only when both noise and calibration residuals satisfy their configured tolerances.
- Failed convergence terminates the run; results are never silently retained.
- Raw results are written before aggregation so that reported numbers remain auditable.
- Worker count affects runtime but not random streams or result ordering.

## Numerical conventions

The finite-sample objectives are

\[
\frac{1}{2}\left\|y-\frac{X\beta}{\sqrt{p}}\right\|_2^2
+\lambda\|\beta\|_1.
\]

Gaussian-Wishart Gram matrices are sampled exactly using Bartlett's decomposition. Stage-I score noise is sampled conditionally on the Gram matrix, preserving the exact joint sufficient-statistics law.

Population risk is evaluated exactly as

\[
\frac{1}{p}\|\widehat{\beta}-\beta^\ast\|_2^2,
\]

without using a finite test set.

Stage-I Gaussian moments and active probabilities are evaluated in closed form. Stage-II soft-threshold moments and active probabilities are evaluated analytically, conditional on the Stage-I scalar input. High-order Gauss-Hermite quadrature is used only for the remaining smooth outer expectation, avoiding quadrature of discontinuous activity indicators.

## Scope and limitations

This package validates a stylized Gaussian, power-law model over the dimensions specified in the experiment configuration.

The experiments do **not** prove oversampled state evolution or establish that the same mechanism applies to nonlinear networks or real-world datasets.

## License

The implementation is released under the MIT License. See [`LICENSE`](LICENSE) for the complete license text.

## Citation

If you use this code in your research, please cite the paper:

```bibtex
@inproceedings{zeinalpour2026threshold,
  title     = {Threshold Renormalization Explains Weak-to-Strong
               Generalization in Diagonal Linear Networks},
  author    = {Zeinalpour, Mohammad and Najafi, Amir},
  booktitle = {AXIOM Workshop at NeurIPS 2026},
  year      = {2026},
  url       = {https://openreview.net/forum?id=MoQnYY6zCh}
}
```

Please verify the official workshop name and citation format against the accepted paper before publication.
