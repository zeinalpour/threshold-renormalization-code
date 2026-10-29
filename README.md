# Threshold Renormalization: Reproduction Package

Official code and reproduction package for:

**Threshold Renormalization Explains Weak-to-Strong Generalization in Diagonal Linear Networks**

**Authors:** Mohammad Zeinalpour and Amir Najafi  
**Workshop:** AXIOM: Foundations of Efficient Deep Learning, NeurIPS 2026  
**Paper:** [OpenReview](https://openreview.net/forum?id=MoQnYY6zCh)

This repository reproduces the synthetic experiments and numerical validation results reported in the paper. It implements the finite-sample two-stage LASSO model, finite-\(p\) scalar state evolution, paired uncertainty estimates, and all tables and figures used in the numerical-validation appendix.

## What Is Reproduced

The reproduction package covers the following experiments:

1. **Empirical versus state-evolution risk:** Comparison of empirical population risk with finite-\(p\) state-evolution predictions.
2. **Weak-to-strong phase boundaries:** Matched tests on both sides of the phase boundaries.
3. **Threshold renormalization:** Evaluation of the Stage-II threshold-renormalization discrepancy and its Lipschitz bound.

All experiments use synthetic data. No external dataset, accelerator, or proprietary solver is required.

## Environment

The verified environment is Python 3.12 on Linux.

### Installation

Create an isolated Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e .
```

The implementation uses public NumPy/SciPy APIs. Tests use Python's standard-library `unittest` runner.

## Quick Verification

To verify the installation and run a small end-to-end experiment:

```bash
python -m unittest discover -s tests -v
python -m threshold_w2sg.cli run --config configs/smoke.yaml
```

Expected outputs are written to:

```text
results/smoke/
```

**Important:** The smoke profile is intended only for installation checks and execution verification. Its numerical output must not be cited as a result of the paper.

## Reproducing the Publication Results

The publication configuration uses six off-critical exponent pairs, four dimensions, and 300 independent paired Monte Carlo repetitions per cell.

Run the full publication experiment with:

```bash
python -m threshold_w2sg.cli run --config configs/publication.yaml
```

The experiment generates the following artifacts:

| Output | Description |
|---|---|
| `raw/empirical_replicates.csv` | One row per independent two-stage fit |
| `raw/state_evolution.csv` | Deterministic finite-\(p\) state-evolution predictions |
| `summary/cell_summary.csv` | Means, confidence intervals, and discrepancies |
| `summary/accuracy_by_dimension.csv` | Accuracy summaries by dimension |
| `summary/phase_at_largest_p.csv` | Phase results at the largest tested dimension |
| `figures/*.pdf` | Figures in PDF format |
| `figures/*.png` | Figures in PNG format |
| `latex/*.tex` | Generated tables and the numerical appendix |
| `run_metadata.json` | Seeds, software versions, hardware, timings, and convergence information |

The publication configuration is the sole source of manuscript numerical results.

### Regenerating Figures and Tables

Figures, tables, summaries, and bootstrap intervals can be regenerated from saved raw results without rerunning the LASSO fits:

```bash
python -m threshold_w2sg.cli render --config configs/publication.yaml
```

### Recomputing State Evolution

If the state-evolution implementation or quadrature configuration changes, recompute the deterministic state evolution and derived artifacts while retaining the registered empirical repetitions:

```bash
python -m threshold_w2sg.cli recompute-se --config configs/publication.yaml
```

## Reproducibility Contract

The implementation follows these reproducibility guarantees:

- The master random seed is recorded in the YAML configuration.
- Every Monte Carlo stream is a deterministic NumPy `SeedSequence`, indexed by case, dimension, repetition, and stream role.
- Bootstrap intervals are pointwise paired percentile intervals for the mean, using 10,000 resamples and separately registered deterministic streams.
- Each LASSO fit is accepted only if its KKT residual satisfies the configured tolerance.
- State-evolution fixed points are accepted only if both noise and calibration residuals satisfy their configured tolerances.
- Failed convergence terminates the run; results are never silently retained.
- Raw results are written before aggregation, making every reported number auditable.
- Worker count affects runtime but not random streams or result ordering.

These guarantees are designed to support transparent numerical validation and reproducibility of the reported experiments.

## Numerical Conventions

### Finite-Sample Objective

The finite-sample optimization objective is

\[
\frac{1}{2}
\left\|y-\frac{X\beta}{\sqrt{p}}\right\|_2^2
+\lambda\|\beta\|_1.
\]

### Gaussian-Wishart Sampling

Gaussian-Wishart Gram matrices are sampled exactly using Bartlett's decomposition. Stage-I score noise is sampled conditionally on the Gram matrix, preserving the exact joint sufficient-statistics law.

### Population Risk

Population risk is evaluated exactly as

\[
\frac{1}{p}
\|\widehat{\beta}-\beta^\ast\|_2^2,
\]

without using a finite test set.

### State-Evolution Calculations

Stage-I Gaussian moments and active probabilities are evaluated in closed form.

Stage-II soft-threshold moments and active probabilities are evaluated analytically, conditional on the Stage-I scalar input. High-order Gauss-Hermite quadrature is used only for the remaining smooth outer expectation. This avoids quadrature of discontinuous activity indicators.

## Runtime Profiles

Two experiment profiles are provided:

- **Smoke profile:** `configs/smoke.yaml` — for installation checks and end-to-end execution verification only. Its numerical output must not be cited.
- **Publication profile:** `configs/publication.yaml` — the sole source of the manuscript's numerical results.

The worker count may be changed to accommodate available computing resources without changing the random streams or result ordering.

## Scope and Limitations

This implementation validates a stylized Gaussian, power-law model over the dimensions specified in the experiment configuration.

The experiments do **not** prove oversampled state evolution and do not establish that the same mechanism applies to nonlinear networks or real-world datasets.

The numerical results should therefore be interpreted within the assumptions and experimental scope of the model studied in the paper.

## License

The implementation is released under the **MIT License**.

See [`LICENSE`](LICENSE) for the complete license text.

## Citation

If you use this code in your research, please cite the paper:

```bibtex
@inproceedings{
zeinalpour2026threshold,
title={Threshold Renormalization Explains Weak-to-Strong Generalization in Diagonal Linear Networks},
author={Mohammad Zeinalpour and Amir Najafi},
booktitle={AXIOM: Foundations of Efficient Deep Learning},
year={2026},
url={https://openreview.net/forum?id=MoQnYY6zCh}
}
```

## Acknowledgments

This repository provides the numerical implementation and reproduction materials associated with the paper presented at the AXIOM workshop at NeurIPS 2026.
