"""Generate publication tables, figures, and manuscript-ready LaTeX."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "comparable_improve": "#0072B2",
    "comparable_deteriorate": "#56B4E9",
    "stronger_improve": "#009E73",
    "stronger_deteriorate": "#E69F00",
    "weaker_improve": "#CC79A7",
    "weaker_deteriorate": "#D55E00",
}


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fixed_metadata = {
        "Creator": "threshold-w2sg reproduction package",
        "CreationDate": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    fig.savefig(
        base.with_suffix(".pdf"), bbox_inches="tight", metadata=fixed_metadata
    )
    fig.savefig(
        base.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "threshold-w2sg reproduction package"},
    )
    plt.close(fig)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "figure.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_risk_accuracy(summary: pd.DataFrame, output: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.65), sharex=True)
    for case_id, group in summary.groupby("case_id", sort=False):
        ordered = group.sort_values("p")
        color = COLORS.get(case_id, None)
        label = str(ordered.iloc[0]["case_label"])
        for axis, stage in zip(axes, [1, 2], strict=True):
            mean = ordered[f"risk{stage}_mean"].to_numpy()
            se = ordered[f"risk{stage}_se"].to_numpy()
            low = ordered[f"risk{stage}_ci_low"].to_numpy()
            high = ordered[f"risk{stage}_ci_high"].to_numpy()
            relative = 100.0 * (mean / se - 1.0)
            lower = 100.0 * (mean - low) / se
            upper = 100.0 * (high - mean) / se
            axis.errorbar(
                ordered["p"],
                relative,
                yerr=np.vstack([lower, upper]),
                marker="o",
                markersize=3.5,
                linewidth=1.0,
                capsize=2,
                color=color,
                label=label,
            )
    for stage, axis in enumerate(axes, start=1):
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted(summary["p"].unique()))
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.set_title(f"Stage {stage}")
        axis.set_xlabel("dimension $p$")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("empirical mean / SE risk - 1 (%)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
    )
    fig.tight_layout(rect=(0.0, 0.16, 1.0, 1.0))
    _save_figure(fig, output / "risk_accuracy")


def _theory_improves(alpha1: np.ndarray, alpha2: np.ndarray, gamma: float):
    nonweaker = (alpha2 >= alpha1) & (alpha1 < 2.0 * gamma)
    weaker_boundary = 1.0 + alpha2 * (1.0 - 1.0 / (2.0 * gamma))
    weaker = (alpha2 < alpha1) & (alpha1 < weaker_boundary)
    return nonweaker | weaker


def plot_phase_diagram(summary: pd.DataFrame, gamma: float, output: Path) -> None:
    _style()
    largest = summary[summary["p"] == summary["p"].max()].copy()
    grid = np.linspace(1.02, 1.82, 350)
    a1, a2 = np.meshgrid(grid, grid)
    improves = _theory_improves(a1, a2, gamma)
    fig, axis = plt.subplots(figsize=(4.2, 4.0))
    axis.contourf(
        a1,
        a2,
        improves.astype(int),
        levels=[-0.5, 0.5, 1.5],
        colors=["#F2D7D5", "#D5F5E3"],
        alpha=0.8,
    )
    for _, row in largest.iterrows():
        improves_emp = bool(row["empirical_improves"])
        marker = "o" if improves_emp else "X"
        axis.scatter(
            row["alpha1"],
            row["alpha2"],
            marker=marker,
            s=58,
            color=COLORS.get(str(row["case_id"]), "black"),
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
            label=str(row["case_label"]),
        )
    axis.plot(grid, grid, color="0.35", linewidth=0.8, linestyle=":")
    critical_alpha1 = 2.0 * gamma
    if 1.0 <= critical_alpha1 <= 1.85:
        axis.plot(
            [critical_alpha1, critical_alpha1],
            [critical_alpha1, 1.85],
            color="0.25",
            linewidth=0.9,
            linestyle="--",
        )
    weaker_line = 1.0 + grid * (1.0 - 1.0 / (2.0 * gamma))
    mask = grid < weaker_line
    axis.plot(weaker_line[mask], grid[mask], color="0.25", linewidth=0.9, linestyle="--")
    axis.set_xlim(1.0, 1.85)
    axis.set_ylim(1.0, 1.85)
    axis.set_xlabel(r"Stage-I exponent $\alpha_1$")
    axis.set_ylabel(r"Stage-II exponent $\alpha_2$")
    axis.set_title("Theory regions and matched finite-$p$ tests")
    axis.text(1.05, 1.78, "improvement region", color="#166534", fontsize=7)
    axis.text(1.57, 1.05, "deterioration region", color="#991B1B", fontsize=7)
    axis.grid(alpha=0.18)
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
        fontsize=6.4,
    )
    fig.tight_layout(rect=(0.0, 0.25, 1.0, 1.0))
    _save_figure(fig, output / "phase_diagram")


def plot_threshold_diagnostic(state: pd.DataFrame, output: Path) -> None:
    _style()
    selected = state
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.05))
    for case_id, group in selected.groupby("case_id", sort=False):
        ordered = group.sort_values("p")
        label = str(ordered.iloc[0]["case_label"])
        color = COLORS.get(case_id, None)
        axes[0].plot(
            ordered["p"], ordered["perturbation_mse"], "o-", color=color, label=label
        )
        axes[0].plot(
            ordered["p"], ordered["sigma2"] ** 2, "--", color=color, alpha=0.75
        )
        axes[1].plot(
            ordered["p"], ordered["perturbation_bound_ratio"], "o-", color=color, label=label
        )
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("dimension $p$")
    axes[0].set_ylabel(r"$p^{-1}\sum_i\mathbb{E}(B_i-\bar B_i)^2$")
    axes[0].set_title(r"Perturbation (solid) and $\sigma_{2,p}^2$ bound (dashed)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("dimension $p$")
    axes[1].set_ylabel(r"perturbation / $\sigma_{2,p}^2$")
    axes[1].set_title("Normalized Lipschitz diagnostic")
    for axis in axes:
        axis.set_xticks(sorted(selected["p"].unique()))
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.grid(alpha=0.25)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
    )
    fig.tight_layout(rect=(0.0, 0.17, 1.0, 1.0))
    _save_figure(fig, output / "threshold_diagnostic")


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def _sci_latex(value: float, digits: int = 1) -> str:
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def _format_memory_gib(value: object) -> str:
    """Format optional system-memory metadata without failing on restricted hosts."""
    if value is None:
        return "not reported"
    try:
        memory_gib = float(value)
    except (TypeError, ValueError):
        return "not reported"
    if not np.isfinite(memory_gib) or memory_gib <= 0.0:
        return "not reported"
    return f"{memory_gib:.1f} GiB"


def write_latex_tables(
    accuracy: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    case_count = int(summary["case_id"].nunique())
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Absolute relative discrepancy between Monte Carlo mean risk and finite-$p$ state-evolution risk across the {case_count} matched cases.}}",
        r"\label{tab:se-risk-accuracy}",
        r"\small",
        r"\begin{tabular}{c@{\qquad}cccc}",
        r"\toprule",
        r"$p$ & Stage I median & Stage I max & Stage II median & Stage II max \\",
        r"\midrule",
    ]
    for _, row in accuracy.iterrows():
        lines.append(
            f"{int(row['p'])} & "
            f"{100*row['stage1_median_relative_error']:.3f}\\% & "
            f"{100*row['stage1_max_relative_error']:.3f}\\% & "
            f"{100*row['stage2_median_relative_error']:.3f}\\% & "
            f"{100*row['stage2_max_relative_error']:.3f}\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _atomic_text(output / "se_accuracy_table.tex", "\n".join(lines) + "\n")

    largest = summary[summary["p"] == summary["p"].max()].sort_values("case_index")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Matched phase test at $p={int(summary['p'].max())}$. Intervals are pointwise paired 95\% percentile-bootstrap intervals for the mean risk difference.}}",
        r"\label{tab:matched-phase-results}",
        r"\scriptsize",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Regime & $(\alpha_1,\alpha_2)$ & Empirical $D_p$ [95\% CI] & SE $D_p$ \\",
        r"\midrule",
    ]
    for _, row in largest.iterrows():
        interval = (
            f"{_fmt(row['risk_difference_mean'])} "
            f"[{_fmt(row['risk_difference_ci_low'])}, {_fmt(row['risk_difference_ci_high'])}]"
        )
        lines.append(
            f"{row['case_label']} & ({row['alpha1']:.2f},{row['alpha2']:.2f}) & "
            f"{interval} & {_fmt(row['risk_difference_se'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _atomic_text(output / "phase_table.tex", "\n".join(lines) + "\n")


def write_compute_report(metadata: dict[str, Any], output: Path) -> None:
    system = metadata["system"]
    memory_text = _format_memory_gib(system.get("memory_gib_available"))
    recompute_line = ""
    if "state_evolution_recompute_seconds" in metadata:
        recompute_line = (
            "- Corrected state-evolution recomputation: "
            f"{metadata['state_evolution_recompute_seconds']:.2f} seconds\n"
        )
    text = f"""# Compute report

- Profile: `{metadata['profile_name']}`
- Two-stage fits: {metadata['two_stage_fits']}
- Workers: {metadata['workers']}
- Wall-clock time: {metadata['wall_seconds']:.2f} seconds
- Aggregate two-stage worker time: {metadata['aggregate_fit_seconds'] / 3600:.4f} worker-hours
{recompute_line.rstrip()}
- Maximum recorded worker RSS: {metadata['max_worker_peak_rss_mib']:.1f} MiB
- CPU: {system['cpu_model']}
- Logical CPUs available: {system['logical_cpus_available']}
- System memory: {memory_text}
- Accelerator: {system['accelerator']}
- Python: {system['python']}
- Package versions: `{json.dumps(system['package_versions'], sort_keys=True)}`

The publication run is CPU-only. Preliminary solver benchmarks, unit tests,
and smoke runs used less than 0.1 additional worker-hour and are not included in
the publication-run total above.
"""
    _atomic_text(output, text)


def write_numerical_section(
    accuracy: pd.DataFrame,
    summary: pd.DataFrame,
    state: pd.DataFrame,
    metadata: dict[str, Any],
    output: Path,
) -> None:
    pmax = int(summary["p"].max())
    largest = summary[summary["p"] == pmax].sort_values("case_index")
    signs_match = int((summary["empirical_improves"] == summary["se_improves"]).sum())
    comparisons = int(len(summary))
    excluded = int(summary["ci_excludes_zero"].sum())
    max_kkt = float(metadata["max_kkt_residual"])
    max_se_residual = float(metadata["max_state_evolution_residual"])
    pmax_accuracy = accuracy[accuracy["p"] == pmax].iloc[0]
    max_composition = float(state["composition_error"].max())
    max_bound_ratio = float(state["perturbation_bound_ratio"].max())
    improvements = largest[largest["case_id"].str.endswith("improve")]
    improvement_text = ", ".join(
        f"{100*(row['risk1_mean']-row['risk2_mean'])/row['risk1_mean']:.1f}\\% "
        f"({row['case_label'].split(':')[0]})"
        for _, row in improvements.iterrows()
    )
    system = metadata["system"]
    model = metadata["model"]
    solver = metadata["solver"]
    state_config = metadata["state_evolution"]
    dimensions = ",".join(str(value) for value in metadata["p_values"])
    case_count = len(metadata["cases"])
    improvement_count = len(improvements)
    results_profile = metadata["profile_name"]
    memory_text = _format_memory_gib(system.get("memory_gib_available"))
    kkt_tolerance_latex = _sci_latex(float(solver["kkt_tolerance"]))
    se_tolerance_latex = _sci_latex(float(state_config["tolerance"]))
    max_kkt_latex = _sci_latex(max_kkt, 2)
    max_se_residual_latex = _sci_latex(max_se_residual, 2)
    max_composition_latex = _sci_latex(max_composition, 2)

    latex = rf"""
\section{{Numerical Validation}}
\label{{app:numerical-validation}}

We test the finite-$p$ two-stage state-evolution description, the
threshold-renormalization mechanism, and the off-critical prediction-risk
criterion. These experiments provide numerical evidence for
Assumption~\ref{{ass:two-stage-state-evolution}}; they do not prove the
oversampled state-evolution extension.

\subsection{{Experimental setup}}
\label{{app:numerical-validation-setup}}

We use the Gaussian model and power-law coefficients of
Section~\ref{{sec:model-main-results}} with
$p\in\{{{dimensions}\}}$, $n_p=\operatorname{{round}}(p^{{\alpha_1}})$,
$m_p=\operatorname{{round}}(p^{{\alpha_2}})$, $\gamma={model['gamma']}$, and
$\Delta={model['delta']}$.
The penalties are
\[
\lambda_{{1,p}}={model['nu1']:.2f}\sqrt{{2n_p/p}},\qquad
\lambda_{{2,p}}={model['nu2']:.2f}\sqrt{{2m_p/p}}.
\]
We evaluate the {case_count} exponent pairs in
Table~\ref{{tab:matched-phase-results}}. Each case--dimension cell contains
{metadata['replicates_per_cell']}
independent paired repetitions, giving {metadata['two_stage_fits']:,} complete
two-stage fits. The population risks are evaluated exactly as
$p^{{-1}}\|\widehat\beta_k-\beta^\ast\|_2^2$; no finite test set is used.

For each stage we sample the exact Gaussian--Wishart sufficient statistics by
Bartlett decomposition. Stage-I score noise is then drawn conditionally on the
sampled Gram matrix, preserving the exact joint sufficient-statistics law.
The Gram-form LASSO objectives are solved by deterministic accelerated proximal
gradient with adaptive restart. A fit is accepted only after its KKT residual
falls below ${kkt_tolerance_latex}$; the largest observed residual was
${max_kkt_latex}$. Stage-I Gaussian moments and active probabilities and the
conditional Stage-II truncated-normal moments are evaluated analytically. The
remaining expectation over the Stage-I scalar input uses
{state_config['quadrature_nodes']}-node Gauss--Hermite quadrature. The
calibrated fixed points are solved to tolerance
${se_tolerance_latex}$;
the largest final fixed-point or calibration residual was
${max_se_residual_latex}$. All random streams are deterministic functions of the
registered master seed, case, dimension, repetition, and stream role.

We report pointwise paired 95\% percentile-bootstrap intervals for mean risk
differences,
using {metadata['bootstrap_resamples']:,} resamples of the independent complete
repetitions. Thus the
intervals capture variation in the signal, both Gaussian designs, and Stage-I
label noise. The complete seeds, raw repetitions, configurations, and generated
outputs are included in the supplementary package.

\subsection{{\texorpdfstring{{Finite-$p$ state-evolution accuracy}}{{Finite-p state-evolution accuracy}}}}
\label{{app:numerical-validation-se}}

For each cell we compare the Monte Carlo mean population risk with its
finite-$p$ state-evolution prediction. Table~\ref{{tab:se-risk-accuracy}}
summarizes the absolute relative discrepancies across the {case_count} cases. At
$p={pmax}$, the median discrepancies are
{100*pmax_accuracy['stage1_median_relative_error']:.3f}\% for Stage I and
{100*pmax_accuracy['stage2_median_relative_error']:.3f}\% for Stage II; the
corresponding maxima are
{100*pmax_accuracy['stage1_max_relative_error']:.3f}\% and
{100*pmax_accuracy['stage2_max_relative_error']:.3f}\%.

\input{{../results/{results_profile}/latex/se_accuracy_table}}

\begin{{figure}}[t]
    \centering
    \includegraphics[width=\linewidth]{{../results/{results_profile}/figures/risk_accuracy.pdf}}
    \caption{{Relative deviation of Monte Carlo mean risks from finite-$p$
    state-evolution risks. Error bars transform the pointwise 95\% bootstrap
    intervals for the empirical means.}}
    \label{{fig:se-risk-accuracy}}
\end{{figure}}

\subsection{{Matched phase test}}
\label{{app:numerical-validation-phase}}

Let $D_p=\mathcal R_p^{{(2)}}-\mathcal R_p^{{(1)}}$, so $D_p<0$ denotes
weak-to-strong improvement. The empirical mean and state evolution agree in
sign in {signs_match} of {comparisons} case--dimension comparisons. The paired
pointwise 95\% interval excludes zero in {excluded} of {comparisons} comparisons.
Table~\ref{{tab:matched-phase-results}} reports the largest dimension. At
$p={pmax}$, Stage II reduces the Monte Carlo mean prediction risk by
{improvement_text} in the
{improvement_count} theory-predicted improvement regimes.

\input{{../results/{results_profile}/latex/phase_table}}

\begin{{figure}}[t]
    \centering
    \includegraphics[width=0.72\linewidth]{{../results/{results_profile}/figures/phase_diagram.pdf}}
    \caption{{Off-critical theoretical regions and matched empirical outcomes
    at $p={pmax}$. Circles denote empirical improvement and crosses denote
    deterioration.}}
    \label{{fig:matched-phase-diagram}}
\end{{figure}}

\subsection{{Threshold-renormalization diagnostic}}
\label{{app:numerical-validation-threshold}}

For the scalar channels we compute
\[
Q_p=\frac1p\sum_{{i=1}}^p\mathbb E[(B_i-\overline B_{{i,p}})^2].
\]
The numerical results satisfy $Q_p\leq\sigma_{{2,p}}^2$ in every tested cell,
and the largest numerical error in the exact composition identity
$\eta(\eta(u;b_1);b_2)=\eta(u;b_1+b_2)$ is ${max_composition_latex}$.
Figure~\ref{{fig:threshold-diagnostic}} reports both the unnormalized
perturbation and its ratio to the Lipschitz bound for all {case_count} cases;
the largest ratio is {max_bound_ratio:.3f}. These are internal scalar-channel
checks of the deterministic composition identity and Lipschitz inequality,
not an independent coordinate-level validation of the finite-sample LASSO
mechanism. The empirical risk agreement in
Figure~\ref{{fig:se-risk-accuracy}} provides separate, indirect
finite-dimensional evidence for the scalar description.

\begin{{figure}}[t]
    \centering
    \includegraphics[width=\linewidth]{{../results/{results_profile}/figures/threshold_diagnostic.pdf}}
    \caption{{Internal Stage-II scalar perturbation diagnostic and its
    Lipschitz bound across all {case_count} matched cases.}}
    \label{{fig:threshold-diagnostic}}
\end{{figure}}

\subsection{{Compute resources and scope}}
\label{{app:numerical-validation-compute}}

The verified run used {metadata['workers']} CPU workers on an anonymized host
with an {system['cpu_model']} CPU. Available system memory was {memory_text};
no GPU was used. Python {system['python']} and exact package versions
are recorded in the supplementary metadata. The reported run took
{metadata['wall_seconds']/60:.1f} minutes wall-clock time and
{metadata['aggregate_fit_seconds']/3600:.2f} aggregate worker-hours; maximum
recorded worker resident memory was {metadata['max_worker_peak_rss_mib']:.0f}
MiB. Preliminary benchmarks, unit tests, and smoke runs used less than 0.1
additional worker-hour.

The evidence is limited to $p\leq{pmax}$, Gaussian designs, the stated power-law
signal, and selected off-critical exponent pairs and regularization constants.
It supports finite-dimensional agreement with the scalar predictions but does
not prove state evolution for the oversampled triangular array. Because the
discrepancies are not monotone in $p$, these experiments also do not establish
a convergence rate. They do not determine signs on critical boundaries or
establish transfer to nonlinear networks or real data. The reported confidence
intervals are pointwise rather than simultaneous across the 24 comparisons.
""".lstrip()
    _atomic_text(output, latex)


def generate_all_artifacts(
    *,
    summary: pd.DataFrame,
    accuracy: pd.DataFrame,
    state: pd.DataFrame,
    metadata: dict[str, Any],
    output_root: Path,
    gamma: float,
) -> None:
    figures = output_root / "figures"
    latex = output_root / "latex"
    plot_risk_accuracy(summary, figures)
    plot_phase_diagram(summary, gamma, figures)
    plot_threshold_diagnostic(state, figures)
    write_latex_tables(accuracy, summary, latex)
    write_numerical_section(
        accuracy,
        summary,
        state,
        metadata,
        latex / "numerical_validation_generated.tex",
    )
    write_compute_report(metadata, output_root / "COMPUTE.md")
    
