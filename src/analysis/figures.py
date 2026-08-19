"""Figure generation for the manuscript.

FIGURE CONSTITUTION: no figure may exist only as an image. Every figure here is
produced by a named function from a preserved source table, and the caller records the source
data, the script, and the exported file. Nothing is drawn by hand.

DESIGN CONSTRAINTS (validated, not eyeballed)
---------------------------------------------
Palette: the categorical slots are capped at THREE. The 3-slot set below was run through the
data-viz validator against a WHITE print surface with the all-pairs gate and passes every
check (worst all-pairs CVD deltaE 9.2 deutan; worst normal-vision deltaE 24.0). Adding a
fourth slot FAILS the normal-vision floor (yellow vs orange, deltaE 13.7), so four
categorical colors are never used -- extra conditions become facets or a single-hue magnitude
encoding instead.

Aqua (#1baf7a) sits at 2.82:1 against white, below the 3:1 contrast gate. The validator's
relief rule therefore applies: wherever aqua carries meaning it is accompanied by a visible
direct label.

Identity NEVER rests on color alone: every multi-series figure also varies marker and line
style. This is required for the CVD case and is doubly necessary here because IEEE Access
papers are frequently read and printed in grayscale.

One axis only -- no dual-axis charts anywhere.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: figures are files, never windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = REPO_ROOT / "results" / "figures"

# --- validated categorical palette (max 3) -------------------------------------------
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")   # blue, orange, aqua
MARKERS = ("o", "s", "^")
LINESTYLES = ("-", "--", ":")

# --- single-hue sequential ramp (blue), for magnitude ---------------------------------
SEQ = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")

# --- chart chrome ---------------------------------------------------------------------
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

#: IEEE Access is a two-column format. A single-column figure is ~3.5in wide;
#: double-column ~7.16in. Sizing figures correctly at creation avoids the
#: scale-down-in-LaTeX step that makes fonts unreadable.
WIDTH_1COL = 3.5
WIDTH_2COL = 7.16


def apply_style() -> None:
    """Publication style: recessive chrome, thin marks, readable at print size."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def _save(fig: plt.Figure, name: str, out_dir: Path | None = None) -> list[Path]:
    """Save both PDF (vector, for LaTeX) and PNG (for quick inspection)."""
    out_dir = out_dir or FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def fig_threshold_divergence(df: pd.DataFrame, name: str = "fig_threshold_divergence",
                             out_dir: Path | None = None) -> list[Path]:
    """Per-language calibrated threshold vs the single source-tuned threshold.

    FORM: the job is *magnitude by identity* across a modest number of named categories, so a
    horizontal dot plot with a reference line beats a bar chart -- it shows the position of
    each language on the threshold scale and the distance to the source threshold, without
    implying that the bars' areas mean anything.

    Expects one row per language with columns: language, tau_language, tau_source.
    """
    apply_style()
    d = df.sort_values("tau_language").reset_index(drop=True)
    tau_source = float(d["tau_source"].iloc[0])

    fig, ax = plt.subplots(figsize=(WIDTH_1COL, 0.18 * len(d) + 1.1))
    y = np.arange(len(d))

    ax.vlines(tau_source, -0.7, len(d) - 0.3, color=INK_MUTED, lw=0.9, ls="--", zorder=1)
    ax.hlines(y, tau_source, d["tau_language"], color=GRID, lw=1.0, zorder=2)
    ax.scatter(d["tau_language"], y, s=22, color=SERIES[0], zorder=3,
               edgecolor=SURFACE, linewidth=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(d["language"])
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlabel("Calibrated decision threshold")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)

    # Direct label on the reference line rather than a legend entry. Anchored BELOW the
    # bottom row so it cannot collide with a data point, whichever language ranks first.
    ax.annotate(f"source threshold ({tau_source:.2f})",
                xy=(tau_source, -0.62), xytext=(4, 0), textcoords="offset points",
                fontsize=6.5, color=INK_SECONDARY, va="center", ha="left")
    return _save(fig, name, out_dir)


def fig_strategy_comparison(df: pd.DataFrame, name: str = "fig_strategy_comparison",
                            out_dir: Path | None = None,
                            value_col: str = "f1_gap_to_full",
                            order: list[str] | None = None,
                            xlabel: str | None = None,
                            reference_line: float | None = None,
                            reference_label: str | None = None) -> list[Path]:
    """Mean shortfall from the achievable per-language optimum, by strategy.

    FORM: the job is *magnitude across an ordered set of methods* with one measure, so a
    horizontal bar chart with direct value labels is correct. A single hue is used because
    the categories are one ordered family, not distinct identities -- color would be
    decoration, and decoration that fails the 4-slot gate at that.

    Expects long-form rows: strategy, <value_col>, and one row per (language, strategy).
    """
    apply_style()
    agg = (df.groupby("strategy")[value_col]
             .agg(["mean", "std", "count"]).reset_index())
    if order:
        agg["__o"] = agg["strategy"].map({s: i for i, s in enumerate(order)})
        agg = agg.dropna(subset=["__o"]).sort_values("__o")
    else:
        agg = agg.sort_values("mean")

    fig, ax = plt.subplots(figsize=(WIDTH_1COL, 0.24 * len(agg) + 1.0))
    y = np.arange(len(agg))
    se = agg["std"] / np.sqrt(agg["count"].clip(lower=1))

    ax.barh(y, agg["mean"], height=0.62, color=SEQ[3], edgecolor=SURFACE, linewidth=0.8)
    ax.errorbar(agg["mean"], y, xerr=se, fmt="none", ecolor=INK_MUTED,
                elinewidth=0.8, capsize=2)

    # Place each value label beyond the END OF THE ERROR BAR, not the end of the bar --
    # otherwise the text sits on top of the whisker cap and both become unreadable.
    for yi, v, e in zip(y, agg["mean"], se.fillna(0.0)):
        ax.annotate(f"{v:.3f}", xy=(v + e, yi), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=6.5, color=INK_SECONDARY)

    if reference_line is not None:
        ax.axvline(reference_line, color=INK_MUTED, lw=0.9, ls="--", zorder=4)
        if reference_label:
            ax.annotate(reference_label, xy=(reference_line, len(agg) - 0.35),
                        xytext=(3, 0), textcoords="offset points",
                        fontsize=6.5, color=INK_SECONDARY, va="center", ha="left")

    ax.set_yticks(y)
    ax.set_yticklabels(agg["strategy"])
    ax.invert_yaxis()
    # The axis label is a parameter, never a default: this renderer is reused for several
    # different measures, and an inherited label silently mislabels the science.
    ax.set_xlabel(xlabel or "Mean F1 shortfall vs per-language optimum\n(lower is better)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.margins(x=0.16)
    return _save(fig, name, out_dir)


def fig_strategy_tradeoff(df: pd.DataFrame, name: str = "fig_strategy_tradeoff",
                          out_dir: Path | None = None, order: list[str] | None = None,
                          budget: float = 0.05) -> list[Path]:
    """Paired panels: realised false-positive rate AND the recall it buys.

    FORM: two measures on different scales for the same categories. A dual-axis chart is
    never acceptable, so this is a SMALL MULTIPLE -- two panels sharing the strategy axis.

    Why this figure is necessary rather than decorative: shown FPR alone, the most
    conservative strategy looks best (it never exceeds the budget). The recall panel reveals
    that it achieves this by barely firing. Presenting only the left panel would actively
    mislead, which is precisely the failure mode the degeneracy guard exists to prevent.
    """
    apply_style()
    agg = (df.groupby("strategy")[["fpr", "recall"]]
             .agg(["mean", "std", "count"]))
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    if order:
        agg["__o"] = agg["strategy"].map({s: i for i, s in enumerate(order)})
        agg = agg.dropna(subset=["__o"]).sort_values("__o")

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_2COL, 0.26 * len(agg) + 1.2), sharey=True)
    y = np.arange(len(agg))

    for ax, (col, label, ref) in zip(axes, [
        ("fpr", f"Realised false-positive rate\n(budget = {budget})", budget),
        ("recall", "Recall achieved\n(higher is better)", None),
    ]):
        se = agg[f"{col}_std"] / np.sqrt(agg[f"{col}_count"].clip(lower=1))
        ax.barh(y, agg[f"{col}_mean"], height=0.62, color=SEQ[3],
                edgecolor=SURFACE, linewidth=0.8)
        ax.errorbar(agg[f"{col}_mean"], y, xerr=se.fillna(0.0), fmt="none",
                    ecolor=INK_MUTED, elinewidth=0.8, capsize=2)
        for yi, v, e in zip(y, agg[f"{col}_mean"], se.fillna(0.0)):
            ax.annotate(f"{v:.3f}", xy=(v + e, yi), xytext=(5, 0),
                        textcoords="offset points", va="center",
                        fontsize=6.5, color=INK_SECONDARY)
        if ref is not None:
            ax.axvline(ref, color=INK_MUTED, lw=0.9, ls="--", zorder=4)
        ax.set_xlabel(label)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.margins(x=0.20)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(agg["strategy"])
    axes[0].invert_yaxis()
    fig.tight_layout()
    return _save(fig, name, out_dir)


def fig_label_efficiency(df: pd.DataFrame, name: str = "fig_label_efficiency",
                         out_dir: Path | None = None,
                         reference_lines: dict[str, float] | None = None,
                         ylabel: str | None = None,
                         value_col: str = "f1_gap_to_full") -> list[Path]:
    """F1 shortfall as a function of the target-language label budget k.

    FORM: change over an ordered quantity -> line chart. At most TWO series (stratified vs
    random sampling), each with its own marker AND line style, so identity survives grayscale
    printing and colorblind viewing.

    Expects columns: k, sampling, f1_gap_to_full.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(WIDTH_1COL, 2.5))

    for i, (label, sub) in enumerate(df.groupby("sampling")):
        s = sub.sort_values("k")
        ax.plot(s["k"], s[value_col], color=SERIES[i], marker=MARKERS[i],
                ls=LINESTYLES[i], label=str(label), markeredgecolor=SURFACE,
                markeredgewidth=0.6)

    for lbl, val in (reference_lines or {}).items():
        ax.axhline(val, color=INK_MUTED, lw=0.9, ls="--")
        ax.annotate(lbl, xy=(ax.get_xlim()[1], val), xytext=(-2, 3),
                    textcoords="offset points", ha="right", fontsize=6.5,
                    color=INK_SECONDARY)

    ax.set_xscale("log", base=2)
    # Label the actual budgets a practitioner would choose, not powers of two.
    ticks = sorted(df["k"].unique())
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(t)) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Labelled target-language examples (k)")
    ax.set_ylabel(ylabel or "F1 shortfall vs per-language optimum")
    ax.legend(loc="upper right")
    return _save(fig, name, out_dir)


def fig_auroc_vs_gap(df: pd.DataFrame, name: str = "fig_auroc_vs_gap",
                     out_dir: Path | None = None, tier_col: str | None = None) -> list[Path]:
    """Threshold-free model quality (AUROC) against the operating-point loss.

    FORM: relationship between two continuous measures -> scatter. This figure carries the
    project's identifiability argument: if the loss were merely "the model is worse in this
    language", the points would lie on a tight downward line. Scatter is therefore the right
    form precisely because it can FALSIFY the claim.

    Uses at most three tier colors (validated all-pairs) plus distinct markers; each point is
    directly labelled with its language, satisfying the relief rule for the low-contrast slot.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(WIDTH_1COL, 2.7))

    if tier_col and tier_col in df.columns:
        for i, (tier, sub) in enumerate(df.groupby(tier_col)):
            ax.scatter(sub["test_auroc"], sub["f1_gap"], s=26,
                       color=SERIES[i % 3], marker=MARKERS[i % 3], label=str(tier),
                       edgecolor=SURFACE, linewidth=0.6, zorder=3)
        ax.legend(loc="best", title=None)
    else:
        ax.scatter(df["test_auroc"], df["f1_gap"], s=26, color=SERIES[0],
                   edgecolor=SURFACE, linewidth=0.6, zorder=3)

    for _, r in df.iterrows():
        ax.annotate(str(r["language"]), xy=(r["test_auroc"], r["f1_gap"]),
                    xytext=(3, 2), textcoords="offset points",
                    fontsize=5.8, color=INK_SECONDARY)

    ax.axhline(0.0, color=AXIS, lw=0.7)
    ax.set_xlabel("Test AUROC (threshold-free model quality)")
    ax.set_ylabel("F1 gained by per-language calibration")
    ax.grid(axis="both", visible=True)
    return _save(fig, name, out_dir)
