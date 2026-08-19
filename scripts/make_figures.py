"""Generate every manuscript figure from preserved result tables.

FIGURE CONSTITUTION: no figure may exist only as an image. Every figure has
source data (a committed CSV), a processing step (this script), a plotting function
(src/analysis/figures.py), an exported image, and a caption -- and must be regenerable
automatically. This script is the "processing + export" link in that chain, and is called by
scripts/reproduce_all.py.

It writes a manifest recording, for each figure, its source table, generating function,
exported files and caption, so the chain is machine-checkable rather than asserted in prose.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.analysis import figures as F  # noqa: E402

TABLES = REPO_ROOT / "results" / "tables"
FIGURES = REPO_ROOT / "results" / "figures"

#: Strategy display order = the information spectrum from M5. Fixed so the figure never
#: silently re-sorts and changes the story it tells.
STRATEGY_ORDER = [
    "fixed_half", "source_global",
    "pooled_global", "worstcase_global",
    "quantile_match", "sld_em", "bbse",
    "few_label_8_rand", "few_label_32_rand", "few_label_128_rand", "few_label_256_rand",
    "target_full",
]

#: Resource tiering, fixed a priori from the corpus documentation -- NOT chosen after seeing
#: results (threat T-C4). Three tiers because the validated palette caps categorical slots
#: at three.
#: Human-readable strategy names, matching the tables. Raw identifiers such as
#: `few_label_8_rand` on a figure axis read as debug output next to a typeset table.
PRETTY_STRATEGY = {
    "fixed_half": "fixed 0.5",
    "source_global": "source-tuned global",
    "pooled_global": "pooled global",
    "worstcase_global": "worst-case global",
    "quantile_match": "quantile matching",
    "sld_em": "SLD/EM prior",
    "bbse": "BBSE label-shift",
    "target_full": "per-language (full)",
    "test_oracle_unachievable": "test oracle (unreachable)",
}


def prettify_strategy(name: str) -> str:
    if name in PRETTY_STRATEGY:
        return PRETTY_STRATEGY[name]
    if name.startswith("few_label_"):
        _, _, k, sampling = name.split("_")
        return f"few-label k={k} ({'random' if sampling == 'rand' else 'strat.'})"
    return name.replace("_", " ")


RESOURCE_TIER = {
    "en": "high", "de": "high", "fr": "high", "es": "high",
    "ru": "mid", "uk": "mid", "it": "mid", "ja": "mid", "zh": "mid",
    "ar": "mid", "hi": "mid", "he": "mid",
    "am": "low", "tt": "low", "hin": "low",
}

#: Captions name languages in full, as the manuscript body does ("0.491 in Tatar"). Kept here
#: so a generated caption cannot disagree with the prose about what a code means.
LANGUAGE_NAMES = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
    "ru": "Russian", "uk": "Ukrainian", "ar": "Arabic", "he": "Hebrew", "hi": "Hindi",
    "hin": "Hinglish", "ja": "Japanese", "zh": "Chinese", "am": "Amharic", "tt": "Tatar",
}


def _load(exp_id: str) -> pd.DataFrame:
    path = TABLES / f"{exp_id}_threshold_study.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing source table: {path}")
    df = pd.read_csv(path)
    df["prev"] = df["prevalence_setting"].astype(float)
    return df


#: The manifest records the PDF export only. Both formats are still produced -- the plotting
#: functions return both and tests/test_figures.py asserts it -- but the PNG is a preview
#: raster of the same figure, is not what the manuscript includes, and is not published, so
#: claiming it as an export would make the manifest describe a file a reader does not have.
#: The manifest's job is to bind each figure to its source data, plotting function and
#: caption; that binding is unchanged.


def fig1_threshold_divergence(manifest: list[dict]) -> None:
    """Per-language calibrated threshold vs the single source-tuned threshold."""
    # SOURCE: EXP-002, the controlled arm this figure is cited from (Section V-A).
    #
    # This figure was previously built from EXP-001, the F1-objective pilot. That was wrong in
    # two ways. The section citing it reports a false-positive-budget result, so the figure
    # showed thresholds selected under a different objective; and 6 of the pilot's 14
    # per-language thresholds are degenerate (predicted-positive rate 0.89-0.99), which are
    # exactly the six largest divergences -- so the headline median was inflated from 0.152 to
    # 0.279 by thresholds Section V-G calls an artefact. The pilot figure also contradicted
    # Table 3: it plotted Amharic at 0.041 where the table prints 0.797.
    src = TABLES / "EXP-002_threshold_study.csv"
    raw = _load("EXP-002")
    at_prev = raw[raw["prev"] == 0.50]
    tau_source = float(at_prev["tau_source"].dropna().unique()[0])
    d = (at_prev[at_prev["strategy"] == "target_full"]
         [["language", "threshold"]]
         .rename(columns={"threshold": "tau_language"})
         .assign(tau_source=tau_source))

    # Caption statistics are computed over TARGET languages only; the source language is
    # plotted (its divergence is zero by construction) but would bias the median downward.
    targets = at_prev[(~at_prev["is_source"]) & (at_prev["strategy"] == "target_full")]
    diffs = (targets["threshold"] - tau_source).abs()
    median_diff, max_diff = float(diffs.median()), float(diffs.max())
    max_lang = LANGUAGE_NAMES.get(
        targets.loc[diffs.idxmax(), "language"], targets.loc[diffs.idxmax(), "language"])

    paths = F.fig_threshold_divergence(d, "fig1_threshold_divergence", FIGURES)
    manifest.append({
        "figure": "fig1_threshold_divergence",
        "source_data": str(src.relative_to(REPO_ROOT)).replace("\\", "/"),
        "plotting_function": "src.analysis.figures.fig_threshold_divergence",
        "exports": [str(p.relative_to(REPO_ROOT)).replace("\\", "/")
                    for p in paths if p.suffix == ".pdf"],
        "caption": (
            "Per-language calibrated decision thresholds for a fixed multilingual classifier, "
            "each selected to meet a 0.05 false-positive budget on that language, against the "
            "single threshold tuned on the English source language (dashed line). Thresholds "
            f"diverge substantially: over the 14 target languages the median absolute "
            f"difference is {median_diff:.3f} on a [0,1] score scale, reaching {max_diff:.3f} "
            f"for {max_lang}. These are the same per-language thresholds tabulated in the "
            "per-language operating-point table."),
    })


def fig2b_strategy_tradeoff(manifest: list[dict]) -> None:
    """Paired panels: budget compliance AND the recall it costs."""
    src = TABLES / "EXP-002_threshold_study.csv"
    d = _load("EXP-002")
    native = d[(d["prev"] == 0.50) & (~d["is_source"])]
    keep = [s for s in STRATEGY_ORDER if s in set(native["strategy"])]
    sub = native[native["strategy"].isin(keep)].copy()
    sub["strategy"] = sub["strategy"].map(prettify_strategy)
    keep_pretty = [prettify_strategy(s) for s in keep]

    paths = F.fig_strategy_tradeoff(sub, "fig2b_strategy_tradeoff", FIGURES,
                                    order=keep_pretty, budget=0.05)
    manifest.append({
        "figure": "fig2b_strategy_tradeoff",
        "source_data": str(src.relative_to(REPO_ROOT)).replace("\\", "/"),
        "plotting_function": "src.analysis.figures.fig_strategy_tradeoff",
        "exports": [str(p.relative_to(REPO_ROOT)).replace("\\", "/")
                    for p in paths if p.suffix == ".pdf"],
        "caption": (
            "The deployment trade-off, shown as paired panels because the two measures are "
            "on different scales and a dual-axis chart would misrepresent them. Left: "
            "realised false-positive rate against the 0.05 budget set on English. Right: the "
            # Captions are escaped verbatim by sync_manuscript_figures.py, so this string
            # must stay plain text: LaTeX markup here reaches the page as literal characters.
            "recall that operating point achieves. Both panels plot means over the 14 target "
            "languages, with standard errors across languages; the tables and the text report "
            "medians, which are the more robust summary of a skewed per-language distribution "
            "and are not the quantity plotted here, so bar heights and table entries differ "
            "and both are correct. The most "
            "conservative strategy never "
            "exceeds the budget, but the right panel shows it does so by barely firing; a "
            "reader shown only the left panel would draw the opposite conclusion."),
    })


def fig3_label_efficiency(manifest: list[dict]) -> None:
    """How many target-language labels buy budget compliance."""
    src = TABLES / "EXP-002_threshold_study.csv"
    d = _load("EXP-002")
    native = d[(d["prev"] == 0.50) & (~d["is_source"])]

    rows = []
    for strategy, g in native.groupby("strategy"):
        if not strategy.startswith("few_label_"):
            continue
        _, _, k, sampling = strategy.split("_")
        rows.append({
            "k": int(k),
            "sampling": {"rand": "random", "strat": "stratified"}[sampling],
            "violation_rate": float((g["fpr"] > 0.05).mean()),
        })
    df = pd.DataFrame(rows)

    full = native[native["strategy"] == "target_full"]
    paths = F.fig_label_efficiency(
        df, "fig3_label_efficiency", FIGURES,
        value_col="violation_rate",
        ylabel="Fraction of languages exceeding\nthe false-positive budget",
        reference_lines={"full-label calibration": float((full["fpr"] > 0.05).mean())})
    manifest.append({
        "figure": "fig3_label_efficiency",
        "source_data": str(src.relative_to(REPO_ROOT)).replace("\\", "/"),
        "plotting_function": "src.analysis.figures.fig_label_efficiency",
        "exports": [str(p.relative_to(REPO_ROOT)).replace("\\", "/")
                    for p in paths if p.suffix == ".pdf"],
        # CAPTION SCOPE. This panel is drawn from ONE prevalence, the corpus's native 50/50
        # balance (the `prev == 0.50` filter above). An earlier caption explained the
        # random-versus-stratified gap by the number of negatives each scheme supplies. That
        # explanation contradicts Section V-E, which states that at 50/50 a random draw of k
        # contains k/2 negatives in expectation, exactly as a stratified draw does, so the
        # negatives mechanism cannot operate in this panel at all. The caption now describes
        # what this panel shows and declines to offer a replacement explanation, because the
        # design does not identify one at this prevalence.
        "caption": (
            "Fraction of target languages exceeding the 0.05 false-positive budget, as a "
            "function of the number of labelled target-language examples used to set the "
            "threshold, at the corpus's native 50/50 class balance. Compliance is unattainable "
            "below k = 32. In this panel random sampling is never worse than class-stratified "
            "sampling and is better at two of the six draw sizes. Both schemes draw the same "
            "expected number of negatives at 50/50, so the negatives mechanism that separates "
            "them at deployment-realistic prevalence cannot account for that difference here; "
            "the results section reports the full sweep over four prevalences."),
    })


def fig4_auroc_vs_gap(manifest: list[dict]) -> None:
    """Identifiability check: is the loss just 'the model is worse here'?"""
    # SOURCE: EXP-003, the F1-objective experiment that Section V-G reports. Previously built
    # from EXP-001 (the pilot), which left that section drawing some numbers from one
    # experiment and some from another; the values are close but not identical, and nothing in
    # the manuscript disclosed that two experiments were in play.
    src = TABLES / "EXP-003_threshold_study.csv"
    raw = _load("EXP-003")
    at_prev = raw[(raw["prev"] == 0.50) & (~raw["is_source"])]
    tf = at_prev[at_prev["strategy"] == "target_full"].set_index("language")
    sg = at_prev[at_prev["strategy"] == "source_global"].set_index("language")
    d = pd.DataFrame({
        "language": tf.index,
        "test_auroc": tf["test_auroc"].to_numpy(),
        # F1 gain from calibrating on the language itself, against the source-tuned threshold.
        "f1_gap": (tf["f1"] - sg["f1"]).to_numpy(),
    })
    d["tier"] = d["language"].map(RESOURCE_TIER).fillna("mid")

    rho, _ = spearmanr(d["test_auroc"], d["f1_gap"])
    paths = F.fig_auroc_vs_gap(d, "fig4_auroc_vs_gap", FIGURES, tier_col="tier")
    manifest.append({
        "figure": "fig4_auroc_vs_gap",
        "source_data": str(src.relative_to(REPO_ROOT)).replace("\\", "/"),
        "plotting_function": "src.analysis.figures.fig_auroc_vs_gap",
        "exports": [str(p.relative_to(REPO_ROOT)).replace("\\", "/")
                    for p in paths if p.suffix == ".pdf"],
        "caption": (
            "Threshold-free model quality (AUROC) against the F1 gained by per-language "
            "calibration at 50/50 class balance, by resource tier, over the 14 target "
            f"languages. The negative association (Spearman correlation {rho:.3f}) is the reason "
            "the F1 objective was rejected as primary: apparent calibration gains concentrate "
            "in languages where the model carries least signal, and it is there that the "
            "F1-optimal threshold degenerates toward flagging all text."),
    })


def fig5_degeneracy_vs_prevalence(manifest: list[dict]) -> None:
    """The artifact is caused by class balance and vanishes at realistic prevalence."""
    src = TABLES / "EXP-003_threshold_study.csv"
    d = _load("EXP-003")
    t = d[(~d["is_source"]) & (d["strategy"] == "target_full")]

    rows = []
    for prev, g in t.groupby("prev"):
        rows.append({"k": float(prev), "sampling": "degenerate languages (of 14)",
                     "f1_gap_to_full": float(g["is_degenerate"].sum())})
    df = pd.DataFrame(rows)

    F.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(F.WIDTH_1COL, 2.3))
    s = df.sort_values("k")
    ax.plot(s["k"], s["f1_gap_to_full"], color=F.SERIES[0], marker=F.MARKERS[0],
            ls=F.LINESTYLES[0], markeredgecolor=F.SURFACE, markeredgewidth=0.6)
    for _, r in s.iterrows():
        ax.annotate(f"{int(r['f1_gap_to_full'])}", xy=(r["k"], r["f1_gap_to_full"]),
                    xytext=(0, 6), textcoords="offset points", ha="center",
                    fontsize=6.5, color=F.INK_SECONDARY)
    ax.set_xlabel("Positive-class prevalence")
    ax.set_ylabel("Languages with a degenerate\noptimal threshold (of 14)")
    ax.set_ylim(-0.5, 7)
    paths = []
    for ext in ("pdf", "png"):
        p = FIGURES / f"fig5_degeneracy_vs_prevalence.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)

    manifest.append({
        "figure": "fig5_degeneracy_vs_prevalence",
        "source_data": str(src.relative_to(REPO_ROOT)).replace("\\", "/"),
        "plotting_function": "scripts/make_figures.py:fig5_degeneracy_vs_prevalence",
        "exports": [str(p.relative_to(REPO_ROOT)).replace("\\", "/")
                    for p in paths if p.suffix == ".pdf"],
        "caption": (
            "Number of target languages whose F1-optimal per-language threshold is degenerate "
            "(flagging over 80% of all text), as a function of positive-class prevalence. The "
            "artefact is produced by the corpus's artificial 50/50 balance and disappears "
            "entirely at deployment-realistic prevalence."),
    })


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for fn in (fig1_threshold_divergence,
               fig2b_strategy_tradeoff, fig3_label_efficiency,
               fig4_auroc_vs_gap, fig5_degeneracy_vs_prevalence):
        try:
            fn(manifest)
            print(f"  OK   {fn.__name__}")
        except FileNotFoundError as exc:
            print(f"  SKIP {fn.__name__}: {exc}")

    out = FIGURES / "figure_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(manifest)} figures -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


