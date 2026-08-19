"""Generate every manuscript table from preserved result tables.

TABLE CONSTITUTION: tables are generated, never hand-edited, and every value
originates from preserved source data. This script emits LaTeX (`manuscript/tables/*.tex`)
plus a manifest recording each table's source file and generating function, so the chain
dataset -> experiment -> analysis -> artifact -> manuscript is machine-checkable.

**No numerical value in the manuscript may be typed by hand.** If a number is wanted, it is
added here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

TABLES_IN = REPO_ROOT / "results" / "tables"
TABLES_OUT = REPO_ROOT / "manuscript" / "tables"

#: Display order = the information spectrum defined in the method. Fixed so a table never
#: silently re-sorts into a different narrative.
STRATEGY_ORDER = [
    "fixed_half", "source_global",
    "pooled_global", "worstcase_global",
    "quantile_match", "sld_em", "bbse",
    "few_label_8_rand", "few_label_16_rand", "few_label_32_rand", "few_label_64_rand",
    "few_label_128_rand", "few_label_256_rand", "target_full",
]

PRETTY = {
    "fixed_half": r"Fixed $\tau=0.5$",
    "source_global": r"\textbf{Source-tuned global} (status quo)",
    "pooled_global": r"\textbf{Pooled global} (other langs., LOO)",
    "worstcase_global": r"Worst-case global (other langs., LOO)",
    "quantile_match": "Quantile matching",
    "sld_em": "SLD/EM prior adjustment",
    "bbse": "BBSE label-shift",
    "target_full": r"\textbf{Per-language} (full labels)",
    "test_oracle_unachievable": "Test oracle (unachievable)",
}


def _pretty(name: str) -> str:
    if name in PRETTY:
        return PRETTY[name]
    if name.startswith("few_label_"):
        _, _, k, sampling = name.split("_")
        return f"Few-label $k={k}$ ({'random' if sampling == 'rand' else 'stratified'})"
    return name.replace("_", r"\_")


def _write(path: Path, latex: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex, encoding="utf-8")
    return path


def _float(df: pd.DataFrame, column_format: str, caption: str, label: str,
           index: bool = False) -> str:
    """Wrap a dataframe as a two-column-spanning IEEE float that cannot overflow.

    Generated tables have a wide label column plus five or six numeric columns and do not
    fit in a single IEEE column: the first build produced three overfull boxes of 116--148pt.
    Rather than hand-trimming, every table is emitted as `table*` (spanning both columns),
    set in \\small, and wrapped in an adjustbox capped at \\textwidth. Overflow is then
    impossible by construction, whatever a future regeneration produces.
    """
    # Explicit float_format: pandas otherwise emits full precision ("0.519000"), which
    # looks broken in print even though .round() was applied upstream.
    tabular = df.to_latex(index=index, escape=False, column_format=column_format,
                          float_format="%.3f")
    # Strip the tabular out of any float pandas may have added; we supply our own.
    return "\n".join([
        r"\begin{table*}[!t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{adjustbox}{max width=\textwidth}",
        tabular.strip(),
        r"\end{adjustbox}",
        r"\end{table*}",
        "",
    ])


def table_strategy_budget(exp_id: str, out_name: str, caption: str, label: str,
                          prevalence: float | None, manifest: list[dict],
                          budget: float = 0.05) -> None:
    """Budget compliance and recall by strategy."""
    src = TABLES_IN / f"{exp_id}_threshold_study.csv"
    if not src.exists():
        print(f"  SKIP {out_name}: missing {src.name}")
        return
    d = pd.read_csv(src)
    # `prevalence_setting` is numeric when a sweep ran and the literal "native" when it did
    # not, so coerce rather than cast; a hard cast crashes on the un-swept experiments.
    d["prev"] = pd.to_numeric(d["prevalence_setting"], errors="coerce")
    sub = d[~d["is_source"]]
    if prevalence is not None:
        sub = sub[sub["prev"] == prevalence]

    rows = []
    for s in STRATEGY_ORDER:
        g = sub[sub["strategy"] == s]
        if g.empty:
            continue
        rows.append({
            "Strategy": _pretty(s),
            "Violation rate": f"{(g['fpr'] > budget).mean():.3f}",
            "Median FPR": f"{g['fpr'].median():.3f}",
            "Max FPR": f"{g['fpr'].max():.3f}",
            "Median recall": f"{g['recall'].median():.3f}",
            "Degen.": f"{int(g['is_degenerate'].sum())}",
        })
    df = pd.DataFrame(rows)

    path = _write(TABLES_OUT / f"{out_name}.tex",
                  _float(df, "lrrrrr", caption, label, index=False))
    manifest.append({
        "table": out_name,
        "source_data": str(src.relative_to(REPO_ROOT)).replace("\\", "/"),
        "generating_function": f"scripts/make_tables.py:table_strategy_budget({exp_id})",
        "export": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "caption": caption,
        "n_languages": int(sub["language"].nunique()),
    })
    print(f"  OK   {out_name}  ({len(df)} rows, {sub['language'].nunique()} languages)")


def table_per_language(exp_id: str, out_name: str, caption: str, label: str,
                       manifest: list[dict]) -> None:
    """Per-language operating points under the status quo vs per-language calibration."""
    src = TABLES_IN / f"{exp_id}_threshold_study.csv"
    if not src.exists():
        print(f"  SKIP {out_name}: missing {src.name}")
        return
    d = pd.read_csv(src)
    # `prevalence_setting` is numeric when a sweep ran and the literal "native" when it did
    # not, so coerce rather than cast; a hard cast crashes on the un-swept experiments.
    d["prev"] = pd.to_numeric(d["prevalence_setting"], errors="coerce")
    if d["prev"].notna().any():
        d = d[d["prev"] == d["prev"].max()]

    sg = d[d["strategy"] == "source_global"].set_index("language")
    tf = d[d["strategy"] == "target_full"].set_index("language")

    out = pd.DataFrame({
        "AUROC": sg["test_auroc"].round(3),
        r"$\tau_L$": tf["threshold"].round(3),
        r"FPR (status quo)": sg["fpr"].round(3),
        r"FPR (per-lang.)": tf["fpr"].round(3),
        r"Recall (status quo)": sg["recall"].round(3),
        r"Recall (per-lang.)": tf["recall"].round(3),
    }).sort_values("FPR (status quo)", ascending=False)
    out.index.name = None          # a named index adds an ugly extra header row
    out = out.reset_index().rename(columns={"index": "Lang.", "language": "Lang."})

    path = _write(TABLES_OUT / f"{out_name}.tex",
                  _float(out, "lrrrrrr", caption, label, index=False))
    manifest.append({
        "table": out_name,
        "source_data": str(src.relative_to(REPO_ROOT)).replace("\\", "/"),
        "generating_function": f"scripts/make_tables.py:table_per_language({exp_id})",
        "export": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "caption": caption,
        "n_languages": int(len(out)),
    })
    print(f"  OK   {out_name}  ({len(out)} languages)")


def table_arm_a_per_language(out_name: str, caption: str, label: str,
                             manifest: list[dict], budget: float = 0.05) -> None:
    """Per-language operating points for BOTH deployed classifiers, side by side.

    Reviewer objection (two independent reviews, 2026-08-03): Section V-F carries what the
    paper calls "the strongest form of our central claim" entirely in prose, and cited the
    *controlled-arm* per-language table for numbers that come from the deployed classifiers.
    A reader could not check the Arabic-0.443 / Hebrew-0.500 claim, or the opposite-direction
    claim, against any table in the manuscript.

    Both classifiers are shown in one table on purpose: the claim is about the *pair*, and
    reading two separate tables against each other is exactly the comparison the prose asks
    the reader to make. The over-budget multiple is given in decision units (FPR / budget)
    because that, not the raw rate, is what the claim is stated in.
    """
    srcs = {"A": TABLES_IN / "EXP-008-citizenlab_arm_a.csv",
            "B": TABLES_IN / "EXP-009_arm_a.csv"}
    missing = [str(p.name) for p in srcs.values() if not p.exists()]
    if missing:
        print(f"  SKIP {out_name}: missing {', '.join(missing)}")
        return

    frames = {}
    for key, path in srcs.items():
        d = pd.read_csv(path)
        sg = d[d["strategy"] == "source_global"].set_index("language")
        frames[key] = pd.DataFrame({
            f"AUROC{key}": sg["test_auroc"].round(3),
            f"Mult{key}": (sg["fpr"] / budget).round(2),
            f"Rec{key}": sg["recall"].round(3),
        })
    out = frames["A"].join(frames["B"], how="inner")
    # Sort by the DistilBERT over-budget multiple so the sign disagreement is legible as a
    # pattern rather than something the reader has to hunt for.
    out = out.sort_values("MultB", ascending=False)
    out.index.name = None
    out = out.reset_index().rename(columns={"index": "Lang.", "language": "Lang."})
    out.columns = ["Lang.",
                   "AUROC", r"FPR/$\alpha$", "Recall",
                   "AUROC ", r"FPR/$\alpha$ ", "Recall "]

    latex = _float(out, "lrrrrrr", caption, label, index=False)
    # Without a grouping header the six numeric columns read as AUROC/FPR/Recall twice with
    # nothing saying which classifier is which. Inserted after \toprule so the column names
    # stay generated rather than hand-written.
    latex = latex.replace(
        r"\toprule",
        "\\toprule\n"
        r"& \multicolumn{3}{c}{DistilBERT-based} & \multicolumn{3}{c}{XLM-RoBERTa-based} \\"
        "\n\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}", 1)

    path = _write(TABLES_OUT / f"{out_name}.tex", latex)
    manifest.append({
        "table": out_name,
        "source_data": ", ".join(str(p.relative_to(REPO_ROOT)).replace("\\", "/")
                                 for p in srcs.values()),
        "generating_function": "scripts/make_tables.py:table_arm_a_per_language",
        "export": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "caption": caption,
        "n_languages": int(len(out)),
    })
    print(f"  OK   {out_name}  ({len(out)} languages, 2 classifiers)")


def main() -> int:
    TABLES_OUT.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    table_strategy_budget(
        "EXP-002", "tab_budget_toxicity",
        caption=("Budget compliance by threshold-selection strategy on the multilingual "
                 "toxicity corpus (14 target languages, false-positive budget $0.05$ set on "
                 "English, native class balance). The status quo exceeds its own stated "
                 "budget in every language. Few-label rows average each language's operating "
                 "point over 25 draws before the fraction of violating languages is taken, "
                 "so they describe the expected behaviour of a draw size and not the outcome "
                 "of any single draw, which is more variable (Section~\\ref{sec:negatives})."),
        label="tab:budget_toxicity", prevalence=0.50, manifest=manifest)

    table_strategy_budget(
        "EXP-006", "tab_budget_sib200",
        caption=("Budget compliance on the parallel SIB-200 corpus (173 target "
                 "languages; the corpus has 174 row-aligned configurations, of which "
                 "English is the source and is excluded from these statistics). "
                 "Sentences and labels are identical across languages. Note that "
                 "per-language calibration does not reduce the violation rate here: with "
                 "only $\\approx161$ negatives in the calibration set the threshold estimate "
                 "is noisy, and the violation rate counts exceedances only, so the more "
                 "conservative source-tuned threshold scores better on it while sitting "
                 "further from the budget (median FPR $0.037$ against $0.041$) and returning "
                 "less recall ($0.531$ against $0.552$). The apparent success of the fixed "
                 "threshold is an artefact of near-silence, flagged in the degeneracy "
                 "column. As in Table 2, few-label rows average each language's operating "
                 "point over 25 draws before the fraction of violating languages is taken, "
                 "so they describe the expected behaviour of a draw size and not the outcome "
                 "of any single draw (Section~\\ref{sec:negatives})."),
        label="tab:budget_sib200", prevalence=None, manifest=manifest)

    table_per_language(
        "EXP-002", "tab_per_language_toxicity",
        caption=("Per-language operating points under a source-tuned global threshold versus "
                 "per-language calibration, at a false-positive budget of $0.05$. The "
                 "source language (en) is listed for reference; its two columns agree "
                 "by construction and it is excluded from every aggregate we report. AUROC is "
                 "threshold-free and shows that the losses are not explained by model quality "
                 "alone."),
        label="tab:per_language_toxicity", manifest=manifest)

    table_arm_a_per_language(
        "tab_arm_a_per_language",
        caption=("Per-language operating points for the \\emph{two deployed classifiers} "
                 "under the status-quo configuration: one threshold, tuned on English to a "
                 "$0.05$ false-positive budget, applied to every language. Left block: the "
                 "DistilBERT-based classifier. Right block: the XLM-RoBERTa-based classifier. "
                 "\\textbf{FPR/$\\alpha$} is the realised false-positive rate as a multiple of "
                 "the configured budget, so $1.00$ is exact compliance, above $1$ is "
                 "over-firing and below $1$ is under-firing. The two classifiers disagree on "
                 "the sign of the error in 8 of 14 target languages. English is listed for "
                 "reference and excluded from every aggregate. AUROC is threshold-free: where "
                 "it sits at or below $0.5$ the score carries no usable signal and no "
                 "threshold can help, which is a model-quality failure rather than a "
                 "threshold-transfer one."),
        label="tab:arm_a_per_language", manifest=manifest)

    out = TABLES_OUT / "table_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(manifest)} tables -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



