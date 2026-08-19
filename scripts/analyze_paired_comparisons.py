"""Paired comparisons between threshold strategies (audit item: reviewer objection O6.2).

A reviewer will ask whether pooled calibration is *significantly* better than the status quo
or whether this is 14 noisy points. The comparison is genuinely paired -- the same languages
under two strategies, evaluated on identical scores -- which is what licenses a paired test
at all (`D-013`).

Reported per comparison: the median paired difference with a paired-bootstrap CI, the number
of languages favouring each strategy, a Wilcoxon signed-rank p-value, and the effect in
decision units. Non-independence of languages is stated with every result.

Only comparisons that answer a stated question are computed. There is no all-pairs matrix.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.statistics import paired_comparison, practical_significance  # noqa: E402

TABLES = REPO_ROOT / "results" / "tables"
BUDGET = 0.05

#: Each entry states the question the comparison answers. A comparison without a question
#: does not belong here.
COMPARISONS = [
    ("pooled_global", "source_global",
     "Does calibrating on other languages beat calibrating on one?"),
    ("target_full", "pooled_global",
     "Does per-language calibration beat pooled calibration?"),
    ("target_full", "source_global",
     "Does per-language calibration beat the status quo?"),
    ("quantile_match", "source_global",
     "Does label-free quantile matching beat doing nothing?"),
]


def main() -> int:
    src = TABLES / "EXP-002_threshold_study.csv"
    if not src.exists():
        print(f"missing {src}")
        return 1

    d = pd.read_csv(src)
    d["prev"] = pd.to_numeric(d["prevalence_setting"], errors="coerce")
    native = d[(d["prev"] == 0.50) & (~d["is_source"])]

    def series(strategy: str, col: str) -> pd.Series:
        return native[native["strategy"] == strategy].set_index("language")[col].sort_index()

    results = []
    for a, b, question in COMPARISONS:
        fa, fb = series(a, "fpr"), series(b, "fpr")
        ra, rb = series(a, "recall"), series(b, "recall")
        if fa.empty or fb.empty:
            print(f"SKIP {a} vs {b}: missing strategy")
            continue
        common = fa.index.intersection(fb.index)

        fpr_cmp = paired_comparison(fa.loc[common].to_numpy(), fb.loc[common].to_numpy(),
                                    seed=11, lower_is_better=True)
        rec_cmp = paired_comparison(ra.loc[common].to_numpy(), rb.loc[common].to_numpy(),
                                    seed=12, lower_is_better=False)

        viol_a = float((fa.loc[common] > BUDGET).mean())
        viol_b = float((fb.loc[common] > BUDGET).mean())

        results.append({
            "question": question, "strategy_a": a, "strategy_b": b,
            "n_languages": int(len(common)),
            "fpr": {
                "median_difference": round(fpr_cmp.median_difference, 4),
                "ci95": [round(fpr_cmp.ci_low, 4), round(fpr_cmp.ci_high, 4)],
                "n_favouring_a": fpr_cmp.n_favouring_a,
                "wilcoxon_p": None if fpr_cmp.p_value is None else round(fpr_cmp.p_value, 5),
            },
            "recall": {
                "median_difference": round(rec_cmp.median_difference, 4),
                "ci95": [round(rec_cmp.ci_low, 4), round(rec_cmp.ci_high, 4)],
                "n_favouring_a": rec_cmp.n_favouring_a,
                "wilcoxon_p": None if rec_cmp.p_value is None else round(rec_cmp.p_value, 5),
            },
            "violation_rate_a": round(viol_a, 4),
            "violation_rate_b": round(viol_b, 4),
            "effect_in_decision_units": {
                "median_over_budget_multiple_a": round(float(fa.loc[common].median() / BUDGET), 2),
                "median_over_budget_multiple_b": round(float(fb.loc[common].median() / BUDGET), 2),
            },
            "caveat": fpr_cmp.note,
        })

        print(f"\n=== {question} ===")
        print(f"  {a}  vs  {b}   (n = {len(common)} languages, paired)")
        print(f"  FPR    : median Δ = {fpr_cmp.median_difference:+.4f} "
              f"[{fpr_cmp.ci_low:+.4f}, {fpr_cmp.ci_high:+.4f}], "
              f"{fpr_cmp.n_favouring_a}/{len(common)} favour {a}, "
              f"Wilcoxon p = {fpr_cmp.p_value:.5f}")
        print(f"  Recall : median Δ = {rec_cmp.median_difference:+.4f} "
              f"[{rec_cmp.ci_low:+.4f}, {rec_cmp.ci_high:+.4f}], "
              f"{rec_cmp.n_favouring_a}/{len(common)} favour {a}, "
              f"Wilcoxon p = {rec_cmp.p_value:.5f}")
        print(f"  Budget violation: {a} {viol_a:.3f}  vs  {b} {viol_b:.3f}")
        print(f"  Over-budget multiple (median): "
              f"{fa.loc[common].median() / BUDGET:.2f}x  vs  "
              f"{fb.loc[common].median() / BUDGET:.2f}x")

    out = TABLES / "paired_comparisons.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print("\n--- practical significance of the status quo ---")
    sq = series("source_global", "fpr")
    print("  " + practical_significance(float(sq.median()), BUDGET))
    print(f"\nWrote {out.name}")
    print("\nNOTE: languages are not independent (shared families, scripts, encoder "
          "capacity); effective n is below the nominal pair count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
