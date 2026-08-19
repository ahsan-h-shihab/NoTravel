"""Connect the observed calibration failures to their statistical cause.

Why this analysis exists (PRESUBMISSION_AUDIT O4.1): a reviewer will say the "it is the
negatives, not the labels" finding is simply the standard error of a proportion. That
objection has force. The answer is not to deny it -- it is to show the empirical failures
land where the theory predicts, which turns "you rediscovered statistics" into "they identify
the cause of a practical failure and quantify when it bites".

The theory, stated plainly
--------------------------
Setting a threshold to achieve a false-positive rate `alpha` means locating the `alpha`
quantile of the negative-class score distribution. The information available for that is the
number of *negatives*, `n_neg`; the expected number of permitted false positives is
`m = alpha * n_neg`. The empirical FPR at a threshold estimated from `n_neg` negatives has
standard error

    SE(FPR) = sqrt(alpha * (1 - alpha) / n_neg)

so the RELATIVE error is ~ 1/sqrt(m). Reliability is therefore governed by `m`, not by the
total label count -- which is exactly the paper's claim. The familiar rule of thumb is that a
binomial estimate needs m >= 10 expected events to be usable.

This script computes `m` for every calibration regime the paper reports and places the
observed budget-violation rates beside it. No new experiments; arithmetic over existing runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

TABLES = REPO_ROOT / "results" / "tables"

#: Rule-of-thumb threshold for a usable binomial rate estimate.
USABLE_EVENTS = 10.0


def expected_false_positives(n_neg: float, alpha: float) -> float:
    return n_neg * alpha


def relative_se(n_neg: float, alpha: float) -> float:
    """Relative standard error of an FPR estimate from `n_neg` negatives."""
    if n_neg <= 0 or alpha <= 0:
        return float("inf")
    return float(np.sqrt(alpha * (1 - alpha) / n_neg) / alpha)


def _load(exp_id: str) -> pd.DataFrame | None:
    p = TABLES / f"{exp_id}_threshold_study.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d["prev"] = pd.to_numeric(d["prevalence_setting"], errors="coerce")
    return d


def main() -> int:
    rows: list[dict] = []

    # ---- Regime 1: few-label sweep on the toxicity corpus (EXP-002, native prevalence) ----
    d2 = _load("EXP-002")
    if d2 is not None:
        native = d2[(d2["prev"] == 0.50) & (~d2["is_source"])]
        prevalence = float(native["dev_prevalence"].median())
        for strategy, g in native.groupby("strategy"):
            if not strategy.startswith("few_label_"):
                continue
            _, _, k, sampling = strategy.split("_")
            k = int(k)
            # Negatives available: stratified spends half the budget on positives.
            n_neg = k / 2 if sampling == "strat" else k * (1 - prevalence)
            rows.append({
                "regime": f"few-label k={k} ({'stratified' if sampling=='strat' else 'random'})",
                "source": "EXP-002",
                "alpha": 0.05,
                "n_neg": round(n_neg, 1),
                "expected_fp": round(expected_false_positives(n_neg, 0.05), 2),
                "relative_se": round(relative_se(n_neg, 0.05), 2),
                "predicted_usable": expected_false_positives(n_neg, 0.05) >= USABLE_EVENTS,
                "observed_violation_rate": round(float((g["fpr"] > 0.05).mean()), 3),
            })

        # ---- Regime 2: full per-language calibration, toxicity ----
        tf = native[native["strategy"] == "target_full"]
        n_neg_full = float(tf["n_dev"].median()) * (1 - prevalence)
        rows.append({
            "regime": "full per-language calibration (toxicity)",
            "source": "EXP-002", "alpha": 0.05,
            "n_neg": round(n_neg_full, 1),
            "expected_fp": round(expected_false_positives(n_neg_full, 0.05), 2),
            "relative_se": round(relative_se(n_neg_full, 0.05), 2),
            "predicted_usable": expected_false_positives(n_neg_full, 0.05) >= USABLE_EVENTS,
            "observed_violation_rate": round(float((tf["fpr"] > 0.05).mean()), 3),
        })

    # ---- Regime 3: full per-language calibration on the parallel corpus (EXP-006) ----
    d6 = _load("EXP-006")
    if d6 is not None:
        t6 = d6[(~d6["is_source"]) & (d6["strategy"] == "target_full")]
        n_neg_sib = float(t6["n_dev"].median()) * (1 - float(t6["dev_prevalence"].median()))
        rows.append({
            "regime": "full per-language calibration (SIB-200)",
            "source": "EXP-006", "alpha": 0.05,
            "n_neg": round(n_neg_sib, 1),
            "expected_fp": round(expected_false_positives(n_neg_sib, 0.05), 2),
            "relative_se": round(relative_se(n_neg_sib, 0.05), 2),
            "predicted_usable": expected_false_positives(n_neg_sib, 0.05) >= USABLE_EVENTS,
            "observed_violation_rate": round(float((t6["fpr"] > 0.05).mean()), 3),
        })

    # ---- Regime 4: budget sweep -- tightening alpha shrinks the permitted events ----
    d7 = _load("EXP-007")
    if d7 is not None:
        t7 = d7[(~d7["is_source"]) & (d7["strategy"] == "target_full")]
        for budget, g in t7.groupby("fpr_budget"):
            n_neg_b = float(g["n_dev"].median()) * (1 - float(g["dev_prevalence"].median()))
            rows.append({
                "regime": f"full per-language calibration @ budget {budget:g}",
                "source": "EXP-007", "alpha": float(budget),
                "n_neg": round(n_neg_b, 1),
                "expected_fp": round(expected_false_positives(n_neg_b, float(budget)), 2),
                "relative_se": round(relative_se(n_neg_b, float(budget)), 2),
                "predicted_usable": expected_false_positives(n_neg_b, float(budget)) >= USABLE_EVENTS,
                "observed_violation_rate": round(float((g["fpr"] > budget).mean()), 3),
            })

    if not rows:
        print("No source tables found; run the experiments first.")
        return 1

    df = pd.DataFrame(rows).sort_values("expected_fp").reset_index(drop=True)
    out_csv = TABLES / "negatives_theory_vs_observed.csv"
    df.to_csv(out_csv, index=False)

    print("Expected permitted false positives m = alpha * n_neg, against observed failure\n")
    print(df.to_string(index=False))

    # ---- Does the m >= 10 rule separate the successes from the failures? ----
    usable = df[df["predicted_usable"]]
    unusable = df[~df["predicted_usable"]]
    verdict = {
        "rule": f"expected permitted false positives m = alpha * n_neg >= {USABLE_EVENTS}",
        "n_regimes": int(len(df)),
        "n_predicted_usable": int(len(usable)),
        "median_violation_when_predicted_usable": (
            round(float(usable["observed_violation_rate"].median()), 3) if len(usable) else None),
        "median_violation_when_predicted_unusable": (
            round(float(unusable["observed_violation_rate"].median()), 3) if len(unusable) else None),
    }
    if len(usable) and len(unusable):
        from scipy.stats import spearmanr
        r = spearmanr(df["expected_fp"], df["observed_violation_rate"])
        verdict["spearman_expected_fp_vs_violation"] = {
            "rho": round(float(r.statistic), 3),
            "p": round(float(r.pvalue), 5),
            "n": int(len(df)),
        }

        # ---- The same association on independent cells only. -------------------------
        # The 18 regimes are not independent: 12 are nested few-label subsamples drawn from
        # one experiment over the same 14 languages and the same frozen scores, so a p-value
        # computed across all 18 would be anti-conservative. Collapsing to one row per
        # (experiment, budget) cell leaves six. The 13 EXP-002 rows at alpha = 0.05 (12
        # few-label draw sizes plus full per-language calibration) collapse to the
        # UNWEIGHTED MEAN of their m and of their violation rate; every other cell is a
        # single regime and passes through unchanged. The rule is recorded here rather than
        # described in prose because the manuscript quotes the resulting rho.
        cells = []
        for (src, alpha), g in df.groupby(["source", "alpha"]):
            cells.append({
                "source": src,
                "alpha": round(float(alpha), 2),
                "n_regimes_collapsed": int(len(g)),
                "m": round(float(g["expected_fp"].mean()), 2),
                "violation": round(float(g["observed_violation_rate"].mean()), 3),
            })
        cells.sort(key=lambda c: c["m"])
        rc = spearmanr([c["m"] for c in cells], [c["violation"] for c in cells])

        # Exact two-sided permutation p. With six cells the whole null distribution is 720
        # orderings, so the exact value is cheap and does not rest on the t-approximation,
        # which is unreliable at this n. Reported instead of it, not alongside it.
        from itertools import permutations
        ms = [c["m"] for c in cells]
        vs = [c["violation"] for c in cells]
        obs = abs(float(rc.statistic))
        n_perm = 0
        n_extreme = 0
        for perm in permutations(vs):
            n_perm += 1
            if abs(float(spearmanr(ms, perm).statistic)) >= obs - 1e-12:
                n_extreme += 1
        verdict["clustered_independent_cells"] = {
            "collapsing_rule": ("one cell per (experiment, budget); the 13 EXP-002 rows at "
                                "alpha=0.05 collapse to the unweighted mean of m and of the "
                                "violation rate; all other cells are single regimes"),
            "n_cells": len(cells),
            "cells": cells,
            "rho": round(float(rc.statistic), 3),
            "p_t_approximation": round(float(rc.pvalue), 4),
            "p_exact_two_sided": round(n_extreme / n_perm, 4),
            "n_permutations": n_perm,
            "n_at_least_as_extreme": n_extreme,
        }

    (TABLES / "negatives_theory_verdict.json").write_text(
        json.dumps(verdict, indent=2) + "\n", encoding="utf-8")

    print("\n--- verdict ---")
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {out_csv.name} and negatives_theory_verdict.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
