"""Trace budget compliance against the number of calibration negatives, on the parallel corpus.

Reviewer request (Stanford review Q4, 2026-08-03): Table 2 shows per-language calibration doing
*worse* than the status quo on SIB-200, and the paper attributes that to the ~161 negatives each
language supplies. That attribution is an inference from the m rule, not a measurement, and the
reviewer asked for the curve itself.

This is a re-analysis of preserved scores, not a new experiment: nothing is re-embedded and no
model is re-run. For each language we subsample the calibration negatives down to a target
count, re-derive the threshold that meets the budget on that subsample, and evaluate it on the
untouched test split. Sweeping the target count traces violation rate against
m = alpha * n_neg on the same corpus and the same scores that produced Table 2.

The prediction under test is the one the paper already makes: compliance should degrade as m
falls, with the transition near m = 10. Nothing here is fitted -- the rule was fixed before this
analysis was written.

Usage: python scripts/analyse_negatives_curve.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
SCORES = REPO_ROOT / "experiments" / "runs" / "EXP-006" / "per_example_scores.parquet"
OUT_CSV = REPO_ROOT / "results" / "tables" / "negatives_curve_sib200.csv"
OUT_JSON = REPO_ROOT / "results" / "tables" / "negatives_curve_verdict.json"

BUDGET = 0.05
#: Calibration-negative counts to sweep. Capped at 128 so that EVERY language contributes to
#: EVERY point: the median language has ~161 dev negatives, so a 161-point would silently
#: restrict to the 98 languages that have that many and stop being comparable to the rest of
#: the curve (it gave 0.411 against Table 2's 0.376 for exactly that reason).
N_NEG_GRID = [8, 16, 24, 32, 48, 64, 96, 128]
N_DRAWS = 25            # subsampling replicates per (language, n_neg)
SEED = 20260803


# Reuse the pipeline's own selector rather than re-implementing it. A hand-rolled quantile
# rule disagreed with Table 2 at the full-data point (0.564 vs 0.376) purely through a
# different tie-break and candidate-grid convention -- exactly the kind of drift this project
# forbids. Importing the real function makes the curve reconcile with the table by
# construction.
import sys as _sys
_sys.path.insert(0, str(REPO_ROOT))
from src.eval.thresholds import tune_threshold  # noqa: E402


def threshold_at_budget(scores: np.ndarray, labels: np.ndarray, alpha: float) -> float:
    return tune_threshold(scores, labels, objective="target_fpr", target_fpr=alpha)


def main() -> int:
    if not SCORES.exists():
        print(f"missing {SCORES}; run the EXP-006 stage of reproduce_all.py first")
        return 1
    d = pd.read_parquet(SCORES)
    rng = np.random.default_rng(SEED)

    dev = d[d.split == "dev"]
    test = d[d.split == "test"]
    languages = sorted(set(dev.language) & set(test.language))

    rows = []
    for n_target in N_NEG_GRID:
        realised = []
        for lang in languages:
            dl, tl = dev[dev.language == lang], test[test.language == lang]
            neg_s = dl.score.to_numpy()[dl.label.to_numpy() == 0]
            pos_s = dl.score.to_numpy()[dl.label.to_numpy() == 1]
            if neg_s.size < n_target:
                continue   # language cannot supply this many; counted in n_languages_used
            ts, tlab = tl.score.to_numpy(), tl.label.to_numpy()
            t_neg = ts[tlab == 0]
            if t_neg.size == 0:
                continue
            for _ in range(N_DRAWS):
                idx = rng.choice(neg_s.size, size=n_target, replace=False)
                sub_s = np.concatenate([neg_s[idx], pos_s])
                sub_y = np.concatenate([np.zeros(n_target, int), np.ones(pos_s.size, int)])
                tau = threshold_at_budget(sub_s, sub_y, BUDGET)
                realised.append(float((t_neg >= tau).mean()))
        realised = np.asarray(realised)
        rows.append({
            "n_neg": n_target,
            "expected_fp": BUDGET * n_target,
            "n_languages_used": int(realised.size // N_DRAWS),
            "n_estimates": int(realised.size),
            "violation_rate": float((realised > BUDGET).mean()),
            "median_realised_fpr": float(np.median(realised)),
            "median_over_budget_multiple": float(np.median(realised) / BUDGET),
        })
        print(f"  n_neg={n_target:4d}  m={BUDGET * n_target:6.2f}  "
              f"violation={rows[-1]['violation_rate']:.3f}  "
              f"median FPR={rows[-1]['median_realised_fpr']:.4f}")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    below = df[df.expected_fp < 10]["violation_rate"]
    at_or_above = df[df.expected_fp >= 10]["violation_rate"]
    # Emitted rather than computed by hand at writing time. An earlier draft quoted the
    # trend statistic from a nine-point version of this grid after the grid was capped at
    # eight, and the manuscript disagreed with its own CSV; deriving it here removes that
    # class of drift entirely.
    rho, p_rho = spearmanr(df["n_neg"], df["violation_rate"])
    verdict = {
        "budget": BUDGET,
        "n_draws_per_language_per_point": N_DRAWS,
        "seed": SEED,
        "n_grid_points": int(len(df)),
        "violation_rate_min": round(float(df.violation_rate.min()), 4),
        "violation_rate_max": round(float(df.violation_rate.max()), 4),
        "spearman_n_neg_vs_violation": {"rho": round(float(rho), 4),
                                        "p": round(float(p_rho), 4),
                                        "n": int(len(df))},
        "median_violation_when_m_below_10": round(float(below.median()), 4),
        "median_violation_when_m_at_least_10": round(float(at_or_above.median()), 4),
        "violation_at_largest_comparable_point": round(float(df.iloc[-1].violation_rate), 4),
        "all_points_below_m10": bool((df.expected_fp < 10).all()),
        "max_m_reachable_on_this_corpus": round(float(df.expected_fp.max()), 2),
        "monotone_decreasing_in_n_neg": bool(
            np.all(np.diff(df.violation_rate.to_numpy()) <= 1e-9)),
    }
    OUT_JSON.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    print(f"\n{json.dumps(verdict, indent=2)}")
    print(f"\n-> {OUT_CSV.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
