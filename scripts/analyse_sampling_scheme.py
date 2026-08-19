"""Does random calibration sampling really beat class-stratified sampling, and where?

Audit finding (claim-to-evidence re-audit, 2026-08-10): Sections V-E and VI-F recommend
sampling calibration data at random rather than class-balanced, and supported it with a single
cell of the design -- `k = 64` at the corpus's native 50/50 prevalence. Two problems, both
visible in the preserved EXP-002 table:

  1. At 50/50 the stated mechanism cannot operate. The mechanism is that stratification spends
     half the annotation budget on positives, which carry no information about a false-positive
     rate. But at 50/50 the *expected* number of negatives in a simple random draw of k is also
     k/2, so the two schemes are matched in expectation and only their variance differs. The
     cited cell is therefore the one place in the design where the explanation does not apply.
  2. The cited cell was also the strongest available at that prevalence, and a one-language
     difference at n = 14. Reporting it alone is selective.

This is re-analysis, not a new experiment: every row is already in
`results/tables/EXP-002_threshold_study.csv`, produced by the pre-registered prevalence sweep.
Nothing is re-embedded, no model is re-run, no data is collected. The script recomputes the
full scheme x k x prevalence grid, pools it, isolates the deployment-realistic regime where the
mechanism does operate, and records the counterexample cells so they can be disclosed rather
than discovered by a reviewer.

Every quantity the manuscript quotes for this claim is emitted here, so it cannot drift from
the grid that produced it (the failure mode recorded in P10).

Usage: python scripts/analyse_sampling_scheme.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "results" / "tables" / "EXP-002_threshold_study.csv"
OUT = REPO_ROOT / "results" / "tables" / "sampling_scheme_verdict.json"

BUDGET = 0.05
K_VALUES = [8, 16, 32, 64, 128, 256]
PREVALENCES = [0.50, 0.25, 0.10, 0.05]
#: Prevalences we treat as deployment-realistic. The corpus's native balance (0.50) is
#: excluded from this subset because no deployment runs at it, and because it is the one
#: setting in which the two sampling schemes draw the same expected number of negatives.
DEPLOYMENT_PREVALENCES = [0.10, 0.05]
#: Below this k no scheme reaches compliance in any condition (violation is 1.000 at k <= 16
#: almost throughout), so scheme comparisons there are uninformative rather than favourable.
USABLE_K_MIN = 32


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC}; run the EXP-002 stage first")
        return 1
    df = pd.read_csv(SRC)
    df = df[(df.fpr_budget == BUDGET) & (~df.is_source)]

    cells = []
    for prev in PREVALENCES:
        sub = df[df.prevalence_setting == prev]
        if sub.empty:
            continue
        # Realised dev prevalence, not the nominal setting: the expected negative count in a
        # random draw follows the corpus, and these differ slightly after subsampling.
        dev_prev = float(sub.dev_prevalence.median())
        for k in K_VALUES:
            r = sub[sub.strategy == f"few_label_{k}_rand"]
            s = sub[sub.strategy == f"few_label_{k}_strat"]
            if r.empty or s.empty:
                continue
            rv, sv = float((r.fpr > BUDGET).mean()), float((s.fpr > BUDGET).mean())
            cells.append({
                "prevalence": prev,
                "dev_prevalence": round(dev_prev, 4),
                "k": k,
                "n_languages": int(r.language.nunique()),
                "expected_negatives_random": round(k * (1 - dev_prev), 1),
                "expected_negatives_stratified": round(k * 0.5, 1),
                "random_violation": round(rv, 4),
                "stratified_violation": round(sv, 4),
                "delta_stratified_minus_random": round(sv - rv, 4),
                "random_median_fpr": round(float(r.fpr.median()), 4),
                "stratified_median_fpr": round(float(s.fpr.median()), 4),
                # Recall and the selected threshold are carried so that a cell favouring
                # stratification can be checked against the compliance-for-recall trade the
                # paper documents elsewhere, rather than being reported as unexplained.
                "random_median_recall": round(float(r.recall.median()), 4),
                "stratified_median_recall": round(float(s.recall.median()), 4),
                "random_median_threshold": round(float(r.threshold.median()), 4),
                "stratified_median_threshold": round(float(s.threshold.median()), 4),
                "favours": "random" if rv < sv else ("stratified" if sv < rv else "tie"),
            })

    def pooled(rows_filter) -> dict:
        sel = [c for c in cells if rows_filter(c)]
        if not sel:
            return {}
        # Pool at the (language, cell) level rather than averaging cell rates, so every
        # language-draw counts once and cells with more languages are not down-weighted.
        n_cells = len(sel)
        rv = float(np.mean([c["random_violation"] for c in sel]))
        sv = float(np.mean([c["stratified_violation"] for c in sel]))
        return {
            "n_cells": n_cells,
            "n_observations_per_scheme": int(sum(c["n_languages"] for c in sel)),
            "random_violation": round(rv, 4),
            "stratified_violation": round(sv, 4),
            "difference": round(sv - rv, 4),
            "cells_favouring_random": sum(c["favours"] == "random" for c in sel),
            "cells_favouring_stratified": sum(c["favours"] == "stratified" for c in sel),
            "cells_tied": sum(c["favours"] == "tie" for c in sel),
        }

    # Observation-level pooling, computed directly from the rows rather than from cell means,
    # as the honest denominator for "across the design".
    def obs_pooled(rows_filter) -> dict:
        keys = {(c["prevalence"], c["k"]) for c in cells if rows_filter(c)}
        r_all, s_all = [], []
        for prev, k in keys:
            sub = df[df.prevalence_setting == prev]
            r_all.append(sub[sub.strategy == f"few_label_{k}_rand"].fpr.values)
            s_all.append(sub[sub.strategy == f"few_label_{k}_strat"].fpr.values)
        r_all, s_all = np.concatenate(r_all), np.concatenate(s_all)
        return {
            "n_observations_per_scheme": int(r_all.size),
            "random_violation": round(float((r_all > BUDGET).mean()), 4),
            "stratified_violation": round(float((s_all > BUDGET).mean()), 4),
            "difference": round(float((s_all > BUDGET).mean() - (r_all > BUDGET).mean()), 4),
        }

    all_cells = pooled(lambda c: True)
    all_obs = obs_pooled(lambda c: True)
    deploy = pooled(lambda c: c["prevalence"] in DEPLOYMENT_PREVALENCES and c["k"] >= USABLE_K_MIN)
    deploy_obs = obs_pooled(
        lambda c: c["prevalence"] in DEPLOYMENT_PREVALENCES and c["k"] >= USABLE_K_MIN)
    balanced = pooled(lambda c: c["prevalence"] == 0.50)

    # Counterexamples are recorded explicitly rather than left for a reviewer to find. In each,
    # we check whether stratification's lower violation rate is bought by a more conservative
    # threshold and lower recall -- the same compliance-for-recall trade the paper documents for
    # the conservative global threshold -- or whether it reflects a genuinely better estimate.
    counter = [c for c in cells if c["favours"] == "stratified"]
    for c in counter:
        c["stratified_threshold_higher"] = bool(
            c["stratified_median_threshold"] > c["random_median_threshold"])
        c["stratified_recall_lower"] = bool(
            c["stratified_median_recall"] < c["random_median_recall"])
    counter_traded = all(c["stratified_threshold_higher"] and c["stratified_recall_lower"]
                         for c in counter) if counter else None

    # The negative-count advantage the mechanism rests on, at deployment prevalence.
    ratio = [round(c["expected_negatives_random"] / c["expected_negatives_stratified"], 2)
             for c in cells if c["prevalence"] in DEPLOYMENT_PREVALENCES]

    # Paired test on the deployment-realistic subset. The unit is the (prevalence, k) cell,
    # not the language-cell pair: the same 14 languages recur in every cell, so pooling pairs
    # would treat repeated measurements on one language sample as independent observations --
    # the error Section "Statistical validity" warns against for the m-association. With the
    # cell as the unit there are only eight of them, so this test is underpowered by
    # construction; we report it because reporting the difference with no uncertainty at all
    # would overstate it, not because eight cells can settle the question.
    deploy_cells = [c for c in cells
                    if c["prevalence"] in DEPLOYMENT_PREVALENCES and c["k"] >= USABLE_K_MIN]
    rv = [c["random_violation"] for c in deploy_cells]
    sv = [c["stratified_violation"] for c in deploy_cells]
    try:
        from scipy.stats import wilcoxon
        stat, pval = wilcoxon(rv, sv)
        paired = {"n_cells": len(deploy_cells), "statistic": float(stat),
                  "p_value": round(float(pval), 4),
                  "mean_random_violation": round(float(np.mean(rv)), 4),
                  "mean_stratified_violation": round(float(np.mean(sv)), 4),
                  "cells_favouring_random": sum(1 for a, b in zip(rv, sv) if a < b),
                  "cells_favouring_stratified": sum(1 for a, b in zip(rv, sv) if a > b),
                  "cells_tied": sum(1 for a, b in zip(rv, sv) if a == b),
                  "significant_at_0.05": bool(pval < 0.05)}
    except Exception as exc:                                    # pragma: no cover
        paired = {"error": str(exc)}

    verdict = {
        "source": "results/tables/EXP-002_threshold_study.csv",
        "analysis_type": "re-analysis of preserved results; no new experiment",
        "budget": BUDGET,
        "k_values": K_VALUES,
        "prevalences": PREVALENCES,
        "deployment_prevalences": DEPLOYMENT_PREVALENCES,
        "usable_k_min": USABLE_K_MIN,
        "cells": cells,
        "pooled_all_cells": all_cells,
        "pooled_all_observations": all_obs,
        "pooled_deployment_prevalence_usable_k": deploy,
        "pooled_deployment_prevalence_usable_k_observations": deploy_obs,
        "paired_test_deployment_prevalence_usable_k": paired,
        "pooled_balanced_corpus_only": balanced,
        "negative_count_ratio_at_deployment_prevalence": sorted(set(ratio)),
        "counterexample_cells": counter,
        "n_counterexample_cells": len(counter),
        "counterexample_k_values": sorted({c["k"] for c in counter}),
        "counterexample_max_k": max((c["k"] for c in counter), default=None),
        "counterexample_all_traded_recall_for_compliance": counter_traded,
    }
    OUT.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")

    print(f"{'prev':>5} {'k':>4} {'E[neg] r/s':>12} {'rand':>6} {'strat':>6} {'delta':>7}  favours")
    for c in cells:
        print(f"{c['prevalence']:>5.2f} {c['k']:>4} "
              f"{c['expected_negatives_random']:>5.1f}/{c['expected_negatives_stratified']:<6.1f} "
              f"{c['random_violation']:>6.3f} {c['stratified_violation']:>6.3f} "
              f"{c['delta_stratified_minus_random']:>+7.3f}  {c['favours']}")
    for name, blk in (("all cells", all_cells), ("all observations", all_obs),
                      ("deployment prevalence, k >= 32", deploy),
                      ("deployment prevalence, k >= 32 (obs)", deploy_obs),
                      ("balanced corpus only", balanced)):
        print(f"\n{name}: {json.dumps(blk)}")
    print(f"\ncounterexample cells ({len(counter)}): "
          f"{[(c['prevalence'], c['k']) for c in counter]}")
    print(f"\n-> {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
