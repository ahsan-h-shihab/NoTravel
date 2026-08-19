"""Emit the Arm A quantities the manuscript states in prose, so they cannot be mistyped.

Audit finding (claim-to-evidence re-audit, 2026-08-10): two prose claims about the deployed
classifiers did not survive recomputation from the preserved per-language tables, and both had
passed every earlier audit because those audits checked *numbers against artifacts* and not
*quantified prose against a per-unit distribution*.

  1. "a 144-fold difference" between the two classifiers' realised operating point in Amharic.
     The ratio of the raw false-positive rates is 141.0. The value 144 is what you get by
     dividing the two-decimal multiples printed in Table 4 (7.19 / 0.05 = 143.8), i.e. a
     rounding artefact of the table, not a property of the data.
  2. "the three strongest target languages" violate the budget under the second classifier.
     They are the second, third and fourth strongest. The strongest target language is French
     (AUROC 0.989), and French is within budget.

A third claim, "more than 99% of harmful content passed", holds only in the worst languages and
was stated without that qualifier in two of its three sites.

This script is re-analysis of preserved results; no experiment is re-run and no data collected.
Everything the manuscript asserts about ranks, counts and ratios in Arm A is emitted here and
checked against the manuscript by `scripts/check_quantified_claims.py`.

Usage: python scripts/analyse_arm_a_claims.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
T = REPO_ROOT / "results" / "tables"
OUT = T / "arm_a_claims.json"

BUDGET = 0.05
#: The two deployed classifiers, in the order the manuscript introduces them.
ARMS = {
    "distilbert": "EXP-008-citizenlab_arm_a.csv",
    "xlmr": "EXP-009_arm_a.csv",
}


def load(fname: str) -> pd.DataFrame:
    d = pd.read_csv(T / fname)
    return d[d.strategy == "source_global"]


def summarise(d: pd.DataFrame) -> dict:
    tgt = d[~d.is_source].copy()
    src = d[d.is_source].iloc[0]
    tgt = tgt.sort_values("test_auroc", ascending=False).reset_index(drop=True)
    tgt["violates"] = tgt.fpr > BUDGET
    tgt["rank_auroc"] = np.arange(1, len(tgt) + 1)
    strongest = tgt.iloc[0]
    # How far down the AUROC ranking you must go to collect the violators the manuscript
    # names. Stated as a rank so the prose cannot drift from the ordering in Table 4.
    viol_ranks = [int(r) for r in tgt.loc[tgt.violates, "rank_auroc"]]
    return {
        "n_target_languages": int(len(tgt)),
        "source_language": str(src.language),
        "source_auroc": round(float(src.test_auroc), 4),
        "source_recall": round(float(src.recall), 4),
        "source_fpr": round(float(src.fpr), 6),
        "violation_rate": round(float(tgt.violates.mean()), 4),
        "n_violators": int(tgt.violates.sum()),
        "median_over_budget_multiple": round(float((tgt.fpr / BUDGET).median()), 4),
        "median_recall": round(float(tgt.recall.median()), 4),
        "median_auroc": round(float(tgt.test_auroc.median()), 4),
        "min_recall": round(float(tgt.recall.min()), 4),
        "min_recall_language": str(tgt.loc[tgt.recall.idxmin(), "language"]),
        "strongest_target_language": str(strongest.language),
        "strongest_target_auroc": round(float(strongest.test_auroc), 4),
        "strongest_target_violates": bool(strongest.violates),
        "strongest_target_multiple": round(float(strongest.fpr / BUDGET), 4),
        "auroc_ranking": [
            {"rank": int(r.rank_auroc), "language": str(r.language),
             "auroc": round(float(r.test_auroc), 4),
             "multiple": round(float(r.fpr / BUDGET), 4), "violates": bool(r.violates)}
            for r in tgt.itertuples()
        ],
        "violator_auroc_ranks": viol_ranks,
        "best_violator_rank": min(viol_ranks) if viol_ranks else None,
        # "the three strongest" is only true if ranks 1-3 all violate.
        "top3_all_violate": bool(set(viol_ranks) >= {1, 2, 3}) if viol_ranks else False,
        # The supportable form: how many of the top n are violators.
        "violators_within_top4": sum(r <= 4 for r in viol_ranks),
        "violators_within_top4_languages": [
            str(x.language) for x in tgt.itertuples() if x.violates and x.rank_auroc <= 4],
    }


def recall_claim(d: pd.DataFrame) -> dict:
    """Support for the 'harmful content passed' claim, per language rather than in aggregate."""
    tgt = d[~d.is_source].copy()
    under = tgt[tgt.fpr <= BUDGET]        # within budget: the under-firing languages
    missed = 1.0 - tgt.recall
    return {
        "n_target_languages": int(len(tgt)),
        "n_under_firing": int(len(under)),
        "median_recall_all_targets": round(float(tgt.recall.median()), 4),
        "median_missed_fraction_all_targets": round(float(missed.median()), 4),
        "median_recall_under_firing": round(float(under.recall.median()), 4),
        "median_missed_fraction_under_firing": round(float((1 - under.recall).median()), 4),
        "n_under_firing_missing_over_99pc": int((under.recall < 0.01).sum()),
        "languages_missing_over_99pc": sorted(
            str(x) for x in under.loc[under.recall < 0.01, "language"]),
        "n_under_firing_missing_over_95pc": int((under.recall < 0.05).sum()),
        "worst_case_recall": round(float(tgt.recall.min()), 4),
        "worst_case_language": str(tgt.loc[tgt.recall.idxmin(), "language"]),
        "worst_case_missed_fraction": round(float(1 - tgt.recall.min()), 4),
        # The universally quantified form the manuscript used in two places.
        "claim_over_99pc_holds_for_every_under_firing_language":
            bool((under.recall < 0.01).all()),
    }


def main() -> int:
    for f in ARMS.values():
        if not (T / f).exists():
            print(f"missing {T / f}; run the Arm A stages first")
            return 1
    d8, d9 = load(ARMS["distilbert"]), load(ARMS["xlmr"])

    a = d8[~d8.is_source][["language", "fpr"]].rename(columns={"fpr": "fpr_distilbert"})
    b = d9[~d9.is_source][["language", "fpr"]].rename(columns={"fpr": "fpr_xlmr"})
    m = a.merge(b, on="language")
    # The cross-classifier ratio is computed from the RAW rates. Deriving it from the
    # two-decimal multiples printed in Table 4 inflates it (7.19 / 0.05 = 143.8 against a true
    # 141.0), which is the defect this field exists to prevent.
    m["ratio"] = m.fpr_xlmr / m.fpr_distilbert
    widest = m.loc[m.ratio.idxmax()]
    disagree = m[(m.fpr_distilbert > BUDGET) != (m.fpr_xlmr > BUDGET)]

    verdict = {
        "source": {k: f"results/tables/{v}" for k, v in ARMS.items()},
        "analysis_type": "re-analysis of preserved results; no new experiment",
        "budget": BUDGET,
        "distilbert": summarise(d8),
        "xlmr": summarise(d9),
        "distilbert_recall_claim": recall_claim(d8),
        "cross_classifier": {
            "n_languages": int(len(m)),
            "widest_ratio_language": str(widest.language),
            "widest_ratio_raw": round(float(widest.ratio), 1),
            "widest_ratio_from_rounded_multiples": round(
                round(float(widest.fpr_xlmr) / BUDGET, 2)
                / round(float(widest.fpr_distilbert) / BUDGET, 2), 1),
            "widest_multiple_distilbert": round(float(widest.fpr_distilbert) / BUDGET, 4),
            "widest_multiple_xlmr": round(float(widest.fpr_xlmr) / BUDGET, 4),
            "n_sign_disagreements": int(len(disagree)),
            "sign_disagreement_languages": sorted(str(x) for x in disagree.language),
        },
    }
    OUT.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")

    cc = verdict["cross_classifier"]
    print(f"widest cross-classifier ratio: {cc['widest_ratio_language']} "
          f"raw {cc['widest_ratio_raw']} vs {cc['widest_ratio_from_rounded_multiples']} "
          f"if derived from rounded multiples")
    for name in ("distilbert", "xlmr"):
        s = verdict[name]
        print(f"\n{name}: violation {s['violation_rate']}, median multiple "
              f"{s['median_over_budget_multiple']}, strongest target "
              f"{s['strongest_target_language']} (AUROC {s['strongest_target_auroc']}, "
              f"multiple {s['strongest_target_multiple']}, violates="
              f"{s['strongest_target_violates']})")
        print(f"  violator AUROC ranks: {s['violator_auroc_ranks']}; top-3 all violate: "
              f"{s['top3_all_violate']}; violators within top 4: "
              f"{s['violators_within_top4']} {s['violators_within_top4_languages']}")
    r = verdict["distilbert_recall_claim"]
    print(f"\nrecall claim: {r['n_under_firing']} under-firing languages; "
          f"{r['n_under_firing_missing_over_99pc']} miss >99% "
          f"({r['languages_missing_over_99pc']}); median missed "
          f"{r['median_missed_fraction_all_targets']}; worst {r['worst_case_language']} "
          f"{r['worst_case_missed_fraction']}")
    print(f"  '>99% in every under-firing language' holds: "
          f"{r['claim_over_99pc_holds_for_every_under_firing_language']}")
    print(f"\n-> {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
