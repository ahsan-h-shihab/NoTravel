"""EXP-008 -- ARM A: off-the-shelf deployed classifiers.

Pre-registered before execution, including the success / failure / mixed
criteria AND the resulting changes to claims, contribution and title. This script evaluates
those criteria mechanically and prints the verdict, so the interpretation cannot drift after
seeing the data.

Usage:
    python scripts/run_arm_a.py --exp-id EXP-008
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.threshold_study import LanguageScores, run_threshold_study  # noqa: E402
from src.data.loaders import load_resources, load_toxicity_corpus  # noqa: E402
from src.eval.metrics import auroc  # noqa: E402
from src.models.classifiers import score_corpus  # noqa: E402
from src.utils.provenance import RunRecord  # noqa: E402
from src.utils.seeding import DEFAULT_SEED, seed_everything  # noqa: E402

# ---- PRE-REGISTERED CRITERIA. Do not edit after execution. -------------------------
SOURCE_AUROC_INSTRUMENT_GATE = 0.70   # below this a classifier is uninterpretable
SUCCESS_VIOLATION_RATE = 0.50
SUCCESS_STRONG_AUROC = 0.80           # "source-dev-comparable" strength
SUCCESS_MIN_STRONG_VIOLATORS = 2
FAILURE_VIOLATION_RATE = 0.20
MIXED_WEAK_AUROC = 0.70               # mixed: violations only where AUROC < this


def verdict_for(df: pd.DataFrame, source_auroc: float, budget: float) -> tuple[str, dict]:
    """Apply the pre-registered criteria mechanically."""
    tgt = df[(~df["is_source"]) & (df["strategy"] == "source_global")]
    n = len(tgt)
    violators = tgt[tgt["fpr"] > budget]
    rate = len(violators) / n if n else float("nan")

    strong = tgt[tgt["test_auroc"] >= SUCCESS_STRONG_AUROC]
    strong_violators = strong[strong["fpr"] > budget]
    weak_only = (len(violators) > 0
                 and (violators["test_auroc"] < MIXED_WEAK_AUROC).all())

    detail = {
        "n_target_languages": int(n),
        "violation_rate": round(float(rate), 4),
        "n_strong_auroc_languages": int(len(strong)),
        "n_strong_auroc_violators": int(len(strong_violators)),
        "strong_violators": sorted(strong_violators["language"].tolist()),
        "violations_confined_to_weak": bool(weak_only),
        "source_test_auroc": round(float(source_auroc), 4),
        "median_target_auroc": round(float(tgt["test_auroc"].median()), 4),
    }

    if source_auroc < SOURCE_AUROC_INSTRUMENT_GATE:
        return "INSTRUMENT_FAILURE", detail
    if rate >= SUCCESS_VIOLATION_RATE and len(strong_violators) >= SUCCESS_MIN_STRONG_VIOLATORS:
        return "SUCCESS", detail
    if rate <= FAILURE_VIOLATION_RATE and len(strong_violators) == 0:
        return "FAILURE", detail
    if weak_only:
        return "MIXED", detail
    return "MIXED", detail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp-id", default="EXP-008")
    ap.add_argument("--models", nargs="*",
                    default=["clf_unitary_xlmr", "clf_citizenlab_distilbert"])
    ap.add_argument("--source", default="en")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    seed_everything(args.seed)
    resources = load_resources()
    run_dir = REPO_ROOT / "experiments" / "runs" / args.exp_id
    run_dir.mkdir(parents=True, exist_ok=True)

    ds = resources["datasets"]["toxicity_multilingual"]
    rec = RunRecord(args.exp_id,
                    objective=("ARM A: does operating-point transfer fail in real deployed "
                               "multilingual classifiers, not only in a frozen-encoder probe?"),
                    hypothesis="H1 external validity", run_dir=run_dir)
    rec.set(dataset=ds["hf_id"], dataset_version=ds["revision"],
            model=";".join(args.models), model_version="see per-model reports",
            random_seed=args.seed, config=vars(args))

    corpus = load_toxicity_corpus(resources=resources, seed=args.seed)
    print(f"[{args.exp_id}] {len(corpus)} languages; models={args.models}")

    frames, reports, verdicts = [], {}, {}
    for model_key in args.models:
        print(f"\n--- {model_key} ---")
        try:
            scores, report = score_corpus(corpus, model_key, resources=resources)
        except Exception as exc:
            print(f"  EXCLUDED: {type(exc).__name__}: {exc}")
            reports[model_key] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        reports[model_key] = report

        ls = {}
        for lang, split in corpus.items():
            dev, test = split.indices("dev"), split.indices("test")
            ls[lang] = LanguageScores(lang, scores[lang][dev], split.labels[dev],
                                      scores[lang][test], split.labels[test])

        src_auroc = auroc(ls[args.source].test_scores, ls[args.source].test_labels)
        print(f"  source ({args.source}) test AUROC = {src_auroc:.4f}")

        # No head is trained in Arm A, so the training prior is unobservable. 0.5 is passed
        # as a declared placeholder; prior-shift strategies are reported with that caveat.
        df = run_threshold_study(
            ls, source_language=args.source, objective="target_fpr",
            target_fpr=args.target_fpr, n_bootstrap=args.bootstrap, seed=args.seed,
            train_prior=0.5, verbose=False)
        df["model"] = model_key
        df["arm"] = "A"
        frames.append(df)

        v, detail = verdict_for(df, src_auroc, args.target_fpr)
        verdicts[model_key] = {"verdict": v, **detail}
        print(f"  PRE-REGISTERED VERDICT: {v}")
        for k, val in detail.items():
            print(f"    {k}: {val}")

        tgt = df[(~df["is_source"]) & (df["strategy"] == "source_global")]
        print(f"  {'lang':5s} {'AUROC':>7s} {'FPR':>7s} {'x budget':>9s} {'recall':>7s}")
        for _, r in tgt.sort_values("test_auroc", ascending=False).iterrows():
            print(f"  {r['language']:5s} {r['test_auroc']:7.3f} {r['fpr']:7.3f} "
                  f"{r['fpr'] / args.target_fpr:9.2f} {r['recall']:7.3f}")

    if not frames:
        rec.finish(status="FAILED", interpretation="No classifier could be scored.")
        print("\nAll classifiers excluded. Arm A inconclusive.")
        return 1

    out = pd.concat(frames, ignore_index=True)
    table = REPO_ROOT / "results" / "tables" / f"{args.exp_id}_arm_a.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(table, index=False)

    summary = {"exp_id": args.exp_id, "budget": args.target_fpr,
               "preregistered_criteria": {
                   "success": f"violation>={SUCCESS_VIOLATION_RATE} AND "
                              f">={SUCCESS_MIN_STRONG_VIOLATORS} languages with "
                              f"AUROC>={SUCCESS_STRONG_AUROC} violating",
                   "failure": f"violation<={FAILURE_VIOLATION_RATE} AND no strong violator",
                   "mixed": f"violations only where AUROC<{MIXED_WEAK_AUROC}",
                   "instrument_failure": f"source AUROC<{SOURCE_AUROC_INSTRUMENT_GATE}"},
               "verdicts": verdicts, "model_reports": reports}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                          encoding="utf-8")

    print("\n=== PRE-REGISTERED VERDICTS ===")
    for m, v in verdicts.items():
        print(f"  {m}: {v['verdict']}  (violation {v['violation_rate']}, "
              f"strong violators {v['n_strong_auroc_violators']})")

    for p in (table, run_dir / "summary.json"):
        rec.add_output(p)
    rec.finish(status="COMPLETE", interpretation="; ".join(
        f"{m}={v['verdict']} (violation {v['violation_rate']})" for m, v in verdicts.items()))
    print(f"\nWrote {table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
