"""Main experiment driver: the cross-lingual threshold-transfer study.

One script serves every experiment in families F1-F5; the experiment
identity comes from the configuration, not from a separate copy of the code. That is
deliberate -- forked scripts are how analysis pipelines silently diverge from each other.

Examples
--------
    # F1/F2/F3: full strategy grid on the primary toxicity corpus
    python scripts/run_threshold_study.py --exp-id EXP-002 --task toxicity --encoder encoder_minilm

    # F5: is the effect an artefact of English being the source?
    python scripts/run_threshold_study.py --exp-id EXP-005 --task toxicity --source de

    # Parallel-corpus control (removes the content confound, risk R-16)
    python scripts/run_threshold_study.py --exp-id EXP-006 --task sib200 --source eng_Latn
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

from src.analysis.threshold_study import (  # noqa: E402
    DEFAULT_K_VALUES, LanguageScores, run_threshold_study,
)
from src.data.loaders import (  # noqa: E402
    load_resources, load_sib200_corpus, load_toxicity_corpus, persist_split_manifest,
)
from src.eval.pipeline import fit_and_score, resample_to_prevalence  # noqa: E402
from src.models.encoders import embed_corpus  # noqa: E402
from src.utils.provenance import RunRecord  # noqa: E402
from src.utils.seeding import DEFAULT_SEED, derived_seed, seed_everything  # noqa: E402

#: Pre-registered gate, identical to EXP-001: below this the head has not learned the task
#: and nothing downstream is interpretable.
SOURCE_DEV_AUROC_GATE = 0.75


def _wilson_interval(successes: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion.

    Used for the budget-violation rate, which is a proportion over languages. Wilson rather
    than the normal approximation because the rate is frequently at or near 0 or 1, where the
    normal interval is badly wrong (and can extend outside [0,1]).
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_task(task: str, args, resources: dict):
    if task == "toxicity":
        return load_toxicity_corpus(languages=args.languages, seed=args.seed,
                                    resources=resources)
    if task == "sib200":
        languages = args.languages
        if languages is None and not args.sib200_all_configs:
            # Default to the AUDITED, row-aligned configs only. SIB-200's scientific role
            # here is as a PARALLEL corpus, and that guarantee holds only for configs whose
            # label vector matches the reference language. Silently including misaligned
            # configs would compare different sentences across languages while appearing
            # healthy -- which is precisely the confound this arm exists to eliminate.
            audit_path = REPO_ROOT / "data" / "processed" / "sib200_audit.json"
            if not audit_path.exists():
                raise FileNotFoundError(
                    f"{audit_path} not found. Run scripts/audit_sib200.py first, or pass "
                    f"--sib200-all-configs to deliberately include unaligned configs.")
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            languages = audit["parallel_aligned_configs"]
            print(f"[{args.exp_id}] using {len(languages)} audited parallel-aligned configs "
                  f"({len(audit['excluded_configs'])} excluded: "
                  f"{audit['excluded_configs'][:6]}...)")
        return load_sib200_corpus(languages=languages, positive_label=args.positive_label,
                                  seed=args.seed, resources=resources)
    raise ValueError(f"Unknown task family: {task}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--task", default="toxicity", choices=["toxicity", "sib200"])
    ap.add_argument("--encoder", default="encoder_minilm")
    ap.add_argument("--source", default=None, help="Source language; defaults per task.")
    ap.add_argument("--objective", default="f1",
                    choices=["f1", "youden", "balanced_accuracy", "target_fpr"])
    ap.add_argument("--target-fpr", nargs="*", type=float, default=None,
                    help="One or more false-positive budgets. Multiple values sweep the "
                         "budget (reviewer sim #1, R2-2: a single budget is arbitrary and a "
                         "budget-dependent effect must be reported as such).")
    ap.add_argument("--positive-label", default="health", help="SIB-200 one-vs-rest positive topic.")
    ap.add_argument("--languages", nargs="*", default=None)
    ap.add_argument("--sib200-all-configs", action="store_true",
                    help="Include SIB-200 configs that failed the row-alignment audit. "
                         "Breaks the parallel-corpus guarantee; use only deliberately.")
    ap.add_argument("--k-values", nargs="*", type=int, default=list(DEFAULT_K_VALUES))
    ap.add_argument("--few-label-repeats", type=int, default=25)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--prevalence", nargs="*", type=float, default=None,
                    help="If given, repeat the whole study at each target prevalence (R-13).")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    source = args.source or ("en" if args.task == "toxicity" else "eng_Latn")
    seed_everything(args.seed)
    resources = load_resources()

    run_dir = REPO_ROOT / "experiments" / "runs" / args.exp_id
    run_dir.mkdir(parents=True, exist_ok=True)

    ds_key = "toxicity_multilingual" if args.task == "toxicity" else "sib200"
    ds_spec = resources["datasets"][ds_key]
    enc_spec = resources["models"][args.encoder]

    rec = RunRecord(
        args.exp_id,
        objective=(f"Measure the operating-point cost of reusing a {source}-tuned decision "
                   f"threshold across languages, and how much of it label-free and few-label "
                   f"adaptation recover ({args.task}, {args.encoder})."),
        hypothesis="H1/H2/H3",
        run_dir=run_dir,
    )
    rec.set(
        dataset=ds_spec["hf_id"], dataset_version=ds_spec["revision"],
        model=enc_spec["hf_id"], model_version=enc_spec["revision"],
        random_seed=args.seed, config=vars(args) | {"resolved_source": source},
    )

    print(f"[{args.exp_id}] task={args.task} encoder={args.encoder} source={source}")
    corpus = load_task(args.task, args, resources)
    manifest = persist_split_manifest(corpus, seed=args.seed,
                                      name=f"split_manifest_{args.task}")
    print(f"[{args.exp_id}] {len(corpus)} languages")

    embeddings, embed_report = embed_corpus(corpus, args.encoder, resources=resources,
                                            verbose=False)
    (run_dir / "embedding_report.json").write_text(
        json.dumps(embed_report, indent=2) + "\n", encoding="utf-8")

    scored = fit_and_score(corpus, embeddings, source_language=source, seed=args.seed)
    print(f"[{args.exp_id}] head C={scored.head.C} "
          f"source dev AUROC={scored.head.source_dev_auroc:.4f}")

    if scored.head.source_dev_auroc < SOURCE_DEV_AUROC_GATE:
        rec.finish(status="FAILED", interpretation=(
            f"PRE-REGISTERED FAILURE: source dev AUROC {scored.head.source_dev_auroc:.4f} < "
            f"{SOURCE_DEV_AUROC_GATE}. Instrument failure, not hypothesis failure."))
        print("PRE-REGISTERED FAILURE: head did not learn the source task.")
        return 1

    prevalences = args.prevalence or [None]
    budgets = args.target_fpr if args.target_fpr else [None]
    frames = []
    for prev in prevalences:
        if prev is None:
            scores = scored.scores
            tag = "native"
        else:
            rng = np.random.default_rng(derived_seed(args.seed, "prev", f"{prev:.4f}"))
            scores = {lang: resample_to_prevalence(ls, prev, rng)
                      for lang, ls in scored.scores.items()}
            tag = f"{prev:.3f}"
            print(f"[{args.exp_id}] prevalence sweep -> {tag}")

        for budget in budgets:
            df = run_threshold_study(
                scores, source_language=source, objective=args.objective,
                target_fpr=budget, k_values=tuple(args.k_values),
                n_few_label_repeats=args.few_label_repeats, n_bootstrap=args.bootstrap,
                seed=args.seed, train_prior=scored.head.train_prior,
                verbose=(prev is None and budget == budgets[0]),
            )
            df["prevalence_setting"] = tag
            df["fpr_budget"] = budget if budget is not None else float("nan")
            frames.append(df)
            if len(budgets) > 1:
                print(f"[{args.exp_id}]   budget {budget}")

    out = pd.concat(frames, ignore_index=True)
    out["task"] = args.task
    out["encoder"] = args.encoder
    out["exp_id"] = args.exp_id

    table_path = REPO_ROOT / "results" / "tables" / f"{args.exp_id}_threshold_study.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(table_path, index=False)

    scores_path = run_dir / "per_example_scores.parquet"
    pd.concat([
        pd.DataFrame({"language": lang, "split": part, "score": s, "label": y})
        for lang, ls in scored.scores.items()
        for part, s, y in (("dev", ls.dev_scores, ls.dev_labels),
                           ("test", ls.test_scores, ls.test_labels))
    ], ignore_index=True).to_parquet(scores_path, index=False)

    # ---- headline summary -----------------------------------------------------------
    native = out[(out["prevalence_setting"] == ("native" if not args.prevalence
                                                else f"{args.prevalence[0]:.3f}"))]
    tgt = native[~native["is_source"]]
    status_quo = tgt[tgt["strategy"] == "source_global"]

    summary = {
        "exp_id": args.exp_id, "task": args.task, "encoder": args.encoder,
        "source_language": source, "objective": args.objective,
        "target_fpr": args.target_fpr,
        "n_languages": int(out["language"].nunique()),
        "source_dev_auroc": round(scored.head.source_dev_auroc, 4),
        "head_train_prior": round(scored.head.train_prior, 4),
        "tau_source": round(float(out["tau_source"].iloc[0]), 4),
        "median_test_auroc": round(float(tgt["test_auroc"].median()), 4),
        "n_degenerate_rows": int(out["is_degenerate"].sum()),
        "degenerate_by_strategy": {
            s: int(g["is_degenerate"].sum()) for s, g in out.groupby("strategy")
            if int(g["is_degenerate"].sum()) > 0
        },
    }

    # Report in the OBJECTIVE'S OWN TERMS. Reporting an F1 shortfall for an FPR-budget
    # objective is meaningless -- strategies that blow the budget trivially "win" on F1 --
    # and an earlier version of this script did exactly that, producing a misleading table.
    if args.objective == "target_fpr" and args.target_fpr:
        budget = args.target_fpr[0]
        tgt = tgt[tgt["fpr_budget"] == budget] if "fpr_budget" in tgt else tgt
        summary["headline_metric"] = f"FPR-budget compliance and recall at FPR<={budget}"
        summary["all_budgets"] = args.target_fpr
        summary["by_strategy"] = {}
        for s, g in tgt.groupby("strategy"):
            n = len(g)
            k = int((g["fpr"] > budget).sum())
            lo, hi = _wilson_interval(k, n)
            summary["by_strategy"][s] = {
                "budget_violation_rate": round(k / n, 4) if n else None,
                # Wilson interval on the violation PROPORTION (reviewer sim #1, R2-10).
                # The headline quantity is a proportion over languages and previously
                # carried no uncertainty at all.
                "violation_rate_ci95": [round(lo, 4), round(hi, 4)],
                "median_fpr": round(float(g["fpr"].median()), 4),
                "max_fpr": round(float(g["fpr"].max()), 4),
                "median_recall": round(float(g["recall"].median()), 4),
                "n_degenerate": int(g["is_degenerate"].sum()),
                "n_languages": n,
            }
        print(f"\n--- FPR budget = {budget}; target languages only ---")
        print(f"  {'strategy':24s} {'viol.rate':>9s} {'95% CI':>15s} {'medFPR':>7s} "
              f"{'maxFPR':>7s} {'medRec':>7s} {'degen':>6s}")
        for s, v in sorted(summary["by_strategy"].items(),
                           key=lambda kv: kv[1]["budget_violation_rate"]):
            ci = f"[{v['violation_rate_ci95'][0]:.2f},{v['violation_rate_ci95'][1]:.2f}]"
            print(f"  {s:24s} {v['budget_violation_rate']:9.3f} {ci:>15s} "
                  f"{v['median_fpr']:7.3f} {v['max_fpr']:7.3f} "
                  f"{v['median_recall']:7.3f} {v['n_degenerate']:6d}")
    else:
        summary["headline_metric"] = "F1 shortfall vs per-language optimum"
        summary["median_f1_gap_status_quo"] = round(float(status_quo["f1_gap_to_full"].median()), 4)
        summary["mean_f1_gap_status_quo"] = round(float(status_quo["f1_gap_to_full"].mean()), 4)
        summary["max_f1_gap_status_quo"] = round(float(status_quo["f1_gap_to_full"].max()), 4)
        summary["by_strategy"] = {
            s: {"mean_f1_gap": round(float(g["f1_gap_to_full"].mean()), 4),
                "median_f1_gap": round(float(g["f1_gap_to_full"].median()), 4),
                "n_degenerate": int(g["is_degenerate"].sum())}
            for s, g in tgt.groupby("strategy")
        }
        print("\n--- mean F1 shortfall vs per-language optimum (lower is better) ---")
        for s, v in sorted(summary["by_strategy"].items(), key=lambda kv: kv[1]["mean_f1_gap"]):
            print(f"  {s:28s} {v['mean_f1_gap']:+.4f}   degenerate={v['n_degenerate']}")
        print(f"\nStatus-quo (source_global) median F1 shortfall: "
              f"{summary['median_f1_gap_status_quo']:+.4f}")

    if summary["degenerate_by_strategy"]:
        print(f"\n⚠ degenerate thresholds: {summary['degenerate_by_strategy']}")

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for p in (table_path, scores_path, run_dir / "summary.json",
              run_dir / "embedding_report.json", manifest):
        rec.add_output(p)
    if args.objective == "target_fpr" and args.target_fpr is not None:
        sg = summary["by_strategy"].get("source_global", {})
        interpretation = (
            f"{summary['n_languages']} languages, FPR budget {args.target_fpr}. "
            f"Source-tuned global threshold violates the budget in "
            f"{sg.get('budget_violation_rate', float('nan')):.1%} of target languages "
            f"(median FPR {sg.get('median_fpr')}, max {sg.get('max_fpr')}).")
    else:
        interpretation = (
            f"{summary['n_languages']} languages. Status-quo source-tuned global threshold "
            f"costs a median {summary['median_f1_gap_status_quo']:.4f} F1 relative to "
            f"per-language calibration; max {summary['max_f1_gap_status_quo']:.4f}. "
            f"{summary['n_degenerate_rows']} degenerate thresholds observed.")
    rec.finish(status="COMPLETE", interpretation=interpretation)
    print(f"\nWrote {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
