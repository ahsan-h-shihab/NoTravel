"""EXP-001 -- Pilot: does cross-lingual threshold divergence exist?

Pre-registered BEFORE execution. This script must not be edited to change
the pre-registered success/failure criteria after seeing results.

Pipeline
--------
1. Load the pinned 15-language corpus with deterministic seeded splits.
2. Embed every language with a frozen multilingual encoder (cached).
3. Fit a logistic head on the SOURCE language's train split only; select C on source dev.
4. Score every language's dev and test splits.
5. Choose thresholds on DEV only:
      tau_source  -- tuned on source dev            (the status quo under test)
      tau_lang    -- tuned on that language's dev    (achievable per-language calibration)
6. Evaluate both on TEST only, alongside threshold-free AUROC as the control.
7. Preserve raw per-example scores, the metric table, and a full provenance record.
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

from src.data.loaders import load_resources, load_toxicity_corpus, persist_split_manifest  # noqa: E402
from src.eval.metrics import auroc, expected_calibration_error, operating_point  # noqa: E402
from src.eval.thresholds import tune_threshold  # noqa: E402
from src.models.encoders import embed_corpus  # noqa: E402
from src.models.heads import train_head  # noqa: E402
from src.utils.provenance import RunRecord  # noqa: E402
from src.utils.seeding import DEFAULT_SEED, seed_everything  # noqa: E402

EXP_ID = "EXP-001"
OBJECTIVE = ("Establish whether per-language optimal decision thresholds differ from a "
             "source-tuned threshold for a fixed multilingual classifier, and validate the "
             "Arm-B pipeline as a scientific instrument.")
HYPOTHESIS = "H1 (pilot scale)"

# Pre-registered gates. Changing these after seeing results is forbidden.
SOURCE_DEV_AUROC_GATE = 0.75
TARGET_AUROC_DEGENERATE = 0.55


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", default="encoder_minilm")
    ap.add_argument("--source", default="en")
    ap.add_argument("--objective", default="f1", choices=["f1", "youden", "balanced_accuracy"])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--languages", nargs="*", default=None)
    args = ap.parse_args()

    seed_everything(args.seed)
    resources = load_resources()
    run_dir = REPO_ROOT / "experiments" / "runs" / EXP_ID
    run_dir.mkdir(parents=True, exist_ok=True)

    rec = RunRecord(EXP_ID, objective=OBJECTIVE, hypothesis=HYPOTHESIS, run_dir=run_dir)

    ds_spec = resources["datasets"]["toxicity_multilingual"]
    enc_spec = resources["models"][args.encoder]
    rec.set(
        dataset=ds_spec["hf_id"], dataset_version=ds_spec["revision"],
        model=enc_spec["hf_id"], model_version=enc_spec["revision"],
        random_seed=args.seed,
        config={
            "encoder_key": args.encoder, "source_language": args.source,
            "threshold_objective": args.objective, "seed": args.seed,
            "split_fractions": {"train": 0.5, "dev": 0.2, "test": 0.3},
            "threshold_selected_on": "dev", "evaluated_on": "test",
            "source_dev_auroc_gate": SOURCE_DEV_AUROC_GATE,
            "target_auroc_degenerate": TARGET_AUROC_DEGENERATE,
        },
    )

    print(f"[{EXP_ID}] Loading corpus ...")
    corpus = load_toxicity_corpus(languages=args.languages, seed=args.seed, resources=resources)
    manifest_path = persist_split_manifest(corpus, seed=args.seed)
    print(f"[{EXP_ID}] {len(corpus)} languages; split manifest -> {manifest_path.name}")

    if args.source not in corpus:
        rec.finish(status="FAILED", interpretation=f"Source language {args.source!r} absent.")
        raise SystemExit(f"Source language {args.source!r} not in corpus")

    print(f"[{EXP_ID}] Embedding with {args.encoder} ...")
    embeddings, embed_report = embed_corpus(corpus, args.encoder, resources=resources)
    (run_dir / "embedding_report.json").write_text(
        json.dumps(embed_report, indent=2) + "\n", encoding="utf-8")

    # ---- Fit the head on the source language only -------------------------------------
    src = corpus[args.source]
    src_emb = embeddings[args.source]
    tr, dv = src.indices("train"), src.indices("dev")
    head = train_head(
        src_emb[tr], src.labels[tr], src_emb[dv], src.labels[dv],
        source_language=args.source, seed=args.seed,
    )
    print(f"[{EXP_ID}] Head: C={head.C}  n_train={head.n_train}  "
          f"source dev AUROC={head.source_dev_auroc:.4f}")

    if head.source_dev_auroc < SOURCE_DEV_AUROC_GATE:
        rec.finish(status="FAILED", interpretation=(
            f"PRE-REGISTERED FAILURE: source dev AUROC {head.source_dev_auroc:.4f} < "
            f"{SOURCE_DEV_AUROC_GATE}. The head did not learn the task, so nothing downstream "
            f"is interpretable. Reconsider the instrument, NOT the hypothesis."))
        raise SystemExit(1)

    # ---- Score every language, then choose thresholds on dev ---------------------------
    scores: dict[str, dict[str, np.ndarray]] = {}
    for lang, split in corpus.items():
        emb = embeddings[lang]
        scores[lang] = {
            part: head.score(emb[split.indices(part)]) for part in ("dev", "test")
        }

    tau_source = tune_threshold(
        scores[args.source]["dev"], corpus[args.source].labels[corpus[args.source].indices("dev")],
        objective=args.objective,
    )
    print(f"[{EXP_ID}] Source-tuned global threshold tau_source = {tau_source:.4f}")

    # ---- Per-language evaluation -------------------------------------------------------
    rows = []
    raw_rows = []
    for lang in sorted(corpus):
        split = corpus[lang]
        dev_y = split.labels[split.indices("dev")]
        test_y = split.labels[split.indices("test")]
        dev_s, test_s = scores[lang]["dev"], scores[lang]["test"]

        tau_lang = tune_threshold(dev_s, dev_y, objective=args.objective)

        op_source = operating_point(test_s, test_y, tau_source)
        op_lang = operating_point(test_s, test_y, tau_lang)
        test_auroc = auroc(test_s, test_y)

        rows.append({
            "language": lang,
            "is_source": lang == args.source,
            "n_dev": int(dev_y.size), "n_test": int(test_y.size),
            "test_positive_rate": round(float(np.mean(test_y == 1)), 4),
            "test_auroc": round(test_auroc, 4),
            "test_ece": round(expected_calibration_error(test_s, test_y), 4),
            "tau_source": round(tau_source, 4),
            "tau_language": round(tau_lang, 4),
            "tau_abs_diff": round(abs(tau_lang - tau_source), 4),
            "f1_source_threshold": round(op_source.f1, 4),
            "f1_language_threshold": round(op_lang.f1, 4),
            "f1_gap": round(op_lang.f1 - op_source.f1, 4),
            "fpr_source_threshold": round(op_source.fpr, 4),
            "fpr_language_threshold": round(op_lang.fpr, 4),
            "fnr_source_threshold": round(op_source.fnr, 4),
            "fnr_language_threshold": round(op_lang.fnr, 4),
            "precision_source_threshold": round(op_source.precision, 4),
            "recall_source_threshold": round(op_source.recall, 4),
            "ppr_source_threshold": round(op_source.predicted_positive_rate, 4),
            "ppr_language_threshold": round(op_lang.predicted_positive_rate, 4),
        })

        for part, s, y in (("dev", dev_s, dev_y), ("test", test_s, test_y)):
            raw_rows.append(pd.DataFrame({
                "language": lang, "split": part, "score": s, "label": y,
            }))

    df = pd.DataFrame(rows).sort_values("f1_gap", ascending=False).reset_index(drop=True)

    # ---- Preserve artifacts -------------------------------------------------------------
    table_path = REPO_ROOT / "results" / "tables" / f"{EXP_ID}_pilot_per_language.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_path, index=False)

    raw_path = run_dir / "per_example_scores.parquet"
    pd.concat(raw_rows, ignore_index=True).to_parquet(raw_path, index=False)

    head_path = run_dir / "head.json"
    head_path.write_text(json.dumps({
        "source_language": head.source_language, "C": head.C, "seed": head.seed,
        "n_train": head.n_train, "source_dev_auroc": head.source_dev_auroc,
        "tau_source": tau_source, "objective": args.objective,
    }, indent=2) + "\n", encoding="utf-8")

    # ---- Pre-registered gate evaluation --------------------------------------------------
    targets = df[~df["is_source"]]
    n_degenerate = int((targets["test_auroc"] <= TARGET_AUROC_DEGENERATE).sum())
    degenerate_fraction = n_degenerate / max(len(targets), 1)
    instrument_failed = degenerate_fraction > 0.5

    median_gap = float(targets["f1_gap"].median())
    max_gap = float(targets["f1_gap"].max())
    median_tau_diff = float(targets["tau_abs_diff"].median())

    summary = {
        "experiment_id": EXP_ID,
        "source_language": args.source,
        "tau_source": round(tau_source, 4),
        "source_dev_auroc": round(head.source_dev_auroc, 4),
        "n_target_languages": int(len(targets)),
        "n_degenerate_auroc": n_degenerate,
        "degenerate_fraction": round(degenerate_fraction, 4),
        "instrument_failed": instrument_failed,
        "median_f1_gap": round(median_gap, 4),
        "mean_f1_gap": round(float(targets["f1_gap"].mean()), 4),
        "max_f1_gap": round(max_gap, 4),
        "median_abs_threshold_diff": round(median_tau_diff, 4),
        "max_abs_threshold_diff": round(float(targets["tau_abs_diff"].max()), 4),
        "median_test_auroc": round(float(targets["test_auroc"].median()), 4),
        "min_test_auroc": round(float(targets["test_auroc"].min()), 4),
        "max_test_auroc": round(float(targets["test_auroc"].max()), 4),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n" + df.to_string(index=False,
                              columns=["language", "test_auroc", "tau_language", "tau_abs_diff",
                                       "f1_source_threshold", "f1_language_threshold", "f1_gap",
                                       "fpr_source_threshold", "fpr_language_threshold"]))
    print("\n--- SUMMARY ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    for path in (table_path, raw_path, head_path, run_dir / "summary.json",
                 run_dir / "embedding_report.json", manifest_path):
        rec.add_output(path)

    if instrument_failed:
        rec.finish(status="FAILED", interpretation=(
            f"PRE-REGISTERED FAILURE: {n_degenerate}/{len(targets)} target languages have "
            f"test AUROC <= {TARGET_AUROC_DEGENERATE}. The classifier does not transfer, so "
            f"there is no meaningful operating point to study with this instrument. "
            f"Reconsider the encoder/head, NOT the hypothesis."))
        print("\nRESULT: PRE-REGISTERED FAILURE CRITERION MET (instrument).")
        return 1

    rec.finish(status="COMPLETE", interpretation=(
        f"Pipeline validated (source dev AUROC {head.source_dev_auroc:.4f} >= "
        f"{SOURCE_DEV_AUROC_GATE}; {n_degenerate}/{len(targets)} degenerate languages). "
        f"Median per-language F1 gain from own-threshold calibration: {median_gap:.4f}; "
        f"max {max_gap:.4f}; median |tau_lang - tau_source| = {median_tau_diff:.4f}."))
    print("\nRESULT: pilot completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
