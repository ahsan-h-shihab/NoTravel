"""PHASE-1 feasibility probe: measure real CPU inference cost before committing.

Why this exists: COMPUTE BUDGET makes CPU-only a permanent constraint, and
the risk register requires that feasibility be MEASURED, not assumed, before the project
commits to a design. This script produces the measurement that closes (or opens) that risk.

It deliberately measures the two candidate inference arms separately:

  Arm A  off-the-shelf public toxicity classifier, zero-shot  (externally valid: what
         deployers actually run)
  Arm B  frozen multilingual sentence encoder, embeddings only (controlled: a linear head
         is fitted afterwards in seconds, so encoding dominates cost)

Output: a JSON record under experiments/logs/ with per-model throughput and an
extrapolation to the full corpus.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Full corpus size, measured from the dataset on 2026-08-02 (15 language splits).
FULL_CORPUS_N = 71_374


def probe_classifier(model_id: str, revision: str, texts: list[str], batch_size: int, max_len: int) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(6)
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())

    # Warm-up: first batch pays lazy-init and allocator costs that are not representative.
    with torch.no_grad():
        warm = tok(texts[:batch_size], padding=True, truncation=True,
                   max_length=max_len, return_tensors="pt")
        model(**warm)

    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = tok(texts[i:i + batch_size], padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
            model(**batch)
    elapsed = time.perf_counter() - t0

    per_s = len(texts) / elapsed
    return {
        "model_id": model_id,
        "revision": revision,
        "arm": "A_classifier",
        "n_params_millions": round(n_params / 1e6, 1),
        "n_texts": len(texts),
        "batch_size": batch_size,
        "max_length": max_len,
        "elapsed_s": round(elapsed, 2),
        "texts_per_s": round(per_s, 2),
        "full_corpus_minutes": round(FULL_CORPUS_N / per_s / 60, 1),
        "num_labels": int(model.config.num_labels),
        "id2label": {str(k): v for k, v in model.config.id2label.items()},
    }


def probe_encoder(model_id: str, revision: str, texts: list[str], batch_size: int, max_len: int) -> dict:
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(6)
    model = SentenceTransformer(model_id, revision=revision, device="cpu")
    model.max_seq_length = max_len
    n_params = sum(p.numel() for p in model.parameters())

    model.encode(texts[:batch_size], batch_size=batch_size, show_progress_bar=False)

    t0 = time.perf_counter()
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                       convert_to_numpy=True, normalize_embeddings=True)
    elapsed = time.perf_counter() - t0

    per_s = len(texts) / elapsed
    return {
        "model_id": model_id,
        "revision": revision,
        "arm": "B_encoder",
        "n_params_millions": round(n_params / 1e6, 1),
        "n_texts": len(texts),
        "batch_size": batch_size,
        "max_length": max_len,
        "embedding_dim": int(emb.shape[1]),
        "elapsed_s": round(elapsed, 2),
        "texts_per_s": round(per_s, 2),
        "full_corpus_minutes": round(FULL_CORPUS_N / per_s / 60, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=256, help="Number of texts to time.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "experiments" / "logs" / "phase1_feasibility_probe.json")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("textdetox/multilingual_toxicity_dataset")
    # Sample across a high-resource, a non-Latin-script and a low-resource split so the
    # timing is not dominated by one tokenizer's behaviour.
    texts: list[str] = []
    for split in ("en", "ru", "am"):
        texts.extend(ds[split]["text"][: args.n // 3])
    texts = texts[: args.n]

    results = []
    # Revision SHAs retrieved from the HuggingFace API on 2026-08-02 and verified to
    # resolve. They are NEVER written from memory: a fabricated revision would poison
    # every downstream provenance record.
    classifiers = [
        ("unitary/multilingual-toxic-xlm-roberta",
         "4ad6f5c104d9ce813a1a2f33cac0c5b579ef6ee5"),
        ("citizenlab/distilbert-base-multilingual-cased-toxicity",
         "b4532a8b095d1886a7b5dff818331ecc88a855ae"),
        ("textdetox/xlmr-large-toxicity-classifier",
         "b9c7c563427c591fc318d91eb592381ae2fbde66"),
    ]
    encoders = [
        ("intfloat/multilingual-e5-small",
         "614241f622f53c4eeff9890bdc4f31cfecc418b3"),
        ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
         "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"),
    ]

    for mid, rev in classifiers:
        try:
            results.append(probe_classifier(mid, rev, texts, args.batch_size, args.max_length))
            print(f"OK  {mid}: {results[-1]['texts_per_s']} texts/s "
                  f"-> {results[-1]['full_corpus_minutes']} min for full corpus")
        except Exception as exc:
            # A pinned revision that no longer resolves is itself a finding (R-03/R-05).
            results.append({"model_id": mid, "revision": rev, "arm": "A_classifier",
                            "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAIL {mid}: {type(exc).__name__}: {exc}")

    for mid, rev in encoders:
        try:
            results.append(probe_encoder(mid, rev, texts, args.batch_size, args.max_length))
            print(f"OK  {mid}: {results[-1]['texts_per_s']} texts/s "
                  f"-> {results[-1]['full_corpus_minutes']} min for full corpus")
        except Exception as exc:
            results.append({"model_id": mid, "revision": rev, "arm": "B_encoder",
                            "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAIL {mid}: {type(exc).__name__}: {exc}")

    record = {
        "probe_run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "PHASE-1 CPU feasibility measurement closing risk R-02",
        "full_corpus_n": FULL_CORPUS_N,
        "settings": {"n_timed": len(texts), "batch_size": args.batch_size,
                     "max_length": args.max_length, "torch_threads": 6},
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
