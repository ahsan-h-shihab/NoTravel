"""Arm A: off-the-shelf deployed classifiers, scored zero-shot.

Why this module exists (EXP-008 pre-registration): every result so far comes from a logistic
head on frozen sentence embeddings. That is a controlled instrument, not a deployment
artefact, and the paper's framing is deployment. No additional Arm-B experiment can close
that gap -- more languages, budgets or prevalences all inherit the same instrument.

Difference from Arm B: **no head is trained**. The classifier is fixed and already has an
opinion; only the threshold varies. So there is no source-language training split here, and
`train_prior` is unknown -- the model's posteriors encode whatever prior its own training set
had, which we cannot observe. Prior-shift strategies are therefore reported for Arm A with
that caveat explicit rather than silently assuming 0.5.

Scores are cached per (model revision, language) with a checksum of the exact text sequence,
identically to the Arm-B embedding cache, so a stale or misaligned score vector can never be
silently reused.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from ..data.loaders import LanguageSplit, resolve

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORE_DIR = REPO_ROOT / "data" / "processed" / "classifier_scores"

MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32


def _text_digest(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()


def _toxic_index(config) -> int:
    """Locate the positive ("toxic") class index from the model's own label map.

    Never assume index 1. Public classifiers disagree about label order, and silently
    scoring the wrong class would invert every result for that model while producing
    perfectly plausible-looking numbers.
    """
    id2label = {int(k): str(v).lower() for k, v in (config.id2label or {}).items()}
    if not id2label:
        raise ValueError("model config exposes no id2label; cannot identify the positive class")

    positive_markers = ("toxic", "offensive", "hate", "label_1", "positive")
    negative_markers = ("non", "neutral", "clean", "not")

    for idx, name in sorted(id2label.items()):
        if any(m in name for m in positive_markers) and not any(
            name.startswith(n) for n in negative_markers
        ):
            return idx

    if len(id2label) == 2:
        # Binary head with uninformative names: fall back to the higher index, which is the
        # near-universal convention, but make the assumption visible to the caller.
        raise ValueError(
            f"cannot identify the positive class from id2label={id2label}; "
            f"resolve explicitly rather than guessing")
    raise ValueError(f"cannot identify the positive class from id2label={id2label}")


def _activation_for(config) -> str:
    """Decide between softmax and sigmoid from the model's own head configuration.

    THIS IS LOAD-BEARING. Public toxicity classifiers use both conventions:

      * multi-class softmax heads (num_labels = 2, e.g. {non-toxic, toxic});
      * single-logit or multi-label SIGMOID heads (e.g. Detoxify exposes
        num_labels = 1 with id2label = {0: 'toxic'}).

    Applying softmax to a single-logit head returns exactly 1.0 for EVERY input, which would
    silently produce a degenerate score vector and a meaningless experiment while raising no
    error at all. The activation is therefore derived from the config, never assumed.
    """
    if getattr(config, "problem_type", None) == "multi_label_classification":
        return "sigmoid"
    if int(getattr(config, "num_labels", 2)) == 1:
        return "sigmoid"
    return "softmax"


def score_language(model, tokenizer, toxic_idx: int, texts: list[str],
                   activation: str = "softmax") -> np.ndarray:
    """Return p(toxic) for each text, using the activation the model's head actually implies."""
    import torch

    out = np.empty(len(texts), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = tokenizer(texts[i:i + BATCH_SIZE], padding=True, truncation=True,
                              max_length=MAX_SEQ_LENGTH, return_tensors="pt")
            logits = model(**batch).logits
            probs = (torch.sigmoid(logits) if activation == "sigmoid"
                     else torch.softmax(logits, dim=-1))
            out[i:i + probs.shape[0]] = probs[:, toxic_idx].numpy()
    return out


def score_corpus(corpus: dict[str, LanguageSplit], model_key: str,
                 resources: dict | None = None, use_cache: bool = True,
                 verbose: bool = True) -> tuple[dict[str, np.ndarray], dict]:
    """Score every language with a pinned off-the-shelf classifier."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(6)
    spec = resolve("models", model_key, resources)

    # A resource may declare `local_dir` when the hub download path is unreliable for it.
    # The files there were fetched from the SAME pinned revision by
    # scripts/fetch_model_direct.py, with per-file SHA-256 recorded, so pinning is unchanged.
    local_dir = spec.get("local_dir")
    if local_dir:
        path = REPO_ROOT / local_dir
        if not path.exists():
            raise FileNotFoundError(
                f"{model_key} declares local_dir={local_dir} but it is absent. Fetch it with "
                f"scripts/fetch_model_direct.py --repo {spec['hf_id']} "
                f"--revision {spec['revision']} --out {local_dir}")
        source, kwargs = str(path), {}
    else:
        source, kwargs = spec["hf_id"], {"revision": spec["revision"]}

    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(source, **kwargs)
    model.eval()
    toxic_idx = _toxic_index(model.config)
    activation = _activation_for(model.config)

    if verbose:
        print(f"  [{model_key}] id2label={dict(model.config.id2label)} "
              f"num_labels={model.config.num_labels} "
              f"-> positive index {toxic_idx}, activation {activation}")

    base = SCORE_DIR / f"{model_key}@{spec['revision'][:12]}"
    scores: dict[str, np.ndarray] = {}
    timing: dict[str, dict] = {}

    for lang in sorted(corpus):
        texts = corpus[lang].texts
        digest = _text_digest(texts)
        sc_path, meta_path = base / f"{lang}.npy", base / f"{lang}.meta.npy"

        if use_cache and sc_path.exists() and meta_path.exists():
            cached = np.load(meta_path, allow_pickle=True).item()
            if str(cached.get("text_sha256")) == digest:
                scores[lang] = np.load(sc_path)
                timing[lang] = {"cached": True, "seconds": 0.0}
                if verbose:
                    print(f"  [{model_key}] {lang:4s} n={len(texts):5d}  cache")
                continue

        t0 = time.perf_counter()
        s = score_language(model, tokenizer, toxic_idx, texts, activation)
        elapsed = time.perf_counter() - t0

        base.mkdir(parents=True, exist_ok=True)
        np.save(sc_path, s)
        np.save(meta_path, np.array({
            "text_sha256": digest, "n": len(texts), "model": spec["hf_id"],
            "revision": spec["revision"], "toxic_index": toxic_idx, "activation": activation,
            "id2label": dict(model.config.id2label), "max_seq_length": MAX_SEQ_LENGTH,
        }, dtype=object), allow_pickle=True)

        scores[lang] = s
        timing[lang] = {"cached": False, "seconds": round(elapsed, 2)}
        if verbose:
            print(f"  [{model_key}] {lang:4s} n={len(texts):5d}  {elapsed:.1f}s")

    report = {
        "model_key": model_key, "hf_id": spec["hf_id"], "revision": spec["revision"],
        "positive_class_index": toxic_idx, "activation": activation,
        "id2label": {str(k): v for k, v in model.config.id2label.items()},
        "max_seq_length": MAX_SEQ_LENGTH, "batch_size": BATCH_SIZE,
        "total_seconds": round(sum(t["seconds"] for t in timing.values()), 2),
        "per_language": timing,
    }
    return scores, report

