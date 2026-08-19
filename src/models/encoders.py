"""Arm-B feature extraction: frozen multilingual sentence encoders.

Why Arm B exists: off-the-shelf classifiers (Arm A) are externally valid but confound
threshold transfer with uneven training-language coverage. Freezing an encoder and fitting a
head on ONE source language isolates the phenomenon under controlled conditions and lets us
vary which language is the source -- something no off-the-shelf classifier permits.

Embeddings are cached to disk because they are deterministic given (pinned model revision,
pinned dataset revision, text) and are expensive relative to everything downstream. The cache
is keyed by model AND revision, so changing the pin can never silently reuse stale vectors.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from ..data.loaders import LanguageSplit, resolve
from ..utils.seeding import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[2]
EMBED_DIR = REPO_ROOT / "data" / "processed" / "embeddings"

#: Sequence length. 128 tokens covers the great majority of these short social-media style
#: texts; it is recorded here because it is a scientific parameter (it bounds how much of a
#: long document the score reflects), not just a performance knob.
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32


def _text_digest(texts: list[str]) -> str:
    """Checksum of the exact text sequence an embedding matrix was built from.

    Guards against the silent-corruption failure mode where a cached matrix is reused for a
    different (or reordered) text list, which would misalign every label.
    """
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()


def _cache_paths(model_key: str, revision: str, language: str) -> tuple[Path, Path]:
    base = EMBED_DIR / f"{model_key}@{revision[:12]}"
    return base / f"{language}.npy", base / f"{language}.meta.npy"


def load_encoder(model_key: str, resources: dict | None = None):
    """Load a pinned sentence encoder on CPU."""
    from sentence_transformers import SentenceTransformer

    spec = resolve("models", model_key, resources)
    model = SentenceTransformer(spec["hf_id"], revision=spec["revision"], device="cpu")
    model.max_seq_length = MAX_SEQ_LENGTH
    return model, spec


def _prepare(texts: list[str], spec: dict) -> list[str]:
    """Apply any model-specific input convention.

    E5 models are trained with an instruction prefix; omitting it measurably degrades their
    representations, so this is a correctness requirement rather than a stylistic choice.
    """
    if "e5" in spec["hf_id"].lower():
        return [f"query: {t}" for t in texts]
    return texts


def embed_language(model, spec: dict, model_key: str, language: str, texts: list[str],
                   use_cache: bool = True) -> tuple[np.ndarray, dict]:
    """Embed one language's texts, using (and populating) the on-disk cache."""
    emb_path, meta_path = _cache_paths(model_key, spec["revision"], language)
    digest = _text_digest(texts)

    if use_cache and emb_path.exists() and meta_path.exists():
        cached_digest = str(np.load(meta_path, allow_pickle=True).item().get("text_sha256"))
        if cached_digest == digest:
            return np.load(emb_path), {"cached": True, "seconds": 0.0}
        # A digest mismatch means the upstream texts changed. Recompute rather than trust it.

    t0 = time.perf_counter()
    emb = model.encode(
        _prepare(texts, spec),
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    elapsed = time.perf_counter() - t0

    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, emb)
    np.save(meta_path, np.array({
        "text_sha256": digest,
        "n": len(texts),
        "dim": int(emb.shape[1]),
        "model": spec["hf_id"],
        "revision": spec["revision"],
        "max_seq_length": MAX_SEQ_LENGTH,
    }, dtype=object), allow_pickle=True)

    return emb, {"cached": False, "seconds": round(elapsed, 2)}


def embed_corpus(corpus: dict[str, LanguageSplit], model_key: str,
                 resources: dict | None = None, use_cache: bool = True,
                 verbose: bool = True) -> tuple[dict[str, np.ndarray], dict]:
    """Embed every language. Returns (embeddings by language, timing report)."""
    seed_everything()
    model, spec = load_encoder(model_key, resources)

    embeddings: dict[str, np.ndarray] = {}
    timing: dict[str, dict] = {}
    for lang in sorted(corpus):
        emb, info = embed_language(model, spec, model_key, lang, corpus[lang].texts,
                                   use_cache=use_cache)
        if emb.shape[0] != len(corpus[lang].texts):
            raise ValueError(
                f"Embedding/text count mismatch for {lang}: "
                f"{emb.shape[0]} vs {len(corpus[lang].texts)}"
            )
        embeddings[lang] = emb
        timing[lang] = info
        if verbose:
            tag = "cache" if info["cached"] else f"{info['seconds']:.1f}s"
            print(f"  [{model_key}] {lang:4s} n={emb.shape[0]:5d} dim={emb.shape[1]:4d}  {tag}")

    report = {
        "model_key": model_key,
        "hf_id": spec["hf_id"],
        "revision": spec["revision"],
        "max_seq_length": MAX_SEQ_LENGTH,
        "batch_size": BATCH_SIZE,
        "embedding_dim": int(next(iter(embeddings.values())).shape[1]),
        "total_seconds": round(sum(t["seconds"] for t in timing.values()), 2),
        "per_language": timing,
    }
    return embeddings, report
