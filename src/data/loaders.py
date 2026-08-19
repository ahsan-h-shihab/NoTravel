"""Dataset loading with pinned revisions and deterministic splits.

Why this exists: DATASET POLICY requires public, versioned, documented,
reproducible datasets, and § REPRODUCIBILITY CONSTITUTION requires that nothing important
exist only as a temporary file. Splits are computed from a seeded permutation and persisted
to disk, so a rerun cannot silently produce different splits even if the upstream row order
changes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

from ..utils.seeding import DEFAULT_SEED, derived_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "resources.yaml"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

#: Split proportions. Rationale:
#:   train (50%) -- fits the Arm-B head, used only for the SOURCE language
#:   dev   (20%) -- ALL threshold selection happens here, for every strategy
#:   test  (30%) -- evaluation only; never touched by any threshold-selection method
#: Keeping test at 30% (~1500 examples/language) keeps bootstrap CIs on F1 usefully narrow
#: while leaving dev large enough that the "full-label" oracle is genuinely well estimated.
SPLIT_FRACTIONS = {"train": 0.5, "dev": 0.2, "test": 0.3}


@dataclass(frozen=True)
class LanguageSplit:
    """One language's texts, labels and split assignment."""

    language: str
    texts: list[str]
    labels: np.ndarray
    split: np.ndarray  # array of "train" / "dev" / "test"

    def subset(self, split_name: str) -> tuple[list[str], np.ndarray]:
        mask = self.split == split_name
        idx = np.flatnonzero(mask)
        return [self.texts[i] for i in idx], self.labels[idx]

    def indices(self, split_name: str) -> np.ndarray:
        return np.flatnonzero(self.split == split_name)


def load_resources(path: Path | None = None) -> dict:
    """Load the pinned resource registry."""
    with open(path or CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(kind: str, key: str, resources: dict | None = None) -> dict:
    """Look up a pinned resource, failing loudly if it has no revision.

    A resource without a revision is a reproducibility defect (risk R-05), so this raises
    rather than defaulting to `main` -- which would silently track upstream changes.
    """
    resources = resources or load_resources()
    try:
        entry = dict(resources[kind][key])
    except KeyError as exc:
        raise KeyError(f"No {kind} entry named {key!r} in resources.yaml") from exc
    if kind == "models" and not entry.get("revision"):
        raise ValueError(f"Resource {kind}/{key} has no pinned revision")
    entry["key"] = key
    return entry


def assign_splits(n: int, language: str, seed: int = DEFAULT_SEED,
                  fractions: dict[str, float] | None = None) -> np.ndarray:
    """Deterministically assign each row to train/dev/test.

    The permutation seed is derived from (base seed, language), so a language's split does
    not depend on how many other languages were processed or in what order.
    """
    fractions = fractions or SPLIT_FRACTIONS
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    rng = np.random.default_rng(derived_seed(seed, "split", language))
    perm = rng.permutation(n)

    n_train = int(round(n * fractions["train"]))
    n_dev = int(round(n * fractions["dev"]))
    out = np.empty(n, dtype=object)
    out[perm[:n_train]] = "train"
    out[perm[n_train:n_train + n_dev]] = "dev"
    out[perm[n_train + n_dev:]] = "test"
    return out.astype("<U5")


def load_toxicity_corpus(languages: Iterable[str] | None = None, seed: int = DEFAULT_SEED,
                         resources: dict | None = None,
                         cache_dir: Path | None = None) -> dict[str, LanguageSplit]:
    """Load the pinned multilingual toxicity corpus with deterministic splits.

    Returns a mapping language -> LanguageSplit. Each HuggingFace "split" in this dataset is
    a *language*, not a train/test partition; the train/dev/test partition is created here.
    """
    from datasets import load_dataset

    resources = resources or load_resources()
    spec = resources["datasets"]["toxicity_multilingual"]

    ds = load_dataset(
        spec["hf_id"],
        revision=spec["revision"],
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    wanted = list(languages) if languages is not None else list(spec["languages"])
    missing = [lang for lang in wanted if lang not in ds]
    if missing:
        raise KeyError(f"Languages absent from the pinned dataset revision: {missing}")

    text_col, label_col = spec["text_column"], spec["label_column"]
    out: dict[str, LanguageSplit] = {}
    for lang in wanted:
        part = ds[lang]
        texts = [str(t) for t in part[text_col]]
        labels = np.asarray(part[label_col], dtype=np.int64)
        if set(np.unique(labels).tolist()) - {0, 1}:
            raise ValueError(f"Non-binary labels in language {lang}: {np.unique(labels)}")
        out[lang] = LanguageSplit(
            language=lang, texts=texts, labels=labels,
            split=assign_splits(len(texts), lang, seed=seed),
        )
    return out


#: Canonical SIB-200 topic labels, in the ClassLabel order used by the majority schema.
SIB200_LABELS = ("entertainment", "geography", "health", "politics",
                 "science/technology", "sports", "travel")


def sib200_snapshot(resources: dict | None = None) -> Path:
    """Download the whole SIB-200 repo once and return the local path.

    WHY A SNAPSHOT RATHER THAN PER-CONFIG `load_dataset`
    ----------------------------------------------------
    SIB-200 has 205 language configs. Loading them one at a time issues hundreds of API
    calls and reliably trips HuggingFace rate limiting: a first audit attempt returned
    HTTP 429 for 79 configs, which looked like data corruption but was purely throttling --
    and Kazakh, a priority language, was among the false failures. Exponential backoff only
    converted that into an unbounded stall.

    The entire repository is ~27 MB of parquet. One snapshot download replaces every
    round-trip, is far faster, and makes the audit deterministic instead of network-dependent.
    """
    from huggingface_hub import snapshot_download

    resources = resources or load_resources()
    spec = resources["datasets"]["sib200"]
    return Path(snapshot_download(
        repo_id=spec["hf_id"], revision=spec["revision"], repo_type="dataset"))


#: SIB-200 config directories follow the FLORES-200 `iso639-3_Script` convention, e.g.
#: `kaz_Cyrl`. The repository also contains non-language directories (an aggregate `test`
#: directory with 41,820 rows was found during the audit). Matching the naming convention
#: keeps those out of the language list, rather than letting an aggregate masquerade as a
#: language and silently distort every per-language statistic.
_SIB200_LANG_RE = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")


def sib200_available_languages(snapshot: Path) -> list[str]:
    """Language config directories present in a snapshot, sorted.

    Only directories matching the `iso639-3_Script` naming convention are returned.
    """
    return sorted(p.name for p in snapshot.iterdir()
                  if p.is_dir() and _SIB200_LANG_RE.match(p.name)
                  and any(p.glob("*.parquet")))


def sib200_read_language(snapshot: Path, language: str) -> tuple[list[str], list[str], str]:
    """Read one language's parquet files. Returns (texts, string labels, schema name).

    Handles SIB-200's inconsistent columns directly rather than guessing: most configs carry
    an integer `label`, a minority carry a string `category`. Raises on anything else -- a
    silently mis-parsed label column would corrupt an entire language while looking normal.
    """
    import pyarrow.parquet as pq

    files = sorted((snapshot / language).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files for {language} in {snapshot}")

    texts: list[str] = []
    labels: list[str] = []
    schema = ""
    for f in files:
        table = pq.read_table(f)
        cols = set(table.column_names)
        if "category" in cols:
            schema = "category_string"
            raw = [str(v) for v in table.column("category").to_pylist()]
        elif "label" in cols:
            values = table.column("label").to_pylist()
            if values and isinstance(values[0], str):
                schema = "label_string"
                raw = [str(v) for v in values]
            else:
                schema = "label_int"
                raw = [SIB200_LABELS[int(v)] for v in values]
        else:
            raise ValueError(f"Unrecognised SIB-200 columns for {language}: "
                             f"{sorted(table.column_names)}")
        texts.extend(str(t) for t in table.column("text").to_pylist())
        labels.extend(raw)

    return texts, labels, schema


def _sib200_schema(split) -> tuple[str, list[str]]:
    """Identify which of SIB-200's inconsistent per-config schemas a split uses.

    Returns (schema_name, label_names). Raises rather than guessing: a silently mis-parsed
    label column would corrupt an entire language's results while looking perfectly normal.
    """
    features = split.features
    if "label" in features and hasattr(features["label"], "names"):
        return "classlabel", list(features["label"].names)
    if "category" in features:
        return "category_string", list(SIB200_LABELS)
    if "label" in features:  # plain string/int label without ClassLabel metadata
        return "label_plain", list(SIB200_LABELS)
    raise ValueError(f"Unrecognised SIB-200 schema; columns were {list(features.keys())}")


def _sib200_labels(split, schema: str) -> list[str]:
    """Extract string topic labels from a split under the identified schema."""
    if schema == "classlabel":
        names = split.features["label"].names
        return [names[int(v)] for v in split["label"]]
    if schema == "category_string":
        return [str(v) for v in split["category"]]
    if schema == "label_plain":
        values = split["label"]
        if len(values) and isinstance(values[0], str):
            return [str(v) for v in values]
        return [SIB200_LABELS[int(v)] for v in values]
    raise ValueError(f"Unknown schema: {schema}")


def load_sib200_corpus(languages: Iterable[str] | None = None, positive_label: str = "health",
                       seed: int = DEFAULT_SEED, resources: dict | None = None,
                       cache_dir: Path | None = None,
                       verbose: bool = True) -> dict[str, LanguageSplit]:
    """Load SIB-200 as a binary one-vs-rest task with deterministic splits.

    SIB-200 is derived from FLORES-200 and is therefore a PARALLEL corpus: the same sentences,
    carrying the same labels, translated into every language. That property is the scientific
    reason for including it -- it removes the content confound, so a cross-language difference
    in optimal threshold cannot be attributed to the languages' texts differing in content or
    difficulty.

    The upstream train/validation/test split is deliberately DISCARDED and re-split with this
    project's own seeded scheme, so the split protocol is identical across both task families
    rather than silently different for one of them.

    Binarization is one-vs-rest over the 7 topic labels. `positive_label` selects which topic
    counts as positive; the analysis runs all seven, since no single topic is privileged.
    """
    resources = resources or load_resources()
    if positive_label not in SIB200_LABELS:
        raise ValueError(f"positive_label={positive_label!r} not in {SIB200_LABELS}")

    snapshot = sib200_snapshot(resources)
    available = sib200_available_languages(snapshot)
    wanted = list(languages) if languages is not None else available
    missing = [lang for lang in wanted if lang not in available]
    if missing:
        raise KeyError(f"Language configs absent from the pinned revision: {missing[:10]}")

    out: dict[str, LanguageSplit] = {}
    schema_counts: dict[str, int] = {}
    for i, lang in enumerate(wanted, 1):
        # Read the pooled parquet files directly; see sib200_snapshot() for why this is not
        # a per-config load_dataset call.
        texts, raw_labels, schema = sib200_read_language(snapshot, lang)
        schema_counts[schema] = schema_counts.get(schema, 0) + 1

        if len(texts) != len(raw_labels):
            raise ValueError(f"text/label length mismatch for {lang}: "
                             f"{len(texts)} vs {len(raw_labels)}")

        labels = np.asarray([1 if lab == positive_label else 0 for lab in raw_labels],
                            dtype=np.int64)
        out[lang] = LanguageSplit(
            language=lang, texts=texts, labels=labels,
            split=assign_splits(len(texts), lang, seed=seed),
        )
        if verbose and (i % 25 == 0 or i == len(wanted)):
            print(f"  SIB-200: {i}/{len(wanted)} languages loaded")

    if verbose:
        print(f"  SIB-200 schemas encountered: {schema_counts}")
    return out


def sib200_label_names(resources: dict | None = None) -> list[str]:
    """The 7 topic labels, read from the pinned revision rather than hard-coded."""
    from datasets import load_dataset

    resources = resources or load_resources()
    spec = resources["datasets"]["sib200"]
    ds = load_dataset(spec["hf_id"], "eng_Latn", revision=spec["revision"])
    return list(ds["train"].features["label"].names)


def persist_split_manifest(corpus: dict[str, LanguageSplit], seed: int = DEFAULT_SEED,
                           out_path: Path | None = None, name: str = "split_manifest") -> Path:
    """Write the split assignment to disk so it is auditable and cannot drift.

    Stores per-language counts and a checksum of the split vector rather than the texts
    themselves: the texts are recoverable from the pinned dataset revision, but a silently
    changed split would be undetectable without this record.
    """
    import hashlib

    out_path = out_path or PROCESSED_DIR / f"{name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "seed": seed,
        "split_fractions": SPLIT_FRACTIONS,
        "languages": {},
    }
    for lang, split in sorted(corpus.items()):
        digest = hashlib.sha256("".join(split.split.tolist()).encode("utf-8")).hexdigest()
        manifest["languages"][lang] = {
            "n": int(len(split.texts)),
            "n_positive": int(np.sum(split.labels == 1)),
            "positive_rate": round(float(np.mean(split.labels == 1)), 6),
            "counts": {s: int(np.sum(split.split == s)) for s in ("train", "dev", "test")},
            "split_sha256": digest,
        }
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out_path
