"""Turn a corpus + encoder + source-trained head into per-language scores.

Kept separate from both scoring and analysis so that the expensive part (embedding) and the
cheap part (thresholding) can be re-run independently. In practice this means the entire
strategy grid, label-budget sweep and bootstrap can be redesigned and re-run without
recomputing a single forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analysis.threshold_study import LanguageScores
from ..data.loaders import LanguageSplit
from ..models.heads import TrainedHead, train_head
from ..utils.seeding import DEFAULT_SEED


@dataclass(frozen=True)
class ScoredCorpus:
    scores: dict[str, LanguageScores]
    head: TrainedHead


def fit_and_score(corpus: dict[str, LanguageSplit], embeddings: dict[str, np.ndarray],
                  source_language: str, seed: int = DEFAULT_SEED) -> ScoredCorpus:
    """Fit the head on the SOURCE language only, then score every language's dev and test.

    The head never sees any target-language label. That is what makes the design a *transfer*
    study rather than a per-language training study, and it is why any cross-language
    difference in the resulting operating point is attributable to transfer rather than to
    differing amounts of supervision.
    """
    if source_language not in corpus:
        raise KeyError(f"Source language {source_language!r} not in corpus")

    src = corpus[source_language]
    src_emb = embeddings[source_language]
    head = train_head(
        src_emb[src.indices("train")], src.labels[src.indices("train")],
        src_emb[src.indices("dev")], src.labels[src.indices("dev")],
        source_language=source_language, seed=seed,
    )

    scores: dict[str, LanguageScores] = {}
    for lang, split in corpus.items():
        emb = embeddings[lang]
        if emb.shape[0] != len(split.texts):
            raise ValueError(
                f"Embedding/text count mismatch for {lang}: {emb.shape[0]} vs {len(split.texts)}")
        dev_idx, test_idx = split.indices("dev"), split.indices("test")
        scores[lang] = LanguageScores(
            language=lang,
            dev_scores=head.score(emb[dev_idx]),
            dev_labels=split.labels[dev_idx],
            test_scores=head.score(emb[test_idx]),
            test_labels=split.labels[test_idx],
        )
    return ScoredCorpus(scores=scores, head=head)


def resample_to_prevalence(ls: LanguageScores, target_prevalence: float,
                           rng: np.random.Generator) -> LanguageScores:
    """Subsample a language's dev and test sets to a target positive prevalence.

    Why this exists (risk R-13): the primary corpus is balanced 50/50 by construction, which
    no deployment is. Prevalence is therefore manipulated as an EXPLICIT controlled factor
    rather than left as a silent confound. Subsampling (rather than reweighting) is used so
    that the resulting sets are genuine samples on which the ordinary metrics are valid
    without correction.

    The majority class is downsampled rather than the minority upsampled, so no example is
    duplicated -- duplication would understate variance in the bootstrap.
    """
    if not 0.0 < target_prevalence < 1.0:
        raise ValueError(f"target_prevalence must be in (0,1), got {target_prevalence}")

    p = target_prevalence
    ratio = (1.0 - p) / p  # negatives per positive at the target prevalence

    def _sub(scores: np.ndarray, labels: np.ndarray, seed_offset: int) -> tuple[np.ndarray, np.ndarray]:
        pos = np.flatnonzero(labels == 1)
        neg = np.flatnonzero(labels == 0)
        if pos.size == 0 or neg.size == 0:
            return scores, labels

        # Largest subsample hitting the target prevalence with no duplication:
        # keep all positives if the required negatives are available, else keep all
        # negatives and cut positives to match.
        if pos.size * ratio <= neg.size:
            n_pos = pos.size
            n_neg = int(np.floor(pos.size * ratio))
        else:
            n_neg = neg.size
            n_pos = int(np.floor(neg.size / ratio))
        n_pos, n_neg = max(1, n_pos), max(1, n_neg)

        local = np.random.default_rng(rng.integers(0, 2**31 - 1) + seed_offset)
        keep = np.concatenate([
            local.choice(pos, size=min(n_pos, pos.size), replace=False),
            local.choice(neg, size=min(n_neg, neg.size), replace=False),
        ])
        local.shuffle(keep)
        return scores[keep], labels[keep]

    dev_s, dev_y = _sub(ls.dev_scores, ls.dev_labels, 0)
    test_s, test_y = _sub(ls.test_scores, ls.test_labels, 1)
    return LanguageScores(
        language=ls.language,
        dev_scores=dev_s, dev_labels=dev_y,
        test_scores=test_s, test_labels=test_y,
    )
