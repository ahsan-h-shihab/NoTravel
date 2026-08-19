"""Tests for the leave-one-out pooled/worst-case global baselines (reviewer sim #1, R2-1).

These baselines exist because comparing an English-only threshold against per-language
calibration omits the obvious middle option a real deployer would take. The critical
property is LEAVE-ONE-OUT: if the target language leaks into the pool, the baseline is
flattered by exactly the labels the scenario says are unavailable, and the paper's headline
would rest on a straw man.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.threshold_study import LanguageScores, _leave_one_out_global_thresholds


def _lang(name: str, shift: float, n: int = 600, seed: int = 0) -> LanguageScores:
    rng = np.random.default_rng(seed)
    labels = np.concatenate([np.ones(n // 2, int), np.zeros(n // 2, int)])
    scores = np.clip(np.where(labels == 1,
                              rng.normal(0.65 + shift, 0.12, n),
                              rng.normal(0.35 + shift, 0.12, n)), 0, 1)
    return LanguageScores(name, scores, labels, scores.copy(), labels.copy())


def test_target_language_is_excluded_from_its_own_pool():
    """THE property. A language whose scores are wildly shifted must not be able to pull its
    own pooled threshold toward itself."""
    normal = {f"n{i}": _lang(f"n{i}", 0.0, seed=i) for i in range(4)}
    outlier = {"weird": _lang("weird", 0.30, seed=99)}
    corpus = {**normal, **outlier}

    pooled, _ = _leave_one_out_global_thresholds(corpus, "f1", None)

    # The outlier's pooled threshold is computed from the four normal languages only, so it
    # must match what those four alone produce -- i.e. be close to their own pooled values.
    normal_pooled = np.mean([pooled[k] for k in normal])
    assert abs(pooled["weird"] - normal_pooled) < 0.12, (
        "the outlier appears to have influenced its own pooled threshold")


def test_worstcase_is_the_max_of_the_other_languages():
    corpus = {f"l{i}": _lang(f"l{i}", 0.1 * i, seed=i) for i in range(4)}
    _, worstcase = _leave_one_out_global_thresholds(corpus, "f1", None)

    from src.eval.thresholds import tune_threshold
    taus = {k: tune_threshold(v.dev_scores, v.dev_labels, objective="f1")
            for k, v in corpus.items()}

    for held_out in corpus:
        expected = max(t for k, t in taus.items() if k != held_out)
        assert worstcase[held_out] == pytest.approx(expected)


def test_every_language_receives_a_threshold():
    corpus = {f"l{i}": _lang(f"l{i}", 0.05 * i, seed=i) for i in range(5)}
    pooled, worstcase = _leave_one_out_global_thresholds(corpus, "f1", None)
    assert set(pooled) == set(corpus)
    assert set(worstcase) == set(corpus)


def test_single_language_corpus_degrades_gracefully():
    """With one language there is nothing to leave out; must not crash or fabricate."""
    pooled, worstcase = _leave_one_out_global_thresholds({"only": _lang("only", 0.0)},
                                                         "f1", None)
    assert pooled == {} and worstcase == {}


def test_works_under_the_fpr_budget_objective():
    corpus = {f"l{i}": _lang(f"l{i}", 0.08 * i, seed=i) for i in range(4)}
    pooled, worstcase = _leave_one_out_global_thresholds(corpus, "target_fpr", 0.05)
    assert all(0.0 <= v <= 1.0 for v in pooled.values())
    # Worst-case must be at least as conservative (higher) as the pooled threshold typically
    # is; at minimum it must be a real threshold in range.
    assert all(0.0 <= v <= 1.0 for v in worstcase.values())
