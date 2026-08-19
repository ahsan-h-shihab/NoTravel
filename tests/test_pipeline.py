"""Unit tests for the scoring/resampling pipeline.

`resample_to_prevalence` underpins the entire prevalence sweep (risk R-13). If it silently
produced the wrong class balance, every conclusion about prior shift would be wrong while
looking perfectly plausible.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.threshold_study import LanguageScores
from src.eval.pipeline import resample_to_prevalence


def _make(n_pos: int, n_neg: int) -> LanguageScores:
    rng = np.random.default_rng(0)
    labels = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
    scores = rng.random(labels.size)
    return LanguageScores("xx", scores, labels, scores.copy(), labels.copy())


@pytest.mark.parametrize("target", [0.05, 0.1, 0.25, 0.5, 0.8])
def test_achieves_target_prevalence(target):
    out = resample_to_prevalence(_make(1000, 1000), target, np.random.default_rng(1))
    for labels in (out.dev_labels, out.test_labels):
        assert np.mean(labels == 1) == pytest.approx(target, abs=0.01)


def test_no_duplication():
    """Duplicating examples would understate bootstrap variance."""
    src = _make(500, 500)
    out = resample_to_prevalence(src, 0.1, np.random.default_rng(2))
    # Scores are distinct by construction, so uniqueness detects duplication.
    assert np.unique(out.dev_scores).size == out.dev_scores.size


def test_keeps_all_of_the_limiting_class_when_possible():
    """At 10% prevalence from a 100/1000 pool, all 100 positives should survive."""
    out = resample_to_prevalence(_make(100, 1000), 0.1, np.random.default_rng(3))
    assert int(np.sum(out.dev_labels == 1)) == 100
    assert int(np.sum(out.dev_labels == 0)) == 900


def test_downsamples_positives_when_negatives_are_scarce():
    """At 10% prevalence from a 900/90 pool, negatives bind: keep 90 neg, 10 pos."""
    out = resample_to_prevalence(_make(900, 90), 0.1, np.random.default_rng(4))
    assert int(np.sum(out.dev_labels == 0)) == 90
    assert int(np.sum(out.dev_labels == 1)) == 10


def test_single_class_input_is_returned_unchanged():
    """Degenerate input must pass through rather than raise or fabricate examples."""
    src = _make(50, 0)
    out = resample_to_prevalence(src, 0.2, np.random.default_rng(5))
    assert out.dev_labels.size == 50


def test_rejects_invalid_prevalence():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            resample_to_prevalence(_make(10, 10), bad, np.random.default_rng(6))


def test_is_deterministic_given_a_seed():
    a = resample_to_prevalence(_make(300, 300), 0.2, np.random.default_rng(7))
    b = resample_to_prevalence(_make(300, 300), 0.2, np.random.default_rng(7))
    np.testing.assert_array_equal(a.dev_scores, b.dev_scores)
