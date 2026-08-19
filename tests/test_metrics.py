"""Unit tests for metric computation.

Why these exist: RESULT TRACEABILITY forbids unexplained numbers, and every
headline result in this project is a function of these primitives. A silent error here would
propagate into every table and figure, so the primitives are tested against hand-computed
values and against independent implementations.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.metrics import auroc, expected_calibration_error, operating_point


def test_operating_point_hand_computed():
    # scores  0.9  0.8  0.4  0.3   threshold 0.5  -> predict 1,1,0,0
    # labels    1    0    1    0
    scores = np.array([0.9, 0.8, 0.4, 0.3])
    labels = np.array([1, 0, 1, 0])
    op = operating_point(scores, labels, 0.5)

    assert (op.tp, op.fp, op.tn, op.fn) == (1, 1, 1, 1)
    assert op.precision == pytest.approx(0.5)
    assert op.recall == pytest.approx(0.5)
    assert op.f1 == pytest.approx(0.5)
    assert op.fpr == pytest.approx(0.5)
    assert op.fnr == pytest.approx(0.5)
    assert op.balanced_accuracy == pytest.approx(0.5)
    assert op.accuracy == pytest.approx(0.5)
    assert op.predicted_positive_rate == pytest.approx(0.5)


def test_threshold_is_inclusive_at_boundary():
    """score == threshold must count as positive; an off-by-one here shifts every result."""
    op = operating_point(np.array([0.5]), np.array([1]), 0.5)
    assert op.tp == 1 and op.fn == 0


def test_degenerate_threshold_gives_zero_not_nan():
    """No predicted positives => precision undefined; we report 0.0, never nan.

    nan would silently destroy any downstream mean over languages.
    """
    op = operating_point(np.array([0.1, 0.2]), np.array([1, 0]), 0.9)
    assert op.precision == 0.0
    assert op.f1 == 0.0
    assert not np.isnan(op.f1)


def test_operating_point_matches_sklearn():
    rng = np.random.default_rng(0)
    scores = rng.random(500)
    labels = (rng.random(500) < 0.3).astype(int)
    from sklearn.metrics import f1_score, precision_score, recall_score

    for t in (0.2, 0.5, 0.8):
        op = operating_point(scores, labels, t)
        pred = (scores >= t).astype(int)
        assert op.f1 == pytest.approx(f1_score(labels, pred, zero_division=0))
        assert op.precision == pytest.approx(precision_score(labels, pred, zero_division=0))
        assert op.recall == pytest.approx(recall_score(labels, pred, zero_division=0))


def test_auroc_matches_sklearn_including_ties():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(1)
    for _ in range(5):
        scores = rng.integers(0, 5, size=300).astype(float)  # many ties on purpose
        labels = (rng.random(300) < 0.4).astype(int)
        assert auroc(scores, labels) == pytest.approx(roc_auc_score(labels, scores))


def test_auroc_perfect_and_constant():
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == pytest.approx(1.0)
    # All-tied scores carry no ranking information -> exactly 0.5.
    assert auroc(np.array([0.5] * 4), np.array([0, 1, 0, 1])) == pytest.approx(0.5)


def test_auroc_undefined_when_one_class_absent():
    """Genuinely undefined must surface as nan, not be silently filled with 0.5."""
    assert np.isnan(auroc(np.array([0.1, 0.9]), np.array([1, 1])))


def test_ece_zero_for_perfectly_calibrated_scores():
    # Scores exactly equal to empirical frequency in each bin => ECE 0.
    scores = np.concatenate([np.full(100, 0.25), np.full(100, 0.75)])
    labels = np.concatenate([
        np.array([1] * 25 + [0] * 75),
        np.array([1] * 75 + [0] * 25),
    ])
    assert expected_calibration_error(scores, labels, n_bins=10) == pytest.approx(0.0, abs=1e-9)
