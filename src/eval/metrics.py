"""Classification metrics at a fixed decision threshold.

Why these metrics (EVALUATION POLICY -- every metric needs a scientific
justification, not popularity):

* F1              -- the standard operating-point summary in the toxicity / hate-speech
                     literature, so it makes results comparable to prior work.
* FPR and FNR     -- the two quantities a deployer actually trades off, and the ones a
                     fairness audit compares across groups. The *disparity* in these across
                     languages is the operational harm this project is about.
* Precision/Recall-- needed to interpret F1 movement (an F1 drop from lost precision means
                     something operationally different from one from lost recall).
* Balanced acc.   -- threshold-sensitive but prevalence-insensitive, so it separates
                     score-distribution effects from prior effects (risk R-13).
* AUROC           -- threshold-INDEPENDENT. Reported to prove that observed operating-point
                     losses are attributable to the threshold rather than to model quality.
                     This is the control that makes the project's central claim identifiable.

Everything here is deliberately implemented over raw counts rather than delegated, so that
each reported number has one visible definition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class OperatingPoint:
    """All metrics for one (scores, labels, threshold) triple."""

    threshold: float
    n: int
    n_pos: int
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    fpr: float
    fnr: float
    balanced_accuracy: float
    accuracy: float
    predicted_positive_rate: float

    def as_dict(self) -> dict:
        return asdict(self)


def _safe_div(num: float, den: float) -> float:
    """Return 0.0 for 0/0 rather than nan.

    Justification: an undefined precision (no predicted positives) is reported as 0.0 so
    that a degenerate threshold is penalised rather than silently propagating nan into
    aggregate statistics, where it would quietly destroy an entire mean.
    """
    return float(num) / float(den) if den else 0.0


def operating_point(scores: np.ndarray, labels: np.ndarray, threshold: float) -> OperatingPoint:
    """Compute all metrics for a fixed threshold. Positive iff score >= threshold."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.shape != labels.shape:
        raise ValueError(f"shape mismatch: scores {scores.shape} vs labels {labels.shape}")

    pred = scores >= threshold
    pos = labels == 1

    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & ~pos))
    tn = int(np.sum(~pred & ~pos))
    fn = int(np.sum(~pred & pos))

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    tnr = _safe_div(tn, tn + fp)

    return OperatingPoint(
        threshold=float(threshold),
        n=int(scores.size),
        n_pos=int(np.sum(pos)),
        tp=tp, fp=fp, tn=tn, fn=fn,
        precision=precision,
        recall=recall,
        f1=_safe_div(2 * precision * recall, precision + recall),
        fpr=_safe_div(fp, fp + tn),
        fnr=_safe_div(fn, fn + tp),
        balanced_accuracy=0.5 * (recall + tnr),
        accuracy=_safe_div(tp + tn, scores.size),
        predicted_positive_rate=_safe_div(tp + fp, scores.size),
    )


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold-independent AUROC via the rank (Mann-Whitney U) identity.

    Implemented directly rather than imported so the tie handling is explicit: ties receive
    average ranks, which is the standard convention and gives 0.5 for constant scores.
    Returns nan when one class is absent, since AUROC is genuinely undefined then -- a case
    that must be surfaced, not silently filled in.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0  # 1-based average rank for the tie group
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    rank_sum_pos = float(np.sum(ranks[labels == 1]))
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def expected_calibration_error(scores: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Equal-width binned ECE.

    Reported as a descriptive diagnostic only. ECE is NOT used to support any headline
    claim: it is known to be sensitive to binning, and the project's claims concern
    operating points rather than probability quality.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (scores > lo) & (scores <= hi) if lo > 0 else (scores >= lo) & (scores <= hi)
        if not np.any(mask):
            continue
        ece += (np.sum(mask) / scores.size) * abs(np.mean(labels[mask]) - np.mean(scores[mask]))
    return float(ece)
