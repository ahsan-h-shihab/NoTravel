"""Decision-threshold selection strategies.

This module is the scientific core of the project. Every strategy takes the SAME scores and
differs ONLY in how it picks the threshold, which is what makes the comparison identify the
operating-point effect rather than a model-quality effect.

Strategy families, in increasing order of information used:

  fixed_half        no information at all                     (naive default)
  source_global     labelled SOURCE-language data only        (THE STATUS QUO UNDER TEST)
  quantile_match    + unlabelled target text                  (label-free)
  sld_em            + unlabelled target text                  (label-free; Saerens+02)
  bbse              + unlabelled target text                  (label-free; Lipton+18)
  few_label(k)      + k labelled target examples              (label-scarce)
  target_full       + fully labelled target calibration set   (achievable upper bound)

`target_full` is the achievable oracle: it tunes on the target DEV split, never on test.
Tuning on test would be a genuine oracle but is not achievable by any deployer, and using it
as the reference would overstate the recoverable gap.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from .metrics import operating_point

Objective = Literal["f1", "youden", "balanced_accuracy", "target_fpr"]

#: Predicted-positive-rate bounds outside which a threshold is treated as DEGENERATE.
#:
#: Why this exists (D-011, forced by EXP-001): on a balanced corpus, maximizing F1 with a
#: weak classifier drives the threshold toward "flag everything". In the pilot, 6 of 14
#: target languages produced an F1-"optimal" threshold flagging >89% of all text as toxic
#: (Amharic 98.7%, FPR 0.978), and corr(AUROC, PPR) was -0.918. Those are not improved
#: operating points -- they are the absence of one, and no deployer would ship them.
#:
#: The bounds are deliberately generous: they catch collapse, not merely aggressive
#: thresholds. This is a REPORTING guard, never a silent correction -- a degenerate result is
#: surfaced and reported, not repaired.
DEGENERATE_PPR_HIGH = 0.80
DEGENERATE_PPR_LOW = 0.01


def predicted_positive_rate(scores: np.ndarray, threshold: float) -> float:
    """Fraction of examples a threshold flags positive. Needs no labels."""
    scores = np.asarray(scores, dtype=np.float64)
    return float(np.mean(scores >= threshold)) if scores.size else 0.0


def is_degenerate(scores: np.ndarray, threshold: float,
                  ppr_high: float = DEGENERATE_PPR_HIGH,
                  ppr_low: float = DEGENERATE_PPR_LOW) -> bool:
    """True if this threshold produces a near-trivial classifier.

    Judged on the predicted-positive rate alone, so it applies identically to every strategy
    and requires no labels -- a deployer could compute exactly this quantity in production.
    """
    if np.asarray(scores).size == 0:
        return True
    ppr = predicted_positive_rate(scores, threshold)
    return ppr > ppr_high or ppr < ppr_low


#: Candidate thresholds are the midpoints between consecutive unique scores, plus the
#: extremes. Sweeping midpoints rather than a fixed grid means the search is exact -- every
#: distinct classification of the calibration set is considered exactly once.
def candidate_thresholds(scores: np.ndarray, max_candidates: int = 2000) -> np.ndarray:
    uniq = np.unique(np.asarray(scores, dtype=np.float64))
    if uniq.size == 0:
        return np.array([0.5])
    if uniq.size > max_candidates:
        # Subsample by quantile rather than uniformly in score space: this keeps resolution
        # where the data actually is, which matters for heavily skewed score distributions.
        qs = np.linspace(0.0, 1.0, max_candidates)
        uniq = np.unique(np.quantile(uniq, qs))
    mids = (uniq[:-1] + uniq[1:]) / 2.0 if uniq.size > 1 else np.array([])
    lo = np.nextafter(uniq[0], -np.inf)
    hi = np.nextafter(uniq[-1], np.inf)
    return np.unique(np.concatenate([[lo], mids, [hi]]))


def _objective_fn(objective: Objective, target_fpr: float | None) -> Callable[[np.ndarray, np.ndarray, float], float]:
    if objective == "f1":
        return lambda s, y, t: operating_point(s, y, t).f1
    if objective == "balanced_accuracy":
        return lambda s, y, t: operating_point(s, y, t).balanced_accuracy
    if objective == "youden":
        return lambda s, y, t: (lambda op: op.recall - op.fpr)(operating_point(s, y, t))
    if objective == "target_fpr":
        if target_fpr is None:
            raise ValueError("objective='target_fpr' requires target_fpr=<value>")

        def _fn(s: np.ndarray, y: np.ndarray, t: float) -> float:
            op = operating_point(s, y, t)
            # Among thresholds meeting the FPR budget, prefer the highest recall; otherwise
            # prefer the smallest FPR overshoot. Encoded as a single scalar so the same
            # argmax sweep works for every objective.
            if op.fpr <= target_fpr:
                return 1.0 + op.recall
            return -abs(op.fpr - target_fpr)

        return _fn
    raise ValueError(f"Unknown objective: {objective}")


def tune_threshold(scores: np.ndarray, labels: np.ndarray, objective: Objective = "f1",
                   target_fpr: float | None = None) -> float:
    """Pick the threshold maximizing `objective` on LABELLED data.

    Ties are broken toward the median of the tied thresholds, which is more stable under
    resampling than taking the first or last -- an arbitrary tie-break would otherwise inject
    avoidable variance into every downstream comparison.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.size == 0:
        return 0.5

    cands = candidate_thresholds(scores)
    fn = _objective_fn(objective, target_fpr)
    values = np.array([fn(scores, labels, t) for t in cands])
    best = values.max()
    tied = cands[values >= best - 1e-12]
    return float(np.median(tied))


# --------------------------------------------------------------------------------------
# Label-free strategies: unlabelled target text only
# --------------------------------------------------------------------------------------

def quantile_match_threshold(source_scores: np.ndarray, source_threshold: float,
                             target_scores_unlabelled: np.ndarray) -> float:
    """Match the *predicted-positive rate* the source threshold induces.

    Rationale: a deployer who cannot measure target accuracy can still measure how often the
    system fires. Holding the firing rate constant across languages is the simplest
    defensible label-free policy, and it is what "keep the moderation volume steady"
    amounts to in practice. It is exact under pure score-distribution shift and wrong when
    true prevalence genuinely differs -- a limitation the experiments must expose, not hide.
    """
    source_scores = np.asarray(source_scores, dtype=np.float64)
    target = np.asarray(target_scores_unlabelled, dtype=np.float64)
    if source_scores.size == 0 or target.size == 0:
        return float(source_threshold)

    ppr = float(np.mean(source_scores >= source_threshold))
    if ppr <= 0.0:
        return float(np.nextafter(target.max(), np.inf))
    if ppr >= 1.0:
        return float(np.nextafter(target.min(), -np.inf))
    return float(np.quantile(target, 1.0 - ppr))


def _adjust_posteriors(p_source: np.ndarray, prior_source: float, prior_target: float) -> np.ndarray:
    """Re-weight source posteriors for a new class prior (the standard prior-shift formula)."""
    eps = 1e-12
    prior_source = float(np.clip(prior_source, eps, 1 - eps))
    prior_target = float(np.clip(prior_target, eps, 1 - eps))
    w_pos = prior_target / prior_source
    w_neg = (1.0 - prior_target) / (1.0 - prior_source)
    num = p_source * w_pos
    return num / np.clip(num + (1.0 - p_source) * w_neg, eps, None)


def sld_em_prior(target_probs: np.ndarray, prior_source: float,
                 max_iter: int = 200, tol: float = 1e-8) -> tuple[float, int]:
    """Saerens-Latinne-Decaestecker (2002) EM re-estimation of the target class prior.

    Uses UNLABELLED target posteriors only. Returns (estimated prior, iterations used).

    IMPORTANT (risk R-14): this assumes label shift -- p(y) changes while p(x|y) does not.
    Across *languages* that assumption is almost certainly false. The method is included
    precisely so its failure mode can be measured rather than assumed away.
    """
    p = np.asarray(target_probs, dtype=np.float64)
    if p.size == 0:
        return float(prior_source), 0

    prior = float(prior_source)
    iters = 0
    for iters in range(1, max_iter + 1):
        adjusted = _adjust_posteriors(p, prior_source, prior)
        new_prior = float(np.mean(adjusted))
        if abs(new_prior - prior) < tol:
            prior = new_prior
            break
        prior = new_prior
    return float(np.clip(prior, 1e-6, 1 - 1e-6)), iters


def bbse_prior(source_probs: np.ndarray, source_labels: np.ndarray, source_threshold: float,
               target_probs: np.ndarray) -> float:
    """Black-Box Shift Estimation (Lipton et al., 2018) for the target positive prior.

    Uses the source confusion matrix of the HARD predictions plus the target predicted-label
    distribution. Falls back to the source prior when the confusion matrix is singular or
    the solution leaves the simplex, which happens when the classifier is near-degenerate on
    a language -- a real occurrence worth reporting rather than papering over.
    """
    sp = np.asarray(source_probs, dtype=np.float64)
    sy = np.asarray(source_labels, dtype=np.int64)
    tp = np.asarray(target_probs, dtype=np.float64)
    prior_source = float(np.mean(sy == 1)) if sy.size else 0.5
    if sp.size == 0 or tp.size == 0:
        return prior_source

    s_pred = (sp >= source_threshold).astype(np.int64)
    # C[i, j] = p_source(y_hat = i, y = j)   -- the JOINT, not the conditional.
    C = np.zeros((2, 2), dtype=np.float64)
    for i in (0, 1):
        for j in (0, 1):
            C[i, j] = np.mean((s_pred == i) & (sy == j))

    t_pred = (tp >= source_threshold).astype(np.int64)
    q = np.array([np.mean(t_pred == 0), np.mean(t_pred == 1)])

    if abs(np.linalg.det(C)) < 1e-10:
        return prior_source
    try:
        # Solving C w = q yields the IMPORTANCE WEIGHTS w[j] = p_target(y=j) / p_source(y=j),
        # NOT the target prior itself. This follows from
        #   sum_j C[i,j] w[j] = sum_j p(y_hat=i | y=j) p_target(y=j) = p_target(y_hat=i) = q[i].
        # Multiplying back by the source prior is therefore required; omitting it returns w[1]
        # (=0.5 in the label-shift unit test) instead of the prior (=0.25).
        w = np.linalg.solve(C, q)
    except np.linalg.LinAlgError:
        return prior_source

    prior_source_vec = np.array([np.mean(sy == 0), np.mean(sy == 1)])
    mu = w * prior_source_vec  # mu[j] = p_target(y = j)

    if not np.all(np.isfinite(mu)) or mu[1] <= 0.0 or mu[1] >= 1.0:
        return prior_source
    return float(np.clip(mu[1], 1e-6, 1 - 1e-6))


def _forward_adjust_scalar(p: float, prior_ref: float, prior_new: float) -> float:
    """Map one posterior from reference prior `prior_ref` to `prior_new`."""
    eps = 1e-12
    pr = float(np.clip(prior_ref, eps, 1 - eps))
    pn = float(np.clip(prior_new, eps, 1 - eps))
    w_pos, w_neg = pn / pr, (1.0 - pn) / (1.0 - pr)
    num = p * w_pos
    return float(num / max(num + (1.0 - p) * w_neg, eps))


def prior_shift_transfer(source_threshold: float, prior_train: float,
                         prior_hat_source: float, prior_hat_target: float) -> float:
    """Transfer a source-tuned raw threshold to a target language via estimated priors.

    WHY TWO STEPS AND NOT ONE (a correctness fix)
    ---------------------------------------------
    The head's posteriors encode the prior it was TRAINED at (`prior_train`), which is not
    the prevalence of whatever validation sample the threshold was tuned on. An earlier
    version passed the resampled dev prevalence as the reference prior, which made SLD/EM
    report an estimated prior of 1.0 *even for the source language itself* -- where there is
    no shift at all. That would have been published as "SLD catastrophically fails
    cross-lingually" when in fact the method was being misused.

    The correct transfer maps the threshold into the shared adjusted space using the SOURCE
    estimate, then back out using the TARGET estimate:

        tau_adjusted = forward(tau_source ; prior_train -> prior_hat_source)
        tau_target   = inverse(tau_adjusted ; prior_train -> prior_hat_target)

    Treating source and target symmetrically gives the method the sanity property it must
    have: when the target distribution equals the source, the threshold is unchanged.
    """
    tau_adjusted = _forward_adjust_scalar(source_threshold, prior_train, prior_hat_source)
    return prior_shift_threshold(tau_adjusted, prior_train, prior_hat_target, np.array([0.5]))


def prior_shift_threshold(source_threshold: float, prior_source: float, prior_target: float,
                          target_probs: np.ndarray) -> float:
    """Convert a prior-shift correction into an equivalent threshold on the RAW scores.

    Instead of adjusting every posterior and re-thresholding at `source_threshold`, we solve
    for the raw score whose adjusted value equals `source_threshold`. This keeps every
    strategy expressed in the same units -- a threshold on the original scores -- so all
    strategies remain directly comparable and no strategy gets to change the score scale.

    Derivation: adjusted(p) is monotone increasing in p, so the raw threshold is
    adjusted^{-1}(source_threshold), which has the closed form below.
    """
    eps = 1e-12
    ps = float(np.clip(prior_source, eps, 1 - eps))
    pt = float(np.clip(prior_target, eps, 1 - eps))
    tau = float(np.clip(source_threshold, eps, 1 - eps))

    w_pos = pt / ps
    w_neg = (1.0 - pt) / (1.0 - ps)
    # Solve tau = (p*w_pos) / (p*w_pos + (1-p)*w_neg)  for p.
    denom = w_pos * (1.0 - tau) + w_neg * tau
    if denom <= eps:
        return float(source_threshold)
    raw = (w_neg * tau) / denom

    if np.asarray(target_probs).size and not np.isfinite(raw):
        return float(source_threshold)
    return float(np.clip(raw, 0.0, 1.0))


# --------------------------------------------------------------------------------------
# Label-scarce strategy
# --------------------------------------------------------------------------------------

def few_label_threshold(target_scores: np.ndarray, target_labels: np.ndarray, k: int,
                        rng: np.random.Generator, objective: Objective = "f1",
                        target_fpr: float | None = None,
                        fallback_threshold: float | None = None,
                        stratified: bool = True) -> float:
    """Tune on a random subsample of k labelled target examples.

    `stratified=True` draws k/2 from each class where possible. This is the OPTIMISTIC
    variant: it assumes the practitioner can find positives cheaply. The pessimistic
    variant (`stratified=False`, simple random sampling) is the realistic one when
    positives are rare. Both are reported, because the gap between them is itself the
    practical answer to "how should I spend my annotation budget?".

    Falls back to `fallback_threshold` when the subsample contains a single class, since a
    threshold tuned on one class is meaningless -- and silently returning one would
    manufacture a spuriously good result at small k.
    """
    scores = np.asarray(target_scores, dtype=np.float64)
    labels = np.asarray(target_labels, dtype=np.int64)
    fallback = 0.5 if fallback_threshold is None else float(fallback_threshold)

    if scores.size == 0 or k <= 0:
        return fallback
    k = min(k, scores.size)

    if stratified:
        pos_idx = np.flatnonzero(labels == 1)
        neg_idx = np.flatnonzero(labels == 0)
        n_pos = min(len(pos_idx), k // 2)
        n_neg = min(len(neg_idx), k - n_pos)
        n_pos = min(len(pos_idx), k - n_neg)  # give leftovers back if one class is short
        idx = np.concatenate([
            rng.choice(pos_idx, size=n_pos, replace=False) if n_pos else np.array([], dtype=int),
            rng.choice(neg_idx, size=n_neg, replace=False) if n_neg else np.array([], dtype=int),
        ])
    else:
        idx = rng.choice(scores.size, size=k, replace=False)

    sub_scores, sub_labels = scores[idx], labels[idx]
    if np.unique(sub_labels).size < 2:
        return fallback
    return tune_threshold(sub_scores, sub_labels, objective=objective, target_fpr=target_fpr)
