"""Unit tests for threshold-selection strategies.

These strategies ARE the scientific contribution's machinery. Each test states the property
that must hold for the corresponding experimental claim to be meaningful.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.metrics import operating_point
from src.eval.thresholds import (
    bbse_prior,
    candidate_thresholds,
    few_label_threshold,
    prior_shift_threshold,
    quantile_match_threshold,
    sld_em_prior,
    tune_threshold,
)


# --------------------------------------------------------------------------- tuning

def test_tuned_threshold_is_at_least_as_good_as_any_grid_point():
    """The sweep must actually find the optimum -- otherwise every 'gap' we report is
    partly just search failure rather than threshold transfer."""
    rng = np.random.default_rng(0)
    scores = rng.random(400)
    labels = (rng.random(400) < (0.2 + 0.6 * scores)).astype(int)

    best_t = tune_threshold(scores, labels, objective="f1")
    best_f1 = operating_point(scores, labels, best_t).f1

    grid_best = max(operating_point(scores, labels, t).f1 for t in np.linspace(0, 1, 501))
    assert best_f1 >= grid_best - 1e-9


def test_perfectly_separable_scores_give_perfect_f1():
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])
    t = tune_threshold(scores, labels, objective="f1")
    assert operating_point(scores, labels, t).f1 == pytest.approx(1.0)


def test_candidate_thresholds_separate_all_distinct_classifications():
    scores = np.array([0.1, 0.4, 0.6])
    cands = candidate_thresholds(scores)
    ppr = {operating_point(scores, np.array([0, 1, 1]), t).predicted_positive_rate for t in cands}
    assert ppr == {0.0, 1 / 3, 2 / 3, 1.0}


def test_target_fpr_objective_respects_the_budget():
    rng = np.random.default_rng(3)
    scores = rng.random(1000)
    labels = (rng.random(1000) < scores).astype(int)
    t = tune_threshold(scores, labels, objective="target_fpr", target_fpr=0.05)
    assert operating_point(scores, labels, t).fpr <= 0.05 + 1e-9


def test_target_fpr_requires_a_budget():
    with pytest.raises(ValueError):
        tune_threshold(np.array([0.1, 0.9]), np.array([0, 1]), objective="target_fpr")


# ------------------------------------------------------------------ label-free: quantile

def test_quantile_match_preserves_predicted_positive_rate_under_a_shift():
    """The defining property: same firing rate in the target as in the source."""
    rng = np.random.default_rng(4)
    source = rng.normal(0.4, 0.1, 5000).clip(0, 1)
    target = rng.normal(0.6, 0.1, 5000).clip(0, 1)  # shifted upward
    source_t = 0.5

    t = quantile_match_threshold(source, source_t, target)
    assert np.mean(target >= t) == pytest.approx(np.mean(source >= source_t), abs=0.02)


def test_quantile_match_is_identity_when_distributions_match():
    rng = np.random.default_rng(5)
    x = rng.normal(0.5, 0.15, 20000).clip(0, 1)
    y = rng.normal(0.5, 0.15, 20000).clip(0, 1)
    assert quantile_match_threshold(x, 0.5, y) == pytest.approx(0.5, abs=0.02)


# ------------------------------------------------------------------- label-free: priors

def test_sld_em_recovers_a_known_prior_under_true_label_shift():
    """Constructed so the label-shift assumption HOLDS. If SLD fails here it is broken;
    if it fails on real language data, that is a finding about languages (risk R-14)."""
    rng = np.random.default_rng(6)
    prior_source, prior_target = 0.5, 0.2
    n = 40000

    # p(x|y) identical in both domains; only the mixing proportion differs.
    def sample(prior):
        y = (rng.random(n) < prior).astype(int)
        x = np.where(y == 1, rng.normal(2.0, 1.0, n), rng.normal(-2.0, 1.0, n))
        return y, x

    _, x_src = sample(prior_source)
    _, x_tgt = sample(prior_target)

    # Bayes-optimal posterior for the SOURCE prior (equal variances => logistic form).
    def posterior(x, prior):
        lo = np.log(prior / (1 - prior)) + 2.0 * x  # 2*(mu1-mu0)/sigma^2 * x with the means above
        return 1.0 / (1.0 + np.exp(-lo))

    est, iters = sld_em_prior(posterior(x_tgt, prior_source), prior_source)
    assert est == pytest.approx(prior_target, abs=0.03)
    assert iters >= 1


def test_bbse_recovers_a_known_prior_under_true_label_shift():
    rng = np.random.default_rng(7)
    prior_source, prior_target = 0.5, 0.25
    n = 60000

    def sample(prior):
        y = (rng.random(n) < prior).astype(int)
        s = np.where(y == 1, rng.normal(0.7, 0.15, n), rng.normal(0.3, 0.15, n)).clip(0, 1)
        return y, s

    y_src, s_src = sample(prior_source)
    _, s_tgt = sample(prior_target)

    est = bbse_prior(s_src, y_src, 0.5, s_tgt)
    assert est == pytest.approx(prior_target, abs=0.03)


def test_bbse_falls_back_when_classifier_is_degenerate():
    """A near-constant classifier gives a singular confusion matrix; we must fall back to
    the source prior rather than emit a garbage estimate."""
    y_src = np.array([0, 1] * 100)
    s_src = np.full(200, 0.5)  # predicts one class for everything
    est = bbse_prior(s_src, y_src, 0.9, np.full(200, 0.5))
    assert est == pytest.approx(0.5)


def test_prior_shift_threshold_inverts_the_posterior_adjustment():
    """The raw-score threshold must classify identically to adjusting posteriors and
    thresholding at the source value. This is what keeps all strategies comparable."""
    rng = np.random.default_rng(8)
    p = rng.random(5000)
    prior_s, prior_t, tau = 0.5, 0.15, 0.5

    raw_t = prior_shift_threshold(tau, prior_s, prior_t, p)

    w_pos, w_neg = prior_t / prior_s, (1 - prior_t) / (1 - prior_s)
    adjusted = (p * w_pos) / (p * w_pos + (1 - p) * w_neg)

    np.testing.assert_array_equal(p >= raw_t, adjusted >= tau)


def test_prior_shift_threshold_is_identity_when_priors_match():
    assert prior_shift_threshold(0.37, 0.4, 0.4, np.array([0.5])) == pytest.approx(0.37)


def test_lower_target_prior_raises_the_threshold():
    """Directional sanity: if positives are rarer in the target, we should fire less."""
    t_low = prior_shift_threshold(0.5, 0.5, 0.1, np.array([0.5]))
    t_same = prior_shift_threshold(0.5, 0.5, 0.5, np.array([0.5]))
    assert t_low > t_same


# --------------------------------------------------------------------------- few-label

def test_few_label_falls_back_when_subsample_is_single_class():
    """Silently returning a tuned value here would manufacture a spuriously good result
    at small k, which is exactly the regime the paper reports on."""
    scores = np.linspace(0, 1, 50)
    labels = np.zeros(50, dtype=int)  # no positives at all
    t = few_label_threshold(scores, labels, k=10, rng=np.random.default_rng(9),
                            fallback_threshold=0.42)
    assert t == pytest.approx(0.42)


def test_few_label_approaches_full_label_as_k_grows():
    rng_data = np.random.default_rng(10)
    scores = rng_data.random(4000)
    labels = (rng_data.random(4000) < scores).astype(int)
    full_t = tune_threshold(scores, labels, objective="f1")

    def mean_abs_err(k, reps=25):
        errs = [abs(few_label_threshold(scores, labels, k, np.random.default_rng(100 + r)) - full_t)
                for r in range(reps)]
        return float(np.mean(errs))

    assert mean_abs_err(2000) < mean_abs_err(20)


def test_few_label_is_deterministic_given_a_seed():
    scores = np.random.default_rng(11).random(500)
    labels = (scores > 0.5).astype(int)
    a = few_label_threshold(scores, labels, 40, np.random.default_rng(12))
    b = few_label_threshold(scores, labels, 40, np.random.default_rng(12))
    assert a == b
