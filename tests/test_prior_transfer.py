"""Tests for prior-shift threshold transfer.

These encode the correctness property whose absence produced a false result: the earlier
implementation used the *resampled dev prevalence* as the reference prior instead of the
prior the head was TRAINED at. That made SLD/EM report an estimated prior of 1.0 for the
SOURCE language -- where no shift exists -- which would have been written up as
"SLD fails cross-lingually" when the method was simply being misused.

The identity test below is the one that would have caught it immediately.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.thresholds import prior_shift_transfer, sld_em_prior


def test_identity_when_target_equals_source():
    """THE regression test. If the target distribution equals the source, the transferred
    threshold must equal the source threshold exactly, for ANY training prior."""
    for train_prior in (0.1, 0.5, 0.9):
        for tau in (0.2, 0.5, 0.8):
            for prior_hat in (0.05, 0.5, 0.95):
                out = prior_shift_transfer(tau, train_prior, prior_hat, prior_hat)
                assert out == pytest.approx(tau, abs=1e-9), (
                    f"not identity at train_prior={train_prior}, tau={tau}, prior={prior_hat}")


def test_lower_target_prior_raises_the_threshold():
    """If positives are estimated rarer in the target, the system should fire less."""
    tau_same = prior_shift_transfer(0.5, 0.5, 0.3, 0.3)
    tau_rarer = prior_shift_transfer(0.5, 0.5, 0.3, 0.05)
    assert tau_rarer > tau_same


def test_higher_target_prior_lowers_the_threshold():
    tau_same = prior_shift_transfer(0.5, 0.5, 0.3, 0.3)
    tau_commoner = prior_shift_transfer(0.5, 0.5, 0.3, 0.7)
    assert tau_commoner < tau_same


def test_sld_recovers_a_shifted_prior_when_given_the_TRAINING_prior():
    """Constructed so label shift genuinely holds. SLD must recover the target prior when
    handed the prior its posteriors encode."""
    rng = np.random.default_rng(0)
    train_prior, target_prior, n = 0.5, 0.10, 60000

    y = (rng.random(n) < target_prior).astype(int)
    x = np.where(y == 1, rng.normal(2.0, 1.0, n), rng.normal(-2.0, 1.0, n))
    # Bayes posterior under the TRAINING prior (equal variances -> logistic form).
    logit = np.log(train_prior / (1 - train_prior)) + 2.0 * x
    posteriors = 1.0 / (1.0 + np.exp(-logit))

    est, _ = sld_em_prior(posteriors, train_prior)
    assert est == pytest.approx(target_prior, abs=0.03)


def test_sld_with_the_wrong_reference_prior_is_badly_wrong():
    """Documents the failure mode explicitly, so nobody 'simplifies' the code back.

    Feeding the *sample* prevalence instead of the training prior drives the estimate far
    from the truth -- this is exactly what produced the spurious 1.0 estimates.
    """
    rng = np.random.default_rng(1)
    train_prior, target_prior, n = 0.5, 0.05, 40000

    y = (rng.random(n) < target_prior).astype(int)
    x = np.where(y == 1, rng.normal(2.0, 1.0, n), rng.normal(-2.0, 1.0, n))
    posteriors = 1.0 / (1.0 + np.exp(-(np.log(train_prior / (1 - train_prior)) + 2.0 * x)))

    correct, _ = sld_em_prior(posteriors, train_prior)
    wrong, _ = sld_em_prior(posteriors, target_prior)   # the bug

    assert abs(correct - target_prior) < 0.03
    assert abs(wrong - target_prior) > abs(correct - target_prior)
