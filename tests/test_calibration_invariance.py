"""The manuscript asserts that source-fitted post-hoc calibration cannot move an operating point.

Section V-D states Eq. (1): for a strictly monotone map g fitted on source data and a threshold
re-derived at the same budget, every language's realised FPR is unchanged. That is an algebraic
claim, so it is testable directly rather than only on our data -- these tests check the property
on synthetic scores where the answer is known, so a regression in the threshold or metric code
would surface here rather than in a reviewer's reading of the paper.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.check_calibration_invariance import (
    fit_platt, fit_temperature, logit, rates, threshold_at_budget,
)

BUDGET = 0.05


def _synthetic(seed: int = 0, n: int = 4000):
    """Two languages whose negatives sit at different places on the score scale."""
    rng = np.random.default_rng(seed)
    def lang(neg_mu, pos_mu):
        z = np.concatenate([rng.normal(neg_mu, 1.0, n), rng.normal(pos_mu, 1.0, n)])
        y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
        return 1.0 / (1.0 + np.exp(-z)), y
    return {"src": lang(-2.0, 1.0), "tgt": lang(-0.5, 2.0)}


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0, 7.5])
def test_temperature_scaling_does_not_move_the_operating_point(temperature):
    data = _synthetic()
    s_src, y_src = data["src"]
    s_tgt, y_tgt = data["tgt"]

    tau = threshold_at_budget(s_src, y_src, BUDGET)
    base = rates(s_tgt, y_tgt, tau)

    g = lambda s: 1.0 / (1.0 + np.exp(-logit(s) / temperature))     # noqa: E731
    tau_g = threshold_at_budget(g(s_src), y_src, BUDGET)
    scaled = rates(g(s_tgt), y_tgt, tau_g)

    assert scaled == pytest.approx(base, abs=1e-12)


def test_platt_scaling_does_not_move_the_operating_point():
    data = _synthetic(seed=1)
    s_src, y_src = data["src"]
    s_tgt, y_tgt = data["tgt"]

    a, b = fit_platt(s_src, y_src)
    g = lambda s: 1.0 / (1.0 + np.exp(-(a * logit(s) + b)))         # noqa: E731

    base = rates(s_tgt, y_tgt, threshold_at_budget(s_src, y_src, BUDGET))
    scaled = rates(g(s_tgt), y_tgt, threshold_at_budget(g(s_src), y_src, BUDGET))
    assert scaled == pytest.approx(base, abs=1e-12)


def test_a_non_monotone_map_is_not_covered_by_the_claim():
    """Guard the scope of Eq. (1): monotonicity is doing the work, not calibration per se.

    Without this, the test suite would pass equally if the implementation happened to ignore
    the map entirely, and the manuscript's claim would be untested rather than verified.
    """
    data = _synthetic(seed=2)
    s_src, y_src = data["src"]
    s_tgt, y_tgt = data["tgt"]

    g = lambda s: np.abs(s - 0.5)                                   # noqa: E731  (not monotone)
    base = rates(s_tgt, y_tgt, threshold_at_budget(s_src, y_src, BUDGET))
    mangled = rates(g(s_tgt), y_tgt, threshold_at_budget(g(s_src), y_src, BUDGET))
    assert mangled != pytest.approx(base, abs=1e-9)


def test_fitted_temperature_is_positive_and_finite():
    s_src, y_src = _synthetic(seed=3)["src"]
    t = fit_temperature(s_src, y_src)
    assert np.isfinite(t) and t > 0
