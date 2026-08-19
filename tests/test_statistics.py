"""Tests for the statistical reporting layer (`D-013`).

These numbers go directly into the manuscript, so each function is checked against a
hand-computable or independently-known value. A silent error here would misstate the
uncertainty on every headline claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.statistics import (
    over_budget_multiple,
    paired_comparison,
    practical_significance,
    spearman,
    wilson_interval,
)


def test_wilson_matches_known_value():
    """Wilson 95% CI for 14/14 is approximately [0.784, 1.000] -- the interval reported for
    the paper's headline violation rate."""
    est = wilson_interval(14, 14)
    assert est.point == pytest.approx(1.0)
    assert est.ci_low == pytest.approx(0.784, abs=0.005)
    assert est.ci_high == pytest.approx(1.0, abs=1e-9)


def test_wilson_stays_inside_unit_interval_at_extremes():
    """The reason Wilson is used rather than the normal approximation: at 0 and 1 the normal
    interval leaves [0,1] entirely."""
    for k, n in [(0, 14), (14, 14), (0, 3), (3, 3)]:
        est = wilson_interval(k, n)
        assert 0.0 <= est.ci_low <= est.ci_high <= 1.0


def test_wilson_is_asymmetric_near_the_boundary():
    est = wilson_interval(1, 20)
    assert (est.ci_high - est.point) > (est.point - est.ci_low)


def test_wilson_handles_empty():
    est = wilson_interval(0, 0)
    assert np.isnan(est.point)


def test_paired_comparison_detects_a_consistent_difference():
    """A strategy uniformly better on every language must be flagged, with all pairs
    favouring it."""
    rng = np.random.default_rng(0)
    b = rng.uniform(0.10, 0.30, 14)
    a = b - 0.08                      # a is uniformly lower (better, since lower_is_better)
    res = paired_comparison(a, b, seed=1)

    assert res.n_pairs == 14
    assert res.n_favouring_a == 14
    assert res.median_difference == pytest.approx(-0.08, abs=1e-9)
    assert res.ci_high < 0
    assert res.p_value is not None and res.p_value < 0.01


def test_paired_comparison_reports_no_difference_when_there_is_none():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, 14)
    res = paired_comparison(x, x.copy(), seed=3)
    assert res.median_difference == pytest.approx(0.0)
    assert res.p_value == pytest.approx(1.0)
    assert res.n_ties == 14


def test_paired_comparison_carries_the_dependence_caveat():
    """T-S4: languages are not independent. The caveat must travel with the number."""
    res = paired_comparison(np.arange(14.0), np.arange(14.0) + 1, seed=4)
    assert "not independent" in res.note


def test_paired_comparison_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        paired_comparison(np.zeros(5), np.zeros(6))


def test_spearman_reports_non_significance_as_a_result():
    """The AUROC vs over-budget null (rho=-0.235, p=0.42) is a FINDING. It must be
    reported as not significant, not silently dropped."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=14)
    y = rng.normal(size=14)
    res = spearman(x, y)
    assert res.n == 14
    assert isinstance(res.significant, bool)
    assert "significant" in res.format()


def test_spearman_detects_a_monotone_relationship():
    x = np.arange(14.0)
    res = spearman(x, x ** 2)          # monotone but non-linear
    assert res.rho == pytest.approx(1.0)
    assert res.significant


def test_over_budget_multiple_is_the_decision_unit():
    assert over_budget_multiple(np.array([0.172]), 0.05)[0] == pytest.approx(3.44)


def test_practical_significance_states_deployment_consequences():
    msg = practical_significance(0.172, 0.05, daily_volume=1_000_000)
    assert "3.4x" in msg
    assert "50,000" in msg      # intended false positives
    assert "172,000" in msg     # actual
    assert "122,000" in msg     # excess
