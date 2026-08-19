"""Statistical reporting for headline results (policy: `D-013`).

Every function here exists because a specific claim needs it. There is deliberately no
generic "compute all the statistics" entry point: STATISTICAL INTEGRITY forbids
computing statistics that answer no question, and `D-013` forbids mechanical p-values.

What is reported, and when:

  * **Uncertainty — always.** Proportions over languages get Wilson intervals (not the normal
    approximation, which is badly wrong near 0 and 1 and can leave [0,1]). Operating-point
    metrics get percentile bootstrap intervals.
  * **Effect size — always, in DECISION units.** The over-budget multiple (realised FPR /
    budget) and the absolute recall difference. Standardised effect sizes are deliberately
    avoided: "Cohen's d = 0.8" tells a deployer nothing, "3.4x the false positives you
    budgeted for" tells them everything.
  * **Paired tests — only where a paired comparison is the actual question.** Comparing two
    strategies across the same languages is genuinely paired, so Wilcoxon signed-rank is
    appropriate (n = 14; normality is not assumed). Comparing unrelated quantities is not,
    and no test is offered for it.
  * **Non-significant results are results.** `interpret_correlation` reports them as such
    rather than as absence of evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- proportions

@dataclass(frozen=True)
class ProportionEstimate:
    successes: int
    n: int
    point: float
    ci_low: float
    ci_high: float
    method: str = "Wilson score"

    def format(self) -> str:
        return f"{self.point:.3f} [{self.ci_low:.3f}, {self.ci_high:.3f}] ({self.successes}/{self.n})"


def wilson_interval(successes: int, n: int, z: float = 1.959963985) -> ProportionEstimate:
    """95% Wilson score interval for a binomial proportion.

    Wilson rather than the normal approximation because violation rates sit at or near 0 and
    1 throughout this project, where the normal interval is badly miscalibrated and can
    extend outside [0, 1].
    """
    if n == 0:
        return ProportionEstimate(0, 0, float("nan"), float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return ProportionEstimate(successes, n, p,
                              float(max(0.0, centre - half)),
                              float(min(1.0, centre + half)))


# ------------------------------------------------------------------------ paired testing

@dataclass(frozen=True)
class PairedComparison:
    n_pairs: int
    median_difference: float
    ci_low: float
    ci_high: float
    n_favouring_a: int
    n_favouring_b: int
    n_ties: int
    p_value: float | None
    test: str
    note: str = ""

    def format(self) -> str:
        p = "n/a" if self.p_value is None else f"{self.p_value:.4f}"
        return (f"median Δ = {self.median_difference:+.3f} "
                f"[{self.ci_low:+.3f}, {self.ci_high:+.3f}], "
                f"{self.n_favouring_a}/{self.n_pairs} favour A, {self.test} p = {p}")


def paired_comparison(a: np.ndarray, b: np.ndarray, n_boot: int = 10000,
                      seed: int = 0, lower_is_better: bool = True) -> PairedComparison:
    """Compare two strategies measured on the SAME languages.

    Returns a paired-bootstrap CI on the median difference *and* a Wilcoxon signed-rank
    p-value. Both are reported because they answer different questions: the interval says how
    large the difference is and how precisely we know it; the test says whether a difference
    of this consistency would be surprising under no effect.

    Wilcoxon rather than a paired t-test because n is small (14 languages) and the
    differences are not plausibly normal. It is a genuinely paired question -- the same
    languages under two strategies -- which is what licenses a paired test at all.

    CAVEAT recorded in the returned note: languages are not independent (shared families,
    scripts, encoder capacity), so the effective sample size is below n. See `T-S4`.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired comparison needs equal shapes, got {a.shape} vs {b.shape}")

    diff = a - b
    n = diff.size
    rng = np.random.default_rng(seed)
    boot = np.array([np.median(diff[rng.integers(0, n, n)]) for _ in range(n_boot)])

    better = diff < 0 if lower_is_better else diff > 0
    worse = diff > 0 if lower_is_better else diff < 0

    p_value: float | None
    try:
        from scipy.stats import wilcoxon
        if np.allclose(diff, 0):
            p_value = 1.0
        else:
            p_value = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
    except Exception:
        p_value = None

    return PairedComparison(
        n_pairs=n,
        median_difference=float(np.median(diff)),
        ci_low=float(np.percentile(boot, 2.5)),
        ci_high=float(np.percentile(boot, 97.5)),
        n_favouring_a=int(np.sum(better)),
        n_favouring_b=int(np.sum(worse)),
        n_ties=int(np.sum(diff == 0)),
        p_value=p_value,
        test="Wilcoxon signed-rank",
        note=("Languages are not independent (shared families, scripts, encoder capacity); "
              "effective n is below the nominal pair count."),
    )


# --------------------------------------------------------------------------- correlation

@dataclass(frozen=True)
class CorrelationResult:
    rho: float
    p_value: float
    n: int
    significant: bool

    def format(self) -> str:
        verdict = "significant" if self.significant else "NOT significant"
        return f"ρ = {self.rho:+.3f}, p = {self.p_value:.4f}, n = {self.n} ({verdict})"


def spearman(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> CorrelationResult:
    """Spearman rank correlation. Rank-based because the relationships here are monotone but
    not assumed linear, and n is small enough that outliers would dominate Pearson."""
    from scipy.stats import spearmanr

    res = spearmanr(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    return CorrelationResult(float(res.statistic), float(res.pvalue),
                             int(np.asarray(x).size), bool(res.pvalue < alpha))


# ------------------------------------------------------------------- practical meaning

def practical_significance(realised_fpr: float, budget: float,
                           daily_volume: int = 1_000_000) -> str:
    """Translate an over-budget multiple into deployment consequences.

    Exists because `D-013` requires practical significance for every headline number. A
    reviewer and a practitioner both need to know what "median FPR 0.172 against a 0.05
    budget" costs in the only units that matter: items wrongly actioned.
    """
    multiple = realised_fpr / budget if budget else float("nan")
    intended = int(round(daily_volume * budget))
    actual = int(round(daily_volume * realised_fpr))
    excess = actual - intended
    return (f"{multiple:.1f}x the configured false-positive rate. "
            f"At {daily_volume:,} items/day an operator expecting ~{intended:,} "
            f"false positives would incur ~{actual:,} — about {excess:,} additional items "
            f"wrongly actioned each day.")


def over_budget_multiple(realised_fpr: np.ndarray, budget: float) -> np.ndarray:
    """Effect size in decision units: realised FPR as a multiple of the configured budget."""
    return np.asarray(realised_fpr, dtype=np.float64) / budget
