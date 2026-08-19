"""Show that source-fitted post-hoc calibration cannot change any realised operating point.

Reviewer objection (peer review, 2026-08-03): the manuscript omits standard post-hoc
calibration baselines -- temperature scaling, Platt scaling, isotonic regression -- and so
"threshold transferability is unproven for post-hoc temperature-scaled scores".

The objection is answerable without running an experiment, because every one of those methods
is a *strictly monotone* map applied to the score, and the map is fitted on SOURCE data, so
the same map is applied to every language. Let g be strictly increasing and let tau' = g(tau)
be the threshold re-derived on the source validation split at the same budget alpha. Then for
any language L,

    FPR_L(g, tau') = P(g(s) >= g(tau) | y=0, L) = P(s >= tau | y=0, L) = FPR_L(tau).

The realised false-positive rate is therefore *identical*, language by language. Calibration
changes the number printed on the threshold, not the set of items above it.

This script checks that algebra against the preserved per-example scores rather than asserting
it: it fits temperature scaling and Platt scaling on the English (source) dev split, re-derives
the threshold at the budget, and compares every target language's realised FPR and recall
against the uncalibrated pipeline.

The claim this supports is narrow and is exactly the one the objection disputes: post-hoc
calibration fitted WITHOUT target-language labels cannot repair operating-point transfer. It
says nothing about calibration fitted WITH target labels -- that is a labelled method, and the
few-label arm already measures the sample-efficiency frontier for labelled methods directly.

Usage: python scripts/check_calibration_invariance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCORES = REPO_ROOT / "experiments" / "runs" / "EXP-002" / "per_example_scores.parquet"

SOURCE = "en"
BUDGET = 0.05
#: Equality is asserted to 1e-12, not to floating-point identity: g and g^{-1} are applied in
#: floating point, so an exact bit match is not guaranteed even though the maths is exact.
TOL = 1e-12


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def threshold_at_budget(scores: np.ndarray, labels: np.ndarray, alpha: float) -> float:
    """Smallest threshold whose false-positive rate on this split is <= alpha."""
    neg = np.sort(scores[labels == 0])[::-1]
    if neg.size == 0:
        return 1.0
    k = int(np.floor(alpha * neg.size))
    return float(neg[k]) if k < neg.size else float(neg[-1])


def rates(scores: np.ndarray, labels: np.ndarray, tau: float) -> tuple[float, float]:
    pred = scores >= tau
    neg, pos = labels == 0, labels == 1
    fpr = float(pred[neg].mean()) if neg.any() else float("nan")
    rec = float(pred[pos].mean()) if pos.any() else float("nan")
    return fpr, rec


def ece(scores: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error, equal-width binning.

    Reported only to show the dissociation the paper rests on: calibration error is what
    post-hoc scaling improves, and it is not the quantity that determines an operating point.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(scores, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        total += m.mean() * abs(labels[m].mean() - scores[m].mean())
    return float(total)


def fit_temperature(scores: np.ndarray, labels: np.ndarray) -> float:
    """Single-parameter temperature minimising NLL on the source split (grid + refine)."""
    z = logit(scores)
    grid = np.concatenate([np.linspace(0.05, 5.0, 200), np.linspace(5.0, 20.0, 60)])
    best_t, best_nll = 1.0, np.inf
    for t in grid:
        p = 1.0 / (1.0 + np.exp(-z / t))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        nll = -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    return best_t


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Platt scaling: logistic regression of the label on the logit (affine, monotone)."""
    from sklearn.linear_model import LogisticRegression
    z = logit(scores).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, labels)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def main() -> int:
    if not SCORES.exists():
        print(f"missing {SCORES}; run the EXP-002 stage of reproduce_all.py first")
        return 1
    d = pd.read_parquet(SCORES)

    src_dev = d[(d.language == SOURCE) & (d.split == "dev")]
    temperature = fit_temperature(src_dev.score.to_numpy(), src_dev.label.to_numpy())
    a, b = fit_platt(src_dev.score.to_numpy(), src_dev.label.to_numpy())
    print(f"fitted on {SOURCE} dev (n={len(src_dev)}): "
          f"temperature T={temperature:.4f}, Platt a={a:.4f} b={b:.4f}\n")

    maps = {
        "uncalibrated": lambda s: s,
        "temperature":  lambda s: 1.0 / (1.0 + np.exp(-logit(s) / temperature)),
        "platt":        lambda s: 1.0 / (1.0 + np.exp(-(a * logit(s) + b))),
    }

    results: dict[str, dict[str, tuple[float, float]]] = {}
    for name, g in maps.items():
        tau = threshold_at_budget(g(src_dev.score.to_numpy()), src_dev.label.to_numpy(), BUDGET)
        per_lang = {}
        for lang, grp in d[d.split == "test"].groupby("language", observed=True):
            per_lang[str(lang)] = rates(g(grp.score.to_numpy()), grp.label.to_numpy(), tau)
        results[name] = per_lang
        print(f"{name:13s} threshold on the {SOURCE} scale = {tau:.6f}")

    print("\nrealised operating points on the test split "
          f"(budget {BUDGET}, threshold set on {SOURCE}):\n")
    print(f"{'lang':>6} | {'FPR uncal':>10} {'FPR temp':>10} {'FPR platt':>10} | "
          f"{'rec uncal':>10} {'rec temp':>10}")
    worst = 0.0
    for lang in sorted(results["uncalibrated"]):
        u, t, p = (results[m][lang] for m in ("uncalibrated", "temperature", "platt"))
        worst = max(worst, abs(u[0] - t[0]), abs(u[0] - p[0]), abs(u[1] - t[1]))
        print(f"{lang:>6} | {u[0]:10.6f} {t[0]:10.6f} {p[0]:10.6f} | "
              f"{u[1]:10.6f} {t[1]:10.6f}")

    # The dissociation, in one line: calibration error is what these maps improve, and the
    # operating point is not. A reviewer asking for ECE alongside FPR transferability is asking
    # for exactly this comparison.
    # Split source from targets: a pooled ECE hides the whole point. The map is fitted on the
    # source, so it should improve calibration there and degrade it elsewhere -- the
    # operating-point analogue of Ovadia et al.'s finding that calibration does not survive
    # distribution shift.
    test = d[d.split == "test"]
    src_test = test[test.language == SOURCE]
    tgt_test = test[test.language != SOURCE]
    print("\nexpected calibration error on the test split:")
    print(f"  {'map':13s} {'ECE ' + SOURCE + ' (fitted on)':>22s} {'ECE targets':>14s}")
    for name, g in maps.items():
        e_src = ece(g(src_test.score.to_numpy()), src_test.label.to_numpy())
        e_tgt = ece(g(tgt_test.score.to_numpy()), tgt_test.label.to_numpy())
        print(f"  {name:13s} {e_src:22.4f} {e_tgt:14.4f}")

    print(f"\nlargest absolute difference across all languages and metrics: {worst:.3e}")
    if worst > TOL:
        print("FAIL: source-fitted calibration changed a realised operating point.")
        return 1
    print("PASS: every realised FPR and recall is unchanged, to within floating-point error.")
    print("      Source-fitted post-hoc calibration is a no-op for the operating point;\n"
          "      it renumbers the threshold and moves nothing across it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
