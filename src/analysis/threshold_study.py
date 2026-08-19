"""The threshold-transfer study: run every strategy over stored scores.

This module is pure arithmetic over per-example scores that some upstream experiment already
produced. Separating it from scoring matters scientifically: it guarantees that every
strategy sees *exactly* the same scores, so any difference between them is attributable to
the threshold alone and not to a different model, split, or preprocessing path.

It is also what makes the study cheap. Scoring is minutes; this is seconds, so the full
strategy grid, the label-budget sweep and the bootstrap can all be re-run freely without
recomputing a single forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..eval.metrics import auroc, operating_point
from ..eval.thresholds import (
    DEGENERATE_PPR_HIGH,
    DEGENERATE_PPR_LOW,
    Objective,
    bbse_prior,
    few_label_threshold,
    is_degenerate,
    prior_shift_transfer,
    quantile_match_threshold,
    sld_em_prior,
    tune_threshold,
)
from ..utils.seeding import DEFAULT_SEED, derived_seed

#: Label budgets swept for H3. Chosen to span the range a practitioner would actually
#: consider: a handful, a coffee-break's worth, an afternoon's worth, and a full day's worth.
DEFAULT_K_VALUES = (8, 16, 32, 64, 128, 256)

#: Bootstrap replicates for uncertainty on every reported comparison.
DEFAULT_N_BOOTSTRAP = 1000


@dataclass(frozen=True)
class LanguageScores:
    """Scores and labels for one language, already split into dev and test.

    dev  -- every threshold-selection method may look at this (label-free methods see only
            the scores; few-label methods see k labels; the full-label reference sees all)
    test -- evaluation only. No strategy may touch it.
    """

    language: str
    dev_scores: np.ndarray
    dev_labels: np.ndarray
    test_scores: np.ndarray
    test_labels: np.ndarray


def _strategy_thresholds(src: LanguageScores, tgt: LanguageScores, tau_source: float,
                         objective: Objective, target_fpr: float | None,
                         k_values: Sequence[int], seed: int,
                         n_few_label_repeats: int,
                         train_prior: float,
                         pooled: dict[str, float] | None = None,
                         worstcase: dict[str, float] | None = None,
                         ) -> dict[str, float | list[float]]:
    """Compute the threshold each strategy would choose for this target language.

    Every strategy is expressed as a threshold on the SAME raw score scale, so none of them
    gets to change the units and quietly become incomparable.

    `train_prior` is the class prior the HEAD WAS TRAINED AT, which is what its posteriors
    encode. It is deliberately a required argument rather than derived from the dev labels:
    deriving it from a resampled dev split is exactly the error that made SLD/EM report an
    estimated prior of 1.0 for the source language itself.
    """
    out: dict[str, float | list[float]] = {}

    # --- no information -----------------------------------------------------------------
    out["fixed_half"] = 0.5

    # --- labelled source only: THE STATUS QUO UNDER TEST --------------------------------
    out["source_global"] = tau_source

    # --- labelled data from OTHER languages, but not this one ---------------------------
    # Added after reviewer simulation #1 (R2-1), which correctly identified that comparing
    # an English-only threshold against per-language calibration omits the obvious middle
    # option: one global threshold calibrated on whatever multilingual data you do have.
    # Without it the headline risks resting on a straw-man baseline.
    #
    # LEAVE-ONE-LANGUAGE-OUT is essential here. Pooling data that includes the target
    # language would leak exactly the information the scenario says you lack, and would
    # flatter the baseline. Excluding it simulates the real case: labels in several
    # languages, none in the one you are about to serve.
    if pooled is not None:
        out["pooled_global"] = pooled.get(tgt.language, tau_source)
    if worstcase is not None:
        out["worstcase_global"] = worstcase.get(tgt.language, tau_source)

    # --- unlabelled target text only ----------------------------------------------------
    out["quantile_match"] = quantile_match_threshold(
        src.dev_scores, tau_source, tgt.dev_scores)

    # Prior-shift methods: estimate the prior for BOTH source and target from the same
    # reference (the head's training prior), then transfer. Treating the two symmetrically
    # gives the required sanity property -- identity when target == source.
    prior_hat_source, _ = sld_em_prior(src.dev_scores, train_prior)
    prior_hat_target, _ = sld_em_prior(tgt.dev_scores, train_prior)
    out["sld_em"] = prior_shift_transfer(tau_source, train_prior,
                                         prior_hat_source, prior_hat_target)

    bbse_source = bbse_prior(src.dev_scores, src.dev_labels, tau_source, src.dev_scores)
    bbse_target = bbse_prior(src.dev_scores, src.dev_labels, tau_source, tgt.dev_scores)
    out["bbse"] = prior_shift_transfer(tau_source, train_prior, bbse_source, bbse_target)

    # --- k labelled target examples ------------------------------------------------------
    # Repeated with different draws because the whole point of small k is that the answer is
    # variable; reporting a single draw would understate the risk a practitioner takes on.
    for k in k_values:
        for stratified in (True, False):
            tag = "strat" if stratified else "rand"
            draws = [
                few_label_threshold(
                    tgt.dev_scores, tgt.dev_labels, k,
                    rng=np.random.default_rng(derived_seed(seed, "few", tgt.language, k, tag, r)),
                    objective=objective, target_fpr=target_fpr,
                    fallback_threshold=tau_source, stratified=stratified,
                )
                for r in range(n_few_label_repeats)
            ]
            out[f"few_label_{k}_{tag}"] = draws

    # --- fully labelled target calibration set (achievable upper bound) -------------------
    out["target_full"] = tune_threshold(
        tgt.dev_scores, tgt.dev_labels, objective=objective, target_fpr=target_fpr)

    # Extra reference, reported separately and never used as the headline oracle: the best
    # achievable ON TEST. No deployer can reach it; it bounds how much of the residual gap is
    # dev/test sampling noise rather than genuine method shortfall.
    out["test_oracle_unachievable"] = tune_threshold(
        tgt.test_scores, tgt.test_labels, objective=objective, target_fpr=target_fpr)

    return out


def _leave_one_out_global_thresholds(
        corpus_scores: dict[str, LanguageScores], objective: Objective,
        target_fpr: float | None) -> tuple[dict[str, float], dict[str, float]]:
    """Global thresholds calibrated on every language EXCEPT the one being served.

    Two deployer strategies that reviewer simulation #1 (R2-1) correctly identified as
    missing from the original baseline set:

    * `pooled`     -- one threshold tuned on dev data POOLED across the other languages.
                      The obvious thing a deployer with some multilingual labels would do.
    * `worstcase`  -- the most conservative of the other languages' own thresholds
                      (the largest, since a larger threshold fires less and so cannot
                      overshoot a false-positive budget). The cautious deployer's choice.

    Leave-one-out is not a technicality. Including the target language would leak precisely
    the labels the low-resource scenario says are unavailable, and would make both baselines
    look better than any deployer could achieve for a genuinely new language.
    """
    languages = sorted(corpus_scores)
    per_language_tau = {
        lang: tune_threshold(ls.dev_scores, ls.dev_labels,
                             objective=objective, target_fpr=target_fpr)
        for lang, ls in corpus_scores.items()
    }

    pooled: dict[str, float] = {}
    worstcase: dict[str, float] = {}
    for held_out in languages:
        others = [lang for lang in languages if lang != held_out]
        if not others:
            continue
        pooled_scores = np.concatenate([corpus_scores[l].dev_scores for l in others])
        pooled_labels = np.concatenate([corpus_scores[l].dev_labels for l in others])
        pooled[held_out] = tune_threshold(pooled_scores, pooled_labels,
                                          objective=objective, target_fpr=target_fpr)
        worstcase[held_out] = float(max(per_language_tau[l] for l in others))

    return pooled, worstcase


def _bootstrap_metric_ci(scores: np.ndarray, labels: np.ndarray, threshold: float,
                         metric: str, n_boot: int, seed: int) -> tuple[float, float]:
    """Percentile bootstrap CI for one metric at a FIXED threshold.

    The threshold is held fixed across replicates because the quantity of interest is
    "how well does this already-chosen operating point perform on new data", not
    "how variable is threshold selection" -- those are different questions and conflating
    them would inflate the interval for reasons unrelated to the claim.
    """
    rng = np.random.default_rng(seed)
    n = scores.size
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        op = operating_point(scores[idx], labels[idx], threshold)
        vals[b] = getattr(op, metric)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def run_threshold_study(corpus_scores: dict[str, LanguageScores], source_language: str,
                        objective: Objective = "f1", target_fpr: float | None = None,
                        k_values: Sequence[int] = DEFAULT_K_VALUES,
                        n_few_label_repeats: int = 25,
                        n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
                        seed: int = DEFAULT_SEED,
                        bootstrap_strategies: Iterable[str] = ("source_global", "target_full"),
                        train_prior: float = 0.5,
                        verbose: bool = True) -> pd.DataFrame:
    """Evaluate every threshold strategy on every language.

    Returns one row per (language, strategy) with the chosen threshold and the test-set
    operating point it produces. Few-label strategies additionally carry the spread across
    repeated draws, since their variability is itself a result.
    """
    if source_language not in corpus_scores:
        raise KeyError(f"Source language {source_language!r} not present in scores")

    src = corpus_scores[source_language]
    tau_source = tune_threshold(src.dev_scores, src.dev_labels,
                                objective=objective, target_fpr=target_fpr)

    pooled, worstcase = _leave_one_out_global_thresholds(
        corpus_scores, objective, target_fpr)

    rows: list[dict] = []
    for lang in sorted(corpus_scores):
        tgt = corpus_scores[lang]
        thresholds = _strategy_thresholds(
            src, tgt, tau_source, objective, target_fpr, k_values, seed,
            n_few_label_repeats, train_prior, pooled, worstcase)

        test_auroc = auroc(tgt.test_scores, tgt.test_labels)
        test_prevalence = float(np.mean(tgt.test_labels == 1))
        dev_prevalence = float(np.mean(tgt.dev_labels == 1))

        for strategy, value in thresholds.items():
            if isinstance(value, list):
                # Few-label: evaluate each draw, then summarise across draws.
                ops = [operating_point(tgt.test_scores, tgt.test_labels, t) for t in value]
                primary = float(np.mean([getattr(op, objective_metric(objective)) for op in ops]))
                row = {
                    "language": lang,
                    "strategy": strategy,
                    "threshold": float(np.mean(value)),
                    "threshold_sd": float(np.std(value)),
                    "n_draws": len(value),
                    "f1": float(np.mean([op.f1 for op in ops])),
                    "f1_sd": float(np.std([op.f1 for op in ops])),
                    "fpr": float(np.mean([op.fpr for op in ops])),
                    "fnr": float(np.mean([op.fnr for op in ops])),
                    "precision": float(np.mean([op.precision for op in ops])),
                    "recall": float(np.mean([op.recall for op in ops])),
                    "predicted_positive_rate": float(np.mean([op.predicted_positive_rate for op in ops])),
                    "objective_value": primary,
                }
            else:
                op = operating_point(tgt.test_scores, tgt.test_labels, float(value))
                row = {
                    "language": lang,
                    "strategy": strategy,
                    "threshold": float(value),
                    "threshold_sd": 0.0,
                    "n_draws": 1,
                    "f1": op.f1,
                    "f1_sd": 0.0,
                    "fpr": op.fpr,
                    "fnr": op.fnr,
                    "precision": op.precision,
                    "recall": op.recall,
                    "predicted_positive_rate": op.predicted_positive_rate,
                    "objective_value": getattr(op, objective_metric(objective)),
                }

                if strategy in bootstrap_strategies and n_bootstrap > 0:
                    lo, hi = _bootstrap_metric_ci(
                        tgt.test_scores, tgt.test_labels, float(value), "f1",
                        n_bootstrap, derived_seed(seed, "boot", lang, strategy))
                    row["f1_ci_low"], row["f1_ci_high"] = lo, hi

            # Degeneracy flag (D-011). Judged on the TEST scores, which is where the metric
            # is reported. Never used to silently drop a row -- it exists so that a
            # flag-everything threshold cannot be averaged into a headline unnoticed.
            row["is_degenerate"] = bool(
                row["predicted_positive_rate"] > DEGENERATE_PPR_HIGH
                or row["predicted_positive_rate"] < DEGENERATE_PPR_LOW
            )
            row.update({
                "is_source": lang == source_language,
                "test_auroc": test_auroc,
                "test_prevalence": test_prevalence,
                "dev_prevalence": dev_prevalence,
                "n_dev": int(tgt.dev_scores.size),
                "n_test": int(tgt.test_scores.size),
                "tau_source": tau_source,
                "source_language": source_language,
                "objective": objective,
            })
            rows.append(row)

        if verbose:
            print(f"  {lang:10s} AUROC={test_auroc:.3f}  "
                  f"tau_full={thresholds['target_full']:.3f}  tau_src={tau_source:.3f}")

    df = pd.DataFrame(rows)

    # Gap relative to the achievable upper bound: THE headline quantity.
    full = (df[df["strategy"] == "target_full"]
            .set_index("language")["f1"].rename("f1_target_full"))
    df = df.join(full, on="language")
    df["f1_gap_to_full"] = df["f1_target_full"] - df["f1"]

    return df


def objective_metric(objective: Objective) -> str:
    """Map a tuning objective to the metrics attribute that reports it."""
    return {
        "f1": "f1",
        "balanced_accuracy": "balanced_accuracy",
        "youden": "balanced_accuracy",  # Youden's J is a monotone recoding of balanced acc.
        "target_fpr": "recall",         # under an FPR budget, recall is what you are buying
    }[objective]
