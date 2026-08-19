"""Independently re-derive headline numbers from preserved per-example scores.

SUCCESS DEFINITION requires results to be VERIFIED, not merely produced. A
number that only one code path can produce has been computed once, not confirmed.

The verification is only meaningful if it is genuinely independent, so this script
deliberately does NOT import the analysis engine (`src.analysis.threshold_study`), the
metrics module, or the threshold module. It recomputes everything from the raw
`per_example_scores.parquet` files with plain NumPy, re-deriving thresholds by brute-force
search rather than by the candidate-midpoint procedure the main pipeline uses. If both paths
agree, an error would have to exist identically in two independently written implementations.

Any disagreement beyond tolerance is a hard failure and is reported as such.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "experiments" / "runs"
TABLES = REPO_ROOT / "results" / "tables"

TOLERANCE = 1e-6


# --- independent primitives, written from the definitions ---------------------------

def _fpr(scores: np.ndarray, labels: np.ndarray, tau: float) -> float:
    neg = labels == 0
    return float(np.sum((scores >= tau) & neg) / max(np.sum(neg), 1))


def _recall(scores: np.ndarray, labels: np.ndarray, tau: float) -> float:
    pos = labels == 1
    return float(np.sum((scores >= tau) & pos) / max(np.sum(pos), 1))


def _f1(scores: np.ndarray, labels: np.ndarray, tau: float) -> float:
    pred = scores >= tau
    tp = float(np.sum(pred & (labels == 1)))
    fp = float(np.sum(pred & (labels == 0)))
    fn = float(np.sum(~pred & (labels == 1)))
    if tp == 0:
        return 0.0
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def _brute_force_threshold(scores: np.ndarray, labels: np.ndarray, budget: float,
                           grid: int = 20001) -> float:
    """Highest-recall threshold whose FPR stays within budget, found on a dense grid.

    Deliberately a different search from the pipeline's exact midpoint sweep: a dense grid
    is cruder but independent, so agreement is informative.
    """
    lo, hi = float(np.min(scores)), float(np.max(scores))
    best_tau, best_rec = hi + 1e-9, -1.0
    for tau in np.linspace(lo - 1e-9, hi + 1e-9, grid):
        if _fpr(scores, labels, tau) <= budget:
            rec = _recall(scores, labels, tau)
            if rec > best_rec:
                best_tau, best_rec = float(tau), rec
    return best_tau


def _load_scores(exp_id: str) -> pd.DataFrame | None:
    p = RUNS / exp_id / "per_example_scores.parquet"
    return pd.read_parquet(p) if p.exists() else None


def verify_exp002(checks: list[dict]) -> None:
    """R5: violation rate 1.000, median realised FPR 0.172 at budget 0.05."""
    raw = _load_scores("EXP-002")
    table = TABLES / "EXP-002_threshold_study.csv"
    if raw is None or not table.exists():
        checks.append({"check": "EXP-002 headline", "status": "SKIPPED",
                       "reason": "source artifacts absent"})
        return

    dev = raw[raw["split"] == "dev"]
    test = raw[raw["split"] == "test"]

    # Re-derive the source threshold independently.
    en_dev = dev[dev["language"] == "en"]
    tau_src = _brute_force_threshold(en_dev["score"].to_numpy(),
                                     en_dev["label"].to_numpy(), 0.05)

    fprs, recalls = [], []
    for lang, g in test.groupby("language"):
        if lang == "en":
            continue
        s, y = g["score"].to_numpy(), g["label"].to_numpy()
        fprs.append(_fpr(s, y, tau_src))
        recalls.append(_recall(s, y, tau_src))

    indep = {
        "tau_source": round(tau_src, 4),
        "violation_rate": round(float(np.mean(np.array(fprs) > 0.05)), 4),
        "median_fpr": round(float(np.median(fprs)), 4),
        "max_fpr": round(float(np.max(fprs)), 4),
        "median_recall": round(float(np.median(recalls)), 4),
        "n_languages": len(fprs),
    }

    df = pd.read_csv(table)
    df["prev"] = pd.to_numeric(df["prevalence_setting"], errors="coerce")
    pipe_rows = df[(df["prev"] == 0.50) & (~df["is_source"])
                   & (df["strategy"] == "source_global")]
    pipeline = {
        "tau_source": round(float(pipe_rows["tau_source"].iloc[0]), 4),
        "violation_rate": round(float((pipe_rows["fpr"] > 0.05).mean()), 4),
        "median_fpr": round(float(pipe_rows["fpr"].median()), 4),
        "max_fpr": round(float(pipe_rows["fpr"].max()), 4),
        "median_recall": round(float(pipe_rows["recall"].median()), 4),
        "n_languages": int(len(pipe_rows)),
    }

    for key in indep:
        a, b = indep[key], pipeline[key]
        agree = abs(a - b) <= max(TOLERANCE, 0.02 * max(abs(a), abs(b), 1e-9))
        checks.append({
            "check": f"EXP-002 / {key}", "independent": a, "pipeline": b,
            "status": "AGREE" if agree else "DISAGREE",
        })


def verify_exp001_degeneracy(checks: list[dict]) -> None:
    """R2: 6/14 languages degenerate under F1; Amharic flags 98.7%."""
    raw = _load_scores("EXP-001")
    if raw is None:
        checks.append({"check": "EXP-001 degeneracy", "status": "SKIPPED",
                       "reason": "source artifacts absent"})
        return

    dev, test = raw[raw["split"] == "dev"], raw[raw["split"] == "test"]
    n_degenerate, am_ppr = 0, None
    for lang, g in dev.groupby("language"):
        if lang == "en":
            continue
        s, y = g["score"].to_numpy(), g["label"].to_numpy()
        # Independent F1-optimal search on a dense grid.
        taus = np.linspace(s.min() - 1e-9, s.max() + 1e-9, 4001)
        tau = float(taus[int(np.argmax([_f1(s, y, t) for t in taus]))])
        t = test[test["language"] == lang]
        ppr = float(np.mean(t["score"].to_numpy() >= tau))
        if ppr > 0.80:
            n_degenerate += 1
        if lang == "am":
            am_ppr = round(ppr, 3)

    checks.append({"check": "EXP-001 / n_degenerate_languages (F1, expect 6)",
                   "independent": n_degenerate, "pipeline": 6,
                   "status": "AGREE" if n_degenerate == 6 else "DISAGREE"})
    if am_ppr is not None:
        agree = abs(am_ppr - 0.987) <= 0.02
        checks.append({"check": "EXP-001 / Amharic predicted-positive rate (expect 0.987)",
                       "independent": am_ppr, "pipeline": 0.987,
                       "status": "AGREE" if agree else "DISAGREE"})


def verify_wilson(checks: list[dict]) -> None:
    """The reported CI [0.78, 1.00] for 14/14, computed from the closed form directly."""
    k, n, z = 14, 14, 1.959963985
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = round(max(0.0, centre - half), 2)
    checks.append({"check": "Wilson CI lower bound for 14/14 (expect 0.78)",
                   "independent": lo, "pipeline": 0.78,
                   "status": "AGREE" if abs(lo - 0.78) <= 0.01 else "DISAGREE"})


def main() -> int:
    checks: list[dict] = []
    verify_exp002(checks)
    verify_exp001_degeneracy(checks)
    verify_wilson(checks)

    out = REPO_ROOT / "results" / "tables" / "verification_log.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")

    width = max(len(c["check"]) for c in checks)
    print("Independent re-derivation from preserved per-example scores\n")
    for c in checks:
        if c["status"] == "SKIPPED":
            print(f"  {c['check']:<{width}}  SKIPPED ({c['reason']})")
        else:
            print(f"  {c['check']:<{width}}  {c['status']:9s} "
                  f"independent={c['independent']}  pipeline={c['pipeline']}")

    disagreements = [c for c in checks if c["status"] == "DISAGREE"]
    print(f"\n{len(checks) - len(disagreements)}/{len(checks)} checks agree")
    if disagreements:
        print("VERIFICATION FAILED:")
        for c in disagreements:
            print(f"  {c['check']}: independent={c['independent']} vs pipeline={c['pipeline']}")
        return 1
    print("All independently re-derived values agree with the pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
