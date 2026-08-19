"""Claim-to-evidence audit: every reported quantity checked against the artifact that made it.

WHAT THIS IS. The permanent, repository-resident reconstruction of the claim-to-evidence audit
that this project ran from P8 through P12 (29 -> 42 -> 47 -> 58 checks). The original
implementation lived outside the repository and was lost. See RECONSTRUCTION below: this is a
conservative rebuild covering the recoverable claim/evidence classes, NOT a byte-for-byte
restoration, and it does not claim to be one.

HOW IT WORKS. Each check locates a claim in the manuscript with a regex that *captures the
printed number*, recomputes the same quantity from the preserved artifact, and compares the two
at the precision the manuscript prints. Manuscript values are therefore never hard-coded here:
if the prose changes, the check still compares prose against artifact. That is what makes these
predicates over propositions rather than string searches.

DESIGN RULES, each answering a failure this project actually had:

  * A missing artifact is a hard failure. There is no SKIP that can be mistaken for a pass.
  * Attribution is checked, not just existence: a float cited by a section must come from the
    experiment that section reports (D-018). Verifying that a number exists somewhere is what
    let a figure from the wrong experiment survive three audits.
  * Statistical rules are imported from the pipeline, never re-implemented. A re-implemented
    threshold rule once disagreed with the main table through a tie-break convention alone.
  * Numbers the manuscript quotes from a derived analysis are checked against that analysis's
    *emitted* verdict file, not recomputed by a second convention. Agreement between two
    conventions is not evidence; agreement with the source of the number is.
  * SELF-00 proves the harness can fail. A validator that has never failed is unverified, and
    this project shipped one that silently became a no-op after a rename.
  * Coverage limits are printed, not implied. A passing audit is not evidence that untested
    defect classes are absent.

SCOPE. This audit covers NUMERIC and ATTRIBUTION claims. Universally and ordinally quantified
prose ("all", "every", "the three strongest", "more than 99%") is the remit of
`scripts/check_quantified_claims.py`, which exists because this class of audit structurally
cannot see quantifier defects. Build, citation and float integrity is `check_manuscript.py`.
Independent re-derivation from raw scores is `verify_results.py`. The four are complementary
and all four are reproduce_all stages.

Usage
-----
    python scripts/check_claims.py            # run the audit
    python scripts/check_claims.py --verbose  # show every check, not only failures
    python scripts/check_claims.py --list     # list check IDs without running
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SECTIONS = REPO_ROOT / "manuscript" / "sections"
MTABLES = REPO_ROOT / "manuscript" / "tables"
TABLES = REPO_ROOT / "results" / "tables"
FIGURES = REPO_ROOT / "results" / "figures"

BUDGET = 0.05


# ----------------------------------------------------------------- artifact loading

class MissingArtifact(RuntimeError):
    """Raised when evidence an audit needs is absent. Never downgraded to a skip."""


def _rel(path: Path) -> str:
    """Repo-relative display path, tolerating paths outside the repo (test fixtures)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _require(path: Path) -> Path:
    if not path.exists():
        raise MissingArtifact(f"required artifact missing: {_rel(path)}")
    return path


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(_require(TABLES / name))


def load_json(path: Path) -> dict | list:
    return json.loads(_require(path).read_text(encoding="utf-8"))


def manuscript_body() -> str:
    """All section prose, LaTeX comments stripped, whitespace normalised."""
    out = []
    for p in sorted(SECTIONS.glob("*.tex")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("%"):
                continue
            out.append(re.sub(r"(?<!\\)%.*$", "", line))
    return " ".join(" ".join(out).split())


def section_text(name: str) -> str:
    p = _require(SECTIONS / name)
    keep = [re.sub(r"(?<!\\)%.*$", "", l)
            for l in p.read_text(encoding="utf-8").splitlines()
            if not l.lstrip().startswith("%")]
    return " ".join(" ".join(keep).split())


# ----------------------------------------------------------------- comparison primitives

def agrees(printed: str, truth: float) -> bool:
    """Compare a manuscript-printed number with the artifact value at the printed precision.

    The manuscript legitimately rounds. Requiring exact equality would fail on correct
    rounding; requiring a fixed tolerance would pass on genuinely wrong values printed to
    many digits. Half-a-unit-in-the-last-place is the only rule that does neither.
    """
    printed = printed.strip().lstrip("+")
    dp = len(printed.split(".")[1]) if "." in printed else 0
    return abs(float(printed) - float(truth)) <= 0.5 * 10 ** (-dp) + 1e-12


def captured(body: str, pattern: str) -> list[str] | None:
    m = re.search(pattern, body)
    return list(m.groups()) if m else None


# ----------------------------------------------------------------- the audit harness

class Audit:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, bool, str, str]] = []

    def check(self, cid: str, category: str, ok: bool, evidence: str, detail: str) -> None:
        self.rows.append((cid, category, bool(ok), evidence, detail))

    def numeric(self, cid: str, category: str, body: str, pattern: str,
                truth: float | list[float], evidence: str, note: str = "") -> None:
        """Extract the printed value(s), compare against the artifact at printed precision."""
        g = captured(body, pattern)
        truths = truth if isinstance(truth, (list, tuple)) else [truth]
        if g is None:
            self.check(cid, category, False, evidence,
                       f"claim not found in manuscript (pattern did not match){' ; ' + note if note else ''}")
            return
        if len(g) != len(truths):
            self.check(cid, category, False, evidence,
                       f"pattern captured {len(g)} value(s), artifact supplies {len(truths)}")
            return
        ok = all(agrees(p, t) for p, t in zip(g, truths))
        self.check(cid, category, ok, evidence,
                   f"manuscript {g} vs artifact {[round(float(t), 6) for t in truths]}"
                   + (f" ; {note}" if note else ""))

    @property
    def failed(self):
        return [r for r in self.rows if not r[2]]


# ----------------------------------------------------------------- checks

def run_audit() -> Audit:
    A = Audit()
    body = manuscript_body()

    # SELF-00. The harness must be able to fail. A checker that has never failed is
    # unverified; this project shipped a validator that silently became a no-op.
    A.check("SELF-00", "harness",
            (agrees("0.173", 0.17253) and not agrees("0.10", 0.5)
             and captured("x = 0.5", r"x = ([\d.]+)") == ["0.5"]
             and captured("nothing here", r"z = ([\d.]+)") is None),
            "internal", "comparison primitives accept correct and reject incorrect values")

    # ---------------------------------------------------------- A. controlled arm, EXP-002
    e2 = load_csv("EXP-002_threshold_study.csv")
    base = e2[(e2.fpr_budget == BUDGET) & (e2.prevalence_setting == 0.5) & (~e2.is_source)]
    sg = base[base.strategy == "source_global"]
    ev2 = "results/tables/EXP-002_threshold_study.csv"

    A.numeric("A-01", "headline", body,
              r"violation rate is \$([\d.]+)\$", float((sg.fpr > BUDGET).mean()), ev2)
    A.numeric("A-02", "headline", body,
              r"median realised false-positive rate is \$([\d.]+)\$", float(sg.fpr.median()), ev2)
    A.numeric("A-03", "headline", body,
              r"or\s+\$\\mathbf\{([\d.]+)\\times\}\$ the configured budget",
              float(sg.fpr.median() / BUDGET), ev2)
    A.numeric("A-04", "headline", body,
              r"the maximum is \$([\d.]+)\$ in Tatar", float(sg.fpr.max()), ev2,
              note=f"max language={sg.loc[sg.fpr.idxmax(), 'language']}")
    A.check("A-05", "headline", sg.loc[sg.fpr.idxmax(), "language"] == "tt", ev2,
            "the maximum-overshoot language is Tatar")
    A.check("A-06", "headline", sg.loc[sg.fpr.idxmin(), "language"] == "hi", ev2,
            "the minimum-overshoot language is Hindi")
    for cid, lang, name in (("A-07", "fr", "French"), ("A-08", "es", "Spanish"),
                            ("A-09", "de", "German")):
        v = float(sg[sg.language == lang].fpr.iloc[0]) / BUDGET
        A.numeric(cid, "headline", body,
                  rf"{name} (?:runs |exceeds the budget by )?at\s*\$([\d.]+)\\times\$", v, ev2)
    A.numeric("A-10", "headline", body,
              r"smallest\s+overshoot anywhere is Hindi at \$([\d.]+)\\times\$",
              float(sg[sg.language == "hi"].fpr.iloc[0]) / BUDGET, ev2)
    A.numeric("A-11", "headline", body,
              r"median recall collapses from \$([\d.]+)\$ to \$([\d.]+)\$",
              [float(sg.recall.median()),
               float(base[base.strategy == "worstcase_global"].recall.median())], ev2)

    # threshold divergence: the correction a deployer with target labels would apply
    tf2 = base[base.strategy == "target_full"]
    dtau2 = float((tf2.threshold - tf2.tau_source).abs().median())
    A.numeric("A-12", "headline", body,
              r"controlled arm \(median \$\|\\Delta\\tau\| = ([\d.]+)\$\)", dtau2, ev2)

    # ---------------------------------------------------------- B. remedies, paired tests
    paired = load_json(TABLES / "paired_comparisons.json")
    evp = "results/tables/paired_comparisons.json"
    pool = next(p for p in paired if p["strategy_a"] == "pooled_global"
                and p["strategy_b"] == "source_global")
    tfp = next(p for p in paired if p["strategy_a"] == "target_full"
               and p["strategy_b"] == "pooled_global")

    A.numeric("B-01", "remedies", body,
              r"cuts the violation rate from \$([\d.]+)\$ to \$([\d.]+)\$",
              [pool["violation_rate_b"], pool["violation_rate_a"]], evp)
    A.numeric("B-02", "remedies", body,
              r"median over-budget multiple from \$([\d.]+)\\times\$ to \$([\d.]+)\\times\$",
              [pool["effect_in_decision_units"]["median_over_budget_multiple_b"],
               pool["effect_in_decision_units"]["median_over_budget_multiple_a"]], evp)
    A.numeric("B-03", "remedies", body,
              r"median paired difference in false-positive\s+rate \$([-\d.]+)\$",
              pool["fpr"]["median_difference"], evp)
    A.numeric("B-04", "remedies", body,
              r"\(95\\% CI \$\[([-\d.]+), ([-\d.]+)\]\$\), Wilcoxon signed-rank \$p = ([\d.]+)\$",
              [pool["fpr"]["ci95"][0], pool["fpr"]["ci95"][1], pool["fpr"]["wilcoxon_p"]], evp)
    pg = base[base.strategy == "pooled_global"]
    A.numeric("B-05", "remedies", body,
              r"worst\s+case nonetheless still runs at \$([\d.]+)\$, or \$([\d.]+)\\times\$",
              [float(pg.fpr.max()), float(pg.fpr.max() / BUDGET)], ev2)
    A.numeric("B-06", "remedies", body,
              r"false-positive\s*rate is \$\+([\d.]+)\$ \(95\\% CI \$\[([-\d.]+), \+([\d.]+)\]\$, \$p = ([\d.]+)\$",
              [tfp["fpr"]["median_difference"], tfp["fpr"]["ci95"][0],
               tfp["fpr"]["ci95"][1], tfp["fpr"]["wilcoxon_p"]], evp)
    A.numeric("B-07", "remedies", body,
              r"recall \$\+([\d.]+)\$ \(95\\% CI \$\[([-\d.]+), \+([\d.]+)\]\$, \$p = ([\d.]+)\$",
              [tfp["recall"]["median_difference"], tfp["recall"]["ci95"][0],
               tfp["recall"]["ci95"][1], tfp["recall"]["wilcoxon_p"]], evp)
    A.numeric("B-08", "remedies", body,
              r"violation rate \(\$([\d.]+)\$ against \$([\d.]+)\$\)",
              [tfp["violation_rate_a"], tfp["violation_rate_b"]], evp)
    # The prose now names the estimator explicitly ("at the median of the paired per-language
    # differences"), because the difference of the two medians in Table 2 is 0.263, not 0.245,
    # and an unlabelled 0.245 read as a difference of medians. The audited quantity is
    # unchanged: it was and remains the median paired difference.
    A.numeric("B-09", "remedies", body,
              r"Pooling reduces recall by \$([\d.]+)\$ at the median",
              abs(pool["recall"]["median_difference"]), evp)
    wc = base[base.strategy == "worstcase_global"]
    A.numeric("B-10", "remedies", body,
              r"does achieve compliance \(\$([\d.]+)\$ violation\)",
              float((wc.fpr > BUDGET).mean()), ev2)
    A.numeric("B-11", "remedies", body, r"an \$([\d.]+)\\%\$ relative loss",
              100 * (1 - float(wc.recall.median()) / float(sg.recall.median())), ev2)

    # ---------------------------------------------------------- C. label-free arm
    for cid, strat, pat in (
        ("C-01", "quantile_match", r"violating in \$([\d.]+)\$ of languages at every budget"),
        ("C-02", "quantile_match", r"median realised rate of \$([\d.]+)\$"),
    ):
        s = base[base.strategy == strat]
        truth = float((s.fpr > BUDGET).mean()) if cid == "C-01" else float(s.fpr.median())
        A.numeric(cid, "label-free", body, pat, truth, ev2)
    A.numeric("C-03", "label-free", body,
              r"fail at \$([\d.]+)\$ \(EM\) and\s*\$([\d.]+)\$ \(black-box shift estimation\)",
              [float((base[base.strategy == "sld_em"].fpr > BUDGET).mean()),
               float((base[base.strategy == "bbse"].fpr > BUDGET).mean())], ev2)

    # ---------------------------------------------------------- D. parallel corpus, EXP-006
    e6 = load_csv("EXP-006_threshold_study.csv")
    ev6 = "results/tables/EXP-006_threshold_study.csv"
    tf6 = e6[(e6.strategy == "target_full") & (~e6.is_source)]
    sg6 = e6[(e6.strategy == "source_global") & (~e6.is_source)]
    d6 = (tf6.threshold - tf6.tau_source).abs()

    A.numeric("D-01", "parallel", body,
              r"per-language optimal thresholds still span\s*\$([\d.]+)\$--\$([\d.]+)\$",
              [float(tf6.threshold.min()), float(tf6.threshold.max())], ev6)
    A.numeric("D-02", "parallel", body, r"a \$([\d.]+)\\times\$ range",
              float(tf6.threshold.max() / tf6.threshold.min()), ev6)
    A.numeric("D-03", "parallel", body,
              r"and \$([\d.]+)\\%\$ of languages sit more than \$0.05\$ from\s+the source threshold",
              100 * float((d6 > 0.05).mean()), ev6)
    A.numeric("D-04", "parallel", body,
              r"\(median \$\|\\Delta\\tau\| = ([\d.]+)\$, maximum \$([\d.]+)\$\)",
              [float(d6.median()), float(d6.max())], ev6)
    A.numeric("D-05", "parallel", body,
              r"it violates the\s+budget \$([\d.]+)\\%\$ of the time against \$([\d.]+)\\%\$ for the source-tuned",
              [100 * float((tf6.fpr > BUDGET).mean()), 100 * float((sg6.fpr > BUDGET).mean())], ev6)
    A.numeric("D-06", "parallel", body, r"with about (\d+) negatives per language",
              float(round(tf6.n_dev.median() * (1 - tf6.dev_prevalence.median()))), ev6)
    A.check("D-07", "parallel", int(tf6.language.nunique()) == 173
            and f"$n = {int(tf6.language.nunique())}$ target languages" in body, ev6,
            f"parallel arm has {tf6.language.nunique()} target languages")

    # Correlations use the project's declared conventions: Spearman with the Bonett-Wright
    # standard error. Both are recomputed here from the same preserved table the manuscript
    # reports, not from a second pipeline.
    from scipy.stats import spearmanr

    def bw_ci(rho: float, n: int) -> tuple[float, float]:
        z, se = np.arctanh(rho), 1.06 / np.sqrt(n - 3)
        return tuple(np.tanh([z - 1.959964 * se, z + 1.959964 * se]))

    r_dt, _ = spearmanr(tf6.test_auroc, d6)
    lo, hi = bw_ci(r_dt, len(d6))
    A.numeric("D-08", "parallel", body,
              r"negatively associated with AUROC \(\$\\rho = ([-\d.]+)\$, \$95\\%\$ CI\s*\$\[([-\d.]+), ?([-\d.]+)\]\$\)",
              [r_dt, lo, hi], ev6, note="Spearman + Bonett-Wright SE, the declared convention")
    mult6 = sg6.fpr / BUDGET
    r_ob, _ = spearmanr(sg6.test_auroc, mult6)
    lo2, hi2 = bw_ci(r_ob, len(sg6))
    A.numeric("D-09", "parallel", body,
              r"opposite direction \(\$\\rho = \+([\d.]+)\$, \$95\\%\$ CI\s*\$\[\+([\d.]+), \+([\d.]+)\]\$\)",
              [r_ob, lo2, hi2], ev6)

    # ---------------------------------------------------------- E. deployed classifiers
    arm = load_json(TABLES / "arm_a_claims.json")
    eva = "results/tables/arm_a_claims.json"
    db, xl, rc, cc = (arm["distilbert"], arm["xlmr"],
                      arm["distilbert_recall_claim"], arm["cross_classifier"])

    A.numeric("E-01", "arm-a", body,
              r"median realised false-positive rate \$([\d.]+)\$, violation rate \$([\d.]+)\$",
              [float(pd.read_csv(TABLES / "EXP-008-citizenlab_arm_a.csv")
                     .query("strategy == 'source_global' and not is_source").fpr.median()),
               db["violation_rate"]], eva)
    A.numeric("E-02", "arm-a", body,
              r"violation rate \$([\d.]+)\$ against \$([\d.]+)\$, median realised false-positive rate \$([\d.]+)\\times\$\s*budget against \$([\d.]+)\\times\$",
              [xl["violation_rate"], db["violation_rate"],
               xl["median_over_budget_multiple"], db["median_over_budget_multiple"]], eva)
    A.numeric("E-03", "arm-a", body,
              r"Median threshold divergence is \$([\d.]+)\$ and\s*\$([\d.]+)\$",
              [_median_dtau("EXP-008-citizenlab_arm_a.csv"),
               _median_dtau("EXP-009_arm_a.csv")], eva)
    A.numeric("E-04", "arm-a", body,
              r"median\s+recall collapses from \$([\d.]+)\$ in English to \$\\mathbf\{([\d.]+)\}\$",
              [db["source_recall"], db["median_recall"]], eva)
    A.numeric("E-05", "arm-a", body,
              r"raise median recall from \$([\d.]+)\$ to \$([\d.]+)\$",
              [db["median_recall"],
               float(pd.read_csv(TABLES / "EXP-008-citizenlab_arm_a.csv")
                     .query("strategy == 'target_full' and not is_source").recall.median())], eva)
    A.numeric("E-06", "arm-a", body,
              r"median target AUROC \$([\d.]+)\$ against \$([\d.]+)\$",
              [db["median_auroc"], db["source_auroc"]], eva)
    A.numeric("E-07", "arm-a", body,
              r"a \$(\d+)\$-fold difference in realised operating point", cc["widest_ratio_raw"], eva,
              note="raw-rate ratio; deriving it from the rounded table multiples gives "
                   f"{cc['widest_ratio_from_rounded_multiples']}")
    A.numeric("E-08", "arm-a", body,
              r"Amharic runs at \$\\mathbf\{([\d.]+)\\times\}\$", cc["widest_multiple_xlmr"], eva)
    A.numeric("E-09", "arm-a", body,
              r"AUROC \$([\d.]+)\$,\s*recall \$([\d.]+)\$\)", [xl["source_auroc"], xl["source_recall"]], eva)
    A.numeric("E-10", "arm-a", body,
              r"scores\s*Arabic at AUROC \$([\d.]+)\$", _auroc("EXP-009_arm_a.csv", "ar"), eva)
    A.check("E-11", "arm-a", cc["n_sign_disagreements"] == 8 and "eight of 14" in body, eva,
            f"{cc['n_sign_disagreements']} of {cc['n_languages']} disagree on the sign of the error")

    # ---------------------------------------------------------- F. F1 degeneracy, EXP-003
    e3 = load_csv("EXP-003_threshold_study.csv")
    ev3 = "results/tables/EXP-003_threshold_study.csv"
    d3 = e3[(e3.strategy == "target_full") & (~e3.is_source) & (e3.prevalence_setting == 0.5)]
    dg = d3[d3.is_degenerate]
    A.check("F-01", "f1-artefact", int(d3.is_degenerate.sum()) == 6 and "In 6 of 14" in body, ev3,
            f"{int(d3.is_degenerate.sum())} of {len(d3)} degenerate at 50/50")
    A.numeric("F-02", "f1-artefact", body, r"Amharic reaches\s*\$([\d.]+)\\%\$",
              100 * float(d3[d3.language == "am"].predicted_positive_rate.iloc[0]), ev3)
    # Anchored to the Amharic F1 sentence: an unanchored "false-positive rate of" also matches
    # the Arm A sentence in Section V-F, which reports a different quantity entirely.
    A.numeric("F-03", "f1-artefact", body,
              r"Amharic reaches\s*\$[\d.]+\\%\$, at a false-positive rate of \$([\d.]+)\$",
              float(d3[d3.language == "am"].fpr.iloc[0]), ev3)
    A.numeric("F-04", "f1-artefact", body, r"\(\$([\d.]+)\$--\$([\d.]+)\$\) bracket the algebraic",
              [float(dg.f1.min()), float(dg.f1.max())], ev3)
    nd = d3[~d3.is_degenerate].predicted_positive_rate
    A.numeric("F-05", "f1-artefact", body,
              r"eight languages falling between \$([\d.]+)\$\s*and \$([\d.]+)\$ and six between \$([\d.]+)\$ and \$([\d.]+)\$",
              [float(nd.min()), float(nd.max()),
               float(dg.predicted_positive_rate.min()), float(dg.predicted_positive_rate.max())], ev3)
    r_ppr, p_ppr = spearmanr(d3.test_auroc, d3.predicted_positive_rate)
    A.numeric("F-06", "f1-artefact", body,
              r"associated with AUROC \(\$\\rho = ([-\d.]+)\$, \$p = ([\d.]+)\\times10\^\{-5\}\$\)",
              [r_ppr, p_ppr * 1e5], ev3)

    # ---------------------------------------------------------- G. negatives / m-rule
    nc = load_json(TABLES / "negatives_curve_verdict.json")
    evn = "results/tables/negatives_curve_verdict.json"
    A.numeric("G-01", "negatives", body, r"\$m \\le ([\d.]+)\$", nc["max_m_reachable_on_this_corpus"], evn)
    A.numeric("G-02", "negatives", body,
              r"the violation rate stays between \$([\d.]+)\$ and \$([\d.]+)\$",
              [nc["violation_rate_min"], nc["violation_rate_max"]], evn)
    A.numeric("G-03", "negatives", body,
              r"slope downward \(\$\\rho = ([-\d.]+)\$\), but eight grid points cannot resolve it\s*\(\$p = ([\d.]+)\$\)",
              [nc["spearman_n_neg_vs_violation"]["rho"], nc["spearman_n_neg_vs_violation"]["p"]], evn)
    nt = load_json(TABLES / "negatives_theory_verdict.json")
    evt = "results/tables/negatives_theory_verdict.json"
    A.numeric("G-04", "negatives", body,
              r"where \$m \\ge 10\$ the median violation rate is \$([\d.]+)\$; where\s*\$m < 10\$ it is \$([\d.]+)\$",
              [nt["median_violation_when_predicted_usable"],
               nt["median_violation_when_predicted_unusable"]], evt)
    A.numeric("G-05", "negatives", body,
              r"association\s*is strong \(Spearman \$\\rho = ([-\d.]+)\$\)",
              nt["spearman_expected_fp_vs_violation"]["rho"], evt)
    A.check("G-06", "negatives", f"all {nt['n_regimes']} calibration regimes" in body, evt,
            f"regime count {nt['n_regimes']} stated in the manuscript")

    # ---------------------------------------------------------- H. topic robustness
    tr = load_json(TABLES / "sib200_topic_robustness.json")
    evr = "results/tables/sib200_topic_robustness.json"
    by = {r["positive_topic"]: r for r in tr["topics"]}
    A.numeric("H-01", "topic", body,
              r"is \$([\d.]+)\$ for \\emph\{science/technology\},\s*\$([\d.]+)\$ for \\emph\{travel\} and \$([\d.]+)\$ for \\emph\{health\}",
              [by["science/technology"]["frac_more_than_0.05_from_source"],
               by["travel"]["frac_more_than_0.05_from_source"],
               by["health"]["frac_more_than_0.05_from_source"]], evr)
    A.numeric("H-02", "topic", body, r"four-topic mean of \$([\d.]+)\$", tr["mean_frac_diverging"], evr)
    A.numeric("H-03", "topic", body,
              r"thresholds cluster in \$\[([\d.]+), ([\d.]+)\]\$",
              [by["politics"]["tau_min"], by["politics"]["tau_max"]], evr)
    A.numeric("H-04", "topic", body,
              r"\(median \$\|\\Delta\\tau\| = ([\d.]+)\$\)\. That is also the topic",
              by["politics"]["median_abs_dtau"], evr)
    A.numeric("H-05", "topic", body,
              r"median target AUROC \$([\d.]+)\$, against \$([\d.]+)\$ for",
              [by["politics"]["median_target_auroc"], by["travel"]["median_target_auroc"]], evr)
    A.numeric("H-06", "topic", body,
              r"Spearman \$\\rho = ([-\d.]+)\$ between median AUROC and median \$\|\\Delta\\tau\|\$",
              tr["spearman_auroc_vs_median_dtau_across_topics"], evr)

    # ---------------------------------------------------------- I. attribution (D-018)
    fman = load_json(FIGURES / "figure_manifest.json")
    evf = "results/figures/figure_manifest.json"
    figs = {f["figure"]: f for f in fman}
    # Every figure must declare its source experiment, and that experiment must be one the
    # manuscript still reports. EXP-001 was removed from the paper by D-018.
    bad_src = [f["figure"] for f in fman if not f.get("source_data")]
    A.check("I-01", "attribution", not bad_src, evf,
            f"every figure declares source_data ({len(fman)} figures)"
            + (f"; missing: {bad_src}" if bad_src else ""))
    exp001 = [f["figure"] for f in fman if "EXP-001" in f.get("source_data", "")]
    A.check("I-02", "attribution", not exp001 and "EXP-001" not in body, evf,
            "no manuscript figure is sourced from EXP-001 and EXP-001 is not named in the prose "
            "(D-018)" + (f"; offenders: {exp001}" if exp001 else ""))
    # Every figure in the manifest is included and referenced; every float label referenced.
    figtex = section_text("figures.tex")
    missing_inc = [f["figure"] for f in fman if f["figure"] not in figtex]
    A.check("I-03", "attribution", not missing_inc, evf,
            "every manifest figure is included in the manuscript"
            + (f"; missing: {missing_inc}" if missing_inc else ""))
    tman = load_json(MTABLES / "table_manifest.json")
    evtm = "manuscript/tables/table_manifest.json"
    gen_tables = {t["table"] if isinstance(t, dict) and "table" in t else None for t in tman} \
        if isinstance(tman, list) else set(tman)
    A.check("I-04", "attribution", len(gen_tables) > 0, evtm,
            f"generated-table manifest present with {len(gen_tables)} entries")

    # ---------------------------------------------------------- J. cross-section consistency
    # The scope limitation on the two deployed classifiers must be stated identically wherever
    # the claim appears. Narrowing it in one place and leaving it broad elsewhere is the
    # defect that recurred across two revision rounds.
    scope_sites = [
        ("01_introduction.tex", "establishes that the direction is model-dependent"),
        ("05_results.tex", "establish that the direction is model-dependent"),
        ("09_conclusion.tex", "establish that the direction is model-dependent"),
    ]
    miss = [f for f, probe in scope_sites if probe not in section_text(f)]
    A.check("J-01", "consistency", not miss, "manuscript sections",
            "the n=2 scope statement appears in Introduction, Results and Conclusion"
            + (f"; missing in {miss}" if miss else ""))
    A.check("J-02", "consistency", "not its distribution over models" in body, "manuscript",
            "the distributional claim is explicitly disclaimed")
    # Acronyms defined in prose, including ones that otherwise appear only in generated tables.
    for cid, acr, defn in (("J-03", "ECE", "expected calibration error"),
                           ("J-04", "SLD/EM", "EM prior re-estimation"),
                           ("J-05", "BBSE", "black-box shift estimation")):
        A.check(cid, "consistency", defn in body, "manuscript",
                f"{acr} expanded at first use in the body prose")
    A.check("J-06", "consistency", r"\ref{sec:magnitude}" in body, "manuscript",
            "the opposite-sign correlations cross-reference their explanation")
    A.check("J-07", "consistency",
            f"${_median_dtau('EXP-008-citizenlab_arm_a.csv'):.3f}$" in body
            and f"${_median_dtau('EXP-009_arm_a.csv'):.3f}$" in body, "manuscript",
            "Contribution 1 quotes both deployed-classifier divergence values")
    # A claim of the null must not be ASSERTED anywhere, in any phrasing. The check must be
    # negation-aware: "we do not present them as evidence that no association exists" is the
    # correct form and must not be flagged, while "because their operating points are
    # uncorrelated" is an assertion and must be. A naive substring test gets both wrong.
    NULL_FORMS = ("statistically uncorrelated", "were uncorrelated", "are uncorrelated",
                  "no association exists", "has no effect", "there is no association")
    NEGATION_CUES = ("do not", "does not", "not present", "rather than", "declines",
                     "cannot", "never", "no longer", "we do not claim")
    asserted = []
    for form in NULL_FORMS:
        for m in re.finditer(re.escape(form), body):
            window = body[max(0, m.start() - 110):m.start()]
            if not any(cue in window for cue in NEGATION_CUES):
                asserted.append((form, body[max(0, m.start() - 60):m.end() + 20]))
    A.check("J-08", "consistency", not asserted, "manuscript",
            "no unnegated assertion of the null in any tracked phrasing"
            + ("" if not asserted else
               "; ASSERTED: " + " || ".join(f"...{c.strip()}..." for _, c in asserted)))

    # ---------------------------------------------------------- K. setup facts
    ev4 = "manuscript/sections/04_experimental_setup.tex + data/processed/sib200_audit.json"
    aud = load_json(REPO_ROOT / "data" / "processed" / "sib200_audit.json")
    setup = section_text("04_experimental_setup.tex")
    A.numeric("K-01", "setup", setup, r"An audit of all (\d+) language", aud["n_configs"], ev4)
    # Markup-agnostic: the audited quantity is the count, not how it is emphasised.
    # Both patterns follow the setup text's move to naming the audit criterion exactly: it
    # compares per-split row counts against the reference language and does not check
    # per-sentence identifiers. The audited counts are unchanged.
    A.numeric("K-02", "setup", setup,
              r"Only the (?:\\textbf\{)?(\d+)(?:\})? configurations whose split sizes match",
              aud["n_parallel_aligned"], ev4)
    A.numeric("K-03", "setup", setup,
              r"(\d+) configurations do not match the reference language's per-split row counts",
              aud["n_row_count_mismatch"], ev4)
    dev = base.groupby("language").n_dev.median()
    A.numeric("K-04", "setup", setup,
              r"validation split ranges from \$(\d+)\$ items in\s*Hebrew to \$(\d+)\$ in Arabic and Italian",
              [float(dev.min()), float(dev.max())], ev2)
    prov = load_json(REPO_ROOT / "experiments" / "runs" / "EXP-002" / "provenance.json")
    A.check("K-05", "setup", "nineteen-field provenance record" in setup and len(prov) == 19,
            "experiments/runs/EXP-002/provenance.json",
            f"provenance record has {len(prov)} fields and the manuscript says so")

    return A


def _median_dtau(csv_name: str) -> float:
    d = pd.read_csv(_require(TABLES / csv_name))
    t = d[(d.strategy == "target_full") & (~d.is_source)]
    return float((t.threshold - t.tau_source).abs().median())


def _auroc(csv_name: str, lang: str) -> float:
    d = pd.read_csv(_require(TABLES / csv_name))
    return float(d[(d.strategy == "source_global") & (d.language == lang)].test_auroc.iloc[0])


# ----------------------------------------------------------------- coverage limits

LIMITS = """KNOWN LIMITS of this audit, stated rather than implied:
  1. It checks NUMERIC and ATTRIBUTION claims. Universally and ordinally quantified prose is
     the remit of check_quantified_claims.py, which exists precisely because this class of
     audit cannot see quantifier defects: it passed 58/58 while two such defects shipped.
  2. It cannot verify claims about the literature, about deployments we did not observe, or
     about what prior work does or does not measure. Those are checked by hand.
  3. It compares prose against the artifact that produced the number. It cannot detect an
     error that is present identically in both. verify_results.py exists for that: it
     re-derives headline quantities from raw per-example scores without importing the
     analysis engine.
  4. A passing run is not evidence that untested defect classes are absent."""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verbose", action="store_true", help="show passing checks too")
    ap.add_argument("--list", action="store_true", help="list check IDs and exit")
    args = ap.parse_args()

    try:
        A = run_audit()
    except MissingArtifact as e:
        print(f"FAIL: {e}")
        print("A missing artifact is a hard failure; this audit never reports a skip as a pass.")
        return 2

    if args.list:
        for cid, cat, _, ev, _ in A.rows:
            print(f"  {cid:8s} {cat:14s} {ev}")
        print(f"\n{len(A.rows)} checks")
        return 0

    width = max(len(r[0]) for r in A.rows)
    for cid, cat, ok, ev, detail in A.rows:
        if ok and not args.verbose:
            continue
        print(f"  {'PASS' if ok else 'FAIL'}  {cid:<{width}}  {cat:14s} {detail}")
        if not ok:
            print(f"        evidence: {ev}")

    n_ok = len(A.rows) - len(A.failed)
    print(f"\n{n_ok}/{len(A.rows)} claim-to-evidence checks pass")
    print(f"\n{LIMITS}")
    if A.failed:
        print("\nFAIL")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
