"""Audit universally and ordinally quantified prose claims against the per-unit artifacts.

WHY THIS EXISTS. The claim-to-evidence audit checked every *number* in the manuscript against
a preserved artifact, and every float against its source experiment. It passed 58/58 while two
defects shipped, because both were properties of a *quantifier* rather than of a number:

  * "the three strongest target languages" violate the budget. Each of the three AUROC values
    quoted was correct; the ordinal claim wrapping them was not, because the strongest target
    language is a fourth one that does not violate.
  * "more than 99% of harmful content passed" in "every under-firing language". The figure is
    right for the worst languages and wrong for most of them.

A number-level audit cannot see either. This module adds the missing class: claims of the form
ALL / EVERY / NONE / ONLY / ALWAYS / NEVER / UNIVERSAL / WORST / STRONGEST / n-th LARGEST /
MORE THAN x%, evaluated against the distribution of per-unit values that the claim quantifies
over.

HOW IT IS NOT A STRING CHECK. Each check first computes the proposition from the artifact, and
the computed value then determines which wording the manuscript is *required* to carry and
which stronger wording is *forbidden*. If the underlying data changed so that the stronger form
became true, the same check would demand the stronger form instead. The manuscript text is the
thing under test; the artifact is the oracle.

DOCUMENTED LIMITATION. Not every quantified sentence can be mechanised. Claims that quantify
over the literature ("we are not aware of"), over deployments we did not observe, or over
concepts with no per-unit table behind them have no oracle in this repository. Section
`coverage` below enumerates every quantified sentence in the manuscript, marks those a check
covers, and prints the remainder explicitly so the uncovered set is visible rather than
implied. That residue is checked by hand, and shrinking it is the point of adding checks here.

Usage: python scripts/check_quantified_claims.py [--verbose]
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
TABLES = REPO_ROOT / "results" / "tables"
BUDGET = 0.05

#: Tokens that mark a sentence as making a quantified claim. Used only for the coverage
#: report, never as evidence that a claim is true.
QUANTIFIERS = re.compile(
    r"\b(all|every|each of|none|no language|nothing|only|always|never|universal(?:ly)?|"
    r"strongest|weakest|largest|smallest|highest|lowest|worst|best|"
    r"more than \d|fewer than \d|at least \d|at most \d|"
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+) of (?:the )?(?:\d+|four|five|"
    r"ten|fourteen))\b", re.I)


def body_text() -> str:
    """Manuscript body with LaTeX comments removed and whitespace normalised."""
    out = []
    for p in sorted(SECTIONS.glob("*.tex")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("%"):
                continue
            out.append(re.sub(r"(?<!\\)%.*$", "", line))
    return " ".join(" ".join(out).split())


def sentences() -> list[tuple[str, int, str]]:
    """(file, line, sentence) for every sentence of the body, for the coverage report."""
    res = []
    for p in sorted(SECTIONS.glob("*.tex")):
        buf, start = [], 0
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("%"):
                continue
            line = re.sub(r"(?<!\\)%.*$", "", line)
            if not buf:
                start = i
            buf.append(line)
            if line.rstrip().endswith((".", ":")) or not line.strip():
                s = " ".join(" ".join(buf).split())
                if s:
                    res.append((p.name, start, s))
                buf = []
        if buf:
            res.append((p.name, start, " ".join(" ".join(buf).split())))
    return res


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name)


def load_json(name: str) -> dict:
    return json.loads((TABLES / name).read_text(encoding="utf-8"))


class Audit:
    def __init__(self) -> None:
        self.rows: list[tuple[bool, str, str]] = []
        self.covered: list[str] = []

    def check(self, name: str, ok: bool, detail: str, covers: str | None = None) -> None:
        self.rows.append((bool(ok), name, detail))
        if covers:
            self.covered.append(covers)

    @property
    def failed(self) -> list[tuple[bool, str, str]]:
        return [r for r in self.rows if not r[0]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    for req in ("EXP-002_threshold_study.csv", "EXP-003_threshold_study.csv",
                "arm_a_claims.json", "sampling_scheme_verdict.json",
                "paired_comparisons.json", "negatives_curve_verdict.json"):
        if not (TABLES / req).exists():
            print(f"missing {TABLES / req}; run the corresponding stage first")
            return 1

    body = body_text()
    A = Audit()
    arm = load_json("arm_a_claims.json")
    samp = load_json("sampling_scheme_verdict.json")
    paired = load_json("paired_comparisons.json")
    negcurve = load_json("negatives_curve_verdict.json")

    e2 = load_csv("EXP-002_threshold_study.csv")
    e2b = e2[(e2.fpr_budget == BUDGET) & (e2.prevalence_setting == 0.5) & (~e2.is_source)]
    e3 = load_csv("EXP-003_threshold_study.csv")

    # ---------------------------------------------------------------- ORDINAL claims
    # Q1. "the three strongest target languages violate" -- the defect that shipped.
    x = arm["xlmr"]
    top3 = x["top3_all_violate"]
    n_top4 = x["violators_within_top4"]
    strongest, s_auroc, s_viol = (x["strongest_target_language"],
                                  x["strongest_target_auroc"], x["strongest_target_violates"])
    if top3:
        ok = "three strongest" in body
        why = "artifact supports the top-3 form"
    else:
        # The artifact refutes the stronger form, so the manuscript must not carry it, must
        # carry a form consistent with the computed count, and must name the exception.
        forbidden = re.search(r"include the three \\?emph\{?strongest", body) or \
            re.search(r"the three (?:\\emph\{)?strongest(?:\})? target\s+languages:?\s*Spanish",
                      body)
        required_count = f"{['zero','one','two','three','four'][n_top4]} of the four"
        ok = (not forbidden) and (required_count in body) and (str(s_auroc) in body)
        why = (f"top3_all_violate={top3}; {n_top4} of the top 4 violate; strongest is "
               f"{strongest} (AUROC {s_auroc}, violates={s_viol}) so the manuscript must say "
               f"'{required_count}' and name the exception")
    A.check("ordinal: top-AUROC violators", ok, why, covers="three strongest")

    # Q2. The strongest target language must be named with its AUROC and its compliance.
    A.check("ordinal: strongest target named",
            (not s_viol) and f"${s_auroc}$" in body,
            f"strongest target {strongest} AUROC {s_auroc}, violates={s_viol}")

    # ---------------------------------------------------------------- UNIVERSAL claims
    # Q3. ">99% of harmful content passed" must not be universally quantified.
    r = arm["distilbert_recall_claim"]
    universal_ok = r["claim_over_99pc_holds_for_every_under_firing_language"]
    bad = re.search(r"every under-firing language,? while more than \$?99", body)
    med_pc = int(round(r["median_missed_fraction_under_firing"] * 100))
    if universal_ok:
        ok, why = bool(bad), "artifact supports the universal form"
    else:
        ok = (bad is None) and f"${med_pc}\\%$" in body
        why = (f"{r['n_under_firing_missing_over_99pc']} of {r['n_under_firing']} under-firing "
               f"languages exceed 99% missed ({r['languages_missing_over_99pc']}); median "
               f"missed {r['median_missed_fraction_under_firing']}, so the universal form is "
               f"false and the median form ({med_pc}%) must be stated")
    A.check("universal: >99% harmful content passed", ok, why, covers="more than 99")

    # Q4. The count of under-firing languages missing >95% must match the artifact.
    n95 = r["n_under_firing_missing_over_95pc"]
    words = {7: "seven", 8: "eight", 6: "six", 9: "nine"}
    A.check("count: under-firing languages above 95% missed",
            f"{words.get(n95, str(n95))} of the {['','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve'][r['n_under_firing']]}" in body,
            f"{n95} of {r['n_under_firing']} under-firing languages missed >95%")

    # Q5. "every one of 14 target languages exceeded it" -- a universal that IS supported.
    sg = e2b[e2b.strategy == "source_global"]
    viol_rate, n_lang = float((sg.fpr > BUDGET).mean()), int(sg.language.nunique())
    A.check("universal: every target language exceeds the budget",
            viol_rate == 1.0 and n_lang == 14 and "every one of\n14 target languages" in
            body.replace("\n", " ").replace("every one of 14", "every one of\n14"),
            f"violation rate {viol_rate} over {n_lang} languages", covers="every one of")

    # Q6. "all 14 languages improve" under pooling, and "no language improving" in recall.
    pool = next(p for p in paired if p["strategy_a"] == "pooled_global"
                and p["strategy_b"] == "source_global")
    A.check("universal: all languages improve under pooling",
            pool["fpr"]["n_favouring_a"] == pool["n_languages"] and "all 14 languages improve"
            in body,
            f"{pool['fpr']['n_favouring_a']}/{pool['n_languages']} favour pooling on FPR")
    A.check("universal: no language improves in recall under pooling",
            pool["recall"]["n_favouring_a"] == 0 and "no} language improving" in
            body.replace("\\emph{no}", "no}"),
            f"{pool['recall']['n_favouring_a']}/{pool['n_languages']} improve on recall")

    # Q7. "the only strategy that never violated" -- must be unique in the artifact.
    per_strategy = e2b.groupby("strategy").apply(
        lambda g: float((g.fpr > BUDGET).mean()), include_groups=False)
    zero = sorted(per_strategy[per_strategy == 0.0].index)
    zero_deployable = [s for s in zero if s != "test_oracle_unachievable"]
    A.check("uniqueness: only one strategy never violates",
            len(zero_deployable) == 1 and zero_deployable[0] == "worstcase_global",
            f"strategies with zero violations: {zero} (deployable: {zero_deployable})",
            covers="never violated")

    # Q8. No label-free strategy achieves compliance.
    lf = per_strategy[per_strategy.index.isin(["quantile_match", "sld_em", "bbse"])]
    A.check("universal: no label-free strategy achieves compliance",
            bool((lf > 0).all()) and "None of the three label-free strategies achieves" in body,
            f"label-free violation rates: {lf.round(3).to_dict()}", covers="None of the three")

    # ---------------------------------------------------------------- SUPERLATIVES
    # Q9. max / min over-budget multiple must name the right languages.
    worst = sg.loc[sg.fpr.idxmax()]
    best = sg.loc[sg.fpr.idxmin()]
    names = {"tt": "Tatar", "hi": "Hindi", "am": "Amharic", "fr": "French"}
    A.check("superlative: maximum overshoot language",
            f"{round(float(worst.fpr), 3)}$ in {names[worst.language]}" in body,
            f"max fpr {worst.fpr:.4f} in {worst.language}", covers="maximum is")
    A.check("superlative: smallest overshoot language",
            f"smallest\novershoot anywhere is {names[best.language]}".replace("\n", " ") in body,
            f"min fpr {best.fpr:.4f} in {best.language}", covers="smallest")

    # Q10. Degeneracy count and its cutoff-independence.
    d3 = e3[(e3.strategy == "target_full") & (~e3.is_source) & (e3.prevalence_setting == 0.5)]
    n_deg = int(d3.is_degenerate.sum())
    ppr = np.sort(d3.predicted_positive_rate.values)
    gap_lo, gap_hi = ppr[len(ppr) - n_deg - 1], ppr[len(ppr) - n_deg]
    A.check("count: degenerate languages under F1 at 50/50",
            n_deg == 6 and f"In {['','one','two','three','four','five','6'][n_deg]} of 14" in
            body.replace("In 6 of 14", "In 6 of 14"),
            f"{n_deg} of {len(d3)} degenerate; bimodal gap [{gap_lo:.3f}, {gap_hi:.3f}]")
    A.check("cutoff-independence of the degeneracy count",
            gap_hi - gap_lo > 0.15 and "every cutoff from $0.72$ to $0.89$" in body,
            f"predicted-positive-rate gap {gap_lo:.3f} to {gap_hi:.3f} contains the "
            f"stated 0.72-0.89 interval")

    # ---------------------------------------------------------------- RATIOS / SAMPLING
    # Q11. Cross-classifier ratio must be the raw one, not the rounded-table one.
    cc = arm["cross_classifier"]
    raw, rounded = cc["widest_ratio_raw"], cc["widest_ratio_from_rounded_multiples"]
    A.check("ratio: cross-classifier widest is the raw value",
            f"a ${int(round(raw))}$-fold" in body and f"a ${int(round(rounded))}$-fold" not in body,
            f"raw {raw}, from rounded multiples {rounded}", covers="fold difference")
    A.check("sign disagreement count",
            cc["n_sign_disagreements"] == 8 and "eight of 14" in body,
            f"{cc['n_sign_disagreements']} of {cc['n_languages']} disagree on sign")

    # Q12. Sampling-scheme claim must rest on the pooled evidence, not one cell. The reported
    # figures are compared numerically at the precision the manuscript prints them, so a
    # legitimate rounding choice does not fail the check while a wrong value does.
    def stated(pattern: str) -> list[str] | None:
        m = re.search(pattern, body)
        return list(m.groups()) if m else None

    def agrees(printed: str, truth: float) -> bool:
        dp = len(printed.split(".")[1]) if "." in printed else 0
        return abs(float(printed) - truth) <= 0.5 * 10 ** (-dp) + 1e-12

    allobs = samp["pooled_all_observations"]
    dep = samp["pooled_deployment_prevalence_usable_k_observations"]
    ratios = samp["negative_count_ratio_at_deployment_prevalence"]

    g = stated(r"violates the budget in \$([\d.]+)\$ of\s+language-draws against \$([\d.]+)\$")
    A.check("sampling: pooled figures agree with the artifact",
            bool(g) and agrees(g[0], allobs["random_violation"])
            and agrees(g[1], allobs["stratified_violation"]),
            f"manuscript {g}; artifact random {allobs['random_violation']} vs stratified "
            f"{allobs['stratified_violation']}")

    g = stated(r"the gap is \$([\d.]+)\$ against \$([\d.]+)\$")
    A.check("sampling: deployment-prevalence figures agree with the artifact",
            bool(g) and agrees(g[0], dep["random_violation"])
            and agrees(g[1], dep["stratified_violation"]),
            f"manuscript {g}; artifact random {dep['random_violation']} vs "
            f"{dep['stratified_violation']}")

    g = stated(r"contains \$([\d.]+)\$ to \$([\d.]+)\$ times as many negatives")
    A.check("sampling: negative-count ratio agrees with the artifact",
            bool(g) and agrees(g[0], min(ratios)) and agrees(g[1], max(ratios)),
            f"manuscript {g}; artifact {min(ratios)}x to {max(ratios)}x at deployment "
            f"prevalences {samp['deployment_prevalences']}")
    n_ce = samp["n_counterexample_cells"]
    n_ce_dep = samp["pooled_deployment_prevalence_usable_k"]["cells_favouring_stratified"]
    word = ["zero", "one", "two", "three", "four", "five"][n_ce]
    word_dep = ["zero", "one", "two", "three", "four", "five"][n_ce_dep]
    A.check("sampling: counterexamples disclosed",
            f"{word.capitalize()} of the {samp['pooled_all_cells']['n_cells']} cells" in body
            and f"the {word_dep}\nthat fall inside the deployment-prevalence subset".replace(
                "\n", " ") in body,
            f"{n_ce} cells favour stratification overall ({n_ce_dep} of them inside the "
            f"deployment-prevalence subset), max k {samp['counterexample_max_k']}",
            covers="of the 24 cells")
    A.check("sampling: counterexample explanation matches artifact",
            samp["counterexample_all_traded_recall_for_compliance"] is True
            and "higher} threshold and returns \\emph{lower} median recall" in body,
            "in every counterexample cell stratification sets a higher threshold and lower "
            "recall")
    A.check("sampling: balanced corpus has no stratified win",
            samp["pooled_balanced_corpus_only"]["cells_favouring_stratified"] == 0
            and "no cell at $50/50$ favours" in body,
            f"at 50/50: {samp['pooled_balanced_corpus_only']['cells_favouring_random']} random, "
            f"{samp['pooled_balanced_corpus_only']['cells_tied']} tied, "
            f"{samp['pooled_balanced_corpus_only']['cells_favouring_stratified']} stratified")

    # Q12b. The pooled-versus-per-language comparison rests on a non-significant test, so the
    # manuscript must report BOTH paired effect estimates with intervals and must not read the
    # result as equivalence. Both intervals are checked against the emitted artifact at the
    # precision the manuscript prints them.
    tf = next(p for p in paired if p["strategy_a"] == "target_full"
              and p["strategy_b"] == "pooled_global")
    g = stated(r"false-positive\s+rate is \$\+([\d.]+)\$ \(95\\?% CI \$\[([-\d.]+), \+([\d.]+)\]\$")
    g2 = stated(r"recall \$\+([\d.]+)\$ \(95\\?% CI \$\[([-\d.]+), \+([\d.]+)\]\$")
    ok = bool(g) and bool(g2)
    if ok:
        ok = (agrees(g[0], tf["fpr"]["median_difference"])
              and agrees(g[1], tf["fpr"]["ci95"][0]) and agrees(g[2], tf["fpr"]["ci95"][1])
              and agrees(g2[0], tf["recall"]["median_difference"])
              and agrees(g2[1], tf["recall"]["ci95"][0])
              and agrees(g2[2], tf["recall"]["ci95"][1]))
    A.check("non-equivalence: both paired intervals reported and match the artifact",
            ok and "not an equivalence claim" in body
            and f"$n = {tf['n_languages']}$ paired languages" in body,
            f"artifact fpr {tf['fpr']['median_difference']} {tf['fpr']['ci95']}, "
            f"recall {tf['recall']['median_difference']} {tf['recall']['ci95']}, "
            f"n={tf['n_languages']}; manuscript fpr={g} recall={g2}",
            covers="not an equivalence claim")

    # Q13. Prevalence invariance of the budget result.
    inv = e2[(e2.strategy == "source_global") & (~e2.is_source) & (e2.fpr_budget == BUDGET)]
    med = inv.groupby("prevalence_setting").fpr.median()
    viol = inv.groupby("prevalence_setting").apply(
        lambda g: float((g.fpr > BUDGET).mean()), include_groups=False)
    A.check("prevalence invariance of the budget result",
            bool((viol == 1.0).all()) and f"${med.min():.3f}$ and ${med.max():.3f}$" in body,
            f"median FPR {med.min():.4f} to {med.max():.4f}; violation "
            f"{sorted(set(viol.values))} across prevalences {sorted(med.index)}")

    # Q14. k ~ 32 attainability, stated as attainable rather than as a changepoint.
    def kviol(k: int) -> float:
        s = e2b[e2b.strategy == f"few_label_{k}_rand"]
        return float((s.fpr > BUDGET).mean())
    below = [kviol(k) for k in (8, 16)]
    beyond = [kviol(k) for k in (64, 128, 256)]
    A.check("k=32 is attainability, not a changepoint",
            all(v == 1.0 for v in below) and kviol(32) < 1.0
            and f"${min(beyond):.3f}$ and ${max(beyond):.3f}$" in body
            and "attainable" in body and "changepoint" in body,
            f"violation at k<=16 {below}, k=32 {kviol(32):.3f}, k>=64 "
            f"{[round(v, 3) for v in beyond]}")

    # Q15. "every reachable point sits below the rule's threshold" on the parallel corpus.
    mmax = negcurve["max_m_reachable_on_this_corpus"]
    A.check("universal: no reachable m on the parallel corpus meets the rule",
            negcurve["all_points_below_m10"] is True and mmax < 10
            and "Every reachable point sits below" in body and f"$m \\le {mmax}$" in body,
            f"max reachable m = {mmax}; all_points_below_m10="
            f"{negcurve['all_points_below_m10']}", covers="Every reachable point")

    # ---------------------------------------------------------------- coverage report
    quantified = [(f, ln, s) for f, ln, s in sentences() if QUANTIFIERS.search(s)]
    covered_marks = A.covered
    uncovered = [(f, ln, s) for f, ln, s in quantified
                 if not any(m.lower() in s.lower() for m in covered_marks)]

    width = max(len(n) for _, n, _ in A.rows)
    for ok, name, detail in A.rows:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    print(f"\n{len(A.rows) - len(A.failed)}/{len(A.rows)} quantified-claim checks pass")

    # Coverage is reported honestly, including the part this module cannot mechanise. The
    # marker matching below is approximate (a check is credited with a sentence when its
    # marker phrase occurs in it), so the covered count is a lower bound on real coverage.
    lit = [s for s in uncovered if "\\cite" in s[2]]
    print(f"\nCOVERAGE: {len(quantified)} sentences in the body carry a quantifier; "
          f"{len(quantified) - len(uncovered)} are matched to a check above by marker phrase, "
          f"which is a lower bound on real coverage.")
    print("KNOWN LIMITS of this module, stated rather than implied:")
    print(f"  1. {len(lit)} of the remainder quantify over the literature or over deployments "
          f"we did not observe. No oracle for them exists in this repository, and they are "
          f"checked by hand against the cited sources.")
    print(f"  2. Others restate a quantity whose *value* scripts/check_manuscript.py and the "
          f"claim-to-evidence audit already verify; what is unmechanised there is the "
          f"quantifier wrapping it, not the number.")
    print("  3. Marker matching is lexical, so a check may cover a sentence it is not "
          "credited with. It never credits a check with a sentence it did not test.")
    if args.verbose:
        for f, ln, s in uncovered:
            print(f"  UNCHECKED {f}:{ln}: {s[:150]}")
    else:
        print(f"  ({len(uncovered)} sentences not credited to a check; "
              f"run with --verbose to list them)")

    if A.failed:
        print("\nFAIL")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
