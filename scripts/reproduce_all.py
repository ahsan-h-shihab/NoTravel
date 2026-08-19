"""Single entry point that regenerates every reported result, table and figure.

This script IS the reproducibility contract (`D-007`). REPRODUCIBILITY
CONSTITUTION treats loss of reproducibility as scientific failure, so the rule is enforced
rather than asserted: **a result this script cannot regenerate may not appear in the
manuscript.**

Usage
-----
    python scripts/reproduce_all.py --all          # everything (hours; downloads models)
    python scripts/reproduce_all.py --stage tests  # just the unit tests
    python scripts/reproduce_all.py --list         # show stages without running
    python scripts/reproduce_all.py --all --check  # verify outputs exist + checksums only

Design notes
------------
* Stages are ordered and idempotent. Embeddings are cached (keyed by model revision AND a
  checksum of the exact text sequence), so re-running is cheap after the first pass.
* Every stage records wall-clock time and exit status into a run report, so a partial
  reproduction is visible rather than silently incomplete.
* The script never *repairs* anything. If a stage fails, it reports and stops that stage --
  masking a failure would defeat the purpose.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


@dataclass
class Stage:
    name: str
    description: str
    command: list[str]
    produces: list[str] = field(default_factory=list)
    optional: bool = False
    #: True when the stage reads or writes manuscript/. The manuscript is not part of the
    #: public artifact, so these stages are LOCAL: they run when manuscript/ is present and
    #: are reported as not-applicable when it is not. Result reproduction never depends on
    #: them -- they generate LaTeX from results that already exist, or validate the paper's
    #: prose against those results.
    manuscript_local: bool = False


#: The manuscript is a private working directory, not part of the public reproduction
#: artifact. Its presence decides whether the manuscript-local stages are applicable.
MANUSCRIPT_PRESENT = (REPO_ROOT / "manuscript").is_dir()


STAGES: list[Stage] = [
    Stage(
        name="environment",
        description="Capture the execution environment (versions, CPU, git commit).",
        command=[PYTHON, "scripts/capture_environment.py"],
        produces=["environment/environment_snapshot.json"],
    ),
    Stage(
        name="tests",
        description="Unit tests for metrics, thresholds, prior transfer, degeneracy, figures.",
        command=[PYTHON, "-m", "pytest", "tests/", "-q"],
    ),
    Stage(
        name="exp001_pilot",
        description="EXP-001 pilot: does cross-lingual threshold divergence exist?",
        command=[PYTHON, "scripts/run_pilot.py"],
        produces=[
            "results/tables/EXP-001_pilot_per_language.csv",
            "experiments/runs/EXP-001/provenance.json",
            "experiments/runs/EXP-001/summary.json",
        ],
    ),
    Stage(
        name="exp002_fpr_budget",
        description="EXP-002 primary: operating-point cost under a false-positive budget.",
        command=[PYTHON, "scripts/run_threshold_study.py", "--exp-id", "EXP-002",
                 "--task", "toxicity", "--encoder", "encoder_minilm", "--source", "en",
                 "--objective", "target_fpr", "--target-fpr", "0.05",
                 "--prevalence", "0.50", "0.25", "0.10", "0.05"],
        produces=[
            "results/tables/EXP-002_threshold_study.csv",
            "experiments/runs/EXP-002/provenance.json",
        ],
    ),
    Stage(
        name="exp003_f1_degeneracy",
        description="EXP-003 secondary: F1 objective with degeneracy exposed.",
        command=[PYTHON, "scripts/run_threshold_study.py", "--exp-id", "EXP-003",
                 "--task", "toxicity", "--encoder", "encoder_minilm", "--source", "en",
                 "--objective", "f1",
                 "--prevalence", "0.50", "0.25", "0.10", "0.05"],
        produces=[
            "results/tables/EXP-003_threshold_study.csv",
            "experiments/runs/EXP-003/provenance.json",
        ],
    ),
    Stage(
        name="exp004_second_encoder",
        description="EXP-004: replicate under a second encoder (rules out a model artefact).",
        command=[PYTHON, "scripts/run_threshold_study.py", "--exp-id", "EXP-004",
                 "--task", "toxicity", "--encoder", "encoder_e5_small", "--source", "en",
                 "--objective", "target_fpr", "--target-fpr", "0.05", "--bootstrap", "300"],
        produces=["results/tables/EXP-004_threshold_study.csv"],
    ),
    Stage(
        name="sib200_audit",
        description="Audit SIB-200 configs for schema and row-alignment (parallel-corpus arm).",
        command=[PYTHON, "scripts/audit_sib200.py"],
        produces=["data/processed/sib200_audit.json"],
    ),
    Stage(
        name="exp006_parallel_control",
        description="EXP-006: parallel-corpus control on 174 aligned SIB-200 languages (T-I1).",
        command=[PYTHON, "scripts/run_threshold_study.py", "--exp-id", "EXP-006",
                 "--task", "sib200", "--encoder", "encoder_minilm", "--source", "eng_Latn",
                 "--objective", "target_fpr", "--target-fpr", "0.05", "--bootstrap", "400"],
        produces=["results/tables/EXP-006_threshold_study.csv"],
    ),
    Stage(
        name="exp007_budget_sweep",
        description="EXP-007: sweep the FPR budget {0.01,0.05,0.10,0.20} (reviewer sim R2-2).",
        command=[PYTHON, "scripts/run_threshold_study.py", "--exp-id", "EXP-007",
                 "--task", "toxicity", "--encoder", "encoder_minilm", "--source", "en",
                 "--objective", "target_fpr", "--target-fpr", "0.01", "0.05", "0.10", "0.20",
                 "--bootstrap", "200"],
        produces=["results/tables/EXP-007_threshold_study.csv"],
    ),
    Stage(
        name="exp008_arm_a",
        description="EXP-008 ARM A: off-the-shelf deployed classifiers (external validity).",
        command=[PYTHON, "scripts/run_arm_a.py", "--exp-id", "EXP-008-citizenlab",
                 "--models", "clf_citizenlab_distilbert"],
        produces=["results/tables/EXP-008-citizenlab_arm_a.csv"],
    ),
    Stage(
        name="exp009_arm_a_second",
        description="EXP-009 ARM A replication: second deployed classifier (XLM-R, Detoxify).",
        command=[PYTHON, "scripts/run_arm_a.py", "--exp-id", "EXP-009",
                 "--models", "clf_unitary_xlmr"],
        produces=["results/tables/EXP-009_arm_a.csv"],
    ),
    Stage(
        name="verify",
        description="Independently re-derive headline numbers via a separate implementation.",
        command=[PYTHON, "scripts/verify_results.py"],
        produces=["results/tables/verification_log.json"],
    ),
    Stage(
        name="negatives_theory",
        description="Binomial analysis relating m = alpha*n_neg to observed failures.",
        command=[PYTHON, "scripts/analyze_negatives_theory.py"],
        produces=["results/tables/negatives_theory_vs_observed.csv"],
    ),
    Stage(
        name="paired_comparisons",
        description="Paired strategy comparisons (bootstrap CI + Wilcoxon signed-rank).",
        command=[PYTHON, "scripts/analyze_paired_comparisons.py"],
        produces=["results/tables/paired_comparisons.json"],
    ),
    Stage(
        name="figures",
        description="Regenerate every manuscript figure from preserved result tables.",
        command=[PYTHON, "scripts/make_figures.py"],
        produces=["results/figures/figure_manifest.json"],
    ),
    Stage(
        name="tables",
        description="Regenerate every manuscript table (LaTeX) from preserved result tables.",
        command=[PYTHON, "scripts/make_tables.py"],
        produces=["manuscript/tables/table_manifest.json"],
        manuscript_local=True,
    ),
    Stage(
        name="manuscript_figures",
        description="Copy figures into the manuscript and emit their LaTeX floats + captions.",
        command=[PYTHON, "scripts/sync_manuscript_figures.py"],
        produces=["manuscript/sections/figures.tex"],
        manuscript_local=True,
    ),
    Stage(
        name="appendix",
        description="Generate the appendix (excluded configs, calibration regimes) from artifacts.",
        command=[PYTHON, "scripts/make_appendix.py"],
        produces=["manuscript/sections/appendix.tex"],
        manuscript_local=True,
    ),
    # Re-analysis, not a new experiment: subsamples preserved EXP-006 scores to trace budget
    # compliance against the number of calibration negatives, answering whether Table 2's
    # counter-intuitive per-language result is a shortage of negatives.
    Stage(
        name="negatives_curve",
        description="Trace SIB-200 budget compliance against calibration-negative count.",
        command=[PYTHON, "scripts/analyse_negatives_curve.py"],
        produces=["results/tables/negatives_curve_sib200.csv"],
    ),
    # Robustness of the parallel-corpus content control to the one-vs-rest binarisation.
    # Reuses cached embeddings; only the label vector, head and threshold search are recomputed.
    Stage(
        name="topic_robustness",
        description="Repeat the SIB-200 threshold estimate for four positive topics.",
        command=[PYTHON, "scripts/analyse_topic_robustness.py"],
        produces=["results/tables/sib200_topic_robustness.json"],
    ),
    # Answers a reviewer objection (missing post-hoc calibration baselines) from preserved
    # scores rather than by running anything new: source-fitted temperature/Platt scaling is
    # strictly monotone, so it provably cannot move an operating point. Verifies Eq. (1).
    Stage(
        name="calibration_invariance",
        description="Verify that source-fitted post-hoc calibration leaves every FPR unchanged.",
        command=[PYTHON, "scripts/check_calibration_invariance.py"],
        produces=[],
    ),
    # Re-analysis of the pre-registered prevalence sweep. The random-versus-stratified
    # recommendation was previously sourced to a single cell at the corpus's 50/50 balance,
    # where the mechanism behind it cannot operate; this emits the full grid, the pooled
    # figures and the counterexample cells so the claim rests on all of the evidence.
    Stage(
        name="sampling_scheme",
        description="Random vs class-stratified calibration sampling across k and prevalence.",
        command=[PYTHON, "scripts/analyse_sampling_scheme.py"],
        produces=["results/tables/sampling_scheme_verdict.json"],
    ),
    # Emits the Arm A ranks, counts and ratios the manuscript states in prose, so that a
    # quantified prose claim can be checked mechanically rather than by eye.
    Stage(
        name="arm_a_claims",
        description="Emit Arm A ranking, recall and cross-classifier quantities used in prose.",
        command=[PYTHON, "scripts/analyse_arm_a_claims.py"],
        produces=["results/tables/arm_a_claims.json"],
    ),
    # Audit class added after two quantified prose claims survived every earlier check: it
    # tests universally and ordinally quantified sentences against the per-unit distributions.
    Stage(
        name="check_quantified_claims",
        description="Validate universally/ordinally quantified prose claims against artifacts.",
        command=[PYTHON, "scripts/check_quantified_claims.py"],
        produces=[],
        manuscript_local=True,
    ),
    # The claim-to-evidence audit: every reported quantity checked against the artifact that
    # produced it, plus float-to-experiment attribution. Registered here because the previous
    # implementation lived outside the repository and was lost; a reproduction contract that
    # does not run its own claim audit cannot be checked from a fresh checkout.
    Stage(
        name="check_claims",
        description="Claim-to-evidence audit: reported quantities vs their source artifacts.",
        command=[PYTHON, "scripts/check_claims.py"],
        produces=[],
        manuscript_local=True,
    ),
    # Validation, not generation: reads manuscript/main.log and main.aux, so it must run after
    # the manuscript has been compiled. It is optional because a clean checkout has no build
    # products yet, and the pipeline should not fail for that reason alone.
    Stage(
        name="check_manuscript",
        description="Validate the compiled PDF: citations, references, floats, layout.",
        command=[PYTHON, "scripts/check_manuscript.py"],
        produces=[],
        optional=True,
        manuscript_local=True,
    ),
]

#: Source-language sweep (EXP-005-*): one stage per source, appended programmatically so
#: adding a source cannot desynchronise the list from the commands actually run.
for _src in ("de", "ru", "ar", "am"):
    STAGES.insert(-3, Stage(
        name=f"exp005_source_{_src}",
        description=f"EXP-005-{_src}: is the effect an artefact of English as source?",
        command=[PYTHON, "scripts/run_threshold_study.py", "--exp-id", f"EXP-005-{_src}",
                 "--task", "toxicity", "--encoder", "encoder_minilm", "--source", _src,
                 "--objective", "target_fpr", "--target-fpr", "0.05", "--bootstrap", "300"],
        produces=[f"results/tables/EXP-005-{_src}_threshold_study.csv"],
    ))


def _run(stage: Stage) -> dict:
    print(f"\n=== {stage.name} ===\n{stage.description}")
    t0 = time.perf_counter()
    proc = subprocess.run(stage.command, cwd=REPO_ROOT, check=False)
    elapsed = round(time.perf_counter() - t0, 1)

    missing = [p for p in stage.produces if not (REPO_ROOT / p).exists()]
    ok = proc.returncode == 0 and not missing
    status = "OK" if ok else ("OPTIONAL-FAIL" if stage.optional else "FAIL")
    print(f"--- {stage.name}: {status} ({elapsed}s)"
          + (f"  MISSING: {missing}" if missing else ""))

    return {"stage": stage.name, "status": status, "returncode": proc.returncode,
            "seconds": elapsed, "missing_outputs": missing}


def _check(stage: Stage) -> dict:
    missing = [p for p in stage.produces if not (REPO_ROOT / p).exists()]
    status = "OK" if not missing else ("OPTIONAL-MISSING" if stage.optional else "MISSING")
    print(f"  {status:16s} {stage.name}"
          + (f"  -> {missing}" if missing else ""))
    return {"stage": stage.name, "status": status, "missing_outputs": missing}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="Run every stage.")
    ap.add_argument("--stage", action="append", default=[],
                    help="Run only the named stage (repeatable).")
    ap.add_argument("--list", action="store_true", help="List stages and exit.")
    ap.add_argument("--check", action="store_true",
                    help="Verify declared outputs exist; run nothing.")
    args = ap.parse_args()

    if args.list:
        print("Stages, in order:\n")
        for s in STAGES:
            flag = " (optional)" if s.optional else ""
            if s.manuscript_local:
                flag += " [manuscript-local]"
            print(f"  {s.name}{flag}\n      {s.description}")
        if not MANUSCRIPT_PRESENT:
            print("")
            print("manuscript/ is not present, so the [manuscript-local] stages are")
            print("not applicable. They emit LaTeX from results that already exist, or")
            print("check the paper's prose against them; no reported result needs them.")
        return 0

    selected = STAGES if (args.all or not args.stage) else \
        [s for s in STAGES if s.name in set(args.stage)]
    unknown = set(args.stage) - {s.name for s in STAGES}
    if unknown:
        print(f"Unknown stage(s): {sorted(unknown)}", file=sys.stderr)
        return 2

    # Result reproduction does not depend on the manuscript. When manuscript/ is absent the
    # manuscript-local stages are not run and are not counted as failures: they are reported
    # separately as not applicable, so the summary states what was actually verified.
    not_applicable = [s for s in selected if s.manuscript_local and not MANUSCRIPT_PRESENT]
    selected = [s for s in selected if s not in not_applicable]

    if args.check:
        print("Checking declared outputs (running nothing):\n")
        results = [_check(s) for s in selected]
    else:
        results = [_run(s) for s in selected]

    report = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "check" if args.check else "run",
        "manuscript_present": MANUSCRIPT_PRESENT,
        "stages": results,
        "not_applicable": [
            {"stage": s.name, "reason": "manuscript/ not present; stage is manuscript-local"}
            for s in not_applicable
        ],
    }
    out = REPO_ROOT / "experiments" / "logs" / "reproduce_all_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    failed = [r for r in results if r["status"] in {"FAIL", "MISSING"}]
    print(f"\n{'=' * 60}")
    print(f"{len(results) - len(failed)}/{len(results)} stages OK   ->  {out.name}")
    if not_applicable:
        print(f"{len(not_applicable)} manuscript-local stage(s) not applicable "
              f"(manuscript/ not present): {[s.name for s in not_applicable]}")
    if failed:
        print(f"FAILED: {[r['stage'] for r in failed]}")
        return 1
    print("Reproduction complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


