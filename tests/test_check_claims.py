"""Regression tests for the claim-to-evidence audit.

A checker that has never been shown to fail is not validated. This project shipped a
validator that silently became a no-op after a rename, and every "PASS" it emitted afterwards
was meaningless. These tests therefore exercise the audit against deliberately broken inputs
and assert that it FAILS, not merely that it runs.

No test mutates the real manuscript or any experiment artifact. Everything that needs a file
uses tmp_path.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The claim-to-evidence audit reads manuscript/sections and manuscript/tables. The
#: manuscript is a private working directory and is not part of the public artifact, so
#: the tests that actually RUN the audit are skipped where it is absent. The tests above
#: them exercise the harness primitives and run everywhere.
MANUSCRIPT_PRESENT = (REPO_ROOT / "manuscript" / "sections").is_dir()
needs_manuscript = pytest.mark.skipif(
    not MANUSCRIPT_PRESENT,
    reason="manuscript/ is not present; the claim audit it validates is manuscript-local")


def _load_module():
    """Import scripts/check_claims.py, which is a script rather than a package module."""
    spec = importlib.util.spec_from_file_location(
        "check_claims", REPO_ROOT / "scripts" / "check_claims.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_claims"] = mod
    spec.loader.exec_module(mod)
    return mod


cc = _load_module()


# --------------------------------------------------------------- comparison primitive

class TestAgrees:
    """`agrees` compares at the precision the manuscript prints. Both halves matter."""

    def test_accepts_correct_rounding(self):
        assert cc.agrees("0.173", 0.17253)      # 3 dp
        assert cc.agrees("3.5", 3.4506)          # 1 dp
        assert cc.agrees("0.016", 0.016436)

    def test_rejects_wrong_value_at_printed_precision(self):
        # The live defect this audit found: 0.016436 must not be printed as 0.017.
        assert not cc.agrees("0.017", 0.016436)
        assert not cc.agrees("0.10", 0.5)
        assert not cc.agrees("141", 143.8)

    def test_precision_is_taken_from_the_printed_value_not_a_fixed_tolerance(self):
        # The same artifact value is accepted at 2 dp and rejected at 3 dp, which a fixed
        # tolerance could not do.
        assert cc.agrees("0.17", 0.1735)
        assert not cc.agrees("0.175", 0.1735)

    def test_handles_signed_values(self):
        assert cc.agrees("+0.064", 0.0637)
        assert cc.agrees("-0.400", -0.4000)


class TestCaptured:
    def test_returns_groups_when_matched(self):
        assert cc.captured("median is $0.173$ here", r"median is \$([\d.]+)\$") == ["0.173"]

    def test_returns_none_when_absent(self):
        assert cc.captured("nothing relevant", r"median is \$([\d.]+)\$") is None


# --------------------------------------------------------------- harness behaviour

class TestHarness:
    def test_numeric_fails_when_claim_absent(self):
        A = cc.Audit()
        A.numeric("T-01", "test", "no such sentence", r"value \$([\d.]+)\$", 1.0, "art")
        assert A.failed and "not found" in A.rows[0][4]

    def test_numeric_fails_on_wrong_value(self):
        A = cc.Audit()
        A.numeric("T-02", "test", "value $0.017$", r"value \$([\d.]+)\$", 0.016436, "art")
        assert A.failed

    def test_numeric_passes_on_right_value(self):
        A = cc.Audit()
        A.numeric("T-03", "test", "value $0.016$", r"value \$([\d.]+)\$", 0.016436, "art")
        assert not A.failed

    def test_arity_mismatch_is_a_failure_not_a_silent_pass(self):
        A = cc.Audit()
        A.numeric("T-04", "test", "a $1$ b $2$", r"a \$([\d.]+)\$ b \$([\d.]+)\$", 1.0, "art")
        assert A.failed and "captured" in A.rows[0][4]


# --------------------------------------------------------------- missing artifacts

class TestMissingArtifact:
    """A missing artifact must fail loudly. There is no skip that can be read as a pass."""

    def test_require_raises_on_missing_file(self, tmp_path):
        with pytest.raises(cc.MissingArtifact):
            cc._require(tmp_path / "definitely_absent.csv")

    def test_require_returns_existing_file(self, tmp_path):
        p = tmp_path / "present.csv"
        p.write_text("x\n", encoding="utf-8")
        assert cc._require(p) == p

    def test_main_reports_failure_exit_code_when_artifact_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cc, "TABLES", tmp_path)          # empty directory
        monkeypatch.setattr(sys, "argv", ["check_claims.py"])
        assert cc.main() == 2                                 # hard failure, not 0


# --------------------------------------------------------------- semantic predicates

class TestNullAssertionPredicate:
    """Equivalent claims phrased differently must be caught; negated forms must not be.

    A literal-string checker passed on this class while a differently-phrased instance of the
    same defect shipped, which is why the predicate is negation-aware rather than a substring
    test.
    """

    NEG_CUES = ("do not", "does not", "not present", "rather than", "declines",
                "cannot", "never", "no longer", "we do not claim")
    FORMS = ("statistically uncorrelated", "were uncorrelated", "are uncorrelated",
             "no association exists", "has no effect", "there is no association")

    def _asserted(self, body: str):
        import re
        out = []
        for form in self.FORMS:
            for m in re.finditer(re.escape(form), body):
                if not any(c in body[max(0, m.start() - 110):m.start()] for c in self.NEG_CUES):
                    out.append(form)
        return out

    def test_flags_a_positive_assertion(self):
        body = "because their realised operating points are uncorrelated, a deployer could not"
        assert self._asserted(body) == ["are uncorrelated"]

    def test_does_not_flag_a_negated_use(self):
        body = ("all three are estimated on 14 languages, and we do not present them as "
                "evidence that no association exists.")
        assert self._asserted(body) == []

    def test_catches_a_rephrasing_a_substring_check_would_miss(self):
        # Same proposition, different words: a checker keyed to one phrase would pass here.
        body = "the two rates were uncorrelated across the languages we tested"
        assert self._asserted(body) == ["were uncorrelated"]

    def test_clean_text_is_clean(self):
        body = ("its interval is $[-0.51, +0.60]$; we report it as uninformative rather than "
                "as evidence of independence.")
        assert self._asserted(body) == []


class TestAttributionPredicate:
    """A float must come from the experiment its citing section reports (D-018).

    Verifying that a quoted number exists *somewhere* is what let a figure generated from the
    F1 pilot survive three audits while the section citing it reported a budget result.
    """

    def test_detects_a_figure_sourced_from_a_retired_experiment(self, tmp_path):
        manifest = [{"figure": "fig1", "source_data": "results/tables/EXP-001_pilot.csv"}]
        p = tmp_path / "figure_manifest.json"
        p.write_text(json.dumps(manifest), encoding="utf-8")
        loaded = cc.load_json(p)
        offenders = [f["figure"] for f in loaded if "EXP-001" in f.get("source_data", "")]
        assert offenders == ["fig1"]

    def test_detects_a_figure_with_no_declared_source(self, tmp_path):
        manifest = [{"figure": "fig2"}]
        p = tmp_path / "m.json"
        p.write_text(json.dumps(manifest), encoding="utf-8")
        loaded = cc.load_json(p)
        assert [f["figure"] for f in loaded if not f.get("source_data")] == ["fig2"]

    def test_accepts_a_correctly_attributed_manifest(self, tmp_path):
        manifest = [{"figure": "fig1", "source_data": "results/tables/EXP-002_threshold_study.csv"}]
        p = tmp_path / "m.json"
        p.write_text(json.dumps(manifest), encoding="utf-8")
        loaded = cc.load_json(p)
        assert not [f["figure"] for f in loaded if "EXP-001" in f.get("source_data", "")]
        assert not [f["figure"] for f in loaded if not f.get("source_data")]


@needs_manuscript
class TestSelfCheck:
    """SELF-00 exists so that a no-op audit cannot masquerade as a passing one."""

    def test_self_check_is_present_and_passes_on_a_working_harness(self):
        A = cc.run_audit()
        self_rows = [r for r in A.rows if r[0] == "SELF-00"]
        assert len(self_rows) == 1
        assert self_rows[0][2] is True

    def test_self_check_would_fail_if_the_primitive_were_neutered(self, monkeypatch):
        # Simulate the historical no-op: make the comparison always return True.
        monkeypatch.setattr(cc, "agrees", lambda printed, truth: True)
        ok = (cc.agrees("0.173", 0.17253) and not cc.agrees("0.10", 0.5))
        assert ok is False    # the SELF-00 conjunction collapses, so the audit reports FAIL


# --------------------------------------------------------------- determinism / end to end

@needs_manuscript
class TestEndToEnd:
    def test_audit_runs_and_is_deterministic(self):
        a1, a2 = cc.run_audit(), cc.run_audit()
        assert [(r[0], r[2]) for r in a1.rows] == [(r[0], r[2]) for r in a2.rows]

    def test_check_ids_are_unique(self):
        ids = [r[0] for r in cc.run_audit().rows]
        assert len(ids) == len(set(ids))

    def test_every_check_names_its_evidence(self):
        assert all(r[3] for r in cc.run_audit().rows)

    def test_audit_has_meaningful_coverage(self):
        A = cc.run_audit()
        assert len(A.rows) >= 60
        assert len({r[1] for r in A.rows}) >= 8      # distinct categories

    def test_no_absolute_paths_hard_coded(self):
        src = (REPO_ROOT / "scripts" / "check_claims.py").read_text(encoding="utf-8")
        assert "C:\\" not in src and "/home/" not in src and "E:\\" not in src

    def test_audit_makes_no_network_calls(self):
        src = (REPO_ROOT / "scripts" / "check_claims.py").read_text(encoding="utf-8")
        for token in ("urllib", "requests", "http://", "https://api", "socket"):
            assert token not in src
