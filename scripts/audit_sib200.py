"""Audit every SIB-200 language config for schema and row-alignment.

Why this exists: SIB-200's scientific value to this project is that it is a PARALLEL corpus
-- the same sentences, the same labels, every language -- which is what removes the content
confound (threat T-I1). That guarantee is only real if the configs are actually row-aligned.

They are not uniform. A spot check found most configs expose an integer `label` and 899 rows,
while some (e.g. `zho_Hans`) expose a string `category` and 1004 rows. A 205-language run
assuming uniformity would either crash or, far worse, silently compare MISALIGNED rows across
languages while looking perfectly healthy.

This script establishes the facts: per-config schema, row count, and whether the label vector
matches the reference language. Its output determines which configs may enter the
parallel-corpus arm, and exclusions are reported rather than quietly applied.

It reads a single repository snapshot (~27 MB) rather than issuing one API call per config.
The per-config approach was tried first and reliably tripped HTTP 429 rate limiting -- 79
configs "failed" for purely network reasons, indistinguishable from genuinely broken ones,
and Kazakh (a priority language) was among the false failures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.loaders import (  # noqa: E402
    load_resources, sib200_available_languages, sib200_read_language, sib200_snapshot,
)

REFERENCE = "eng_Latn"


def main() -> int:
    resources = load_resources()
    spec = resources["datasets"]["sib200"]

    print(f"SIB-200 @ {spec['revision'][:12]} — downloading repository snapshot ...")
    snapshot = sib200_snapshot(resources)
    languages = sib200_available_languages(snapshot)
    print(f"  snapshot at {snapshot}")
    print(f"  {len(languages)} language configs")

    ref_texts, ref_labels, ref_schema = sib200_read_language(snapshot, REFERENCE)
    print(f"  reference {REFERENCE}: schema={ref_schema} n={len(ref_labels)}")

    records = []
    for lang in languages:
        try:
            texts, labels, schema = sib200_read_language(snapshot, lang)
            same_n = len(labels) == len(ref_labels)
            aligned = same_n and all(a == b for a, b in zip(labels, ref_labels))
            records.append({
                "config": lang, "schema": schema, "n": len(labels),
                "n_matches_reference": same_n,
                "labels_match_reference": bool(aligned),
                "n_empty_texts": int(sum(1 for t in texts if not t.strip())),
                "error": None,
            })
        except Exception as exc:  # noqa: BLE001
            records.append({"config": lang, "schema": None, "n": None,
                            "n_matches_reference": False, "labels_match_reference": False,
                            "n_empty_texts": None, "error": f"{type(exc).__name__}: {exc}"})

    ok = [r for r in records if r["labels_match_reference"]]
    bad_schema = [r for r in records if r["schema"] not in (None, ref_schema)]
    bad_n = [r for r in records if r["n"] is not None and not r["n_matches_reference"]]
    errored = [r for r in records if r["error"]]
    aligned_but_odd_schema = [r for r in ok if r["schema"] != ref_schema]

    summary = {
        "revision": spec["revision"],
        "reference_language": REFERENCE,
        "reference_schema": ref_schema,
        "reference_n": len(ref_labels),
        "n_configs": len(languages),
        "n_parallel_aligned": len(ok),
        "n_non_reference_schema": len(bad_schema),
        "n_row_count_mismatch": len(bad_n),
        "n_errored": len(errored),
        "schema_counts": {s: sum(1 for r in records if r["schema"] == s)
                          for s in sorted({r["schema"] for r in records if r["schema"]})},
        "n_aligned_despite_other_schema": len(aligned_but_odd_schema),
        "non_reference_schema_configs": [r["config"] for r in bad_schema],
        "row_count_mismatch_configs": [
            {"config": r["config"], "n": r["n"]} for r in bad_n],
        "errored_configs": [{"config": r["config"], "error": r["error"]} for r in errored],
        "parallel_aligned_configs": [r["config"] for r in ok],
        "excluded_configs": sorted(
            {r["config"] for r in records if not r["labels_match_reference"]}),
    }

    out_dir = REPO_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sib200_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n--- SIB-200 AUDIT ---")
    for k in ("n_configs", "n_parallel_aligned", "n_non_reference_schema",
              "n_row_count_mismatch", "n_errored", "schema_counts",
              "n_aligned_despite_other_schema"):
        print(f"  {k}: {summary[k]}")
    if bad_n:
        print(f"  row-count mismatches: "
              f"{[(r['config'], r['n']) for r in bad_n][:20]}")
    print(f"  EXCLUDED from the parallel arm ({len(summary['excluded_configs'])}): "
          f"{summary['excluded_configs'][:20]}")

    for priority in ("kaz_Cyrl", "ben_Beng"):
        state = "INCLUDED" if priority in set(summary["parallel_aligned_configs"]) else "EXCLUDED"
        print(f"  priority language {priority}: {state}")

    print(f"\nWrote {out_dir / 'sib200_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
