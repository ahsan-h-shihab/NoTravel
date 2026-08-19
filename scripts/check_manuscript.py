"""Validate the compiled manuscript: citations, references, floats, and layout warnings.

The final-validation checks in this project were previously done by eye, and that missed
things a build log states plainly: a dangling `\\ref`, a bibliography entry no longer cited,
a float that overflows the column. This script turns each of those into a pass/fail.

One class of warning is expected and is reported separately rather than counted as a defect:
`ieeeaccess.cls` emits two overfull \\hbox warnings per page at exactly \\textwidth from its
running-head \\parbox, and one at the title. The class's own sample document (access.tex,
shipped with the template) produces the same warnings, so they are a property of the class,
not of this manuscript. Only overfulls that are NOT of that shape are treated as defects.

Usage: python scripts/check_manuscript.py [--tex manuscript/main.tex]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Overfull boxes produced by the class itself, in points. 505.12177pt is exactly \textwidth
#: (the running-head parbox); 9.2679pt is the title block.
CLASS_INTRINSIC_PT = {"505.12177", "9.2679"}


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def check_citations(aux: str, bib: str) -> list[str]:
    problems = []
    in_bib = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited: set[str] = set()
    for group in re.findall(r"\\citation\{([^}]*)\}", aux):
        cited.update(k.strip() for k in group.split(",") if k.strip())
    cited.discard("IEEEtran")            # emitted by the bibliography style itself

    orphan = sorted(in_bib - cited)
    missing = sorted(cited - in_bib)
    if orphan:
        problems.append(f"bibliography entries never cited ({len(orphan)}): "
                        f"{', '.join(orphan)}")
    if missing:
        problems.append(f"cited but absent from refs.bib: {', '.join(missing)}")
    print(f"  citations: {len(cited)} distinct cited, {len(in_bib)} in refs.bib")
    return problems


def check_labels(aux: str, log: str) -> list[str]:
    problems = []
    defined = set(re.findall(r"\\newlabel\{([^}]+)\}", aux))
    undefined = set(re.findall(r"Reference `([^']+)' on page", log))
    if undefined:
        problems.append(f"undefined references: {', '.join(sorted(undefined))}")
    if re.search(r"Citation `[^']+' (on page .* )?undefined", log):
        problems.append("log reports undefined citations")
    if "There were undefined references" in log:
        problems.append("log: 'There were undefined references'")
    # A label defined but never referenced is not an error, but a float nobody points at
    # usually means a lost cross-reference, so it is worth surfacing.
    referenced: set[str] = set()
    for group in re.findall(r"\\(?:ref|autoref|eqref)\{([^}]*)\}", log):
        referenced.add(group)
    print(f"  labels: {len(defined)} defined")
    return problems


def check_floats(tex_dir: Path, aux: str) -> list[str]:
    """Every figure and table label must exist and be referenced from the body text."""
    problems = []
    body = "\n".join(read(p) for p in sorted(tex_dir.glob("sections/*.tex")))
    body += "\n".join(read(p) for p in sorted(tex_dir.glob("tables/*.tex")))
    body += read(tex_dir / "main.tex")

    float_labels = set(re.findall(r"\\label\{((?:fig|tab):[^}]+)\}", body))
    refs = set()
    for group in re.findall(r"\\ref\{([^}]*)\}", body):
        refs.add(group.strip())

    unref = sorted(float_labels - refs)
    dangling = sorted(r for r in refs if r.startswith(("fig:", "tab:"))
                      and r not in float_labels)
    if unref:
        problems.append(f"floats never referenced in the text: {', '.join(unref)}")
    if dangling:
        problems.append(f"\\ref to a non-existent float: {', '.join(dangling)}")
    print(f"  floats: {len(float_labels)} labelled, {len(unref)} unreferenced")
    return problems


def check_layout(log: str) -> list[str]:
    problems = []
    overfull = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", log)
    ours = [pt for pt in overfull if pt not in CLASS_INTRINSIC_PT]
    intrinsic = len(overfull) - len(ours)
    if ours:
        problems.append(f"content overfull boxes: {', '.join(pt + 'pt' for pt in ours)}")
    print(f"  layout: {intrinsic} class-intrinsic overfull (expected), "
          f"{len(ours)} attributable to content")

    for pattern, label in [(r"LaTeX Warning: (?!Citation|Reference|Label)(.+)", "warning")]:
        seen = {m.strip() for m in re.findall(pattern, log)}
        for w in sorted(seen):
            if "Float specifier changed" in w or "float specifier changed" in w:
                problems.append(f"float placement rejected by the class: {w}")
            elif "rerun" in w.lower():
                problems.append(f"needs another pass: {w}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", type=Path, default=REPO_ROOT / "manuscript" / "main.tex")
    args = ap.parse_args()

    tex_dir = args.tex.parent
    stem = args.tex.stem
    log_p, aux_p, bib_p = (tex_dir / f"{stem}.log", tex_dir / f"{stem}.aux",
                           tex_dir / "refs.bib")
    for p in (log_p, aux_p, bib_p):
        if not p.exists():
            print(f"missing {p}; compile first")
            return 1
    log, aux, bib = read(log_p), read(aux_p), read(bib_p)

    pages = re.search(r"Output written on .*?\((\d+) pages", log)
    print(f"manuscript: {pages.group(1) if pages else '?'} pages")

    problems: list[str] = []
    problems += check_citations(aux, bib)
    problems += check_labels(aux, log)
    problems += check_floats(tex_dir, aux)
    problems += check_layout(log)

    if problems:
        print("\nFAIL")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nPASS: no undefined references, no uncited entries, no content overfull boxes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
