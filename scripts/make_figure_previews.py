"""Rasterise selected publication figures to PNG previews for the README.

The PDFs in results/figures/ are the publication artefacts and are never modified. This script
only reads them and writes raster copies to results/figures/previews/, so the README can show
them inline on GitHub (which does not render PDFs inside Markdown). The previews are pure
rasterisations: nothing is redrawn, re-styled, cropped or re-plotted.

Only the figures the README shows are produced. Regenerating the figures with
scripts/make_figures.py does not touch the previews; rerun this script afterwards.

Rasteriser: PyMuPDF if it is installed, otherwise Poppler's `pdftoppm`. Neither is needed for
any reported result, which is why neither is in environment/requirements.txt.

Usage: python scripts/make_figure_previews.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "results" / "figures"
DEST = SRC / "previews"

#: Figures 1, 2 and 5 are the main README figures; Figures 3 and 4 are its compact
#: "supporting figures" row.
PREVIEWS = [
    "fig1_threshold_divergence",
    "fig2b_strategy_tradeoff",
    "fig5_degeneracy_vs_prevalence",
    "fig3_label_efficiency",
    "fig4_auroc_vs_gap",
]

#: Chosen so the narrowest figure (a single IEEE column, ~216 pt) still yields ~900 px.
DPI = 300


def _with_pymupdf(pdf: Path, png: Path) -> bool:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return False
    with fitz.open(pdf) as doc:
        # alpha=False composites onto white, so the preview reads on light and dark themes.
        doc[0].get_pixmap(dpi=DPI, alpha=False).save(png)
    return True


def _with_pdftoppm(pdf: Path, png: Path) -> bool:
    exe = shutil.which("pdftoppm")
    if exe is None:
        return False
    stem = png.with_suffix("")
    subprocess.run([exe, "-png", "-r", str(DPI), "-singlefile", str(pdf), str(stem)], check=True)
    return True


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in PREVIEWS:
        pdf = SRC / f"{name}.pdf"
        png = DEST / f"{name}.png"
        if not pdf.exists():
            print(f"missing {pdf}; run scripts/make_figures.py first")
            return 1
        if not (_with_pymupdf(pdf, png) or _with_pdftoppm(pdf, png)):
            print("no rasteriser available: install PyMuPDF (`pip install pymupdf`) "
                  "or Poppler (`pdftoppm`)")
            return 1
        print(f"  OK   {png.relative_to(REPO_ROOT).as_posix()}  ({png.stat().st_size / 1024:.0f} KiB)")
    print(f"\n{len(PREVIEWS)} previews -> {DEST.relative_to(REPO_ROOT).as_posix()}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
