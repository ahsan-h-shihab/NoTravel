"""Smoke tests for figure generation.

These do not check that a figure is *beautiful* -- they check that it can be produced at all
from a realistically-shaped frame, that both the vector and raster outputs are written, and
that the validated palette constraints hold. A figure that crashes during the final
manuscript build is a schedule risk; catching it here is cheap.

The palette assertion is the important one: the categorical slots were validated against a
white print surface with the all-pairs gate, and adding a fourth slot FAILS the
normal-vision floor. Encoding that cap as a test stops a future contributor from quietly
appending a fourth series colour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import figures as F

LANGS = ["en", "de", "fr", "es", "ru", "uk", "it", "ja", "zh", "ar", "hi", "he", "am", "tt", "hin"]
STRATEGIES = ["fixed_half", "source_global", "quantile_match", "sld_em", "bbse",
              "few_label_32_strat", "target_full"]


def test_categorical_palette_is_capped_at_three():
    """Validated all-pairs on a white surface for 3 slots; a 4th fails the
    normal-vision floor (yellow vs orange, deltaE 13.7). The cap is load-bearing."""
    assert len(F.SERIES) == 3
    assert len(F.MARKERS) >= len(F.SERIES)
    assert len(F.LINESTYLES) >= len(F.SERIES)


def test_sequential_ramp_is_single_hue_and_monotone():
    """A sequential ramp must darken monotonically; a rainbow would misencode magnitude."""
    def luminance(hex_color: str) -> float:
        r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    lums = [luminance(c) for c in F.SEQ]
    assert all(a > b for a, b in zip(lums, lums[1:])), "sequential ramp is not monotone"


def test_threshold_divergence_figure(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"language": LANGS,
                       "tau_language": rng.uniform(0.2, 0.8, len(LANGS)),
                       "tau_source": 0.47})
    paths = F.fig_threshold_divergence(df, "t", tmp_path)
    assert {p.suffix for p in paths} == {".pdf", ".png"}
    assert all(p.exists() and p.stat().st_size > 1000 for p in paths)


def test_strategy_comparison_figure(tmp_path):
    rng = np.random.default_rng(1)
    df = pd.DataFrame([{"strategy": s, "f1_gap_to_full": max(0.0, rng.normal(0.05, 0.03))}
                       for s in STRATEGIES for _ in range(15)])
    paths = F.fig_strategy_comparison(df, "s", tmp_path)
    assert all(p.exists() for p in paths)


def test_strategy_comparison_respects_explicit_order(tmp_path):
    """Strategy order encodes the information spectrum; it must not be silently re-sorted."""
    rng = np.random.default_rng(2)
    df = pd.DataFrame([{"strategy": s, "f1_gap_to_full": rng.random()}
                       for s in STRATEGIES for _ in range(5)])
    paths = F.fig_strategy_comparison(df, "s2", tmp_path, order=STRATEGIES)
    assert all(p.exists() for p in paths)


def test_label_efficiency_figure(tmp_path):
    rng = np.random.default_rng(3)
    df = pd.DataFrame([{"k": k, "sampling": s,
                        "f1_gap_to_full": 0.08 / np.sqrt(k) + rng.normal(0, 0.002)}
                       for k in (8, 16, 32, 64, 128, 256) for s in ("stratified", "random")])
    paths = F.fig_label_efficiency(df, "k", tmp_path, reference_lines={"label-free": 0.02})
    assert all(p.exists() for p in paths)


def test_auroc_vs_gap_figure(tmp_path):
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"language": LANGS,
                       "test_auroc": rng.uniform(0.65, 0.95, len(LANGS)),
                       "f1_gap": rng.uniform(0.0, 0.15, len(LANGS)),
                       "tier": rng.choice(["high", "mid", "low"], len(LANGS))})
    assert all(p.exists() for p in F.fig_auroc_vs_gap(df, "a", tmp_path, tier_col="tier"))
    # Must also work without a tier column -- the single-series case needs no legend.
    assert all(p.exists() for p in F.fig_auroc_vs_gap(df, "a2", tmp_path))


def test_auroc_vs_gap_label_offsets_separate_near_identical_points(monkeypatch, tmp_path):
    """Two languages that score almost identically must be able to have readable labels.

    In the real data fr and ru differ by 0.003 AUROC and 0.0005 F1, so the default label
    offset prints both on top of each other and reads "fru". `label_offsets` moves a text
    label and nothing else; this pins both halves of that contract.
    """
    import matplotlib.pyplot as plt

    df = pd.DataFrame({"language": ["fr", "ru", "en"],
                       "test_auroc": [0.9034, 0.9061, 0.60],
                       "f1_gap": [-0.0005, 0.0, 0.20],
                       "tier": ["high", "mid", "low"]})
    grabbed = {}
    monkeypatch.setattr(F, "_save", lambda fig, name, out_dir=None: grabbed.setdefault("fig", fig) and [])

    def label_boxes(**kw):
        plt.close("all")
        F.fig_auroc_vs_gap(df, "x", tmp_path, tier_col="tier", **kw)
        fig = grabbed.pop("fig")
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        ax = fig.axes[0]
        boxes = {t.get_text(): t.get_window_extent(r) for t in ax.texts}
        offsets = {t.get_text(): t.xyann for t in ax.texts}
        data = [(np.asarray(c.get_offsets())).tolist() for c in ax.collections]
        return boxes, offsets, data

    def overlap(a, b):
        return not (a.x1 <= b.x0 or b.x1 <= a.x0 or a.y1 <= b.y0 or b.y1 <= a.y0)

    default_boxes, default_off, default_data = label_boxes()
    fixed_boxes, fixed_off, fixed_data = label_boxes(label_offsets={"fr": (-7, 5), "ru": (3, -6)})

    assert overlap(default_boxes["fr"], default_boxes["ru"]), "premise: default offsets collide"
    assert not overlap(fixed_boxes["fr"], fixed_boxes["ru"]), "offsets must separate the labels"
    assert fixed_off["fr"] == (-7, 5) and fixed_off["ru"] == (3, -6)
    assert fixed_off["en"] == default_off["en"] == (3, 2), "unlisted languages keep the default"
    assert fixed_data == default_data, "moving a label must not move any plotted point"


def test_single_language_does_not_crash(tmp_path):
    """Degenerate input appears in per-encoder subsets; it must degrade, not explode."""
    df = pd.DataFrame({"language": ["en"], "tau_language": [0.5], "tau_source": [0.47]})
    assert all(p.exists() for p in F.fig_threshold_divergence(df, "one", tmp_path))
