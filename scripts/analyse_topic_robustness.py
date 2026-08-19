"""Is the parallel-corpus content control specific to one one-vs-rest binarisation?

Reviewer objection (Reviewer #2, 2026-08-03): Section V-B rules out content as an explanation
for threshold divergence using SIB-200, but the binarisation actually run was a single positive
topic (`health`), which the manuscript did not disclose. The control therefore rested on one
arbitrary binary task, and the phrase "binarised one-vs-rest from seven topic labels" invites
the broader reading that all seven were used.

This is analysis, not a new experiment: the SIB-200 texts are identical whichever topic is
positive, so the cached sentence embeddings are reused unchanged. Only the label vector, the
logistic head and the per-language threshold search are recomputed. No model is re-downloaded,
no text is re-embedded, no data is collected.

For each positive topic we recompute exactly the three quantities Section V-B reports:
  * the span of per-language budget-optimal thresholds,
  * the fraction of languages whose threshold sits more than 0.05 from the source threshold,
  * the median and maximum |dtau| against the source threshold.

If those are stable across topics, the content-control claim is not an artefact of the
`health` binarisation and needs no narrowing. If they are not, the claim must be scoped to the
binarisation it was measured under.

Usage: python scripts/analyse_topic_robustness.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.loaders import SIB200_LABELS, load_sib200_corpus     # noqa: E402
from src.models.encoders import embed_corpus                        # noqa: E402
from src.models.heads import train_head                             # noqa: E402
from src.eval.thresholds import tune_threshold                      # noqa: E402
from src.eval.metrics import auroc                                  # noqa: E402

BUDGET = 0.05
SOURCE = "eng_Latn"
ENCODER = "encoder_minilm"
SEED = 20260802
#: `health` is the binarisation reported in the paper. The comparison topics are chosen to
#: span the prevalence range of the label set rather than to be favourable.
TOPICS = ["health", "science/technology", "politics", "travel"]
OUT = REPO_ROOT / "results" / "tables" / "sib200_topic_robustness.json"


def main() -> int:
    audit_path = REPO_ROOT / "data" / "processed" / "sib200_audit.json"
    if not audit_path.exists():
        print(f"missing {audit_path}; run scripts/audit_sib200.py first")
        return 1
    languages = json.loads(audit_path.read_text(encoding="utf-8"))["parallel_aligned_configs"]

    rows = []
    for topic in TOPICS:
        corpus = load_sib200_corpus(languages=languages, positive_label=topic, seed=SEED)
        # Cache is keyed by model revision and a checksum of the exact text sequence, and the
        # texts do not depend on which topic is positive -- so this is a cache read.
        emb, _ = embed_corpus(corpus, ENCODER, verbose=False)

        def split_of(lang, name):
            """Embeddings and labels for one split, indexed the same way the corpus is."""
            ls = corpus[lang]
            mask = ls.split == name
            return emb[lang][mask], ls.labels[mask]

        src_tr_x, src_tr_y = split_of(SOURCE, "train")
        src_dv_x, src_dv_y = split_of(SOURCE, "dev")
        head = train_head(src_tr_x, src_tr_y, src_dv_x, src_dv_y,
                          source_language=SOURCE, seed=SEED)
        tau_source = tune_threshold(head.score(src_dv_x), src_dv_y,
                                    objective="target_fpr", target_fpr=BUDGET)

        taus, prevalence, aurocs = {}, [], []
        for lang in languages:
            if lang == SOURCE:
                continue
            dv_x, dv_y = split_of(lang, "dev")
            scores = head.score(dv_x)
            taus[lang] = tune_threshold(scores, dv_y,
                                        objective="target_fpr", target_fpr=BUDGET)
            prevalence.append(float(np.mean(dv_y)))
            # Recorded so the topics can be read against the paper's own AUROC-divergence
            # relationship rather than treated as unexplained variation.
            aurocs.append(float(auroc(scores, dv_y)))

        t = np.array(list(taus.values()))
        d = np.abs(t - tau_source)
        rows.append({
            "positive_topic": topic,
            "reported_in_paper": topic == "health",
            "n_target_languages": int(t.size),
            "mean_positive_prevalence": round(float(np.mean(prevalence)), 4),
            "tau_source": round(float(tau_source), 4),
            "tau_min": round(float(t.min()), 4),
            "tau_max": round(float(t.max()), 4),
            "frac_more_than_0.05_from_source": round(float((d > 0.05).mean()), 4),
            "median_abs_dtau": round(float(np.median(d)), 4),
            "median_target_auroc": round(float(np.median(aurocs)), 4),
            "min_target_auroc": round(float(np.min(aurocs)), 4),
            "max_abs_dtau": round(float(d.max()), 4),
        })
        r = rows[-1]
        print(f"  {topic:20s} prev={r['mean_positive_prevalence']:.3f}  "
              f"tau in [{r['tau_min']:.3f}, {r['tau_max']:.3f}]  "
              f">0.05 from source: {r['frac_more_than_0.05_from_source']:.3f}  "
              f"median |dtau|={r['median_abs_dtau']:.3f}  "
              f"median AUROC={r['median_target_auroc']:.3f}")

    # The paper reports rho(AUROC, |dtau|) = -0.400 WITHIN the health binarisation. If the
    # across-topic variation obeys the same relationship, the one topic without divergence is
    # predicted by the paper's own mechanism rather than being unexplained.
    from scipy.stats import spearmanr
    rho_topics, _ = spearmanr([r["median_target_auroc"] for r in rows],
                              [r["median_abs_dtau"] for r in rows])
    fracs = [r["frac_more_than_0.05_from_source"] for r in rows]
    verdict = {
        "budget": BUDGET, "source": SOURCE, "encoder": ENCODER, "seed": SEED,
        "topics": rows,
        "frac_diverging_min": min(fracs),
        "frac_diverging_max": max(fracs),
        "median_abs_dtau_min": min(r["median_abs_dtau"] for r in rows),
        "median_abs_dtau_max": max(r["median_abs_dtau"] for r in rows),
        "divergence_present_in_every_topic": bool(all(f > 0.30 for f in fracs)),
        "n_topics_with_divergence": int(sum(f > 0.30 for f in fracs)),
        "mean_frac_diverging": round(float(sum(fracs) / len(fracs)), 4),
        "spearman_auroc_vs_median_dtau_across_topics": round(float(rho_topics), 4),
    }
    OUT.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    print(f"\n{json.dumps({k: v for k, v in verdict.items() if k != 'topics'}, indent=2)}")
    print(f"\n-> {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
