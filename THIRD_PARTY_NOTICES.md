# Third-party notices

This repository contains material we did not author. This file records, for every such item,
where it came from and what its terms are. Terms were read from the artefact itself or from
the authoritative upstream record, never inferred from the fact that a file is downloadable.

Nothing here is licensed by us. Our own licence, when added, applies only to the material
listed under "Our own work" and never to anything in this file.

---

## Referenced by the public pipeline, not redistributed

| Path | Upstream | Terms | Redistribution | Modified by us | Needed to reproduce |
|---|---|---|---|---|---|
| `data/models/unitary-multilingual-toxic-xlm-roberta/` | `unitary/multilingual-toxic-xlm-roberta` | **Apache-2.0** (declared on the model repository) | Permitted, notice required | No | Fetched at run time by `scripts/fetch_model_direct.py`; not committed here |

No third-party file is redistributed in this repository. Every dataset and model is
downloaded at run time from the revision pinned in `configs/resources.yaml`.

## Referenced by pinned revision, **not** redistributed

No text, image or model weight from any of these is contained in this repository. The
pipeline downloads them at run time from the pinned revision recorded in each experiment's
`provenance.json`.

| Resource | Identifier | Revision | Declared licence |
|---|---|---|---|
| Toxicity corpus | `textdetox/multilingual_toxicity_dataset` | `01907546…` | **OpenRAIL++** |
| Parallel corpus | `mteb/sib200` | `83d03204…` | **CC-BY-SA-4.0** |
| Encoder 1 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | pinned | **Apache-2.0** |
| Encoder 2 | `intfloat/multilingual-e5-small` | pinned | **MIT** |
| Deployed classifier 1 | `unitary/multilingual-toxic-xlm-roberta` | `4ad6f5c1…` | **Apache-2.0** |
| Deployed classifier 2 | `citizenlab/distilbert-base-multilingual-cased-toxicity` | pinned | **none declared upstream** |

Two of these deserve a note rather than silence. `mteb/sib200` is CC-BY-SA-4.0, a
share-alike licence; we redistribute none of its text, only derived numeric scores and an
audit of its configuration counts. `citizenlab/distilbert-base-multilingual-cased-toxicity`
declares no licence at all on its model repository; we redistribute none of it and reference
it by revision only.

The per-example score files under `experiments/runs/*/per_example_scores.parquet` contain
four columns — `language`, `split`, `score`, `label` — and no corpus text.

## Not published

The manuscript is a private working directory until publication: its LaTeX source, the
submitted PDF, the bibliography, the IEEE Access class and style files, `IEEEtran.bst` and
the manuscript figures are not in this repository. With them go the third-party items the
LaTeX build requires and which we have no right to redistribute in any case: the Adobe/ITC
Formata, ITC Giovanni and Times font programs, whose own headers read *"All Rights
Reserved"*, and the IEEE Access logo and branding assets, which carry no licence.

No reported result depends on any of this. `scripts/reproduce_all.py` never invokes LaTeX,
needs no font, and passes 24/24 stages without the manuscript present.

## Unresolved

Nothing remains unresolved. `ieeeaccess.cls` and `spotcolor.sty`, whose redistribution
rights could not be established from any authoritative source, are inside the unpublished
manuscript directory and are therefore not distributed.

---

## Our own work

Everything in this repository was authored for this project: `scripts/`, `src/`, `tests/`,
`configs/`, `environment/`, and all generated artefacts under `results/` and `experiments/`.

A repository licence has **not** been chosen. The proposed boundary, for decision:

- **Code** — `scripts/`, `src/`, `tests/`, `configs/`, `analysis/`: a permissive licence
  (Apache-2.0 or MIT).
- **Result artefacts** — `results/`, `experiments/`: CC-BY-4.0 is the usual choice for this
  venue.
- **Third-party material** — everything in this file: unchanged, under its own terms, never
  covered by ours.

Until a licence file exists, default copyright applies and no reuse rights are granted.
