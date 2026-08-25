# Licence scope and third-party terms

This repository is **not** uniformly licensed. It contains original code, generated result
artefacts derived from third-party corpora, and references to third-party datasets and models
that are downloaded at run time and are not redistributed here. Those three categories have
different terms, and this file states which applies where.

Every licence recorded below was read from the upstream record itself — the HuggingFace
repository metadata for each dataset and model — and not inferred from the fact that an
artefact is downloadable.

---

## 1. Original work — Apache License 2.0

The `LICENSE` file at the repository root is the Apache License 2.0 and applies to the
material this project authored:

| Path | Contents |
|---|---|
| `scripts/` | reproduction runner, analysis and verification scripts |
| `src/` | data loading, model loading, evaluation protocol, statistics, figures |
| `tests/` | unit tests |
| `configs/` | experiment and resource configuration |
| `environment/` | dependency pins and the captured environment snapshot |
| root files | `README.md`, `CITATION.cff`, `THIRD_PARTY_NOTICES.md`, this file, `.gitignore` |

Every file above was written for this project. No third-party source file is vendored into
any of these directories.

## 2. Derived result artefacts — upstream terms apply

| Path | Contents |
|---|---|
| `experiments/runs/*/per_example_scores.parquet` | four columns: `language`, `split`, `score`, `label` |
| `experiments/runs/*/provenance.json` | revisions, seeds, versions, SHA-256 of every output |
| `results/tables/`, `results/figures/` | aggregates and plots computed from the above |

These are measurements produced by running the pinned models over the pinned corpora. **No
corpus text is present** — the score files carry no sentence, document or token from any
dataset. What they do carry is a model score and the corpus's own label for each example, so
they are derived from the upstream corpora and remain subject to the upstream terms in
section 3. The Apache-2.0 grant in section 1 does **not** extend to them.

## 3. Third-party datasets and models — referenced, not redistributed

Nothing in this table is contained in this repository. Each is downloaded at run time from
the revision pinned in `configs/resources.yaml` and recorded in each run's `provenance.json`.

| Resource | Identifier | Declared licence | Verified from |
|---|---|---|---|
| Parallel corpus | `mteb/sib200` | **CC-BY-SA-4.0** | HF repo metadata (`license:cc-by-sa-4.0`) |
| Toxicity corpus | `textdetox/multilingual_toxicity_dataset` | **OpenRAIL++** | HF repo metadata (`license:openrail++`) |
| Encoder 1 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | **Apache-2.0** | HF repo metadata |
| Encoder 2 | `intfloat/multilingual-e5-small` | **MIT** | HF repo metadata |
| Deployed classifier 1 | `unitary/multilingual-toxic-xlm-roberta` | **Apache-2.0** | HF repo metadata |
| Deployed classifier 2 | `citizenlab/distilbert-base-multilingual-cased-toxicity` | **none declared** | HF repo metadata carries no licence tag |

Two of these carry obligations that reach beyond attribution, which is why section 2 exists:

- **`mteb/sib200` is CC-BY-SA-4.0**, a share-alike licence. Whether a table of model scores
  and corpus labels, carrying none of the corpus text, constitutes Adapted Material under
  that licence is a question this repository does not purport to settle. Section 2 is
  deliberately conservative: it asserts no permissive grant over those artefacts.
- **`textdetox/multilingual_toxicity_dataset` is OpenRAIL++**, which carries use-based
  restrictions rather than being a pure open-source licence. Those restrictions attach to
  derivatives.

`citizenlab/distilbert-base-multilingual-cased-toxicity` declares no licence at all. Nothing
from it is redistributed here; it is referenced by revision only.

## 4. Dependencies

Declared in `environment/requirements.txt` and installed by the user, not vendored:
numpy, scipy, pandas, pyarrow (Apache-2.0/BSD-3), scikit-learn, statsmodels (BSD-3),
torch (BSD-3), transformers, tokenizers, datasets, huggingface-hub, sentence-transformers,
nltk (Apache-2.0), sentencepiece (Apache-2.0), PyYAML, pytest, matplotlib, seaborn, tqdm
(MIT/BSD/PSF-family). None is copyleft and none is redistributed, so none constrains the
Apache-2.0 grant in section 1.

## 5. What is not in this repository

The manuscript source, the submitted PDF, the bibliography, the IEEE Access class and style
files, the fonts the LaTeX build requires and the IEEE branding assets are a private working
directory until publication. No reported result depends on any of them:
`scripts/reproduce_all.py` never invokes LaTeX and needs no font.
