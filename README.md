# NoTravel: Cross-Lingual Threshold Transfer in Multilingual AI

Reproduction artifact for *"The Operating Point Does Not Travel: Cross-Lingual Threshold
Transfer Fails by Over- and Under-Firing in Multilingual Text Classifiers"* (submitted to
IEEE Access).

Everything needed to regenerate and independently verify every numerical result, statistical
test and figure reported in the paper. CPU-only; no GPU, no LaTeX, no paid API.

The manuscript source and PDF are not published here before publication. Nothing in the
reproduction pipeline depends on them.

## Reproduce

```bash
python -m pip install -r environment/requirements.txt
python scripts/reproduce_all.py --all      # regenerate everything (hours, CPU-only)
python scripts/reproduce_all.py --check    # verify an existing checkout (fast)
python scripts/reproduce_all.py --list     # show the stages without running them
```

`--all` runs 24 stages end to end: it downloads the pinned datasets and models, recomputes
every score, and rebuilds every result table and figure. `--check` verifies the committed
artifacts without recomputing them. Both must end `24/24 stages OK`.

One model is fetched outside the HuggingFace hub API because that path is unreliable for it.
If a run stops with a `local_dir ... is absent` error, follow the command it prints:

```bash
python scripts/fetch_model_direct.py --repo unitary/multilingual-toxic-xlm-roberta \
  --revision 4ad6f5c104d9ce813a1a2f33cac0c5b579ef6ee5 \
  --out data/models/unitary-multilingual-toxic-xlm-roberta
```

## Verify

```bash
python -m pytest tests/                 # 116 tests; 108 run here, 8 skip (see below)
python scripts/verify_results.py        # re-derive headline quantities from raw scores
```

`verify_results.py` recomputes the headline numbers directly from the preserved per-example
scores, without importing the analysis engine, so an error common to both would still show.
Every experiment output additionally carries a SHA-256 in its `provenance.json` (55
checksums), verified by `reproduce_all.py --check`.

Six further stages validate the manuscript itself: three emit its LaTeX tables, figures and
appendix from results that already exist, and three check its prose against those results.
They require `manuscript/`, which is not published, so `reproduce_all.py` reports them as not
applicable rather than as failures, and the eight tests covering the claim audit skip for the
same reason. No reported result depends on any of them.

## Layout

```
scripts/                reproduce_all.py, the analysis scripts, the verification scripts
src/                    data loading, model loading, evaluation protocol, statistics, figures
configs/                resources.yaml pins every dataset and model by revision hash
experiments/runs/       per experiment: provenance.json (revisions, seeds, versions,
                        SHA-256 of every output) and per_example_scores.parquet
results/{tables,figures}  every artifact a reported number is computed from
tests/                  unit tests for the metrics and statistics
environment/            requirements.txt and a captured environment snapshot
```

Per-example scores are released as Parquet (`language`, `split`, `score`, `label`), so every
result can be re-derived independently of this analysis code.

## Third-party material

Datasets and models are referenced by pinned revision and downloaded at run time; none is
redistributed here. Sources and exact terms: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
No repository licence has been chosen yet, so default copyright currently applies.

## Citation

See [`CITATION.cff`](CITATION.cff).
