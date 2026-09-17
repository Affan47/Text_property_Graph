# EPSS-TPG

Navigation: [Start Here](../README.md) | [Current folder map](../07_maintenance/01_LAYOUT_AND_COMPATIBILITY.md).
This practical guide predates the numbered layout. Its older paths remain available through compatibility links; use the current folder map when browsing the implementation.

A graph neural network for predicting whether a CVE will be exploited in
the wild, trained on the CVE's own description text plus the usual
vulnerability metadata. The interesting bit is *how* the text gets into
the model: each CVE is turned into a Text Property Graph that captures
syntactic, sequential, semantic, discourse, and security relations, and
the GNN reads the graph rather than a flat embedding.

This README is the practical guide. It explains what's in the repo,
where to put the source data, and the exact command line for every
script you might want to run.

A note on running the tools. Every shell script in `scripts/` figures
out its own location and resolves paths from there, so you can launch
them from any working directory. Python entry points use
`python -m epss.X` or `python <script>.py` and behave the same way as
long as the project root is on `PYTHONPATH` — which it is whenever you
run them from the project root or use the `-m` form.

## Setup

You need Python 3.10 or newer, PyTorch 2.x (CUDA build if you have a
GPU, CPU build otherwise), and PyTorch Geometric 2.5+. The TPG
construction also needs spaCy with the small English model
(`python -m spacy download en_core_web_sm`) and Hugging Face
Transformers to load SecBERT (`jackaduma/SecBERT`). Everything else is
the usual scientific Python stack — numpy, pandas, scikit-learn, tqdm.

Quick sanity check that PyTorch sees your GPU:

```bash
python -c "import torch; print('cuda available:', torch.cuda.is_available())"
```

If it prints `False`, training will fall back to CPU and run roughly
five to ten times slower. Inference is fine either way.

## How the project is laid out

```
EPSS_TPG/
│
├── epss/            Main Python package. Training, dataset construction,
│                    the GNN itself, the inference entry points, and the
│                    profiling utilities. Importable as epss.X or runnable
│                    as python -m epss.X.
│
├── tpg/             The Text Property Graph backend. Builds the per-CVE
│                    graph using a spaCy frontend plus a hybrid SecBERT +
│                    rule-based security overlay, then exports it as a
│                    PyG tensor for the GNN. Also hosts the generic
│                    domain layer (tpg/schema/domain.py + PatternFrontend)
│                    that lets any domain be declared as data, and the
│                    NetworkX/GraphML/Cypher exporters + GraphSON import.
│
├── tpg_app/         The deployable TPG document-intelligence platform:
│                    universal extractors (PDF/DOCX/HTML/MD/CSV/JSON/URL),
│                    SQLite+FTS5 knowledge store, REST API, web UI, and
│                    CLI. See tpg_app/README.md. Start with:
│                    python -m tpg_app.cli serve
│
├── tpg_chatbot/     The original CLI question-answering chatbot over a
│                    JSON graph store (predecessor of tpg_app).
│
├── analysis/        Stand-alone analysis scripts (CSV profiling, feature
│                    verification, plot regeneration for an existing
│                    checkpoint). Run as python analysis/<name>.py.
│
├── inference/       Stand-alone CLI for scoring CVEs you fetch live from
│                    NVD, with five modes: by ID, by file of IDs, by date
│                    range, by recent days, or temporal evaluation against
│                    ground truth.
│
├── scripts/         Shell-script batch entry points, grouped by purpose:
│                    training/ for end-to-end retrains, inference/ for
│                    test-only evaluations across saved checkpoints, and
│                    analysis/ for maintenance.
│
├── examples/        Small standalone scripts that exercise the TPG and
│                    GNN APIs in isolation. Useful as a smoke test after
│                    install.
│
├── data/            Source data and labelled records. The per-experiment
│                    pyg graph caches that training generates are large
│                    and regeneratable, so they're not tracked.
│
├── outputs/         Where every training run drops its artefacts, grouped
│                    by dataset family — social_media/, megavul/, nvd_kev/,
│                    and security_ablation/. Each run directory holds the
│                    test_results.json metrics summary, the experiment
│                    config that produced it, and (for the security
│                    ablation) the full set of checkpoints and predictions.
│
├── inference_results/   Per-family summaries produced by the inference
│                        batch scripts under scripts/inference/.
│
├── datasets_info/   Dataset profiling artefacts: per-LLM profiles,
│                    ablation summaries, per-source CSV statistics.
│
├── SummTPGVul/      Git submodule holding the source CSVs for both
│                    dataset families. See "Source data" below.
│
└── README.md        You are here.
```

Two things the layout doesn't make obvious. First, the training batches
under `scripts/training/` are thin wrappers around
`python -m epss.run_pipeline` with flags pre-filled for each ablation.
Second, there are two `infer.py` files and they do different things:
`inference/infer.py` is the user-facing CLI for scoring fresh CVEs from
NVD, while `epss/infer.py` is the package-internal temporal scorer you
invoke as `python -m epss.infer`. Mixing them up is the most common
papercut on a fresh checkout.

## Source data

The training scripts read the source CSVs from the `SummTPGVul` git
submodule, which lives one directory above the project root. After
cloning the parent repository, pull the submodule contents:

```bash
git submodule update --init --recursive
```

The CSVs are Git LFS objects, so install LFS and fetch them:

```bash
sudo apt install -y git-lfs && git lfs install
cd ../SummTPGVul && git lfs pull && cd -
```

The submodule layout the scripts expect:

```
../SummTPGVul/SummVul/
├── Social_Media_Dataset/Data_Files/
│   ├── gpt_combined_summ.csv
│   ├── gemma_combined_summ.csv
│   └── mistral_combined_summ.csv
└── Data_Files/megavul/
    ├── gpt.csv
    ├── gemma.csv
    └── mistral.csv
```

If you keep the submodule somewhere else, point the scripts at it via
two environment variables:

```bash
export EPSS_TPG_DATA_REPO=/your/path/to/SummTPGVul/SummVul/Social_Media_Dataset/Data_Files
export EPSS_TPG_MEGAVUL_REPO=/your/path/to/SummTPGVul/SummVul/Data_Files/megavul
```

Drop them into `~/.bashrc` to make the override permanent.

## Training

### A single configuration

`python -m epss.run_pipeline` is the canonical entry point. It reads
the source CSV, builds the labelled records, constructs the TPG cache,
trains the multi-view GGNN, evaluates on the held-out test split, and
writes everything to the output directory you give it.

```bash
python -m epss.run_pipeline \
    --source-csv ../SummTPGVul/SummVul/Social_Media_Dataset/Data_Files/gpt_combined_summ.csv \
    --data-dir   data/epss_gpt_v2_ALL \
    --output-dir outputs/social_media/gpt/ALL \
    --backbone multiview --hybrid --label-mode soft --epochs 100 \
    --no-epss-feature --include-summary-in-tpg --summary-source combined
```

`--help` documents every flag. The ones you'll touch most are
`--backbone` (gcn, gat, multiview), `--hybrid` (turn on the tabular
branch with CVSS, CWE, age, and exploit-availability features),
`--label-mode` (binary KEV versus soft EPSS), `--no-epss-feature` (drop
EPSS from the tabular branch — required when training against the soft
EPSS target, otherwise you leak the label), and the three text-source
switches `--summary-only-tpg`, `--include-summary-in-tpg`, and
`--summary-source`.

### The batch scripts

Four pre-baked training batches live under `scripts/training/`. They
all share three conveniences: rerun with `--no-overwrite` to skip
completed runs, pass `--dry-run` to preview without executing, and
give a regex as a positional argument to subset the run list.

The 15-run social-media baseline — GPT, Gemma and Mistral crossed with
the five text-source variants `D`, `SMP`, `S_git`, `S_cvss`, `ALL`:

```bash
scripts/training/run_social_media_retrain.sh                  # all 15
scripts/training/run_social_media_retrain.sh gpt              # GPT block only
scripts/training/run_social_media_retrain.sh 'gemma|mistral'  # two LLMs
scripts/training/run_social_media_retrain.sh --threads 16     # cap OMP threads
```

The 5-run GPT NOSEC ablation that pairs with the social-media baseline
to produce the WITH-vs-NOSEC comparison:

```bash
scripts/training/run_gpt_nosec_retrain.sh
scripts/training/run_gpt_nosec_retrain.sh 'D|ALL'    # subset by variant
```

The full 33-run text-source ablation matrix covering both dataset
families:

```bash
scripts/training/run_all_summary_experiments.sh
scripts/training/run_all_summary_experiments.sh S_cvss   # all S_cvss variants
```

The 33-run security-frontend ablation, mirroring the matrix above with
`--no-security-frontend` on every run, plus three NVD/KEV reference
runs. Outputs land in `outputs/security_ablation/` rather than on top
of the WITH-sec baseline:

```bash
scripts/training/run_all_no_security_experiments.sh
scripts/training/run_all_no_security_experiments.sh nvd_kev   # NVD/KEV only
```

By default the batches stream their per-run output to stdout. Pass
`--quiet` to redirect to a log file and just print the batch banner.

## Inference and evaluation

There are two distinct things you might want to do with a trained
model: re-evaluate a saved checkpoint against its own test split
(test-only), or score CVEs that the model has never seen before
(fresh inference). Both have a Python entry point and, where it makes
sense, a shell wrapper that handles batches of checkpoints.

### Re-evaluating a saved checkpoint

For one run at a time:

```bash
python -m epss.test_only \
    --run-dir outputs/social_media/gpt/ALL \
    --batch-size 16 \
    --threshold 0.5
```

The script reads `experiment_config.json` from the run directory,
rebuilds the dataset the same way the training did, loads
`best_model.pt`, and writes a fresh `test_results.json` and
`predictions_test.csv`. Device is auto-detected; force it with
`--device cuda` or `--device cpu` if needed.

For batch re-evaluation across every saved checkpoint, there are two
wrappers under `scripts/inference/`. `test_all_datasets.sh` reruns
in place — it overwrites each run's `test_results.json` and
`predictions_test.csv`, then drops a per-family summary CSV:

```bash
scripts/inference/test_all_datasets.sh                # all four families
scripts/inference/test_all_datasets.sh social_media   # just one family
DEVICE=cuda BATCH_SIZE=16 scripts/inference/test_all_datasets.sh
```

`run_inference_on_all.sh` does the same evaluation but routes the
results into a fresh tree under
`inference_results/<family>/<llm>/<variant>/`, leaving the originals
alone:

```bash
scripts/inference/run_inference_on_all.sh
OUT_ROOT=/tmp/eval scripts/inference/run_inference_on_all.sh
```

### Scoring CVEs the model never saw

`inference/infer.py` is the user-facing CLI. It fetches CVE
descriptions from NVD, builds a TPG, and scores them with a saved
checkpoint. Five modes:

```bash
# Specific CVE IDs
python inference/infer.py \
    --checkpoint outputs/social_media/gpt/ALL/best_model.pt \
    --config     outputs/social_media/gpt/ALL/experiment_config.json \
    --cve-ids CVE-2024-1234 CVE-2024-5678

# File of CVE IDs (one per line)
python inference/infer.py --cve-file ids.txt --checkpoint ... --config ...

# Everything published in the last N days
python inference/infer.py --recent-days 30 --checkpoint ... --config ...

# A specific date range
python inference/infer.py --date-range 2024-01-01 2024-01-31 \
                          --checkpoint ... --config ...

# Temporal evaluation: train cutoff vs ground truth (KEV) today
python inference/infer.py --temporal-eval --train-cutoff 2024-01-01 \
                          --eval-days 30 --checkpoint ... --config ...
```

Each call writes a CSV sorted by exploit probability, with columns
`cve_id, prob, tier, binary_pred, cvss_score, published, in_kev,
description`.

If you want to compare predictions against the FIRST EPSS API
programmatically rather than producing a user-facing CSV, use the
package-internal scorer `python -m epss.infer` instead.

### Cross-distribution evaluation

To score a trained model against an entirely different labelled corpus
(one the model never saw during training):

```bash
python -m epss.cross_distribution_eval \
    --checkpoint outputs/social_media/gpt/ALL/best_model.pt \
    --config     outputs/social_media/gpt/ALL/experiment_config.json \
    --eval-data  data/epss_mv_mistral_ALL/labeled_cves.json
```

## Dataset profiling and analysis

Three profiling scripts live inside the `epss` package (they import
other `epss.*` modules, so they belong there) and write their outputs
into `datasets_info/`.

`per_llm_full_profile` is the headline one. It reads the labelled
records for every (LLM, variant) social-media dataset, reruns the
rule-only security pipeline over each one, and writes a JSON / CSV /
Markdown triple with mean graph size, mean SEC overlay density, and
the full per-entity and per-edge breakdowns:

```bash
python -m epss.per_llm_full_profile \
    --output-dir datasets_info/Per_LLM_profile_new \
    --workers 8
```

`per_llm_graph_dims` is the fast version when you only want node and
edge counts straight from the PyG cache:

```bash
python -m epss.per_llm_graph_dims --output-dir datasets_info/Per_LLM_profile
```

`security_edges_stats` tabulates SEC_* edge firing rates on a single
labelled corpus:

```bash
python -m epss.security_edges_stats \
    --labeled-cves data/epss_gpt_v2_ALL/labeled_cves.json \
    --variant ALL
```

Three more analysis scripts live under `analysis/`. `analyze_dataset.py`
profiles a source CSV (schema, missingness, EPSS distribution):

```bash
python analysis/analyze_dataset.py \
    --csv ../SummTPGVul/SummVul/Social_Media_Dataset/Data_Files/gpt_combined_summ.csv \
    --output-dir datasets_info/gpt_combined_summ
```

`verify_features.py` compares the labelled-record dtypes between two
pipelines (handy when you want to confirm the NVD pipeline and the
CSV-derived pipeline agree on every field):

```bash
python analysis/verify_features.py \
    --a data/epss/labeled_cves.json \
    --b data/epss_sec4ai/labeled_cves.json
```

`generate_visualizations.py` regenerates the full plot suite for an
existing checkpoint without retraining — useful when you've changed
the plot code and don't want to wait for a full retrain:

```bash
python analysis/generate_visualizations.py --run-dir outputs/social_media/gpt/ALL
```

And as a smoke test of the TPG and GNN APIs in isolation:

```bash
python examples/demo.py                # build a TPG from a single CVE
python examples/compare_frontends.py   # spaCy vs hybrid security frontend
python examples/experiment.py          # end-to-end mini experiment
```

## The TPG as a generic document platform

Beyond the EPSS experiments, the TPG backend now doubles as a general
document-intelligence system. Domains are declarative `DomainSpec` JSON
(security, medical, legal, financial, scientific ship built-in; new ones
need zero code), and `tpg_app/` wraps the whole thing into a deployable
application — universal ingestion (PDF, DOCX, HTML, Markdown, CSV, JSON,
URLs), a SQLite+FTS5 knowledge store with hybrid graph+BM25 retrieval,
entity/relation analytics, and a REST API with a web UI:

```bash
pip install -r tpg_app/requirements.txt
python -m tpg_app.cli --domain security ingest ./reports/
python -m tpg_app.cli query "What does CVE-2024-1234 affect?"
python -m tpg_app.cli serve     # web app at http://localhost:8742
```

Graphs persist as GraphSON and re-export to NetworkX, GraphML (Gephi,
yEd, Cytoscape) and Cypher (Neo4j) via `tpg.exporters`. Full guide:
`tpg_app/README.md`.

## Maintenance

Training batches generate large per-experiment PyG graph caches —
roughly 5 to 40 GB per social-media run. Once a batch finishes,
those caches can be freed without losing the saved models,
predictions, or labelled records:

```bash
scripts/analysis/cleanup_old_data.sh --dry-run    # preview what would go
scripts/analysis/cleanup_old_data.sh              # delete (asks first)
scripts/analysis/cleanup_old_data.sh --force      # skip the prompt
```

Only the regeneratable cache directories are removed — everything
under `outputs/` is left alone.

## Where things end up

Every training run drops a small set of artefacts into its output
directory, with consistent naming across families:

| File | What it is |
|---|---|
| `best_model.pt` | The checkpoint with the lowest validation loss |
| `experiment_config.json` | Every hyperparameter and flag the run used (replays cleanly with `test_only`) |
| `test_results.json` | PR-AUC, ROC-AUC, F1, precision, recall, Brier, threshold, sample counts |
| `val_results.json` | Same metrics on the validation split |

The inference batches under `scripts/inference/` drop their per-family
summaries at `outputs/<family>/test_only_summary.csv` and the
cross-family aggregate at `outputs/test_only_combined_summary.csv`.

The profiling scripts in `epss.per_llm_*` and `epss.security_edges_stats`
write their JSON / CSV / Markdown reports under `datasets_info/`.

## Reproducing the headline results

A handful of result blocks come up most often. Each one is produced
by a single batch script:

| Result block | How to reproduce |
|---|---|
| GPT WITH-vs-NOSEC ablation | `scripts/training/run_gpt_nosec_retrain.sh` (NOSEC half) plus the GPT runs in the social-media baseline |
| Per-LLM Megavul characterisation | `scripts/training/run_all_summary_experiments.sh mv_*` |
| 15-run social-media baseline | `scripts/training/run_social_media_retrain.sh 'gpt\|gemma\|mistral'` |
| Per-LLM graph and SEC overlay profile | `python -m epss.per_llm_full_profile --output-dir datasets_info/Per_LLM_profile_new` |

The headline numbers were collected with `--epochs 100`, `--backbone
multiview`, `--hybrid`, `--label-mode soft`, and `--no-epss-feature`.
Keep those constant and the metrics should reproduce within stochastic
noise.
