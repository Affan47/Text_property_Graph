# EPSS-TPG

This is the code for an experiment that asks a simple question: can a graph
neural network read a CVE's text description, combine it with the usual
vulnerability metadata, and predict whether the CVE will be exploited in the
wild?

The short answer is "yes, and the gap between the GNN and the standard EPSS
score is real". The longer answer lives in the methodology write-ups under
[docs/](docs/), but this README is purely about getting your hands on the code
and running it. It walks through what the project contains, where to put the
source data, and the exact command line for every script that ships in this
tree.

Every shell script in here resolves its own paths from `BASH_SOURCE`, so you
can run them from anywhere. Python is invoked as `python -m epss.X` or
`python <script_path>.py` and works the same way from any cwd as long as the
project root is on `PYTHONPATH` (which it is automatically if you run from the
root or call `python -m`).

## What you need

Python 3.10 or newer, PyTorch 2.x (CUDA build if you have a GPU), and
PyTorch Geometric 2.5+. On top of that, the TPG construction needs spaCy
with an English model (`python -m spacy download en_core_web_sm`) and the
Hugging Face Transformers library to load SecBERT (`jackaduma/SecBERT`).
Everything else (numpy, pandas, scikit-learn, tqdm) is standard.

Quick sanity check:

```bash
python -c "import torch; print('cuda available:', torch.cuda.is_available())"
```

If it prints `False`, the code will still run on CPU — just five to ten times
slower for training. Inference is fine on CPU.

## How the project is laid out

```
EPSS_TPG/
│
├── epss/            Main Python package. Training, dataset construction,
│                    the GNN itself, the inference entry points, and the
│                    profiling utilities live here. Everything is importable
│                    as epss.X or runnable as python -m epss.X.
│
├── tpg/             The Text Property Graph backend. Builds the per-CVE
│                    graph with a spaCy frontend and a hybrid SecBERT +
│                    rule-based security overlay, then exports it as a
│                    PyG tensor for the GNN to consume.
│
├── analysis/        Stand-alone analysis scripts (CSV profiling, feature
│                    verification, plot generation for an existing
│                    checkpoint). Run as python analysis/<name>.py.
│
├── inference/       Stand-alone CLI for scoring CVEs you fetch live from
│                    NVD, with five different modes (by ID, by file of IDs,
│                    by date range, by recent days, or temporal evaluation
│                    against ground truth).
│
├── scripts/         All shell-script batch entry points, grouped by
│                    purpose: training/ for end-to-end retrains,
│                    inference/ for test-only evaluations, and analysis/
│                    for maintenance tasks.
│
├── examples/        Small standalone examples that exercise the TPG and
│                    GNN APIs in isolation. Useful as a smoke test after
│                    a fresh install.
│
├── docs/            Markdown documentation: the technical report, the
│                    complete TPG guide, the security-frontend reference,
│                    plus per-topic write-ups under tpg_architecture/,
│                    epss_model/, experiments/, and domain_examples/.
│
├── data/            Source data and labelled records. The per-experiment
│                    pyg graph caches that the training scripts generate
│                    are not tracked (they're large and regeneratable).
│
├── outputs/         Where every training run drops its artefacts, grouped
│                    by dataset family: social_media/, megavul/, nvd_kev/,
│                    and security_ablation/. Each run directory holds the
│                    test_results.json metrics summary, the experiment
│                    config that produced it, and (for the security
│                    ablation) the full set of checkpoints and predictions.
│
├── inference_results/   Where the inference batches in scripts/inference/
│                        land their summaries.
│
├── datasets_info/   Dataset profiling artefacts: per-LLM profiles, ablation
│                    summary tables, and per-source CSV statistics.
│
├── SummTPGVul/      Git submodule holding the source CSVs for both
│                    dataset families. See "Source data" below.
│
└── README.md        You are here.
```

A few cross-references that the layout doesn't make obvious. The training
scripts in `scripts/training/` are thin wrappers around
`python -m epss.run_pipeline`, just with the right flags pre-filled for
each ablation. The two `inference/` directories (one for Python, one for
shell scripts under `scripts/inference/`) are intentional — the Python
one is the user-facing CLI for scoring new CVEs, the shell one is for
sweeping evaluation over your saved checkpoints. Don't confuse
`inference/infer.py` (CLI for fresh CVEs) with `epss/infer.py`
(package-internal temporal scorer called via `python -m epss.infer`).

## Source data

The training scripts expect to find the source CSVs in the `SummTPGVul`
git submodule, cloned one directory above the project root. After cloning
the parent repo, fetch the submodule contents:

```bash
git submodule update --init --recursive
```

If your clone uses Git LFS for the large CSVs (the submodule's
`.gitattributes` will tell you), pull those too:

```bash
sudo apt install -y git-lfs && git lfs install
cd ../SummTPGVul && git lfs pull && cd -
```

The layout the scripts expect under the submodule is:

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

If you keep the submodule somewhere else, point the scripts at it with
two environment variables:

```bash
export EPSS_TPG_DATA_REPO=/your/path/to/SummTPGVul/SummVul/Social_Media_Dataset/Data_Files
export EPSS_TPG_MEGAVUL_REPO=/your/path/to/SummTPGVul/SummVul/Data_Files/megavul
```

Add them to `~/.bashrc` to make the override permanent.

## Training

### One configuration end-to-end

`python -m epss.run_pipeline` is the canonical training entry point. It
reads the source CSV, builds the per-CVE labelled records, constructs the
TPG cache, trains the multi-view GGNN, evaluates on the held-out test
split, and writes everything to the output directory you point it at.

A minimal social-media ALL invocation:

```bash
python -m epss.run_pipeline \
    --source-csv ../SummTPGVul/SummVul/Social_Media_Dataset/Data_Files/gpt_combined_summ.csv \
    --data-dir   data/epss_gpt_v2_ALL \
    --output-dir outputs/social_media/gpt/ALL \
    --backbone multiview --hybrid --label-mode soft --epochs 100 \
    --no-epss-feature --include-summary-in-tpg --summary-source combined
```

`python -m epss.run_pipeline --help` documents every flag. The ones you'll
reach for most often are `--backbone` (gcn, gat, multiview), `--hybrid`
(turn on the tabular branch with CVSS, CWE, age and exploit-availability
features), `--label-mode` (binary KEV vs soft EPSS), `--no-epss-feature`
(removes EPSS from the tabular branch — required when training against
the soft EPSS target to prevent leakage), and the text-source switches
(`--summary-only-tpg`, `--include-summary-in-tpg`, `--summary-source`).

### The batch scripts

There are four pre-baked training batches under `scripts/training/`. Each
one is idempotent (rerun with `--no-overwrite` to skip completed runs),
supports `--dry-run` for previewing without executing, and accepts a
regex filter as a positional argument to subset the run list.

`run_social_media_retrain.sh` is the 15-run social-media baseline that the
paper reports on — GPT, Gemma and Mistral crossed with the five
text-source variants (`D`, `SMP`, `S_git`, `S_cvss`, `ALL`).

```bash
scripts/training/run_social_media_retrain.sh                  # all 15 runs
scripts/training/run_social_media_retrain.sh gpt              # GPT block only
scripts/training/run_social_media_retrain.sh 'gemma|mistral'  # two LLMs
scripts/training/run_social_media_retrain.sh --threads 16     # cap OMP threads
```

`run_gpt_nosec_retrain.sh` is the 5-run GPT NOSEC ablation that pairs
with the social-media baseline to produce the WITH-vs-NOSEC table in
the methodology document.

```bash
scripts/training/run_gpt_nosec_retrain.sh
scripts/training/run_gpt_nosec_retrain.sh 'D|ALL'    # subset by variant
```

`run_all_summary_experiments.sh` is the full 38-run ablation matrix
covering both dataset families (social-media + Megavul) and every LLM,
without the security-frontend ablation.

```bash
scripts/training/run_all_summary_experiments.sh
scripts/training/run_all_summary_experiments.sh S_cvss   # all S_cvss variants
```

`run_all_no_security_experiments.sh` mirrors that same 38-run matrix
with `--no-security-frontend` on every run, plus the three NVD/KEV
reference runs. Outputs land in `outputs/security_ablation/` rather than
on top of the WITH-sec baseline.

```bash
scripts/training/run_all_no_security_experiments.sh
scripts/training/run_all_no_security_experiments.sh nvd_kev   # just NVD/KEV
```

All four batches stream their progress to stdout by default. Pass
`--quiet` to send the per-run output to a log file and only print the
batch banner.

## Inference and evaluation

There's a clear split between *test-only evaluation* (load a saved
checkpoint, re-score the test split, write the metrics) and *fresh
inference* (score CVEs that the model has never seen, fetched live from
NVD). Both modes have a Python entry point and a shell wrapper.

### Re-evaluating a saved checkpoint

For a single run:

```bash
python -m epss.test_only \
    --run-dir outputs/social_media/gpt/ALL \
    --batch-size 16 \
    --threshold 0.5
```

The script reads `experiment_config.json` from the run directory,
rebuilds the dataset the same way the training did, loads `best_model.pt`,
and writes a fresh `test_results.json` and `predictions_test.csv`. Device
is auto-detected; pass `--device cuda` or `--device cpu` to force it.

To sweep every saved checkpoint at once, use one of the two batch
wrappers under `scripts/inference/`. `test_all_datasets.sh` reruns
test-only evaluation in place, overwriting each run's
`test_results.json` and `predictions_test.csv` and producing a
per-family summary CSV:

```bash
scripts/inference/test_all_datasets.sh                # all four families
scripts/inference/test_all_datasets.sh social_media   # just one family
DEVICE=cuda BATCH_SIZE=16 scripts/inference/test_all_datasets.sh
```

`run_inference_on_all.sh` does the same evaluation but routes the
results into a fresh tree under `inference_results/<family>/<llm>/<variant>/`
so the originals are left alone:

```bash
scripts/inference/run_inference_on_all.sh
OUT_ROOT=/tmp/eval scripts/inference/run_inference_on_all.sh
```

### Scoring CVEs the model never saw

`inference/infer.py` is the user-facing CLI. It fetches CVE descriptions
from NVD, builds a TPG, and scores them with a saved checkpoint. Five
modes:

```bash
# By CVE ID
python inference/infer.py \
    --checkpoint outputs/social_media/gpt/ALL/best_model.pt \
    --config     outputs/social_media/gpt/ALL/experiment_config.json \
    --cve-ids CVE-2024-1234 CVE-2024-5678

# By file of IDs
python inference/infer.py --cve-file ids.txt --checkpoint ... --config ...

# By recent days
python inference/infer.py --recent-days 30 --checkpoint ... --config ...

# By date range
python inference/infer.py --date-range 2024-01-01 2024-01-31 \
                         --checkpoint ... --config ...

# Temporal evaluation against ground truth
python inference/infer.py --temporal-eval --train-cutoff 2024-01-01 \
                         --eval-days 30 --checkpoint ... --config ...
```

Each call writes a CSV sorted by exploit probability with columns
`cve_id, prob, tier, binary_pred, cvss_score, published, in_kev,
description`.

The `python -m epss.infer` module is a complementary, package-internal
temporal scorer that additionally verifies predictions against the FIRST
EPSS API. Use it when you want a programmatic comparison rather than a
user-facing CSV.

### Cross-distribution evaluation

If you want to score a trained model against an entirely different
labelled corpus (one the model never saw at training time), use:

```bash
python -m epss.cross_distribution_eval \
    --checkpoint outputs/social_media/gpt/ALL/best_model.pt \
    --config     outputs/social_media/gpt/ALL/experiment_config.json \
    --eval-data  data/epss_mv_mistral_ALL/labeled_cves.json
```

## Dataset profiling and analysis

The methodology document leans on a few profile-generation scripts. They
all live under `epss/` (because they import other `epss.*` modules) and
write their outputs into `datasets_info/`.

The headline profiler is `per_llm_full_profile`, which reads the labelled
records for every (LLM, variant) social-media dataset, runs the rule-only
security pipeline over each, and emits a JSON / CSV / Markdown triple
with mean graph size, mean SEC overlay density, and the full entity- and
edge-type breakdowns.

```bash
python -m epss.per_llm_full_profile \
    --output-dir datasets_info/Per_LLM_profile_new \
    --workers 8
```

`per_llm_graph_dims` is the fast version — node and edge counts only,
read straight from the pyg cache:

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

Three stand-alone scripts live under `analysis/`. `analyze_dataset.py`
profiles a source CSV (schema, missingness, EPSS distribution):

```bash
python analysis/analyze_dataset.py \
    --csv ../SummTPGVul/SummVul/Social_Media_Dataset/Data_Files/gpt_combined_summ.csv \
    --output-dir datasets_info/gpt_combined_summ
```

`verify_features.py` compares the labelled-record dtypes between two
data-source pipelines (useful when you want to confirm the NVD pipeline
and the CSV-derived pipeline agree on every field):

```bash
python analysis/verify_features.py \
    --a data/epss/labeled_cves.json \
    --b data/epss_sec4ai/labeled_cves.json
```

`generate_visualizations.py` regenerates the full plot suite for an
existing checkpoint without retraining — handy when you've changed the
plot code and don't want to wait two hours for a rerun:

```bash
python analysis/generate_visualizations.py --run-dir outputs/social_media/gpt/ALL
```

For sanity checks of the TPG and GNN APIs in isolation, the
`examples/` directory has three small scripts:

```bash
python examples/demo.py                # build a TPG from a single CVE
python examples/compare_frontends.py   # spaCy vs hybrid security frontend
python examples/experiment.py          # end-to-end mini experiment
```

## Maintenance

The training batches generate large per-experiment pyg graph caches
(roughly 5–40 GB per social-media run). After a batch finishes, the
caches can be freed without losing the saved models, predictions, or
labelled records:

```bash
scripts/analysis/cleanup_old_data.sh --dry-run    # preview what would go
scripts/analysis/cleanup_old_data.sh              # delete (asks first)
scripts/analysis/cleanup_old_data.sh --force      # skip the prompt
```

Only the regeneratable pyg cache directories are removed — everything in
`outputs/` is left untouched.

## Where things end up

Each training run drops a small set of artefacts into its output
directory. The naming convention is consistent across families:

| File | What it is |
|---|---|
| `best_model.pt` | The trained checkpoint with the lowest validation loss |
| `experiment_config.json` | Every hyperparameter and flag the run used (replays cleanly with `test_only`) |
| `test_results.json` | PR-AUC, ROC-AUC, F1, precision, recall, Brier, threshold, sample counts |
| `val_results.json` | Same metrics on the validation split |

For the batch scripts under `scripts/inference/`, the per-family CSVs
land at `outputs/<family>/test_only_summary.csv` and the cross-family
aggregate at `outputs/test_only_combined_summary.csv`.

The profiling scripts in `epss/per_llm_*` and `epss/security_edges_stats`
write their JSON/CSV/Markdown reports under `datasets_info/`.

## Reproducing the paper

The methodology document under [docs/](docs/) refers to three result
blocks. Each one is produced by a single batch script:

| Result block | How to reproduce |
|---|---|
| GPT WITH-vs-NOSEC ablation | `scripts/training/run_gpt_nosec_retrain.sh` (NOSEC half) plus the GPT runs in the social-media baseline |
| Per-LLM Megavul characterisation | `scripts/training/run_all_summary_experiments.sh mv_*` |
| 15-run social-media baseline | `scripts/training/run_social_media_retrain.sh 'gpt\|gemma\|mistral'` |
| Per-LLM graph + SEC overlay profile | `python -m epss.per_llm_full_profile --output-dir datasets_info/Per_LLM_profile_new` |

The numbers reported in the methodology document were collected with
`--epochs 100`, `--backbone multiview`, `--hybrid`, `--label-mode soft`,
and `--no-epss-feature`. Hold those constant and the headline metrics
should reproduce within stochastic noise.
