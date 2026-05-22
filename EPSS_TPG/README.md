# EPSS-TPG: CVE Exploitation Prediction with Text Property Graphs

A graph neural network that predicts whether a CVE will be exploited
in the wild, from the CVE's text description plus structured
vulnerability metadata. Each CVE is converted into a Text Property
Graph (TPG) that captures syntactic, sequential, semantic, discourse,
and security relations; a multi-view GGNN reads that graph; an
optional tabular branch reads the CVSS, CWE, age and exploit
availability fields; a small head produces the predicted exploitation
probability.

This README covers the **practical use** of the code: the directory
layout, the dataset assumptions, and the exact command line for every
script in the repository (training, inference, dataset analysis).
The underlying methodology and the empirical evidence are written up
under [docs/papers/](docs/papers/) (LaTeX) and [docs/](docs/) (Markdown).

> Every command in this README is given relative to the project root
> and resolves its own paths from `${BASH_SOURCE[0]}`. You can invoke
> any shell script from any working directory.

---

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Repository layout](#2-repository-layout)
3. [Dataset assumptions and external repos](#3-dataset-assumptions-and-external-repos)
4. [Training scripts](#4-training-scripts)
5. [Inference-only scripts](#5-inference-only-scripts)
6. [Dataset analysis and statistics scripts](#6-dataset-analysis-and-statistics-scripts)
7. [Maintenance scripts](#7-maintenance-scripts)
8. [Where results and artefacts live](#8-where-results-and-artefacts-live)
9. [Reproducing the paper](#9-reproducing-the-paper)

---

## 1. Prerequisites

- Python 3.10 or newer
- PyTorch 2.x (CUDA build recommended; CPU works but is 5--10x slower)
- PyTorch Geometric 2.5+
- spaCy + an English model: `python -m spacy download en_core_web_sm`
- Hugging Face Transformers (for SecBERT, `jackaduma/SecBERT`)
- Standard scientific Python: numpy, pandas, scikit-learn, tqdm

Quick CUDA check:

```bash
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('torch:', torch.__version__)"
```

If `cuda available: False`, install a CUDA-enabled wheel from the
PyTorch website. Every script in this README falls back to CPU
automatically when CUDA is not available.

---

## 2. Repository layout

```
EPSS_TPG/
│
├── epss/                         # Main Python package (training + model + dataset)
│   ├── run_pipeline.py           # Train a single configuration end-to-end
│   ├── train.py                  # Trainer class (loss, eval, checkpointing)
│   ├── test_only.py              # Run test on a saved checkpoint (no train)
│   ├── infer.py                  # Score CVEs by ID, range, or recent days
│   ├── cve_dataset.py            # PyG dataset (one graph per CVE)
│   ├── gnn_model.py              # Model factory
│   ├── edge_aware_layers.py      # Multi-view GGNN encoder + attention fusion
│   ├── tabular_features.py       # CVSS/CWE/age/exploit feature builder
│   ├── csv_adapter.py            # Convert source CSVs to labelled records
│   ├── data_collector.py         # NVD + KEV + EPSS + ExploitDB downloader
│   ├── prepare_dataset.py        # Generic CSV → labelled-records adapter
│   ├── make_temporal_splits.py   # Create explicit temporal train/test splits
│   ├── visualize.py              # Save predictions CSVs and plots
│   ├── threshold_analysis.py     # Recompute metrics at multiple thresholds
│   ├── backfill_val_predictions.py  # Rebuild predictions_val.csv for old runs
│   ├── cross_distribution_eval.py   # Evaluate on a held-out labelled corpus
│   ├── per_llm_full_profile.py   # Per-LLM graph + security overlay profile
│   ├── per_llm_graph_dims.py     # Per-LLM mean/median graph size
│   └── security_edges_stats.py   # SEC_* edge firing-rate statistics
│
├── tpg/                          # Text Property Graph backend
│   ├── pipeline.py               # spaCy + SecBERT TPG construction
│   ├── schema/                   # Node and edge type schema
│   ├── frontends/                # spaCy / security / hybrid frontends
│   ├── passes/                   # Security pass + cross-modal pass
│   └── exporters/                # PyG tensor export
│
├── analysis/                     # Top-level dataset-analysis Python scripts
│   ├── analyze_dataset.py        # Full schema + statistical profile of a CSV
│   ├── generate_visualizations.py  # Re-generate plots for a saved checkpoint
│   └── verify_features.py        # Cross-check labelled-record dtypes
│
├── inference/                    # Top-level inference-only Python scripts
│   └── infer.py                  # CLI for scoring new / specific CVEs from NVD
│
├── scripts/                      # All batch (.sh) entry points
│   ├── training/                 # End-to-end training batches
│   │   ├── run_social_media_retrain.sh         # 19-run social-media matrix
│   │   ├── run_gpt_nosec_retrain.sh            # 5 GPT NOSEC runs
│   │   ├── run_all_summary_experiments.sh      # 38-run full ablation
│   │   └── run_all_no_security_experiments.sh  # 38-run security-ablation
│   ├── inference/                # Inference-only batches (no training)
│   │   ├── run_inference_on_all.sh             # Score every saved checkpoint
│   │   └── test_all_datasets.sh                # Test-only evaluation per family
│   └── analysis/                 # Maintenance / cleanup batches
│       └── cleanup_old_data.sh   # Free pyg-cache disk after runs finish
│
├── examples/                     # Stand-alone usage examples
│   ├── demo.py                   # Minimal TPG build + GNN forward demo
│   ├── compare_frontends.py      # spaCy-only vs hybrid security frontend
│   ├── experiment.py             # End-to-end experiment skeleton
│   └── TPG_sample_output.json    # Example graph dump
│
├── docs/                         # Documentation
│   ├── papers/                   # LaTeX papers + compiled PDFs
│   │   ├── security_frontend_rationale.tex  # Main methodology document
│   │   ├── tpg_chapter.tex                  # TPG chapter
│   │   ├── model_architecture.tex           # Model architecture chapter
│   │   ├── experiment_results_35_runs.tex   # Results chapter
│   │   ├── dataset_features.tex             # Dataset feature dictionary
│   │   └── *.pdf                            # Compiled outputs
│   ├── EPSS_GNN_Technical_Report.md
│   ├── TPG_COMPLETE_GUIDE.md
│   ├── Security_TPG_Complete_Reference.md
│   ├── tpg_architecture/                    # Architecture deep-dives
│   ├── chatbot/                             # Chatbot module docs
│   ├── domain_examples/                     # Worked examples
│   ├── epss_model/                          # EPSS model notes
│   └── experiments/                         # Experiment write-ups
│
├── data/                         # Source data + per-run pyg graph caches
│   ├── epss/                     # NVD/KEV labelled records and supporting JSONs
│   ├── epss_<llm>_v2_<variant>/  # Per-run datasets for the social-media ablation
│   ├── epss_mv_<llm>_<variant>/  # Per-run datasets for the Megavul ablation
│   └── (every dataset directory contains pyg_dataset/processed/cve_graphs_*.pt
│       — the multi-GB graph tensor cache; deleted after evaluation by the
│       disk-cleanup hook unless --keep-cache is passed)
│
├── outputs/                      # Trained checkpoints + per-run predictions + metrics
│   ├── social_media/             # Social-media corpus runs
│   │   ├── deepseek/{D, S_git, ALL}
│   │   ├── gemma/{D, SMP, S_git, S_cvss, ALL}
│   │   ├── gpt/{D, SMP, S_git, S_cvss, ALL}
│   │   ├── mistral/{D, SMP, S_git, S_cvss, ALL}
│   │   └── test_only_summary.csv
│   ├── megavul/                  # Megavul corpus runs
│   │   ├── gpt/{D, S_url, S_code, S_cvss, ALL}
│   │   ├── gemma/{D, S_url, S_code, S_cvss, ALL}
│   │   ├── mistral/{D, S_url, S_code, S_cvss, ALL}
│   │   └── test_only_summary.csv
│   ├── nvd_kev/                  # NVD/KEV reference runs
│   │   ├── multiview_hybrid_rerun/
│   │   ├── temporal_2020_2022_to_2023_2024/
│   │   ├── temporal_2020_to_2021_2026_noepss/
│   │   └── temporal_multiview_hybrid/
│   ├── security_ablation/        # NOSEC runs (security frontend disabled)
│   │   └── gpt_v2_{D,S_smp,S_git,S_cvss,ALL}_nosec/
│   ├── _legacy/                  # Older / exploratory runs (preserved, not deleted)
│   │   ├── cross_eval/           # Cross-distribution evaluation (~48 GB)
│   │   ├── no_security_ablation/ # Old 38-run security-ablation batch
│   │   ├── epss_{gcn,gat,sage,rgat,multiview,edge_type}_{hybrid,text}/   # Old arch sweep
│   │   ├── epss_full_*/          # Legacy full-NVD runs
│   │   ├── epss_sec4ai*/         # Early Sec4AI4Aec runs (pre-baseline)
│   │   └── (logs, old inference outputs, graphson exports, ...)
│   └── test_only_combined_summary.csv   # Cross-family aggregate summary
│
├── inference_results/            # Test outputs from scripts/inference/run_inference_on_all.sh
│   ├── social_media/<llm>/<variant>/   # test_results.json + predictions_test.csv
│   ├── megavul/<llm>/<variant>/        # same
│   ├── nvd_kev/<llm>/<variant>/        # same
│   ├── social_media_summary.csv
│   ├── megavul_summary.csv
│   └── combined_summary.csv
│
├── datasets_info/                # Dataset-information artefacts + ablation logs
│   ├── Per_LLM_profile/                 # Original per-LLM security profile
│   ├── Per_LLM_profile_new/             # Refreshed 15-run social-media profile
│   ├── Summary_in_TPG_ablation/         # Run logs + summary tables (.md, .json)
│   ├── Security_ablation/               # Security-frontend ablation logs
│   ├── CVSS_ablation/                   # CVSS-ablation logs
│   ├── TPG_ablation/                    # TPG-mode ablation logs
│   ├── gpt_combined_summ/               # Per-source profiling (GPT)
│   ├── gemma_combined_summ/             # Per-source profiling (Gemma)
│   ├── deepseek_combined_summ/          # Per-source profiling (DeepSeek)
│   └── OVERALL_ANALYSIS.md              # Cross-cut summary
│
├── logs/                         # Run logs from inference/test-only batches
├── tpg_chatbot/                  # Chatbot persistence
├── README.md                     # This file
└── .gitignore
```

### What each top-level folder is for

| Folder | Purpose |
|---|---|
| `epss/` | The main Python package. Importable as `epss.X`; runnable as `python -m epss.X`. Contains everything that trains the model, scores it, and analyses its outputs. |
| `tpg/` | The TPG core library. Builds the per-CVE graph and exports it as PyG tensors. `epss/` depends on this. |
| `analysis/` | Stand-alone dataset-analysis scripts that do not need to be in the package. Run as `python analysis/X.py`. |
| `inference/` | Stand-alone inference CLI. Distinct from `epss/infer.py` (which is the package-internal version called via `python -m epss.infer` for temporal evaluation). |
| `scripts/` | All `.sh` batch entry points, grouped by purpose: `training/`, `inference/`, `analysis/`. Every script resolves its own paths from `${BASH_SOURCE[0]}`. |
| `examples/` | Small stand-alone examples to verify the install and demonstrate the API. |
| `docs/` | All Markdown documentation. `docs/papers/` holds the LaTeX sources and compiled PDFs. |
| `data/` | Source data and per-run pyg graph caches. Heavy (~250 GB). Caches are auto-deleted by the disk-cleanup hook unless you pass `--keep-cache`. |
| `outputs/` | All training run outputs: `best_model.pt`, `test_results.json`, `predictions_*.csv`, `training_history.json`, `experiment_config.json`. |
| `inference_results/` | Per-family inference summaries written by `scripts/inference/run_inference_on_all.sh`. |
| `datasets_info/` | Dataset profiling artefacts: per-LLM profiles, ablation run logs, summary tables. |
| `logs/` | Top-level log files written by the inference/test-only batches. |

---

## 3. Dataset assumptions and external repos

The training scripts expect the source CSVs to live in a single
sibling repo (`SummTPGVul`) one directory above the project root:

```
../SummTPGVul/
└── SummVul/
    ├── Social_Media_Dataset/Data_Files/
    │   ├── gpt_combined_summ.csv
    │   ├── gemma_combined_summ.csv
    │   ├── mistral_combined_summ.csv
    │   └── deepseek_combined_summ.csv
    └── Data_Files/megavul/
        ├── gpt.csv
        ├── gemma.csv
        └── mistral.csv
```

Clone it as a sibling of `EPSS_TPG/`:

```bash
cd /home/ayounas/Text_property_Graph
git clone https://github.com/observatio/SummTPGVul.git
# if the CSVs are LFS-tracked:
sudo apt install -y git-lfs && git lfs install
cd SummTPGVul && git lfs pull
```

If your clone lives elsewhere, export these env vars before running
any training script:

```bash
export EPSS_TPG_DATA_REPO=/your/path/to/SummTPGVul/SummVul/Social_Media_Dataset/Data_Files
export EPSS_TPG_MEGAVUL_REPO=/your/path/to/SummTPGVul/SummVul/Data_Files/megavul
```

---

## 4. Training scripts

### 4.1 Single-run training (one configuration)

`epss/run_pipeline.py` is the canonical entry point — it builds the
labelled records, constructs the per-CVE graphs, trains the GGNN,
evaluates it on the held-out test split, and writes everything to
the output directory.

```bash
# minimum: source CSV in, output dir out
python -m epss.run_pipeline \
    --source-csv ../SummTPGVul/SummVul/Social_Media_Dataset/Data_Files/gpt_combined_summ.csv \
    --data-dir   data/epss_gpt_v2_ALL \
    --output-dir outputs/social_media/gpt/ALL \
    --backbone multiview --hybrid --label-mode soft --epochs 100 \
    --no-epss-feature --include-summary-in-tpg --summary-source combined
```

Most important flags (see `python -m epss.run_pipeline --help` for the full list):

| Flag | Effect |
|---|---|
| `--backbone {gcn,gat,multiview}` | GNN backbone. `multiview` (default in batches) uses 5 GGNN heads. |
| `--hybrid` | Add the tabular branch (CVSS, CWE, age, exploit). |
| `--label-mode {binary,soft}` | Binary KEV label or soft EPSS regression target. |
| `--epochs N` | Number of training epochs. |
| `--no-epss-feature` | Remove EPSS from the tabular branch (prevents leakage when target is soft EPSS). |
| `--no-security-frontend` | Disable the security pipeline entirely (NOSEC mode). |
| `--include-security-edges` | Emit the typed SEC_* edges (default off). |
| `--summary-only-tpg` | Build the TPG from a single summary column only (no description). |
| `--include-summary-in-tpg` | Concatenate description with summary columns. |
| `--summary-source {description,combined,all_sources,github_urls,cvss_metrics,social_media_post,commit_url,code}` | Which text source to feed to the TPG. |

### 4.2 Batch training: full ablation matrix

The full 38-run text-source ablation (social-media + Megavul, all four LLMs, all five variants each):

```bash
scripts/training/run_all_summary_experiments.sh

# common variants:
scripts/training/run_all_summary_experiments.sh gpt            # only the GPT runs
scripts/training/run_all_summary_experiments.sh S_cvss         # only the S_cvss variants
scripts/training/run_all_summary_experiments.sh --dry-run      # preview without executing
scripts/training/run_all_summary_experiments.sh --no-overwrite # skip runs already done
scripts/training/run_all_summary_experiments.sh --quiet        # suppress per-run streaming
```

### 4.3 Batch training: social-media 19-run baseline

The refreshed 19-run social-media batch (GPT/Gemma/Mistral × {D, SMP, S_git, S_cvss, ALL} plus 4 DeepSeek runs, disk-cleaning after each run):

```bash
scripts/training/run_social_media_retrain.sh                  # all 19 runs
scripts/training/run_social_media_retrain.sh gpt              # 5 GPT runs only
scripts/training/run_social_media_retrain.sh 'gemma|mistral'  # 10 runs (Gemma + Mistral)
scripts/training/run_social_media_retrain.sh --keep-cache     # keep pyg caches (~40 GB per run)
scripts/training/run_social_media_retrain.sh --threads 16     # OMP/MKL thread count (default 32)
```

### 4.4 Batch training: GPT NOSEC ablation (5 runs)

Mirrors the 5 GPT social-media variants with `--no-security-frontend` so the WITH-vs-NOSEC comparison can be refreshed:

```bash
scripts/training/run_gpt_nosec_retrain.sh                # all 5 NOSEC runs
scripts/training/run_gpt_nosec_retrain.sh --dry-run      # preview
scripts/training/run_gpt_nosec_retrain.sh 'D|ALL'        # filter to D and ALL
scripts/training/run_gpt_nosec_retrain.sh --no-overwrite # skip if test_results.json already exists
```

### 4.5 Batch training: full security ablation (38 runs)

The 38-run security ablation (every run with `--no-security-frontend`), kept in a separate output tree so it does not collide with the WITH-sec batch:

```bash
scripts/training/run_all_no_security_experiments.sh           # all 38 NOSEC runs
scripts/training/run_all_no_security_experiments.sh gpt       # only GPT
scripts/training/run_all_no_security_experiments.sh nvd_kev   # only the NVD/KEV reference runs
scripts/training/run_all_no_security_experiments.sh --dry-run # preview
```

NOSEC outputs go to `outputs/security_ablation/{social_media,megavul,nvd_kev}/...`.

---

## 5. Inference-only scripts

These never train. They load a saved checkpoint, score either the
original test split or fresh CVEs, and write metrics / predictions.

### 5.1 Test-only evaluation on a single saved checkpoint

```bash
python -m epss.test_only \
    --run-dir outputs/social_media/gpt/ALL          # required: dir with best_model.pt + experiment_config.json
    --batch-size 16                             # optional override
    --threshold 0.5                             # decision threshold for F1 / precision / recall
    --device cuda                               # auto-detect by default
```

Writes `test_results.json` and `predictions_test.csv` next to the checkpoint.

### 5.2 Batch test-only across every saved checkpoint

```bash
scripts/inference/test_all_datasets.sh                # all three families (social, megavul, nvd_kev)
scripts/inference/test_all_datasets.sh social_media   # one family only
scripts/inference/test_all_datasets.sh megavul nvd_kev  # multiple families

# env overrides:
DEVICE=cuda BATCH_SIZE=16 THRESHOLD=0.5 scripts/inference/test_all_datasets.sh
```

Each run's `test_results.json` and `predictions_test.csv` are rewritten in place;
per-family `outputs/test_only_<family>_summary.csv` summaries are produced.

### 5.3 Test all checkpoints and route results into a fresh tree

```bash
scripts/inference/run_inference_on_all.sh                  # all families → inference_results/
scripts/inference/run_inference_on_all.sh social_media     # just one family
OUT_ROOT=/tmp/my_inf  scripts/inference/run_inference_on_all.sh  # override output root
```

Produces `inference_results/<family>/<llm>/<variant>/test_results.json` plus
per-family and combined summary CSVs.

### 5.4 Score arbitrary / unseen CVEs (NVD-fetched)

The user-facing inference CLI in `inference/infer.py`:

```bash
# 5.4.1 Specific CVE IDs
python inference/infer.py \
    --checkpoint outputs/social_media/gpt/ALL/best_model.pt \
    --config     outputs/social_media/gpt/ALL/experiment_config.json \
    --cve-ids CVE-2024-1234 CVE-2024-5678

# 5.4.2 File of CVE IDs (one per line)
python inference/infer.py \
    --checkpoint outputs/social_media/gpt/ALL/best_model.pt \
    --config     outputs/social_media/gpt/ALL/experiment_config.json \
    --cve-file ids.txt

# 5.4.3 Recent N days (fetched live from NVD)
python inference/infer.py \
    --checkpoint outputs/social_media/gpt/ALL/best_model.pt \
    --config     outputs/social_media/gpt/ALL/experiment_config.json \
    --recent-days 30

# 5.4.4 Date range
python inference/infer.py \
    --checkpoint outputs/social_media/gpt/ALL/best_model.pt \
    --config     outputs/social_media/gpt/ALL/experiment_config.json \
    --date-range 2024-01-01 2024-01-31

# 5.4.5 Temporal evaluation (train cutoff vs. KEV ground truth today)
python inference/infer.py \
    --checkpoint outputs/social_media/gpt/ALL/best_model.pt \
    --config     outputs/social_media/gpt/ALL/experiment_config.json \
    --temporal-eval --train-cutoff 2024-01-01 --eval-days 30
```

Each invocation writes a CSV sorted by exploit probability (descending),
with `cve_id, prob, tier, binary_pred, cvss_score, published, in_kev, description`.

### 5.5 Temporal inference with ground-truth verification (package version)

`epss/infer.py` is a complementary, package-internal temporal scorer
that verifies predictions against KEV + the FIRST EPSS API:

```bash
python -m epss.infer --mode post-dataset \
    --after-date 2025-07-01 --before-date 2025-09-30 \
    --checkpoint outputs/_legacy/epss_sec4ai/best_model.pt \
    --config     outputs/_legacy/epss_sec4ai/experiment_config.json

python -m epss.infer --mode pre-dataset \
    --before-date 2021-11-01 \
    --checkpoint outputs/_legacy/epss_sec4ai/best_model.pt
```

### 5.6 Cross-distribution evaluation

Score a trained model against a held-out labelled corpus (one the model never saw):

```bash
python -m epss.cross_distribution_eval \
    --checkpoint outputs/social_media/gpt/ALL/best_model.pt \
    --config     outputs/social_media/gpt/ALL/experiment_config.json \
    --eval-data  data/epss_mv_mistral_ALL/labeled_cves.json
```

### 5.7 Threshold sweep (post-processing only, no inference)

```bash
python -m epss.threshold_analysis \
    --runs-root outputs \
    --output    outputs/threshold_analysis
```

Reads every `predictions_test.csv` it finds, recomputes F1 / precision /
recall / accuracy at multiple decision thresholds, and writes
`outputs/threshold_analysis/REPORT.md` plus per-run threshold curves.

### 5.8 Backfill `predictions_val.csv` for older runs

For runs that finished before the `visualize.py` patches were in
place, regenerate the validation predictions without retraining:

```bash
python -m epss.backfill_val_predictions --root outputs
```

---

## 6. Dataset analysis and statistics scripts

### 6.1 Full per-LLM graph + security profile (15-run social-media)

Builds the complete per-(LLM, variant) profile used in the LaTeX
methodology document — mean graph size, mean SEC_* edge count, every
entity-type and edge-type breakdown:

```bash
python -m epss.per_llm_full_profile \
    --output-dir datasets_info/Per_LLM_profile_new \
    --workers 8       # multiprocess pool (default: half the available cores)
```

Outputs `per_llm_full_profile.{json,csv,md}` plus per-dataset raw JSONs in
`datasets_info/Per_LLM_profile_new/raw/`.

### 6.2 Per-LLM graph dimensions only (fast)

If you only need mean/median nodes and edges per graph (no security
pipeline re-run, just a read of the pyg cache):

```bash
python -m epss.per_llm_graph_dims --output-dir datasets_info/Per_LLM_profile
```

### 6.3 SEC_* edge firing statistics on a single corpus

```bash
python -m epss.security_edges_stats \
    --labeled-cves data/epss_gpt_v2_ALL/labeled_cves.json \
    --variant      ALL
```

Reports per-edge firing rates and per-entity counts for one labelled corpus.

### 6.4 Profile a source CSV (schema, missingness, EPSS distribution)

```bash
python analysis/analyze_dataset.py \
    --csv ../SummTPGVul/SummVul/Social_Media_Dataset/Data_Files/gpt_combined_summ.csv \
    --output-dir datasets_info/gpt_combined_summ
```

### 6.5 Verify labelled-record dtypes across two source pipelines

```bash
python analysis/verify_features.py \
    --a data/epss/labeled_cves.json \
    --b data/epss_sec4ai/labeled_cves.json
```

Useful when comparing the NVD pipeline against the Sec4AI4Aec CSV-derived
pipeline to confirm dtypes and value distributions match.

### 6.6 Regenerate plots for an existing checkpoint

```bash
python analysis/generate_visualizations.py \
    --run-dir outputs/social_media/gpt/ALL
```

Detects the trained `tabular_dim` from the saved weights, slices the
dataset to match, and writes the full plot suite (confusion matrix,
PR curve, ROC curve, calibration, prediction histogram) into the run dir.

### 6.7 Standalone TPG examples

```bash
python examples/demo.py                # build a TPG from a single CVE
python examples/compare_frontends.py   # spaCy-only vs hybrid security frontend
python examples/experiment.py          # end-to-end mini experiment
```

---

## 7. Maintenance scripts

### 7.1 Free per-run pyg graph caches (recover disk)

After a training batch finishes, the per-run pyg caches (5--40 GB each
for the dense social-media datasets) can be freed without losing the
saved model, predictions, or labelled records:

```bash
scripts/analysis/cleanup_old_data.sh --dry-run   # preview what would be deleted (run this first)
scripts/analysis/cleanup_old_data.sh             # actually delete (asks for confirmation)
scripts/analysis/cleanup_old_data.sh --force     # skip confirmation
```

Only `pyg_dataset/processed/cve_graphs_*.pt` and `pyg_dataset/raw/` are
removed. Everything in `outputs/` is preserved.

---

## 8. Where results and artefacts live

| Artefact | Path |
|---|---|
| Trained model checkpoint | `outputs/<family>/<llm>/<variant>/best_model.pt` |
| Per-CVE test predictions | `outputs/<family>/<llm>/<variant>/predictions_test.csv` |
| Per-CVE val predictions | `outputs/<family>/<llm>/<variant>/predictions_val.csv` |
| Headline metrics (PR, ROC, F1, Brier, ...) | `outputs/<family>/<llm>/<variant>/test_results.json` |
| Per-epoch training curve | `outputs/<family>/<llm>/<variant>/training_history.json` |
| Hyperparameters / config | `outputs/<family>/<llm>/<variant>/experiment_config.json` |
| Generated plots | `outputs/<family>/<llm>/<variant>/*.png` |
| Per-family inference summaries | `inference_results/<family>_summary.csv` |
| Combined inference summary | `inference_results/combined_summary.csv` |
| Per-LLM dataset profile | `datasets_info/Per_LLM_profile_new/per_llm_full_profile.{json,csv,md}` |
| Batch run logs | `datasets_info/<ablation>/run_logs/<run_id>.log` |
| Threshold-sweep report | `outputs/threshold_analysis/REPORT.md` |

---

## 9. Reproducing the paper

The methodology document `docs/papers/security_frontend_rationale.tex`
references three result blocks. To reproduce each:

| Section | Batch |
|---|---|
| §6 GPT WITH-vs-NOSEC ablation | `scripts/training/run_gpt_nosec_retrain.sh` (NOSEC) + WITH-sec runs from §8 |
| §7 Per-LLM characterisation of Megavul | `scripts/training/run_all_summary_experiments.sh` filter `mv_*` |
| §8 Social-media 15-run baseline | `scripts/training/run_social_media_retrain.sh 'gpt\|gemma\|mistral'` |
| §7 + §8 graph-characterisation tables | `python -m epss.per_llm_full_profile --output-dir datasets_info/Per_LLM_profile_new` |

Compile the LaTeX with any TeX Live installation:

```bash
cd docs/papers
latexmk -pdf -interaction=nonstopmode security_frontend_rationale.tex
```
