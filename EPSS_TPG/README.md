# EPSS-TPG: CVE Exploitation Prediction with Text Property Graphs

A graph neural network that predicts whether a CVE will be exploited
in the wild, from the CVE's text description plus structured
vulnerability metadata. Each CVE is converted into a Text Property
Graph (TPG) that captures syntactic, sequential, semantic, discourse,
and security relations; a multi-view GNN reads that graph; an
optional tabular branch reads the CVSS, CWE, age and exploit
availability fields; a small head produces the predicted exploitation
probability.

This README covers the **practical use** of the code: how to train,
how to test a saved model, and how to score fresh CVEs. The
underlying methodology and full results are written up in the LaTeX
documents under `EPSS_Latex/`.

> **All paths in this README are relative to the project root**
> (`/home/ayounas/Text_property_Graph/EPSS_TPG`). Run every command
> from there.

---

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Repository layout](#2-repository-layout)
3. [Datasets](#3-datasets)
4. [Training](#4-training)
5. [Testing a trained model (no retraining)](#5-testing-a-trained-model-no-retraining)
6. [Inference on unseen CVEs](#6-inference-on-unseen-cves)
7. [Threshold analysis and reporting](#7-threshold-analysis-and-reporting)
8. [Where the results live](#8-where-the-results-live)
9. [Reproducing the paper](#9-reproducing-the-paper)

---

## 1. Prerequisites

- Python 3.10 or newer
- PyTorch 2.x (CUDA build recommended; CPU works but is 5-10x slower)
- PyTorch Geometric 2.5+
- spaCy + an English model (`python -m spacy download en_core_web_sm`)
- Hugging Face Transformers (for SecBERT, `jackaduma/SecBERT`)
- Standard scientific Python: numpy, pandas, scikit-learn, tqdm

Quick CUDA check:

```bash
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('torch version:', torch.__version__)"
```

If `cuda available: False`, install a CUDA-enabled wheel from the
PyTorch website. Everything in this README falls back to CPU
automatically when CUDA is not available.

---

## 2. Repository layout

```
EPSS_TPG/
├── epss/                          Main Python package
│   ├── run_pipeline.py            Train a single configuration end-to-end
│   ├── train.py                   Trainer class (loss, eval, checkpoints)
│   ├── test_only.py               Run test on a saved checkpoint (no train)
│   ├── infer.py                   Score arbitrary CVEs (fetches from NVD)
│   ├── threshold_analysis.py      Post-process predictions at multiple thresholds
│   ├── backfill_val_predictions.py  Rebuild predictions_val.csv for old runs
│   ├── csv_adapter.py             Convert source CSVs to labelled records
│   ├── cve_dataset.py             PyG dataset (one graph per CVE)
│   ├── data_collector.py          NVD + KEV + EPSS + ExploitDB downloader
│   ├── gnn_model.py               Model factory
│   ├── edge_aware_layers.py       Multi-view GGNN encoder + attention fusion
│   ├── tabular_features.py        CVSS/CWE/age/exploit feature builder
│   └── visualize.py               Saves predictions CSVs and plots
│
├── tpg/                           Text Property Graph backend
│   ├── pipeline.py                spaCy + SecBERT TPG construction
│   ├── schema/                    Node and edge type schema
│   ├── passes/                    Security pass + cross-modal pass
│   └── exporters/                 PyG tensor export
│
├── data/                          Source datasets and labelled records
│   ├── epss/                      NVD/KEV labelled records and supporting JSONs
│   ├── epss_<llm>_v2_<variant>/   Per-run datasets for the social-media ablation
│   └── epss_mv_<llm>_<variant>/   Per-run datasets for the Megavul ablation
│
├── output/                        Trained models, predictions, and metrics
│   ├── epss_<llm>_v2_<variant>/   One directory per social-media run
│   ├── epss_mv_<llm>_<variant>/   One directory per Megavul run
│   ├── epss_nvd_kev_*/            NVD/KEV binary classification runs
│   ├── epss_temporal_*/           NVD/KEV temporal-shift runs
│   ├── inference/                 Inference outputs from epss/infer.py
│   ├── threshold_analysis/        Aggregated reports (REPORT.md, per-run CSV)
│   └── test_only_*.csv            Per-family summaries (legacy wrapper)
│
├── Inference_results/             Test outputs from scripts/run_inference_on_all.sh
│   ├── social_media/<llm>/<variant>/   test_results.json + predictions_test.csv
│   ├── megavul/<llm>/<variant>/        same
│   ├── nvd_kev/<llm>/<variant>/        same
│   ├── social_media_summary.csv
│   ├── megavul_summary.csv
│   ├── nvd_kev_summary.csv
│   └── combined_summary.csv
│
├── scripts/
│   ├── run_inference_on_all.sh    Wrapper that organises test outputs by family
│   └── test_all_datasets.sh       Older wrapper that overwrites in-run outputs
│
├── Datasets_information/
│   └── Summary_in_TPG_ablation/
│       └── run_all_summary_experiments.sh   The 35-run training launcher
│
├── EPSS_Latex/                    Paper sources (multiview formula, dataset
│                                  features, experiment results)
│
├── docs/                          Background documentation
└── logs/                          Training and test logs
```

For each completed training run, `output/<run-name>/` contains:

| File | Purpose |
|---|---|
| `best_model.pt` | The best checkpoint by validation PR-AUC |
| `experiment_config.json` | All training arguments (used by test_only) |
| `test_results.json` | Test-set metrics at threshold 0.5 |
| `predictions_test.csv` | Per-CVE test predictions, ranked by probability |
| `predictions_val.csv` | Per-CVE validation predictions (for threshold tuning) |
| `val_results.json` | Validation-set metrics |
| `training_history.json` | Per-epoch train/val loss and PR-AUC |

---

## 3. Datasets

The project trains on three dataset families:

### 3.1 Social-media (Sec4AI4Aec-EPSS-Enhanced)

Four variants, one per LLM that generated the summaries: GPT, Gemma,
Mistral, DeepSeek. Each dataset is one row per CVE (9,218 CVEs) with
the NVD description, CVSS metadata, EPSS score, social-source counts,
and three LLM-generated summary columns.

Source data (cloned externally, not in this repo): the
`Sec4AI4Aec-EPSS-Enhanced` repository main branch.

### 3.2 Megavul

Three variants: Mega-GPT, Mega-Mistral, Mega-Gemma. Each dataset is
9,486 rows over 9,240 unique CVEs. Rows are commit-grounded: each row
carries a commit URL, the pre-patch and post-patch function code, and
LLM summaries of the commit URL, the vulnerable code, and the CVSS
metadata.

Source data: the `megavul-based-with-summarizations` branch of the
same upstream repository.

### 3.3 NVD/KEV

The full NVD-derived corpus, labelled by CISA KEV membership. Used
for the binary classification rerun and the temporal-shift
evaluations. Sources: NVD API, CISA KEV catalog, FIRST EPSS API,
ExploitDB.

The local NVD/KEV labelled records live at
`data/epss/labeled_cves.json` (full file, 132,322 CVEs) and the two
balanced files (`labeled_cves_balanced.json` and
`labeled_cves_balanced_v2.json`, both 4,015 CVEs).

> **History.** The original `output/epss_nvd_kev_*` and
> `output/epss_temporal_*` checkpoints were deleted on 2026-05-12 to
> free disk space, and the original metrics are archived in
> `output/threshold_analysis/nvd_kev_archived_results.json`. Section
> 4.4 below has the retraining commands; running them rebuilds the
> three NVD/KEV experiments end-to-end.

---

## 4. Training

Every training run goes through `epss/run_pipeline.py`. The same
script handles all three dataset families; the differences are flags
and the source data path.

### 4.1 Train one social-media run (description-only baseline on GPT)

```bash
python -m epss.run_pipeline \
    --source-csv path/to/Sec4AI4Aec-EPSS-Enhanced/Data_Files/gpt_combined_summ.csv \
    --data-dir   data/epss_gpt_v2_D \
    --output-dir output/epss_gpt_v2_D \
    --backbone multiview --hybrid \
    --label-mode soft --epochs 100 \
    --no-epss-feature \
    --summary-source description
```

Variants for the same LLM:

| Variant | Extra flags |
|---|---|
| `D` (description) | `--summary-source description` |
| `S_all` | `--summary-only-tpg --summary-source all_sources` |
| `S_git` | `--summary-only-tpg --summary-source github_urls` |
| `S_cvss` | `--summary-only-tpg --summary-source cvss_metrics` |
| `ALL` | `--include-summary-in-tpg --summary-source combined` |

### 4.2 Train one Megavul run (description-only baseline on Mega-GPT)

```bash
python -m epss.run_pipeline \
    --source-csv path/to/megavul/gpt.csv \
    --data-dir   data/epss_mv_gpt_D \
    --output-dir output/epss_mv_gpt_D \
    --backbone multiview --hybrid \
    --label-mode soft --epochs 100 \
    --no-epss-feature \
    --summary-source description
```

Megavul variants use a different summary column map:

| Variant | Extra flags |
|---|---|
| `D` | `--summary-source description` |
| `S_url` | `--summary-only-tpg --summary-source commit_url` |
| `S_code` | `--summary-only-tpg --summary-source code` |
| `S_cvss` | `--summary-only-tpg --summary-source cvss_metrics` |
| `ALL` | `--include-summary-in-tpg --summary-source combined` |

### 4.3 Train all 35 ablation runs at once

```bash
bash Datasets_information/Summary_in_TPG_ablation/run_all_summary_experiments.sh
```

Filter to a subset with the `FILTER` environment variable (regex
matched against the run name):

```bash
# only the megavul gemma runs
FILTER='^epss_mv_gemma_' \
    bash Datasets_information/Summary_in_TPG_ablation/run_all_summary_experiments.sh

# dry-run, print commands without executing
DRY_RUN=true \
    bash Datasets_information/Summary_in_TPG_ablation/run_all_summary_experiments.sh
```

### 4.4 Train the NVD/KEV runs

There are three NVD/KEV experiments. The first is a balanced
binary-classification rerun, the second and third are temporal-shift
evaluations where the model is trained on earlier years and tested on
later ones. All three take a few hours combined on a single GPU.

The temporal runs use a separate held-out test set produced by
`epss/make_temporal_splits.py`. That script takes one source labelled
file and writes three artefacts into `--output-dir`:

```
labeled_cves_temporal_train.json     records with publication date <= --train-end-date
labeled_cves_temporal_test.json      records with publication date in [--test-start-date, --test-end-date]
temporal_split_manifest.json         provenance: dates, per-year counts, seed
```

It refuses to write anything if either the train or the test split is
empty, so do not try to call it once for "train only" and once for
"test only". Run it once with both windows.

**4.4.1 NVD/KEV binary rerun (target = KEV membership, EPSS kept as input)**

```bash
python -m epss.run_pipeline \
    --skip-collect \
    --labeled-file data/epss/labeled_cves_balanced_v2.json \
    --data-dir     data/epss_nvd_kev_multiview_hybrid_rerun \
    --output-dir   output/epss_nvd_kev_multiview_hybrid_rerun \
    --backbone multiview --hybrid \
    --label-mode binary --epochs 100
```

This trains on the older 4,015-CVE balanced file (20% positive
prevalence). A random 70/15/15 split is used for train/val/test,
since no external test set is supplied.

**4.4.2 NVD/KEV temporal shift (train 2020-2022, test 2023-2024)**

The negative caps below land on the paper's reported sizes (7,487
train at 6.5% positive, 3,316 test at 9.5% positive). Without them
the script keeps every negative, giving roughly 63k train and 68k
test records at sub-1% positive, which is not what the paper used.

```bash
# 1. build the temporal splits in one call
python -m epss.make_temporal_splits \
    --source              data/epss/labeled_cves.json \
    --output-dir          data/epss_temporal_2020_2022_train \
    --train-end-date      2022-12-31 \
    --test-start-date     2023-01-01 \
    --test-end-date       2024-12-31 \
    --max-train-negatives 7000 \
    --max-test-negatives  3000

# 2. train + evaluate on the held-out window
python -m epss.run_pipeline \
    --skip-collect \
    --labeled-file       data/epss_temporal_2020_2022_train/labeled_cves_temporal_train.json \
    --test-labeled-file  data/epss_temporal_2020_2022_train/labeled_cves_temporal_test.json \
    --data-dir           data/epss_temporal_2020_2022_train \
    --output-dir         output/epss_temporal_2020_2022_to_2023_2024 \
    --backbone multiview --hybrid \
    --label-mode binary --epochs 100
```

For the EPSS-removed variant of the same window, add
`--no-epss-feature` to step 2 and change the output directory to
`output/epss_temporal_2020_2022_to_2023_2024_noepss`.

**4.4.3 NVD/KEV no-EPSS temporal extrapolation (train 2020, test 2021-2026)**

The local labelled file already starts in 2020, so a single call to
the temporal-split script gives both halves in one go. The negative
caps below match the paper's 1,500 train / 9,303 test sizes;
without them you get roughly 18k train and 114k test.

```bash
# 1. build the splits in one call
python -m epss.make_temporal_splits \
    --source              data/epss/labeled_cves.json \
    --output-dir          data/epss_temporal_2020_train_noepss \
    --train-end-date      2020-12-31 \
    --test-start-date     2021-01-01 \
    --test-end-date       2026-12-31 \
    --max-train-negatives 1355 \
    --max-test-negatives  8518

# 2. train without EPSS in the tabular branch
python -m epss.run_pipeline \
    --skip-collect --no-epss-feature \
    --labeled-file       data/epss_temporal_2020_train_noepss/labeled_cves_temporal_train.json \
    --test-labeled-file  data/epss_temporal_2020_train_noepss/labeled_cves_temporal_test.json \
    --data-dir           data/epss_temporal_2020_train_noepss \
    --output-dir         output/epss_temporal_2020_to_2021_2026_noepss \
    --backbone multiview --hybrid \
    --label-mode binary --epochs 100
```

The split script does not have a `--train-start-date`. If your source
file extends earlier than 2020 and you want training restricted to a
single year, pre-filter the source first:

```bash
python3 - <<'PY'
import json
from datetime import datetime
with open("data/epss/labeled_cves.json") as f:
    records = json.load(f)
items = list(records.items()) if isinstance(records, dict) \
        else [(r.get("cve_id", str(i)), r) for i, r in enumerate(records)]
def year_of(rec):
    for k in ("published", "datePublished", "date"):
        v = rec.get(k)
        if not v: continue
        try:
            return datetime.fromisoformat(str(v).replace("Z", "+00:00")).year
        except Exception:
            continue
    return None
filt = {cid: rec for cid, rec in items if (year_of(rec) or 0) >= 2020}
with open("data/epss/labeled_cves_2020_onwards.json", "w") as f:
    json.dump(filt, f, indent=2)
print(f"Wrote {len(filt)} records.")
PY
```

Then point `--source` at the filtered file. Do not try to invoke
`make_temporal_splits` once for the train half and once for the test
half; the script refuses to write any output when either half is
empty, so a "test-only" or "train-only" call exits with no files
written.

### 4.5 Useful training flags

| Flag | What it does |
|---|---|
| `--label-mode soft` | Train against the raw EPSS score in $[0, 1]$ |
| `--label-mode binary` | Train against KEV membership (0 / 1) |
| `--no-epss-feature` | Drop EPSS score and percentile from the tabular branch (required when EPSS is the target, to avoid leakage) |
| `--no-hybrid` | Graph-only branch, no tabular MLP |
| `--include-security-edges` | Add the 10 SEC_* edge types and the security view |
| `--two-view-tpg` | Pool description and summary nodes separately |
| `--add-source-labels` | Append a 3-dim source one-hot to node features |
| `--seed N` | Reproducibility seed (default 42) |
| `--device cuda` / `--device cpu` | Force device |
| `--epochs N`, `--batch-size N`, `--lr X` | Training schedule overrides |

Full list: `python -m epss.run_pipeline --help`.

---

## 5. Testing a trained model (no retraining)

To validate a trained model without retraining, use `epss/test_only.py`.
It loads `best_model.pt` together with `experiment_config.json`,
rebuilds the dataset the same way training did, runs the test forward
pass, and writes the metrics JSON and per-CVE predictions CSV. It does
not train, does not fetch anything from NVD, and does not modify any
source data.

The labelled records are picked up automatically. If the run trained
with `--labeled-file` (the NVD/KEV binary rerun, the temporal
experiments) test_only follows that path. Otherwise it falls back to
`<data-dir>/labeled_cves.json`. If the run also trained with
`--test-labeled-file` (the temporal experiments), test_only uses that
held-out file as the test set, exactly as the training run did.

### 5.1 Run inference on every saved checkpoint, organised by dataset

This is the recommended entry point. One command, organised output:

```bash
bash scripts/run_inference_on_all.sh                   # all families
bash scripts/run_inference_on_all.sh megavul           # one family
bash scripts/run_inference_on_all.sh social_media nvd_kev   # multiple
```

The wrapper walks every `output/<run>/best_model.pt`, parses the run
name into `(family, llm, variant)`, and routes results into a fresh
tree under `Inference_results/`:

```
Inference_results/
├── social_media/
│   ├── deepseek/{D, S_all, S_git, ALL}/                  no S_cvss for deepseek
│   ├── gemma/{D, S_all, S_git, S_cvss, ALL}/
│   ├── gpt/{D, S_all, S_git, S_cvss, ALL}/
│   └── mistral/{D, S_all, S_git, S_cvss, ALL}/
├── megavul/
│   ├── gpt/{D, S_url, S_code, S_cvss, ALL}/
│   ├── mistral/{D, S_url, S_code, S_cvss, ALL}/
│   └── gemma/{D, S_url, S_code, S_cvss, ALL}/
├── nvd_kev/
│   ├── nvd_kev/multiview_hybrid_rerun/
│   └── temporal/{2020_2022_to_2023_2024, 2020_to_2021_2026_noepss}/
├── social_media_summary.csv
├── megavul_summary.csv
├── nvd_kev_summary.csv
└── combined_summary.csv
```

Each variant directory contains `test_results.json` (metrics at
threshold 0.5) and `predictions_test.csv` (per-CVE predictions ranked
by probability). The summary CSVs aggregate the metrics across runs,
one row per run, with columns family / llm / variant / run plus the
usual PR-AUC / ROC-AUC / F1 / precision / recall / Brier / prevalence.

The original `output/<run>/test_results.json` and
`predictions_test.csv` files are not touched; the wrapper passes
`--results-dir` to test_only so the new outputs go into
`Inference_results/` instead.

Useful environment overrides:

```bash
DEVICE=cuda    bash scripts/run_inference_on_all.sh    # force GPU
BATCH_SIZE=64  bash scripts/run_inference_on_all.sh    # bigger batches
THRESHOLD=0.10 bash scripts/run_inference_on_all.sh    # FIRST EPSS operational threshold
OUT_ROOT=/some/other/path bash scripts/run_inference_on_all.sh
```

Per-family logs land at `logs/inference_<family>.log`.

### 5.2 Test a single run

```bash
python -m epss.test_only --run-dir output/epss_mv_gpt_D
```

By default this overwrites the run's own `test_results.json` and
`predictions_test.csv` with values that match the originally saved
ones bit-for-bit. To keep the originals intact and route the outputs
elsewhere, pass `--results-dir`:

```bash
python -m epss.test_only \
    --run-dir output/epss_mv_gpt_D \
    --results-dir Inference_results/megavul/gpt/D
```

Or keep them in the run dir but with a suffix:

```bash
python -m epss.test_only \
    --run-dir output/epss_mv_gpt_D \
    --output-suffix _verify
# writes test_results_verify.json and predictions_test_verify.csv
```

### 5.3 Test the same model on a different held-out dataset

```bash
python -m epss.test_only \
    --run-dir output/epss_mv_gpt_D \
    --external-test-labeled path/to/some_other_labeled_cves.json \
    --results-dir Inference_results/cross_distribution/mv_gpt_D_on_other
```

The external file must follow the same schema as the labelled records
the model trained against. Use this for cross-distribution evaluation:
a model trained on social-media data tested on Megavul, or the
reverse.

### 5.4 Batch-test a glob of runs (skip the wrapper)

```bash
python -m epss.test_only \
    --runs-glob 'output/epss_*_v2_*' 'output/epss_mv_*' \
    --summary-out output/test_only_combined_summary.csv \
    --device cuda
```

This is equivalent to running test_only once per matching run,
writing each result back into the run dir, and aggregating the
metrics into the summary CSV. Use this when you want to overwrite
the in-run files in one shot and do not need the
`Inference_results/` tree.

### 5.5 What test_only does NOT do

- It does not train, does not fetch from NVD, and does not modify
  the source data or the dataset cache.
- It does not compute val-tuned thresholds. For threshold sweeps and
  the val-tuned best-F1 reporting, use `epss/threshold_analysis.py`
  (Section 7).
- It does not visualise. For ROC and calibration plots, train through
  `run_pipeline.py` once; the visualisation is part of the training
  end-stage.

---

## 6. Inference on unseen CVEs

`epss/infer.py` scores arbitrary CVEs that the model has never seen,
fetching the latest NVD descriptions, KEV labels and EPSS scores live
from the public APIs.

### 6.1 Score CVEs published in a date range

```bash
python -m epss.infer \
    --checkpoint output/epss_mv_gpt_D/best_model.pt \
    --config     output/epss_mv_gpt_D/experiment_config.json \
    --after-date  2026-01-01 \
    --before-date 2026-04-01 \
    --max-cves 500 \
    --output-dir output/inference/2026_q1
```

Output: `output/inference/2026_q1/predictions.csv` with one row per
CVE (CVE-ID, predicted probability, risk tier, predicted label,
current EPSS, in-KEV flag, description).

### 6.2 Score a specific list of CVEs

```bash
# inline list
python -m epss.infer \
    --checkpoint output/epss_mv_gpt_D/best_model.pt \
    --config     output/epss_mv_gpt_D/experiment_config.json \
    --cve-ids "CVE-2024-23897,CVE-2024-9474,CVE-2023-50164" \
    --output-dir output/inference/handpicked

# from a text file (one CVE-ID per line)
python -m epss.infer \
    --checkpoint output/epss_mv_gpt_D/best_model.pt \
    --config     output/epss_mv_gpt_D/experiment_config.json \
    --cve-file my_cves.txt \
    --output-dir output/inference/handpicked
```

### 6.3 Useful inference flags

| Flag | What it does |
|---|---|
| `--max-cves N` | Cap the number of CVEs fetched in date-range mode |
| `--threshold X` | Decision threshold for the predicted-label column (default 0.5) |
| `--skip-fetch` | Use already-built labelled JSON instead of calling the NVD API |
| `--labeled-json PATH` | Pre-built labelled records to score directly |
| `--keep-work-dir` | Keep the temporary graph cache after the run (for debugging) |
| `--nvd-api-key XYZ` | NVD API key to avoid the 50-request-per-30-second public throttle |

Full list: `python -m epss.infer --help`.

---

## 7. Threshold analysis and reporting

`epss/threshold_analysis.py` post-processes the saved predictions
without re-running anything. It walks every run directory you point
it at and writes a unified report.

```bash
python -m epss.threshold_analysis \
    --runs-glob 'output/epss_*_v2_*' 'output/epss_mv_*' \
    --root output \
    --out-dir output/threshold_analysis
```

It writes:

- `output/threshold_analysis/per_run_metrics.csv` (one row per run, with PR-AUC, ROC-AUC, Brier, F1 at multiple operational thresholds, and val-tuned best-F1 threshold)
- `output/threshold_analysis/per_dataset_summary.csv` (means and standard deviations by dataset)
- `output/threshold_analysis/REPORT.md` (human-readable summary)

The report covers: PR-AUC and lift over chance baseline, F1 at FIRST
EPSS operational thresholds (0.05, 0.10, 0.20, 0.50), val-tuned
best-F1 threshold scored on the test set (the principled choice for
imbalanced binary reporting), and a test-tuned best-F1 ceiling
clearly flagged as upward-biased.

---

## 8. Where the results live

| Asset | Path |
|---|---|
| Per-run trained models and metrics | `output/epss_<run-name>/` |
| Per-variant test outputs (organised by family/llm/variant) | `Inference_results/<family>/<llm>/<variant>/` |
| Per-family aggregated summary | `Inference_results/<family>_summary.csv` |
| Combined summary across all 37 runs | `Inference_results/combined_summary.csv` |
| Aggregated threshold report | `output/threshold_analysis/REPORT.md` |
| Per-run unified metrics CSV | `output/threshold_analysis/per_run_metrics.csv` |
| Top-10 KEV cross-reference data | `output/threshold_analysis/top10_full_kev.json` |
| Live FIRST EPSS snapshot used by the paper | `output/threshold_analysis/epss_live.json` |
| NVD/KEV metrics archived from the LaTeX (pre-deletion) | `output/threshold_analysis/nvd_kev_archived_results.json` |
| Inference outputs on freshly-fetched CVEs | `output/inference/<run>/` |
| Methodology paper sources | `EPSS_Latex/multiview_tpg_formula.tex`, `EPSS_Latex/dataset_features.tex`, `EPSS_Latex/experiment_results_35_runs.tex` |

The current headline numbers (in-distribution test sets):

| Dataset family | Best mean PR-AUC | Best variant per LLM |
|---|---|---|
| Social-media (4 LLMs, 19 runs) | 0.837 | description-only |
| Megavul (3 LLMs, 15 runs) | 0.423 | description-only or ALL, depending on LLM |
| NVD/KEV binary rerun | 0.94 | hybrid with EPSS as input |
| NVD/KEV temporal shift (2020-22 -> 2023-24) | 0.80-0.85 | hybrid, with EPSS as input |
| NVD/KEV no-EPSS extrapolation (2020 -> 2021-26) | 0.20-0.49 | depends on training run |

The third NVD/KEV row varies by retrain because the training set is
small (about 1,500 records with 145 positives) and the test window is
wide (2021-2026). Different random seeds and different KEV snapshots
produce visibly different numbers; the LaTeX-archived run reached
0.49, recent retrains land around 0.20-0.30.

PR-AUC times chance lift is the right way to compare across families
because prevalence differs sharply: 15.5% on social-media, 2.2% on
Megavul, 20% on NVD/KEV balanced, 8-10% on the temporal NVD/KEV
splits.

---

## 9. Reproducing the paper

The full numbers, methodology and ablation results are in three
LaTeX documents. Compile each one twice (longtables need a second
pass to settle):

```bash
cd EPSS_Latex
pdflatex -interaction=nonstopmode multiview_tpg_formula.tex && pdflatex -interaction=nonstopmode multiview_tpg_formula.tex
pdflatex -interaction=nonstopmode dataset_features.tex && pdflatex -interaction=nonstopmode dataset_features.tex
pdflatex -interaction=nonstopmode experiment_results_35_runs.tex && pdflatex -interaction=nonstopmode experiment_results_35_runs.tex
```

To regenerate the threshold-analysis report against the current saved
predictions:

```bash
python -m epss.threshold_analysis \
    --runs-glob 'output/epss_*_v2_*' 'output/epss_mv_*' 'output/epss_nvd_kev_*' 'output/epss_temporal_*' \
    --out-dir output/threshold_analysis
```

To regenerate the per-family inference outputs under
`Inference_results/` (the recommended way to validate the test-set
numbers reported in the paper):

```bash
bash scripts/run_inference_on_all.sh
```

The run produces `Inference_results/combined_summary.csv` (one row
per run) which can be diffed against the per-run metrics from the
threshold-analysis step; the two should agree at threshold 0.5
within floating-point noise.
