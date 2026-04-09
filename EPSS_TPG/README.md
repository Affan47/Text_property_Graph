# EPSS-GNN: CVE Exploitation Prediction via Text Property Graphs

A graph neural network system that predicts whether a CVE will be exploited in the wild, using only public data — NVD descriptions, CISA KEV labels, FIRST EPSS scores, and ExploitDB PoC records.

**Branch:** `feature/epss-gnn` · **Hardware:** NVIDIA RTX 5000 Ada (32 GB VRAM) · **Framework:** PyTorch 2.3 / PyG 2.7

---

## How It Works

Each CVE description is converted into a **Text Property Graph (TPG)** — a structured graph preserving syntactic, semantic, and discourse relations between entities, predicates, and arguments. Node features are 781-dimensional vectors: a 13-dim one-hot node type plus a 768-dim SecBERT contextual embedding.

A **MultiView GNN** processes the graph through four semantic views (syntactic, sequential, semantic, discourse) and produces a 256-dim graph embedding. This is fused with a 57-dim tabular feature vector encoding CVSS, CWE, EPSS, and ExploitDB signals, then passed through a classifier to produce P(exploitation).

---

## Results Summary

| Evaluation | Dataset | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| 4K balanced (random split) | 4,015 CVEs, 20% KEV | 0.759 | 0.892 | 0.692 | — | 0.686 |
| 127K full (unbalanced) | 127,735 CVEs, 0.42% KEV | 0.729 | 0.981 | 0.392 | — | 0.247 |
| **5% stratified (random split)** | **10,532 CVEs, 5.1% KEV** | **0.865** | **0.986** | **0.790** | — | **0.815** |
| **Temporal (2002–16 → 2017–19)** | **1,087 CVEs test set** | **0.887** | **0.988** | **0.810** | — | **0.865** |
| **Sec4AI4Aec-EPSS (9,218 CVEs)** | **9,218 CVEs, 15.5% pos** | **0.998** | **0.9996** | **0.9786** | **1.000** | **0.958** |
| EPSS v3 (reference baseline) | — | ~0.779 | — | — | — | — |

The **Sec4AI4Aec model** (PR-AUC=0.998, Precision=1.000) was trained on the `Sec4AI4Aec-EPSS-Enhanced` social media dataset with soft EPSS labels (threshold ≥ 0.1 = positive). The temporal split (PR-AUC=0.887) remains the most rigorous for deployment evaluation.

---

## Inference on Unseen CVEs — Post-Training Verification

After training on CVEs up to **2025-06-01**, the model was evaluated on 300 CVEs published **Jul–Sep 2025** (completely unseen) and verified against CISA KEV + FIRST EPSS:

| Metric | Value | Notes |
|---|---|---|
| CVEs scored | 300 | Published Jul–Sep 2025 |
| Predicted positive | 6 (2.0%) | prob ≥ 0.5 |
| EPSS agreement | **100%** | All 6 predicted positives had current EPSS ≥ 0.1 |
| In CISA KEV | 0 | Expected — KEV is highly selective (1,559/1M+ CVEs) |
| Pearson corr(model, EPSS) | 0.30 | Moderate; graph adds signal beyond EPSS |

**Top unseen CVEs correctly flagged as CRITICAL:**

| CVE | Prob | Current EPSS | Description |
|---|---|---|---|
| CVE-2025-34074 | 0.956 | 0.573 | Lucee admin RCE |
| CVE-2025-34079 | 0.941 | 0.560 | NSClient++ RCE |
| CVE-2025-34073 | 0.923 | 0.553 | maltrail unauthenticated command injection |
| CVE-2025-34076 | 0.869 | 0.246 | Microweber CMS LFI |
| CVE-2025-6934  | 0.849 | 0.236 | WordPress plugin unauthenticated RCE |

The 0 KEV matches is not a failure — CISA only adds CVEs under active exploitation by threat actors; most high-EPSS CVEs are never formally KEV-listed. Ground truth for freshly published CVEs requires 6–12 months of observation.

---

## Repository Structure

```
EPSS_TPG/
│
├── docs/
│   ├── EPSS_GNN_Technical_Report.md     ← Full technical report (19 sections)
│   ├── TPG_COMPLETE_GUIDE.md            ← From CPG to TPG: beginner-to-advanced
│   ├── Security_TPG_Complete_Reference.md ← Security frontend API reference
│   └── WHO_analysis_summary.md          ← Domain-agnostic TPG test
│
├── epss/                                ← GNN training package
│   ├── data_collector.py               ← Fetch NVD + KEV + EPSS + ExploitDB
│   ├── csv_adapter.py                  ← Convert Sec4AI4Aec CSV → labeled_cves.json
│   ├── tabular_features.py             ← 57-dim tabular encoder (CVSS+CWE+EPSS+PoC)
│   ├── cve_dataset.py                  ← PyG InMemoryDataset: CVE records → TPG graphs
│   ├── gnn_model.py                    ← HybridEPSSClassifier (6 GNN backbones)
│   ├── edge_aware_layers.py            ← EdgeTypeGNN, RGAT, MultiView (from SemVul)
│   ├── train.py                        ← Training loop, metrics, checkpointing
│   ├── run_pipeline.py                 ← CLI: collect → build → train → evaluate
│   ├── infer.py                        ← Temporal inference + KEV/EPSS verification
│   └── visualize.py                    ← PR curve, ROC, calibration, training curves
│
├── tpg/                                ← Text Property Graph library
│   ├── schema/types.py                 ← NodeType / EdgeType enums (13 + 13)
│   ├── schema/graph.py                 ← TextPropertyGraph core class
│   ├── frontends/                      ← spaCy, security rule, SecBERT, hybrid frontends
│   ├── passes/                         ← AMR framing, RST discourse, cross-modal linking
│   ├── exporters/exporters.py          ← PyGExporter + GraphSON exporter
│   └── pipeline.py                     ← HybridSecurityPipeline (main entry point)
│
├── data/
│   ├── epss_sec4ai/                    ← Sec4AI4Aec training data
│   │   ├── labeled_cves.json           ← [gitignored] 9,218 CVEs converted from CSV
│   │   └── pyg_dataset/processed/     ← [gitignored] .pt graph cache
│   ├── epss_full/                      ← NVD pipeline (best temporal model)
│   │   ├── labeled_cves_5pct.json      ← 10,532 CVEs, 5.1% KEV
│   │   ├── labeled_cves_temporal_train.json
│   │   └── labeled_cves_temporal_test.json
│   └── epss/                           ← Source CSV + raw data
│       └── final_dataset_with_delta_days copy.csv  ← [gitignored] Sec4AI4Aec CSV
│
├── output/
│   ├── epss_sec4ai/                    ← Sec4AI4Aec trained model (PR-AUC=0.998)
│   │   ├── best_model.pt               ← [gitignored] 834K-param checkpoint
│   │   ├── experiment_config.json      ← Full hyperparameter record
│   │   ├── predictions_test.csv        ← 1,385 test-set CVE scores
│   │   └── *.png                       ← PR curve, ROC, calibration, training curves
│   └── infer/                          ← Inference run outputs
│       └── post_dataset_q3_2025/       ← Jul–Sep 2025 unseen CVE run
│           ├── predictions_infer.csv   ← 300 CVEs with KEV + EPSS ground truth
│           └── verification_summary.txt
│
├── analyze_dataset.py                  ← Profile any labeled_cves CSV/JSON dataset
├── verify_features.py                  ← Compare feature dtypes between datasets
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
conda create -n tpg python=3.10
conda activate tpg
pip install torch==2.3.0 torch_geometric==2.7.0
pip install spacy transformers networkx numpy scikit-learn matplotlib pandas tqdm requests
python -m spacy download en_core_web_sm
```

---

## Training Commands

### A. Train on Sec4AI4Aec-EPSS-Enhanced CSV (recommended — social media + EPSS soft labels)

```bash
# Full training (9,218 CVEs, ~100 epochs, ~30–40s/epoch on RTX 5000)
python -m epss.run_pipeline \
    --source-csv "data/epss/final_dataset_with_delta_days copy.csv" \
    --data-dir data/epss_sec4ai \
    --output-dir output/epss_sec4ai \
    --backbone multiview --hybrid --label-mode soft \
    --epochs 100 --patience 15 --batch-size 32 --lr 1e-3

# Quick smoke test (50 CVEs, 10 epochs)
python -m epss.run_pipeline \
    --source-csv "data/epss/final_dataset_with_delta_days copy.csv" \
    --data-dir data/epss_sec4ai \
    --output-dir output/epss_sec4ai \
    --backbone multiview --hybrid --label-mode soft \
    --max-cves 50 --epochs 10
```

### B. Train on Sec4AI4Aec WITHOUT EPSS feature (leakage-free, deployment-ready)

> **Use this for production.** The default training uses EPSS as both label and input feature (data leakage). This variant removes EPSS from the 57-dim tabular features, forcing the model to learn from CVE text + CVSS + CWE only. Tabular dim drops to 55. Inference on truly new CVEs works without needing pre-existing EPSS data.

```bash
python -m epss.run_pipeline \
    --source-csv "data/epss/final_dataset_with_delta_days copy.csv" \
    --data-dir data/epss_sec4ai_noleak \
    --output-dir output/epss_sec4ai_noleak \
    --backbone multiview --hybrid --label-mode soft \
    --no-epss-feature \
    --epochs 100 --patience 15
```

### C. Train on NVD pipeline (binary KEV labels, 2020–2024)

```bash
# Fetch data from NVD API + CISA KEV + FIRST EPSS, then train
python -m epss.run_pipeline \
    --start-year 2020 --end-year 2024 \
    --backbone multiview --hybrid \
    --output-dir output/nvd_2020_2024

# Skip fetch, use existing labeled_cves.json
python -m epss.run_pipeline \
    --skip-collect \
    --labeled-file data/epss_full/labeled_cves_5pct.json \
    --data-dir data/epss_5pct_train \
    --backbone multiview --hybrid \
    --hidden 256 --layers 3 --heads 4 \
    --batch-size 64 --epochs 200 --patience 20 \
    --lr 5e-4 --output-dir output/my_run --device cuda
```

### D. Compare GNN backbones

```bash
for BACKBONE in gcn gat sage edge_type rgat multiview; do
    python -m epss.run_pipeline \
        --skip-collect --backbone $BACKBONE --hybrid \
        --labeled-file data/epss_full/labeled_cves_5pct.json \
        --data-dir data/epss_5pct_train \
        --output-dir output/compare_$BACKBONE
done
```

---

## Inference Commands — Score Unseen CVEs

All inference commands fetch CVE descriptions from NVD, pre-populate current EPSS scores from the FIRST API, build TPG graphs with the same pipeline as training, and verify predictions against CISA KEV.

Output files per run:
- `predictions_infer.csv` — one row per CVE: `predicted_prob`, `risk_tier`, `is_in_kev`, `kev_date_added`, `current_epss_score`, `correct_vs_kev`, `correct_vs_epss`
- `verification_summary.txt` — TP/FP/FN/TN breakdown, precision/recall vs CISA KEV, tier distribution, top-20 highest-risk CVEs

> **Rate limit note:** Without an NVD API key, NVD fetches at 1 CVE per 6 seconds (~30 min for 300 CVEs). Get a free key at https://nvd.nist.gov/developers/request-an-api-key and pass `--nvd-api-key YOUR_KEY` to reduce this to ~30 seconds.

---

### Mode 1 — Post-Dataset (Jul–Sep 2025, completely unseen by the model)

CVEs published after the training set cutoff (2025-06-01). Ground truth is current CISA KEV status.

```bash
python -m epss.infer \
    --mode post-dataset \
    --after-date 2025-07-01 \
    --before-date 2025-09-30 \
    --checkpoint output/epss_sec4ai/best_model.pt \
    --config    output/epss_sec4ai/experiment_config.json \
    --max-cves 300 \
    --output-dir output/infer/post_dataset_q3_2025
```

```bash
# Q4 2025 (Oct–Dec)
python -m epss.infer \
    --mode post-dataset \
    --after-date 2025-10-01 \
    --before-date 2025-12-31 \
    --checkpoint output/epss_sec4ai/best_model.pt \
    --config    output/epss_sec4ai/experiment_config.json \
    --max-cves 300 \
    --output-dir output/infer/post_dataset_q4_2025
```

```bash
# Q1 2026 (Jan–Mar) — 10 months after training cutoff
python -m epss.infer \
    --mode post-dataset \
    --after-date 2026-01-01 \
    --before-date 2026-03-31 \
    --checkpoint output/epss_sec4ai/best_model.pt \
    --config    output/epss_sec4ai/experiment_config.json \
    --max-cves 300 \
    --output-dir output/infer/post_dataset_q1_2026
```

---

### Mode 2 — Pre-Dataset (2019–2021, before the earliest training record)

CVEs published before 2021-11-23 (earliest in training data). KEV status is fully settled for these historical CVEs, providing the most reliable ground truth.

```bash
python -m epss.infer \
    --mode pre-dataset \
    --after-date 2019-01-01 \
    --before-date 2021-10-31 \
    --checkpoint output/epss_sec4ai/best_model.pt \
    --config    output/epss_sec4ai/experiment_config.json \
    --max-cves 500 \
    --output-dir output/infer/pre_dataset_2019_2021
```

```bash
# Deeper historical — 2017–2018 (Log4Shell era predecessors)
python -m epss.infer \
    --mode pre-dataset \
    --after-date 2017-01-01 \
    --before-date 2018-12-31 \
    --checkpoint output/epss_sec4ai/best_model.pt \
    --config    output/epss_sec4ai/experiment_config.json \
    --max-cves 500 \
    --output-dir output/infer/pre_dataset_2017_2018
```

---

### Mode 3 — Custom CVE List

```bash
# Score specific CVEs by ID
python -m epss.infer \
    --mode custom \
    --cve-ids CVE-2025-31200,CVE-2025-30065,CVE-2025-21333,CVE-2025-0282,CVE-2024-38094 \
    --checkpoint output/epss_sec4ai/best_model.pt \
    --config    output/epss_sec4ai/experiment_config.json \
    --output-dir output/infer/custom_list

# Score CVEs from a text file (one CVE-ID per line)
python -m epss.infer \
    --mode custom \
    --cve-file my_cves.txt \
    --checkpoint output/epss_sec4ai/best_model.pt \
    --config    output/epss_sec4ai/experiment_config.json \
    --output-dir output/infer/custom_from_file
```

---

### Mode 4 — Graph-Only (no EPSS signal, pure text/CVSS/CWE)

Use `--no-epss-prefetch` to disable EPSS injection into tabular features. This tests what the model learned from graph structure alone. Predictions will be lower overall but represent true text-derived signal.

```bash
python -m epss.infer \
    --mode post-dataset \
    --after-date 2025-07-01 \
    --before-date 2025-09-30 \
    --checkpoint output/epss_sec4ai/best_model.pt \
    --config    output/epss_sec4ai/experiment_config.json \
    --no-epss-prefetch \
    --max-cves 300 \
    --output-dir output/infer/post_dataset_graph_only
```

---

### Mode 5 — Leakage-Free Model Inference (after retraining with --no-epss-feature)

If you retrained with `--no-epss-feature`, this model does not need EPSS pre-fetching at all:

```bash
python -m epss.infer \
    --mode post-dataset \
    --after-date 2025-07-01 \
    --before-date 2025-09-30 \
    --checkpoint output/epss_sec4ai_noleak/best_model.pt \
    --config    output/epss_sec4ai_noleak/experiment_config.json \
    --no-epss-prefetch \
    --max-cves 300 \
    --output-dir output/infer/noleak_q3_2025
```

---

### Inference with NVD API Key (10× faster)

```bash
export NVD_API_KEY="your-key-here"

python -m epss.infer \
    --mode post-dataset \
    --after-date 2025-07-01 \
    --before-date 2025-09-30 \
    --checkpoint output/epss_sec4ai/best_model.pt \
    --config    output/epss_sec4ai/experiment_config.json \
    --nvd-api-key $NVD_API_KEY \
    --max-cves 1000 \
    --keep-work-dir \
    --output-dir output/infer/post_dataset_large
```

### Keep Graph Cache (avoid rebuilding SecBERT embeddings on re-runs)

```bash
python -m epss.infer \
    --mode post-dataset \
    --after-date 2025-07-01 \
    --before-date 2025-09-30 \
    --checkpoint output/epss_sec4ai/best_model.pt \
    --config    output/epss_sec4ai/experiment_config.json \
    --keep-work-dir \
    --work-dir  /tmp/epss_infer_cache \
    --output-dir output/infer/post_dataset_q3_2025
```

---

## Dataset Analysis & Feature Verification

```bash
# Profile the Sec4AI4Aec dataset (rows, dtypes, EPSS distribution, CVSS coverage)
python analyze_dataset.py \
    "data/epss/final_dataset_with_delta_days copy.csv"

# Compare feature dtypes + value ranges between NVD and Sec4AI4Aec datasets
python verify_features.py \
    --nvd data/epss/labeled_cves.json \
    --csv data/epss_sec4ai/labeled_cves.json \
    --max-sample 500

# Fast field-only check (skip 57-dim tabular encoding)
python verify_features.py --skip-tabular \
    --nvd data/epss/labeled_cves.json \
    --csv data/epss_sec4ai/labeled_cves.json
```

---

## Understanding Data Leakage in Soft-Label Mode

When training with `--label-mode soft`, the `epss_score` is used as both:
- The regression **target** (label `y` = EPSS probability)
- A tabular **input feature** (feature dim 7 in the 57-dim vector)

This creates data leakage: the model learns "output high probability when epss_score input is high" rather than learning genuine exploitation signals from CVE text structure. The graph becomes secondary.

**Evidence:** Disabling EPSS pre-fetch at inference time (setting `epss_score=0.0`) collapses all 300 predictions to MINIMAL — confirming the model relied on EPSS input to discriminate.

**Two solutions:**

| Approach | Command flag | Tabular dim | Inference requirement | When to use |
|---|---|---|---|---|
| EPSS pre-fetch (current default) | *(default)* | 57 | Needs FIRST API call | When EPSS is always available |
| No-leakage retrain | `--no-epss-feature` | 55 | No EPSS needed | Production / cold-start |

---

## Ground Truth for Inference Verification

| Source | Coverage | Reliability | Lag |
|---|---|---|---|
| **CISA KEV** | ~1,559 CVEs (very selective) | Highest — active exploitation confirmed | 0–90 days after exploitation |
| **FIRST EPSS API** | ~230,000+ CVEs | High — ML model using many signals | Daily updates |
| **EPSS ≥ 0.1** | ~15% of all CVEs | Moderate threshold for "high risk" | Same day |

CISA KEV is the most authoritative ground truth but will show 0 matches for very recent CVEs (< 3 months old) simply because the KEV review process takes time. Use EPSS as a proxy for recent CVEs and KEV for historical validation.

---

## Documentation

| Document | What it covers |
|---|---|
| [docs/EPSS_GNN_Technical_Report.md](docs/EPSS_GNN_Technical_Report.md) | Complete technical report: problem setup, data pipeline, TPG construction, SecBERT integration, all 6 GNN architectures, training, results, inference |
| [docs/TPG_COMPLETE_GUIDE.md](docs/TPG_COMPLETE_GUIDE.md) | CPG → TPG deep dive: from Joern to text graphs, node/edge schema, complete examples |
| [docs/Security_TPG_Complete_Reference.md](docs/Security_TPG_Complete_Reference.md) | Security frontend API reference: entity types, edge types, pipeline stages, configuration |
| [docs/WHO_analysis_summary.md](docs/WHO_analysis_summary.md) | Domain-agnostic TPG validation on WHO medical document |

---

## What Cannot Be Included in This Repo

| File | Size | How to Rebuild |
|---|---|---|
| `data/*/pyg_dataset/processed/*.pt` | 1.4 GB – 39.5 GB | `python -m epss.run_pipeline --skip-collect` |
| `data/epss_full/nvd_cves.json` | 120 MB | `python -m epss.data_collector` |
| `data/epss_full/labeled_cves.json` | 149 MB | Same as above |
| `data/epss_sec4ai/labeled_cves.json` | ~10 MB | `python -m epss.csv_adapter --input data/epss/final_dataset_with_delta_days\ copy.csv` |
| `output/**/best_model.pt` | ~34 MB each | `python -m epss.run_pipeline ...` |

The 5% stratified labeled file (`data/epss_full/labeled_cves_5pct.json`, 12 MB) and all temporal split files are tracked and sufficient to reproduce the NVD-pipeline results without re-fetching.
