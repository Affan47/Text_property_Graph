# EPSS-GNN: CVE Exploitation Prediction via Text Property Graphs

A graph neural network system that predicts whether a CVE will be exploited in the wild, using only public data — NVD descriptions, CISA KEV labels, FIRST EPSS scores, and ExploitDB PoC records.

**Branch:** `feature/epss-gnn` · **Hardware:** NVIDIA RTX 5000 Ada (32 GB VRAM) · **Framework:** PyTorch 2.3 / PyG 2.7

---

## How It Works

Each CVE description is converted into a **Text Property Graph (TPG)** — a structured graph preserving syntactic, semantic, and discourse relations between entities, predicates, and arguments. Node features are 781-dimensional vectors: a 13-dim one-hot node type plus a 768-dim SecBERT contextual embedding.

A **MultiView GNN** processes the graph through four semantic views (syntactic, sequential, semantic, discourse) and produces a 256-dim graph embedding. This is fused with a 57-dim tabular feature vector encoding CVSS, CWE, EPSS, and ExploitDB signals, then passed through a classifier to produce P(exploitation).

Best result: **PR-AUC = 0.865** on a 5% stratified test split, **0.887** on a strict temporal split (trained on 2002–2016, tested on 2017–2019). Both exceed EPSS v3's self-reported PR-AUC of ~0.779.

---

## Repository Structure

```
TPG_TextPropertyGraph/
│
├── docs/                              ← All documentation
│   ├── EPSS_GNN_Technical_Report.md  ← Full technical report (19 sections, ~3400 lines)
│   ├── TPG_COMPLETE_GUIDE.md         ← From CPG to TPG: beginner-to-advanced guide
│   ├── Security_TPG_Complete_Reference.md  ← Security frontend API reference
│   └── WHO_analysis_summary.md       ← Domain-agnostic TPG test (WHO document)
│
├── epss/                              ← GNN training package
│   ├── data_collector.py             ← Fetch NVD + KEV + EPSS + ExploitDB → labeled JSON
│   ├── tabular_features.py           ← 57-dim tabular encoder (CVSS+CWE+EPSS+PoC)
│   ├── cve_dataset.py                ← PyG InMemoryDataset: CVE records → TPG graphs
│   ├── gnn_model.py                  ← HybridEPSSClassifier (6 GNN backbones)
│   ├── edge_aware_layers.py          ← EdgeTypeGNN, RGAT, MultiView (ported from SemVul)
│   ├── train.py                      ← Training loop, metrics, checkpointing
│   ├── run_pipeline.py               ← CLI: collect → build graphs → train → evaluate
│   └── visualize.py                  ← Plots: PR curve, ROC, calibration, training curves
│
├── tpg/                               ← Text Property Graph library
│   ├── schema/
│   │   ├── types.py                  ← NodeType / EdgeType enums (13 + 13)
│   │   └── graph.py                  ← TextPropertyGraph core class
│   ├── frontends/
│   │   ├── spacy_frontend.py         ← spaCy → TPG translation (deps, NER, SRL, coref)
│   │   ├── security_frontend.py      ← Rule-based security entity extractor
│   │   ├── model_security_frontend.py ← SecBERT-based entity extractor
│   │   └── hybrid_security_frontend.py ← Combined rule + model frontend
│   ├── passes/
│   │   ├── enrichment.py             ← AMR-style framing, RST discourse relations
│   │   └── cross_modal.py            ← TPG ↔ CPG cross-modal linking
│   ├── exporters/
│   │   └── exporters.py              ← PyGExporter + GraphSON exporter
│   └── pipeline.py                   ← HybridSecurityPipeline (main entry point)
│
├── examples/
│   ├── demo.py                       ← End-to-end TPG pipeline demo
│   ├── compare_frontends.py          ← Compare rule vs model vs hybrid frontends
│   ├── experiment.py                 ← Single-CVE graph construction experiment
│   └── TPG_sample_output.json        ← Sample GraphSON output for a CVE description
│
├── tests/
│   └── __init__.py
│
├── data/                              ← All datasets (large files gitignored — see below)
│   ├── epss_full/                    ← Authoritative source: raw + labeled files
│   │   ├── cisa_kev.json             ← CISA KEV catalog (1,554 entries, 2002–2026)
│   │   ├── exploitdb.json            ← ExploitDB PoC database (24,936 entries)
│   │   ├── epss_scores_full.json     ← EPSS bulk scores (323,611 CVEs)
│   │   ├── labeled_cves_5pct.json    ← PRIMARY training file (10,532 CVEs, 5.1% KEV)
│   │   ├── labeled_cves_5pct_noepss.json  ← Cold-start variant (EPSS fields zeroed)
│   │   ├── labeled_cves_temporal_train.json ← Temporal train (2002–2016 KEV)
│   │   ├── labeled_cves_temporal_test.json  ← Temporal test (2017–2019 KEV)
│   │   ├── nvd_cves.json             ← [gitignored] Raw NVD fetch, 135K records, 120 MB
│   │   └── labeled_cves.json         ← [gitignored] Master merged dataset, 127K CVEs, 149 MB
│   │
│   ├── epss/                         ← 4K balanced experiment (12 backbone runs)
│   │   ├── labeled_cves_balanced_v2.json ← 4,015 CVEs, 20% KEV positive rate
│   │   ├── epss_scores.json          ← EPSS snapshot (132K entries)
│   │   ├── epss_scores_full.json     ← EPSS bulk (323K entries, copy)
│   │   ├── epss_scores-2026-03-28.csv ← Raw EPSS CSV download
│   │   ├── cisa_kev.json             ← KEV catalog (copy)
│   │   ├── exploitdb.json            ← ExploitDB (copy)
│   │   └── pyg_dataset/
│   │       ├── raw/                  ← Input JSON for PyG pipeline
│   │       └── processed/            ← Vocab JSONs tracked; .pt files gitignored
│   │
│   ├── epss_5pct_train/              ← 5% stratified graphs (best model)
│   │   └── pyg_dataset/
│   │       ├── raw/labeled_cves.json ← Copy of labeled_cves_5pct.json
│   │       └── processed/            ← edge/node vocab JSONs + [gitignored] 3.3 GB .pt
│   │
│   ├── epss_temporal_train/          ← Temporal split graphs (strictest eval)
│   │   └── pyg_dataset/
│   │       ├── raw/labeled_cves.json ← Copy of labeled_cves_temporal_train.json
│   │       └── processed/            ← Vocab JSONs + [gitignored] 2.3 GB .pt
│   │
│   ├── epss_full_train/              ← Full 127K unbalanced graphs
│   │   └── pyg_dataset/
│   │       ├── raw/                  ← [gitignored] 150 MB JSON copy
│   │       └── processed/            ← Vocab JSONs + [gitignored] 39.5 GB + 2.5 GB .pt
│   │
│   ├── epss_balanced/                ← Legacy (superseded by epss/)
│   │   └── pyg_dataset/raw/          ← 2.7 MB balanced JSON; no processed/ graphs
│   │
│   ├── epss_test/                    ← 30-CVE smoke test dataset
│   │   ├── labeled_cves.json         ← 30 CVEs for pipeline sanity checking
│   │   ├── cisa_kev.json / epss_scores.json
│   │   └── pyg_dataset/processed/    ← [gitignored] tiny .pt cache
│   │
│   ├── epss_qtest/                   ← Empty scratch directory
│   ├── text/                         ← Plain text samples for TPG pipeline testing
│   └── pdfs/                         ← PDF frontend test files
│
├── output/                            ← All experiment results
│   ├── epss_<backbone>_{text,hybrid}/ ← One dir per experiment (12 runs on 4K dataset)
│   │   ├── experiment_config.json    ← All hyperparameters
│   │   ├── test_results.json         ← PR-AUC, ROC-AUC, F1, Precision, Recall
│   │   ├── training_history.json     ← Per-epoch train/val metrics
│   │   ├── predictions_test.csv      ← Per-CVE scores on test set
│   │   ├── *.png                     ← PR curve, ROC, calibration, training curves
│   │   └── best_model.pt             ← [gitignored] Saved weights
│   │
│   ├── epss_full_5pct_multiview_hybrid/  ← Best model (PR-AUC=0.865)
│   │   ├── cwe_vocab.json            ← Fitted CWE vocabulary (top-25 CWEs)
│   │   └── ...                       ← Same structure as above
│   │
│   ├── epss_temporal_multiview_hybrid/   ← Temporal model (PR-AUC=0.887)
│   ├── epss_full_multiview_hybrid/       ← Full 127K unbalanced run
│   ├── epss_full_10k_test/               ← 10K subset fast-iteration run
│   ├── comparison/                       ← Cross-backbone comparison JSONs
│   ├── graphson/                         ← GraphSON-format TPG samples (general/medical/security)
│   ├── pyg/                              ← PyG-format TPG samples (same domains)
│   ├── analysis/                         ← (legacy, docs moved to docs/)
│   └── inference/                        ← Inference run outputs
│       ├── predictions_20260406.csv      ← April 2026 run: 6,109 CVEs scored
│       ├── temporal_eval_jan2024.csv     ← Jan 2024 eval (broken EPSS — historical)
│       └── temporal_eval_jan2024_fixed.csv ← Jan 2024 eval (fixed local EPSS)
│
├── infer.py                           ← Operational inference script (score new CVEs)
└── generate_visualizations.py         ← Re-generate all plots from any checkpoint
```

---

## Quick Start

### 1. Install dependencies

```bash
conda create -n tpg python=3.12
conda activate tpg
pip install torch==2.3.0 torch_geometric==2.7.0
pip install spacy transformers networkx numpy scikit-learn matplotlib
python -m spacy download en_core_web_sm
```

### 2. Fetch data and build the training dataset

```bash
python -m epss.run_pipeline --backbone multiview --hybrid \
    --output-dir output/my_run
```

Or use the pre-built 5% stratified dataset (already in `data/epss_full/`):

```bash
python -m epss.run_pipeline --backbone multiview --hybrid \
    --skip-collect \
    --labeled-file data/epss_full/labeled_cves_5pct.json \
    --data-dir data/epss_5pct_train \
    --hidden 256 --layers 3 --heads 4 \
    --batch-size 64 --epochs 200 --patience 20 \
    --lr 5e-4 --output-dir output/my_run --device cuda
```

### 3. Score new CVEs with the trained model

```bash
# Score all CVEs published in the last 30 days
python infer.py --recent-days 30 \
    --epss-file data/epss_full/epss_scores_full.json \
    --output output/inference/predictions_$(date +%Y%m%d).csv

# Score specific CVEs
python infer.py --cve-ids CVE-2024-1234 CVE-2024-5678

# Temporal evaluation (train cutoff → score next N days)
python infer.py --temporal-eval \
    --train-cutoff 2024-01-01 --eval-days 30 \
    --epss-file data/epss_full/epss_scores_full.json
```

---

## Results Summary

| Evaluation | Dataset | PR-AUC | ROC-AUC | F1 | Recall |
|---|---|---|---|---|---|
| 4K balanced (random split) | 4,015 CVEs, 20% KEV | 0.759 | 0.892 | 0.692 | 0.686 |
| 127K full (unbalanced) | 127,735 CVEs, 0.42% KEV | 0.729 | 0.981 | 0.392 | 0.247 |
| **5% stratified (random split)** | **10,532 CVEs, 5.1% KEV** | **0.865** | **0.986** | **0.790** | **0.815** |
| **Temporal (2002–16 → 2017–19)** | **1,087 CVEs test set** | **0.887** | **0.988** | **0.810** | **0.865** |
| Inference Jan 2024 (fixed EPSS) | 2,647 CVEs, 15 KEV | 0.328 | 0.901 | 0.378 | 0.467 |
| EPSS v3 (reference baseline) | — | ~0.779 | — | — | — |

The temporal split result (PR-AUC=0.887) is the most rigorous: the model is trained on CVEs from 2002–2016 and evaluated on 2017–2019 CVEs it has never seen. It correctly identifies Shellshock, Heartbleed, PHPMailer RCE, and Jenkins CLI RCE purely from text structure and tabular metadata patterns learned in an earlier era.

---

## What Cannot Be Included in This Repo

The following are gitignored due to size but can be rebuilt:

| File | Size | How to Rebuild |
|---|---|---|
| `data/*/pyg_dataset/processed/*.pt` | 1.4 GB – 39.5 GB | `python -m epss.run_pipeline --skip-collect` |
| `data/epss_full/nvd_cves.json` | 120 MB | `python -m epss.data_collector` (fetches NVD API) |
| `data/epss_full/labeled_cves.json` | 149 MB | Same as above (merges NVD + KEV + EPSS) |
| `output/**/best_model.pt` | ~34 MB each | `python -m epss.run_pipeline ...` |

The 5% stratified labeled file (`data/epss_full/labeled_cves_5pct.json`, 12 MB) and all temporal split files are tracked and sufficient to reproduce the best results without re-fetching NVD.

---

## Documentation

| Document | What it covers |
|---|---|
| [docs/EPSS_GNN_Technical_Report.md](docs/EPSS_GNN_Technical_Report.md) | Complete technical report: problem setup, data pipeline, TPG construction, SecBERT integration, all 6 GNN architectures, training, results, inference |
| [docs/TPG_COMPLETE_GUIDE.md](docs/TPG_COMPLETE_GUIDE.md) | CPG → TPG deep dive: from Joern to text graphs, node/edge schema, complete examples |
| [docs/Security_TPG_Complete_Reference.md](docs/Security_TPG_Complete_Reference.md) | Security frontend API reference: entity types, edge types, pipeline stages, configuration |
| [docs/WHO_analysis_summary.md](docs/WHO_analysis_summary.md) | Domain-agnostic TPG validation on WHO medical document |
