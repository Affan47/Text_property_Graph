# EPSS-TPG: Exploit Prediction with Text Property Graphs

Artifact branch: `artifact/epss-tpg`.

This repository holds the EPSS-TPG model, the pipeline that builds a
Text Property Graph (TPG) from a CVE input text, the multi-view GNN
that reads the graph, the hybrid graph-plus-tabular head, and the
batch scripts that reproduce the training and evaluation runs.

## Repository layout

```
Text_property_Graph/
├── README.md                                  (this file)
│
├── SummTPGVul/                                submodule: source CSVs for both
│                                              dataset families (social-media
│                                              + Megavul), tracked via Git LFS
│
└── EPSS_TPG/                                  main project
    ├── README.md                              project-level instructions
    ├── epss/                                  training / inference / dataset code
    ├── tpg/                                   TPG backend (spaCy frontend,
    │                                          security frontend, hybrid frontend,
    │                                          schema, passes, exporters)
    ├── analysis/                              top-level dataset-analysis scripts
    ├── inference/                             user-facing inference CLI for
    │                                          scoring fresh CVEs
    ├── scripts/                               batch shell scripts, grouped by
    │                                          purpose (training/, inference/,
    │                                          analysis/)
    ├── examples/                              standalone examples and demos
    ├── datasets_info/                         per-dataset profiling artefacts
    │                                          and ablation summaries
    ├── docs/                                  in-project documentation
    ├── inference_results/                     test-only outputs per saved
    │                                          checkpoint
    ├── data/                                  per-dataset working trees
    │                                          (per-experiment caches not tracked)
    └── outputs/                               training outputs per run, grouped
                                               by family (social_media/, megavul/,
                                               nvd_kev/, security_ablation/)
```

## Datasets (submodule)

The source CSVs for both dataset families live in a single git submodule
pinned to a specific commit so the dataset version is reproducible and
the model code stays small.

| Submodule | Holds |
|---|---|
| `SummTPGVul` | Both dataset families under one tree. Social-media CSVs (GPT, Gemma, Mistral LLM summaries per CVE) under `SummVul/Social_Media_Dataset/Data_Files/`. Megavul commit-based CSVs (GPT, Gemma, Mistral) under `SummVul/Data_Files/megavul/`. The submodule ships a DeepSeek CSV too, but it is not used by the baseline. |

To clone the full artifact with the dataset submodule:

```bash
git clone --recurse-submodules https://github.com/<owner>/Text_property_Graph.git
cd Text_property_Graph
git checkout artifact/epss-tpg
```

If the repository is already cloned without the submodule:

```bash
git submodule update --init --recursive
```

The dataset CSVs are tracked via Git LFS inside `SummTPGVul`, so a
working `git-lfs` installation is required to pull them:

```bash
sudo apt install -y git-lfs && git lfs install
cd SummTPGVul && git lfs pull && cd ..
```

## Reproducing the experiments

Three classes of run are scripted:

1. **Social-media ablation (15 runs):** three LLMs (GPT, Gemma,
   Mistral) crossed with five text-source variants
   (`D`, `SMP`, `S_git`, `S_cvss`, `ALL`), for a total of
   `3 × 5 = 15` runs. DeepSeek is intentionally excluded.
2. **Megavul ablation (15 runs):** three LLMs (GPT, Gemma, Mistral)
   crossed with five variants (`D`, `S_url`, `S_code`, `S_cvss`,
   `ALL`), for a total of `3 × 5 = 15` runs.
3. **NVD/KEV reference runs:** binary KEV classification and the
   two temporal-shift configurations.

The security-frontend ablation reruns the same matrix (the 15-run
social-media block plus the 15-run Megavul block plus the three
NVD/KEV runs) with `--no-security-frontend` on every run, landing
in `EPSS_TPG/outputs/security_ablation/`.

## Running the scripts

All commands below are runnable from any cwd — the shell scripts
resolve the project root from `${BASH_SOURCE[0]}`. Python entry
points expect to be invoked from `EPSS_TPG/`.

### Training

```bash
cd EPSS_TPG

# Full 15-run social-media baseline (GPT, Gemma, Mistral × D/SMP/S_git/S_cvss/ALL)
scripts/training/run_social_media_retrain.sh

# 5-run GPT NOSEC ablation (paired with the social-media baseline)
scripts/training/run_gpt_nosec_retrain.sh

# Full 33-run text-source ablation (social-media + Megavul)
scripts/training/run_all_summary_experiments.sh

# Full 33-run security-frontend ablation (everything with --no-security-frontend)
scripts/training/run_all_no_security_experiments.sh

# Common flags accepted by every batch script:
#   --dry-run       preview without executing
#   --no-overwrite  skip runs whose test_results.json already exists
#   --quiet         stream per-run output to log file instead of stdout
#   <regex>         positional filter, e.g. 'gpt', 'S_cvss', 'mv_*'
```

Single-configuration training (without a batch wrapper):

```bash
python -m epss.run_pipeline \
    --source-csv ../SummTPGVul/SummVul/Social_Media_Dataset/Data_Files/gpt_combined_summ.csv \
    --data-dir   data/epss_gpt_v2_ALL \
    --output-dir outputs/social_media/gpt/ALL \
    --backbone multiview --hybrid --label-mode soft --epochs 100 \
    --no-epss-feature --include-summary-in-tpg --summary-source combined
```

### Inference and test-only evaluation

```bash
cd EPSS_TPG

# Re-evaluate every saved checkpoint, overwriting in-place test_results.json
scripts/inference/test_all_datasets.sh                # all four families
scripts/inference/test_all_datasets.sh social_media   # one family only

# Same evaluation but route results into a fresh tree under inference_results/
scripts/inference/run_inference_on_all.sh

# Single saved checkpoint:
python -m epss.test_only --run-dir outputs/social_media/gpt/ALL

# Score fresh CVEs from NVD (five modes):
python inference/infer.py --checkpoint outputs/social_media/gpt/ALL/best_model.pt \
                          --config     outputs/social_media/gpt/ALL/experiment_config.json \
                          --cve-ids CVE-2024-1234 CVE-2024-5678
python inference/infer.py --cve-file ids.txt        --checkpoint ... --config ...
python inference/infer.py --recent-days 30          --checkpoint ... --config ...
python inference/infer.py --date-range 2024-01-01 2024-01-31 --checkpoint ... --config ...
python inference/infer.py --temporal-eval --train-cutoff 2024-01-01 --eval-days 30 \
                          --checkpoint ... --config ...

# Score a trained model against a held-out labelled corpus:
python -m epss.cross_distribution_eval \
    --checkpoint outputs/social_media/gpt/ALL/best_model.pt \
    --config     outputs/social_media/gpt/ALL/experiment_config.json \
    --eval-data  data/epss_mv_mistral_ALL/labeled_cves.json
```

### Dataset analysis

```bash
cd EPSS_TPG

# Per-(LLM, variant) graph + SEC overlay profile for the current baseline
python -m epss.per_llm_full_profile --output-dir datasets_info/Per_LLM_profile_new --workers 8

# Fast version: mean/median graph size only (reads pyg cache directly)
python -m epss.per_llm_graph_dims --output-dir datasets_info/Per_LLM_profile

# SEC_* edge firing-rate statistics on a single labelled corpus
python -m epss.security_edges_stats --labeled-cves data/epss_gpt_v2_ALL/labeled_cves.json --variant ALL

# Profile a source CSV (schema, missingness, EPSS distribution)
python analysis/analyze_dataset.py \
    --csv ../SummTPGVul/SummVul/Social_Media_Dataset/Data_Files/gpt_combined_summ.csv \
    --output-dir datasets_info/gpt_combined_summ

# Recompute metrics at multiple thresholds (no inference needed)
python -m epss.threshold_analysis --runs-root outputs --output outputs/threshold_analysis

# Cross-check labelled-record dtypes between two pipelines
python analysis/verify_features.py --a data/epss/labeled_cves.json \
                                   --b data/epss_sec4ai/labeled_cves.json

# Regenerate plots for an existing checkpoint without retraining
python analysis/generate_visualizations.py --run-dir outputs/social_media/gpt/ALL
```

### Maintenance

```bash
cd EPSS_TPG

# Free per-run pyg graph caches (5--40 GB per run); keeps best_model.pt and metrics
scripts/analysis/cleanup_old_data.sh --dry-run    # preview
scripts/analysis/cleanup_old_data.sh              # delete (asks for confirmation)
scripts/analysis/cleanup_old_data.sh --force      # skip the prompt
```


