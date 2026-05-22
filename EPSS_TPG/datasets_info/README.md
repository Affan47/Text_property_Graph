# Dataset And Experiment Reports

This folder keeps the experiment reports, dataset profiles, batch scripts, run
logs, and generated statistics in one place. The name is kept as
`Datasets_information` because existing commands and scripts already refer to
this path.

For a cleaner top-level documentation entry point, see
[../docs/experiments/README.md](../docs/experiments/README.md).

## Folder Groups

| Group | Folders |
|---|---|
| Dataset reports | `gpt_combined_summ`, `gemma_combined_summ`, `final_dataset_with_llama_summ`, `deepseek_combined_summ` |
| Ablation studies | `TPG_ablation`, `CVSS_ablation`, `Summary_in_TPG_ablation` |
| Overall synthesis | `OVERALL_ANALYSIS.md` |

Each dataset-report subdirectory documents one incoming dataset:
- `README.md` — features, distributions, critical findings, schema vs `csv_adapter` expectations, reproduction commands
- `profile.json` — machine-readable profile produced by `epss/prepare_dataset.py`
- `profile.txt`  — human-readable summary of the profile

## Dataset Index

| Dataset | Source | Rows | Unique base CVEs | Status |
|---|---|---:|---:|---|
| [gpt_combined_summ](gpt_combined_summ/README.md) | `Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/gpt_combined_summ.csv` | 9,218 | 5,692 | ⚠ 8-run ablation completed; PR-AUC stays ≥ 0.97 even at "max-clean" — signal source not yet identified ([details](gpt_combined_summ/ablation_results.md)) |
| [gemma_combined_summ](gemma_combined_summ/README.md) | `Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/gemma_combined_summ.csv` | 9,218 | 5,692 | ✅ 8-run ablation completed; cross-dataset A/B with gpt_combined definitively rejects LLM-summary leakage hypothesis ([details](gemma_combined_summ/ablation_results.md)) |
| [final_dataset_with_llama_summ](final_dataset_with_llama_summ/README.md) | `Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/final_dataset_with_llama_summ.csv` | 9,218 | 5,692 | ✅ 8-run ablation completed; 3-way GPT vs Gemma vs Llama comparison **triple-rejects** LLM-summary leakage hypothesis (Run D spread = 0.0002 across 3 datasets) ([details](final_dataset_with_llama_summ/ablation_results.md)) |
| [deepseek_combined_summ](deepseek_combined_summ/README.md) | `Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/deepseek_combined_summ.csv` | 9,218 | 5,692 | ✅ 8-run ablation completed; 4-way GPT vs Gemma vs Llama vs DeepSeek comparison **quadruple-rejects** LLM-summary leakage hypothesis (Run D spread = 0.0010 across 4 datasets) ([details](deepseek_combined_summ/ablation_results.md)) |

## Overall Synthesis

[**OVERALL_ANALYSIS.md**](OVERALL_ANALYSIS.md) — Full 32-run synthesis (4 datasets × 8 ablation configurations). Headline: all four ablatable hypotheses (CVE duplication, LLM summary text, tabular proxies, imputed labels) are now rejected on all four datasets. The leakage source must be in features identical across the datasets — most likely the NVD `description` text, CVSS components, or sample-selection bias.

## Ablation Studies

| Study | Status | Description |
|---|---|---|
| [TPG_ablation](TPG_ablation/README.md) | ✅ 7 runs completed — see [results](TPG_ablation/tpg_ablation_results.md) | **Major finding:** TPG-only PR-AUC = 0.81 on full data, 0.34 on max-clean — vs hybrid 0.998 / 0.974. The tabular branch (dominated by CVSS) carries 0.19-0.63 PR-AUC of the model's signal. CVSS is now the prime suspect for the inflated metric across the 32 prior runs. |
| [CVSS_ablation](CVSS_ablation/README.md) | 4-run plan ready | Tests whether the CVSS components are the missing ~0.19 PR-AUC the TPG ablation localised. New `--drop-cvss` flag in `prepare_dataset.py` removes all 10 CVSS columns. |
| [Summary_in_TPG_ablation](Summary_in_TPG_ablation/README.md) | ✅ 16 clean runs completed — see [results](Summary_in_TPG_ablation/results.md) | Production-honest matrix with `--no-epss-feature` across 4 datasets × 4 variants. Post-TPG-fix rerun on 2026-05-04: baseline mean PR-AUC = 0.8333; summary mean Δ = −0.0120; security-edge mean Δ = +0.0057; both features mean Δ = −0.0068. No feature gain exceeds bootstrap noise. |

## Adding a new dataset

```bash
cd /home/ayounas/Text_property_Graph/EPSS_TPG
python -m epss.prepare_dataset \
    --input  /path/to/<new>.csv \
    --output-dir data/epss_<tag>
```

This generates `<stem>_profile.json` and `<stem>_profile.txt` under the output dir. Copy them into a new subdirectory of `Datasets_information/<stem>/` and write a `README.md` summarising features and any critical findings.
