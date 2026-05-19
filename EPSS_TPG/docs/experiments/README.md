# Experiment Documentation

Experiment reports remain in [../../Datasets_information](../../Datasets_information/README.md)
because that folder also contains the batch scripts, run logs, and generated
statistics used by the experiments.

## Main Reports

| Report | What it contains |
|---|---|
| [Dataset feature dictionary](DATASET_FEATURE_DICTIONARY.md) | Explanation of every raw, converted, tabular, and TPG-derived feature used in the datasets |
| [Overall analysis](../../Datasets_information/OVERALL_ANALYSIS.md) | Full synthesis across ablations and clean Summary-in-TPG runs |
| [Dataset index](../../Datasets_information/README.md) | Per-dataset report index and reproduction notes |
| [Summary-in-TPG clean matrix](../../Datasets_information/Summary_in_TPG_ablation/README.md) | 16-run production-honest matrix and commands |
| [Summary-in-TPG results](../../Datasets_information/Summary_in_TPG_ablation/results.md) | Verified metrics from the clean matrix |
| [TPG-only ablation](../../Datasets_information/TPG_ablation/README.md) | TPG branch contribution without tabular fusion |
| [CVSS ablation](../../Datasets_information/CVSS_ablation/README.md) | CVSS feature contribution analysis |

## Dataset Reports

| Dataset | Report |
|---|---|
| GPT summaries | [gpt_combined_summ](../../Datasets_information/gpt_combined_summ/README.md) |
| Gemma summaries | [gemma_combined_summ](../../Datasets_information/gemma_combined_summ/README.md) |
| Llama summaries | [final_dataset_with_llama_summ](../../Datasets_information/final_dataset_with_llama_summ/README.md) |
| DeepSeek summaries | [deepseek_combined_summ](../../Datasets_information/deepseek_combined_summ/README.md) |

## Current Focus

Recent summary experiments added these run types through flags, without changing
the existing 16-run batch script:

- `--summary-only-tpg`
- `--two-view-tpg`
- `--add-source-labels`
- `--summary-pooling-node`
- `--graph-diagnostics`
