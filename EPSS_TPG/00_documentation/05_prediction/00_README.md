# Prediction Reading Guide

Start with [the code reading order](../01_start_here/02_CODE_READING_ORDER.md). The [old EPSS GNN report](../../99_archive/03_previous_dataset_work/01_documentation/05_prediction/02_EPSS_GNN_TECHNICAL_REPORT.md) remains historical background.

The new revision dataset is not directly compatible with the existing adapter. Read [the current dataset guide](../03_datasets/00_README.md) before preparing a training run. No trained checkpoint remains in the active results folders.

| Folder | Contents |
|---|---|
| `02_prediction/01_package/epss/` | Normalization, graph dataset, models, training and evaluation |
| `02_prediction/02_inference/` | Fresh-CVE inference CLI |
| `02_prediction/03_analysis/` | Standalone feature checks and plot generation |

From the project root, package commands retain their normal names:

```bash
python -m epss.run_pipeline --help
python -m epss.make_temporal_splits --help
python -m epss.test_only --help
```

`--hybrid` enables the tabular model branch. `--no-hybrid` selects the rule-only security frontend; these flags control different things. Check [the project findings](../01_start_here/01_PROJECT_SCHEMA_AND_FINDINGS.md) for label leakage, date semantics and split limitations before starting a new comparison.
