# Text Property Graph Project

Start with [the documentation reading order](00_documentation/README.md).
For the implementation, follow [the code reading order](00_documentation/01_start_here/02_CODE_READING_ORDER.md).

| Order | Folder | Contents |
|---|---|---|
| 00 | [Documentation](00_documentation/README.md) | Architecture, datasets, experiments, papers and maintenance notes |
| 01 | [TPG](00_documentation/02_tpg/00_README.md) | Graph engine, examples, document application, chatbot and workspace |
| 02 | [Prediction](00_documentation/05_prediction/00_README.md) | EPSS/KEV dataset processing, GNN training, inference and analysis |
| 03 | [Scripts](00_documentation/07_maintenance/02_SCRIPT_GUIDE.md) | Numbered training, evaluation and maintenance entry points |
| 04 | [Data](00_documentation/03_datasets/00_README.md) | Normalized records/caches and a link to the source repository |
| 05 | [Results](00_documentation/04_experiments/00_README.md) | Training, evaluation and dataset profiling artifacts |
| 06 | Runtime | Logs |
| 99 | Archive | TeX installer and compatibility paths for old documents |

The unnumbered names `tpg`, `epss`, `data`, `outputs`, `docs` and similar entries are compatibility symlinks, not duplicate copies. Browse the numbered folders; old commands and saved configuration paths can continue to use the aliases. Python package names and experiment IDs retain their original spelling.

From this directory:

```bash
python -m epss.run_pipeline --help
bash 03_scripts/01_training/run_social_media_retrain.sh --dry-run
```

See [the migration record](00_documentation/07_maintenance/01_LAYOUT_AND_COMPATIBILITY.md) for exact paths and verification details.

