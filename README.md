# Text Property Graph

Open [Start Here](EPSS_TPG/00_documentation/README.md) for the documentation reading order, then follow [the code reading order](EPSS_TPG/00_documentation/01_start_here/02_CODE_READING_ORDER.md).

The main project is organized as follows:

```text
EPSS_TPG/
  00_documentation/  Start here: architecture, datasets, experiments, papers
  01_tpg/            Graph engine, examples, applications and workspace
  02_prediction/     EPSS/KEV models, training, inference and analysis
  03_scripts/        Training, evaluation and maintenance scripts
  04_data/           Records, graph caches and source-repository link
  05_results/        Training results, evaluation results and profiles
  06_runtime/        Runtime logs
  99_archive/        Installer and legacy document paths
```

`SummTPGVul/` is the source-dataset Git submodule, now selected from `tpg-paper-revision`. You can also browse it through `EPSS_TPG/04_data/02_source_repository/`. See [the current dataset guide](EPSS_TPG/00_documentation/03_datasets/00_README.md).

The previous local datasets and saved runs were retired. Their analyses are separated into `EPSS_TPG/99_archive/03_previous_dataset_work/`. The replacement dataset requires a new adapter and target definition before training; the old experiment commands are not compatible.

The unnumbered entries inside `EPSS_TPG/` are compatibility links, not extra copies. They preserve Python imports, old commands and saved paths. See [the layout guide](EPSS_TPG/00_documentation/07_maintenance/01_LAYOUT_AND_COMPATIBILITY.md).

```bash
cd EPSS_TPG
python -m epss.run_pipeline --help
python 03_scripts/03_maintenance/verify_current_layout.py --expect-empty
```

For setup and publication commands, read [the dataset transition guide](EPSS_TPG/00_documentation/07_maintenance/04_DATASET_TRANSITION.md).
