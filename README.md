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

`SummTPGVul/` is the source-dataset Git submodule. Its registered name and upstream directory structure are preserved. You can also browse it through `EPSS_TPG/04_data/02_source_repository/`.

The unnumbered entries inside `EPSS_TPG/` are compatibility links, not extra copies. They preserve Python imports, old commands and saved paths. See [the layout guide](EPSS_TPG/00_documentation/07_maintenance/01_LAYOUT_AND_COMPATIBILITY.md).

```bash
cd EPSS_TPG
python -m epss.run_pipeline --help
bash 03_scripts/01_training/run_social_media_retrain.sh --dry-run
```

The earlier repository guide is preserved in [the documentation archive](EPSS_TPG/00_documentation/01_start_here/04_ORIGINAL_REPOSITORY_GUIDE.md).
