# Running the Project

From `EPSS_TPG/`, inspect the entry points without starting an experiment:

```bash
python -m epss.run_pipeline --help
python -m tpg_app.cli --help
```

See [the transition guide](../07_maintenance/04_DATASET_TRANSITION.md) for source-data setup. The training adapter is not yet compatible with the revision dataset, so no current batch-training command is claimed here.

The [old practical guide](../../99_archive/03_previous_dataset_work/01_documentation/01_start_here/03_RUNNING_THE_PROJECT.md) is retained for historical dependency and command context. Its dataset names, recipes and checkpoint paths are retired.
