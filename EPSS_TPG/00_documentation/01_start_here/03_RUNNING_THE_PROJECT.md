# Running the Project

Activate your Python environment and install the local TPG packages once per
environment. From `EPSS_TPG/`:

```bash
python -m pip install --no-deps --no-build-isolation -e ./01_tpg
```

This uses the existing dependencies without downloading or upgrading them and
requires setuptools 64 or newer. It registers `tpg`, `tpg_app`, and `tpg_chatbot`
directly from their numbered folders. Keep this checkout in place; repeat the
installation after moving it. This is a checkout installation, not a standalone
wheel deployment. NLP/application dependencies and spaCy models must already be
available in the environment.

Inspect the entry points without starting an experiment:

```bash
python -m epss.run_pipeline --help
python -m tpg_app.cli --help
python -m tpg_chatbot.ingest --help
python 01_tpg/02_examples/01_scripts/experiment.py --help
```

TPG defaults now use absolute paths under `01_tpg/`, independent of the shell's
working directory. Example inputs live in `02_examples/04_inputs/`; generated
graphs live in `02_examples/03_generated/`. Databases, uploads and generated
chatbot stores live in `05_workspace/`. Explicit CLI paths and `TPG_DB` /
`TPG_UPLOADS` environment overrides remain supported.

See [the transition guide](../07_maintenance/04_DATASET_TRANSITION.md) for source-data setup. The training adapter is not yet compatible with the revision dataset, so no current batch-training command is claimed here.

The [old practical guide](../../99_archive/03_previous_dataset_work/01_documentation/01_start_here/03_RUNNING_THE_PROJECT.md) is retained for historical dependency and command context. Its dataset names, recipes and checkpoint paths are retired.
