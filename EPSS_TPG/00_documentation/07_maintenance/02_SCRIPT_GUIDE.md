# Script Guide

1. `03_scripts/03_maintenance/setup_revision_data.sh` reproduces the pinned sparse dataset checkout and downloads the three selected LFS files.
2. `03_scripts/03_maintenance/verify_current_layout.py` checks the revision, data, aliases and retired experiment locations.
3. `03_scripts/03_maintenance/build_paper.sh` builds general TPG manuscripts still in the active paper directory.

Old training/evaluation wrappers and cleanup tools are in [the archived workflows](../../99_archive/03_previous_dataset_work/04_workflows/). They are reference material, not runnable recipes for the new dataset.

The model CLI remains available through `python -m epss.run_pipeline --help`. A task-specific adapter and target definition are needed before training on the replacement data.
