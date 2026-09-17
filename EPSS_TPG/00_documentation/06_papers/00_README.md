# Paper Reading Guide

LaTeX sources, compiled PDFs and build files are separate:

- `01_sources/`: editable `.tex` manuscripts and figures.
- `02_pdf/`: existing compiled PDFs.
- `03_build_files/`: auxiliary files from earlier compilations.

Suggested reading order:

| Read | Source |
|---|---|
| 1 | [TPG chapter](01_sources/tpg_chapter.tex) |
| 2 | [Security frontend rationale](01_sources/security_frontend_rationale.tex) |
| 3 | [Multiview formulas](01_sources/multiview_tpg_formula.tex) |
| 4 | [Historical dataset features](../../99_archive/03_previous_dataset_work/01_documentation/06_papers/01_sources/dataset_features.tex) |
| 5 | [Training/testing context](01_sources/training_testing_context.tex) |
| 6 | [Historical experiment results](../../99_archive/03_previous_dataset_work/01_documentation/06_papers/01_sources/experiment_results_35_runs.tex) |

Additional sources include architecture figures, CVE graph figures and `tpg_chapter_grounded.tex`. Existing PDFs were moved without regeneration, so their build time may precede source changes.

To build a paper from the project root:

```bash
bash 03_scripts/03_maintenance/build_paper.sh tpg_chapter
```

The build script keeps temporary LaTeX output in a separate build directory and updates only the requested PDF after a successful compilation.

Dataset and result manuscripts were archived because they describe retired inputs and runs. They have not been rewritten to imply results for the new revision dataset. General TPG chapters and architecture figures remain here; any historical experimental discussion in them should be read in that context.
