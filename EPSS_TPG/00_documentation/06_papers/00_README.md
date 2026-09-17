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
| 4 | [Dataset features](01_sources/dataset_features.tex) |
| 5 | [Training/testing context](01_sources/training_testing_context.tex) |
| 6 | [Experiment results](01_sources/experiment_results_35_runs.tex) |

Additional sources include architecture figures, CVE graph figures and `tpg_chapter_grounded.tex`. Existing PDFs were moved without regeneration, so their build time may precede source changes.

To build a paper from the project root:

```bash
bash 03_scripts/03_maintenance/build_paper.sh dataset_features
```

The build script keeps temporary LaTeX output in a separate build directory and updates only the requested PDF after a successful compilation.

