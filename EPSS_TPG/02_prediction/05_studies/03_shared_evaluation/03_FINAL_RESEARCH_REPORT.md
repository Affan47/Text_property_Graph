# Final Research Report

## Current Consolidation: 5 November

The master PDF now adds TRV01-TRV05, CF01 Stages 1-3 and CF02 while preserving
the original final regression/classification tables and gallery assets. New
chapters are development evidence, not new test evaluations. No temporal-text
training is authorized by the CF02 provenance gate.

Use the CPU consolidation builder rather than regenerating the legacy assets:

```bash
conda activate CodeBERTFusion
export PYTHONPATH="/tmp/tpg_pdf_review_deps${PYTHONPATH:+:$PYTHONPATH}"
python EPSS_TPG/00_documentation/06_papers/03_build_files/primary_tpg_final_research_report/consolidation_v1/build_report.py --stage build
python EPSS_TPG/00_documentation/06_papers/03_build_files/primary_tpg_final_research_report/consolidation_v1/build_report.py --stage verify
```

The builder directory contains its prerequisites, eight regression tests,
source/publication hash locks and the preserved prior report. No experiment
runner, checkpoint or historical result is rewritten.

## Original Final Evaluations

The earlier static-score approach, completed retrospective Day-60 study and
completed VulnCheck classification study are presented together in the
[final PDF](../../../00_documentation/06_papers/02_pdf/primary_tpg_final_research_report.pdf)
and its [editable LaTeX source](../../../00_documentation/06_papers/01_sources/primary_tpg_final_research_report.tex).

The 2 November update covers the dataset, TPG architecture, source-level fusion, structured inputs,
chronological CVE roles, validation diagnostics, all frozen configurations,
held-out metrics, real prediction examples and limitations. The final conclusion
is negative: neither the validation-selected pooled model nor the designated
all-source TPG model improves on the training-median test reference. The report
does not select a different winner after inspecting test results. The added
classification results likewise do not establish an all-source TPG advantage
over pooled text on the reused evaluation cohort. A real saved graph for
CVE-2023-41591 explains nodes, edges, embeddings, message passing and extraction
limitations. All 38 final classifier figures accompany the original regression figures.

Generated tables, figures, `report_evidence.json`, `report_evidence.txt` and PDF
layout checks are stored under
`00_documentation/06_papers/03_build_files/primary_tpg_final_research_report/`,
relative to `EPSS_TPG`. Original results and checkpoints remain unchanged.

The following commands document the original, pre-consolidation asset pipeline.
Do not run them over the consolidated publication; use the builder above instead:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/14_build_primary_final_report.py
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/15_extend_final_research_report.py
PDFLATEX=/home/ayounas/texlive/2026/bin/x86_64-linux/pdflatex bash EPSS_TPG/03_scripts/03_maintenance/build_paper.sh primary_tpg_final_research_report
```

Optional PDF checks, in an environment with PyMuPDF and Pillow:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/14_build_primary_final_report.py --check-pdf
```

The builder checks saved fit-report hashes, matches all test CVEs and targets,
and recomputes MAE/RMSE from stored predictions. It does not train a model,
produce new predictions, alter the test cohort or change the frozen selection.
The extension additionally checks final classifier artifact hashes and recalculates
AP, ROC-AUC and confusion matrices for all 36 predictors. It verifies the real
graph against its original cache hashes and text against the raw dataset.
The editable chapter includes are alongside the main LaTeX file; generated
`updated_report_evidence.json`, `.txt` and `real_cve_example.json` are in the
existing report build folder.
