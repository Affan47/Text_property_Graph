# Frozen Final Evaluation

VK10 executes the existing VK09 protocol. It is not a training experiment.
The original protocol, models, thresholds, development caches and reports remain
unchanged. The user authorized final execution after reviewing development.

## Task and Interpretation

The target is `in_vckev`: listed or not listed in the saved VulnCheck catalogue.
It is not EPSS regression, CISA KEV membership, future inclusion or an exploitation
probability. The exact 651 existing test-role CVEs remain in the comparison.
They were previously exposed during EPSS research, so this is a **reused-cohort
retrospective evaluation**, not a fresh independent test.

The primary model family is all-source TPG plus structured inputs; its matched
comparator is all-source pooled text plus the same structured inputs. All three
seeds remain reportable. All 36 frozen predictors are evaluated, including the
description, summary, post, logistic, edge-free and prevalence controls.

## What the Evaluator Does

1. Checks the VK09 freeze and all pinned dependencies.
2. Separates accepted predictors from boolean labels. It uses the unchanged
   source policy and saved training-fitted 29-column structured encoder.
3. References graph chunks only when their text, source, offsets, CVE role and
   processing identities match. Missing chunks use the frozen builder.
4. Loads each authenticated checkpoint and predicts without gradient updates.
   Original batch boundaries, pooling and per-batch sigmoid are preserved.
5. Reports every predictor at 0.5, and the 24 source-neural fits at their existing
   validation-selected cutoffs. There is no new threshold search or calibration.
6. Reports all seeds, class counts, source availability, shared-wording subsets,
   predictions, errors, ten-bin reliability tables and paired bootstrap intervals.

Missing labels, invalid text/graphs, changed identities or incomplete predictions
abort the run. They do not silently remove a CVE. Failures are recorded separately
under `19_final_evaluation_v1/failure_logs/`.

## Commands

From the repository root with `CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study -p 'test_vulncheck_final_evaluation_v1.py'
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/17_run_final_evaluation.py --stage prepare --scale smoke
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/17_run_final_evaluation.py --stage cache --scale smoke --workers 4
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/17_run_final_evaluation.py --stage evaluate --scale smoke
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/17_run_final_evaluation.py --stage evaluate --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/17_run_final_evaluation.py --stage prepare --scale full --release-test
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/17_run_final_evaluation.py --stage cache --scale full --release-test --workers 4
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/17_run_final_evaluation.py --stage evaluate --scale full --release-test
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/17_run_final_evaluation.py --stage evaluate --scale full --release-test --verify-only
```

Use `--verify-only` for an already completed stage; it does not overwrite it.
Use `--resume` only for interrupted cache/evaluation stages. Completed fit outputs
are reused only after their locks and evaluation identities pass verification.
Do not delete completed results to run different test-driven settings.

## Artifacts

All paths below are relative to `EPSS_TPG`:

- `04_data/01_records_and_graphs/vulncheck_final_evaluation_v1/`: separate inputs,
  labels, filtering decisions, structured quality flags and shared-wording audit.
- `01_tpg/05_workspace/vulncheck_final_evaluation_v1/`: evaluation document/chunk
  index, authenticated references, any newly built graphs and security audit.
- `05_results/05_vulncheck_study/19_final_evaluation_v1/`: smoke parity results,
  full predictions, metrics, report, figures, provenance and locks.
- `18_final_protocol_v1/`: original frozen development report and protocol.
  Its statement that evaluation was not yet performed remains a historical record;
  the new evaluation report records the later execution.

Read `full/final_report.md` first, then `summary.json` and `metrics.csv`.
`predictions.csv` contains actual versus predicted listing for every CVE/model.
`evaluation_figures.pdf` shows every predictor, not just a selected winner.
The separate completed EPSS regression report remains
`00_documentation/06_papers/02_pdf/primary_tpg_final_research_report.pdf`.

## How This Relates to the Earlier Studies

These are different questions, not successive measurements of one target:

| Study | Inputs and target | What its result means |
|---|---|---|
| EPSS forecasting | Earlier daily EPSS values predict later EPSS values or a sequence | Temporal regression; compare with persistence and training-only corrections |
| Completed primary Day-60 study | Current text and structured features estimate the EPSS value 60 days after CVE publication | Retrospective scalar regression; not proof of historical forecasting |
| Current VulnCheck study | Current text and structured features classify the saved `in_vckev` label | Retrospective binary classification; not an EPSS score or future exploitation target |

The completed Day-60 regression report records test MAE 0.017413276 for its
validation-selected pooled description/post model and 0.017076295 for the
all-source TPG comparator, versus 0.013457419 for the training-median reference.
Lower MAE is better. Neither selected neural comparison improves on that
reference; the report does not substitute a different model after seeing test
results. This conclusion is not altered by the later classification study.

The reusable outcome is a source-aware graph/data pipeline with typed relations,
frozen embeddings, provenance, matched non-graph controls and reproducible
predictions. These are engineering achievements. Whether they improve prediction
must be judged separately for each target and evaluation setting.
