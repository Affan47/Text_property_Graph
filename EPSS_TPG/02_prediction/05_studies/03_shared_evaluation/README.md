# Shared Evaluation and Final Test

The completed [final research report](03_FINAL_RESEARCH_REPORT.md) brings
together the architecture, protocol, results, figures and limitations in an
editable LaTeX manuscript and compiled PDF. It preserves the negative held-out
finding and reports every frozen configuration without changing model selection.

## Three distinct phases

1. **Training:** fit scaling, any learned vocabulary and model parameters on
   training CVEs only. Fixed neural seeds are 17, 29 and 43. The initial budget
   is 100 epochs, patience 15, batch size 8, learning rate 0.001 and weight
   decay 0.0001, using MAE. Final runner details must be frozen before fitting.
2. **Validation:** compare every registered variant on matched examples.
   Select using the declared MAE criterion and retain all seed-level results,
   including poorer models and alternative metrics.
3. **Test:** freeze the protocol, preprocessing, target/missingness rules,
   thresholds, variants and checkpoints. Then release the reserved test data
   for one final evaluation. Report all frozen comparisons without retuning.

The [PS05 final evaluator](01_FROZEN_FINAL_TEST.md) is implemented and the
primary-study protocol was frozen before release. All 27 saved fits have now
been evaluated on 651 test CVEs and verified. Read the
[final results and limitations](02_FINAL_RESULTS.md); this test set is now
observed and must not be used for further tuning. The older CS01 design command still has no
test access, and its guards have not been weakened.

## Existing chronological roles

We preserve the complete CVE role manifest instead of repeatedly resplitting
after viewing validation results. Its boundaries are validation beginning
2025-03-02, test beginning 2025-06-01, and test ending 2025-12-02, with an
observation cutoff of 2026-09-29. Existing exclusion rules remain unchanged.
The primary study retains the parent Day-30/Day-60 date-admissibility rule
for cohort continuity; Day 30 is not an EPSS input in that study.

The manifest contains 2,870 training, 1,104 validation, 651 test and 1,343
excluded CVEs. Per-task eligibility can reduce those counts. PS05 confirmed
that all 651 reserved test CVEs have observed eligible Day-60 targets. No CVE
ID crosses roles. Similar wording or shared
advisories between different CVEs remain a documented overlap concern; they
do not prove that the vulnerabilities are the same. Do not claim that this
identity-based split eliminates all related-evidence leakage.

Current-text/CVSS inputs in the primary study are explicitly retrospective.
Chronological CVE roles alone do not make those features historical. Models
using them cannot be presented as prospective Day-60 forecasts.

## Metrics and figures

| Result | What it tells us | Planned figure |
| --- | --- | --- |
| MAE | Average absolute EPSS error; main selection metric | Matched source-ablation heatmap |
| RMSE and upper error percentiles | Whether a few large errors dominate | Residual histogram and error by score range |
| Median absolute error and signed bias | Typical error and systematic over/underestimation | Residual versus actual score |
| R-squared and Spearman correlation | Explained variation and rank association, where defined | Actual versus predicted hexbin |
| Paired baseline differences and CVE-bootstrap intervals | Improvement on the same CVEs, with sampling uncertainty | Paired-effect plot |
| Lead-specific and anchor-specific errors | Where continuous forecasts succeed or fail | Lead curves and anchor/lead error/skill heatmaps |
| Fixed example trajectories | How predictions differ from the recorded sequence | Actual/predicted daily trajectories |
| Threshold diagnostics at EPSS 0.01 and 0.1 | Agreement on explicit score categories | Confusion matrices and ROC/PR curves |
| Training/validation losses and clipping | Learning behavior and output-boundary effects | Learning curves and clipping summaries |

Accuracy, precision, recall, F1, balanced accuracy, ROC-AUC and average
precision are secondary **EPSS-category** diagnostics. They are not
classification metrics for an unthresholded regression target, and they do
not establish observed exploitation. Undefined metrics return null with a
reason rather than a fabricated zero.

The primary development runners now save regression/category metrics,
bootstrap comparisons, prediction exports and figures. PS04 full development
is complete; read [the results](../01_primary_tpg/10_SOURCE_COMPARISON_RESULTS.md).
PS05 supplies the separate inference-only test reports. CS01 itself remains
a design/coverage command, and the consolidated continuous-forecasting
runner is still a separate unfinished study.

## Required saved outputs from future runners

- Exact input identities, eligible CVE IDs, source masks, target dates and
  experiment configuration, plus train-only preprocessing state.
- Every seed's checkpoint, epoch history and reproduction checks.
- Actual and predicted values, raw and clipped outputs, target masks, and
  baseline predictions on exactly the same evaluated records.
- Source availability, graph sizes, edge vocabulary and missingness counts.
- Numerical metrics in JSON and plain-text interpretation, with figures and
  separate training, validation and final-test reports.
- Learning, residual and clipping diagnostics sufficient to check whether
  improvements are CVE-specific rather than a constant global adjustment.

A result that does not beat its matched baseline remains a valid negative
finding. Attention weights alone are not feature importance. Correlation
with SMP activity is not evidence that SMP caused EPSS to change.
