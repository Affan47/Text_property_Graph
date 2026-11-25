# PS02 Full Core Results

All 21 fits completed on the unchanged 2,546 training and 1,104 validation
CVEs. Checkpoints, prediction outputs, source masks, metrics and paired reports
were independently reproduced. The run and its caches were not edited. No
test scores were inspected and no new model was trained during this review.

## Main Result

The lowest average validation MAE comes from **description pooled text plus
CVSS/reference count**, not a TPG configuration. The difference is modest,
and paired intervals do not establish a reliable graph advantage. This is
a result of the present task and implementation, not proof that graph-based
text representations can never help.

The table averages each neural model's metrics across seeds 17, 29 and 43.
It does not evaluate an ensemble. Lower MAE/RMSE is better. Relative gain
compares MAE with the training-median baseline; negative means worse.

| Configuration | Mean MAE | Mean RMSE | Mean R2 | MAE gain over median |
| --- | ---: | ---: | ---: | ---: |
| Training mean | 0.052431 | 0.111014 | -0.0136 | -122.13% |
| Training median | 0.023604 | 0.112460 | -0.0402 | Reference |
| Structured ridge | 0.044725 | 0.106951 | 0.0592 | -89.48% |
| Description pooled text + structured | **0.022792** | **0.106405** | 0.0686 | 3.44% |
| Description TPG + structured | 0.022996 | 0.107473 | 0.0500 | 2.57% |
| All-source pooled text, structured off | 0.023159 | 0.111081 | -0.0149 | 1.88% |
| All-source pooled text + structured | 0.023051 | 0.109603 | 0.0117 | 2.34% |
| All-source TPG, structured off | 0.023369 | 0.112162 | -0.0347 | 0.99% |
| All-source TPG + structured | 0.022969 | 0.107239 | 0.0532 | 2.69% |

All-source means description, GitHub summaries, non-GitHub summaries and SMP.
Structured means the existing CVSS encoding plus NVD reference count. These
are current retrospective values, not verified historical Day-60 inputs.

## Results Across Seeds

| Neural configuration | Seed 17 MAE | Seed 29 MAE | Seed 43 MAE |
| --- | ---: | ---: | ---: |
| Description pooled text + structured | 0.022773 | 0.022773 | 0.022831 |
| Description TPG + structured | 0.022932 | 0.022927 | 0.023130 |
| All-source pooled text, structured off | 0.023398 | 0.022720 | 0.023358 |
| All-source pooled text + structured | 0.023232 | 0.022507 | 0.023415 |
| All-source TPG, structured off | 0.023389 | 0.023360 | 0.023357 |
| All-source TPG + structured | 0.023434 | 0.022806 | 0.022667 |

The single best seed is all-source pooled text + structured, seed 29. It is
not substituted for that configuration's three-seed mean. Selecting a lucky
seed would hide the variation the fixed-seed protocol was intended to measure.

## What the Matched Comparisons Say

Each difference below is left-model MAE minus right-model MAE. Negative
favors the left model. Intervals resample CVEs, using seed-averaged absolute
errors for each CVE. They are conditional development intervals, not final
test evidence or corrections for many comparisons.

| Left versus right | MAE difference | Paired 95% interval |
| --- | ---: | --- |
| Description TPG versus description pooled text, structured on | +0.000204 | [-0.000170, +0.000599] |
| All-source TPG versus all-source pooled text, structured on | -0.000083 | [-0.000945, +0.000627] |
| All-source TPG versus all-source pooled text, structured off | +0.000210 | [-0.000147, +0.000685] |
| All-source versus description pooled text, structured on | +0.000259 | [-0.000520, +0.001468] |
| All-source versus description TPG, structured on | -0.000028 | [-0.000795, +0.000945] |
| Description pooled text + structured versus median | -0.000811 | [-0.002644, +0.000304] |
| All-source TPG + structured versus median | -0.000635 | [-0.002070, +0.000365] |

These intervals include zero. Thus the mean rankings are useful descriptions,
but they do not establish reliable superiority for those comparisons.
Some other model-versus-median intervals exclude zero, including the very
small gain for all-source TPG without structured features. That is not
evidence that TPG beats its matched non-graph control or predicts larger
scores adequately.

Core adds SMP and both summary sources together. It cannot isolate which one
helps or hurts. Likewise, the structured-on/off comparison adds or removes
CVSS and reference count together; it does not isolate either structured block.

## Why a Small Overall Error Can Be Misleading

| Recorded Day-60 score | Training CVEs | Validation CVEs |
| --- | ---: | ---: |
| Below 0.01 | 1,953 | 996 |
| From 0.01 to below 0.1 | 461 | 58 |
| From 0.1 to below 0.5 | 56 | 30 |
| From 0.5 through 1 | 76 | 20 |

The training median is 0.00166; the validation median is 0.00071. Predicting
small values can therefore achieve a low aggregate MAE while missing many
larger scores. This behavior is consistent with a skewed target and the
MAE objective, which favors conditional medians. It does not prove that loss
choice alone causes the observed errors.

For description pooled text + structured, the 50 validation CVEs with scores
at least 0.1 have mean absolute error **0.4350** and contribute **86.4%** of
total absolute error. All-source TPG + structured gives a similar high-score
error of **0.4373** and an error share of **86.2%**.

The figure `target_and_error_distribution.png` in the separate review folder
shows the fraction of CVEs in each score range and each range's share of total
absolute error. Its error totals average per-CVE errors across seeds, not
predictions before calculating error.

## Real Example: Seed Variation and Concentrated Gains

For **CVE-2025-0108**, the stored Day-60 EPSS is **0.93704**:

| Model | Seed 17 prediction | Seed 29 prediction | Seed 43 prediction |
| --- | ---: | ---: | ---: |
| Description pooled text + structured | 0.84997 | 0.97377 | 0.66404 |
| Description TPG + structured | 0.92423 | 0.91274 | 0.90688 |
| All-source TPG + structured | 0.00660 | 0.92274 | 0.85553 |

This example was identified by its large error reduction relative to the
median baseline; it is not a representative sample or a new experiment.
It contributes about **89.7%** of the description pooled-text model's net
error reduction and **84.7%** of the all-source TPG model's net reduction.
For description TPG, its contribution exceeds the total net reduction because
errors on other CVEs offset some of the gain.

A diagnostic calculation excluding this one case leaves description TPG
worse than the median baseline on the remaining cases. This does not change
the official results: no CVE has been dropped. It illustrates why reporting
only overall average gains would overstate their breadth.

Separately, `real_prediction_examples.csv` selects examples at fixed recorded-
score ranks rather than selecting successful predictions. It preserves the
actual target date and all three seed predictions.

## Accuracy and F1 Are EPSS-Category Diagnostics

At the fixed threshold EPSS >= 0.1, there are 50 positive-category CVEs and
1,054 negative-category CVEs. Predicting every CVE below the threshold already
gives **95.47% accuracy**, with zero recall and zero F1.

| All-source TPG + structured | True positives / 50 | False positives | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seed 17 | 0 | 0 | 0.00 | 0.000 | 0.824 |
| Seed 29 | 4 | 6 | 0.08 | 0.133 | 0.791 |
| Seed 43 | 3 | 3 | 0.06 | 0.107 | 0.832 |

Ranking can be informative even when predicted scores rarely cross the fixed
threshold. ROC-AUC therefore does not contradict poor recall here. Thresholds
were not tuned after inspecting these results. These categories describe
EPSS scores, not confirmed exploitation or VulnCheck status.

## Did Training Run Correctly?

The 18 neural fits completed 609 epochs in total, with 21 to 43 epochs per
fit. Selected epochs range from 6 to 28. All fits stopped under the fixed
15-epoch patience rule, and no training batch recorded zero total data-loss
gradient. Checkpoint reloads reproduce the saved validation predictions.

The previously monitored all-source TPG + structured seed 29 improved after
epoch 19: its best epoch became 25, and it stopped at 40. The continuation
was therefore consistent with the early-stopping rule, not a stuck loop.

These checks support computational reproducibility, not a guarantee of no
bugs, perfect extraction or useful generalization. They also do not show
that each branch contributes effectively merely because its gradients exist.

## Recommended Next Step

Keep description pooled text + structured as the development reference, but
do not present its small average gain as a conclusive finding. Do not promote
TPG as superior on these results.

Before spending another large training budget, use the saved models to check
source/edge sensitivity and inspect the source relevance of high-error CVEs.
This can distinguish ineffective graph use from weak or diluted textual
evidence without retraining. The remaining registered source-removal
experiments can then isolate SMP and the two summary sources. Core alone
cannot justify claims about those individual sources.

Keep the test set closed until the final comparisons and selection rules are
frozen. Do not remove difficult CVEs, choose the best seed, or alter the
completed run to improve the reported result. No additional fit is launched
by this review.

## Reports and Reproduction

The original run is unchanged at:
`EPSS_TPG/05_results/04_consolidated_studies/01_primary_day60/training_v1/full/`.

The separate review is at:
`EPSS_TPG/05_results/04_consolidated_studies/01_primary_day60/core_review_v1/`.
It contains `analysis.txt`, `analysis.json`, `real_prediction_examples.csv`,
the target/error figure and its own artifact lock. The JSON includes every
fit's diagnostics, prediction distribution, threshold metrics and source weights.

From the repository root with `CodeBERTFusion` active:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/07_review_primary_core.py
```

This reproduces the saved-output analysis only. It does not retrain, generate
new predictions, inspect test data or alter the completed fit files.
