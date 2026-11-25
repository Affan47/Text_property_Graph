# PS04 Full Results: SMP and URL-Summary Comparisons

## What was verified

The completed study contains 24 neural fits and three deterministic baselines.
Twelve neural fits were newly trained; twelve neural fits and three baselines
were reused unchanged. All use 2,546 training and 1,104 validation CVEs.

The user-run verification reloaded checkpoints and reproduced predictions.
The separate review verified all 27 fit artifact locks and identities,
recomputed the metrics and paired intervals from saved predictions, and
reproduced the combined report. It also checked epoch histories against the
selected validation checkpoints. No new inference or training was performed
by this review, and the test set remains closed.

The twelve new fits completed 454 epochs in total, stopping after 21--60
epochs. Their selected checkpoints are from epochs 6--45. No training batch
in the reviewed neural histories has zero total data-loss gradient. This
supports execution and reproducibility, not semantic quality or generalization.

## Main result

**Description + SMP with pooled text is the validation leader. TPG has not
demonstrated a reliable advantage over the matched pooled-text models.**

Every neural configuration below includes CVSS and reference count. MAE is
the mean absolute difference between predicted and recorded Day-60 EPSS;
smaller is better. Values average the three fixed seeds, not an ensemble.

| Text inputs | Pooled-text MAE | TPG MAE |
| --- | ---: | ---: |
| Description | 0.02279227 | 0.02299644 |
| Description + SMP | **0.02248772** | 0.02327050 |
| Description + both URL summaries | 0.02279122 | 0.02302962 |
| Description + both URL summaries + SMP | 0.02305140 | **0.02296873** |

The training-median reference has MAE 0.02360351. The best pooled-text model
has 4.73% lower MAE; the best graph model has 2.69% lower MAE. The pooled
leader's mean RMSE is 0.10511792 and mean R2 is only 0.0910. These are modest
results, not accurate reconstruction of the full score distribution.

## Does adding SMP help?

For pooled text without summaries, **yes descriptively**: DM improves over D
in all three seeds. MAE changes are -0.00022940, -0.00038156 and -0.00030268.
The mean change is -0.00030455, about a 1.34% reduction relative to D.
However, its paired CVE-bootstrap 95% interval is [-0.00103634, +0.00039654],
which includes zero. The evidence does not establish a reliable population gain.

For TPG without summaries, the mean effect is worse: +0.00027406. Two seeds
worsen and one improves. With summaries present, adding SMP has mixed seed
directions for both representations. There is no general SMP benefit that
holds across these model choices.

## Do URL summaries help?

Adding both summaries to description-only pooled text changes mean MAE by
just -0.00000105, with mixed seed directions. Adding them to description +
SMP pooled text worsens MAE in all three seeds, averaging +0.00056367.

For TPG, adding summaries to description-only slightly worsens the mean,
while adding summaries to description + SMP improves all three seeds by
an average 0.00030178. These different effects suggest that additional text
is not automatically useful to every representation and fusion model.
They do not establish that the summaries are factually wrong.

All eight source-addition intervals include zero. GitHub and non-GitHub
summaries were added together, so their separate effects remain unknown.

## Does the graph outperform pooled text?

TPG has higher mean MAE for D, DM and DGN, and slightly lower mean MAE for
DGNM. All four graph-versus-pooled paired intervals include zero. Therefore,
the scientifically defensible conclusion is **no demonstrated TPG superiority
in this study**, not that TPG can never help other tasks or architectures.

Two median-baseline comparisons have unadjusted intervals narrowly below zero:
DM TPG [-0.00086241, -0.00001224] and all-source pooled text
[-0.00127654, -0.00003756]. Those are not the lowest-MAE models. Interval width
depends on the distribution of paired errors, not only the mean. They are
conditional validation findings without multiple-testing correction, not
independent final-test confirmations.

## High EPSS scores remain difficult

There are 50 validation CVEs with actual EPSS >= 0.1 and 1,054 below 0.1.
For the best pooled DM configuration:

| Seed | MAE | Accuracy at EPSS 0.1 | Precision | Recall | F1 | High-score CVEs detected out of 50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 0.02254350 | 95.56% | 55.56% | 10% | 0.1695 | 5 |
| 29 | 0.02239182 | 95.47% | 50.00% | 8% | 0.1379 | 4 |
| 43 | 0.02252785 | 95.56% | 66.67% | 4% | 0.0755 | 2 |

Predicting every CVE below 0.1 already gives 95.47% accuracy. Thus the high
accuracy must not be presented as strong high-score detection. The categories
are EPSS scores, not exploitation labels. No threshold was tuned here.

Those 50 CVEs contribute 85.60% of the pooled DM model's absolute error.
CVE-2025-0108 contributes 74.63% of its total net error reduction versus the
median reference. Excluding this CVE only as an influence calculation still
leaves a small MAE advantage of 0.00028328; the official cohort is unchanged.

Three previously inspected PS03 cases illustrate both success and failure:

| CVE | Recorded Day-60 EPSS | Pooled DM seed 17 | Pooled DM seed 29 | Pooled DM seed 43 |
| --- | ---: | ---: | ---: | ---: |
| CVE-2025-0108 | 0.937040 | 0.940414 | 0.917269 | 0.912130 |
| CVE-2024-48248 | 0.934500 | 0.000735 | 0.001179 | 0.002748 |
| CVE-2025-1974 | 0.870260 | 0.062446 | 0.040896 | 0.006782 |

These are illustrative diagnostic cases, not a representative sample. The
second row shows a severe miss even though another similarly high-score CVE
is predicted well. The saved CSV also includes the matched TPG predictions.

## What to do next

Close this bounded development stage rather than starting another large
architecture or feature search. Before any test access:

1. Freeze all eight completed S1 configurations, all three seeds and the three
   baselines as the final comparison set. Do not choose the best seed.
2. Designate pooled DM as the validation-selected leader and all-source TPG
   as the lowest-MAE graph comparator. Report the other configurations too.
3. Freeze target eligibility, preprocessing, source masks, checkpoint use,
   metrics, fixed category thresholds and paired-comparison rules.
4. Implement test-data/cache preparation and one held-out evaluation under
   that frozen protocol. Reuse the saved training-fitted preprocessing and
   selected checkpoints; do not retune on test results.

No test-release code or test artifacts were created during this review. The
subsequent [PS05 implementation](../03_shared_evaluation/01_FROZEN_FINAL_TEST.md)
now provides a frozen, development-smoke-tested release procedure; actual
held-out evaluation remains a user-run step. If
the final test remains weak, report that result as a limitation of this
implementation/task rather than changing the experiment after seeing test data.

This is retrospective Day-60 estimation, not future exploitation prediction
or a causal study of SMP. Current text and CVSS may postdate the target.
Shared documents can create dependence between distinct CVEs. The CVE
bootstrap does not model that dependence or correct for repeated validation
use and multiple comparisons. Security-edge usefulness is not established
by these source comparisons; no positive edge-benefit claim should be made.

## Saved analysis

Original results remain in `05_results/04_consolidated_studies/01_primary_day60/source_comparison_v1/full/`.
The separate review is in `05_results/04_consolidated_studies/01_primary_day60/source_comparison_review_v1/`,
with `analysis.txt`, `analysis.json`, `real_prediction_examples.csv` and an
artifact hash lock. Paths are relative to `EPSS_TPG`.

Reproduce the saved-prediction analysis without training or test scoring:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/11_review_primary_source_comparison.py
```
