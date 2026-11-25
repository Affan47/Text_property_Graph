# CT05: Completed Training-Only Assessment

## Status and Meaning

Both full datasets completed all 18 neural fits and their linear controls.
Fresh-process checkpoint, coefficient, prediction and report replay passed.
The separate review recalculated all 104 saved metric rows per dataset and
checked role dates, evidence-component separation, labels and best-checkpoint
selection. It did not train, generate new predictions, or score official
validation/test CVEs. Original experiments and artifact locks are unchanged.

**D** means standalone description text. **DM** means description combined with
accepted post text. Both models also contain the same structured CVSS/reference
features. These are pooled SecureBERT vector models, **not GNNs**.

An out-of-fold (OOF) prediction is made for a CVE by a model fitted on earlier,
evidence-separated CVEs from the original training set. Nested selection chooses
the checkpoint; assessment labels do not. The first chronological warmup cannot
be assessed without future-data training. The older OOF cohort contains 1,531
CVEs and 87 high scores; Reddit contains 14,548 CVEs and 413 high scores.
Here, a high score means Day-60 EPSS >= 0.1, not confirmed exploitation.

## Overall Error

Numbers below average the three seed metrics, not predictions across seeds.
Lower RMSE and MAE are better. RMSE gives larger errors more weight.

| Dataset | Model | RMSE | MAE | R-squared |
|---|---|---:|---:|---:|
| Older | Optimization-mean reference | 0.166331 | 0.061636 | -0.009920 |
| Older | Optimization-median reference | 0.168012 | 0.043227 | -0.030441 |
| Older | D | 0.148769 | 0.052606 | 0.191968 |
| Older | DM | 0.146654 | 0.058498 | 0.214849 |
| Reddit | Optimization-mean reference | 0.060879 | 0.022632 | -0.017896 |
| Reddit | Optimization-median reference | 0.062109 | 0.014171 | -0.059461 |
| Reddit | D | 0.058139 | 0.017796 | 0.071650 |
| Reddit | DM | 0.057614 | 0.016738 | 0.087849 |

Both neural models have lower overall RMSE point estimates than the constant
mean. Neither beats the constant median on overall MAE. This is an objective
tradeoff, not permission to replace the frozen primary metric after results.
No new baseline significance test is implied by these point estimates.

| Dataset | DM minus D RMSE | 95% CVE interval | 95% evidence-component interval |
|---|---:|---:|---:|
| Older | -0.002115 | [-0.007237, +0.003147] | [-0.006992, +0.005670] |
| Reddit | -0.000524 | [-0.001128, +0.000073] | [-0.001579, +0.000405] |

The point reductions are about 1.42% and 0.90%. **Neither dataset establishes
an overall DM advantage under either saved interval procedure.** DM improves
overall RMSE in two of three seeds on each dataset. The intervals use 20,000
resamples at seed 6029 and condition on fitted models; they do not include
refitting uncertainty or correct repeated research use.

## Reddit: Posts Present versus Posts Absent

| Assessment group | CVEs | High-score CVEs | D RMSE | DM RMSE | DM minus D |
|---|---:|---:|---:|---:|---:|
| Accepted posts present | 1,649 | 84 | 0.105331 | 0.099060 | -0.006271 |
| No accepted posts | 12,899 | 329 | 0.048926 | 0.049877 | +0.000950 |

The post-present group improves in all three seeds. Its component-resampled
interval is [-0.010651, -0.002264]. Without posts, the mean RMSE worsens, with
component interval [+0.000397, +0.001472]; seed 29 accounts for most of that harm.
These are prespecified but exploratory subgroup findings, not a new primary win.

The total mean-seed squared-error saving is 0.858056. Post-present CVEs save
2.113334, while post-absent CVEs lose 1.255279. Thus a benefit in one group is
partly offset in another. This is an additive **squared-error** accounting, not
an allocation of RMSE or evidence that posts caused a score change.

For post-absent CVEs, D and DM predictor vectors and source flags were checked
and are identical. Their predictions can still differ because separately
training the two models changes weights and checkpoint selection. These changes
cannot be attributed to feeding nonexistent posts. Within the post-present
group, the comparison still combines text/context, availability and learned-weight
effects; a pure content mechanism has not been isolated.

## Chronological Folds

| Dataset | Fold | CVEs | Scores >= 0.1 | D RMSE | DM RMSE |
|---|---:|---:|---:|---:|---:|
| Older | 1 | 512 | 63 | 0.206808 | 0.202073 |
| Older | 2 | 508 | 22 | 0.144384 | 0.143554 |
| Older | 3 | 511 | 2 | 0.052146 | 0.054863 |
| Reddit | 1 | 4,832 | 324 | 0.077500 | 0.078521 |
| Reddit | 2 | 4,781 | 48 | 0.043294 | 0.041036 |
| Reddit | 3 | 4,935 | 41 | 0.047572 | 0.045997 |

Reddit fold 2 favors DM under both interval types. Fold 3 favors DM under the
CVE interval, but its component interval crosses zero. Fold 1's mean worsens;
its component interval also crosses zero. Only two high-score cases occur in
older fold 3, so its tail-ranking results are especially fragile. No fold was
removed or redrawn.

Outcome and source support vary substantially across time. For example, Reddit
fold 1 has only 6 high-score/post-present optimization CVEs and 2 in selection,
versus 41 in assessment. Older fold 1 selection has 5 high scores among 237 CVEs,
versus 63 among 512 assessment CVEs. These differences characterize the task;
they do not prove a causal explanation for the model errors.

## Ranking and Large Errors

| Dataset | Model | AUROC for score >= 0.1 | Average precision | High-score hits among top 50 |
|---|---|---:|---:|---:|
| Older | D | 0.864032 | 0.394906 | 26.33 |
| Older | DM | 0.885028 | 0.443746 | 29.33 |
| Reddit | D | 0.703232 | 0.148412 | 26.67 |
| Reddit | DM | 0.708327 | 0.152349 | 24.33 |

The older ranking point estimates improve with DM. Reddit's changes are small
and mixed: top-50 hits decline despite slightly higher AUROC and AP. Across
all 413 Reddit high-score cases, RMSE worsens from 0.315558 to 0.318995. The
overall squared-error improvement comes from scores below 0.1, offsetting a
loss on high scores. This is not evidence of a general rare-high-score fix.

In the older cohort, high-score RMSE improves from 0.579811 to 0.528865, but
low-score RMSE worsens from 0.056260 to 0.076341. Its top three positive
contributors supply about 96.01% of net squared-error saving. Reddit's top
three supply about 56.13%. All records remain in the official analysis.

| Dataset | CVE | Actual Day-60 score | D mean prediction | DM mean prediction |
|---|---|---:|---:|---:|
| Older | CVE-2023-3519 | 0.911990 | 0.079183 | 0.321504 |
| Older | CVE-2023-38035 | 0.970820 | 0.126093 | 0.320599 |
| Older | CVE-2024-55956 | 0.967520 | 0.362832 | 0.143456 |
| Reddit | CVE-2023-26360 | 0.874250 | 0.039157 | 0.185251 |
| Reddit | CVE-2022-21907 | 0.712520 | 0.163528 | 0.018951 |

Mean predictions above illustrate three seeds. Official metrics and error
contributions are calculated per seed before averaging, not from these displayed
means. Several high-score CVEs remain heavily underestimated in both models.

## Controls and Selection

The nested-selection ridge uses alpha 1000 in every dataset/fold/source arm,
the upper endpoint of the frozen grid. Regularization improves those text-ridge
controls relative to alpha 1, but they still trail the neural models in overall
RMSE. Do not extend the grid after inspecting these assessment outcomes.

An early best epoch is not an untrained model. For example, Reddit fold 1 DM
seed 29 selects epoch 1, already 951 optimization steps. That fit worsens fold 1
RMSE compared with D. It remains in the results; assessment cannot replace the
selection decision. Fit histories and update counts are included in the review.

Constants are fitted separately per fold. Pooled OOF rankings can reflect
differences in their fold-level constants, so AUROC below 0.5 for the pooled
constant is not evidence of within-fold ranking. Constant top-k ties use CVE ID.
Rankings across learned fold models also compare different score scales.

## Decision

Keep CT05 complete and frozen. It finds a repeatable **exploratory Reddit
post-present benefit**, but not an established overall source-addition benefit
or TPG/GNN advantage. These are retrospective text/CVSS estimates, not historical
forecasts, and previously studied training records are not independent test data.

The next useful gate is a **bounded, zero-training diagnostic of the saved
Reddit DM models**, using already cached standalone-D inputs to distinguish
content replacement from changing the M flag. Replacing DM with standalone D
also changes embedding context and must be described accordingly. Do not shuffle
whole DM vectors and call that a post-only permutation: it would change description
information too. Do not yet launch a broad architecture search, re-split the data,
discard seed 29, select the favorable subgroup as the primary result, or open test.

## Reproducible Outputs and Commands

Each dataset now has `full_review_v1/` beside its unchanged `full/` directory,
under `05_results/08_fresh_studies/11_training_only_pooled_mse_v1/`.
It contains TXT/JSON findings, metric/interval tables, role support, fit histories,
ridge choices, all CVE error contributions, group savings, a figure and hash locks.
The result folders are `01_reddit` and `02_previous_multisource`.

Run these individually. They use CPU only and do not train:

```bash
cd /home/ayounas/Text_property_Graph
```
```bash
conda activate CodeBERTFusion
```
```bash
python EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/11_review_training_only_pooled.py --dataset both --verify-only
```
```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/07_fresh_studies -p 'test_training_only_review.py'
```

For the longer original checkpoint and 20,000-resample report replay:

```bash
python EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/10_run_training_only_pooled.py --dataset both --stage train --scale full --jobs 3 --verify-only
```
