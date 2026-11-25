# PS05 Final Held-Out Results

## Scope and verification

All **651 reserved test CVEs** have eligible observed Day-60 EPSS targets.
None was removed for a missing target. Each neural variant was evaluated on
the same CVEs using its three frozen checkpoints, alongside three baselines:
27 fits in total, with no model fitting or retuning during evaluation.

The completed evaluator independently reproduced its predictions and reports.
The separate interpretation verified the frozen protocol, prepared-data and
graph-cache locks, all fit provenance, target alignment and unchanged-state
diagnostics, and recomputed every reported metric and paired comparison.
It did not rerun inference or select new models. The NVML warning did not
prevent evaluation; these models run on CPU.

Test inputs include descriptions and SMP for all 651 CVEs, GitHub summaries
for 205 and non-GitHub summaries for 448. The cache contains 2,365 source
documents and 2,652 newly built chunks under the unchanged extraction and
embedding policy. Target dates range from 2025-07-01 through 2025-11-15, but
633 of 651 fall in July 2025. This is not broad coverage of the later months.

## Main conclusion

**The validation-selected models do not generalize better than the simple
training-median reference on this held-out cohort.** This is a negative
result for the present retrospective Day-60 estimation implementation, not
proof that all graph representations or social-media data are ineffective.

The validation-selected leader remains description + SMP pooled text. The
designated graph comparator remains all-source TPG. They are not replaced
after examining test outcomes.

| Configuration | Validation MAE | Test MAE | Test MAE versus median reference |
| --- | ---: | ---: | ---: |
| Frozen training median | 0.02360351 | 0.01345742 | Reference |
| Description, pooled text | 0.02279227 | 0.01352576 | 0.51% worse |
| Description, TPG | 0.02299644 | 0.01336559 | 0.68% lower |
| Description + SMP, pooled text: selected leader | 0.02248772 | 0.01741328 | **29.40% worse** |
| Description + SMP, TPG | 0.02327050 | 0.01427174 | 6.05% worse |
| Description + both URL summaries, pooled text | 0.02279122 | 0.01379090 | 2.48% worse |
| Description + both URL summaries, TPG | 0.02302962 | 0.01327620 | 1.35% lower |
| All sources, pooled text | 0.02305140 | 0.01557981 | 15.77% worse |
| All sources, TPG: designated comparator | 0.02296873 | 0.01707629 | **26.89% worse** |

All neural rows include CVSS and reference count. MAE averages absolute EPSS
error across CVEs and then the three fixed seeds. Lower is better. The
percentage column compares test MAE with the same frozen training-median
predictor evaluated on test, not with the test-set median fitted after release.

The training-mean and structured-ridge baselines have test MAE 0.04394112
and 0.04943072. They also fail to improve on the median reference.

The lowest observed test MAE belongs to description + summaries TPG, but
its difference from the median is -0.00018122 with conditional 95% interval
[-0.00034618, +0.00001604]. That interval includes zero. **No neural model's
paired interval versus the median lies entirely below zero.** Do not promote
this nominal test leader to a new selected model.

The selected pooled leader is worse than the median by +0.00395586, interval
[+0.00231180, +0.00669874]. The designated graph comparator is worse by
+0.00361888, interval [+0.00303744, +0.00435613]. Their direct comparison
includes zero: neither is established as better than the other.

## Adding SMP does not help on this test cohort

| Matched change | Change in test MAE | Conditional 95% interval |
| --- | ---: | --- |
| Add SMP to description, pooled text | +0.00388751 | [+0.00243367, +0.00634380] |
| Add SMP to description + summaries, pooled text | +0.00178891 | [+0.00107413, +0.00229557] |
| Add SMP to description, TPG | +0.00090615 | [+0.00075005, +0.00104992] |
| Add SMP to description + summaries, TPG | +0.00380010 | [+0.00323594, +0.00460285] |

Positive means adding SMP increases error. All four intervals are above zero
under the existing CVE-bootstrap procedure. These are unadjusted, conditional
comparisons of the trained systems, not proof that SMP causes harm or that
the posts are false. This reverses the validation tendency for pooled DM.

URL-summary effects are mixed. Adding them to pooled DM reduces test error,
but adding them to TPG DM increases error. TPG beats pooled text for DM and
DGN under the unadjusted comparisons, yet loses for all-source inputs. This
does not establish general graph superiority or a useful contribution from
security edges specifically.

## High scores are still missed

The test cohort contains 609 scores below 0.01 and 21 scores at least 0.1.
Predicting every CVE below 0.1 already gives **96.77% accuracy**.

| Frozen model | Seed | High-score CVEs detected out of 21 | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Selected description + SMP pooled text | 17 | 3 | 14.29% | 0.2222 |
| Same | 29 | 0 | 0% | 0 |
| Same | 43 | 0 | 0% | 0 |
| Designated all-source TPG | 17 | 0 | 0% | 0 |
| Same | 29 | 1 | 4.76% | 0.0909 |
| Same | 43 | 1 | 4.76% | 0.0769 |

These categories concern recorded EPSS >= 0.1, not observed exploitation.
High accuracy therefore must not be presented as successful high-score
detection. Undefined precision for no predicted positives remains null in
the saved metrics, not a fabricated zero.

The following are the three largest recorded test scores, selected by actual
score rather than by how good or bad a model looks:

| CVE | Actual Day-60 EPSS | Selected pooled model, seed 17 | Seed 29 | Seed 43 |
| --- | ---: | ---: | ---: | ---: |
| CVE-2025-47916 | 0.874400 | 0.005883 | 0.012629 | 0.007004 |
| CVE-2025-4427 | 0.832630 | 0.027435 | 0.002362 | 0.007417 |
| CVE-2025-48827 | 0.754480 | 0.043956 | 0.011550 | 0.004660 |

All configurations have negative mean test R2. That means their average
squared-error performance is worse than the constant test-mean reference
used in the R2 definition. The test mean is a descriptive reference, not a
deployable baseline fitted on training.

## Why smaller raw test MAE is not an improvement

The selected pooled model's raw MAE drops from 0.02248772 on validation to
0.01741328 on test, but the median baseline drops even more, from 0.02360351
to 0.01345742. Its relative advantage on validation becomes a disadvantage
on test. The cohorts have different score distributions: the test median
actual score is 0.00051, with 93.55% of scores below 0.01.

Distribution change, weak rare-high-score coverage, source relevance and
seed-dependent fusion are possible contributing factors. These results do
not identify their individual causes. Reproducible checkpoints also do not
prove the absence of every modeling or extraction flaw.

## Evidence overlap and limitations

There are 166 source-document occurrences across 79 test CVEs whose normalized
text also appears in development: 165 SMP documents and one description.
Of those occurrences, 123 match training and 106 match validation; the counts
overlap because a document can match both. Shared text does not imply that
the CVEs are duplicates. It does limit claims of evidence-independent testing.

Current text and CVSS remain retrospective and may postdate the target. The
EPSS-disclosure quarantine is not proof of complete leakage prevention. The
bootstrap treats CVEs as sampling units, does not model dependence from
shared text and does not correct for multiple comparisons. Validation has
already informed development. State these limitations beside any claims.

## What is complete and what follows

The frozen primary experiment is now complete. Preserve all checkpoints,
predictions and reports. Do not change source filters, target eligibility,
thresholds, seeds or architecture to improve this test result.

The appropriate next deliverable is the final research report: architecture,
data policy, validation selection, full test table and negative generalization
finding. Any later modeling change is a new study, with a new independently
justified evaluation plan; this test set is now observed and cannot be called
unseen again. The separate continuous-forecasting proposal is not evaluated
by these primary-study results.

## Saved files

Original results: `05_results/04_consolidated_studies/01_primary_day60/final_evaluation_v1/full/`.
Separate interpretation: `05_results/04_consolidated_studies/01_primary_day60/final_evaluation_review_v1/`.
The review contains `analysis.txt`, `analysis.json`, `real_prediction_examples.csv`,
`validation_test_comparison.png` and an artifact lock. Paths are relative to `EPSS_TPG`.

Reproduce the saved-result interpretation without new inference or fitting:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/13_review_primary_final_evaluation.py
```
