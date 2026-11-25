# VK07 Results: Summaries, Posts and Description Graphs

## 1. What Was Evaluated

The full development run contains 18 new fits and six unchanged description-only
references. All use 2,870 training CVEs and 1,104 validation CVEs, of which 74 are
listed in the saved VulnCheck snapshot. The target is that listing, not an EPSS
threshold, future exploitation, or CISA KEV status. Structured inputs remain the
same 29 columns. No test scoring or threshold tuning was performed.

The separate review recalculates all 24 reports from 26,496 saved validation
probabilities, checks labels and CVE order, verifies selected epochs and stopping
rules, and reproduces the original mean metrics. The original completed run is
unchanged. Checkpoint inference is verified separately by the original CLI.
That full checkpoint/D-reference/prediction verification also passed. No model
was retrained during either check.

## 2. Overall Results

Each number is the arithmetic mean of three separately trained seeds (17, 29,
43), not an ensemble. Precision, recall and F1 use the fixed threshold of 0.5.
AP means average precision: it measures how well listed CVEs are ranked ahead
of unlisted CVEs across score thresholds. Higher AP, ROC-AUC and F1 are better.

| Text used, always with structured features | Representation | AP | ROC-AUC | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| Description | Pooled tokens | 0.3015 | 0.7909 | 31.25% | 24.77% | 27.47% |
| Description | Security TPG | 0.2992 | 0.8010 | 29.07% | 32.43% | 29.69% |
| Description + URL summaries | Pooled tokens | 0.2750 | 0.7878 | 28.19% | 29.73% | 28.63% |
| Description + URL summaries | Security TPG | 0.2776 | 0.7863 | 31.39% | 33.78% | 28.57% |
| Description + posts | Pooled tokens | 0.3479 | 0.8276 | 23.27% | 63.06% | 33.31% |
| Description + posts | Security TPG | 0.3731 | 0.8357 | 35.03% | 49.55% | 39.91% |
| Description + summaries + posts | Pooled tokens | 0.3781 | 0.8395 | 40.00% | 48.65% | 43.60% |
| Description + summaries + posts | Security TPG | **0.4149** | **0.8439** | **43.19%** | **49.10%** | **45.85%** |

The all-source TPG has the highest mean AP and F1 in this comparison. Relative
to description-only TPG, mean AP rises by 0.1157 and F1 by 16.15 percentage
points. All-source AP improves over the corresponding description-only model
in every seed for both representations. This is a development improvement, not
an independent-test result or a claim that TPG is universally superior.

The balanced description-TF-IDF/structured logistic reference has AP 0.2587 and
F1 33.16%. All-source models improve both mean measures, but use additional
text. That comparison does not isolate the effect of graph relationships.

## 3. What Each Source Adds

**Summaries without posts do not improve mean AP.** Adding the two URL-summary
groups to description reduces mean AP for both pooled and graph models. Each
representation improves in only one seed. This does not prove that summaries
contain no signal: these models must learn to use them alongside description.
GitHub and non-GitHub sources were added together, so their individual effects
are not separately identified here.

**Posts improve the main comparison, with important qualifications.** Adding
posts improves pooled AP in all three seeds and TPG AP in two. Pooled recall
rises strongly, but so do false alarms: the three post-plus-description pooled
fits flag 255, 98 and 154 unlisted CVEs. High recall alone is not a clean win.

**Summaries help the TPG comparison when posts are already included.** All-source
TPG improves over description-plus-posts TPG in all three seeds: AP differences
are +0.0325, +0.0523 and +0.0405. Pooled text improves in two seeds, with a small
decline in seed 43. This supports a context-dependent benefit, not a causal
claim about summaries or a measured interaction mechanism.

## 4. Does TPG Beat Matched Pooled Text?

| Seed | All-source pooled AP | All-source TPG AP | TPG minus pooled AP | All-source pooled F1 | All-source TPG F1 |
|---|---:|---:|---:|---:|---:|
| 17 | 0.3926 | 0.4268 | +0.0342 | 45.86% | 45.45% |
| 29 | 0.3872 | 0.3866 | -0.0006 | 44.74% | 45.03% |
| 43 | 0.3546 | 0.4313 | +0.0767 | 40.21% | 47.06% |

Mean AP favors all-source TPG by 0.0368, but not in every seed. Mean F1 favors
TPG by 2.25 percentage points, again not in every seed. Seed 17 has higher TPG
AP but slightly lower F1 and ROC-AUC than pooled text. There is no uniform
dominance across metrics. Three-seed standard deviations are not confidence
intervals, and this review performs no significance test.

The pooled control and TPG use different node populations and encoder paths;
added-source models also introduce source attention compared with D alone.
These are matched system comparisons, not isolated proof of security-edge value.
The earlier description-only edge-free control does not answer that question
for the all-source model.

## 5. How Many CVEs Are Found or Missed?

| All-source TPG seed | Listed CVEs found, out of 74 | Listed CVEs missed | Unlisted CVEs falsely flagged, out of 1,030 | Accuracy |
|---|---:|---:|---:|---:|
| 17 | 35 | 39 | 45 | 92.39% |
| 29 | 34 | 40 | 43 | 92.48% |
| 43 | 40 | 34 | 56 | 91.85% |

The model still misses roughly half the listed CVEs and makes more false than
true positive flags in each seed. A false flag means unlisted in this snapshot,
not proof that the CVE has never been exploited.

Always predicting unlisted would achieve **93.30% accuracy**, but zero recall
and F1. Thus the model's mean 92.24% accuracy is not the main evidence of useful
detection. AP, recall, precision and the actual error counts are more informative
for this imbalanced cohort.

## 6. Shared Wording Changes the Interpretation

The existing normalized-text audit identifies 86 validation CVEs with some
wording shared with training: seven description groups and 57 post groups.
These are distinct CVEs; the audit does not merge them or assert that their
underlying vulnerabilities are the same. The 86 CVEs include 21 of the 74
listed validation CVEs. No original split, prediction or model was changed.

| Validation subset | CVEs | Listed CVEs | Description TPG AP | Description + posts TPG AP | All-source pooled AP | All-source TPG AP |
|---|---:|---:|---:|---:|---:|---:|
| All validation CVEs | 1,104 | 74 | 0.2992 | 0.3731 | 0.3781 | 0.4149 |
| No recorded wording shared with training | 1,018 | 53 | 0.2713 | 0.2703 | 0.2757 | 0.3182 |
| Some recorded wording shared with training | 86 | 21 | 0.5580 | 0.6520 | 0.7262 | 0.6978 |

On the 1,018-CVE subset, the mean posts-only TPG gain over description disappears.
All-source TPG still improves mean AP over description TPG by 0.0469 and over
matched all-source pooled text by 0.0425. In that subset, TPG beats matched
all-source pooled AP in all three seeds, but beats description TPG in only two.

These subset results qualify, rather than erase, the main result. The subsets
have very different listed fractions: 53/1,018 versus 21/86. AP depends on class
composition, and these are not matched random groups. Their difference cannot
be assigned entirely to memorization. Absence of exact normalized shared wording
does not establish independence of advisories, vendors, paraphrases or evidence.

## 7. Probability Quality and Real Examples

All-source TPG mean Brier score is 0.06273, better than description TPG (0.08203)
and the training-prevalence constant (0.06663). Its mean log loss is 0.27376,
better than description TPG (0.36005) but slightly worse than that constant
(0.26725). The unweighted combined logistic model has lower Brier (0.05756) and
log loss (0.21694), despite lower AP. The new scores should not be presented as
calibrated exploitation probabilities.

The following examples use seed 17, fixed for illustration rather than selected
as the strongest seed. Values are the saved classifier scores, not EPSS values.

| CVE | Listed in saved snapshot? | Description TPG score | All-source TPG score | Interpretation at 0.5 |
|---|---|---:|---:|---|
| CVE-2024-13365 | Yes | 0.07133 | 0.99871 | Found by the all-source model; missed by D |
| CVE-2025-21590 | Yes | 0.00858 | 0.99862 | Found by the all-source model; missed by D |
| CVE-2025-0160 | No | 0.07121 | 0.99827 | Highly confident false flag in this snapshot |
| CVE-2024-55963 | Yes | 0.000000215 | 0.00001921 | Missed by both models despite available text |

Seed 17 finds 25 listed CVEs that D misses, but also loses nine listed CVEs
that D found: the net increase is 16, from 19 to 35. These are comparisons of
separately trained models, not demonstrations that a particular post caused
the change. The TXT/JSON review preserves exact scores, accepted source masks
and description excerpts for these and additional high-confidence examples.

## 8. Training and Source Quality

All 18 new runs stop with AP below their selected checkpoint. In 16, final
training loss is also below training loss at selection. This is consistent
with overfitting after the useful checkpoint; simply adding epochs is not
supported by these curves. The all-source TPG selected epochs are 10, 3 and
31, with stopping at 25, 18 and 46. The saved best checkpoints are evaluated.

The full cache contains 19,624 chunks: 3,974 descriptions, 787 GitHub summaries,
2,684 non-GitHub summaries and 12,179 post chunks. Their pooled representations
remain separated by document and source. The 17,231 reused chunks were not
copied; the remaining 2,393 were built under the frozen processing policy.

Training post graphs contain 9,422,962 nodes across 2,808 CVEs, compared with
851,146 nodes across 1,096 validation CVEs. Document lengths and source content
therefore differ substantially across roles. This is an observation, not proof
that graph size explains the scores.

Security-relation candidates are dominated by product/version links. For example,
validation descriptions have 441 version links and five cause links; validation
posts have 1,635 version links, 34 severity links, eight classification links and
two threat links. Only 119 of the 1,096 validation CVEs with posts have such
security edges in those posts. A larger score gain cannot therefore be described
as demonstrated learning of exploitation relationships. Linguistic edges, node
representations, source fusion and structured inputs also contribute to the system.

The disclosure filter remains important but incomplete. It removes all available
posts for 41/376 listed training CVEs versus 19/2,494 unlisted training CVEs; in
validation the counts are 0/74 and 8/1,030. Availability and evidence content can
encode dataset-specific shortcuts. The masks-only logistic baseline has weak
AP (0.06588), but that alone cannot exclude such shortcuts in a nonlinear model.

## 9. Conclusion and Next Step

This is a positive **development result for the all-source system**, with better
mean ranking and F1 than the description-only systems. The evidence for TPG over
matched pooled text is smaller and seed-dependent. The gain is reduced when
recorded shared wording is excluded, while the posts-only TPG gain does not
persist on that subset. Historical availability, filtering and prior validation
reuse prevent a strong forecasting or independent-generalization claim.

The next concrete step should be one bounded diagnostic of the saved all-source
models: remove or shuffle post and summary representations while retaining CVEs,
masks, structured inputs and the fixed threshold; inspect changes on both wording
subsets; and review the source documents behind confident errors. This can test
whether the gains depend on CVE-specific content without another training sweep.
Representation removal is an out-of-distribution sensitivity test, not causal
proof. Do not retune from these checks or open the test set automatically.

After that evidence review, explicitly freeze any final evaluation plan. The
previously observed test is not an untouched new holdout merely because the
target changed from EPSS to VulnCheck. No new diagnostic or training is launched
by this results review.

## 10. Files and Reproduction

Original results remain in `05_results/05_vulncheck_study/13_multisource_ablation_v1/full/`.
The separate `14_multisource_results_review_v1/` folder contains `analysis.txt`,
`analysis.json`, `per_fit_metrics.csv` and `source_comparison_by_seed.png`.
The figure links the same seed across separately trained source variants;
its lines are neither a time sequence nor confidence intervals.

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/13_review_multisource_results.py --verify-only
```
