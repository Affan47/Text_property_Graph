# VK08 Findings: Thresholds Help a Little; Source Quality Still Matters

## 1. Scope and Checks

This review covers the completed validation analysis of 1,104 CVEs: 74 listed
and 1,030 not listed in the saved VulnCheck snapshot. The original training
cohort remains 2,870 CVEs. There was no new training or test scoring.
Both original full `--verify-only` commands and the separate review
reproduction passed during this analysis.

The review checks all 24 selected thresholds, recalculates 99,360 saved
intervention predictions across six models and 15 conditions, and compares
the three predefined wording subsets. All six original all-source probability
vectors reproduce exactly. Actual description/summary/post excerpts in the
error examples were checked against the accepted predictor export. Original
models, thresholds, source documents and experiment reports remain unchanged.

## 2. Did Selecting the Threshold Improve F1?

F1 balances precision (how many flagged CVEs are listed) and recall (how many
listed CVEs are found). Each model has one selected cutoff, not a different
cutoff for every CVE. All values below are means of three separately trained
seeds, not ensemble scores.

| Text inputs and representation, always with structured features | F1 at original 0.5 cutoff | Highest F1 on the same validation data | Cross-fold threshold F1 |
|---|---:|---:|---:|
| Description, pooled text | 27.47% | 32.69% | 29.86% |
| Description, TPG | 29.69% | 33.69% | 29.65% |
| Description + summaries, pooled text | 28.63% | 31.24% | 26.95% |
| Description + summaries, TPG | 28.57% | 32.32% | 26.68% |
| Description + posts, pooled text | 33.31% | 42.95% | 40.54% |
| Description + posts, TPG | 39.91% | 44.51% | 40.63% |
| All sources, pooled text | 43.60% | 44.81% | 40.09% |
| All sources, TPG | **45.85%** | **48.23%** | **45.76%** |

The middle column selects and measures a cutoff on the same labels, so it is
optimistic. The last column classifies each validation fold using a cutoff
selected on the other four folds. This checks threshold sensitivity only:
these CVEs already influenced checkpoint selection and earlier development.
It is not independent cross-validation of model training or final testing.

For all-source TPG, the apparent gain is 2.39 percentage points, but cross-fold
F1 is essentially the original result. Threshold choice is therefore not a
complete explanation for the remaining errors. Some other models benefit
more, notably pooled description-plus-posts. That does not establish a single
best cutoff that can be transferred between models.

## 3. Highest Individual F1 and Its Cost

| All-source TPG seed | Selected cutoff | Original F1 | Selected-validation F1 | Cross-fold F1 | Listed CVEs found, out of 74 | Unlisted CVEs flagged, out of 1,030 |
|---|---:|---:|---:|---:|---:|---:|
| 17 | 0.3731103 | 45.45% | 47.20% | 44.85% | 38 | 49 |
| 29 | 0.5724091 | 45.03% | 47.83% | 46.72% | 33 | 31 |
| 43 | 0.6832602 | 47.06% | **49.67%** | 45.71% | 38 | 41 |

The highest observed threshold-selected validation F1 is **49.67%**, from seed
43. Its precision is 48.10% and recall 51.35%; 36 listed CVEs are still missed.
Compared with the original cutoff, false flags fall from 56 to 41, but listed
CVEs found also fall from 40 to 38. The higher F1 reflects that tradeoff, not
uniform improvement in every metric.

The rounded cutoffs here are for reading only. Reproduction uses the full
values bound to checkpoint hashes in `thresholds.json`. No final model or seed
is automatically chosen. Across the five threshold folds, seed 17 cutoffs
range from 0.1769 to 0.3943, seed 29 from 0.5724 to 0.5733, and seed 43 from
0.6833 to 0.8553. Cutoff stability differs by checkpoint.

## 4. Which Sources Do the Saved Models Use?

AP measures ranking over score thresholds. It does not change just because
the decision cutoff changes. This table alters inputs to already trained
all-source models; it does not retrain a new model without that source.

| Input change | Pooled-text mean AP | TPG mean AP | Interpretation |
|---|---:|---:|---|
| No change | 0.3781 | 0.4149 | Original all-source ranking |
| Exchange post vectors between CVEs with the same source availability | 0.1268 | 0.1213 | Strong dependence on which posts belong to which CVE |
| Remove posts and reweight remaining sources | 0.1678 | 0.1679 | Remaining inputs do not preserve the learned all-source behavior |
| Exchange description vectors | 0.2928 | 0.2983 | Description identity also matters |
| Zero the encoded structured features | 0.3351 | 0.3206 | Structured information also affects the saved model |
| Exchange both summary vectors | 0.3719 | 0.3962 | Smaller sensitivity to CVE-specific summaries |
| Remove both summary sources | 0.3704 | 0.4098 | Small ranking change; decision metrics still change |
| Remove GitHub summaries only | 0.3772 | 0.4168 | No consistent AP loss across seeds |
| Remove non-GitHub summaries only | 0.3726 | 0.4113 | Small mean AP loss |

Swapping posts, swapping descriptions and zeroing structured representations
each reduce AP in all three seeds of both representations. Swapping summaries
also reduces AP in all seeds, but by a much smaller amount. In the TPG model,
post swapping lowers F1 at the frozen selected thresholds from 48.23% to
15.32%. Thresholds were not reselected to rescue any altered condition.

There are 1,102 non-self donors and two unchanged singleton cases. Donors
preserve the complete source-mask pattern, so the post-swap effect cannot be
explained simply by adding or removing a source. However, the swapped content
can be semantically implausible, and zero/remove conditions can be outside the
training distribution. These are reliance checks, not causal effects or
proof that the model understands exploitation.

Do not compare removal from the all-source model with a separately trained
description-only model as if they were identical experiments. The all-source
model learned its parameters with posts present.

## 5. Is This Only Shared Wording?

The existing audit identifies 86 validation CVEs with recorded wording shared
with training; 21 are listed. The other 1,018 include 53 listed CVEs. These
are distinct vulnerabilities; shared wording is not a duplicate-CVE claim.

On the 1,018-CVE subset, all-source TPG AP falls from **0.3182 to 0.1062** when
posts are exchanged, and to **0.1486** when posts are removed. The source
dependence is therefore not confined to the recorded exact-wording subset.
This does not exclude common advisories, paraphrases, platform templates,
vendor patterns or indirect target disclosures.

## 6. Attention and Source Availability

Mean text-attention weights across all 1,104 CVEs and three TPG seeds are:

| Text source | Mean share of text attention |
|---|---:|
| Description | 17.06% |
| GitHub summary | 0.61% |
| Non-GitHub summary | 2.30% |
| Posts and related text | 80.02% |

These weights apply only to text fusion, not to the separate structured branch.
They are not percentages of causal importance. Missing sources receive zero
weight. Among CVEs where the source exists, mean weights are 3.35% for GitHub
summaries, 3.43% for non-GitHub summaries and 80.61% for posts. Posts exist for
1,096 CVEs, compared with 201 with GitHub summaries and 741 with non-GitHub
summaries. Availability alone does not establish information quality.

## 7. Real Source-Quality Problems in Error Examples

These observations concern the accepted saved text. They do not require an
assumption that the remote pages still contain the same text.

| CVE | Saved description | Observed issue in other accepted text |
|---|---|---|
| CVE-2025-21253 | Microsoft Edge for iOS/Android spoofing | Non-GitHub summary describes Windows Print Spooler remote code execution and names CVE-2024-12345 instead. The post is a broad Microsoft Patch Tuesday bulletin. |
| CVE-2025-26630 | Microsoft Office Access use-after-free allowing local code execution | Non-GitHub summary describes Windows Print Spooler remote code execution and contains the unresolved identifier CVE-2023-XXXXX. |
| CVE-2025-2748 | Kentico Xperience stored XSS through file upload | Non-GitHub summary says its excerpt contains only the heading "Hotfixes" and lacks technical detail. That is a source-insufficiency response, not a useful vulnerability summary. |
| CVE-2024-55963 | Appsmith access-control failure allowing a server restart | GitHub summary describes the access-control omission, while an accepted related post is headed as an AppSmith RCE exploit. The text needs CVE-specific relevance review; this observation alone does not resolve whether the exploit is related. |

CVE-2025-21253 is unlisted in this snapshot but receives TPG scores 0.99783,
0.94410 and 0.999997. Removing summaries leaves all three false flags in place.
Swapping its post representation removes the false flag in all three seeds,
but that replaces evidence with another CVE's content; it is not a repair.
CVE-2025-26630 also remains a false flag in all seeds after summary removal.
The mismatched summaries are genuine quality concerns, but are not proven to
be the sole cause of these predictions.

CVE-2024-55963 and CVE-2025-2748 are listed, yet all three all-source TPG models
miss them at their selected thresholds. More available text does not ensure
that the model uses the relevant evidence successfully. A false flag means
unlisted in the snapshot, not proof that exploitation never occurred.

## 8. Decision

The bounded development comparison is complete. Do not launch another broad
training sweep or search more thresholds merely to improve the reported F1.
Report the original, selected-validation and cross-fold threshold results
together. AP remains 0.4149 for the unchanged all-source TPG scores.

The concrete issue now is evidence relevance: some summary content is
inconsistent with its CVE, some summaries are scraping-failure responses, and
some posts discuss many vulnerabilities rather than the individual CVE. Any
correction policy should be reviewed independently of whether it improves
these validation scores, versioned, and assessed under a new evaluation plan.
Do not silently edit these completed experiments or delete their hard cases.

For the present models, the next decision is to freeze the final reporting and
evaluation protocol, including model selection and checkpoint-specific cutoffs.
The earlier EPSS test cohort is not wholly untouched merely because the target
has changed to VulnCheck. A new external or future snapshot offers a stronger
generalization assessment. No new annotation, training or test release is
implemented by this review.

## 9. Reproducible Review

The separate review is in
`05_results/05_vulncheck_study/17_threshold_source_review_v1/`:
`analysis.txt`, `analysis.json`, `threshold_comparison.csv`,
`source_interventions.csv`, `threshold_and_source_results.png`, and provenance
locks. Original outputs remain in folders 15 and 16.

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/15_review_threshold_and_source_results.py --verify-only
```
