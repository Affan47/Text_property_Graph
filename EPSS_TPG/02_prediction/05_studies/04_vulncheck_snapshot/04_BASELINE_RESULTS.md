# VK03: Full Baseline Results

## 1. Verified Scope

Full baseline training is complete: 2,870 training CVEs (376 listed) and 1,104
validation CVEs (74 listed). All eight logistic checkpoints and the prevalence
reference reproduced. Metrics were also independently recalculated from the
saved predictions. No retraining, threshold adjustment or test scoring occurred.

The target is listing in the saved VulnCheck snapshot. A label of zero means not
listed in that snapshot, not never exploited. These are retrospective development
results, not an independent final evaluation. This experiment uses descriptions
and CVSS/reference-count features, not TPG/GNN, URL-summary text, post text or EPSS.

## 2. All Fixed Comparisons

| Model | AP | ROC-AUC | F1 at 0.5 | Precision at 0.5 | Recall at 0.5 | Log loss |
|---|---:|---:|---:|---:|---:|---:|
| Prevalence constant | 0.0670 | 0.5000 | 0 | Undefined | 0 | 0.2672 |
| Structured, ordinary weights | 0.1596 | 0.6911 | 0.0482 | 0.2222 | 0.0270 | 0.2419 |
| Description, ordinary weights | 0.1807 | 0.7315 | 0.0267 | 1.0000 | 0.0135 | 0.2466 |
| Description + structured, ordinary weights | 0.2494 | 0.7726 | 0.0706 | 0.2727 | 0.0405 | 0.2169 |
| Source masks, ordinary weights | 0.0659 | 0.4941 | 0 | 0 | 0 | 0.2708 |
| Structured, balanced weights | 0.1676 | 0.7079 | 0.2277 | 0.1474 | 0.5000 | 0.4803 |
| Description, balanced weights | 0.1691 | 0.7354 | 0.2162 | 0.1441 | 0.4324 | 0.4828 |
| Description + structured, balanced weights | 0.2587 | 0.7826 | 0.3316 | 0.2743 | 0.4189 | 0.3272 |
| Source masks, balanced weights | 0.0659 | 0.4941 | 0.1162 | 0.0644 | 0.5946 | 0.6613 |

AP (average precision) measures ranking across score thresholds. Higher AP and
ROC-AUC are better; lower log loss is better. F1, precision and recall here use
the frozen 0.5 threshold. The constant predicts no listed CVEs, so its precision
is undefined. The ordinary description model's precision of 1.0 comes from
predicting only one CVE listed and getting that one right; it misses the other 73.

Combined inputs have the highest observed AP in both weighting modes. These are
descriptive comparisons without paired confidence intervals, not demonstrated
statistical superiority or an automatically selected model.

## 3. What Balancing Changes

For description + structured inputs:

| At threshold 0.5 | Ordinary weights | Balanced weights |
|---|---:|---:|
| Listed CVEs correctly detected | 3 | 31 |
| Listed CVEs missed | 71 | 43 |
| Not-listed CVEs incorrectly flagged | 8 | 82 |
| Not-listed CVEs correctly rejected | 1,022 | 948 |

F1 rises from 7.06% to 33.16%, while AP rises only from 0.2494 to 0.2587.
Balancing changes the fitting objective and score distribution, making more
CVEs cross 0.5; the large F1 change is not a comparably large ranking improvement.
It is not merely a threshold adjustment, because the models were fitted with
different class weights.

The ordinary combined model has the lowest observed log loss (0.21694) and
Brier score (0.05756), both better than the prevalence reference (0.26725 and
0.06663). The balanced combined model has worse probability-error metrics
(0.32718 and 0.09632), despite its higher recall. Its scores should not be
interpreted as calibrated probabilities of listing or exploitation.

## 4. Important Limitations

The mask-only classifier's smoke advantage did not persist. Its full-validation
AP is 0.0659, versus prevalence 0.0670, and ROC-AUC is 0.4941. With balancing,
it flags 683 CVEs, of which 639 are not listed. This is not useful evidence of
semantic discrimination. It does not establish that richer models are free of
source-availability shortcuts.

Training-to-validation ranking gaps remain substantial. Balanced description
has training AP 0.8151 versus validation 0.1691; balanced description + structured
has 0.8192 versus 0.2587. This is consistent with overfitting and/or differences
between development cohorts; these metrics alone do not identify the cause.
Listed prevalence differs too: 13.10% in training versus 6.70% in validation.
Raw accuracy and cross-split losses therefore need care.

Shared wording, class-associated filtering, current rather than historical
structured values, incomplete upstream label validation and prior test exposure
remain as documented in the frozen protocol. No old test results were inspected
or changed during this review.

## 5. Actual Prediction Examples

These are the balanced combined model's scores, not exploitation probabilities:

| CVE | Saved VulnCheck label | Predicted score | Decision at 0.5 |
|---|---:|---:|---|
| CVE-2025-24201 | Listed (1) | 0.971290 | Correctly flagged |
| CVE-2025-25291 | Not listed (0) | 0.925309 | Incorrectly flagged against this snapshot |
| CVE-2024-55963 | Listed (1) | 0.008575 | Missed |

## 6. Corrected Figure and Saved Review

The original PR plot linearly connects the constant classifier's endpoints,
creating a misleading diagonal. This does not change its correct saved AP of
0.0670. The separate review figure uses stepwise curves and a horizontal
prevalence reference. Its horizontal axis is the fraction of listed CVEs found;
its vertical axis is the fraction of flagged CVEs that are actually listed.
The plotted thresholds are for display, not new threshold selection.

Original models, metrics and figures remain unchanged. The separate TXT/JSON
analysis and `precision_recall_corrected.png` are under
`EPSS_TPG/05_results/05_vulncheck_study/06_baseline_results_review_v1/`.

Reproduce the review without inference or training:

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/05_review_development_baselines.py --verify-only
```

## 7. Next Bounded Experiment

Move to the actual TPG question: a separately frozen description-TPG + structured
classifier versus a matched pooled-description + structured classifier, with the
same CVEs, source text, structured encoding, training budget and reporting rules.
Retain all completed logistic references. Freeze weighting/threshold rules before
running; do not choose a new threshold to make these validation numbers look better.

Reuse existing graph chunks only if text and processing identities match. This
first graph comparison should isolate representation effects before adding URL
summaries and SMP text. No additional tiny baseline smoke or repeated logistic
tuning is needed. No graph advantage has yet been demonstrated for this target.
