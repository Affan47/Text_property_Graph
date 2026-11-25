# VK04: Description Graph Classifier Results

## 1. What Was Verified

All six full-development checkpoints were reloaded and their saved validation
predictions and reports reproduced. The separate review also recomputed metrics
from those predictions, checked the label joins and original artifact hashes,
and inspected saved training histories and graph coverage. No retraining,
threshold tuning or test scoring occurred. Original results remain unchanged.

The target is whether a CVE is listed in the saved VulnCheck KEV snapshot. It is
not an EPSS regression target, future exploitation probability or CISA KEV label.
There are 2,870 training CVEs (376 listed) and 1,104 validation CVEs (74 listed).
Current descriptions and structured features are retrospective inputs.

## 2. Main Comparison

Both neural models use description text, frozen SecBERT embeddings and the same
29 encoded CVSS/reference features. No SMP, URL-summary text or EPSS inputs are
used. Each model was trained with seeds 17, 29 and 43. The table reports the
arithmetic mean of the three separate runs, not an ensemble prediction.

| Model | Average precision | ROC-AUC | Precision at 0.5 | Recall at 0.5 | F1 at 0.5 | Accuracy at 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| Pooled description + structured | 0.3015 | 0.7909 | 31.25% | 24.77% | 27.47% | 91.24% |
| Description TPG-GNN + structured | 0.2992 | 0.8010 | 29.07% | 32.43% | 29.69% | 89.64% |
| Balanced description TF-IDF + structured logistic reference | 0.2587 | 0.7826 | 27.43% | 41.89% | 33.16% | 88.68% |

Average precision measures how well listed CVEs are ranked ahead of unlisted
CVEs, across decision thresholds. Here it is sklearn average precision, not
trapezoidal PR-curve area. Precision answers: of the CVEs flagged as listed,
how many are actually listed? Recall answers: of the 74 listed CVEs, how many
are found? F1 combines precision and recall. Reported decisions retain the
predeclared 0.5 score threshold; there was no search for a better threshold.

Both neural representations have higher observed AP than the combined balanced
logistic reference in every saved seed. However, the graph model does not show
a consistent ranking advantage over pooled embeddings. The logistic reference
still has higher F1 and recall at the fixed threshold. Neither observation is
an independent-test or statistical-significance claim.

## 3. Every Seed Matters

| Seed | Pooled AP | TPG AP | TPG minus pooled AP | TPG listed CVEs found | TPG unlisted CVEs incorrectly flagged |
|---|---:|---:|---:|---:|---:|
| 17 | 0.3318 | 0.2933 | -0.0385 | 19 / 74 | 39 |
| 29 | 0.2743 | 0.3366 | +0.0623 | 23 / 74 | 48 |
| 43 | 0.2983 | 0.2676 | -0.0307 | 30 / 74 | 106 |

TPG wins on AP for one seed and loses for two. Mean AP differs by only -0.0023.
Across-seed population standard deviations are 0.0236 for pooled and 0.0285 for
TPG; these are not confidence intervals. Reporting only TPG seed 29 would hide
the instability. Seed 43 finds more listed CVEs but also produces far more false
alarms. It is not an unqualified improvement.

Always predicting "not listed" would achieve 93.30% accuracy while finding
none of the listed CVEs. Accuracy alone is therefore misleading here.

## 4. Scores Are Not Calibrated Probabilities

| Model | Mean Brier score, lower is better | Mean log loss, lower is better |
|---|---:|---:|
| Pooled description + structured | 0.07495 | 0.50153 |
| Description TPG-GNN + structured | 0.08203 | 0.36005 |
| Constant training-prevalence reference | 0.06663 | 0.26725 |
| Unweighted TF-IDF + structured logistic reference | 0.05756 | 0.21694 |

These metrics penalize disagreement between scores and actual binary labels.
All six neural runs have worse Brier scores than the prevalence reference.
Pooled seed 17 assigns scores of at least 0.8 to 55 CVEs, with a mean score of
0.9712, but only 19 of those CVEs are listed. Good ranking and overconfidence can
coexist. Class-weighted training and AP-based checkpoint selection do not promise
calibrated probabilities.

## 5. Training and Graph Coverage

The selected epochs were 29/17/6 for pooled text and 18/8/5 for TPG. All six
runs stopped 15 epochs after their last improvement. In every run, final-epoch
AP was below the selected checkpoint's AP. Falling training loss in most fits
without sustained validation improvement is consistent with overfitting; it
does not prove that one particular architecture component caused it. Simply
training for longer is not supported by these curves.

The cache has 3,974 description graphs: 3,650 exact verified reuses and 324 new
builds. One graph corresponds to each development CVE in this cache.

| Development split | CVEs | CVEs with at least one security edge | Product-version edges | Cause edges | Severity edges |
|---|---:|---:|---:|---:|---:|
| Training | 2,870 | 1,031 | 1,459 | 35 | 1 |
| Validation | 1,104 | 304 | 441 | 5 | 0 |

Across both splits, 1,900 of 1,941 security edges are product-version links.
They are extracted candidates, not independently verified facts. Graphs without
security edges still contain linguistic/document edges. The current results do
not demonstrate the usefulness of a rich exploitation-relation graph, nor do
they demonstrate that security relations are useless. Pooled and graph paths
also differ in active capacity and node populations, so this is not an isolated
security-edge ablation.

## 6. Decision and Next Step

The defensible conclusion is: these neural models improve observed validation
ranking over the logistic reference, but adding the present graph representation
does not consistently outperform pooled description embeddings. This is not a
failure of classification to learn any signal, and not evidence of TPG superiority.

Before another training run, inspect the frozen models' reliance on text versus
structured inputs and their sensitivity to relation removal. Do not change the
threshold or retune the saved runs. Such interventions describe model sensitivity,
not the result of matched retraining or a causal effect of the removed feature.
Use that bounded diagnostic to decide whether a predeclared source ablation is
worth running. Do not launch the whole source matrix merely to search for a win.

Repeated use of validation, shared wording across CVEs, source filtering,
retrospective features and prior test exposure remain limitations. A new target
does not turn the already observed test cohort into a fresh independent holdout.

## 7. Saved Review

Read `05_results/05_vulncheck_study/08_description_graph_review_v1/analysis.txt`
or `analysis.json` for all six metrics, all logistic references, training-curve
summaries and graph coverage. `ranking_and_training.png` shows each seed's
precision-recall curve and validation AP by epoch. Its axes explain the quantities;
curves display thresholds without selecting a new reporting threshold.

Reproduce the review without inference, training or test access:

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/07_review_description_graph.py --verify-only
```
