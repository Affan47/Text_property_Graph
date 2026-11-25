# VK06 Results: Matched Edge-Free Training

## What Was Verified

All three full edge-free checkpoints, saved predictions and reports reproduced.
The separate review recalculated their metrics and selected-epoch checks from
saved artifacts. Original VK04 models and results remain unchanged. No new
training, threshold tuning or test scoring was performed during review.

The control retains the original 2,870 training and 1,104 validation CVEs, with
74 listed validation CVEs. It preserves 302,654 training nodes and 123,130
validation nodes. It removes 926,672 training edges and 381,652 validation edges,
leaving zero edges in both roles. Frozen embeddings, node types, encoder layers,
pooling, structured inputs, initialization rule and training settings are matched.

## Overall Comparison

Neural results are arithmetic means of three separately trained seeds, not an
ensemble. Classification decisions use the unchanged score threshold of 0.5.

| Model with structured features | Average precision | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Pooled description, VK04 | 0.3015 | 0.7909 | 31.25% | 24.77% | 27.47% |
| Original description TPG, VK04 | 0.2992 | 0.8010 | 29.07% | 32.43% | 29.69% |
| Same graph encoder trained without edges, VK06 | 0.2873 | 0.7838 | 31.75% | 27.93% | 27.49% |
| Balanced combined TF-IDF/structured logistic reference | 0.2587 | 0.7826 | 27.43% | 41.89% | 33.16% |

Average precision (AP) measures ranking of listed CVEs ahead of unlisted CVEs
across thresholds. Precision is the listed fraction among flagged CVEs; recall
is the fraction of the 74 listed CVEs found. F1 combines precision and recall.
The logistic reference still has higher F1 at 0.5, although neural ranking AP
is higher. These metrics need not choose the same model.

## Seed-by-Seed Evidence

| Seed | Original TPG AP | Edge-free AP | Edge-free minus TPG AP | Original TPG ROC-AUC | Edge-free ROC-AUC |
|---|---:|---:|---:|---:|---:|
| 17 | 0.2933 | 0.3016 | +0.0083 | 0.8133 | 0.7717 |
| 29 | 0.3366 | 0.2904 | -0.0463 | 0.8015 | 0.7962 |
| 43 | 0.2676 | 0.2700 | +0.0024 | 0.7882 | 0.7836 |

Original TPG has a mean AP advantage of 0.0119, driven by its larger advantage
in seed 29. Edge-free AP is slightly better in the other two seeds. Original
TPG ROC-AUC is higher in all three seeds. This is some evidence that edges help
this setup, but not a stable AP advantage across seeds or proof of statistical
significance. Three-seed standard deviations are not confidence intervals.

## Decisions and Probability Errors

| Seed | Listed found with TPG / edge-free | Unlisted falsely flagged with TPG / edge-free |
|---|---:|---:|
| 17 | 19 / 27 | 39 / 99 |
| 29 | 23 / 23 | 48 / 45 |
| 43 | 30 / 12 | 106 / 18 |

Seed 17 finds eight more listed CVEs but adds 60 false alarms. Seed 43 avoids
88 false alarms but misses 18 listed CVEs that the original model found. There
is no uniform operating-point improvement. These false positives are relative
to the saved snapshot; they are not proof that those CVEs were never exploited.

Mean edge-free Brier score is 0.08106 versus TPG 0.08203, slightly better. Its
mean log loss is 0.38187 versus TPG 0.36005, worse. Both mean probability-error
measures remain worse than the training-prevalence constant (Brier 0.06663,
log loss 0.26725). Neither model's scores should be assumed calibrated.

## Training and the Earlier Diagnostic

Edge-free selected epochs are 10, 6 and 6; the runs ended after 25, 21 and 21
epochs under the fixed patience rule. Final AP was lower than selected AP in
every seed. For seeds 29 and 43, training loss continued decreasing after the
selected epoch while validation AP worsened. Extending identical training is
not supported by those histories.

VK05 removed edges at inference from models trained with edges. VK06 trains
fresh models without edges, allowing optimization and checkpoint selection to
change. The two experiments need not agree. Inference-time edge removal alone
could not predict whether retraining without edges would improve performance.

## Conclusion and Next Decision

The control does not support removing graph edges as a general improvement.
It provides some evidence of a benefit from edges, particularly in ROC-AUC,
but the original TPG still does not consistently outperform pooled description
embeddings. Because all edge types were removed together, this result cannot
attribute a benefit specifically to security edges.

Close this connectivity ablation rather than continue tuning architecture or
thresholds against the same validation set. The next planned question can be
a bounded source ablation: description only; description plus URL summaries;
description plus SMP; and all sources. Retain structured features and matched
pooled/TPG controls, the same CVEs, missing-source masks and reviewed disclosure
filtering. Verify and document source availability/filtering by class before
training; no new source experiment has been launched by this review.

Current features remain retrospective, and repeated validation, shared text,
source filtering and prior test exposure limit generalization claims. This is
not an independent final evaluation or evidence of state-of-the-art performance.

## Review Artifacts

`05_results/05_vulncheck_study/12_edge_free_results_review_v1/` contains TXT/JSON
analysis and `matched_edge_comparison.png`. Each figure line connects the same
seed's original and edge-free trained models. Higher vertical values indicate
better ranking; connecting lines are not confidence intervals.

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/11_review_edge_free_control.py --verify-only
```
