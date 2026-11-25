# CT02: Completed Older-Dataset MAE Comparison

## Scope

The older collection's `mae_fast_v1/full` run is complete: 39 neural fits
across seeds 17, 29 and 43, plus three training-only controls. Of those neural
fits, 22 reuse verified CT01 checkpoints and 17 were trained by CT02. Every
fit uses the same 2,546 training and 1,104 validation CVEs. No test scoring
was performed. These are not full Reddit or MSE results.

The review verified the original and referenced artifact hashes, checkpoint
identities, earliest-best validation-MAE epochs, all saved prediction metrics,
source partitions and paired bootstrap intervals. It did not rerun model
inference: the user's completed runner verification covers checkpoint inference.
The review itself was rerun in a fresh process and reproduced its outputs.
Nothing in the completed training directories was changed.

## Main Results

D means description; M means screened social-media text; U means generated
GitHub and non-GitHub webpage summaries, not URL strings. Combined variants
join their text before building their TPG. All text models also receive the
same structured features and source masks. Thus D means description-only
among the textual sources, not a model without CVSS or other structured data.

The following values average the three seeds' validation metrics, not their
predictions. MAE measures average absolute EPSS error. RMSE gives more weight
to large errors. Lower is better for both.

| Text combination | Pooled MAE | TPG MAE | Pooled RMSE | TPG RMSE |
|---|---:|---:|---:|---:|
| D | 0.022424 | 0.022827 | 0.104874 | 0.107337 |
| M | 0.023466 | 0.023412 | 0.111975 | 0.112177 |
| U | 0.023926 | 0.023673 | 0.112213 | 0.112448 |
| DM | 0.023380 | 0.023169 | 0.111709 | 0.109720 |
| DU | 0.023713 | 0.023938 | 0.112231 | 0.112154 |
| DMU | 0.023559 | 0.023630 | 0.112066 | 0.112029 |

The training-median baseline has MAE 0.023604 and RMSE 0.112460. The structured
network has mean MAE 0.024251; structured ridge has MAE 0.054357.

Description-pooled has the lowest mean MAE and RMSE. Its MAE is nominally
5.00% lower than the median baseline, but its paired difference is -0.001179
with a 95% interval of [-0.003522, +0.000379]. That interval includes zero.
It is not an established MAE improvement over the baseline.

Adding M or U did not improve seed-mean MAE over D within either model family.
This does not prove that posts or summaries are useless. It describes their
performance under this source policy, architecture, objective and cohort.

## Graph Versus Pooling

| View | TPG minus pooled MAE | Unadjusted paired 95% interval |
|---|---:|---|
| D | +0.000402 | [-0.000161, +0.001166] |
| M | -0.000054 | [-0.000171, +0.000088] |
| U | -0.000253 | [-0.000445, -0.000098] |
| DM | -0.000211 | [-0.001005, +0.000429] |
| DU | +0.000225 | [+0.000167, +0.000282] |
| DMU | +0.000071 | [-0.000172, +0.000297] |

Negative differences favor TPG. U favors TPG under its unadjusted interval,
but U-TPG still has slightly worse point MAE than the constant median. DU
favors pooled text. The other four intervals include zero. M-TPG improves
over the median by about 0.81%, with an unadjusted interval below zero.
These intervals resample CVEs conditional on the saved three fits. They do
not cover all retraining uncertainty, correct multiple comparisons, or undo
repeated use of this validation set. This is exploratory development evidence.

## High Scores and Ranking

Of 1,104 validation CVEs, 996 have targets below 0.01 and 50 have targets at
or above 0.1. D-pooled's MAE on those 50 CVEs is 0.422857; these CVEs contribute
85.40% of its total absolute error across seeds. Only 3, 4 and 3 of those 50
CVEs receive predictions at or above 0.1 in seeds 17, 29 and 43 respectively.
These counts describe regression outputs; they are not the original task's
classification targets or tuned F1 thresholds.

D-pooled has the highest mean R-squared (0.0953). M-TPG has the highest neural
mean Spearman correlation (0.2589), but its scores are poorly calibrated in
absolute terms. Ranking and score accuracy are different properties.

For a real example, CVE-2024-48248 has saved Day-60 EPSS 0.934500. Its mean
prediction across the three D-pooled fits is 0.003578, and across the three
DMU-TPG fits it is 0.002124. Conversely, CVE-2025-0108 has actual 0.937040 and
D-pooled mean prediction 0.998926. The models are not literally constant, but
many high-score cases are severely underestimated.

The reviewed report includes both these examples and a deterministic selection
of high-score and near-median cases. Example averages are illustrative, not
an ensemble evaluated in the main metrics.

## Runtime and Next Step

New pooled fits took approximately 25-60 seconds per saved session, with zero
graph loads. DU-TPG needed roughly 22 seconds per epoch and benefited from its
cache. DM/DMU recorded roughly 405-467 seconds per epoch, zero cache hits, and
79-80% of epoch time in loading. Loading includes hashing, decoding and tensor
construction, not just physical disk reads. Profiles after resume cover only
the last invocation and must not be summed as total experiment wall time.

Keep this completed MAE study unchanged. The next engineering step is a
versioned, parity-tested TPG loading/cache improvement and live timing. The
subsequent bounded MSE comparison should retain the same cohorts, source masks,
seeds and architecture while selecting checkpoints on validation RMSE. MSE
tests a different objective; it is not guaranteed to improve all metrics.

## Reports and Reproduction

Review directory, relative to `EPSS_TPG`:

`05_results/08_fresh_studies/08_combined_training_v2/02_previous_multisource/01_securebert2/mae_fast_v1/full_review_v1/`

Read `REPORT.txt` first. `findings.json` includes all review values and hashes.
The CSVs provide model metrics, high-score diagnostics, seed/epoch details,
runtime profiles, additional paired comparisons and actual CVE examples.
Four PNG figures show errors by source view, high-score errors, actual versus
predicted scores and loading cost. Axis meanings are explained in the report.

From the project directory in `CodeBERTFusion`, reproduce the review without
model inference, fitting or test access:

```bash
python EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/03_review_combined_training.py --verify-only
```

## Additional Review: Ranking and Source Fusion

The following is a post-hoc analysis of the same saved predictions. It does
not change the official MAE objective, selected checkpoints, cohort or test
policy. No additional model fitting or inference was performed.

### Four-CVE Sensitivity

D-pooled's net MAE reduction of 0.001179188 is concentrated in four CVEs:

| CVE | Share of net absolute-error reduction |
|---|---:|
| CVE-2025-0108 | 67.10% |
| CVE-2025-30066 | 36.52% |
| CVE-2025-24813 | 28.96% |
| CVE-2025-26794 | 11.36% |

The four contribute 143.94% because losses on other CVEs offset some of the
gain. Excluding these four after seeing the predictions makes D-pooled worse
than the median baseline by 0.000519997 MAE. This is a sensitivity calculation,
not permission to remove the four from the experiment. On all 1,054 CVEs below
0.1, D-pooled has MAE 0.003428447 versus the median's 0.002820066.

### Ranking Is Not Score Accuracy

These metrics use the existing regression predictions to rank the 50 CVEs
with saved Day-60 EPSS >= 0.1 above the remaining 1,054. This is not a new
exploitation label or VulnCheck task. AUROC measures high-versus-low ordering;
average precision (AP) emphasizes the precision/recall ranking; top-50 hits
counts high-score CVEs in the 50 largest predictions. Rows average three seed
metrics, not ensemble predictions. The positive prevalence is 4.53%.

| View | Pooled AUROC | TPG AUROC | Pooled AP | TPG AP | Pooled top-50 hits | TPG top-50 hits |
|---|---:|---:|---:|---:|---:|---:|
| D | 0.7162 | 0.7214 | 0.2013 | 0.1824 | 11.33 | 12.00 |
| M | 0.8155 | 0.8322 | 0.1883 | 0.1735 | 10.33 | 9.33 |
| U | 0.5756 | 0.5108 | 0.0847 | 0.0713 | 5.67 | 5.33 |
| DM | 0.7406 | 0.7996 | 0.1260 | 0.2072 | 7.67 | 12.33 |
| DU | 0.6418 | 0.6216 | 0.1243 | 0.1073 | 9.00 | 7.67 |
| DMU | 0.7306 | 0.6779 | 0.1593 | 0.1196 | 11.67 | 7.33 |

M has the highest AUROC, but not the highest AP or top-50 count. DM-TPG leads
those latter two seed-mean summaries among the neural configurations. Therefore
"M is best at ranking" is too broad. M models also include structured features
and masks, so this does not isolate a causal contribution of social-media text.
U is weak here; its AUROC/AP values do not prove an absence of all useful signal.

Early checkpoint selection is real, but "almost untrained" is not established:
there are 319 optimizer steps per full epoch. Even epoch 2 follows 638 updates.
The ranking threshold and source comparisons were inspected after development
results were available and remain exploratory, without new confirmatory claims.

### What Was Verified in the External Probe

The supplied `fusion_probe.py`, `fusion_probe.json` and `pooled_views.npz` were
inspected without executing their training code. All cached CVE IDs, roles,
targets, structured vectors, source masks and pooled vectors matched the saved
development inputs. The JSON seed means reproduce from its three seed records.
However, fusion checkpoints and per-CVE predictions were not provided, so the
new training outcomes could not be independently replayed. Logistic metrics
are printed by that script but are not saved in its JSON; the claimed 0.87
AUROC and 21 top-50 hits remain externally reported rather than verified here.

The supplied per-source MAE probe reports DM AUROC 0.85459 and MAE 0.02265,
and DMU AUROC 0.86434 and MAE 0.02261. These motivate a controlled test, but
the probe changes several factors together:

- Standalone D/M/U embeddings replace combined-text contextual embeddings.
- Source-specific projections and a wider head increase parameters from
  28,705 to 54,401 or 80,097 for one, two or three branches.
- Text/head initialization order differs, even for the same numerical seed.
- Batched arithmetic replaces the official per-CVE gradient accumulation.
- Thirty neural and six logistic fits further reuse the validation cohort.
- The logistic models predict `EPSS >= 0.1`, a different target from score regression.

The observed decrease in the four CVEs' predictions under merged models is
verified. It does not by itself prove that merging caused every degradation,
or that source pooling will preserve D predictions after joint training.

### Reusing the Existing Graphs

An outcome-blind sample of 23 graphs covers both development splits and all
six views, selecting shortest and middle-sized graphs by node count. Every
sampled tensor node row mapped correctly through `document_index` to its
document and D/M/G/N source in the graph JSON and frozen plan. This is a bounded
feasibility check, not a completed all-graph mapping audit.

The current training loader drops `document_index`; a new adapter can retain
it without rerunning spaCy or SecureBERT. The global DOCUMENT node has index
`-1` and must not be arbitrarily assigned to D. A new readout must declare
whether to retain a separate global channel or exclude that node consistently.
Cross-source messages and combined-context embeddings remain even when the
final pooling is source-specific. Such a readout is not a fully isolated
source encoder and is not identical to the supplied standalone-vector probe.

### Updated Next-Step Recommendation

Keep the completed MAE run frozen. Before expensive MSE graph fits, specify a
small pooled merged-versus-source-readout comparison using the same combined
cached embeddings, roles, seeds and source masks. Share projection parameters
or include a parameter-matched control. Treat MAE/MSE as separate objectives;
for MSE, select on validation RMSE and declare MAE, ranking and high-score errors
as secondary results. Optimize graph loading with a separate parity gate before
extending the controlled readout comparison to TPG. This supersedes a proposal
to immediately run only the existing merged MSE architecture.

Reddit can support a later locked test of the selected procedure, but its test
identity, prior-exposure exclusions and no-retuning rule must be checked. Its
different cohort and substantial M source shift are not automatically evidence
of generalization. Its supported full text arms remain D, M and DM because
accepted validation webpage summaries are absent.

The addendum is in the sibling `ranking_fusion_review_v1` directory. Read its
`REPORT.txt`, then `findings.json`, `ranking_per_seed.csv`,
`ranking_seed_means.csv`, `d_gain_contributions.csv` and
`top_contributor_predictions.csv`. CSV ordering is canonical, not gain-ranked;
sort `absolute_error_reduction` numerically to inspect the largest contributors.
Copies of the supplied probe script and JSON are retained as external evidence,
not official training artifacts. Original completed runs remain unchanged.

Reproduce the addendum with the supplied probe evidence still available:

```bash
python EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/04_review_ranking_and_fusion.py --verify-only
```
