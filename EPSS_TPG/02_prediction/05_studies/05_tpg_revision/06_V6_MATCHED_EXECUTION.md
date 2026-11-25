# TRV05 execution: v6 cache, matched fits and partitioned diagnostics

## Completed fits and findings

All nine full fits completed on the fixed 2,870 training / 1,104 validation
CVEs. Fresh-process checkpoint replay reproduced every saved prediction and
metric. Training stopped after 20--33 epochs, with selected checkpoints from
epochs 5--18. The nine frozen v5 references and the test cohort were not changed.
The smoke gates and 37 focused v5/v6 regression tests passed.

Read the [full diagnostic tables and figures](../../../05_results/06_tpg_revision/11_v6_comparison_v1/diagnostics/full/REPORT.md).
The figures, predictions, intervals and checkpoint diagnostics are saved beside
that report. The figures show all three seeds; no best seed is substituted for
the mean. The full diagnostic reproduced in a fresh process, including all
prediction tables, 1,000-resample intervals and seven figure hashes.

### Full validation comparison

Each entry is the mean across seeds 17, 29 and 43, not an ensemble. AP measures
ranking quality; larger is better. F1 uses the unchanged 0.5 cutoff. Brier is
the mean squared error of the predicted listing probabilities; smaller is better.

| Matched model family | v5 AP | v6 AP | v5 F1 | v6 F1 | v5 Brier | v6 Brier |
|---|---:|---:|---:|---:|---:|---:|
| Node mean, no subtypes | 0.296638 | 0.291011 | 0.272766 | 0.292982 | 0.078711 | 0.079923 |
| Subtypes, node mean | 0.286329 | 0.297985 | 0.276958 | 0.301091 | 0.070785 | 0.085908 |
| Subtypes, type-balanced | 0.281423 | 0.292690 | 0.303865 | 0.321890 | 0.077889 | 0.090156 |

Both subtype variants improve AP and F1 in all three seeds. The plain node-mean
variant improves AP in only one seed. This does not establish a reliable overall
AP improvement: the full-cohort paired 95% AP intervals include zero for all
three families, respectively [-0.029726, 0.019101], [-0.019868, 0.039114] and
[-0.013447, 0.031105]. These are exploratory intervals on repeatedly used
validation data, not independent final-test evidence.

Brier error rises on average in all three families. For the two subtype families,
the intervals for the Brier increase are [0.009028, 0.021270] and
[0.008053, 0.017282], above zero under this saved resampling procedure. Ranking
and probability error therefore tell different stories. In subtype node-mean,
recall rises from 0.238739 to 0.351351, but precision falls from 0.358858 to
0.328924. Its log loss improves from 0.514188 to 0.408848 despite worse Brier;
log loss penalizes very confident mistakes more strongly. Do not describe this
as an across-the-board probability-quality improvement.

### Where the changes occur

The table shows mean AP differences, v6 minus the same-seed v5 control. Positive
means improved ranking within that subset. These subset values cannot be
averaged to reconstruct full-cohort AP.

| Model family | 16 graph-repaired CVEs | 39 severity-restored CVEs | 1,049 unchanged CVEs |
|---|---:|---:|---:|
| Node mean, no subtypes | -0.007937 | -0.001159 | -0.001048 |
| Subtypes, node mean | -0.087302 | -0.000989 | +0.013368 |
| Subtypes, type-balanced | -0.074359 | -0.006968 | +0.011524 |

The subtype AP gains occur in the unchanged-input group, not in the two repaired
groups. Changing training graphs and subtype inputs changes the learned weights
and the selected epoch, which can affect every validation prediction. This is
not evidence that the validation repairs directly improved those unchanged CVEs.
Even the plain node-mean model can change on severity-group CVEs after retraining,
although it never uses the subtype projection.

Only CVE-2025-31125 is listed in the graph-repaired group; only CVE-2025-2783 is
listed in the severity-restored group. Both remain below the fixed 0.5 cutoff
in every v5/v6 fit. Each group's stratified bootstrap repeats its one observed
positive CVE, so apparently narrow or one-sided subgroup intervals must not be
interpreted as evidence across a population of positive CVEs.

### Frozen weights separate the direct graph effect

With the original v5 weights and subtype meaning retained, repaired graphs
produce exactly zero probability difference on all 1,088 non-graph-repaired
validation CVEs, in all nine comparisons. No 0.5-threshold decision changes
anywhere in validation. On the 16 repaired graphs, mean AP change is zero for
node-mean and subtype node-mean, and -0.035714 for subtype type-balanced.
There is no demonstrated direct ranking benefit from this intervention.

For example, in subtype type-balanced seed 17, the listed CVE-2025-31125 changes
from 0.4001595 on its old graph to 0.2942987 on its repaired graph with identical
weights. Retraining v6 produces 0.1483665. These are model listing scores, not
verified exploitation probabilities. The first change is a topology effect;
the second comparison also includes new weights and checkpoint selection.

The restored SEVERITY column learns nonzero weights: selected norms are
0.707121, 1.326131 and 0.531235 for subtype node-mean, and 0.541598, 0.857197
and 0.642708 for subtype type-balanced (seeds 17/29/43). The plain arm remains
zero. This verifies that the restored channel is active, not that its learned
meaning is correct or predictively helpful by itself. The full subtype norms
range from 2.910244 to 5.506257; norms alone do not prove instability.

Overconfidence is not uniformly resolved. Subtype node-mean seed 17 changes its
selected epoch from 28 to 5 and reduces validation logits with absolute value
above 10 from 804 to 2. Seed 29 still has 665 such logits, one extremely confident
false positive and 17 extremely confident false negatives. Type-balanced seed
43 increases its above-10 logit count from 174 to 481. All these diagnostics are
retained rather than reporting only the favorable seed.

### Decision

Keep v6 as an explicit extraction/schema correction and retain all nine fits as
bounded development evidence. Do not promote it as a proven predictive upgrade,
reopen the test cohort, tune a new threshold, or launch another decay search from
these results. The experiment separates valid representation repairs from the
unproven claim that those repairs improve classification. Any further predictive
study needs a separately agreed hypothesis and evaluation protocol.

## Approved scope

This implements the approved [v6 audit and specification](05_V6_FRONTEND_AND_SCHEMA.md).
The original audit, v5 cache, nine v5 controls and completed TRV03/TRV04 studies
remain unchanged. The task is retrospective classification of inclusion in the
saved VulnCheck catalogue. No EPSS predictor, additional text source, threshold
tuning, new regularization setting or test scoring is introduced.

Full development remains 2,870 training / 1,104 validation CVEs. The three
validation partitions are frozen from the graph audit, not selected from model
errors or outcomes:

| Partition | Validation CVEs | Listed CVEs | Meaning |
|---|---:|---:|---|
| graph_repaired | 16 | 1 | Version-shaped predicates and dependent annotations removed |
| severity_restored | 39 | 1 | Same graph; SEVERITY now has its explicit numeric channel |
| unchanged | 1,049 | 72 | Same graph and numeric subtype inputs |

These groups do not overlap. The full-cohort AP cannot be calculated by averaging
their AP values. All groups retain listed/unlisted counts and undefined metrics.
Each changed subgroup has only one positive example. Stratified bootstrapping
reuses that same positive in every resample and therefore cannot estimate
variation across different positive CVEs. Subgroup AP/recall changes and their
intervals are descriptive, not reliable evidence of subgroup generalization.

## Cache and initialization gate

The new cache copies surviving tensor rows and edge features exactly from v5,
remaps only edge indices after removals, and uses the approved append-only
SEVERITY channel. No parser or embedding encoder runs again. Every source and
output artifact is hash-checked; publication is atomic and existing outputs are
never overwritten. The smoke cohort is the old integration cohort plus the
first two IDs in each audit-defined partition, selected without score inspection:
28 training / 8 validation CVEs. It is not a performance benchmark.

For every model variant and seed 17, 29 and 43, the parity gate checks:

- Common initial parameters are bit-identical to v5; the subtype projection is zero.
- On the same v5 inputs, initial v5/v6 probabilities are exactly equal.
- On v6 inputs, unaffected cases remain equal within 1e-7 float32 tolerance.
- Initial prediction changes on repaired graphs are recorded, not forced to zero.
- The corresponding saved v5 checkpoint reproduces its reference predictions.

Zero initial difference is not required for graphs with removed nodes/edges.
Changing topology can change message passing and pooling before any training.
Forcing equal outputs there would defeat the intended graph intervention.

Frozen v5 references always use their original full-validation order and batch
size before selecting smoke rows. A first smoke gate correctly stopped when
smaller-batch inference differed by 5.96e-7. No gate or fit was published from
that attempt; canonical batching fixes the replay without relaxing tolerance.
Smoke graph building and training remain bounded. Frozen reference inference
uses already-observed full development validation only, never the test cohort.

## Matched training

The three variants are `v6_node_mean`, `v6_subtypes_node_mean` and
`v6_subtypes_type_balanced`. Each uses seeds 17, 29 and 43: nine fresh fits.
The nine matching v5 fits remain frozen references and are not retrained.

Settings are inherited from the approved specification: hidden width 32, two
typed-edge layers, frozen embeddings, unchanged structured branch, AdamW with
learning rate 0.001 and weight decay 0.0001, batch size 8, at most 100 epochs,
patience 15, training-only balanced class weights and weighted BCE. Checkpoint
selection remains earliest strict best full-validation AP. The cutoff stays 0.5.
Gradient replay, packet budgets and two CPU threads per fit are unchanged.

Independent fits may execute concurrently. Worker count affects scheduling, not
the fit settings. Completed fit directories are immutable; `--resume` reloads
and verifies them and starts only missing fits. Failed/incomplete attempts are
preserved outside the completed result tree. There is no mid-epoch resume.

Smoke uses all three variants at seed 17 for two epochs. The full-development
structured encoder and class weights are reused for integration only. Its v5
references were full-trained; smoke comparisons are not matched evidence.

## Diagnostics

Every v6/v5 seed pair has full-cohort and disjoint-partition AP, ROC-AUC,
precision, recall, F1, accuracy, balanced accuracy, Brier, log-loss and confusion
counts. Reports include per-CVE probabilities/logits, selected epochs, epoch
curves, subtype norms and the selected SEVERITY column-31 norm.

The frozen-checkpoint graph intervention uses the saved v5 model with v6 graph
topology, while remapping SEVERITY back to UNKNOWN for its old 31-channel schema.
This isolates inference-time graph removal. No weights change; the 1,049
unchanged and 39 severity-only cases must retain their probabilities within
1e-7 floating-point tolerance. Their exact maximum discrepancy is reported.
The 16 graph-repaired cases retain their paired metric and score changes.

Full reports use 1,000 paired class-stratified CVE bootstrap samples (seed 6029)
for v6-minus-v5 on all four reporting groups and for the frozen graph intervention
on the repaired group. Smoke uses 20 resamples for integration. Undefined metrics
remain null, including ranking metrics in single-class groups. Intervals hold
observed class composition fixed, are not estimates of seed uncertainty, and
do not correct for repeated development, multiple comparisons or advisory groups.

Changes on unchanged inputs after retraining reflect global learned-model and
checkpoint-selection differences. They are not evidence of a direct local graph
repair. The subtype arms still combine predicate removal and schema restoration;
this is not a full factorial attribution of the two repairs. No subgroup winner
is automatically promoted.

## Repository locations

- CLI: `02_prediction/03_analysis/04_tpg_revision/10_run_v6_comparison.py`
- Adapter: `02_prediction/01_package/epss/tpg_revision/v6_cache.py`
- Runner: `02_prediction/01_package/epss/tpg_revision/v6_training.py`
- Diagnostic: `02_prediction/01_package/epss/tpg_revision/v6_diagnostic.py`
- Cache: `01_tpg/05_workspace/security_product_v6_subtypes_v1/{smoke,full}`
- Results: `05_results/06_tpg_revision/11_v6_comparison_v1/`
- Subfolders: `parity`, `training`, `diagnostics`, each with `smoke` and `full`.
- Live per-fit logs: `11_v6_comparison_v1/execution_logs/{smoke,full}`.
- Final report: `11_v6_comparison_v1/diagnostics/full/REPORT.md`.

## Commands

Run these from `/home/ayounas/Text_property_Graph` in `CodeBERTFusion`.
Use `--verify-only` for an existing completed stage; do not delete outputs.

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/04_tpg_revision -p 'test_v6*.py'
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/10_run_v6_comparison.py --stage cache --scale smoke
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/10_run_v6_comparison.py --stage cache --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/10_run_v6_comparison.py --stage parity --scale smoke
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/10_run_v6_comparison.py --stage train --scale smoke --workers 1
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/10_run_v6_comparison.py --stage train --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/10_run_v6_comparison.py --stage diagnose --scale smoke
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/10_run_v6_comparison.py --stage diagnose --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/10_run_v6_comparison.py --stage cache --scale full
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/10_run_v6_comparison.py --stage cache --scale full --verify-only
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/10_run_v6_comparison.py --stage parity --scale full
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/10_run_v6_comparison.py --stage train --scale full --workers 6
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/10_run_v6_comparison.py --stage train --scale full --verify-only
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/10_run_v6_comparison.py --stage diagnose --scale full
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/10_run_v6_comparison.py --stage diagnose --scale full --verify-only
```

For interrupted training, use the same training command with `--resume`.
