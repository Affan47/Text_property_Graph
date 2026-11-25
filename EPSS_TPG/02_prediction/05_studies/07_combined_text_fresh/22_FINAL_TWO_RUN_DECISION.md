# Final Two-Run Decision

Date: 16 November 2026.

Status: recommended study design, not an executable training/release lock.
No training, test-cache construction or test-outcome inspection was performed
for this decision. Existing results and source data remain unchanged.

## Decision

Yes: reserve the Reddit official test set for one final, frozen evaluation.
Complete exactly the two proposed comparisons, then close this dataset's
experimental series. Do not reopen early warning after the DS01 stop decision.
Do not add a catalogue, summary, loss-search or architecture-search branch.

Run 1 is development evidence: its assessment CVEs have already informed many
analyses. Run 2 is a held-out retrospective, cross-version transfer evaluation.
Neither is a historical exploitation forecast: current text, CVSS and reference
counts may postdate the Day-60 score. EPSS estimates exploitation probability;
estimating EPSS is not observing whether exploitation occurred.

## Same Six Rows

| Row | Predictor | Fits per seed/fold |
| --- | --- | --- |
| 1 | Training mean; also report training median as a diagnostic | None |
| 2 | CVSS and allowed reference-count features, with observed-value flags | 1 |
| 3 | Structured features plus pooled D | 1 |
| 4 | Structured features plus D TPG | 1 |
| 5 | Structured features plus pooled DM | 1 |
| 6 | Structured features plus DM TPG | 1 |

D means standalone description. M means the existing accepted prose-post input,
not bare CVE identifiers. DM uses the existing combined-text graph/embedding
cache; pooled models average its embedded text tokens, while the GNN performs
typed message passing and graph readout. Freeze the existing source masks,
document boundaries, graph schema and encoder identities. No cache regeneration
or alternate readout is chosen from the new results.

Use the existing two-layer, hidden-32 typed-edge GNN, including node types,
32 entity-subtype channels, relation IDs and 88 edge properties. Preserve
directed edges and exact vocabularies. Keep the structured branch and prediction
head settings matched. Record parameter counts: this is a comparison of graph
and pooling systems, not an isolated causal test of edge information.

Run 1 retains CT07's three main-policy folds, group purges and 8,053 assessment
CVEs. Do not rerun its publication-date sensitivity policy. Fit every scaler,
category mapping and constant only on that fold's optimization IDs. Use the
fixed selection IDs only for early stopping.

Run 2 retains 24,213 official training, 972 validation and 3,977 test CVEs.
Keep all 4,987 prior-test-connected exclusions. Preprocessing and constants use
training IDs only; validation selects checkpoints; no train+validation refit.
Build test predictors only after the model/transform lock; never fit vocabulary
or scaling on test. Store test labels separately until the release stage.

## Proposed Fixed Settings

- Five explicit seeds: 17, 29, 43, 59, 71. This is a new five-seed choice; CT07
  used three seeds and DS01 used ten, not five.
- Raw-score MSE, equal CVE contribution, AdamW, learning rate 0.001,
  weight decay 0.0001, batch size 8, at most 100 epochs, patience 15.
- Earliest strict minimum selection/validation RMSE; save only the best
  checkpoint, with optimizer/RNG state needed for deterministic recovery.
- Frozen SecureBERT 2.0 embeddings, hidden size 32, two graph layers, mean
  readout, no new dropout or hyperparameter tuning. CPU execution with four
  computation threads per fit; record and pin BLAS/runtime settings.
- One integration gate: one epoch for each of the five learned rows on fold 1,
  seed 17. Verify input/label separation, update/reload parity, source/vocabulary
  identities and RAM consumption. Discard its weights and start scientific
  fits fresh. Its outcomes cannot select scientific settings.
- Estimate runtime and concurrency from the full-fold integration measurements
  and the complete resident working-set size. The 24-graph synthetic benchmark
  establishes neither full-epoch runtime nor that concurrent full caches fit.

Run 1 has 75 fits (30 graph, 45 non-graph). Run 2 has 25 fits (10 graph,
15 non-graph). Total: 100 learned fits, plus five integration checks and no-fit
constants. All six rows proceed to Run 2, irrespective of Run 1's ranking.

## Metrics and Claims

Primary metric: RMSE. Compute each seed's errors on the same CVEs, then report
their mean and individual results. Do not silently switch to an ensemble of
predictions. Run 1 reports pooled out-of-fold errors and each fold separately.
Fold-specific constants can discriminate between folds, so pooled ranking
metrics for constants need not equal their within-fold values.

Declare six primary contrasts: 4 minus 3 and 6 minus 5 (graph versus pooling),
and 3/4/5/6 minus 2 (text systems versus structured). Use paired evidence-group
and seed resampling: 20,000 draws, seed 6029, retaining matched seed/fold pairs.
Report nominal 95% intervals and simultaneous-family protection via six
Bonferroni-adjusted 99.1667% intervals. CVE-resampled intervals are a sensitivity
analysis, not a substitute selected when the grouped interval is unfavorable.
Five seeds still leave limited information about optimization variability;
resampling saved fits does not measure all uncertainty from retraining data.

Report MAE, Spearman, high-score (EPSS >= 0.1) AUROC/AP and top-50/top-100
retrieval, with supports and undefined metrics explicit. Report full-cohort
results first; target-version and M-present/M-absent slices are descriptive.
Existing overlap flags remain documented. Secondary metrics cannot rescue a
failed primary RMSE contrast. Keep every CVE, including influential cases.

- A favorable adjusted Run-1 interval plus the same sign in Run 2 is
  development support with directional replication, not established test gain.
- A held-out improvement claim requires the matching adjusted test interval
  to support it too. Publish all six contrasts regardless of significance.
- A graph-versus-pooling advantage does not automatically make either system
  useful: also show differences from the training mean/median on the same CVEs.
  Any formal baseline-superiority claim needs its own prespecified uncertainty
  rule; no favorable unadjusted baseline comparison becomes a headline.
- Rows 5/6 add post availability as well as wording. With no flag-only arm in
  this final design, do not claim that post wording alone caused a difference.
- Within-version targets avoid an EPSS formula change in Run 1; they do not
  remove cohort, source-coverage, calendar or retrospective-input differences.
- Inconclusive is not evidence of equivalence or a universal graph failure.

## One Test Release

Before any new scientific fitting, write the executable protocol and hashes
for both runs, these six rows, all settings, metrics, contrasts and partitions.
No result from Run 1 may change Run 2's design. After official development,
lock all 25 selected checkpoints, transforms, constants and evaluator code.
Only then build fixed-policy test predictors and score all rows in one release.
Unresolvable test-input failures must be documented; do not silently remove
hard cases or revise source/graph rules after inspecting test errors.

A one-time scientific release permits crash recovery and deterministic replay
of identical checkpoints. It does not permit retuning, choosing a test winner,
new thresholds, new seeds, alternate losses or another test-directed fit.
Report negative and inconclusive outcomes and stop further experiments here.

The existing CLIs do not yet implement this entire locked two-run evaluation.
Do not launch an old runner as if it did. The next implementation is this
runner and its one integration/resource gate, not another exploratory study.

## Report Commands

These update/verify documentation only, on CPU. They do not train or open test.

```bash
cd /home/ayounas/Text_property_Graph
conda activate CodeBERTFusion
REPORT=EPSS_TPG/00_documentation/06_papers/03_build_files/primary_tpg_final_research_report/consolidation_v2
python -B "$REPORT/build_report.py" --build
python -B "$REPORT/build_report.py" --verify-only
```
