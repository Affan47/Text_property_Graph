# TRV03: fixed deterministic regularization comparison

## Question and frozen design

Does stronger global AdamW weight decay improve validation ranking and probability
errors for description-based VulnCheck listing classifiers? This is retrospective
classification, not EPSS regression or forecasting first exploitation.

| Setting | Fixed choice |
| --- | --- |
| Families | v5_node_mean; v5_subtypes_node_mean; v5_subtypes_type_balanced |
| Weight decay | 0.0001 (reuse original checkpoints); 0.001; 0.01 |
| Seeds | 17, 29, 43 for every family and decay |
| CVEs | 2,870 training and 1,104 validation; unchanged roles |
| Fits | Nine reused controls and 18 new fits |
| Selection | Original earliest best validation AP; patience 15; at most 100 epochs |
| Other settings | Original weighted BCE, learning rate, batches, graph cache, initialization and replay |
| Execution | Up to six independent fit processes, two Torch threads each |
| Test access | None |

The configuration and implementation hashes are saved before training. Completed
original controls are reproduced, not overwritten or retrained. Resuming verifies
finished fits and restarts an unfinished fit from its fixed seed.

## What this can establish

Report all seeds and same-seed differences in AP, ROC-AUC, F1 at the fixed 0.5
cutoff, Brier score and log loss. No threshold optimization is part of this run.
Selected-checkpoint diagnostics include raw logits, extreme probabilities,
confident mistakes, subtype weight norms, RMS weights and selected epochs.

A norm near 5.9 over 992 weights is not by itself an explosion: its RMS is about
0.19. The decay affects all model parameters, not only subtype weights. A benefit
therefore cannot be attributed solely to the subtype projection. Selected epochs
can differ, so weight norms also reflect different training durations. Original
controls do not contain per-epoch weight snapshots; norms are compared at the
selected checkpoints, not reconstructed across unavailable intermediate states.

## Known extraction gaps kept separate

The spaCy frontend accepts content tokens tagged VERB without checking for version
strings. In saved graphs, `18.x` in CVE-2019-13024 is tagged VERB and becomes a
PREDICATE. The v5 corrections cover selected nominal predicates and the words
`version`/`versions`, not this general version-string case.

The security frontend does produce SEVERITY entities. The frozen 31-category
subtype vocabulary omits SEVERITY, so its lookup falls back to UNKNOWN for 91
entities. Their text and raw entity type remain available; the explicit subtype
feature is what loses that distinction. Fixing either gap requires a separately
versioned graph/schema experiment, not a silent change during this comparison.

## Commands and outputs

Run from the repository root with CodeBERTFusion activated:

```bash
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/04_run_regularization.py --check-only
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/04_run_regularization.py --workers 6
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/04_run_regularization.py --verify-only
```

After interruption use `--resume --workers 6`, not a new run directory. Results
are stored in `05_results/06_tpg_revision/06_v5_regularization_v1/full` relative
to EPSS_TPG: REPORT.md, summary.txt/JSON, per-CVE predictions, metrics,
diagnostics, reference hashes, checkpoints, histories and per-fit logs.

## Subsequent source experiment

DG means description plus GitHub summary only, excluding non-GitHub summaries
and SMP. Previous DGN comparisons do not isolate DG. A later matched D versus DG
study should retain missing-source cases, audit target-CVE relevance and compare
pooled text with graphs on the same CVEs. This is a hypothesis, not a promised
improvement or proof that descriptions have reached an information ceiling.
Saved current descriptions are not automatically publication-time descriptions.
DG training and extraction/schema changes are outside this run.

## Execution status

Full execution and independent checkpoint verification are complete. All nine
original controls and 18 new fitted configurations reproduce; no test scoring.

The initial launch stopped before the first optimizer update because JSON had
converted the class-weight keys to strings. That attempt is preserved in
`06_v5_regularization_v1/failed_before_class_weight_key_fix`. The new adapter now
normalizes keys to integers without changing weight values. The synthetic
fit/reload test now exercises JSON-loaded weights, and all 38 revision tests
pass. No original model or cache file was edited.

A second repair concerns diagnostics only: applying sigmoid after concatenating
logits changed float32 rounding compared with original small-batch prediction.
The diagnostic now uses exactly the original sigmoid batch shape, with the same
strict tolerance. The predecessor source and identity are preserved. Thirteen
new checkpoints are reused; five removed temporary outputs restart with identical
seeds and settings. Their first-attempt logs remain available for comparison.
Failed worker outputs are now retained rather than deleted. No checkpoint or
graph is silently rewritten to accommodate the repair.

An independent saved-prediction review and same-seed figure can be generated
after completion, then verified without training:

```bash
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/05_review_regularization.py
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/05_review_regularization.py --verify-only
```

## Completed results

Values are three-seed means on the same 1,104 validation CVEs. AP and F1 are
higher-is-better; Brier and log loss are lower-is-better. F1 uses the unchanged
0.5 cutoff. Norm is the mean subtype projection norm at selected checkpoints;
zero indicates a disabled subtype branch.

| Model | Decay | AP | F1 | Brier | Log loss | Subtype norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original node mean | 0.0001 | 0.296638 | 0.272766 | 0.078711 | 0.362941 | 0 |
| Original node mean | 0.001 | 0.295859 | 0.301163 | 0.085959 | 0.422165 | 0 |
| Original node mean | 0.01 | 0.305847 | 0.284254 | 0.082494 | 0.448871 | 0 |
| Subtypes + node mean | 0.0001 | 0.286329 | 0.276958 | 0.070785 | 0.514188 | 4.9023 |
| Subtypes + node mean | 0.001 | 0.293537 | 0.323036 | 0.088470 | 0.444369 | 4.5501 |
| Subtypes + node mean | 0.01 | 0.293179 | 0.280434 | 0.091266 | 0.526151 | 5.0692 |
| Subtypes + balanced mean | 0.0001 | 0.281423 | 0.303865 | 0.077889 | 0.362600 | 3.5771 |
| Subtypes + balanced mean | 0.001 | 0.283897 | 0.317529 | 0.090075 | 0.379776 | 3.6396 |
| Subtypes + balanced mean | 0.01 | 0.288057 | 0.291506 | 0.092718 | 0.469460 | 3.7626 |

There is no overall fix. Subtype node mean at 0.001 reduces mean log loss and
norm, but worsens Brier in every seed; its AP gain comes only from seed 43.
Original node mean at 0.01 has the highest mean AP here, but only seed 43 improves
AP and all three seeds worsen log loss. Subtype balanced mean at 0.01 improves
AP in all seeds, but worsens mean F1 and both probability-error metrics.
Every stronger-decay condition worsens mean Brier relative to its own control.
The subtype/node-mean extreme false-negative count falls from 20.67 to 8.00
per seed at 0.001, but this does not establish calibrated probabilities or
subtype-specific causality. Do not promote a nominal winner or extend the grid.

The separate [full review](../../../05_results/06_tpg_revision/07_v5_regularization_review_v1/REPORT.md)
includes all seeds, unchanged baselines, probability extremes, selected epochs,
recovery details and a paired comparison figure. TXT and JSON accompany it.
Five technical restarts reproduce every recorded epoch from the original
attempt; the other thirteen new checkpoints and nine controls were retained.
Next: a separately declared D-versus-DG source study with relevance and
class-wise coverage checks, not simultaneous extraction or threshold tuning.
