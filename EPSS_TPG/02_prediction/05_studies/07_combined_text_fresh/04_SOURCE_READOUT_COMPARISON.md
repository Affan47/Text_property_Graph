# CT03: A Controlled Source-Readout Comparison

Full training is complete and checkpoint/report replay passed. Read
[the full results and review](05_SOURCE_READOUT_RESULTS.md) before proposing
new fits. The design below remains the original specification. For exact
linear-control replay, set `OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4`.

## Decision and Scope

Keep CT01/CT02 and all six completed MAE source variants unchanged. CT03 is
one bounded development comparison on the older dataset: 2,546 training and
1,104 validation CVEs. No test records are adapted, scored or used for tuning.
The target remains the observed Day-60 EPSS score. Current text and CVSS are
retrospective inputs, not a verified historical forecasting experiment.

The new evidence supports testing source readout, not calling it a repair
that is guaranteed to improve prediction. The official comparison below is
pooled text only. It does not train a GNN, optimize the existing GNN loader,
rebuild TPGs or regenerate SecureBERT embeddings. Those are separate decisions
after this bounded comparison. Reddit remains a later protocol, not an
automatic expansion of this repeatedly inspected development experiment.

## What the External Pilot Verified

The supplied controlled pilot contains 36 frozen neural checkpoints and saved
per-CVE predictions. All 36 checkpoints reproduced their predictions within
the declared 1e-7 tolerance; saved metrics and parameter counts also matched.
The six logistic metric sets reproduced from saved probability outputs,
including the reported M ranking result. The logistic estimators/scalers were
not supplied, so their fitted parameter states were not replayed or refitted.
Four outcome-blind sampled DM graphs reproduced the supplied source means and
token counts; this is not a complete extraction audit of the pilot's NPZ file.

Equal parameter counts support a narrower statement than "capacity is not the
cause": the benefit survives one equal-count configuration. Splitting 32 text
channels into two groups of 16 still changes projections and layer-normalization
groups. Similarly, combined versus separate embeddings are not numerically
equivalent: their reported matched-width MSE RMSE is approximately 0.0988 versus
0.1010. Neither result rules out those factors in every architecture.

No pilot checkpoint is promoted into CT03. Every official neural fit starts
fresh with the locked seed and objective. The pilot remains additional
exploratory use of the same validation set, not independent confirmation.

## Exactly What Is Compared

Both arms use the same observed TOKEN rows from each existing combined DM
graph, the same frozen contextual embeddings, the same structured values,
the same source-presence masks and the same CVEs.

```text
Existing combined Description + social-media TPG
                 |
Frozen SecureBERT token rows, linked to node/document/source identity
                 |
        +--------+------------------+
        |                           |
MERGED ARM                     SOURCE ARM
Mean of all observed tokens    Mean D tokens       Mean M tokens
        |                           |                  |
768 -> 32                      768 -> 16           768 -> 16
        |                           +--------+---------+
        |                               concatenate
        +---------------------------+-------+
                                    |
                         32-value text representation
                                    |
                  Concatenate: structured branch (32)
                           and source masks (4)
                                    |
                    Common 68 -> 32 -> 1 sigmoid head
                                    |
                         Estimated Day-60 EPSS
```

Each arm has exactly **28,705 parameters** with the current 53-value structured
input. Common structured/head modules initialize identically for each seed.
The merged arm passes exact same-weight forward parity with CT02. The source
arm has a different text projection layout; it is not claimed to have the
same initial predictions or optimization trajectory as the merged arm.

The GLOBAL DOCUMENT node and all non-TOKEN nodes are excluded from both pooled
readouts. A missing source has a zero representation after its projection,
so its bias does not create an artificial source vector. The original four
source flags retain D/M/G/N identity; G/N are absent in this DM study.
Embedding availability is recorded separately from text presence.

Source means are extracted from the **combined DM embeddings**, not separately
contextualized D/M graphs. No token truncation or longest-post deletion is
introduced. The extractor reads compact embedding rows and source metadata
without expanding the 88-dimensional edge-feature matrix or GNN input tensors.
Every full merged vector must exactly match its locked CT02 pooled counterpart.
Document indices and source masks are checked for every included TOKEN node.

## Fixed Training and Evaluation Rules

| Item | Locked choice |
|---|---|
| Neural fits | 2 readouts x 2 objectives x 3 seeds = 12 |
| Primary objective | Raw-score MSE, selected on validation RMSE |
| Secondary objective | Raw-score MAE, selected on validation MAE |
| Seeds | 17, 29, 43 |
| Optimizer | AdamW; learning rate 0.001; weight decay 0.0001 |
| Batch and budget | 8 CVEs; at most 100 epochs; patience 15 |
| Weighting | Equal CVE; per-CVE gradient accumulation |
| Checkpoint selection | Earliest strict improvement among trained epochs |
| Checkpoints | One best checkpoint per fit, including optimizer/RNG for resume |
| Primary contrast | Source minus merged RMSE under MSE training |
| Secondary outputs | MAE, R-squared, correlations, signed bias, tail errors, AUROC/AP and top-k hits |
| Reporting partitions | Existing missing-source, overlap, version and target-range slices |
| Paired intervals | 1,000 CVE resamples at seed 6029; conditional on the saved three fits |
| Test | Disabled; no release-test CLI option |

Ranking uses `actual Day-60 EPSS >= 0.1` as the high-score group. This is not
exploitation or VulnCheck membership. Top-k ties are resolved by CVE ID and
boundary tie counts are reported. AUROC/AP intervals use paired CVE samples
with both classes; secondary comparisons are exploratory and unadjusted.
Repeated validation use prevents an independent-confirmation claim even when
a new interval excludes zero. The shuffle schedule remains `seed + epoch` to
retain the CT02 arithmetic control; this is not a new shuffle experiment.

## Linear and Constant Controls

There are twelve controls: training mean, training median, five ridge
regressions and five logistic classifiers. Ridge and logistic use the same
five feature sets: S, D, M, DM and D_M. S is the structured-only input. All
other sets add their declared text vector(s), source flags and observed-value
flags to the structured input. D/M are split from the combined graph here,
not standalone-context embeddings. DM is the single merged mean; D_M is the
concatenation of source means.

- Ridge estimates the EPSS score, with fixed alpha 1.0. Both raw and clipped
  score errors are reported.
- Logistic estimates the probability that Day-60 EPSS is at least 0.1, with
  fixed C=0.01 and a maximum of 2,000 iterations. These outputs are explicitly
  named `probability_high_epss`, never EPSS estimates. Ranking, Brier, log loss
  and F1 at the fixed 0.5 probability threshold are reported separately.
- All scaling and coefficients are fitted on training CVEs only. A convergence
  warning fails the run rather than silently accepting an unfinished solver.
- Linear coefficients, scaler state, training-ID hashes and per-CVE predictions
  are saved, making future replay possible without fitting the estimators again.

The tiny smoke training cohort has no high-score class. Its logistic controls
therefore use an explicitly labelled `SMOKE_SINGLE_CLASS_NO_CLASSIFIER_FIT`
constant for I/O testing. Synthetic tests exercise an actual two-class fit.
Full classifiers refuse a one-class training cohort. Smoke scores are not
evidence of predictive quality and must not be used to tune settings.

## Files and Their Roles

All locations below are relative to `EPSS_TPG`.

- `02_prediction/01_package/epss/combined_readout_v1/`: separate adapter,
  model and runner. Existing training packages are not edited.
- `02_prediction/04_configs/07_fresh_studies/03_source_readout_v1.json`:
  the locked experiment settings and interpretation.
- `02_prediction/03_analysis/07_fresh_studies/05_run_source_readout.py`:
  plan, vector preparation, readiness, training and verification CLI.
- `02_prediction/03_analysis/07_fresh_studies/06_verify_source_readout_pilot.py`:
  external checkpoint and prediction verification; never executes its training loops.
- `01_tpg/05_workspace/05_source_readout_v1/02_previous_multisource/01_securebert2/`:
  `smoke`/`full` derived vector caches. Each CVE row contains only predictor
  vectors, source masks/counts and provenance, not EPSS targets. Target values
  remain in the existing separately locked label source.
- `05_results/08_fresh_studies/09_source_readout_v1/02_previous_multisource/01_securebert2/`:
  `pilot_review_v1`, frozen `protocol/full`, and official `smoke`/`full` outputs.

Within each completed official scale, read `report/REPORT.txt`,
`report/findings.json`, `report/seed_mean_metrics.csv` and
`report/paired_comparisons.csv`. The plot `report/validation_errors.png` labels
the objective/readout on x and mean score error on y. Per-fit directories hold
`best_checkpoint.pt`, `history.json`, `evaluation.json`, `predictions.csv` and
hash locks. `controls/` holds training-fitted states and separate predictions
for each reference. Progress bars cover extraction, parity, controls, fits,
epochs, optimizer batches and paired bootstrap calculations.

## CPU Commands, One at a Time

Smoke and full vector preparation, all twelve full neural fits, the twelve
controls and fresh-process report replay are complete. The commands below
document the original sequence; use verification commands for completed
artifacts. No old checkpoint or graph needs to be deleted. The historical
implementation-time free-space estimate is not a current capacity measurement.

```bash
cd /home/ayounas/Text_property_Graph
```

```bash
conda activate CodeBERTFusion
```

```bash
RUNNER=EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/05_run_source_readout.py
```

Verify the smoke results already produced:

```bash
python "$RUNNER" --stage train --scale smoke --verify-only
```

Verify the saved full protocol:

```bash
python "$RUNNER" --stage plan --scale full --verify-only
```

Extract the full vectors once, using eight workers with four CPU threads each:

```bash
python "$RUNNER" --stage vectors --scale full --workers 8 --resume
```

Independently reproduce the full vector extraction and merged-vector parity:

```bash
python "$RUNNER" --stage vectors --scale full --workers 8 --verify-only
```

Check the full training gate:

```bash
python "$RUNNER" --stage ready --scale full
```

Run twelve neural fits and the twelve controls. Three neural workers use four
CPU threads each; the controls run first. No GPU is required or enabled.

```bash
python "$RUNNER" --stage train --scale full --jobs 3 --resume
```

Replay the saved best checkpoints and linear states in a fresh process:

```bash
python "$RUNNER" --stage train --scale full --jobs 3 --verify-only
```

If interrupted, repeat the same training command with `--resume`. Completed
fits are preserved; partial fits replay deterministically after their best
checkpoint. Do not change the settings or pinned code during a run. This
release is CPU-only; moving to GPU requires a separately verified protocol.
