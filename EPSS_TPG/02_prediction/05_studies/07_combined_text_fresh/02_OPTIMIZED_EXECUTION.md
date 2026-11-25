# CT02: Faster Execution Without Changing the MAE Experiment

## Decision and Current Status

The original `combined_training_v1` package and its running `baseline_v2`
experiment are unchanged. Do not delete its checkpoints: they are reference
evidence and the new runner can reuse completed fits without copying or
retraining their models. The active incomplete fit is not imported. If it is
stopped, its best checkpoint remains available to the original runner; the
new runner starts that incomplete fit afresh.

At the storage inspection on 2026-11-12, approximately 307 GiB was available.
The older full run's checkpoints occupied only 4.34 MiB. Deleting them would
save negligible space and break reference verification. Nothing was deleted
or stopped by this implementation.

Implementation and smoke verification are complete. Full pooled-vector
preparation, six-checkpoint full-D parity and full CT02 training are left for
the user to run. Do not interpret smoke metrics as research results.

## What Changes

1. For the **pooled control**, the exact mean of observed TOKEN embeddings is
   computed once per combined graph. A small vector is read during training,
   rather than repeatedly expanding a whole graph and its edge features.
   This uses the original frozen SecureBERT embeddings and original reduction
   expression. It does not rerun the language model, parser or graph builder.
2. For the **TPG model**, graph tensors, relation IDs, all 88 edge features,
   node types, subtype channels and message-passing arithmetic are unchanged.
   The default per-worker graph cache grows from 1 GiB to 8 GiB. The cache is
   an LRU, not a promise that every large source view fits in memory.
3. Independent fits run in separate processes with per-fit locks. The default
   is two jobs, four CPU threads each. A CPU-affinity and conservative RAM
   check runs first. A coordinator lock protects report publication. Process
   count is an execution choice; it does not change each fit's identity or RNG.
4. Each fresh fit saves just **one best checkpoint**. Epoch histories and an
   execution profile remain available. Resume restores the best checkpoint
   and reproduces subsequent recorded epochs, rather than keeping another
   model file for every epoch.
5. Completed original MAE fits can be linked through `reference.json` only
   after exact identity and prediction checks. Original checkpoint and artifact
   hashes are recorded. These are references, not renamed fresh CT02 fits.

The CPU smoke comparison covered 39 fits per dataset, all six views and seeds
17, 29 and 43. All 78 MAE training histories, including gradient-norm summaries,
and prediction CSV bytes equaled the original smoke artifacts. Fresh-process
verification replayed the parallel fits sequentially. Unit tests additionally
check exact parameter updates, missing embeddings, vocabulary guards, input
separation, interrupted resume, locking and best-checkpoint selection.

## Objective Pair

This version supports a bounded, explicitly separate comparison:

| Run | Training loss | Best checkpoint selected by | Interpretation |
|---|---|---|---|
| `mae_fast_v1` | Absolute error on the original EPSS scale | Lowest validation MAE; earliest epoch wins ties | Performance-preserving continuation of the MAE study |
| `mse_fast_v1` | Squared error on the original EPSS scale | Lowest validation RMSE; earliest epoch wins ties | A different objective that puts more weight on large errors |

Both retain hidden size 32, two graph layers, original AdamW settings,
three seeds, identical cohorts and 100 epochs with patience 15. Neither uses
log/logit targets. MSE cannot import an MAE checkpoint. MSE is not a bug fix or
a promised improvement: it can improve RMSE while worsening MAE.

Report MAE, RMSE, R-squared, Spearman, Pearson, signed bias and the existing
high-score (actual EPSS >= 0.1) partition for both objectives. Existing
per-CVE CSVs make prediction spread and ranking analysis inspectable.

The planned main contrasts are TPG versus pooled within each available view:
six contrasts in the older dataset, three in Reddit, separately for each
objective. Existing paired intervals are **unadjusted and exploratory**; this
release does not claim multiplicity-corrected confirmatory inference. Repeated
validation inspection and retrospective text/CVSS remain limitations. It does
not reset past test exposure or prove temporally leakage-free forecasting.

## Deliberately Unchanged

- The shuffle remains `seed + epoch` for exact MAE parity. Some seed/epoch
  combinations share an order, but changing this during a performance-only
  comparison would change the experiment. A different seed schedule requires
  its own protocol and cannot reuse these checkpoints as fresh fits.
- No graph readout, attention, dropout, subtype or source-filter changes.
- No sparse message-passing rewrite without numerical and gradient testing.
- No GPU training in this parity release. CPU and GPU results are not assumed
  bit-identical. A failing NVML query alone does not prove CUDA is unusable.
- Reddit U/DU/DMU full comparisons remain blocked by zero accepted validation
  webpage summaries. The small smoke tests still exercise all six input views.
- No VulnCheck classifier or catalogue predictor is added by CT02.
- No test records are adapted or scored.

## File Map

All paths below are relative to `EPSS_TPG`.

| Location | Purpose |
|---|---|
| `02_prediction/01_package/epss/combined_training_v2/pooled.py` | Frozen TOKEN means, source hashes and representation-specific input store |
| `02_prediction/01_package/epss/combined_training_v2/training.py` | MAE/MSE fits, resume, best-only saving and original-checkpoint replay |
| `02_prediction/01_package/epss/combined_training_v2/runner.py` | Per-fit scheduling, resource gate, smoke qualification and reports |
| `02_prediction/03_analysis/07_fresh_studies/02_run_combined_training.py` | New CLI; the old CLI remains unchanged |
| `02_prediction/04_configs/07_fresh_studies/02_combined_training_v2.json` | Versioned execution overrides on the frozen base settings |
| `01_tpg/05_workspace/04_training_acceleration_v1/` | Small per-dataset, per-CPU-thread-count pooled-vector caches |
| `05_results/08_fresh_studies/08_combined_training_v2/` | New dataset/encoder/run/scale/fit output hierarchy |

Within each run, `report/metrics.csv` contains metrics and
`report/paired_comparisons.csv` contains development comparisons. Each fit has
`predictions.csv`, and either `best_checkpoint.pt` or `reference.json` pointing
to a verified original model. Fresh fits also have `history.json` and
`execution_profile.json`. Graph-load time, loads, hits, cache bytes and epoch
times are recorded. Timing is not compared for exact replay.

## Handover Commands: Older Dataset, CPU

Run individually in the project directory. These commands keep the original
MAE task and reuse completed original fits after verification.

```bash
cd /home/ayounas/Text_property_Graph
```
```bash
conda activate CodeBERTFusion
```
```bash
RUNNER=EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/02_run_combined_training.py
```

The smoke has already been run. Recheck its saved checkpoints and parity:

```bash
python "$RUNNER" --dataset previous --stage train --scale smoke --reference-run baseline_v2 --verify-only
```

Compute the small pooled cache once. This reads the existing full graph cache
and can take time; it does not generate graphs or train models. It resumes
atomic completed rows automatically.

```bash
python "$RUNNER" --dataset previous --stage vectors --scale full --resume
```
```bash
python "$RUNNER" --dataset previous --stage vectors --scale full --verify-only
```

Replay all six frozen full-D checkpoints on the new input path. Every saved
validation prediction and CSV must match exactly. This is not another full
training job and does not access the test set.

```bash
python "$RUNNER" --dataset previous --stage parity --scale full --reference-run baseline_v2
```
```bash
python "$RUNNER" --dataset previous --stage parity --scale full --reference-run baseline_v2 --verify-only
```
```bash
python "$RUNNER" --dataset previous --stage ready --scale full --reference-run baseline_v2
```

**Only after these checks pass**, stop the original foreground run with
`Ctrl+C`, preferably at a completed-fit boundary. Do not remove its output
directory or use `kill -9`. A partially completed original fit is preserved
but is not imported into the new run.

```bash
python "$RUNNER" --dataset previous --stage train --scale full --reference-run baseline_v2 --reuse-completed --jobs 2 --resume
```
```bash
python "$RUNNER" --dataset previous --stage train --scale full --reference-run baseline_v2 --jobs 2 --verify-only
```

Do not add `--reuse-completed` to verification: verification must not import
new fits. Keep `--reference-run baseline_v2` for the exact full reference gate.

## Reddit and the Separate MSE Comparison

The same architecture handles both datasets. Reddit has no completed original
full-D checkpoint set to import. Its full run is qualified by the original
runner's matched smoke histories, not a claimed full-D parity result.

```bash
python "$RUNNER" --dataset reddit --stage train --scale smoke --reference-run baseline_v2 --verify-only
```
```bash
python "$RUNNER" --dataset reddit --stage vectors --scale full --resume
```
```bash
python "$RUNNER" --dataset reddit --stage ready --scale full
```
```bash
python "$RUNNER" --dataset reddit --stage train --scale full --jobs 2 --resume
```
```bash
python "$RUNNER" --dataset reddit --stage train --scale full --jobs 2 --verify-only
```

MSE smoke runs have been completed separately for both datasets. After the
MAE handover and before launching another objective, verify the desired one:

```bash
python "$RUNNER" --dataset previous --stage train --scale smoke --objective mse --run-name mse_fast_v1 --verify-only
```
```bash
python "$RUNNER" --dataset previous --stage ready --scale full --objective mse --run-name mse_fast_v1
```
```bash
python "$RUNNER" --dataset previous --stage train --scale full --objective mse --run-name mse_fast_v1 --jobs 2 --resume
```
```bash
python "$RUNNER" --dataset previous --stage train --scale full --objective mse --run-name mse_fast_v1 --jobs 2 --verify-only
```

Use `--dataset reddit` for its corresponding MSE commands, after preparing its
full pooled cache. Do not run multiple full coordinators merely because 88
cores are available. Large graph caches and storage bandwidth are shared.
Inspect execution profiles first. `--jobs` may change without changing fit
arithmetic; changed cache capacity or training settings require a fresh named
run and matching smoke. No full CT02 fits were launched during implementation.
