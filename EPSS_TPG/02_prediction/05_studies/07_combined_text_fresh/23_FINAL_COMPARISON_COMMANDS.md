# FC01 Final Two-Run Comparison

**Current status, 17 November:** the corrected integration gate and all 75
Run-1 fits are complete. None of the six primary comparisons establishes an
RMSE improvement under the frozen interval rule. Do not promote a point winner
or change settings. Next is the already declared Run-2 official-development
preparation and 25 fits (section 5 below), followed by the guarded test release.
Review: `30_fc01_run1_review_v1/review/`. To reproduce its read-only analysis:

```bash
python EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/35_review_final_run1.py --verify-only
```

**Loader correction verified, 16 November:** the authorized integration job was
stopped before editing its pinned code. Four completed one-epoch fits (structured,
pooled D, pooled DM and TPG D) are preserved. Each of the nine predictor arrays
is now decoded once, with a progress bar, instead of once per CVE row.
All three full-fold inputs are byte-identical and all four saved checkpoints
reproduce their prediction CSVs exactly. Scientific settings remain unchanged.
New protocol hashes record the runtime correction; old locks were not rewritten.
The five-row integration gate was subsequently restarted and completed before Run 1.
Evidence: `29_fc01_loader_fix_v1/review/`; the earlier diagnosis remains in
`28_fc01_startup_review_v1/review/`.

Implementation of the [final decision](22_FINAL_TWO_RUN_DECISION.md). No new
target search, summary arm, loss search, catalogue fit or split selection.
Scientific training is user-run. The official test remains closed until the
explicit final release command below.

The protocol and Run-1 adapter are now prepared. Predictor vectors and fitted
transforms match CT07 exactly: 13,378 unique original-training CVEs and 14,817
reused graph objects. The largest conservative input-tensor working set is
11.995 GiB. The commands below use a 32-GiB allowance with two fit processes;
this passed the current memory preflight. It is not a training-time estimate.
No real integration fit or scientific fit has been run by the implementation.

## What Runs

- Six reporting rows: training constants, structured, pooled D, TPG D,
  pooled DM, TPG DM. The training mean is primary; median is a no-fit diagnostic.
- Five seeds: 17, 29, 43, 59, 71. Raw-score MSE, AdamW 0.001, decay 0.0001,
  batch 8, 100 maximum epochs, patience 15, earliest best selection RMSE.
- Run 1: original CT07 main-policy folds; 8,053 assessment CVEs; 75 fits.
- Run 2: 24,213 training / 972 validation CVEs; 25 fits, then all models
  evaluated once on the 3,977 official test CVEs. No train+validation refit.
- One integration gate: all five learned rows, fold 1, seed 17, one epoch.
  No assessment scoring in this gate. Its weights are never reused by full fits.
- Same typed two-layer GNN, 13 node types, 23 edge types, 32 subtypes and
  88 edge properties from the frozen graph contract. SecureBERT stays frozen.
- D is standalone description; DM is combined description and accepted prose
  posts. No new graphs or embeddings for development. Missing posts stay masked.
- Structured features are fitted separately on each optimization set, and
  once on the official training set. EPSS/KEV outcomes are not model inputs.

All fits are CPU-only, four computation/BLAS threads per fit, pinned by the
launcher. `--jobs` controls independent fit processes. `--budget-gib` is the
resident input-tensor allowance per fit, not total machine RAM. Complete
working sets must fit; otherwise execution stops before fitting. No CVE,
token, edge or long graph is silently dropped. The RAM gate also reserves
space for activations, Python overhead and the operating system.

The GNN preloads optimization/selection tensors once, then reuses them across
epochs. Assessment is loaded only after selecting the best checkpoint. Pooled
models reuse hash-verified TOKEN means. Only the best checkpoint is saved,
including optimizer/RNG state. An interrupted fit resumes from that checkpoint
and exactly replays recorded later epochs; `--resume` is not an approximation.
Completed outputs never get retrained by resume.

## Directory Layout

```text
02_prediction/01_package/epss/final_comparison_v1/       implementation
02_prediction/04_configs/07_fresh_studies/11_final_comparison_v1.json
05_results/08_fresh_studies/27_final_comparison_v1/01_reddit/
  protocol/                         both runs and code/input hashes
  pooled_rows/                      reusable sufficient statistics
  prepared/run1/fold_{1,2,3}/        separate predictors, labels, transforms
  prepared/run2/fold_0/              official development only
  integration/fits/                 one-epoch gate checkpoints
  integration/gate/                 all-five completion receipt
  runs/run1/fits/                    75 best checkpoints
  runs/run1/report/                  within-version OOF comparison
  runs/run2/fits/                    25 best checkpoints
  runs/run2/report/                  official validation comparison
  release_lock.json                 all official checkpoints/transforms locked
  test_cache/                       only after release_lock exists
  test_evaluation_release.json      irreversible scientific release marker
  test_report/                      all fixed held-out results
```

Existing graph objects, source datasets, earlier checkpoints and reports stay
unchanged. Do not edit pinned implementation files while fits are running.
The first untrained FC01 export is retained in
`preserved/pre_interface_review_20261116/`. A report-interface correction and
additional release guards were tested before the final protocol was frozen;
no research fit existed in that predecessor.
`preserved/pre_release_report_lock_20261116/` retains the subsequent untrained
export before adding the development-report hashes to the final release lock.
`preserved/pre_npz_loading_fix_20261116/` retains the stopped job's code,
protocol, preparation and four completed integration checkpoints. They are
historical parity references, not checkpoints to copy into the new run identity.
The corrected gate starts fresh; its weights are not reused for scientific fits.
The new review resolves the old startup snapshot's source paths to these
preserved copies. Use the new verification command, not the superseded startup
diagnostic against current source paths.

## Commands, In Order

Run each command separately. These are CPU commands; no working CUDA driver is
required. The NVML initialization warning does not select or require a GPU.

```bash
cd /home/ayounas/Text_property_Graph
```
```bash
conda activate CodeBERTFusion
```
```bash
RUNNER=EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/32_run_final_comparison.py
```

### 1. Verify the implementation

Synthetic unit tests only; fixtures named test do not refer to held-out CVEs:

```bash
python -B -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/07_fresh_studies -p 'test_final_comparison*.py'
```

The protocol is frozen before any integration/scientific fitting. For the
already-prepared installation, verify it:

```bash
python "$RUNNER" --stage freeze --verify-only
```

On a clean installation without FC01 output, omit `--verify-only` once.
Never replace a frozen protocol to accommodate changed source code or data.
The documented loader correction preserved the complete predecessor before
publishing its successor; it did not change the scientific configuration.

Verify the correction and replay the preserved checkpoints without training:

```bash
python EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/34_verify_fc01_loader_fix.py --verify-only
```

### 2. Prepare and check Run 1

```bash
python "$RUNNER" --stage prepare --run run1 --verify-only
```

On a clean installation without this prepared adapter, omit `--verify-only`
once. Predictor and label files are separate; CT07 predictors must reproduce.

```bash
python "$RUNNER" --stage preflight --run run1 --jobs 2 --budget-gib 32
```

### 3. Run the only integration gate

```bash
python "$RUNNER" --stage integrate --run run1 --jobs 2 --budget-gib 32 --resume
```
```bash
python "$RUNNER" --stage integrate --run run1 --jobs 2 --budget-gib 32 --verify-only
```

The five full-fold one-epoch checks report real preload and epoch timings. They
do not select a loss, architecture, seed, readout or source policy. See each
`execution_profile.json` for memory/load timing. The 24-graph benchmark is not
used to promise a whole-run completion time.

### 4. Run all 75 within-version fits

```bash
python "$RUNNER" --stage train --run run1 --jobs 2 --budget-gib 32 --resume
```
```bash
python "$RUNNER" --stage train --run run1 --jobs 2 --budget-gib 32 --verify-only
```

Reports are published automatically when every fit is complete. Verification
reloads all best checkpoints, reproduces predictions and reruns the report.
No new optimization occurs with `--verify-only`.

### 5. Prepare and run official development

All five learned rows proceed regardless of Run 1's ranking.

```bash
python "$RUNNER" --stage prepare --run run2
```
```bash
python "$RUNNER" --stage prepare --run run2 --verify-only
```
```bash
python "$RUNNER" --stage preflight --run run2 --jobs 2 --budget-gib 32
```
```bash
python "$RUNNER" --stage train --run run2 --jobs 2 --budget-gib 32 --resume
```
```bash
python "$RUNNER" --stage train --run run2 --jobs 2 --budget-gib 32 --verify-only
```

### 6. Lock all official models, then build test predictors

This requires all 25 official fits and their fresh-process replay receipt.
It does not read test outcomes or choose a winner.

```bash
python "$RUNNER" --stage lock-test
```
```bash
python "$RUNNER" --stage test-cache --cache-stage inputs
```
```bash
python "$RUNNER" --stage test-cache --cache-stage graphs --workers 8
```
```bash
python "$RUNNER" --stage test-cache --cache-stage embeddings
```
```bash
python "$RUNNER" --stage test-cache --cache-stage verify
```

These immutable cache stages resume already completed objects automatically.
The encoder uses the exact saved development device/thread identity; it cannot
silently switch to GPU or another model. Structural/embedding failures stop
publication rather than removing difficult test cases. Test rows remain
labeled test in their receipts; no development role is forged to reuse code.

### 7. Release test outcomes once

This command scores all fixed rows and closes further fitting in this study:

```bash
python "$RUNNER" --stage evaluate --release-test --budget-gib 32
```

Verify the identical release, without retraining or changing models:

```bash
python "$RUNNER" --stage evaluate --release-test --budget-gib 32 --verify-only
```

If evaluation was interrupted, use the same command with `--resume` instead.
The existing release marker must match the same protocol/checkpoints/cache.
Crash recovery and deterministic replay are not a new scientific test.

## What the Reports Mean

`predictions.csv` contains per-CVE, per-seed predictions. `metrics.csv`,
`seed_metrics.csv`, `supports.csv`, `intervals.csv` and `baseline_differences.csv`
retain all declared rows and controls. Figures show RMSE and the six paired
family-adjusted differences. Primary uncertainty jointly resamples evidence
groups and matched seeds, 20,000 times at seed 6029; CVE resampling is sensitivity.
The six-contrast family has 95% nominal and 99.1667% Bonferroni intervals.
Metrics average seed-wise results; no prediction ensemble is silently substituted.

Ranking, version, post-availability, fold and existing validation-overlap slices
are descriptive. A same-direction test result is not automatically a supported
held-out gain. `cross_run_decisions.json` applies the fixed distinction. Baseline
differences remain descriptive, so an unadjusted favorable comparison cannot
be promoted as formal baseline superiority.

Run 1 has repeatedly examined cases. Run 2 is retrospective cross-version
transfer, not an early-warning forecast. Neither a weak result nor an interval
including zero proves all graphs or text useless. Whatever the outcome, retain
all rows, uncertainties and influential CVEs, write the final findings and stop.
