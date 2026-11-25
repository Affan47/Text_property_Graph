# CT07: Full Pooled Execution and Constant-Control Gate

## Completed Run

All **54 full fits** are now complete. A fresh process reproduced their selected
checkpoints, predictions, constant controls and 20,000-resample reports. Read
[the completed-results review](12_WITHIN_VERSION_FULL_RESULTS.md) before any new
experiment. The main DM-minus-D benefit is not established; the sensitivity
does not pass the constant-control reporting gate. No official validation or
test scoring was performed, and no GNN expansion is authorized by these results.

## Purpose and Boundaries

The full CT07 adapter is verified. This execution layer does not replace its
code, configuration, prepared inputs or 54 small smoke results. It adds the
full-training gate and freezes the constant-control comparisons before any
full CT07 neural fit. Older v3 remains audit-only. No official validation or
test scoring, GNN expansion, encoder training, ridge search or threshold tuning
is introduced.

This is retrospective within-v2 EPSS estimation on already inspected original
training records. It is an exploratory re-estimation, not new independent
evidence. The same-source text can postdate the target score. Having separated
labels and predictors does not establish historically available text or the
absence of all possible leakage.

## Fixed Experiment

Both `main` and `published_since_v2` use the exact CT07 prepared roles. The
second policy restricts only optimization to publication >=2022-02-04; it
changes sample size, recency, composition and preprocessing, not just target
version. Selection and assessment IDs and labels are the same between policies.

Each policy has three folds, three arms and seeds 17, 29 and 43: **27 fits per
policy, 54 full fits total**. All use optimization-fitted structured features.

| Arm | Text supplied | Post flag supplied |
|---|---|---|
| D | Standalone description mean | No |
| DF | Exactly the same description mean | Yes, no post text |
| DM | Combined description + accepted-post mean | Yes |

No model capacity, feature dimensions or source masks are changed. The pooled
model retains width 32, AdamW, raw-score MSE, batch size 8, learning rate 0.001,
weight decay 0.0001, maximum 100 epochs and patience 15. Selection uses earliest
strict best selection RMSE among trained epochs, never assessment error. There
is no refitting after selection. Best checkpoint only, full history and
optimizer/RNG state are retained. Interrupted fits resume from the best
checkpoint and replay subsequent recorded epochs deterministically.

The original fitting engine remains hash-pinned; its epoch messages still use
`[CT05]`. The launcher prints `[CT07-FULL]` with policy, fold, arm and seed, and
writes only to the new execution directory. Four numerical threads per fit are
set inside the CLI and workers. Jobs can run independently, with CPU/RAM budget
checks and per-fit locks. Job count does not change model settings or the shuffle.

## Constant Controls and Claims

For every policy and fold, compute the arithmetic mean and median of that
policy's **optimization** scores. Assign each constant to all assessment CVEs
in the fold. These are fold-specific constants, not one value fitted to the
combined assessment cohort. Neither selection nor assessment estimates them.

The pre-training snapshot records all constant values, their fit-ID hashes,
per-CVE predictions and RMSE/other descriptive metrics. The full runner must
reproduce these states and predictions exactly before fitting any neural model.

The primary contrast stays **main-policy DM minus D OOF RMSE**, averaging the
three seed-wise differences. Secondary comparisons are DM minus DF and DF
minus D. For each policy, also compare every arm against both of its own
constants on all CVEs, post-present CVEs and post-absent CVEs.

All these contrasts use 20,000 paired resamples, seed 6029, with both CVEs and
evidence components as resampling units. A constant's error vector is broadcast
across the three neural seeds; seeds are not treated as independent CVEs.
Shared bootstrap draws accelerate the calculation without changing the
existing estimator or its resampling procedure. Regression tests compare the
new batched calculation exactly against the old pair-at-a-time procedure.

Outcome language is fixed:

- If DM beats D but not the constants, report **relative improvement over D**,
  not useful estimation beyond a trivial predictor. A relative improvement is
  still a valid result; failing the constants does not erase that comparison.
- A conservative useful-estimation reporting gate requires DM-minus-D and
  DM-minus-both-constants upper RMSE bounds below zero under both interval units
  on the same population.
- Attribution specifically to post content, beyond post availability, also
  requires the DM-minus-DF comparison. Presence and content are different inputs.
- A subgroup or sensitivity result cannot replace an unsupported primary
  comparison. An interval crossing zero is inconclusive, not evidence that text
  contains no information.
- These are exploratory nominal 95% intervals, not family-wise adjusted tests.
  Requiring multiple favorable comparisons does not undo repeated dataset use.
  Intervals condition on fitted models and do not include retraining uncertainty.

The report includes raw/clipped error summaries, MAE, RMSE, R2, Pearson,
Spearman, signed bias, AUROC/AP for score >=0.1, and top-10/25/50/100 hits.
The 0.1 boundary is descriptive, not an exploitation label or a tuned decision
threshold. Report full OOF, each fold, post-present/absent and score >=0.1/<0.1
metrics. Ranking pooled over folds also compares different models' score scales.

## Frozen CT05 Reference

The reference snapshot uses the already-saved predictions for the exact 8,053
CT07 assessment CVEs, not new fits. Each CVE must be in its original CT05
assessment role, with no evidence component shared with that model's
optimization or selection roles. Target values and post masks must agree.
The snapshot retains the old-to-new fold crosswalk.

D and DM predictions must match the official CT05 OOF CSV byte-value for
byte-value. DF comes from the separately documented external flag-model probe;
its nine selected checkpoints, selection scores and histories were replayed
and hash-verified. It is explicitly an external exploratory reference, not an
original preregistered CT05 arm.

| Reference on CT07 assessment CVEs | All 8,053 | With posts: 834 | Without posts: 7,219 |
|---|---:|---:|---:|
| Main optimization median | 0.036641 | 0.088418 | 0.024383 |
| Main optimization mean | 0.037002 | 0.086970 | 0.025564 |
| Restricted optimization median | 0.036641 | 0.088418 | 0.024383 |
| Restricted optimization mean | 0.036393 | 0.087413 | 0.024386 |
| CT05 D | 0.039419 | 0.093089 | 0.027050 |
| CT05 external DF | 0.039819 | 0.091246 | 0.028372 |
| CT05 DM | 0.037585 | 0.083304 | 0.027767 |

Cells are RMSE, lower is better. Neural values average three seed-wise RMSEs,
not predictions averaged into an ensemble. Full-precision tables including DF
and paired intervals are generated directly from the frozen artifacts in
`protocol/`. The constant table demonstrates why policy-specific controls
matter: the restricted mean has lower overall error but higher post-present
error than the main mean.

This reference does not establish statistical superiority by point estimates
alone. CT05 used different fitting/selection cohorts and date boundaries;
comparing its errors with CT07 cannot isolate target-version restriction as
the cause. No reference number chooses new hyperparameters or checkpoints.
The former precision extrapolation is not used as a power calculation.

The fixed 20,000-resample evidence-component calculation gives historical
DM-minus-D overall RMSE difference -0.001833746 (95% interval -0.003810132 to
-0.000213515). Post-present DM-minus-DF is -0.007941860 (-0.014833858 to
-0.002403369). These reproduce the direction of the external analysis; all
full-precision values and resampling settings are saved rather than rounded
claims being treated as exact. On post-present cases, DM also beats both main
constants under the recorded grouped intervals. Overall, its higher RMSE than
the constants is a point-estimate result: the corresponding grouped intervals
cross zero. Do not describe overall inferiority as statistically established.

The crosswalk is also fixed: 2,680 CT07 fold-1 CVEs came from CT05 fold 2;
CT07 fold 2 combines 595 CT05-fold-2 and 2,083 CT05-fold-3 CVEs; CT07 fold 3
contains 2,695 CT05-fold-3 CVEs. This is why the reference is not a matched
change to training target version alone.

## Artifacts and Verification

Implementation checks: all 40 within-version regression tests passed. All 54
new smoke fits matched the original CT07 histories, selected weights and
predictions exactly. A fresh process replayed the new reports using one worker
after training smoke with three workers. The full preflight passed with three
CPU workers and four numerical threads per worker. At implementation handover,
no full fits had been launched; the later completed run is recorded above.
The worker also checks the frozen code/runtime before each fit, so editing code
between fits cannot silently create a mixed-identity run. The initial integration
outputs and source snapshot, before this additional guard, are preserved under
`14_within_version_training_v1_initial_smoke_preserved/`; they are not full runs.

New package: `02_prediction/01_package/epss/within_version_training_v1/`.
New config: `04_configs/07_fresh_studies/09_within_version_training_v1.json`.
New CLI: `03_analysis/07_fresh_studies/17_run_within_version_training.py`.
Results root: `05_results/08_fresh_studies/14_within_version_training_v1/01_reddit/v2/`.

- `protocol/`: fixed specification, policy-specific constant states/predictions,
  historical CT05 predictions, provenance hashes, fold crosswalk, metrics,
  20,000-resample intervals, TXT/JSON report and reference figure.
- `smoke/fits/`: 54 two-epoch integration fits under the new launcher. Their
  histories, selected weights and predictions must match the original CT07
  smoke exactly before full execution is allowed.
- `full/fits/{policy}/fold_N/{D,DF,DM}/seed_N/`: best checkpoint, history,
  identity, predictions and evaluation per completed fit.
- `{smoke,full}/report/`: canonical OOF predictions, metrics, arm/control
  intervals, reporting-gate checks, figure and TXT/JSON findings.

Training consumes the durable protocol snapshot, not files under `/tmp`.
Reconstructing the *historical snapshot* with `--stage protocol --verify-only`
still needs the original recorded external-probe files while they are present.
Existing protocol hash verification during training does not require those
external originals. No previous output is deleted or overwritten.

## CPU Commands, One at a Time

```bash
cd /home/ayounas/Text_property_Graph
```
```bash
conda activate CodeBERTFusion
```
```bash
RUNNER=EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/17_run_within_version_training.py
```

The new specification and smoke are prepared during implementation. Verify the
smoke (no retraining) and then check the full preflight:

```bash
python "$RUNNER" --stage train --scale smoke --jobs 3 --verify-only
```
```bash
python "$RUNNER" --stage ready --scale full --jobs 3
```

Launch the **54 full fits yourself**. Three simultaneous CPU fits use four
numerical threads each. Keep the terminal/session alive until completion.

```bash
python "$RUNNER" --stage train --scale full --jobs 3 --resume
```

Then replay saved checkpoints and regenerate the reports without retraining:

```bash
python "$RUNNER" --stage train --scale full --jobs 3 --verify-only
```

The original `15_run_within_version_pooled.py` deliberately still rejects full
training. Use the new launcher above; no caches need rebuilding. To resume an
interrupted full run, rerun its `--resume` command. Completed fits are replayed,
not trained again. A separate fresh process verifies the full run after it ends.

For a fresh workspace without these new outputs, first run `--stage protocol`,
then `--stage train --scale smoke --jobs 3`, then the verification/preflight above.
Smoke uses 20 bootstrap resamples only to test the report path, not for inference.

Focused regression suite:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/07_fresh_studies -p 'test_within_version*.py'
```
