# CT05: Description versus Description Plus Posts

Status update: both full runs are complete and verified. See the
[completed-results review](08_TRAINING_ONLY_POOLED_RESULTS.md). The specification
and implementation-time smoke record below remain unchanged.

## What the Completed Bootstrap Audit Establishes

The CT03 precision audit changes the strength of the baseline comparison, not
any saved prediction. At seed 6029 with 20,000 paired CVE resamples:

| Comparison | RMSE difference | 95% interval | Interpretation |
|---|---:|---:|---|
| Merged MSE minus training mean | -0.007987 | [-0.016669, +0.000295] | Improvement not established by this interval |
| Per-source MSE minus training mean | -0.012869 | [-0.024099, -0.003243] | Below zero under this conditional procedure |
| Per-source MSE minus merged MSE | -0.004882 | [-0.012224, +0.001822] | No established advantage between these readouts |

Negative RMSE differences favor the first model; units are EPSS score units.
With bootstrap seeds 0 through 199 and 1,000 resamples each, the merged-minus-mean
upper endpoint is below zero in 13/200 cases, versus 200/200 for per-source minus
mean. Per-source minus merged includes zero for all 200. The external review's
18/200 used an unspecified seed list; it is not our reproducible count. Fractions
of bootstrap seeds measure numerical sensitivity, not the probability an effect
is real. More draws do not repair repeated validation use or refitting uncertainty.

All original CT03 predictions, checkpoints and intervals remain unchanged.
CT03 did not answer whether posts improve on a description encoded by itself.
CT05 tests that question without using official validation scores for another fit.

## Architecture

```text
D:  standalone description TOKEN mean -----------------------+
                                                            | choose one arm
DM: merged description + accepted posts TOKEN mean -----------+
                                                            |
                                             linear -> LN -> GELU (32)
                                                            |
fold-fitted CVSS/reference features -> 32 features -----------+-> concatenate
                                                            |
                                         D/M/G/N flags -----+
                                                            |
                                            32-unit head -> sigmoid EPSS
```

Both arms include the same structured branch and parameter count within a fold.
D uses standalone D context, not D tokens embedded alongside posts. Its source
flags are always `[1,0,0,0]`; DM permits only D and accepted M. G/N summaries are
excluded. The question concerns accepted posts, not every raw Reddit attachment.

This is pooled-text training, not GNN training. Existing frozen SecureBERT TOKEN
vectors supply the means. No graph or embedding is regenerated. Hash-verified
CT02 means can be reused; smoke independently reduces every selected compact
cache and checks exact equality. Full preparation reads each required mean once,
so graph loading is not repeated during epochs.

## Nested Roles and Dates

Official train/validation/test IDs stay unchanged. Only original training CVEs
are used. The three CT04 evidence-grouped chronological folds are retained;
the unpurged alternative is not trained. The date is the **Day-60 target score
date**, not the CVE-ID year or a post date.

Inside each CT04 fitting set, the latest 20% by date, keeping ties together,
becomes checkpoint selection. Earlier optimization CVEs that share an exact-text
component with selection are purged. No optimization or selection component
touches assessment. Shared evidence does not mean distinct CVEs are duplicates.

| Dataset | Fold | Optimization CVEs | Selection CVEs | Assessment CVEs | Additional optimization purge |
|---|---:|---:|---:|---:|---:|
| Older | 1 | 703 | 237 | 512 | 2 |
| Older | 2 | 1,009 | 259 | 508 | 15 |
| Older | 3 | 1,393 | 355 | 511 | 24 |
| Reddit | 1 | 7,605 | 1,914 | 4,832 | 15 |
| Reddit | 2 | 11,389 | 2,924 | 4,781 | 47 |
| Reddit | 3 | 15,031 | 3,816 | 4,935 | 56 |

Optimization fits weights and preprocessing. Selection chooses the earliest
strict best RMSE checkpoint. Assessment is scored only after fitting and never
chooses epochs or ridge penalties. There is **no refit after selection**.

OOF means one out-of-fold assessment prediction per covered CVE and seed:
**1,531 older CVEs (87 scores >= 0.1)** and **14,548 Reddit CVEs (413 >= 0.1)**.
Initial warmup CVEs cannot be assessed without future-data training. Do not claim
OOF coverage of all 132 older training high-score CVEs. Older fold 3 has only two
high-score cases: retain it, but treat its ranking metrics as weak evidence.

Shared development JSON containers are decoded for integrity and projected to
training IDs; official validation values/features do not enter these fits or
metrics. No test source is opened. Text and CVSS are retrospective snapshots;
chronological target splitting does not establish historical text availability.

## Fixed Settings

- CPU; numerical libraries and PyTorch pinned to four threads inside the CLI.
- Raw-score MSE, equal-CVE batches of eight, width 32, no dropout.
- AdamW: learning rate 0.001, weight decay 0.0001, 100 epochs maximum, patience 15.
- Seeds 17/29/43: 18 neural fits per dataset, matched batch orders between arms.
- Seven control predictors per fold: optimization mean/median, structured ridge,
  fixed D/DM ridge and separately selection-tuned D/DM ridge.
- Fixed ridge alpha 1; selection-only grid `[0.1,1,10,100,1000]`, smallest alpha
  wins ties. This is nested chronological holdout tuning, not multi-fold alpha CV.
  Scalers and coefficients use optimization only. No refit after selection.
- Only `best_checkpoint.pt` per fit, including optimizer/RNG state. Resume starts
  there and checks replay of previously recorded post-best epochs.
- Verification reloads neural checkpoints and saved linear coefficients, without fitting.

Primary contrast: DM minus D pooled OOF RMSE, mean of three seed-wise differences,
not seed-ensemble performance. Report each fold too. Use **20,000** paired
resamples at seed 6029: CVE-level and exact-evidence-component-level. Component
sampling retains all members together. Intervals condition on fitted models.

Secondary metrics: MAE, R-squared, Spearman, Pearson, bias, raw/clipped error,
AUROC/AP for scores >= 0.1 and top-10/25/50/100 hits. The 0.1 threshold describes
high EPSS, not exploitation or VulnCheck listing. M-present/M-absent and score-tail
partitions are exploratory. Pooled OOF ranking compares score scales from different
fold models. No threshold tuning or subgroup winner selection is allowed.

These previously studied training records are not independent confirmation.
The motive is a protected source comparison, not a better-looking interval.
TPG expansion waits for this result. Smoke outcomes are wiring checks only.

## Files

- Config: `02_prediction/04_configs/07_fresh_studies/06_training_only_pooled_mse_v1.json`.
- Package: `02_prediction/01_package/epss/training_only_pooled_v1/`.
- Results: `05_results/08_fresh_studies/11_training_only_pooled_mse_v1/{01_reddit,02_previous_multisource}/`.
- Means: `01_tpg/05_workspace/06_training_only_pooled_mse_v1/{01_reddit,02_previous_multisource}/`.
- Outputs: `full/report/REPORT.txt`, `findings.json`, `oof_predictions.csv`,
  `metrics.csv`, `paired_intervals.csv`, `fold_rmse.png`, and SHA-256 artifact locks.
- `protocol/` stores all roles; `full/prepared/fold_N/` separates predictors,
  labels and preprocessing state; `full/fits/` stores best-only checkpoints.

The first smoke attempt, stopped by numerical-library identity ordering before
neural fits, is retained in sibling `_implementation_attempt_01` directories.
Runtime identity now sorts library entries but still requires exact versions,
thread settings and file paths. No completed historical run was changed.

## CPU Commands, One at a Time

Implementation verification completed: 26 focused tests passed (13 CT04 and
13 CT05); both protocols and smoke preparations reproduced; 18 smoke neural
fits per dataset and all linear controls passed fresh-process replay. All 97
older and 64 Reddit smoke graph means matched fresh compact-cache reductions.
Each fold used 16 optimization, 8 selection and 8 assessment smoke CVEs with
two epochs. These small outcomes are not used to tune settings. Full fitting
has not been launched.

Protocol SHA-256 values:

- Older: `63b2a8cc39520884cf35e55a47b99a5b4a96c44fc183cb2838889ea16a231ff2`.
- Reddit: `76ff5ad01d652a389a6726115b5a71ff9278ba3f1bf5a6a213b8d44015bde2e2`.

Smoke report lock SHA-256 values:

- Older: `5321a3da4c95da8609f38243c71455da900d3fb3c5df991bb6f8028545176b98`.
- Reddit: `2f7f5dc42c7ee23596f431b4f1cb69edb93289f94e100bf2660751d881a0dae8`.

```bash
cd /home/ayounas/Text_property_Graph
```
```bash
conda activate CodeBERTFusion
```
```bash
RUNNER=EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/10_run_training_only_pooled.py
```
```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/07_fresh_studies -p 'test_training_only_pooled.py'
```
```bash
python "$RUNNER" --dataset both --stage plan --verify-only
```
```bash
python "$RUNNER" --dataset both --stage prepare --scale smoke --verify-only
```
```bash
python "$RUNNER" --dataset both --stage train --scale smoke --jobs 3 --verify-only
```
```bash
python "$RUNNER" --dataset previous --stage prepare --scale full --workers 8 --resume
```
```bash
python "$RUNNER" --dataset previous --stage train --scale full --jobs 3 --resume
```
```bash
python "$RUNNER" --dataset previous --stage train --scale full --jobs 3 --verify-only
```
```bash
python "$RUNNER" --dataset reddit --stage prepare --scale full --workers 8 --resume
```
```bash
python "$RUNNER" --dataset reddit --stage train --scale full --jobs 3 --resume
```
```bash
python "$RUNNER" --dataset reddit --stage train --scale full --jobs 3 --verify-only
```

Up to eight CPU fits may run concurrently if resource checks pass. Changing
`--jobs` does not alter mathematical settings. Model/code changes require a new
protocol; do not bypass identity locks. Full training is left for the user.
