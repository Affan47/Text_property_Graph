# DS01: Description-Only Surrogate

## Decision

Implement one bounded retrospective screening test: Reddit v2, CT07 main-policy
roles, three folds and ten fixed seeds, 30 scientific fits. No new historical
forecast claim, no additional NVD collection, no Reddit v1 arm and no graph fit.
The 7,957 assessment histories are complete; downloads are no longer the gate.
The remaining risks are historical wording, encoder/model dates, future-label
maturity and the limited number of risers, not unfinished requests.

Reddit v1 remains a separately specified option. Its 283 candidate assessment
risers come from the already inspected fixed-epoch support audit; its first
fitting blocks have very few negatives. It is not untouched confirmation and
cannot be appended to CT07 without a distinct design and support decision.

## Frozen Model and Rules

- Predictor: only the existing standalone description SecureBERT vector (768).
- Network: linear 768-to-32, LayerNorm, GELU, linear 32-to-32, GELU, linear
  32-to-1, sigmoid. 25,761 parameters; no structured branch or source mask.
- Day-60 labels remain separate from predictors. Raw-score MSE, AdamW,
  learning rate 0.001, decay 0.0001, batch 8, at most 100 epochs, patience 15.
- Select the earliest strict best Day-60 selection RMSE. Only that checkpoint
  is saved. Resume replays epochs after the best checkpoint deterministically.
- Seeds: 17, 29, 43, 59, 71, 89, 101, 127, 149, 173. The same seed identities
  span all three folds. Every fit records input, code and checkpoint hashes.
- Reference controls: fold-optimization mean/median and frozen CT07 D with
  structured inputs. The three-seed reference is not a matched ten-seed fit.
- The removed structured branch changes parameter count. This is a clean
  text-only input experiment, not a capacity-controlled causal attribution.
- BLAS/OpenMP/Torch threads are set to four by the launcher before numerical
  imports. Jobs are a resource choice, not a changed scientific setting.

After every fit is complete, evaluate the fixed ranking rule
`logit(mean_seed_description_prediction) + logit(past_EPSS)`. Clipping to
`[1e-6, 1-1e-6]` is solely for finite logits. The sum is not a calibrated
crossing probability. No crossing label fits the model or the combination.

The primary post-fit progression metric is AP gain over raw past EPSS on all
7,797 isolated-day-assumption eligible cases (62 risers). Strict observed
labels (3,516 cases, same 62 risers), AUROC, fold/seed results and Day-60
errors are mandatory diagnostics. One-sided subset selection is prohibited.
Resample evidence groups and seed identities 20,000 times at seed 6029.
The same seed draw applies across folds; fixed-fit group intervals are also
reported. These intervals do not cover every possible training-data change,
label assumption or provenance uncertainty.

Positive AP difference with a positive lower joint interval permits only a
new historical-protocol decision. Otherwise stop this surrogate branch as
inconclusive or unsupported. Neither outcome proves all text forecasting
possible or impossible. No automatic GNN expansion or test release follows.

## Status and Artifacts

The user completed smoke and all 30 full fits. Checkpoint/report replay and
independent metric review pass. The frozen primary progression rule does not
pass; this surrogate branch stops. No additional fits, threshold tuning or
cohort extension follow automatically. See `21_DESCRIPTION_SURROGATE_RESULTS.md`
for the completed findings and verification-only commands. The commands below
retain the original execution sequence; completed outputs should only be verified.

Results: `05_results/08_fresh_studies/25_description_only_surrogate_v1/01_reddit/v2/`.
`protocol/` pins settings, roles and dependencies; `<scale>/prepared/` stores
only predictor arrays plus separate labels/IDs. `<scale>/fits/fold_N/seed_N/`
holds one best checkpoint, history and predictions. `<scale>/report/` holds
Day-60 metrics, early-warning metrics/predictions, paired intervals and the
explicit decision. Fresh verification replays checkpoints and reports.

## Commands, One at a Time

All commands use CPU. Do not use `--verify-only` until the relevant stage exists.

```bash
cd /home/ayounas/Text_property_Graph
```
```bash
conda activate CodeBERTFusion
```
```bash
RUNNER=EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/30_run_description_surrogate.py
```
```bash
python -B -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/07_fresh_studies -p test_description_surrogate.py
```
```bash
python "$RUNNER" --stage freeze --verify-only
```
```bash
python "$RUNNER" --stage prepare --scale smoke --verify-only
```
```bash
python "$RUNNER" --stage prepare --scale full --verify-only
```
```bash
python "$RUNNER" --stage train --scale smoke --jobs 3 --resume
```
```bash
python "$RUNNER" --stage train --scale smoke --jobs 3 --verify-only
```
```bash
python "$RUNNER" --stage train --scale full --jobs 3 --check-only
```
```bash
python "$RUNNER" --stage train --scale full --jobs 3 --resume
```
```bash
python "$RUNNER" --stage train --scale full --jobs 3 --verify-only
```

For a clean installation without prepared artifacts, omit `--verify-only`
on the freeze and prepare commands. Existing locked output is never overwritten.

## Separate Engineering Benchmark

The user completed this benchmark and source-tensor verification passes.
This is not a model fit. It uses 24 bounded training graphs, compares loading
and resident memory, and applies synthetic backward passes without an optimizer
or labels. It does not approve an MSE graph protocol or establish full-epoch RAM.
It is independent of the early-warning result. No job has been scheduled.

```bash
python EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/25_benchmark_graph_loading.py --view DM --limit 24 --budget-gib 8 --threads 4 --repetitions 3 --run-name dm24_v1
```
```bash
python EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/25_benchmark_graph_loading.py --view DM --limit 24 --budget-gib 8 --threads 4 --repetitions 3 --run-name dm24_v1 --verify-only
```

## Report Rebuild

```bash
REPORT=EPSS_TPG/00_documentation/06_papers/03_build_files/primary_tpg_final_research_report/consolidation_v2
```
```bash
python -B -m unittest discover -s "$REPORT" -p test_report.py
```
```bash
python -B "$REPORT/build_report.py" --build
```
```bash
python -B "$REPORT/build_report.py" --verify-only
```
