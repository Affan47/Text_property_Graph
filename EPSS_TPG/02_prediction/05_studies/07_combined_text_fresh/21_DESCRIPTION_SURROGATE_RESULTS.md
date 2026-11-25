# DS01 Completed Review

## Decision

The frozen decision is `STOP_THIS_SURROGATE_BRANCH_INCONCLUSIVE_OR_NO_GAIN`.
All 30 best checkpoints and original reports reproduced. The separate review
recomputes 60 Day-60 and 104 ranking metric rows from saved predictions; it
changes no experiment artifact. Official validation/test records stay closed.

On 7,797 primary cases with 62 risers:

| Ranking | AUROC | AP |
|---|---:|---:|
| Past EPSS | 0.850182 | 0.064131 |
| Description only, mean of ten seeds | 0.744542 | 0.058694 |
| Fixed description + past-EPSS logit sum | 0.845174 | 0.082676 |

The primary AP change is +0.018545, with joint group-and-seed interval
[-0.010515, +0.076495]. AUROC changes by -0.005009
[-0.048767, +0.034192]. Twenty thousand draws, seed 6029; all original
predictions and labels retained. Nine individual seeds improve AP, none
improves AUROC. AP has positive point differences in all three folds, but the
declared uncertainty requirement is still not met.

Strict observed labels retain 3,516 cases and the same 62 risers. AP change
is +0.041881 [+0.003092, +0.105670], but AUROC change is +0.034000
[-0.004899, +0.071659]. This was a sensitivity, not the primary gate. Dropping
4,281 assumed negatives changes case composition and prevalence. Do not
replace the primary cohort to obtain a positive result.

## Day-60 Score Estimation

| Predictor | Mean seed-wise RMSE | Mean seed-wise MAE |
|---|---:|---:|
| Description only, 10 seeds | 0.036237 | 0.008350 |
| Frozen CT07 D + structured, 3 seeds | 0.036313 | 0.009791 |
| Optimization mean | 0.037002 | 0.013175 |
| Optimization median | 0.036641 | 0.004611 |

These use all 8,053 assessment CVEs. The small RMSE point advantage does not
establish an error improvement under a new paired interval; none was added.
MAE is substantially worse than the median control. Seed-wise metrics are
not metrics of ensemble predictions. The reference differs in capacity,
inputs and seed count, so this does not prove CVSS, references or leakage
caused the old ranking gain. Model-time and wording constraints remain.

## Graph-Loading Benchmark

The 24 graphs span 13 to 19,956 nodes. Resident tensors occupy 93.963 MiB;
peak process RSS was 1.145 GiB. All selected tensors verified; saved
representations matched and parameters stayed unchanged.

| Mean measured segments, 24 graphs | Uncached | Resident RAM |
|---|---:|---:|
| Load seconds | 0.479737 | 0.001083 |
| Forward/backward seconds | 0.160598 | 0.153200 |
| Total seconds | 0.640335 | 0.154283 |

Repeated-segment speedup: 4.15x. Including the 0.436952-second preload across
three passes: 2.13x. These are recorded segments, not complete process wall
time or full-epoch measurements. Verification checks hashes/tensors, not
repeatability of clock timings. No labels, optimizer steps or scientific fits.

## Next Step

Close the early-warning surrogate under its original rule, not all possible
future text forecasting. Do not append Reddit v1, choose a new cutoff or
collect the remaining histories to rescue it.

Proceed toward the separately motivated within-version Day-60 MSE graph vs.
pooled comparison. Before training, inventory actual fold/view RAM working
sets and worker budgets; require identical text coverage, structured features
and compatible pooling/readout; freeze the model, objective, folds, seeds,
constants and reporting. The 24-graph check is not the full memory inventory.
This remains retrospective estimation. No new training job is started here.

## Saved Evidence

- Original: `05_results/08_fresh_studies/25_description_only_surrogate_v1/01_reddit/v2/full/report/`.
- Review: `05_results/08_fresh_studies/26_description_surrogate_review_v1/01_reddit/review/`.
- Benchmark: `05_results/08_fresh_studies/20_graph_loading_benchmark_v1/01_reddit/dm24_v1/`.
- PDF includes the new tables and interval figure; its preceding edition is
  preserved under `consolidation_v2/preserved/before_ds01_completed_20261116/`.

## Commands, One at a Time

All commands below are CPU-only verification; none trains or downloads.

```bash
cd /home/ayounas/Text_property_Graph
```
```bash
conda activate CodeBERTFusion
```
```bash
python EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/30_run_description_surrogate.py --stage train --scale full --jobs 3 --verify-only
```
```bash
python EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/25_benchmark_graph_loading.py --view DM --limit 24 --budget-gib 8 --threads 4 --repetitions 3 --run-name dm24_v1 --verify-only
```
```bash
python EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/31_review_description_surrogate.py --verify-only
```
```bash
python -B -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/07_fresh_studies -p test_description_surrogate_review.py
```
```bash
python -B EPSS_TPG/00_documentation/06_papers/03_build_files/primary_tpg_final_research_report/consolidation_v2/build_report.py --verify-only
```
