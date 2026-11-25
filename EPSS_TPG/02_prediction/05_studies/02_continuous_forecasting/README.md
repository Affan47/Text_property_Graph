# Secondary Study: Daily-Updated EPSS Forecasting

## CF02: historical-text eligibility gate

The saved-evidence audit blocks temporal-text training for now. This means
historical availability of the saved wording is **not established**, not that
every description or post is proven to contain future information. No new
training, prediction, download or test adaptation is part of this gate.

The audit uses exactly the CF01 2,563 training and 1,104 validation CVEs.
Counts below mean nonempty saved text, before relevance or quality filtering.
They are not the accepted-source counts of the earlier classifier studies.

| Source | Training CVEs with text | Validation CVEs with text | CVEs with verified historical text |
|---|---:|---:|---:|
| D: description | 2,563 | 1,104 | 0 |
| G: GitHub summary | 542 | 201 | 0 |
| N: non-GitHub summary | 1,799 | 749 | 0 |
| M: social/related posts | 2,562 | 1,104 | 0 |

There are 1,765 training and 1,019 validation CVEs with at least one
deduplicated M document passing the existing strict-before-Day-30 date screen.
These are candidates, not verified historical inputs. All 5,731 selected raw
post associations match the earlier provenance audit by text hash, platform
and date. They contain `post`, `platform` and `date`, but no original post
ID/permalink; none has an explicit timezone. Sixty training associations have
lossy raw-platform text candidates, not authenticated historical versions.

All 3,667 current NVD update timestamps are after Day 30. That does not prove
that descriptions changed after Day 30; neither `description` nor
`orig_description` supplies a version-linked historical text record. NVD
reference-addition dates identify when links were recorded, not the content
of the linked pages. G/N summaries and their string logs lack authenticated
earlier source-content versions and summary-to-version lineage. A later
generated summary could be valid if recomputed solely from authenticated
pre-anchor evidence under a fixed procedure; the present artifacts do not
establish that condition.

For example, CVE-2022-46908 was published on 2022-12-12, so Day 30 is
2023-01-11. Its saved NVD update timestamp is 2026-06-17. An early reference
link to `https://sqlite.org/src/info/cefc032473ac5ad2` was recorded on
2022-12-12, but that alone does not establish what inputs produced the saved
non-GitHub summary. The record's saved Mastodon post is dated 2025-05-14,
well after Day 30. The audit preserves these actual values in `findings.json`.

Artifacts: `05_results/04_consolidated_studies/02_continuous_epss/cf02_text_eligibility_v1/`:
`findings.txt`, `findings.json`, `cve_source_evidence.csv`, `artifact_lock.json`.
The CSV has one row per development CVE/source (14,668 rows), source text
hashes, Day-30 dates and separate presence/date-screening/evidence decisions.
Primary Day-30-to-Day-60 eligibility remains 2,519 training / 1,104 validation;
the additional 44 training CVEs are retained in the overall evidence audit.

CPU verification of the saved audit:

```bash
cd /home/ayounas/Text_property_Graph
conda activate CodeBERTFusion
python EPSS_TPG/02_prediction/03_analysis/05_continuous_forecasting/cf02/07_audit_text_availability.py --verify-only
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/05_continuous_forecasting/cf02 -p 'test_*.py'
```

### Recommended consolidation

Proceed to the master report/manuscript, not a temporal-text training run:

1. Retrospective Day-60 estimation: target, source ablations, constants,
   validation selection and the already completed held-out evaluation.
2. Snapshot VulnCheck classification: catalogue-specific labels, baselines,
   frozen thresholds, completed evaluation and TRV01-TRV05 development revisions.
3. CF01 daily forecasting: eligible windows, masks, persistence and training-only
   corrections, all three seeds, per-slice results and Stage 3 diagnostics.
4. CF02 evidence limitations: text present is not text verified at the anchor;
   specify the content versions and timestamp provenance needed to reopen it.

Do not describe a structural performance ceiling or a universal impossibility
of EPSS forecasting as proved. Flat-history sensitivity is very small for
these checkpoints; last-only jointly changes values, masks, age and
known-history, not just one observation flag. The constant-offset decomposition
uses validation predictions and is not a deployment baseline. Keep the
frozen-D residual-DG proposal as an untested hypothesis: zero mean logit
correction does not imply zero mean probability change after sigmoid.
The eligibility audit itself did not regenerate the PDF. The subsequent
5 November master-report consolidation now includes this gate and CF01 Stages
1-3, preserving original results and identifying the constant shift as a
validation-derived diagnostic. See the paper build guide under
`00_documentation/06_papers/03_build_files/primary_tpg_final_research_report/consolidation_v1/`.

## CF01 stage 3: frozen-model history reliance

The bounded full diagnostic is complete. No new training was performed or
authorized. It uses the three saved Stage 2 full checkpoints,
1,104 validation CVEs and the same 3,067,429 eligible cells per seed. Test data
and all Stage 1/2 code, configurations and artifacts remain untouched.

| Diagnostic | What changes | What stays fixed |
|---|---|---|
| `intact` | Nothing; exactly replay the saved predictions | Everything |
| `flat_history` | Earlier observed scores become the most recent observed score | Observation masks, ages, known-history flags, day indices, prefix length, anchor, lead and persistence value |
| `last_obs_only` | Remove observations before the last one; reset their value/mask/age/known-history channels | Last observation, subsequent age, day indices, prefix length, anchor, lead and persistence value |
| `constant_shift_control` | Add one fixed per-seed offset to persistence | Same offset at every anchor and lead; same eligible cells |

`last_obs_only` changes the number of retained observations, not the number of
GRU time steps. For missing observations before the retained last observation,
age and known-history are zero because no history is then available.

The constant is each seed's full-validation Day-30-to-Day-60 mean raw residual,
approximately 0.000102987 / 0.000070019 / 0.000059358. It is fixed across all
slices and bootstrap resamples, including the smoke diagnostic. This is an
explicit **post-hoc validation-prediction decomposition**, not a training-fitted
control or a deployable predictor. The stored bias differences cancel the
target algebraically; full replay also checks the direct prediction-minus-
persistence mean. No per-slice shift is fitted to improve another horizon.

Outputs contain raw/clipped predictions and exact differences in canonically
ordered compressed CSVs, per-CVE JSON metrics, and five predeclared slices.
MAE/RMSE comparisons include each intervention minus intact, intact minus
persistence, and the constant shift minus persistence. Paired intervals use
1,000 CVE resamples at seed 6029, separately per seed and for mean seed effects
(not an ensemble). They are conditional on the checkpoints and fixed offsets,
unadjusted for multiple diagnostics, and not an independent generalization test.

Ten new tests and nineteen existing loader tests passed. Six-CVE smoke using
the full-trained checkpoints reproduced exactly. Reloading JSON in sorted-key
order also reproduced report hashes, exercising the Stage 2 ordering failure
without modifying Stage 2. Every intact cell is compared with the original
prediction CSV; parameter tensors and source hashes are checked unchanged.

Implementation is in `02_prediction/01_package/epss/continuous_forecasting/stage3/`;
the frozen specification is `02_prediction/04_configs/02_continuous_forecasting/02_stage3.json`.
Outputs are under `05_results/04_consolidated_studies/02_continuous_epss/stage3_v1/`.
Each `seed_*/predictions_and_differences.csv.gz` contains the inspectable cells;
`report/` contains `report.txt`, `summary.json`, `metrics.csv`,
`paired_intervals.json` and `paired_changes.csv`. Original artifacts are not overwritten.

Commands use **CPU with two PyTorch threads**; there is no GPU mode in this
frozen diagnostic. Run commands one at a time. Do not start a second process
while a diagnostic on the same scale is active.

```bash
cd /home/ayounas/Text_property_Graph
conda activate CodeBERTFusion
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/05_continuous_forecasting/stage3 -p 'test_history_reliance.py'
python EPSS_TPG/02_prediction/03_analysis/05_continuous_forecasting/stage3/05_diagnose_history_reliance.py --scale full --check-only
# First execution only. Use --resume after interruption; completed seeds are reused.
python EPSS_TPG/02_prediction/03_analysis/05_continuous_forecasting/stage3/05_diagnose_history_reliance.py --scale full
# After completion: no fitting, just deterministic replay and hash checks.
python EPSS_TPG/02_prediction/03_analysis/05_continuous_forecasting/stage3/05_diagnose_history_reliance.py --scale full --verify-only
```

These interventions can create inputs unlike training histories. They measure
frozen-model sensitivity, not a causal effect or retrained reduced-input model
performance. A constant-gain ratio above 100% means that the constant performed
better than intact for that metric; it is not a causal explained percentage.

### Stage 3 findings

The complete full diagnostic reproduced in a fresh process: all three intact
replays, intervened predictions, compressed CSV bytes, JSON metrics, paired
intervals and canonical report CSVs matched. There is no Stage 2-style row-order
exception in Stage 3. The separate findings TXT/JSON review also reproduced.
An additional full report replay from sorted-key JSON (the resume case)
reproduced identical CSV/report hashes without rerunning model inference.

The current GRUs barely respond to removing earlier score movements while
preserving observation availability. On Day30->60, the mean absolute prediction
changes are:

| Seed | Flat history | Last observation only | Fixed shift's share of intact MAE gain |
|---|---:|---:|---:|
| 17 | 4.276662894e-8 | 7.307344989e-5 | 95.9821% |
| 29 | 1.194948182e-8 | 4.655844545e-5 | 101.4652% |
| 43 | 4.799100376e-9 | 1.602659480e-5 | 99.6294% |

Each change compares the intervention with the same seed's intact prediction
on the same CVE. The ratio compares error improvements over persistence, not a
causal attribution. A value above 100% means the constant did slightly better
than the intact model for that seed. The last-only mean signed change is upward
for seeds 17/29 but downward for seed 43; it is not a uniform directional effect.

Across seeds, the fixed shifts reproduce **99.0225% of the primary MAE gain**,
**97.4827% of its RMSE gain**, and **88.1864% of the complete-matrix RMSE gain**.
The mean intact primary MAE is 0.010223739651; fixed-shift MAE is
0.010224019535. Intact retains a very small average advantage, not evidence of
substantial useful trajectory modeling. The means are across seed metrics, not
an ensemble of predictions.

| Primary-slice comparison against intact | MAE change | Paired 95% interval | RMSE change |
|---|---:|---|---:|
| Flat history | -5.377714754e-9 | [-1.277085911e-8, +7.609304720e-10] | -1.600603541e-9 |
| Last observation only | +6.837394039e-6 | [+4.886203301e-6, +8.661417570e-6] | -4.891976198e-6 |
| Fixed constant shift | +2.798844412e-7 | [+1.232826463e-7, +4.442964032e-7] | +2.934995579e-7 |

Negative error changes favor the intervention. Flat-history intervals include
zero for primary and complete-matrix MAE/RMSE. Last-only changes the prediction
more, but it worsens mean MAE while improving mean RMSE. This intervention
changes masks/age/known-history as well as score content, so it cannot establish
that older score fluctuations were beneficial. Exact per-seed and all-slice
changes, RMSE intervals and maximum differences are in the saved CSV/JSON files.

The actual target directions also support the descriptive offset interpretation:
47,618 of 74,551 seven-day cells are unchanged, 20,987 increase and 5,946 decrease.
For Day30->60, 919 CVEs increase, 179 decrease and 6 are unchanged. These are
unweighted observed-cell counts, not hierarchical loss weights or evidence of
why EPSS changed.

Keep the fixed baseline; do not promote an intervention or tune another GRU
from these diagnostics. This finding concerns the current checkpoints, not
whether EPSS is predictable in principle. Any subsequent TPG study should test
additional timestamp-supported evidence against the unchanged controls on
matched cells; these results do not establish that text fusion will help.

The separate interpretation is saved in
`05_results/04_consolidated_studies/02_continuous_epss/stage3_review_v1/findings.txt`
and `findings.json`. Reproduce it without model inference or training:

```bash
python EPSS_TPG/02_prediction/03_analysis/05_continuous_forecasting/reviews/06_review_history_reliance.py --verify-only
```

## CF01 stage 2: approved full-development comparison

The user approved one full comparison with the Stage 1 settings unchanged.
The original `01_stage1.json`, code, preflight and smoke artifacts remain
hash-locked. Full training is enabled only by the separate versioned runner
in `continuous_forecasting/stage2/`; the old smoke CLI remains bounded.

The cohort is 2,563 training and 1,104 validation CVEs, with 9,576,458 and
3,067,429 eligible forecast cells. The Day-30 to Day-60 slice contains
2,519 training and 1,104 validation CVEs. No test records are adapted or scored.

Settings remain GRU width 16, head width 32, AdamW learning rate 0.001,
weight decay 0.0001, eight CVEs per optimizer batch, up to 100 epochs,
patience 15, and seeds 17/29/43. There are 321 updates per complete full epoch.
The loss and evaluation weights are unchanged. Every epoch evaluates the full
validation matrix and all predefined slices. Selection uses only the earliest
strict best clipped validation Day-30 to Day-60 MAE among **trained epochs**.

Epoch 0 is saved separately in each history and verified as exact persistence
on every eligible validation cell. It cannot win checkpoint selection. Histories
include raw training MAE, gradient norms, cumulative optimizer updates, and
raw/clipped validation errors. `clipped_fraction` is the hierarchical share of
forecasts outside [0,1]. `signed_bias` means clipped prediction minus observed
EPSS; `raw_signed_bias` uses predictions before clipping. Negative bias means
underprediction. These are reported for every slice, including epoch 0.

Completed epochs have atomic recovery checkpoints; completed seed outputs are
immutable. Resume restores both model and AdamW state and retains the same
epoch-specific shuffle. Verification reconstructs training-only corrections,
checks epoch-zero parity, reloads all selected checkpoints, and reproduces
prediction CSVs, reports, bootstrap intervals and figures in a fresh process.
It does not replay every training update or retune a checkpoint.

Stage 2 smoke and its fresh-process replay passed. All selected smoke weights
and predictions follow the original settings. The original 19 loader tests
and four Stage 2 selection/diagnostic tests passed. Smoke overshoot is a
plausible explanation of its clipping, not a proven causal diagnosis.

The full run is complete. It used CPU with two PyTorch threads; this version
has no GPU CLI option. Output root (relative to `EPSS_TPG`):
`05_results/04_consolidated_studies/02_continuous_epss/stage2_v1/full/`.
`controls/` stores persistence and lead-correction outputs; each `seed_*/`
stores its checkpoint, history and predictions; `report/` stores the final
JSON/TXT summary, CSV slice table, paired intervals and figures.

### Completed results

All three seeds slightly improve the selected Day-30 to Day-60 slice, but all
three worsen complete-matrix MAE. The lead-median control equals persistence
on validation: its only nonzero correction is +0.000020 at lead 90, which has
no eligible validation cells. The table averages neural seed metrics, not
predictions; lower MAE is better.

| Validation slice | Persistence MAE | Mean neural MAE | Relative change in MAE |
|---|---:|---:|---:|
| All eligible forecast days and leads | 0.006077860 | 0.006092635 | +0.2431% worse |
| Day 30 to Day 60 | 0.010252373 | 0.010223740 | -0.2793% better |
| Seven days ahead, across eligible anchors | 0.003527679 | 0.003561429 | +0.9567% worse |
| Fourteen days ahead, across eligible anchors | 0.006346233 | 0.006360325 | +0.2221% worse |
| Thirty days ahead, across eligible anchors | 0.010491413 | 0.010472152 | -0.1836% better |

For neural minus persistence MAE, the primary-slice 95% paired interval is
[-0.000031957, -0.000025132]; the complete-matrix interval is
[+0.000013225, +0.000016380]. These exclude zero in opposite directions.
RMSE improves slightly in all five slices. These are conditional development
comparisons, not independent generalization evidence: the bootstrap does not
account for checkpoint selection or earlier reuse of the validation cohort.

| Seed | Last trained epoch | Selected epoch | Optimizer updates | Selected forecast clipping |
|---|---:|---:|---:|---:|
| 17 | 41 | 26 | 13,161 | 0% |
| 29 | 17 | 2 | 5,457 | 0% |
| 43 | 17 | 2 | 5,457 | 0% |

The all-zero smoke outcome did not persist. All selected primary forecasts
are slightly above persistence: average increases are 0.000102987,
0.000070019 and 0.000059358 for seeds 17, 29 and 43. Their between-CVE
standard deviations are 0.000010541, 0.000004544 and 0.000001721. These small,
relatively similar changes suggest a modest upward adjustment; they do not
establish rich CVE-specific trajectory learning or prove constant predictions.
The correction control fits all anchors per lead, whereas neural checkpoint
selection uses Day30->60 specifically. That objective difference also matters.

Interpretation and provenance are saved separately, leaving the run unchanged:
`05_results/04_consolidated_studies/02_continuous_epss/stage2_review_v1/review.txt`
and `review.json`. No text, TPG, CVSS or VulnCheck predictors were used here.
Keep this bounded study complete. Before more training or text fusion, a
separately specified frozen-model history-reliance diagnostic could test whether
past trajectory information adds anything beyond the latest score, age and lead.
No such intervention or new training has been run in this review.

### Verification outcome and report-order caveat

Fresh-process replay reproduced both controls, all three checkpoints and their
prediction CSVs, and all epoch-zero metrics. Intervals, figures, summary JSON
and summary TXT also reproduced byte-for-byte. The final original CLI check
does **not** finish successfully: only `metrics.csv` has a different row order.
The resumed run reloaded JSON with sorted slice keys, whereas fresh evaluation
uses the declared slice insertion order. All 25 unique predictor/slice rows,
their header and every value match exactly; this is not a numerical discrepancy.

The original runner and reports stay unchanged because they are hash-locked.
A separate strict report-order audit requires every other artifact hash to
match, permits no missing/duplicate rows, and uses no numerical tolerance.
Its TXT/JSON receipt is in
`05_results/04_consolidated_studies/02_continuous_epss/stage2_report_order_audit_v1`.
It checks report equivalence, not a new model replay. Do not describe the entire
original byte-level verification CLI as passing until a separately versioned
serialization change is approved.

The outputs already exist. To reproduce the analysis on CPU:

```bash
cd /home/ayounas/Text_property_Graph
conda activate CodeBERTFusion
python EPSS_TPG/02_prediction/03_analysis/05_continuous_forecasting/reviews/03_review_stage2.py --verify-only
python EPSS_TPG/02_prediction/03_analysis/05_continuous_forecasting/reviews/04_audit_report_order.py --reproduced-report EPSS_TPG/05_results/04_consolidated_studies/02_continuous_epss/stage2_v1/full/failed_attempts/.cf01-weywzvmo --verify-only
```

## CF01 stage 1: implemented and smoke-verified

The lazy loader, persistence, training-only lead correction and shared temporal
GRU now have a separate versioned adapter. The completed VulnCheck revision
results and original CS01 modules/configuration are unchanged. Full-development
training is deliberately unavailable in this CLI. No test CVEs are adapted,
scored, or moved into development.

The full preflight and bounded end-to-end smoke have reproduced in fresh
processes, including checkpoint predictions, report and figure hashes. Nineteen
new tests and all nineteen existing consolidated-contract tests passed.
An additional actual-data same-seed smoke training replay matched every selected
weight and every saved history value exactly for all three seeds, without
replacing any artifacts.

| Scope | Training CVEs | Training cells | Validation CVEs | Validation cells |
|---|---:|---:|---:|---:|
| Full preflight, no training | 2,563 | 9,576,458 | 1,104 | 3,067,429 |
| ID-selected smoke | 12 | 47,719 | 6 | 13,775 |

A cell is one CVE, forecast day and future target day. Cells from one CVE are
not independent samples. The full Day-30 to Day-60 slice has 2,519 training /
1,104 validation CVEs; the daily study admits 44 additional training CVEs with
other usable past/future pairs. Both cohorts retain their original roles.

Training contains 70,192 date-admissible cells with usable history but missing
targets, and 1,608,700 with no observed history. Validation has respectively
0 and 31,887. These excluded cells are not assigned zero labels. The counts
partition date-admissible cells: missing-target counts apply only after a usable
history exists, so they do not double-count the no-history category.

### Smoke findings, not performance evidence

Each neural smoke fit uses seeds 17/29/43, two epochs and the same 12/6 CVEs.
All five predictors (two controls plus three seeds) score identical validation
cells. The lead corrections are all zero on this small training sample, so that
control equals persistence. The two-epoch selected neural predictions are all
negative before clipping and become zero afterwards. This is explicitly reported,
not treated as a successful forecasting result or a reason to tune from smoke.

| Predictor | Full-matrix raw MAE | Full-matrix clipped MAE | Day-30 to Day-60 clipped MAE | Clipped fraction |
|---|---:|---:|---:|---:|
| Persistence | 0.000045982 | 0.000045982 | 0.000073333 | 0% |
| Training-only lead correction | 0.000045982 | 0.000045982 | 0.000073333 | 0% |
| Temporal seed 17 | 0.002540728 | 0.000538839 | 0.000518333 | 100% |
| Temporal seed 29 | 0.002352217 | 0.000538839 | 0.000518333 | 100% |
| Temporal seed 43 | 0.002840558 | 0.000538839 | 0.000518333 | 100% |

The raw-output loss still supplies gradients outside [0,1]; this is not a
clipped-loss dead-gradient failure. The sample is tiny, selected by CVE IDs,
and each fit sees only four optimizer updates. No useful prediction signal is
established by it. Review this finding and freeze the full-run settings before
enabling the one full comparison. Do not change the sample to obtain nicer scores.

### Adapter and evaluation details

- One sanitized 91-day series per development CVE is stored, not millions of
  expanded windows. The source JSON container is decoded once for ID selection;
  only development records have their dates/scores adapted and retained. Test
  records have no eligibility or numerical output. Raw files remain unchanged.
  `series.json` is a backing store, not a model input: observations serve as
  history or targets at different anchors. Only the separately constructed
  past-prefix tensor and requested lead enter the model.
- Training target dates must be before 2025-03-02. Validation anchors must be on
  or after 2025-03-02 and its target dates before 2025-06-01. Out-of-role dates
  are screened before score conversion and become unavailable internally.
- Prefix channels are observed value, observation flag, age divided by 90,
  known-history flag and day index divided by 90. Missing input placeholders
  are zero only with their mask off. They are not invented observations.
- Predictor inputs and requested leads are separate from target tensors. Padded
  targets remain NaN under a false mask; loss/evaluation index only valid targets.
- At most 32 anchor requests are assembled at once. Gradients accumulate over
  those packets, with one optimizer update per batch of at most eight CVEs.
  Weight each observed lead by `1 / (CVEs_in_batch * eligible_anchors * leads_at_anchor)`.
- Lead corrections are weighted medians of training residuals. For each lead,
  use the same hierarchical loss weights, conditioned on that lead. Process one
  lead's residuals at a time. An unsupported lead receives a documented zero
  correction, not an estimate fitted to validation.
- The unchanged CS01 GRU has hidden width 16 and a 32-unit horizon-conditioned
  head. Its final layer starts at zero. Source-precision persistence is added to
  the float32 learned correction, making initialization exactly persistence.
- Train on raw MAE. Select the earliest strict best clipped validation
  Day-30 to Day-60 MAE. Raw/clipped errors, clipping rate and clipping magnitude
  remain visible. AdamW uses learning rate 0.001 and weight decay 0.0001.
- Report hierarchical MAE, RMSE, signed bias and clipping for the full matrix,
  Day-30 to Day-60 and leads 7/14/30. Per-CVE summaries and per-cell matrices are
  retained. Slice metrics renormalize over eligible anchors/CVEs in that slice.
- Paired intervals use 1,000 CVE bootstrap resamples (seed 6029), keeping each
  CVE's entire summarized forecast history together and matching all predictors.
  Report correction minus persistence, and the mean same-seed neural difference
  against each control. This is not an ensemble or a bootstrap over individual
  forecast cells; smoke intervals have no generalization interpretation.

### Files and commands

All paths below are relative to `EPSS_TPG`:

| Path | Purpose |
|---|---|
| `02_prediction/01_package/epss/continuous_forecasting/data.py` | Compact preparation, frozen roles, lazy past/target batches |
| `02_prediction/01_package/epss/continuous_forecasting/controls.py` | Persistence and weighted training-only lead corrections |
| `02_prediction/01_package/epss/continuous_forecasting/evaluation.py` | Matched CSV predictions, per-CVE metrics, matrices, paired intervals |
| `02_prediction/01_package/epss/continuous_forecasting/runner.py` | Three-seed smoke fitting, checkpoint replay and figures |
| `02_prediction/04_configs/02_continuous_forecasting/01_stage1.json` | Explicit settings and full-training/test gates |
| `02_prediction/03_analysis/05_continuous_forecasting/` | CLI and focused tests |
| `05_results/04_consolidated_studies/02_continuous_epss/stage1_v1/preflight/full/` | Verified full coverage, compact series and provenance |
| `05_results/04_consolidated_studies/02_continuous_epss/stage1_v1/preflight/smoke/` | ID-selected bounded sample |
| `05_results/04_consolidated_studies/02_continuous_epss/stage1_v1/smoke/` | Checkpoints, training histories, CSV predictions, JSON/TXT reports and figures |

The outputs already exist. Run verification, not a second publication:

```bash
cd /home/ayounas/Text_property_Graph
conda activate CodeBERTFusion
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/05_continuous_forecasting -p 'test_daily_forecasting.py'
python EPSS_TPG/02_prediction/03_analysis/05_continuous_forecasting/01_run_daily_forecasting.py --stage preflight --scale full --verify-only
python EPSS_TPG/02_prediction/03_analysis/05_continuous_forecasting/01_run_daily_forecasting.py --stage preflight --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/05_continuous_forecasting/01_run_daily_forecasting.py --stage smoke --scale smoke --verify-only
```

`--verify-only` does not update model parameters. It reconstructs controls using
training data and replays saved checkpoints, metrics, predictions and figures.
On a new checkout without outputs, omit `--verify-only` in the same order.
`--stage smoke --scale full` remains rejected in the Stage 1 CLI. The separately
approved Stage 2 full-fit command is listed above.

## What changes

The model is no longer restricted to Day 30 or three selected anchors. For
each CVE, an anchor is the day when a forecast is made. At Day 10, inputs end
at Day 10; eligible targets can be Days 11 through 90. At Day 60, targets
can be Days 61 through 90. At Day 89, only Day 90 remains.

There are at most 4,095 anchor/lead pairs per complete 91-score history.
Missing scores and chronological boundaries reduce that number. A request
for a score beyond Day 90 is unsupported by this dataset, not a missing zero.

## Inputs and model

```text
Scores from Day 0 through today's anchor
  + observed/missing flags
  + age of the most recent observation
  + day index
                |
      Variable-length temporal GRU
                |
     Requested anchor and future lead
                |
     Persistence + learned correction
                |
 Raw estimate and estimate clipped to [0,1]
```

One shared model learns across eligible daily forecasts. It does not train
thousands of independent models and does not choose the easiest future day
after seeing the answer. The residual head starts at persistence. Raw outputs,
clipping frequency and the effect of clipping must remain visible in the
eventual training/evaluation reports.

The variable-length model and lazy data loader are implemented and tested with
padding and future-input invariance checks. The bounded smoke runner is complete;
the full fitting runner remains gated. Do not materialize millions of duplicate
text embeddings to represent repeated forecasts.

## Matched comparisons

1. Persistence: predict the most recent observed EPSS value.
2. A correction estimated from training data only, indexed by lead. Its
   weighting must follow the same CVE/anchor contract as the neural model;
   undefined leads must have an explicit fallback.
3. The shared temporal model, with the same observed input prefixes and
   target masks.
4. A matched temporal-plus-TPG extension only where text availability at the
   prediction anchor can be supported. A current summary or a post date alone
   does not establish that the saved wording was available then.

All methods must predict the same eligible CVE/anchor/lead cells. Report the
full matrix, with leads 7, 14 and 30 singled out in advance. The declared
primary validation comparison is Day-30 to Day-60 on common eligible CVEs.
Recent 10/20/31-day histories are secondary ablations, not an automatically
discovered optimal window. No observed future change determines eligibility,
sampling weights or the history used for a forecast.

## Missingness and weighting

Past missing values use explicit masks and observation age. An anchor needs
at least one observed past score. Missing future targets do not contribute
to the loss. They must never become zero-valued labels.

Average observed leads within each anchor, then eligible anchors within each
CVE, then CVEs. Otherwise CVEs with longer, more complete histories would
dominate the study. Validation intervals must resample CVEs, not pretend the
many forecasts of the same vulnerability are independent observations.

## Full preflight coverage

| Split | CVEs with eligible forecasts | Observed, date-eligible anchor/lead pairs |
| --- | ---: | ---: |
| Training | 2,563 | 9,576,458 |
| Validation | 1,104 | 3,067,429 |

The training count includes CVEs with useful forecasts even if Day 60 is
missing. It excludes assigned CVEs with no usable past/future pair. The two
heatmaps saved by CS01 show, for each anchor and lead, how many CVEs can be
evaluated. Their colors represent data coverage, **not model accuracy**.

Training targets must precede the validation boundary. Validation anchors
must be in the validation period and targets must precede the test boundary.
Changing the anchor never moves a CVE to another role. Numeric scores beyond
a role's target boundary are not adapted into development sequences.

## Limits

This study measures future EPSS scores, not actual exploitation events. A
model-version change may affect those scores. Daily dates do not establish
exact score-release times, so this is date-level forecasting, not an intraday
trading-style claim. Current CVSS/text cannot be used to support a historical
forecast merely because they refer to the same CVE.

If the temporal model does not improve on persistence and training-only
corrections, report that result and retain the simpler control. Do not use
the test set to search for a favorable horizon.
