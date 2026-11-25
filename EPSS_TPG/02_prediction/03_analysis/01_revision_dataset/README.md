# Revision Dataset Analysis

## Current: GD01 Instrumented Gain Diagnostic

`diagnose_forecast_gain_learning.py` compares the frozen MS06 residual gain
with gain 1. It uses fresh EPSS-only and all-source TPG models, identical
starting weights/batches, and unchanged loss/optimizer settings. It saves
batch-level losses, gradients, decay terms, updates, clipping, source weights,
epoch metrics, checkpoints, predictions and branch-intervention results.

The complete diagnostic has **96 training-only CVEs, 12 fits and 30 fixed
epochs per fit**. These CVEs are deliberately balanced across unchanged,
increasing and decreasing outcomes. It is not another full-cohort forecast
evaluation and performs no validation/test scoring. The four-fit smoke uses
nine training examples and three epochs. Previous code and results stay fixed.

See [GD01 in the experiment register](../../EXPERIMENT_REGISTER.md#gd01-instrumented-training-only-gain-diagnostic)
for commands, output files and interpretation limits. Results go to
`05_results/02_evaluation/epss_day30_to_day60/gain_learning_diagnostic_v1/`.

## New: SF01 Structured Timing Audit And SF02 CVSS Comparison

`audit_forecast_structured_features.py` checks structured fields for the fixed
Day-30 development cohort. Of 3,623 CVEs, 3,601 have parseable current CVSS
vectors and scores; 22 are missing. **None has verified field-level Day-30
availability.** The CVE ID establishes which vulnerability a value describes,
not when that value was available. Current reference counts, VulnCheck fields
and full-window EPSS version-change flags are not admitted as predictors.

`run_forecast_structured_day60.py` implements a separate, explicit
**retrospective** comparison: all-source TPG + EPSS, with and without a current
CVSS branch. It retains the same 2,519 training / 1,104 validation CVEs and
missing-value masks. The two configurations share architecture, initialization
and settings. Historical mode blocks unverified values rather than silently
backdating them. Existing models/caches and raw data are not changed.

The development audit is saved as TXT/JSON in `structured_availability_v1`;
new predictions and checkpoints use `retrospective_cvss_v1`, both under
`05_results/02_evaluation/epss_day30_to_day60/`. The small two-fit integration
run and its checkpoint reproduction are complete; the full-data preflight
passes. The six full fits remain for the user. No test scoring.
See [SF01/SF02 in the experiment register](../../EXPERIMENT_REGISTER.md#sf01sf02-structured-availability-and-current-cvss)
for counts, limitations, file purposes and commands. Twenty focused synthetic
tests include missing/numeric CVSS formats, branch gradients, checkpoint
reproduction and refusal of unacknowledged retrospective runs.

## Current: MS05 Cache and MS06 Day-60 Runner

The Day-30-aware cache is built for all **2,519 training / 1,104 validation
CVEs**. It references 10,110 unchanged CD03 v2 chunks and stores 21 newly built
chunks. The new indexes preserve MS04 Day-30 source masks. Do not delete the
original cache: the reused graph and tensor locations are read-only references.

`run_forecast_day60_multisource.py` implements the matched **Days 0-30 -> Day-60**
comparison: EPSS-only, description TPG + EPSS, and all-source TPG + EPSS, with
persistence. Its three real smoke fits and predictions reproduce. The nine
full fits (three configurations, seeds 17/29/43) are ready for the user to run;
they have not been launched. No test scoring is performed.

See [MS05](../../EXPERIMENT_REGISTER.md#ms05-day-30-aware-graph-cache) for cache
locations and [MS06](../../EXPERIMENT_REGISTER.md#ms06-matched-day-60-training)
for the fixed model settings, full-run/resume commands and result files.
The smoke metrics are an integration check, not evidence of useful learning:
all 16 smoke training CVEs have unchanged input/target endpoints.

## Completed and Verified: MS04 Day-60 Preparation

`prepare_forecast_day60_multisource.py` prepares **Days 0-30 EPSS -> Day-60
EPSS**, with descriptions, summaries and newly screened Day-30 SMP sources.
The completed exports contain **2,519 training / 1,104 validation CVEs**.
Partial histories are kept: 2,254 training and 867 validation CVEs have missing
input days. Targets must be observed; a Day-90 score is not required.

Split boundaries move 30 days earlier with the prediction anchor, preserving
every existing CD02 CVE role and all 651 reserved test identities. Their Day-60
eligibility is not inspected. No old data, graphs or results are overwritten.
Descriptions and summaries remain retrospective, not verified historical text.

Fifteen new tests and eleven existing CD02 tests pass. Smoke/full creation and
verification commands are in [MS04 in the experiment register](../../EXPERIMENT_REGISTER.md#ms04-day-60-multi-source-preparation).
The script writes separate histories, labels, source inputs, date checks,
coverage reports and provenance under
`04_data/01_records_and_graphs/cve_identity_day30_to_day60_v1/{smoke,full}/`.
The user's full and smoke exports independently reproduce. Explicit checks
confirm smoke/full agreement for histories, labels, source routing, IDs and
the training registry. All 79 prerequisite hashes and both output locks pass.
The original production exports are unchanged. Use `--verify-only` for these
completed outputs rather than repeating the creation commands.

MS04 itself does not create TPGs or train models. MS05 and MS06 now consume
these exports. Do not feed them to the existing MS03 Day-90 runner. Final test
scoring remains disabled.

## Completed: MS03 Matched Training

`run_forecast_matched_multisource.py` now compares EPSS only, description TPG
plus EPSS, and description/summary/SMP TPGs plus EPSS. Persistence is the common
reference. All use the same 2,548 training / 1,104 validation CVEs and observed
Day-90 target, with Day 30-60 EPSS inputs. The full protocol is nine fresh fits
(three models, three seeds), MAE loss, up to 100 epochs and patience 15.
The test set stays closed; retrospective text limitations are explicit.

Read [MS03 in the experiment register](../../EXPERIMENT_REGISTER.md#ms03-matched-multi-source-training)
for the architecture, exact settings, source coverage, output files and ordered
smoke/full CLI commands. All nine full checkpoints and prerequisite smoke
checkpoints now reproduce. Mean MAE is 0.0045173701 for EPSS-only,
0.0045175334 for description + EPSS, and 0.0045175095 for all text + EPSS,
against persistence at 0.0045179801. These tiny differences do not establish
useful text benefit: seven fits have constant corrections, and the other two
vary only across source masks in saved predictions. Learned GRU and active
node projections shrink near zero. The original run is unchanged.

`review_forecast_matched_multisource.py` saves a separate TXT/JSON review and
figure under `05_results/02_evaluation/epss_day60_to_day90/matched_multisource_review_v1/`.
Its four tests check endpoint counts, mask-group analysis and parameter norms.
Use `--verify-only` to reproduce the completed review. There is no retraining,
new prediction generation or test scoring in the review script.

## Completed: CD03 V2 Multi-Source Graph Cache

The user's full cache and tensor/graph verification are complete: 3,652 CVEs,
10,438 document associations and 11,121 chunks. Independent hash/report review
passed without repeating the 44-minute tensor traversal. The review script is
`review_identity_multisource_cache.py`; TXT/JSON findings are saved under
`01_tpg/05_workspace/cve_identity_multisource_v2/completed_review_v1/`.
The cache contains 3,006,451 nodes and 9,517,311 no-COREF edges. No model training
has run in CD03; the MS03 matched source-aware training runner is now implemented.

See the [artifact guide](../../../01_tpg/00_ARTIFACT_GUIDE.md) for real
existing graph links and the individual smoke/full commands, or
[CD03 in the register](../../EXPERIMENT_REGISTER.md#cd03-inspectable-multi-source-graph-cache).
The implementation preserves CD02 source masks, v4 security logic, frozen
embedding resources and every source character; test graphs remain excluded.

## Verified: CD02 CVE-Based Preparation

The next implementation retains distinct CVE records rather than quarantining
whole CVEs because of shared text/URLs. Unique IDs and unique wording are not
the same: the pinned dataset has 113 exactly repeated description-string
groups covering 421 CVEs; normalization gives the earlier documented 116/430.
No claim of duplicate vulnerabilities follows from these counts.

New fixed cohort: **2,548 training / 1,104 validation / 651 eligible test**,
using the reviewed April/July 2025 boundaries. Partial histories remain.
Descriptions and imperfect nonempty summaries are retained. Optional validation
documents already present as normalized training text are masked at source
level; no CVE is dropped. Repeated descriptions are reported and retained.
Test exports and training remain disabled. The July-heavy test coverage and
retrospective text limitations are explicit.

The user's smoke/full exports are independently reproduced and mutually
consistent. They retain 10,438 source-document associations. Two validation
posts are masked; no summary or CVE is removed by the overlap rule. Training
includes 342 partial recent histories. Eleven synthetic tests also passed.

The separate TXT/JSON completed review is under
`EPSS_TPG/05_results/02_evaluation/epss_day60_to_day90/cve_identity_preparation_review_v1/`.
The next graph/cache implementation must handle long retained SMP documents
explicitly (maximum 395,061 characters), without silently truncating or dropping
CVEs. No further cohort-selection round is needed. For completed exports use
the verification commands below, not the creation commands:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_identity_development.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_identity_development.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_identity_development.py --scale smoke
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_identity_development.py --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_identity_development.py --scale full
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_identity_development.py --scale full --verify-only
```

Outputs: `EPSS_TPG/04_data/01_records_and_graphs/cve_identity_day60_to_day90_v1/{smoke,full}/`.
Read `analysis.txt`/`analysis.json`, `description_identity_check.json` and
`source_decisions.json`. Numerical targets are separate from past histories
and text inputs. Full export checks the saved smoke's inputs and source masks.
No original results, raw data or legacy cohorts are modified.

See [CD02 in the register](../../EXPERIMENT_REGISTER.md#cd02-cve-based-development-preparation)
for source registries, field contracts, date rules and limitations. Graph-cache
expansion and the fixed full-training runner follow this preparation.

## Verified: CD01 Cohort-Design Review

The next requested step reviews the restrictive training allocation before
expanding the multi-source cache. It considers all 5,968 CVEs using dates,
observation availability and grouping metadata, not score magnitudes or model
performance. Partial inputs remain eligible; missing Day-90 targets are not
filled. Four fixed chronological windows are compared without adopting any.

Frozen groups remain the main comparison. A weaker text-only grouping and ten
individual shared-URL sensitivity checks quantify possible over-grouping, but
do not authorize removing links. Former test-role movements and independent
group counts are explicit. The user's full review has been independently
reproduced; nine synthetic tests and the hash/date-policy preflight also pass.

5,645 of 5,968 CVEs are eligible before splitting. Later-Q2 frozen grouping
gives 1,998 training / 884 validation / 342 test CVEs, with 334 test anchors
in July 2025. Shared references/posts are not proof of duplicate CVEs.
The remaining recommendation is a versioned identity/source-overlap policy,
not a demand for perfect summaries or another optimizer diagnostic.

Read the separate `analysis.txt` and `analysis.json` under
`EPSS_TPG/05_results/02_evaluation/epss_day60_to_day90/cohort_design_interpretation_v1/`.
They explain the options, label-boundary exclusions and repeated stored-post
examples. The original review files, code and split assignments are unchanged.
Use `--verify-only` below for an already completed run, not default creation.

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_cohort_design.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_cohort_design.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_cohort_design.py
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_cohort_design.py --verify-only
```

Outputs go to `EPSS_TPG/05_results/02_evaluation/epss_day60_to_day90/cohort_design_review_v1/`:
`analysis.txt` and `analysis.json` explain the counts; `cve_availability.json`
accounts for every CVE; `candidate_assignments.json` is non-operative audit
metadata; `url_review_queue.json` contains pending review items;
`cohort_counts.png` compares eligible CVE counts under unchanged grouping.
The protocol and lock record configuration and integrity. No training, score
prediction, original data changes or existing split changes occur.

Read [CD01 in the register](../../EXPERIMENT_REGISTER.md#cd01-bounded-cohort-design-review)
before deciding on a new cohort. Full multi-source preparation follows that
explicit decision; no further optimizer-tuning round is planned.

## Verified: MS02 Small Multi-Source TPG Preview

The next graph stage is implemented without changing H02 or the fixed
security extractor. It uses the **same 16 training / eight validation preview
CVEs**, Day-60 source masks and 65 accepted documents. Each description,
summary and post gets an independent graph. Frozen embeddings are shared
across the ten existing source-removal views; missing sources never remove
a CVE. The primary graph view is no-COREF. No training or full expansion.

The user completed the preview; all 331 artifact hashes and 65 document bundles
pass independent verification. There are 13,604 nodes, 42,568 no-COREF edges
and 19 typed security edges, with no missing ordinary-token embeddings.
The separate review accounts for all 5,968 CVEs and distinguishes completed
R07/H02 full training from the still-untrained multi-source fusion model.

Read `01_tpg/05_workspace/multisource_day60_v1/preview_review_v1/analysis.txt`
and `analysis.json`, relative to `EPSS_TPG`. Reproduce the review with:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_multisource_graph_preview.py --verify-only
```

Next: prepare the existing 1,029 training / 173 validation source graphs,
implement a fixed fusion runner, run one integration smoke, then full fits.
This does not require another EPSS-only tuning round or perfect input histories.
The full multi-source training CLI is not implemented yet.

Eleven synthetic tests and the real four-source preflight pass. The original
creation/verification CLI is under TPG (do not recreate the completed cache):

```bash
python -m unittest discover -s EPSS_TPG/01_tpg/08_tests -p 'test_multisource_graph_preview.py'
python EPSS_TPG/01_tpg/02_examples/01_scripts/prepare_multisource_graph_preview.py --check-only
python EPSS_TPG/01_tpg/02_examples/01_scripts/prepare_multisource_graph_preview.py
python EPSS_TPG/01_tpg/02_examples/01_scripts/prepare_multisource_graph_preview.py --verify-only
```

Read `analysis.txt` and `analysis.json` in
`EPSS_TPG/01_tpg/05_workspace/multisource_day60_v1/graph_preview/` after it
finishes. GraphSON, tensor bundles and checked document receipts are in the
same directory. `--resume` retains completed documents; `--verify-only`
does not rerun NLP/embeddings. Models load offline by default.

See the [MS02 register](../../EXPERIMENT_REGISTER.md#ms02-small-multi-source-graph-preview)
for scope, graph isolation, source routing and output details. Historical
text availability remains unverified; the test set stays closed.

## Verified: H02 Full Matched History Controls

The user completed preparation, smoke and all six full fits. Checkpoints
and saved predictions reproduce. No automated retraining or test scoring.

**Finding:** recent and longer history produce exactly the same final
validation predictions within each seed. Both have mean MAE 0.008303950178;
persistence MAE is 0.008303930636. The tiny differences do not demonstrate
an improvement. Seeds 17/29 learn constant corrections; seed 43 has only
negligible input-dependent corrections. Distinct tensor lengths, separate
checkpoint weights and recorded gradients were verified.

150/173 validation endpoints equal their latest observed input. Two CVEs,
`CVE-2023-45249` and `CVE-2023-7028`, account for 94.99% of persistence's
total absolute error. They remain in the primary results. Endpoint equality
does not establish an unchanged intervening trajectory.

Read the separate review in
`EPSS_TPG/05_results/02_evaluation/epss_day60_to_day90/history_controls_review_v1/`:
`analysis.txt`, `analysis.json`, `learning_curves.png`, and
`error_concentration.png`. Original run artifacts remain unchanged.

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_later_controls.py --verify-only
```

Next: a small source-aware TPG graph/cache preview with the frozen Day-60
masks and source-removal plan. No further EPSS-only tuning or final-test
evaluation is recommended from these results.

H01's verified preview now has a separate full-development preparation and
EPSS-only training runner. It keeps **1,029 training / 173 validation CVEs**
and compares Days 30-60 with Days 0-60, both predicting observed Day-90 EPSS.
No source data, old result, old graph cache or test score is changed.

- `prepare_forecast_later_development.py` reproduces H01 and exports the full
  matched inputs, separate labels, source masks and coverage audit.
- `run_forecast_later_controls.py` implements both 31- and 61-day loaders,
  small smoke and fixed six-fit full comparison, persistence, resume and
  checkpoint/prediction verification.
- `test_forecast_later_development.py` has fourteen synthetic tests.
- `../../04_configs/day60_to_day90_controls_v1.json` fixes the protocol.

The full run is complete and independently verified. Settings are MAE,
Adam at 0.001, decay 0.0001,
batch eight, seeds 17/29/43, 100 epochs maximum and patience 15. Both histories
share initial weights, epoch orders, CVEs and persistence. Full training
requires a verified smoke; smoke performance does not change settings.

The full source audit finds GitHub summaries for 236 training / 51 validation
CVEs, non-GitHub summaries for 705 / 121, and screened SMP for 880 / 53.
Descriptions cover everyone. All stored post-date entries lack timezone;
84 post documents have conflicting dates. These are retrospective candidate
sources, not verified historical inputs. Source-aware TPG fusion follows
the temporal controls and still requires a small graph preview.

The original run sequence is below. It has already completed; use the
verification commands for existing outputs instead of recreating them.
Run individually from the repository root with `CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_later_development.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_later_development.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_later_development.py
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_later_development.py --verify-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_later_controls.py --scale smoke --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_later_controls.py --scale smoke
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_later_controls.py --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_later_controls.py --scale full --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_later_controls.py --scale full
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_later_controls.py --scale full --verify-only
```

Exports go to
`EPSS_TPG/04_data/01_records_and_graphs/epss_day60_to_day90_v1/development/`.
Read `analysis.txt` and `source_coverage.txt`; JSON counterparts retain the
machine-readable counts. Training reports go to
`EPSS_TPG/05_results/02_evaluation/epss_day60_to_day90/history_controls_v1/{smoke,full}/`.
Read `summary.txt` and `summary.json` after each run. `--resume` retains
locked fits and restarts only incomplete fits; `--verify-only` performs no
training. Progress bars cover preparation, tensor loading, fits, epochs
and batches.

See the [H02 register](../../EXPERIMENT_REGISTER.md#h02-full-matched-history-controls)
for the model, files, missingness contract and future fusion comparisons.

## Verified: H01 Day-60 to Day-90 Preview

`prepare_forecast_later_anchor.py` implements the separate later-anchor task:
**Days 30-60 versus Days 0-60, both predicting Day 90**. This is a
preparation preview, not training. The original R07/MS01 artifacts stay fixed.

The shared split helper reassigns roles at Day 60 with Day-90 label maturity
and unchanged related-CVE groups/calendar boundaries. Preflight finds
**1,029 matched eligible training / 173 validation CVEs**. Both history
variants use the same CVEs and latest-observation persistence reference.
Only 16 train / 8 validation records are exported in this first preview.

Source timing is also moved to Day 60 through the existing source-aware
adapter; this does not authenticate historical text. Target values are
stored separately from inputs. Numerical test histories, targets and test
text are not exported or analyzed. Test metadata is assigned a reserved role.

The user's preview has been independently reproduced. All ten artifacts
pass verification, as do ten preparation tests and three review tests.
No H01 graphs, embeddings or models have been generated.

The separate review is in
`EPSS_TPG/05_results/02_evaluation/epss_day60_to_day90/preview_review_v1/`:
read `analysis.txt` for explanations and `analysis.json` for counts,
per-CVE input checks and cohort changes. Original H01 artifacts are unchanged.

| Split | Eligible CVEs | Complete Days 30-60 | Complete Days 0-60 |
|---|---:|---:|---:|
| Training | 1,029 | 787 | 77 |
| Validation | 173 | 166 | 37 |

All other eligible histories are partial, not discarded. The sparsest
training history has four observed scores. Longer histories include older
missing days; twelve training CVEs gain no observations from those older days.

The preview has no GitHub summary for any of its eight validation CVEs.
Screened social-media text remains for 14/16 training and 4/8 validation
CVEs. All 34 post-date entries lack timezone information; two Telegram text
records with conflicting dates remain excluded. Passing the screen does not
verify when the text was originally published. These are small-sample
readiness findings, not performance or full-cohort source-coverage estimates.

For this completed run, use verification only from the repository root
with `CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_later_anchor.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_later_anchor.py --verify-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_later_anchor.py --verify-only
```

Output: `EPSS_TPG/04_data/01_records_and_graphs/epss_day60_to_day90_v1/preview/`.
Read `analysis.txt` and `analysis.json` first. `inputs.json` has both past
histories, `labels.json` has separate targets, and `source_inputs.json`
has the new source masks. `split_manifest.json` and `cohort_audit.json`
explain new role assignments and development eligibility.

The H01 CLI remains preview-only. H02 above now supplies separate
full-development and training commands; it does not alter the frozen H01
implementation. Do not run the old 31-step-only loader on the longer history.
See the [H01 register](../../EXPERIMENT_REGISTER.md#h01-later-anchor-preparation).

## Verified: MS01 Preview and Window Feasibility

The user preview is complete and reproduced. Original files are unchanged.
Read the separate `multisource_window_review_v1/analysis.txt` and
`analysis.json` under `EPSS_TPG/05_results/02_evaluation/epss_day30_to_day60/`.

The review flags non-GitHub summaries for CVE-2022-28706 and CVE-2024-22024
that describe loading/CSS errors rather than useful vulnerability evidence.
Only 1/8 validation-preview CVEs retains a post after date screening.
These are bounded coverage/quality findings, not reasons to change cohorts
or claim all summaries/posts are unusable.

The separate EPSS comparison uses original training CVEs only. Five CVEs
whose Day-90 targets reach the validation period are excluded. On 1,011
matched CVEs with at least 20 adjacent observed pairs in each window,
Days 30-60 have more movement for 51, Days 0-30 for 419, and 541 are equal.
The complete-data subset has only 41 CVEs and a different mean pattern;
both sensitivity views are retained.

On 1,013 matched endpoint CVEs, persistence MAE is 0.00256757 for
Days 0-30 -> Day 60 and 0.00287774 for Days 30-60 -> Day 90.
A later prediction anchor is feasible, but neither greater input movement
nor easier prediction is established. This is descriptive training-only
analysis, not a new trained model or authorization to change the task.

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_multisource_forecast_windows.py --verify-only
```

Five synthetic window tests pass, covering gaps, missing observations,
endpoint equality despite intervening movement, date-boundary exclusion
and matched-cohort coverage.

### Source Adapter

We will complete the planned multi-source development work before final test
evaluation. R07 trained description + EPSS, not summaries or social-media
text. Its results remain unchanged.

`prepare_forecast_multisource.py` is the first step: a source-aware adapter
for **description, GitHub summaries, non-GitHub summaries and individual
posts**, with source-presence and eligibility masks. The bounded preview
uses 16 training / 8 validation CVEs from the frozen cohorts. Eleven
synthetic tests and the real-data preflight passed before the user run.
The saved preview is now verified. No graphs, embeddings or training have
been generated at this stage.

Current descriptions/summaries stay explicitly retrospective. Posts must
have consistent stored dates strictly before Day 30 to enter the initial
input view. Missing, same-day, later and conflicting dates are excluded
from that view but retained in the audit. Passing this screen does not
prove authentic publication time. The strict verified-historical text
view remains empty; dates alone do not authorize historical evidence.

Missing sources do not remove CVEs or generate placeholder text. Ten
planned matched conditions cover EPSS-only, each text source with EPSS,
all sources, and removal of each source in turn. Training is not enabled.

The preview is complete: use `--verify-only` now. Original commands are
retained below; run separately from the repository root with `CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_multisource.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_multisource.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_multisource.py
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_multisource.py --verify-only
```

Output: `EPSS_TPG/01_tpg/05_workspace/multisource_development_v1/source_preview/`.
Read `analysis.txt` and `analysis.json`, then inspect `records.json`.
Only `model_inputs.json` contains the candidate allowlisted text payload;
the audit record is not a model input. `protocol.json` defines the planned
comparisons and limitations; `run_identity.json` and `preview_lock.json`
track source and output checksums. Existing output is not overwritten.

The next step after review is a small multi-source TPG graph preview.
Neither final test scoring nor full multi-source training is authorized
by this adapter. See the [MS01 register](../../EXPERIMENT_REGISTER.md#ms01-bounded-multi-source-adapter).

## Verified: R07 Full Eligible-Cohort MAE Training

The user's six full-cohort fits are complete. All 50 artifacts and
checkpoint predictions reproduce. The separate review did not retrain,
generate embeddings or score the test set; original results are unchanged.

| Model/reference | Mean validation MAE across seeds |
|---|---:|
| Persistence / zero-offset scalar reference | 0.00909675127 |
| EPSS-only | 0.00909686292 |
| No-COREF TPG+EPSS | 0.00909687284 |

All six fits improve over matched full R04 Smooth L1 MAE, but none beats
persistence. Five corrections are exactly constant across validation CVEs;
the sixth varies by only `2.842e-14`. Strong encoder-weight shrinkage remains.
This is not evidence of useful added TPG contribution for this target.

162/197 validation endpoints equal the last observed input score.
CVE-2024-23692 rises from 0.00210 to 0.95432 and CVE-2024-8963 from
0.30991 to 0.95343; models stay near their input scores. Those two cases
account for 89.05% of baseline absolute error. Missing input days remain
allowed, and neither model type improves seed-mean MAE over persistence
within any of the three saved input-completeness groups.

Read `residual_mae_development_v1_review/analysis.txt` and `analysis.json`
under the evaluation root. Reproduce the saved review without training:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_mae_development.py --verify-only
```

The description-plus-EPSS training stage is finished. The subsequent user
decision is to complete multi-source development through MS01 above before
final test evaluation. Report R07 persistence and all six fits without
selecting a winning seed or changing splits. No more R07 loss/decay tuning
is planned.

`run_forecast_mae_development.py` implements the completed R06 handoff:
all **1,023 eligible training / 197 validation CVEs**, EPSS-only and
no-COREF TPG+EPSS, seeds 17/29/43, MAE loss, original weight decay
`0.0001`, fresh initialization, up to 100 epochs and patience 15.
Seven synthetic tests and the real-data preflight passed before training.
The full run is now complete, not another bounded comparison.

Missing input days are already permitted and masked. The task needs at
least one score during Days 0-30 and an observed Day-60 target. The other
source CVEs include 2,670 eligible reserved test CVEs, 136 assigned records
without required observations, and 1,942 policy exclusions. They cannot
simply be added to training while preserving this experiment's evaluation.
The source file still contains all 5,968 CVEs.

Use `--verify-only` for the completed run. Original commands are retained
below; run individually from the repository root with `CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_mae_development.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_mae_development.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_mae_development.py
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_mae_development.py --verify-only
```

Only if interrupted, use `--resume`. Completed fits are verified and skipped;
an unfinished fit restarts from its fixed initialization. Completed runs
use `--verify-only`. Progress bars cover source checks, fits, epochs,
batches and prediction checks.

Results go to `residual_mae_development_v1/` under
`EPSS_TPG/05_results/02_evaluation/epss_day30_to_day60/`. Read `summary.txt`
first. Original runs and caches are unchanged. Compare all seeds against
persistence and matched full R04 results, not the smaller R06 cohort's error.

After reviewing validation results, freeze final choices before separately
authorizing test evaluation. No test-training or test-scoring command is
part of R07. Historical description availability remains unverified, so
the graph experiment remains retrospective.

See the [R07 register](../../EXPERIMENT_REGISTER.md#r07-full-eligible-cohort-mae-training)
for the complete count breakdown, protocol, limitations and output structure.

## Verified: R06 Final Bounded Loss Comparison

The user's run is complete. All 80 locked artifacts, 12 checkpoint
predictions and six R05 Smooth L1 control reproductions are independently
verified. The fixed rule selects **MAE loss** for full development training.

| Model | Smooth L1 mean validation MAE | MAE-loss mean validation MAE |
|---|---:|---:|
| EPSS-only | 0.01512880375 | 0.01494139428 |
| No-COREF TPG+EPSS | 0.01516128976 | 0.01494129092 |

MAE improves all six paired comparisons. Persistence remains slightly
better at **0.01494125000**. The learned corrections are now nearly zero;
this is improved alignment with MAE, not a demonstrated forecasting or
TPG advantage. The large increase in CVE-2024-23692 remains uncaptured.

Read `residual_loss_comparison_v1_review/analysis.txt` and `analysis.json`
under the evaluation root. The review did not train or alter original
artifacts. Optional reproduction:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_loss_comparison.py --verify-only
```

**R07 above is the full-training runner**, not another bounded diagnostic or
hyperparameter sweep. It uses MAE, the existing 1,023/197 development split,
both models, three seeds, original decay and the recorded full budget.
No full run or test scoring was started by this review.

`run_forecast_loss_comparison.py` compares **MAE loss versus Smooth L1**
using exactly R05's 128 training / 64 validation CVEs, two models,
seeds 17/29/43 and 30 fixed epochs. Weight decay stays at `0.0001`.
The graph cache, initial weights, epoch orders, loss-independent settings
and test exclusion remain unchanged.

Seven synthetic tests and the real-data preflight passed before the user
trained the models. The new Smooth L1 controls now exactly reproduce frozen
R05 original-decay results. MAE loss is the only scientific change.

This is the **last planned bounded comparison before full development
training**. Select one common loss by the lower mean validation MAE
across both models and all three seeds; numerical ties retain Smooth L1.
The next stage uses the existing 1,023 training / 197 validation CVEs,
the same two models/seeds, up to 100 epochs and patience 15. No test
scoring, split expansion or additional bounded tuning round is planned.
Full training remains a separate user-run stage, not an automatic action.

The comparison is complete: use **`--verify-only`** now. Original execution
commands are retained below. Run individually from the repository root with
`CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_loss_comparison.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_loss_comparison.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_loss_comparison.py
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_loss_comparison.py --verify-only
```

Only if interrupted, use the same runner with `--resume`. Completed fits
are verified and skipped; an unfinished fit restarts from initialization.
Completed runs use `--verify-only`, not `--resume`.

Results are saved in `residual_loss_comparison_v1/` under
`05_results/02_evaluation/epss_day30_to_day60/`. Read `summary.txt` first,
then `summary.json` and `full_training_decision.json`. Selecting a
development loss is not a claim that the model beats persistence.
Original R05 outputs and all frozen code stay unchanged.

See the [R06 register and full-training decision rule](../../EXPERIMENT_REGISTER.md#r06-final-bounded-loss-comparison).

## Verified: R05 Bounded Weight-Decay Comparison

The user's run is complete. All 79 artifacts and 12 checkpoint predictions
reproduce independently. Zero decay preserves encoder norms but worsens
validation MAE in five of six paired comparisons. All learned models lose
to persistence MAE. Three-seed means are:

| Model | Original decay MAE | Zero decay MAE |
|---|---:|---:|
| EPSS-only | 0.01512880 | 0.01517969 |
| No-COREF TPG+EPSS | 0.01516129 | 0.01519152 |

Persistence and the zero-offset scalar reference both score **0.01494125**.
57/64 validation targets are unchanged, and CVE-2024-23692 contributes
99.58% of persistence's total absolute error. Graph models fit training data
better over later epochs while validation error worsens. Retaining encoder
weights therefore does not, by itself, solve generalization.

Read `residual_decay_comparison_v1_review/analysis.txt` and `analysis.json`
under the evaluation root. Original outputs are unchanged. Do not expand
zero-decay training on these results. R06 above now implements the proposed
loss-only comparison; the R05 review itself did not train or implement it.

Reproduce the saved analysis without training or new predictions:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_decay_comparison.py --verify-only
```

`run_forecast_decay_comparison.py` implements the next controlled experiment:
128 training / 64 validation CVEs selected by fixed ID hash within the
existing splits. Compare EPSS-only and no-COREF TPG+EPSS, each with Adam weight
decay `0.0001` versus `0`, for seeds 17/29/43 and 30 fixed epochs. This is
12 small fits, not full-cohort training. No test scoring or new embeddings.

Both decay conditions use identical initial weights and batch schedules.
The loss, learning rate, residual gain and graph cache remain unchanged.
References are persistence and a scalar correction fitted to training data
only. The original R04 run stays untouched. Seven synthetic tests and the
real-data preflight passed before the user trained the models. The automated review
reproduced validation predictions without retraining or test scoring.

The run is complete: **use `--verify-only` now**. Original execution commands
are retained below. Run individually from the repository root with
`CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_decay_comparison.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_decay_comparison.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_decay_comparison.py
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_decay_comparison.py --verify-only
```

Only after interruption, use the same runner with `--resume`. It skips
verified completed fits and restarts unfinished fits from initialization.
Completed runs use `--verify-only`, not `--resume`.

Results are saved in `residual_decay_comparison_v1/` under
`05_results/02_evaluation/epss_day30_to_day60/`. Read `summary.txt` first;
`summary.json` and per-fit files retain all seeds, paired validation
differences, correction variability, gradient counters and parameter norms.
Source/identity and output locks prevent silent overwriting or provenance
changes. Review this bounded comparison before deciding on full training.

See the [R05 protocol and register](../../EXPERIMENT_REGISTER.md#r05-bounded-weight-decay-comparison).

## Verified: DQ19 Training-Only Gradient Check

`diagnose_forecast_residual_gradients.py` compares data-loss gradients with
the decay term in the saved R04 v2 models. It uses 32 fixed training CVEs in
four batches, all four model configurations and three seeds, at initialization
and at the saved selected checkpoint. There are 96 backward passes and **zero
optimizer steps**. No validation/test predictions, embedding generation or
training are performed.

At selected checkpoints, the decay norm exceeds the data-gradient norm in
48/48 EPSS-encoder checks and 36/36 GNN-layer checks. The final output bias
shows the opposite pattern in all 48 checks. No gradient/output clipping
occurs. This is consistent with encoder shrinkage and nearly bias-only
corrections, but does not prove causality or reconstruct Adam's past updates.
30/32 sampled training targets equal persistence; keep this sample limitation
visible. The diagnostic is complete and reproduces exactly; eight tests pass.

Read `coreference_residual_gradient_audit_v1_review/analysis.txt` first, then
the detailed TXT/JSON reports in `coreference_residual_gradient_audit_v1/`,
both under `05_results/02_evaluation/epss_day30_to_day60/`.
Original results, caches, splits and model code remain unchanged.

Run individually from the repository root with `CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_residual_gradients.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/diagnose_forecast_residual_gradients.py --verify-only
```

The next candidate was a simple scalar-correction control followed by a
separately registered, small weight-decay-only ablation with a matched
EPSS-only control. R05 above now implements that comparison; DQ19 itself did
not implement or train it.
See the [DQ19 register](../../EXPERIMENT_REGISTER.md#dq19-training-only-gradientdecay-check).

## Verified: R04 v2 Full Development

The user's 12 fits now reproduce independently on all 197 validation CVEs.
Mean MAE is about 0.009217 for graph modes and 0.009221 for EPSS-only, versus
persistence at 0.00909675. All fits improve over matched R03 but none beats
persistence MAE. RMSE improves slightly; no recorded clipping remains.

Corrections are positive and nearly constant, especially for seeds 29/43.
162/197 cases are unchanged from their latest observed input; three large
increases account for 97.79% of persistence's absolute error. Checkpoint
inspection finds strong GRU/head parameter shrinkage, despite gradient flow.
This motivates a bounded training-only gradient-versus-regularization check,
not another blind full run or a claim that the TPG extractor is broken.

Read `coreference_residual_development_v2_review/analysis.txt` and
`analysis.json` under the evaluation root. Original outputs stay unchanged;
no retraining or test scoring. **The commands below document a completed run:
use `--verify-only` now.**

`run_forecast_coreference_residual_development.py` implements the fixed full
comparison: 1,023 training / 197 validation CVEs, four models, seeds 17/29/43,
maximum 100 epochs and patience 15. No split/grouping changes or test scoring.
The model is the same as v2 smoke; gain is fitted once from full training input
histories only. Cached graphs/embeddings are reused and R03 references verified.
The real-data preflight passes with full-training gain **0.00885**. The combined
COREF regression suite passes **57 tests**. The user subsequently completed
the dataset run; the automated review verified saved results without retraining.

Run individually from the repository root with `CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_coreference_residual_development.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_residual_development.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_residual_development.py
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_residual_development.py --verify-only
```

Results go to `coreference_residual_development_v2/` under the evaluation root.
Reports retain every seed, raw corrections, clipping, gradient diagnostics,
missingness strata and matched R03/persistence comparisons. No dataset fitting
was performed by the automated review. Preflight reproduces existing validation
references but performs no new fitting or output writes.

If interrupted, use the same runner with `--resume`: verified completed fits
are skipped and unfinished fits restart from fixed initialization. This is not
optimizer-state resume. For completed runs use `--verify-only`. The frozen
JSON's old planning status remains for hash compatibility; the full-run
identity records current implementation status.
See the [full R04 v2 register](../../EXPERIMENT_REGISTER.md#r04-v2-full-development).

## Verified: R04 v2 Scaled-Residual Smoke

The user run now reproduces independently. All four models select epoch 1,
with zero clipped predictions. Graph MAE is approximately 0.0001252 versus
v1's 0.0007433, but persistence remains lower at 0.0000875. EPSS-only MAE is
0.0001282. RMSE improves slightly versus persistence; the models help the two
changed CVEs but add errors to the 22 unchanged cases. This is not a graph-win
claim. Read `coreference_residual_smoke_v2_review/analysis.txt` and `analysis.json`
under the evaluation root. Original models and outputs remain unchanged.

The review also reproduces DQ07 and explains why all 5,968 source records are
not training examples. The 1,023/197 figures are the current eligible train/
validation cohort after chronological, related-group and observation rules;
2,670 eligible CVEs remain reserved for testing. Broad grouping may be overly
restrictive and deserves separate, outcome-independent review, not silent
relaxation. No full runner or new experiment is implemented by this review.
**The commands below document the completed smoke: use `--verify-only` now.**

`run_forecast_coreference_residual_v2.py` keeps R04's cohort, encoders and
training settings. It changes the correction to `tanh(s * head_output)` where
`s` is the median latest observed EPSS across the 32 training CVEs, bounded to
[0.001, 0.05]. This uses past training inputs only, not outcomes or validation
values. It reduces initial correction sensitivity without imposing a small
permanent correction cap. It can still clip or underfit; compare the saved
results with v1 and persistence before interpreting a benefit.
The real-data preflight passes with gain **0.00885**; the combined regression
suite passes **46 tests**. V2 dataset training was run by the user; the automated review
only reproduced saved results, without retraining or test scoring.

Run these individually from the repository root with `CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_coreference_residual_v2.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_residual_v2.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_residual_v2.py
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_residual_v2.py --verify-only
```

Results stay separate in `coreference_residual_smoke_v2/` under the evaluation
root. The runner first verifies v1, including its existing validation results;
preflight performs no v2 optimizer updates or output writes. Raw predictions,
corrections, clipping, gradient counters and TXT/JSON summaries remain available.
Do not rerun training into a completed output directory.

The subsequent full plan is saved in
`04_configs/coreference_residual_development_v2.json`: the unchanged 1,023/197
split, four models, seeds 17/29/43, up to 100 epochs and patience 15. The
separate full runner above now implements it; the smoke CLI stays smoke-only.
There is no requirement to beat persistence on the tiny smoke cohort.
See the [R04 v2 register](../../EXPERIMENT_REGISTER.md#r04-v2-training-history-scaled-residual-smoke).

## Verified: R04 Persistence-Residual Smoke

The user run is complete and independently reproduced. On 24 validation CVEs,
persistence MAE is 0.00008750, compared with approximately 0.000743 for each
graph model and 0.000552 for EPSS-only. All three graph models clip 20/24
validation predictions to zero. Training forward passes record no clipping;
encoder gradients flow on all 11 steps after the expected first-step zero.
This is not evidence of a disconnected branch. Persistence is exact on 22/24
CVEs, and none of the trained models beats it overall.

Read the separate `coreference_residual_smoke_v1_review/analysis.txt` and
`analysis.json` under the evaluation-results root. Original models, outputs
and caches remain unchanged. Before full expansion, the recommendation is a
separately versioned smaller-correction smoke comparison, with its scale
chosen from training data only and the other settings held fixed.
**The commands below document the completed run; use `--verify-only` now.**

`run_forecast_coreference_residual.py` trains four matched smoke models:
three fixed COREF graph configurations and an EPSS-only control. Each predicts
the latest observed past EPSS plus a learned correction, bounded to [0,1].
The final correction layer starts at zero, so initial predictions exactly
match float32 persistence, including when Day 30 is missing. No target mean
or future score enters initialization. Epoch zero is a check, not a selectable
checkpoint.

Use the existing 32 training / 24 validation CVEs, seed 17 and three epochs.
The graphs come from the verified DQ17 v2 full cache. Original R03 models,
DQ18 diagnostics and all caches stay unchanged. Full training is not exposed.

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_residual.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_residual.py
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_residual.py --verify-only
```

Results go to `05_results/02_evaluation/epss_day30_to_day60/coreference_residual_smoke_v1/`.
TXT/JSON reports include metrics, raw/clipped outputs, corrections, initialization
checks and gradient diagnostics. Output directories cannot be overwritten.
The zero final layer initially blocks upstream data-loss gradients; tests and
reports distinguish this expected first-step behavior from later gradient flow.
See the [R04 register](../../EXPERIMENT_REGISTER.md#r04-persistence-residual-smoke)
for the protocol, files, constraints and review criteria. Dataset training is
left to the user's commands; synthetic tests do not train on the real CVEs.

## Verified: Full DQ18 Saved R03 Branch Diagnostic

The user's full 197-CVE diagnostic now reproduces all 26,004 CVE/probe
predictions. All 12 checkpoints remain unchanged and all 2,364 EPSS-only
graph-control outputs are exact no-ops. No training or test scoring.

Unlike smoke, the full repeated-only and all-COREF models have larger mean
responses to EPSS swaps than graph swaps. Sensitivity is strongly seed-dependent.
Repeated-only graph swaps modestly worsen error; all-COREF graph-swap effects
have mixed signs. This does not overturn R03's failure to beat persistence.
Read `coreference_branch_diagnostic_v1/full_review/analysis.txt` and
`analysis.json` under the evaluation-results root for the separate interpretation.

The recommended next step is to specify a small persistence-plus-correction
fusion experiment with matched EPSS-only control, not expand the diagnostic.
The subsequent R04 smoke implementation is now documented above; DQ18 itself
does not train, and full residual development remains a later step.

The smoke history and original full commands below are retained for reference.
**Full output already exists: use `--scale full --verify-only`, not regeneration.**

The user's 16-CVE smoke run is verified and independently reproduced. All 12
model states remain unchanged, and all 192 EPSS-only graph-control predictions
are exact no-ops. Fusion models respond more to graph swaps than EPSS swaps on
this low-score sample, but this does not establish useful graph contribution.
The sample has no high-score targets or CVEs with 29 or fewer observed past
scores. The separate `smoke_review/analysis.txt` and `analysis.json` explain
these findings and the minus-0.01 clipping limitation.

Original full-run commands (now completed), without training:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/diagnose_forecast_coreference_branches.py --scale full
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/diagnose_forecast_coreference_branches.py --scale full --verify-only
```

`diagnose_forecast_coreference_branches.py` tests responses to swapping or
zeroing the EPSS/graph representations and shifting observed past EPSS scores.
It uses all 12 frozen R03 models. It does not train, change graphs, regenerate
embeddings or score the test set. Graph interventions on EPSS-only models must
have exactly zero effect. Input masks and ages remain intact under score shifts.

Original smoke commands are retained below for reference. The smoke folder
already exists; use its `--verify-only` command instead of regenerating it:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/diagnose_forecast_coreference_branches.py --scale smoke --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/diagnose_forecast_coreference_branches.py --scale smoke
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/diagnose_forecast_coreference_branches.py --scale smoke --verify-only
```

Reports are saved separately under
`05_results/02_evaluation/epss_day30_to_day60/coreference_branch_diagnostic_v1/smoke/`.
The plain-language TXT report explains what every probe and response metric
means. JSON retains per-CVE predictions, all seeds, donor maps and subgroup
results. Full analysis uses all 197 validation CVEs and requires matching smoke
artifacts first; see the [DQ18 register entry](../../EXPERIMENT_REGISTER.md#dq18-r03-branch-diagnostic).
Changing predictions does not itself prove useful or causal branch contribution.
Persistence-residual fusion training remains a subsequent, separate decision.

## Verified: R03 Matched Development Training

`run_forecast_coreference_development.py` consumes the verified full DQ17 v2
cache without changing it. It implements the fixed 1,023/197 comparison:
three COREF edge modes plus EPSS-only, each with seeds 17/29/43, and persistence.
Budget: maximum 100 epochs, patience 15. The user dataset run is complete;
all 12 checkpoints and their validation predictions reproduce.

Persistence MAE is 0.00909675, below every trained fit. Mean MAEs across
all three seeds are 0.01903434 (EPSS-only), 0.02392563 (repeated-only),
0.02707772 (all-COREF), and 0.03177332 (no-COREF). Seed-dependent narrow
prediction ranges and poor generalization need investigation; these results
do not establish that TPG improves forecasting. No test results were examined.

`review_forecast_coreference_training.py` creates the separate TXT/JSON review
in `05_results/02_evaluation/epss_day30_to_day60/coreference_training_development_v1_review/`.
It verifies the saved models, compares all seeds, examines missingness and
COREF groups, and records the largest persistence errors without excluding them.
The review is already generated. Reproduce it without overwriting:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_coreference_training.py --verify-only
```

Original training commands below are retained for reference; **do not rerun
training into the completed folder**. Use `--verify-only` for saved results.

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_development.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_development.py
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_development.py --verify-only
```

The preflight checks small and largest development graph batches without
fitting or scoring. If interrupted, `--resume` skips verified completed fits
and restarts an unfinished fit from its original initialization. No test
scoring, graph regeneration or hyperparameter overrides. Outputs go to
`05_results/02_evaluation/epss_day30_to_day60/coreference_training_development_v1/`.
See [R03](../../EXPERIMENT_REGISTER.md#r03-matched-development-training) for
the complete protocol, artifact descriptions, safeguards and recovery rules.

## Verified: DQ17 Development Cache

The new preparation CLI is under the TPG examples, where graph-generation
scripts belong: `01_tpg/02_examples/01_scripts/prepare_security_coreference_development_v2.py`.
It prepares three matched edge configurations for the existing 1,023 training
and 197 validation CVEs, using shared frozen embeddings and the unchanged v4
extractor. Start with its seven-CVE preview. Full preparation requires a verified
preview; unfinished runs can resume after checking completed artifacts.

The user's full v2 cache now passes verification for all 1,220 development
CVEs. Each configuration has the same 125,884 nodes; only the selected COREF
edges differ. Cache verification itself performs no training or test scoring;
the separate R03 training run is now complete.

The [experiment register](../../EXPERIMENT_REGISTER.md#dq17-matched-development-cache-and-r03-protocol)
contains commands, output files and the fixed R03 protocol: seeds 17/29/43,
maximum 100 epochs, patience 15, three graph variants, EPSS-only and persistence.
DQ17 itself does not train. The R03 entry point above now implements the next
stage; no real-data fitting was run during implementation. R02 stays frozen.

V1 stopped at `CVE-2021-22822` because SecBERT removes two U+FFFD replacement
characters that spaCy retains as token nodes. V2 keeps these nodes with zero
vectors, zero presence masks and explicit source-span audits. Other missing
text-token embeddings still fail. The 39 completed v1 entries are verified
before byte-for-byte reuse. Keep the old folders; use the v2 CLI, not a v1
resume. See the register for the corrected commands and findings.

`review_forecast_coreference_development.py` verifies the completed full cache
and writes separate TXT/JSON coverage findings to
`05_results/02_evaluation/epss_day30_to_day60/coreference_development_v2_review/`.
It counts node embeddings, edge types and CVEs affected by each ablation, and
checks the replacement-character audit. It does not generate predictions or
change the cache. The review output is protected from overwriting.

## Verified: R02 Matched COREF Training Smoke

`run_forecast_coreference_smoke.py` consumes the verified DQ16 cache, without
changing it. It compares three graph-edge configurations with EPSS-only and
persistence references on the same 32 training and 24 validation CVEs.
The three graph models share initial weights and batch order. Seed 17 and
three epochs are fixed; dropout is disabled for this controlled smoke check.
The user's run is complete and saved predictions reproduce exactly. Persistence
MAE is 0.00008750, compared with approximately 0.017875 for the graph models
and 0.04816025 for EPSS-only. All monitored branches received gradients, but
predictions remain nearly constant after three epochs. This small check does
not justify selecting a COREF configuration. No test scoring or full
expansion is exposed. Read the [R02 register entry](../../EXPERIMENT_REGISTER.md#r02-matched-coref-training-smoke)
for the full protocol, limitations, outputs and commands.

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_smoke.py --verify-only
```

This command reproduces saved validation predictions without retraining.
Do not overwrite the completed output. The separate interpretation is in
`05_results/02_evaluation/epss_day30_to_day60/coreference_training_smoke_v1_review/`.
Reports and checkpoints are saved separately in
`05_results/02_evaluation/epss_day30_to_day60/coreference_training_smoke_v1/`.

## Verified: DQ11 Packet and Source Findings

The user's packet reproduces exactly on the same 56 CVEs: 43 version links
and 67 unresolved mentions across 20 CVEs. Source inspection found a filter
bug that omits `16.1.x` after `F5 BIG-IP`, plus known unsupported release
forms and incomplete range/list structures. All human judgments remain
pending; the observations are separate automated analysis, not gold labels.

```bash
python EPSS_TPG/01_tpg/02_examples/01_scripts/analyze_security_product_packet.py --check-only
```

Findings: `01_tpg/09_annotation/07_product_version_review/packet_analysis_v1/`.
The original `packet_v1/` is unchanged. The narrow IP/product-name collision
is now corrected by the opt-in v3 extension; no full expansion or training. Read
[DQ11](../../EXPERIMENT_REGISTER.md#dq11-saved-productversion-source-review)
for review decisions and scope. No items are marked correct automatically.

## Verified: DQ10 v2 Small Preview

The separate v2 correction under `01_tpg/10_extensions/` and the same 56-CVE
preview are complete. All 31 combined tests pass. The four Desigo links now
name the actual products, `Snyk CLI` has a cleaner boundary, and the incomplete
Oracle name is rejected rather than guessed. Trailing bounds and patch
numbers are retained. Version mentions increase from 94 to 110; candidate
links change from 44 to 43. Unresolved mentions increase from 50 to 67 because
16 formerly missed forms are now explicit and one incomplete owner is rejected.
These counts do not measure accuracy. No full expansion or training.

Check the completed v2 preview without rebuilding graphs:

```bash
python EPSS_TPG/01_tpg/02_examples/01_scripts/audit_security_product_versions_v2.py --verify-only
```

Run from the project root with `CodeBERTFusion` active. Outputs, including
TXT/JSON analysis and corrected GraphSON, are stored under
`01_tpg/05_workspace/security_product_version_v2/smoke/`. There are no new
embeddings, model-ready graph tensors or test descriptions. Version bounds
are preserved as metadata, not silently added to the GNN feature contract.
The original v1 preview/review, extractor and caches remain unchanged;
all 56 v1 graphs reproduced exactly during the v2 run. Read
[DQ10](../../EXPERIMENT_REGISTER.md#dq10-productversion-extraction-preview)
for the results, remaining limitations and review required before any expansion.

## Verified: DQ09 Full Fusion Diagnostic

The full `tpg_fusion_diagnostic_full_v2/` run is complete and independently
reproduced by `review_forecast_tpg_fusion_diagnostic_v2.py`. All 7,683 numerical
probe rows agree; source inspection covers 1,220 development CVEs and 122 typed
security edges. Reconstruction IDs can differ between processes, but the
restricted structural correspondence and cached source evidence agree.
Original outputs and models are unchanged; no test scoring or training.

Key findings on all 197 validation CVEs:

- EPSS swaps increase error in 9/9 comparisons; graph swaps in 8/9.
- Removing all edges changes outputs much less than swapping whole graphs.
- Only 11 validation CVEs have typed security edges. Removing those edges has
  very small effects; generic copies of their relations remain.
- All three fusion MAEs (0.02610255, 0.02551741, 0.01892104) are worse than
  matched persistence (0.00909675).
- Eight validation version links raise product/version concerns. Examples
  include `php -> 7.18` for a sentence about `Pluck-CMS v4.7.18`, and plugin
  versions attached to WordPress. These are local source-based assessments,
  not independently labelled extraction accuracy.

Read the saved TXT/JSON review in
`05_results/02_evaluation/epss_day30_to_day60/tpg_fusion_diagnostic_full_v2_review/`
and [DQ09 in the experiment register](../../EXPERIMENT_REGISTER.md#dq09-saved-fusion-branch-and-security-relation-diagnostic).

Repeat read-only verification from the project root:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_tpg_fusion_diagnostic_v2.py --check-only
```

No new training command is needed now. The recommended extraction correction
is now implemented as the opt-in DQ10 preview above; a persistence-residual
fusion comparison remains a separately declared future experiment. Keep extractor
and model changes distinguishable. Do not overwrite completed graph caches,
add more text sources yet, or open the test set.

The frozen v1 scripts and failed full directory are retained for provenance.
The v2 diagnostic uses `tpg_cached_structure.py`; the v2 reviewer checks
numerical reproduction and source evidence without demanding process-identical
reconstruction IDs. Do not use the v1 reviewer for v2 outputs.

## Verified: R01 Full TPG Experiment

Model integration passed its smoke checks and the full run is now verified. See
[R01](../../EXPERIMENT_REGISTER.md#r01-current-description-tpg-regression) for
the shared-core architecture, fixed comparisons and user-run CLI commands.
The full run uses 1,023 training and 197 validation CVEs, five neural variants
and three fixed seeds. `review_forecast_tpg_full.py` reproduces all 15 selected
checkpoints, predictions, metrics, graph counts, embedding coverage and
early-stopping decisions. Its TXT/JSON findings are in the separate
`tpg_description_full_v1_review/` result folder; original runs are unchanged.

Persistence MAE is 0.00909675, lower than every neural fit. Fusion mean MAE is
0.02351366 versus 0.02412072 without graph messages; this small mean improvement
is not consistent across seeds. 162/197 endpoints match the latest input, and
five CVEs account for 98.36% of persistence error. Only 11 validation graphs
have SEC_ edges. Unembedded TOKEN nodes are 323 whitespace tokens and four
source replacement characters, which the saved tokenizer removes.

Diagnose saved-model branch use and relation coverage before a new training run.
The retrospective text limitation, existing splits and reserved-test restriction
remain in force. The earlier 32-training/16-validation smoke review stays in
`tpg_description_smoke_v1_review/` and is reproduced by `review_forecast_tpg.py`.

Implementation lives in `02_prediction/01_package/epss/forecast_tpg_data.py`
and `forecast_tpg.py`; generated graphs belong to `01_tpg/05_workspace/forecast_graphs/`.
Tests here: `python -m unittest test_forecast_tpg_full_review test_forecast_tpg_review test_forecast_tpg` (19 tests).

Read-only reproduction from the project root:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_tpg_full.py --check-only
```

## DQ08: Packet Verified; Evidence Decisions Pending

`review_grouping_keys.py` prepares DQ07's complete queue for human review.
The packet contains exact normalized evidence, source CVE/field locations and
source/code hashes. Full text is exposed only for already-excluded groups;
active-split text stays withheld. Editable decisions start unresolved, never
automatically excluded because of a homepage hint or repeated phrase.

The real packet and saved JSON/TXT validation reports reproduce. All 1,212
decisions are still untouched/unresolved: 629 URLs, 539 post texts and 44
descriptions, associated with 1,700 distinct CVEs. This completes setup, not
approval to remove evidence or change splits. From the repository root:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_grouping_key_packet.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_grouping_keys.py show --rank 1 --full
```

Run one command at a time. Output is the separate
`05_results/02_evaluation/epss_day30_to_day60/grouping_key_review_v1/` folder.
Packet generation is complete; do not rerun `prepare` into that folder.
Separate verified findings are in the sibling `grouping_key_review_v1_review/`
folder (`analysis.txt`, `analysis.json`, `README.md`). No decisions have been
recorded yet. The examples identify homepage links, long stored pages with
navigation, and identical short descriptions for bounded evidence review.
The generated README/TXT guide explains `record`, direct JSON editing and
validation. Validation writes `validation/analysis.txt/json`; it reports exact
key coverage, recorded decisions and unresolved counts, not model performance.

Reviewed decisions need reasons, evidence notes, reviewer identity and an explicit
timezone timestamp. Exclusions additionally need a predictor-safety plan.
`record` saves prior decisions by checksum; direct edits do not create history.
Partial review is allowed, and unresolved keys remain protected. Even a fully
resolved file does not authorize a split change; no adapter-ready exclusions
are exported. See [DQ08](../../EXPERIMENT_REGISTER.md#dq08-exact-key-grouping-decisions).

Tests: `python -m unittest test_grouping_key_packet_review test_grouping_key_review`
from this directory.

## Completed and Verified: Full-Dataset Coverage and Grouping Review (DQ07)

`audit_forecast_cohort_coverage.py` accounts for every source CVE using the
existing locked preparation. It separates assigned roles from task eligibility
and lists shared URLs/text hashes behind connected groups. No model is fitted,
no test outcomes are analyzed, and no split or source data is modified.

The source has 5,968 CVEs, not just the 797 training/163 validation CVEs used
by F04. Other records are reserved, split-excluded or lack observations needed
for that target. The largest connected group has 1,377 CVEs; the new queue
helps review whether some grouping links are too broad. It does not infer
that those CVEs are duplicates or that they can safely be returned to training.

The researcher completed the audit. `review_forecast_cohort_coverage.py`
reproduces all seven tables, checks hashes/CSV/text and reconstructs connected
groups independently by adjacency traversal. From the repository root:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_cohort_coverage.py
```

Grouping removes 459 candidate training, 208 candidate validation and 881
candidate test CVEs. The largest group contributes 1,329 of the 1,548 removals;
its other 48 CVEs were already outside date boundaries. Shared homepages,
generic description titles and repeated scraped/page text need manual review.
The 1,212 queued keys are not automatically invalid or independent recovery
opportunities. Actual examples are retained in the separate review.

Original outputs remain at `05_results/02_evaluation/epss_day30_to_day60/cohort_coverage_audit_v1/`:
TXT/JSON explanations, seven CSV/JSON tables, a coverage figure, status and hashes.
Shared evidence counts overlap and are not estimates of recoverable CVEs.
The reserved test is inspected for structure/availability only, never numerical
score summaries or performance. The loader checks raw/prepared values only
for integrity. Read the [protocol](../../EXPERIMENT_REGISTER.md#dq07-full-dataset-coverage-and-grouping-review)
before proposing any separately versioned grouping change.

The sibling `cohort_coverage_audit_v1_review/` stores the TXT/JSON explanation
and README; original scores, splits and audit files remain unchanged. Do not
rerun the original audit into its existing destination. Review text excerpts
come only from already-excluded groups, not active test records.

Synthetic tests: `python -m unittest test_forecast_cohort_coverage test_forecast_cohort_coverage_review`.
No TPG-GNN training on the revised forecast target is claimed by this audit.

## Completed and Verified: Remaining Errors and Temporal Shifts (DQ06)

`audit_forecast_extrema_shift.py` was run by the researcher and its results
reproduce. It compares training/validation score distributions, repeated values,
future-change timing, and validation error concentration for all nine saved F04
comparators. It also checks stored EPSS version-date consistency and reports
date coincidences, without treating them as causes.

No training, new forecasts, source repair, downloads or test analysis. The
preflight replays the completed residual run for verification; the audit itself
uses its saved validation predictions. Future outcomes are descriptive only,
never new model inputs or exclusion rules. Reproduce the separate review from
the repository root:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_extrema_shift.py
```

`review_forecast_extrema_shift.py` checks hashes/provenance, reproduces all eight
tables, checks CSV/JSON/text agreement and independently checks errors and timing.
Median latest scores differ: 0.00885 in training and 0.00051 in validation.
130/163 validation CVEs have future scores identical to their latest input;
five CVEs account for 98.27% of persistence's total joint error. All nine reported
training future-version transitions coincide with the largest adjacent change,
but no validation CVE reports a transition. Coincidence is not causality.

Original results stay in `05_results/02_evaluation/epss_day30_to_day60/extrema_shift_audit_v1/`:
TXT/JSON explanations, eight CSV/JSON tables, three figures, status and hashes.
The sibling `extrema_shift_audit_v1_review/` holds the separate TXT/JSON review
and README. Do not rerun the original audit into its existing destination. See
[DQ06](../../EXPERIMENT_REGISTER.md#dq06-remaining-extrema-errors-and-temporal-shifts)
for the exact protocol and limitations. Synthetic tests from this directory:
`python -m unittest test_forecast_extrema_shift test_forecast_extrema_shift_review`.

## Completed and Verified: Ordered Residual Extrema (F04-Residual)

`02_prediction/01_package/epss/forecast_minmax_residual.py` was run by the researcher
and its results reproduce. It reuses the current F04 cohort, masks,
loss, optimizer and three fixed seeds, with a persistence-initialized two-output
correction head. Outputs are sorted and clipped; raw proposals and correction
diagnostics remain visible. No original run files are changed and no test set
is scored. From the repository root, reproduce the read-only review:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_minmax_residual.py
```

Joint MAEs are 0.00552955/0.00557464/0.00556700 for seeds 17/29/43, versus
persistence 0.00551816. Forecast levels vary across CVEs again, but the learned
corrections do not beat persistence overall. The review verifies initialization,
selected checkpoints, stopping rules, raw/final outputs, metrics and hashes.

Original artifacts remain in `05_results/02_evaluation/epss_day30_to_day60/minmax_residual_validation/`.
The sibling `minmax_residual_validation_review/` holds TXT/JSON findings, real
examples and constant/changing-future comparisons. Full design and limits are in
[F04-Residual](../../EXPERIMENT_REGISTER.md#f04-residual-ordered-corrections-to-the-latest-score).
Run synthetic tests here with
`python -m unittest test_forecast_minmax_residual test_forecast_minmax_residual_review`.

## Completed and Verified: GRU Diagnostic (DQ05)

`diagnose_forecast_minmax_sequence.py` investigates the narrow F04-GRU forecast
range using the existing checkpoints for all three seeds. It measures variation
inside the GRU, output-layer derivatives, sensitivity to observed input scores,
and responses to four fixed input changes. Six artificial constant-score histories
provide an additional check. They are clearly separated from real CVEs.

No model is trained, no parameter is updated, and no test prediction is scored.
Training-loss gradients are computed only as measurements at the saved weights;
they are not used to update weights. Validation-loss gradients are not computed.
Missing-score masks, observation ages, splits and the original run remain unchanged.

The researcher completed DQ05. `review_forecast_minmax_diagnostic.py` reproduces
all measurements and checks hashes, CSV/JSON agreement, reports and independent
scalar derivatives without fitting. Reproduce the review from the repository root:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_minmax_diagnostic.py
```

Original results remain under
`05_results/02_evaluation/epss_day30_to_day60/minmax_gru_diagnostic_v1/`.
The separate sibling `minmax_gru_diagnostic_v1_review/` holds the TXT/JSON
interpretation. Do not rerun the original diagnostic into its existing destination.

Artificial constant-score histories at 0 and 1 give maximum forecasts of
0.01045/0.01059 (seed 17), 0.01037/0.01329 (29) and 0.01050/0.01094 (43).
These probes have no actual future target. Weights changed and gradients remain
nonzero, but sensitivity is weak. Typical training targets also exceed typical
validation targets: median future minima 0.00885 versus 0.00050.
The checks found no reproduction discrepancy and do not establish a unique cause.
The ordered residual-extrema comparison is now completed and reviewed above.
DQ05 alone did not validate it. Preserve persistence and all fixed seeds.
See [DQ05](../../EXPERIMENT_REGISTER.md#dq05-fixed-checkpoint-gru-sensitivity-diagnostic).
Synthetic tests:
`python -m unittest test_forecast_minmax_diagnostic test_forecast_minmax_diagnostic_review`.

## Completed and Verified: F04-GRU

`02_prediction/01_package/epss/forecast_minmax_sequence.py` adds the declared
ordered-head sequence model on the same F04 cohort and targets. It reuses the
five-channel sequence loader, reports all three fixed seeds, and compares saved
references without refitting them. The researcher completed training and
`review_forecast_minmax_sequence.py` verified selected checkpoints, predictions,
initialization, stopping rules, metrics and hashes without training/test scoring.
Joint MAE is about 0.0292 for each seed versus persistence 0.00551816. The model
outputs very similar small extrema across CVEs; ordered outputs alone did not help.

From the repository root:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_minmax_sequence.py
```

The `minmax_gru_validation_review` folder contains `analysis.txt/json` and
`README.md`. Original run artifacts are unchanged.

See the [experiment register](../../EXPERIMENT_REGISTER.md#f04-gru-ordered-sequence-extrema)
for the fixed protocol, CLI and artifacts. Synthetic tests from this directory:
`python -m unittest test_forecast_minmax_sequence test_forecast_minmax_sequence_review`.

## Completed and Verified: F04-Linear

`02_prediction/01_package/epss/forecast_minmax.py` implements the first F04
comparison: predict the minimum and maximum EPSS in Days 31-60 using only
Days 0-30. Persistence and past observed range are matched references for
fixed two-output ridge. Full future coverage is required, producing 797 training
and 163 validation CVEs without altering the endpoint cohort or split boundaries.
Every future target score is checked against the frozen archive comparison.
The researcher completed the run. `review_forecast_minmax.py` reproduces saved
predictions, eligibility, target/archive alignment, corrections and metrics without
refitting or test scoring. TXT/JSON findings are in `minmax_linear_validation_review`.

Persistence joint MAE is 0.00551816 versus linear 0.00687337. The past-range
reference improves maximum error but worsens minimum error; no overall gain.
The review explains the constant future scores and 12 corrected linear pairs.

From the repository root:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_minmax.py
```

See the [experiment register](../../EXPERIMENT_REGISTER.md#f04-linear-future-minimum-and-maximum-epss)
for CLI commands, output files and interpretation. Synthetic tests from this
directory: `python -m unittest test_forecast_minmax test_forecast_minmax_review`.

## Completed and Verified: F01-Tree

The researcher completed the fixed shallow tree on the unchanged DQ04 cohort.
`review_forecast_tree.py` checks saved hashes, predictions, node counts and mean
training targets, reports and independent error metrics without fitting or test
scoring. Findings are in `tree_validation_review/analysis.txt/json` and `README.md`.
MAE is 0.02477230, compared with persistence 0.00909675. The four leaf averages
do not improve overall forecasting; do not automatically retune the model.

From the repository root:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_tree.py
```

See the [experiment register](../../EXPERIMENT_REGISTER.md#f01-tree-fixed-nonlinear-comparator).
Synthetic-only tests in this directory:
`python -m unittest test_forecast_tree test_forecast_tree_review`.

## Completed and Verified: F01-Huber

The implementation is in `02_prediction/01_package/epss/forecast_huber.py`, not
this analysis directory. It uses the completed DQ04 freeze and existing ridge
model. The full specification and CLI are in [F01-Huber](../../EXPERIMENT_REGISTER.md#f01-huber-fixed-robust-linear-baseline).
The researcher completed the run. `review_forecast_huber.py` reproduces saved
training diagnostics and validation predictions without fitting or test scoring,
checks artifact hashes and calculates independent error metrics. Its TXT/JSON
findings are in `huber_validation_review` beside the original run.

Huber MAE is 0.00909609 versus persistence 0.00909675 on the same 197 CVEs.
This tiny gain does not resolve the large EPSS increases; keep persistence as
the reference. The fitted residual scale is unusually small and must not be
interpreted as forecast uncertainty.

From the repository root:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_huber.py
```

Synthetic-only tests from this analysis directory:
`python -m unittest test_forecast_huber test_forecast_huber_review`.

## Completed: Missing-Data Sensitivity (DQ04)

`forecast_missingness_sensitivity.py` freezes the current missing-data policy
and audits saved validation predictions by Days 0-30 input completeness. It
keeps the original source, targets and split unchanged, does not impute scores,
and does not fit or select a model. Test performance remains closed.

From the repository root in `CodeBERTFusion`:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/forecast_missingness_sensitivity.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/forecast_missingness_sensitivity.py
```

The full diagnostic writes only to the new `missingness_sensitivity_v1` result
folder and refuses existing destinations. It produces TXT/JSON findings,
per-CVE coverage and missing-reason tables, all-model/all-seed matched subgroup
metrics, a coverage figure, a freeze manifest and output checksums. It does not
download archives or rewrite raw/processed data. Recovery of the nine archive
dates is optional, not a prerequisite. Policy and experiment details are in
[DQ04](../../EXPERIMENT_REGISTER.md#dq04-frozen-missing-data-policy-and-validation-sensitivity).

The researcher has completed this run. `review_forecast_missingness.py` checks
the frozen input/output hashes, reconstructs every coverage/metric table, and
independently checks MAE/RMSE/group-MAE. It writes separate TXT/JSON/Markdown
findings to `missingness_sensitivity_v1_review` without changing the locked run.
The finding is that persistence retains lower MAE in every declared subset;
this is not a causal test of missingness. All model seeds remain reported.

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_missingness.py
```

Tests from this script directory:

```bash
python -m unittest test_forecast_missingness_sensitivity test_forecast_missingness_review
```

## Completed: Legacy-Aware EPSS Verification (DQ03)

`verify_forecast_snapshots_v2.py` is a separate follow-up to the locked DQ02 run.
It skips pre-EPSS dates, validates old-format archives with explicit URL/manifest
date evidence, and keeps modern header checks. Only the original nine in-range
HTTP-403 dates can be downloaded, with explicit opt-in. The original dataset,
collector, verifier and completed results are not modified.

From the repository root with `CodeBERTFusion` activated:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/verify_forecast_snapshots_v2.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/verify_forecast_snapshots_v2.py
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/verify_forecast_snapshots_v2.py --resume --download-missing
```

Run the offline second command before the opt-in third command. Both write to
the new `epss_snapshot_verification_train-validation_v2` results folder, not the
original run. A full offline run reparses all available archives and may take
several minutes. Resume checks checkpoint and archive hashes. Missing values
stay missing; newly found scores go to separate recovery-candidate tables only.
Failures remain unresolved; no alternative source is silently substituted.
Full instructions and scope are in [DQ03](../../EXPERIMENT_REGISTER.md#dq03-legacy-aware-verification-and-targeted-retries).

The completed v2 run and its opt-in retry have been reviewed: 72,708 matches, 1,300 entries
missing in both sources, 77 pre-EPSS entries and 335 unresolved entries; no
new recovery candidates. The latest saved invocation has download permission
enabled and nine new HTTP-403 attempt records. No archives were recovered.
The commands above are now the completed procedure, not a recommendation to
retry indefinitely. Investigate the exact missing dates via the documented
historical archive repository or the provider before attempting another source.

`review_forecast_snapshots_v2.py` checks pinned inputs/code, archive/manifest
hashes, checkpoint hashes, reconstructed comparisons and saved CSV/JSON/reports.
It compares v2 with DQ02 and writes TXT/JSON/Markdown and per-CVE summary tables
to a separate sibling `_v2_review` folder. It does not reparse archive CSVs,
use the network, change source values, train models or compare test scores.
Run it after the next completed v2 invocation to refresh the review:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_snapshots_v2.py
```

Synthetic tests, from this script directory:

```bash
python -m unittest test_forecast_snapshot_v2_review test_forecast_snapshots_v2 test_forecast_download_audit test_forecast_snapshot_review test_forecast_snapshots
```

## Historical EPSS Verification (DQ02)

`verify_forecast_snapshots.py` compares stored and archive values without
modifying either. It reuses the repaired collector, reads the old cache without
writing, and permits downloads only with `--download-missing`. Missing/invalid
archives do not become zero scores or evidence that a CVE was absent.

From the repository root in `CodeBERTFusion`:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/verify_forecast_snapshots.py --scope diagnostic --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/verify_forecast_snapshots.py --scope diagnostic
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/verify_forecast_snapshots.py --scope diagnostic --resume --download-missing
```

The first command does no archive parsing or writes. The second is offline;
the third explicitly allows network downloads for unresolved dates. Checkpoints
allow interrupted runs to resume, but source/code/scope identity must match.
Use `--scope train-validation` for all endpoint training/validation CVEs,
not just the three selected diagnostic cases. Default output folders are
`epss_snapshot_verification_<scope>/` under
`EPSS_TPG/05_results/02_evaluation/epss_day30_to_day60/`.

Read the [experiment register](../../EXPERIMENT_REGISTER.md#dq02-historical-epss-snapshot-comparison)
for scope counts, comparison statuses, commands and limitations. The output
has TXT/JSON/Markdown explanations, CSV/JSON comparisons, CVE summaries,
disagreements and unresolved rows, plus archive provenance and output checksums.
Full archives may require substantial disk space. The tests in
`test_forecast_snapshots.py` use tiny synthetic archives and mocked downloads.
No source correction or training was performed during implementation.
The researcher has since completed the diagnostic: all 176 stored scores match
their archives; seven entries are absent in both sources. The 183 archives were
reparsed offline by `review_forecast_snapshots.py`, which also verifies locks,
CSV/JSON agreement, comparisons and summaries. Its sibling review folder contains
TXT/JSON/Markdown and `uniform_verification_plan.json` for the next 1,220-CVE run.

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_snapshots.py
```

Pass `--run-dir` to review a completed uniform run later. The review never fetches
archives or edits a completed run. Regression tests are in
`test_forecast_snapshot_review.py`. Read DQ02 in the experiment register for
the full-scope commands; this remains data verification, not new training.

The uniform run is now completed but not fully resolved. Before resuming it,
read DQ02's download diagnosis: legacy files were transferred successfully but
rejected by the date-header requirement, some requested dates predate EPSS,
and only nine in-range dates have unresolved HTTP-403 responses.
`audit_forecast_downloads.py` hashes downloaded files and reparses legacy payloads,
writing TXT/JSON/Markdown and separate comparison tables in the sibling
`epss_snapshot_verification_train-validation_download_audit/` folder.
It preserves all original run statuses. `test_forecast_download_audit.py`
covers date boundaries, HTTP-error classification and legacy payload handling.

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/audit_forecast_downloads.py
```

## Day-30 Evidence Availability Audit

`audit_forecast_evidence.py` is the next diagnostic step after the completed
residual GRU. Read the current [experiment register](../../EXPERIMENT_REGISTER.md)
for DQ01 and the prior runs. The audit verifies the locked data and completed
run, inspects all endpoint validation CVEs, and reports large EPSS changes and
the dates associated with descriptions, summaries, reference URLs and posts.
An early stored date is not automatically proof of historical text availability.
No training, model inference or test error analysis is performed.

From the repository root, with `CodeBERTFusion` active:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/audit_forecast_evidence.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/audit_forecast_evidence.py
```

The default output is
`EPSS_TPG/05_results/02_evaluation/epss_day30_to_day60/evidence_availability_audit/`.
It contains TXT/JSON/Markdown explanations, CVE-level and evidence-level CSV/JSON,
coverage and platform tables, two figures, and output checksums. Existing
destinations are refused; raw sources and prior results are never overwritten.
`test_forecast_evidence.py` checks dates, missingness, groups, evidence semantics,
duplicate flags, protected outputs and synthetic figure/report generation.

The user run is complete and verified. `review_forecast_evidence.py` checks all
audit artifacts, reproduces records and tables, recalculates endpoint groups
from raw scores and explains error concentration and provenance limitations.
Its TXT/JSON/Markdown outputs live in the sibling
`evidence_availability_audit_review/` folder. It does not overwrite the audit.
Tests are in `test_forecast_evidence_review.py`. The current interpretation is
explicitly tied to the reviewed preparation checksum, not arbitrary new cohorts.

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_evidence.py
```

## Target Definitions and Adapter (11 October 2026)

The [target plan](../../../05_results/03_dataset_analysis/01_revision_dataset/13_epss_targets/README.md)
defines Day-60 regression and the separate change, direction and min/max tasks.
`test_forecast_data.py` tests the new isolated `epss/forecast_data.py` adapter;
`verify_epss_target_plan.py` checks all prepared records against the raw source,
reruns tests and publishes the short plan and real examples. No models train.
The conservative candidate split and the missing direction class in validation
need review before experiments start. The older classifier is unchanged.

## Training Feasibility (10 October 2026)

Read the [feasibility report guide](../../../05_results/03_dataset_analysis/01_revision_dataset/12_training_feasibility/README.md)
before designing new training runs. `assess_training_feasibility.py` recounts
the repaired EPSS data, checks usable forecast windows, matches posts to source
dates, verifies the earlier temporal rows and reproduces legacy adapter hazards.
`build_training_feasibility_report.py` produces the TXT/JSON/Markdown report,
28-experiment matrix and LaTeX; `verify_training_feasibility_report.py` checks
the compiled PDF and reruns the focused tests. None trains a model or changes
the raw data or training implementation. Earlier audits retain their own scope.

## Revised Summary Samples (9 October 2026)

Read the [sample comparison guide](../../../05_results/03_dataset_analysis/01_revision_dataset/09_summary_quality/sample_comparison/README.md)
first. The six revised files contain the same 50 CVEs, selected from 956 records
with both GitHub and non-GitHub URLs. This is a paired follow-up, not a new
full-dataset accuracy estimate.

Run these in the CodeBERTFusion environment, from `EPSS_TPG/`:

```bash
python 02_prediction/03_analysis/01_revision_dataset/compare_summary_samples.py
python 02_prediction/03_analysis/01_revision_dataset/check_revised_summary_generator.py
python 02_prediction/03_analysis/01_revision_dataset/build_summary_sample_report.py
bash 03_scripts/03_maintenance/build_paper.sh summary_quality_findings
python 02_prediction/03_analysis/01_revision_dataset/verify_summary_sample_pdf.py
```

The comparison script reads all six files, the original 50 rows and both full
source versions; it verifies hashes, unchanged source fields, paired counts,
saved prompts, fetching logs and duplicate text. The second script reproduces
current generator problems offline. The third writes the updated TXT/JSON
findings, figures and LaTeX chapter. No sources are edited or fetched by these
analysis scripts; no API calls, training, commits or pushes are performed.

From this script directory, the applicable tests are:

```bash
python -m unittest test_summary_sample_comparison test_summary_quality.QualityRuleTests -v
```

The old `OfflineScraperCharacterizationTests` describe the September generator;
they are not the test suite for the revised source.

## EPSS Recollection Verification

`compare_epss_recollection.py` compares the latest enriched JSON with the
separate output of the repaired collector. It verifies all original scores,
all non-EPSS fields, the 91 expected dates per CVE, status counts and archive
provenance. `test_epss_recollection.py` contains the offline regression tests.
Read [the recollection guide](../../../05_results/03_dataset_analysis/01_revision_dataset/11_epss_recollection/README.md)
for outputs, limitations and commands. This is a missing-score collection,
not a replacement of every upstream dataset or a revalidation of every score.

## EPSS Collector Audit (30 September 2026)

`audit_epss_collector.py` reviewed the original upstream collector with offline
fixtures and optional checks of explicitly selected daily archives. It does
not fill scores or repair the collector. Read the
[findings and code walkthrough](../../../05_results/03_dataset_analysis/01_revision_dataset/10_epss_collector_review/README.md)
for the historical TXT/JSON reports and subsequent repair verification.
The original audit is hash-pinned and must not be rerun against the repaired
collector as though it were the same source version.

## Summary Quality Extension (29 September 2026)

`analyze_summary_quality.py` checks every generated-summary field in the current
5,968-CVE snapshot. `summary_scraper_checks.py` tests selected upstream functions
offline; `test_summary_quality.py` tests the audit rules and characterizes
scraper behavior. No source files are modified and no LLM is called.
Read the [summary-quality report guide](../../../05_results/03_dataset_analysis/01_revision_dataset/09_summary_quality/README.md)
for the findings, exact output filenames, limitations and rerun commands.
The updated dataset PDF includes a dedicated plain-language chapter.

## Latest Snapshot: EPSS, Summaries and Posts

For source commit `857fbf6533a68ede6c21b53ed57784720300b038`, start with
`analyze_enrichment_update.py`. It checks the nine downloaded data files,
profiles fields, compares the new enrichment stages, verifies the post join,
and tests selected collection functions offline. `test_enrichment_update.py`
contains its focused regression tests. From `EPSS_TPG/`, run:

```bash
python -m unittest discover -s 02_prediction/03_analysis/01_revision_dataset -p 'test_enrichment_update.py' -v
python 02_prediction/03_analysis/01_revision_dataset/analyze_enrichment_update.py
```

Read the paired reports through the
[new report index](../../../05_results/03_dataset_analysis/01_revision_dataset/07_epss_smp_update/README.md).
This is initial schema, lineage and readiness verification, not a replacement
for the earlier statistical figures or temporal study. It does not train models.

**Historical scope below:** the six-file analyzer and source audit are pinned
to `a1e32ca3f16052b987767c013a49e2441b972e55`. Their default source checks are
not valid rerun commands for the newer checkout and may reject it. Keep their
results associated with the original snapshot; do not change their expected
commit just to bypass those checks.

This analysis reads all six selected revision CSV/JSON files without modifying
them. It does not fetch URLs, call APIs, build TPGs or train a model. The files
represent related dataset stages, not six independent samples.

## 1. Understand What Is Being Analyzed

The unit of analysis is **one CVE record**. These files describe a selected
vulnerability cohort and successive updates to it, not six independent datasets
or a random sample of all NVD vulnerabilities. A field count is not a count of
usable model inputs.

The source directory is `SummTPGVul/Sec4AI4Sec-EPSS/Data_Files/`, relative to the
main repository. The analysis reads these six files in full:

| Individual file | Type | Role and observed size |
|---|---|---|
| `cves_unique_with_source_links_clean.csv` | CSV table | Base cohort: 5,692 CVEs, 15 columns. URL lists are stored as JSON strings inside cells. |
| `cves_unique_with_source_links_clean.json` | JSON array of objects | The same base cohort, with native lists and numeric values. |
| `cves_merged_refetched.csv` | CSV table | Refreshed cohort: 5,968 CVEs, 26 columns. Earlier values are retained under `orig_` names. |
| `cves_merged_refetched.json` | JSON array of objects | The refreshed cohort in JSON. Contains 2,445 non-standard `NaN` tokens. |
| `cves_merged_with_url_dates.json` | JSON array with nested reference-history objects | Refreshed records plus history metadata: 5,968 CVEs, 28 fields. Inherits the same `NaN` tokens. |
| `cves_merged_with_url_dates_vc_kev.json` | JSON array with nested history and catalogue evidence | Final enrichment: 5,968 CVEs; 29 keys on unlisted records and 39 on listed records. Inherits the same `NaN` tokens. |

`fetch_url_dates.py` in that source directory is a **Python preparation script**,
not another dataset. The audit inspects it and tests its history-fetch function
with synthetic responses. It does not run a real fetch.

The pinned source commit is
`a1e32ca3f16052b987767c013a49e2441b972e55`. The counts above describe that
snapshot; the generated reports include its hashes and the time of analysis.

## 2. Read the Scripts in This Order

| File | Type | Purpose |
|---|---|---|
| `README.md` | Markdown documentation | This guide: inputs, methods, commands and reading order. |
| `analyze_revision_dataset.py` | Python entry point | Decodes and validates records, profiles all observed fields, compares stages, checks potential leakage and writes paired reports. Start at `main()`, then `run_analysis()`. |
| `plots.py` | Python plotting module | Creates eleven figures in PNG and SVG. The main report also contains their plotted values. |
| `test_analysis.py` | Python unittest suite | Synthetic tests for bad input, missingness, dates, joins, empty groups, plots and failed report publication. |
| `03_scripts/03_maintenance/audit_revision_data.py` (relative to `EPSS_TPG/`) | Python audit | Checks the pinned snapshot against Git LFS hashes, expected counts and stage preservation; probes selected upstream functions offline. |
| `03_scripts/03_maintenance/test_revision_audit.py` (relative to `EPSS_TPG/`) | Python unittest suite | Regression tests for the source audit. |

Dependencies: Python 3.10+, NumPy, pandas, SciPy and matplotlib. These are
available in the existing `CodeBERTFusion` environment. No NLP model is needed.

## 3. Run the Checks and Analysis

Run these commands one by one:

```bash
cd /home/ayounas/Text_property_Graph/EPSS_TPG
conda activate CodeBERTFusion
python -m unittest discover -s 02_prediction/03_analysis/01_revision_dataset -p 'test_*.py' -v
python -m unittest discover -s 03_scripts/03_maintenance -p 'test_revision_audit.py' -v
python 02_prediction/03_analysis/01_revision_dataset/analyze_revision_dataset.py
```

Defaults resolve relative to the checkout, not the current working directory.
For a fixed future-date cutoff, add `--as-of 2026-09-19T00:00:00Z`.
Otherwise the report records execution time in UTC as its cutoff.
This option changes date diagnostics, not the dataset snapshot or training split.

`--source` accepts a directory containing the six named input files; `--output`
selects another report directory outside the source dataset repository. For the
default source, the existing pinned revision integrity/code audit also runs.
The custom source must follow the same CVE-level schema and contain the six
named files; it is not a generic reader for arbitrary datasets. The pinned
Git-LFS/code audit is skipped for a custom source, and that is stated in the
report. The normal command replaces the existing named outputs.

## 4. Read the Reports in This Order

The default output is
`05_results/03_dataset_analysis/01_revision_dataset/`:

For every pair below, `.txt` is a plain-text report for reading and `.json` is
the structured counterpart for code. They contain the same statistics.

| Read next | Individual files | What to look for |
|---|---|---|
| 1 | `00_analysis.txt`, `00_analysis.json` | Overall findings, dataset lineage, methods, recommendations, software versions, hashes and plotted values. |
| 2 | `05_findings_explained.txt`, `05_findings_explained.json` | The separately reviewed, plain-language explanation of each finding and warning. This interpretation is updated after review, not automatically by the script. |
| 3 | `02_feature_dictionary.txt`, `02_feature_dictionary.json` | Each field's meaning, observed types, missingness and possible role. Roles are cautions for task design, not a training configuration. |
| 4 | `03_source_integrity_audit.txt`, `03_source_integrity_audit.json` | What passed, which upstream checks failed, and what these checks cannot establish. |
| 5 | The six per-file pairs below | Full details for the particular snapshot or format you plan to use. |

All these per-file reports are in `01_file_profiles/`:

| Individual files | Dataset being profiled |
|---|---|
| `cves_unique_with_source_links_clean_csv.txt`, `cves_unique_with_source_links_clean_csv.json` | Base CSV after explicit decoding. |
| `cves_unique_with_source_links_clean_json.txt`, `cves_unique_with_source_links_clean_json.json` | Base JSON. |
| `cves_merged_refetched_csv.txt`, `cves_merged_refetched_csv.json` | Refreshed CSV after explicit decoding. |
| `cves_merged_refetched_json.txt`, `cves_merged_refetched_json.json` | Refreshed JSON, including non-standard missing values. |
| `cves_merged_with_url_dates_json.txt`, `cves_merged_with_url_dates_json.json` | Date-enriched JSON and nested history fields. |
| `cves_merged_with_url_dates_vc_kev_json.txt`, `cves_merged_with_url_dates_vc_kev_json.json` | Final JSON, conditional enrichment and nested evidence lists. |

These profile files are **reports about data**, not normalized copies of the
dataset. Their `.json` extension should not be confused with the source JSON.

Each text report includes the complete structured statistics as well as its
readable introduction. JSON reports reject nonfinite output numbers. Source
`NaN` tokens are counted and handled only in memory, never rewritten upstream.
Input hashes are checked again at the end of the run.

The script first generates outputs in a temporary directory. Input or analysis
failures leave existing reports untouched. After all checks pass, matching
report and figure files are replaced one at a time; `00_analysis.json` is
published last. This is not a whole-directory atomic transaction. Unrelated
files, this README and the reviewed `05_findings_explained` pair are preserved.

The generated reports and figures are local, Git-ignored results. This script,
its tests and the navigation README files can be committed normally. A fresh
clone must download the selected LFS files and rerun the analysis to recreate
the outputs; the source data is not copied into the reports folder.

## 5. Read the Figures

Every row names two individual files inside `04_figures/`. **PNG** is a raster
image for viewing; **SVG** is a scalable image for documents. The charts do not
contain trained-model results.

| Individual files | Question answered |
|---|---|
| `01_file_overview.png`, `01_file_overview.svg` | How many CVEs and distinct top-level fields does each file contain? |
| `02_feature_missingness.png`, `02_feature_missingness.svg` | Which fields are unavailable in each file? Gray means a field is absent from the entire schema. |
| `03_membership_missingness.png`, `03_membership_missingness.svg` | Does field availability itself reveal catalogue membership? An absent class has an undefined percentage. |
| `04_publication_membership.png`, `04_publication_membership.svg` | How do CVE counts and observed membership vary by publication year? Calendar gaps keep their actual spacing. |
| `05_cvss_coverage.png`, `05_cvss_coverage.svg` | How do stored scores and missing severity differ by CVSS version? Invalid or missing scores are excluded from histograms and counted in the report. |
| `06_numeric_associations.png`, `06_numeric_associations.svg` | Which eligible numeric quantities vary together in the CVSS 3.1 cohort? Read pairwise sample counts alongside the coefficients. |
| `07_reference_hosts.png`, `07_reference_hosts.svg` | Which hosts are referenced by the largest number of distinct CVEs? |
| `08_text_reference_sizes.png`, `08_text_reference_sizes.svg` | How long are descriptions and reference lists within each membership group? The horizontal axis is log-transformed. |
| `09_score_revision.png`, `09_score_revision.svg` | Which stored CVSS 3.1 scores changed between original and refreshed records? |
| `10_temporal_intervals.png`, `10_temporal_intervals.svg` | How do recorded publication, modification, reference and catalogue dates differ? These are different clocks, not verified prediction horizons. |
| `11_reference_history_matching.png`, `11_reference_history_matching.svg` | How many reference-history entries used exact or case-insensitive matching? Counts are entries, not CVEs. |

## 6. Understand the Checks and Their Limits

The input contract rejects duplicate JSON keys, duplicate or malformed CVE IDs,
empty datasets, ambiguous CSV headers, short/long CSV rows, failed numeric/list
decoding, non-boolean membership flags and incompatible known field types.
Required identifiers, descriptions and reference containers must exist.
Optional missing metric values remain missing; unknown additional fields are
profiled without inventing their meaning. An actual Git LFS pointer is rejected
by its header, not by an arbitrary small-file size threshold.

Within JSON, preservation checks distinguish an absent key, `null`, `NaN`, a
blank string and different numeric types. CSV/JSON equivalence is a different
check: it deliberately normalizes CSV blanks and JSON missing values for their
storage formats. Removed schema fields are also reported explicitly.

`integrity_errors: []` means the selected snapshot checks passed. It does not
mean the data are ready for prediction. Readiness warnings now follow observed
schema, documentation and isolated code checks rather than unconditional old
warning strings. The general warning about historical availability remains a
limitation of this offline analysis.

## 7. Interpret the Findings Carefully

- A union field count is not the number of model predictors. Target-linked fields, IDs and audit metadata need separate treatment.
- Nested denominators count enclosing objects or list elements, not necessarily CVEs.
- Absent keys, nulls, nonfinite numbers and blanks are separate missing states. Empty lists remain observed empty lists; false and zero are not missing.
- CVSS versions are categorical. The correlation figure uses current CVSS 3.1 records and masks original scores from other versions. Pairwise sample sizes are saved alongside coefficients.
- Missingness conditioned on catalogue membership can expose label leakage. Catalogue nonmembership does not establish that a CVE is unexploited.
- URL checks distinguish GitHub subdomains, archive hosts, FTP references and repeated schemes. No link availability or evidence validity is established without fetching it.
- Negative event intervals can reflect publication/enrichment clocks or date precision. They are not automatically bad records or valid time-to-exploitation targets.
- Description/URL overlap is a reason to inspect split contamination, not a reason to automatically delete distinct CVEs.
- Extreme values are flagged descriptively, not removed. When IQR is zero, the outlier count is `null` with `iqr_rule_status: not_applicable_zero_iqr`. High-cardinality frequency tables are truncated with omitted-category counts; all records contribute to statistics.

Read the main TXT report first, then the field dictionary and per-file profiles.
Use the JSON companions for further checks instead of scraping the text tables.

## 8. Run the Publication-to-VulnCheck Temporal Extension

This separate study reads the verified final enriched JSON, checks it against
`00_analysis.json`, and writes only into the new `06_temporal_vckev/` report
subfolder. It does not rerun the baseline, change raw files or train a model.

| File | Type | Read it for |
|---|---|---|
| `temporal_analysis.py` | Python entry point | Timestamp normalization, exact deltas, exclusive categories, cumulative labels, cutoff evidence, sensitivity scenarios, CVSS/reference/CISA summaries and provenance. Start at `main()`, then `run()`, `derive_row()` and `horizon_label()`. |
| `temporal_plots.py` | Python plotting module | Nine descriptive figures, each saved as PNG/SVG, with plotted data in the summary JSON. |
| `test_temporal_analysis.py` | Python unittest suite | Synthetic regression cases for interval boundaries, cumulative labels, precision, censoring, metadata conflicts, safe feature roles and reproducibility. |

From `EPSS_TPG/`, in `CodeBERTFusion`:

```bash
python -m unittest discover -s 02_prediction/03_analysis/01_revision_dataset -p 'test_*.py' -v
python 02_prediction/03_analysis/01_revision_dataset/temporal_analysis.py
```

Read the [temporal report guide](../../../05_results/03_dataset_analysis/01_revision_dataset/06_temporal_vckev/README.md)
for every output name, type, figure and finding. The main table contains 5,968
CVE rows; `prediction_windows` contains three horizon rows per CVE. Tables have
CSV/TXT/strict-JSON versions; the summary, schema and feature policy have paired
TXT/JSON reports. Unknown is `NA` in text tables and `null` in JSON, never zero.

The temporal script's `--source` is a **single final JSON file**, unlike the
baseline script's directory argument. `--baseline` identifies its verified
manifest, `--source-readme` supplies documentary cutoff evidence and `--output`
selects a separate derived folder. Default paths resolve from the checkout.

There is no implicit primary observation cutoff. The upstream documented run
day is used only in explicitly named `documented_snapshot_target_*` sensitivity
columns. Primary labels account for midnight precision; point-timestamp labels
are also provided separately. All negative deltas retain unknown targets.

Only with actual capture evidence may a caller provide `--snapshot-cutoff`
(full timestamp with timezone), `--cutoff-provenance` and
`--confirm-complete-snapshot` together. These are recorded as operator assertions,
not independent verification. The baseline's `--as-of` diagnostic option does
not establish observation completeness and is not a temporal-label cutoff.

Input hashes and all derived rows are checked before staged outputs replace
matching files, with the summary JSON published last. This is file-by-file
publication, not an atomic directory swap. Existing manual documentation and
all baseline outputs remain untouched. The main table includes outcome-linked
audit fields and must not be fed wholesale into a model.
