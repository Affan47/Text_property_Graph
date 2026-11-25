# Experiment Register

## ET01 closed: observed high Day-60 EPSS, 18 November 2026

**Decision: STOP_NO_TPG.** In the locked Reddit v2 development comparison,
the direct description-plus-structured classifier found 31 of 62 positives
in 403 reviews versus 33 on average for the saved regression rankings.
Neither paired lower bound exceeded zero. The structured-only result (35
hits) was secondary. No graph fit, alternate threshold, or official holdout
release follows. This task concerns EPSS >= 0.1, not catalogue membership or
future exploitation. [Results brief](../00_documentation/06_papers/02_pdf/et01_threshold_results_brief.pdf)
and [study and replay guide](05_studies/09_epss_threshold/README.md).

## CC02 EPSS Publication Delay, 17 November 2026

The required 2026-11-17 file was unavailable at 11:40:52 UTC: configured GitHub
archive HTTP 404, dated publisher URL HTTP 403, publisher current-file redirect
to 2026-11-16. This supports waiting for publication, not a negative/missing-score
classification. The valid date/path convention and previous-day availability
were checked separately; no replacement score date is authorized. The failed
collection created no completed archive, so the subsequent verification had
nothing to verify. No classifier preparation or training was released.

A separate companion, `04_check_epss_publication.py`, records bounded HEAD-only
checks and can delegate to the unchanged CC02 collector only when the configured
dated source is available. HTTP 200 is not content verification: the original
date/header/gzip/CSV checks must still pass. Completed resume and verification
are offline. `WAITING_FOR_REQUIRED_DATE` returns exit status 2 without a traceback
or a fabricated missing-score export. There is no automatic polling loop.

Receipt: `05_results/09_catalogue_classification/04_epss_publication_v1/checks/diagnosis_20261117_v1/`.
Frozen source policy, role hash, labels, collector and config are unchanged.
All 57 catalogue/input/publication regression tests passed. The three recorded
status requests downloaded no scores and sent no CVE IDs or credentials.

## CC02 Screened Inputs and Grouped Inner Roles, 17 November 2026

Separate package `catalogue_inputs_v1`; frozen CC01 code/results are unchanged.
Only original training-role candidates are adapted. Nominal 70/15/15 roles use
SHA-256 seed 6029 on whole evidence components, without target stratification,
seed search or redraw. Exact accepted D/M text, identical CVEs across datasets,
and existing Reddit evidence groups are joined. URLs alone do not merge CVEs.
This is retrospective group-held-out development, not a chronological forecast
or a new independent test. The two collections share 817 training CVEs.

| Dataset | Fitting (listed) | Selection (listed) | Assessment (listed) |
| --- | ---: | ---: | ---: |
| Reddit | 17,105 (379) | 3,870 (88) | 3,624 (78) |
| Older | 2,102 (294) | 381 (43) | 387 (46) |

All 26,652 unique CVEs are retained in 23,760 components; largest = 233.
The screen quarantines 69 Reddit and 386 older documents, excludes 23,336 bare-ID
Reddit posts, and retains D for every candidate. Prose M remains for 2,396 Reddit
and 2,808 older candidates. Text rejection does not delete a CVE. Only CVSS v3
fields and cleaned, deduplicated reference lists enter structured preparation;
all catalogue enrichment and original EPSS metadata are excluded. General
exploitation wording can remain, so this is not a proof against all answer proxies.

Role SHA-256: `a1c36ff45143627a115f79602117e186fe94075d7406dbde4a98527896365a9a`.
Output: `05_results/09_catalogue_classification/03_matched_inputs_v1/freeze/`.
CLI: `03_analysis/08_catalogue_classification/03_prepare_classification_inputs.py`.
The `collect-epss` stage is opt-in and public, requires no API key, and accepts
only a verified internal date of 2026-11-17. No matching saved archive was found
during implementation. The final `prepare` stage is blocked until collection;
its synthetic replay passed with fitting-only preprocessing and explicit nulls.
No training, embeddings, official validation/test joins or FC01 edits occurred.

## CC01 Snapshot Collected and Independently Reviewed, 17 November 2026

User collection succeeded. Backup UTC 2026-11-17T10:42:42.355Z; 5,954 unique
catalogue CVEs. Exact archive, receipt and training labels reproduce offline.
Reddit: 24,599 candidates, 545 LISTED, 24,054 NOT_LISTED, zero UNKNOWN.
Previous: 2,870 candidates, 383 LISTED, 2,487 NOT_LISTED, zero UNKNOWN.
Listed prevalence is 2.22% / 13.34%. The all-negative accuracy baseline would
be 97.78% / 86.66%, with zero positive recall and F1; no classifier was fitted.

Every legacy positive appears in the new snapshot; 1 Reddit and 7 older saved
unverified negatives are listed now. These are snapshot disagreements, not
established historical transitions or collection-error diagnoses. The training
populations overlap in 817 CVEs, 39 listed, with identical current labels.
Absence from this complete backup is not proof of no exploitation.

Receipt SHA-256: `84f353e468672a6e0d5a7fa6010685719a3353c2e2a2bebc64e070f6ac133ac7`.
Review: `05_results/09_catalogue_classification/02_snapshot_review_v1/catalogue_v1/`.
CLI: `03_analysis/08_catalogue_classification/02_review_catalogue_snapshot.py --verify-only`.
All 24 snapshot/review tests pass. No new requests, training, FC01 modifications
or held-out label joins. Next: dated EPSS, predictor disclosures, then fixed
grouped inner roles/settings. The label gate alone does not release training.

## CC01 Catalogue Classification Approved and Preparation Implemented, 17 November 2026

Separate target: retrospective membership in one authenticated complete VulnCheck
backup. No EPSS-threshold label, future-exploitation claim or modification of
FC01. Registered comparisons: prevalence, dated EPSS-only, EPSS/structured
logistic and tree, and matched pooled/TPG D and DM with EPSS/structured inputs.
AP primary; precision/recall, F1, calibration and fixed-budget metrics retained.

The new protocol, official-backup collector, immutable receipts and training-only
label adapter pass 18 offline tests. Frozen scope: 24,599 Reddit and 2,870 older
training candidates. All labels remain UNKNOWN before an authenticated download;
old false flags are not adopted. No network requests, fits or held-out label joins
were performed. Date-aligned EPSS, source-disclosure review and the fixed inner
training protocol remain required before classifiers can run.

Implementation: `epss/catalogue_classification_v1/`. CLI and full commands:
[CC01 guide](05_studies/08_catalogue_classification/README.md).
Outputs: `05_results/09_catalogue_classification/01_snapshot_v1/catalogue_v1/plan/`.
FC01 Run 2 remains the unchanged regression close-out; it was not launched here.

## Research Direction Review, 17 November 2026 (No New Experiment)

Master-report Section "Research Direction and Classification" separates
high-EPSS labels from VulnCheck snapshot membership and future exploitation.
It records that predictive superiority remains unestablished, while verified
graph/tensor infrastructure and data-quality findings are reusable contributions.
Classification alone does not solve the target or evidence limitations: an
all-low classifier achieves 99.23% accuracy on FC01's 8,053 cases while finding
none of the 62 high-score CVEs. The official EPSS and VulnCheck definitions,
threshold distinction and catalogue-label requirements are linked in the report.

FC01 settings and completed artifacts remain unchanged. Its Run 2 is still the
bounded close-out, not a venue for new classification arms. Any later catalogue
study requires verified snapshot outcomes and separate approval; CT09's export
does not authenticate original negatives. A complete authorized backup with
dated receipts is proposed instead of thousands of per-CVE requests. No new
catalogue download, fit, label change or held-out scoring was performed.

## FC01 Run 1 Completed, Reviewed 17 November 2026

All 75 fixed fits are complete: five model rows x five seeds x three original
CT07 folds, 8,053 out-of-fold CVEs and EPSS-v2 Day-60 targets. No official
validation/test scoring. The independent review verifies all fit locks and
recomputes 245 seed/partition metric rows from the saved predictions.

Mean seed-wise RMSE: structured 0.03575746, pooled D 0.03566554, TPG D
0.03636248, pooled DM 0.03527905, TPG DM 0.03524434; mean constant 0.03700242,
median constant 0.03664138. All six primary group/seed intervals contain zero
at nominal 95% and family-adjusted 99.1667% coverage; the CVE/seed sensitivity
agrees. TPG D improves over pooling in 1/5 seeds; TPG DM in 2/5. The latter's
0.00003471 point advantage is not an established improvement.

All learned models improve overall point RMSE against the constants, but every
one has higher MAE than the median constant. Pooled D leads AP (0.122765),
pooled DM AUROC (0.895839), and TPG DM top-100 hits (16.2 of 62). Ranking is not
classification accuracy or exploitation detection. High-score targets account
for 86-89% of neural squared error and remain substantially underpredicted.
Fold/source slices are descriptive, not alternate primary gates. No models
or seeds are promoted, and no new target/loss/readout is selected.

Next: unchanged Run 2, all 25 official-development fits, then lock all models
before the planned single test release. No new training is launched by review.
Original output: `27_final_comparison_v1/01_reddit/runs/run1/report/`.
Separate review: `30_fc01_run1_review_v1/review/` under fresh-study results.
CLI: `03_analysis/07_fresh_studies/35_review_final_run1.py --verify-only`.
The master PDF includes full-cohort, paired, source/fold, rare-score, seed,
training-history and real-CVE example tables plus interval/error figures.

## FC01 Loader Correction Verified, 16 November 2026

Stopped only the user-authorized integration job. Preserved its code, protocol,
prepared data and four completed one-epoch checkpoints under
`27_final_comparison_v1/01_reddit/preserved/pre_npz_loading_fix_20261116/`.
The loader now decodes each of nine NPZ members once and displays progress.
All 98 unit tests pass, including archive-read-count, exact float bytes,
empty assessment, forbidden target-array and invalid-input regression tests.
Full contexts load in 1.12, 1.37 and 1.58 seconds in the recorded observation;
these are input-loading timings, not epoch estimates. All three fold payloads,
roles, labels, masks and scientific settings are unchanged. Four frozen
checkpoints (structured, pooled D, pooled DM, TPG D) reproduce all 1,072
selection predictions and their CSVs exactly. No parameter updates were made.

The new protocol changes code pins only; original locks remain preserved.
Restart the five-row integration gate under the corrected identity before
scientific fitting. No assessment or official test scoring was performed.
Review: `05_results/08_fresh_studies/29_fc01_loader_fix_v1/review/`.
CLI: `03_analysis/07_fresh_studies/34_verify_fc01_loader_fix.py --verify-only`.
The master PDF and reproducible report builder include the correction.

## FC01 Integration Startup Defect, 16 November 2026

The user started the five-row integration gate. Both first workers were CPU-
active, repeatedly reading `fold_1/predictors.npz`; structured and pooled-D
then completed. `data.load_context` accesses a full compressed array inside
each per-CVE row loop. A bounded 32-row predictor-only probe is identical after
decoding once (32 archive accesses versus one). This is a loading-performance
defect, not a scientific result or evidence of model convergence.

The running pinned code and checkpoints remain unchanged. Hold the 75-fit run
pending a verified, provenance-preserving loader-only correction. No test access.
Diagnosis and verification CLI: `03_analysis/07_fresh_studies/33_diagnose_fc01_startup.py`.
Saved review: `05_results/08_fresh_studies/28_fc01_startup_review_v1/review`.

## FC01: Final Two-Run Implementation, 16 November 2026

Implemented separately from every completed study. Both settings and reporting
rules are frozen before fitting: five learned rows (structured, pooled D,
TPG D, pooled DM, TPG DM), training mean/median controls, five seeds
17/29/43/59/71, raw-score MSE, earliest strict best selection RMSE, best-only
checkpoints and deterministic recovery. Six primary paired contrasts use
20,000 joint evidence-group/seed resamples with family adjustment; CVE/seed
resampling is a sensitivity analysis. No ensemble substitution.

Run 1 keeps the original CT07 main folds and 8,053 assessment CVEs: 75 fits.
Run 2 uses the official 24,213 train / 972 validation roles: 25 fits, followed
by the single locked 3,977-CVE test evaluation. Run 1 is previously examined
development evidence, not independent confirmation. Both tasks remain
retrospective Day-60 score estimation, not historical exploitation forecasts.

Run-1 preparation reuses 14,817 graph objects across 13,378 distinct original-
training CVEs. Fold vectors and optimization-only structured transforms match
CT07 exactly. The largest conservative D/DM tensor working set is 11.995 GiB,
before activation/runtime overhead. Graphs remain resident in RAM per fit;
insufficient budgets stop rather than dropping records or evicting graphs.
CPU/BLAS threads are fixed at four per fit; concurrency is memory-gated.

Synthetic unit checks pass. No research-data integration fit, full training,
test adaptation or test scoring has been launched by this implementation.
The single required integration gate is fold 1, seed 17, one epoch for each
of the five learned rows. Scientific fits initialize fresh afterward.

All 25 official-development checkpoints and their replay receipt must be
locked before any test predictor cache can be built. Test labels require
the separate explicit `--release-test`; afterward new fitting is blocked.
All earlier experimental artifacts remain unchanged. The preceding master
PDF is preserved and the current report documents the implemented status.

CLI: `03_analysis/07_fresh_studies/32_run_final_comparison.py`.
[Complete CPU commands and release sequence](05_studies/07_combined_text_fresh/23_FINAL_COMPARISON_COMMANDS.md).

## DS01 Completed: Primary Progression Rule Failed

All 30 full fits and reports reproduced; 8,053 original-training assessment
CVEs. No official validation/test scoring. On the primary 7,797 crossing cases
(62 risers), description-plus-past-EPSS AP is 0.082676 versus 0.064131 for past
EPSS; joint group/seed change +0.018545 [-0.010515, +0.076495]. AUROC is
0.845174 versus 0.850182. The frozen progression gate fails. Positive strict-
label AP sensitivity does not replace the primary rule. Stop this surrogate
branch without a universal claim that all text forecasting must fail.

Day-60 mean seed-wise RMSE/MAE is 0.036237/0.008350; the median control is
0.036641/0.004611. These are point estimates, not a new paired significance
claim. The prior structured-D reference differs in capacity and seed count.

The 24-graph DM benchmark verified sources, saved representation parity and
unchanged parameters. Measured-segment speedup is 4.15x warm, 2.13x including
preload across three passes. No whole-epoch speed or whole-cohort RAM claim.
Next: a separate retrospective Day-60 within-version MSE graph-versus-pooled
protocol, full working-set inventory and matched-input/readout gate. No new fit.

Review CLI: `03_analysis/07_fresh_studies/31_review_description_surrogate.py`.
[Results, limits and CPU verification commands](05_studies/07_combined_text_fresh/21_DESCRIPTION_SURROGATE_RESULTS.md).

## DS01: Frozen Description-Only Surrogate, 16 November 2026

All 7,957 assessment NVD histories are complete; no further download is needed
for that scope. The remaining 5,421 histories are not requested. Complete
receipts do not establish historical wording or a model available at Day 7.

Implemented the bounded Reddit-v2 current-description test: 3 existing CT07
main folds x 10 seeds = 30 fits, standalone 768-value D vectors, no CVSS,
reference counts, source flags or EPSS in the neural predictor. Day-60 MSE,
earliest strict best selection-RMSE checkpoint, fixed AdamW settings. Best-only
checkpoints, deterministic replay, CPU/BLAS pins and canonical CSVs are included.

The post-fit primary AP check uses 20,000 paired evidence-group/seed resamples
on 7,797 eligible cases. Strict labels, AUROC, per-fold/seed metrics, constants
and the frozen CT07 structured-D reference remain visible. Passing is only
exploratory; an inconclusive outcome stops this surrogate branch, not every
possible historical text model. Reddit v1 and graph MSE require separate plans.

Ten synthetic tests and offline predictor-export verification pass. No dataset
smoke or scientific fit has been launched by this implementation. Full training
requires a completed matching smoke. The user runs the commands.

CLI: `03_analysis/07_fresh_studies/30_run_description_surrogate.py`.
Outputs: `05_results/08_fresh_studies/25_description_only_surrogate_v1/01_reddit/v2/`.
[Frozen specification and CPU commands](05_studies/07_combined_text_fresh/20_DESCRIPTION_ONLY_SURROGATE.md).
The master PDF is updated with the complete-history summary and DS01 design;
the preceding edition and the untrained pre-integration export are preserved.

## Day-7 Availability Audit and No-Fit Graph Loading Gate

Implemented separately from the frozen CT studies. Public NVD history collection
is opt-in and resumable, with raw request receipts, pagination and source hashes.
The locked original-training scope is 7,957 assessment CVEs first and 13,378
cohort CVEs total. No long download or training was launched. Description/CVSS
reconstruction distinguishes dated support, no recorded assignment by cutoff,
undated origin and unresolved history. Reference counts/flags and model-time
availability remain separate blockers; a description/CVSS-only subset ranking
must not be described as fully historical validation.

The one-isolated-day assumption remains 7,797 eligible / 136 unresolved, with
62 observed risers. CVE-2022-30525 was already above 0.1 by Day 7 and is not an
eligible riser; its later CVSS addition/EPSS jump is a timing example, not proof
of the cause of the 62-riser gain. Completed EPSS, catalogue and graph results,
checkpoints and raw data remain unchanged.

A six-graph Reddit training-role RAM-loading benchmark passed exact tensor/GNN
representation parity and parameter-immutability checks. No optimizer or targets
were used. This is a bounded performance gate, not another scientific fit or
a full-cohort speed claim. Twenty-four availability tests and four RAM-loader
tests passed. Public transport still needs the user's bounded live pilot.

CLI: `03_analysis/07_fresh_studies/24_audit_day7_availability.py`.
Benchmark: `03_analysis/07_fresh_studies/25_benchmark_graph_loading.py`.
Results: `05_results/08_fresh_studies/19_day7_input_availability_v1/` and
`20_graph_loading_benchmark_v1/`.
[Specification, safeguards and individual commands](05_studies/07_combined_text_fresh/15_DAY7_AVAILABILITY_AUDIT.md).

## CT09: Recovered Catalogue Export and Bounded Outcomes

The broader repository search found a 5,251-record VulnCheck-shaped export on
`modify-summ-generation`, Git LFS SHA-256
`3a1e4fbaf9f982ac4df927b5dbb252098edec2e5c902878963e86b2d11969a39`.
Its byte identity and both training-only comparisons reproduce. It corroborates
528 of 544 Reddit and all 376 older saved-listed candidates, with matching
addition dates. Sixteen Reddit saved positives are absent; older CVE-2017-5753
is saved false but present. There is no original export-time/completeness
envelope, so absence does not release negative labels or overwrite originals.

Implemented a separate opt-in receipt collector: the fixed CT08 pilot plus,
with `--include-disagreements`, the locked export discrepancies yields 35 Reddit
and 21 older training-only requests. HTTP/network failures remain UNKNOWN;
verification is offline; credentials are not stored. No actual VulnCheck API
requests were sent because no key was available. Fifteen new synthetic tests
passed; 32 together with CT08. No training or holdout outcome analysis occurred.

Review CLI: `03_analysis/07_fresh_studies/21_review_saved_catalogue.py`.
Collection CLI: `03_analysis/07_fresh_studies/20_verify_catalogue_outcomes.py`.
Results: `05_results/08_fresh_studies/16_catalogue_checks_v1/`.
[Findings and individual commands](05_studies/07_combined_text_fresh/14_CATALOGUE_OUTCOME_VERIFICATION.md).

## CT08: Training-Only Catalogue Target Readiness

Status: implemented, both audits reproduced and 17 focused tests passed. No
training, downloads, official validation/test analysis or role changes. CT07
and all earlier completed results are preserved.

Scope removes the EPSS-target requirement, not the role/exposure protections:
24,599 Reddit candidates (544 saved listed) and 2,870 older candidates (376
saved listed). The original EPSS training subsets remain 24,213/537 listed and
2,546/303 listed. Additional candidates are audit rows, not approved new splits.

The configured receipt locations provide zero usable verified checks. This
does not prove every saved flag wrong or that no receipts exist elsewhere.
Failed requests remain unknown, never negative. Snapshot classification needs
a defined verified collection-time policy; future inclusion additionally needs
event-time semantics, at-risk/follow-up support and cutoff-valid content.
Negative deltas and uncertain midnight order remain explicit. No future target
is released. Stored post dates are not exact-wording history verification.

Runner: `03_analysis/07_fresh_studies/19_audit_catalogue_readiness.py`.
Config: `04_configs/07_fresh_studies/10_catalogue_readiness_v1.json`.
Results: `05_results/08_fresh_studies/15_catalogue_readiness_v1/{01_reddit,02_previous_multisource}/audit/`.
Guide and individual CPU commands:
[CT08 readiness](05_studies/07_combined_text_fresh/13_CATALOGUE_TARGET_READINESS.md).

## CT07 Full Execution: Policy-Specific Controls

Status: **all 54 full fits completed and independently reproduced**, including
checkpoint predictions, policy-specific constants, and 20,000-resample reports.
The earlier 40 regression tests and 54 smoke replays passed; three additional
saved-review tests pass. No official validation/test scoring or new training
was performed during the results review.

Separate package/config/CLI and output root, preserving the CT07 adapter and all
earlier artifacts. Fixed main and publication-restricted D/DF/DM comparisons:
27 fits per policy, three seeds, unchanged pooled architecture/MSE/AdamW and
selection settings. No GNN training, older-v3 fitting or official holdout access.
Full training was run by the user after the smoke-parity gate.

Completed OOF assessment covers 8,053 CVEs: 834 with accepted posts, 7,219
without; 62 targets >=0.1, including 26 with posts. Main mean-seed RMSE is
D 0.036312768, DF 0.035576757, DM 0.036631561. Main DM-minus-D is +0.000318793
(grouped 95% interval -0.000679354 to +0.001442220). DM is effectively level
with the optimization-median control (0.036641380); superiority over both
constants is not established for any overall neural arm. DF beats the mean
control but not the median under both interval units.

The restricted-publication sensitivity favors DM over D by -0.000985926
(-0.001748007 to -0.000205012 grouped), but not over DF or both constants.
The supported relative gain is concentrated in post-absent cases, not a
demonstrated inference-time benefit from post text. All six predefined DM
useful-estimation/content gates remain false. Keep this exploratory study
frozen; no TPG expansion, re-splitting or subgroup promotion follows.

Review: [interpretation, tables and CPU commands](05_studies/07_combined_text_fresh/12_WITHIN_VERSION_FULL_RESULTS.md).
Saved review: `14_within_version_training_v1/01_reddit/v2/full_review_v1/`.
Review CLI: `03_analysis/07_fresh_studies/18_review_within_version_training.py`.

Constants are computed from each policy/fold's optimization scores only and
frozen before full neural fits. Mean-control overall RMSE is 0.037002422 for
main and 0.036393000 for the restricted policy; median is 0.036641380 for both.
Full/post-present/post-absent arm-versus-own-control contrasts use 20,000 paired
CVE/component resamples, seed 6029, beside unchanged DM-minus-D primary and
secondary DM-minus-DF/DF-minus-D contrasts. Intervals are nominal exploratory
intervals, not multiplicity-adjusted independent confirmation.

Freeze the CT05 reference on the exact 8,053 assessment CVEs, preserving its
old fold mapping and external DF provenance. D/DM RMSE is 0.039418785/0.037585039;
both exceed the main constants overall. Post-present DM RMSE is 0.083303764,
below the constants by point estimate. Different training/selection roles mean
this is not a matched version-only intervention. Relative arm improvement is
still reportable when constants win, but cannot establish useful estimation
beyond those controls. A content-specific claim also needs DM versus DF.

Guide: [frozen settings, interpretation gates, artifacts and CPU commands](05_studies/07_combined_text_fresh/11_WITHIN_VERSION_FULL_EXECUTION.md).
Runner: `03_analysis/07_fresh_studies/17_run_within_version_training.py`.
Output: `05_results/08_fresh_studies/14_within_version_training_v1/01_reddit/v2/`.

## CT07: Reddit-v2 Pooled Adapter and Smoke Gate

Status: 29 CT06/CT07 tests passed; fresh-process protocol, adapter, all 54 smoke
checkpoints/predictions and report reproduction passed. All 58 smoke graph
means matched fresh compact reductions. Full preparation subsequently completed
and reproduced independently; full training remains disabled.

Full-preparation review: 14,817 cached representations cover 13,378 unique CVEs;
25 of the 13,403 eligible CVEs are purged in every fold. All six policy/fold
combinations have 53 structured features. Main/restricted selection and
assessment IDs, scores, text and masks match exactly; numeric scaling is fitted
separately on each optimization cohort. Potential OOF assessment is 8,053 CVEs,
62 high scores, 834 with posts and 26 with both. No full fitted-model metrics.
TXT/JSON/CSV findings and the support chart are in `full/preparation_review_v1/`;
`16_review_within_version_adapter.py --verify-only` reproduces that review.

New versioned package/config/CLI; CT05/CT06 artifacts and official roles remain
unchanged. Older-v3 training is deferred, with its audit retained. Reddit main
optimization counts remain 4,228/6,278/8,371. A separately declared sensitivity
keeps publication >=2022-02-04 in optimization only, giving 1,188/3,247/5,340
CVEs (16/36/53 high scores). Selection and assessment IDs and labels stay fixed.
This sensitivity follows inspected cohort support and changes size/composition
and update count; it is not an outcome-blind original hypothesis or causal test.

D, DF and DM use identical pooled heads and optimization-fitted structured
features. DF reuses standalone D vectors with the post-presence flag; DM uses
the frozen combined description/post mean. No G/N text, EPSS predictors, GNN
training or new embeddings. Graph/embedding receipts and cached vector hashes
are checked, with fresh compact-reduction parity on smoke. The unchanged CT05
fitting engine is reused with distinct CT07 output identities.

Smoke: 54 two-epoch fits across main/sensitivity x three folds x D/DF/DM x
seeds 17/29/43; best-only checkpoints and exact replay. Main/sensitivity use
16/8 optimization CVEs per fold and the same 8 selection/8 assessment CVEs.
Smoke intervals use 20 resamples for wiring, not scientific inference. Future
full reporting retains the main DM-minus-D primary contrast and 20,000-resample
CVE/component intervals. Full training remains disabled; official validation
and test are not adapted or scored. No smoke outcome changes the settings.

Guide: [review, exact source contracts, outcome rules and commands](05_studies/07_combined_text_fresh/10_WITHIN_VERSION_POOLED.md).
Runner: `03_analysis/07_fresh_studies/15_run_within_version_pooled.py`.
Results: `05_results/08_fresh_studies/13_within_version_pooled_v1/01_reddit/v2/`.

## CT06: Within-Version Fold and Support Preflight

Implemented and reproduced, without training. The new primary pool is 13,403
Reddit training CVEs with v2 Day-60 targets; the secondary pool is 1,511 older
training CVEs with v3 targets. All original roles, CT05 artifacts and other-version
records remain unchanged. Date-only 40/60/80% expanding assessment blocks and
nested latest-20% selection keep ties together. CT05's whole-training evidence
components are retained, including links through excluded-version records.

Potential assessment covers 8,053 Reddit CVEs (62 high scores, 834 with posts)
and 909 older CVEs (14 high scores, all with posts). Minimum total role-count
gates pass. Reddit fold 2 has only 14 high scores, 5 with posts; all older folds
have thin tails and no post-absent assessment cases. Both datasets retain
`SUPPORT_REVIEW_REQUIRED`; no failed fold is dropped and no cut is adjusted.
Sixteen focused tests and real fresh-process plan/audit reproduction passed.

The recorded future comparison is pooled D/DF/DM plus identically fold-fitted
structure, MSE and seeds 17/29/43. DF supplies D text plus the M-presence flag,
not post text. New preprocessing and a bounded smoke gate must precede training.
No GNN, EPSS-history predictor, test scoring or wider ridge search is enabled.
The design is exploratory, not fresh confirmation or historical-text forecasting.

Guide: [exact rules, verified counts, limitations and CPU commands](05_studies/07_combined_text_fresh/09_WITHIN_VERSION_PREFLIGHT.md).
Runner: `03_analysis/07_fresh_studies/14_audit_within_version.py`.
Results: `05_results/08_fresh_studies/12_within_version_audit_v1/{01_reddit/v2,02_previous_multisource/v3}/{plan,audit}/`.

## CT05: Training-Only Standalone D versus DM Pooled MSE

CT03's 20,000-resample merged-minus-mean RMSE interval crosses zero; per-source
minus mean stays below zero; per-source minus merged crosses zero. Original
predictions and intervals remain frozen. CT05 tests source addition on CT04
grouped training folds, with separate chronological optimization, selection and
assessment roles. No official roles change. Standalone D and merged DM use
matched pooled heads plus optimization-fitted structured features, raw MSE and
three seeds: 18 neural fits per dataset, plus fixed and selection-tuned ridge
controls. Best-only checkpoints, internally pinned numerical threads, canonical
OOF predictions and 20,000-resample CVE/component intervals are implemented.

Both full datasets are now complete: 18 neural fits each plus linear controls.
Fresh-process checkpoint/prediction/report replay passed. Older D/DM OOF RMSE
is 0.148769/0.146654; Reddit is 0.058139/0.057614. Both overall paired intervals
include zero with CVE and evidence-component resampling. Reddit's 1,649 accepted-
post CVEs improve in all seeds (RMSE difference -0.006271), while its 12,899
post-absent CVEs worsen on average (+0.000950). The subgroup result is exploratory,
not an overall victory or a GNN result. Rare-high-score error remains substantial.
Older fold 3 has only two high-score CVEs; all folds and seeds remain included.
Original results are unchanged. A separate hash-verified saved-result review
recalculates all 104 metric rows per dataset and preserves CVE error contributions.

Results: [full interpretation and read-only commands](05_studies/07_combined_text_fresh/08_TRAINING_ONLY_POOLED_RESULTS.md).
Review CLI: `03_analysis/07_fresh_studies/11_review_training_only_pooled.py`.
Recommended next gate is a saved-Reddit-model source-use diagnostic, not new training.

Subsequent review: external D-plus-flag checkpoints replayed and a target-date
version audit confirmed that chronological folds mix EPSS versions. CT06 above
supersedes that next-step recommendation with an outcome-blind within-version
plan and support audit. CT05's completed results and interpretation stay frozen.

Guide: [findings, role counts and CPU commands](05_studies/07_combined_text_fresh/07_TRAINING_ONLY_POOLED_COMPARISON.md).
Runner: `03_analysis/07_fresh_studies/10_run_training_only_pooled.py`.
Results: `05_results/08_fresh_studies/11_training_only_pooled_mse_v1/`.

## CT04: Training-Only Fold Feasibility

Implemented and reproduced for older and Reddit training cohorts only. Three
expanding date-based folds use Day-60 score dates and fixed 40/60/80% positions,
with a chronological-only reference and normalized exact-text component purge.
No official roles change; no training, test adaptation or scoring. Thread
settings are enforced inside the new CLI and recorded in the identity.

Potential OOF assessment coverage is 1,531 older CVEs (87 high-score, not all
132 training positives) and 14,548 Reddit CVEs (413 high-score). Older fold 3
has only two high-score assessment CVEs and fails the frozen tail-support
threshold. All Reddit folds pass marginal D/M comparison count gates, but joint
high-score/post support remains small in places; no automatic training approval.
Existing outer-training structured vectors cannot be reused as fold-fitted
preprocessing. Standalone D must use D context, not D tokens from DM embeddings.

Guide: [rules, findings and CPU commands](05_studies/07_combined_text_fresh/06_TRAINING_ONLY_FOLD_PLAN.md).
Runner: `03_analysis/07_fresh_studies/08_audit_training_folds.py`.
Results: `05_results/08_fresh_studies/10_training_fold_audit_v1/`.

## CT03: Parameter-Matched Pooled Source Readout

Implemented as a separate package/config/CLI. Older dataset only: 2,546 train
and 1,104 validation CVEs, test disabled. Twelve neural fits compare merged
versus source readout, MAE versus MSE, and seeds 17/29/43. Both arms have
28,705 parameters and use observed TOKEN embeddings from the same combined
DM graph. Primary endpoint is source-minus-merged validation RMSE under MSE;
MAE and high-score ranking remain secondary/exploratory. Twelve fixed controls
include two constants, five ridge regressions and five logistic classifiers.
Logistic outputs P(EPSS >= 0.1), never an EPSS score or exploitation label.

The 36 supplied external pilot checkpoints reproduced saved predictions;
six logistic metric sets matched saved scores, but logistic estimator states
were not supplied. Capacity/context are not ruled out universally: equal-count
architectures still differ, and combined/separate MSE errors are not identical.
This is repeatedly inspected development evidence, not independent confirmation.

Official smoke vector preparation, twelve fits, best-checkpoint reloads and
fresh-process report replay passed. Full vector extraction and all twelve full
neural fits plus twelve controls are now complete; full checkpoint and report
replay passed with BLAS threads fixed to four. The primary MSE source-minus-merged
RMSE contrast is -0.004882 [ -0.012340, +0.001730 ], so superiority is not established.
Both MSE arms beat the training-mean reference under the saved conditional
CVE bootstrap, but the supplemental precision check makes the merged arm's
nominal exclusion of zero unstable: its 20,000-resample interval includes zero.
Only the source-versus-mean interval stays below zero across the declared 200
bootstrap seeds. The source-versus-merged primary interval still includes zero.
Original intervals remain unchanged. Secondary MAE/ranking gains remain exploratory. Reddit has
9 high-score validation CVEs and 8.58%/99.59% train/validation accepted-prose
coverage; do not re-split after outcomes to obtain favorable significance.
Existing graph caches and completed CT01/CT02 results remain unchanged.
No GNN training or GNN-loader optimization is included in this step.

Verified full review: [results and methodological qualifications](05_studies/07_combined_text_fresh/05_SOURCE_READOUT_RESULTS.md).
Review CLI: `03_analysis/07_fresh_studies/07_review_source_readout.py --verify-only`.

Guide: [CT03 design and CPU commands](05_studies/07_combined_text_fresh/04_SOURCE_READOUT_COMPARISON.md).
Runner: `03_analysis/07_fresh_studies/05_run_source_readout.py`.
Results: `05_results/08_fresh_studies/09_source_readout_v1/`.

## CT02: Post-Hoc Ranking and Fusion Review

Saved-prediction analysis only; original 39 fits and test policy unchanged.
D-pooled's four largest contributors supply 143.94% of its net MAE gain,
including 67.10% from CVE-2025-0108. Removing those four post-hoc makes its
MAE 0.000520 worse than the median; the official cohort retains them.
M-TPG has the highest neural AUROC for EPSS >= 0.1 (0.8322), but DM-TPG has
higher AP (0.2072 versus 0.1735) and top-50 hits (12.33 versus 9.33).
These are exploratory high-score ranking metrics, not exploitation prediction.

The supplied fusion probe's input arrays and aggregate arithmetic were
verified; no new probe training was executed. Missing saved fusion checkpoints
and predictions prevent replay of its reported training outcomes. Standalone
source context, parameter count, initialization order and batching all differ
from the official run. Its logistic metrics are not present in the supplied
JSON and remain externally reported. A 23-graph outcome-blind audit supports
reusing document/source metadata, with a full mapping gate still required.

Recommendation: a bounded, parameter-controlled pooled source-readout test
before costly MSE TPG fits, plus parity-qualified graph loading optimization.
No new training or model implementation is included in this review.
Guide: [Completed results and addendum](05_studies/07_combined_text_fresh/03_MAE_FULL_RESULTS.md).
Script: `03_analysis/07_fresh_studies/04_review_ranking_and_fusion.py`.
Outputs: sibling `ranking_fusion_review_v1` beside `mae_fast_v1/full`.

## CT02: Older-Dataset Full MAE Results

The user's `mae_fast_v1/full` development run is complete: 39 neural fits
(22 referenced CT01 fits and 17 new CT02 fits) plus three controls, on
2,546 training and 1,104 validation CVEs. Test scoring remains disabled.
This completion supersedes the implementation-stage status below.

All six source variants were evaluated. D-pooled has the lowest seed-mean
MAE (0.022424) and RMSE (0.104874). The training-median MAE is 0.023604;
D-pooled's nominal 5.00% reduction has a paired MAE interval including zero.
TPG versus pooled MAE favors U under its unadjusted interval, disfavors DU,
and includes zero in D/M/DM/DMU. Adding M or U did not improve seed-mean MAE
over D within either representation family. M-TPG beats the median by only
0.81% under its unadjusted interval. No general TPG superiority is established.

High-score underprediction is material: D-pooled MAE is 0.422857 on the same
50 CVEs with actual EPSS >= 0.1. Those CVEs contribute 85.40% of its summed
absolute error. All findings remain development-only and retrospective.
No new inference, fitting, source-policy change or test scoring was done in
the review. Original hashes, best-epoch metadata, saved prediction metrics
and all report intervals reproduced; a fresh process reproduced the review.

Results guide: [Completed MAE comparison](05_studies/07_combined_text_fresh/03_MAE_FULL_RESULTS.md).
Review script: `03_analysis/07_fresh_studies/03_review_combined_training.py`.
TXT/JSON/CSV/figure artifacts: the sibling `full_review_v1` directory beside
the completed `mae_fast_v1/full` run. The original run is unchanged.
The next step is parity-tested TPG loading improvement before the separately
specified MSE study, not deletion, retuning or promotion of a single seed.

## CT02: Versioned CPU Execution and Objective Comparison

Implemented 2026-11-12 without modifying CT01 code, caches or checkpoints.
The original full `baseline_v2` process was left running. No cleanup or full
CT02 training was performed. Approximately 307 GiB was free; the active older
full run's checkpoints occupied only 4.34 MiB and remain useful references.

CT02 precomputes exact frozen TOKEN means for pooled controls, retains the
unchanged typed-edge GNN, increases the default per-worker LRU to 8 GiB and
uses independently locked process fits (default two jobs, four threads each).
It can reuse completed CT01 MAE fits through immutable, hash-verified references
after exact checkpoint/prediction parity. Partial fits are not imported.
One best checkpoint per new fit is retained. A separate `mse_fast_v1` uses
raw-scale MSE and validation-RMSE selection; it never imports MAE checkpoints.

Verification: 35 tests passed. The 78 MAE smoke fits across both datasets
reproduced all original training histories and prediction CSV bytes. New
parallel fits also passed sequential fresh-process checkpoint replay. Full
vector preparation and full-D reference parity remain user-run gates before
the older full MAE handover. The original shuffle schedule is retained for
parity; no architecture, source-policy or test-access changes were made.
The separate 78 MSE smoke fits also completed and passed fresh-process replay;
their two-epoch scores are not full-study findings.

Runner: `03_analysis/07_fresh_studies/02_run_combined_training.py`.
Config: `04_configs/07_fresh_studies/02_combined_training_v2.json`.
Outputs: `05_results/08_fresh_studies/08_combined_training_v2/`.
Guide and individual CPU commands:
[Optimized execution](05_studies/07_combined_text_fresh/02_OPTIMIZED_EXECUTION.md).
All paired development intervals remain exploratory and unadjusted, not a
multiplicity-corrected claim of predictive superiority.

## CT01: Fresh Combined-Text Training Adapter

Gate correction, 2026-11-11: the initial `baseline_v1` full attempt failed
before training because smoke/full encoder-generation thread counts were 4/8.
All other encoder and feature-contract fields matched. Only that generation
thread count is now exempted in cross-scale qualification; training settings,
weights, vocabularies, exact per-cache identities and checkpoint identities
remain strict. `--stage ready --scale full` checks the actual full-start gate
without fitting. Corrected runs use `--run-name baseline_v2`, preserving all
`baseline_v1` artifacts and their recorded hashes. No cache rebuild is needed.
The correction passed 25 tests, both new smoke replays, and the full readiness
gate for both datasets. All 78 smoke neural evaluations equal the original
evaluations, and all 514 original run files retained their SHA-256 hashes.
No full training was launched during this repair.

Current runner: `03_analysis/07_fresh_studies/01_run_combined_training.py`.
Configuration: `04_configs/07_fresh_studies/01_combined_training_v1.json`.
Guide: [Combined-text training](05_studies/07_combined_text_fresh/README.md).

This reads the corrected v3 caches with frozen SecureBERT 2.0 embeddings,
without changing the graphs, embeddings, role locks or source policies.
The typed message-passing GNN uses the exact saved 13-node-type, 23-edge-type
and 32-subtype vocabularies and all 88 edge properties. It concatenates the
graph representation with the existing 53-value structured representation and
source masks. Pooled-text and structured-only controls share the same cohorts.
The target is retrospective Day-60 EPSS, not VulnCheck classification.

Status on 2026-11-11: both six-CVE smoke runs completed across all six text
variants and seeds 17, 29 and 43. Each has 39 neural fits plus three controls.
All 78 neural checkpoints reproduced their validation predictions in a fresh
process. Exactly one best checkpoint per fit is present; there are no separate
last-epoch or per-epoch model files. Full-data preflights passed for all 32,578
Reddit and 19,698 older cache receipts, with payload probes and mandatory
per-load checks. This is not another complete graph-generation replay.
Full training and test scoring have not been started.

Full cohorts remain 2,546/1,104 for the older collection and 24,213/972 for
Reddit. Older full variants: D/M/U/DM/DU/DMU; Reddit full variants: D/M/DM,
with U/DU/DMU deferred because validation has no accepted webpage summaries.
Defaults are two GNN layers, hidden width 32, MAE loss, AdamW, 100 epochs and
patience 15. CLI overrides require a separate named run and matching smoke.
The best checkpoint is selected by minimum validation MAE, keeping the
earliest epoch on ties. Completed runs and input identities are hash-locked.

Artifacts: `05_results/08_fresh_studies/07_combined_training_v1/` under EPSS_TPG,
organized by dataset, encoder, run, scale, view, representation and seed.
The implementation does not restore deleted historical checkpoints or reset
prior test exposure.

## Fresh Start: Combined Text + SecureBERT 2.0

This supersedes active-run instructions below. The user requested removal of
old checkpoints/results/graph caches for earlier and Reddit datasets. Raw data,
documentation, dataset analysis and role locks remain. Removal receipts:
`00_documentation/07_maintenance/fresh_start_v1/`.

New code: `01_tpg/10_extensions/tpg_combined_v1/`.
Cache: `01_tpg/05_workspace/01_reddit/01_combined_securebert2_v1/`.
Verified inputs: `04_data/03_reddit_collection/04_combined_text_inputs_v1/`.
Views D/M/U/DM/DU/DMU concatenate text before extraction. Encoder:
`cisco-ai/SecureBERT2.0-base`, revision `7f7c16d1b2316c5046759667ed97f527aa1b7709`.
The 24,213/972 roles remain fixed; test is not adapted. The full plan has
32,578 distinct graph objects. Smoke: 6 CVEs, 14 graphs, exact fresh-process
graph/embedding replay on the largest example. Full expansion is user-run.
New training is not implemented by this cache stage. Reddit summary ablation remains
blocked by zero accepted validation summaries.

Both-dataset extension: `01_tpg/10_extensions/tpg_combined_datasets_v1/` and
`01_tpg/02_examples/01_scripts/build_dataset_tpgs.py --dataset both` now wrap
the unchanged shared graph/embedding engine. The older repaired 5,968-CVE
multisource collection retains 2,546 train / 1,104 validation CVEs, with 651
previously evaluated test IDs excluded from the adapter. The other 1,667
records are outside the preserved primary roles. No roles are changed.
Older full plan: 19,698 distinct objects, separate from Reddit's 32,578.
Both use D/M/U/DM/DU/DMU, v6, 32 subtypes, pinned SecureBERT 2.0 and the same
53 structured columns with dataset-specific training-fitted preprocessing.
Four older post documents across two training CVEs are additionally quarantined
for EPSS disclosure; no CVE is dropped. Accepted older summaries cover 885
validation CVEs. The 808 shared training CVE IDs mean these are not independent
replications. Verified cohort accounting and coverage:
`05_results/08_fresh_studies/00_dataset_coverage_v1/REPORT.txt` and `findings.json`.

Execution-only many-core extension: `tpg_combined_parallel_v1/` and
`build_dataset_tpgs_parallel.py` preserve the frozen extraction/receipt contract
while supporting more than the old 16-worker limit. A 32-process, one-thread
configuration passed preflight on the 44-physical-core / 88-logical-CPU host.
29 regression tests and two real-example compressed-byte parity checks passed.
Status and validation records are under
`06_runtime/02_fresh_studies/03_parallel_validation_v1/` (TXT and JSON).
Completed production graphs were verified, not regenerated. No full-workload
speedup is claimed before measurement, and the existing user-run job was not
stopped automatically. The new runner logs per-graph timings and active long
jobs separately from immutable cache artifacts. This is not a model experiment.

Guide: `00_documentation/02_tpg/11_COMBINED_TEXT_FRESH_START.md`.
Below are historical findings; old output links no longer provide replayable
artifacts. Deletion does not reset test exposure.

## RD02 Summary Extension: Preparation, Not Training

The full RD02 reference study has completed: 21 predictors, 24,213 train /
972 validation, test closed. The saved review is
`05_results/07_reddit_studies/02_rd02_primary_v1/full_review_v1` under EPSS_TPG.
The historical smoke-only entries below remain records of earlier stages.

The user requested GitHub and non-GitHub webpage summaries as model inputs.
`03_analysis/06_reddit_dataset/prepare_rd02_summary_extension.py` implements
an isolated predictor overlay, reference/CVE matching, EPSS-disclosure screening,
source masks, a nine-file inventory and a development-only collection queue.
Output: `05_results/07_reddit_studies/04_rd02_summary_extension_v1/preparation`.
The 808 matching older CVEs are all training records: 177 G / 538 N summaries
before screen, 177 G / 533 N afterward. Validation G/N coverage is zero.
No source scores, split IDs, completed fits or test records were changed.

The proposed comparison includes D/DG/DN/DGN with and without M_prose,
both pooled text and v6 TPG, matched structured inputs and seeds 17/29/43.
Six added views x two representations x three seeds = 36 proposed new fits;
completed D/DM_prose fits are reference candidates subject to parity checks.
Training is not enabled. Collection/source review and a new cache/smoke gate
remain required. Read `03_analysis/06_reddit_dataset/SUMMARY_EXTENSION.txt`.

## RD02/RD04: Reddit Development Smoke Gates

RD02 full export: 24,213 training / 972 validation CVEs, separate Day-60
labels and 53 training-fitted structured columns. The 943/29 validation
description-overlap partition remains fixed. Training-only v6 graph gate:
32 CVEs / 71 chunks. Smoke cache: 12 train / 6 validation CVEs / 47 chunks.
Training mean, median and ridge plus pooled/v6 TPG D, DM_all and DM_prose
across seeds 17/29/43 give 21 smoke predictors. All checkpoints/predictions
replayed; same-weight masked-M wrapper parity is exact. Two-epoch smoke
metrics are not full-development performance estimates.

RD04's dedicated lazy adapter reproduced 24,548 / 1,106 CVEs and
96,289,074 / 3,692,438 eligible pairs. Existing three-seed smoke artifacts
replayed without retraining. Both tasks keep test adaptation/scoring disabled.

The user authorized smoke execution, but requested only bounded graph caching
until durable storage is provided. Full RD02 cache plan: 66,800 chunks / 53,052
documents; not built. No full-development model training has been enabled.
No original split IDs, raw datasets or completed historical fits were changed.

Commands and interpretation: `05_studies/06_reddit_expansion/README.md`.
Reproducible TXT/JSON/CSV report and hashes:
`05_results/07_reddit_studies/02_rd02_primary_v1/gate_review_v1` under EPSS_TPG.
CPU CLIs: `03_analysis/06_reddit_dataset/run_rd02_primary.py`,
`run_rd04_daily.py` and `review_reddit_smoke_gates.py`.

## CF02: Saved Historical-Text Evidence Gate

No temporal-text fit authorized or run. Exact CF01 cohort: 2,563 training /
1,104 validation. Nonempty D/G/N/M CVE counts are respectively
2,563/542/1,799/2,562 in training and 1,104/201/749/1,104 in validation,
before relevance filtering. Date-screened M is present for 1,765 training /
1,019 validation CVEs, but no saved source establishes historical availability
of its wording. All 5,731 selected raw post associations match the saved
provenance audit; no original post identities or explicit timezones accompany
the saved post objects. Later NVD updates do not prove later wording, and
NVD link-addition dates do not authenticate linked content or summaries.

CPU audit CLI: `03_analysis/05_continuous_forecasting/cf02/07_audit_text_availability.py`.
TXT/JSON/CSV output and immutable artifact lock:
`05_results/04_consolidated_studies/02_continuous_epss/cf02_text_eligibility_v1`.
Recommend consolidating the completed regression, classification/revision and
continuous-forecasting findings, not weakening the temporal evidence policy.
This gate leaves all original datasets, checkpoints, results and test cohorts
unchanged. It does not regenerate the master PDF or establish that every text
is actually future-leaking; eligibility is unproven under the strict policy.

## CF01: Stage 3 Frozen History Diagnostic

Completed with no training or test access. Three
full Stage 2 checkpoints, the same 1,104 validation CVEs and 3,067,429 cells
per seed. `flat_history` preserves masks/age; `last_obs_only` consistently erases
older observations but preserves prefix length and the last-score baseline.
One fixed full-validation primary residual mean per seed defines the explicitly
post-hoc `constant_shift_control`, never a newly fitted deployment baseline.
Canonical streamed CSV keys, fixed gzip headers and canonically sorted report
rows prevent the Stage 2 resume-order mismatch. Ten new plus nineteen existing
tests pass; full-checkpoint smoke, fresh replay and sorted-JSON report replay
pass. CLI: `03_analysis/05_continuous_forecasting/stage3/05_diagnose_history_reliance.py`.
Outputs: `05_results/04_consolidated_studies/02_continuous_epss/stage3_v1`.
All original Stage 2 artifacts remain fixed.

Flat-history mean primary prediction changes are 4.2767e-8 / 1.1949e-8 /
4.7991e-9 for seeds 17/29/43. Paired primary and complete-matrix MAE/RMSE
intervals for flat minus intact include zero. Last-only causes larger changes,
but worsens mean MAE while improving mean RMSE; it also changes availability
channels, so this is not evidence of beneficial past score fluctuations.
Fixed offsets reproduce 99.0225% of the primary MAE gain, 97.4827% of primary
RMSE gain and 88.1864% of complete-matrix RMSE gain. These are descriptive
ratios of mean gains, not causal fractions or deployable training-fit controls.
Do not promote a seed or tune another GRU from this repeatedly inspected
validation cohort. TXT/JSON interpretation: `02_continuous_epss/stage3_review_v1`.
Fresh-process full replay passed for all frozen checkpoints, intervened
predictions, compressed CSV bytes, canonical reports and paired intervals.
The findings TXT/JSON review reproduced too; no row-order exception was needed.
Full report replay from JSON-reloaded results also reproduced identical hashes,
separately exercising the resumed publishing path without model inference.

## CF01: Stage 2 Full Development

Explicitly approved with unchanged Stage 1 settings; full run completed.
The separate `03_analysis/05_continuous_forecasting/stage2/02_run_full_development.py`
imports the hash-locked settings without changing Stage 1 artifacts or gates.
Same 2,563 / 1,104 CVEs, all eligible cells, three seeds, 100-epoch cap and
patience 15. Epoch zero is exact persistence, reported but excluded from
checkpoint selection. Full histories expose clipping and signed error for all
predeclared slices; each full epoch has 321 optimizer updates.
Stage 2 smoke and fresh-process checkpoint/report replay passed, along with
19 loader and four runner tests. Output:
`05_results/04_consolidated_studies/02_continuous_epss/stage2_v1`.
No test adaptation or scoring. This is one fixed development comparison,
not a new hyperparameter search. The completed revisions remain unchanged.

Seeds 17/29/43 stopped at epochs 41/17/17 and selected 26/2/2. Selected
forecasts have zero clipping; epoch zero equals persistence. Mean primary
Day30->60 MAE is 0.010223740 versus persistence 0.010252373 (0.2793% lower).
Complete-matrix MAE is 0.006092635 versus 0.006077860 (0.2431% higher).
Seven-/fourteen-day lead MAE worsens; thirty-day lead MAE improves slightly.
RMSE improves slightly in every slice. Paired primary/all MAE intervals exclude
zero in opposite directions, conditional on validation-selected checkpoints.
The lead correction equals persistence on all validation cells. This is not
broad forecasting superiority and is not a TPG result. A separate immutable
TXT/JSON review is in `02_continuous_epss/stage2_review_v1`, reproducible with
`03_analysis/05_continuous_forecasting/reviews/03_review_stage2.py --verify-only`.

Fresh-process controls, all three checkpoint/prediction files, epoch-zero checks,
intervals and figures reproduced. Original final byte verification rejects only
`metrics.csv` row order after JSON reload during resume. All 25 keyed rows and
every value match exactly. The separate strict report-order TXT/JSON audit in
`02_continuous_epss/stage2_report_order_audit_v1` reproduced. Original frozen code
and reports were not rewritten; the original full verification CLI still reports
the formatting mismatch, not a model or numeric mismatch.

## CF01: Daily Forecasting Stage 1

The researcher closed TRV01--TRV05 and deferred Frozen-D residual DG as a
hypothesis, not another classification fit. The separate daily forecasting
adapter implements compact 91-day storage, lazy past-only windows, masks and
observation age, strict role/date boundaries, persistence, training-only weighted
lead corrections and the shared CS01 GRU. Full training and test adaptation are
blocked. See the [stage-one specification and commands](05_studies/02_continuous_forecasting/README.md).

Full preflight reproduces 2,563 / 1,104 eligible CVEs with 9,576,458 / 3,067,429
training/validation cells. The original Day-30 to Day-60 slice is 2,519 / 1,104.
Smoke uses 12 / 6 ID-selected CVEs, all seeds 17/29/43 and two epochs each.
The 13,775 validation cells match across every predictor. Nineteen new tests and
nineteen existing contracts pass; full preflight and smoke checkpoint/report
reproduction passed. All 1,000-resample CVE-paired intervals and figures replay.
Actual-data repeat fitting of the bounded smoke also exactly reproduced every
selected parameter and epoch-history value for seeds 17, 29 and 43; saved
artifacts remained unchanged.

Smoke is not useful-performance evidence: lead corrections equal zero, and all
selected neural raw predictions are negative and clip to zero. Clipped full-matrix
MAE is 0.000538839 versus persistence 0.000045982. Raw errors and clipping are
retained; no tuning or full run follows automatically. Results are under
`05_results/04_consolidated_studies/02_continuous_epss/stage1_v1`.

## TRV05: Matched V6 Execution

Completed and verified: nine fresh description-only v6 fits against nine frozen
v5 controls; same seeds, decay, cohort and selection. New CLI:
`03_analysis/04_tpg_revision/10_run_v6_comparison.py`. Results belong in
`05_results/06_tpg_revision/11_v6_comparison_v1`; the original audit stays fixed.
Diagnostics separate the 16 graph-repaired, 39 severity-restored and 1,049
unchanged validation CVEs. A frozen-v5 topology intervention preserves old
subtype encoding and weights. See the [execution guide](05_studies/05_tpg_revision/06_V6_MATCHED_EXECUTION.md).
Smoke cache, initialization parity, training and diagnostics have reproduced.
The full cache and all nine initialization/reference gates passed. All nine full
fits completed and their checkpoints/predictions reproduced. Mean v5-to-v6 AP:
node mean 0.296638 to 0.291011; subtype node mean 0.286329 to 0.297985;
subtype type-balanced 0.281423 to 0.292690. All full-cohort AP intervals include
zero. Both subtype arms worsen mean Brier with positive paired error intervals.
Frozen graph repair leaves all 1,088 unrepaired validation probabilities exactly
unchanged and changes no threshold decisions. The changed subgroups each have
only one positive. Full diagnostic reproduction passed, including all tables,
bootstrap intervals and seven figure hashes; no test scoring or automatic
promotion. Full interpretation and commands are in the
execution guide above.

## TRV05: V6 Frontend and Subtype Diff Audit

Full audit of the unchanged 2,870 training / 1,104 validation descriptions:
63 version-shaped predicates, 106 exclusively owned arguments and 449 edges
removed across 44 CVEs. All 91 existing SEVERITY entities across 85 CVEs gain
an explicit channel instead of UNKNOWN. Tokens, embeddings, surviving node
and edge properties, old extractors, v5 caches and completed results are
unchanged. Exact per-CVE diffs and a nine-fit specification are locked in
`05_results/06_tpg_revision/10_v6_diff_audit_v1`.
The full audit reproduced in a fresh process and 32 focused regression tests
passed. The initial CSV publication is preserved separately; its line-ending
verification defect was repaired without changing any numerical or graph finding.

Planned arms: `v6_node_mean`, `v6_subtypes_node_mean`,
`v6_subtypes_type_balanced`; seeds 17/29/43; wd 0.0001; original settings and
AP selection; nine frozen v5 references. This stage runs no dataset training
or test scoring. The immutable v6 tensor cache and matched runner follow audit
review. See the [guide](05_studies/05_tpg_revision/05_V6_FRONTEND_AND_SCHEMA.md)
and CLI `03_analysis/04_tpg_revision/09_audit_v6_schema.py --verify-only`.

## TRV04: D Versus DG Execution, Phases 2/3

The researcher accepted Phase 1 and the unchanged-mask six-fit comparison is
complete. All six checkpoints and D references reproduced; exact masked-wrapper
parity passed. The adapter, matched runner and frozen-model diagnostic are
implemented; see the execution
status in the [TRV04 guide](05_studies/05_tpg_revision/04_D_VS_DG.md).
The six frozen D references, all TRV03 fits, extractor v5 and source policy are
unchanged. Subtypes and v6 fixes are excluded. Ten fixed label-blind G
derangements complement G-masked inference, with full/G-present/G-absent
metrics and paired CVE intervals. Availability bias is a hypothesis, not an
established mechanism. Entry: `03_analysis/04_tpg_revision/07_run_d_vs_dg.py`.
New output root: `05_results/06_tpg_revision/09_d_vs_dg_v1/`.

Overall AP: pooled D 0.301493 -> DG 0.272080; graph D 0.296638 -> DG 0.279375.
For the 201 G-present CVEs, graph AP rises 0.436579 -> 0.485612 in all three
seeds, but the paired AP-change interval [-0.035277, 0.120322] includes zero.
Graph G-present recall falls 0.450980 -> 0.313725 against D. Within frozen DG,
G-present graph AP is 0.485612 intact, 0.472999 masked, and 0.406294 averaged
over the ten fixed shuffles. Content matching matters descriptively, but does
not establish an overall improvement or isolate an availability-bias cause.
Do not promote DG or change filtering from these results. Full per-seed metrics,
confusion counts, predictions, donor maps and figures are retained in
`09_d_vs_dg_v1/{training,diagnostics,review}/full/`. Review CLI:
`03_analysis/04_tpg_revision/08_review_d_vs_dg.py`. No new test scoring.

## TRV04: D Versus DG Readiness, Phase 1

Phase 1 completed without graphs, embeddings, training or test scoring. Frozen
VK02/VK07 exports contain one accepted G document for 586/2,870 training CVEs
(55 listed, 531 unlisted) and 201/1,104 validation CVEs (17 listed, 184 unlisted).
G-absent validation is 903 CVEs (57 listed, 846 unlisted). Original masks and
policy remain unchanged. Availability is class-associated in opposite directions
across the two splits; it is not evidence that the text predicts listing.

Sixteen training-only, label-blind selected summaries were reviewed against saved
descriptions/URLs: 9 aligned, 4 mixed/unresolved, 1 off-target, 2 insufficient.
This automated review is risk-enriched, not independent annotation or a population
error estimate. Saved records lack the original scraped page bodies, so causal
attribution to scraping versus generation is limited. Unicode CVE IDs, legitimate
other-CVE context, repository/PDF metadata and multi-CVE summaries are covered.

The exact specification is six new DG fits (pooled text and v5 node mean, seeds
17/29/43, original decay 0.0001), six reused D references, identical structured
features and full-validation AP selection. Report paired DG-D metrics separately
on all validation, G-present and G-absent CVEs. Same-weight wrapper parity is
required; G-absent differences can otherwise also reflect changed learned weights.
No new training runner is launched by the audit; v6 fixes stay separate.

Read [the guide and protocol](05_studies/05_tpg_revision/04_D_VS_DG.md),
[the report](../05_results/06_tpg_revision/08_dg_readiness_v1/REPORT.md) and
[the machine-readable specification](../05_results/06_tpg_revision/08_dg_readiness_v1/trv04_runner_spec.json).
Entry point: `03_analysis/04_tpg_revision/06_audit_github_relevance.py`.
Original evidence/review: `01_tpg/09_annotation/10_dg_relevance_v1/`.

## TRV03: Global Weight Decay With Subtype Controls

Full execution and independent checkpoint replay completed at the researcher's
request. Nine locked 0.0001 controls are reused; 18 new fits compare 0.001 and
0.01 across v5_node_mean, v5_subtypes_node_mean and
v5_subtypes_type_balanced, each with seeds 17/29/43. Same 2,870 training and
1,104 validation CVEs; original AP selection, weighted BCE, graph cache and
fixed 0.5 reporting cutoff. No test access or extraction/schema changes.

All 38 revision tests pass. Selected-checkpoint norm and logit diagnostics are
reported alongside all seed metrics. Global decay cannot isolate subtype-only
causality, and a Frobenius norm near 5.9 does not by itself prove explosion.
See [the frozen comparison guide](05_studies/05_tpg_revision/03_DETERMINISTIC_REGULARIZATION.md).
Runner: `03_analysis/04_tpg_revision/04_run_regularization.py`.
Outputs: `05_results/06_tpg_revision/06_v5_regularization_v1/full`.

No uniform regularization improvement: all six stronger-decay conditions worsen
mean Brier versus their own control. Subtype node mean at 0.001 changes mean
AP 0.286329 -> 0.293537, log loss 0.514188 -> 0.444369 and norm 4.9023 -> 4.5501,
but AP improves only for seed 43 and Brier worsens in all seeds. Original node
mean at 0.01 reaches AP 0.305847, with AP improving only in seed 43 and log loss
worsening in all seeds. Subtype balanced at 0.01 improves AP in all three seeds
(mean 0.288057), but mean F1, Brier and log loss worsen. No variant is promoted.
Thirteen new checkpoints were preserved through a diagnostic-only sigmoid
batch-shape repair; five removed temporary outputs were deterministically
restarted, with identical recorded epoch results. Original controls and graphs
remain unchanged. See [all results and the paired figure](../05_results/06_tpg_revision/07_v5_regularization_review_v1/REPORT.md).

## TRV02: Bounded v5 Corrections and Explicit Subtypes

The fixed 16-training-CVE audit is complete as an automated source-grounded
review, not independent annotation. It records 23 primary correction events,
eight dependent argument annotations and four dependent relation assertions.
Nine CVEs change: 30 nodes and 87 edges removed, one hardware version retyped.
Source text and token nodes remain. Missed product boundaries, complex release
ranges and unresolved contextual scope are documented rather than guessed.

The opt-in v5 extension leaves core/v4 code and completed results unchanged.
A new cache verifies source graphs/tensors and reuses surviving embeddings
exactly. `v5_subtypes` adds explicit entity categories with missing/unknown states.
Four variants separate corrections, subtype inputs and original/type-balanced
readout. Full mode is 12 fits on unchanged 2,870 train / 1,104 validation CVEs,
seeds 17/29/43, with locked v4 references. Weight decay, loss, AP checkpoint
selection and 0.5 reporting cutoff remain fixed. No test access is implemented.

Status: audit reproduced; 24-training/4-validation corrected cache reproduced;
four model preflights passed. The researcher completed four smoke fits. Their
final report initially failed because precision was undefined (zero predicted
positives at 0.5). The reporting-only repair preserves nulls, records defined
seed counts, verifies the exact predecessor identity and reuses all four fits.
All 21 original smoke files remain byte-identical; no recovery retraining or
test scoring. There are now 32 passing revision tests, including null-metric and
strict recovery regressions. No full-development fit was launched during that repair. The proposed later
weight-decay control is not enabled in this graph/subtype comparison.
Full cache update: all 3,974 description graphs reproduced and the four-variant
full preflight passed. Corrections affect 1,500 CVEs (1,089 train / 411 validation),
remove 2,480 nodes and 7,350 edges, and retype five hardware versions. Embeddings
are reused exactly. There are 91 raw SEVERITY nodes mapped to UNKNOWN across 85
CVEs; residual version-shaped predicates also remain. These are documented
limitations, not changes to the frozen schema. No full training or test scoring
was performed during review. Findings: `05_results/06_tpg_revision/04_v5_cache_review_v1/`.
Full findings, exact taxonomy, limitations, artifacts and user-run CLI commands:
[TRV02 guide](05_studies/05_tpg_revision/02_V5_SEMANTIC_AUDIT_AND_SUBTYPES.md).
Audit: `01_tpg/09_annotation/09_v5_semantic_audit_v1/` (TXT/JSON and graph copies).
Runner: `03_analysis/04_tpg_revision/02_run_v5_subtypes.py`.
Completed fits: `05_results/06_tpg_revision/03_v5_subtypes_comparison_v1/`.

**Full training completed and verified:** all twelve v5 checkpoints and validation
predictions reproduced. A separate review also recomputed metrics for all v5,
v4/readout, pooled-description and logistic references on the same validation IDs.
No retraining, test access, threshold tuning or original artifact changes.

| Three-seed mean | AP | ROC-AUC | F1 at 0.5 | Brier | Log loss |
|---|---:|---:|---:|---:|---:|
| v4 original mean | 0.299201 | 0.801006 | 0.296945 | 0.082034 | 0.360052 |
| v4 type-balanced | 0.283586 | 0.782389 | 0.287666 | 0.104663 | 0.484936 |
| v5 original mean | 0.296638 | 0.792465 | 0.272766 | 0.078711 | 0.362941 |
| v5 type-balanced | 0.282558 | 0.790842 | 0.322898 | 0.103602 | 0.450278 |
| v5 subtypes + mean | 0.286329 | 0.784545 | 0.276958 | 0.070785 | 0.514188 |
| v5 subtypes + balanced | 0.281423 | 0.792443 | 0.303865 | 0.077889 | 0.362600 |

No new graph variant exceeds original v4 mean AP or pooled-description mean AP
0.301493. v5 type-balanced has the highest new mean F1 (0.322898), below combined
balanced logistic F1 0.331551. Subtypes reduce AP in all three original-pooling
seeds. With balanced pooling, seed-29 FP falls 155 -> 25 but TP falls 40 -> 16.
The extraction correction alone changes that seed's v4-balanced FP only 158 ->
155. The proposed explanation of seed instability is therefore not established.
All six enabled subtype branches have nonzero gradients and saved weights.
All twelve final training losses fall below their selected-epoch losses; Brier
worsens in 7/12 and ROC-AUC in 8/12 relative to selected checkpoints. These are
descriptive development comparisons, not significance or causal claims.

The complete overhaul explanation, architecture, every seed, paired differences,
real mistakes, curves and limitations are in
[the full v5 review](../05_results/06_tpg_revision/05_v5_training_review_v1/REPORT.md).
TXT/JSON/CSV and figures are preserved beside it. Reproduce with
`03_analysis/04_tpg_revision/03_review_v5_results.py --verify-only`.
The original narrow decay proposal is superseded by the explicitly requested
three-family comparison in TRV03 above. Completed variants and the original
final cohort remain frozen; no simultaneous extractor/threshold changes.

## TRV01: Separate TPG Readout Revision

New exploratory development path; completed studies remain frozen. See
[the revision guide](05_studies/05_tpg_revision/README.md).
Preservation archives the existing results/code and records cache manifests.
The first controlled change is type-balanced graph readout instead of all-node
mean pooling, with identical nodes, edges, embeddings, parameters and training
settings. Training-only review packets retain pending semantic judgments.
The CLI is `03_analysis/04_tpg_revision/01_run_type_readout.py`.
Full mode adds three description-TPG + structured fits against reproduced VK04
references on 2,870 training / 1,104 validation CVEs. No new final-test scoring,
extractor changes or threshold tuning. Full training is user-run, not automatic.
Status: full fits completed; all three revised and original-reference checkpoints
and validation predictions reproduced, with preserved results unchanged.
Mean AP 0.299201 -> 0.283586; F1@0.5 0.296945 -> 0.287666;
ROC-AUC 0.801006 -> 0.782389. AP improves in two seeds but seed 29's larger
decline lowers the mean. ROC-AUC, Brier score and log loss worsen in all seeds.
Do not promote the changed readout. The follow-up source-grounded review and
bounded corrections are recorded separately under TRV02 above. No new test scoring.
Detailed TXT/JSON review: `05_results/06_tpg_revision/02_type_readout_review_v1/`.

This is the single current index of completed and planned experiment types.
Update this file as each new type is implemented. Detailed predictions, models
and reports stay in their existing result folders; this index does not replace
them. Older feasibility reports describe the proposal at the time they were
written, not the latest implementation status.

**Current scope:** repaired EPSS history and the SMP-dated revision dataset.
Raw source: `SummTPGVul/Sec4AI4Sec-EPSS/Data_Files/epss_recollected_2026-09-30_smp.json`.
There are 5,968 CVEs. Its SHA-256 is
`506ae7da7c9202ca132e4e7ff8511f1fbbea9ab534630374dde5370e0ecfd43c`.
The additional dates do not themselves verify original post publication times.

**VK10: Final evaluation completed and reproduced, without training or tuning.**
Read the [final findings](05_studies/04_vulncheck_snapshot/17_FINAL_RESULTS.md)
and [execution guide](05_studies/04_vulncheck_snapshot/16_FINAL_EVALUATION.md).
The exact 651 evaluation CVEs (36 listed, 615 not listed) were scored by all
36 frozen predictors. All-source TPG mean AP=0.260112, ROC-AUC=0.794429,
F1@0.5=0.239042, frozen-validation-cutoff F1=0.245084. Matched pooled text:
0.273424, 0.806323, 0.283226 and 0.312189 respectively. The validation advantage
did not transfer. Paired TPG-minus-pooled AP interval [-0.080184, 0.046758]
includes zero; the two declared F1 intervals favor pooled text conditionally.
No test-selected winner replaces the primary family. On 590 CVEs without
shared training wording, TPG/pooled mean AP is 0.135996/0.152936; differing
class proportions and evidence overlap prevent causal interpretation.
All 2,530 graph chunks were reused and structurally verified; no new embeddings.
113 tests passed; smoke reproduced all 36 saved probability vectors exactly.
Full predictions and reports were independently rerun with --verify-only.
Outputs: `05_results/05_vulncheck_study/19_final_evaluation_v1/full/`, including
23,436 prediction rows, 60 metric rows, all seeds, conditional paired intervals,
ten-bin reliability, overlap subsets, source coverage and 38 figures/PDF pages.
The evaluation cohort was exposed in prior EPSS research: this remains a
reused-cohort retrospective comparison, not a new independent holdout.
Original models, data, caches, VK09 freeze and completed regression results
remain unchanged. No fitting, threshold search, calibration or source repair.

**VK09: Earlier development report and final protocol freeze.**
Read the [final report and protocol guide](05_studies/04_vulncheck_snapshot/15_FINAL_REPORT_AND_EVALUATION_PROTOCOL.md).
Bundle: `05_results/05_vulncheck_study/18_final_protocol_v1/`, including full
MD/TXT/JSON report, all 36 validation records, exact role IDs, 35 checkpoint
hashes plus the prevalence reference, per-checkpoint thresholds, feature-state
and dependency hashes. Primary DGNM security TPG is selected by mean validation
AP; matched comparator DGNM pooled text; seeds 17/29/43 all retained, no ensemble.
Report 0.5 for every model and existing VK08 thresholds for the 24 source-neural
fits. No newly tuned cutoffs for logistic/edge-free references. Primary metric
AP, with all classification/calibration metrics and fixed paired bootstrap rules.
651 existing test-role IDs are planned for a reused-cohort retrospective
evaluation, not a fresh independent test. No cohort reassignment or source repair.
At this stage no final scoring had been performed. VK10 subsequently implemented
and executed the separate authorized evaluator; the VK09 CLI still only freezes
or verifies its historical bundle. All 99 tests at the VK09 stage
passed, including seven new freeze/selection/integrity contract checks.

**Completed and reproduced: VK08 frozen source diagnostic and validation-selected thresholds.**
Read the [full findings](05_studies/04_vulncheck_snapshot/14_THRESHOLD_AND_SOURCE_FINDINGS.md).
All-source TPG mean F1: fixed 0.5=0.458488, selected-validation=0.482348,
threshold cross-fold=0.457594. Best individual selected F1=0.496732, seed 43,
cutoff 0.6832602024, TP38/FP41/FN36. Pooled all-source means are
0.436028/0.448101/0.400895. Threshold selection is not an independent evaluation.
TPG mean AP under frozen interventions: original 0.414902, swap posts
0.121293, swap description 0.298322, zero structured 0.320621, swap summaries
0.396170. Post dependence persists without recorded shared wording (AP
0.318209 to 0.106192 under swapping). Summary/CVE mismatches and generic posts
are documented; removing summaries does not repair the highlighted false flags.
Separate TXT/JSON/CSV/figure review: `17_threshold_source_review_v1/` under
`05_results/05_vulncheck_study/`. 99,360 saved intervention predictions checked;
all six original probability vectors reproduce exactly. No retraining/test scoring.
Read the [protocol and CLI sequence](05_studies/04_vulncheck_snapshot/13_SOURCE_DIAGNOSTIC_AND_THRESHOLDS.md).
Original VK07 artifacts and fixed-0.5 metrics remain unchanged. For all 24 saved
fits, maximize positive-class F1 across distinct validation scores, breaking
exact F1 ties at the highest threshold. Save per-checkpoint thresholds and a
five-fold threshold-only sensitivity check; this does not undo validation reuse
or establish independent test performance. Diagnose six full-trained all-source
models with zero/remove/swap source interventions, structured-vector zeroing,
shared-wording subsets and confident-error excerpts. Thresholds stay fixed
through all interventions. No training, new pretrained embeddings, test scoring,
or automatic final model/seed selection. Full runs were completed by the user.
Outputs: `15_threshold_selection_v1/<scale>/` and
`16_multisource_diagnostic_v1/<scale>/` under `05_results/05_vulncheck_study/`.
Verification: 92 tests passed. Both smoke stages completed and independently
reproduced with `--verify-only`; six saved all-source
models reproduce probabilities exactly on 64 CVEs. Only two are listed, so
smoke F1 is not a performance estimate. Full analysis is now complete.

**Completed and verified: VK07 bounded multi-source ablation.**
Read the [full findings and limitations](05_studies/04_vulncheck_snapshot/12_MULTISOURCE_RESULTS.md).
All 18 new full fits and six frozen references reproduce; 26,496 validation
probabilities and all selected epochs/metrics were independently checked.
Mean AP: D pooled/TPG 0.301493/0.299201; DGN 0.275018/0.277592;
DM 0.347898/0.373139; DGNM 0.378135/0.414902. All-source TPG F1 is 0.458488
versus pooled 0.436028, at fixed 0.5. All-source AP improves over D in all
seeds for both representations, but TPG beats matched all-source pooled AP in
only two seeds. All-source TPG finds 35/34/40 of 74 listed validation CVEs.
Among 1,018 CVEs without recorded shared training wording (53 listed), mean
AP is 0.271302 for D TPG, 0.270307 for DM TPG, 0.275661 for all-source pooled
and 0.318209 for all-source TPG. These are descriptive sensitivity results,
not evidence of duplicate CVEs or a repaired independent split. All 18 new
runs finish below selected AP, with lower final training loss in 16 of them.
Separate TXT/JSON/CSV/figure review: `05_results/05_vulncheck_study/14_multisource_results_review_v1/`.
Read the [protocol and CLI commands](05_studies/04_vulncheck_snapshot/11_MULTISOURCE_ABLATION.md).
D / DGN / DM / DGNM, pooled text versus security TPG, structured branch always on;
unchanged 2,870 training / 1,104 validation CVEs and VK02 v2 accepted-source policy.
Six VK04 D checkpoints are immutable references; 18 new fits use seeds 17/29/43,
balanced BCE, AdamW, 100 epochs/patience 15, validation AP selection and fixed 0.5
decision threshold. Full training was launched by the user after implementation;
no new test scoring or threshold tuning occurred. Source coverage/filtering is reported by class.
Documents are encoded separately, pooled within source, then combined with masked
attention and structured late fusion. Exact reusable chunks use authenticated
external references rather than duplicate storage; missing chunks use the frozen
PS01 v2 builder. Results: `05_results/05_vulncheck_study/13_multisource_ablation_v1/`.
This does not newly train a relation extractor or establish historical source availability.
Verification: 80 tests passed; 200 smoke chunks checked; six new smoke fits and
two frozen D references reproduced independently. D probability parity is exact
(maximum difference 0.0). Undefined precision remains null with explicit defined/
undefined seed counts, not a fabricated zero. The initial reporting failure is
archived within the new result root; the corrected run retains identical graph
artifacts and per-fit predictions/metrics/histories. Full export reproduced with
12,808 accepted documents; completed full cache has 19,624 chunks, including
17,231 exact externally reused chunks and 2,393 newly built chunks.

**Completed and verified: VK06 matched edge-free training control.**
Read the [full results review](05_studies/04_vulncheck_snapshot/10_EDGE_FREE_RESULTS.md).
All three checkpoints/predictions reproduced. Mean AP: edge-free 0.287339 versus
TPG 0.299201 and pooled 0.301493; same-seed edge-free AP differences are +0.008276,
-0.046251, +0.002389. Original TPG ROC-AUC is higher in every seed. No consistent
AP win across seeds or advantage over pooled text is established. Mean F1 is
0.274904 edge-free versus 0.296945 TPG. Separate TXT/JSON/figure review:
`05_results/05_vulncheck_study/12_edge_free_results_review_v1/`.
Read the [frozen protocol and CLI commands](05_studies/04_vulncheck_snapshot/09_EDGE_FREE_CONTROL.md).
Three fresh seeds 17/29/43, 2,870 training/1,104 validation CVEs, unchanged VK04
settings. The original classifier and training/replay/inference functions are
reused; only the edge arrays are emptied after original graph packets form.
Every node, embedding, node type, layer, pooling operation and structured input
is retained. Preflight audits all nodes and edge removal, and saved VK04
graph/pooled/logistic references are preserved. Separate results:
`05_results/05_vulncheck_study/11_edge_free_control_v1/<scale>/`.
This is retraining without edges, distinct from VK05 frozen-model perturbation.
User-run full training is complete; no test scoring or threshold search.
Verification passed: 68 VulnCheck tests, real-data smoke training and independent
checkpoint/prediction reproduction, plus full 3,974-graph preflight. Subsequent
full fits preserve 425,784 nodes and remove all 1,308,324 development edges.

**Implemented: VK05 frozen branch/edge sensitivity and error inspection.**
Full v2 [results review](05_studies/04_vulncheck_snapshot/08_DIAGNOSTIC_RESULTS.md):
all 84 condition reports and affected-subset metrics reproduce from 92,736 saved
predictions. TPG mean AP: baseline 0.299201, zero description 0.173686, zero
structured 0.131059; both branches matter in all seeds. Security-edge removal
affects 304 CVEs/15 listed but changes zero decisions; all-edge removal gives
AP changes +0.014037/-0.013302/+0.005052. Connectivity has no consistent observed
ranking advantage. Separate TXT/JSON/figure review:
`05_results/05_vulncheck_study/10_graph_diagnostic_review_v1/`.
Proposed next controlled fit: identical node/encoder setup with edges disabled,
not an expanded model/source sweep. No new fits were launched by this review.
The CLI now defaults to v2, correcting sigmoid evaluation to occur per original
batch rather than after concatenation. V1 full failed for pooled seed 43 on a
1.1920928955078125e-7 score difference; original per-batch scoring reproduced it
exactly. The 1e-7 check remains unchanged. V1 smoke/code and all VK04 artifacts
are preserved. V2 outputs are separate under `09_graph_diagnostic_v2/<scale>`;
use `--version v1` only for archived v1 verification.
V2 full diagnostic completed and independently reproduced on 1,104 validation
CVEs; all six baseline score vectors match saved VK04 exactly (maximum difference
0.0). All 62 VulnCheck tests pass. No retraining or test scoring. Full v2 output
already exists; use `--version v2 --scale full --verify-only` to reproduce it.
Read the [diagnostic guide](05_studies/04_vulncheck_snapshot/07_FROZEN_GRAPH_DIAGNOSTIC.md).
Six full-trained VK04 classifiers; description/structured latent zeroing and
fixed label-independent swaps; eight edge-removal conditions from the saved
vocabulary, including an all-edge condition. Nodes/embeddings are unchanged.
Pooled edge-removal and unaffected-CVE predictions are checked as negative
controls. Saved baseline scores must reproduce before interpreting perturbations.
Balanced/unweighted combined logistic references are retained for bounded
confident-mistake comparisons. No tuning, fitting, parser/embedding rerun or
test scoring. Separate TXT/JSON/CSV outputs:
`05_results/05_vulncheck_study/09_graph_diagnostic_v2/<scale>/`.
Smoke uses 64 validation CVEs in hash-selected original batches and full-trained models; full uses
the existing 1,104 validation CVEs. This is frozen-model sensitivity, not an
isolated retraining ablation, causal analysis or independent test.
Verification: 59 tests passed; real-data smoke and independent reproduction
passed (64 CVEs, two listed, 5,376 condition predictions); full preflight passed.
These checks describe v1 readiness; the full v1 failure and versioned correction
are recorded above. The numerical fix does not itself imply a scientific result.

**Completed and verified: VK04 description TPG versus pooled-description classification.**
Read the [full results review](05_studies/04_vulncheck_snapshot/06_GRAPH_RESULTS.md).
All six full checkpoints and validation predictions reproduced. Mean AP is
0.301493 pooled versus 0.299201 TPG; AP differences by seed are -0.038507,
+0.062323 and -0.030692. Mean F1 at 0.5 is 0.274737 versus 0.296945; the balanced
combined logistic reference remains higher at 0.331551. No consistent graph
ranking gain or independent-test advantage is established. Separate TXT/JSON
review and ranking/training figure are in
`05_results/05_vulncheck_study/08_description_graph_review_v1/`.
Read the [label/threshold explanation and run guide](05_studies/04_vulncheck_snapshot/05_THRESHOLDS_AND_GRAPH_COMPARISON.md).
Two encoders use the same accepted descriptions, frozen SecBERT node embeddings,
29 structured columns, balanced BCE and seeds 17/29/43. The edge-type-aware GNN
uses two layers and no COREF; the pooled control uses embedded tokens without
edges. Head design, optimizer/budget and validation-AP selection are matched;
node populations and active encoder capacities differ. Threshold stays 0.5.
Full preflight: 2,870 training / 1,104 validation CVEs, 3,974 description chunks,
3,650 exact reuse candidates and 324 missing chunks. All 51 tests pass. The
8/4 real-data smoke built eight chunks, reused four and trained/reloaded both
two-epoch paths. Independent smoke cache and checkpoint/report verification passed;
its ten SEC_HAS_VERSION edges
are not a semantic-accuracy assessment. Full cache/training subsequently completed:
3,650 chunks reused, 324 built; 1,941 security edges, including 1,900 version links.
Cache: `01_tpg/05_workspace/vulncheck_description_v1/<scale>/`; results:
`05_results/05_vulncheck_study/07_description_graph_v1/<scale>/`.
All completed logistic references remain unchanged. No test scoring or threshold
tuning. Old static-EPSS soft supervision/reporting and current binary VulnCheck
targets are explicitly distinguished in the guide; historical F1 is not directly
comparable without checking the adapter, labels, features and split.

**Completed: VK03 full-development baseline comparison and review.**
Read the [verified results](05_studies/04_vulncheck_snapshot/04_BASELINE_RESULTS.md).
All eight checkpoints and the prevalence reference reproduce on 1,104 validation
CVEs after training on all 2,870 training CVEs. Combined description + structured
AP is 0.249377 unweighted and 0.258694 balanced, versus prevalence 0.067029.
Balanced combined F1 is 0.331551, precision 0.274336 and recall 0.418919:
31 true positives, 82 false positives and 43 missed listed CVEs. Unweighted
combined has lower log loss/Brier; balancing improves recall but not probability
accuracy. Mask-only AP is 0.065879, so its tiny-smoke advantage did not persist.
Large train/validation AP gaps and prior data limitations remain. No model was
automatically selected and no threshold or test tuning occurred. Separate review
TXT/JSON and a corrected stepwise PR figure are in
`05_results/05_vulncheck_study/06_baseline_results_review_v1/`. The original PR
figure's constant-baseline diagonal is a plotting artifact, not its true AP;
all original artifacts remain unchanged. Next: a bounded matched pooled-text
versus description-TPG comparison with structured inputs, not more baseline tuning.

**Implementation record: VK03 full-development baseline preparation.**
Read the [full cohort, source audit and commands](05_studies/04_vulncheck_snapshot/03_FULL_DEVELOPMENT_BASELINES.md).
The full VK02 export reproduces: 2,870 training CVEs (376 listed), 1,104 validation
(74 listed). All descriptions remain; 452 of 13,260 documents are excluded.
Any-document exclusion affects 29.79% of listed versus 2.57% of not-listed
training CVEs, and 32.43% versus 2.23% in validation. Accepted shared wording
crosses roles in seven description and 57 post groups, involving 86 validation
CVEs; no CVEs were merged or reassigned. The frozen eight-fit protocol tests
structured, description TF-IDF, their concatenation and source masks, each
unweighted and training-balanced, plus a prevalence constant. Threshold 0.5,
seed 17, no search, no automatic winner selection, no test scoring. All 42 tests
pass, including synthetic fits/reloads and CSV round trips. Protocol reproduction
and full training preflight passed. Full real-data training has not been run
during preparation; it is ready without another dataset smoke. The separate
plan is in `05_results/05_vulncheck_study/04_baseline_plan_v1/`; the training runner
will publish `05_results/05_vulncheck_study/05_baseline_development_v1/`.
Independent annotation, label provenance and final evaluation remain limitations,
not silently approved conditions. No TPG/GNN or EPSS inputs enter these baselines.

**Completed: VK02 versioned source policy, separate adapter and classifier smoke.**
Read the [implementation and findings](05_studies/04_vulncheck_snapshot/02_POLICY_ADAPTER_AND_SMOKE.md).
Policy v2 adds a span-local function-name exception and a narrow explicit
wrong-CVE summary rule, retaining conservative catalogue/EPSS screening.
Review is engineering-only, not independent annotation. Predictor text, masks
and structured vectors are separate from labels and audit decisions. The smoke
uses 64 training / 32 validation CVEs (8 / 2 listed), train-only preprocessing,
four logistic controls and a training-prevalence reference. All four checkpoint
reloads and saved predictions reproduce; all 33 VK01/VK02 tests pass. Every model
has zero F1 at the fixed 0.5 threshold; these tiny-sample scores are integration
diagnostics, not a performance comparison. No model or threshold was selected.
Results are in `05_results/05_vulncheck_study/03_classifier_smoke_v2/`.
Full development export is available but not run; full training is disabled.
No test scoring, new graphs, embeddings or changes to original results occurred.

**Completed: VK01 full VulnCheck snapshot-classification readiness audit.**
Read the [new study guide](05_studies/04_vulncheck_snapshot/README.md).
The separate audit checks strict boolean labels, stored-date anomalies, current
source coverage, candidate catalogue disclosures and cross-role shared text.
It proposes bounded source/representation/structured comparisons without
training, graph generation, split changes or test scoring. EPSS is excluded
from the initial predictor plan; missing Day-60 scores do not invalidate a
VulnCheck label. Existing test exposure remains explicit. Full audit findings
must be reviewed before approving training or a new evaluation protocol.
Sixteen focused tests and the 24-CVE / 60-document smoke reproduction pass.
All-record labels are valid: 548 listed and 5,420 not listed. The full audit
subsequently reproduced across 20,605 documents. Candidate filtering flags
781 documents across 316 CVEs, retaining at least one source for every CVE.
Read the [full findings](05_studies/04_vulncheck_snapshot/01_READINESS_RESULTS.md):
direct label disclosures, a keyword/function-name ambiguity, a concrete summary
mismatch, label-associated filtering and remaining shared text need explicit
treatment. Accepted-text overlap still crosses old roles in 188 groups across
594 CVEs. The separate TXT/JSON interpretation is in
`05_results/05_vulncheck_study/02_readiness_review_v1/`. No models were trained
and original audit/review statuses were not changed.

**Completed: PS05 final held-out evaluation and interpretation.**
Read the [final findings](05_studies/03_shared_evaluation/02_FINAL_RESULTS.md).
All 651 reserved test CVEs are eligible; all 27 frozen fits were scored and
independently reproduced. The review verified data/cache and model-output
locks, provenance, metrics and paired comparisons without new inference.
Selected pooled DM test MAE is 0.01741328, 29.40% worse than the training-median
reference (0.01345742). Designated all-source TPG is 0.01707629, 26.89% worse.
No neural model's interval versus median is entirely below zero. SMP addition
increases test MAE in all four matched comparisons; high-score recall remains
weak. Nominal test-best DGN TPG is reported but not promoted to a new selected
winner. Shared development text occurs in 166 documents across 79 test CVEs.
Separate TXT/JSON analysis, actual/predicted examples and a comparison figure
are in `05_results/04_consolidated_studies/01_primary_day60/final_evaluation_review_v1/`.
The primary experiment and its [final research report](05_studies/03_shared_evaluation/03_FINAL_RESEARCH_REPORT.md)
are complete. The report contains an architecture diagram, matched source
comparisons, all 27 fit results and real prediction examples. Its builder
checks saved artifacts and prediction arithmetic without new inference.
Do not retune against the now-observed test set. No original results were changed.

**Implementation reference: PS05 final held-out evaluation.**
Read the [release and evaluation guide](05_studies/03_shared_evaluation/01_FROZEN_FINAL_TEST.md).
The protocol fixes all eight PS04 S1 configurations, seeds 17/29/43 and three
baselines (27 saved fits). The validation-selected leader is pooled DM; the
designated graph comparator is all-source TPG. There is no new training,
preprocessing fit, seed selection or ensemble. Every full data/cache/scoring
stage requires `--release-test`. Test roles remain explicitly test; existing
development-only guards are unchanged. All 78 consolidated tests pass. The
two-validation-CVE smoke reused nine verified development chunks and reproduced
the original full-trained predictions for all 27 fits. Independent freeze,
data, cache and inference/report verification passed. No reserved test records,
eligibility or numerical scores have been inspected during implementation.
The protocol is stored under
`05_results/04_consolidated_studies/01_primary_day60/final_evaluation_v1/frozen_protocol/`.
Final test scoring subsequently completed as reported above. Report all frozen comparisons,
including weak outcomes, without retuning after release.

**Completed: PS04 full bounded source comparison and review.**
Read the [full findings](05_studies/01_primary_tpg/10_SOURCE_COMPARISON_RESULTS.md).
All 27 reused/new fit artifact locks and matched report arithmetic reproduce.
The 12 new fits completed 454 epochs. Pooled description + SMP has the lowest
mean MAE, 0.02248772, versus 0.02360351 for the training-median baseline (4.73%
lower). Best TPG is all-source + structured, MAE 0.02296873. All 12 source-addition
and graph-versus-text paired intervals include zero. High-score recall for
pooled DM is only 4--10% at EPSS 0.1; 50 high-score CVEs contribute 85.60% of
its error. CVE-2025-0108 contributes 74.63% of net gain over median. Separate
TXT/JSON review and real examples are under
`05_results/04_consolidated_studies/01_primary_day60/source_comparison_review_v1/`.
Recommended next: close bounded development, freeze all eight S1 configurations,
three seeds and baselines, then implement one held-out evaluation. No test
scoring or new training occurred during review.

**Implementation reference: PS04 bounded source comparison.**
Read the [run guide](05_studies/01_primary_tpg/09_BOUNDED_SOURCE_COMPARISON.md).
The separate runner fixes D, DM, DGN and DGNM, pooled-text and security-TPG
representations, CVSS + reference count, and seeds 17/29/43. It reuses 12
completed D/DGNM neural fits and three original baselines read-only; only 12
DM/DGN fits are newly trained. All 2,546 training / 1,104 validation CVEs,
PS02 settings and source pooling are unchanged. Reports include matched source
and representation effects, per-seed metrics, conditional intervals and figures.
All 68 consolidated tests pass. The four-new-fit smoke and independent
checkpoint/report verification passed; these are integration results only.
The full-data preflight passed; the subsequently completed full run is reported
above. Results belong in
`05_results/04_consolidated_studies/01_primary_day60/source_comparison_v1/<scale>/`
under `EPSS_TPG`. Original code, trained fits and caches remain unchanged.
No final-test access is enabled.

**Completed: PS03 full saved-model diagnostic and interpretation.**
Read the [full findings](05_studies/01_primary_tpg/08_DIAGNOSTIC_RESULTS.md).
All 18 frozen checkpoints and 1,104 validation CVEs passed diagnostic
reproduction. The separate review verified 62 artifact hashes and recomputed
saved response statistics. Security-edge removal has negligible average
effect; source dependence varies strongly by seed. In all-source TPG +
structured models, SMP receives 99.304% and 90.514% mean attention at seeds
17/29, whereas description receives 96.010% at seed 43. Text permutation and
structured removals worsen MAE in every seed, but do not establish TPG benefit.
The 16 selected evidence cases contain 215 document occurrences; 62 mention
other CVEs and 41 share text with training CVEs. Neither proves duplicate
vulnerabilities. See exact excerpts and limitations in the separate TXT/JSON
review at `05_results/04_consolidated_studies/01_primary_day60/diagnostic_review_v1/`
under `EPSS_TPG`. Recommended next: the bounded D/DM/DGN/DGNM comparison,
with structured features fixed, both representations and three seeds. Twelve
fits can be reused; twelve new fits use the subsequently implemented PS04
selector above. The complete sources suite is larger. No training or test
scoring occurred during PS03 review.

**Implementation reference: PS03 saved-model diagnostic.**
Read the [diagnostic guide](05_studies/01_primary_tpg/07_FROZEN_MODEL_DIAGNOSTIC.md)
for the full-run commands. This inference-only stage measures responses to
source removal, structured-block removal, mask-matched text permutation and
edge interventions. A bounded evidence review retains exact text excerpts,
other-CVE mentions and candidate security-edge evidence without automatically
declaring them irrelevant or erroneous. All 59 consolidated regression tests
pass. The smoke run and its independent reproduction passed on 12 validation
CVEs using six full-trained seed-17 checkpoints; model states stayed unchanged.
The subsequently completed full diagnostic covers all 1,104 validation
CVEs and all 18 neural checkpoints. No training, new embeddings or test scoring
is performed. Diagnostic effects are not retrained ablation results.

**Completed: PS02 full core primary study and verified interpretation.**
All 21 fits and their validation predictions independently reproduce on the
unchanged 2,546 training / 1,104 validation CVEs. Read the
[core results](05_studies/01_primary_tpg/06_CORE_RESULTS.md).
Description pooled text + structured features has the lowest mean MAE,
0.02279227, versus the training-median baseline 0.02360351 (3.44% lower).
All-source TPG + structured has mean MAE 0.02296873 (2.69% lower).
Paired intervals for these baseline gains and graph-versus-text comparisons
include zero. The 50 validation CVEs with EPSS >= 0.1 account for about 86%
of these models' absolute error, and CVE-2025-0108 accounts for most of their
net gain. High category accuracy must not be presented as high recall or
exploitation prediction. No reliable TPG superiority is established.
The 18 neural fits completed 609 epochs; checkpoint selection and predictions
reproduce. No new training or test scoring occurred during review. The
separate TXT/JSON analysis is at
`05_results/04_consolidated_studies/01_primary_day60/core_review_v1/`
under `EPSS_TPG`. Remaining source/edge/structured suites are not yet completed.
Recommended next: bounded saved-model source/edge sensitivity and source
relevance checks before another large training suite; do not retune on test.

**Implementation reference: PS02 full primary-study training runner.**
Read [the full training guide](05_studies/01_primary_tpg/05_FULL_TRAINING.md)
for the one-by-one commands. The fixed core run contains six neural variants
at seeds 17/29/43 plus three deterministic baselines: 21 fits using all 2,546
training / 1,104 validation CVEs. MAE, AdamW, 100 maximum epochs and patience
15 retain the registered budget. Bounded graph packets and deterministic
two-pass gradient replay preserve all chunks and hierarchical source pooling.
Full source, edge and structured-block suites are available with shared-fit
reuse and epoch resume; the normal-text frontend cache remains unavailable.
All 68 supported configurations completed the four-training/two-validation
integration smoke and prediction verification. Fifty regression tests pass;
the full-data preflight passes. No full primary fit or test scoring was launched
during implementation; the subsequently completed core run is reported above.
Smoke artifacts are under
`05_results/04_consolidated_studies/01_primary_day60/training_v1/smoke/`
under `EPSS_TPG`. Smoke performance is not evidence for model selection.

**Completed: PS01 v2 full primary cache and independent review.**
The [full-cache review](05_studies/01_primary_tpg/04_FULL_CACHE_REVIEW.md)
confirms 2,546 training / 1,104 validation CVEs, 12,221 documents and 22,669
chunks: 11,266 reused and 11,403 new. All 125,573 artifact hashes passed;
all receipt/no-COREF graph counts and security provenance were checked;
twelve selected chunks passed repeated tensor validation (not a second
full tensor-semantic pass). Only one token uses the new mark exception.
The graphs contain 11,568 candidate security edges in 1,910 CVEs; 88.5% are
product/version links. SMP contributes 91.0% of node occurrences and includes
multi-CVE text, so count/coverage checks are not semantic-quality evidence.
Separate TXT/JSON review files are in
`05_results/04_consolidated_studies/01_primary_day60/cache_v2_full_review/`
under `EPSS_TPG`. The subsequently implemented v2-aware, memory-bounded
primary-training runner is documented above. No new model training,
test graphs or test scoring occurred in this review.

**Previous repair: PS01 v2 nonspacing-token coverage.**
The full v1 cache stopped on U+06E7 in a stored SMP for CVE-2021-30769. The
pinned tokenizer removes that isolated mark. The separate v2 policy retains
the text/node and permits a zero embedding only after exact-span, Unicode,
normalizer and contextual-offset checks. Ordinary words still require
embeddings. The real failing chunk passes and 41 combined tests pass. The
original v1 files are unchanged; the [v2 continuation guide](05_studies/01_primary_tpg/03_TOKENIZER_MARK_CORRECTION.md)
provides commands that reuse verified completed v1 chunks. No full expansion
or training was run during this correction.

**Implemented: PS01 primary adapter and matched training smoke.**
The [adapter guide](05_studies/01_primary_tpg/01_ADAPTER_AND_SMOKE.md) contains
paths and CLI commands. Full predictor/label export covers 2,546 training /
1,104 validation CVEs, 12,221 accepted documents and 22,669 planned chunks.
The six-CVE smoke reused 16 verified chunks and built four, then completed
seven three-epoch fits with checkpoint/prediction reproduction. Only nine
SEC_HAS_VERSION edges occur in its real graphs; synthetic qualifier tests
are separate and are not semantic-accuracy evidence. Thirty-one combined
tests pass. Additional structured fields and GNN options are documented,
not silently activated. That stage performed no full graph expansion, full
training or test scoring; the subsequent full-cache completion is recorded above.

**Implemented foundation: CS01 consolidated study design and preflight.**
Read [the two-study guide](05_studies/00_README.md) before starting another fit.
The agreed primary target is retrospective Day-60 EPSS, with all 15 text-source
combinations, matched pooled-text/TPG representations, structured on/off, and
additional controls. The secondary study covers every eligible daily anchor
and future lead through Day 90. CS01 implements contracts, model components,
structured encoders and coverage preparation; PS01 adds the primary adapters
and small training runner. Full training and final-test runners remain later.
The full preflight finds 2,546 / 1,104 primary CVEs and
2,563 / 1,104 continuous-forecast CVEs. No test scores have been inspected.
This consolidated design supersedes the pending narrow RO01 training proposal
below; it does not replace or reinterpret any completed experiment.

**Completed: [RO01 rolling-origin feasibility audit](#ro01-rolling-origin-feasibility-audit).**
The full audit independently reproduces: 21,741 eligible training windows and
8,585 validation windows, with all original 2,519 / 1,104 CVEs represented.
Later anchors add 46 training CVEs with a large endpoint change, increasing
the distinct count from 99 to 145 across the checked horizons. However,
91.80% of added training windows are unchanged; later anchors do not
consistently increase change counts on matched CVEs. All 2,182 date-blocked
windows have no numeric score exports, and exact CVE-weight totals pass.
One bounded matched training comparison is feasible: Day-30-only versus
pooled-anchor training, fixed 30-day lead, evaluated on the same Day-30
validation CVEs. This is a recommendation, not an implemented training run.
Existing results remain unchanged; no new forecast or test scoring occurred.

**Completed: [TR01 daily trajectories and training-only corrections](#tr01-daily-trajectories-and-training-only-corrections).**
All nine full checkpoints, controls and validation predictions independently
reproduce on the unchanged 2,519 training / 1,104 validation CVEs. Every neural
fit has worse joint validation MAE than persistence and the training-only
daily correction. Mean MAEs are 0.0072355168, 0.0072373439 and 0.0072436857
for 10/20/31-day histories, versus persistence 0.0072151528. The correction
control is effectively persistence: all source-precision residual medians
are zero. Learned corrections remain constant or nearly constant across CVEs.
TR01 is not promoted under the declared MAE criterion. No further full fit,
adaptive gate, TPG fusion or test scoring is authorized by this review.
The bounded RO01 rolling-origin feasibility check is now implemented, with
unchanged CVE roles and target-maturity rules, before deciding on more training.

**Completed: [MH01 fixed history lengths and multiple horizons](#mh01-fixed-history-lengths-and-multiple-horizons).**
All nine full fits and their validation predictions independently reproduce
on the unchanged 2,519 training / 1,104 validation CVEs. The 31-day history
has the lowest mean Day-37 MAE; 20 days has the lowest Day-44 and Day-60 MAE.
These improvements over persistence are only 0.235%, 0.319% and 0.221%.
Five models add exactly constant corrections across validation CVEs; the
remaining four have very small correction variation. No selected checkpoint
clips predictions. This does not establish an optimal history length or
useful CVE-specific trajectory learning. Its daily-trajectory successor,
TR01, is now complete and reviewed above.
No adaptive selection, daily trajectory, rolling anchors, TPG fusion or test
scoring is implemented in MH01. Do not restart its completed full fits.

**Completed: [GB01 group-balanced MAE](#gb01-group-balanced-mae).**
All six full fits are complete on the unchanged 2,519 training / 1,104 validation
CVEs. Every fit has worse ordinary validation MAE than its matched GV01 control.
Mean MAE is 0.0102259255 for EPSS-only and 0.0102931307 for all-source fusion;
all three fusion seeds are worse than persistence (0.0102523732). Mean fusion
RMSE improves slightly to 0.0602854461, so the result is a metric trade-off,
not deterioration on every measure. GB01 is not promoted under the declared
primary MAE criterion. No new experiment or test scoring is authorized.

**Completed: [GV01 fixed-cohort gain validation](#gv01-fixed-cohort-gain-validation).**
All six full fits reproduced. Mean validation MAE is 0.0102232855 for EPSS-only
and 0.0102243637 for all-source TPG + EPSS, versus persistence 0.0102523732.
The relative improvements are only 0.284% and 0.273%. No selected checkpoint
clips a validation prediction. Corrections are constant or nearly constant
within source-mask groups, and fusion attention is effectively uniform.
This does not establish useful text/edge learning. Training has 2,141 unchanged
endpoints, 258 increases and 120 decreases; validation has 6, 919 and 179.
GD01's small balanced-subset fitting improvement did not transfer strongly.

GD01's 12 full diagnostic fits and training probes were reproduced. Mean
best-training MAE changed from 0.03189114 to 0.01237729 for all-source fusion,
but only from 0.03197353 to 0.03184369 for EPSS-only. At the selected unit-gain
fusion checkpoints, 62-83 of 96 raw predictions were below zero and clipped.
These outcome-balanced training-subset results show improved fitting, not
validation improvement. Text interventions matter more than edge removal;
the learned EPSS-history correction remains weak. Test remains closed.

**Available: [SF01 structured timing audit and SF02 current-CVSS comparison](#sf01sf02-structured-availability-and-current-cvss).**
The audit covers the same 2,519 training / 1,104 validation CVEs. Current CVSS
vectors and scores are usable for 3,601 CVEs; 22 are missing. None has verified
field-level Day-30 availability. SF02 therefore requires explicit retrospective
opt-in. It compares two matched all-source TPG + EPSS models, with and without
current CVSS. Both smoke checkpoints reproduce and the full-data preflight
passes; the six full fits remain for the user. Twenty new tests and all 28
existing Day-60 tests pass.
Raw data, MS03/MS06, graph caches and held-out test data are unchanged.

**Existing: [MS05 Day-30 graph cache](#ms05-day-30-aware-graph-cache) is built;
[MS06 Day-60 matched training](#ms06-matched-day-60-training) completed all nine full fits and verification.**
The full cache covers 2,519 training / 1,104 validation CVEs: 10,110 chunks
reference verified existing artifacts and only 21 required new graphs and
embeddings. All three smoke fits and nine full fits reproduce. The mean
validation MAEs are 0.0102520316 (EPSS-only), 0.0102520436 (description fusion)
and 0.0102519955 (all-source fusion), versus persistence 0.0102523732.
The improvement is negligible; learned corrections remain nearly constant.
Of 2,519 training CVEs, 2,141 have unchanged endpoints, compared with only
6 of 1,104 validation CVEs. Existing MS03 results are unchanged; no test
scoring has been performed. Do not restart completed MS06 training.

**Completed and verified: [MS04 Day-60 multi-source preparation](#ms04-day-60-multi-source-preparation).**
The user's full and smoke exports independently reproduce: 2,519 eligible
training / 1,104 validation CVEs for Days 0-30 EPSS inputs and the observed
Day-60 target. The smoke histories, labels, source routing, IDs and training
registry agree with full. All 79 prerequisite hashes and both output locks
pass. Every existing CD02 CVE role is preserved. Fifteen new synthetic tests
and the eleven existing CD02 tests passed during implementation. MS04 itself
does not train models. The MS05 cache and MS06 runner now consume its outputs;
no test evaluation has run.

**Completed and verified: [MS03 matched multi-source training](#ms03-matched-multi-source-training).**
All nine full checkpoints and their validation predictions reproduce, along
with the three smoke checkpoints. EPSS-only, description TPG + EPSS, and all
accepted text TPGs + EPSS are almost persistence. Text adds no consistent MAE
benefit. Seven fits have constant corrections; the other two vary only across
source-availability masks in these saved predictions. The learned temporal
and active node-projection weights shrink close to zero. Test remains closed.

**Completed cache: [CD03 v2 multi-source graph cache](#cd03-inspectable-multi-source-graph-cache).**
The user's full graph/tensor verification passed. An independent review also
rechecked all 69,702 file hashes and reproduced report totals without repeating
the 44-minute tensor traversal. The cache covers 2,548 training / 1,104 validation
CVEs, 10,438 document associations and 11,121 chunks: 3,006,451 nodes and
9,517,311 no-COREF edges. Exactly 5,700 chunks retain v1 receipts; 5,421 were
built under v2. The original v1 cache remains unchanged. No training occurred.
MS03 training with this cache is now complete and reviewed below.
Read the [artifact guide](../01_tpg/00_ARTIFACT_GUIDE.md).

**Verified preparation: [CD02 CVE-based development preparation](#cd02-cve-based-development-preparation).**
The new version uses separate CVE identities, without transitive URL/post/text
quarantine. It prepares 2,548 training / 1,104 validation CVEs and reserves 651
eligible test CVEs under the reviewed April/July 2025 dates. The user's smoke
and full exports now reproduce; smoke histories/labels/source routing match
full exactly. There are 10,438 document associations and only two masked
validation posts, with no CVE removed for overlap. The source-aware cache is
now complete; the fixed MS03 fusion runner is implemented. No model training has occurred
in CD02/CD03.

## RO01: Rolling-Origin Feasibility Audit

### Full Results Review

The full export was independently reproduced. Additional checks confirm
32,607 unique candidate windows, no changes to CVE roles, no numeric fields
exported for the 2,182 date-ineligible windows, and exact total weight one per
eligible CVE and per CVE/lead. No training or new model predictions occurred.
The NVML warning is non-fatal for this CPU audit; verification exited successfully.

| Split | Original CVEs | Candidate windows | Eligible windows | Date exclusions | Missing endpoints |
|---|---:|---:|---:|---:|---:|
| Training | 2,519 | 22,671 | 21,741 | 831 | 99 |
| Validation | 1,104 | 9,936 | 8,585 | 1,351 | 0 |

All data-availability exclusions here are missing endpoints; no additional
window is excluded because all observations in a compared history are absent.
Every original CVE retains at least one eligible window. These 30,326 usable
windows are not 30,326 independent CVEs.

**Additional changing examples exist, but remain limited.** Across the fixed
7/14/30-day leads, later anchors reveal endpoint changes in CVEs that were
unchanged at the checked Day-30 endpoints:

| Training-CVE count | Day-30 anchor only | All three anchors | Added by later anchors |
|---|---:|---:|---:|
| Any nonzero endpoint change | 379 | 530 | 151 |
| Any absolute endpoint change at least 0.01 | 99 | 145 | 46 |

A change of 0.01 means one EPSS percentage point, not exploitation evidence.
This compares only the audit's selected endpoints. TR01 inspected every
future day; its count of 100 training CVEs with any large daily change is
therefore not the same quantity as this audit's 99 Day-30-anchor CVEs.
Likewise, the corresponding validation counts have different scopes.

Of 7,550 eligible training windows starting on Day 30, 88.70% are unchanged
relative to the latest input score. The 14,191 added windows starting on
Days 45/60 are 91.80% unchanged. Pooled training is 90.72% unchanged,
compared with 30.91% of pooled validation windows. Thus more windows do not
automatically resolve the earlier predominance of unchanged training scores.
Large changes occur in 524/21,741 training windows (2.41%) and 454/8,585
validation windows (5.29%). Equal-CVE weighting gives corresponding shares
of 2.32% and 5.24%. These remain descriptive outcomes, not sampling rules.

**Matched comparisons do not show that later is consistently better.**
The following rows keep the same CVEs and a 30-day future lead while changing
the anchor. Each row has its own matched CVE set:

| Split | Later anchor | Matched CVEs | Large changes at Day-30 anchor | Large changes at later anchor | CVEs with large change only at later anchor |
|---|---:|---:|---:|---:|---:|
| Training | Day 45 | 2,354 | 96 | 89 | 32 |
| Training | Day 60 | 2,043 | 95 | 80 | 44 |
| Validation | Day 45 | 800 | 59 | 52 | 9 |
| Validation | Day 60 | 514 | 40 | 30 | 6 |

The extra later-only cases are useful to identify, but some cases change
only at the earlier anchor. For example, the training Day-30/Day-60 comparison
has 44 later-only and 59 earlier-only large-change CVEs. Do not choose an
anchor or horizon just because its future outcome counts look attractive.

**Weights and masks are needed.** Training CVEs contribute 5-9 eligible
windows each; 1,951 have all nine. Validation also has 5-9, with 514 having
all nine. Exact rational checks pass for 2,519 training and 1,104 validation
CVE weight groups, plus 7,557 training and 3,312 validation CVE/lead groups.
Twelve eligible training windows lack a score on the anchor date: six use
a score one day earlier, three two days earlier and three four days earlier.
No eligible validation window has this condition. These are windows, not
necessarily distinct CVEs; observation age is recorded, not silently reset.

**Real example:** CVE-2021-21797 has EPSS 0.00731 at both Days 30 and 60.
Its score reaches 0.03197 at Day 90, an increase of 0.02466 from Day 60.
Both windows remain in training. This illustrates additional later change,
not evidence that a model could predict it or that exploitation caused it.

**Recommended next step:** one bounded EPSS-only comparison is now feasible.
Use fresh matched models with fixed 31-day histories and a scalar 30-day lead:
train one on Day-30 windows only and the other on eligible Day-30/45/60 windows.
This yields 2,519 versus 6,916 training windows from the same 2,519 CVEs.
Use RO01's lead-specific CVE weights for the pooled model. Keep the primary
validation comparison fixed at the same 1,104 Day-30 CVEs predicting Day 60,
so the outcome is not made easier by evaluating a different anchor population.
Retain persistence and training-only correction controls, preserve date
exclusions, and freeze the architecture, loss, seeds and selection rule before
implementation. The 30-output TR01 checkpoint is not a matched scalar-task
control; both neural controls need fresh initialization. Start with smoke.
No adaptive gate, TPG fusion, resplitting or final test scoring is authorized
by this review, and no training implementation has been added here.

The separate review is saved as `summary.txt` and `summary.json` in
`../05_results/02_evaluation/epss_rolling_origin/feasibility_v1/full_review`.
It records these derived counts, exact checks, source hashes and limitations.
Original locked audit files remain unchanged.

### Scope

TR01 did not improve joint MAE over persistence. RO01 checks whether later
decision times supply additional usable, changing training examples before
we define another model run. It is an audit, not a training-data export or
automatic approval to train. It does not select windows by future change.

| Item | Fixed rule |
|---|---|
| CVEs | Same 2,519 training / 1,104 validation CVEs as TR01 |
| Prediction anchors | Days 30, 45 and 60 after publication |
| Future leads | 7, 14 and 30 days after each anchor |
| Histories checked | Latest 10, 20 and 31 days, including the anchor date |
| Target | Observed endpoint score at anchor + lead |
| Date eligibility | Anchor and endpoint must remain within the CVE's original chronological role |
| Input eligibility | At least one observed score in each compared history; missing days otherwise retained |
| Outcome selection | None: increase, decrease and size of change do not affect eligibility or weights |
| Numerical comparison | Endpoint minus latest observed input score, in source precision |
| Large-change description | Absolute difference at least 0.01, meaning one EPSS percentage point |

For example, the Day-45 anchor with a 14-day lead examines the Day-59
endpoint and only histories ending on Day 45. The Day-60 anchor with a
30-day lead examines Day 90. All endpoints lie in the existing 91-day
source record. This endpoint audit does not count a temporary increase
that reverses before the endpoint; it is not an analysis of every intermediate
future score or a new 30-output trajectory-training protocol.

### Split and Missingness Checks

Each CVE retains its original role, even if a later anchor would now fall
in another period. The window is excluded rather than moved. An endpoint
equal to the next partition's start is also excluded. The existing observation
cutoff is checked. Date-ineligible windows record dates and reasons but do
not inspect or export numeric input/target scores for that window.

Only development CVEs are selected before numerical adaptation. The source
file is parsed and hashed as a whole for integrity, but no test CVE's score
history or text is adapted or exported. The audit checks the completed TR01
artifact hashes and frozen dependencies; it does not rerun a neural model.

For admissible windows, an unobserved endpoint is not imputed or treated as
unchanged. If the shortest history is empty, a score outside that history
cannot rescue it. A missing anchor-day score is allowed when an earlier
visible score exists in every history; the actual latest day and age are
recorded. The three histories must have identical latest observed scores.

The full preflight and subsequent completed audit agree on the following counts:

| Split | Unique CVEs | Candidate anchor/lead windows | Eligible windows | Excluded by dates | Other excluded windows |
|---|---:|---:|---:|---:|---:|
| Training | 2,519 | 22,671 | 21,741 | 831 | 99 |
| Validation | 1,104 | 9,936 | 8,585 | 1,351 | 0 |

There are nine candidate windows per CVE, not nine independent CVEs. All
2,519 training and 1,104 validation CVEs retain at least one eligible window.
Other exclusions concern data availability; inspect the full audit's reason
counts for detail. Multiple missingness reasons may describe one window, so
reason counts are explicitly nonexclusive. These totals do not establish
that the new windows contain more useful predictive information.

### Matched Comparisons and Weights

Reports include coverage, unchanged endpoints, increases, decreases, large
changes, partial-history counts and missing anchor-day scores for every
split/anchor/lead combination. A second set of comparisons uses only CVEs
eligible at both Day 30 and the later anchor for the same lead. It reports
large changes at each anchor, at both, and at only one. There is also a
common-CVE comparison across all three anchors. These matched comparisons
help distinguish changes in behavior from changes in the eligible population.
They remain descriptive, not evidence that a later prediction is an earlier
warning or that a particular anchor should be selected after seeing outcomes.

Two proposed loss-weight plans are recorded, without using them to train:

- Pooled leads: each window receives `1 / eligible windows for that CVE`.
- Separate lead-specific tasks: each window receives
  `1 / eligible anchors for that CVE and lead`.

Exact rational checks establish total weight one per represented CVE (or
CVE/lead). Both the denominator and floating-point weight are saved. Window
counts and a CVE-balanced large-change percentage are reported separately.
The weights depend on availability and date eligibility, never future
score magnitude. They address unequal window counts, not statistical
independence between repeated observations of one CVE.

### Files and Commands

Read these files in order:

1. `04_configs/rolling_origin_audit_v1.json`: fixed candidate grid and rules.
2. `01_package/epss/forecast_rolling_origin_audit.py`: date-first inspection,
   availability checks, paired summaries, weights and artifact reproduction.
3. `03_analysis/01_revision_dataset/test_forecast_rolling_origin.py`: 13 tests
   covering blocked access, date boundaries, no role changes, input/target
   separation, missingness, weights, matched cohorts and tamper detection.
4. `03_analysis/01_revision_dataset/audit_forecast_rolling_origin.py`: CLI.

Outputs belong to
`../05_results/02_evaluation/epss_rolling_origin/feasibility_v1/{smoke,full}`:

- `summary.txt`: plain-language tables and interpretation rules.
- `summary.json`: grouped counts, paired comparisons, CVE coverage and limitations.
- `window_audit.json`: one diagnostic row for each CVE/anchor/lead, including
  exclusions, allowed scores, observed-input counts, differences and weights.
  This file contains targets and is explicitly **not model input**.
- `protocol.json`, `run_identity.json`, `audit_lock.json`: fixed scope, original
  CVE roles, split boundaries, source hashes and output integrity.

The 16-training / 8-validation smoke audit is already saved and reproduced:
138/144 training windows and 60/72 validation windows are eligible. Five
training and twelve validation windows are date-excluded; one other training
window lacks its endpoint. No model was trained. The subsequent full audit is
also complete and reproduced; its findings are reviewed above.

The full audit is complete; use `--verify-only` to check it. The sequence below
is retained for reproduction on a fresh checkout, from the repository root
in `CodeBERTFusion`:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_rolling_origin.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/audit_forecast_rolling_origin.py --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/audit_forecast_rolling_origin.py --scale full
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/audit_forecast_rolling_origin.py --scale full --verify-only
```

`--check-only` performs the audit checks without saving files. Existing audit
directories are never overwritten; use `--verify-only` after success. This
CLI has no training mode. Review the full results before defining a future
training protocol, and do not shift the partition boundaries to admit more
later windows. The parent Day-60 eligibility selection and repeated use of
validation remain limitations.

## TR01: Daily Trajectories and Training-Only Corrections

### Full Results Review

All nine full fits and both baseline controls reproduced using the saved
checkpoints. No optimizer updates, retraining or test scoring occurred during
the review. The original locked outputs remain unchanged. Additional findings
and source hashes are saved separately in
`../05_results/02_evaluation/epss_day30_trajectory/history_controls_v1/full_review/summary.json`
and `summary.txt`.

All 1,104 validation CVEs have 30 observed future scores. Joint MAE therefore
averages over 33,120 CVE/day predictions. Each neural row below is the mean
of three seed metrics, not an ensemble prediction. Lower error is better.

| Method | Joint validation MAE | Relative MAE change from persistence | Mean pooled RMSE |
|---|---:|---:|---:|
| Persistence | 0.0072151528 | Reference | 0.0490265415 |
| Training-only daily correction | 0.0072151527 | Effectively identical | Effectively identical |
| GRU, 10-day history | 0.0072355168 | 0.282% worse | 0.0490212471 |
| GRU, 20-day history | 0.0072373439 | 0.308% worse | 0.0490213930 |
| GRU, 31-day history | 0.0072436857 | 0.395% worse | 0.0490199206 |

Pooled RMSE is the square root of mean squared error across all CVE/day
predictions, calculated per fit and then averaged across seeds. It is not
the average of 30 daily RMSEs. These very small RMSE gains do not reverse
the negative result under the declared joint-MAE criterion. Every individual
neural fit, not just each history average, has worse joint MAE than both
controls. Mean daily MAE improves on 10 of 30 days for the 10-day history,
11 of 30 for 20 days and 7 of 30 for 31 days. Do not describe every metric
or every day as worse.

The original checkpoint rule selects the best trained epoch starting at
epoch 1. It does not select the untrained persistence initialization as an
epoch-0 checkpoint. A correctly selected trained checkpoint can therefore
still lose to the separate persistence reference; verification confirms the
selection rule, not predictive superiority.

| History days | Seed | Selected epoch | Epochs run | Joint validation MAE | Clipped CVE/day predictions |
|---|---:|---:|---:|---:|---:|
| 10 | 17 | 3 | 18 | 0.0072383212 | 65 |
| 10 | 29 | 14 | 29 | 0.0072393606 | 181 |
| 10 | 43 | 11 | 26 | 0.0072288685 | 81 |
| 20 | 17 | 32 | 47 | 0.0072319850 | 168 |
| 20 | 29 | 7 | 22 | 0.0072395093 | 182 |
| 20 | 43 | 15 | 30 | 0.0072405373 | 86 |
| 31 | 17 | 19 | 34 | 0.0072502150 | 83 |
| 31 | 29 | 4 | 19 | 0.0072361448 | 173 |
| 31 | 43 | 8 | 23 | 0.0072446972 | 257 |

Clipping counts are individual future-day predictions out of 33,120 per fit,
not distinct CVEs. All selected clipping is at zero. Raw, unclipped MAE is
also worse than persistence for every fit. Some unselected epochs clipped
more predictions, up to 3,867 in one epoch of the 20-day, seed-17 run.

**The simple correction is effectively zero.** All 30 fitted offsets are
`1.4500692476198535e-11`. Recomputing the same weighted medians with the
original input-score precision, rather than the float32 tensor representation,
gives exactly zero for every day. The minute numerical difference from
persistence is rounding, not a useful learned signal. This is a legitimate
outcome of the control, not a missing fitting step.

**The neural corrections still barely distinguish CVEs.** The 10-day models
for seeds 29 and 43 and the 20-day model for seed 29 have exactly constant
corrections across validation CVEs at all 30 future days. Four other models
vary only at numerical scales: their largest daily correction ranges are
between about 7e-12 and 4e-10. The remaining two vary more, but still very
little: the 10-day seed-17 model's largest daily correction standard deviation
is 5.14e-8, and the 31-day seed-17 model's is 2.06e-6. Final predictions differ
between CVEs because their latest observed scores differ. That is not evidence
of learning CVE-specific future trajectories.

**Training and validation have different change distributions.** Compare
each future score to that CVE's latest observed input score, using the original
saved numerical precision. Missing future scores are excluded, not treated
as unchanged. The last column below counts CVEs whose every observed future
score equals the latest input; training CVEs may still have missing days.

| Split | Observed future scores | Unchanged future scores | Unchanged share | CVEs with every observed future score unchanged |
|---|---:|---:|---:|---:|
| Training | 75,245 | 67,250 | 89.37% | 2,140 / 2,519 |
| Validation | 33,120 | 2,693 | 8.13% | 0 / 1,104 |

The training scores include 5,222 increases and 2,773 decreases; validation
has 26,322 increases and 4,105 decreases. Most differences can be small:
only 100 training CVEs and 116 validation CVEs have any absolute change of
at least 0.01 in the 30-day future period. This threshold means one EPSS
percentage point. These observations explain why a training median correction
can be zero and identify a substantial distribution difference. They do not
prove the reason for neural failure, prove a source-data bug, or justify
moving validation CVEs into training.

**Real examples.** These are selected illustrations from `history10__17`,
not a random sample or a best-seed claim. Each forecast uses only information
through Day 30.

| CVE | Latest input EPSS | Actual Day 37 | Predicted Day 37 | Actual Day 60 | Predicted Day 60 |
|---|---:|---:|---:|---:|---:|
| CVE-2025-1094 | 0.000430 | 0.818900 | 0.000622 | 0.840910 | 0.000403 |
| CVE-2025-27363 | 0.053690 | 0.094880 | 0.053882 | 0.754640 | 0.053663 |
| CVE-2025-24989 | 0.257230 | 0.234420 | 0.257422 | 0.077020 | 0.257203 |

For this named fit, the ten CVEs with the largest average trajectory error
account for 47.88% of total absolute error and 79.41% of squared error across
all 30 days. Major increases and decreases remain poorly captured.

**Decision and next step.** Retain persistence and the completed comparisons;
do not promote TR01 or repeat full training with more outputs alone. This
experiment is EPSS-only: it does not establish whether TPG, summaries or SMPs
can help. No adaptive history gate is supported by these findings yet.

Before another model run, preflight the already proposed rolling-origin task:
fixed prediction anchors and future leads, unchanged CVE partition roles,
explicit target maturity and missingness, and equal total training weight per
CVE when it supplies multiple windows. First check whether later anchors
actually provide additional changing training trajectories. Do not assume
they do, select windows using future changes, or let windows from the same
CVE cross roles. If the resulting data still offers little changing training
signal, report that limitation before adding model complexity. No new task,
training run or final test evaluation was implemented by this review.

### Question and Fixed Design

MH01 produced mostly constant or nearly constant adjustments to the latest
EPSS score. TR01 asks whether predicting the whole next month learns useful
differences between CVEs, beyond repeating the last score or adding a simple
day-specific correction. It predicts the 30 future values together. It does
not feed a real future score back into the model to predict the next day.

| Item | Fixed choice |
|---|---|
| Prediction time | Day 30 after publication |
| History controls | Days 21-30, Days 11-30 and Days 0-30: 10, 20 and 31 slots |
| Outputs | 30 EPSS scores: Day 31, Day 32, ..., Day 60 |
| Cohort | Existing 2,519 training / 1,104 validation CVEs, unchanged roles |
| Persistence | Repeat the latest observed input score for all 30 days |
| Simple correction | Fit one offset per future day using training CVEs only |
| Neural model | GRU (5 input channels, hidden size 16), hidden head size 32, 30 outputs |
| Neural output | Latest score + tanh(head), gain 1, clipped to [0,1] |
| Full fits | 3 history lengths x seeds 17, 29, 43 = 9 fresh fits |
| Optimization | Adam 0.001, weight decay 0.0001, batch 8, gradient norm clip 1 |
| Budget | Up to 100 epochs, patience 15; CPU, 4 threads, deterministic algorithms |
| Selection | Earliest strictly best validation mean per-CVE observed-day MAE |

No history length is selected separately for each validation CVE. All fixed
seeds and all daily outputs are reported. The 30-output head differs from
MH01's three-output head, so this is a new joint task, not a pure history
length comparison against MH01. Within TR01 the three histories share CVEs,
targets, batch orders and initial weights per seed. Day-to-day increases and
decreases are both allowed; there is no monotonic trajectory constraint.

### How the Simple Correction Works

For each future day, compute the observed training residual:

```text
residual = actual future EPSS - latest EPSS visible by Day 30
future prediction = clip(latest visible EPSS + fitted daily offset, 0, 1)
```

The offset is a weighted median of these residuals. A training CVE contributes
weight `1 / number of observed future days` to each of its observed targets.
This matches the neural loss: average errors over available days within a
CVE, then average CVEs. Missing targets are not assigned zero or imputed.
With complete trajectories, all CVEs have equal weight for each daily median.
At an exact half-weight tie, choose the lower median deterministically.
Weights are accumulated as exact fractions to avoid rounding the tie rule.

For example, if a fitted offset is 0.002, a latest score of 0.010 gives a
prediction of 0.012 for that day. This is only an arithmetic illustration,
not a reported dataset result. The offset is shared across CVEs, unlike a
useful neural correction that would respond to each CVE's own history.

The fitting function receives only training IDs, training histories and
training labels; it rejects extra label records. Validation labels are used
to measure its error, never to fit or select offsets. One shared control is
enough because the latest observed score is identical across the three
matched history views. The saved baseline includes training IDs, offsets,
counts, weights, predictions and clipping. The weighted median minimizes raw
weighted absolute residuals; it is not separately optimized for clipping.
Baseline arithmetic is float64 on the same float32 persistence values used
by the neural model. Tiny rounding differences are not meaningful gains.

### Coverage and Leakage Boundaries

| Split | CVEs | CVEs with all 30 future scores | CVEs missing at least one future score | Observed CVE/day targets | Missing CVE/day targets |
|---|---:|---:|---:|---:|---:|
| Training | 2,519 | 2,215 | 304 | 75,245 | 325 |
| Validation | 1,104 | 1,104 | 0 | 33,120 | 0 |

These counts were checked in the full preparation preflight. A CVE/day target
means one future EPSS score for one CVE: the training maximum is
2,519 x 30 = 75,570 values. Partial trajectories remain in the experiment.
All 30 validation scores are present, so every daily comparison uses the same
1,104 CVEs. Input-history missingness is unchanged from MH01; input masks and
observation ages are recomputed within each history window.

Preparation checks the frozen parent records and raw-file hash, filters to
development IDs before adapting scores, and validates every future date
against the inherited split rules. Inputs and labels are separate files.
No future targets, target-derived weights, descriptions, summaries, SMPs,
CVSS values or KEV fields are model inputs in TR01. No test histories, labels
or text are exported. The parent cohort still requires an observed Day-60
target; this is not a newly expanded cohort of every CVE with any short-term
target. Related CVEs are not guaranteed independent, and repeated validation
development is not final test evaluation.

### Results to Inspect

The full run will save:

- Per-day MAE, RMSE, signed error and rank correlation, with observed counts.
- Mean per-CVE trajectory MAE and complete-trajectory comparisons.
- Each model's daily and joint error differences from the training-only
  correction control, plus daily differences from persistence.
- Every validation CVE's actual scores or nulls, target masks, dates,
  persistence values, raw forecasts, clipped forecasts and learned corrections.
- Daily correction variability and counts of forecasts clipped at zero or one.
- Descriptive daily errors for increases of at least 0.01, decreases of at
  least 0.01, and smaller changes relative to persistence. Here 0.01 means
  one EPSS percentage point. These outcome groups are never predictive inputs
  or training-selection rules; their membership may change across days.
- Epoch-level masked training loss, raw training loss, observed target counts,
  clipping, data-gradient magnitudes and validation metrics.
- Checkpoints, fixed batch orders and file hashes for reproduction.

Console progress bars cover preparation, batches, epochs and model fits.
Epoch messages show joint error, selected daily errors (+1/+7/+14/+30),
clipping and gradient totals. Every daily metric is retained in JSON even
though not all 30 are printed in each epoch message.

The implementation smoke run has 16 training / 8 validation CVEs, one seed
and three epochs for each history length. Three training targets are missing
and masked. All three checkpoints and both baseline controls reproduce.
This is an execution check, not evidence of useful forecast accuracy. The
neural smoke forecasts equal persistence; no model benefit is claimed.
Seventeen focused tests also pass, including train-only fitting, weighted
median ties, missing-target gradients, dates, clipping, matched initialization,
manual optimizer parity, resume, tamper detection and a synthetic nine-fit
lifecycle. The implementation itself ran only smoke; the user's subsequent
full run is now complete and verified in the full results review above.

### Files and Commands

Read the files in this order:

1. `04_configs/trajectory_v1.json`: fixed protocol and training settings.
2. `01_package/epss/forecast_trajectory_data.py`: matched histories, daily
   targets, coverage, chronological maturity and preparation locks.
3. `01_package/epss/forecast_trajectory.py`: controls, GRU, masked loss,
   comparisons, diagnostics, training and reproduction.
4. `03_analysis/01_revision_dataset/test_forecast_trajectory.py`: regression tests.

These are versioned successors of MH01. The old scripts, prepared files,
models, graph caches and results remain unchanged.

Prepared data: `../04_data/01_records_and_graphs/day30_trajectory_v1/{smoke,full}`.
`inputs.json` contains past histories; `labels.json` holds the 30 future
values and masks. `coverage.json` and `summary.txt` describe missingness.
`split_manifest.json`, `protocol.json`, `run_identity.json` and the preparation
lock record the roles and provenance.

Results: `../05_results/02_evaluation/epss_day30_trajectory/history_controls_v1/{smoke,full}`.
Start with `summary.txt` and `summary.json`. `baselines.json` includes both
controls and the training-only fitted offsets. Each `fits/history{length}__{seed}`
directory contains `checkpoint.pt`, `history.json`, `predictions.json`,
`metrics.json` and a lock. The top-level evaluation lock protects the complete
run. Verification refits the simple deterministic training baseline and
reproduces saved validation predictions without neural retraining or test scoring.

The full run is now complete. Use `--verify-only` to check it; do not rerun
training. The following sequence is retained for reproducibility on a fresh
checkout, from the repository root in `CodeBERTFusion`:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_trajectory.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_trajectory.py --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_trajectory.py --scale full
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_trajectory.py --scale full --verify-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_trajectory.py --scale full --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_trajectory.py --scale full
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_trajectory.py --scale full --verify-only
```

The smoke artifacts already exist locally. On a fresh checkout, first run
`prepare_forecast_trajectory.py --scale smoke`, followed by
`run_forecast_trajectory.py --scale smoke`. Full execution requires a verified
smoke run. Existing preparation is not overwritten; verify it instead.
For an interrupted full run, use `run_forecast_trajectory.py --scale full --resume`.
Completed fits are preserved; an incomplete fit restarts from fixed
initialization. Keep the default four threads for exact reproduction.

Review the full daily comparisons before deciding whether adaptive history
selection, rolling forecasts or matched TPG fusion is justified. None is
implemented by this stage, and no final test evaluation is authorized here.

## MH01: Fixed History Lengths and Multiple Horizons

The fixed-task gain and weighting comparisons are complete. This is the next
stage, not another loss-weighting experiment. We first measure whether the
amount of observed history affects forecasts at several useful future leads.
There is no automatic claim of an optimal history or horizon.

### Fixed Design

| Item | Definition |
|---|---|
| Prediction anchor | Day 30 after CVE publication, unchanged for every view |
| Latest 10 daily slots | Days 21-30 inclusive |
| Latest 20 daily slots | Days 11-30 inclusive |
| Latest 31 daily slots | Days 0-30 inclusive |
| Future leads | +7, +14 and +30 days |
| Actual target days | Days 37, 44 and 60 after publication |
| Full development cohort | Existing 2,519 training / 1,104 validation CVEs |
| Full fits | Three history lengths x seeds 17, 29, 43 = nine fits |
| Outputs from each fit | Three EPSS forecasts together, not three separately trained models |
| Reference | Latest observed visible EPSS repeated for each horizon |
| Model | GRU with five past-input channels, hidden size 16; head hidden size 32 with three outputs |
| Output mapping | Persistence + tanh(head), gain 1; final predictions clipped to [0,1] |
| Optimizer/budget | Adam 0.001, weight decay 0.0001, batch 8, gradient norm clip 1; at most 100 epochs, patience 15 |
| Selection | Earliest strictly best validation mean per-CVE observed-horizon MAE from epoch 1 |

The model may predict either increases or decreases, independently at each
horizon. There is no monotonic trajectory constraint. The head has three
outputs, so this is a new joint task, not a pure history-length ablation
against single-output GV01. Within MH01, all history lengths have identical
parameter layouts, starting weights per seed, CVEs, target masks, batch
schedules and training settings. No text, CVSS, KEV or future EPSS enters
the encoder. TPG caches and earlier checkpoints are unchanged.

### Masks and Temporal Safety

All current development CVEs have at least one observed score within Days
21-30, so all three history views can use the same cohort and persistence
base. Shorter windows recompute observation age and age-known flags inside
their own window. A score or observation age before Day 21 cannot leak into
the 10-day view. Empty windows would fail explicitly rather than silently
fall back to earlier history or remove CVEs.

| Split | CVEs | Observed Day 37 | Observed Day 44 | Observed Day 60 | All three observed |
|---|---:|---:|---:|---:|---:|
| Training | 2,519 | 2,514 | 2,517 | 2,519 | 2,512 |
| Validation | 1,104 | 1,104 | 1,104 | 1,104 | 1,104 |

These counts were checked in the pinned source for the current development
IDs only. The five missing Day-37 and two missing Day-44 training values
remain null in the exported labels. Training averages absolute error over
observed horizons within each CVE, then averages CVEs. A CVE with fewer
observed targets therefore does not receive less total loss weight merely
because some targets are missing. An internal masked tensor placeholder is
not an imputed target and contributes no loss or gradient.

The full parent cohort already requires an observed Day-60 score. Therefore
MH01 is conditional on that selection; it does not represent every CVE that
could support a shorter forecast. No new eligibility expansion occurs here.
Parent chronological roles are preserved, and each target must mature inside
its assigned partition. Preparation checks the raw-file hash and filters to
the selected train/validation IDs before adapting histories or targets.
No numerical test histories, target scores or test text are exported.

### Evaluation and Interpretation

Reports give MAE, RMSE and improvement over persistence separately for each
future lead, with observed target counts. Across history lengths, each lead
uses exactly the same CVEs and scores. Complete-case metrics additionally
compare all horizons on the same CVEs; currently all validation cases qualify.
All fixed seeds are reported, together with paired errors relative to the
31-day view. Aggregate means are means of seed metrics, not ensemble predictions.

The joint checkpoint-selection metric averages observed-horizon errors within
each CVE and then averages CVEs. The report also records macro horizon MAE.
They can differ with missing targets, so the selection rule is explicit and
not changed after results arrive. Shorter leads may be easier; lower error
alone does not make a short horizon a more useful warning. No automatic
horizon selection, uncertainty calibration or test-based tuning is performed.

### Full Results Review

The nine full fits, checkpoint selection, input locks, masks, persistence
references and saved validation predictions were independently reproduced.
No retraining or test scoring occurred during this review. All 1,104
validation CVEs have observed targets at each of the three horizons, so
differences below are not caused by changing validation populations.

Each cell is mean validation MAE over seeds 17, 29 and 43, not an ensemble
prediction. Lower is better. Persistence repeats the latest observed input
score. Day numbers count from CVE publication; all forecasts start at Day 30.

| Score being predicted | Days ahead of Day 30 | Persistence MAE | Latest 10 days MAE | Latest 20 days MAE | Latest 31 days MAE |
|---|---:|---:|---:|---:|---:|
| Day 37 | 7 | 0.004506812 | 0.004511672 | 0.004504576 | 0.004496213 |
| Day 44 | 14 | 0.007478913 | 0.007461075 | 0.007455043 | 0.007468290 |
| Day 60 | 30 | 0.010252373 | 0.010230716 | 0.010229679 | 0.010240988 |

The best mean gains relative to persistence are 0.235%, 0.319% and 0.221%,
respectively. These are relative error reductions, not EPSS percentage-point
changes. The 10-day model is slightly worse than persistence at +7 days on
average. All nine fits improve the joint three-horizon MAE, but not every
fit improves every individual horizon. Mean joint MAE is 0.007401154 for
10 days, 0.007396433 for 20 days and 0.007401830 for 31 days, compared with
0.007412699 for persistence.

The apparent winner changes with the seed. At +7 days, 31 days wins for
seeds 17 and 29, but 20 days wins for seed 43. At +14 days, 10 days wins
for seed 17 and 20 days for the other two. At +30 days, 10 days wins for
seeds 17 and 43, and 20 days for seed 29. Twenty days beats 31 days at
+30 for all three seeds, but the absolute MAE differences are only
0.00002737, 0.00000433 and 0.00000223. These small differences do not
establish a generally optimal history length. No significance claim is made.

| History days | Seed | Selected epoch | Epochs run | Joint validation MAE |
|---|---:|---:|---:|---:|
| 10 | 17 | 10 | 25 | 0.007402925 |
| 10 | 29 | 1 | 16 | 0.007405880 |
| 10 | 43 | 14 | 29 | 0.007394658 |
| 20 | 17 | 1 | 16 | 0.007400794 |
| 20 | 29 | 11 | 26 | 0.007393174 |
| 20 | 43 | 18 | 33 | 0.007395331 |
| 31 | 17 | 16 | 31 | 0.007398667 |
| 31 | 29 | 21 | 36 | 0.007397539 |
| 31 | 43 | 15 | 30 | 0.007409284 |

The checkpoint was selected jointly across all three outputs, not separately
for each target day. These epochs follow the original patience rule.

**What the forecasts actually do.** Five selected models produce one exactly
constant correction per horizon across all 1,104 validation CVEs:
`history10__17`, `history10__43`, `history20__29`, `history31__17` and
`history31__43`. This does not mean their final EPSS predictions are identical:
each CVE still contributes its own last observed score. It means the learned
adjustment does not distinguish CVEs in these saved predictions. The other
four models vary slightly; the largest per-horizon correction standard
deviation is only 0.000001462 (`history20__17`, +7). This is evidence of
near-persistence behavior on validation, not proof that every internal
representation is identical or that EPSS history has no predictive value.

No selected checkpoint clips any validation output at zero or one. Other
epochs do clip outputs: the largest count is 956 out of 3,312 CVE/horizon
predictions in one epoch of `history20__17`. These are predictions, not
956 distinct CVEs. Final MAE and raw-output MAE are equal because the selected
forecasts do not clip. Mean RMSE gains across the nine history/horizon cells
are only about 0.009-0.036% relative to persistence, so large errors remain.

**Real examples.** The following are saved predictions from the 20-day,
seed-17 model, chosen as an explicitly named illustration, not as the best
seed. The three predictions are made together from information ending on
Day 30. Values are EPSS scores on the 0-1 scale.

| CVE | Latest input EPSS | Actual Day 37 | Predicted Day 37 | Actual Day 44 | Predicted Day 44 | Actual Day 60 | Predicted Day 60 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CVE-2025-1094 | 0.000430 | 0.818900 | 0.000562 | 0.840240 | 0.000566 | 0.840910 | 0.000562 |
| CVE-2025-27363 | 0.053690 | 0.094880 | 0.053821 | 0.082830 | 0.053827 | 0.754640 | 0.053821 |
| CVE-2025-24989 | 0.257230 | 0.234420 | 0.257359 | 0.234420 | 0.257369 | 0.077020 | 0.257359 |

These examples show missed increases and a missed decrease; they are not a
random or representative sample. In this same named fit, the ten largest
absolute errors account for 60.74%, 55.07% and 47.51% of total absolute error
at +7, +14 and +30 days. Their shares of squared error are 94.97%, 88.40%
and 77.74%. Each horizon ranks its own ten cases. This concentration helps
explain why small average MAE gains do not amount to capturing major changes.

**Missingness and scope.** Training has 50 partial 10-day histories, 347
partial 20-day histories and 2,254 partial 31-day histories. Validation has
no partial 10- or 20-day histories, but 867 partial 31-day histories. All
histories retain masks; missing values are not actual zero scores. Longer
history therefore adds both earlier observations and more missing slots;
this is not a comparison of equally complete histories. Five Day-37 and two
Day-44 training targets are absent and masked; none of the validation targets
is absent. The same parent cohort still requires an observed Day-60 target.

**Decision.** Keep all MH01 results as fixed-window controls. Do not call
20 days optimal, choose the easiest horizon as the research target, or claim
that multi-horizon learning has solved the earlier near-constant correction
problem. A shorter lead is easier even for persistence. The best MH01 mean
Day-60 MAE (0.010229679) is also slightly worse than the earlier single-output
GV01 EPSS-only mean (0.010223286), although their heads and selection objectives
differ, so this is not a pure architectural ablation.

The next proposed stage is the planned direct daily-trajectory task, Days
31-60, with fixed histories, masked targets and the same CVE roles. Include
persistence and a per-horizon constant-correction control fitted on training
data only: the present models' near-constant adjustments make that control
especially important. Define its protocol before training; do not select
offsets or history lengths separately for each validation CVE. Report daily
errors and large-change cases, not only an average across days. Adaptive
history selection and TPG fusion should then be separate matched comparisons.
No additional training, graph construction or test evaluation was performed
or authorized by this result review. Repeated validation use remains a
limitation; none of these results is an independent final test estimate.

### Read and Run Order

1. `04_configs/multihorizon_v1.json`: frozen scope, settings and limitations.
2. `01_package/epss/forecast_multihorizon_data.py`: source checks, windows,
   target masks, date maturity, matched coverage and preparation locks.
3. `01_package/epss/forecast_multihorizon.py`: variable-length GRU, masked
   multi-output loss, matched fits, diagnostics and checkpoint reproduction.
4. `03_analysis/01_revision_dataset/test_forecast_multihorizon.py`: regression
   tests for target isolation, exact offsets, age reset, masked gradients,
   maturity, matched persistence, optimizer parity, resume and corruption.

Prepared data goes to `../04_data/01_records_and_graphs/day30_multihorizon_v1/{smoke,full}`:
`inputs.json` contains only past histories; `labels.json` contains separate
future dates, values and masks; `split_manifest.json` contains development
roles only. `coverage.json` and `summary.txt` explain counts and limitations.

Results go to `../05_results/02_evaluation/epss_day30_multihorizon/history_controls_v1/{smoke,full}`:
`summary.txt` / `summary.json` cover all history/horizon/seed combinations;
`baselines.json` saves persistence. Per-fit `predictions.json` includes real
scores or null, forecasts, raw forecasts, corrections, anchor/target dates
and masks. `history.json` records losses, per-horizon errors, gradients and
clipping; `metrics.json` contains selected-checkpoint metrics. Checkpoints,
orders and locks permit reproduction. Verification re-scores validation,
without retraining or test scoring.

Run from the repository root with `CodeBERTFusion` active, one at a time:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_multihorizon.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_multihorizon.py --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_multihorizon.py --scale full
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_multihorizon.py --scale full --verify-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_multihorizon.py --scale full --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_multihorizon.py --scale full
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_multihorizon.py --scale full --verify-only
```

Smoke preparation and training are already saved. On a fresh checkout without
those artifacts, first run `prepare_forecast_multihorizon.py --scale smoke`
and `run_forecast_multihorizon.py --scale smoke`. Existing preparation refuses
overwrite; use `--verify-only` after a successful export. Interrupted training
uses `run_forecast_multihorizon.py --scale full --resume`: completed fits stay
unchanged and an incomplete fit restarts from fixed initialization. Keep the
default `--threads 4` for exact reproduction. No full training has been run
by the implementation step.

### Remaining Stages

After reviewing MH01, a separate protocol can add daily targets (Days 31-60)
and compare direct multi-step forecasts. Adaptive history gates can then be
compared against the fixed-window controls; weights are not proof of minimum
necessary history. Rolling forecasts require new prediction anchors and
fresh as-of feature checks while retaining CVE partition roles and target
maturity. TPG fusion must be a matched, source-aware extension, not an input
mixed into this EPSS-only baseline. None of these later stages is claimed
implemented here. Repeated validation development and unverified daily score
release times remain limitations throughout.

## GB01: Group-Balanced MAE

The question is whether giving changing training cases more influence helps
the model learn CVE-specific corrections. No CVEs are removed, duplicated or
resampled. The histories, source masks, embeddings, graph edges, batch orders,
initial weights, gain 1, Adam, decay, clipping and validation selection remain
the same as GV01. Saved GV01 controls are verified and reused, not retrained.

### Full Results Review

The six full fits completed. Each epoch still used all 2,519 training CVEs.
Early stopping ended fits after 16-42 epochs; stopping before the maximum
100 epochs is the specified patience rule, not a reduction of the cohort.

| Model | Mean ordinary validation MAE | Mean validation RMSE |
|---|---:|---:|
| Persistence | 0.0102523732 | 0.0603332762 |
| GV01 EPSS-only | 0.0102232855 | 0.0603226511 |
| GB01 EPSS-only | 0.0102259255 | 0.0603169686 |
| GV01 all-source fusion | 0.0102243637 | 0.0603202013 |
| GB01 all-source fusion | 0.0102931307 | 0.0602854461 |

Means average the three seeds' metrics, not ensemble predictions. All six
GB01 fits have higher ordinary MAE than their matched GV01 controls. All
three fusion fits also have higher MAE than persistence. GB01 EPSS-only
still improves MAE over persistence by 0.258%, but GV01 was better. Fusion
MAE is 0.673% worse than GV01 fusion and 0.398% worse than persistence.
Fusion RMSE is 0.079% lower than persistence: a small improvement on that
secondary metric, not grounds to replace the predefined MAE criterion.

| Validation endpoint group | CVEs | Persistence MAE | GB01 EPSS-only mean MAE | GB01 fusion mean MAE |
|---|---:|---:|---:|---:|
| Increased | 919 | 0.01150000 | 0.01144580 | 0.01149019 |
| Decreased | 179 | 0.00419061 | 0.00430204 | 0.00447767 |
| Unchanged | 6 | approximately 0 | 0.00011145 | 0.00043861 |

Both models modestly improve the increasing group on average but worsen
decreasing and unchanged examples. Fusion error for decreasing examples is
6.85% higher than persistence despite increased training weight for decreases.
The six unchanged validation cases are too few for a broad group-level claim.

EPSS-only corrections remain almost constant, approximately +0.000095 to
+0.000128 depending on seed. Fusion corrections now vary within the same
source-availability group; seeds 17/29 also have nonuniform source attention.
This shows less constant behavior, not proof of useful text or graph-edge
learning. Attention alone is not an explanation of predictive importance.

Selected fusion checkpoints are epochs 1, 2 and 14, with 3, 0 and 7 of 1,104
validation predictions clipped below zero. EPSS-only selected checkpoints
clip none. Other epochs are much less stable: the highest validation clipping
fraction across all fits is 90.67%. These are separate epoch evaluations,
not additional unique CVEs. No selected checkpoint has tanh saturation.

For the fixed illustrative fusion seed 29, CVE-2025-1094 is predicted at
0.00061394 versus true Day-60 EPSS 0.84091; CVE-2025-27363 is predicted at
0.05393811 versus 0.75464. CVE-2025-24989 declines from the latest input
0.25723001 to 0.07702, but is predicted at 0.25750977. The examples demonstrate
persistent large-change failures; they are not a representative sample or
a reason to exclude those CVEs.

**Decision:** retain GB01 as a negative result for the primary MAE comparison.
Reweighting alone did not solve the problem. Keep GV01 and persistence as
references; do not promote GB01 or change the selection metric after observing
these results. The experiment does not establish that imbalance is irrelevant,
that text has no signal, or that TPG cannot work. Training traces are online
losses under changing parameters/dropout, not fixed-checkpoint training scores;
they do not on their own establish overfitting. Validation has been repeatedly
used for development and is not independent final evaluation.

### Training Weights

Groups compare each CVE's **original stored Day-60 score** with its latest
observed **original stored Day 0-30 score**. Day-30 missingness uses the same
latest-observation fallback as persistence. Group equality is checked before
float32 conversion, so rounding cannot turn unchanged endpoints into changes.
An unchanged endpoint does not mean the intervening daily trajectory was flat.

| Group | Training CVEs | Weight per training example | Total group weight |
|---|---:|---:|---:|
| Unchanged | 2,141 | approximately 0.3922 | 839.6667 |
| Increased | 258 | approximately 3.2545 | 839.6667 |
| Decreased | 120 | approximately 6.9972 | 839.6667 |

The exact rule is `weight = N / (K * group_count)`, with `N = 2519` and
`K = 3` in full training. Average training weight is one. Each batch uses
`mean(weight * absolute_error)`, not division by that batch's weight sum.
Thus every group has equal total influence across an epoch without changing
the batches. Groups are not equally numerous and gradients need not be equal.

Weights use training outcomes only. Validation groups are calculated separately
for descriptive error reports and do not affect the weights. Neither group
labels nor weights are passed into the temporal encoder, GNN or fusion inputs.
The full run refuses to proceed if any training group is absent. The existing
smoke cohort contains only unchanged training examples, so its weights are one;
this is an integration test only. Synthetic tests cover all three groups and
nonuniform weights, including a short final batch.

### Fixed Experiment

- Same 2,519 training / 1,104 validation CVEs and Day 0-30 -> Day 60 task.
- EPSS-only and all-source TPG + EPSS; seeds 17, 29, 43: six fresh fits.
- Gain 1, learning rate 0.001, weight decay 0.0001, batch size 8.
- Up to 100 epochs, patience 15. Earliest strictly lowest **ordinary validation
  MAE** from epoch 1 selects the checkpoint, not balanced validation MAE.
- Persistence and unweighted GV01 remain the references. Report every seed.
- No new graphs, embeddings, raw-data changes, CVSS inputs or test scoring.

Reweighting changes the training objective. It can worsen ordinary MAE, does
not guarantee prediction of large changes, and does not repair the difference
between historical training and validation outcomes. There is no automatic
promotion to test evaluation. Text availability remains retrospective where
historical content has not been authenticated.

### Files and Outputs

- CLI: `03_analysis/01_revision_dataset/run_forecast_group_balanced.py`.
- Implementation: `01_package/epss/forecast_group_balanced.py`.
- Protocol: `04_configs/group_balanced_validation_v1.json`.
- Tests: `03_analysis/01_revision_dataset/test_forecast_group_balanced.py`.
- Results: `../05_results/02_evaluation/epss_day30_to_day60/group_balanced_validation_v1/{smoke,full}`.
- `weighting.json`: exact training counts/weights and train/validation group
  assignments. This is audit metadata, not a model-input file.
- `history.json`: weighted and ordinary online training MAE, group errors,
  ordinary validation MAE, validation group errors, gradients, clipping and
  correction variability. Training online errors include dropout and parameter
  changes; they are not fixed-checkpoint evaluation errors.
- `metrics.json` and `predictions.json`: selected-checkpoint scores and actual
  versus predicted Day-60 EPSS for every validation CVE, including raw outputs,
  persistence, corrections and source attention.
- `comparison.json`: per-CVE differences against GV01/persistence and GV01
  group-level metrics. Negative differences favor the new model.
- `summary.txt` / `summary.json`: all fits and paired comparisons. Missing
  group metrics are `null`, never zero. Raw/clipped errors remain distinct.
- Checkpoints, batch schedules and locks support deterministic reproduction.

### Commands

Run from the repository root with `CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_group_balanced.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_group_balanced.py --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_group_balanced.py --scale full --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_group_balanced.py --scale full
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_group_balanced.py --scale full --verify-only
```

The real smoke fits are already saved. A fresh checkout without the artifacts
must first run `--scale smoke`. Full training requires successful smoke
verification. Defaults remain `--threads 4 --cache-gib 12`; use the same settings
for reproduction. Use `--scale full --resume` only after interruption: completed
fits are kept and verified, while an interrupted fit restarts. Prerequisite
verification reproduces MS06/GV01 predictions before any new training begins.

The full GB01 run is now complete. On this workspace use `--scale full
--verify-only` for reproduction; do not restart training or resume a completed run.

## GV01: Fixed-Cohort Gain Validation

This is the next fixed-task comparison, not an adaptive-window experiment.
It keeps MS06's Day 0-30 input histories, observed Day-60 targets, Day-30 source
masks, frozen no-COREF TPG cache, missing-input masks and cohort assignments.
It does not add current CVSS, regenerate embeddings, open the test set or
overwrite prior results. Text remains retrospective where historical
availability has not been authenticated.

| Setting | Fixed choice |
|---|---|
| Full cohort | 2,519 training / 1,104 validation CVEs |
| Models | EPSS history only; description + summaries + SMP TPGs with EPSS |
| Change | Gain 1 inside `tanh(gain * head_output)` |
| Original controls | Locked MS06 fits, matched by variant and seed |
| Full fits | Six: two models x seeds 17, 29, 43 |
| Loss and optimizer | MAE; Adam, learning rate 0.001, weight decay 0.0001 |
| Budget | Up to 100 epochs; patience 15; batch size 8 |
| Checkpoint selection | Earliest strictly lowest validation MAE from epoch 1 |
| Initialization and schedules | Fresh matched initial weights and unchanged MS06 batch orders |
| Persistence | Latest observed input EPSS, including fallback when Day 30 is missing |
| Smoke | Existing 16 training / 8 validation CVEs; seed 17; three epochs |

Clipping and MAE selection deliberately remain unchanged, so this isolates the
gain change. There is no epoch-zero selection, gain sweep, seed selection or
automatic move to test evaluation. A lower selected validation error is still
development evidence, not independent final performance. The smoke cohort has
zero training objective at persistence; it checks integration, not learning
ability or gain selection. GD01 separately tested changing training examples.

### Files and Diagnostics

- Entry point: `03_analysis/01_revision_dataset/run_forecast_gain_validation.py`.
- Implementation: `01_package/epss/forecast_gain_validation.py`.
- Frozen comparison: `04_configs/gain_validation_v1.json`, inheriting MS06 settings.
- Regression tests: `03_analysis/01_revision_dataset/test_forecast_gain_validation.py`.
- Results: `../05_results/02_evaluation/epss_day30_to_day60/gain_validation_v1/{smoke,full}`.
- `summary.txt` / `summary.json`: all seeds, MAE/RMSE, persistence, paired model
  differences, original-control differences, and source coverage.
- Each fit's `history.json`: online training raw/clipped MAE, gradient sums,
  clipping/saturation counts, validation MAE and validation clipping every epoch.
- `predictions.json`: each validation CVE's true score, prediction, raw prediction,
  correction, persistence, source mask and source attention.
- `comparison.json`: each CVE's absolute error and differences from the matching
  original MS06 prediction and persistence. Negative differences favor gain 1.
- `metrics.json`: selected checkpoint metrics and clipping diagnostics. Exact
  boundary predictions are counted separately from out-of-range raw predictions.
- Checkpoints, batch schedules and artifact locks support reproduction. Verification
  re-scores saved checkpoints on validation, without retraining or test scoring.

Raw MAE uses predictions before clipping to [0,1]. Large raw MAE alongside low
clipped MAE warns that clipping hides large excursions. Training counts are
online exposures under dropout; validation counts are a fixed evaluation at
the end of each epoch. They are not directly comparable learning curves.

### Commands

Run from the repository root in `CodeBERTFusion`, one command at a time:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_gain_validation.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_gain_validation.py --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_gain_validation.py --scale full --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_gain_validation.py --scale full
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_gain_validation.py --scale full --verify-only
```

The smoke run is already saved. On a fresh checkout without that artifact,
run `--scale smoke` before its verification. Full training requires a verified
smoke. Keep runtime settings consistent (defaults: `--threads 4 --cache-gib 12`).
If interrupted, use `--scale full --resume`: completed fits are verified and
retained; an interrupted fit restarts from its fixed initialization. Never use
resume for a completed run. Prerequisite/control checks can take time before
the training bars appear; each major stage reports progress.

Judge the full comparison by all seeds' paired validation error against both
original MS06 and persistence, together with raw error, clipping, seed variation
and source availability. Do not conclude that graph edges are useful merely
because the all-source model fits better; that requires a separate matched
edge-removal experiment. Do not add adaptive histories/horizons in this run.

## GD01: Instrumented Training-Only Gain Diagnostic

The purpose is to distinguish failure to fit training examples from failure
to generalize. The original MS06 code, configurations, graphs and checkpoints
are unchanged. This runner constructs fresh copies of the frozen model and
changes only its non-parameter gain multiplier after initialization. Gain 1
means `tanh(head)` instead of `tanh(0.00143 * head)`; it does not remove the
tanh or output clipping. The original gain is not a hard limit on corrections.

### Fixed Scope

| Setting | Smoke | Complete diagnostic |
|---|---|---|
| Training CVEs | 9: three per movement group | 96: 32 per movement group |
| Movement groups | Unchanged, increased, decreased | Same |
| Models | EPSS-only; all-source TPG + EPSS | Same |
| Gain conditions | Frozen MS06 gain; gain 1 | Same |
| Seeds | 17 | 17, 29, 43 |
| Fixed training epochs | 3 | 30 |
| Fits | 4 | 12 |
| Checkpoint selection | Lowest subset training MAE, including epoch zero | Same |
| Validation/test scoring | None | None |

Selection uses only original training CVEs and their training outcomes.
Changing examples span their observed change magnitudes; unchanged examples
use deterministic hashed IDs. **This sample is intentionally not representative
of the full population.** Its training errors cannot be compared as validation
performance or used to claim that gain 1 improves forecasting. All full-cohort
split roles remain unchanged. This stage does not automatically launch a
full-cohort model, collect data, generate graphs or regenerate embeddings.

MAE loss, Adam, original weight decay, learning rate, dropout, batching and
graph settings remain fixed. Paired gain conditions have identical initial
parameters and batch orders for each seed. Eight focused synthetic tests cover
selection, gain derivatives, skipped parameters, nonmutating graph probes,
optimizer-step parity with uninstrumented training, checkpoint reproduction,
resume and corruption detection.

### What Gets Inspected

- Full-training movement counts and history completeness, selected IDs and
  magnitudes, missing-Day-30 cases, identical selected histories and their
  empirical EPSS-only training-error floor.
- Selected graph sizes, embedding masks, source-document counts and edge-type
  coverage, with selected-ID/tensor access restrictions.
- Every batch: clipped and unclipped loss, raw head output, correction range,
  tanh derivative/saturation, prediction clipping and source attention/masks.
- Every model component: loss-gradient norms before/after gradient clipping,
  Adam's coupled decay-term norms, ratios, gradient/decay cosine, missing
  gradients, actual parameter updates and optimizer moment summaries.
- Every epoch: evaluation-mode training MAE/RMSE, increasing/decreasing/unchanged
  group errors, correction variation, parameter norms and near-zero counts.
- Best-training and final predictions/checkpoints, with exact reload checks.
- Training-only branch probes: swap EPSS histories while restoring recipient
  persistence; swap graphs; remove text; remove edges; remove each text source.
  The donor map, per-CVE predictions, mask changes and error/movement summaries
  are saved. These are artificial diagnostic interventions, not causal evidence.

A zero gradient in the encoder on the very first step can be expected because
the last prediction layer starts at zero. Gradient ratios are null when the
data gradient is zero; a separate flag distinguishes nonzero decay in that
case. Parameters with no gradient are skipped by Adam. The data gradient is
clipped before Adam adds its decay term. These details matter when interpreting
the trace; a large ratio alone does not prove the cause of poor learning.

### Files And Commands

Implementation: `01_package/epss/forecast_gain_diagnostic.py`.
Protocol: `04_configs/gain_learning_diagnostic_v1.json`.
CLI: `03_analysis/01_revision_dataset/diagnose_forecast_gain_learning.py`.
Tests: `03_analysis/01_revision_dataset/test_forecast_gain_diagnostic.py`.

Outputs are under
`05_results/02_evaluation/epss_day30_to_day60/gain_learning_diagnostic_v1/{smoke,diagnostic}/`.

| Artifact | Contents |
|---|---|
| `summary.txt`, `summary.json` | Comparison and interpretation guide; explicitly training-only |
| `run.log` | Major events, selected batch diagnostics, epochs, failures and verification |
| `cohort_audit.json` | Training selection, movement and missingness evidence |
| `graph_audit.json` | Selected-CVE graph/source/edge statistics |
| `training_diagnostics.png` | Training error and correction variation against completed epochs |
| `run_identity.json`, `diagnostic_lock.json` | Fixed protocol, selected IDs, source hashes and output hashes |
| `fits/<model>__<gain>__<seed>/steps.jsonl` | Every batch's losses, gradients, decay, updates and source weights |
| `history.json` in each fit | Per-epoch training metrics and parameter norms |
| `predictions.json` in each fit | Actual Day-60 scores and best/final training predictions |
| `interventions.json` in each fit | Branch-probe predictions, donor map and measured responses |
| `checkpoint.pt` in each fit | Initial, best-training and final model states |
| `orders.json`, `fit_lock.json` in each fit | Deterministic batch schedules and artifact integrity |

From the repository root, in `CodeBERTFusion`:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/diagnose_forecast_gain_learning.py --mode diagnostic --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/diagnose_forecast_gain_learning.py --mode diagnostic --log-every-batches 1
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/diagnose_forecast_gain_learning.py --mode diagnostic --verify-only
```

Every batch is saved regardless of console frequency. Use the default
`--log-every-batches 10` for fewer console lines. The standard execution uses
four Torch threads and a 12 GiB tensor cache. To recover an interrupted run,
add `--resume` with the same execution settings; completed fits are verified
and reused, while an incomplete fit restarts from its fixed seed. Completed
runs require `--verify-only`. Verification reproduces checkpoints and probes
and checks trace hashes; it does not retrain to replay historical gradients.

## SF01/SF02: Structured Availability and Current CVSS

### What The Timing Audit Found

An ID connects a CVSS vector to the correct CVE. It does **not** establish
that the vector was available when the CVE was 30 days old. Both conditions
matter for a historical forecast. A current-value study is still useful, but
answers a different question: does today's saved CVSS add information in this
dataset?

| Development-only check | Training | Validation | Total |
|---|---:|---:|---:|
| CVEs kept in the comparison | 2,519 | 1,104 | 3,623 |
| Parseable current CVSS vector and score | 2,511 | 1,090 | 3,601 |
| Missing current CVSS vector | 8 | 14 | 22 |
| CVSS availability independently verified by Day 30 | 0 | 0 | 0 |
| Saved NVD update later than Day 30 | 2,519 | 1,104 | 3,623 |

There are 3,518 v3.1 vectors and 83 v3.0 vectors. The dataset stores the
version as a JSON number; the new parser handles both numeric and text forms
and checks agreement with the vector prefix. Missing vectors stay masked;
no CVE is dropped to improve coverage. These checks validate syntax and score
range, not an independent recomputation or historical verification of CVSS.

For example, training record **CVE-2013-10001** has Day 30 on **2022-06-16**,
but its saved NVD update is **2026-06-16T23:50:30.143**. The saved vector is
`CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:H/I:N/A:N` and its score is 5.9.
The later update does not prove that this vector changed after Day 30. It
also does not prove that this value was already available by then.

The collector selects metrics from a current NVD response. It records the
record-level NVD update time, but no per-metric availability time. Its
column-wise fallback can retain older collected values when new values are
null. `orig_*` values therefore must not be interpreted as publication-time
snapshots. Historical admission would need separately verified historical
values and their evidence, not a blanket assumption that CVSS is static.

The 9,344 saved URL-history objects describe current links using `url`,
`earliest_added_date`, `last_modified_url_date` and `match_mode`. They do not
contain the complete removal/re-addition history required to reconstruct a
historical reference count. VulnCheck-derived fields are also excluded from
the structured input, including their missingness. The only CWE field in
these records is `vckev_cwes`; it is not substituted for an independently
dated CWE annotation. Full-window EPSS version-change flags can describe
events after Day 30, so they are excluded too.

### New Model Comparison

SF02 is a **retrospective sensitivity experiment**, not a validated
historical forecast. It is separate from MS06 and does not alter its results.

1. All accepted description, summary and SMP TPGs + Days 0-30 EPSS; CVSS
   branch and its masks disabled.
2. The same inputs and architecture, with current CVSS enabled.

The structured view has **27 inputs**: 22 one-hot base-vector categories,
two version indicators, score divided by ten, and two observed-value masks.
These encode attack vector, attack complexity, privileges required, user
interaction, scope, and confidentiality/integrity/availability impacts.
An eight-dimensional MLP representation is concatenated with the temporal
GRU representation, pooled TPG representation, source masks and CVSS masks.
The residual head predicts a correction to the latest observed past EPSS;
the final score is clipped to [0, 1]. TPG edge types remain GNN inputs, not
CVSS/tabular categories.

Both variants have the same parameter layout, starting weights per seed,
CVEs, batch order and source masks. The control keeps the extra CVSS inputs
at zero. Compare these **new matched fits** to each other; the head is larger
than MS06's, so an older MS06 fit is not an exact architecture control.

The target remains observed Day-60 EPSS. Settings remain MAE loss, Adam,
learning rate 0.001, weight decay 0.0001, batch size 8, maximum 100 epochs,
patience 15, and seeds **17, 29, 43**, giving **six full fits**. Persistence
is reported on the same validation CVEs. Final test scoring is disabled.
These outputs are score predictions, not exploitation probabilities or
binary classification metrics.

The small real run uses 16 training / 8 validation CVEs and seed 17. Its
training endpoints all equal persistence, so zero training MAE and identical
validation MAE (about 0.00030875) only verify integration. Synthetic tests
separately verify gradient flow through CVSS, temporal and graph branches.
The smoke result is not evidence that CVSS helps or fails to help.

### Files And Commands

- `01_package/epss/forecast_structured_features.py`: timing audit, allowlist,
  fixed CVSS encoding and historical/retrospective policy gate.
- `01_package/epss/forecast_structured_model.py`: CVSS encoder and matched
  source-aware late fusion, using the existing graph layers and caches.
- `01_package/epss/forecast_structured_training.py`: fixed comparison,
  checkpoint reproduction, progress bars and interrupted-run resume.
- `04_configs/structured_day60_training_v1.json`: experiment meaning and
  limitations; numerical training settings inherit frozen MS06 settings.
- `03_analysis/01_revision_dataset/test_forecast_structured_features.py`:
  20 synthetic tests, including numeric versions, masks, timing rejection,
  matched initialization, gradients, six-fit lifecycle and corrupted outputs.

Audit files are in
`05_results/02_evaluation/epss_day30_to_day60/structured_availability_v1/{smoke,full}/`:
`summary.txt` and `summary.json` explain the findings, `cve_audit.json`
contains development-only timing evidence, and `retrospective_inputs.json`
contains only the fixed numeric allowlist. The latter is not a historical
feature export. Source hashes and output locks make the audit reproducible.

Training outputs use
`05_results/02_evaluation/epss_day30_to_day60/retrospective_cvss_v1/{smoke,full}/`.
Each fit saves a checkpoint, epoch history, metrics and validation predictions
with `actual_day60_epss`, `prediction` and the persistence reference. Summaries
state the retrospective limitation and give paired CVSS-minus-control MAE.

From the repository root, in `CodeBERTFusion`, run these individually:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/audit_forecast_structured_features.py --scale full --verify-only

python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_structured_day60.py --scale smoke --structured-policy retrospective --allow-retrospective-structured --verify-only

python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_structured_day60.py --scale full --structured-policy retrospective --allow-retrospective-structured --check-only

python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_structured_day60.py --scale full --structured-policy retrospective --allow-retrospective-structured

python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_structured_day60.py --scale full --structured-policy retrospective --allow-retrospective-structured --verify-only
```

For an interrupted full run, use the full-training command with `--resume`.
Completed runs require `--verify-only`, not `--resume`. Defaults are four CPU
threads and a 12 GiB tensor cache; keep these settings identical when verifying
or resuming the run. No parser, embeddings or graph expansion is needed.
Omitting the retrospective opt-in blocks the run before model/data loading.
The historical mode intentionally remains unavailable until trustworthy
historical structured values can be supplied and reviewed.

## MS05: Day-30-Aware Graph Cache

The new cache uses the **MS04 Day-30 source masks**, not the older Day-60
masks. It preserves every accepted document, its source type, its character
offsets and its chunk boundaries. It uses the frozen v4 security extractor,
spaCy resources, SecBERT embeddings, type vocabulary and control-token policy.
Neither EPSS values nor targets are passed to the graph builder.

| Full-cache property | Count |
|---|---:|
| Training CVEs | 2,519 |
| Validation CVEs | 1,104 |
| Source-document associations | 10,058 |
| Graph chunks | 10,131 |
| Existing chunks reused unchanged | 10,110 |
| Newly built chunks | 21 |
| Nodes | 2,034,752 |
| Nodes with embeddings | 1,942,321 |
| Nodes without embeddings | 92,431 |
| Edges in the no-COREF view | 6,323,038 |
| Security edges, included in that edge total | 3,688 |

Missing embeddings follow the frozen structural/whitespace/replacement/control
mask policy; missing ordinary-token embeddings still fail validation. These
counts describe stored representations, not semantic extraction accuracy.

Reuse requires the exact CVE, split, source type, document identity, chunk
text hash, parent-text hash, offsets and receipt file set. Every referenced
artifact must match the original completed cache lock. The old code/resource
identities are also checked. A reused graph stays in the old cache: it is not
copied, relabelled, edited or re-encoded. **Do not delete the CD03 v2 cache.**

The new cache contains source documents and indexes with explicit absolute
graph/tensor locations. The 21 new chunks are stored locally. For every model,
the new index determines which sources are allowed; retaining an older graph
file on disk does not make it an allowed input. Separate chunks have no invented
cross-chunk relations. The fixed training edge configuration is `no_coref`.

Cache locations:
`EPSS_TPG/01_tpg/05_workspace/day30_multisource_v1/{smoke,full}/`.

| File | What it provides |
|---|---|
| `cve_index.json`, `by_cve/<CVE>.json` | New Day-30 source masks, documents and absolute graph/tensor locations |
| `documents/` | Original accepted text with complete chunk coverage |
| `chunk_locations.json` | Reused/local storage roots, artifact hashes and node/edge statistics |
| `tpg_json/`, `tensors/`, `receipts/` | Only newly built chunks; reused chunks stay at their recorded original locations |
| `analysis.txt`, `analysis.json` | Coverage, counts, limitations and reuse totals |
| `run_identity.json`, `protocol.json`, `cache_lock.json` | Frozen resources, preparation identity and output hashes |

Both cache scales are already built. All 65 smoke chunks were reused and
passed deep verification, which reloads graphs and tensors. The full build
rechecked every reused file hash and validated every new graph/tensor. Normal
verification repeats these checks; `--deep-verify` additionally reloads every
reused tensor and graph. It is optional and slower, without rerunning encoders.

```bash
python EPSS_TPG/01_tpg/02_examples/01_scripts/prepare_day30_multisource_graphs.py --scale smoke --verify-only
python EPSS_TPG/01_tpg/02_examples/01_scripts/prepare_day30_multisource_graphs.py --scale full --verify-only
```

Construction commands, for an environment where these outputs do not yet exist:

```bash
python EPSS_TPG/01_tpg/02_examples/01_scripts/prepare_day30_multisource_graphs.py --scale full --plan-only
python EPSS_TPG/01_tpg/02_examples/01_scripts/prepare_day30_multisource_graphs.py --scale smoke --workers 4
python EPSS_TPG/01_tpg/02_examples/01_scripts/prepare_day30_multisource_graphs.py --scale smoke --verify-only --deep-verify
python EPSS_TPG/01_tpg/02_examples/01_scripts/prepare_day30_multisource_graphs.py --scale full --workers 4
```

Use `--resume` only for an incomplete build. Completed outputs reject creation
and resume. New chunks use independent CPU workers and local model resources;
no download is initiated. Reuse validation and important stages show progress.

## MS06: Matched Day-60 Training

### Fixed Experiment

**Days 0-30 EPSS -> observed Day-60 EPSS**, using the same **2,519 training and
1,104 validation CVEs** for all configurations. Missing sources and partial
EPSS histories remain explicit rather than causing model-specific CVE removal.
The 651 reserved test identities are not exported or evaluated.

| Configuration | Past EPSS | Graph inputs | Output |
|---|---|---|---|
| Persistence reference | Latest observed score in Days 0-30 | None | Repeat that score for Day 60 |
| `epss_only` | All observed Days 0-30 with missingness and age masks | None | Predicted Day-60 EPSS |
| `description_epss` | Identical history | Description TPGs only | Predicted Day-60 EPSS |
| `all_sources_epss` | Identical history | Description, GitHub/non-GitHub summaries and accepted SMP TPGs | Predicted Day-60 EPSS |

The tested Day-30 loader produces 31 time steps with five channels. No Day-31
or later score enters that sequence. The temporal encoder is the existing
16-unit GRU. Text uses the existing 32-unit, two-layer typed-edge GNN, then
mean pooling within chunks/documents/sources and masked source attention.
The model concatenates temporal state, fused text and source masks, and learns
a bounded correction to persistence. The target is separate from those inputs.
The residual gain uses only latest observed scores from the training inputs.

There are **nine fresh full fits**: three configurations times seeds **17, 29,
43**. The fixed settings remain MAE loss, Adam, learning rate `0.001`, weight
decay `0.0001`, batch size 8, dropout 0.1, gradient clipping at 1, at most 100
epochs and patience 15. Checkpoints use the earliest strictly best validation
MAE from epoch 1. Epoch zero is not a candidate; persistence remains separately
reported. All seeds are retained rather than selecting a winning seed.

The complete initial state and batch ordering are matched across configurations
within each seed. EPSS-only does not load graph tensors in its forward pass.
Description-only does not visit summary or SMP graphs. No absent source receives
attention. Read-only tensor caches share a bounded memory budget across the
original and new cache roots. Test-role and unselected tensor requests fail.

### Verification and Smoke Results

Thirteen new pipeline tests pass, together with 15 MS04 tests and 14 existing
MS03 tests. Coverage includes target separation, missing-Day-30 persistence,
training-only gain, exact reuse identity, corruption detection, forbidden graph
access, resume recovery, checkpoint reloads and the full nine-fit lifecycle on
synthetic data. For all three variants, the new training loop produces exactly
the same synthetic histories and state tensors as the frozen MS03 loop when
given identical tensors and labels. The old implementation was not edited.

The real smoke uses 16 training and eight validation CVEs, seed 17 and three
epochs. All three checkpoints and their validation predictions reproduce.
Their validation MAE equals persistence at `0.0003087500084051862`. All 16
training targets exactly equal their latest input score; none of the eight
validation endpoints does. The zero training objective therefore does not
establish a useful learned correction. This small run checks execution, not
model selection or predictive benefit. It does not change the full protocol.

The full-data preflight also passed: it reproduced the full cache, rechecked
its referenced artifacts, verified the matched cohort and graph memory budgets,
and checked persistence initialization for all three seeds. It did not train
the full models or score the test set.

The full comparison has **not been trained**. As requested, its commands are
provided for the user to run. Completing this endpoint does not repair or
reinterpret the learned-branch collapse observed in MS03.

### Run and Inspect

Use the repository root and the `CodeBERTFusion` environment. Smoke is already
complete; verify it rather than rerunning creation. Keep `--threads 4` and
`--cache-gib 12` consistent with the completed smoke identity.

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_day60_multisource.py --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_day60_multisource.py --scale full --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_day60_multisource.py --scale full
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_day60_multisource.py --scale full --verify-only
```

If interrupted, run the following instead of the creation command. Completed
fits are verified and retained; the interrupted fit restarts from its fixed
seed rather than pretending to resume an unsaved optimizer state:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_day60_multisource.py --scale full --resume
```

The full command first reproduces all smoke checkpoints. It then trains all
nine configured fits with progress bars, without expanding the cohort or
changing hyperparameters. There is no test-evaluation CLI mode.

Results:
`EPSS_TPG/05_results/02_evaluation/epss_day30_to_day60/matched_multisource_v1/{smoke,full}/`.

| File | Meaning |
|---|---|
| `summary.txt`, `summary.json` | Per-fit regression metrics, across-seed MAE and paired model differences |
| `baselines.json` | Persistence predictions and metrics on the identical validation CVEs |
| `fits/<variant>__<seed>/predictions.json` | CVE, `actual_day60_epss`, prediction, persistence, correction, source masks and attention |
| `fits/<variant>__<seed>/metrics.json` | MAE, RMSE, signed error, Spearman correlation/status, clipping and correction variation |
| `fits/<variant>__<seed>/history.json` | Training/validation curves and data-gradient sums |
| `fits/<variant>__<seed>/checkpoint.pt` | Selected weights and configuration |
| `run_identity.json`, `orders_seed_*.json`, lock files | Inputs, settings, fixed ordering and reproducibility checks |

Regression metrics use the 0-1 EPSS scale. F1, precision and classification
accuracy are not appropriate without a separately defined classification
target. Attention is not proof of causal influence or useful text learning.
Descriptions/summaries remain retrospective; stored SMP dates are not proof
of historical content availability. Review the full Day-60 results alongside
the fixed Day-90 results before separately authorizing final held-out scoring.

## MS04: Day-60 Multi-Source Preparation

### Completed Verification

Independent execution of `--scale full --verify-only` reproduced the saved
full export and its TXT/JSON reports. The smoke export also reproduced;
explicit smoke/full checks confirmed identical histories, targets, source
routing and training registry for the selected smoke records. All 79 pinned
prerequisites and the 13 artifacts in each export's lock passed their hash
checks. The saved production exports, raw dataset and existing models were
not modified.

The preparation is ready for the Day-30-aware graph stage. This verifies
data preparation, not predictive performance or historical text availability.
No training or test scoring was performed in this review.

### Purpose and Scope

Complete the other planned regression endpoint using the current CVE-identity
policy: **EPSS from Days 0 through 30 -> observed EPSS on Day 60**. Day numbers
are measured from CVE publication. This is a 30-day-ahead prediction, not a
prediction 60 days after the input window ends. The completed MS03 experiment
instead uses Days 30-60 to predict Day 90. Earlier R07 Day-60 results used a
different, smaller cohort and description-only text; they do not substitute
for the new matched multi-source comparison.

This implementation prepares data only. It does not retrain MS03, fix its
learned-branch collapse, create embeddings or evaluate the final test set.
The planned comparison remains EPSS-only, description TPG + EPSS, and all
accepted text TPGs + EPSS, with persistence as the common reference. A new
graph-cache adapter and training runner follow this verified preparation.

### Chronological Roles and Eligibility

Moving the prediction anchor from Day 60 to Day 30 also moves the three split
boundaries 30 days earlier. This preserves the original publication cohorts
and **all 5,968 saved CVE roles**, including the reserved test records. It does
not use the same forecast cutoff dates for the two endpoints. The original
Day-90 files and boundaries remain unchanged.

| Boundary | MS03 Day-90 task | New Day-60 task |
|---|---|---|
| Validation starts | 2025-04-01 | 2025-03-02 |
| Test starts | 2025-07-01 | 2025-06-01 |
| Test ends, exclusive | 2026-01-01 | 2025-12-02 |
| Dataset observation cutoff, unchanged | 2026-09-29 | 2026-09-29 |

Training targets must precede the validation boundary; validation targets
must precede the test boundary. Boundary-crossing targets remain excluded.
The shift is determined by the anchor change, not by prediction performance.
A metadata-only check showed that blindly reusing the April/July boundaries
would leave only 21 CVEs assigned to testing; that alternative is not adopted.

| Role | Assigned CVEs | Eligible development CVEs | Eligible CVEs with some missing Day 0-30 scores |
|---|---:|---:|---:|
| Training | 2,870 | 2,519 | 2,254 |
| Validation | 1,104 | 1,104 | 867 |
| Reserved test | 651 | Not inspected for this task | Not inspected |
| Previously excluded by split policy | 1,343 | Not reconsidered | Not inspected |

These counts reproduce from the user's completed production exports.
Eligibility requires at least one observed Day 0-30 input score and an
observed Day-60 target. Partial inputs are retained with masks; targets are
never filled. No Day-90 score is required. Among assigned training CVEs, 324
lack the target and 343 lack all input scores; these reasons overlap and
exclude 351 CVEs in total. All assigned validation CVEs are eligible.

The Day-60 and Day-90 training cohorts share 2,512 CVEs: seven are eligible
only for Day 60 and 36 only for Day 90. All 1,104 validation identities match.
Thus future within-task comparisons are matched, but cross-task training
cohorts are not identical. The 651 reserved identities are **not a verified
Day-60 eligible test count**: their scores, text and eligibility are not
inspected or exported here.

### Text Sources and Input Contract

| Source | Training CVEs with retained text | Validation CVEs with retained text |
|---|---:|---:|
| CVE description | 2,519 | 1,104 |
| GitHub summary | 528 | 201 |
| Non-GitHub summary | 1,761 | 749 |
| SMP | 1,746 | 1,019 |

Sources overlap within CVEs; do not add the rows as distinct CVE counts.
There are 1,918 retained training SMP documents and 1,278 validation SMP
documents. Posts must have consistent, parseable stored dates strictly before
Day 30. Same-day, later, missing, invalid or conflicting dates are excluded
from this view. Day-60 masks are not reused. These checks do not authenticate
the post publication dates or prove that today's post text existed then.

Nonempty descriptions and summaries are retained despite imperfect wording.
They remain retrospective evidence, not verified Day-30 snapshots. Missing
sources produce source masks, not CVE deletion. The training-text registry is
rebuilt using eligible Day-60 training CVEs; optional matching validation text
is masked. This run finds zero such retained optional-document overlaps.
Repeated descriptions are retained: shared wording does not establish that
two CVE IDs represent the same vulnerability.

The tested sequence adapter produces 31 rows with five values per row:
score-or-zero, observation mask, observation age divided by 60, age-known mask,
and day offset divided by 60. Zero placeholders are distinguished from observed
zero scores by the observation mask. The normalization denominator matches the
existing model convention; it does not include future days. Inputs contain
only Days 0-30. Day-60 targets are stored separately and never passed to this
adapter. Changes after Day 30 cannot change the input sequence.

### Files and Commands

Implementation: `01_package/epss/forecast_day60_multisource.py`.
Protocol: `04_configs/day60_multisource_preparation_v1.json`.
CLI: `03_analysis/01_revision_dataset/prepare_forecast_day60_multisource.py`.
Completed outputs:
`EPSS_TPG/04_data/01_records_and_graphs/cve_identity_day30_to_day60_v1/{smoke,full}/`.

| File | Purpose |
|---|---|
| `inputs.json` | Past EPSS values, dates and missingness/age masks |
| `labels.json` | Separate observed Day-60 targets |
| `source_inputs.json` | Accepted text, source types and source masks |
| `source_date_checks.json` | Stored-date evidence and screening decisions; not model input |
| `source_decisions.json` | Optional validation-text overlap decisions |
| `training_text_registry.json` | Text identities from eligible training records |
| `retained_validation_text_registry.json` | Retained validation text identities for later test screening |
| `split_manifest.json` | All CVE dates and reserved roles, without test scores or text |
| `development_eligibility.json` | Input/target availability for assigned development records |
| `protocol.json`, `run_identity.json`, `preparation_lock.json` | Rules, selected IDs and artifact hashes |
| `analysis.txt`, `analysis.json` | Eligibility, source coverage, comparison counts and limitations |

Run from the repository root in the `CodeBERTFusion` environment, in order:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_day60_multisource.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_day60_multisource.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_day60_multisource.py --scale smoke
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_day60_multisource.py --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_day60_multisource.py --scale full
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_day60_multisource.py --scale full --verify-only
```

Smoke selects 16 training and eight validation CVEs deterministically while
using the full eligible-training registry. Full creation requires the matching
saved smoke export. Existing outputs cannot be overwritten; use `--verify-only`
for reproduction. Important stages show progress bars. Verification checks
upstream hashes and reproduces all exported JSON and the TXT report.

The next graph stage may reuse existing chunks only when text and graph/encoder
resource identities match exactly. New Day-30 masks and source assignments
must still be used. Do not pass these exports to the existing Day-90 training
CLI. The separately versioned Day-60 runner will use matched configurations
and seeds, followed by review and a separately authorized frozen test evaluator
for both endpoints. No new optimizer search is implemented here.

## MS03: Matched Multi-Source Training

### Completed Results and Interpretation

The user's nine-fit full run is complete. Independent execution of the original
`--scale full --verify-only` command reproduced all saved full and prerequisite
smoke checkpoint predictions and checked all 69,828 pinned inputs. No retraining
or test scoring was performed. The original training code, configurations,
graphs, checkpoints, predictions and locks remain unchanged.

| Model | Mean validation MAE across three seeds | Sample standard deviation across seeds |
|---|---:|---:|
| Persistence reference | 0.004517980106 | Not fitted |
| EPSS history only | 0.004517370091 | 0.000000292877 |
| Description TPG + EPSS | 0.004517533436 | 0.000000050063 |
| Description, summaries and SMP TPGs + EPSS | 0.004517509546 | 0.000000070109 |

MAE is mean absolute prediction error on the 0-1 EPSS scale; lower is better.
These are means of separate seed results, not ensemble predictions. Every fit
improves on persistence numerically, but the relative MAE reduction is only
0.00856%-0.02084%. EPSS-only has the lowest mean MAE. Adding description helps
in two of three paired seeds, but the differences are tiny and its mean is
worse. All-source text beats EPSS-only in one seed and loses in two. No
statistical significance, practical forecasting benefit or causality follows
from these differences. Each fit stopped early: 20-45 epochs, with selected
checkpoints at epochs 5-30.

The model behavior is more informative than ranking these nearly equal scores:

- All three EPSS-only fits, all three description fits, and all-source seed 29
  make exactly one learned correction for every validation CVE.
- All-source seeds 17 and 43 produce eight correction values each. Within each
  of the eight source-presence patterns, the correction is exactly constant.
  Differences between distinct CVE texts therefore produce no visible
  correction variation within those groups at saved numerical precision.
- All-source attention equals uniform weighting over the available sources.
  Its average values across sources reflect source coverage, not evidence that
  the model learned which text is more useful.
- Learned corrections range from approximately 0.000000513 to 0.000001249.
  The latest observed EPSS, supplied directly by the persistence connection,
  accounts for practically all of the predictions' variation.
- Saved GRU weight norms are at most 1.28e-12; active node-projection norms
  are at most 9.70e-12. Their initial norms were approximately 4.8 and 3.3.
  Selected-epoch data gradients are correspondingly tiny. Unused graph modules
  in EPSS-only remain at initialization, as intended; they are not evidence of
  an active text branch.

These checks support learned-branch collapse in this experiment. They do not
isolate whether weight decay, the small residual scale (0.00166), training
target distribution or their interaction caused it. Saved attention and
within-mask observations are not counterfactual interventions on text.

There is also a substantial target-distribution difference:

| Split | CVEs | Day-90 score equals latest input score | Day-90 score is higher | Day-90 score is lower | Partial Day 30-60 histories |
|---|---:|---:|---:|---:|---:|
| Training | 2,548 | 2,079 (81.59%) | 327 | 142 | 342 |
| Validation | 1,104 | 18 (1.63%) | 968 | 118 | 0 |

These counts compare original stored scores, not float32 rounding differences.
An unchanged endpoint does not imply that the entire Day 60-90 path was flat.
Training median absolute endpoint change is zero; validation median absolute
change is 0.00008. All validation histories are complete, so missing validation
input scores cannot explain this result. This does not rule out an effect of
missing training inputs. The largest ten validation persistence errors account
for 47.29% of total absolute error, not the 94.99% two-case concentration of the
earlier, different H02 cohort.

Examples of substantial observed changes that the nearly persistent models
do not capture well:

| CVE | Latest observed input EPSS | Actual Day-90 EPSS | Absolute change |
|---|---:|---:|---:|
| CVE-2024-52875 | 0.21922 | 0.60436 | 0.38514 |
| CVE-2025-21418 | 0.44574 | 0.07435 | 0.37139 |
| CVE-2025-24016 | 0.16746 | 0.49431 | 0.32685 |

**Decision:** multi-source training is operational, but useful learned text
contribution is not demonstrated. This does not establish that TPGs or summaries
are inherently unhelpful, nor prove a graph-extraction bug. Avoid another
unchanged full run or using the final test set to choose settings. A bounded
training-only capacity check on CVEs with changing targets can establish
whether matched EPSS-only/text models can learn nonconstant corrections;
inspect gradient scale relative to regularization before any new full protocol.
That diagnostic selection must not replace the primary cohort or be reported
as unbiased validation performance. No new training is implemented in this review.

Saved review: [TXT](../05_results/02_evaluation/epss_day60_to_day90/matched_multisource_review_v1/analysis.txt),
[JSON](../05_results/02_evaluation/epss_day60_to_day90/matched_multisource_review_v1/analysis.json),
and [learning curves and target-change figure](../05_results/02_evaluation/epss_day60_to_day90/matched_multisource_review_v1/learning_and_target_shift.png).
The curve figure uses a narrow MAE axis to show very small differences; it must
not be read as showing large performance improvements. JSON includes all nine
fits, paired CVE errors, source-present/absent subgroups, per-epoch histories,
initial/saved parameter norms and selected-epoch gradient sums. Four synthetic
review tests pass. Reproduce this saved-artifact review without new inference:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_matched_multisource.py --verify-only
```

### Question and Fixed Comparison

Does description text improve the Day-90 EPSS forecast beyond past EPSS alone,
and do summaries and social-media posts add anything beyond the description?
All three models predict the **observed EPSS score 90 days after publication**
using EPSS from **Days 30 through 60**, inclusive: 31 possible daily scores.
This is regression, not exploitation or VulnCheck classification.

| Configuration | EPSS input | Text input | Trainable components |
|---|---|---|---|
| `epss_only` | Day 30-60 history, observation masks and ages | None; graph files are not loaded in this model's forward pass | GRU and residual prediction head |
| `description_epss` | Identical history | Description TPG only | GRU, typed-edge GNN, source fusion and residual head |
| `all_sources_epss` | Identical history | Description, GitHub summary, non-GitHub summary and accepted SMP TPGs | Same GRU/GNN/fusion/head design, with available sources enabled |
| Persistence reference | Most recent observed score in Day 30-60 | None | No fitted parameters; repeat that score at Day 90 |

The comparison uses **2,548 training and 1,104 validation CVEs**. The existing
651 eligible test CVEs remain reserved: no test histories, text, targets,
graphs or predictions enter this runner. Missing past scores keep their masks
and ages. A missing summary/post does not remove its CVE. The Day-90 target
must be observed; it is never filled in. Existing split and source decisions
remain unchanged. Distinct CVE IDs are not declared duplicate vulnerabilities
merely because their wording or links overlap.

### Model Flow

```text
Past EPSS + observation masks/ages -> GRU -----------------------------+
                                                                    |
Accepted source text -> existing frozen TPG/node-embedding cache       |
                    -> trainable typed-edge GNN for each chunk        |
                    -> mean node vectors within each chunk            |
                    -> mean chunk vectors within each document        |
                    -> mean document vectors within each source       |
                    -> attention over present sources, using GRU state |
                                                                    v
                          concatenate GRU state + text vector + masks
                                      -> learned correction
                latest observed EPSS + correction -> clip to [0, 1]
                                      -> predicted Day-90 EPSS
```

The GNN uses the saved edge-type vocabulary and edge attributes, including
security relations, directly in message passing. The fixed graph view is
`no_coref`; it does not remove security edges. The vocabulary is not replaced
by hardcoded relation indices. Node embeddings and graph construction stay
frozen; GNN weights are trained. No parser or transformer encoder rerun is
needed. Chunks are encoded independently, so relations across chunk boundaries
are not available to this model.

Pooling happens in stages so a 97-chunk post does not receive 97 times the
source weight of a one-chunk description just because it is longer. All chunks
are retained. Attention ignores absent sources; an all-missing text row has a
zero text vector. Source masks explicitly tell the model what was available.
Attention weights are diagnostics, not evidence of causality or reliable
feature attribution. Text quality and historical availability remain separate
limitations even when a graph passes structural checks.

### Source Coverage

These are the accepted cached sources used without any new filtering:

| Source | Training CVEs with that source, out of 2,548 | Validation CVEs with that source, out of 1,104 |
|---|---:|---:|
| Description | 2,548 | 1,104 |
| GitHub summary | 539 | 201 |
| Non-GitHub summary | 1,785 | 749 |
| Social-media/related posts | 1,798 | 1,054 |

The full cache has 11,121 chunks across 10,438 CVE/document associations.
Descriptions and summaries are **retrospective text**, not authenticated
Day-60 snapshots. Passing the stored SMP date rules does not authenticate when
a post or its exact wording became available. Results must therefore be
reported as retrospective development comparisons, not a proven leakage-free
historical deployment experiment. Exact optional training-text overlap is
handled by the already frozen CD02 source masks; descriptions are retained.

### Fixed Training Settings

- Seeds 17, 29 and 43, each applied to all three models: **nine full fits**.
- Fresh initialization for every fit; identical initial parameter values and
  CVE batch order within each seed. Graph modules exist in the common layout
  but are inactive, unloaded and without gradients in EPSS-only mode.
- MAE loss, Adam with learning rate 0.001 and weight decay 0.0001; CVE batch
  size 8; gradient-norm clipping at 1.0.
- GRU hidden size 16; GNN/head width 32; two GNN layers; dropout 0.1.
- Up to 100 epochs, patience 15. Select the earliest strictly best validation
  MAE checkpoint from epoch 1 onward. Epoch zero is the separate persistence
  reference, not a selectable trained checkpoint.
- The final linear layer starts at zero, so every model initially predicts
  persistence. Residual scaling is fitted from training inputs only, using
  the existing clipped-median rule; validation targets do not set it.
- Report MAE, RMSE, mean signed error and Spearman correlation, plus paired
  per-seed MAE differences and mean/sample-standard-deviation MAE across seeds.
  All seeds are retained, not only the best result.
- Save per-epoch losses, pre-optimizer data-gradient absolute sums by branch,
  output clipping counts, and per-CVE predictions, masks and attention weights.
  Gradient sums do not include Adam's weight-decay term.

The smoke run uses the fixed CD02 16-training/8-validation subset, one seed
(17), and three epochs per model. It checks integration only. Full training
requires completed, reproducible smoke checkpoints and does not tune settings
from smoke scores. The nine full fits start from scratch, not smoke weights.

Implementation checks passed: 14 focused synthetic tests, the 11 existing
CD02 preparation tests, and smoke/full preflights against the real locked
cache. Tests cover source removal, absent-source attention, document pooling,
typed-edge sensitivity, checkpointed gradients with dropout, target separation,
deterministic fits, checkpoint replay, resume and concurrent-writer exclusion.
Synthetic optimizer steps use invented fixtures only. The real preflights
check initialization on two training CVEs, with no parameter updates or test
access; they are not completed real-data smoke or full training runs.

### Commands, In Order

Run from the repository root with `CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_matched_multisource.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_matched_multisource.py --scale smoke --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_matched_multisource.py --scale smoke
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_matched_multisource.py --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_matched_multisource.py --scale full
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_matched_multisource.py --scale full --verify-only
```

Use `--scale full --resume` after an interruption. Completed fits are verified
and retained; an incomplete fit restarts from its original seed. Resume is
between fits, not from an arbitrary middle epoch. A process lock prevents
concurrent writers. Completed runs refuse overwrite; use `--verify-only`.

CPU execution defaults to four Torch threads and a 12-GiB tensor LRU cache.
This is a cache budget, **not a total-process memory limit**. Graph microbatches
contain at most 8,192 nodes and 32,768 edges; activation checkpointing reduces
training activation memory without removing graph nodes. `--threads` and
`--cache-gib` are configurable before starting smoke, but must be kept the same
for smoke, full, resume and verification because they are recorded in the run
identity. There is no full-cache tensor reconstruction at every epoch: the
runner checks locked file hashes, then reads pinned tensors lazily. Progress
bars cover file checks, fits, epochs, CVE batches and validation.

### Where the Outputs Go

`EPSS_TPG/05_results/02_evaluation/epss_day60_to_day90/matched_multisource_v1/{smoke,full}/`

| Output | Purpose |
|---|---|
| `summary.txt`, `summary.json` | Matched results, persistence, source coverage, all seeds and limitations |
| `baselines.json` | Persistence predictions and metrics for the exact same validation CVEs |
| `fits/<variant>__<seed>/checkpoint.pt` | Selected model parameters, settings and graph provenance |
| `fits/<variant>__<seed>/history.json` | Epoch losses, branch gradient sums and clipping counts |
| `fits/<variant>__<seed>/predictions.json` | Individual validation predictions, actual Day-90 scores, corrections, masks and attention |
| `fits/<variant>__<seed>/metrics.json` | Selected checkpoint's metrics and correction spread |
| `run_identity.json`, `orders_seed_*.json` | Fixed inputs, code hashes, settings, selected CVEs and batch schedules |
| `fit_lock.json`, `evaluation_lock.json` | Integrity checks for completed fits and the completed comparison |

Implementation: `01_package/epss/forecast_matched_multisource.py` is the runner;
`forecast_matched_multisource_model.py` contains the source-aware GNN/data
adapter; `04_configs/matched_multisource_training_v1.json` freezes the protocol.
The CLI and focused tests are in `03_analysis/01_revision_dataset/`.
Original prepared inputs, graph caches and earlier results are not modified.
After these runs, inspect paired performance and branch gradients before
deciding whether any final test evaluation is justified.

## Earlier Development Status

**Completed review: [CD01 bounded cohort-design review](#cd01-bounded-cohort-design-review).**
The user's completed review is independently reproduced. 5,645 of 5,968 CVEs
are data-eligible before splitting; missing histories are not the main loss.
Later-Q2 frozen grouping offers 1,998/884/342 eligible CVEs, but 334 test anchors
are in July 2025 and 1,328 records cross label-maturity boundaries. Shared
evidence groups are not verified duplicate vulnerabilities. See the completed
interpretation before freezing a revised cohort/source policy. No split is
adopted and no training or test-performance scoring occurred.

**Earlier description graph cache:** [DQ17 v2 full development cache](#dq17-matched-development-cache-and-r03-protocol).
All 1,220 development CVEs pass source, tensor, edge-mask and hash verification.
The three configurations share 125,884 nodes. No encoder rerun, model training
or test scoring is performed during this review.

**Earlier trained full-cohort run:** [H02 full matched history controls](#h02-full-matched-history-controls).
All six full checkpoints and both smoke checkpoints reproduce their validation
predictions. Neither history improves MAE over persistence. Recent and longer
history give identical final validation predictions within each seed. The
separate review records collapsed or negligible corrections and the two CVEs
responsible for 94.99% of absolute persistence error. The test set stays closed.

**Previous description-plus-EPSS run:** [R07 full eligible-cohort MAE training](#r07-full-eligible-cohort-mae-training).
All 50 artifacts and six checkpoint predictions are independently verified.
Training uses 1,023 CVEs; validation uses 197. All six fits improve on matched
R04 Smooth L1 MAE, but none beats persistence. Five learned corrections are
exactly constant across validation CVEs; the sixth is effectively constant.
Useful added TPG contribution remains unproven. The full planned training is
complete; no further bounded tuning is planned. The test set remains closed.
Historical-description availability restrictions remain in force.

**Latest diagnostic:** [DQ19: training-only gradient/decay check](#dq19-training-only-gradientdecay-check).
The bounded check is complete and reproduced. Encoder decay-term norms exceed
data-loss gradient norms on every sampled selected-checkpoint batch. This
supports a controlled regularization experiment, not a causal conclusion.
No optimizer steps, validation scoring or test scoring were performed.

The earlier [DQ14 bounded semantic review](#dq14-bounded-semantic-review-with-v4-fixed)
remains unchanged: 44 links, with 17 supported in their stated scope,
20 unsupported as interpreted and seven limited or ambiguous. These are
source-review observations, not gold labels or accuracy estimates.
**Latest verified preview: [MS02 small multi-source graph preview](#ms02-small-multi-source-graph-preview).**
The user's completed 24-CVE, 65-document cache passes all 331 artifact hashes
and graph/tensor/source-mask checks. It contains 13,604 nodes, 42,568 no-COREF
edges and 19 typed security edges. No ordinary-token embeddings are missing.
The separate TXT/JSON review explains source coverage and accounts for all
5,968 CVEs. The user has now requested CD01 before full expansion to reconsider
the restrictive cohort allocation. After an explicit cohort decision: full
multi-source cache plus a fixed fusion runner, one integration smoke, then
full training. No new optimizer-tuning loop is recommended.
No training, expansion or test scoring occurred in MS02 or its review.

**Completed temporal controls: [H02 full matched history controls](#h02-full-matched-history-controls).**
The user's full-development export, smoke and six-fit comparison are complete
and independently verified. Fourteen implementation tests and three new review
tests pass. The automated review performed no retraining. Full source coverage and
the completed-results review are documented below.

**Earlier completed preview: [H01 later-anchor preview](#h01-later-anchor-preparation).**
The user requested the recommended separate Day-60-to-Day-90 experiment.
The user's bounded preview is complete and independently reproduced.
All ten artifacts pass verification; a separate readiness review checks
cohort changes, missing observations and source masks. The matched cohort
has 1,029 training and 173 validation CVEs. The H01 preview itself exports
16 training and eight validation CVEs; the subsequent H02 export covers the
full development cohort. No H01 training occurred.
MS01 is complete and reusable; its source timing is reapplied at Day 60.
No test scores/text are exported.
R07 remains the completed Day-30-to-Day-60 description-plus-EPSS experiment.

The preceding R04 version changes only the correction response scale, derived from past
training inputs. Its user smoke run is complete and independently verified.
A full 1,023/197, three-seed runner now implements the recorded protocol,
without changing smoke results, graph caches, grouping rules or test access.
DQ18 smoke and full diagnostics are reproduced and reviewed. A separately
versioned residual-head smoke run is now complete and independently verified.
Full R04 training uses the new separate runner, not an override on the smoke
CLI. R03, DQ18 and all graph caches remain unchanged.

A separate v2 preparation CLI extends the fixed graph/embedding contract to the
existing 1,023 training and 197 validation endpoint CVEs. V1 stopped on two
tokenizer-removed replacement characters after 39 completed entries. V2 keeps
those nodes with zero embeddings and explicit masks/audits, and recovers the
verified entries without changing them. Its seven-CVE preview includes the
failure case and must pass before full preparation.
The user's v2 full preparation is complete and verified. R03's longer
three-seed protocol is implemented without changing its frozen definition.
R03 results and their separate review are now verified below.
The latest extractor correction remains DQ13:
V4 narrowly removes source-verified same-occurrence repeated_surface candidates
while preserving existing distinct-span and pronoun links. The user's 56-CVE
preview and separate review are verified. Human judgments remain pending;
the later R02 training uses only the fixed small cache. DQ17 expands the same
fixed policy to the existing development cohort without new semantic labels.

## CD03: Inspectable Multi-Source Graph Cache

**Status:** v2 full graph generation and user-run graph/tensor verification
completed. Independent hash/statistics review passed. No training has run in
this stage. CD02, MS02, the shared core, v4 extractor and old caches remain
unchanged. No test graphs or scores were accessed.

| Scope | CVEs | Source documents associated with CVEs | Planned graph chunks |
|---|---:|---:|---:|
| Smoke | 24 | 70 | 111 |
| Full development | 3,652 | 10,438 | 11,121 |

The full development cohort is 2,548 training / 1,104 validation CVEs. All
accepted description, GitHub-summary, non-GitHub-summary and SMP documents
are retained. Identical wording does not merge distinct CVE IDs. No additional
overlap filter is introduced; CD02 source masks remain fixed.

### Completed Full-Cache Findings

| Source | Training CVEs with graphs | Validation CVEs with graphs |
|---|---:|---:|
| Description | 2,548 | 1,104 |
| GitHub summary | 539 | 201 |
| Non-GitHub summary | 1,785 | 749 |
| SMP | 1,798 | 1,054 |

The 11,121 chunks contain 3,006,451 nodes, including 2,906,755 with embeddings,
and 9,517,311 no-COREF edges. Missing TOKEN embeddings are fully accounted for
in the verified statistics: 15,076 whitespace, ten replacement-character and
two allowlisted control-only tokens. The other 84,608 unembedded nodes are
non-token nodes; this count is not missing word embeddings.

There are 5,459 typed security edges across 1,683 CVEs. Of these edges, 5,128
are SEC_HAS_VERSION. The extractor therefore provides mainly product/version
structure; coverage is not semantic accuracy or evidence of forecasting value.

**Modeling constraint:** SMP constitutes 21.92% of training nodes but 73.60%
of validation nodes. The same 395,061-character post is associated with
CVE-2025-26465 and CVE-2025-26466; each association produces 97 chunks and
127,465 nodes. Those remain distinct CVEs. Pool chunks within documents and
documents within sources; avoid treating every chunk/node as an equal CVE-level
observation. No source or CVE was removed by this review.

The review is saved as TXT/JSON in
`01_tpg/05_workspace/cve_identity_multisource_v2/completed_review_v1/`.
Reproduce it without re-encoding, model execution or the full tensor traversal:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_identity_multisource_cache.py --verify-only
```

Next: implement a fixed matched comparison of EPSS-only, description TPG plus
EPSS, and all-source TPG plus EPSS, with source masks and bounded graph batching.
Run one integration smoke before full fixed-seed development fits. Earlier
H02/R07 numbers are not matched baselines for this changed cohort. Source
removal ablations follow the same CVEs. The test set remains closed, and
retrospective-text limitations still apply.

### Graph Contract

**V2 coverage correction:** the v1 parallel run failed on node 208 of
`CD03-de0b13b82adab6b4c808f66874d7e40a26ec5df9cd2bf88db148aa6b404cb571`,
the non-GitHub summary of CVE-2024-49125. Its text is U+0000 at chunk offsets
556:557. V1 incorrectly required an embedding for this tokenizer-removed
control. At investigation, 5,700 completed chunk receipts were present.

The prepared chunks contain 52 occurrences across six control types: U+200B
(35), U+0000 (3), U+FEFF (1), U+0080 (6), U+0099 (6), U+2063 (1). These are
character occurrences, not counts of missing word embeddings or unique CVEs.
The actual pinned tokenizer returns no token IDs for each of these controls.

Bounded real verification passed for all 34 development chunks containing
these controls. Only two control-only token nodes lacked embeddings: the
reported U+0000 node, and U+2063 at 1979:1980 in an SMP for CVE-2025-30065
(node 544). Ordinary lexical token coverage passed. The reproducible TXT/JSON
report is in `01_tpg/05_workspace/cve_identity_multisource_v2/control_character_check/full/`.
Twenty-three focused tests passed, covering both old and new paths, source
preservation, strict ordinary-word coverage, masks/pooling, worker dispatch
and unchanged v1 receipt reuse. No full v2 cache was built by these checks.

V2 is separately versioned at `01_tpg/05_workspace/cve_identity_multisource_v2/`.
It copies verified v1 graphs/tensors/receipts unchanged and builds only missing
chunks under the extended coverage policy. Source text, graph topology,
source masks, chunking and encoder weights are unchanged. Control-only tokens
can have zero embeddings and explicit masks; ordinary words and mixed
word/control tokens cannot. Controls are excluded from text pooling and
audited separately from whitespace. The original strict validator is unchanged.

Run `prepare_identity_multisource_graphs_v2.py --scale full --workers 8
--threads-per-worker 1` from `01_tpg/02_examples/01_scripts/` using its full
repository-relative path as in the artifact guide. The first v2 run does not
use `--resume`; subsequent interrupted v2 runs do. Verify using the same v2
script with `--scale full --verify-only`. No full v2 generation or training
has been run by the automated review.

- Independent source documents, split into non-overlapping chunks of at most
  4,096 characters, preferring whitespace. Every original character is covered.
- Forced boundaries and non-graph whitespace/replacement-only segments are
  explicit. Empty sources have masks and empty lists, not fabricated graphs.
- Shared core plus unchanged v4 security extraction and frozen SecBERT;
  no-COREF is primary. Three edge views share the same node embeddings.
- Chunk-local and full-document offsets, source labels, original document
  text, CVE/document/chunk identities and per-chunk receipts are retained.
- Streaming generation and per-CVE loading avoid retaining all embeddings
  in memory. Verified smoke chunks are reused in the full run. Resume verifies
  receipts and rejects changed sources or artifacts.
- Numeric loading allowlists graph tensors, keeping text, identifiers,
  source-audit metadata and targets out of numerical predictor features.

Cross-chunk relations may be lost; none are invented. Future pooling must
respect document/chunk nesting rather than treating chunks as independent
CVEs. Current text remains retrospective and is not authenticated historical
evidence. This stage does not train or change the extractor.

### Commands and Outputs

**CPU parallel execution added:** an independent runner wraps the unchanged
checksum-pinned CD03 builder. It supports resuming the user's partial full
cache, verifies existing receipts, bounds pending work to twice the worker
count, and keeps tensors out of inter-process queues. Each worker owns its
local parser/encoder. Graph policies, text, cohorts and targets are unchanged.
Execution JSON/TXT provenance is outside the frozen cache file set, under
`01_tpg/05_workspace/cve_identity_multisource_v1/parallel_execution/`.

Stop the old command with Ctrl+C and wait for it to exit, then run:

```bash
python EPSS_TPG/01_tpg/02_examples/01_scripts/prepare_identity_multisource_graphs_parallel.py --scale full --resume --workers 8 --threads-per-worker 1
```

Use the original `--verify-only` command afterward. Seven focused tests cover
CPU limits, environment restoration, writer locking, spawned execution,
hash-preserving resume, encoder-free smoke reuse, worker failure recovery and
corruption rejection. A full parallel dataset run has not been performed by
the automated review. More workers require additional RAM; speed is not assumed to
scale linearly. The sequential runner must never run concurrently on the same
output directory.

Bounded real verification: 24 training-source chunks, 26.29 seconds sequential
(excluding encoder loading) versus 12.07 seconds with four spawned workers
(including startup/loading). Node/edge content and statistics matched; maximum
node-feature difference was zero. Temporary check artifacts were removed;
the user's original smoke and partial full caches were not modified by this
check. These timings do not establish the speedup of the complete dataset.

From the repository root, with `CodeBERTFusion` active, run individually:

```bash
python -m unittest discover -s EPSS_TPG/01_tpg/08_tests -p 'test_identity_multisource_graphs.py'
python EPSS_TPG/01_tpg/02_examples/01_scripts/prepare_identity_multisource_graphs.py --scale smoke
python EPSS_TPG/01_tpg/02_examples/01_scripts/prepare_identity_multisource_graphs.py --scale smoke --verify-only
python EPSS_TPG/01_tpg/02_examples/01_scripts/prepare_identity_multisource_graphs.py --scale full
python EPSS_TPG/01_tpg/02_examples/01_scripts/prepare_identity_multisource_graphs.py --scale full --verify-only
```

Use `--resume` only for an interrupted incomplete cache; use `--plan-only`
to inspect counts without building. Model downloads are opt-in through
`--allow-model-download`. Verification never reruns the parser or encoder.

Outputs: `01_tpg/05_workspace/cve_identity_multisource_v1/{smoke,full}/`.
Open `README.md`, `cve_index.json`, then `by_cve/<CVE>.json`. Actual node/edge
graphs are under `tpg_json/no_coref/<CVE>/<source>/<document-hash>/`; matching
embedding tensors are under `tensors/`. Full text is stored in `documents/`.
`analysis.txt` and `analysis.json` report graph and embedding counts.

The focused tests cover lossless chunking, forced boundaries, source routing,
test/target rejection, graph identity, saved artifact verification, numeric
isolation, interruption/resume, smoke reuse, corruption and parent offsets.
Bounded real checks use training documents only and temporary outputs, not a
completed CD03 cache. The next stage after cache verification is the fixed
multi-source fusion runner with a matched EPSS-only control, one integration
smoke and full-development training, not another extractor-tuning cycle.

## CD02: CVE-Based Development Preparation

**Status:** user's smoke and full preparation completed and independently
reproduced. Eleven synthetic regression tests and real preflight also passed.
No graphs, embeddings or training were generated by this preparation. CD01,
H02, MS02 and all original artifacts remain unchanged.

### Completed Preparation

| Source | Training CVEs with text | Validation CVEs with text |
|---|---:|---:|
| Description | 2,548 | 1,104 |
| GitHub summary | 539 | 201 |
| Non-GitHub summary | 1,785 | 749 |
| SMP | 1,798 | 1,054 |

There are 6,865 training and 3,573 validation document associations (10,438
total), not unique texts or already built graphs. Two validation posts on
CVE-2025-1094 were masked; no summary or CVE was removed by overlap screening.
Ten validation description overlaps remain intentionally. Training includes
342 partial recent histories and 2,308 partial longer histories. Validation
includes 867 partial longer histories. All exported targets are observed.

The largest retained training document has 14,929 characters; twelve validation
document associations exceed that size, up to 395,061 characters in an SMP
document associated with CVE-2025-26465 and CVE-2025-26466. The next graph stage
needs bounded chunking with offsets/coverage checks and resume, not silent
truncation or whole-CVE exclusion. Length does not itself establish bad content.
The test distribution remains narrow: 633/651 eligible anchors are July 2025.

Read the separate completed review at
`05_results/02_evaluation/epss_day60_to_day90/cve_identity_preparation_review_v1/analysis.txt`
and `analysis.json`, relative to `EPSS_TPG`. No further cohort-selection round
is requested; implement the graph/cache adapter and fixed fusion comparison.

### Verified Identity Clarification

There are 5,968 unique CVE IDs in the pinned SMP-dated dataset. Rechecking
`description` gives 5,660 distinct exact strings, with 113 repeated-string
groups covering 421 records. Lowercasing/whitespace normalization gives 116
groups covering 430 records, agreeing with the existing full-dataset analysis
in `05_results/03_dataset_analysis/01_revision_dataset/README.md` item 7.
For example, CVE-2024-38106 and CVE-2024-21338 both have the complete stored
description "Windows Kernel Elevation of Privilege Vulnerability". These are
distinct CVE records; repeated short wording is not proof of duplicate
vulnerabilities. The earlier 69/66/61 examples concerned stored post content,
not a claim that those CVEs all share the same descriptions.

The full-dataset summary review and its later revised 50-CVE sample comparison
remain separate. This stage uses the pinned 5,968-CVE dataset and does not
silently replace full-data summaries with sample-only regenerated summaries.
No new factuality filter or perfect-summary requirement is introduced.

### Cohort and Source Rules

- One record per unique CVE ID. Different CVEs are not grouped because of a
  shared URL, current/original description or post. This is CVE-disjoint, not
  proven vulnerability-family-disjoint or leakage-free evaluation.
- Anchor is publication + 60 days, target is the observed publication + 90-day
  score. Training labels precede 2025-04-01; validation anchors start then and
  their labels precede 2025-07-01. Test anchors start in July and precede
  2026-01-01, with target maturity by 2026-09-29. Label-boundary exclusions
  remain unchanged. Cohort: **2,548 train / 1,104 validation / 651 eligible test**.
- Partial input histories remain masked, not dropped. Targets are not imputed.
  Both 31-day and 61-day histories are exported with the existing tested
  time/mask/age contracts, separately from numerical labels.
- All nonempty descriptions and summaries remain retrospective. The existing
  source adapter screens posts strictly before Day 60 with consistent dates.
- Build a registry from accepted training text only. If an optional validation
  summary/post matches a training document after lowercasing and whitespace
  normalization, mask that validation document, not the CVE. Comparison keys
  are normalized; actual text is preserved. Cross-source equality is included.
- Keep descriptions even when their wording repeats and report the overlap.
  Keep all nonempty imperfect summaries unless masked by the explicit overlap
  rule; no undocumented failure/quality heuristic is added.
- Missing sources have empty document lists and explicit masks. Source choices
  never change the CVE cohort. Retain the old strict-group experiment as an
  available harder comparison, not proof that its groups are duplicates.
- Smoke exports 16 training / eight validation CVEs selected by a fixed ID hash.
  Ownership/coverage uses the full development text population even in smoke,
  ensuring identical source decisions in the later full export.

The 651 eligible test anchors are concentrated in July 2025. Do not infer broad
later-year coverage. Only test IDs/dates/availability metadata are exported;
no numerical test labels, histories or post/summary documents. Global
description-string counts are checked without exporting test descriptions.
A future test adapter must use the saved training and retained-validation
text registries before evaluation. That adapter is not implemented here.

### Execution and Files

From the project root with `CodeBERTFusion` active, run individually:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_identity_development.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_identity_development.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_identity_development.py --scale smoke
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_identity_development.py --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_identity_development.py --scale full
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_identity_development.py --scale full --verify-only
```

Full preparation requires the matching saved smoke export and verifies its
histories and source routing against the full export. Existing output paths
are never overwritten. Verification reproduces data and checks file hashes.

Paths relative to `EPSS_TPG`:

| File/directory | Purpose |
|---|---|
| `02_prediction/01_package/epss/forecast_identity_development.py` | Versioned preparation, source masks and validation |
| `02_prediction/03_analysis/01_revision_dataset/prepare_forecast_identity_development.py` | CLI |
| `02_prediction/03_analysis/01_revision_dataset/test_forecast_identity_development.py` | Synthetic regression checks |
| `02_prediction/04_configs/cve_identity_development_v1.json` | Fixed new cohort/source contract |
| `04_data/01_records_and_graphs/cve_identity_day60_to_day90_v1/{smoke,full}/` | Separate outputs |

Each output contains `analysis.txt` and `analysis.json`, a protocol/identity,
all-CVE split/availability metadata, description-string counts, separate
development `inputs.json`/`labels.json`, allowlisted `source_inputs.json`,
source/date decision audits, training and retained-validation text registries,
and `preparation_lock.json`. Registries/audits are not numerical model inputs.

**Next:** review the completed export, then implement its full source-aware
graph cache and fixed fusion runner using the existing TPG. Do not feed the new
cohort into H02/MS02 fixed-cohort runners unchanged. One integration smoke
precedes full training; no additional optimizer-tuning round is planned.

## CD01: Bounded Cohort-Design Review

**Status:** user's full review complete and independently reproduced. Nine
synthetic tests and source/prerequisite preflight also pass. H02/MS02 and all
earlier models, caches, data exports and split assignments remain unchanged.

### Completed Findings

| Data availability before splitting | CVEs |
|---|---:|
| Observed Day-90 score plus some Days 30-60 input | 5,645 |
| Observed Day-90 score but no recent input | 6 |
| Missing Day-90 score | 317 |
| Total unique CVE IDs | 5,968 |

| Frozen grouping candidate | Eligible train | Eligible validation | Eligible test |
|---|---:|---:|---:|
| Existing | 1,029 | 173 | 2,984 |
| Six-month validation | 1,102 | 91 | 2,984 |
| Later Q1 validation | 1,230 | 672 | 1,756 |
| Later Q2 validation | 1,998 | 884 | 342 |

The largest-training option has a narrow test concentration: 334/342 anchors
are in July 2025. It excludes 1,242 records by grouping, 619 at the training
label boundary, 709 at the validation label boundary and 15 outside the test
anchor period. Boundary exclusions are not duplicate records. Partial input
histories remain eligible (287 recent-partial training CVEs in that option).

Shared-evidence components must not be called duplicate vulnerabilities.
Distinct CVEs can share an advisory or post without being the same issue.
The grouping uses descriptions/original descriptions, normalized stored posts
and URLs, not generated summary text. Saved metadata includes the same stored
post associated with 69 CVEs, another with 66, and an aerospace-CVE-list item
with 61. Even without URL grouping, a text-connected excluded component has
798 eligible CVEs. These are content-overlap observations, not semantic proof
that those CVEs are duplicates or that every association is wrong.

The separate interpretation recommends separating identity/strong-relatedness
rules from optional-source quality and overlap handling. Do not discard whole
CVEs solely for an imperfect summary or shared broad URL/post. Do not ignore
repeated actual model inputs either: document source-level overlap handling
and retain a strict grouped sensitivity comparison. Current text remains
retrospective. No revised policy or new counts are adopted by this advice.

Read `05_results/02_evaluation/epss_day60_to_day90/cohort_design_interpretation_v1/analysis.txt`
and `analysis.json` (relative to `EPSS_TPG`). They preserve the original locked
review and record source hashes, counts, grouping examples and limitations.

### Question and Limits

How much of the dataset could support Day-60-to-Day-90 forecasting under a
better-balanced chronological design? Incomplete input histories are allowed.
Every ordinary endpoint example still needs an observed Day-90 score and at
least one observed input. No target imputation, performance-based split search,
model training, graph generation or adoption of a new split is implemented.

This deliberately extends metadata inspection to all records, including
previously reserved test CVEs. Only dates, observation availability, description
presence and grouping evidence are used. Score magnitudes are validated and
discarded, never exported, summarized or compared. Existing numerical labels,
test predictions and model checkpoints are not loaded. Group evidence may
contain shared text; it is used only to rebuild components, not as model input,
and no text excerpts are exported by this review.

### Fixed Comparisons

| Candidate | Training target must precede | Validation anchors start | Test anchors start |
|---|---|---|---|
| Existing | 2024-01-01 | 2024-01-01 | 2025-01-01 |
| Six-month validation | 2024-07-01 | 2024-07-01 | 2025-01-01 |
| Later Q1 validation | 2025-01-01 | 2025-01-01 | 2025-04-01 |
| Later Q2 validation | 2025-04-01 | 2025-04-01 | 2025-07-01 |

Anchor means publication plus 60 days. Validation targets must precede test
start. Test anchors must precede 2026-01-01 and targets must be no later than
the unchanged observation cutoff, 2026-09-29. These fixed candidates are not
selected from model outcomes. A candidate with inadequate validation/test
coverage must not be adopted just because it increases training count.

Each window is checked with unchanged frozen groups and, separately, an
explicitly weaker **text-only sensitivity** grouping. The sensitivity retains
repeated description/post links but omits URL links; it is not a leakage-safe
recommended training split. Shared evidence crossing roles is counted. The
ten most widely shared active URLs also get individual omission sensitivity
counts and a pending-review status. High frequency or a homepage hint does not
prove a URL should be ignored. No URL exception is adopted automatically.

Results include role counts, independent grouping-component counts, largest
eligible groups, partial-history counts, missing-target/input counts, date
coverage, exclusion reasons and movements from previous roles. Longer-history
and description-only potential are reported separately, not silently included
in the matched recent-history cohort. Group counts are independence under that
rule, not proof that all semantic relatedness is detected.

### Files and Execution

Paths below are relative to `EPSS_TPG`:

- `02_prediction/03_analysis/01_revision_dataset/review_forecast_cohort_design.py`: CLI and metadata review.
- `02_prediction/03_analysis/01_revision_dataset/test_forecast_cohort_design.py`: synthetic regression tests.
- `02_prediction/04_configs/cohort_design_review_v1.json`: frozen candidate windows and review limits.
- `05_results/02_evaluation/epss_day60_to_day90/cohort_design_review_v1/`: new review output only.

The output contains `analysis.txt`, `analysis.json`, `protocol.json`,
`cve_availability.json`, `candidate_assignments.json`, `url_review_queue.json`,
`cohort_counts.png` and `review_lock.json`. Candidate assignments are audit
metadata, not training manifests. The figure shows eligible CVE counts by
candidate window and role with the frozen grouping retained.

Run from the project root with `CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_cohort_design.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_cohort_design.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_cohort_design.py
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_cohort_design.py --verify-only
```

Default execution refuses an existing output directory. Verification checks
hashes and reproduces the numerical metadata tables and narrative without
overwriting the first report. Progress bars cover record projection, candidate
windows, bounded URL checks and artifact saving.

**Next:** review these counts and grouping risks, explicitly choose and freeze
a separate cohort protocol (or keep H02), then prepare the full multi-source
cache and fixed training comparison. Do not silently recycle prior test CVEs,
remove difficult outcomes, or compare changed-cohort metrics as a matched
architecture improvement. No additional optimizer diagnostic is required.

## MS02: Small Multi-Source Graph Preview

**Status:** user's cache complete and independently verified. All 331 artifact
hashes, 65 graph/tensor bundles, source labels and Day-60 masks pass. Eleven
synthetic tests and the earlier real four-source preflight also passed.
Verification does not rerun NLP/embeddings or train a prediction model.

### Completed Findings and Training Status

| Preview result | Verified value |
|---|---:|
| CVEs, training / validation | 16 / 8 |
| Independent source documents | 65 |
| Nodes / no-COREF edges | 13,604 / 42,568 |
| Nodes with contextual embeddings | 12,979 |
| Nodes without contextual embeddings | 625 |
| Typed security edges | 19: 17 SEC_HAS_VERSION, 2 SEC_CAUSES |

The unembedded nodes are 65 DOCUMENT, 113 PARAGRAPH, 325 TOPIC and 122
whitespace TOKEN nodes. No ordinary-token embeddings are missing. Typed
security edges occur in 15 documents across seven CVEs; these are coverage
counts, not semantic accuracy. All sources together have 4.29 times the nodes
of description-only in this preview. Pool by document/source to avoid simply
letting long posts dominate. The largest document has 1,872 nodes.

R07 and H02 already completed full eligible-cohort training. Multi-source
TPG fusion has not been trained. H02's 5,968 records are accounted for by
1,029 eligible training, 173 eligible validation, 2,985 reserved test,
1,659 split-policy exclusions and 122 development records missing a required
input or target. Partial input histories remain eligible. Of the split-policy
exclusions, 1,565 concern related groups crossing time folds, 79 concern label
boundaries and 15 concern later anchors outside the primary test policy.
The reserved test number is an assignment count, not scored eligibility.

Full source preparation would cover 2,869 training and 434 validation documents
on the unchanged 1,029/173 cohort. Recommended initial comparison: EPSS-only,
description+EPSS and all-text+EPSS, seeds 17/29/43 (nine fits), plus persistence.
This is a proposed next protocol, not an implemented training command. Run one
integration smoke before those full fits; source-removal ablations can follow.
Current descriptions/summaries and date-screened posts remain retrospective
evidence, not authenticated Day-60 inputs. Do not claim historical prediction
validity from source masks alone.

Review: `01_tpg/05_workspace/multisource_day60_v1/preview_review_v1/analysis.txt`
and `analysis.json` (paths relative to `EPSS_TPG`). Reproduce without training:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_multisource_graph_preview.py --verify-only
```

### Scope and Inputs

Use **exactly H01's 16 training / eight validation CVEs**, retained in the
H02 full development export. No resampling, completeness filtering, source
fallback, expansion to 1,202 CVEs or numerical test-data access.
The source adapter's **Day-60** candidate masks are reproduced from its
saved source audit. Graph construction receives only allowlisted document
text and source identity, not EPSS scores, targets, CVSS or VulnCheck fields.
The entry point hash-verifies prerequisites; it does not read numerical
history/label contents or run the trained H02 models.

| Source | Training CVEs with accepted text | Training documents | Validation CVEs with accepted text | Validation documents |
|---|---:|---:|---:|---:|
| Description | 16 | 16 | 8 | 8 |
| GitHub summary | 3 | 3 | 0 | 0 |
| Non-GitHub summary | 8 | 8 | 8 | 8 |
| Social-media posts | 14 | 16 | 4 | 6 |
| Total documents | Not additive across sources | 43 | Not additive across sources | 22 |

The 65 documents are not 65 CVEs. Some CVEs have multiple accepted posts.
The missing GitHub validation source is intentional: the fixed preview can
check missing-source behavior but cannot assess GitHub-summary benefit.

### Graph and Cache Design

1. Build one independent graph per document through the existing shared
   TPG core and opt-in v4 product/version security pipeline. Do not alter
   v4, earlier caches or saved runs.
2. Preserve the exact source wording and offsets. Mark every node and edge
   with `source_text_type`: description, github_summary, non_github_summary
   or smp. A globally unique node reference is `(graph_id, local_node_id)`.
3. Do not invent cross-document entity, dependency or coreference links.
   Descriptions, summaries and individual posts remain separate graphs.
4. Embed each accepted document once with the existing frozen SecBERT
   encoder and saved vocabulary. Keep contextual token/span embeddings,
   node-type channels and embedding-present masks. No fine-tuning or silent
   truncation. Missing ordinary-token embeddings stop the run. The existing
   audited whitespace and U+FFFD-only exceptions remain unchanged.
5. Use the **no-COREF** view for every source comparison. Other edge masks
   and unembedded GraphSON variants are kept for compatibility/inspection,
   not to reopen coreference selection. Security relation IDs come from the
   saved exporter vocabulary, not hardcoded indices.
6. Route documents into the fixed four-source order. Missing sources have
   an empty graph list and mask zero; no fake sentence or empty graph is
   created. Retain all 24 CVEs even for an EPSS-only or source-empty view.
7. Source-removal views reuse the same node tensors and embeddings. Numeric
   views contain only `x`, `edge_index`, `edge_type`, `edge_features` and
   `text_embedding`, grouped by source with a mask. Graph/audit identifiers
   stay outside these numeric graph inputs.

The internal shared-cache compatibility fields `cve` and
`description_sha256` carry the document graph ID and document text hash,
respectively. They do not imply that summaries/posts are descriptions and
are stripped by the numeric view adapter.

The ten existing MS01 source combinations are retained: EPSS-only;
description-only, GitHub-summary-only, non-GitHub-summary-only and SMP-only
text views; all text; and each of four leave-one-source-out views. These
are graph-routing checks, **not trained ablation experiments**. No EPSS
history is copied into this graph cache.

Future fusion can pool document graph embeddings within each source, then
combine source representations/masks with the temporal encoder. That model
and its training protocol are not implemented by this preview.

### Files and Commands

| File or directory | Purpose |
|---|---|
| `01_tpg/02_examples/01_scripts/prepare_multisource_graph_preview.py` | Preview-only CLI |
| `02_prediction/01_package/epss/forecast_multisource_graphs.py` | H02-to-TPG adapter, construction, cache verification and source views |
| `02_prediction/04_configs/multisource_graph_preview_v1.json` | Fixed preview scope and limitations |
| `01_tpg/08_tests/test_multisource_graph_preview.py` | Eleven synthetic tests, including date-mask exclusion and resume |
| `01_tpg/05_workspace/multisource_day60_v1/graph_preview/` | New output directory, separate from existing caches |

Paths above are relative to `EPSS_TPG`. Within the new output, `manifest.json`
and `run_identity.json` map CVEs to source document hashes and resources;
`graphson/{mode}/{document_key}.json` stores inspectable unembedded graphs;
`graphs/{document_key}.pt` stores shared tensors and edge masks;
`receipts/{document_key}.json` permits document-level checked resume;
`analysis.txt` and `analysis.json` report graph sizes, embeddings, typed
security edges, source masks and all ten source views. `cache_lock.json`
protects the completed file set.

Run individually from the repository root with `CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/01_tpg/08_tests -p 'test_multisource_graph_preview.py'
python EPSS_TPG/01_tpg/02_examples/01_scripts/prepare_multisource_graph_preview.py --check-only
python EPSS_TPG/01_tpg/02_examples/01_scripts/prepare_multisource_graph_preview.py
python EPSS_TPG/01_tpg/02_examples/01_scripts/prepare_multisource_graph_preview.py --verify-only
```

`--check-only` computes four temporary training-source graphs/embeddings
without saving files. Downloads are disabled unless explicitly enabled with
`--allow-model-download`; the tested local resources were available.
`--verify-only` never loads a parser/encoder or regenerates graphs. It checks
saved graphs, source tags, masks, vocabulary, embeddings, reports and hashes.
For an interrupted build, use `--resume`: completed document receipts are
checked and retained; only unfinished documents are rebuilt.
There is no `--scale full` or training mode.

**Next:** the produced graphs and source sizes are reviewed. Prepare the full
matched development cache and a source-aware fusion runner. Security edges remain candidates, and
current descriptions/summaries plus supplied post dates remain retrospective
evidence, not authenticated historical inputs. Text itself may contain later
events or exploitation statements. No forecasting claim follows from cache
creation. The reserved test remains closed.

## H02: Full Matched History Controls

**Status:** user export, smoke and full training complete; all saved
checkpoint predictions independently reproduced. No automated retraining.
H01, R07, all previous graph caches and the reserved test remain unchanged.

### Verified Results

The separate review is in
`EPSS_TPG/05_results/02_evaluation/epss_day60_to_day90/history_controls_review_v1/`.
It contains `analysis.txt`, `analysis.json`, `learning_curves.png`,
`error_concentration.png` and `review_lock.json`. Original results are unchanged.

| Model/history | Seed | Selected epoch / epochs run | Validation MAE | Validation RMSE |
|---|---:|---:|---:|---:|
| Persistence | Not trained | Not applicable | 0.008303930636 | 0.07386211956 |
| Recent and longer, identical final predictions | 17 | 12 / 27 | 0.008303947254 | 0.07386211799 |
| Recent and longer, identical final predictions | 29 | 31 / 46 | 0.008303959727 | 0.07386212318 |
| Recent and longer, identical final predictions | 43 | 1 / 16 | 0.008303943554 | 0.07386211794 |

Each seeded row represents **two fits**, not one. Both histories have mean
MAE 0.008303950178 and sample SD 0.000000008474 across seeds. All six MAEs
are slightly worse than persistence, by only about 1.3e-8 to 2.9e-8.
The tiny RMSE differences are not a meaningful gain either. Longer history
has demonstrated no advantage under this fixed architecture and protocol.

**Endpoint behavior:** in training, 949/1,029 targets equal the last observed
input, 50 increase and 30 decrease. In validation, 150/173 are equal, 18
increase and five decrease. These compare endpoints only; the intervening
trajectory need not be flat. With MAE, needless corrections on many unchanged
CVEs can outweigh improvements on a few changing ones.

| CVE | Last observed input EPSS | Actual Day-90 EPSS | Absolute persistence error | Share of total validation absolute error |
|---|---:|---:|---:|---:|
| CVE-2023-45249 | 0.12209 | 0.88105 | 0.75896 | 52.83% |
| CVE-2023-7028 | 0.20153 | 0.80716 | 0.60563 | 42.16% |

Together these two cases account for **94.99%** of total absolute validation
error. The models remain near persistence and miss both increases. All CVEs
remain in all primary metrics. Spearman correlation near 0.978 does not mean
these large changes were forecast.

**Mechanism checks:** the real loader produces distinct `[N,31,5]` and
`[N,61,5]` tensors, with correctly matching recent observations. Older input
scores exist for 1,017 training CVEs and all 173 validation CVEs. The saved
paired weights are not identical; recorded temporal data gradients are
nonzero and weights changed. There is no evidence here of both variants
accidentally using the same 31-day input or a completely disconnected GRU.

Nevertheless, seeds 17/29 produce constant corrections across validation.
Their selected temporal parameter norms shrink from about 4.8 to around
1e-11 and 1e-37 respectively. Seed 43 selects epoch one; its corrections
vary by CVE but remain below 4.5e-8 in absolute value. Tiny between-history
correction differences disappear in the final float32 sum. This documents
an effectively persistence-only solution; it does not isolate weight decay
as the sole causal mechanism or prove older history has no usable signal.

**Missingness:** neither history improves on persistence on the same
complete-history or partial-history validation subsets. Those subsets have
different CVEs and error concentration, so their absolute MAEs do not prove
that missingness caused higher error. Detailed counts and metrics remain in
the review JSON/TXT. No completeness threshold or split was changed.

**Next decision:** retain persistence and move to a small source-aware TPG
graph/cache preview using the fixed Day-60 sources and masks. Recent history
is the simpler provisional EPSS control, not a statistically established
winner. Avoid another outcome-driven EPSS tuning loop. Freeze matched
fusion/source-removal comparisons after the graph preview and retain the
distinction between retrospective text evidence and historically verified
forecasting. Final test remains closed.

Reproduce this review without changing the runs:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_later_controls.py --verify-only
```

### Fixed Comparison

| Item | Fixed decision |
|---|---|
| Prediction time / target | Day 60 after stored NVD publication / observed Day-90 EPSS |
| Training / validation | Exact H01 matched IDs: 1,029 / 173 |
| Recent history | Days 30-60 inclusive: 31 daily observations, including missing values |
| Longer history | Days 0-60 inclusive: 61 daily observations, including missing values |
| Shared reference | Repeat the latest observed input EPSS at Day 90 (persistence) |
| Temporal model | One 16-unit GRU; 32-unit GELU head; scalar residual output |
| Prediction | `clip(persistence + tanh(gain * head(hidden)), 0, 1)` |
| Gain | Median latest observed score across training CVEs, clipped to [0.001, 0.05]; no labels or validation inputs |
| Initial output | Exactly persistence; final linear weights and bias start at zero |
| Objective / optimizer | MAE / Adam, learning rate 0.001, weight decay 0.0001 |
| Batch / clipping | Eight CVEs; gradient norm clipped to 1.0 |
| Full schedule | Seeds 17, 29, 43; two histories per seed; up to 100 epochs, patience 15 |
| Selection | Earliest strictly best validation MAE from epochs 1 onward; epoch zero excluded |
| Execution | Deterministic CPU, one PyTorch thread |
| Smoke | H01's 16 training / eight validation IDs; seed 17, three epochs |
| Full-run gate | Smoke checkpoints and predictions reproduce; no smoke-score threshold or tuning |
| Evaluation | MAE, RMSE, mean signed error, Spearman correlation when defined; paired MAE differences and all-seed summaries |

Weights and batch orders are identical between the two histories for each
seed. The full run starts fresh, not from smoke checkpoints. Gain is fitted
on the relevant training subset for each scale, identically for both histories.
The GRU/head family follows R07's residual EPSS control, but uses a new loader
and newly derived gain; R07 weights and graph caches are not reused.

### Input Contract

Five channels per day: zero-padded score, observed mask, within-window
observation age divided by 60, age-known mask, and day since publication
divided by 60. Both variants use the **same denominator**. The recent window
therefore starts at time coordinate 0.5, not zero. An observed zero has mask
one; padding for a missing score has mask zero. Before the first observation
inside a window, age padding is zero and the age-known mask is zero.

The recent view never inherits an earlier observation or its age from Days
0-29. Both variants end at Day 60 and use the same persistence score. Labels,
text, CVE IDs, VulnCheck fields and audit metadata are not tensor channels.
Targets remain separate. A missing source never removes a CVE.

### Full Source Coverage

These counts were calculated during the no-write real-data preflight.
They describe the full matched development cohort, not just H01's preview.

| Source | Training with text | Training passing screen | Validation with text | Validation passing screen |
|---|---:|---:|---:|---:|
| Description | 1,029 | 1,029 | 173 | 173 |
| GitHub summary | 236 | 236 | 51 | 51 |
| Non-GitHub summary | 705 | 705 | 121 | 121 |
| Social-media posts | 1,029 | 880 | 172 | 53 |

Descriptions and summaries pass the retrospective candidate view, **not**
a historical-availability verification. Social posts must pass the existing
strictly-before-Day-60 stored-date screen. There are 1,527 post-date entries:
988 before the anchor, 538 after it and one on it; all omit timezone.
These are date-evidence counts, not distinct CVE counts. There are 84
conflicting post documents. Exact candidate text hashes shared across
training and validation: zero. This narrow check does not rule out
paraphrases, common sources or other dependence.

Multi-source TPG fusion remains the next graph stage, not an input to H02
training. It must retain these IDs, source masks and Day-60 cutoff, with
shared embeddings and matched source-removal controls for description,
GitHub summaries, non-GitHub summaries, SMP and the combined model.
Start its graph work on a small preview. Current sources support
retrospective comparisons; authenticated historical text forecasting remains
unsupported by these checks.

### Artifacts and Commands

Implementation:

- `01_package/epss/forecast_later_development.py`: reproduce H01, export full histories and separate labels, audit sources.
- `01_package/epss/forecast_later_controls.py`: validated variable-length loader, fixed GRU controls, persistence, verification and resume.
- `04_configs/day60_to_day90_controls_v1.json`: fixed settings and limitations.
- `03_analysis/01_revision_dataset/test_forecast_later_development.py`: fourteen focused synthetic tests.

Paths above are relative to `EPSS_TPG/02_prediction`.

Preparation output:
`EPSS_TPG/04_data/01_records_and_graphs/epss_day60_to_day90_v1/development/`.
It contains `inputs.json`, `labels.json`, `source_inputs.json`,
`source_audit.json`, `source_coverage.json`, `source_coverage.txt`,
`split_manifest.json`, `cohort_audit.json`, `analysis.json`, `analysis.txt`,
`protocol.json`, `run_identity.json`, and `preparation_lock.json`.
The full split manifest contains reserved-test metadata only.

Training output:
`EPSS_TPG/05_results/02_evaluation/epss_day60_to_day90/history_controls_v1/`,
with separate `smoke/` and `full/` directories. Each fit saves
`checkpoint.pt`, `history.json`, `predictions.json`, `metrics.json`,
and `fit_lock.json`. Root files include `baselines.json`, fixed per-seed
batch orders, `run_identity.json`, `summary.json`, `summary.txt`,
and `evaluation_lock.json`.

Run commands individually, in order:

The user has already completed this sequence. For the existing outputs,
use `--verify-only`; do not rerun creation commands into completed directories.

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

For interrupted training, repeat the relevant training command with
`--resume`. Locked completed fits are verified and retained; an incomplete
fit restarts from its original seed. Completed runs use `--verify-only`,
not `--resume`. Verification reloads checkpoints and reproduces validation
predictions without optimizer updates. No command exposes test scoring.

Review all six full fits against their shared persistence reference, report
all seeds and compare histories on paired CVEs. Neither a later anchor nor
lower error establishes earlier warning, causality or useful text evidence.

## H01: Later-Anchor Preparation

**Status:** user preview complete and independently reproduced; all ten
artifacts verified. Ten preparation tests and three review tests pass.
This step creates a small data preview, not graphs or trained models.
R07, MS01, their original cohorts and saved results remain unchanged.

### New Question

At Day 60 after NVD publication, does the model benefit from:

| History variant | Available EPSS inputs | Target | Forecast horizon |
|---|---|---|---|
| `recent_30_60` | Days 30-60 inclusive, 31 values | Day-90 EPSS | 30 days |
| `long_0_60` | Days 0-60 inclusive, 61 values | Day-90 EPSS | 30 days |

Both variants use **identical CVEs**: an observed Day-90 target and at least
one observed input in both histories are required. No target imputation.
Missing input scores have masks and observation-age fields. A true zero
remains observed. The recent variant cannot use early observations for its
age calculation. The common latest observed score, hence the persistence
reference, is identical across these two variants.

This does not claim that later forecasting is easier or provides an earlier
warning. R07 predicts at Day 30; H01 predicts at Day 60. Their headline errors
are not directly comparable because both labels and cohort membership differ.
No further loss/decay sweep is authorized by this preparation.

### Recomputed Splits

The existing shared split helper uses the new input cutoff and target date:

- Train: Day-60 anchor and Day-90 target before `2024-01-01`.
- Validation: Day-60 anchor in 2024; Day-90 target before `2025-01-01`.
- Reserved test: Day-60 anchor in 2025, within the frozen observation cutoff.
- Exclude targets crossing development boundaries and groups spanning active
  periods. Keep the same group IDs, links and calendar boundaries.
- Reject any previously reserved test CVE entering development.

All **5,968** source CVEs are accounted for by metadata assignment:

| Role | Assigned CVEs |
|---|---:|
| Train | 1,150 |
| Validation | 174 |
| Reserved test | 2,985 |
| Excluded | 1,659 |

The reserved-test number is an **assignment count**, not an eligibility or
performance result. No test numerical histories, targets or text are
exported or analyzed. Existing whole-source integrity checks still verify
the frozen files.

Exclusions: 1,565 related-group conflicts, 24 training label-boundary
crossings, 55 validation label-boundary crossings and 15 later anchors.
There are 30 conflicting groups at this anchor.

| Development split | Assigned | Recent-history eligible | Longer-history eligible | Matched eligible | Overlap with R07 eligible CVEs |
|---|---:|---:|---:|---:|---:|
| Train | 1,150 | 1,029 | 1,029 | **1,029** | 1,012 |
| Validation | 174 | 173 | 173 | **173** | 166 |

These counts were checked by preflight, not by training. Numerical histories
and labels are exported only for the **16 train / 8 validation CVEs** in the
bounded preview. Preview selection is a fixed ID hash, independent of scores.

### Verified Readiness Findings

The separate review is stored in
`05_results/02_evaluation/epss_day60_to_day90/preview_review_v1/`,
relative to `EPSS_TPG`, as `analysis.txt` and `analysis.json`.
Original preparation files remain unchanged.

Of the 121 ineligible training CVEs, 111 lack both a recent input and a
Day-90 target, six lack only the target, and four lack only a recent input.
The one ineligible validation CVE lacks both. These are disjoint counts;
adding the original overlapping missing-input and missing-target counters
would double-count some CVEs.

Compared with R07, training retains 1,012 CVEs, gains 17 and loses 11.
The losses are five missing Day-90 targets, five label-boundary crossings
and one related-group conflict. Validation retains 166, gains seven and
loses 31: 28 label-boundary crossings and three related-group conflicts.
These are deliberate eligibility and maturity consequences, not dropped
records caused by a model.

| Split | Input days | Eligible CVEs | All input scores present | Some input scores missing | Fewest observed scores |
|---|---|---:|---:|---:|---:|
| Training | 30-60 (31 days) | 1,029 | 787 | 242 | 4 |
| Training | 0-60 (61 days) | 1,029 | 77 | 952 | 4 |
| Validation | 30-60 (31 days) | 173 | 166 | 7 | 27 |
| Validation | 0-60 (61 days) | 173 | 37 | 136 | 50 |

The longer window adds older days; it does not remove recent observations.
Twelve training CVEs gain no observed scores from Days 0-29. Partial
histories remain eligible with their masks and observation ages; this review
does not introduce a new completeness threshold.

| Source in the 24-CVE preview | Training CVEs with text / passing screen | Validation CVEs with text / passing screen |
|---|---:|---:|
| Description | 16 / 16 | 8 / 8 |
| GitHub summary | 3 / 3 | 0 / 0 |
| Non-GitHub summary | 8 / 8 | 8 / 8 |
| Social-media posts | 16 / 14 | 8 / 4 |

The validation preview cannot measure GitHub-summary benefit because it has
none. Audit the full development cohort before source comparisons; do not
resample this preview to obtain favorable coverage.

All 34 stored post-date entries omit timezone information; 22 precede the
anchor and 12 follow it under the existing date interpretation. Two Telegram
text records have conflicting dates: `CVE-2022-22776` and `CVE-2024-35250`.
Both stay excluded from the screened candidate view. These might represent
reposts or collection issues; conflicting dates alone do not establish which.
No preview summary contains the checked phrases "loading message", "css error"
or "no technical details". This narrow check does not establish factual quality.
All text remains historically unverified.

**Decision:** structurally ready for full matched development preparation,
not evidence that later forecasting or longer history improves prediction.
No model-performance metrics were calculated.

### Text and Planned Controls

The source adapter is reused with Day 60 as its cutoff. Descriptions and
summaries remain retrospective; posts need consistent supplied dates
strictly before Day 60. Passing the screen does not authenticate publication.
The new preview is not the same sample as MS01, so different source counts
do not establish the effect of moving the anchor.

First establish matched EPSS-only recent-versus-longer-history controls
and persistence on this new task. Source-aware TPG+EPSS variants follow
the same task and cohort contract; they do not reuse Day-30 source masks.
The old fixed-31-day tensor loader is not a valid loader for the 61-day
history. A new tested tensor/training contract is a subsequent step.

Tests cover future-score isolation, early-history separation, zero versus
missing values, within-window age calculation, exact sequence lengths,
target maturity, related-group exclusions, reserved-test access rejection,
Day-60 source screening, output reproduction and tamper protection.

### Commands

Run individually from the repository root with `CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_later_anchor.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_later_anchor.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_later_anchor.py
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_later_anchor.py --verify-only
```

Output: `04_data/01_records_and_graphs/epss_day60_to_day90_v1/preview/`,
relative to `EPSS_TPG`. The original `epss_day30_to_day60` directory is not
modified. No full-export or training mode is exposed in this CLI.

| File | Purpose |
|---|---|
| `inputs.json` | Both past-history variants for the 24 preview CVEs; no future scores |
| `labels.json` | Their separately stored observed Day-90 targets |
| `source_inputs.json` | Source-aware text payload and masks at Day 60 |
| `source_audit.json` | Text/date provenance and quality flags; not a direct model input |
| `split_manifest.json` | Dates, group IDs and reassigned roles for all CVEs; no score/text values |
| `cohort_audit.json` | Train/validation eligibility and observation counts, not numerical target values |
| `analysis.json` / `analysis.txt` | Counts, definitions, restrictions and next steps |
| `protocol.json` | Fixed task, window and split definitions |
| `run_identity.json` | Preview/full-development IDs and source/code hashes |
| `preparation_lock.json` | Checksums for the ten output artifacts |

Implementation: `02_prediction/01_package/epss/forecast_later_anchor.py`.
Protocol: `02_prediction/04_configs/day60_to_day90_preview_v1.json`.
After the user runs and verifies this preview, review the exact cohort
changes before preparing the full matched development data and training
controls. That review is now complete. The next stage is matched development
preparation and a tested variable-length loader, followed by fixed EPSS-only
controls; multi-source TPG fusion remains planned. Final test remains closed.

To reproduce the saved review without changing the preparation:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_later_anchor.py --verify-only
```

## MS01: Bounded Multi-Source Adapter

**Decision:** keep the final test closed while completing the originally
planned multi-source development experiments. R07 completed description
plus EPSS training, not the entire multi-source research plan.

**Status:** user preview complete and independently reproduced. Its six
artifacts and locked source/code identities match. The separate review
did not alter the preview, generate graphs/embeddings or train models.
Eleven adapter tests and five new window-analysis tests pass.

### Preview Review

The 16 training / 8 validation preview CVEs retain description coverage
16/8, GitHub-summary coverage 5/3 and non-GitHub-summary coverage 12/6.
Only 14/16 training and 1/8 validation CVEs retain SMP after the stored-date
screen. These are preview counts, not full-development estimates.

Two non-GitHub summaries, for **CVE-2022-28706** and **CVE-2024-22024**,
describe a generic loading message and CSS error rather than useful
vulnerability evidence. This demonstrates why nonempty summary text alone
is not a quality check. The review flags them for source inspection but
does not alter their masks or claim that every bad summary is detected.
No websites were re-fetched.

For CVE-2022-28233 and CVE-2022-30974, identical Telegram text has supplied
dates `2025-04-28T00:29:00` and `2025-04-30T19:01:00`, both after the
current anchor. Identical wording can be reposted; a date conflict does not
prove corrupt data. Both associations remain in the audit and are excluded
from the date-screened predictor view.

Maximum candidate document lengths in this training preview are 1,194
description characters, 1,941 GitHub-summary characters, 2,303 non-GitHub
summary characters and 4,359 post characters. These are not token counts.
Future graph preparation must handle long documents explicitly.

### Later-Window Feasibility

The user asked whether observing Days 30-60 instead could make Day-90 EPSS
easier to predict. We compared stored scores on the **original training
CVEs only**, without fitting or evaluating a model. Of 1,023 training
CVEs, five have Day 90 reaching the validation period and are excluded
before analyzing their scores. Of the remaining 1,018, five lack Day 90.
No original split or experiment target was changed.

For daily variability, 1,011 matched CVEs have at least 20 observed adjacent
day pairs in **each** of the three windows. Missing days are not filled
and daily changes are not calculated across gaps.

| Observed window | Matched CVEs | CVEs with at least one observed daily change | Mean absolute daily EPSS change |
|---|---:|---:|---:|
| Days 0-30 | 1,011 | 447 | 0.0004678342 |
| Days 30-60 | 1,011 | 94 | 0.0001267535 |
| Days 60-90 | 1,011 | 65 | 0.0000857510 |

The last column first averages absolute observed day-to-day changes within
each CVE, then across CVEs. Median per-CVE movement is zero in all windows.
Days 30-60 are more variable for 51 CVEs; Days 0-30 are more variable for
419; 541 are equal. This does not support a general increase in variability
during Days 30-60 in this training cohort.

**Sensitivity matters:** only 41 CVEs have all 91 observations. In that
small complete subset, mean daily movement is higher during Days 30-60
(0.00106845 versus 0.00073998), although only five CVEs have more middle-window
movement and 32 have more early-window movement. Large changes can dominate
means; neither a selected complete subset nor the main partial-history
comparison establishes a universal temporal pattern.

There are 34 in-period CVEs with a scheduled EPSS-version transition in
Days 0-90. Excluding them leaves 982 CVEs meeting the adjacent-pair rule:
41 have greater middle-window movement, 405 greater early-window movement
and 536 equal movement. The result does not disappear under this sensitivity.
Version dates are schedule metadata, not independently verified archive
version headers.

A **separate matched set of 1,013 CVEs** has observed Day-60 and Day-90
endpoints and usable inputs. It need not satisfy the adjacent-pair rule.

| Available input history | Target | Horizon from input cutoff | Unchanged / increased / decreased | Persistence MAE on these training records |
|---|---|---|---|---:|
| Days 0-30 | Day 60 | 30 days | 916 / 76 / 21 | 0.00256757 |
| Days 30-60 | Day 90 | 30 days | 934 / 49 / 30 | 0.00287774 |
| Days 0-30 | Day 90 | 60 days | 860 / 111 / 42 | 0.00516371 |

Persistence retains the last observed input score. Its later-interval MAE
is **12.08% higher**, despite more unchanged endpoints. These are descriptive
training-record reference errors, not held-out forecast results or evidence
that a learned later-window model must fail.

Example: **CVE-2023-28432** has last observed input EPSS 0.01489 through
Day 30, then 0.42964 on both Day 60 and Day 90. Waiting until Day 60 makes
that later target easy for persistence, but does not forecast the earlier
increase. Observing a jump is not the same as predicting it in advance.

**Decision:** Day-60-to-Day-90 is a feasible separate development question,
not an automatic improvement and not a replacement for R07. A new task
would require anchor-based split/group reassessment, label-maturity rules,
Day-60 text-availability masks and fixed matched baselines. A Days 0-60
history is an additional longer-history ablation. Scores beyond Day 90
are not available in this source window.

Source review, sensitivity tables, direction transitions and per-CVE
training-window statistics are saved in:
`05_results/02_evaluation/epss_day30_to_day60/multisource_window_review_v1/analysis.txt`
and `analysis.json`.

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_multisource_forecast_windows.py --verify-only
```

No training, new model predictions, validation-window outcome comparison,
test analysis or split changes were performed. Original MS01/R07 artifacts
remain unchanged.

### Inputs and Timing

The adapter reads the same hash-verified source dataset as R07. It does not
silently merge revised 50-record summary samples or scrape current pages.

| Source label | Exact field | Candidate-view rule |
|---|---|---|
| `description` | `description` | Preserve current text; mark historical availability unverified |
| `github_summary` | `summ_github_only_urls_fast.gpt-oss:120b` | Preserve current summary; no authenticated source-content snapshot |
| `non_github_summary` | `summ_non_github_urls_fast.gpt-oss:120b` | Same retrospective restriction |
| `smp` | `social_media_posts[].post` | Retain only posts with consistent stored dates strictly before Day 30 |

The default view is **retrospective_date_screened**, not historical
forecasting. A date screen cannot authenticate a post's original publication.
Undated descriptions and summaries remain retrospective even when late
posts are excluded. The strict `historical_verified` view has no text
enabled; no supplied field alone promotes a document to verified evidence.

Day-30 posts are excluded because within-day ordering is not established.
Missing, invalid, later or conflicting dates also exclude that post from
the candidate view. Aware timestamps are normalized to UTC; naive timestamps
keep an unknown-timezone flag. Exact same-platform/text posts within a CVE
are combined with all provenance and dates retained, never with an
earliest-date shortcut. Different timestamps for the same text are flagged
even if they fall on the same day. Cross-CVE identities/conflicts still
need the future full-development source audit; this is a bounded preview.

Each CVE has separate **source presence** and **view eligibility** masks.
Empty or unavailable text never becomes invented wording, and no CVE is
dropped just because a source is absent. `model_input()` applies the
source-removal condition to both documents and their effective masks.
Metadata dates, raw URL lists, structured CVSS, VulnCheck and future EPSS
values do not cross that input boundary. Text can still mention later
exploitation, so this allowlist is not a guarantee against textual leakage.

### Bounded Preview

Select **16 training / 8 validation CVEs** by a fixed ID hash within the
existing 1,023/197 cohorts, without inspecting outcomes. Test CVEs are
not selected or analyzed. Shared preparation integrity checks still validate
the frozen input files; they do not evaluate test performance.

The real-data preflight found the following CVEs with candidate text:

| Source | Training preview (16 CVEs) | Validation preview (8 CVEs) |
|---|---:|---:|
| Description | 16 | 8 |
| GitHub summary | 5 | 3 |
| Non-GitHub summary | 12 | 6 |
| SMP after stored-date screen | 14 | 1 |

All 24 preview CVEs have stored posts before screening. These small-sample
counts must not be extrapolated to the full cohort. Timing masks can remove
most validation SMP inputs; no change of cohort is authorized to improve
their apparent availability.

Tests cover partial/empty sources, ablation masks, missing and late dates,
timezone normalization, same-text date conflicts, duplicate retention,
target-field isolation, strict field types, deterministic selection,
test-role rejection, saved-preview reproduction and tamper detection.

### Planned Comparisons

All variants retain the same CVEs and the same observed EPSS Days 0-30:

1. EPSS-only.
2. Description TPG + EPSS.
3. GitHub-summary TPG + EPSS.
4. Non-GitHub-summary TPG + EPSS.
5. SMP TPG + EPSS.
6. All four text sources + EPSS.
7. All sources minus description.
8. All sources minus GitHub summaries.
9. All sources minus non-GitHub summaries.
10. All sources minus SMP.

These configurations are defined in one protocol file, not runnable
training yet. A changed fusion architecture needs newly matched controls;
do not attribute an R07-versus-new-model difference solely to extra text.
Keep comparable seed schedules, optimizer/loss/budget and checkpoint rules.
No complete-case-only filtering or best-seed reporting.

Future graphs must use the fixed security TPG, independent document
namespaces and `source_text_type` on nodes and edges. Pool within each
source before source-level fusion with EPSS. Missing sources are masked.
No invented links across documents. Document/token budgets must be
established from training-only evidence and documented before fitting.

### Commands and Artifacts

Run separately from the repository root with `CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_multisource.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_multisource.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_multisource.py
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/prepare_forecast_multisource.py --verify-only
```

Protocol: `02_prediction/04_configs/multisource_source_adapter_v1.json`.
Implementation: `02_prediction/01_package/epss/forecast_multisource.py`.
Output: `01_tpg/05_workspace/multisource_development_v1/source_preview/`,
all relative to `EPSS_TPG`.

| File | Purpose |
|---|---|
| `records.json` | Audit records, exact source text, dates, provenance, flags and both kinds of masks; not direct model input |
| `model_inputs.json` | Candidate allowlisted text documents and effective masks; CVE/split are routing metadata |
| `analysis.json` / `analysis.txt` | Source coverage, date-screen counts, duplicate/conflict counts and explanations |
| `protocol.json` | Timing policy, future graph/fusion contract and the ten matched source conditions |
| `run_identity.json` | Frozen source/code hashes and selected IDs |
| `preview_lock.json` | Checksums for all six output artifacts |

The source preview and separate review are complete. The next multi-source
implementation stage is a **small
multi-source graph preview**, not full training. The adapter does not yet
parse text, run SecBERT, construct graph tensors or fit a fusion model.
Final test evaluation remains deferred until the multi-source development
decision and final model/evaluation protocol are frozen.

## R07: Full Eligible-Cohort MAE Training

**Status:** user training complete; all 50 locked artifacts and six
checkpoint predictions independently verified. Review did not retrain,
generate embeddings or score test CVEs. Original results remain unchanged.

### Verified Results

| Model/reference | Mean validation MAE | Sample SD across seeds |
|---|---:|---:|
| Persistence | 0.00909675127 | Not seeded |
| Scalar correction (offset zero) | 0.00909675127 | Not seeded |
| EPSS-only | 0.00909686292 | 0.00000010352 |
| No-COREF TPG+EPSS | 0.00909687284 | 0.00000007313 |

MAE measures average absolute EPSS score error on the 0-1 scale; lower is
better. The very small differences do not establish useful graph benefit.
Every learned fit has worse MAE than both the original persistence reference
and its float32 equivalent. Thus the result is not solely a baseline
numerical-representation difference.

| Model | Seed | Selected epoch | Epochs completed | Validation MAE |
|---|---:|---:|---:|---:|
| EPSS-only | 17 | 20 | 35 | 0.00909679947 |
| EPSS-only | 29 | 8 | 23 | 0.00909680690 |
| EPSS-only | 43 | 13 | 28 | 0.00909698238 |
| TPG+EPSS | 17 | 16 | 31 | 0.00909680972 |
| TPG+EPSS | 29 | 4 | 19 | 0.00909695298 |
| TPG+EPSS | 43 | 7 | 22 | 0.00909685583 |

All fits obey patience 15 and stop before the 100-epoch cap. Matched R04 to
R07 mean MAE decreases from 0.00921877690 to 0.00909686788, a **1.322%**
relative improvement across both models/seeds. All six pairs improve.
This removes harmful corrections; it does not beat simply retaining the
last observed score. Graph-minus-EPSS-only MAE is positive at seeds 17/29
and negative at 43, with tiny differences in every case.

Five fits have exactly one learned correction over the 197 validation
CVEs. Graph seed 29 has two values separated by only `2.842e-14`.
Mean corrections range from approximately `-1.124e-7` to `3.230e-7`.
Predictions vary because the persistence input varies, not because the
learned correction meaningfully distinguishes CVEs on this validation set.

Temporal, graph-projector and first-head-layer norms shrink strongly.
Nonzero data-loss gradients occur during training, but that does not prove
effective use of those branches by the selected model. The review does
not isolate the cause of shrinkage, and aggregate gradient counts include
epochs after checkpoint selection. No training or selected-validation
output clipping is recorded.

**Real examples:** 924/1,023 training and 162/197 validation endpoints equal
the last observed input score. Equality of endpoints does not imply a
completely flat trajectory between them.

| CVE | Last observed input score | Actual Day-60 score | Share of persistence absolute error |
|---|---:|---:|---:|
| CVE-2024-23692 | 0.00210 | 0.95432 | 53.14% |
| CVE-2024-8963 | 0.30991 | 0.95343 | 35.91% |
| CVE-2023-7028 | 0.04478 | 0.20153 | 8.75% |

The first two cases account for 89.05% of persistence's total absolute
error. Every R07 model remains close to the input score on these cases.
They are retained. This helps explain why RMSE stays near 0.0826445 despite
the smaller MAE; neither metric establishes reliable large-change prediction.

Partial histories are represented: validation has 38 CVEs with all 31
input scores, 130 with 30 scores and 29 with at most 29. Persistence MAEs
are respectively 0.00015000000, 0.01246200000 and 0.00573448276.
Both model types' seed-mean MAEs are slightly worse in every group.
Groups contain different CVEs, so this is not a causal missingness estimate.

Detailed TXT/JSON review:
`05_results/02_evaluation/epss_day30_to_day60/residual_mae_development_v1_review/`.
It retains all six fit histories, gradients, metrics, case errors, missingness
comparisons, parameter-norm ratios and source hashes.

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_mae_development.py --verify-only
```

**Original post-R07 recommendation:** freeze a final test protocol reporting
persistence, the scalar reference and all six learned fits on identical eligible test CVEs.
Do not select a best seed, extend epochs to seek a win, or train on test
records. Test graph preparation and scoring require a separate authorized
stage. A future redesign is a new experiment, not a reinterpretation of
these negative validation findings.

**Subsequent scope decision:** the user has requested the planned multi-source
development stage before final testing. Follow MS01 above. R07 remains an
unchanged description-only graph comparison, not a completed test of all text.

### Cohort Accounting

"Full" means all eligible CVEs in the existing training split, not the entire
source file. The current task uses observed EPSS scores from Days 0-30 to
predict the score on Day 60, counting from NVD publication.

| Current use | CVEs | Why |
|---|---:|---|
| Eligible training | 1,023 | Used to update model weights |
| Eligible validation | 197 | Used to select the checkpoint |
| Eligible reserved test | 2,670 | Held for separate final evaluation |
| Assigned to train/validation/test but missing required observations | 136 | No observed input and/or no Day-60 target |
| Excluded by the frozen split policy | 1,942 | Related-group conflicts, boundary-crossing targets or anchors outside the period |
| Total source | 5,968 | No source records deleted |

Partial input histories are already allowed: at least one of the 31 input
scores and an observed Day-60 target are required. Missing input days retain
masks and observation-age features. All 91 scores are not required. Missing
future targets are not fabricated. Among policy-excluded CVEs, 1,726 have
sufficient observations; missing EPSS alone does not explain their exclusion.

The prediction anchor is Day 30. Training anchors and target dates precede
2024; validation anchors are in 2024 with targets before 2025; test anchors
are in 2025. Related-group quarantine accounts for 1,548 exclusions. Shared
links can create broad groups, so this is a conservative policy, not proof
that every grouped CVE is a duplicate. A separately justified future policy
revision can recover data, but requires new cohorts and matched comparisons.
See the saved DQ07 coverage audit for the complete accounting.

### Fixed Training Design

- Models: EPSS-only and no-COREF TPG+EPSS, both persistence-plus-correction.
- Seeds: 17, 29, 43, giving six fresh fits over the same 1,023 training CVEs.
- Loss: MAE; Adam learning rate `0.001`, weight decay `0.0001`.
- Budget: up to 100 epochs, patience 15, batch size 8, CPU.
- Correction gain: `0.00885`; model sizes, cached embeddings and edge
  configuration are unchanged.
- Select the earliest strictly best validation-MAE checkpoint. Epoch zero
  is not selectable. All seeds are reported, not only the best seed.
- References: persistence, a scalar correction fitted on training data,
  and the matched full-cohort R04 Smooth L1 results. R06's smaller-cohort
  errors are not directly comparable to this full-cohort error.

The preflight checks frozen sources, all full-cohort graph tensors, disjoint
training/validation groups, matched R04 initial states and epoch schedules,
and persistence initialization on ordinary and large training graphs.
Tests cover exact short-run R06 MAE parity, early stopping/ties, deterministic
reloads, input/target separation, partial histories, resume and tamper checks.

### Run and Verify

The user run is complete: use `--verify-only` now. Original execution
commands are retained below. Run individually from the repository root
with `CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_mae_development.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_mae_development.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_mae_development.py
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_mae_development.py --verify-only
```

Only after interruption use `--resume`: completed fits are checked and
skipped; the unfinished fit restarts from its fixed initialization.
Completed runs accept `--verify-only`, not `--resume`.

Protocol: `04_configs/residual_mae_development_v1.json`.
Runner: `03_analysis/01_revision_dataset/run_forecast_mae_development.py`.
Implementation: `01_package/epss/forecast_mae_development.py`.
Results: `05_results/02_evaluation/epss_day30_to_day60/residual_mae_development_v1/`
relative to `EPSS_TPG`. Read `summary.txt`, then `summary.json`.
The final lock covers 50 artifacts: identities, handoff, baselines, initial
states, batch schedules, checkpoints, histories, gradient counters,
validation predictions, metrics and per-fit locks.

### After This Run

Review all six fits on the same 197 validation CVEs and compare against
persistence. Freeze final model and evaluation choices before a separately
authorized test evaluation on the 2,670 eligible reserved CVEs. Do not
train on those test CVEs first. A later deployment refit is a different
activity and cannot be evaluated on records it used for fitting.

Current descriptions are retrospective snapshots: their availability at
each historical prediction anchor is not established. R07 preserves that
limitation and does not claim a fully time-safe text forecast. Repeated
validation comparisons also limit how independently validation can assess
generalization. No model benefit is assumed before results are reviewed.

## R06: Final Bounded Loss Comparison

**Status:** user run complete; all 80 artifacts, 12 checkpoint predictions
and six exact Smooth L1 control reproductions independently verified.
The review did not retrain, regenerate embeddings, change splits or score
the test set. Original result files remain unchanged.

This is the **last planned bounded comparison before full development
training**, as requested. It changes the training loss only. No further
bounded loss/decay search is planned after a valid, verified run.

### Verified Results

| Model | Training loss | Three-seed mean validation MAE | Sample SD |
|---|---|---:|---:|
| Persistence | Not trained | 0.01494125000 | Not seeded |
| Scalar correction (offset zero) | Training residual median | 0.01494125000 | Not seeded |
| EPSS-only | Smooth L1 | 0.01512880375 | 0.00005249121 |
| EPSS-only | MAE | 0.01494139428 | 0.00000010540 |
| No-COREF TPG+EPSS | Smooth L1 | 0.01516128976 | 0.00004774360 |
| No-COREF TPG+EPSS | MAE | 0.01494129092 | 0.00000001447 |

MAE loss improves all six matched model/seed comparisons. The mean over both
models and all seeds falls from 0.015145046755 to 0.014941342601, a 1.345%
relative reduction. The preregistered rule therefore selects **MAE** as the
one common loss for full development training.

**This is not a persistence win.** All 12 learned fits still have higher
MAE than persistence; the six MAE-trained excesses are only approximately
`2.60e-8` to `2.58e-7`. The improvement over Smooth L1 primarily removes
unnecessary corrections rather than demonstrating useful forecasting.
Average absolute corrections in the MAE fits are about `3.08e-8` to
`3.06e-7`. The small graph-versus-EPSS-only difference is not evidence of a
practically useful graph benefit.

The same 57/64 validation endpoints remain unchanged. CVE-2024-23692 still
rises from 0.00210 to 0.95432 and contributes 99.58% of baseline absolute
error. MAE-model predictions are about 0.00209987-0.00210031, so the large
increase is not captured. The case is retained, not removed or used to
change the split.

MAE selected epochs are 14/12/25 for EPSS-only and 17/15/28 for graph models
at seeds 17/29/43. All 30 epochs run for every fit. No MAE-training output
clipping or selected-validation output clipping is recorded. Encoder
gradients are nonzero in 478 or 479 of 480 steps, so this is not an entirely
disconnected branch.

Original decay still shrinks weights; the selected MAE graph input-projector
norms retain only about 0.011-0.834% of initialization. These checkpoints are
later than the Smooth L1 checkpoints, so this is not a matched-step causal
norm comparison. Do not use it to reopen the finished decay sweep.

**Next stage:** full eligible development training using the already defined
MAE handoff: 1,023 training / 197 validation CVEs, EPSS-only and no-COREF
TPG+EPSS, seeds 17/29/43, fresh initialization, original decay `0.0001`,
gain `0.00885`, up to 100 epochs and patience 15. No further bounded
comparison is needed. The separate R07 full-training run above is now
complete and verified; the automated review did not retrain it. Keep the test set
closed and report persistence alongside every model.

Detailed TXT/JSON review:
`05_results/02_evaluation/epss_day30_to_day60/residual_loss_comparison_v1_review/`.
It retains every fit's metrics, case-level predictions/errors, histories,
gradient counters, norm ratios, source hashes and the unchanged decision.

Optional reproduction of the saved interpretation, without new predictions:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_loss_comparison.py --verify-only
```

### Fixed Comparison

| Component | Setting |
|---|---|
| Cohort | Exact R05 128 training / 64 validation IDs, unchanged groups and split roles |
| Models | EPSS-only and no-COREF TPG+EPSS |
| Losses | MAE (`L1Loss`) versus Smooth L1 (`beta=0.01`), both mean-reduced |
| Weight decay | Original Adam `0.0001` for every fit |
| Seeds | 17, 29, 43; 12 fresh fits in total |
| Budget | All 30 epochs; batch eight; 480 updates per fit; no early stopping |
| Selection | Earliest strictly lowest validation MAE at epochs 1-30; epoch zero excluded |
| Unchanged settings | Learning rate 0.001, gradient clip norm one, dropout zero, residual gain 0.00885 |
| Initial state and orders | Identical to R05 original-decay counterpart for each model/seed |
| Graphs / embeddings | Existing fixed cache; no regeneration |
| References | Same R05 persistence and training-fitted scalar correction |
| Execution | CPU, one Torch thread, deterministic algorithms |

The graph architecture, feature set, forward function and targets are
unchanged. MAE loss uses absolute prediction error directly; Smooth L1
reduces the gradient for small errors through its quadratic region.
This intentionally changes the learning signal, not the target.
Training-objective values cannot be compared numerically across the two
losses as if they were the same metric. Validation MAE remains the common
primary measure for every model.

The primary comparison is **MAE-trained minus Smooth-L1-trained validation
MAE**, paired by model and seed. Negative values favor MAE training.
Retain all seeds, RMSE, correction variability, parameter norms, gradient
counters, histories and clipping counts. No best-seed selection.

### Control Reproduction

Frozen R04/R05 modules, protocols, caches and outputs are not edited.
R06 uses a separate version of the fixed-epoch loop that changes only the
criterion and names the stored loss field `training_objective_loss`.
Each new Smooth L1 control must exactly reproduce the matched R05
original-decay checkpoint tensor state, selected epoch, history values,
gradients, predictions and metrics. The renamed history field is mapped
back to `training_huber_loss` only for this equality check.

A failed control comparison stops the experiment. Fix a technical problem
without relaxing the parity requirement or changing the scientific protocol.
Synthetic tests already establish exact algorithm parity for both model
types. Real-data full-control parity is checked when the user runs training;
the preflight does not retrain a control.

### Full Training Next

The decision rule is fixed before R06 training:

1. Average the six selected-checkpoint validation MAEs for each loss:
   both models and all three seeds receive equal weight.
2. Choose MAE only if its average is more than `1e-12` lower than Smooth L1.
   Otherwise retain Smooth L1. Use one common loss for both model types.
3. After the complete comparison and its control checks are verified, move
   to the existing **1,023 training / 197 validation CVEs**, with the same
   two models and seeds 17/29/43.
4. Full development training uses fresh fixed-seed initialization, no
   R05/R06 warm start, original decay `0.0001`, gain `0.00885`, up to 100
   epochs and patience 15, as in the preceding full-development budget.
   Keep the original chronological/grouped splits and leave the test set closed.

The decision is saved as `full_training_decision.json`. Full training is
not launched automatically, and its runner is a separate next stage.
There is no additional bounded tuning gate requiring the learned model
to beat persistence before that full development experiment. However,
baseline superiority must still be reported honestly: choosing a loss
does not establish a winning model, statistical significance or deployment
readiness. Persistence remains a valid reference even when every trained
checkpoint performs worse. Full training means all currently eligible
development CVEs, not all raw CVEs or the reserved test cohort.

### Files And Commands

- CLI: `02_prediction/03_analysis/01_revision_dataset/run_forecast_loss_comparison.py`.
- Implementation: `02_prediction/01_package/epss/forecast_loss_comparison.py`.
- Protocol: `02_prediction/04_configs/residual_loss_comparison_v1.json`.
- Tests: `02_prediction/03_analysis/01_revision_dataset/test_forecast_loss_comparison.py`.
- Output: `05_results/02_evaluation/epss_day30_to_day60/residual_loss_comparison_v1/`.

The R06 output is complete: use **`--verify-only`**, not another training run.
The original execution commands are retained below for reference.
Run individually from the repository root with `CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_loss_comparison.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_loss_comparison.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_loss_comparison.py
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_loss_comparison.py --verify-only
```

Only after interruption:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_loss_comparison.py --resume
```

Recovery verifies and skips completed fits, restarting an unfinished fit
from fixed initialization. Completed outputs refuse retraining/resume;
use `--verify-only`. Verification reloads saved checkpoints and reproduces
validation predictions and R05 control parity without training.

Outputs include run identity/source hashes, shared seed schedules,
baselines, all 12 fit folders, TXT/JSON summaries and the full-training
decision. Per-fit files contain checkpoints, histories, gradients,
predictions, metrics and locks. `evaluation_lock.json` protects the complete
80-artifact result set. Prediction arrays follow the validation IDs in
`run_identity.json`. Progress bars cover source checks, fits, epochs,
batches and prediction verification.

Tests cover fixed settings, the actual loss functions and zero-error
derivative, exact R05 loop parity, deterministic MAE fitting and reloads,
input/target separation, complete-result loss selection and tie handling,
preflight without training, interrupted recovery, immutable completed
outputs and decision-file tampering.

## R05: Bounded Weight-Decay Comparison

**Status:** user training complete. All 79 locked artifacts and all 12
checkpoint predictions independently verified. Seven synthetic tests and
the real-data preflight passed before training. The automated review reproduced
saved validation predictions but did not retrain or score the test set.

This is the controlled comparison motivated by DQ19. It is not full training,
does not change the original R04 experiment, and does not access the test set.

### Verified Findings

| Model | Weight decay | Three-seed mean validation MAE | Sample SD |
|---|---:|---:|---:|
| Persistence | Not applicable | 0.01494125 | Not seeded |
| Training-fitted scalar correction | Not applicable | 0.01494125 | Not seeded |
| EPSS-only | 0.0001 | 0.01512880 | 0.00005249 |
| EPSS-only | 0 | 0.01517969 | 0.00006894 |
| No-COREF TPG+EPSS | 0.0001 | 0.01516129 | 0.00004774 |
| No-COREF TPG+EPSS | 0 | 0.01519152 | 0.00000669 |

The fitted scalar offset is zero. **All 12 learned fits lose to persistence
on MAE.** Removing decay worsens MAE in five of six model/seed pairs; only
graph seed 29 improves. Each selected RMSE is slightly better than
persistence, but this is not evidence of superiority on the primary MAE
criterion. There is no consistent graph-fusion advantage.

All 30 epochs ran: 480 updates per fit. Ten fits select epoch one; both
seed-43 graph fits select epoch two. The paired conditions happen to select
the same epoch for each model/seed, so their selected-norm comparisons here
are at equal update counts.

**What decay changed:** original decay leaves selected EPSS encoder norms at
about 82-91% of initialization, versus approximately 100% without decay.
The graph input projection retains about 18-45% with original decay and
101-104% without it. Thus removing decay preserves weight magnitude, but
does not reliably improve forecasting. Zero-decay EPSS-only corrections
still have very little CVE-specific variation.

**What the curves show:** graph training error decreases while validation
error later increases, consistent with overfitting on this small subset.
For example, no-COREF seed 17 ends with training/validation MAE
0.00348265/0.01658905 under original decay and 0.00214157/0.01883897 under
zero decay. These are epoch-30 history values, not selected-model scores.
EPSS-only Smooth L1 loss falls slightly while training MAE generally rises:
the training objective and the main evaluation metric have different
tradeoffs. This is not proof that changing the loss will solve the problem.

**Target distribution:** 115/128 training CVEs and 57/64 validation CVEs have
unchanged endpoints relative to persistence. The validation set has five
increases and two decreases. CVE-2024-23692 rises from 0.00210 to 0.95432 and
accounts for 99.58% of persistence's total absolute error. The selected models
predict only about 0.00226-0.00253 for that CVE. Small, mostly positive
corrections improve only three or four cases per model and worsen 60 or 61.
Do not remove this CVE or alter splits based on these outcomes.

Gradients are nonzero in each encoder for 478 or 479 of 480 updates:
the branches are not entirely disconnected. Selected validation predictions
have no clipping. Later graph training does have clipping: original-decay
counts are 39/33/44 and zero-decay counts 33/19/28 for seeds 17/29/43.
These are repeated sample evaluations, not distinct CVEs. No EPSS-only
training clipping is recorded.

**Decision:** do not promote zero decay to full training. A next candidate is
a controlled loss-only comparison of MAE loss versus current Smooth L1,
keeping original decay and matched EPSS-only/graph controls. Keep persistence
as a valid reference and do not force a learned model to replace it. This is
a proposal at the time of review, not a guaranteed fix. R06 above now
implements the loss comparison as the final bounded stage before full
development training.

Detailed review TXT/JSON:
`05_results/02_evaluation/epss_day30_to_day60/residual_decay_comparison_v1_review/`.
It includes all fit metrics, case-level errors, full histories, gradient
counters, norm ratios and limitations. The original result folder is unchanged.

Review reproduction, without training or new model predictions:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_decay_comparison.py --verify-only
```

### Fixed Design

| Choice | Setting |
|---|---|
| Training subset | 128 CVEs from the existing 1,023 eligible training CVEs |
| Validation subset | 64 CVEs from the existing 197 validation CVEs |
| Selection | Fixed CVE-ID hash ordering separately within the frozen splits; independent of labels |
| Models | EPSS-only and no-COREF TPG+EPSS |
| Decay comparison | Original Adam weight decay `0.0001` versus `0` |
| Seeds | 17, 29, 43 for every model/condition: 12 fits |
| Budget | All 30 epochs, batch eight: 480 optimizer steps per fit; no early stopping |
| Checkpoint | Earliest strictly lowest validation MAE from epochs 1-30; epoch zero excluded |
| Unchanged settings | Adam learning rate 0.001, Smooth L1 beta 0.01, clip norm one, dropout zero |
| Residual gain | Fixed at 0.00885, inherited from R04's training-input-only calculation |
| Inputs / target | EPSS Days 0-30; description graph for fusion; stored Day-60 EPSS target |
| Reference predictions | Persistence and a constant correction fitted only to the 128 training targets |
| Execution | CPU, one Torch thread, deterministic algorithms; existing shared graph embeddings |

No-COREF is chosen in advance as the graph control with no coreference edges,
not as the best graph mode from previous outcomes. All other cached graph
features remain as in R04. Related groups must remain separate between the
training and validation subsets. No chronology or eligibility rule is relaxed.

For each model/seed, the two decay settings start from identical parameter
values and use identical epoch orders. Both are trained freshly, not resumed
from R04 or smoke checkpoints. Thirty fixed epochs give the conditions equal
total update counts. Selected checkpoints may nevertheless come from
different epochs; interpret their norm comparisons accordingly.

The scalar reference uses the median of `training Day-60 EPSS - persistence`
and clips corrected predictions to [0,1]. This fits an un-clipped residual MAE
criterion, not the neural Smooth L1 objective. It may equal persistence if the
median correction is zero. No validation targets are used to fit this scalar.

### Readout And Decision

The primary comparison is **zero-decay minus original-decay validation MAE**,
paired by model and seed. Negative values favor zero decay. Report every seed,
three-seed mean MAE and sample standard deviation, without choosing a best seed.
Also inspect RMSE, correction variability, parameter norms at initialization
and the selected checkpoint, gradient counters and output clipping.

Compare all models to persistence and the scalar reference on the same 64
validation CVEs. Do not compare this subset's MAE directly to the previous
197-CVE full validation MAE. These are exploratory development results, not
independent confirmation: validation has already informed prior decisions.
Current description graphs still carry the historical-availability limitation.

Expansion is a separate decision after review. Lower decay may preserve
encoder signal without improving prediction. Do not automatically proceed
to a full run or open the test set.

### Files And Commands

- CLI: `02_prediction/03_analysis/01_revision_dataset/run_forecast_decay_comparison.py`.
- Implementation: `02_prediction/01_package/epss/forecast_decay_comparison.py`.
- Protocol: `02_prediction/04_configs/residual_decay_comparison_v1.json`.
- Tests: `02_prediction/03_analysis/01_revision_dataset/test_forecast_decay_comparison.py`.
- Output: `05_results/02_evaluation/epss_day30_to_day60/residual_decay_comparison_v1/`.

The package reuses the frozen R04 fixed-epoch training algorithm, with a
separate protocol and result identity. The original extractor, graph cache,
runner, checkpoints, splits and reports are not edited.

The output is now complete: **use `--verify-only`**, not another training
command. Original execution commands are retained below for reference.
Run individually from the repository root with `CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_decay_comparison.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_decay_comparison.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_decay_comparison.py
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_decay_comparison.py --verify-only
```

Only if training is interrupted, run:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_decay_comparison.py --resume
```

Resume verifies and skips locked fits; an unfinished fit restarts from its
fixed initialization, not from an optimizer-state checkpoint. Completed
outputs refuse retraining and resume; use `--verify-only` instead.

Output includes `run_identity.json` (IDs, protocol and source hashes),
`orders_seed_*.json`, `baselines.json`, and `fits/<model>__<condition>__<seed>/`
with checkpoint, history, gradients, prediction details, metrics and file lock.
Prediction arrays follow the validation IDs saved in `run_identity.json`.
`summary.txt` and `summary.json` collect all fits and paired comparisons.
`evaluation_lock.json` protects the complete 79-artifact result set.
Verification reloads checkpoints and reproduces validation predictions and
metrics without training or test scoring.

The seven tests cover fixed settings, split/group isolation, matched initial
weights and batch schedules, training-only scalar fitting, deterministic fits
and reloads, preflight without training, interrupted-fit recovery, completed
output protection and artifact tampering.

## DQ19: Training-Only Gradient/Decay Check

**Status:** executed and reproduced exactly; eight focused tests pass.
This is a diagnostic, not a new training run. R04, its 91 locked artifacts,
all graph caches and the existing splits remain unchanged.

### Bounded Method

- Select 32 of the existing 1,023 training CVEs by a fixed hash of CVE ID,
  without looking at targets or predictions. Reuse four batches of eight.
- Inspect all four model configurations and seeds 17/29/43, at saved
  initialization and at the previously selected checkpoint: 96 backward passes.
- Use the original Smooth L1 loss, residual gain 0.00885 and clipping norm one.
  No optimizer is instantiated or stepped.
- Compare the data-loss gradient after clipping with `0.0001 * parameter`.
  Installed ordinary Adam adds this decay term after external clipping.
  Record raw and clipped gradients, decay, their sum, direction alignment,
  zero/unused gradients, and per-parameter and per-module norms.
- Use only selected training inputs and endpoint labels. Shared JSON containers
  are parsed and restricted to these IDs. Existing result files are hashed,
  not rescored. Checkpoints themselves were previously selected on validation.

### Findings

| Selected-checkpoint component | Comparison | Sampled checks |
|---|---|---:|
| EPSS GRU encoder | Decay norm exceeds data-gradient norm | 48/48 |
| GNN message-passing layers | Decay norm exceeds data-gradient norm | 36/36 |
| Final output bias | Data-gradient norm exceeds decay norm | 48/48 |
| Input projection / other head parameters | Mixed; see detailed report | Not uniform |

Each check is one model, seed and training batch, not an independent experiment.
No gradient clipping or output clipping occurred in any of the 96 passes.

The following examples are the median of the four batch ratios for the
no-COREF model. A ratio above one means the decay term has the larger norm;
it is **not** a ratio of actual Adam parameter updates.

| Seed | EPSS encoder: decay/data ratio | GNN layers: decay/data ratio | Final bias: decay/data ratio |
|---|---:|---:|---:|
| 17 | 1,059 | 1,988 | 0.0109 |
| 29 | 15,738,566 | 416,867 | 0.0128 |
| 43 | 340,736 | 6,959 | 0.0128 |

For seed 29, the EPSS encoder's median data-gradient norm is about `1.80e-19`
and its decay norm is `2.83e-12`. The enormous ratio therefore compares two
tiny quantities; it does not imply a large absolute update.

At initialization, the zero final-layer weights block upstream data gradients,
as expected. Nonzero upstream weights still receive a decay term. In this
sample, 30/32 targets equal float32 persistence, and two whole batches have
zero initial loss. These are the fixed audit batches, not the original first
training batches. Sparse change signal and loss scaling remain plausible
contributors alongside regularization.

**Interpretation:** the results support a local gradient imbalance consistent
with the previously observed encoder shrinkage and nearly bias-only
corrections. They do not prove that decay caused the behavior: the sample is
small, intermediate training states and Adam moment buffers are unavailable,
and stable targets can also favor small or common corrections.

**Next candidate, not implemented here:** first add a train-fitted scalar
correction control, then a separately registered small matched experiment
changing only weight decay (original value versus zero). Keep the residual
gain, initialization, batches, optimizer, loss and budget fixed; include the
EPSS-only control and decide graph configurations before running. Do not
simultaneously change gain, optimizer or split, and do not open the test set.

### Files And Reproduction

Script: `02_prediction/03_analysis/01_revision_dataset/diagnose_forecast_residual_gradients.py`.
Tests: `test_forecast_residual_gradients.py` in the same folder.
Results: `05_results/02_evaluation/epss_day30_to_day60/coreference_residual_gradient_audit_v1/`.
Plain-language interpretation:
`05_results/02_evaluation/epss_day30_to_day60/coreference_residual_gradient_audit_v1_review/`.

- `identity.json`: fixed protocol, selected training CVEs/batches and source hashes.
- `gradient_details.json`: every batch, state and parameter measurement.
- `analysis.json` / `analysis.txt`: module-level summaries and interpretation limits.
- `diagnostic_lock.json`: hashes of the four diagnostic artifacts.
- Review `analysis.txt` / `analysis.json`: findings, sample limitations and next-step rationale.

The diagnostic already exists. Run these commands individually with
`CodeBERTFusion` active, from the repository root:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_residual_gradients.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/diagnose_forecast_residual_gradients.py --verify-only
```

Without `--verify-only`, the script creates the initial diagnostic only when
its destination is absent. It refuses to overwrite existing results.
Verification repeats the same 96 backward passes without optimizer updates
and requires exact report reproduction and unchanged source hashes.

## R04 v2: Full Development

**Status:** user full run complete; all 91 locked artifacts and all 12 selected
checkpoints/predictions independently verified. This uses the full existing
eligible development cohort, not all 5,968 source CVEs. Preserve the 1,023
training / 197 validation IDs, chronological boundaries and related groups.
The reserved test remains closed. Current descriptions remain retrospective,
not authenticated historical Day-30 text.

### Verified Results

| Model | Seed 17 MAE | Seed 29 MAE | Seed 43 MAE | Three-seed mean MAE | Sample SD |
|---|---:|---:|---:|---:|---:|
| Persistence | 0.00909675 | Same | Same | 0.00909675 | Not seeded |
| No COREF | 0.00917751 | 0.00923345 | 0.00924016 | 0.00921704 | 0.00003440 |
| Repeated-only | 0.00917750 | 0.00923345 | 0.00924014 | 0.00921703 | 0.00003440 |
| All COREF | 0.00917740 | 0.00923345 | 0.00924005 | 0.00921697 | 0.00003442 |
| EPSS-only | 0.00918790 | 0.00923341 | 0.00924022 | 0.00922051 | 0.00002845 |

Every fit improves MAE over matched R03, but remains 0.89-1.58% worse than
persistence. All RMSEs are slightly lower than persistence, reflecting small
improvements on larger errors rather than overall MAE superiority. All three
graph modes and EPSS-only are very similar. Three seeds describe seed
variation, not independent dataset uncertainty or statistical significance.

There is no recorded training-time or selected-validation clipping. All
variants select/stop at epochs 1/16 (seed 17), 10/25 (29) and 7/22 (43), as
specified by patience 15. No fit reaches the 100-epoch cap.

**Prediction behavior:** persistence is exact for 162/197 CVEs; 28 increase
and seven decrease. Every selected model adds a positive correction. Seed-29
graph corrections are approximately 0.000182130 with standard deviation only
about 4e-11; seed-29/43 EPSS-only corrections are exactly constant in saved
float32 outputs. Final predictions still vary because the persistence input
varies. Learned CVE-specific correction is weak; useful graph contribution
is not demonstrated by these results.

Three CVEs account for 97.79% of persistence absolute error: CVE-2024-23692,
CVE-2024-8963 and CVE-2023-7028. For example, the first rises from 0.00210 to
0.95432 while all_coref seed 17 predicts 0.00221160. All fits remain close to
persistence on these large changes. None of these cases is removed.

**Parameter inspection:** seed-29 no_coref GRU L2 norm shrinks from 4.79991
to 2.83054e-8; node-projector norm shrinks from 3.26458 to 0.00057530. The
other seed-29 models show the same pattern. Gradients did flow through every
encoder during training, so this is not evidence of a disconnected branch.
Seed 43's first-step head gradient is zero because all eight first-batch targets
equal float32 persistence; this was checked directly. Some later encoder
gradients are also zero, unlike the short smoke's all-later-step pattern.

The observed shrinkage and nearly bias-only corrections motivate a bounded
training-only check of data-loss gradients versus unchanged Adam weight decay.
That check is now completed in DQ19 above.
The smaller correction scale may change their balance, but the causal role
of regularization is not proved by norms alone. Stable targets and Smooth L1
may also favor a common positive correction. Do not simply increase epochs
or claim a security-extractor bug. Consider a train-fitted scalar-correction
control and then one justified small regularization ablation, separately
registered, before another full run.

Detailed TXT/JSON review:
`05_results/02_evaluation/epss_day30_to_day60/coreference_residual_development_v2_review/`.
It retains all fit statistics, missingness groups, changed/unchanged cases,
matched model differences, top-error examples and initial/selected module
norms. No retraining, new experiment implementation or test scoring occurred.

### Fixed Experiment

| Component | Choice |
|---|---|
| Inputs / target | EPSS Days 0-30 and description TPG; stored Day-60 EPSS |
| Models | No COREF, repeated-only COREF, all COREF, EPSS-only |
| Seeds | 17, 29, 43 for every model: 12 fits |
| Budget | Up to 100 epochs; patience 15; earliest strictly lowest validation MAE |
| Head | Same v2 `clip(persistence + tanh(gain * head), 0, 1)` |
| Gain | Fit once on past inputs of all 1,023 training CVEs; shared across models/seeds |
| Initialization | Fresh per seed; graph variants share initial weights/orders; no smoke warm start |
| Optimization | Same Adam, learning rate 0.001, weight decay 0.0001, Smooth L1 beta 0.01, batch eight, clip norm one |
| Execution | CPU, one Torch thread, deterministic algorithms, dropout zero |
| References | Persistence and all matched R03 variants/seeds |

Gain is the median latest observed training-input EPSS, bounded to [0.001,0.05].
Its full-cohort value may differ from smoke's 0.00885 because the population is
larger, not because validation selects a scale. Gain and contributing training
IDs are saved in the run identity; every model spec includes the gain. Epoch
zero equals float32 persistence but cannot win checkpoint selection.

### Checks And Recovery

The runner verifies DQ17 v2, both completed residual smokes, and all saved R03
checkpoints/predictions. Comparisons therefore use identical CVEs, targets,
graph configurations and seeds. No parsing, embeddings or downloads are rerun.
`--check-only` reproduces existing validation references and checks exact
persistence on small/largest train and validation graph batches. It performs
no new fitting, new model outcome scoring or output writes. Seed-specific
initialization checks are saved during training and reproduced on verification.

Completed fits have separate locks and cannot be overwritten. `--resume`
checks sources, initialization, batch schedules and finished fits; it skips
verified completed fits and restarts an unfinished fit from its fixed initial
state. This is fit-level recovery, not optimizer-state resume. Completed runs
accept `--verify-only`, not `--resume`. A failed final verification preserves
locked artifacts rather than rewriting their status or checksums.

### Files And Commands

Runner: `02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_residual_development.py`.
Implementation: `02_prediction/01_package/epss/forecast_coreference_residual_development.py`.
Protocol: `02_prediction/04_configs/coreference_residual_development_v2.json`.
The JSON's original `planned_not_runnable_pending_smoke_verification_and_review`
status is historical and deliberately unchanged because smoke pins its hash.
The new runner records current implementation status in its own run identity.

Output: `05_results/02_evaluation/epss_day30_to_day60/coreference_residual_development_v2/`.

- `initialization/`, `orders_seed_*.json`, `epoch_zero_seed_*.json`: initial states, schedules and persistence checks per seed.
- `fits/<variant>__<seed>/`: selected checkpoint, history, first-step/accumulated gradients, raw/corrected/clipped predictions, metrics and lock.
- `summary.json` / `summary.txt`: all 12 fits, persistence, three-seed means/sample standard deviations, paired differences and limitations.
- `predictions.json`: CVE-level targets, baseline and all 12 predictions with correction details.
- `training_history.json` / `training_curves.png`: x-axis epoch, y-axis EPSS MAE; solid training, dashed validation.
- `run_identity.json`, `run_status.json`, `evaluation_lock.json`: provenance, completion and hashes for 91 output artifacts.

The full output already exists. Use `--verify-only`, not another training or
resume command. Original execution commands remain below for reference.
Run individually from the repository root with `CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_coreference_residual_development.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_residual_development.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_residual_development.py
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_residual_development.py --verify-only
```

Only after interruption, instead of a fresh training command:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_residual_development.py --resume
```

Eleven synthetic tests cover fixed settings, seed/state matching, strict-tie
early stopping, deterministic reloads, input/target isolation, all-seed
aggregation, shared lazy graph storage, smoke prerequisite rejection, exact
algorithm parity with v2 smoke, complete matched references, interruption
recovery, changed-source rejection, no overwriting and artifact tampering.
No best-seed selection, split revision or test scoring is part of this run.
The combined COREF regression suite passes **57 tests**. The real-data full
preflight also passes, reproducing frozen references and checking exact
persistence on the selected small/largest batches. The median across the
1,023 training histories is **0.00885**, so the full gain happens to equal
the smoke gain. No full output directory, optimizer run or test score was
created during this implementation/preflight.

## R04 v2: Training-History-Scaled Residual Smoke

**Status:** user smoke training complete and independently verified. This follows the verified
v1 smoke, where each graph model clipped 20/24 validation predictions to zero.
Keep v1 code, checkpoints, outputs, graph caches and cohort unchanged. V2 uses
the existing encoder and diagnostic helpers without modifying or monkey-patching
v1. Versioned orchestration keeps the new artifacts and model contract separate.

### One Controlled Change

```text
p = latest observed input EPSS on or before Day 30
s = median(p) over training CVEs, bounded to [0.001, 0.05]
z = zero-initialized correction head(TPG/GNN + EPSS/GRU)
prediction = clip(p + tanh(s * z), 0, 1)
```

The same scalar `s` is used for all four models. Every training CVE gets one
vote in the median, including observed zero scores. The fallback for missing
Day 30 is the latest earlier observed score, not zero or a future value.
No labels, validation inputs or test values determine the scale. The limits
0.001 and 0.05 are fixed engineering choices, not fitted optima; no grid search
is introduced. The chosen value and the contributing training IDs are recorded
in `run_identity.json`, and the value is saved in each checkpoint's model spec.

Scaling occurs **inside** tanh. For a small head output, the correction responds
more gently than v1, but it is not permanently limited to plus/minus `s`.
For example, an illustrative scale of 0.01 and head output 0.1 gives a correction
near 0.001 rather than v1's approximately 0.0997. These are explanatory numbers,
not observed predictions. Large corrections and clipping remain possible.
Smaller initial gradients may also slow learning. This experiment does not
guarantee lower error, less clipping or useful text information.

### Fixed Smoke Contract

- Same 32 training / 24 validation CVEs, seed 17, three epochs and batch size eight.
- Same three COREF configurations, EPSS-only control and persistence reference.
- Same GNN/GRU, initial weights, loss, optimizer, learning rate, weight decay,
  missingness masks, batch orders and earliest-best-trained-epoch selection.
- Exact float32 persistence at initialization; epoch zero remains ineligible.
- Same clipping, raw-correction and gradient diagnostics. No new graphs,
  embeddings, text sources, split changes or test scoring.
- Before a v2 run, verify the completed v1 artifacts and reproduce its validation
  predictions. Thus `--check-only` performs existing v1 validation verification,
  but no optimizer steps, v2 outcome scoring or output writes.

New protocol: `02_prediction/04_configs/coreference_residual_smoke_v2.json`.
Runner: `02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_residual_v2.py`.
Outputs: `05_results/02_evaluation/epss_day30_to_day60/coreference_residual_smoke_v2/`.
The TXT/JSON summaries explain the scale and retain all four configurations.
Hashes pin both the original run and the new protocol/code. Existing output
directories cannot be overwritten. Progress bars cover the important stages.

The original commands are retained below. V2 output now exists: use
`--verify-only`, not another training run into the same directory.
Run individually with `CodeBERTFusion` active, from the repository root:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_coreference_residual_v2.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_residual_v2.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_residual_v2.py
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_residual_v2.py --verify-only
```

Eleven synthetic tests cover scale isolation from targets/validation, median
bounds and true zeros, missing Day 30, future-input rejection, identical initial
encoder weights, gradient routing, scale placement inside tanh, graph-control
invariance, deterministic fits, reloads, selection, tampering and no overwriting.
The combined COREF regression suite passes **46 tests**. The real-data preflight
passes after reproducing v1, with median training-input EPSS and gain both
**0.00885**. Exact float32 persistence holds for all four new models on all 56
selected CVEs. No v2 dataset fitting, outcome scoring or output writing occurred.

### Verified V2 Findings And Cohort Scope

All 16 artifacts and selected-checkpoint predictions reproduce. The training
gain remains 0.00885. All models select epoch 1. No selected prediction or
recorded training forward pass clips outside [0,1]. Encoder gradients are
nonzero on 11/12 steps after the expected first-step zero.

| Model | Validation MAE | Validation RMSE |
|---|---:|---:|
| Persistence | 0.00008750 | 0.00037418 |
| No COREF | 0.00012521 | 0.00036741 |
| Repeated-only | 0.00012520 | 0.00036741 |
| All COREF | 0.00012521 | 0.00036742 |
| EPSS-only | 0.00012822 | 0.00036585 |

Graph MAE improves about 83% over v1 but remains worse than persistence. All
four models improve the two changed CVEs and worsen the 22 unchanged cases.
RMSE improves slightly because larger errors receive more weight; this does
not establish overall superiority or a winning COREF policy. Validation MAE
increases at epochs 2 and 3; three epochs do not establish convergence.

The separate `coreference_residual_smoke_v2_review/analysis.txt` and
`analysis.json` also explain all 5,968 source CVEs. DQ07 was independently
reproduced again: assigned roles are 1,156 training, 199 validation, 2,671
reserved test and 1,942 excluded. Observation requirements reduce the first
three to 1,023, 197 and 2,670 respectively. Partial histories are allowed;
an observed Day-60 target and at least one past input are required, not 91
complete scores. Excluded records are not deleted from the raw source.

The largest related group accounts for 1,329 of 1,548 cross-period group
exclusions. Broad URLs and generic repeated text can overconnect distinct CVEs;
these groups are not verified duplicate clusters. The current sizes preserve
matched comparisons, not an assertion that this is the optimal cohort. A
larger study requires outcome-independent grouping review, separately versioned
preparation and matched baselines. Do not silently repurpose the reserved test.
No full runner, training or test scoring was added by this review.

### Full Training After Smoke Review

`02_prediction/04_configs/coreference_residual_development_v2.json` records the
full protocol. Following smoke verification and review, the separate full
runner above now implements it. There is still no `--full` flag on the smoke
CLI, and the pinned protocol's original planning metadata stays unchanged.

| Setting | Full-development plan |
|---|---|
| Cohort | Existing 1,023 training / 197 validation CVEs; test remains closed |
| Models | No COREF, repeated-only, all COREF, EPSS-only; persistence reference |
| Seeds | 17, 29, 43 for every model: 12 fits, report all of them |
| Budget | Maximum 100 epochs, patience 15; earliest strictly lowest validation MAE |
| Scale | Same median rule, refitted once using only the 1,023 training input histories |
| Initialization | Fresh per seed; no smoke checkpoint warm start; matched graph initial states/orders |
| Reporting | Individual fits plus three-seed mean and sample standard deviation; no best-seed selection |
| Comparisons | Persistence and matched full-cohort R03 results, not 24-CVE smoke headline errors |

The full gain may differ because its training cohort is larger, not because
validation error selects a new hyperparameter. Verify smoke artifacts and
initialization, then review clipping, gradients and corrections before proceeding.
Beating persistence on 24 CVEs is **not** an eligibility requirement: that would
encourage tuning to a small development sample. Any subsequent design change
requires a new version, not retrospective editing of this plan. Historical
description availability and repeated-validation limitations remain explicit.

## R04: Persistence-Residual Smoke

**Status:** user smoke run complete; all 16 locked artifacts, checkpoint
predictions and initialization checks independently verified without retraining
or test scoring. Eleven synthetic tests
cover fixed settings, persistence initialization, missing Day 30, true zero
scores, encoder parity, shared initialization, gradient routing, clipping,
EPSS-only graph invariance, future-input rejection, checkpoint reloads,
earliest-checkpoint selection, saved-result reproduction and no overwriting.
The combined COREF suite passes **35 tests**. The real-data preflight also
passes exact float32 persistence checks for all four models on the selected
32 training and 24 validation CVEs. No optimizer steps, label scoring or
experiment output writes were performed during that preflight.

### Verified Smoke Findings

MAE is the average absolute difference between predicted and stored Day-60
EPSS; lower is better. All rows use the same 24 validation CVEs.

| Model | Selected epoch | Validation MAE | Predictions clipped to zero |
|---|---:|---:|---:|
| Latest observed EPSS (persistence) | Not trained | 0.00008750 | Not applicable |
| No COREF | 2 | 0.00074358 | 20/24 |
| Repeated-mention COREF only | 2 | 0.00074326 | 20/24 |
| All COREF | 2 | 0.00074331 | 20/24 |
| EPSS-only residual | 3 | 0.00055154 | 0/24 |

Persistence is exact for 22/24 CVEs. Graph corrections are negative on every
validation CVE and worsen every individual error relative to persistence.
EPSS-only corrections are positive on all 24, improving the two changed cases
but worsening the other 22. The tiny differences among graph configurations
do not establish a preferred COREF policy.

All models start at exact float32 persistence. Each encoder receives nonzero
data-loss gradients on 11/12 steps; the first-step zero is expected from the
zero-initialized final layer. Head gradients are nonzero on all 12 steps.
There is no clipping in the recorded training forward passes. Therefore the
validation clipping is not evidence of training-wide gradient saturation or
a disconnected branch. It shows corrections too negative for many small
validation scores. The saved logs do not establish why those corrections arose.

On matching validation IDs, dates and targets, R04 MAEs are much lower than
R02's direct-head smoke MAEs (about 0.01787 for fusion and 0.04816 for EPSS-only),
but still worse than persistence. Three epochs and one seed are an engineering
check, not evidence of convergence or generalized model quality.

Separate review: `05_results/02_evaluation/epss_day30_to_day60/`
`coreference_residual_smoke_v1_review/analysis.txt` and `analysis.json`.
Original outputs and pinned code remain unchanged. Recommended next step:
specify a separately versioned, smaller bounded-correction smoke comparison,
with the same cohort, controls and optimizer settings. Choose its scale using
training data only, not the 24 validation targets. Keep v1 as the reference;
do not silently change clipping, loss or selection rules. Full training is
not yet recommended or implemented.

### Fixed Comparison

This is an engineering smoke run, not a performance claim. Reuse the **same
32 training and 24 validation CVEs as DQ16/R02**, taking their unchanged
graphs from the verified DQ17 v2 full cache. No reselection by error or target.
The full cache verification still checks all 1,220 development graphs; the
smoke optimizer and metric calculations use only the selected 56 CVEs.

| Component | Fixed choice |
|---|---|
| Inputs | Description TPG; EPSS Days 0-30; masks and observation age |
| Target | Stored Day-60 EPSS, 30 days after the Day-30 anchor |
| Models | `no_coref`, `repeated_only`, `all_coref`, `epss_only` |
| Reference | Latest observed past EPSS, including fallback when Day 30 is absent |
| Seed and budget | Seed 17; all three epochs; batch size eight |
| Graph encoder | Same two 32-unit EdgeTypeGNN layers and mean pooling as R03 |
| Temporal encoder | Same 16-unit GRU and five input channels as R03 |
| Optimization | Adam 0.001; weight decay 0.0001; Smooth L1 beta 0.01; clip norm 1.0 |
| Execution | CPU; one Torch thread; deterministic algorithms; dropout zero |
| Selection | Earliest strictly lowest validation MAE among trained epochs 1-3 |
| Full development | Planned only: 1,023/197 CVEs, seeds 17/29/43, 100 epochs, patience 15 |

The protocol is separate:
`02_prediction/04_configs/coreference_residual_smoke_v1.json`.
There is no seed, epoch or full-expansion override in this CLI.

### What Changes

```text
Description -> existing TPG/GNN ----\
                                   concatenation -> existing MLP -> tanh correction
Past EPSS -> existing GRU ---------/                                  |
Latest observed past EPSS ------------------------------------------- + -> clip to [0,1]
```

The direct-score sigmoid becomes **latest observed EPSS plus a learned
correction**, bounded to [0,1]. The MLP structure is unchanged; its final
linear layer starts with zero weights and bias. Fresh models are used, not
fine-tuned R03 checkpoints. No target mean is used to initialize the head.
The three graph modes share one saved initial state and the same batch orders.
EPSS-only uses its own saved initial state on identical CVEs and orders.

The new class inherits the existing model modules. Its encoder calls follow
the frozen R03 forward path; synthetic parity tests confirm identical
concatenated representations before the changed head. Original model code
is not edited. SecBERT node embeddings remain frozen; the GNN/GRU are trained.

### Persistence and Gradient Checks

- Before training, every model must exactly equal its **float32 persistence
  input**, with zero correction, on the selected training and validation
  records. The source-precision persistence reference is kept separately;
  only float32 rounding may differ, by at most 6e-8 in this check.
- This epoch-zero check does not score labels or select a checkpoint.
  Epoch zero is **not eligible** for selection. The experiment may therefore
  perform worse than persistence; that result must remain visible.
- Missing Day 30 uses the latest observed input, never a future score or
  an unmarked zero. A real observed score of zero remains valid. All-missing
  histories are rejected under the existing input contract.
- Zero final-layer weights initially block upstream **data-loss** gradients.
  The output layer can learn on the first update and then pass gradients to
  the encoders. The first-step and accumulated gradients are recorded.
  Adam weight decay can still change weights when data-loss gradients are zero.
- Hard clipping has zero gradient outside [0,1]. Save raw predictions,
  corrections, training clipping counts and selected-checkpoint clipping counts;
  do not hide clipping or assume every branch must have nonzero gradients on
  every step. Boundary and saturation tests cover this behavior.

### Outputs and Safety

Default results:
`05_results/02_evaluation/epss_day30_to_day60/coreference_residual_smoke_v1/`.

| Artifact | Meaning |
|---|---|
| `run_identity.json` | Protocol, 32/24 cohort, source/cache/DQ18/code hashes |
| `initialization/*.pt`, `batch_orders.json` | Saved initial weights and common epoch orders |
| `epoch_zero.json` | Exact float32-persistence checks and source-rounding differences |
| `checkpoints/*.pt` | Earliest best trained checkpoint for each model |
| `history.json`, `training_curves.png` | Training/validation errors; epoch on x, EPSS MAE on y |
| `gradients.json` | First-step data-loss gradients and accumulated branch counters |
| `predictions.json` | Actual score, persistence, learned correction, raw and clipped model outputs |
| `summary.json`, `summary.txt` | Metrics, missingness groups, ranges, clipping and limitations |
| `run_status.json`, `evaluation_lock.json` | Completion state and artifact checksums |

Progress bars cover provenance checks, initialization, models, epochs, batches,
checkpoint evaluation and report writing. `--check-only` verifies existing
dependencies and persistence initialization without optimizer updates, label
scoring or output writes. `--verify-only` reloads and reproduces saved results
without training. Completed outputs cannot be overwritten. There is no resume
option for this bounded smoke run; an interrupted run stays preserved and a
new output must use a new sibling directory under the evaluation-results root.

R03/DQ18 results and source caches remain unchanged. No SMP, summaries, CVSS,
VulnCheck predictors, new parsing, encoder downloads or test scoring are added.
The current-description/historical-availability limitations remain unchanged.

### Commands

Original execution commands are retained below. The smoke output now exists:
use only `--verify-only` to check it, not the training command again.
Run individually from the repository root with `CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_coreference_residual.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_residual.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_residual.py
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_residual.py --verify-only
```

Review smoke initialization, clipping, gradients and saved predictions before
implementing the full multi-seed run. Do not compare these 24-CVE metrics
directly with the 197-CVE R03 headline metrics, choose the best seed, or change
loss/stopping rules/graphs together. Prior temporal-only residual models did
not beat persistence; this design change is a controlled experiment, not a
promised improvement.

## DQ18: R03 Branch Diagnostic

**Status:** the user's smoke and full intervention runs are complete and
independently reproduced. Eight synthetic tests
cover deterministic donor mapping, input selection, latent/input swap
equivalence, missingness-preserving score shifts, branch boundaries, the
EPSS-only negative control, immutable models/tensors, saved-output reproduction,
tamper detection and the full-run smoke requirement.
The combined R02/R03/DQ18 suite passes all 24 tests. The real-data smoke
preflight also passes, reproducing original predictions for all 12 checkpoints
on the 16 selected CVEs. No real-data interventions or diagnostic output files
were produced during implementation.

### Verified Full Findings

Separate interpretation: [TXT](../05_results/02_evaluation/epss_day30_to_day60/coreference_branch_diagnostic_v1/full_review/analysis.txt)
and [JSON](../05_results/02_evaluation/epss_day30_to_day60/coreference_branch_diagnostic_v1/full_review/analysis.json).
All **26,004 CVE/probe predictions** reproduce across 197 validation CVEs and
12 checkpoints. These are repeated observations, not independent samples.
All model states remain unchanged; no gradients are created. All **2,364
EPSS-only graph-control predictions** are exact no-ops. No training or test scoring.

**The full finding differs from smoke:** repeated-only and all-COREF models
now respond more to EPSS swaps than graph swaps on average. Smoke lacked
high-score cases; the full recipient and donor pools are different. This is
not a controlled causal test of why the averages change. Use full results for
the main interpretation, not the smoke graph-dominance observation.

The table reports mean absolute prediction changes and mean MAE differences
after each swap. Positive MAE difference means worse recipient-target error.
Donor repetitions are averaged within each model seed, then across all three
seeds. These are artificial interventions, not retrained ablations.

| Model | Prediction change: EPSS swap | MAE change: EPSS swap | Prediction change: graph swap | MAE change: graph swap |
|---|---:|---:|---:|---:|
| No COREF | 0.00001285 | 0.00001222 | 0.00211217 | -0.00004108 |
| Repeated-text COREF | 0.01481832 | 0.01479203 | 0.00410835 | 0.00048189 |
| All COREF | 0.00999755 | 0.00997825 | 0.00520028 | 0.00003971 |
| EPSS-only | 0.02105683 | 0.02028398 | 0.00000000 | 0.00000000 |

1. **Some fusion fits use EPSS effectively within the fixed model.** Swapping
   EPSS substantially worsens error for repeated-only seeds 17/43 and all-COREF
   seeds 17/29. The no-COREF fits, repeated-only seed 29 and all-COREF seed 43
   have very small responses. EPSS-only seed 43 is much less responsive than
   its other seeds. Keep all seeds; the diagnostic does not identify one bug.
2. **Graph contribution is weaker and less consistent.** Repeated-only graph
   swaps worsen MAE in all nine seed/donor combinations, by 0.00048189 on
   average. These reuse the same data and are not nine independent statistical
   tests. All-COREF mean graph-swap MAE changes are +0.00182087, -0.00151771
   and -0.00018402 for seeds 17/29/43; the small aggregate hides opposing effects.
3. **Zeroing is not a retrained comparison.** Zeroing the graph worsens MAE
   in every fusion fit, but zero is an artificial input to the trained head.
   This does not establish graph usefulness over a separately trained EPSS-only
   model. R03 still shows higher MAE than persistence for every trained fit.
4. **Coverage and clipping remain relevant.** The full run contains 38/130/29
   CVEs with 31/30/at-most-29 past scores and 88/109 with/without reference
   COREF. EPSS-swap responses are largest in the least-complete group, but
   differing scores and donors prevent a causal missingness interpretation.
   Subtracting 0.01 clips 5,539/5,840 observed scores (94.85%) to zero; adding
   0.01 causes no upper clipping. The shifts are not symmetric local derivatives.

**Next decision:** DQ18 is complete; no further diagnostic expansion is needed
before specifying a controlled persistence-plus-learned-correction fusion
experiment. Include a matched EPSS-only residual control and all three graph
configurations/seeds, using the same 1,023/197 cohorts. Start with a small smoke
run and zero-initialized correction output, with tests for exact persistence
initialization, missing Day 30, clipping and gradient routing. Change the head
without simultaneously changing extraction, loss, stopping rules or evidence
sources. Compare against frozen R03 and persistence. Earlier temporal-only
residual models did not beat persistence, so improvement is not assured.
That analysis did not train a new model. The subsequent R04 smoke implementation
is documented above; its dataset run is still pending.

### Verified Smoke Findings

Historical software-check findings follow. Their graph-versus-EPSS sensitivity
ordering is not the main conclusion after the full diagnostic above.

Separate interpretation: [TXT](../05_results/02_evaluation/epss_day30_to_day60/coreference_branch_diagnostic_v1/smoke_review/analysis.txt)
and [JSON](../05_results/02_evaluation/epss_day30_to_day60/coreference_branch_diagnostic_v1/smoke_review/analysis.json).
The locked smoke artifacts and all R03 checkpoints remain unchanged.

- All 12 original model outputs reproduce and all model states remain
  unchanged. No gradients are created. All **192 EPSS-only graph-control
  predictions** are exactly unchanged, as required.
- Fusion predictions respond more to graph swaps than to EPSS swaps on
  these 16 CVEs. This is sensitivity, not proof of useful graph information.
  Graph-swap MAE differences are small and of mixed sign, so these findings
  do not establish accurate CVE-specific graph/target alignment.
- Repeated-only seed 29 and all-COREF seed 43 show nearly no response to EPSS
  swaps. EPSS-only seed 43 is much less responsive than seeds 17 and 29.
  These observations are consistent with the earlier compressed outputs,
  but do not identify a cause or justify selecting seeds.

The table shows **mean absolute changes in predicted EPSS**, not MAE or
improvements. Donor repetitions are averaged within each seed before averaging
all three model seeds. Values are rounded to eight decimals.

| Model | Swap EPSS representation | Swap graph representation | Add 0.01 to observed past scores |
|---|---:|---:|---:|
| No COREF | 0.00000040 | 0.00262602 | 0.00000363 |
| Repeated-text COREF | 0.00003492 | 0.00410010 | 0.00035495 |
| All COREF | 0.00004403 | 0.00330281 | 0.00035744 |
| EPSS-only | 0.00028963 | 0.00000000 | 0.00293011 |

**Coverage limitation:** Day-60 scores range only from **0.00043 to 0.00756**.
Five CVEs have 31 observed past scores, 11 have 30, and none has 29 or fewer.
Eight reference graphs have COREF and eight do not. The sample is too small
and lacks the high-score cases needed for the main interpretation.

**Clipping limitation:** 484 of the 485 observed past scores are below 0.01.
The minus-0.01 probe therefore clips 484 values to zero. Plus/minus probes
are not symmetric local derivatives; missing padding and masks remain intact.

**Decision at smoke review:** proceed with the already implemented full DQ18 diagnostic and
verification, not a new training run. Preserve the fixed smoke selection,
protocol and all seeds. Review the full findings before specifying residual
fusion. The test set remains unopened.

### Question and Fixed Scope

Do the saved R03 models respond to their EPSS and graph branches, and does
that response correspond to better or worse prediction of the recipient CVE's
stored Day-60 EPSS? This is a post-hoc sensitivity check, not another training
experiment or proof of causal contribution. It cannot establish that TPG is
useful merely because predictions change.

- All four R03 configurations and all three seeds remain included: 12 models.
- Smoke uses 16 validation CVEs selected by a fixed CVE-ID hash, independent
  of target values, errors or branch responses. Full uses the same 197 R03
  validation CVEs. Neither stage selects or scores test records.
- The frozen R03 verifier checks the complete source/cache chain and reproduces
  all saved predictions before the diagnostic. Even smoke/preflight therefore
  spend time verifying the full cache; this is not graph regeneration.
- CPU, one Torch thread, deterministic algorithms and evaluation mode.
  No gradients, optimizer, epochs, new weights or embedding downloads.
- Each model separately encodes EPSS and graph inputs. Capture the actual
  concatenated representation immediately before its existing prediction head.
  Reusing that representation avoids rewriting either encoder.

### Interventions

| Probe | What changes | What remains fixed |
|---|---|---|
| `original` | Nothing; reproduce the saved output | Both branches and all weights |
| `swap_epss` | Encoded past EPSS history from another CVE, including masks/age information | Recipient graph, weights and recipient target |
| `swap_graph` | Encoded graph from another CVE | Recipient EPSS, weights and recipient target |
| `zero_epss` | Temporal representation becomes zero immediately before the head | Recipient graph and weights |
| `zero_graph` | Graph representation becomes zero immediately before the head | Recipient EPSS and weights |
| `observed_plus` | Add 0.01 to observed input scores, bounded at 1 | Missing padding, masks, ages, day indexes, graph and weights |
| `observed_minus` | Subtract 0.01 from observed input scores, bounded at 0 | Missing padding, masks, ages, day indexes, graph and weights |

Swaps use three fixed bijections (seeds 101/211/307) shared across model variants
and model seeds. Every donor must belong to a different related-CVE group; no
relaxation is allowed if a bijection is impossible. Donors are not matched on
date or score, so swaps deliberately break the CVE-history pairing. They are
not realistic time-travel counterfactuals. With separate deterministic encoders,
latent swaps match direct input swaps, as checked in synthetic tests.

Graph swaps and graph zeroing must leave EPSS-only predictions **exactly
unchanged**. This negative control catches intervention routing errors.
EPSS shifts must leave the captured graph branch exactly unchanged.
Checkpoint state hashes are checked before and after interventions; parameter
gradients must remain absent. Original predictions must match with absolute
tolerance 1e-7 and relative tolerance 1e-6.

### Reporting and Interpretation

There are 11 evaluated probe conditions per model: original, six swaps, two
zero-branch probes and two score shifts. Full output therefore contains
**12 x 11 x 197 = 26,004 CVE-level prediction rows**; smoke contains 2,112.

- Mean absolute prediction change measures response strength, not improvement.
- MAE difference is intervened MAE minus original MAE against recipient labels.
  Positive means worse error; negative means lower error under the artificial
  intervention. Neither sign establishes causality or a retrained ablation result.
- Per-seed and per-donor results are retained. Aggregate donor repetitions
  within each model seed first, then describe all three model seeds with mean
  and sample standard deviation. Do not treat nine repetitions as independent
  CVE samples or choose the best seed.
- Fixed completeness groups (31, 30, at most 29 past scores) and with/without
  COREF groups are reported. COREF presence is measured in the all-COREF
  reference graph, not necessarily in the currently selected edge view.
  Subgroups overlap; counts below 20 are flagged; empty errors remain null.
- Latent norms describe scale, not importance. Near-zero response does not
  prove a branch contains no useful information. Zeroing/swaps may produce
  representations outside the training distribution. Clipping may reduce the
  requested score shift near 0 or 1.

Default root:
`05_results/02_evaluation/epss_day30_to_day60/coreference_branch_diagnostic_v1/`.
Each `smoke/` or `full/` folder contains:

| File | Purpose |
|---|---|
| `identity.json` | Protocol, selected CVEs, donor maps, source and code hashes |
| `analysis.json` | All seed/subgroup responses, aggregates and integrity checks |
| `analysis.txt` | Plain-language probe definitions, metric table and limitations |
| `predictions.json` | Recipient CVE, donor, original/intervened score and actual score |
| `diagnostic_lock.json` | Checksums protecting the separate diagnostic outputs |

Existing output folders cannot be overwritten. `--verify-only` checks hashes
and reproduces the diagnostic. Full mode requires matching smoke identities
and artifacts; it does not silently create them. Progress bars cover source
verification, model encodings, interventions and report writing.

### Commands

Run each command individually from the repository root with `CodeBERTFusion`
active. First run and review smoke; full commands are for the subsequent stage.

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_coreference_branches.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/diagnose_forecast_coreference_branches.py --scale smoke --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/diagnose_forecast_coreference_branches.py --scale smoke
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/diagnose_forecast_coreference_branches.py --scale smoke --verify-only
```

After reviewing smoke:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/diagnose_forecast_coreference_branches.py --scale full
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/diagnose_forecast_coreference_branches.py --scale full --verify-only
```

**Subsequent decision:** inspect R03 branch response, particularly narrow-output
versus wider-range seeds. Then specify a small, separately versioned
persistence-plus-correction fusion comparison with a matched EPSS-only residual
control. Keep cohorts and all seeds fixed. Earlier EPSS-only residual forecasts
did not beat persistence, so this is not a promised improvement. Any revised
early-stopping policy must be a separately declared comparison, not a selective
extension of weak seeds. Historical-description and repeated-validation limits
remain; the test set stays unopened.

## R03: Matched Development Training

**Status:** user run complete; all three seeds, 12 checkpoints and validation
predictions independently reproduced. No retraining or test scoring. Eight synthetic tests
cover fixed settings, seed matching, stopping/selection, checkpoint reloads,
input/target separation, missingness groups, bounded graph loading, interruption
recovery and tampering. R02, DQ17 and the protocol JSON remain unchanged.
All 16 R02/R03 training tests pass. The real-data preflight also passes:
the full cache is verified and all 12 small/largest-graph forward batches
complete without fitting, scoring or writing experiment outputs.

### Verified R03 Findings

Separate review: [TXT](../05_results/02_evaluation/epss_day30_to_day60/coreference_training_development_v1_review/analysis.txt)
and [JSON](../05_results/02_evaluation/epss_day30_to_day60/coreference_training_development_v1_review/analysis.json).
Original result files, checkpoints, protocol and caches are unchanged.

All rows use the same **197 validation CVEs**. MAE is average absolute EPSS
score error; lower is better. An MAE of 0.01 means one percentage point.
The last column is the sample standard deviation across seeds, not a confidence
interval over CVEs. Persistence is a single deterministic reference.

| Configuration | Seed 17 MAE | Seed 29 MAE | Seed 43 MAE | Mean MAE | Seed standard deviation |
|---|---:|---:|---:|---:|---:|
| Persistence: latest observed input score | - | - | - | **0.00909675** | - |
| EPSS-only | 0.01280439 | 0.01245079 | 0.03184784 | 0.01903434 | 0.01109822 |
| TPG + EPSS, no explicit COREF edges | 0.03196582 | 0.03139446 | 0.03195968 | 0.03177332 | 0.00032812 |
| TPG + EPSS, repeated-text COREF only | 0.02062356 | 0.03198427 | 0.01916904 | 0.02392563 | 0.00701679 |
| TPG + EPSS, repeated-text and pronoun COREF | 0.02405972 | 0.02522857 | 0.03194487 | 0.02707772 | 0.00425540 |

1. **No demonstrated TPG forecasting gain in this run.** Persistence beats
   every trained fit on MAE and equal-group MAE. EPSS-only has lower mean MAE
   than all graph modes. EPSS-only seed 29 has slightly better RMSE than
   persistence (0.08164098 versus 0.08264449); metric choice matters.
2. **COREF improvements are seed-dependent.** Repeated-only beats no-COREF
   in seeds 17 and 43, but not 29. All-COREF beats repeated-only only in seed 29.
   Keep all seeds; do not report only the best run as the main result.
3. **Several fits compress their outputs into low scores.** Across all
   no-COREF seeds, predicted scores stay between approximately 0.00538 and
   0.02347, while actual validation scores reach 0.95432. EPSS-only seed 43
   predicts 0.00837-0.01092. Repeated-only seed 29 and all-COREF seed 43 also
   have narrow ranges. Other seeds produce substantially wider ranges.
4. **Early stopping was followed, but does not prove convergence.** All fits stop under patience 15,
   after 22-67 epochs, with selected epochs 7-52. None reaches the 100-epoch
   budget. EPSS-only seeds 17 and 29 improve sharply at epoch 26, whereas
   seed 43 stops at epoch 22. We cannot know its later behavior from this run;
   merely raising the maximum epoch limit without changing patience would
   not extend that fit. Any stopping-policy comparison must be a separate
   predefined experiment, not a selective rerun of the weakest seed.
   Training/validation gaps and later worsening validation errors
   are consistent with poor generalization; they do not establish its cause.
   Every monitored branch has nonzero gradients at every recorded step.
   That confirms gradient flow, not effective use of EPSS or text.
5. **Persistence is exact on 162/197 CVEs (82.23%).** Their Day-60 score equals
   the latest observed input score; this does not imply a flat intervening
   curve. Three CVEs account for 97.79% of persistence absolute error:
   `CVE-2024-23692` (0.00210 to 0.95432), `CVE-2024-8963` (0.30991 to 0.95343),
   and `CVE-2023-7028` (0.04478 to 0.20153). Keep these cases in evaluation.
6. **Missing inputs do not explain the whole problem.** The three fixed
   history-completeness groups contain 38, 130 and 29 CVEs for 31, 30 and at
   most 29 observed past scores. Persistence beats every fit in each group,
   including the complete-history group. This is descriptive, not causal.
7. **The COREF comparison has limited coverage.** 88 validation graphs have
   explicit COREF edges; 109 do not; 27 have pronoun COREF. These separately
   trained models can give different predictions even on identical no-COREF
   validation graphs, because their learned weights differ.

**Next recommendation, not implemented here:** diagnose the saved R03
EPSS/text branch response across seeds before another training run. Then
predefine a small persistence-residual fusion comparison with a matched
EPSS-only residual control. Earlier temporal-only residual results did not
beat persistence, so this is a testable design change, not a promised gain.
Keep the fixed splits and all seeds. Do not simultaneously expand graphs,
change loss and add new evidence sources. Leave the test set unopened.

Reproduce the review without changing its files:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_coreference_training.py --verify-only
```

The reviewer reuses the frozen verifier, recalculates metrics from saved
predictions, and writes separate TXT/JSON findings with a review lock. Five
focused arithmetic/provenance tests pass. Outcome-based subsets are post-hoc
descriptions only, never a new cohort or predictor feature. Repeated validation
inspection and unverified historical descriptions remain limitations.

### Fixed Comparison

The completed DQ17 v2 full cache supplies all **1,023 training and 197 validation
CVEs**. Predict EPSS on Day 60 using Days 0-30 plus the current description.
This is retrospective score forecasting, not an observed exploitation outcome.

- Four trained variants: `no_coref`, `repeated_only`, `all_coref`, `epss_only`.
- Seeds **17, 29, 43**: 12 fits, plus one deterministic persistence reference.
- Two-layer 32-unit EdgeTypeGNN, 16-unit GRU, concatenated direct-score sigmoid
  head. SecBERT and graph embeddings are frozen and never regenerated.
- Adam 0.001, weight decay 0.0001, Smooth L1 beta 0.01, batch eight, gradient
  norm clip 1.0, dropout zero. CPU, one Torch thread, deterministic algorithms.
- Maximum 100 epochs; stop after 15 consecutive epochs without strictly lower
  validation MAE. Keep the earliest best checkpoint; ties do not reset patience.
- Per seed, all three graph modes share one saved initialization. EPSS-only
  has a separate initial state. Head bias uses training-target mean only.
  All four variants share the 100-epoch batch schedule; stopping may differ,
  but common epochs use identical orders. No best-seed selection or tuning.

Existing regressor, temporal/missingness contract, batching, initialization,
metrics and restore functions are reused. A bounded 128-bundle CPU cache
exposes three edge views without storing three copies of node features.

### Preflight and Recovery

`--check-only` verifies sources/cache/protocol and runs forward passes on a
small training batch and the largest eight training and validation graphs.
Outputs are discarded: no optimizer steps, validation metrics, result files
or test scoring. This checks forward memory, not a guaranteed backward-memory
footprint or runtime. There is no silent truncation or cohort dropping.

`--resume` requires unchanged sources, code, protocol and initial states. It
reproduces predictions for completed, locked fits and skips their training.
An unfinished fit restarts from its original initialization and schedule,
**not from its interrupted epoch**. Incomplete fit artifacts may be replaced
under explicit resume; locked completed fits cannot be overwritten. Completed
experiments use `--verify-only`, not `--resume`. Keep unrelated logs outside
the result folder.

### Outputs and Metrics

Default folder: `05_results/02_evaluation/epss_day30_to_day60/coreference_training_development_v1/`.

| File | Purpose |
|---|---|
| `run_identity.json` | Frozen cache, code, source, protocol and cohort identities |
| `initialization/seed_*/` | Shared graph-fusion and separate EPSS-only initial states |
| `orders_seed_*.json` | Complete planned training schedules |
| `fits/<variant>__<seed>/checkpoint.pt` | Earliest best validation-MAE checkpoint |
| `fits/.../history.json`, `gradients.json` | Epoch history and branch-gradient diagnostics |
| `fits/.../predictions.json`, `metrics.json` | Reloaded-checkpoint validation predictions and metrics |
| `fits/.../fit_lock.json` | Completed-fit checksums for verification/resume |
| `summary.json`, `summary.txt` | Every seed, persistence, aggregates and limitations |
| `predictions.json` | Same 197 CVEs, dates, actual scores and all fixed-seed predictions |
| `training_history.json`, `training_curves.png` | Epoch on x; training/validation EPSS MAE on y |
| `run_status.json`, `evaluation_lock.json` | Status and final artifact hashes |

Report MAE, RMSE, signed error, Spearman when defined, and equal-group MAE.
JSON includes each seed's matched MAE differences versus persistence,
EPSS-only and no-COREF; negative favors the named variant. Aggregate every
seed with mean and sample standard deviation, not best-seed selection.
Undefined correlations remain null with explicit defined-seed counts.
Persistence is one reference, not three independent observations.

Missingness reports retain the fixed input groups: 31, 30, or at most 29
observed past scores. Groups below 20 CVEs are flagged; empty groups have
null metrics. Main-cohort membership is unchanged. Prediction ranges,
selected epochs and budget-boundary flags help identify constant outputs or
incomplete convergence. Gradient flow is not branch importance. Current-text
availability and repeated validation inspection remain research limitations.

### Commands

Run individually from the repository root with `CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_coreference_development.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_development.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_development.py
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_development.py --verify-only
```

Only for an unfinished interrupted/failed output:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_development.py --resume
```

There is no test option, individual-seed selection or epoch override. The
training command runs all fixed comparisons. Existing completed outputs are
never overwritten. Implementation: `02_prediction/01_package/epss/forecast_coreference_training.py`.

## DQ17: Matched Development Cache and R03 Protocol

**Status:** v1 preview complete; v1 full preparation failed after 39 CVEs.
The user's v2 preview and full preparation are complete and independently
verified. All 1,023 training and 197 validation endpoint CVEs are present.
R02 and all earlier artifacts remain frozen. This step creates graphs and
embeddings, not model predictions. The R03 definition below is fixed; its
runner and completed dataset findings are documented above.

Implementation verification: 28 COREF/cache tests pass, including five new
coverage/recovery tests. The v2 real-resource preflight passes on development
descriptions including the failing CVE, without creating cache files. No full
v2 graph preparation or dataset training was performed by these checks.

### V2 Coverage Correction

The v1 failure was reproduced for `CVE-2021-22822`. Two token nodes contain
only the Unicode replacement character U+FFFD, displayed as a question-mark
diamond. Their source spans are [70, 71) and [91, 92). spaCy retains them as
tokens, but the frozen SecBERT tokenizer normalizes them to empty strings.
The original all-nonwhitespace-token coverage check therefore incorrectly
treated this source-text/normalization case as possible truncation.

V2 permits missing embeddings only for tokens consisting entirely of U+FFFD.
Their original text, nodes, offsets and edges remain. Their vectors are zero,
their embedding-present mask is zero, and they are excluded from text-vector
pooling. The audit records every affected node and span. No replacement meaning
is invented and raw descriptions are not repaired. Ordinary text, mixed
replacement-character/word tokens and unexplained missing embeddings still
fail. Missing whitespace and replacement-character counts are separate.

The 39 completed v1 entries have been checked against their receipts and
source text. V2 copies their tensors byte-for-byte into a new output, alongside
the verified DQ16 reuse. Counts can overlap; reports also give the distinct
reused total. Preserve both v1 folders for provenance verification. Do not
resume the failed v1 command or edit its locked code, tests or reports.
The training protocol, v4 extractor, cohort and encoder remain unchanged.

### Matched Cache

- Preserve the existing endpoint cohort: **1,023 training / 197 validation
  CVEs**. No reselection based on errors, no split changes, no test graphs.
- Reuse the 56 DQ16 tensor bundles and 39 completed v1 entries byte-for-byte.
  Three entries overlap, giving 92 distinct reused CVEs. The other 1,128
  descriptions were parsed and embedded with the fixed v4 pipeline.
  The same frozen SecBERT weights, tokenizer, dimensions and spaCy resource
  are required; changes fail rather than silently replace the resources.
- Each CVE has shared node features and three masks: `no_coref`,
  `repeated_only`, `all_coref`. Nodes and non-COREF edges do not vary.
  No summaries, SMP, structured CVSS, VulnCheck fields or EPSS enter graphs.
- The v2 preview includes the six original hash-selected cases plus the failing
  training CVE: four training-role and three validation-role CVEs. The added
  case is a source-quality regression example, not outcome-based sampling.
  Full preparation requires this preview to pass verification first.
- Progress bars cover resource loading, preparation and verification.
  `--resume` verifies completed per-CVE receipts before continuing an unfinished
  output. Changed inputs, policies or completed files are rejected. Completed
  caches cannot be overwritten or resumed; use `--verify-only` to check them.
- Original DQ16 and DQ17 v1 caches remain unchanged. New outputs live in
  `01_tpg/05_workspace/forecast_graphs/coreference_development_v2/{smoke,full}/`.
  `graphs/*.pt` contain shared tensors and masks; `graphson/<mode>/*.json`
  contain unembedded inspectable graphs. `analysis.txt` and `analysis.json`
  report counts; `protocol.json` records the future experiment specification.
  Per-CVE receipts, source pins and `cache_lock.json` record provenance.

### Verified Full V2 Findings

The completed cache independently passes all frozen source, tensor, mask,
vocabulary and artifact checks. All 1,220 endpoint CVEs are present, with no
silent exclusions or test graphs. The original cache is unchanged.

| Edge configuration | Shared nodes | Total edges | Explicit COREF edges |
|---|---:|---:|---:|
| No COREF | 125,884 | 385,269 | 0 |
| Repeated-text COREF | 125,884 | 386,303 | 1,034 |
| All fixed v4 COREF | 125,884 | 386,611 | 1,342 |

The extra 308 edges in all-COREF are pronoun candidates. Only the selected
edges differ; node tensors, other edge types and retained edge properties match.

| CVEs with at least one relation | Training (1,023) | Validation (197) | Total (1,220) |
|---|---:|---:|---:|
| Any COREF | 397 | 88 | 485 |
| Repeated-text COREF | 253 | 70 | 323 |
| Pronoun COREF | 180 | 27 | 207 |
| Typed security relation | 463 | 84 | 547 |
| Product/version relation | 452 | 84 | 536 |

These groups overlap. There are no COREF edges in 735 CVEs, so all three
input graphs are identical for those CVEs. All-COREF and repeated-only differ
for only 27 validation CVEs. This limits the direct scope of that ablation;
models trained on different edges can still produce different predictions on
CVEs whose own graphs are identical, because their learned weights differ.

**Embeddings:** 116,943 of 125,884 nodes have embeddings. The 8,941 without
embeddings are fully accounted for: 1,220 document nodes, 1,296 paragraph
nodes, 6,098 topic nodes, 323 whitespace tokens and four U+FFFD-only tokens.
All ordinary text-token nodes and all sentence, entity, predicate, argument,
phrase, clause and mention nodes in this cache have embeddings. There are no
CONCEPT nodes. Feature width is 782: 13 type indicators, 768 contextual values
and one embedding-present mask. An absent embedding is not a missing node.

The four audited replacement-character tokens are in `CVE-2021-22822` (two),
`CVE-2021-22826` (one) and `CVE-2021-22827` (one), all training-role CVEs.
Source wording and graph structure remain intact. No ordinary missing text
embeddings were hidden by the exception.

**Security coverage:** 750 typed security edges comprise 725 `SEC_HAS_VERSION`,
24 `SEC_CAUSES` and one `SEC_HAS_SEVERITY`. Product/version links therefore
account for 96.7% of typed security edges. These are extraction candidates,
not verified facts; the cache does not establish broad security-relation
coverage or semantic accuracy. Generic copies of relations can coexist with
typed edges and should not be counted as independent facts.

**Sizes:** median nodes per CVE are 84 in training and 83 in validation, but
the means are 98.72 and 126.36 respectively. Validation has a longer large-graph
tail: maximum 1,327 nodes versus 636 in training. `CVE-2024-26626` is the largest
with 1,327 nodes and 4,103 edges. The R03 preflight should exercise large graphs
and batch memory without changing cohorts or silently truncating text.

**Conclusion:** the matched cache is ready for the fixed R03 development
experiment. These checks do not establish a winning edge configuration
or better forecasts. Keep the planned persistence and EPSS-only references,
three seeds and unchanged splits. No R03 fit or test scoring was performed.

Separate review: `05_results/02_evaluation/epss_day30_to_day60/coreference_development_v2_review/analysis.txt`
and `analysis.json`. The JSON includes per-CVE counts, node/edge-type totals,
size distributions and the original cache-lock checksum. Reproduction script:
`02_prediction/03_analysis/01_revision_dataset/review_forecast_coreference_development.py`.
Its existing output is protected; pass a fresh `--output-dir` under the same
evaluation-results directory to repeat the review.

### R03 Fixed Experiment Definition

Definition: `02_prediction/04_configs/coreference_development_v1.json`.

| Setting | Fixed choice |
|---|---|
| Task | Predict stored Day-60 EPSS using EPSS Days 0-30 and current description |
| Prediction anchor and horizon | Publication plus 30 days; forecast 30 days ahead |
| Comparisons | Three graph-edge configurations + EPSS-only GRU + persistence |
| Trained runs | Four variants times seeds 17, 29, 43: 12 fits |
| Graph branch | Existing two-layer, 32-unit EdgeTypeGNN, mean node pooling |
| EPSS branch | 16-unit GRU with existing observation/age masks |
| Fusion and output | Concatenation, direct-score sigmoid head; not residual prediction |
| Budget | Maximum 100 epochs per fit; patience 15 |
| Selection | Earliest strictly lowest validation MAE; min_delta = 0 |
| Optimizer | Adam; learning rate 0.001; weight decay 0.0001 |
| Loss and batches | Smooth L1 beta 0.01; batch eight; gradient norm clipped at 1.0 |
| Dropout | Zero for every variant, preserving the matched R02 setting |
| Randomness | CPU deterministic execution; one Torch thread; identical graph initial state and epoch order within each seed |
| References | Same EPSS-only model contract; persistence repeats last observed input score |
| Metrics | MAE primary; RMSE, signed error, Spearman where defined, equal-group MAE |

Stopping is per model; common epochs retain identical batch orders even when
stop times differ. Report every seed plus the three-seed mean and sample
standard deviation, not the best seed. Persistence is one deterministic
reference, not three independent runs. Compare paired errors on the same CVEs;
report prediction ranges, curves, selected epochs and input-missingness groups.
Checkpoint selection and all metrics use development data only. No test
performance, hyperparameter search or automatic selection of a COREF winner.

This is a retrospective experiment: current descriptions and SecBERT are not
proven available at the original Day-30 anchors. Candidate security relations
are not verified facts. Validation has informed development repeatedly and is
not a final unbiased estimate. Explicit COREF-edge changes do not remove all
coreference-derived information from the other graph structures.

### Commands

Run one at a time from the repository root with `CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/01_tpg/08_tests -p 'test_security_coreference*.py'
python EPSS_TPG/01_tpg/02_examples/01_scripts/prepare_security_coreference_development_v2.py --check-only
python EPSS_TPG/01_tpg/02_examples/01_scripts/prepare_security_coreference_development_v2.py --scale smoke
python EPSS_TPG/01_tpg/02_examples/01_scripts/prepare_security_coreference_development_v2.py --scale smoke --verify-only
python EPSS_TPG/01_tpg/02_examples/01_scripts/prepare_security_coreference_development_v2.py --scale full --smoke-dir EPSS_TPG/01_tpg/05_workspace/forecast_graphs/coreference_development_v2/smoke
python EPSS_TPG/01_tpg/02_examples/01_scripts/prepare_security_coreference_development_v2.py --scale full --verify-only
```

Stop if a check fails; do not bypass provenance checks. If preparation is
interrupted, repeat its preparation command with `--resume`. For full resume,
retain the same `--smoke-dir`. Offline resources are used by default;
`--allow-model-download` is an explicit opt-in, and fetched weights must still
match DQ16. These cache-preparation commands do not train. The separate R03
runner and commands are documented above.

## R02: Matched COREF Training Smoke

**Status:** user run complete; saved predictions and metrics independently
reproduced. This is a small training/serialization check, not a final performance comparison.
No full-data mode, test-split option or hyperparameter search is exposed.
Verification: all eight new synthetic tests and 11 existing TPG forecasting
tests pass. The real DQ16 preflight also passes, without optimizer steps,
validation scoring or result writes. No dataset training has been run during
implementation. The subsequent user run trained four models; this review
only reloads their saved checkpoints.

| Comparison | Inputs | Trainable components | Output |
|---|---|---|---|
| `no_coref` | DQ16 nodes and non-COREF edges + EPSS Days 0-30 | Relation-aware GNN + temporal GRU + regression head | Predicted Day-60 EPSS |
| `repeated_only` | Same nodes + repeated-text COREF and other edges + same EPSS | Same architecture and initial weights | Predicted Day-60 EPSS |
| `all_coref` | Same nodes + all fixed v4 edges + same EPSS | Same architecture and initial weights | Predicted Day-60 EPSS |
| `epss_only` | Same EPSS Days 0-30, no graph/text representation | Separate GRU + regression head | Predicted Day-60 EPSS |
| `persistence` | Last observed EPSS in Days 0-30 | None | Repeat that score for Day 60 |

All five use exactly the same **32 training-role and 24 validation-role CVEs**.
The target is Day-60 EPSS, 30 days after the Day-30 prediction anchor. It is
not KEV listing or a newly observed exploitation outcome. The three graph
variants share 768-dimensional frozen SecBERT embeddings. No embedding or
extractor is trained or rerun.

### Fixed Settings

- Seed 17; exactly three epochs; batch size eight; CPU with one Torch thread
  and deterministic algorithms enabled by the CLI.
- Two edge-type-aware GNN layers, 32 hidden units, mean node pooling; a
  16-unit mask-aware GRU processes the 31 daily input steps.
- Graph and temporal representations are concatenated into the existing
  sigmoid **direct-score** head. This is not the earlier persistence-residual
  model or the separate multiview classifier.
- Adam: learning rate 0.001, weight decay 0.0001; Smooth L1 beta 0.01;
  gradient-norm clipping at 1.0.
- **Dropout is disabled for all models in this smoke check.** The earlier
  R01 smoke used 0.1. Different edge counts otherwise produce different
  random message-dropout masks even with the same seed. This choice is fixed
  before results and is not a production hyperparameter decision.
- One saved GNN initial state is loaded by all three graph variants. A
  separately saved EPSS-only initial state is used for the temporal reference.
  Head bias uses only the mean training target, never validation targets.
- The exact three epoch orders are generated once and shared across all
  four trained models. Each completes all three epochs. The earliest lowest
  validation-MAE checkpoint is selected; no best-seed selection is possible.

The implementation reuses the existing `TPGRegressor`, graph batching,
temporal input channels, checkpoint restore and metric functions. The small
training loop records actual initial-state checksums, batch orders and branch
gradient diagnostics. It does not change frozen R01 code. Masks and observation
ages remain present for missing past EPSS; no missing target is imputed.
Only selected development records enter fitting and evaluation. Shared source
JSON files contain other records, but they are not selected for this run.

### Outputs and Interpretation

Default output: `05_results/02_evaluation/epss_day30_to_day60/coreference_training_smoke_v1/`.

| File | Purpose |
|---|---|
| `summary.json`, `summary.txt` | Settings, five matched validation results, limitations; JSON includes branch-gradient diagnostics |
| `predictions.json` | One row per validation CVE, actual Day-60 EPSS, anchor/target dates and five predictions |
| `training_history.json` | Training loss and training/validation MAE for all epochs |
| `training_curves.png` | Epoch on x; EPSS mean absolute error on y; solid training and dashed validation lines |
| `initialization/*.pt` | Two saved initial states: shared GNN-fusion state and EPSS-only reference state |
| `batch_orders.json` | Actual shared training-CVE order in each epoch |
| `checkpoints/*.pt` | Four selected checkpoints, initialization identity and DQ16 cache-lock checksum |
| `evaluation_lock.json`, `run_status.json` | Artifact checksums and completion/failure state |

Report MAE, RMSE, signed error, Spearman correlation where defined and
equal-group MAE on the same 24 validation CVEs. An error of 0.01 is one EPSS
percentage point. Gradient diagnostics count steps with available/nonzero
gradients and their pre-clipping absolute sums for the temporal encoder,
node projection, graph layers and output head. They indicate gradient flow,
not incremental predictive value or causality.

Only six sample CVEs differ between repeated-only and all-COREF graphs.
The sample includes earlier reviewed examples; validation has been inspected
repeatedly. Current-description historical availability and pretrained-model
cutoff limitations remain. Do not declare a winning architecture from this
one-seed, three-epoch smoke test. Review output integrity and training behavior
before deciding on any larger matched development run. Test scoring remains
disabled, and expansion is not automatic.

### Verified Results

All results below use the same 24 validation CVEs. MAE is the average absolute
difference between the predicted and stored Day-60 EPSS score; lower is better.

| Model or reference | Validation MAE | Validation RMSE | Selected epoch |
|---|---:|---:|---:|
| Persistence | 0.00008750 | 0.00037418 | Not trained |
| No COREF | 0.01787610 | 0.03076149 | 3 |
| Repeated-text COREF only | 0.01787481 | 0.03076099 | 3 |
| All fixed v4 COREF | 0.01787397 | 0.03076063 | 3 |
| EPSS-only GRU | 0.04816025 | 0.04967015 | 3 |

- Persistence exactly matches the stored Day-60 score for 22 of 24 CVEs.
  This does not establish that their entire future EPSS curves were flat.
  No trained model has lower absolute error than persistence on any of these
  24 CVEs. The other two are `CVE-2024-23749` and `CVE-2024-24495`.
- Graph-model predictions span approximately 0.01336-0.01438, while actual
  scores span 0.00042-0.15116. The EPSS-only predictions span
  0.04691-0.04713. These are nearly constant predictions, not useful evidence
  of accurate CVE-specific forecasts.
- Every monitored branch has finite, nonzero gradients in all 12 optimizer
  steps. This verifies gradient flow, not useful predictive contribution.
  Gradient sums should not be compared as branch-importance scores.
- Training and validation errors decrease through epoch three for every
  model. Every selected checkpoint is at the budget boundary; convergence
  has not been demonstrated. The training target mean is 0.05327969 versus
  0.00765042 in validation. The short budget, direct-score head and different
  target distributions are relevant observations, not a proven explanation.
- The all-COREF versus no-COREF MAE difference is only 0.00000213.
  Do not select a COREF policy or claim a general improvement from this sample.

Detailed review: `05_results/02_evaluation/epss_day30_to_day60/coreference_training_smoke_v1_review/analysis.txt`
and `analysis.json`. Original predictions, checkpoints and locks are unchanged.

Recommended next stage, **not run or implemented by this review**: prepare a
new versioned development cache for the existing 1,023 training and 197
validation endpoint CVEs, retaining all three edge configurations. Define a
longer, fixed multi-seed matched protocol before training. Keep persistence
and EPSS-only references, frozen splits and the unopened test set. The old
full graph cache must not be presented as the corrected v4 cache.

### Commands

Run from the repository root with CodeBERTFusion active:

The default output now exists. To recheck this completed run, use only the
`--verify-only` command below. The creation commands are retained as protocol
documentation, not instructions to overwrite or retrain the completed run.

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/01_revision_dataset -p 'test_forecast_coreference_smoke.py'
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_smoke.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_smoke.py
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_smoke.py --verify-only
```

`--check-only` verifies the cache, initializes models and runs small
training-role forward batches; it takes no optimizer steps, scores no
validation outcomes and writes no results. The command without flags trains
the four small models and reports the matched persistence reference.
`--verify-only` reloads selected checkpoints and reproduces saved validation
predictions/metrics; it does not retrain. Existing outputs are never
overwritten. Preserve a failed output and choose a new `--output-dir` under
the evaluation-results directory if a rerun is needed.

Implementation: `02_prediction/01_package/epss/forecast_coreference_smoke.py`.
The CLI and synthetic regression tests live beside the other revision
experiment scripts in `02_prediction/03_analysis/01_revision_dataset/`.

## DQ16: Shared-Embedding Cache for COREF Variants

**Status:** user's cache completed and verified on 2026-10-19.
The offline preflight verified the fixed DQ15 graphs, frozen development
roles, Day-60 endpoint eligibility and locally available SecBERT resources.
It did not encode dataset text or create a cache. Synthetic tests cover
embedding sharing, masks, save/reload, source preservation, leakage guards,
missing token embeddings, split eligibility/group separation and the GNN
forward interface. Synthetic forward checks are not dataset predictions.
All 96 security regression tests pass, including ten new cache tests.
A read-only tokenizer-offset check across the fixed sample found 2,851 TOKEN
nodes: 2,843 have text-piece coverage; the other eight contain only line
breaks. This checks token alignment, not generated embedding quality.

### Verified Cache Results

The independent `--verify-only` run passed all 60 artifact checksums, source
and code pins, all 56 tensor-bundle checks and their three edge configurations.
The encoder was not rerun. A separate read-only inspection of the saved
tensors reproduced the following coverage. No embedded node has an all-zero
embedding, and the verifier found no non-finite values or token coverage
failures. These are numerical/integrity checks, not semantic-quality scores.

| Node type | Total nodes | With embeddings | Without embeddings |
|---|---:|---:|---:|
| Document | 56 | 0 | 56 |
| Paragraph | 57 | 0 | 57 |
| Sentence | 127 | 127 | 0 |
| Token | 2,851 | 2,843 | 8 |
| Entity | 569 | 569 | 0 |
| Predicate | 243 | 243 | 0 |
| Argument | 440 | 440 | 0 |
| Noun phrase | 616 | 616 | 0 |
| Verb phrase | 123 | 123 | 0 |
| Clause | 115 | 115 | 0 |
| Mention | 6 | 6 | 0 |
| Topic | 280 | 0 | 280 |
| **Total** | **5,483** | **5,082** | **401** |

There are no CONCEPT nodes in this sample. Document, paragraph and topic
nodes are deliberately excluded from contextual span embedding by the shared
core. The eight remaining unembedded nodes contain line breaks only. These
401 nodes are not discarded: they retain their node-type indicators, edges,
zero embedding vectors and an embedding-present indicator of zero. Every
entity and every token containing text has an embedding.

All 56 node matrices have **782 columns: 13 node-type indicators + 768
SecBERT values + one embedding-present indicator**. The encoder is the frozen
`jackaduma/SecBERT` revision `7c603df5bc4c5ba9c731bcc2ea0ab2db36e104cb`.
Unused checkpoint prediction-head and pooler keys are recorded in the
manifest; these components are not used to produce contextual token vectors.

| Saved role | CVEs | Nodes | Embedded nodes | Edges: no COREF | Edges: repeated only | Edges: all COREF |
|---|---:|---:|---:|---:|---:|---:|
| Training | 32 | 3,775 | 3,547 | 11,878 | 11,962 | 11,964 |
| Validation | 24 | 1,708 | 1,535 | 5,190 | 5,198 | 5,202 |
| **Total** | **56** | **5,483** | **5,082** | **17,068** | **17,160** | **17,166** |

These are three views of the same cohort, not additional CVEs. One shared
node matrix and one full edge table are stored per CVE; masks select the
appropriate edges. The result is ready for a matched training smoke test,
not proof that COREF links or embeddings improve predictions. No validation
metrics were calculated and no test graphs were opened.

The cohort remains 56 CVEs: 32 training-role and 24 validation-role records.
There is no resampling, silent dropping, full expansion or test graph creation.
Each CVE's text is embedded once with frozen SecBERT on CPU. The shared core
`tpg.embeddings.embed_graph` performs sentence-window encoding and source-span
pooling. Neither spaCy nor security extraction runs again. The unused BERT
pooler is disabled so no randomly initialized pooler enters the model identity;
only contextual token representations are used. Missing encoder weights fail
the run instead of creating random fallback parameters.

| Stored component | Purpose | Shared between variants? |
|---|---|---|
| Node features `x` | Node-type indicators, contextual embeddings and an embedding-present indicator | Yes, stored once |
| Pooled text embedding | Mean of embedded TOKEN vectors, compatible with the text baseline interface | Yes, stored once |
| Full edge table | Endpoints, named-relation vocabulary IDs and existing edge features | Yes, stored once |
| `no_coref` mask | Select every edge except COREF | Separate boolean mask |
| `repeated_only` mask | Select non-COREF edges plus repeated-text candidate links | Separate boolean mask |
| `all_coref` mask | Select all fixed v4 edges | Separate boolean mask |

A mask value of true keeps an edge. Edge order and attributes are checked
against each saved DQ15 graph. The vocabulary is fixed even when a variant
has no COREF edges. TOKEN nodes containing text must have embeddings; missing
ones stop generation rather than silently truncating input. Whitespace-only
tokens, such as line breaks, may lack embeddings and remain as nodes with a
zero indicator; they are excluded from pooled text averages. The 56 saved
graphs contain eight such line-break tokens. Structural nodes without
embeddings remain present with zero vectors and an embedding-present value
of zero. Counts by node type are saved for inspection.

No EPSS scores, prediction targets or VulnCheck fields enter graph bundles.
The cache records CVE IDs for a later separate join to the frozen temporal
inputs and targets. Those identifiers are not forward-pass features. Current
descriptions remain retrospective: historical availability at Day 30 is not
established. Frozen pretrained weights also do not establish a historical
pretraining-data cutoff. This stage makes no forecasting-performance claim.

Implementation: `02_prediction/01_package/epss/forecast_coreference_cache.py`.
CLI: `01_tpg/02_examples/01_scripts/prepare_security_coreference_cache.py`.
Default output: `01_tpg/05_workspace/forecast_graphs/coreference_v1/smoke/`.

| Output | Contents |
|---|---|
| `graphs/<CVE>.pt` | One safe tensor bundle per CVE: shared features, full edges, three masks |
| `manifest.json` | Cohort roles, vocabulary, encoder/tokenizer fingerprints, package versions and source/code checksums |
| `analysis.json` | Per-CVE node-embedding coverage and edges retained in each configuration |
| `analysis.txt` | Plain-language explanation and aggregate counts |
| `cache_lock.json` | Checksums of all completed artifacts |
| `run_status.json` | Running, complete or failed state |

The new schema has its own verified `load_cache(directory, mode)` adapter. It
returns a manifest and the existing graph-item format for that mode. It does
not start training or load targets. Shared node tensors should be treated as
read-only. Do not point the older training CLI directly at this new schema;
a matched-training entry point is a subsequent step. Existing checkpoints
are not retroactively compatible with the corrected graph experiment.

The original creation commands are retained below. The default cache now
exists: use only `--verify-only` to recheck it. Do not rerun creation over it.
Commands run from the project root in the CodeBERTFusion environment:

```bash
python -m unittest discover -s EPSS_TPG/01_tpg/08_tests -p 'test_security_coreference*.py'
python EPSS_TPG/01_tpg/02_examples/01_scripts/prepare_security_coreference_cache.py --check-only
python EPSS_TPG/01_tpg/02_examples/01_scripts/prepare_security_coreference_cache.py
python EPSS_TPG/01_tpg/02_examples/01_scripts/prepare_security_coreference_cache.py --verify-only
```

`--check-only` verifies inputs and loads/fingerprints the frozen encoder but
does not generate embeddings or output files. `--verify-only` checks an
existing cache without loading the encoder, including tensor/source alignment
and exact edge masks. Neither flag trains anything. Downloads are disabled
unless `--allow-model-download` is explicitly supplied during preflight or
creation; current local SecBERT resources are available. An existing output
is never overwritten. After a failed run, preserve it and choose a new
`--output-dir` below `forecast_graphs/coreference_v1/`.

Embedding coverage and cache verification are complete. The next step is to
implement a small matched training run. Keep the three configurations'
CVEs, EPSS inputs, targets, seeds, architecture and optimization budget fixed.
The reviewed small sample is for debugging, not an unbiased evaluation set.

## DQ15: Matched Coreference Edge Preflight

**Status:** user's preview completed and verified on 2026-10-19. This prepares
an edge comparison, not a new forecasting experiment or performance result.
Verification: eight new regression tests and all 78 existing product/version
tests pass. The read-only preflight passed on all 56 saved CVEs; it created no
new dataset artifacts. Synthetic save/reload and tampering checks also pass.
The completed preview's file checksums, source/code pins, all 168 graph
reconstructions, per-CVE counts and text report now pass `--verify-only`.

| Variant | Nodes retained | Total edges retained | COREF edges retained | Product/version links retained | CVEs with edges removed versus v4 |
|---|---:|---:|---:|---:|---:|
| `no_coref` | 5,483 | 17,068 | 0 | 43 | 23 |
| `repeated_only` | 5,483 | 17,160 | 92 | 43 | 6 |
| `all_coref` | 5,483 | 17,166 | 98 | 43 | 0 |

These are totals over the same 56 CVEs for each variant, not three distinct
cohorts. Removing all COREF edges removes 86 training-role edges and 12
validation-role edges. Removing pronoun edges alone removes two training-role
edges and four validation-role edges, one in each of six CVEs. The remaining
33 CVEs have no COREF edges, so their node/edge content is identical across
all variants. Provenance metadata differs intentionally.

The preview establishes controlled edge differences, not predictive benefit.
Only six graphs differ between `repeated_only` and `all_coref`, so this bounded
sample is especially limited for assessing the effect of pronoun candidates.

| Variant | COREF edges retained | What it isolates |
|---|---|---|
| `no_coref` | None | Graph without explicit candidate coreference edges |
| `repeated_only` | Method `repeated_surface` only | Candidate links between repeated text mentions |
| `all_coref` | All fixed v4 COREF edges | Repeated-text and pronoun candidates together |

Each variant uses exactly the same 56 saved development CVEs (32 training-role,
24 validation-role), node IDs, node features and non-COREF edges. Security
product/version relations are not removed. The selection uses extraction
provenance, not DQ14 verdicts, prediction errors or targets. Unknown methods
or non-candidate COREF edges stop the preview for review. The full security
type vocabulary remains fixed, even when a relation has no edges.

This is **edge-only**, not removal of all coreference information: mention
nodes, cluster properties and other derived structure remain. Unembedded PyG
exports are checked in memory; these are not model-ready embedding caches.
The old trained models/caches cannot serve as matched v4 training runs.

The DQ14 counts (17/20/7) remain provisional automated judgments. Five links
had unsupported pronoun antecedents; 15 lacked product-specific release
identity evidence. Equal values can still be useful weak associations.
The FTP/SCM judgments used different conceptual scopes, so a consistent human
rubric is needed before treating these judgments as a reference standard.
The original DQ14 ledger and reports are preserved, not retroactively edited.

Implementation: `01_tpg/10_extensions/tpg_coreference_ablation_v1.py`.
CLI: `01_tpg/02_examples/01_scripts/preview_security_coreference_ablation.py`.
Output: `01_tpg/05_workspace/coreference_ablation_v1/smoke/` with `analysis.txt`,
`analysis.json`, three graph folders, completion status and checksums.
The command checks every saved variant by reconstructing it from fixed v4.
It refuses overwriting an existing output. No NLP rerun, embeddings, training,
model inference, cohort expansion or test inspection is performed.

The original creation commands, from the project root, are below. The default
output now exists: use only `--verify-only` to recheck this completed run.
Do not delete or overwrite it to rerun creation.

```bash
python -m unittest discover -s EPSS_TPG/01_tpg/08_tests -p 'test_security_coreference_ablation.py'
python EPSS_TPG/01_tpg/02_examples/01_scripts/preview_security_coreference_ablation.py --check-only
python EPSS_TPG/01_tpg/02_examples/01_scripts/preview_security_coreference_ablation.py
python EPSS_TPG/01_tpg/02_examples/01_scripts/preview_security_coreference_ablation.py --verify-only
```

Next, prepare a small separate model-ready cache
with shared embeddings and these three edge masks. Any later training should
hold CVEs, targets, EPSS histories, architecture, seeds, optimizer, budget and
validation rules fixed. Compare persistence and EPSS-only references on the
same records. The sample is for debugging, not an unbiased accuracy estimate;
historical description availability remains unverified. Training and full
expansion require a subsequent decision.

## 1. Read the Status Correctly

- **Completed, validation only:** implemented and run on the current prepared
  data; no final test performance was calculated.
- **Planned, labels available:** required targets can be constructed or already
  exist, but the experiment has not been trained. Feasibility is not a result.
- **Conditional:** additional input-time, label, coverage or adapter checks are
  required before a defensible experiment.
- **Blocked:** an essential scientific requirement is unresolved.
- **Archived:** documentation from previous datasets exists. It was not rerun
  or independently revalidated for the current dataset and target.

Future experiments are listed to define their questions, not to promise that
they will outperform simple baselines. A future EPSS score, future VulnCheck
listing, and actual exploitation are three different targets. We do not have
a generally verified first-exploitation outcome in this adapter.

## 2. Shared Rules for Current Forecasting

Day 0 is the stored NVD publication date. Inputs cover Days 0-30, with 31
possible daily scores. The prediction anchor is Day 30; the target is the
score on Day 60, **30 days after the prediction anchor**.

| Split | Rule | Endpoint-eligible CVEs |
|---|---|---:|
| Training | Anchor and target before 2024-01-01 | 1,023 |
| Validation | Anchor from 2024-01-01; anchor and target before 2025-01-01 | 197 |
| Reserved test | Anchor from 2025-01-01 up to, but excluding, 2026-01-01 | 2,670 |

The score-date observation cutoff is 2026-09-29. A missing endpoint is never
made into zero. At least one input score must exist. The loader verifies the
source and prepared-data checksums, task eligibility and split policy.
Related CVEs linked by the declared description, post-text and URL rules
cannot cross active splits: crossing groups are excluded, not moved backward
in time. These rules do not detect every semantic duplicate and broad links
can exclude many CVEs. Grouping evidence is used for splitting, not prediction.

The split remains the conservative development design. It has not been
changed after inspecting errors. Training is concentrated in June 2022;
validation is small. Do not optimize split boundaries against observed errors.
Test labels are checked for data integrity but **test predictions and metrics
remain disabled** in the new baseline entry points.

### Why Not Train on All 5,968 CVEs?

All source CVEs were processed, not randomly downsampled. The source dataset
is larger than the training set because it also contains validation/test data,
records excluded by time/group rules, and records with insufficient observations
for a particular target. These are separate decisions.

| Assigned role | All assigned CVEs | Eligible for Day-60 endpoint | Eligible for Days 31-60 minimum/maximum |
|---|---:|---:|---:|
| Training | 1,156 | 1,023 | 797 |
| Validation | 199 | 197 | 163 |
| Reserved test | 2,671 | 2,670 | 2,670 |
| Excluded by split rules | 1,942 | 1,726 | 1,682 |
| Total source records | 5,968 | 5,616 | 5,312 |

These are existing preparation counts, not new test results. The first four
rows partition the source; the last row is a total. Data-eligible excluded
records still cannot enter an active experiment. Test records remain reserved.
The minmax task requires every future day to be observed because a missing
score could be the true minimum or maximum. Missing input days are allowed
provided at least one input score exists. No missing score becomes zero.

The 1,942 split exclusions comprise 1,548 CVEs linked across periods, 358 whose
target windows cross the validation-to-test boundary, 21 whose target windows
cross the training-to-validation boundary, and 15 beyond the chosen test period.
These reasons partition exclusions; do not add missing-data counts to them.

The grouping rule deserves review: the largest connected group has 1,377
CVEs. A common URL or text can connect other records transitively; that does
not establish that every member is a duplicate. **DQ07** below audits coverage
and creates a manual evidence queue. It does not automatically loosen grouping.
Any scientifically justified correction requires a separately versioned
preparation and matched reruns, not replacement of these results. Training
on reserved test CVEs would remove the independent evaluation, not improve
the validity of the experiment.

Scores are dated observations; their exact historical release times have not
been recovered. Current results are retrospective daily-score forecasts, not
proof that all inputs were available at a particular intraday decision time.

### Common Metrics

| Code | Metrics | Meaning |
|---|---|---|
| R | MAE, RMSE, Spearman; error relative to persistence; equal-group MAE | MAE is average absolute score error. RMSE emphasizes large errors. Spearman measures ordering. Group MAE gives each related group equal weight. |
| C | Macro-F1, balanced accuracy, per-class recall and confusion matrix | Used for discrete direction/trend classes; report every class, not only overall accuracy. |
| B | Average precision, ROC-AUC, Brier score, calibration; precision/recall at a validation-chosen threshold | Used only for defined binary outcomes. Handle undefined metrics when a class is absent. |

Regression errors use the 0-1 EPSS scale: 0.01 is one percentage point.
Persistence skill is `1 - model_MAE / persistence_MAE`; it is undefined when
the persistence MAE is zero. Negative skill means worse error than persistence.
Use identical CVEs, anchors and targets for each comparison. A source-available
subset needs a newly matched baseline; it cannot be compared directly with a
larger-cohort headline. Related CVEs and repeated posts are not independent
replicates. Seed standard deviations are not confidence intervals.

## 3. Current Completed Runs

| ID | Input -> model -> target/output | Fixed design | Status and evidence |
|---|---|---|---|
| F00 | Last available Day 0-30 EPSS -> persistence -> one Day-60 score | No fitting; records age of last score | Completed, validation only. [Report](../05_results/02_evaluation/epss_day30_to_day60/persistence_validation/README.md). |
| F01-Ridge | Seven past-score statistics -> scaled ridge -> one Day-60 score | Training-only scaling; alpha=1; SVD solver; clip output to [0,1]; no search | Completed, validation only. [Report](../05_results/02_evaluation/epss_day30_to_day60/ridge_validation/README.md). |
| F01-Huber | Same seven past-score statistics -> fixed robust linear regression -> one Day-60 score | Same frozen cohort/scaler; epsilon=1.35, alpha=0.0001; no search | Completed by user, validation only; verified. [Run](../05_results/02_evaluation/epss_day30_to_day60/huber_validation/README.md), [review](../05_results/02_evaluation/epss_day30_to_day60/huber_validation_review/README.md). |
| F01-Tree | Same seven past-score statistics -> fixed shallow tree -> one Day-60 score | Depth <=3, minimum leaf size 20, seed 17; no search | Completed by user, validation only; verified. [Run](../05_results/02_evaluation/epss_day30_to_day60/tree_validation/README.md), [review](../05_results/02_evaluation/epss_day30_to_day60/tree_validation_review/README.md). |
| F04-Linear | Past-score statistics -> two-output ridge and matched references -> Days 31-60 minimum/maximum | Complete future required; separate 797/163 cohort; fixed alpha=1, clipping/sorting | Completed by user, validation only; verified. Different target from endpoint rows. [Review](../05_results/02_evaluation/epss_day30_to_day60/minmax_linear_validation_review/README.md). |
| F04-GRU | Mask-aware score sequence -> GRU with ordered head -> Days 31-60 minimum/maximum | Same 797/163 cohort; seeds 17/29/43; per-seed validation checkpoint selection | Completed by user, validation only; verified. All seeds retained. [Review](../05_results/02_evaluation/epss_day30_to_day60/minmax_gru_validation_review/README.md). |
| F04-Residual | Latest observed EPSS plus two GRU corrections -> sorted bounded future extrema | Same 797/163 cohort, loss, optimizer, budget and seeds; persistence initialization; epoch zero diagnostic only | Completed by user, validation only; verified. Much better than direct GRU, but all seeds slightly worse than persistence on joint MAE. [Review](../05_results/02_evaluation/epss_day30_to_day60/minmax_residual_validation_review/README.md). |
| F02-GRU | Past daily scores, missingness and age -> small GRU -> one Day-60 score | One layer, 16 hidden units; sigmoid head; Smooth L1 beta=0.01; three fixed seeds; validation checkpoint selection | Completed, validation only; user rerun verified. [Run report](../05_results/02_evaluation/epss_day30_to_day60/gru_validation_user_run/README.md), [post-run findings](../05_results/02_evaluation/epss_day30_to_day60/gru_validation_user_run_review/README.md). |
| F02-Residual | Same past history -> GRU correction added to last observed EPSS -> one Day-60 score | Zero-initialized correction head, bounded output; same split, seeds, optimizer and budget; epoch zero diagnostic only | Completed by user, validation only; verified. [Run report](../05_results/02_evaluation/epss_day30_to_day60/gru_residual_user_run/README.md), [post-run findings](../05_results/02_evaluation/epss_day30_to_day60/gru_residual_user_run_review/README.md). |

F01-Ridge is the squared-error linear member of the originally proposed F01
family. The separate Huber and shallow-tree members are also completed and verified.
F02-GRU is a mask-aware input model: it processes all days with missingness
flags, rather than claiming to skip missing days or use a specialized GRU-D.

### Matched Endpoint Results

| Model | Validation CVEs | MAE (lower is better) | RMSE (lower is better) | Spearman (higher is better) |
|---|---:|---:|---:|---:|
| F00 persistence | 197 | 0.00909675 | 0.08264449 | 0.89710785 |
| F01 ridge | 197 | 0.01075409 | 0.08224354 | 0.72362091 |
| F01 Huber | 197 | 0.00909609 | 0.08264314 | 0.88349212 |
| F01 tree | 197 | 0.02477230 | 0.09641363 | 0.45751343 |
| F02 GRU seed 17 (epoch 6) | 197 | 0.03298903 | 0.14073863 | 0.70692951 |
| F02 GRU seed 29 (epoch 75) | 197 | 0.01413642 | 0.08320599 | 0.50848971 |
| F02 GRU seed 43 (epoch 80) | 197 | 0.01470110 | 0.08312206 | 0.52466022 |
| F02 mean of three seed metrics | 197 per seed | 0.02060885 | 0.10235556 | 0.58002648 |
| F02-Residual seed 17 (epoch 2) | 197 | 0.00914799 | 0.08263532 | 0.89221966 |
| F02-Residual seed 29 (epoch 5) | 197 | 0.00911757 | 0.08264288 | 0.89187624 |
| F02-Residual seed 43 (epoch 15) | 197 | 0.00909911 | 0.08264372 | 0.89366613 |
| F02-Residual mean of three seed metrics | 197 per seed | 0.00912156 | 0.08264064 | 0.89258734 |

Huber reduces MAE by only 0.00000066 versus persistence (about 0.00727%
relative), with lower rank correlation. This tiny validation difference does
not establish a practically meaningful or statistically supported gain. Its
predictions remain very close to persistence and still miss the large increases.

The shallow tree has 172.32% higher MAE than persistence, higher RMSE and lower
rank correlation. Its four leaf averages replace individual scores with a small
set of forecasts. Do not promote this run or tune its depth from these results.

Ridge has higher average absolute error despite slightly lower RMSE. Retain
persistence as the reference. These results do not establish exploitation
prediction, a causal effect, or the performance of a TPG. All three GRU runs
have higher MAE than persistence. The sample standard deviation across seeds
is 0.01072527 for MAE, 0.03324074 for RMSE and 0.11019826 for Spearman. This
is not a confidence interval or the performance of an ensemble.

Seed 17 stopped after 21 epochs; seeds 29 and 43 ran to the fixed 80-epoch
limit. The latter two improved much later in training. This shows sensitivity
to initialization and the stopping rule; it does not prove every GRU is
inferior. Do not silently extend the budget or drop seed 17. A new budget,
head or loss is a separately named experiment. Validation selected these
checkpoints, so independent final performance still requires the sealed test.

### User Rerun Verified

The user-run models, training histories, predictions and summary exactly match
the earlier run. This is a reproducibility check, not an additional independent
sample or three more independent seeds. All 197 validation predictions per seed
were reproduced from saved model states. Source/output/code hashes, recorded
targets, paired baseline predictions, metrics and stopping rules passed review.
No training or test scoring was performed during this review.

The central findings are:

- The last observed input score equals the saved Day-60 score for 162 of 197
  CVEs. Persistence is exact on these cases; the intervening daily scores need
  not have remained constant.
- Seed 17 predicts only 0.00971514 to 0.01131528, despite actual Day-60 scores
  ranging from 0.00042 to 0.95432. It has learned little score variation at
  its selected checkpoint. This observation does not identify a single cause.
- Seeds 29 and 43 improve later, but have higher absolute error than
  persistence on 189 and 190 CVEs respectively. Seed 17 is worse on 194.
- For CVE-2024-24919, persistence correctly predicts 0.94504; the GRU predicts
  0.01057, 0.79256 and 0.77313 across the three seeds. Even a large score that
  is already visible in the input can be underestimated by the direct head.
- No prediction/alignment/checkpoint discrepancy was found in these checks.
  This is not proof that every part of the pipeline or source data is correct.
  TPG and SMP were not inputs, so these runs do not test their value.

The [review script](03_analysis/01_revision_dataset/review_forecast_sequence.py)
saves analysis in TXT/JSON and Markdown without editing the locked run files.
To repeat only verification and analysis (no training), run from the repository
root with CodeBERTFusion active:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_sequence.py
```

## 4. EPSS Experiment Families

All F00-F10 IDs from the original feasibility catalogue are retained here.
Unless stated otherwise, inputs are available through Day 30 and the future
window is Days 31-60. Planned experiments below have not been trained.

| ID | Inputs and model | Required target -> output | Horizon | Metrics | Status / key requirement |
|---|---|---|---|---|---|
| F00 | Latest past EPSS; persistence | Observed endpoint -> one score | Day 30 -> Day 60 | R | Completed. Day-90 variant is only planned. |
| F01 | Past-score statistics; ridge, Huber regression or a small tree model | Observed endpoint -> one score | Day 30 -> Day 60 | R | Ridge, Huber and shallow tree completed and verified. Training-only preprocessing; validation only. |
| F02 | Score sequence plus masks and age; GRU, later temporal CNN | Observed endpoint -> one bounded score | Day 30 -> Day 60 | R | GRU implemented; CNN and matched MSE-loss comparison planned. |
| F03 | Past EPSS; statistics or sequence classifier | Exact Day-60 minus Day-30 change -> increase/stable/decrease probabilities | 30 future days | C | Labels available, but current validation has no decrease cases. Do not claim adequate three-class evaluation. Fixed epsilon=0.01. |
| F04 | Past EPSS; extrema baselines, linear model and GRUs with direct/residual ordered heads | All 30 future scores -> minimum and maximum | Days 31-60 | MAE/RMSE per bound, range-width error | F04-Linear, F04-GRU and F04-Residual completed and verified. Require every future day; validation only. Not a confidence interval. |
| F05 | Past EPSS with anchor below a justified threshold; binary head | Future threshold crossing -> one probability | 30 future days; later 60 | B | Conditional. Threshold must be predeclared; one crossing resolves a positive, but a zero requires complete non-crossing follow-up. Not exploitation. |
| F06 | Past EPSS; sequence encoder and 30-output decoder | Per-day future scores/masks -> 30 scores | Days 31-60 | Per-lead-day and per-CVE MAE/RMSE, persistence skill, coverage | Planned, masked targets available. Average loss within each CVE before across CVEs; never feed future targets to the encoder. |
| F07 | Time-valid description -> shared TPG -> GNN -> regression head, no EPSS | Observed future endpoint -> one score | Publication -> Day 30/60/90; separate cold-start tasks | R; matched non-graph text baseline | Blocked as historical forecasting until description availability is established. Do not compare a publication-time model with Day-30 forecasts as the same task. |
| F08 | Time-valid description TPG-GNN + EPSS encoder -> fusion | Observed endpoint -> one score | Day 30 -> Day 60 | R on matched F02 cohort | Main future TPG experiment; blocked on historical description evidence and new regression fusion integration. |
| F09 | Time-valid description, SMP, GitHub/non-GitHub summaries, EPSS and legitimate structured data -> source-aware fusion | Observed endpoint -> one score; extrema only as separate task | Day 30 -> Day 60 | R and source-removal ablations | MS01 implements a bounded four-text-source adapter with explicit retrospective restrictions and masks. Multi-source graph/fusion training is not implemented yet; authenticated historical use remains conditional. Structured predictors are not included in MS01. |
| F10 | Past EPSS -> history encoder and trend classifier | Full future-window slope with a fixed tolerance -> rising/flat/falling probabilities | Days 31-60 | C | Planned. Needs explicit slope/tolerance and coverage policy. Future-window trend is not the same as endpoint change. |
| D01 (Delta) | Past EPSS -> regression head | Exact Day-60 minus Day-30 score -> signed change | 30 future days | Change MAE/RMSE; reconstructed endpoint errors | Planned, labels available. Day 30 itself must be observed; last available score is not an interchangeable anchor. |
| H90 | Repeat F00/F01/F02 on unchanged historical inputs | Day-90 score -> one score | Day 30 -> Day 90, 60 future days | R | Planned. Regenerate target eligibility, maturity and exclusions. Do not silently reuse Day-60 cohorts. |
| F02-Residual | Same GRU input and encoder; last observed score plus a learned correction | Observed Day-60 endpoint -> bounded score | Day 30 -> Day 60 | R, all fixed seeds and matched baseline | Completed by user and verified. Much better than the direct GRU, but all seeds have slightly higher MAE than persistence. Head and initialization changed together. |

Before splitting, Day-60 endpoint labels with input exist for 5,616 CVEs;
delta/direction for 5,612; exact future min/max for 5,312; at least one future
trajectory score for 5,624. These are not training counts. The current
min/max cohort has 797 training and 163 validation CVEs; delta/direction has
1,023 training and 195 validation CVEs. Each task needs its own eligible IDs.
Forecasting 90 days *after* Day 30 would need Day 120, outside the saved window.

### F04-Residual: Ordered Corrections to the Latest Score

**Status: completed by the researcher and verified; validation only.** No training
on the actual dataset was performed during implementation or review. Entry point:
`01_package/epss/forecast_minmax_residual.py`.

This follow-up was declared after inspecting F04-GRU and DQ05. The question is
whether giving the model a direct connection to the latest observed EPSS helps
it distinguish CVEs, rather than reconstructing the score level from the GRU
alone. This is a new comparison, not a repair to the saved F04-GRU results.

#### Fixed Design

| Component | Rule |
|---|---|
| Task | Inputs through Day 30 after NVD publication; predict the minimum and maximum EPSS over Days 31-60. |
| Cohorts | Same 797 training and 163 validation CVEs, unchanged chronological/group-safe splits and complete-future eligibility. |
| Missing data | Original masks and observation-age channels. No target filling, new exclusions or dataset repair. |
| Encoder | Same one-layer, 16-unit GRU and five input channels as F04-GRU. |
| Base score | Latest observed EPSS in Days 0-30. This may be earlier than Day 30; an observed zero remains valid. |
| Head | Two linear values transformed by tanh into signed corrections, each added to the base score. |
| Ordering and bounds | Stable ascending sort of the two proposals, followed by clipping each to [0,1]. Loss is computed on these final outputs. |
| Initialization | Both head weights and biases zero. Both outputs initially equal the base in float32; no target statistics are used. |
| Seeds | 17, 29, 43, all retained. Initial encoder weights and batch-order generator match the corresponding direct-GRU seed. |
| Optimizer | Adam; learning rate 0.001; weight decay 0.0001; batch size 64; gradient norm clipped at 1. |
| Loss | Smooth L1, beta 0.01, equally averaged over both extrema and all CVEs in a batch. |
| Budget | At most 80 epochs; patience 15. |
| Selection | Lowest validation joint MAE among trained epochs for each seed; earliest ties. Epoch zero is diagnostic only. |
| References | Persistence, past observed range, fixed ridge, and every saved F04-GRU seed, on identical validation CVEs and dates. |
| Evaluation | Minimum/maximum MAE and RMSE, range-width errors, joint MAE, equal-related-group joint MAE; all seeds and seed mean/sample SD. |
| Exclusions | No test scoring, hyperparameter search, TPG, SMP, CVSS, KEV or new external features. |

In plain terms, if `p` is the latest observed score and the GRU produces
`a` and `b`, form `p + tanh(a)` and `p + tanh(b)`. Sort these two numbers and
clip them to the valid EPSS range. The smaller final value is the forecast
minimum; the larger is the forecast maximum. Both can lie above or below `p`.
The model does not force the future range to contain the latest score.

At initialization the proposals are equal. Stable sorting sends the
minimum-target loss to proposal A and the maximum-target loss to B at ties;
their gradients can therefore differ when the targets differ. Away from ties,
either proposal can become the minimum. Sorting is piecewise differentiable,
not a smooth operation at every point. Zero head weights initially block
encoder data-loss gradients until the head weights change, as in F04-GRU.

#### Comparison Rules and Limitations

This changes the head parameterization, initialization and ordering/bounding
mechanism together. It is not an isolated skip-connection-only ablation.
The encoder, cohorts, optimizer, loss and budget are held fixed. Identical
seeds pair initial encoders and sample order, not subsequent learning paths.

Persistence initialization is exact in float32, including base values 0 and 1.
Reference metrics retain original score precision, so tiny round-off differences
are possible. Epoch zero cannot be selected, so this model can still be worse
than persistence. No improvement is promised.

Clipping has zero gradient outside [0,1], and tanh may saturate. Every epoch
records the number of raw values outside [0,1], raw proposal crossings and
mean absolute correction for training and validation. Selected per-CVE outputs
keep both corrections and raw proposals, not just the ordered final forecasts.
Raw A/B identity can swap, so correction A is not always the minimum's change.
Crossing counts are diagnostic, not evidence of invalid final outputs.

Report all seeds, paired differences versus persistence and the matching
direct-GRU seed, and the existing references. Do not replace an unfavorable
seed, change the target population or use the test set to choose a variant.
Repeated validation inspection, complete-future selection, EPSS version changes
and uncertainty about exact daily release times remain limitations. These are
EPSS extrema, not confidence intervals or actual exploitation outcomes.

#### Commands and Artifacts

The creation commands below are retained for reproducibility. The default run
now exists, so they refuse to overwrite it. Use the review command in the
verified-findings subsection to inspect the completed run without refitting.

From the repository root, first run the read-only preflight:

```bash
python EPSS_TPG/02_prediction/01_package/epss/forecast_minmax_residual.py --check-only
```

Then start the fixed three-seed experiment:

```bash
python EPSS_TPG/02_prediction/01_package/epss/forecast_minmax_residual.py
```

Default source: `05_results/02_evaluation/epss_day30_to_day60/minmax_gru_validation/`.
The preflight verifies that run's hashes, original preparation/freeze/archive
evidence, saved predictions and metrics. It does not fit any model. Source
directory overrides are explicit with `--reference-dir`; output overrides use
`--output-dir`. There are no CLI tuning flags or test-evaluation option.

Default output: `05_results/02_evaluation/epss_day30_to_day60/minmax_residual_validation/`.
Existing destinations are refused, and outputs must not overlap source or
reference directories. The raw dataset and completed models remain unchanged.

| File | Purpose |
|---|---|
| `models.json` | All selected residual checkpoints, settings, fit CVE IDs and provenance; portable JSON tensors. |
| `initialization.json` | Per-seed epoch-zero joint MAE and verified float32 persistence initialization. |
| `training_history.json` | Every epoch's loss, training/validation errors, clipping, crossings and correction magnitude. |
| `predictions.csv/json` | Same validation rows as F04-GRU, preserving every comparator plus each residual seed, raw proposals and corrections. |
| `summary.json`, `summary.txt`, `README.md` | Per-target and joint metrics, paired comparisons, seed variation, output diagnostics and limitations. |
| `training_curves.png` | One panel per seed: epoch on x, joint MAE on y, training/validation lines and selected epoch marker. |
| `run_status.json`, `evaluation_lock.json` | Completion/failure status and successful artifact hashes; failed runs receive no success lock. |

Synthetic tests cover bounds, ordering, ties and head gradients, exact
persistence and its latest-score input derivative, missing Day 30, deterministic
training/reload, future-input separation, held-out isolation, unchanged cohorts,
independent metrics, reference integrity, output protection and failure status.
Tests use small synthetic datasets only, not the real experiment cohort.

#### Verified F04-Residual Findings

`review_forecast_minmax_residual.py` reproduced the selected predictions,
initialization, metrics, output diagnostics and saved checkpoint errors. Input,
reference and output hashes agree, as do CSV/JSON/text reports. The epoch logs
obey selection and stopping rules. Other epoch weights were not saved, so their
losses and optimizer updates are not replayed. The training figure was visually
inspected and is readable. Original artifacts remain unchanged; no test scoring.

| Model | Minimum MAE | Maximum MAE | Joint MAE | Maximum RMSE | Range-width MAE |
|---|---:|---:|---:|---:|---:|
| Persistence | 0.00008331 | 0.01095301 | 0.00551816 | 0.09085521 | 0.01095521 |
| Residual seed 17 | 0.00009485 | 0.01096425 | 0.00552955 | 0.09085080 | 0.01096442 |
| Residual seed 29 | 0.00009391 | 0.01105538 | 0.00557464 | 0.09083571 | 0.01105045 |
| Residual seed 43 | 0.00010628 | 0.01102772 | 0.00556700 | 0.09084401 | 0.01104052 |

All rows use the same 163 validation CVEs. Joint MAE averages minimum and maximum
absolute errors equally. Lower is better; 0.01 means one EPSS percentage point.
The slight maximum-RMSE decreases do not imply better maximum MAE or joint MAE.

| Seed | Selected epoch | Epochs run | Joint error increase versus persistence | Joint error reduction versus matching direct GRU |
|---|---:|---:|---:|---:|
| 17 | 18 | 33 | 0.2064% | 81.0992% |
| 29 | 13 | 28 | 1.0236% | 80.8797% |
| 43 | 8 | 23 | 0.8851% | 80.9934% |

All runs stopped after 15 epochs without improving their best validation joint
MAE. Mean seed joint MAE is 0.00555706 (sample SD 0.00002413), about 0.7050%
above persistence. These are descriptive relative error changes, not significance
tests or confidence intervals. No best-seed selection or ensemble is reported.

The prediction range across CVEs is no longer concentrated near 0.01. Predicted
maxima span 0.00043038-0.94526088, 0.00056483-0.94566637 and
0.00052741-0.94504017 for seeds 17/29/43. The direct base-score connection
preserves score-level variation, but the corrections do not improve the primary
metric. Each seed has lower joint error than persistence for 24 CVEs and higher
error for 139, with no exact ties. These counts are not significance tests.

| Future-score pattern | CVEs | Persistence joint MAE | Seed 17 | Seed 29 | Seed 43 |
|---|---:|---:|---:|---:|---:|
| All 30 future scores identical | 132 | 0.00004735 | 0.00006139 | 0.00012497 | 0.00011215 |
| At least two future scores differ | 31 | 0.02881323 | 0.02881334 | 0.02877972 | 0.02879409 |

These groups use future outcomes only for explanation. Do not use them as input
features or exclusion rules. Seeds 29/43 make small gains on changing futures,
but their added errors on constant futures outweigh those gains overall.

For CVE-2024-23692, the latest observed input is 0.00210 and the actual future
range is [0.00210, 0.95432]. Residual maximum forecasts are 0.00211075,
0.00224385 and 0.00220722. For CVE-2024-8963, the input is 0.30991 and actual
future maximum is 0.95343; forecasts remain near 0.310. Large increases remain
unpredicted, despite fixing the earlier narrow range across CVEs.

No selected validation output needs clipping. Raw proposals cross in 161, 156
and 0 CVEs, respectively; sorting resolves these and every final pair is valid.
Zero clipping at selection does not mean clipping never occurred: the largest
logged validation clipping count is 106 raw values for each seed. Peaks in
training are 39, 42 and 42 values, respectively. A CVE has two raw values, so
these counts are not numbers of CVEs. They describe logged epoch-end outputs,
not every intermediate minibatch state.

**Decision:** retain persistence as the joint-error reference, preserve all
seeds and record the residual result without claiming a forecasting gain over
persistence. This is not proof that all neural models or TPG features cannot
help. Do not automatically add another architecture or keep tuning against
this validation period. A model-independent review of the remaining future
changes and temporal distribution differences is a useful next step before
declaring further training. TPG/SMP extensions still require historical evidence.

The separate [review](../05_results/02_evaluation/epss_day30_to_day60/minmax_residual_validation_review/README.md)
contains TXT/JSON findings and all three real CVE examples. Reproduce it with:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_minmax_residual.py
```

### F04-GRU: Ordered Sequence Extrema

**Status: completed by the researcher and verified; validation only.**
Entry point: `01_package/epss/forecast_minmax_sequence.py`.

This is the planned neural extension of F04. It tests whether a mask-aware
sequence encoder can predict the future minimum and maximum more accurately
than the three completed F04 references. It is not a retry of endpoint
forecasting, and it does not add TPG/SMP or assume that ordering alone will
improve accuracy. The settings are declared after inspecting F04-Linear;
this remains development on an already inspected validation period.

| Design item | Fixed choice |
|---|---|
| Inputs | 31 daily steps, each with score, observed mask, observation age/30, age-known mask, day index/30. Reuses the existing sequence loader. |
| Missing inputs | Zero is storage padding only when the observed mask is 0. Observed zero stays distinct; no future interpolation or score filling. |
| Encoder | One-layer GRU, 16 hidden units. No new statistical scaler; raw score channels and fixed age/day normalization follow the earlier GRU. |
| Head | Two linear logits `a,b`; `low = sigmoid(a)` and `high = low + (1-low)*sigmoid(b)`. Guarantees `0 <= low <= high <= 1`; no sorting or output clipping. |
| Initialization | Head weights zero; biases derived only from the eligible training mean minimum/maximum. Fixed epsilon 0.00001 keeps logits finite. Epoch zero is diagnostic, never selectable. |
| Cohort/targets | Same 797 training and 163 validation CVEs as F04-Linear; all 30 future scores verified. No changed targets, groups or split boundaries. |
| Loss | Smooth L1, beta=0.01; equal weight to minimum and maximum, equal weight per training CVE. |
| Optimization | Adam, learning rate 0.001, weight decay 0.0001, batch size 64, gradient norm cap 1. CPU, one training thread, deterministic operations. |
| Budget | At most 80 epochs per seed; patience 15 epochs without a strict validation joint-MAE improvement. |
| Seeds | 17, 29, 43, all saved and reported. No best-seed selection. |
| Checkpoint | Lowest validation joint MAE within each seed, earliest checkpoint on a tie. Validation participates in selection, not gradient updates. |
| Comparators | Saved persistence, past observed range and F04-Linear ridge, reproduced without refitting on identical CVEs and targets. |
| Metrics | Per-target MAE/RMSE/rank/signed error; range-width errors; joint MAE; equal-related-group joint MAE. Every seed plus mean/sample SD of joint metrics. |

Joint MAE gives the minimum and maximum equal weight within each CVE, then
averages CVEs. The mean of three seed metrics is not the score of an ensemble,
and their sample standard deviation is not a confidence interval. All references
remain in the report, including past range's favorable maximum-target result.

Ordered outputs are a structural constraint, not a guarantee that future scores
will lie inside the predicted range. Sigmoids approach boundaries and can
saturate numerically; exact zero-width targets can be difficult to fit. The
initialization approximates mean extrema when they are on boundaries or equal.
Loss, initialization, sequence representation and output structure all differ
from ridge, so this is not an isolated architecture or ordered-head ablation.
No claim about TPG/SMP, actual exploitation or calibrated uncertainty follows.

The original training commands are retained below for reproducibility. They
refuse the existing result directory; do not rerun them over this completed run:

```bash
python EPSS_TPG/02_prediction/01_package/epss/forecast_minmax_sequence.py --check-only
```

```bash
python -u EPSS_TPG/02_prediction/01_package/epss/forecast_minmax_sequence.py
```

The first verifies frozen data, all future target scores, the F04-Linear run and
saved comparator predictions without training or writing output. The second
trains all three fixed seeds, selects validation checkpoints, and reports
validation only. No test-scoring, tuning, overwrite or resume flags are exposed.
Preserve failed runs; a deliberate rerun needs a separately documented output.

Default output: `05_results/02_evaluation/epss_day30_to_day60/minmax_gru_validation/`.

| Artifact | Contents |
|---|---|
| `models.json` | All three selected checkpoints, tensor weights as JSON, settings, fit IDs and provenance. |
| `training_history.json` | Every epoch's training loss and training/validation joint MAE per seed. |
| `initialization.json` | Training-only mean extrema, bounded initialization values and epoch-zero validation diagnostics. |
| `predictions.csv/json` | Existing three references and minimum/maximum/range/error for every GRU seed on each validation CVE. |
| `summary.txt/json`, `README.md` | Matched results, seed variability, output definitions and limitations. |
| `training_curves.png` | One panel per seed: epoch on x, joint MAE on y, training/validation curves and selected epoch. |
| `run_status.json`, `evaluation_lock.json` | Run state and completed-artifact hashes. Failures produce `failure.txt` without a success lock. |

Real-data preflight passed; no real-data GRU fit was performed during
implementation. Synthetic tests cover ordering at extreme logits, degenerate
initialization, gradients, missing/zero channels, incomplete-future exclusion,
train/test/future isolation, deterministic updates, checkpoint selection and
reloads, all-seed reporting, reference integrity and failed-run preservation.

#### Verified F04-GRU Findings

All selected checkpoints reproduce their training/validation errors and saved
predictions. The review checked initialization from training means, the cohort,
target/archive reference, ordering, per-epoch selection and stopping rules,
CSV/JSON/text, independent error metrics and source/code/output hashes. It did
not refit, replay optimizer updates or score the test set. Unselected epoch
weights were not saved, so the full logged trajectory was checked for consistency,
not independently regenerated. No checkpoint/alignment/report discrepancy was found.

All rows below use the same 163 validation CVEs and Days 31-60 extrema:

| Method | Selected epoch | Minimum MAE | Maximum MAE | Joint MAE |
|---|---:|---:|---:|---:|
| Persistence | - | 0.00008331 | 0.01095301 | 0.00551816 |
| Past observed range | - | 0.01500301 | 0.00939675 | 0.01219988 |
| Ridge | - | 0.00040545 | 0.01334129 | 0.00687337 |
| GRU seed 17 | 7 | 0.02335781 | 0.03515351 | 0.02925566 |
| GRU seed 29 | 47 | 0.02326456 | 0.03504662 | 0.02915559 |
| GRU seed 43 | 7 | 0.02338989 | 0.03518986 | 0.02928987 |

Joint MAE averages both extrema equally. The mean of the three GRU joint MAEs
is 0.02923371 (sample SD 0.00006978), about 5.30 times persistence. This is not
an ensemble score or confidence interval. Every seed is worse than all three
references on joint MAE, and each has lower joint error than persistence on
only 2 of the 163 CVEs. The ordered-head constraint worked, but forecasting
accuracy did not improve.

Seeds 17 and 43 ran 22 epochs and selected epoch 7. Seed 29 ran 62 epochs and
selected epoch 47, whose joint MAE was 0.02915559; its final epoch was worse at
0.02921360. All stopped after 15 epochs without beating their previous best.
An improvement over the immediately preceding epoch does not reset patience
unless it also beats the best value so far. The logs therefore agree with the
declared stopping rule.

The predicted extrema have a much narrower spread than the targets:

| Seed | Predicted minima across CVEs | Predicted maxima across CVEs |
|---|---|---|
| 17 | 0.00891828 to 0.00923060 | 0.01045145 to 0.01082039 |
| 29 | 0.00884198 to 0.01096054 | 0.01036746 to 0.01308483 |
| 43 | 0.00896242 to 0.00930927 | 0.01050254 to 0.01091221 |

Actual future minima span 0.00042 to 0.94504; maxima span 0.00042 to 0.95432.
This is weak differentiation between CVEs in the fitted models, not proof of
a specific optimization bug or sigmoid saturation. For example, seed 29 predicts
CVE-2024-8963 as minimum 0.00943260 and maximum 0.01111800, while its actual
future extrema are 0.30991 and 0.95343. Enforcing order does not preserve the
individual CVE's score level or ensure accurate extrema.

The 132 constant-future and 31 changing-future CVEs remain included. Each seed
has substantial errors in both groups; the review gives their separate values.
These outcome-based groups are explanatory only, not predictor features or
new model-selection cohorts.

Retain persistence as the joint-error reference and keep past range's distinct
maximum-target result visible. Before more training, investigate the narrow
prediction spread with a separately declared diagnostic; do not silently change
loss, initialization or head structure and reuse this result's identity.
This run does not measure TPG/SMP usefulness or actual exploitation.

Reproduce verification without fitting or test scoring:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_minmax_sequence.py
```

The [review folder](../05_results/02_evaluation/epss_day30_to_day60/minmax_gru_validation_review/README.md)
contains `analysis.txt/json` and `README.md`, including checkpoint checks,
prediction distributions, subgroup errors and real CVE examples. Original
model files, predictions, histories and hashes remain unchanged.

### F04-Linear: Future Minimum and Maximum EPSS

**Status: completed by the researcher and verified; validation only.**
Entry point: `01_package/epss/forecast_minmax.py`.

This begins the already planned F04 family with simple references and a linear
model before a neural encoder. The question is different from the endpoint task:
at Day 30, what will the lowest and highest EPSS scores be during Days 31-60?
For example, a future sequence with a low of 0.02 and a high of 0.40 has target
`[0.02, 0.40]`, regardless of its last score. This is an illustrative example,
not a measured CVE result. The two outputs are score extrema, not confidence
limits, exploitation probabilities or times of exploitation.

| Design item | Fixed choice |
|---|---|
| Inputs | Same seven statistics of observed Days 0-30 EPSS: latest score, mean, variability, slope, count, latest-score age and slope availability. |
| Target | Exact minimum and maximum of all 30 daily scores in Days 31-60. |
| Eligibility | At least one input score and all 30 future scores; no imputation or observed-only substitute for future extrema. |
| Splits | Existing chronological/group-safe manifest and Day-60 maturity rules. No new boundaries or group reassignment. |
| Persistence reference | Last observed input score repeated as both predicted extrema. |
| Past observed range reference | Lowest and highest available input scores, reused as future extrema. Missing input days are explicit; this is not necessarily the full past-window range. |
| Linear model | Two-output ridge regression, alpha=1, SVD solver, intercept, equal weight per training CVE. No parameter search. |
| Preprocessing | New scaler fitted on F04-eligible training CVEs only. The endpoint scaler is not modified or reused as the extrema scaler. |
| Output constraint | Clip each raw prediction to [0,1], then sort. Save raw outputs, crossing flags and clipping counts. This is fixed postprocessing, not an inherently ordered neural head. |
| Evaluation | Minimum/maximum MAE and RMSE; range-width error; joint MAE (equal weight to both extrema per CVE); equal-related-group joint MAE; rank and signed-error diagnostics. |
| Selection | No validation stopping, search, seed selection, test scoring or automatic neural training. |

The frozen source and prepared data remain unchanged. F04 creates a new,
explicit task-specific subset of the endpoint cohorts:

| Split | Endpoint CVEs | F04-eligible CVEs | Excluded from F04: incomplete future window |
|---|---:|---:|---:|
| Training | 1,023 | 797 | 226 |
| Validation | 197 | 163 | 34 |

All 28,800 future target scores for these 960 CVEs match the DQ04-pinned archive
verification. The CLI checks those matches before permitting training. It does
not download, alter or fill scores. A per-CVE cohort manifest records each
inclusion or exclusion and missing future dates. Missing *input* scores remain
permitted under the existing policy. Complete-future selection may favor a
different population; the reports make this limitation explicit.

The main comparison is between the three methods on the same 163 CVEs for the
same two targets. Do not directly compare these errors with previous endpoint
MAE on 197 CVEs, or call a lower number an improvement to endpoint prediction.
The period has already been inspected for earlier tasks, so this is development
evidence rather than untouched validation. Neither this model nor its target
implements TPG, SMP or exploitation prediction.

The original commands below are retained for reproducibility. They refuse the
existing result directory; do not rerun them over the completed run:

```bash
python EPSS_TPG/02_prediction/01_package/epss/forecast_minmax.py --check-only
```

```bash
python -u EPSS_TPG/02_prediction/01_package/epss/forecast_minmax.py
```

The first checks frozen inputs, task cohorts and score verification without
fitting a new model or writing results. The second fits the linear model on
training CVEs and evaluates all three methods on validation only. Existing
output folders are refused; preserve failed runs as well. Deliberate reruns
require a separately documented `--output-dir`.

Default output: `05_results/02_evaluation/epss_day30_to_day60/minmax_linear_validation/`.

| Artifact | Purpose |
|---|---|
| `model.json` | Extrema-specific scaler, training IDs, two coefficient vectors, intercepts, correction rule and provenance. |
| `cohort_manifest.csv/json` | Every endpoint training/validation CVE, F04 eligibility, group and missing future dates. No test CVEs. |
| `predictions.csv/json` | Future window dates, exact target extrema, all three methods, raw linear outputs, corrections, range widths and paired errors. |
| `summary.txt/json`, `README.md` | Counts, metrics, definitions, model settings and limitations. |
| `validation_comparison.png` | Three method rows, minimum/maximum columns; actual extrema on x, predicted extrema on y. |
| `run_status.json`, `evaluation_lock.json` | Execution state and successful artifact hashes. Failures create `failure.txt` without a success lock. |

The real-data preflight passed. Synthetic tests cover complete-future eligibility,
missing input/observed-zero handling, task-specific scaling, future/nontraining
isolation, portable predictions, ordering/clipping, target verification, test
guards, output protection and failed-run preservation. No real-data model was
fitted during its implementation. The separate F04-GRU extension has since
been completed by the researcher and verified, as recorded above.

#### Verified F04 Findings

All saved predictions and correction flags reproduce from the coefficients.
The cohort manifest, future extrema, 28,800 archive score matches, task-specific
training-only scaler, metrics, CSV/JSON/text and file hashes were checked.
Independent decimal-based per-target and joint errors agree. No refitting or
test scoring was performed; source and completed artifacts remain unchanged.

These results use the same 163 complete-future validation CVEs, not the 197-CVE
endpoint cohort. Lower errors are better; joint MAE averages the minimum and
maximum absolute errors equally for each CVE, then averages CVEs.

| Method | Minimum MAE | Maximum MAE | Joint MAE | Maximum RMSE | Range-width MAE |
|---|---:|---:|---:|---:|---:|
| Persistence | 0.00008331 | 0.01095301 | 0.00551816 | 0.09085521 | 0.01095521 |
| Past observed range | 0.01500301 | 0.00939675 | 0.01219988 | 0.07977386 | 0.02370761 |
| Two-output ridge | 0.00040545 | 0.01334129 | 0.00687337 | 0.08961527 | 0.01343277 |

Persistence has the lowest joint MAE. The linear model has 24.56% higher joint
MAE, despite slightly lower maximum RMSE. The past observed range has the lowest
maximum MAE/RMSE among these methods, but much higher minimum and joint error.
Do not choose a favorable target or metric after seeing these results and call
it an overall improvement. No statistical significance has been established.

Of the 163 validation CVEs, 132 have an identical score on all 30 future days;
130 also equal the latest input score. Constant future scores do not necessarily
equal the last input. On the 31 changing-future CVEs, persistence joint MAE is
0.02881323 and linear joint MAE is 0.02938859. These outcome-based groups are
diagnostics, never model inputs or reasons to exclude records.

| CVE | Actual future minimum | Actual future maximum | Linear predicted minimum | Linear predicted maximum |
|---|---:|---:|---:|---:|
| CVE-2024-23692 | 0.00210 | 0.95432 | 0.00199038 | 0.00478219 |
| CVE-2024-8963 | 0.30991 | 0.95343 | 0.30725862 | 0.33651817 |
| CVE-2023-7028 | 0.04478 | 0.20153 | 0.04410621 | 0.04552294 |

These are the three largest persistence maximum errors. Large future increases
remain poorly predicted. Reusing an old minimum also has a clear limitation:
CVE-2024-24919 has past observed minimum 0.00043 but future minimum 0.94504;
CVE-2024-4358 has past observed minimum 0.00043 but future minimum 0.93837.
An earlier low score need not recur during the forecast window.

The linear outputs crossed (raw minimum greater than raw maximum) for 12 CVEs.
Those same 12 CVEs each had one out-of-range output; all 12 pairs still needed
sorting after clipping. The fixed correction worked and raw values remain
saved. This is a limitation of independent outputs, not a silent data repair
or evidence of confidence-interval calibration.

Persistence also has lower joint MAE than both alternatives in each of the
three fixed input-completeness groups. Different groups contain different CVEs,
so this does not establish that missingness caused model errors.

Keep persistence as the joint-error reference, retain the useful separate
maximum-target comparison, and do not reinterpret F04 errors as improvements
to endpoint results. Neural extensions remain separately planned; none is
started automatically. Historical TPG/SMP availability is still a separate gate.

To reproduce verification without fitting or test scoring:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_minmax.py
```

The [review folder](../05_results/02_evaluation/epss_day30_to_day60/minmax_linear_validation_review/README.md)
contains `analysis.txt/json` and `README.md`, including all correction cases,
real CVE examples, constant/changing-future groups and input-coverage groups.

### F01-Huber: Fixed Robust Linear Baseline

**Status: completed by the researcher and verified; validation only.**
Entry point: `01_package/epss/forecast_huber.py`. This implements the Huber member
of the existing F01 family after DQ04. The family was planned earlier; the
specific configuration is declared now after inspecting previous validation
results. This is another development experiment, not independent confirmation.

**Question:** can a fixed robust linear model improve the Day-60 forecast relative
to persistence and ridge using the same input information? It does not test TPG
or SMP, predict actual exploitation, or promise improvement. Temporal text
availability still blocks a defensible historical TPG fusion experiment.

| Component | Fixed specification |
|---|---|
| Inputs | Seven observed-score statistics over Days 0-30: latest, mean, population standard deviation, slope, observed-day count, latest-observation age and slope-defined flag. |
| Data policy | DQ04 freeze and source/preparation hashes required; retain nulls and primary cohorts; no imputation or new exclusions. |
| Scaling | Existing training-only standardized features; constant-feature handling unchanged. |
| Estimator | scikit-learn `HuberRegressor`; coefficients, intercept and residual scale fitted on training CVEs only. |
| Settings | `epsilon=1.35`, `alpha=0.0001`, `max_iter=1000`, `tol=1e-8`, `fit_intercept=True`, `warm_start=False`. No search or runtime tuning flags. |
| Target and output | Observed Day-60 EPSS; one score clipped to [0,1], with raw predictions retained. |
| Cohorts | Same 1,023 training and 197 validation CVEs; same chronological/group-safe split. |
| Comparators | Saved ridge model predictions reproduced without refitting; matched persistence predictions. |
| Evaluation | MAE, RMSE, signed error, Spearman, equal-related-group MAE, paired MAE difference and persistence-relative skill. Validation only. |
| Selection | One fixed deterministic optimizer run; no validation stopping, parameter search, seed selection or test evaluation. |
| Failure policy | Convergence warning or invalid parameters fails the run and preserves its status/error; no automatic budget increase or alternate optimizer. |

Huber fitting limits the influence of large scale-normalized training residuals
without deleting those CVEs. Its `epsilon` is not a CVSS/EPSS cutoff or the
direction-label tolerance. The training-only residual scale is saved. The
estimator's linear-loss-region flags are diagnostics, not proof that data are
wrong, and every training CVE is retained. See the [official estimator documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html).

The epsilon and alpha settings use the documented defaults. The 1,000-iteration
ceiling and 1e-8 tolerance are fixed numerical settings, not a validation search.
The ridge comparison changes the objective, regularization strength and solver;
do not present it as an isolated loss-only ablation. Robustness may suppress
genuine important changes, so retain all errors and do not assume success.
Installed NumPy, SciPy, scikit-learn and Python versions are recorded.

The following are the original training commands, retained for reproducibility.
They refuse the existing result directory. Do not rerun them over this completed run:

```bash
python EPSS_TPG/02_prediction/01_package/epss/forecast_huber.py --check-only
python -u EPSS_TPG/02_prediction/01_package/epss/forecast_huber.py
```

The first validates the frozen artifacts and reproduces the existing ridge
predictions without fitting Huber or writing files. The second fits Huber on
training data and evaluates validation only. Test integrity checks remain part
of loading the prepared bundle, but no test predictions or metrics are computed.

Default output: `05_results/02_evaluation/epss_day30_to_day60/huber_validation/`.
Existing destinations are refused. A deliberate rerun requires a different
`--output-dir`; there is no overwrite or automatic failure-retry option.

| Artifact | Purpose |
|---|---|
| `model.json` | Portable coefficients, scaler, residual scale, training IDs, fixed settings, iterations and provenance. |
| `predictions.csv/json` | Every validation CVE, actual target, persistence/ridge/Huber predictions, raw Huber score, clipping indicator and paired errors. |
| `training_diagnostics.csv/json` | Training residuals and linear-loss-region flags; all rows retained. Never used as input features. |
| `summary.txt/json`, `README.md` | Matched results, definitions, configuration and limitations. |
| `validation_comparison.png` | Actual versus predicted score and paired absolute-error differences; axes explained in the report. |
| `run_status.json` | Running, complete or failed state with provenance. |
| `evaluation_lock.json` | Written only for a completed run; hashes successful artifacts. |
| `failure.txt` | Created if execution fails; no successful result is certified. |

The real-data preflight passed. Ten new tests cover synthetic training isolation,
future-input isolation, portable predictions against scikit-learn, observed zero,
clipping, convergence failure, comparator alignment, test guards, immutable inputs,
output hashes and failed-run preservation. Only synthetic model fits were performed
during implementation. The researcher subsequently completed the real run.

#### Verified Findings

The review reproduced all saved validation predictions from the coefficients,
checked training residual diagnostics, CSV/JSON agreement, reports, source/code
hashes and cohort alignment, and independently calculated MAE/RMSE/group MAE.
It did not refit the model, modify the completed run, or score the test set.

| Comparison on the same 197 CVEs | Persistence | Huber |
|---|---:|---:|
| Average absolute error (MAE) | 0.00909675 | 0.00909609 |
| Error emphasizing large misses (RMSE) | 0.08264449 | 0.08264314 |
| Rank correlation (Spearman) | 0.89710785 | 0.88349212 |
| MAE giving each related-CVE group equal weight | 0.00772734 | 0.00772632 |

The mean absolute difference between the two forecasts is only 0.00001446;
the median is approximately 0.000000004. Huber is effectively very close to
repeating the last input score. Exact comparisons show 18 lower errors and
179 higher errors, many differing only at very small numerical scales. Those
counts are not a statistical significance test.

The solver completed in 123 iterations without warnings. Its fitted residual
scale is about 1.14e-9; this is a loss parameter, not forecast uncertainty or
proof of extraordinary accuracy. The Day-60 target equals the latest input
score exactly for 924/1,023 training and 162/197 validation CVEs. Of the training
records, 480 fall in the linear-loss region and all remain included. The tiny
scale warrants caution about numerical sensitivity; no optimizer retuning or
separate numerical-stability experiment was performed.

| CVE | Latest input score | Actual Day-60 score | Huber prediction |
|---|---:|---:|---:|
| CVE-2024-23692 | 0.00210 | 0.95432 | 0.00210049 |
| CVE-2024-8963 | 0.30991 | 0.95343 | 0.30994643 |
| CVE-2023-7028 | 0.04478 | 0.20153 | 0.04476196 |

The large changes remain unpredicted. In the fixed DQ04 input-coverage groups,
Huber MAE is slightly higher for the 38 complete histories and 130 histories
missing one score, and slightly lower for the 29 missing two or more. These
are different CVEs, so this does not show that missing data helped or caused errors.
Keep persistence as the reference and retain all original cohorts. No claim
about exploitation prediction or TPG usefulness follows from this EPSS-only run.

To reproduce the review without training or test scoring:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_huber.py
```

The [review folder](../05_results/02_evaluation/epss_day30_to_day60/huber_validation_review/README.md)
contains the explanation and machine-readable findings (`analysis.txt/json`).

### F01-Tree: Fixed Nonlinear Comparator

**Status: completed by the researcher and verified; validation only.**
Entry point: `01_package/epss/forecast_tree.py`.

This implements the remaining small-tree alternative in F01. The question is
whether nonlinear combinations of the same historical EPSS statistics help
predict Day 60. It does not change the target, add unavailable historical text,
or assume that a more complex model will predict the large jumps. The tree
family was planned earlier; the detailed settings below are declared now,
after inspecting earlier validation results. This remains development evidence.

| Item | Fixed design |
|---|---|
| Inputs | Same seven Days 0-30 statistics as ridge and Huber. No TPG, SMP, CVSS, summary or KEV input. |
| Target | Observed Day-60 EPSS; prediction anchor is Day 30. |
| Cohort | Unchanged 1,023 training and 197 validation CVEs; chronological/group-safe split and label maturity unchanged. |
| Model | One scikit-learn `DecisionTreeRegressor`, squared-error criterion, best split. |
| Size | Maximum depth 3 (at most eight leaves); minimum 20 training CVEs per leaf. A root-only tree is allowed. |
| Other settings | `min_samples_split=2`, all seven features available at each split, `random_state=17`, `ccp_alpha=0`. No search or seed selection. |
| Preprocessing | Reuse the exact training-only scaler and DQ04 missing-data policy. No new filling or exclusions. Scaling is retained for a matched input representation, not because trees require it. |
| Output | One score per CVE, produced by the training-target mean in the reached leaf; raw score and leaf ID retained. Fixed [0,1] output clipping. |
| Comparators | Saved persistence, ridge and Huber, reproduced without refitting on exactly the same validation IDs. |
| Metrics | MAE, RMSE, signed error, Spearman, equal-related-group MAE, paired difference from persistence. No significance claim. |
| Selection | No validation-based tree growth, pruning, early stopping, parameter search or test scoring. |

Depth 3 and minimum leaf size 20 are conservative complexity limits, not values
chosen by a validation search. The fixed seed makes tie-breaking reproducible;
it is not a three-seed neural experiment. This is a nonlinear comparator, not
an isolated architecture/loss ablation: its capacity and regularization also
differ from the linear models. Leaf averages cannot extrapolate beyond observed
training-target values. A small tree can underfit, and large changes may remain
unpredicted. See the [estimator documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html).

The original training commands below are retained for reproducibility. They
refuse the existing result directory; do not rerun them over the completed run:

```bash
python EPSS_TPG/02_prediction/01_package/epss/forecast_tree.py --check-only
```

```bash
python -u EPSS_TPG/02_prediction/01_package/epss/forecast_tree.py
```

Preflight checks frozen data/code and comparator hashes and reproduces saved
predictions without fitting or writing output. The second command fits only
training CVEs and evaluates validation. Both refuse an existing output folder;
preserve completed and failed runs. A deliberate separately documented rerun
requires a new `--output-dir`.

Default output: `05_results/02_evaluation/epss_day30_to_day60/tree_validation/`.

| Artifact | Contents |
|---|---|
| `model.json` | Full tree nodes, training-only scaler, fit IDs, settings and source/code provenance. Portable JSON; no pickle loading. |
| `tree_nodes.csv` | Child indices, feature index, standardized threshold, predicted value and training count per node. Leaf feature -2 and child -1 are explained in the report. |
| `predictions.csv/json` | Aligned validation targets, predictions of all four comparators, tree leaf IDs and errors. |
| `summary.txt/json`, `README.md` | Metrics, configuration, definitions and limitations. |
| `validation_comparison.png` | Four matched panels: actual Day-60 EPSS on x, predicted Day-60 EPSS on y. |
| `run_status.json`, `evaluation_lock.json` | Execution state and hashes for completed artifacts. Failures create `failure.txt` without a success lock. |

Synthetic tests cover training isolation, future-score isolation, missing inputs
and observed zero, portable predictions against scikit-learn including float32
threshold boundaries, malformed trees, test guards, comparator tampering,
protected outputs, artifact hashes and failure preservation. No real tree fit
was performed during implementation.

#### Verified Tree Findings

The saved predictions, metrics, JSON/CSV tables, report text, provenance and
file hashes reproduce. Training CVEs were routed through the saved tree to
verify every node count and mean target, without fitting the tree again.
Independent decimal-based MAE/RMSE/group-MAE checks also passed. The test set
was not scored. This verifies these artifacts, not every possible data issue.

| Model | Validation CVEs | MAE | RMSE | Rank correlation |
|---|---:|---:|---:|---:|
| Persistence | 197 | 0.00909675 | 0.08264449 | 0.89710785 |
| Huber | 197 | 0.00909609 | 0.08264314 | 0.88349212 |
| Tree | 197 | 0.02477230 | 0.09641363 | 0.45751343 |

The tree has about 2.72 times persistence MAE (172.32% higher). It has smaller
absolute error on 5 CVEs and larger error on 192. These comparisons are
descriptive, not significance tests. All original CVEs remain included.

The fitted tree contains seven nodes and four leaves:

| Leaf ID | Training CVEs | Forecast assigned to every CVE in the leaf | Validation CVEs |
|---|---:|---:|---:|
| 3 | 919 | 0.00931071 | 182 |
| 4 | 48 | 0.03279125 | 5 |
| 5 | 36 | 0.06702083 | 4 |
| 6 | 20 | 0.47327000 | 6 |

For 162 validation CVEs, Day-60 EPSS exactly equals the latest input score.
Persistence has zero error on these CVEs, but the tree has MAE 0.01610502.
The small number of leaf averages loses distinctions between individual CVEs;
this helps explain the errors but does not prove that every tree model fails.

The tree reduces errors on the three previously identified large changes, but
still misses their actual scores: CVE-2024-23692 is predicted as 0.00931071
instead of 0.95432, CVE-2024-8963 as 0.47327 instead of 0.95343, and
CVE-2023-7028 as 0.06702083 instead of 0.20153. These improvements are outweighed
by larger errors elsewhere. Tree MAE is higher than persistence in all three
frozen input-coverage groups, including complete histories. This does not
establish missingness as a cause, nor justify dropping any group.

Keep persistence as the reference and the tree as a completed negative
comparison. Do not automatically deepen or retune it on the same validation
set. A future target or model needs a separately declared question and protocol;
this result says nothing about TPG/SMP value because neither was an input.

Reproduce verification without fitting or test scoring:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_tree.py
```

Detailed explanations and machine-readable findings are in
[`tree_validation_review`](../05_results/02_evaluation/epss_day30_to_day60/tree_validation_review/README.md)
as `README.md`, `analysis.txt` and `analysis.json`. Original artifacts are unchanged.

### F02-Residual: Completed and Verified

The [implementation](01_package/epss/forecast_residual.py) is a separate entry
point, so the earlier direct-head code and its pinned results remain unchanged.
This follow-up was motivated by validation findings; it is not a preplanned
independent confirmation. The user completed the three-seed run; its source,
predictions, metrics, checkpoint selection and initialization have been verified.
No retraining or test scoring was performed during the post-run review.

The output is `clip(base + tanh(linear(GRU(history))), 0, 1)`, where `base` is
the most recent observed EPSS in Days 0-30. If Day 30 is missing, use the last
observed earlier day, not padded zero. Both correction-head weights and bias
start at zero, so initialization equals persistence in float32, including
observed scores of 0 and 1. Float32 rounding can differ slightly from the
original JSON score precision used for reporting metrics.

The five input channels, 16-unit encoder, seeds 17/29/43, Smooth L1 beta=0.01,
Adam learning rate 0.001, weight decay 0.0001, batch size 64, gradient clipping
at 1, 80-epoch maximum and patience of 15 are unchanged. Encoder initialization
and batch-order pairing use the same seeds; optimization paths will differ.
Head formula and initialization change together, so this is not an isolated
skip-connection-only ablation. No epoch-zero persistence fallback is silently
selected: epoch 0 is diagnostic, and eligible checkpoints are trained epochs
1-80. All three seeds must be retained even if they perform worse.

Loss is computed on the bounded output. Clipping can produce zero gradients
when raw scores go outside [0,1]; the histories save training and validation
clipping counts, and predictions retain raw scores and corrections. Zero head
weights initially give zero encoder gradients from the loss; that is expected
until the head learns nonzero weights. This design is not guaranteed to beat
persistence or solve the direct GRU's optimization limitations.

Preflight has verified 1,023 training and 197 validation CVEs, the preparation
lock, source and reference hashes, and saved direct-head predictions. Checks
and short synthetic unit tests are not experimental results on the dataset.
The original run command is retained below for reproducibility, not as a
request to train again. The output already exists; a deliberate repeat should
use a separate destination to preserve it.

```bash
cd /home/ayounas/Text_property_Graph
conda activate CodeBERTFusion
python -u EPSS_TPG/02_prediction/01_package/epss/forecast_residual.py \
  --prepared-dir EPSS_TPG/04_data/01_records_and_graphs/epss_day30_to_day60 \
  --reference-dir EPSS_TPG/05_results/02_evaluation/epss_day30_to_day60/gru_validation_user_run \
  --output-dir EPSS_TPG/05_results/02_evaluation/epss_day30_to_day60/gru_residual_user_run
```

Append `--check-only` to verify without fitting or writing results. The actual
run prints epoch progress, saves all three models, epoch-zero diagnostics,
learning curves, per-CVE paired predictions and TXT/JSON reports. It compares
with persistence, ridge and the direct GRU on the same validation CVEs. There
is no test-scoring mode. The destination is separate; do not overwrite the
direct-head results.

#### Findings From the Completed Run

- Selected epochs are 2, 5 and 15; training stopped after 17, 20 and 30 epochs.
  Every stop follows the fixed 15-epoch patience rule, and epoch zero was not
  selected. Initial predictions reproduce persistence to float32 precision.
- Residual MAE is above persistence by 0.5632%, 0.2289% and 0.0260% for seeds
  17, 29 and 43. These are relative percentages of error, not EPSS percentage
  points. Tiny RMSE reductions do not establish a useful or significant gain.
- The mean MAE is 0.00912156 with seed sample SD 0.00002468. This is much less
  variation than the direct head, not a confidence interval or ensemble score.
- Mean absolute corrections are only 0.00007286, 0.00002570 and 0.00000356.
  The selected models largely preserve persistence rather than demonstrating
  a useful new predictive signal.
- No selected-checkpoint training or validation scores were clipped. Earlier
  epoch-end logs reach 170 and 148 clipped validation CVEs for seeds 17 and
  29 respectively (training maxima 90 and 60); seed 43 records zero. These
  are per-epoch maximum counts, not distinct-CVE totals. Earlier states were
  not all saved, so those earlier counts are log evidence, not independently
  regenerated inference. Clipping can suppress gradients, but causation has
  not been established here.
- Persistence matches 162 of 197 endpoints exactly. The residual improves
  absolute error for 28, 9 and 28 CVEs respectively, but worsens the others
  at stored precision. Many changes are very small.
- The large jumps remain missed. For CVE-2024-23692 the recorded Day-60 score
  is 0.95432 and persistence is 0.00210; residual predictions are 0.00217148,
  0.00207885 and 0.00210236. These cases remain in all metrics.

No prediction, metric or selection inconsistency was found in the checks;
this does not certify the entire dataset or pipeline as bug-free. Keep
persistence as the primary MAE reference and keep the test set sealed.
Review large-change CVEs and historical evidence availability before another
training configuration. A TPG/SMP benefit has not been tested in these runs.

The [post-run review](../05_results/02_evaluation/epss_day30_to_day60/gru_residual_user_run_review/README.md)
also has TXT/JSON versions. Regenerate analysis only, without training, using:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_residual.py
```

## 5. SMP-Centered Studies

These use a post event as the anchor, not automatically CVE publication plus
30 days. They require a separate event dataset and split audit. The dated
file has 9,699 CVE/post associations and 9,574 after exact repeat removal.
All timestamps omit timezone; the date enrichment reused source-row dates.
Treat their publication meaning as an explicitly stated assumption, not as
independently authenticated platform timestamps. Text can repeat across CVEs
and dates. Historical text availability and post identity remain separate checks.

| ID | Inputs -> method | Target / output | Window | Metrics or analysis | Status |
|---|---|---|---|---|---|
| S00 | Supplied post dates + EPSS -> descriptive alignment | Before/after scores, slopes, changes, counts | +/-7 first; +/-14 and +/-30 sensitivity | Coverage, distributions, pre/post slope groups, real examples | Completed descriptive study under date assumptions; not a trained model or causal effect. [Date analysis](../05_results/03_dataset_analysis/01_revision_dataset/14_smp_date_verification/SMP_DATED_EPSS_ANALYSIS.md). |
| S01 / A | EPSS strictly available before the event -> temporal model | Future EPSS -> one score per declared lead | +1/+7/+14/+30 days | R | Conditional prediction baseline. Define event-time anchor and daily-release lag; same events as S02/S03. |
| S02 / B | Same past EPSS + occurrence/counts known at anchor -> temporal/structured fusion | Same future score as S01 | Same leads | R, paired difference against S01 | Conditional. An occurrence flag is constant if every example is a post event; include matched non-event anchors to test occurrence or use varying past counts to test intensity. |
| S03 / C | Same past EPSS + time-valid post text -> TPG-GNN + temporal fusion | Same future score as S01 | Same leads | R, paired difference against S01/S02 | Conditional. Exclude posts after anchor; prove text availability; group repeated posts/CVEs across folds. |
| S04 | First observed post versus repeated discussion; counts in past 7/14/30 days | Same endpoint, or descriptive post-intensity association | Same future leads | Matched R; descriptive counts and slopes | Planned. First observed is not first ever. Counts must end at the anchor; future post counts leak. |

Within publication through Day 60, the +/-7 analysis has 1,141 complete
event records covering 477 CVEs; +/-14 has 668/255 and +/-30 only 7/6.
These are descriptive before/after cohorts, not ready-made training counts.
Do not require observed future scores to define a predictor; target availability
may filter evaluation, but future post activity must not filter the input cohort.
Use +/-7 first and explicitly report censoring, time uncertainty, EPSS-version
changes, repeated events, and whether the score was already rising before the
post. Compare with/without SMP on identical CVEs and windows. An association
after posting does not show that posting caused the score to change.

## 6. VulnCheck Experiment Families

**All K01-K13 future-listing experiments remain blocked** pending verified
first-entry meaning, status at the anchor, successful negative checks through
the horizon and a defensible observation cutoff. A stored `false` flag is
not proof of never exploited. Negative publication-to-addition deltas require
review, not conversion to zero. Right-censored cases are not negative labels.
CISA KEV and VulnCheck KEV are different catalogues.

For K01-K13, the target is **first future inclusion in VulnCheck within a fixed
30-day horizon**, with separate 60/90-day versions later. Output is one
probability of that catalogue event, not universal exploitation probability.
Use B metrics and masked binary loss. A zero requires adequate verified
follow-up. Choose either publication or Day 30 as the anchor before constructing
labels; do not mix them in one task. This shared target/metric definition
applies to every row below.

| ID | Input features -> model | Additional input requirement / purpose |
|---|---|---|
| K00 | Current description, optionally current CVSS -> TPG-GNN + optional tabular head | Separate snapshot diagnostic: target is current `in_vckev`, no future horizon. Technically possible after an adapter, not trained. False-flag quality and current-text disclosures limit interpretation. B metrics. |
| K01 | Past EPSS -> logistic/history baseline | Time-valid scores; reference before graph experiments. |
| K02 | Description -> text encoder without graph -> binary head | Historical text; matched non-graph comparator for K03. |
| K03 | Description -> shared TPG -> relation-aware GNN -> binary head | Historical text; isolate graph structure. |
| K04 | Description TPG-GNN + CVSS structured branch | CVSS components/version as available at anchor, not later updates. |
| K05 | Description TPG-GNN + past EPSS encoder -> fusion | Compare with K01 and K03 on the same cohort. |
| K06 | SMP text -> source-labelled TPG-GNN | Verified post availability, grouping and source masks. |
| K07 | Description and SMP -> separate TPG views -> attention fusion | Both evidence sources available before anchor; missing-view masks. |
| K08 | GitHub summary -> TPG-GNN | Grounded summary from source content available by anchor; not a later generated disclosure. |
| K09 | Non-GitHub summary -> TPG-GNN | Same source-time and grounding checks; match K08 cohort for comparisons. |
| K10 | Description + GitHub summary -> two TPG views | Compare against description alone on exactly the same eligible CVEs. |
| K11 | Description + non-GitHub summary -> two TPG views | Same matched-cohort condition; do not insert future advisories. |
| K12 | Description + SMP + both summary sources -> multi-source TPG fusion, no EPSS | Explicit source labels/masks; new four-source adapter required. |
| K13 | All time-valid text + EPSS + legitimate structured inputs -> complete fusion model | Compare with K05/K12 and one-source-removed variants; not implemented. |

For future listing, exclude `in_vckev`, VulnCheck addition/update dates,
vendor/project, VulnCheck CWEs, exploitation/ransomware/canary/XDB fields,
CISA addition dates and due dates from predictors. Also exclude their
missingness, derived counts, graph nodes and edges. These are label/audit
fields, not safe inputs merely because they have been turned into a graph.
Actual exploitation prediction needs independently defined exploitation labels;
neither an EPSS threshold nor a catalogue flag establishes that target.

## 7. Multi-Task Families

| ID | Inputs -> model | Targets -> outputs | Horizon | Metrics / status |
|---|---|---|---|---|
| M01 | Time-valid K03/K05/K13 inputs -> shared encoders and cumulative event heads | Future VulnCheck inclusion -> ordered p30 <= p60 <= p90 | 30/60/90 days from one common anchor | B per horizon, ordering checks. Blocked on catalogue labels. Separate valid-label mask per horizon. |
| M02 | Time-valid text + past EPSS -> fusion with regression and binary heads | Future EPSS and future VulnCheck inclusion -> one score plus one distinct event probability | Day 30 -> Day 60 initially | R and B separately; masked losses normalized per task. Blocked on event labels/input provenance. Compare with both single-task models. |
| M03 | Past EPSS -> shared history encoder and consistent endpoint/min/max heads | Day-60 score and Days 31-60 extrema -> min <= endpoint <= max | 30 future days | R per output, range error, bound checks. Planned, labels available; endpoint and complete-window masks differ. |

Task weights must be declared/selected using validation, not test performance.
Missing labels remain masked; one task's targets never substitute for another's.
Equal training budgets are needed when comparing single-task and multi-task runs.

## 8. Controlled Ablations Still to Run

| ID | Comparison | What it tests / prerequisite |
|---|---|---|
| A01 | Text embedding without graph vs TPG-GNN | Added value of structure, with identical text/encoder/cutoff/cohort. Historical text gate applies. |
| A02 | Core linguistic edges vs security edges vs learned/candidate security relations | Added value of relation types; preserve uncertainty and evidence. Extractors trained only on training evidence. |
| A03 | Typed-edge GNN vs type-collapsed edges; single relation-aware GNN vs multiview | Whether typed message passing helps; match graph inputs and model budget. |
| A04 | EPSS only vs TPG only vs their fusion | Contribution of each branch on matched CVEs; do not conflate cold-start and Day-30 tasks. |
| A05 | Description only, summary only, concatenated text, separate source views, pooled summary node | Summary contribution and graph construction; inspect graph size and validation curves for excessive smoothing. |
| A06 | Complete model minus SMP, GitHub summary, other summary, EPSS, or CVSS one at a time | Source contribution with explicit missing-source masks; same examples for each paired comparison. |
| A07 | Statistics vs GRU vs temporal CNN; Huber vs MSE; input channels removed | Temporal architecture/loss/missingness contribution. Do not change multiple components at once. Current ridge/GRU loss differences are not an isolated architecture ablation. |
| A08 | Single-task vs multi-task, equal budget | Benefit or negative transfer from sharing. M01/M02 remain blocked. |
| A09 | Post-date +/-1-day sensitivity, first-post-only, repeated-post controls | Sensitivity to SMP timing and repeated evidence. Not evidence of causal effects. |
| A10 | Fixed EPSS-version strata / change-window sensitivity | Whether model-version transitions affect forecast errors. Full-window version-change flags are audit-only, not predictors. |

TPG uses the shared core builder, with source labels, document/linguistic
structure, embeddings and the chosen security relations. The GNN uses graph
edge types during message passing; pooling yields a graph vector. That vector
can later join the GRU history vector and legitimate structured features.
No cross-CVE training message passing may reach validation or test graphs.
A current-text retrospective diagnostic must be labelled as such, not as
historically leakage-safe simply because the graph builder is shared.

## 9. Archived Experiment Families

These exist in older local documentation. Their targets, data and selection
rules differ from current EPSS forecasting. **No archived numerical result is
revalidated or adopted as a current result here.** Old labels described as KEV
need catalogue-specific interpretation; historical NVD/KEV work used CISA,
not the new VulnCheck flag.

| Family | Documented experiment types | Evidence |
|---|---|---|
| Legacy 16-run clean matrix | Four LLM datasets x B, B+summary, B+security edges, B+both; hybrid features; EPSS target removed from inputs | [Archived clean matrix](../99_archive/03_previous_dataset_work/01_documentation/04_experiments/05_summary_ablation/02_RESULTS.md) |
| Legacy leaky/non-leaky hybrid batches | Summary graph variants with and without EPSS as a feature; historical A-H feature settings | Same archived report; target-feature leakage variants are not valid future-EPSS benchmarks. |
| Legacy graph and CVSS ablations | TPG-only branch, graph variants, tabular/CVSS removal | [Graph ablation](../99_archive/03_previous_dataset_work/01_documentation/04_experiments/06_graph_ablation/01_README.md), [CVSS ablation](../99_archive/03_previous_dataset_work/01_documentation/04_experiments/07_cvss_ablation/01_README.md) |
| Legacy text-source and summary experiments | Description-only, summary-only by source, description+summary; social-media and Megavul corpora | [Archived results source](../99_archive/03_previous_dataset_work/01_documentation/06_papers/01_sources/experiment_results_35_runs.tex) |
| Legacy NVD/CISA reruns | NVD/KEV binary task; 2020-2022 -> 2023-2024 temporal runs with/without EPSS; 2020 -> 2021-2026 no-EPSS extrapolation | Same archived results source; these are not current VulnCheck-horizon labels. |
| Legacy cross-distribution evaluation | Curated-corpus model evaluated on a larger NVD held-out corpus | Archived clean-matrix report; different checkpoint/cohort, not a matched current forecast comparison. |
| Legacy SMP/EPSS analyses | Before/after association, platform exclusions, forward matched studies | [Archive index](../99_archive/03_previous_dataset_work/README.md). These descriptive/modeling records do not establish causality. |

## 10. Order and Maintenance

### Execution Workflow

For future steps, implement the code and run offline tests first, then provide
the CLI commands. The researcher runs training and reports completion; analyze
the saved results afterward. Do not automatically launch another training run.
The GRU results above were produced before this workflow was requested.

To repeat the fixed GRU experiment yourself without replacing those results,
run these commands from a terminal:

```bash
cd /home/ayounas/Text_property_Graph
conda activate CodeBERTFusion
python -u EPSS_TPG/02_prediction/01_package/epss/forecast_sequence.py \
  --prepared-dir EPSS_TPG/04_data/01_records_and_graphs/epss_day30_to_day60 \
  --ridge-dir EPSS_TPG/05_results/02_evaluation/epss_day30_to_day60/ridge_validation \
  --output-dir EPSS_TPG/05_results/02_evaluation/epss_day30_to_day60/gru_validation_user_run
```

This uses CPU, runs all three fixed seeds, and evaluates validation only.
Progress is printed after each seed finishes. The output includes TXT/JSON
reports, paired predictions, model states and training curves. The command
refuses an existing nonempty destination; use a new run directory to preserve
previous results, or `--overwrite` only for a deliberate replacement.

### Research Order

1. Keep F00 and F01-Ridge as fixed references. F02-GRU is now completed with all three seeds retained.
2. F02-Residual is now complete and verified. Review large-change CVEs and
   historical feature availability using the completed **DQ01** audit below.
   DQ02/DQ03 verified present development scores; the researcher has accepted
   retaining documented missing values. **DQ04** is now completed and reviewed:
   the frozen policy and matched input-coverage diagnostics retain the main
   cohorts, without fitting or selecting another model. Further archive recovery is optional.
   **F01-Huber** is now completed and verified on the same frozen cohort.
   Its tiny validation MAE gain does not solve the large-change forecasting
   problem; keep persistence as the reference. **F01-Tree** is also completed
   and verified, with substantially higher validation error. Preserve these
   comparisons; do not automatically tune more endpoint models. **F04-Linear**
   is now complete and verified on its separate complete-future cohort.
   Persistence retains the lowest joint error; past observed range is better
   only on the maximum-target measures. Preserve this distinction. **F04-GRU**
   is also complete and verified; every seed has substantially higher joint error
   and a narrow prediction spread. **DQ05** is now completed and reproduced:
   sensitivity to observed scores is weak despite changed weights and nonzero
   gradients. **F04-Residual** is now complete and verified: score-level variation
   is restored, but all seeds still have slightly higher joint error than
   persistence. Retain this result. **DQ06** is now completed and reproduced:
   five CVEs account for 98.27% of persistence's validation error, and typical
   training/validation score levels differ. Keep the difficult CVEs and verify
   historical feature availability before declaring further training. Do not silently change the model
   or use the test set to decide a fix. A different model is not an automatic win or
   a correction to earlier endpoint results. Do not reinterpret the current
   validation period as untouched.
3. **DQ07** is completed and reproduced. The largest group accounts for 1,329
   of 1,548 group exclusions; shared homepages and repeated page/generic text
   warrant context review. **DQ08** implements a source-bound human review packet
   and decision validator. Prepare written exact-key decisions before considering
   a separate split-policy revision. Keep test values/performance sealed and
   review links without reference to model errors.
4. **R01 smoke and full development runs are completed and verified** as the
   separate retrospective current-description study below. The full run uses
   1,023 training/197 validation CVEs and all three fixed seeds. No neural fit
   beats persistence on MAE; DQ09 below implements the next branch-use and
   extraction-coverage diagnostic. Its small subset is completed and reproduced;
   the unchanged full diagnostic is next. DQ08
   decisions are not required to run R01 because it keeps all current grouping
   protections and split boundaries. F07/F08 remain conditional on historical
   description evidence; R01 must not be relabelled as those experiments.
5. Complete SMP event provenance and event-specific splits before S01-S04
   prediction; S00 can remain a conditional descriptive study.
6. Repair catalogue-label evidence before K01-K13. Run single-task studies
   before M01/M02; M03 can proceed earlier because it uses EPSS labels only.
7. Freeze the final design before one deliberate final test evaluation.

### R01: Current-Description TPG Regression

**Status: smoke and full training completed by the researcher; all selected
checkpoints and saved metrics independently reproduced. Test scoring remains disabled.**

#### Verified Full Results

The full run uses 1,023 training CVEs, 197 validation CVEs in 169 related
groups, and seeds 17, 29 and 43. `review_forecast_tpg_full.py` verifies source,
code, cache and output hashes; all 15 checkpoint identities; exact validation
predictions and metrics; selected-epoch training errors; and the stopping rules.
Every graph's tensor counts and mean-token baseline embedding agree with the
manifest. Earlier training trajectories are not replayed and contextual
embeddings are not regenerated. Original artifacts remain unchanged.

| Method | Seed 17 MAE | Seed 29 MAE | Seed 43 MAE | Mean MAE | Seed sample SD |
|---|---:|---:|---:|---:|---:|
| Persistence (not trained) | 0.00909675 | 0.00909675 | 0.00909675 | 0.00909675 | Not applicable |
| EPSS-only GRU | 0.01280439 | 0.01245079 | 0.03184784 | 0.01903434 | 0.01109822 |
| Mean token embedding + EPSS | 0.03244024 | 0.03236324 | 0.03107370 | 0.03195906 | 0.00076771 |
| TPG only | 0.03172284 | 0.03159807 | 0.03177944 | 0.03170012 | 0.00009280 |
| TPG + EPSS | 0.02610255 | 0.02551741 | 0.01892104 | 0.02351366 | 0.00398808 |
| TPG without graph messages + EPSS | 0.02085888 | 0.03087261 | 0.02063067 | 0.02412072 | 0.00584842 |

MAE is average absolute error in the Day-60 EPSS score; lower is better.
Every model uses the same validation CVEs. Seed means are not ensemble
predictions. Persistence is one deterministic reference, repeated here only
for ease of comparison. No best seed is selected.

**What the comparison establishes:**

- All 15 neural fits lose to persistence on MAE and equal-group MAE. EPSS-only
  seed 29 slightly improves RMSE (0.08164098 versus 0.08264449), so do not say
  every metric is worse. Full per-seed metrics are in the review.
- Graph fusion beats its no-edge counterpart for seeds 29 and 43, but loses
  for seed 17. Mean error is only 2.52% lower with edges. Three seeds do not
  establish a reliable graph advantage, and this removes all edges, not just
  security edges.
- Fusion beats mean-token-text + EPSS for all three seeds, but adding TPG to
  EPSS-only worsens seeds 17 and 29 and improves seed 43. The overall mean is
  worse than EPSS-only. This is not a universal verdict on TPGs.
- 162/197 validation endpoints (82.23%) equal their latest observed input;
  35 differ. Persistence MAE on those 35 is 0.05120171, still lower than
  every neural fit on that same subset. Equality at the endpoints does not
  prove the intermediate EPSS curve is constant.
- Five CVEs contribute 98.36% of persistence's absolute error. The largest
  case is `CVE-2024-23692`: 0.00210 at the prediction anchor and 0.95432 at
  Day 60. Fusion predicts only 0.00867, 0.00570 and 0.00579. Keep the case;
  its difficulty does not justify exclusion.
- TPG-only predictions never exceed 0.03286 across the three seeds, although
  the largest observed score is 0.95432. The graph-only model has not learned
  to represent the high-score cases in this validation period.

All fits stopped by patience after 21-67 epochs, rather than reaching the
80-epoch ceiling. EPSS-only seeds 17 and 29 improve sharply around epoch 26,
whereas seed 43 stops at epoch 22. Text and graph models often reduce training
error without corresponding validation gains. This warrants a diagnosis of
optimization and generalization, not an unsupported claim of over-smoothing
or a silent change to patience after seeing the results.

| Split | CVEs | Nodes | Edges | Embedded nodes | Unembedded nodes | CVEs with SEC_ edges |
|---|---:|---:|---:|---:|---:|---:|
| Training | 1,023 | 100,603 | 307,474 | 93,401 | 7,202 | 105 (10.26%) |
| Validation | 197 | 24,778 | 76,689 | 23,039 | 1,739 | 11 (5.58%) |

Together these contain 125,381 nodes, 384,163 edges and 116,440 embedded
nodes. The unembedded nodes are 1,220 documents, 1,296 paragraphs, 6,098 topics
and 327 tokens. Source-span checks on 44 CVEs identify 323 whitespace tokens
and four U+FFFD replacement characters already in the raw descriptions of
`CVE-2021-22822`, `CVE-2021-22826` and `CVE-2021-22827`. The hash-matched local
tokenizer produces no subwords for those texts. This is not evidence of
embedding truncation. Preserve the source and explicit embedding masks;
do not invent the characters that were lost upstream.

There are just 122 typed security edges: 97 `SEC_HAS_VERSION`, 24 `SEC_CAUSES`
and one `SEC_HAS_SEVERITY`. Other security relation types are absent from
this run. The adapter uses shared rule extraction, not a trained relation
extractor. Exact exporter/cache agreement does not establish semantic
extraction accuracy. Sparse coverage limits what this experiment can say
about the security-aware part of the GNN.

**Next decision:** retain this completed result and diagnose the saved models
before another training experiment. Check whether predictions respond to
past EPSS versus text and inspect representative security-edge extractions.
Such perturbations are sensitivity checks, not fair retrained ablations or
causal evidence. A later persistence-residual fusion is a candidate, not an
assumed fix; specify its settings before running a small engineering subset.
Do not add summaries/SMP or enlarge the model simply because this run loses.
Keep the test set reserved and retain the current chronological/group splits.
Current descriptions and frozen encoder knowledge remain historically
unverified; this is retrospective analysis, not leakage-free deployment evidence.

Saved review: `05_results/02_evaluation/epss_day30_to_day60/tpg_description_full_v1_review/`
contains `analysis.txt`, `analysis.json` and `README.md`. The original
`tpg_description_full_v1/` and `description_full_v1/` cache are unchanged.
The implementation plus smoke/full-review tests pass (19 synthetic/local tests).

Read-only verification from the repository root:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_tpg_full.py --check-only
```

#### Verified Smoke Results

The separate `review_forecast_tpg.py` script checked source/code/output hashes,
all 48 cached graphs, checkpoint contracts, exact validation predictions and
metrics, CSV/JSON/TXT agreement, and selected-epoch training/validation MAE.
The earlier epoch histories were checked for finite values and valid selection,
not reproduced by retraining. Token embeddings were checked for consistency
and coverage, not recomputed with the language model. Original outputs remain
unchanged. The reviewer plus R01 implementation tests pass (15 tests).

| Method | Selected epoch | Validation MAE | Validation RMSE |
|---|---:|---:|---:|
| Last observed EPSS (persistence) | Not trained | 0.00001813 | 0.00007250 |
| EPSS-only GRU | 3 | 0.04906421 | 0.05124552 |
| Text embedding + EPSS | 3 | 0.02948061 | 0.03923228 |
| TPG only | 3 | 0.01794760 | 0.03642987 |
| TPG + EPSS | 3 | 0.02030968 | 0.03652087 |
| TPG without graph messages + EPSS | 3 | 0.02181669 | 0.03674895 |

Every row uses the same 16 validation CVEs. MAE is the average absolute score
error; RMSE weights large errors more strongly. Lower is better. All neural
results are seed 17 after at most three epochs, not final architecture rankings.

Persistence exactly matches the Day-60 score for 15/16 CVEs. The only different
endpoint is `CVE-2024-23749`: latest input (Day 30) 0.00046, Day 60 0.00075.
Matching endpoints do not prove that every intermediate score was unchanged.
The training target mean is 0.0532796875 versus 0.011085 in validation; their
medians are 0.00885 and 0.00057. The direct neural heads are not initialized
to per-CVE persistence. For example, TPG-only predictions span just
0.01033-0.01103, while actual validation scores span 0.00042-0.15116.
`CVE-2024-8190` has actual and persistence score 0.15116, but TPG-only predicts
0.01047911 and fusion predicts 0.01371306. Thus lowering near-constant outputs
can improve average error without learning the important between-CVE variation.

Every neural model's training and validation MAE decreases over the three
recorded epochs, and every selected checkpoint is epoch 3. This does not show
convergence, overfitting, or that longer training will beat persistence. TPG-only
has the lowest neural MAE here, but it still loses to persistence. The graph
fusion/no-edge difference is descriptive only, not proof of a graph benefit.

| Split | CVEs | Nodes | Edges | Embedded nodes | Unembedded nodes | CVEs with typed security edges |
|---|---:|---:|---:|---:|---:|---:|
| Training | 32 | 3,751 | 11,843 | 3,523 | 228 | 2 |
| Validation | 16 | 1,142 | 3,441 | 1,025 | 117 | 0 |

The totals are 4,893 nodes, 15,284 edges and 4,548 embedded nodes. The 345
unembedded nodes consist of 48 document nodes, 49 paragraph nodes, 240 topic
nodes and 8 newline-only TOKEN nodes. Re-parsing the relevant source descriptions
verified four newline tokens each in `CVE-2023-1966` and `CVE-2023-34332`.
Their zero embedding and zero mask are expected; no unembedded non-whitespace
token was found in this subset. The baseline text vector reproduces exactly
from the cached embedded TOKEN vectors.

Only two typed security relations occur: one `SEC_HAS_VERSION` edge each for
`CVE-2022-1418` and `CVE-2022-27167`, both training CVEs. No validation graph
contains a `SEC_*` edge. Consequently this run cannot demonstrate the value
of typed security relations. Security entity text and linguistic structure
remain present. Sparse extraction is a coverage limitation, not evidence of
dropped export edges; do not force unsupported relations into the graphs.

**Decision at the smoke stage (full run now completed above):** the engineering
smoke test passed. Continue with the already
specified full development cohort, all fixed seeds and unchanged comparisons.
Do not tune to these 16 validation CVEs, revise splits, or open the test set.
More training is not a guaranteed improvement. Historical text availability
remains unresolved, so the retrospective limitation still applies.

Saved review: `05_results/02_evaluation/epss_day30_to_day60/tpg_description_smoke_v1_review/`
contains `analysis.json`, `analysis.txt` and `README.md`. To reproduce without writes:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_tpg.py --check-only
```

The initial two-real-CVE check verified graph validation, GraphSON export,
tensor reload and all five untrained forward passes. `CVE-2020-22983` had
81 nodes, 284 edges and 74 embedded nodes; `CVE-2023-29445` had 61 nodes,
184 edges and 54 embedded nodes. Neither description produced a typed
security relation under the current rules. Security-edge enablement therefore
must not be described as evidence that every graph contains such a relation.
The synthetic relation test separately checks a qualified `SEC_AFFECTS` edge.

This moves to the first TPG forecasting implementation without waiting for a
perfect dataset. It uses the current description as a retrospective input.
Its wording and the frozen language model's pretraining history are not verified
as available at the Day-30 anchor. Therefore results cannot establish deployable
historical forecasting performance, even with chronological CVE splits.

#### Target and Cohorts

The label is the observed EPSS score on Day 60. Past EPSS covers Days 0-30.
There is one bounded regression output, not a KEV probability, exploitation
label, predicted score range or causal SMP effect.

| Scale | Training CVEs | Validation CVEs | Seeds | Maximum epochs | Purpose |
|---|---:|---:|---|---:|---|
| `smoke`, default | 32 | 16 | 17 | 3 | Engineering check only |
| `full`, later | 1,023 | 197 | 17, 29, 43 | 80 | Matched retrospective validation study |

Smoke records are selected by a fixed hash ordering of eligible CVE IDs, not
scores, errors, severity or graph size. Selection is deterministic. Existing
group exclusions and chronological splits remain unchanged. Both scales require
at least one observed past score and an observed Day-60 target. Missing
descriptions or graph failures stop preparation; they do not silently drop CVEs.
All model variants use exactly the same selected records. Test graphs, predictions
and performance are disabled; the existing loader still checks source/prepared
values for integrity, including reserved records.

#### Shared TPG Construction

Read these components in order:

1. `01_tpg/01_core/tpg/pipeline.py`: shared `SecurityPipeline` and enrichment passes.
2. `frontends/spacy_frontend.py` and `frontends/security_frontend.py`, under that
   same `tpg/` directory: linguistic structure and security entity candidates.
3. `passes/security_relations.py` and `contracts.py`: source-supported candidate
   relations, qualifiers, source spans and relation-property features.
4. `embeddings.py`: frozen SecBERT contextual vectors aligned to sentence/token
   spans, with windowing for long sentences.
5. `exporters/exporters.py`: the shared numerical graph layout and type vocabulary.
6. `02_prediction/01_package/epss/forecast_tpg_data.py`: new description-only
   forecasting adapter and versioned graph cache.
7. `02_prediction/01_package/epss/forecast_tpg.py`: new regression and fusion runner,
   reusing `edge_aware_layers.py` and the existing past-EPSS sequence adapter.

The core is not copied or modified. spaCy must provide the parser, tagger and
NER; its blank-English fallback is rejected. Rule-based security relations are
enabled, including their polarity/modality properties; they remain candidates,
not verified facts. Optional learned extraction/annotation work is not silently
enabled or trained by this runner.

Only the raw `description` string enters graph construction. No ID is prefixed
to it, and no structured CVSS, KEV, EPSS, summary or SMP field is passed through.
An ID or severity already written in the description remains part of that text.
Metadata uses CVE IDs for alignment, not as learned input features. Nodes and
edges retain `source_text_type=description` in the inspected GraphSON examples.

Node input is node-type one-hot + frozen embedding + embedding-present mask.
Structural nodes without a stored embedding retain zero vectors and mask zero.
Edge type IDs use the saved vocabulary, including `SEC_*`; relation IDs and
64-dimensional property features enter the GNN message function, not a tabular
feature vector. Original directed edges are retained; no undocumented reverse
edges are added. Source spans, graph integrity, dimensions, finite features,
embedding coverage and undropped security edges are checked at preparation.

SecBERT is loaded locally by default, frozen and in evaluation mode on CPU.
The cache records weights/tokenizer/spaCy fingerprints, package versions,
construction-code hashes, per-CVE graph sizes and input description hashes.
`--allow-model-download` is an explicit opt-in only if local resources are missing.
No external model download was needed for the initial preflight.

#### Matched Comparisons

| Mode | Text/graph component | EPSS component | Question |
|---|---|---|---|
| Persistence | None | Last observed Days 0-30 score | Fixed reference |
| `epss_only` | None | Mask-aware GRU | New matched neural EPSS reference |
| `text_epss` | Mean embedded TOKEN vector, no graph messages | Same GRU | Does the text representation help? |
| `tpg_only` | Two relation-aware GNN layers and mean node pooling | None | What does the graph provide alone? |
| `tpg_epss` | Same graph encoder | Same GRU, concatenated with graph vector | Does fusion help? |
| `tpg_no_edges_epss` | Same nodes/layers/pooling, empty edge list | Same GRU | Do inter-node messages help beyond node representations? |

The no-edge control keeps the same initial parameter values at a given seed;
training diverges as expected. The mean-token baseline uses the same frozen
token vectors as the graph, not a different text model. It does not include
all graph node types, so it is not alone an isolated edge-structure ablation.

Graph hidden size is 32 with two existing `EdgeTypeGNNLayer` blocks, dropout 0.1,
and residual/LayerNorm updates. The temporal GRU has 16 hidden units and the
existing five channels: score, observation mask, normalized observation age,
age-known mask and normalized day index. A small fusion MLP produces a sigmoid
Day-60 score. The output is direct regression, not the earlier residual head.

All neural modes use batch size 8, Adam at 0.001, weight decay 0.0001,
Smooth L1 beta 0.01 and gradient-norm clipping at 1. The last-layer bias is
initialized from the training target mean only. Validation MAE chooses the
earliest best trained epoch; patience is 3 for smoke and 15 for full. No
hyperparameter search or best-seed selection is performed. Every seed is reported.
These settings are not a promise to beat persistence.

Report MAE, RMSE, signed error, Spearman correlation where defined, and
equal-related-group MAE. The old complete-cohort Huber/ridge metrics must not
be compared numerically with a 16-CVE smoke result as though cohorts matched.
Validation is already repeatedly inspected; leave the test set unopened.

#### Run the Small Subset First

From the repository root with `CodeBERTFusion` active, run one at a time:

```bash
python EPSS_TPG/02_prediction/01_package/epss/forecast_tpg_data.py --scale smoke --check-only
python EPSS_TPG/02_prediction/01_package/epss/forecast_tpg_data.py --scale smoke
python EPSS_TPG/02_prediction/01_package/epss/forecast_tpg.py --check-only
python -u EPSS_TPG/02_prediction/01_package/epss/forecast_tpg.py
```

The first command constructs two real graphs without writes or training.
The second prepares all 48 smoke graphs. The third checks all five forward
passes without parameter updates. The fourth trains five small models for up
to three epochs each, plus computes persistence on the same 16 validation CVEs.
Progress bars cover source verification, resource loading, graph construction,
cache validation/loading, model checks, models/seeds, epochs, training/evaluation
batches and report saving. Finite-loss/gradient checks stop numerical failures.

| Location under `EPSS_TPG/` | Contents |
|---|---|
| `01_tpg/05_workspace/forecast_graphs/description_smoke_v1/` | Label-free tensor graph files, manifest, two GraphSON inspection examples, TXT/Markdown guide and cache lock |
| `05_results/02_evaluation/epss_day30_to_day60/tpg_description_smoke_v1/` | Selected checkpoints, predictions CSV/JSON, history JSON, summaries TXT/JSON, README, training curves, completion status and output hashes |
| `02_prediction/03_analysis/01_revision_dataset/test_forecast_tpg.py` | Synthetic data/model/cache/serialization regression tests |

Existing output directories are refused, never overwritten. Use a deliberate new
`--output-dir` for a repeat; a new cache also needs the runner's `--cache-dir`.
Failed runs retain their failure status; a successful output lock is written last.
Checkpoints are reloaded and must reproduce their saved validation predictions.

After the smoke run, inspect graph sizes, embedding coverage, actual security
edge counts, finite losses and prediction/checkpoint alignment. Few or zero
security edges on some descriptions are possible; do not fabricate relations
or append the CVE ID to force an edge. Do not judge model quality from smoke MAE.

The commands below document the completed scaling step after smoke review.
Do not rerun them into the existing full-run folders:

```bash
python EPSS_TPG/02_prediction/01_package/epss/forecast_tpg_data.py --scale full
python -u EPSS_TPG/02_prediction/01_package/epss/forecast_tpg.py \
  --cache-dir EPSS_TPG/01_tpg/05_workspace/forecast_graphs/description_full_v1
```

The full run uses a separate `tpg_description_full_v1` result folder. SMP,
summary fusion, learned-relation evaluation, graph-depth/security-edge ablations
and any revised grouping policy remain later, separately specified experiments.

### DQ14: Bounded Semantic Review With v4 Fixed

**Status: completed source-only automated review. V4 and all original artifacts
remain unchanged. No expansion, embeddings, inference, training or test inspection.**

Scope: 44 links across 22 development CVEs (35 links from training CVEs and
nine from validation CVEs). Review all 22 different-type repeated-text links
and all six pronoun links, plus six same-type controls and ten product/version
links in fixed SHA-256 item-ID order. The risk flags were already known, and
this development sample is not preregistered or independent evaluation.

| Review group | Links | Supported in stated scope | Unsupported as interpreted | Limited or ambiguous |
|---|---:|---:|---:|---:|
| Different-type repeated-text candidates | 22 | 4 | 13 | 5 |
| Pronoun candidates | 6 | 0 | 5 | 1 |
| Same-type controls | 6 | 3 | 2 | 1 |
| Product/version links | 10 | 10 | 0 | 0 |
| Total | 44 | 17 | 20 | 7 |

Interpretation and concrete source examples:

- CVE-2021-25094: `which` refers to the rogue zip file that is uncompressed,
  not `Tatsu WordPress`. CVE-2024-35539: `This vulnerability` refers to the
  race condition, not `Typecho v1.3.0`. Five pronoun links are unsupported;
  `its` in CVE-2022-3604 remains ambiguous due to the saved wording.
- CVE-2022-27167: `11.2` appears under different ESET products. CVE-2022-30525:
  equal firmware bounds appear under different Zyxel models. These support
  value equality, not product-release identity. Same-type VERSION controls
  show that matching types do not fix this interpretation problem.
- `IpMatcher`, `SMU` and `Expedition` have supported repeated-name links despite
  differing parser types. A blanket type-mismatch deletion rule would lose
  supported associations. Other links support only family or concept scope.
- The ten sampled product/version links have source-supported owner/release
  associations and appropriate local bounds. The selected ESET, Acrobat and
  Ruby edges do not encode every range/branch. This is not a claim that all
  43 links are correct, or that all 111 versions have complete relations.

No recall or population precision is computed. Unselected links, missed
relations and all unresolved versions are not exhaustively reviewed. Every
human status remains pending, each training_label is null, and all judgments
are explicitly attributed to the automated review. Programmatic tests verify source
binding and scope; they do not prove the judgments semantically correct.

Script: `01_tpg/02_examples/01_scripts/review_security_product_v4_semantics.py`.
Observation ledger: `01_tpg/09_annotation/08_v4_semantic_review/automated_observations_v1.json`.
Reports: `01_tpg/09_annotation/08_v4_semantic_review/bounded_review_v1/`
(`analysis.txt`, `analysis.json`, `README.md`, `review_lock.json`). Source text,
exact spans, excerpts, rationale and interpretation scope are saved for each link.
Four focused tests cover fixed selection, forbidden splits, source quotes,
qualifier evidence and separation from human/model labels.
All 78 combined focused tests pass; the original v4 artifacts and completed
bounded-review checksums also verify.

```bash
python EPSS_TPG/01_tpg/02_examples/01_scripts/review_security_product_v4_semantics.py --check-only
```

Decision: retain v4 as a frozen candidate graph, not a verified identity graph.
Review these findings and document accepted limitations before a separately
approved development expansion or matched COREF-on/off ablation. Do not merge
entities using COREF as trusted identity, silently delete the flagged links,
or launch another training run on the strength of this bounded review alone.

### DQ13: Same-Occurrence Coreference Correction

**Status: user's 56-CVE preview verified and reproduced exactly; separate
TXT/JSON review saved. No expansion, embeddings, training or test inspection.**

Extension: `01_tpg/10_extensions/tpg_security_product_v4.py`.
CLI: `01_tpg/02_examples/01_scripts/audit_security_product_versions_v4.py`.
Tests: `test_security_product_v4.py` and `test_security_product_v4_audit.py`
in `01_tpg/08_tests/` (nine new tests).
The implementation passed 72 combined tests. Two additional saved-review tests
check reproduction, removal evidence and scope. Frozen v3 preview/review checks
still pass; no original artifacts were modified.
All 74 combined focused tests pass after adding the saved-review checks.

The rule removes a COREF edge only when its method is `repeated_surface`,
status is `candidate`, both endpoints are ENTITY nodes, their spans are
identical and nonempty, and both texts exactly match the source substring.
This separates multiple annotations of one occurrence from separate mentions.
Both annotation nodes remain. No links are added or rewired. Existing
distinct-span edges, pronoun links, asserted/manual links, unknown-span cases,
version ownership, bounds and abstentions are preserved.

The pipeline wrapper calls the unchanged v3 pipeline, then the new pure graph
filter. The CLI uses that same filter directly on the locked v3 GraphSON,
so no parser or encoder is instantiated. It checks exact round trips, node
content and IDs, retained edge ordering/properties, security evidence and
label-free exports. The expected removals are computed from each edge's
source evidence rather than a fixed total. Any off-scope change fails the run.

`security_extraction_version` remains v3 because the extractor is unchanged.
`coreference_filter` records the v4 policy and individual removed edges, and
the graph pass history includes v4. Node IDs are preserved; edge IDs can shift.
This is not a new coreference model and does not prove that retained matches
across separate mentions identify the same real-world entity.

Original creation commands (output now exists; use `--verify-only` to verify it):

```bash
python EPSS_TPG/01_tpg/02_examples/01_scripts/audit_security_product_versions_v4.py --check-only
python EPSS_TPG/01_tpg/02_examples/01_scripts/audit_security_product_versions_v4.py
python EPSS_TPG/01_tpg/02_examples/01_scripts/audit_security_product_versions_v4.py --verify-only
```

Output: `01_tpg/05_workspace/security_product_version_v4/smoke/`.
TXT/JSON reports explain the counts; `comparisons.json` includes descriptions,
removed-edge evidence and before/after extraction; `comparison_counts.csv`
contains per-CVE counts. `graphs/` contains separate unembedded GraphSON,
protected by `preview_lock.json`. Existing outputs cannot be overwritten;
`--output-dir` supports a new destination. No full-mode option exists.

The original core, extractors, graphs, caches, models, source records and human
review decisions remain frozen. Completion of this preview will not itself
authorize expansion or establish an accuracy/forecast improvement.

#### Verified v4 Results

| Measure | Saved v3 | v4 |
|---|---:|---:|
| Development CVEs | 56: 32 train, 24 validation | Same 56 |
| Release mentions | 111 | 111 |
| Product/release links | 43 | 43 |
| Coreference edges | 190 | 98 |
| Same-span coreference edges | 92 | 0 |
| Nodes | 5,483 | 5,483 |
| Total edges | 17,258 | 17,166 |

Every saved v4 GraphSON object reproduces exactly by applying the frozen v4
filter to its saved v3 input, including metadata and removal provenance.
Checks also reproduce the saved extraction rows and individual removed-edge
lists. All 92 removals satisfy the source-verified rule. All nodes and retained
edge properties remain unchanged; edge IDs can shift as documented.

The changed set is 34 CVEs: 24 training CVEs (79 removals) and 10 validation
CVEs (13 removals). The other 22 retain identical node/edge content.
CVE-2022-26130 retains its recovered `16.1.x` node and loses six same-span
coreference edges overall; its total coreference count changes from seven to one.

Of the 98 retained coreference edges, 92 use `repeated_surface` between
different positions and six use `unambiguous_type_heuristic` for pronouns.
Among the 92 repeated-text candidates, 22 have different entity types across
their endpoints. This is a review flag, not proof of a wrong relationship;
parser type differences may describe one real object. No candidate has been
automatically accepted as semantic ground truth. No forecast benefit is measured.

Review: `01_tpg/02_examples/01_scripts/review_security_product_v4.py`.
Reports: `01_tpg/05_workspace/security_product_version_v4/smoke_review/`
(`analysis.txt`, `analysis.json`, `README.md`, `review_lock.json`). The JSON
includes per-CVE retained edges, exact spans, methods and type-difference flags.

```bash
python EPSS_TPG/01_tpg/02_examples/01_scripts/review_security_product_v4.py --check-only
```

Next: freeze v4 for bounded source review of product/release evidence and the
retained candidates. Document accepted limitations before a separately scoped
larger development preview or matched training ablation. Do not chase perfect
coverage with additional unvalidated rules or open the test set.

### DQ12: Narrow IP Context Correction

**Status: user's 56-CVE v3 preview verified; separate saved-graph review complete.
The recovery works, but the edge-count expectation was too strict and same-span
coreference remains a semantic concern. No model cache, expansion or training.**

Implementation: `01_tpg/10_extensions/tpg_security_product_v3.py`.
Tests: `01_tpg/08_tests/test_security_product_v3.py`.

The v1/v2 address-context regex uses a word boundary, which also matches after
the hyphen in `BIG-IP`. V3 rejects compound-name suffixes as address cues while
retaining explicit `IP`, `IP address` and `address` contexts. It checks against
the original text so a truncated prefix cannot create a false boundary.
The release regex, CVSS filtering, v2 ownership decisions, bounds and explicit
abstentions are otherwise retained. V3 reuses existing graph construction and
validation; no monkey-patching or modification of frozen modules is used.

On the actual CVE-2022-26130 source description, `16.1.x` at `[13, 19)` is
recovered. Recognized version mentions change from seven to eight, with zero
new owner links. The recovered mention remains `unsupported_wildcard_release`.
This checks one development description only; the DQ10 v2 counts and locked
DQ11 packet are not rewritten as v3 results.

Eight focused v3 tests include genuine IPv4 negative cases (standalone cues,
colon/equal notation and addresses following product mentions), compound-name
positive cases, four-component software releases, long whitespace, CVSS
exclusion, source spans, policy metadata, generic-only/typed export behavior,
three hash seeds and unchanged core/v1/v2 hashes. A number is not classified
as an address merely because its four components form a valid IPv4 value.
The filter is contextual; comprehensive detection of unlabelled IP addresses
is not claimed.

Run from the project root with `CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/01_tpg/08_tests -p 'test_security_product_v3.py'
```

No embeddings, prediction weights, labels, split criteria, original results
or review judgments change. V3 is not enabled in a default pipeline or saved
model. Existing v2 audit commands remain v2.

#### Same-Sample v3 Preview

CLI: `01_tpg/02_examples/01_scripts/audit_security_product_versions_v3.py`.
Tests: `01_tpg/08_tests/test_security_product_v3_audit.py` (six new tests).
All 60 combined focused tests pass. The saved DQ11 analysis also passes its
read-only integrity check; human decisions remain unchanged.
The preview has no full mode. It reuses the exact frozen v2 IDs, split roles
and descriptions, reconstructs the complete v2 GraphSON exactly, then checks
v3 source spans, owner abstentions, generic/typed evidence and label-free export.
The frozen v2 preview, DQ11 packet and code/source pins are verified before and
after the run. Outputs cannot overwrite or overlap protected inputs.

The original expectation was the additional unresolved `16.1.x` mention at
`[13, 19)` in CVE-2022-26130, with one extra node and two edges. The run instead
adds three edges, explained below. Changed release
text, owners, bounds, software mentions, security evidence and graph counts
are reported; deviations are flagged for investigation. Only extractor-policy
tags are ignored when comparing security evidence. A completed run is not a
semantic accuracy certification or authorization for full expansion.

Run one command at a time:

```bash
python EPSS_TPG/01_tpg/02_examples/01_scripts/audit_security_product_versions_v3.py --check-only
python EPSS_TPG/01_tpg/02_examples/01_scripts/audit_security_product_versions_v3.py
python EPSS_TPG/01_tpg/02_examples/01_scripts/audit_security_product_versions_v3.py --verify-only
```

Output: `01_tpg/05_workspace/security_product_version_v3/smoke/`.
Read `analysis.txt` and `analysis.json`, then the full source/evidence comparison
in `comparisons.json`. Per-CVE counts are also in `comparison_counts.csv`;
separate graphs are in `graphs/`, protected by `preview_lock.json`.
Existing outputs are refused; `--output-dir` allows a new location. This does
not rewrite the earlier review packet or assign human semantic judgments.

#### Verified v3 Results

| Measure | Frozen v2 | v3 |
|---|---:|---:|
| Development CVEs | 56 (32 train, 24 validation) | Same 56 |
| Release mentions | 110 | 111 |
| Candidate product/release links | 43 | 43 |
| Release mentions with no assigned product | 67 | 68 |
| Same-span candidate coreference edges | 91 | 92 |

Only CVE-2022-26130 changes in the saved node/edge content comparison. The new
VERSION node is `16.1.x` at `[13, 19)`, still `unsupported_wildcard_release`.
There are no removed nodes or edges, changed existing node/edge properties,
or new security relations after accounting for ID shifts and policy-version
tags. The remaining 55 graphs match. Graph metadata/pass history are excluded
from this comparison, not claimed identical.

The three added edges are sentence CONTAINS, VERSION-to-token BELONGS_TO, and
CARDINAL-to-VERSION COREF. Both endpoints of the last edge occupy `[13, 19)`.
This is one occurrence annotated twice, not two separate mentions. The shared
core's repeated_surface rule groups equal text without excluding identical
spans or checking compatible entity types. V3 exposed an existing limitation.

The audit's `expected_big_ip_recovery_present=false` is caused by its combined
requirement for exactly two added edges, not by failure to recover the release.
That expectation was based on a shorter synthetic sentence. The original
report, flag and checksums are preserved; the separate review does not simply
declare every graph edge semantically correct.

Review script: `01_tpg/02_examples/01_scripts/review_security_product_v3.py`.
Results: `01_tpg/05_workspace/security_product_version_v3/smoke_review/`
(`analysis.txt`, `analysis.json`, `README.md`, `review_lock.json`).
Three new tests verify ID/policy normalization, retained evidence, edge
multiplicity, missing-endpoint rejection and same-span detection. All 63
combined focused tests pass.

```bash
python EPSS_TPG/01_tpg/02_examples/01_scripts/review_security_product_v3.py --check-only
```

Next recommended change: separately version same-span coreference handling,
preserve legitimate repeated mentions, and preview on this same sample before
considering expansion. Do not edit the frozen core, audit or saved graphs.
Human review remains pending; no forecasting benefit follows from these counts.

### DQ11: Saved Product/Version Source Review

**Status: packet generated by the user and reproduced on all 56 saved
descriptions; separate automated source analysis completed. Human semantic
review remains pending.**

The records, summary counts, source/code pins and checksums match.
`analyze_security_product_packet.py` saves TXT/JSON findings under
`01_tpg/09_annotation/07_product_version_review/packet_analysis_v1/`, outside
the locked packet. It does not modify `review_template.json` or any decisions.
All 56 full-source confirmations and 110 item judgments remain pending.

Findings from the actual saved descriptions:

- 43 linked mentions, 67 unresolved mentions across 20 CVEs, and 12 CVEs with
  no recognized versions. Four descriptions account for 38 unresolved mentions:
  Zyxel (16), ESET (9), F5 BIG-IP (7), Ruby (6).
- **Confirmed filter bug:** in CVE-2022-26130, the raw regex recognizes
  `16.1.x` but the IP-address context filter drops it after `BIG-IP`. A
  synthetic product-name-only substitution preserves the same release,
  while a genuine `IP address 192.168.1.1` is correctly excluded. No extractor
  was changed. The saved packet has no item for this omitted release.
- **Unsupported release shapes:** Java's `7u291`, `8u281`, and integer `16`
  are missed. Ivanti EPM's `2022 SU6` and `2024 September update` are also
  missed in a description with zero version nodes. MitraStar's `100VNZ0b33`
  needs role clarification, not an automatic firmware label.
- **Incomplete structures:** ESET and Splunk links to initial releases do not
  encode complete ranges. A later Hawk `9.0.1` patch sentence must not become
  an affected-version claim. Lowercase names/descriptors can prevent matching
  even when a product is clear from the source.

These are automated source/code observations, not authenticated human labels
or precision/recall measurements. The retained Desigo, Snyk, patch and bound
corrections remain present. Next implementation should fix the narrow filter
collision with paired controls, not try to force all 67 abstentions into links.
No full expansion or training yet; any broader source-only run remains a
separately specified decision after the bounded fix and review.

Verify the completed analysis:

```bash
python EPSS_TPG/01_tpg/02_examples/01_scripts/analyze_security_product_packet.py --check-only
```

The original packet-generation specification follows for provenance.

Script: `01_tpg/02_examples/01_scripts/review_security_product_v2.py`.
Outputs belong under `01_tpg/09_annotation/07_product_version_review/`,
separate from the immutable DQ10 preview and existing learned-claim annotations.
Eleven new tests cover span/quote mismatches, generic/typed edge disagreement,
test-cohort exclusion, source tampering, missing decisions, reviewer/timezone
requirements, missed mentions, incomplete reviews, packet checksums and
original-file preservation. The combined focused suite has 42 passing tests.

Scope: exactly the same 56 development CVEs, 43 candidate product/version
links and 67 unresolved version mentions. Import the saved GraphSON through
the core importer; do not reparse text, load model weights or generate graphs.
Check every source quote, sentence, qualifier and ownership decision, ensure
each version has exactly one linked owner or an explicit abstention, and
verify that the generic and typed copies carry matching evidence. Changes
are grouped into new mentions, removed/added owners, changed qualifiers or
release text, and unchanged items. Include all descriptions, even when no
version was recognized, to allow missed-release review.

The packet begins entirely pending. Separate review decisions require full
source inspection, a reviewer name, a timezone-aware timestamp, exact spans
for missed releases and explanations for concerns. An unchecked case is never
treated as correct or as a negative label. Partial completed-CVE coverage is
reported explicitly; no accuracy or recall is invented from candidate counts.
This remains development-informed engineering evidence, not held-out gold.
HAS_VERSION is not coerced into affected-version/fixed-version claims from the
existing learned-extraction schema.

Run from the project root with `CodeBERTFusion` active:

```bash
python EPSS_TPG/01_tpg/02_examples/01_scripts/review_security_product_v2.py prepare --check-only
python EPSS_TPG/01_tpg/02_examples/01_scripts/review_security_product_v2.py prepare
```

The [DQ11 guide](../01_tpg/09_annotation/07_product_version_review/README.md)
describes `edit-copy` and `summarize`, file purposes and decision meanings.
Original artifacts and both extractors remain unchanged. No full expansion,
embeddings, model training, new forecasts or test inspection. Completion of
this worksheet does not automatically authorize an expanded experiment.

### DQ10: Product/Version Extraction Preview

**Current status: v2 implemented, 31 combined tests pass, and the separate
56-CVE preview completed. No full expansion, training, embeddings, model-ready
cache or test scoring.**

The new files are `01_tpg/10_extensions/tpg_security_product_v2.py` and
`01_tpg/02_examples/01_scripts/audit_security_product_versions_v2.py`, with
14 new extraction/lifecycle tests in `01_tpg/08_tests/test_security_product_v2*.py`.
V2 reuses the core graph/NLP/provenance/export machinery and the unmodified v1
software-placement and edge-rebuild helpers. It is not enabled in any existing
model or default pipeline. The new CLI intentionally has no full mode.

Output: `01_tpg/05_workspace/security_product_version_v2/smoke/`.
The same 56 source descriptions and 32/24 split counts are retained. Before
writing the new output, v1 is rebuilt and all 56 GraphSON objects match their
original saved values. Source/code and original preview/review checksums pass
again after the run. Only a separate v2 output is created.

| Measure on the identical sample | v1 | v2 |
|---|---:|---:|
| Recognized version mentions | 94 | 110 |
| Candidate product/version links | 44 | 43 |
| Recognized versions without an owner | 50 | 67 |

The extra 17 unresolved mentions comprise 14 newly recognized wildcard
releases, one integer build, one product-prefixed release and one rejected
incomplete Oracle owner. The original 50 no-adjacent-product cases remain.
Nine existing release mentions also retain Patch wording. Candidate counts
are not extraction accuracy, and these reviewed examples are not independent
validation of the new rules.

Observed corrections:

- CVE-2022-24042: the four versions now attach to Desigo DXR2/PXC3/PXC4/PXC5,
  not `All`.
- CVE-2021-2163: the incomplete `Enterprise Edition` owner is rejected with
  `parser_split_product_requires_review`; no cross-sentence name is invented.
- CVE-2024-48964: `Snyk CLI` no longer includes the generic word `package`.
- CVE-2021-33318, CVE-2022-27790/27801/28237, CVE-2024-23749 and
  CVE-2024-8190: trailing or parenthesized upper bounds are retained.
- CVE-2024-8190: `4.6 Patch 518` remains a complete version mention.
- Wildcard, integer-build and product-prefixed forms are preserved as version
  nodes with explicit unsupported-form reasons, not fabricated ownership.

Four CVEs have changed relation endpoints; other changes include bound
metadata and unsupported release nodes. Non-version typed relations are
unchanged. Complex release lists/ranges and missing names remain unresolved;
the added metadata is still not encoded by the frozen 64-value edge features.
**Do not expand or train yet.** Review these changes and remaining gaps before
a separately approved next experiment.

Verify the completed v2 run from the project root with `CodeBERTFusion` active:

```bash
python EPSS_TPG/01_tpg/02_examples/01_scripts/audit_security_product_versions_v2.py --verify-only
```

The [extension guide](../01_tpg/10_extensions/README.md) includes test commands
and optional replay instructions using a new destination. The original v1
implementation, preview and review are retained below as historical context.

**Previous v1 status: smoke completed and reproduced; do not expand v1.**

The separate review under
`01_tpg/05_workspace/security_product_version_v1/smoke_review/` contains
`analysis.txt`, `analysis.json`, `README.md` and `review_lock.json`.
`review_security_product_preview.py` checks the original artifact hashes and
source/code identity, then temporarily repeats the 56-CVE extraction. All
comparison rows agree after list-order normalization; all 56 new GraphSON
objects agree exactly. The original preview and its pinned files remain unchanged.

| Measure | Before | After |
|---|---:|---:|
| Typed product/version candidate links | 10 | 44 |
| CVEs with such links | 10 | 32 |
| Recognized version mentions | 105 | 94 |

There are 50 recognized versions without owners across 17 CVEs. Unsupported
release forms are not included in this count. Thirty-two CVEs have changed
typed relation endpoints; the one non-version typed relation is unchanged.
These are extraction counts, not measured accuracy or forecast improvements.

The known Pluck-CMS, plugin/theme, ESET and python-jose corrections are present,
but new source-based problems prevent recommending a full preview:

- **False owner:** CVE-2022-24042 attaches four Desigo versions to `All`.
- **Incomplete name:** CVE-2021-2163 uses `Enterprise Edition`; the saved NLP
  sentence boundary separates it from `Oracle GraalVM`.
- **Overbroad name:** CVE-2024-48964 includes `package` in `package Snyk CLI`.
- **Missing detail:** trailing `and below`, parenthesized `and earlier`, and
  `4.6 Patch 518 and before` are not fully represented as qualifiers/releases.
- **Coverage gaps:** product-prefixed versions, integer builds and wildcard
  releases can create no version node and therefore no abstention count.

Next: preserve v1, introduce a versioned correction with focused negative
controls, and repeat the small preview. Do not expand v1 or create a training
cache yet. The reviewer can be checked without rerunning extraction:

```bash
python EPSS_TPG/01_tpg/02_examples/01_scripts/review_security_product_preview.py --check-only
```

The original implementation and run specification follow for provenance.

The next step follows the confirmed DQ09 product/version problems. It is a
bounded rule correction, not learned extraction or a claim of better forecasts.
The implementation is under `01_tpg/10_extensions/tpg_security_product_v1.py`;
it reuses the existing spaCy frontend, graph classes, enrichment, provenance
and exporters. No file is added to or edited inside the frozen `01_core/tpg`:
even a new Python file there would invalidate the old cache's code manifest.

Changes are limited to source-bound software/version extraction and replacing
both generic and typed `HAS_VERSION` copies with the same ownership decision:

- Keep complete dotted versions, including `4.7.18` from `v4.7.18`, with literal
  prefix and source offsets preserved. Avoid CVSS/IP contexts and truncated
  suffixes. Unsupported wildcard versions are not converted to shorter numbers.
- Prefer an adjacent full product phrase over embedded platform names. Do not
  treat filename extensions as standalone software products.
- Sort software mentions deterministically. Regression checks use three Python
  hash seeds without changing any frozen cache or global core constants.
- Retain `before`, `through` and supported bounds as edge metadata, including
  the exact qualifier text and span. Bare `up to` leaves inclusivity unspecified.
- Record abstentions when a recognized version has no supported unambiguous
  owner. Do not fabricate a CVE entity from the record ID or a version owner
  from mere sentence proximity.

All ten previously flagged source descriptions were checked locally. For
example, `php -> 7.18` becomes `Pluck-CMS -> 4.7.18`, and the ESET version is
attached to the complete ESET product phrase instead of Windows alone. These
are targeted regression observations, not independent accuracy measurements.
Thirteen focused tests pass, covering source spans, qualifiers, ambiguity,
cross-sentence rejection, other-relation negation, deterministic construction,
export, source/output preservation, test-cohort exclusion and preview lifecycle.

**Selection:** 56 unique CVEs: the frozen hash-selected 32 training and 16
validation sample plus eight additional reviewed validation cases. Two reviewed
training cases already occur in the sample. Both selection reasons are recorded;
known validation problems are not presented as unseen evaluation examples.
The preflight passed without creating preview artifacts. The entire old cache
is checked for integrity, but the smoke preview reparses only its selected CVEs.

Original commands, already completed; do not overwrite the saved preview:

```bash
python -m unittest discover -s EPSS_TPG/01_tpg/08_tests -p 'test_security_product*.py'
python EPSS_TPG/01_tpg/02_examples/01_scripts/audit_security_product_versions.py --scale smoke --check-only
python EPSS_TPG/01_tpg/02_examples/01_scripts/audit_security_product_versions.py --scale smoke
```

Output: `01_tpg/05_workspace/security_product_version_v1/smoke/`.
`analysis.txt`, `analysis.json` and `README.md` explain scope and counts;
`comparisons.json` retains all descriptions and before/after entity/relation
evidence; `comparison_counts.csv` is the compact table; `graphs/<CVE>.json`
contains corrected GraphSON; status and lock files protect reproducibility.
Existing outputs are refused. The full source-only preview is gated by a
completed matching smoke report; its command is in the
[extension guide](../01_tpg/10_extensions/README.md).

**Important limitations:** these graphs have no new contextual embeddings and
are not a training cache. The frozen 64-value edge-feature encoder does not
encode the new `version_operator`/`qualifier_text` metadata. No feature contract
or model consumer was silently changed. Unknown lower-case single-word products,
integer-only releases and complex ranges still need review. Changed entities
can affect other unchanged core rules, so compare all typed relations as well
as version links. Other application and model pipelines are unchanged.

Review smoke owner boundaries, complete versions, qualifiers, deleted/new
links and abstentions before expansion. Keep a later graph-cache migration
separate from a persistence-residual prediction-head experiment so the sources
of any performance difference remain distinguishable. Do not add summaries/SMP
or inspect test descriptions during this step.

### DQ09: Saved Fusion Branch and Security-Relation Diagnostic

**Status: full v2 diagnostic completed by the researcher and independently
reproduced. No new training or test scoring.**

#### Verified Full-Run Findings

`review_forecast_tpg_fusion_diagnostic_v2.py` verifies the original file hashes,
completion status, source/code/cache/checkpoint provenance and all three fixed
models. It repeats every numerical probe on the same 197 validation CVEs:
7,683 per-CVE rows, 117 response rows and 144 representation-coordinate rows
agree exactly. It reparses all 1,220 development descriptions and verifies 122
typed security edges and 20 selected examples. Saved CSV/JSON agree and the
saved prose matches its generator; both figures were inspected visually.
The combined R01, graph-alignment, diagnostic and review suite passes 58 tests.

Source tables agree after removing only process-dependent reconstruction IDs
and index maps, retaining cached indices, text, entity types, spans, evidence
and qualifiers. Saved maps pass bijection/count/entity-index checks. The saved
run had 1,217 directly matching graphs and three permutations; reconstruction
under `PYTHONHASHSEED=4` has 1,215 direct matches and five permutations. This
difference is expected process numbering, not a difference in graph structure.

| Model seed | Original fusion MAE | Mean prediction movement after EPSS swap | Mean prediction movement after graph swap | Mean prediction movement after removing all edges |
|---|---:|---:|---:|---:|
| 17 | 0.02610255 | 0.01600652 | 0.01047818 | 0.00000485 |
| 29 | 0.02551741 | 0.01322999 | 0.00446225 | 0.00005306 |
| 43 | 0.01892104 | 0.02429255 | 0.00433356 | 0.00023313 |

Each row is a fixed training seed, not a newly fitted model. Movement is the
mean absolute change from that model's original prediction; swaps average all
three donor maps. EPSS units run from 0 to 1, so 0.01 is one percentage point.
Matched persistence MAE is **0.00909675**, lower than all three fusion MAEs.
For **162/197** validation CVEs, the Day-60 score equals the latest observed
input score. These results do not show a predictive improvement from fusion.

EPSS swaps increase recipient-target error in **9/9** comparisons, averaging
MAE increases of 0.01498756, 0.01320751 and 0.02420362 by model seed.
Graph swaps increase error in **8/9**, averaging 0.00240403, 0.00004095 and
0.00116106. The exception is seed 29 / donor map 307, reducing MAE by 0.00098770.
Thus both inputs affect forecasts, but useful graph alignment is less consistent
than EPSS alignment. These artificial swaps are not causal importance estimates.

Removing all edges moves forecasts far less than swapping the whole graph.
This suggests the saved predictions depend more on graph/node content than on
inter-node messages in this probe. It does not prove the GNN is disconnected
or that relationships never mattered during training. Swapping graphs changes
embeddings, content, size and structure together; zeroing branch representations
is also out-of-distribution and is not an importance percentage.

**Security coverage:** 105/1,023 training CVEs have 108 typed edges (84 versions,
23 causes, one severity). Only 11/197 validation CVEs have typed edges: 13
`SEC_HAS_VERSION` and one `SEC_CAUSES`. All 122 typed edges also have generic
`ENTITY_REL` copies. Most descriptions have no textual CVE-ID entity:
1,005/1,023 training and 192/197 validation, limiting rules that require it.

| Model seed | Validation CVEs with SEC edges | Mean prediction movement after SEC removal | Largest movement | CVEs moving more than 1e-7 |
|---|---:|---:|---:|---:|
| 17 | 11 | 0.0000000508 | 0.0000001825 | 2 |
| 29 | 11 | 0.0000001324 | 0.0000002179 | 6 |
| 43 | 11 | 0.0000006239 | 0.0000013160 | 11 |

These effects are very small, not exactly zero. Parallel generic links remain,
so this does not test removal of all security information. Within the 11-CVE
SEC subset, graph swaps reduce error on average for every seed; only one of
three swaps per seed increases error. Do not extend the whole-cohort alignment
finding to this subgroup without qualification.

**Eight of the 13 validation version links raise source-based concerns:**
five attach plugin/theme versions to WordPress, two attach application versions
to `php` found in filenames, and one attaches `python-jose`'s version to `python`.
Examples:

- `CVE-2022-3604`: `WordPress -> 1.3.0`; the wording names the Contact Form
  Entries WordPress plugin before 1.3.0, not a WordPress platform release.
- `CVE-2023-50564`: `php -> 7.18`; the wording names
  `/inc/modules_install.php of Pluck-CMS v4.7.18`. Both the owner and version
  boundary are problematic. The current version regex can start after `v4.`
  and match only `7.18`.
- `CVE-2024-24495`: `php -> 1.0`; the named product is Daily Habit Tracker,
  while `php` comes from `delete-tracker.php`.
- `CVE-2024-33664`: `python -> 3.3.0`; the named package is `python-jose`.

The separate review preserves all eight observations and source sentences.
These are local wording-based concerns, not a gold-labelled error rate. The
other links are not automatically certified correct. Version bounds such as
`before` and `through` also need explicit review. Original `not_reviewed`
statuses and cached graphs remain unchanged.

**Decision:** do not expand to summaries/SMP or rerun the same training yet.
First turn the confirmed source examples into small extractor tests, including
deterministic numbering, complete versions and product-versus-platform ownership.
Keep any correction/versioned graph build separate from the frozen R01 cache.
Then predeclare a persistence-residual fusion experiment with matched EPSS-only
and no-edge controls. Isolate extractor changes from prediction-head changes
rather than attributing a combined change to either one. No improvement is
guaranteed; current descriptions remain retrospective, not verified Day-30 text.

Saved review: `05_results/02_evaluation/epss_day30_to_day60/tpg_fusion_diagnostic_full_v2_review/`
contains `analysis.json`, `analysis.txt` and `README.md`. Original full outputs
are unchanged. To repeat verification without writing another report:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_tpg_fusion_diagnostic_v2.py --check-only
```

#### Graph-Ordering Failure and v2 Recovery

The original full diagnostic stopped at `CVE-2022-25866` with
`Reconstructed edge_index differs`. Reproduced with `PYTHONHASHSEED=4`:
the security extractor sorts a set of software names by length alone.
Equal-length `git` and `php` can switch order, changing four ENTITY IDs and
eight edge positions for this CVE. Its 125-node, 392-edge graph passes an exact
directed structural comparison after alignment. This is not an EPSS-data or
model-weight failure.

`diagnose_forecast_tpg_fusion_v2.py` preserves the original probe protocol,
cohorts, three model seeds and three swap seeds. A separate version is necessary
because completed v1 artifacts pin the original script's hash. The new helper
`tpg_cached_structure.py` checks equivalence before any model probes:

- Non-ENTITY node indices stay fixed, including token positions.
- ENTITY permutations must preserve all connections to fixed nodes. Unanchored
  entities cannot move. No approximate embedding similarity is used.
- Directed edge endpoints, types, all 64 numerical edge properties and counts
  of parallel edges must agree exactly after mapping. Real differences stop
  the diagnostic; they are not discarded or repaired.
- Cached tensors and embeddings are never reordered or regenerated. Models
  still receive exactly the original cached graph inputs.
- Reports retain reconstructed IDs and matched cache indices separately. These
  are structural correspondences, not independently verified semantic identities.
  The cache lacks original per-entity labels/spans, and no claim is made that
  tensor equivalence establishes semantic extraction accuracy.

Keep `tpg_fusion_diagnostic_smoke_v1/`, its review, and the failed
`tpg_fusion_diagnostic_full_v1/` unchanged. The new full output is
`tpg_fusion_diagnostic_full_v2/`. Do not use the v1 reviewer on the v2 schema.
Changing the extractor's ordering itself belongs in a versioned future graph
build, not a silent modification of this frozen experiment.

The regression suite now passes 52 tests, including changed-edge rejection,
parallel-edge multiplicities, fixed token anchors, ENTITY permutations,
checkpoint preservation and synthetic v2 end-to-end output checks.

Read-only verification with `PYTHONHASHSEED=4` reparsed all **1,220** frozen
development descriptions: **1,215 exact matches and five entity permutations**,
with no structural failures. The five CVEs were `CVE-2022-25866`,
`CVE-2022-29176`, `CVE-2022-29180`, `CVE-2022-30950` and `CVE-2022-30951`.
Counts were 1,023 training and 197 validation descriptions, retaining all 122
typed security edges. Alignment-mode counts depend on the process hash seed;
these are reconstruction checks, not results of new model probes. No test
graphs were inspected or scored.

#### Historical Small-Run Findings

The following records the completed smoke review and the decision made then.
The full diagnostic and its current decision are now verified above; the old
run commands are retained as history, not as requests to repeat completed runs.

`review_forecast_tpg_fusion_diagnostic.py` reproduced all 624 per-CVE rows,
117 response/subgroup rows, 144 branch-coordinate rows, 48 source-inspection
rows, two security-edge cases and 12 description examples. JSON/CSV/TXT agree;
source/code/checkpoint/cache/output hashes and completion status pass. Figures
were inspected visually. Original files and checkpoints remain unchanged.
The combined diagnostic, review and R01 tests pass (32 tests).

These are the **full-trained** fusion checkpoints applied to the fixed
16-validation-CVE engineering subset, not models trained on 32 CVEs.
Original fusion MAEs are 0.01640326, 0.01528559 and 0.01424011 for seeds
17, 29 and 43; matched persistence is 0.00001813. This subset has 15/16
endpoints equal to the latest input. Branch response is not evidence that
the fusion model beats persistence.

| Model seed | Mean forecast movement after EPSS swap | Mean forecast movement after graph swap | Mean forecast movement after removing all edges |
|---|---:|---:|---:|
| 17 | 0.00141746 | 0.00822062 | 0.00000366 |
| 29 | 0.00163161 | 0.00342619 | 0.00003928 |
| 43 | 0.00232756 | 0.00472335 | 0.00019788 |

Movement is mean absolute difference from the original forecast, in EPSS
units. Swaps average all three fixed donor maps. EPSS swaps increase
recipient-target MAE in **9/9** comparisons; graph swaps increase it in
**8/9**. Mean increases are 0.00141135/0.00162210/0.00231420 for EPSS swaps
and 0.00089732/0.00017702/0.00180355 for graph swaps, ordered by model seed.
The seed-29 graph swap with donor-map seed 307 reduces MAE by 0.00067416;
do not hide this exception or call every graph pairing useful.

Both branches affect predictions. Much smaller movement after removing edges
than after swapping whole graphs is consistent with limited message-passing
contribution to these selected forecasts. Graph swaps also change node content,
embeddings, counts and graph structure, so they cannot isolate relationship
value. This is not proof that edges were irrelevant during training or that
all TPG uses fail. Zeroing the graph representation causes larger movement
(0.03970770/0.03965451/0.02050477), but zero is out-of-distribution, not a
calibrated feature-importance reference.

The inspected training descriptions contain two `SEC_HAS_VERSION` edges;
the 16 validation descriptions contain none. Removing SEC_ edges is therefore
an exact no-op in validation. SEC-positive subgroup metrics correctly remain
null. Of 32 training descriptions, 31 have no textual CVE-ID entity; all 16
validation descriptions lack one. That prevents many CVE-entity-dependent
rules from firing, but does not prove a relationship should have been extracted.
Both typed training links also exist as generic ENTITY_REL links.

**New semantic concerns from the actual source wording:**

- `CVE-2022-1418`: `WordPress -> HAS_VERSION -> 2.2.9`. The description says
  "Social Stickers WordPress plugin through 2.2.9". The likely version owner
  is the Social Stickers plugin, not WordPress itself. The generic relation
  also loses the version bound, and its recorded predicate is `plugin`,
  warranting parser review.
- `CVE-2022-27167`: `Windows -> HAS_VERSION -> 8.0.12013.0`. The source phrase
  is "ESET File Security for Microsoft Windows Server 8.0.12013.0". Linking
  the version to Windows alone likely confuses the product with its platform.

These are source-based reviewer interpretations, not online verification or
human gold annotations. They demonstrate why valid spans and tensors do not
guarantee a correct relation. The original `not_reviewed` labels remain intact.
Neither case establishes a population error rate or the cause of forecast
errors. Both are training examples, not SEC-positive validation cases.

**Decision at the smoke stage (full run now completed above):** run the already specified full diagnostic without changing the
extractor, model, seeds or splits. It includes the 11 SEC-positive validation
graphs excluded from this small engineering subset. Preserve the complete
diagnosis before deciding on a separately versioned product/version extraction
repair or persistence-residual fusion. No new training or test scoring yet.

Verified interpretation is saved separately in
`05_results/02_evaluation/epss_day30_to_day60/tpg_fusion_diagnostic_smoke_v1_review/`
as `analysis.json`, `analysis.txt` and `README.md`.

Current read-only preflight (checkpoint reproduction; not full source inspection):

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/diagnose_forecast_tpg_fusion_v2.py --scale full --check-only
```

Full-run command used after the ordering repair (now completed):

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/diagnose_forecast_tpg_fusion_v2.py --scale full
```

Read `03_analysis/01_revision_dataset/diagnose_forecast_tpg_fusion_v2.py` and
`test_forecast_tpg_fusion_diagnostic_v2.py`. This follows the completed R01 full
run; it does not replace its results, alter its code or select new checkpoints.
The core TPG, cached graphs, missing-data policy and chronological/group splits
remain unchanged.

#### Questions

1. Does changing the EPSS input move the saved fusion model's prediction?
2. Does changing the description graph move its prediction?
3. Does breaking either CVE-to-input association increase error against the
   original CVE's Day-60 score? Response alone is not evidence of usefulness.
4. Are security relations present, source-aligned and qualified correctly in
   the extracted evidence? Structural checks are automatic; semantic judgments
   remain unreviewed for human inspection.

#### Scope and Fixed Design

| Item | Fixed choice |
|---|---|
| Model | Existing full R01 `tpg_epss` checkpoints; seeds 17, 29, 43 |
| Small run | Same deterministic 16 validation CVEs as the earlier smoke subset; inspect 32 training descriptions plus those 16 validation descriptions |
| Full diagnostic | All 197 validation CVEs; inspect all 1,023 training and 197 validation descriptions |
| Training scope | None. Training descriptions are inspected, not scored; existing full-trained weights are used even for the small diagnostic |
| Prediction task | Original Day-60 EPSS given Days 0-30 and current description graph |
| Swap maps | Three fixed maps, salts 101/211/307; same maps for all model seeds |
| Donors | Bijections within selected validation CVEs, always from a different related-CVE group; no target/error-based selection |
| Probe count | Original input plus 12 probes per model seed (13 conditions) |
| Precision check | Saved baseline prediction comparison: absolute tolerance 1e-7, relative tolerance 1e-6; allows small batch-size/order numerical differences |
| Test data | No test graphs, donor CVEs, predictions, loss or performance analysis |
| Updates | No parameter updates, gradients, retraining, source changes or split changes |

The default is `--scale smoke`. It is an engineering check, not a new
16-CVE training experiment. The small validation subset previously had no
typed security edges: a no-op security-edge removal there checks consistency
but cannot assess security-edge value. The full diagnostic is needed for the
11 SEC-positive validation graphs. No automatic scale-up is performed.

#### Probes and Interpretation

| Probe | What changes | What stays fixed |
|---|---|---|
| `swap_epss` | Whole past EPSS sequence, masks and observation ages from a different validation CVE | Recipient graph and target |
| `swap_graph` | Whole description graph, node embeddings and edges from a different validation CVE | Recipient EPSS sequence and target |
| `zero_epss_branch` | Final 16-dimensional EPSS representation is zeroed just before fusion | Graph representation and saved weights |
| `zero_graph_branch` | Final 32-dimensional graph representation is zeroed just before fusion | EPSS representation and saved weights |
| `remove_security_edges` | SEC_ edge messages are removed using saved vocabulary IDs | Nodes, security wording, generic ENTITY_REL links and all other inputs |
| `remove_all_edges` | All inter-node messages are removed | Node representations, saved layers, pooling and EPSS input |
| `observed_plus` / `observed_minus` | Observed past scores shift by +/-0.01, clipped to [0,1] | Missing padding, masks, observation ages and relative day positions |

Swapping complete branches is artificial: combinations may not correspond to
a realistic CVE. Zeroing a representation is also not a valid missing-input
policy. Error changes against recipient targets are dependence diagnostics,
not new forecast accuracy claims, ground truth for synthetic pairings, causal
effects or fair retrained ablations. Removing all edges from a trained model
is distinct from R01's separately trained no-edge comparison.

Record mean absolute prediction movement, signed movement, maximum movement,
original and probe recipient-target MAE, and their difference. Keep every
seed/repeat; no best swap or checkpoint is selected. Report all validation
CVEs and separate with/without-SEC subsets. An empty subset has null metrics,
not zero error. Response >1e-7 is a numerical-change count, not a threshold
for declaring a branch useful. No significance test is claimed.

If swaps move outputs but do not consistently increase recipient-target error,
that does not demonstrate useful CVE-specific alignment. If they increase
error, it supports dependence on the original association within this inspected
validation cohort, not causality or guaranteed future performance. Component
variation is measured separately for the two branches; learned coordinate
scales cannot be treated as directly comparable importance scores.

#### Security Inspection

The saved spaCy pipeline reparses descriptions without rerunning or downloading
the embedding model. Node-type order, edge endpoints/types and 64-dimensional
edge-property values must exactly reproduce the cache after the restricted
v2 ENTITY/edge-order alignment described above. Source sentence and
entity spans are checked against raw wording. Every extracted security edge
is recorded with source/target text, relation type, provenance, negation,
modality and any parallel generic relation. Cases remain `not_reviewed`.

The CVE-level coverage table includes extracted entity counts, relation counts
and whether a CVE-ID entity occurs in the text. Many current rules require that
entity; a metadata identifier is not automatically a textual mention. Absence
does not justify prepending an ID or inventing relations. Entity co-occurrence
does not establish that a relation was missed. Human annotation is needed to
measure extraction correctness/recall. Stable-ID examples include descriptions
with and without typed edges, selected without model-error information.

SEC_ removal is also checked to leave SEC-free graphs' predictions unchanged
within numerical tolerance. A small whole-cohort response can reflect limited
coverage; inspect the SEC-positive subset. Generic security relations, wording
and embeddings remain after removing typed edges, so this is not removal of
all security information.

#### User Commands

The following records the completed smoke workflow. Do not rerun it into
the existing destination. From the repository root with `CodeBERTFusion` active:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/diagnose_forecast_tpg_fusion.py --scale smoke --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/diagnose_forecast_tpg_fusion.py --scale smoke
```

The first command verifies saved models, data/code hashes and fixed donor maps
without probing or writing reports. The second produces the small diagnostic.
Both have completed. The original full attempt then failed on reconstruction
ordering. Use the v2 recovery for the full diagnostic:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/diagnose_forecast_tpg_fusion_v2.py --scale full
```

The preserved original outputs are `tpg_fusion_diagnostic_smoke_v1/` and
the failed `tpg_fusion_diagnostic_full_v1/`. The v2 commands instead default to
`tpg_fusion_diagnostic_smoke_v2/` or `tpg_fusion_diagnostic_full_v2/` under
`05_results/02_evaluation/epss_day30_to_day60/`. Existing destinations are
refused; a deliberate rerun needs a new `--output-dir`. Failed runs retain a
failure status. Key verification, per-seed probes, source inspection and
report writing display progress bars. No network downloads are needed.

| Files | Purpose |
|---|---|
| `analysis.json`, `analysis.txt`, `README.md` | Protocol, source/checkpoint hashes, coverage and interpretation |
| `responses.json/csv` | Per-seed/probe/repeat response and recipient-target errors, including SEC subgroups |
| `per_cve.json/csv` | Actual baseline/probe predictions, donor identity and changes for each validation CVE |
| `branch_components.json/csv` | Across-CVE spread of each saved EPSS/graph representation coordinate |
| `security_coverage.json/csv` | Per-CVE counts, textual CVE-entity availability and exact structure-alignment index maps |
| `security_edge_cases.json/csv` | Every typed relation with source evidence, qualifiers and manual-review status |
| `inspection_examples.json/csv` | Source descriptions with/without typed edges, chosen without outcomes |
| `branch_sensitivity.png` | Probe on x; mean absolute forecast movement on y; each line is a model seed |
| `security_coverage.png` | Split on x; CVE count with/without typed security edges on y |
| `run_status.json`, `analysis_lock.json` | Completion/failure state and successful output hashes |

Tests cover group-safe bijections, masks/padding/clipping, vocabulary-based
edge removal, branch-specific hooks, immutable inputs/weights, empty-subset
metrics, test-CVE rejection, qualified evidence round trips, cache tampering,
saved forecast reproduction, output locks and destination protection. Synthetic
tests do not train on the real data. Small-run findings are verified above;
full-cohort findings and the separate v2 review are now verified above.

After review, decide whether to propose persistence-residual fusion, a bounded
extraction revision or no model change. Do not automatically add SMP/summaries,
increase model size, remove difficult CVEs or inspect the held-out test set.

### DQ08: Exact-Key Grouping Decisions

**Status: packet generation and structural validation completed and reproduced.
All 1,212 decisions remain unreviewed; evidence decisions are pending.** This step
organizes scientific review, not another model, data cleaning pass or split change.

#### Verified Packet Results

`review_grouping_key_packet.py` reconstructs the packet from the source and
reproduces both saved validation reports without modifying them. The decisions
file still exactly matches the initial unresolved template. Nothing was removed.

| Evidence type | Different shared values | Different associated CVEs |
|---|---:|---:|
| URLs | 629 | 1,013 |
| Stored post texts | 539 | 1,178 |
| Descriptions | 44 | 178 |
| All types | 1,212 | 1,700 |

A value is one normalized URL or text, not one CVE. CVEs overlap between types;
the last CVE count is a deduplicated total, not the sum above. Of these 1,700 CVEs,
1,609 are currently excluded, 4 are assigned to train, 6 to validation and 81 to
reserved test. Only URL context includes active-split records; every exposed
post/description value belongs entirely to excluded records. No test outcomes
were analyzed. The queue contains 23 homepage hints, not 23 approved removals.

Examples explain why decisions need context:

- Item 1, `https://phpgurukul.com/`, is shared by 66 CVEs. A broad homepage
  connection is not proof that the vulnerabilities are duplicates.
- Items 24 and 25 contain 175,357 and 170,158 normalized characters, respectively,
  including navigation and cookie notices. They are associated with 69 and 66
  CVEs, and all those IDs actually occur in their respective full texts. An ID
  occurrence does not establish relevance to the main article. Review extraction
  and article boundaries before proposing any removal.
- Item 29 is the identical 51-character description "windows kernel elevation
  of privilege vulnerability", shared by 38 CVEs. They may be different flaws,
  but identical predictor text across splits still requires duplicate protection.

Separate findings: `05_results/02_evaluation/epss_day30_to_day60/grouping_key_review_v1_review/`
contains `analysis.json`, `analysis.txt` and `README.md`, including exact source
locations and hashes. The original packet, decisions and reports remain unchanged.
The reviewer and DQ07/DQ08 tests pass (26 tests). To verify again without writing:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_grouping_key_packet.py --check-only
```

Next: review a bounded set of these examples and document evidence-based
decisions. This run did not calculate how many excluded CVEs could be recovered;
other shared links may still keep their groups connected. Any applied policy
requires a separate approved, versioned preparation.

`review_grouping_keys.py` prepares all 1,212 DQ07 queued keys with exact identities,
source CVE/field locations and unresolved decisions. URLs are preserved in the
existing normalized form. Full normalized description/post content is supplied
only when every associated CVE is currently excluded; active-split text is
withheld. The exact full key is checked against the raw source, not inferred
from the 240-character excerpt. No structured EPSS targets, catalogue labels,
predictions or performance measures enter the packet. The stored text may
itself mention scores/exploitation and is not authorized as a predictor.

| Command | Purpose | Changes permitted |
|---|---|---|
| `prepare --check-only` | Reproduce DQ07, reconstruct exact evidence and check paths/hashes | None |
| `prepare` | Create a separate immutable evidence packet and editable decisions | New review folder only; every decision starts unresolved |
| `show --rank N` or `show --key HASH` | Display one context with source locations | None; text limited to 2,400 characters unless `--full` |
| `record` | Record retain/exclude/unresolved for one exact key | One decision row; previous decisions saved by checksum |
| `validate` | Check packet/source/code hashes, all decision identities and required metadata | Refresh TXT/JSON validation reports only |

#### Decision Rules

- `retain`: propose continuing to use this exact key for grouping, with a reason.
- `exclude`: propose dropping this exact grouping key, with source evidence and
  a plan to prevent its content from leaking across future predictor splits.
- `unresolved`: insufficient evidence; no implicit removal. Partial review is
  allowed and all untouched keys remain unresolved/protected.

A reviewed decision requires a reviewer, an ISO timestamp with explicit timezone,
a reason and an evidence note identifying the source context examined. Exclusion
also requires a `predictor_safety_plan`. A written plan does not prove that
cleaning is implemented. A withheld text key cannot be excluded in this packet.
Generic URL shapes or repeated text do not generate automatic decisions.

Validation rejects missing/duplicate/unknown keys, changed kind/context hashes,
unsupported decisions, incomplete documentation and unsupported exclusions.
Immutable context is reconstructed from source even if someone changes a file
and updates its checksum. The packet is tied to the current source/preparation,
DQ07 artifacts and code. This provides reproducibility, not cryptographic proof
of a reviewer's identity or proof that their scientific claims are correct.

**A valid file may contain 1,212 unresolved keys.** Even all-resolved decisions
are not authorization to apply exclusions. There is no adapter-ready exclusion
export, recovered-CVE calculation, re-preparation, training or test scoring.
No existing source, split or checkpoint is modified.

#### Setup Commands (Already Completed)

From the repository root, with CodeBERTFusion active:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_grouping_keys.py prepare --check-only
```

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_grouping_keys.py prepare
```

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_grouping_keys.py validate
```

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_grouping_keys.py show --rank 1 --full
```

Default folder: `05_results/02_evaluation/epss_day30_to_day60/grouping_key_review_v1/`.
Do not rerun `prepare` into this completed folder. Creation refuses an existing
folder. Override `prepare --audit-dir/--output-dir`
or `show/record/validate --review-dir` for a deliberate separate packet.

The following is command syntax, not a pre-approved decision. Replace the
bracketed text after inspecting the source context; choose the decision on evidence:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_grouping_keys.py record \
  --rank 1 --decision unresolved --reviewer "<reviewer>" \
  --reason "<why evidence is insufficient>" \
  --evidence-note "<CVE/field locations examined and what they show>"
```

For a proposed `--decision exclude`, also supply
`--predictor-safety-plan "<how shared content will be excluded/cleaned and duplicates rechecked>"`.
Run `validate` after edits. `record` is single-reviewer, not a concurrent service;
it checks for intervening changes and retains the prior decisions snapshot.
Direct JSON edits are allowed but do not automatically produce history.

| File | Purpose |
|---|---|
| `review_packet.json` | Locked, source-bound context for every queued key; no approval decisions |
| `decisions.json` | Editable exact-key decisions, initially unresolved |
| `decision_history/*.json` | Previous exact decisions bytes saved by `record`, named by SHA-256 |
| `README.md`, `review_guide.txt` | Plain-language instructions and safety limits |
| `packet_lock.json` | Hashes of immutable packet/guide files; deliberately excludes editable decisions |
| `validation/analysis.json`, `validation/analysis.txt` | Structural validity and progress, bound to the exact decisions checksum |

Offline tests cover full versus truncated context, source-key association,
active-text withholding, structured-target isolation, unresolved defaults,
missing/duplicate/unknown keys, reviewer metadata, timezone requirements,
predictor safety plans, immutable context tampering, overwrite protection,
history preservation and unchanged original inputs. Run from the analysis folder:
`python -m unittest test_grouping_key_review`.

After packet generation, review bounded sets of keys and document their decisions.
Do not require every key to be decided before discussing a limited policy
proposal, and do not treat unresolved keys as excluded. Only a later separately
approved/versioned preparation can apply a justified revision. Historical text
availability and matched EPSS-only/TPG-only/fused comparisons remain subsequent
steps, not results of this workflow.

### DQ07: Full-Dataset Coverage and Grouping Review

**Status: completed by the researcher and reproduced locally.**
This is the next data-use check, not another EPSS model. EPSS-only references,
sequence models and F04 diagnostics are complete. R01 now has verified
current-dataset TPG smoke and full development runs; no neural fit beats
persistence on full validation MAE. The older graph experiments do not
establish its performance on this revised target and dataset.

#### Verified Findings

`review_forecast_cohort_coverage.py` verifies all hashes and provenance,
reproduces seven JSON tables, checks CSV/text agreement, and independently
reconstructs the connected groups using adjacency traversal rather than the
adapter's union-find. No count or reproduction discrepancy was found. The
coverage figure was inspected visually and remains unchanged.

| Candidate period | CVEs before grouping, after date rules | CVEs after grouping | Removed by cross-period grouping |
|---|---:|---:|---:|
| Training | 1,615 | 1,156 | 459 |
| Validation | 407 | 199 | 208 |
| Reserved test | 3,552 | 2,671 | 881 |

These counts precede target-availability checks. Grouping excludes 51.11% of
date-eligible validation CVEs. The source file still contains all 5,968 CVEs;
none were deleted or randomly downsampled.

The largest connected group has **1,377 CVEs**. Of these, 48 already failed
the date rules; 391 candidate training, 164 candidate validation and 774
candidate test records are excluded because the group crosses periods.
Those **1,329 group exclusions account for 85.85%** of all 1,548 group-related
exclusions. Its 1,117 shared keys comprise 562 URLs, 512 post-text keys and 43
description keys. Eleven are homepage URLs. The largest direct shared key
links 69 CVEs, not all 1,377 directly.

Examples requiring context review, not automatic deletion:

- `https://phpgurukul.com/` directly links 66 CVEs. These are all candidate-test
  records; the larger component crosses periods through other connections.
- `https://www.dlink.com/` and `https://www.netgear.com/` directly link 23 and
  16 CVEs respectively. A common vendor homepage is not by itself proof of
  duplicate vulnerabilities.
- A repeated text headed by a Serbian student/Cellebrite article connects
  69 CVEs; its saved excerpt includes website navigation. This suggests a
  page-relevance/scraping review, not proof that the full page is irrelevant.
- An aerospace CVE-list page connects 61 CVEs. An aggregated list can mention
  many distinct vulnerabilities legitimately.
- The normalized description `windows kernel elevation of privilege vulnerability`
  is shared by 38 CVEs. A shared generic title need not mean identical flaws.

The manual queue contains **1,212 keys**, including **23 homepage URL hints**.
Counts overlap. Do not add their connected-group counts or interpret them as
recoverable CVEs. Other links can keep a group connected after one is removed.
Text examples in the separate review come only from already-excluded groups;
active test text and numerical outcomes are not analyzed.

**Interpretation:** the grouping implementation reproduces its declared rules,
but broad references and repeated generic/page text may make the policy too
restrictive. This is not yet a confirmed software bug or an approved relaxation.
Review exact keys with original context and written reasons. Do not remove
shared text from grouping while still feeding identical text across splits.
Feature cleaning and duplicate protection must remain consistent.

Reproduce the completed review without rerunning the original audit:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_cohort_coverage.py
```

The separate [review](../05_results/02_evaluation/epss_day30_to_day60/cohort_coverage_audit_v1_review/README.md)
contains TXT/JSON findings, real normalized excerpts, exact URL examples and
scope limitations. Original data, grouping rules, splits and runs are unchanged.

| Question | Output and rule |
|---|---|
| Where is every source CVE? | One row per CVE with candidate/assigned split, group ID, dates and final exclusion reason. |
| Why is a CVE unavailable for a task? | Separate score-availability reasons for endpoint and minmax, using masks rather than numerical target values. |
| Does this inspect reserved test performance? | No. Only identities, dates, group structure and availability masks/counts are summarized. Shared preparation validation checks raw values for integrity only. |
| Why can a group become large? | All connected groups and shared keys are listed; transitive links are distinguished from direct shared evidence. |
| Which evidence should be examined? | A manual queue of used homepage URLs, direct cross-period links and keys inside quarantined groups. All shared evidence is retained separately. |
| What do review counts mean? | Direct counts identify CVEs sharing a key. Connected counts include transitive neighbors. They overlap across keys and are not recoverable-CVE estimates. |
| Are any links or CVEs removed automatically? | No. No new URL exclusions, target definitions, split assignments or predictor features are authorized. |

`audit_forecast_cohort_coverage.py` verifies the prepared/source hashes and
reconstructs the grouping-evidence list from raw data. The analysis itself
does not access numerical input/target values, model checkpoints, predictions,
errors, catalogue labels, or post text. Shared URLs are retained exactly as
normalized by the existing grouping contract; shared text is represented by
key hashes and CVE IDs. Grouping evidence is not historical predictor evidence.

The manual queue orders homepage URL hints first, then larger quarantined
connected groups and directly shared counts, without looking at model outcomes.
A homepage is not automatically an invalid link; context must decide. Broad
links may over-group, while genuine multi-CVE advisories may correctly connect
records. No number of recovered CVEs is promised by this audit.

#### Commands

The original creation commands are retained for reproducibility. The default
destination already exists; use the review command above for verification,
or a new `--output-dir` only for a deliberate separate audit. Run from the
repository root with CodeBERTFusion active, one command at a time:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/audit_forecast_cohort_coverage.py --check-only
```

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/audit_forecast_cohort_coverage.py
```

Default output: `05_results/02_evaluation/epss_day30_to_day60/cohort_coverage_audit_v1/`.
The script refuses an existing destination. Overrides are `--prepared-dir`
and `--output-dir`. No downloads, training or test scoring are performed.

| File | Contents |
|---|---|
| `cve_dispositions.csv/json` | Every CVE, assigned role, dates, group, observed-day counts, task eligibility and separate split/missing-score reasons. |
| `split_counts.csv/json` | Counts before/after related-group quarantine. |
| `task_coverage.csv/json` | Task eligibility and mutually exclusive missing-observation reasons within each role/task. |
| `exclusion_counts.csv/json` | Mutually exclusive final split-exclusion reasons. |
| `group_sizes.csv/json` | All connected groups, their CVE IDs, assigned/candidate roles and evidence types. |
| `shared_evidence.csv/json` | All shared keys, exact normalized URLs, text hashes and existing exclusion decisions. |
| `review_queue.csv/json` | Evidence needing manual context review; not a list of authorized deletions. |
| `coverage.png` | Assigned role on x; CVE count on y. Each task panel stacks eligible and ineligible counts. |
| `analysis.txt/json`, `README.md` | Plain-language explanation, counts, scope, limitations and input hashes. |
| `run_status.json`, `analysis_lock.json` | Completion state and artifact checksums. |

Tests cover complete accounting, zero versus missing scores, task-specific
eligibility, chronological exclusions, transitive grouping, review-count overlap,
previously reviewed URL exclusions, target-value isolation, source/evidence
tampering, original-file preservation, protected paths and failure handling.
Run `python -m unittest test_forecast_cohort_coverage` from the analysis directory.

Reviewer tests additionally verify independent component reconstruction, split
count arithmetic, absence of active-test text in excerpts, outcome-value
isolation, file preservation and rejection of rehashed altered tables:
`python -m unittest test_forecast_cohort_coverage_review`.

After this verified audit, inspect the queue and document any justified policy
proposal before implementation. Keep current results intact. A policy revision
must not be selected because it yields better validation errors. Then address
historical text availability for F07/F08, with EPSS-only/text-only/fused models
on identical eligible CVEs. Do not keep adding GRU variants while these data
questions are unresolved.

### DQ06: Remaining Extrema Errors and Temporal Shifts

**Status: completed by the researcher and reproduced locally.** This is a post-hoc
descriptive analysis following F04-Residual, not another trained model or a
new predictive evaluation. It did not alter scores, missingness, cohorts,
split boundaries, model checkpoints or existing results.

#### Verified Findings

`review_forecast_extrema_shift.py` verifies the original hashes and provenance,
replays saved model predictions, reproduces all eight JSON tables, checks CSV
and report agreement, and independently checks per-CVE extrema, dates, errors,
error rankings and group partitions. No fitting or test analysis is performed.
Original figures were also inspected visually; automated verification checks
their hashes rather than regenerating pixels.

| Finding | Training (797 CVEs) | Validation (163 CVEs) |
|---|---:|---:|
| Median latest observed EPSS on or before Day 30 | 0.00885 | 0.00051 |
| All Days 31-60 scores equal the latest input | 706 | 130 |
| Days 31-60 scores are constant, possibly different from latest input | 707 | 132 |
| At least one future score differs from latest input by 0.01 or more | 26 | 3 |
| Reported version transition during Days 31-60 | 9 | 0 |

The constant-future and unchanged-from-input rows overlap, not separate groups.
The threshold 0.01 is one EPSS percentage point. The latest input score 0.00885
appears in 325 training CVEs; repetition alone is not evidence of bad data.

Persistence's single largest-error CVE, **CVE-2024-23692**, accounts for
**52.93%** of its total joint validation error. Its latest input is 0.00210,
but its actual future maximum is 0.95432. The five largest-error CVEs account
for **98.27%**: CVE-2024-23692, CVE-2024-8963, CVE-2023-7028,
CVE-2024-23334 and CVE-2024-45506. They belong to five existing related-CVE
groups. The last two involve decreases; not every large error is an increase.
Residual seeds 17/29/43 concentrate **98.07%/97.27%/97.41%** of their error in
the same five CVEs. These are descriptive ranks, not exclusion rules.

All nine training CVEs with reported future version transitions have their
largest nonzero adjacent-day change on that date; eight also have their first
future difference from the latest input on that date. All 163 validation CVEs
report no version transition. Stored metadata therefore does not supply a
reported transition explanation for the validation jumps. This does not
authenticate version history or establish causes.

The previous model ranking is unchanged: persistence joint MAE is 0.00551816;
residual seeds give 0.00552955, 0.00557464 and 0.00556700. Do not remove hard
CVEs, resplit after inspecting outcomes, or infer that TPG/SMP will fix the errors.
Verify information available by Day 30 across the full eligible cohort before
declaring the next feature experiment. Validation is already repeatedly examined.

Reproduce the completed review, not the original audit into its existing folder:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_extrema_shift.py
```

The separate [review](../05_results/02_evaluation/epss_day30_to_day60/extrema_shift_audit_v1_review/README.md)
contains `analysis.txt`, `analysis.json` and `README.md`, including actual CVEs,
publication-year summaries, model comparisons and figure explanations.

#### Questions and Scope

| Question | Fixed analysis |
|---|---|
| Do typical scores differ between splits? | Compare latest input, observed-history mean/SD/slope, input coverage/age, future extrema/range and future deviations across the same 797 training and 163 validation CVEs. |
| Are specific score values repeated? | Count up to ten most frequent exact stored latest scores, future minima and future maxima separately per split. Repetition is not evidence of corruption or imputation. |
| Where does error concentrate? | On the 163 validation CVEs, report each saved model's top 1/5/10/20 largest-error CVEs and their share of total joint absolute error. |
| Which future-change patterns have higher errors? | Group validation errors by departure from the latest input, constant/changing future scores, publication year and stored version-date status. |
| When do changes happen? | Record first future difference from the latest input, first dates of future extrema and the largest consecutive-day future jump. |
| Do changes coincide with stored version dates? | Validate flag/date/day consistency and count same-date occurrences for a reported Days 31-60 transition; no causal interpretation. |
| Which date defines yearly groups? | Publication year from the stored publication date, and anchor year from Day 30, separately. Never use the CVE ID's year as a substitute. |
| What models are compared? | Persistence, past observed range, ridge, direct GRU seeds 17/29/43 and residual GRU seeds 17/29/43. All nine use identical validation records. |
| What remains excluded? | Test outcome/error analysis, training, new forecasts, optimization, new predictors, online downloads and data correction. |

The preflight replays existing selected predictions to verify the completed
residual run, then the audit reads its saved validation predictions. Training
data are used only for distribution summaries; no new training predictions or
model fits are generated. Shared integrity checks still validate prepared files,
but the audit does not summarize or evaluate the reserved test cohort.

#### Timing and Missing-Data Rules

The input is Days 0-30 after NVD publication and the future is Days 31-60.
The latest input may predate Day 30. Its date and age stay explicit.
First future difference means the first observed future score unequal to that
latest input, not proof of a change at the prediction anchor.

Adjacent-day jumps require both consecutive scores. Missing Day 30 prevents a
Day-30-to-31 comparison; an older input is never substituted for Day 30. Future
Days 31-60 are complete by the frozen F04 eligibility rule. Equal largest
absolute jumps use the earliest date. An all-zero set of adjacent changes has
no jump date. Input slope is unavailable with only one observed input score.

These mutually exclusive diagnostic groups use the largest absolute departure
of any future score from the latest input:

| Group | Exact rule |
|---|---|
| `unchanged_at_base` | Every future score equals the latest input. |
| `nonzero_below_0_01` | At least one differs, but the largest absolute departure is below 0.01. |
| `at_least_0_01` | At least one future score differs by 0.01 or more, including equality. |

Here 0.01 means one EPSS percentage point, not a 1% relative change. Decimal
arithmetic prevents a stored 0.11 minus 0.10 being misclassified at the boundary.
These are post-hoc groups about any future day, not the earlier endpoint
direction labels. Constant future scores need not equal the latest input.

Version fields are retained as stored metadata. A usable reported transition
requires a true Boolean flag, an integer day 1-90, an ISO date and agreement
between the day offset and publication date. The categories are before Day 30
within the input, on Day 30, Days 31-60, or Days 61-90 outside this task. A false
flag with no date/day is `reported_no_transition`, not independently verified
absence of a model change. Missing or contradictory fields become
`metadata_missing` or `metadata_inconsistent`; no date is invented.

Same-date comparisons require a usable reported future transition and a
nonzero jump or first difference, respectively. Both denominators are recorded.
These flags cover stored dates only, not intraday ordering, authenticated model
headers, first exploitation or causal effects. Full-window version metadata
must not become a historical predictor.

#### Commands and Outputs

The original creation commands are retained below for reproducibility. The
default output already exists; use the review command above for verification.
To create a separate audit, choose a new `--output-dir`. Verify prerequisites first:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/audit_forecast_extrema_shift.py --check-only
```

Then run the descriptive audit:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/audit_forecast_extrema_shift.py
```

Default input is `minmax_residual_validation`; default output is the separate
`05_results/02_evaluation/epss_day30_to_day60/extrema_shift_audit_v1/` folder.
Explicit overrides are `--run-dir` and `--output-dir`. Existing destinations
are refused, and the output cannot overlap raw/prepared/reference directories.
The preflight writes no results and does not construct new diagnostic tables.

| File | Contents |
|---|---|
| `analysis.json`, `analysis.txt`, `README.md` | Fixed protocol, input hashes, date ranges, cohort counts and readable explanations. |
| `cve_cases.csv/json` | One row for each training/validation CVE: observed input history, future extrema/timing, stored version fields, quality flags and Days 0-60 series. |
| `split_distributions.csv/json` | Counts, missingness, mean, SD, median, quartiles and tail percentiles for each numeric quantity and split. |
| `frequent_scores.csv/json` | Exact repeated values and CVE counts, not counts of daily observations. |
| `temporal_groups.csv/json` | Publication-year and anchor-year summaries, separately for each split. |
| `validation_errors.csv/json` | Minimum, maximum and joint absolute error per CVE/model for all nine saved comparators. |
| `model_group_metrics.csv/json` | Group sizes, per-target/joint MAE, error sums and shares of each model's full validation error. Empty-group means are unavailable. |
| `concentration.csv/json` | Error shares and CVE IDs for fixed top-k ranks. Zero total error gives an undefined share, not zero. |
| `version_timing.csv/json` | Metadata-status counts, comparable-case denominators and same-date coincidence counts. |
| `distribution_shift.png` | EPSS on x, cumulative percentage of CVEs on y; training/validation curves for latest input and future maximum. |
| `error_concentration.png` | Count of largest-error CVEs on x, percentage of total joint error on y, separately ranked for each model. |
| `largest_future_changes.png` | Up to three largest validation departures; days since publication on x, observed EPSS on y, with missing input gaps preserved. |
| `run_status.json`, `analysis_lock.json` | Status and successful output hashes; failed runs have no success lock. |

#### Interpretation Limits and Next Decision

Future outcomes define the error ranks and groups. They are not deployable
filters or predictor inputs. Error shares partition only within a single
grouping dimension; do not sum publication-year shares with version-date shares.
Each model has its own top-error CVEs. Related CVEs are not independent samples.

No significance tests, causal conclusions, improved prediction claims or
automatic next-model selection are made. A temporal shift does not justify
resplitting after seeing outcomes, and a repeated score is not automatically
bad data. Keep the existing cohorts and difficult CVEs. The completed review
does not authorize further training; any TPG/SMP extension still needs historical evidence.

Synthetic tests cover date boundaries, inconsistent metadata, exact 0.01
classification, missing Day 30, earliest ties, test-data exclusion, Days 61-90
isolation, error-share partitions, zero-error denominators, source alignment,
CSV/JSON consistency, hashes, overwrite protection and failure handling.
Reviewer tests additionally check independent corruption detection, zero-error
denominators, exact table reproduction despite changed checksums, and preservation
of all original files: `python -m unittest test_forecast_extrema_shift_review`.

### DQ05: Fixed-Checkpoint GRU Sensitivity Diagnostic

**Status: completed by the researcher and reproduced locally.** This is a post-hoc
diagnostic motivated by the completed F04-GRU results, not a new trained model
or an untouched validation experiment. Do not infer its numerical findings
from the training logs alone.

#### Question and Scope

Why do the saved forecasts vary so little across CVEs? The diagnostic examines
four stages: observed EPSS histories, the final GRU hidden state, the two linear
output values, and their ordered sigmoid transformation. These measurements can
locate weak sensitivity, but cannot uniquely explain how it developed during
training. Intermediate epoch checkpoints and historical optimizer states are
not available for this study.

| Item | Fixed rule |
|---|---|
| Models | Selected F04-GRU checkpoints for seeds 17, 29 and 43; retain all three. |
| CVEs | Same 797 training and 163 validation CVEs; no new exclusions or test scoring. |
| Inputs | Days 0-30 scores with the original masks, observation ages and day indices. |
| Targets | Days 31-60 minimum/maximum, used for descriptive spread and training-loss gradients only. |
| Hidden-state check | Population standard deviation across CVEs, for each of the 16 hidden coordinates. |
| Output check | Spread of both linear values and both forecasts; exact derivatives of the ordered transformation. |
| Input sensitivity | Derivative of each forecast with respect to each observed score; exclude missing padding from reported aggregates. |
| Fixed changes | Add/subtract 0.01 to observed scores, clipped to [0,1]; repeat the latest score in observed positions; reverse observed scores. |
| Preserved channels | All masks, missing padding, observation ages and day indices remain unchanged in these four changes. |
| Synthetic probes | Six fully observed 31-day histories, constant at 0, 0.001, 0.01, 0.1, 0.5 or 1. These are not real CVEs. |
| Weight check | Saved parameter norms and distance from the corresponding seeded, training-mean initialization. |
| Loss gradients | Full-training SmoothL1 loss at saved weights, beta 0.01. No validation-loss gradients, optimizer steps, weight decay or Adam-state reconstruction. |
| Integrity | Verify completed-run files, reproduce its selected predictions and metrics, then check pinned inputs/code again before saving diagnostic results. |
| Selection | No new checkpoint, seed, perturbation or hyperparameter selection. |

For ordered outputs `l=sigmoid(a)`, `q=sigmoid(b)` and `u=l+(1-l)q`,
the diagnostic records `dl/da=l(1-l)`, `du/da=(1-q)l(1-l)` and
`du/db=(1-l)q(1-q)`. Small derivatives can attenuate local changes; this alone
does not establish harmful saturation or an optimizer bug.

Input derivatives are absolute sensitivities averaged over observed days within
each CVE, then summarized across CVEs. They are not causal feature importance.
The hidden-state coordinates have different units from EPSS: their spread is
not a directly comparable percentage of information retained. The report also
keeps each coordinate separately so averaging does not conceal all detail.

#### Commands and Outputs

The following are the original creation commands, retained for reproducibility.
They refuse the now-existing default output directory; use the review command
below to verify the completed run instead. For a separate new diagnostic folder,
use `--output-dir` consistently with both commands.

From the repository root, first verify the prerequisites:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/diagnose_forecast_minmax_sequence.py --check-only
```

Then run the diagnostic, without fitting any model:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/diagnose_forecast_minmax_sequence.py
```

Default destination:
`EPSS_TPG/05_results/02_evaluation/epss_day30_to_day60/minmax_gru_diagnostic_v1/`.
An existing destination is refused. Alternative destinations are explicit via
`--output-dir`; outputs cannot overlap the source dataset directory, prepared
data or completed model run.

| File | Contents |
|---|---|
| `analysis.json` | Protocol, provenance, six seed/split summaries and derivative distributions. |
| `analysis.txt`, `README.md` | Tables, plain-language explanations and interpretation limits. |
| `per_cve.csv/json` | 2,880 rows: each of 960 CVEs for each seed; predictions, derivatives and signed perturbation responses. |
| `feature_spread.csv/json` | 132 rows describing individual hidden coordinates, linear outputs, predictions and targets. |
| `parameter_diagnostics.csv/json` | 18 parameter-tensor rows across three seeds: norms, initialization differences and training-loss gradients. |
| `perturbations.csv/json` | 24 seed/split/change summaries, including actual input changes after clipping. |
| `synthetic_probes.csv/json` | 18 predictions for artificial constant-score histories; no real future labels. |
| `diagnostic.png` | Validation hidden-state spread, forecast spread and maximum-forecast sensitivity, showing every seed. |
| `run_status.json`, `analysis_lock.json` | Completion state and output hashes; a failed run has no success lock. |

The row counts above are confirmed in the completed DQ05 artifacts.
Synthetic tests cover analytic derivatives against
autograd, input derivatives against finite differences, a constant-head control,
preserved masks/weights/random state, future-target separation, test exclusion,
output hashes and failure handling.

#### Interpretation and Next Decision

Perturbed histories may be unrealistic or outside the observed distribution.
They have no newly observed future outcomes, so we do not score them against
the original labels. Training-loss gradients are measurements at the selected
weights, not a replay of past updates or proof of convergence. Computing these
gradients does not train the model.

Review the diagnostic before proposing another experiment. Any change to the
loss, initialization, output layer or feature set needs its own declared protocol
and matched persistence comparison. Do not select a change from test results or
claim that this diagnostic establishes a unique cause. No TPG/SMP data are added
and the historical-evidence limitations recorded in DQ01 still apply.

#### Verified DQ05 Findings

`review_forecast_minmax_diagnostic.py` reproduced all six seed/split summaries,
2,880 CVE/seed rows, 132 feature-spread rows, 18 parameter rows, 24 perturbation
rows and 18 synthetic probes. Original run/diagnostic hashes, CSV/JSON/text,
scalar output derivatives and response/spread aggregates agree. No fitting,
parameter updates, source edits or test scoring were performed. The original
figure was also visually inspected; its axes and labels are readable.

| Seed | Maximum forecast for constant-0 history | Maximum forecast for constant-1 history | Mean absolute maximum-forecast change after +0.01 to observed validation scores |
|---|---:|---:|---:|
| 17 | 0.01045141 | 0.01059003 | 0.00000097 |
| 29 | 0.01036652 | 0.01328704 | 0.00002262 |
| 43 | 0.01050238 | 0.01094006 | 0.00000374 |

The first two columns use artificial, fully observed 31-day histories, not real
CVEs with measured futures. The last column uses the actual validation histories
with the declared perturbation. A 0.01 EPSS change means one percentage point.
These measurements demonstrate weak input sensitivity, not a new accuracy result.

Mean hidden-coordinate standard deviations across validation CVEs are 0.005515,
0.015892 and 0.005637. The hidden states and linear outputs vary, but only weakly
affect the final forecasts. The minimum-head local derivatives average about
0.0088; the maximum's gap-component derivatives average about 0.00153. These
small nonzero derivatives attenuate local changes. Hidden/score units differ,
so these numbers must not be interpreted as a fraction of information retained.

All six parameter tensors changed from initialization for each seed, and all
have finite positive training data-loss gradient norms. The model is not still
at initialization, nor does it have an entirely zero-gradient checkpoint.
The recurrent weight norm for seed 29 falls from 4.067342 to 0.737465, but
this does not establish that regularization caused the weak sensitivity.
There is no saved optimizer replay to isolate historical causes.

| Split | Median actual Days 31-60 minimum | Median actual Days 31-60 maximum |
|---|---:|---:|
| Training | 0.00885 | 0.00885 |
| Validation | 0.00050 | 0.00054 |

Near-constant predictions stay close to a common training level, whereas typical
validation targets are much lower. This is consistent with weak input-dependent
learning, not proof of a particular optimizer bug or corrupted scores. Do not
change the split or remove difficult CVEs to improve these results.

**Decision:** keep persistence as the joint-error reference and all three seeds.
No reproducibility or derivative discrepancy was found. The ordered
residual-extrema comparison motivated here has now been completed as F04-Residual
above, with its bounds, ordering and initialization specified before training.
Targets, cohorts, loss, budget and seeds are unchanged.
Changes to regularization or loss should be separate ablations. The residual
comparison does not establish a joint-error gain over persistence.
Repeated validation inspection remains a limitation; do not open the test set.

The separate [review report](../05_results/02_evaluation/epss_day30_to_day60/minmax_gru_diagnostic_v1_review/README.md)
contains TXT/JSON findings. Reproduce it without modifying original artifacts:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_minmax_diagnostic.py
```

### DQ01: Day-30 Evidence Availability Audit

**Status: completed by the user and independently reproduced locally.**
This is a diagnostic study, not a new trained model. It checks all 197 endpoint
validation CVEs against the locked source, prepared records and residual run.
The completed audit contains 1,677 evidence records; this count includes separate
description fields, summary fields, reference/history entries and post associations.
It is not a count of verified historical texts or independent observations.

Inputs -> locked validation endpoints and raw textual evidence -> descriptive
audit -> CVE-level changes, evidence dates, tables and figures. The anchor stays
Day 30 and the endpoint stays Day 60. No training, model inference, threshold
tuning, new split or test error analysis is performed. The shared preparation
loader still checks all splits for integrity, as in previous steps.

- All validation CVEs are retained. Endpoint groups are unchanged, nonzero
  changes below 0.01, and changes of at least 0.01 in absolute EPSS units.
  The latter means at least one percentage point, not a one-percent relative rise.
- Comparisons use the latest observed input. If Day 30 is missing, its age is
  reported and the exact-Day-30 difference stays missing. Future gaps are not
  interpolated or treated as observed daily changes.
- Descriptions, original descriptions, generated summaries, URLs/history and
  posts are inspected separately. Stored dates before the anchor do **not**
  authenticate historical text. An NVD reference date is not webpage publication.
- Same-day evidence, missing timezones, repeated posts and conflicting post
  dates remain explicit. Summaries without dated source snapshots stay unverified.
- Future scores and outcome-defined groups are diagnostic only. No predictor
  feature list is authorized by this audit. No causal or predictive-benefit claim
  can follow from a few large-change examples.

Run from the repository root in `CodeBERTFusion`:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/audit_forecast_evidence.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/audit_forecast_evidence.py
```

The second command creates
`05_results/02_evaluation/epss_day30_to_day60/evidence_availability_audit/`
under `EPSS_TPG`. It writes TXT/JSON/Markdown reports, CVE and evidence tables,
platform and group-coverage tables, two figures and an artifact checksum file.
The script refuses an existing destination; use a new `--output-dir` for an
intentional rerun. It never changes the source, preparation or model runs.
Synthetic tests: `test_forecast_evidence.py` in the same analysis folder.

#### Verified Findings

Read the [original audit](../05_results/02_evaluation/epss_day30_to_day60/evidence_availability_audit/README.md)
and the [post-run interpretation](../05_results/02_evaluation/epss_day30_to_day60/evidence_availability_audit_review/README.md).
The review checks checksums, reconstructs every CVE/evidence record, compares
all CSV/JSON tables and narrative, and independently calculates endpoint changes
from the raw date-keyed scores. The two figures were also visually inspected.
Original audit files remain unchanged; the sibling review has TXT/JSON/Markdown.

| Finding | Verified result | Meaning |
|---|---|---|
| Endpoint changes | 162 unchanged, 32 smaller, 3 at least one percentage point | Equal endpoints do not prove a flat intervening curve. |
| Error concentration | Three large-change CVEs account for 97.79% of total absolute persistence error | This is post-run diagnosis, not a reason to exclude difficult cases. |
| Posts for those three CVEs | All eight stored dates are after Day 60 | Not eligible at Day 30 under these dates; missing earlier discussion is still possible. |
| Earlier posts | 76 associations across 50 CVEs before Day 30; two on Day 30; 241 later | All 319 timestamps omit timezone. Exact deduplication leaves 316 associations. |
| Reference history | 469 of 570 entries before Day 30, across 195 CVEs | NVD link-addition evidence, not proof of historical page content. |
| Descriptions | Present for all 197 CVEs; all saved NVD updates later than Day 30 | Does not prove descriptions changed; historical versions remain unresolved. |
| Summaries | 196 nonempty fields out of 394, across 172 CVEs | No verified generation/source-snapshot date; empty fields are not usable summaries. |
| EPSS provenance | All 11,712 non-missing Day-0-to-Day-60 values, including all 197 targets, marked `EXISTING_UNVERIFIED` | Retained scores have no successful historical-snapshot match recorded in this status. This is not proof they are wrong. |

The three large changes are CVE-2024-23692 (0.00210 -> 0.95432),
CVE-2024-8963 (0.30991 -> 0.95343), and CVE-2023-7028 (0.04478 -> 0.20153).
All have exact Day-30 scores. Stored metadata reports v3 without a version
change for each; that metadata is not independent verification of score accuracy.

Two text caveats matter: the CVE-2024-8963 non-GitHub summary describes a portal
error and makes an unsupported low-exploitation conclusion. CVE-2023-7028 has
an Exploit-DB body date of 2024-01-12 but a stored date of 2024-03-14, placing
the two on different sides of the forecast window. Neither date should be
silently substituted for independently verified publication evidence.

Reproduce the review without training or changing the original audit:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_evidence.py
```

**Next step implemented as DQ02 below, awaiting the researcher run:** verify retained EPSS against
dated snapshots in a separate derived comparison. Diagnose the three large
curves first, then use a uniform policy across training and validation, not
only high-error cases. Filling missing scores did not necessarily verify scores
already present. The collector preserves mismatches and logs them; it does not
automatically correct conflicting old values. Do not overwrite locked data.
If verification changes scores, version and rebuild the prepared bundle before
new experiments. Keep test performance closed. Historical TPG/SMP inputs remain
conditional, and current-text association requires a separately named study.

### DQ02: Historical EPSS Snapshot Comparison

**Status: both researcher runs are completed. The diagnostic resolved all dates.
The uniform run finished with unresolved dates: see the download diagnosis below
before repeating its commands. A finished process is not complete verification.**
Script: `03_analysis/01_revision_dataset/verify_forecast_snapshots.py`.

Stored EPSS -> validated dated archive comparison -> one status and both values
per CVE/day. This is a data-quality study, not a trained experiment. It reuses
the repaired collector's archive parser and optional downloader. It validates
cache URL, requested date, payload hash, successful HTTP status, archive date
header, model header and score rows. The configured release schedule is recorded,
not silently revised. A missing date header remains unresolved.

| Scope | CVEs | CVE/day comparisons | Required dates | Local archive/manifest pairs present | Missing pairs |
|---|---:|---:|---:|---:|---:|
| `diagnostic` | 3 | 183 | 183 | 142 | 41 |
| `train-validation` | 1,220 | 74,420 | 1,384 | 620 | 764 |

These are preflight file-presence counts, **not** validated archive counts.
The diagnostic CVEs are the three large-change examples selected after model
evaluation. They cannot establish full-dataset quality. The uniform scope covers
all 1,023 training and 197 validation endpoint-eligible CVEs, Days 0-60 inclusive,
including missing stored scores. Test CVEs are excluded from comparisons; the
shared preparation loader still performs its existing all-split integrity checks.

Run from the repository root in `CodeBERTFusion`. Start offline:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/verify_forecast_snapshots.py --scope diagnostic
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/verify_forecast_snapshots.py --scope train-validation
```

Reports go to sibling folders under
`EPSS_TPG/05_results/02_evaluation/epss_day30_to_day60/`:
`epss_snapshot_verification_diagnostic/` and
`epss_snapshot_verification_train-validation/`.

To explicitly permit missing/invalid archive downloads after an offline run:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/verify_forecast_snapshots.py --scope diagnostic --resume --download-missing
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/verify_forecast_snapshots.py --scope train-validation --resume --download-missing
```

These commands download full daily archives, not only the selected CVEs.
The uniform run may require substantial time and disk space. Existing archives
remain read-only; new downloads go into the selected output's
`downloaded_snapshots/` directory. No network request is made without
`--download-missing`. `--check-only` only validates inputs and counts file pairs.
Use `--resume` after interruption too: it checks source/code/scope identity and
checkpoint hashes, retains completed valid dates, and retries unresolved dates.
Do not use one scope's folder for another. A fresh run refuses an existing output.

Outputs include `analysis.txt/json`, `README.md`, all comparisons and per-CVE
summaries in CSV/JSON, disagreements, unresolved records, `snapshots.json` with
provenance, per-date checkpoints, and `verification_lock.json`. The state file
distinguishes `running`, `complete`, and `completed_with_unresolved_dates`.
An interrupted resume can leave earlier aggregate reports: trust a finished
state with matching output checksums, not an old report left beside new checkpoints.

| Comparison | Meaning |
|---|---|
| `MATCH` | Both scores exist and are numerically equal; no tolerance is applied. |
| `MISMATCH` | Both exist but disagree; no automatic replacement. |
| `MISSING_STORED_AVAILABLE` | Stored score missing, valid archive contains a score. |
| `STORED_PRESENT_CVE_ABSENT` | Stored score exists, valid archive has no such CVE. |
| `BOTH_ABSENT` | Stored score missing and CVE absent from valid archive; not zero. |
| `UNRESOLVED_SNAPSHOT` | Missing, failed or invalid archive; no match/absence conclusion. |

Report inputs (Days 0-30), intermediate future days (31-59), and Day-60 targets
separately for each split. Difference is archive score minus stored score.
No raw values, dataset statuses, prepared labels, splits or models are changed.
Archive hashes/headers do not establish historical intraday availability or
signed authenticity. A complete run can still contain score disagreements.
Review the comparison before approving a correction policy; accepted corrections
would require a separately versioned derived dataset and rebuilt preparation.
Tests: `test_forecast_snapshots.py`, entirely synthetic with no real requests.

#### Completed Diagnostic and Uniform Handoff

The [diagnostic review](../05_results/02_evaluation/epss_day30_to_day60/epss_snapshot_verification_diagnostic_review/README.md)
rechecks output/checkpoint hashes, all archive/manifest hashes, date/model headers,
selected scores from the actual archives, independent comparison labels and
differences, CSV/JSON agreement and all summary/report content. It makes no
network requests. The original verifier and completed artifacts were preserved.

| CVE | Days checked | Stored scores matching archives | Missing in both sources | Disagreements | Unresolved dates |
|---|---:|---:|---:|---:|---:|
| CVE-2023-7028 | 61 | 56 | 5 | 0 | 0 |
| CVE-2024-23692 | 61 | 60 | 1 | 0 | 0 |
| CVE-2024-8963 | 61 | 60 | 1 | 0 | 0 |
| Total | 183 | 176 | 7 | 0 | 0 |

The seven absent entries are CVE-2023-7028 on 2024-01-12 through 2024-01-16,
CVE-2024-23692 on 2024-05-31, and CVE-2024-8963 on 2024-09-19. They are absent
from valid dated archives, not failed downloads or zero scores. Keep them null.
All three Day-60 targets match. The large EPSS jumps exist in the retrieved
archives too; this comparison does not establish their cause or validate all
other CVEs. No numerical correction is indicated for the diagnostic scope.

The earlier `EXISTING_UNVERIFIED` finding described the source status fields.
Those fields remain unchanged, but the 176 diagnostic scores now have separate
archive-verification evidence. Do not keep describing these checked scores as
having no verification evidence, or generalize their matches to the entire cohort.

New script: `03_analysis/01_revision_dataset/review_forecast_snapshots.py`.
It writes TXT/JSON/Markdown in a sibling review folder, plus
`uniform_verification_plan.json` fixing the next scope to all 1,023 training and
197 validation CVEs. The full-scope engine was already implemented; this handoff
adds a reproducible review and explicit cohort plan without modifying its code.
The plan is not a completed verification or authorization to repair data/train.

Start the full comparison from the repository root in `CodeBERTFusion`:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/verify_forecast_snapshots.py --scope train-validation --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/verify_forecast_snapshots.py --scope train-validation --download-missing
```

After interruption or an existing full-scope run, add `--resume` to the second
command. For an offline-only start, omit `--download-missing`; unresolved dates
will remain explicit. Downloads may be substantial. The old cache has 620 of
1,384 required pairs; 764 pairs are missing there. Diagnostic-only downloads
remain separate and are not automatically merged into the old cache.

When finished, the same offline review can check the full scope:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_snapshots.py --run-dir EPSS_TPG/05_results/02_evaluation/epss_day30_to_day60/epss_snapshot_verification_train-validation
```

Tests in `test_forecast_snapshot_review.py` cover independent numeric comparisons,
all-development-CVE selection, locked-artifact tampering and incomplete runs.
Do not move to data correction or new model training until the uniform results
are reviewed. Test score comparisons and performance evaluation remain closed.

#### Uniform Run: Downloading Versus Verification

The researcher completed the 1,220-CVE run. Its original strict results are
69,676 exact score matches, 1,184 CVE/day entries absent in both sources, and
3,560 unresolved CVE/day entries. No compared score disagreed. All 197 validation
endpoints match; 973 training endpoints match and 50 are unresolved under the
blanket date-header rule. Fully resolved windows: 923 CVEs; at least one unresolved
day: 297 CVEs. These counts do not certify the unresolved records.

| Archive-date outcome | Dates | Explanation |
|---|---:|---|
| Accepted by original strict policy | 1,060 | 599 newly downloaded files plus 461 reused cache files. |
| Downloaded legacy files rejected for missing header | 289 | EPSS v1 files have no internal date/model comment; the rule is incompatible with their documented format. |
| Requests before archives started | 26 | 2021-03-19 through 2021-04-13; the verifier should not request them. |
| In-range requests returning HTTP 403 | 9 | Unresolved retrievals, not proof of CVE absence or archive nonexistence. |
| Total requested dates | 1,384 | The four preceding outcomes do not overlap. |

There are 888 downloaded archive/manifest pairs, comprising 599 accepted modern
files and 289 legacy files. Therefore downloading largely succeeded: `False`
does not mean all files failed to transfer. FIRST explicitly documents archives
starting on 2021-04-14 and no comment header in v1; date/model comments begin
with v2 on 2022-02-04. [Official data documentation](https://www.first.org/epss/data).

Two verifier limitations are confirmed: it requests dates before the first
archive and then misclassifies the expected lack of historical coverage as a
generic download error; it rejects documented legacy files using a blanket
date-header requirement. Do not simply disable modern-header checks. Legacy
date provenance must remain explicitly based on the dated URL and manifest.

Unresolved CVE/day entries separate into 77 before archive availability, 3,148
under the legacy-header rule, and 335 on the nine in-range HTTP-403 dates.
All 77 pre-start entries and all 335 in-range HTTP-failure entries have null
stored scores. The legacy group contains 3,032 present scores and 116 nulls.
Groups can share CVEs; do not add their CVE counts as independent populations.

The separate legacy payload inspection found all 3,032 present scores match and
all 116 null entries correspond to CVEs absent from their downloaded files.
This includes the 50 training endpoints held back by the blanket header rule.
Together with modern-header comparisons, all 72,708 stored non-null values in
the scope match the retrieved archives. Preserve the distinction: the legacy
dates rely on URLs/manifests, not internal date headers. The original locked run
still reports 3,560 unresolved entries; no statuses or score values were changed.
The nine in-range HTTP failures affect 335 null entries, not mismatched scores.

The nine in-range HTTP-403 dates are 2021-04-22 through 2021-04-26,
2021-06-07, 2021-06-18, 2022-07-14 and 2024-12-01. Their server-side cause has
not been established. FIRST documents an official historical archive repository;
check that documented source selectively rather than inferring scores or
redownloading every archive.

Read the [download diagnosis](../05_results/02_evaluation/epss_day30_to_day60/epss_snapshot_verification_train-validation_download_audit/README.md)
and [original-policy review](../05_results/02_evaluation/epss_day30_to_day60/epss_snapshot_verification_train-validation_review/README.md).
`audit_forecast_downloads.py` checks every downloaded payload checksum and
reparses all rejected legacy payloads. Its separate CSV/JSON legacy comparisons
do not alter the locked original statuses. `review_forecast_snapshots.py` reparses
accepted archives and reconstructs the original comparisons and tables.

The full offline review has now completed successfully: all 1,060 accepted
archives were reparsed, and the reconstructed CSV/JSON tables, summaries and
report text agree with the saved run. Source, preparation, code, checkpoint,
archive and manifest hashes also passed their checks. The separate download
audit verified all 888 downloaded archive/manifest pairs and parsed the 289
legacy archives. All 26 focused verifier/review/audit tests passed. These checks
used no network, changed no source scores and performed no training or test
comparison. Passing the review confirms that the saved results are reproducible;
it does not turn unresolved dates into resolved ones.

Repeat the download diagnosis offline, without any new transfer or score change:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/audit_forecast_downloads.py
```

**Next action:** run the versioned pre-EPSS/legacy-aware verifier in DQ03 below,
which limits network attempts to the nine in-range HTTP failures. Preserve this completed run, its source
hash and its checkpoints. Do not recommend blindly repeating the unchanged
`--resume --download-missing` command: it cannot solve the pre-start dates or
the legacy-format policy. No correction or new training has been authorized by
this analysis. Tests for the diagnosis are in `test_forecast_download_audit.py`.

### DQ03: Legacy-Aware Verification and Targeted Retries

**Status: offline verification and opt-in retry completed and reviewed; nine dates still return HTTP 403.**
This is a data check, not a trained experiment or a dataset replacement.
Script: `03_analysis/01_revision_dataset/verify_forecast_snapshots_v2.py`.
The original DQ02 script and completed run remain unchanged and hash-verifiable.

| Case | New behavior | Meaning |
|---|---|---|
| Date before 2021-04-14 | Skip cache access and HTTP requests; retain `BEFORE_EPSS` | No daily EPSS history was available then; missing is not zero. |
| Headerless archive during 2021-04-14 to 2022-02-03 | Validate URL, requested date, HTTP status, checksum, row count and all scores | Accept with `legacy_requested_url_and_manifest_only`; no invented internal date/model header. |
| Modern archive | Require valid internal date/model headers as well as URL/manifest checks | Modern checks are not weakened to accommodate legacy files. |
| One of the original nine in-range HTTP-403 dates | Permit a targeted retry only with `--download-missing` | A retry may still fail; no score is inferred from a failed request. |
| Missing stored score found in a valid archive | Write a recovery candidate | No automatic insertion, replacement or dataset publication. |
| Stored score before EPSS history | Flag `STORED_SCORE_BEFORE_EPSS` | An anomaly to investigate, not an accepted match. |

The new run reuses both DQ02's downloaded files and the older cache, read-only.
Only its own `downloaded_snapshots/` can receive downloads. Its independent
output folder is `05_results/02_evaluation/epss_day30_to_day60/epss_snapshot_verification_train-validation_v2/`.
It preserves the same 1,023 training and 197 validation CVEs and Days 0-60.
Test-score comparison remains disabled. Source values, labels, split decisions
and completed model runs are not edited.

The researcher has completed the following command sequence. It is retained
for reproducibility, not as a recommendation to repeat the failing requests:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/verify_forecast_snapshots_v2.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/verify_forecast_snapshots_v2.py
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/verify_forecast_snapshots_v2.py --resume --download-missing
```

The first command validates the preparation and original-run hashes and displays
the allowed retry dates; it does not validate every archive. The second performs
the complete offline verification and creates the new run. The third rechecks
checkpoint/archive hashes, reuses accepted results and permits requests only for
the nine dates that still lack valid archives. It uses the collector's existing
source URL; it does not silently switch to an alternative mirror. If these
requests still return HTTP 403, those dates remain unresolved for investigation.
After interruption, rerun the second command with `--resume` for offline-only
continuation, or rerun the third for the opt-in network continuation.

Outputs include `comparisons.csv/json`, `recovery_candidates.csv/json`,
`unresolved.csv/json`, `snapshots.json`, `analysis.txt/json`, `README.md`, date
checkpoints and `verification_lock.json`. Empty recovery tables retain column
headers. Recovery candidates are proposals for a later versioned dataset, not
applied corrections. `all_in_range_dates_resolved` excludes pre-EPSS dates;
it does not mean every CVE has a score on every day.

Verification during implementation: 37 focused tests passed; the real-data
preflight confirms 74,420 CVE/day entries, 1,384 dates, 26 pre-EPSS dates and
nine permitted retry dates. Real archives from 2021-04-14 and 2022-02-04 passed
their respective legacy and modern rules. All 1,060 originally accepted archive
manifests contain row counts required by the new checks. These were pre-run
implementation checks; completed-run findings follow below.

#### DQ03 Results Before and After the Download Retry

The initially reviewed v2 invocation was offline, with
`download_permission_this_invocation=false` and no new HTTP attempt records.
The researcher subsequently ran `--resume --download-missing`. The latest
saved report now has `download_permission_this_invocation=true`; each of the
nine dates has one recorded HTTP-403 response, checked on 2026-10-16 between
10:02:53 and 10:02:58 UTC. Their date-level status is now `DOWNLOAD_ERROR`,
while their CVE/day outcome remains `UNRESOLVED_SNAPSHOT`. There are no new
archive payloads in the v2 download cache and no recovery candidates.
The counts below are unchanged by the retry.

| Outcome for one CVE on one date | Entries | Interpretation |
|---|---:|---|
| Existing score matches archive | 72,708 | Includes 3,032 legacy matches previously excluded by the strict header rule. |
| Missing in both dataset and valid archive | 1,300 | Includes 116 legacy cases; retain missing values, not zero. |
| Date before daily EPSS history | 77 | No request needed; not an unresolved server failure. |
| Archive still unavailable for verification | 335 | Nine dates; all stored values are missing. |
| Total | 74,420 | Same 1,220 CVEs and Days 0-60 as DQ02. |

There are 1,349 accepted archive dates, 26 pre-EPSS dates and nine unresolved
dates. No score disagreement or newly recoverable score was found. All 1,023
training and 197 validation Day-60 targets match. The nine unresolved dates
affect 271 distinct CVEs; 949 CVEs have no unresolved archive date, which does
not mean every daily score exists. Only 117 CVEs have all 61 stored scores;
1,103 have partial histories and none have entirely missing histories in this
endpoint-eligible subset. These are 61-day development-window counts, not
91-day coverage figures for the complete raw dataset.

| Unresolved archive date | CVEs affected on that date |
|---|---:|
| 2021-04-22 | 9 |
| 2021-04-23 | 9 |
| 2021-04-24 | 9 |
| 2021-04-25 | 9 |
| 2021-04-26 | 9 |
| 2021-06-07 | 25 |
| 2021-06-18 | 28 |
| 2022-07-14 | 209 |
| 2024-12-01 | 28 |

A CVE can occur on several dates. These rows sum to 335 CVE/day entries,
not 335 distinct CVEs. None of these unresolved entries is a Day-60 target.

The independent review checked all 1,349 archive/manifest checksums, pinned
source/code/preparation hashes, checkpoint hashes, comparisons reconstructed
from the preparation and checkpoints, original-run transitions, CSV/JSON
agreement and report text. It did not decompress/reparse archive CSVs a second
time; that was done by the completed v2 run. The review was refreshed after the
download-enabled invocation, and all 42 focused tests passed.
Review artifacts: [TXT report](../05_results/02_evaluation/epss_day30_to_day60/epss_snapshot_verification_train-validation_v2_review/analysis.txt),
[JSON report](../05_results/02_evaluation/epss_day30_to_day60/epss_snapshot_verification_train-validation_v2_review/analysis.json)
and per-CVE `cve_summary.csv/json` in the same review folder.

**Next action:** investigate these exact dates against the documented historical
archive repository or contact the archive provider. Do not repeatedly rerun
the unchanged failing requests. HTTP 403 records a refusal but does not prove
that the archive is absent or identify the cause. Any alternative source must
be explicitly recorded and its payload validated before use. No alternative
source was downloaded during this review.
No dataset merge is justified by the current run: recovery candidates are empty.
No raw values, splits, model results or original DQ02 statuses were changed.

Repeat the review after the next completed run:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_snapshots_v2.py
```

Before updating GitHub data, review the new report and recovery candidates.
If no new scores are found, publish verified code/report changes rather than
claiming the dataset has acquired new EPSS values. Keep the original source and
large local archive caches separate from a deliberate publication decision.

### DQ04: Frozen Missing-Data Policy and Validation Sensitivity

**Status: completed by the researcher and independently reproduced; validation only.**
The researcher accepted keeping the documented gaps rather than repeatedly
retrying the nine failing dates. DQ03's recovery investigation is now optional,
not a prerequisite for this endpoint diagnostic. No raw dataset or prepared
split has been edited, and the reserved test performance remains closed.

Policy: `04_configs/epss_missing_data_policy.json`.
Script: `03_analysis/01_revision_dataset/forecast_missingness_sensitivity.py`.

| Item | Fixed decision |
|---|---|
| Input | Existing Days 0-30 scores, observation masks, observation age and age-known masks. |
| Missing scores | Retain nulls in data; masked numeric padding is not an observed zero. No imputation. |
| Audit reasons | Retain pre-EPSS, absent-from-archive and unresolved-archive reasons in a separate table; never use them as predictors. |
| Primary target | Observed Day-60 EPSS, 30 days after the prediction anchor. Missing targets remain ineligible. |
| Main cohort | Same 1,023 training and 197 validation CVEs; no boundary changes or new exclusions. |
| Models | Saved persistence, ridge, direct GRU and residual GRU; all three fixed neural-model seeds (17, 29, 43). |
| Outputs | Frozen provenance/policy manifest, input-coverage tables/figure, matched validation metrics and per-subset CVE IDs. |
| New training/inference | None. The diagnostic reads previously saved predictions. |
| Test access | Prepared-data integrity checks remain; no test predictions, metrics or subgroup analysis. |

The three disjoint groups use only input-day availability, not future EPSS
values, prediction errors or the model that happens to look best:

| Input-history group | Observed input scores | Training CVEs (preflight) | Validation CVEs (preflight) |
|---|---:|---:|---:|
| All input scores present | 31 | 91 | 38 |
| One input score missing | 30 | 872 | 130 |
| Two or more input scores missing | 1-29 | 60 | 29 |

These are input-only counts, different from DQ03's completeness counts across
all 61 days. Each model is evaluated on the exact same IDs within a group.
An additional overlapping sensitivity subset, `no_unresolved_input_archive`,
excludes only CVEs with an unresolved archive date among Days 0-30. An unresolved
future date does not exclude a CVE from that subset. This is a retrospective
audit restriction, not a new operational prediction rule. Histories can remain
incomplete even when every relevant archive date has been verified.

Metrics: MAE, RMSE, mean signed error, Spearman correlation, equal-related-group
MAE, paired MAE difference from persistence, and persistence-relative MAE skill.
All seed-level metrics are retained without choosing a winner or constructing
an undeclared ensemble. Undefined quantities remain null, including correlation
for constant/tiny samples and relative skill when persistence MAE is zero.
Empty groups are retained, and groups below 20 CVEs are explicitly flagged.

This is a post-hoc sensitivity diagnostic on previously inspected validation
data, not an untouched evaluation or evidence that removing records improves
training. Coverage can correlate with publication dates, model versions and
archive availability. Report the unchanged full-cohort results as primary;
do not optimize missingness thresholds, split dates or model choice from these
subgroup results. There is no claim that missingness causes forecast errors.

Run from the repository root in `CodeBERTFusion`, one command at a time:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/forecast_missingness_sensitivity.py --check-only
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/forecast_missingness_sensitivity.py
```

The first checks frozen hashes, verification artifacts, target eligibility,
input masks, cohort membership and saved predictions, but calculates no subgroup
metrics and writes no reports. The second produces the full diagnostic in
`05_results/02_evaluation/epss_day30_to_day60/missingness_sensitivity_v1/`:

| File | Purpose |
|---|---|
| `freeze_manifest.json` | Hash-pins source, preparation, verification, policy, code, model runs and development cohort IDs. Does not rewrite the source or create a Git release. |
| `coverage.csv/json` | Per-CVE input completeness, missingness reasons, observation age and label coverage. Audit metadata, not an expanded model feature list. |
| `coverage.png` | Number of CVEs with zero, one or multiple missing input scores; separate training/validation panels. |
| `missing_score_reasons.csv/json` | Existing missing CVE/day entries and their verification reasons; values stay null. |
| `metrics.csv/json` | Matched validation metrics for every fixed subset and saved model/seed. |
| `analysis.txt/json`, `README.md` | Findings, definitions, subset IDs and limitations. |
| `analysis_lock.json` | Checksums of generated artifacts. |

Existing output folders are refused; use a new `--output-dir` for a deliberate
repeat rather than overwriting findings. No network access or training is
performed. Implementation preflight and 53 focused tests passed initially,
including synthetic end-to-end artifact checks. After the researcher run,
the post-run review passed and the expanded suite has 56 passing tests.

This freeze supports the current endpoint task only. Min/max targets still
require complete future windows; trajectories need separate masked-loss rules.
Historical description, summary, SMP and catalogue-label gates remain in force.
After DQ04, review the findings before declaring the next model/task experiment;
do not rerun all existing models merely because the missingness policy is documented.

#### DQ04 Verified Results

The review reproduced coverage, missing-score reasons, frozen provenance,
all 40 subset/model metric rows, subset membership, CSV/JSON and report text.
Independent decimal-based calculations agree on MAE, RMSE and equal-related-group
MAE. The original coverage figure was visually checked; its checksum was verified,
not reproduced pixel-for-pixel. No fitting, new predictions or test scoring occurred.

| Validation input group | CVEs | Persistence MAE | Trained models with lower MAE |
|---|---:|---:|---|
| All 31 input scores present | 38 | 0.00015000 | None |
| One input score missing | 130 | 0.01246200 | None |
| Two or more input scores missing | 29 | 0.00573448 | None |
| All validation CVEs (primary) | 197 | 0.00909675 | None |
| No unresolved input-archive date (overlapping sensitivity subset) | 196 | 0.00914316 | None |

Persistence means predicting the most recent observed input EPSS again at Day 60.
The trained models are ridge and every saved direct/residual GRU seed. On all
197 CVEs, ridge MAE is 0.01075409; direct GRU seed MAEs range from 0.01413642
to 0.03298903, and residual seed MAEs range from 0.00909911 to 0.00914799.
These ranges report all seeds, not a selected winner or ensemble. Some models
have slightly lower RMSE; the MAE finding is not dominance on every metric.

Of the 130 validation CVEs missing exactly one input score, 129 are missing
Day 0 (the stored NVD publication date), and one is missing Day 22. This is not
the same pattern as losing many scores near the Day-30 prediction anchor.
The three largest persistence errors remain:

| CVE | Missing input days | Latest input EPSS | Actual Day-60 EPSS | Absolute error |
|---|---|---:|---:|---:|
| CVE-2024-23692 | Day 0 | 0.00210 | 0.95432 | 0.95222 |
| CVE-2024-8963 | Day 0 | 0.30991 | 0.95343 | 0.64352 |
| CVE-2023-7028 | Days 0-4 | 0.04478 | 0.20153 | 0.15675 |

Together these three CVEs account for 97.79% of total absolute persistence error.
This explains concentration, not causality or a reason to remove those CVEs.
The complete-history group contains different vulnerabilities; its lower error
does not demonstrate that filling missing values would improve forecasts.
Performance is not monotonically worse with the number of missing inputs:
the one-missing-score group has higher persistence MAE than the multiple-missing
group, partly reflecting the large errors above.

The unresolved-input sensitivity subset removes only CVE-2024-48352. Its
persistence error is exactly zero. Removing it raises persistence MAE from
0.00909675 to 0.00914316 because the denominator decreases, not because any
remaining prediction changes. No model gains an MAE advantage over persistence
in the restricted subset. This is a validation subset comparison, not a test
of retraining without affected cases or of all missing archive dates.

**Decision:** retain the unchanged primary cohort, nulls, masks and observation
age. Do not tune exclusions to obtain smaller errors. Missingness remains a
limitation, but this diagnostic does not support treating it as the sole reason
the trained models underperform: none improves MAE even within complete inputs.
The analysis cannot isolate effects of missing training inputs or establish
that missingness has no effect. Keep persistence as the reference and declare
the next task/model separately. Historical text/SMP availability must still be
resolved before a temporally defensible TPG fusion experiment.

Review outputs: [TXT findings](../05_results/02_evaluation/epss_day30_to_day60/missingness_sensitivity_v1_review/analysis.txt)
and [JSON findings](../05_results/02_evaluation/epss_day30_to_day60/missingness_sensitivity_v1_review/analysis.json).
The locked original diagnostic remains unchanged. Reproduce the review without
training or creating another sensitivity run:

```bash
python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/review_forecast_missingness.py
```

For every completed run, record the ID, source/preparation hash, exact settings,
training/validation CVEs, checkpoint rule, seed(s), saved artifacts, metrics,
limitations and test-access status. Record unsuccessful runs too. Do not
overwrite one model's result folder with another experiment or erase earlier
results to make a comparison look better. Each result directory contains its
own TXT/JSON report and per-CVE predictions; this file stays the central index.

Sources for the catalogue: [original 28-experiment feasibility matrix](../05_results/03_dataset_analysis/01_revision_dataset/12_training_feasibility/experiment_matrix.csv),
[target plan](../05_results/03_dataset_analysis/01_revision_dataset/13_epss_targets/TARGET_DEFINITION_AND_TRAINING_PLAN.md),
[Step 1 readiness](../05_results/03_dataset_analysis/01_revision_dataset/13_epss_targets/STEP1_READINESS.md),
and the completed-run and archived evidence linked above. Historical proposal
counts are superseded by the prepared task cohorts where this register reports
current counts; raw source files and old reports have not been silently changed.

## CC03: Frozen-Architecture Catalogue Classifier Implementation

The separate `catalogue_training_v1` package preserves the corrected combined
TPG, SecureBERT 2.0, typed-edge GNN and fusion, but fits binary verified catalogue
membership with BCE and selects by inner-selection AP. Five seeds cover six
learned arms per dataset: EPSS/structured logistic and tree, pooled D/DM, and
TPG D/DM. Prevalence and raw EPSS are references. Thirty fits per dataset are
specified; no full classifier training was run during implementation.

CC02 roles and disclosure screening remain frozen. Cache reuse requires exact
screened text and processing identities; missing or changed graphs use the
existing extractor. Source masks, fitting-only numerical preprocessing, EPSS
missingness, best-only checkpoints, deterministic resume, and selection-only
F1 thresholds are enforced. All classifiers and cutoffs are locked before
inner assessment. Neither official validation nor test is adapted. The dated
EPSS input is explicitly pending, not silently replaced by an older score.

The synthetic suite verifies encoder parity, node/edge/subtype gradient paths,
checkpoint replay, interrupted resume and the release/assessment workflow.
A two-object real cache probe verifies both rebuilding and exact reuse, frozen
embeddings and TOKEN means. These checks are not scientific classifier results.
Commands and architecture: [CC03 guide](05_studies/08_catalogue_classification/02_TRAINING_COMMANDS.md).
