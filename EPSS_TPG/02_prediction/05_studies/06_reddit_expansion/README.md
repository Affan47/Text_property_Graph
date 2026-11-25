# Reddit Expansion: Pre-modeling Gates

## Active Work Supersedes the Historical Runs Below

Use the [combined-text SecureBERT 2.0 guide](../../../00_documentation/02_tpg/11_COMBINED_TEXT_FRESH_START.md).
The user requested a fresh start with D/M/U/DM/DU/DMU concatenation before TPG
extraction, removing old graph caches/results/checkpoints. Verified inputs now
live in `04_data/03_reddit_collection/04_combined_text_inputs_v1/`.
Old results linked below are historical and retired, not current runs.

## Current: Full RD02 Completed; Webpage Summaries Being Added

The full RD02 cache and 21-predictor development comparison have completed.
Read the [saved-result review](../../../05_results/07_reddit_studies/02_rd02_primary_v1/full_review_v1/REPORT.txt).
It retains 24,213 train / 972 validation CVEs and does not score test CVEs.
The older gate sections below are preserved as historical records.

The summary-inclusive extension now has a versioned preparation adapter and
[runbook](../../03_analysis/06_reddit_dataset/SUMMARY_EXTENSION.txt).
Exact CVE/reference matches recover 177 G and 538 N training summaries before
screening (177 G / 533 N afterward), but zero validation summaries. These
counts are for the fixed new cohort, not all 5,968 older CVEs. Three full saved
sources and six regenerated sample files were inventoried without choosing a
generator from outcome performance. The missing-source collection queue is
development-only; no website or LLM has been contacted.

Proposed views are D, DG, DN, DGN, DM_prose, DM_proseG, DM_proseN and DM_proseGN
for pooled text and v6 TPG, with unchanged structured features. Completed D
and DM_prose references remain fixed. Source review, summary coverage, a new
cache adapter and smoke parity must precede the proposed 36 additional fits.
Do not run summary training with zero validation G/N coverage. No temporal-text
forecasting, test adaptation or new training is authorized by this preparation.

## Latest: RD02 Export, Graph Gate and Smoke Verified

The user authorized bounded smoke execution, then explicitly selected
**smoke-only graph caching while durable storage is arranged**. No full
training or test evaluation was performed. RD04's existing preflight and
smoke artifacts reproduced unchanged; they were not retrained.

Read the [combined gate report](../../../05_results/07_reddit_studies/02_rd02_primary_v1/gate_review_v1/REPORT.txt),
[machine-readable findings and hashes](../../../05_results/07_reddit_studies/02_rd02_primary_v1/gate_review_v1/findings.json),
and [all 21 smoke metric rows](../../../05_results/07_reddit_studies/02_rd02_primary_v1/gate_review_v1/rd02_smoke_metrics.csv).

| RD02 artifact | Scope | Status |
| --- | --- | --- |
| Full predictor/label export and structured vectors | 24,213 train / 972 validation; 53 train-fitted structured columns | Saved and reproduced |
| Training-only graph gate | 32 CVEs; 65 documents; 71 chunks; 6,209 nodes; 17,858 edges | Saved and verified |
| Smoke graph cache | 12 train / 6 validation; 38 documents; 47 chunks; 6,499 nodes; 19,426 edges | Saved and verified |
| Matched smoke fits | 3 deterministic controls plus 6 neural variants x 3 seeds | All 21 predictors replayed |
| Full graph cache | 25,185 development CVEs; 53,052 documents; 66,800 planned chunks | Not built; waiting for durable storage |

The full export is `05_results/07_reddit_studies/02_rd02_primary_v1/export/full`.
It separates `predictor_inputs.json.gz`, `labels.json`, and
`audit_not_model_input.json`. `structured_state.json` records training-only
category vocabularies, medians, scaling and column names;
`structured_vectors.json` stores the resulting per-CVE vectors. The predictor
allowlist excludes EPSS/catalogue fields, their missingness, CVE IDs and dates.
Reference lists are deduplicated within each CVE, not across CVEs. Current
CVSS/text/reference counts remain explicitly retrospective inputs.

Actual graph artifacts are under `01_tpg/05_workspace/rd02_primary_v1/gate`
and `.../smoke`: `cve_index.json` joins documents to `graphs/<chunk-hash>.json`
and `tensors/<chunk-hash>.pt`. Receipts pin exact text, offsets, frontend/pass
code, encoder identity and file hashes. The v6 frontend uses the version-shaped
predicate guard; corrections through v6 and the 32-channel subtype vocabulary
are applied. SEVERITY is index 31. The training-only graph gate contains one
SEVERITY entity; the smoke sample contains none. A valid cache is not a claim
of perfect semantic extraction. Only SEC_HAS_VERSION occurs among security
edge types in these small samples (22 in the gate, 7 in smoke).

Pooled and graph variants share frozen SecBERT embeddings. Documents remain
separate; chunks are averaged within documents, then documents within D or M.
`M_all` and `M_prose` select from the same cached attachments. Absent prose M
sets its mask to zero without removing the CVE. Two typed-edge GNN layers
(width 32, no COREF/dropout) precede node-mean readout in the graph arms.
Masked source attention and a structured branch feed a scalar sigmoid
regression head. There is no EPSS history predictor in RD02.

Smoke settings: 12 training / 6 validation CVEs, two epochs, batch size two,
seeds 17/29/43, AdamW learning rate 0.001 and weight decay 0.0001. Sample
selection used IDs and source/overlap coverage, not target values. Same-weight
D versus masked-M DM predictions match exactly for both representations and
all seeds. Deterministic chunk-gradient replay matches ordinary backpropagation
in a regression test. The complete local test suite passes 75 tests.

These are integration results only. Smoke validation includes four CVEs
without training-description overlap and two with overlap, not the full
943/29 partition. All six are local single-version examples; empty crossing
slices have null metrics. The saved whole-91-day version flag is separately
reported, not substituted for the local publication-to-Day-60 release check.
Training median smoke MAE is 0.00833833, versus 0.00851083 for the mean and
0.00867854 for ridge. The two-epoch neural fits are worse on this tiny sample;
no model is promoted and no settings are retuned from smoke performance.

An initial structured-export bug changed column order after JSON reload.
Explicit sorted ordering plus a JSON round-trip regression check fixed it
before any training. The untrained first exports are preserved under
`export_before_column_order_fix`; they are not usable inputs. Initial graph
chunks were reused only after exact processing/text identity checks. Completed
historical datasets, role locks and experiment artifacts remain unchanged.

### Commands for the Completed Gates

CPU only, from `/home/ayounas/Text_property_Graph`, with `CodeBERTFusion` active.
Run each command separately. These commands verify, not retrain:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset -p 'test_*.py'
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/run_rd02_primary.py --stage export --scale full --verify-only
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/run_rd02_primary.py --stage cache --scale gate --verify-only
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/run_rd02_primary.py --stage cache --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/run_rd02_primary.py --stage smoke --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/run_rd04_daily.py --stage preflight --scale full --verify-only
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/run_rd04_daily.py --stage smoke --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/review_reddit_smoke_gates.py --verify-only
```

For a fresh checkout, export gate/smoke/full without `--verify-only`, build
gate then smoke caches, and run smoke once. Completed outputs must use
verification. An interrupted cache supports `--resume`; it does not silently
overwrite a completed cache. `--cache-dir` supports a user-selected durable
filesystem; `--reuse-cache` checks exact chunk identities before copying.
Do not start full caching until the durable location is supplied. The cache
space guard currently budgets about 43.55 GiB, an estimate rather than a
measured full-cache size. Full-training/test modes are deliberately absent.

## Earlier Gate: RD04 Smoke Passed and RD02 Overlap Review Accepted

The user explicitly authorized the bounded RD04 smoke fits for this step.
They completed on CPU, followed by fresh-process checkpoint/prediction replay.
**Full training remains for the user to run after a separate full-run stage is
implemented and approved.** No test CVEs were adapted or scored. No RD02 graph
cache or training was started.

### RD04 Results and Locations

The dedicated CLI is `03_analysis/06_reddit_dataset/run_rd04_daily.py` under
`02_prediction`. It is not the old CF01 CLI. The package is
`01_package/epss/reddit_forecasting`; its frozen configuration is
`04_configs/05_reddit_studies/RD04_stage1_v1.json`.

Read [the full preflight](../../../05_results/07_reddit_studies/01_rd04_daily_v1/preflight/full/REPORT.txt)
and [the smoke report](../../../05_results/07_reddit_studies/01_rd04_daily_v1/smoke/REPORT.txt).
JSON summaries, per-CVE metrics, canonical prediction CSVs, raw and clipped
predictions, epoch-zero metrics, histories, checkpoints and paired bootstrap
intervals are saved alongside the reports. Completed bundles have SHA-256 locks.

| Verified quantity | Training | Validation |
| --- | ---: | ---: |
| Development CVEs | 24,548 | 1,106 |
| Eligible anchor/target pairs | 96,289,074 | 3,692,438 |
| Pairs crossing a release between anchor and target | 5,448,048 | 0 |
| Pairs crossing a release between first observed input and target | 10,004,810 | 0 |
| CVEs eligible for Day 30 to Day 60 | 24,179 | 972 |

The two release rows describe different spans and must not be added together.
Validation stays within 2024, so its absence of release crossings is expected
under the frozen schedule. Empty diagnostic slices report null metrics, not
zero error. Release dates never enter the encoder. Training target dates stop
at 2023-12-31; validation anchors and targets remain in 2024. Out-of-role scores
are nulled in the derived adapter only; original scores are unchanged.

The smoke sample was selected by CVE IDs alone: 12 training CVEs / 20,284
pairs and 6 validation CVEs / 10,274 pairs. Only three smoke validation CVEs
have the primary Day-30-to-Day-60 pair. Persistence and the training-fitted
lead correction each have primary MAE 0 on those three cases. All three GRUs
select epoch 2, clip all validation predictions to zero, and have primary
clipped MAE 0.000716667. Their raw primary MAEs differ. This is an integration
gate, not evidence of useful forecasts or grounds to retune settings from
the smoke sample. Epoch zero matches persistence exactly for every seed.

### RD02 Review Decision

Read [the overlap report](../../../05_results/03_dataset_analysis/02_reddit_dataset/10_rd02_overlap_review_v1/REPORT.txt)
and [supplemental source approval](../../../05_results/03_dataset_analysis/02_reddit_dataset/10_rd02_overlap_review_v1/source_approval.json).
The original blocked manifest is retained unchanged; the approval links to
its exact hash and changes the source-review disposition only.

- The 23 normalized-exact groups contain 22 short product/vulnerability titles
  and one longer shared mechanism description. These are repeated inputs,
  not proof that the CVE records describe the same vulnerability.
- The one near pair links train and test, has Jaccard 0.9010989, and differs
  by one inserted word token under the audit tokenizer. Held-out wording is
  not published. This remains a shared-input risk, not proof of independent
  root causes.
- RD02-specific affected counts are 104 train, 30 validation and 10 test
  CVEs across any role. The older audit flags pooled RD02/RD04 signatures;
  those broader flags are not substituted for task-specific counts here.
- The predefined validation sensitivity partition is **29 with a description
  overlap with training / 943 without detected overlap with training**. One
  additional validation CVE shares a signature with test only and is not put
  in the training-overlap partition.
- Distinct CVEs, shared URLs, disclosure screening and D/M_all/M_prose masks
  are retained. No outcome file was opened for this review. Acceptance is a
  documented source-policy decision, not proof of leakage-free text.

The future RD02 adapter must validate both the old manifest and supplemental
approval. It must retain the whole primary cohort and report overlap
sensitivity without selecting a subgroup winner. Cache preparation remains
the next RD02 implementation step. RD03 and historical text fusion stay blocked.

### Reproduce These Completed Checks

CPU only, from the project root with `CodeBERTFusion` active. Run individually:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset -p 'test_*.py'
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/run_rd04_daily.py --stage preflight --scale full --verify-only
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/run_rd04_daily.py --stage preflight --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/run_rd04_daily.py --stage smoke --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/review_rd02_overlap.py --verify-only
```

For a fresh checkout without these outputs, run the same commands without
`--verify-only`, in the same order. The smoke write command performs bounded
training; verification reloads saved checkpoints without optimizer updates.
There is deliberately no full-training or test-release CLI mode yet.

## Earlier Stage: Frozen Roles, Source Gate and Development Events

At this earlier stage, all training was reserved for the user. The later,
explicit smoke-only authorization is recorded above. The commands in this
historical section were source audits, role freezes and descriptive analyses.

The accepted `retain_all` + `retain_matching_prior_roles` CVE role IDs are
frozen in [rd01_frozen_manifest.json](../../../05_results/03_dataset_analysis/02_reddit_dataset/08_rd01_source_lock_v1/rd01_frozen_manifest.json).
The manifest deliberately separates frozen role identities from source/cache
approval: RD02 was blocked for text-overlap review, and RD04 needed its new
development loader/preflight. The newer section above records their resolution.
No old CF01 command should be used as though it runs this new cohort.

| Frozen task | Train CVEs | Validation CVEs | Held-out CVEs |
| --- | ---: | ---: | ---: |
| RD02 | 24,213 | 972 | 3,977 |
| RD04 | 24,548 | 1,106 | 3,978 |

RD04 cell counts remain 96,289,074 / 3,692,438 / 16,060,201. All original
4,987 prior-test-linked CVEs stay excluded. Test release is disabled.

### Source Audit Results

Read the [source report](../../../05_results/03_dataset_analysis/02_reddit_dataset/08_rd01_source_lock_v1/REPORT.txt)
and [source coverage](../../../05_results/03_dataset_analysis/02_reddit_dataset/08_rd01_source_lock_v1/source_coverage.csv).
RD02 has 23 cross-role normalized-exact text groups and one verified lexical
near-duplicate pair at five-word-shingle Jaccard >=0.90. RD04 has 25 exact
groups and one near pair, but text is not an RD04 predictor. These are review
candidates, not proof that different CVEs are identical vulnerabilities.
No test text excerpts or numerical test targets were exported.

| RD02 role | M_all attachments after screening | M_prose attachments | CVEs with M_prose | CVEs with mask_M_prose=0 |
| --- | ---: | ---: | ---: | ---: |
| Train | 26,578 | 3,309 | 2,077 | 22,136 |
| Validation | 1,289 | 1,283 | 968 | 4 |
| Test | 4,426 | 4,425 | 3,976 | 1 |

Two test attachments were quarantined by the fixed EPSS-topic screen. Thus
the pre-screen test prose count of 4,427 becomes 4,425. Attachment counts
are not CVE counts. All descriptions pass this disclosure screen, but the
remaining cross-role wording overlaps required the now-completed review above.
Generic EPSS mentions are conservatively quarantined; they are not all
claimed to be proven numeric-score leaks. No target value drives the screen.

The registered `RD02_registered_v2.json` compares three deterministic controls
and six neural variants over three seeds: pooled and v6 32-subtype TPG models
with D, DM_all and DM_prose, each with the same structured branch. This is a
21-fit **registered specification**, not an implemented training/cache runner.

### Development-Only Event Findings

Read [the descriptive event report](../../../05_results/03_dataset_analysis/02_reddit_dataset/09_reddit_dev_events_v1/REPORT.txt),
[all statistics](../../../05_results/03_dataset_analysis/02_reddit_dataset/09_reddit_dev_events_v1/event_summary.csv),
and [counts](../../../05_results/03_dataset_analysis/02_reddit_dataset/09_reddit_dev_events_v1/event_counts.csv).
The study uses frozen RD04 train/validation only and requires the complete
window to remain within its role boundary. The previously reported all-corpus
coverage counts are not its denominator. No test score changes were analyzed.

| Days each side | Train events / CVEs | Validation events / CVEs |
| --- | ---: | ---: |
| 1 | 1,263 / 911 | 399 / 307 |
| 7 | 680 / 524 | 168 / 129 |
| 14 | 392 / 302 | 108 / 82 |
| 30 | 132 / 112 | 38 / 31 |

There are 3,180 event-window records over 1,218 CVEs. A single post can qualify
at multiple widths. Primary statistics first average events within each CVE,
then weight CVEs equally. Event-weighted statistics and first-eligible-event
sensitivity are secondary. Local release crossing and attachment type are
reported separately; slopes at +/-1 day are undefined, not invented.

Version-clean +/-7-day endpoint-change example (equal CVE weights):

| Role | CVEs | Mean change before post | Mean change after post | Median before / after |
| --- | ---: | ---: | ---: | ---: |
| Train | 521 | 0.00859517 | 0.00408845 | 0 / 0 |
| Validation | 129 | 0.00054111 | 0.01531757 | 0 / 0 |

Changes are in EPSS score units on the 0-1 scale, not percentage points or
exploitation rates. The before/after relationship differs between periods;
these results do not establish causation or justify tuning a prediction
horizon. Positive daily-step fractions, absolute changes, means, medians,
slopes, extrema and direction patterns are retained in the detailed tables.

### Hashes and Commands

[manifest_hashes.json](../../../05_results/03_dataset_analysis/02_reddit_dataset/08_rd01_source_lock_v1/manifest_hashes.json)
contains the complete SHA-256 strings for the master manifest and both task-ID
files. The source audit and event study reproduced byte-for-byte. No models,
graph caches or embeddings were built in these stages.

CPU only, one command at a time, from the repository root in `CodeBERTFusion`:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset -p 'test_*.py'
```
```bash
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/lock_rd01_sources.py --verify-only
```
```bash
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/analyze_reddit_dev_events.py --verify-only
```

Write-stage commands (without `--verify-only`) have already completed.
These earlier next steps are now completed as recorded above. RD03 and
temporal text fusion remain gated as before.

## RD01 Diagnostic Completed: Read This First

The missing-score policy has been accepted: preserve nulls and statuses, allow
partial observed histories, and never impute targets. Unrecovered archive dates
are not a blanket block on RD02 or RD04. RD03 still requires authenticated
receipts; strict historical text fusion still requires content lineage.

The [RD01 PDF](../../../05_results/03_dataset_analysis/02_reddit_dataset/07_rd01_review_v1/rd01_split_qualification.pdf)
and [plain-text report](../../../05_results/03_dataset_analysis/02_reddit_dataset/07_rd01_review_v1/rd01_split_qualification.txt)
explain the completed bridge, split-qualification and event-window diagnostics.
The immutable [JSON findings](../../../05_results/03_dataset_analysis/02_reddit_dataset/06_rd01_diagnostic_v1/findings.json)
contain all six source/exposure combinations. No source policy or split IDs
were frozen; no model was trained and no numerical test scores were exported.

The 4,718-CVE giant contains 3,976 post-launch CVEs. Quarantining fingerprints
attached to more than 10 or more than 25 CVEs leaves largest components of
1,681 or 2,581 CVEs. Original exposure is retained: all 4,987 prior-test-linked
CVEs remain excluded from all roles under the existing stronger RD01 proposal.
Removing a shared post cannot undo earlier inspection of its content.

Recommended **candidate**, retaining all posts and compatible prior roles:

| Task | Training CVEs | Validation CVEs | Test CVEs |
| --- | ---: | ---: | ---: |
| RD02, observed Day-60 target within role boundary | 24,213 | 972 | 3,977 |
| RD04, at least one observed past/future pair within role boundary | 24,548 | 1,106 | 3,978 |

These are not frozen allocations. The matching RD02 attachment counts are
26,578 / 1,289 / 4,428. Identifier-only attachments account for
23,269 / 6 / 1 respectively. This text-source shift needs explicit disclosure;
all retained posts are not equally informative. Shared URLs across CVEs remain
preserved; only duplicate URL entries within each CVE were removed earlier.

Deduplicated complete event windows:

| Required days before and after post | Complete attachments | CVEs | Windows crossing a local EPSS release |
| --- | ---: | ---: | ---: |
| 1 | 12,725 | 5,216 | 41 |
| 7 | 6,728 | 2,756 | 132 |
| 14 | 4,231 | 1,729 | 106 |
| 30 | 1,797 | 814 | 123 |

These are descriptive coverage counts before split qualification. Each window
requires every daily score, including the post date. Unique-CVE counts across
window lengths or text categories overlap. A full-history version-change flag
is reported separately from a release inside the local event window.

### RD01 Artifact Guide

Under `05_results/03_dataset_analysis/02_reddit_dataset/06_rd01_diagnostic_v1/`:

- `giant_posts.csv`, `giant_post_degree_distribution.csv`: every giant-component
  fingerprint, attachment degree, publication span and bridge classification.
- `top_hub_posts.csv`, `top_cross_year_posts.csv`: bounded review lists with
  real saved excerpts; a cross-year link is not necessarily a graph bridge.
- `giant_incidence_bridges.csv`, `bounded_bridge_removals.json`: actual
  bipartite bridges and the measured component effect of selected post removals.
- `component_policy_comparison.csv`: the structural and text-availability cost
  of retaining all posts versus the fixed `>10` / `>25` thresholds.
- `candidate_split_counts.csv`: all candidate role counts, retained text
  categories, source absence and eligible daily anchor/target cells.
- `candidate_exclusion_flow.json`: mutually exclusive first exclusion reasons.
- `post_window_summary.csv`: all four windows, text groups and post policies.
- `post_window_coverage.csv.gz`: one row per attachment/window, without score
  magnitudes; distinguishes out-of-range dates from missing in-range scores.
- `protocol.json`, `findings.json`, `REPORT.txt`, `artifact_lock.json`: declared
  rules, findings, explanations and hash identities.

### RD01 Commands: CPU Only

Run one at a time from the repository root in `CodeBERTFusion`:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset -p 'test_*.py'
```
```bash
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/audit_rd01_splits.py --verify-only
```
```bash
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/build_rd01_review.py --verify-only
```

The write commands (the same audit/report commands without `--verify-only`)
have already completed. Verification replays the diagnostic without overwriting
it. Before a split freeze, explicitly select the source/exposure policy and
complete the already-specified duplicate-text and target-answer source checks.

## Current Decision

For the explanatory report after verification, read the
[PDF](../../../05_results/03_dataset_analysis/02_reddit_dataset/05_gate_findings_report_v1/reddit_gate_findings.pdf),
[TXT](../../../05_results/03_dataset_analysis/02_reddit_dataset/05_gate_findings_report_v1/reddit_gate_findings.txt),
and [JSON](../../../05_results/03_dataset_analysis/02_reddit_dataset/05_gate_findings_report_v1/findings.json).
This adds interpretation and real examples without replacing the locked gate artifacts.

Stage 0 is complete. Stage 1 deduplication, field quarantine and shared-evidence
components are complete and reproduced. EPSS recovery and authenticated
VulnCheck verification have not passed. No new split, model, graph cache,
embedding or test evaluation has been created.

Read the full [TXT review](../../../05_results/03_dataset_analysis/02_reddit_dataset/04_premodeling_gates_v1/stage2_4_specifications_v2/review.txt)
and [JSON review](../../../05_results/03_dataset_analysis/02_reddit_dataset/04_premodeling_gates_v1/stage2_4_specifications_v2/review.json).
Version 2 makes component-size keys JSON-stable. The initial report and builder
are preserved; counts, specifications and all Stage 0/1 artifacts are unchanged.

## Verified Results

| Check | Result |
| --- | ---: |
| Durable copied source files | 13 |
| Source bytes, excluding the extra manifest copy | 1,601,148,355 |
| Original CVEs retained | 44,985 |
| CVEs published on/after 2021-04-14 | 35,113 |
| CVEs with at least one EPSS score, all publication years | 35,862 |
| Post-launch CVEs with at least one score | 34,995 |
| Saved VulnCheck true flags, all years | 2,187 |
| Saved true flags among post-launch CVEs | 1,418 |
| Scores recovered from nine failed dates | 0 |
| Unresolved daily scores retained as null | 17,112 |
| Exact within-CVE duplicate post entries removed | 154 |
| Retained CVE-post entries | 74,032 |
| Retained entries consisting only of a CVE ID | 25,261 |
| Repeated entries removed from all_sources_urls | 125,371 |
| Repeated entries removed from github_only_urls | 14,337 |
| Repeated entries removed from non_github_urls | 111,034 |
| Shared multi-CVE post fingerprints | 3,915 |
| Multi-CVE-mention entries after deduplication | 23,570 |
| Conservative connected components | 35,950 |
| Largest conservative component, current CVEs | 4,718 |
| Current CVEs connected to prior test CVEs | 4,987 |
| New CVE IDs among those test-connected CVEs | 4,085 |

The three URL removal counts are not disjoint: the GitHub/non-GitHub lists
are subsets of the all-reference list. The original 23,663 multi-mention
post-entry count includes duplicates; 23,570 is the derived deduplicated count.
No CVEs were merged because they share wording or an advisory.

Connected components are transitive. A post may connect A to B, and another
connect B to C. All three need the same split or quarantine even if A and C
never appear together. The exact shared-post-only graph has 35,953 components
and a largest component of 4,716 CVEs. The conservative graph additionally
uses mentioned CVE IDs and exact old Reddit text reuse. This is not a TPG or
a neural model: it is a grouping tool to avoid shared evidence across splits.

## What Remains Unresolved

- All nine direct archive requests returned HTTP 403. All nine date paths
  are absent from the official archive tree pinned at
  `8fdfde5e090e31ce53f78dfbc24c53e2b3ef7df2`. The
  [FIRST data guide](https://www.first.org/epss/data) links to that official
  archive. No API-wide score harvesting or missing-value imputation was used.
- A single public FIRST lookup for CVE-2021-26855 on 2021-04-22 returned
  HTTP 422/listNoResults. That does not establish that all requested CVEs
  lack scores. It was a diagnostic query, not a substitute data source.
- No VulnCheck API key or saved authenticated receipts were available.
  All 44,985 CVEs therefore have unknown outcomes under the **new receipt
  verification contract**. This does not erase the 2,187 saved true flags.
- The exact originally executed EPSS collector remains unrecovered. The
  retries used the separately identified uploaded parser.
- Post IDs, exact timestamps, edit history and capture-time wording remain
  unavailable. Retrospective DM studies must not become historical text
  forecasts by changing their name.

The v2 files are **not improved score datasets**: no values changed. Original
BEFORE_EPSS and CVE_NOT_IN_SNAPSHOT entries remain missing, never zero. A later
retry must use new output paths and retain the current receipts.

## Inputs and Outcomes Stay Separate

Under `04_premodeling_gates_v1/stage1_sanitization/`:

| File | Purpose |
| --- | --- |
| `predictor_candidates.jsonl.gz` | Exact-deduplicated D/M text, reference lists and retrospective NVD/CVSS metadata. Contains no catalogue or EPSS fields. IDs/dates are control metadata, not automatic predictors. |
| `outcomes_restricted.jsonl.gz` | Original catalogue flag, all ten catalogue fields, new verified outcome (currently null), and EPSS scores/statuses. Never pass the whole object to an encoder. |
| `post_audit_not_model_input.jsonl.gz` | Bare-ID and multi-CVE flags with original indices and fingerprints. |
| `shared_post_graph.jsonl.gz` | CVE attachments, mentioned identifiers and bulk-advisory candidate tags. |
| `connected_components.json` | Conservative components, publication ranges and previous-study exposure. |
| `exact_shared_post_components.json` | The narrower shared-fingerprint-only groups for comparison. |
| `cve_component_map.json` | Deterministic CVE-to-component lookup. |
| `report.json`, `report.txt`, `artifact_lock.json` | Counts, rules and source/output SHA-256 checks. |

All ten catalogue-only fields and their missingness remain outside candidate
inputs. No derived catalogue counts, masks, nodes or edges are added. For future
forecasting, a separate loader must select **past** EPSS observations from the
restricted history while keeping future targets out of its inputs.

## Proposed Runner Specifications

All configurations are in `02_prediction/04_configs/05_reddit_studies/` and
have training and test access disabled. They are specifications, not runnable
training implementations.

| Contract | Fixed comparison proposed | Status |
| --- | --- | --- |
| RD01 | Publication-chronological, component-disjoint roles; proposed train through 2023, validation 2024, later-publication test from 2025; quarantine crossing components and prior exposure | Not frozen; no assigned IDs |
| RD02 | Mean, median, ridge; pooled D/DM; v6 graph D/DM with/without 32-channel subtypes; all text arms matched with structured features | 18 neural fits + 3 deterministic controls proposed |
| RD03 | Three logistic references; pooled D/DM; v6 subtype graph D/DM; verified true/false labels only | 12 neural fits + 3 deterministic controls proposed; labels blocked |
| RD04 | Persistence, training-only lead correction, three GRU seeds; separate descriptive 7/14-day post-centered analyses | No temporal text training; no causal claims |

Seeds are 17, 29 and 43. RD02/RD03 use the pinned SecBERT identity, two existing
edge-type-aware GNN layers, hidden size 32, no COREF, node-mean readout and
source-aware pooling. No new GNN-family search is proposed. CVSS categories
use training-fitted one-hot encoding; allowed numeric features use train-only
imputation/scaling. Current CVSS is explicitly retrospective.

RD01 must check target dates, not just publication dates. A training target
cannot extend beyond the training period. Missing targets stay missing. For
RD02, the version-clean subset tests actual release changes within Days 0-60;
the global 91-day changed flag alone is insufficient. RD04 similarly checks
actual input/anchor/target spans.

Previously exposed test components are quarantined from the new study.
Components with any previous-study role cannot enter an untouched new test.
Other old train/validation roles are retained only when compatible with the
chronological proposal. Cross-period components are quarantined, never moved
backward in time. Component and exposure counts must be reviewed before
accepting this potentially restrictive design. Score bands and verified-class
distributions are reported **after rule freeze**, not used to optimize cutoffs.

## Commands: Run One at a Time, CPU Only

```bash
cd /home/ayounas/Text_property_Graph
```
```bash
conda activate CodeBERTFusion
```
```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset -p 'test_*.py'
```
```bash
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/reddit_gates.py --stage preserve --verify-only
```
```bash
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/recover_reddit_epss_v2.py --verify-only
```
```bash
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/reddit_official_recovery.py --stage recover --verify-only
```
```bash
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/reddit_gates.py --stage sanitize --verify-only
```
```bash
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/specify_reddit_studies.py --verify-only
```
```bash
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/build_gate_findings_report.py --verify-only
```

The write-stage commands already executed were `reddit_gates.py --stage
preserve`, `recover_reddit_epss_v2.py --download`,
`reddit_official_recovery.py --stage recover --download`,
`reddit_gates.py --stage sanitize`, and `specify_reddit_studies.py`.
Do not repeat write commands against completed outputs; use verification.

## Next Action: Bounded Authenticated VulnCheck Check

This uses durable source/output paths. The deterministic 20-CVE sample tests
the receipt machinery, not prevalence or model readiness. It contains ten old
true and ten old false flags, selected by sorted CVE ID. Review it before any
full collection. No key belongs in code or Git.

```bash
read -rsp "VulnCheck API key: " VC_API_KEY
```
```bash
export VC_API_KEY
```
```bash
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/repair_provenance.py kev --source EPSS_TPG/04_data/03_reddit_collection/01_source_70615ef/extracted/reddit_posts_vc_kev_epss.json --output-dir EPSS_TPG/05_results/03_dataset_analysis/02_reddit_dataset/04_premodeling_gates_v1/stage1_vulncheck_bounded --limit 20 --allow-network
```
```bash
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/repair_provenance.py kev --source EPSS_TPG/04_data/03_reddit_collection/01_source_70615ef/extracted/reddit_posts_vc_kev_epss.json --output-dir EPSS_TPG/05_results/03_dataset_analysis/02_reddit_dataset/04_premodeling_gates_v1/stage1_vulncheck_bounded --limit 20 --verify-only
```
```bash
unset VC_API_KEY
```

Use `--resume` only after an interrupted collection; add `--retry-unknown` to
retry failed receipts explicitly. New checks are dated **now**, not historical
November 4 labels. Do not freeze RD03 labels until the intended cohort has
adequate successful receipts and the collection-time policy is reviewed.
