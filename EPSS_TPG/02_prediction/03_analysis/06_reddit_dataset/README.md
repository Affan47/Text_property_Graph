# Reddit Dataset Audit

## Current: Summary Extension Preparation

RD02 full development has completed: 21 predictors on 24,213 training / 972
validation CVEs, with test scoring still disabled. The earlier smoke-only
notes below describe historical gates, not the current full-run status.
The saved-result review is `05_results/07_reddit_studies/02_rd02_primary_v1/full_review_v1`
under EPSS_TPG.

`prepare_rd02_summary_extension.py` adds a separate, versioned G/N summary
overlay, source inventory, reference-match audit and development collection
queue. It does not train, fetch webpages, call an LLM or modify RD02 results.
Read [SUMMARY_EXTENSION.txt](SUMMARY_EXTENSION.txt) for the counts, ablations,
remaining coverage gap and commands. Existing summaries are preserved;
missing summaries are not replaced by URL strings or invented text.

The original audits are CPU-only and do not train models or change source
scores. The later `run_rd04_daily.py` is a separate development adapter and
bounded smoke-training CLI; its `--verify-only` mode never retrains. See the
[current study status](../../05_studies/06_reddit_expansion/README.md) first.

## Latest Development Gates

- `run_rd02_primary.py` implements separate screened predictor/Day-60 label
  exports, train-fitted structured preprocessing, v6 TPG caches and the
  21-predictor smoke gate. Full export, the 32-training-CVE graph gate and
  12-train/6-validation smoke artifacts are saved and verified. Full graph
  caching waits for the user's durable storage location; no full training
  or test mode is enabled.
- `review_reddit_smoke_gates.py --verify-only` reproduces the readable combined
  report, all RD02 smoke metric rows and artifact hashes under
  `05_results/07_reddit_studies/02_rd02_primary_v1/gate_review_v1`.
- `run_rd04_daily.py --stage preflight --scale full --verify-only` reproduces
  24,548 train / 1,106 validation CVEs and 96,289,074 / 3,692,438 forecast pairs.
- `run_rd04_daily.py --stage smoke --scale smoke --verify-only` replays the
  completed three-seed smoke checkpoints, matched controls, raw/clipped
  predictions and release-partition diagnostics. It does not train.
- `review_rd02_overlap.py --verify-only` reproduces the 24-signature structural
  review and hash-linked source approval. It reads no outcome file and exports
  no raw test text. RD02 bounded cache construction is now complete;
  full caching is waiting for durable storage.
- `test_rd04_daily.py` and `test_rd02_overlap_review.py` test temporal input
  isolation, weights, masks, release boundaries, metadata-only output and CSV
  reproducibility. `test_rd02_primary.py` additionally checks structured JSON
  reloads, strict feature allowlists, subtype indexing, masked wrapper parity
  and replay gradients. The complete local suite passes 75 tests.

RD04 results: `05_results/07_reddit_studies/01_rd04_daily_v1` under `EPSS_TPG`.
RD02 review: `05_results/03_dataset_analysis/02_reddit_dataset/10_rd02_overlap_review_v1`.
Full training and test scoring remain disabled. The original locks, data and
past experimental results were not overwritten.

## Read First

The illustrated guide is now available as
`00_documentation/06_papers/02_pdf/reddit_dataset_guide.pdf` under `EPSS_TPG`.
Its complete bundle (LaTeX, TXT, JSON evidence and nine figures) is in
`05_results/03_dataset_analysis/02_reddit_dataset/03_dataset_guide_v1`.
It explains all 34 possible top-level fields and all observed nested fields.
Important correction: records have 24 fields when saved as unlisted and 34
when listed; absent keys are not the same as explicit null values.

1. The supplied `Reddit dataset report.pdf` describes the new collection.
2. `audit_reddit.py` fetches and checks the pinned files, then audits their
   content and source-code provenance.
3. Read `findings.txt` in the output folder for the findings in plain English.
   `findings.json` contains the machine-readable counts and evidence.

## Locations

- Branch: `observatio/SummTPGVul`, `new_data_collection`.
- Pinned commit: `70615ef948b6063ea47dc5479174e1b2e6a1eb54`.
- Default download directory: `/tmp/tpg_reddit_audit_70615ef`.
- Project results: `EPSS_TPG/05_results/03_dataset_analysis/02_reddit_dataset/01_readiness_v1`.

The default source location is temporary because the project filesystem has
less than 200 MB free. `/tmp` is not durable storage. Before deleting it, move
the verified source bundle to a sufficiently large persistent disk. The
`--data-root` option supports another location; a changed absolute source path
requires a new report output directory because source identities include paths.
Never place this download over the existing `SummTPGVul` submodule.

## Commands, One at a Time

```bash
cd /home/ayounas/Text_property_Graph
```
```bash
conda activate CodeBERTFusion
```
```bash
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/audit_reddit.py --stage fetch
```
```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset -p 'test_audit_reddit.py'
```
```bash
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/audit_reddit.py --stage audit
```
```bash
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/audit_reddit.py --stage audit --verify-only
```

If the audit already exists, run the final verification command instead of
creating it again. Verification recomputes the analysis and checks every output
hash; it does not overwrite the saved report. GPU is not used.

Fetching uses the existing Git credential through the GitHub LFS batch API.
Credentials and signed download URLs are not recorded. Git blob hashes, LFS
sizes/SHA-256 hashes, ZIP contents and extracted member hashes are checked.
No branch switch, push or update of the existing project submodule occurs.

## Interpretation

- One CVE can have several posts, and one post can mention several CVEs.
- Text/date/platform fingerprints are not authenticated Reddit post IDs.
- The date parser accepts only the saved ISO date format; no local timezone is
  guessed. The scraper's `created_utc` conversion provides UTC-date lineage,
  not independently authenticated historical wording.
- EPSS windows cover publication day through Day 90. Events outside this
  range may be useful, but require additional score history, not imputation.
- Before/after counts are conditional coverage counts, not effects, causation,
  or training eligibility. Same-day score availability does not establish
  whether the score was published before or after the post.
- VulnCheck error-handling checks run AST-extracted functions with simulated
  errors only. They show a possible failure path, not the number of bad labels
  in the saved collection.
- Prior cohort IDs are used to identify exposure. Old test predictions and
  metrics are never opened. There is no new train/validation/test split.

The current task neither modifies collectors nor retries missing EPSS scores.
Those actions need separate versioned outputs after reviewing this audit.

## Collector Search and Recovery Tools

Read `EPSS_TPG/05_results/03_dataset_analysis/02_reddit_dataset/RECOVERY_STATUS.md`
next. It explains the repository search, the remaining evidence gaps, and the
difference between a fresh VulnCheck check and repairing an old snapshot.

- `recover_collectors.py`: searches available Git refs, historical collector
  blobs and working copies; saves TXT/JSON evidence and verifies its hashes.
- `repair_provenance.py kev`: checks VulnCheck with explicit listed, not-listed,
  and unknown outcomes. Requests require `--allow-network`; the default only
  prints the plan. Original labels and datasets are not overwritten.
- `repair_provenance.py reddit`: preserves raw post identity, timestamps,
  wording, edit metadata and archive provenance from a supplied raw file.
  It cannot reconstruct fields already discarded from the enriched dataset.
- `test_recovery.py`: offline regression tests, including simulated HTTP
  failures, response replay, identity conflicts and text-version preservation.

Run these CPU-only checks one at a time:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset -p 'test_*.py'
```
```bash
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/recover_collectors.py --verify-only
```
```bash
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/repair_provenance.py kev --limit 20
```

For a fresh, bounded API check, supply your key without placing it in shell
history, then run and verify the receipts:

```bash
read -rsp "VulnCheck API key: " VC_API_KEY
```
```bash
export VC_API_KEY
```
```bash
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/repair_provenance.py kev --limit 20 --allow-network
```
```bash
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/repair_provenance.py kev --limit 20 --verify-only
```
```bash
unset VC_API_KEY
```

If collection is interrupted, repeat the collection command with `--resume`.
Previously failed requests are retried only with `--resume --retry-unknown`.
Authentication failures and rate limiting stop collection; remaining CVEs stay
unknown. Review the bounded result before using `--full` with a separate output
directory. These results describe the new lookup times, not the old snapshot.

The default recovery outputs are under `/tmp/tpg_reddit_recovery_v1` because
the project disk is nearly full. Move verified evidence to persistent storage
before clearing `/tmp`. No raw Reddit recovery has been run: it requires an
original archive file with identity fields, not the current enriched CVE file.

## Full RD02 and RD04 Development Runs

The separate `run_reddit_development.py` CLI checks the 11 approved upstream
hashes before building the full RD02 cache, running either development study,
or reproducing reports. It does not change the original smoke runners or
their locked outputs. Full training is run by the user, not automatically.

Read [RUN_FULL_DEVELOPMENT.txt](RUN_FULL_DEVELOPMENT.txt) for the ordered commands,
restart rules, resource checks, output locations and scientific limits.
RD02 covers 24,213 training / 972 validation CVEs; RD04 covers 24,548 / 1,106.
Both keep their held-out test cohorts closed. The frozen chunk size is 1,200
characters; 4,096 is the node budget per graph packet, not a text chunk limit.

## Reproduce the Illustrated Guide

The CPU-only builder checks the saved audit hashes and exact dataset identity,
extracts real examples, compiles LaTeX three times and checks text boundaries.
It refuses to overwrite a completed report. Verify the completed bundle:

```bash
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/build_dataset_guide.py --verify-only
```

To reproduce it separately without changing the published bundle:

```bash
python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/build_dataset_guide.py --output-dir /tmp/reddit_dataset_guide_reproduction
```

This requires the previously downloaded `/tmp/tpg_reddit_audit_70615ef` sources
and the installed TeX Live. `--resume-build` is only for an incomplete build
without an artifact lock. The original supplied PDF and experiments are untouched.
