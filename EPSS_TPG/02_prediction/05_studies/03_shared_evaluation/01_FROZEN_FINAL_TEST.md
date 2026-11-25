# PS05: Frozen Final Evaluation

**Status:** full evaluation completed and independently reproduced on all 651
reserved test CVEs. Read the [final findings](02_FINAL_RESULTS.md). Commands
below document the original procedure; use verification commands for completed
artifacts, not fresh-run commands or a revised protocol to chase test scores.

## Purpose

Evaluate the completed primary-study models on the reserved test CVEs, once,
without changing models after seeing their test performance. This is the final
assessment of retrospective Day-60 EPSS estimation, not another training run.

The development results selected description + SMP pooled text as the leading
configuration and all-source TPG as the graph comparator. The report will still
include **all eight PS04 S1 configurations, all three seeds and three baselines**:
27 saved fits. No best-seed selection or new ensemble is permitted.

The new evaluator is separate from development code. Test records stay labelled
`test` in inputs, graph indexes, receipts and predictions. Existing development
validators remain unchanged and continue to reject test inputs.

## What is frozen before release

- All checkpoint files and their originating run identities.
- The completed PS04 comparison set and seeds 17, 29 and 43.
- MAE as the primary metric; the existing regression metrics and fixed EPSS
  category thresholds 0.01 and 0.1 as additional diagnostics.
- Recorded EPSS exactly 60 days after publication as the target.
- The training-fitted structured encoder, frozen text encoder, graph extractor,
  relation vocabulary, chunk boundaries, pooling and inference settings.
- The complete existing CVE role manifest and date-admissibility rules.
- The current-text policy, including the existing explicit-EPSS-disclosure
  quarantine; no new summary-quality or optional-text-overlap filter.
- Code, data and model hashes. Changing these causes a failure, not silent mixing.

The freeze command checks completed development results and metadata only.
It does not inspect numerical test scores, test descriptions, summaries or posts.
Every full data/cache/evaluation stage requires `--release-test`.

## Which CVEs are evaluated?

Start with the 651 CVEs assigned to the reserved test role in the existing
manifest. That is a role count, not a promise that all 651 have usable targets.
The eligible count was calculated only after explicit release: all 651 qualify.

Retain the inherited publication-plus-30/publication-plus-60 date rule for
cohort continuity. Day 30 is not an input score in this primary study. The
policy boundaries are validation starting 2025-03-02, test starting 2025-06-01,
test anchors before 2025-12-02, and target observation ending 2026-09-29.
Those rules are read from the saved protocol, not redefined by the evaluator.

A missing Day-60 target excludes the record from scoring with an explicit
reason. An observed zero remains a valid target. Missing descriptions,
summaries, SMP or CVSS do not remove a CVE; the existing masks remain in use.
Malformed dates or graph/embedding failures stop the stage for investigation
rather than silently dropping difficult records.

Inputs and labels are separate files. Graph construction receives only
allowlisted text, source identity and provenance, never EPSS targets. The
structured encoder is loaded, not fitted again. Baseline means, medians and
ridge coefficients are read from their saved checkpoints, not recalculated
using test records.

## Retrospective text and overlap

Preserve the primary study's accepted current text. Earlier temporal adapters
had different date/overlap rules; importing those into this final primary
evaluation would change its input policy. An audit reports text shared with
training or validation without removing it or relabelling distinct CVEs as
duplicates. Consequently, this is CVE-disjoint evaluation, not a guarantee of
unrelated vulnerability families or independent textual evidence.

Current descriptions, summaries and CVSS may postdate the target score. Do
not present these results as a historical forecast, exploitation classification
or causal SMP effect. The EPSS-disclosure screen is conservative and does not
prove that all indirect target information has been removed from text.

## Run the final stages

Run commands one at a time from the repository root with `CodeBERTFusion`
active. The protocol is already frozen during implementation; start by verifying it:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/12_run_primary_final_evaluation.py --stage freeze --verify-only
```

**The next command explicitly releases held-out records.** It prepares
predictors and targets but does not score a model:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/12_run_primary_final_evaluation.py --stage data --scale full --release-test
```

Optional independent reproduction of this export:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/12_run_primary_final_evaluation.py --stage data --scale full --release-test --verify-only
```

Build test TPGs and frozen embeddings with four CPU workers:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/12_run_primary_final_evaluation.py --stage cache --scale full --release-test --workers 4
```

This is graph preparation, not model training. Each worker uses one Torch
thread. Progress bars cover chunk reuse/building and tensor/span verification.
All original graph policies remain fixed. Reserved CVEs have different chunk
identities from development CVEs, so new test chunks normally need building.
An interrupted cache can resume verified completed chunks:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/12_run_primary_final_evaluation.py --stage cache --scale full --release-test --workers 4 --resume
```

A completed cache can be independently checked without regenerating embeddings:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/12_run_primary_final_evaluation.py --stage cache --scale full --release-test --verify-only
```

Score all frozen models on the same eligible test CVEs:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/12_run_primary_final_evaluation.py --stage evaluate --scale full --release-test
```

Reproduce the saved predictions and report without refitting:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/12_run_primary_final_evaluation.py --stage evaluate --scale full --release-test --verify-only
```

Use `--resume` on the evaluation command only if it was interrupted. Completed,
hash-verified fit outputs are reused; an incomplete fit's inference is repeated.
Verification is reproducibility checking, not another opportunity for selection.

## Files and reports

| Location under `EPSS_TPG` | Purpose |
| --- | --- |
| `05_results/04_consolidated_studies/01_primary_day60/final_evaluation_v1/frozen_protocol/` | Decisions, checkpoint hashes, reserved IDs and integrity lock saved before release |
| `04_data/01_records_and_graphs/primary_day60_final_v1/full/inputs.json` | Test predictors only |
| `04_data/01_records_and_graphs/primary_day60_final_v1/full/labels.json` | Separate observed Day-60 targets |
| Same data folder: `eligibility_audit.json`, `overlap_audit.json` | Included/excluded CVEs and shared-text evidence |
| `01_tpg/05_workspace/primary_day60_final_v1/full/` | Actual GraphSON, tensors, source documents, CVE index and per-chunk receipts |
| `05_results/04_consolidated_studies/01_primary_day60/final_evaluation_v1/full/fits/` | Per-fit actual/predicted CSV/JSON, metrics, source masks/attention, unchanged-state diagnostics and figures |
| Same result folder: `reports/` | All frozen configurations, per-seed metrics, paired comparisons, TXT/JSON analysis and comparison figure |

No trained checkpoints are overwritten or newly trained here. The source
checkpoint remains in its original PS02 or PS04 folder. The test result records
which checkpoint was used.

Regression figures show actual versus predicted scores, signed errors and
errors by actual score. Category figures show confusion matrices, ROC and
precision-recall curves at the two frozen thresholds. F1 and accuracy describe
EPSS categories, not confirmed exploitation. Undefined metrics remain explicit.

Reports average seed-level errors, not predictions into an unregistered ensemble.
Paired intervals compare the same CVEs. They do not correct for multiple testing
or dependence from shared text. The designated graph-versus-leader comparison
is reported alongside the complete source/representation/baseline comparisons.

## Smoke checks and stopping rule

Implementation verification passed: **78 consolidated tests**, plus the
independent freeze, data, cache and evaluation verification commands. The smoke
uses two existing validation CVEs and nine reused chunks. All 27 full-trained
checkpoint predictions reproduce their saved development values; model state
hashes remain unchanged. The report figure was inspected for readable layout.
No full held-out data, graphs or scores were produced during implementation.

The smoke stages use only the two existing PS01 smoke-validation CVEs and the
full-trained checkpoints. They must reproduce saved development predictions.
They do not read reserved test records. Synthetic tests exercise release guards,
observed-zero versus missing targets, target/predictor separation, role handling,
matching chunk plans, no-refit baselines and immutable neural inference.

Once test results are available, report them without changing thresholds,
source filters, seeds, eligible cases or checkpoints to improve the score.
Any later research change is a new study and needs an appropriate independent
evaluation. A negative result remains useful evidence about this implementation.

Implementation: `02_prediction/01_package/epss/primary_final_evaluation/`.
Protocol: `02_prediction/04_configs/01_consolidated/05_primary_final_evaluation.json`.
