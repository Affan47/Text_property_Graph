# VK08: Source Reliance and Validation-Selected Thresholds

Full analysis is complete. Read the [verified findings](14_THRESHOLD_AND_SOURCE_FINDINGS.md).

## 1. Purpose

VK07 full training is complete. VK08 does not train another model. It uses the
saved classifiers to answer two questions:

1. Which score cutoff gives the highest F1 on the available validation CVEs?
2. How do the all-source classifiers respond when their source information is
   removed, zeroed, or exchanged between CVEs?

The target is still `in_vckev`: listed or not listed in the saved VulnCheck
snapshot. It is not an EPSS threshold or a future exploitation label. The
original VK07 results, including F1 at 0.5, remain unchanged.

Implementation checks: all 92 VulnCheck regression tests passed, including 12
new threshold/intervention tests. Both smoke stages completed and independently
reproduced with `--verify-only` on 64 validation CVEs; the six all-source models reproduced the saved probabilities exactly
(maximum difference 0.0). This smoke subset has only two listed CVEs, so its
selected F1 values are integration evidence, not useful performance estimates.
The user subsequently completed full validation threshold selection and full
diagnostic execution; no new dataset training or test evaluation was performed.

The first smoke verification exposed a JSON identity comparison issue: tuple
definitions become lists when read from JSON. The comparison now uses canonical
JSON on both sides, without relaxing artifact hashes or numerical tolerances.
The initial smoke report is preserved as `smoke_initial_identity_check/` in the
diagnostic result root. The corrected report is in `smoke/`. A regression test
covers the JSON round trip, and baseline decisions must match the saved scores
at both 0.5 and the selected threshold, not just within a probability tolerance.
Corrected and initial smoke predictions and summary metrics are byte-for-byte
identical; only the report-identity handling and additional guard changed.

## 2. What Dynamic Threshold Selection Means

Each saved checkpoint receives **one validation-selected threshold**. A CVE is
predicted listed when its score is greater than or equal to that threshold.
There is no separate threshold for each CVE, no threshold learned from test
labels, and no promise that the chosen threshold will remain optimal on new data.

The objective is explicitly **positive-class F1**, not accuracy or AP. The
algorithm searches every distinct validation score, treating equal scores as
one group. It also includes an all-negative decision boundary. This exhausts
the possible decision sets for a scalar threshold, instead of using a coarse
0.01-spaced grid. If multiple cutoffs have exactly equal F1, the highest cutoff
wins; this gives the more conservative decision set. AP and ROC-AUC do not
change when the decision threshold changes.

All 24 saved fits are included: 18 new multi-source fits and six unchanged
description references. No seed, checkpoint, source combination, or ensemble
is automatically selected as the final model. The AP-selected checkpoints are
not replaced by checkpoints selected for F1.

## 3. Reading the Three F1 Columns

| Column | Meaning | Limitation |
|---|---|---|
| Fixed-0.5 F1 | Decisions made using the original cutoff | Useful unchanged comparison; not necessarily optimal |
| Same-validation selected F1 | Highest F1 after searching cutoffs on these validation labels | Optimistic: these same labels selected and measured the cutoff |
| Cross-fold F1 | Each CVE is classified using a cutoff selected on the other four validation folds | Checks cutoff sensitivity, but does not undo earlier checkpoint selection or validation reuse |

Five folds are assigned deterministically by a hash of the CVE identifier,
without using labels or scores. Within each fold, the cutoff is selected only
on the other folds. Fold thresholds, selection counts, predictions, and fold
scores are saved. If a selection partition lacks either class, that fold's
cutoff is explicitly undefined; the overall cross-fold F1 is then null rather
than calculated on a silently reduced cohort. Undefined precision is also null.

These are **threshold cross-fold checks**, not retrained cross-validation
models. CVEs with related wording can still cross these folds. There is no new
independence or leakage-free claim. A larger selected F1 must not be published
as a new independently measured test result. Final use requires a frozen model
and threshold plus an explicitly agreed evaluation plan; prior EPSS test
exposure remains relevant.

## 4. Frozen Source Interventions

The diagnostic loads the six full-trained all-source models: pooled text and
security TPG, each at seeds 17, 29, and 43. It reuses verified cached node
embeddings, runs the saved graph/text encoders, and pools chunks into documents
and documents into sources exactly as in VK07. No parser, pretrained embedding
encoder, optimizer, or new graph construction is run.

Before interpreting anything, the diagnostic must reproduce saved probabilities
within the existing absolute tolerance of `1e-7`. The original batch boundaries
and batch-local sigmoid calculation are retained. Model state and absence of
gradients are checked afterward.

| Intervention | What changes | What stays fixed |
|---|---|---|
| Zero description, summaries, posts, or all optional sources | Corresponding post-encoder source vectors become zero | Original source masks and structured representation |
| Swap those source groups | Another validation CVE provides the corresponding vectors | Recipient mask, structured representation, label, and threshold |
| Remove GitHub summaries, non-GitHub summaries, both summaries, posts, or all optional sources | Vectors are zeroed and corresponding mask bits disabled; attention reweights remaining sources | Description remains present; structured representation and threshold |
| Zero structured representation | Encoded structured vector becomes zero | Text, masks, graph nodes, edges, and threshold |

Swaps use one deterministic, label-blind cycle within each complete source-mask
pattern. Missing sources are never invented. A singleton mask group cannot
swap with another CVE and is left unchanged, explicitly recorded in the donor
table. The same donor mapping is used for all six models. This is one sensitivity
intervention, not a permutation significance test.

Zeroing is different from removing: zeroing preserves availability indicators,
whereas removal changes which sources receive attention. Both can create inputs
unlike those seen during training. Neither identifies a causal source effect.

Results use both 0.5 and the already selected checkpoint threshold. **No cutoff
is reselected after an intervention or for a wording subset.** Results are
reported for all selected validation CVEs, CVEs with recorded wording shared
with training, and those without that recorded overlap. Shared wording does
not imply that the vulnerabilities are duplicates.

## 5. Files Written

Paths below are relative to `EPSS_TPG/05_results/05_vulncheck_study/`.

| Folder/file | Purpose |
|---|---|
| `15_threshold_selection_v1/<scale>/summary.txt` and `.json` | Threshold objective, fixed/selected/cross-fold metrics, interpretation |
| `thresholds.json` | One selected cutoff per checkpoint, bound to its SHA-256 |
| `comparison.csv` | Convenient model-by-model metrics table |
| `threshold_curves.csv` and `.png` | Candidate cutoffs and F1; plots show the all-source models |
| `predictions.json` | Original scores plus fixed, selected, and cross-fold decisions |
| `16_multisource_diagnostic_v1/<scale>/summary.txt` and `.json` | Frozen source-intervention results and limitations |
| `comparisons.csv` | Model, intervention, subset, AP/F1 changes, score changes, decision flips |
| `predictions.json` | CVE-by-CVE probabilities and decisions for every intervention |
| `source_donors.json` | Exact donor CVEs, source masks, and unchanged singleton cases |
| `baseline_attention.json` | Original source weights; descriptive, not causal importance |
| `confident_cases.json` | Up to three false flags and misses per model with actual accepted source excerpts |
| `run_identity.json` and `artifact_lock.json` | Input/code/checkpoint provenance and output hashes |

False flags mean unlisted in the saved snapshot, not never exploited. Evidence
excerpts are bounded and marked as truncated when necessary. They are examples
for review, not independently verified factual source claims.

## 6. Run in Order

Run from the repository root in `CodeBERTFusion`. The smoke scale uses 64 CVEs
from complete original validation batches, selected without examining labels
or scores, and all six full-trained all-source models. Smoke thresholds are
integration checks only and must never be used for deployment.

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study -p 'test_vulncheck_threshold_diagnostic_v1.py'
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/14_diagnose_sources_and_select_thresholds.py --stage thresholds --scale smoke
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/14_diagnose_sources_and_select_thresholds.py --stage thresholds --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/14_diagnose_sources_and_select_thresholds.py --stage diagnose --scale smoke
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/14_diagnose_sources_and_select_thresholds.py --stage diagnose --scale smoke --verify-only
```

Then run the full 1,104-CVE validation analysis:

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/14_diagnose_sources_and_select_thresholds.py --stage thresholds --scale full
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/14_diagnose_sources_and_select_thresholds.py --stage thresholds --scale full --verify-only
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/14_diagnose_sources_and_select_thresholds.py --stage diagnose --scale full
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/14_diagnose_sources_and_select_thresholds.py --stage diagnose --scale full --verify-only
```

Existing completed outputs are never overwritten. Use `--verify-only` to
reproduce them. `--check-only` performs a preflight without the analysis or
output writes; the diagnostic preflight requires completed thresholds of the
same scale. Hashing referenced chunks may take time even for smoke: it verifies
the frozen cache and does not rebuild it. A failed unfinished run writes no
completed report; rerun the same command after investigating the failure.
