# VulnCheck Snapshot Classification

**VK10 final evaluation completed and reproduced.**
Start with the [final findings](17_FINAL_RESULTS.md), then the
[execution and artifact guide](16_FINAL_EVALUATION.md).
All 36 frozen predictors were evaluated on the exact 651 CVEs, with 36 listed
and 615 not listed. No training, threshold tuning, source repair or model
replacement occurred. All-source TPG mean AP/F1 at frozen validation cutoffs
are 0.260112/24.51%, versus 0.273424/31.22% for matched pooled text.
The validation advantage did not transfer. The AP difference interval includes
zero; both declared F1 difference intervals favor pooled text conditionally.
This is a reused-cohort retrospective evaluation, not a fresh independent test.
Results, per-CVE predictions and a 38-page figure PDF are in
`05_results/05_vulncheck_study/19_final_evaluation_v1/full/`.

**Earlier VK09 stage: development reporting and final protocol freeze.**
Read the [frozen development report and evaluation rules](15_FINAL_REPORT_AND_EVALUATION_PROTOCOL.md).
All-source security TPG is the validation-AP-selected primary family; all three
seeds and 36 completed predictors remain in the report. Exact cohort IDs,
checkpoint hashes, source/feature rules and existing thresholds are locked in
`05_results/05_vulncheck_study/18_final_protocol_v1/`.
Final scoring had not been run when VK09 was frozen; VK10 records the later
authorized execution without changing that historical bundle. The 651 CVEs
were previously exposed in the EPSS study and are not a new independent test. No further
training, threshold tuning, source repair or test-based model replacement is
permitted within this frozen version.

**VK08 full results are available:** read the
[threshold and source findings](14_THRESHOLD_AND_SOURCE_FINDINGS.md).
All-source TPG F1: 45.85% at 0.5, 48.23% selected on the same validation data,
45.76% in threshold-only cross-fold checks. Highest individual selected F1 is
49.67%, not an independent-test result. Posts dominate source sensitivity;
real summary/CVE mismatches are documented without claiming they caused every
error. No further broad training is recommended from this review.

**VK08 source diagnostic and validation threshold selection:** read the
[policy, outputs and CLI sequence](13_SOURCE_DIAGNOSTIC_AND_THRESHOLDS.md).
This extends the completed models without retraining: one maximum-validation-F1
cutoff per checkpoint, fixed-0.5 references, cross-fold cutoff checks, and frozen
all-source interventions. No test scoring or automatic final-model selection.

**VK07 full multi-source ablation completed and verified:** read the
[results, source effects and shared-wording sensitivity](12_MULTISOURCE_RESULTS.md), then the
[source-fusion design, storage policy and CLI sequence](11_MULTISOURCE_ABLATION.md).
D, DGN, DM and DGNM compare pooled text against security TPG with the same
structured inputs and 2,870/1,104 development CVEs. Six D checkpoints remain
frozen references; all 18 new full fits are complete. No test scoring.
Verification passed: 80 tests, 200 checked smoke chunks, six new smoke fits and
independent checkpoint/report reproduction. The subsequent full checkpoint
verification and separate 24-configuration results review also reproduce.
All-source TPG mean AP/F1 are 0.4149/45.85%, versus 0.3781/43.60% for matched
pooled text. TPG AP wins two of three seeds. On the 1,018-CVE subset without
recorded shared training wording, all-source TPG AP is 0.3182 versus description
TPG 0.2713; the posts-only TPG gain disappears there. These are development
results, not independent-test or causal findings.

**VK06 full edge-free training completed and verified:** read the
[results and interpretation](10_EDGE_FREE_RESULTS.md) and
[frozen design and commands](09_EDGE_FREE_CONTROL.md). Three fresh full-seed fits
retain VK04 nodes, embeddings, encoder layers, pooling, structured inputs and
settings; only edges are disabled. Existing graph/pooled/logistic runs remain
references. Mean AP is 0.2873 edge-free versus 0.2992 original TPG; effects vary
by seed, while original TPG ROC-AUC is higher in all three seeds. No test scoring.

**VK05 v2 full diagnostic reviewed:** read the
[branch reliance, edge sensitivity and real error examples](08_DIAGNOSTIC_RESULTS.md).
Both input branches affect ranking. Security-edge removal changes no fixed-threshold
decisions in any seed; all-edge removal has mixed ranking effects. This is
frozen-model sensitivity, not proof of causal importance or a new training result.

**VK05 frozen-model diagnostic implemented:** read the
[branch, edge and confident-mistake diagnostic guide](07_FROZEN_GRAPH_DIAGNOSTIC.md).
It uses saved full VK04 checkpoints, unchanged validation roles and fixed 0.5
threshold. No training, new graph/text embeddings or test scoring is performed.

**VK04 full matched graph comparison completed and verified:** read the
[six-fit results and interpretation](06_GRAPH_RESULTS.md), followed by the
[threshold explanation and architecture](05_THRESHOLDS_AND_GRAPH_COMPARISON.md).
Mean validation AP is 0.3015 for pooled descriptions and 0.2992 for TPG, both
with structured features. No consistent graph ranking advantage is established.
The original runs remain unchanged; no test scoring or threshold tuning occurred.

**VK03 full baseline training is complete and verified:** read the
[results and next bounded graph comparison](04_BASELINE_RESULTS.md).
The [frozen eight-fit protocol](03_FULL_DEVELOPMENT_BASELINES.md) preserves the
2,870 training / 1,104 validation cohort. These are logistic baseline results,
not yet TPG-GNN results. No test scoring occurred.

**Full audit complete and reproduced:** read the
[verified findings and next decision](01_READINESS_RESULTS.md). The separate
interpretation preserves the original outputs and pending review entries.

**VK02 adapter and classifier smoke complete:** read the
[versioned policy, separate exports and smoke findings](02_POLICY_ADAPTER_AND_SMOKE.md).
This is an engineering-reviewed integration stage, not full training or a new
independent evaluation. The sections below retain VK01's audit context.

## 1. The Question

Can descriptions, URL summaries, social/related posts and current CVSS features
distinguish CVEs **listed in the saved VulnCheck KEV snapshot** from CVEs not
listed in that snapshot?

This is a new binary classification study. It is not CISA KEV prediction,
prediction of never/ever being exploited, or prediction of addition within a
future time window. Current text and structured values remain retrospective.

## 2. What VK01 Implements

The readiness audit checks all 5,968 CVEs' label types and publication/addition
dates, then checks text-source availability and potential target disclosure.
It does not train models, generate embeddings/graphs, change splits, or score
the test set. It writes separate labels and source-audit metadata, not an
approved model-input matrix.

The existing roles are retained **for auditing**, not declared a new untouched
evaluation split. The earlier primary-study test has already been observed.
The audit requires an explicit new evaluation decision before training; a fresh
independent final test cannot be claimed just by changing the target.

The original source and all EPSS results remain unchanged. Missing or nonboolean
labels remain invalid rather than becoming negative. Negative addition/publication
differences are retained as date warnings, not used to discard valid snapshot
labels. No EPSS history or Day-60 observation is required for this binary label.

## 3. Source Rules and Their Limits

Candidate quarantine applies to a complete document when it mentions VulnCheck,
KEV, the known-exploited-vulnerability catalogue phrase, a CISA catalogue link,
or an explicit numeric EPSS disclosure. This intentionally conservative screen
can flag benign context; **a flag is not proof of leakage**. Review the saved
examples before approving the source policy.

Other-CVE references and exploitation statements are separate review flags.
They do not automatically mean that a document is wrong or should be removed.
Repeated normalized text is reported across CVEs and previous roles, without
merging distinct vulnerabilities or reassigning their roles.

Each document keeps its source, platform, exact-text hash, normalized-text hash
and review flags. Source masks show raw availability and availability after the
candidate quarantine. A missing source never automatically removes the CVE.
Bounded review excerpts are drawn only from previous training/validation roles.

## 4. Proposed Experiments, Not Completed Fits

The machine-readable experiment matrix contains two baselines and 16 neural
configurations: D / DM / DGN / DGNM, pooled text versus security TPG, and
structured features off versus on. Three seeds would produce 48 neural fits;
this is a bounded proposal, **not a command to launch them now**.

- D: description; G: GitHub summary; N: non-GitHub summary; M: posts.
- Baselines: training-prevalence constant and structured logistic regression.
- Structured inputs: fixed CVSS v3 encoding plus NVD reference count; fitting
  any scaling or class weights uses training records only.
- TPG candidate: existing edge-type-aware GNN with the fixed no-coreference
  extraction policy. No claim that relation extraction itself is learned.
- Initial predictors exclude EPSS and all catalogue-derived fields, including
  their masks, counts, nodes, edges and embeddings.
- Select checkpoints using validation average precision (AP). Report both
  fixed 0.5 and validation-selected F1 decision thresholds; freeze thresholds
  before a designated final evaluation.
- Report AP, ROC-AUC, precision, recall, F1, balanced accuracy, accuracy,
  confusion matrices, Brier score, log loss and calibration. A classifier score
  is not assumed to be calibrated just because it lies between zero and one.

The two URL-summary groups are combined in DGN. This bounded matrix does not
separately identify their individual effects or include every possible subset.
Existing TPG chunks are only reuse candidates: accepted text, source and
processing identities must match after the new disclosure screen.

## 5. Run One Command at a Time

From the repository root:

```bash
conda activate CodeBERTFusion
```

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study -p 'test_*.py'
```

Optional preflight without writing reports:

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/01_audit_readiness.py --scale smoke --check-only
```

The implementation smoke checks all labels/field presence but only 24 CVEs'
text (16 training, eight validation), selected to cover available sources
without using labels. It is not a representative sample for text percentages.

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/01_audit_readiness.py --scale smoke
```

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/01_audit_readiness.py --scale smoke --verify-only
```

Then audit every CVE's text:

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/01_audit_readiness.py --scale full
```

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/01_audit_readiness.py --scale full --verify-only
```

An existing output directory is not overwritten. Verification checks hashes and
recomputes the JSON/TXT audit from the pinned source. Figures have saved hash
checks; verification does not rerender them. Progress bars cover every major step.

## 6. Where to Read the Output

Results are under `EPSS_TPG/05_results/05_vulncheck_study/01_readiness_v1/<scale>/`.

| Read order | File | Purpose |
|---|---|---|
| 1 | `summary.txt` | Readable findings, counts, limitations and decisions still needed. |
| 2 | `summary.json` | Full machine-readable counts by previous role, year and source. |
| 3 | `review_examples.json` | Real bounded excerpts with pending human-review status. |
| 4 | `feature_policy.json` | Proposed allowlist and prohibited direct/derived predictors. |
| 5 | `protocol.json`, `experiment_matrix.json` | Frozen audit specification and proposed training comparisons. |
| 6 | `labels_audit.json` | One label/date-audit row per CVE, separate from source information. |
| 7 | `source_audit.json`, `source_masks_audit.json` | Document hashes, warnings and per-CVE source presence. |
| 8 | `shared_text_groups.json` | Shared wording across distinct CVEs and previous roles. |
| 9 | `run_identity.json`, `artifact_lock.json` | Source/code identities and saved-output checksums. |

Figure `previous_role_labels.png`: the horizontal axis is the **previous** CVE
role, the vertical axis counts CVEs, and stacked colors distinguish listed from
not listed. These bars do not approve new splits.

Figure `candidate_source_coverage.png`: rows are previous roles, columns are
text sources, and each number/color counts CVEs retaining that source after
the candidate screen. In smoke mode, unexamined roles show zero examined
records, not evidence of missing text throughout those roles.

## 7. What Must Be Decided Next

Implementation verification passed: 16 focused tests, a real-data smoke run
and independent smoke-output reproduction. All-record label checks found:

| Previous role | CVEs | Listed | Not listed |
|---|---:|---:|---:|
| Training | 2,870 | 376 | 2,494 |
| Validation | 1,104 | 74 | 1,030 |
| Previously observed test | 651 | 36 | 615 |
| Previously excluded | 1,343 | 62 | 1,281 |
| Total | 5,968 | 548 | 5,420 |

All labels are booleans. The 324 training-role CVEs without an eligible Day-60
EPSS label have valid VulnCheck labels. This removes an EPSS-specific obstacle;
it does not automatically approve a new split. The stored-date audit retains
219 negative differences, including 104 cases on the publication day.

The source-aware smoke examined only 24 CVEs / 60 documents and flagged one
candidate numeric EPSS disclosure. Five bounded review excerpts remain pending.
Those text counts are smoke findings, not whole-dataset estimates. The full
text audit subsequently completed; its separate findings are linked above.

The subsequent VK02 stage implements an engineering-reviewed source policy,
separate predictors/labels and a small classifier smoke; see the linked guide.
It does not implement a graph cache or approve full training. Label
acquisition/mapping provenance and a transparent evaluation plan acknowledging
prior test exposure still need review. New independent final evaluation remains
a separate requirement; the audit does not fabricate a new holdout.

## 8. Did the Old Dataset Have Only One EPSS Value?

The legacy CSV-to-model path used **one scalar `epss_score` per record**, rather
than a daily history. The adapter passes it to the record, and
`CVEGraphDataset._label_from_record` returns that scalar in soft mode. This
does not prove every historical raw file had only one numeric EPSS-related
column, or that repeated CVEs could not occur in different rows/files.

One continuous target per CVE is sufficient for regression. A timeline is not
required. The legacy soft-mode model used continuous-score supervision with
weighted binary cross-entropy and classification-oriented reporting: actual
EPSS was thresholded at 0.1 and predictions at 0.5 by default. The older
binary KEV mode instead used a separate binary label.

The completed primary study also predicts one scalar, specifically Day-60
EPSS, but uses MAE and an explicit dated-label contract. The earlier
history-to-future studies are the time-series forecasting experiments. These
three settings should not be described as interchangeable based on F1 alone.
