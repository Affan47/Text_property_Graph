# VK02: Reviewed Source Policy and Classifier Smoke

**Subsequent status:** the full export is now complete. VK03 provides a separate
[frozen full-development baseline runner](03_FULL_DEVELOPMENT_BASELINES.md).
The original VK02 smoke remains unchanged and still cannot launch full training.

## 1. What Is Complete

VK02 adds a versioned source policy, separate predictor and label exports, and
a small matched classifier smoke. It does not replace VK01 or change any
original dataset, split, graph cache or earlier result.

The target is `in_vckev`: listed (1) or not listed (0) in the saved VulnCheck
snapshot. Not listed does not mean never exploited. This is not CISA KEV
classification or prediction of future catalogue inclusion. Current descriptions,
summaries, CVSS and reference counts remain retrospective inputs.

The source policy has **engineering review and regression tests**, not independent
semantic annotation. Its approval is limited to adapter and integration checks;
it does not establish that all remaining text is relevant or leakage-free. The
original VK01 pending review entries have not been relabelled as approved.

## 2. Source Policy v2

Each document receives a decision before model input is built:

| Case | Decision and reason |
|---|---|
| Catalogue statements, KEV phrases, CISA catalogue URLs or numeric EPSS disclosures | Quarantine the complete document as a potential target disclosure or proxy. A flagged phrase is not proof that the document is wrong. |
| Python definition `def vulncheck(...)` | Exempt only that function-name span. A separate VulnCheck mention in the same document still triggers quarantine. |
| Summary explicitly identifies a different CVE and never mentions its assigned CVE | Quarantine as a narrow relevance mismatch. |
| Other CVEs mentioned as context | Retain unless another rule applies. Distinct CVEs can legitimately describe related vulnerabilities. |
| Missing source | Preserve the CVE and mark that source absent; never fabricate replacement text. |

The reviewed mechanisms include the function-name collision in CVE-2023-23333,
the catalogue disclosure in the CVE-2024-9644 summary, the unrelated identifier
in the CVE-2019-1367 summary, and legitimate other-CVE context in CVE-2006-2450.
Regression fixtures test these mechanisms; they are not a new independent
annotation of the complete dataset. Rule decisions use text, not the target label.

This remains a conservative lexical screen. It cannot detect every paraphrased
disclosure or every wrong-product summary. Filtering itself can change source
availability differently for listed and unlisted CVEs. That concern remains
visible through a separate source-mask-only control.

## 3. Separate Inputs and Labels

The adapter exports only existing training and validation roles. It does not
export or reassign previously observed test or excluded records. EPSS history
and an observed Day-60 EPSS score are not requirements for this binary label.

| File | Contents and use |
|---|---|
| `inputs.json` | Accepted source-labelled text, source-presence masks and encoded structured features. CVE ID and role are join metadata, not numerical predictors. |
| `labels.json` | CVE ID, role and binary target; passed only to supervision and scoring. |
| `decisions.json` | Exclusion reasons, evidence spans and text hashes; audit only, never model input. |
| `structured_encoder.json` | Training-fitted structured encoding and feature names. |
| `structured_audit.json` | Missing or unsupported CVSS information. |
| `policy_review.json` | Versioned rules, bounded review fixtures and review limitations. |
| `summary.txt`, `summary.json` | Counts and scope. |
| `run_identity.json`, `artifact_lock.json` | Source/code/configuration identities and output checksums. |

Structured inputs are the existing CVSS v3 categorical indicators, score and
missing/validity information, plus the NVD reference count. There are 29 encoded
columns. Reference-count scaling is fitted on training records only. Unsupported
CVSS vectors are represented through the existing missing/validity mechanism,
not interpreted as valid CVSS v3 vectors.

EPSS values, catalogue fields and catalogue-derived features are not structured
predictors. Filtering reasons/counts are not predictors. The content classifiers
do not receive source masks; only the explicitly named mask diagnostic does.

## 4. Small Matched Training Check

The fixed seed is 17. Sampling preserves existing roles, uses approximately
proportional label strata, and requires at least two CVEs of each class. It is
an integration sample, not a representative performance study.

| Role | CVEs | Listed | Not listed |
|---|---:|---:|---:|
| Training | 64 | 8 | 56 |
| Validation | 32 | 2 | 30 |

The adapter examined 328 documents and accepted 318. A rejected document can
trigger multiple rules, so reason counts must not be summed as document counts.
All 96 CVEs retained a description. The structured audit reports two unsupported
or invalid CVSS-version cases and one missing vector; these flags may overlap.

Four logistic classifiers use exactly these CVEs:

1. Structured features only.
2. Description TF-IDF only.
3. Description TF-IDF concatenated with structured features.
4. Accepted-source presence masks only, as a shortcut diagnostic.

The fifth control always predicts the training listed fraction, 8/64 = 0.125.
TF-IDF vocabulary is fitted on training descriptions only, with unigrams and
bigrams and at most 4,096 features. Summaries and posts are exported and screened
but their text is not consumed by these description-based classifiers. No TPG,
GNN, transformer embedding or graph cache is trained, generated or loaded here.

The classifiers use fixed regularization, no class weighting and a fixed 0.5
decision threshold. No model, threshold or hyperparameter is selected from this
smoke. Checkpoints were saved, reloaded and independently verified against
saved validation predictions. All 33 VK01/VK02 tests pass.

## 5. Smoke Findings, Not Model Selection

| Model | Validation average precision | Validation log loss | F1 at 0.5 |
|---|---:|---:|---:|
| Training-prevalence constant | 0.0625 | 0.255151 | 0 |
| Structured only | 0.1292 | 0.236064 | 0 |
| Description TF-IDF | 0.0590 | 0.272909 | 0 |
| Description + structured | 0.1325 | 0.229563 | 0 |
| Source masks only | 0.2917 | 0.254168 | 0 |

Average precision measures how well listed CVEs rank ahead of unlisted CVEs;
larger is better. Log loss penalizes incorrect confidence; smaller is better.
With only two listed validation CVEs, neither supports a reliable ranking of
models. The mask control's high nominal ranking score is not a reason to select
it: source-availability shortcuts were precisely why this diagnostic was included.

Every model predicts below 0.5 for all 32 validation CVEs. Thus all have zero
recall and F1, 93.75% accuracy and 50% balanced accuracy. Precision is undefined
because no CVE is predicted listed; the saved metric is `null`, not a fabricated
precision value. This illustrates why accuracy alone is misleading. No threshold
was adjusted to improve these small-sample results.

`metrics.json` also contains ROC-AUC, confusion matrices, Brier score and
calibration bins. `diagnostics.json` records matrix sizes, fitting iterations,
training metrics and checkpoint checks. `predictions.csv` and `predictions.json`
contain actual binary labels and predicted scores for the same validation CVEs.

In `smoke_diagnostics.png`, the left panel names each model on the vertical axis
and shows its validation log loss on the horizontal axis. The right panel groups
CVEs by actual listing status horizontally and shows predicted scores vertically.
Small horizontal offsets separate models; they are not additional data values.

## 6. Locations and Commands

All paths below are relative to `EPSS_TPG`:

| Location | Purpose |
|---|---|
| `02_prediction/04_configs/02_vulncheck/02_adapter_smoke_v2.json` | Fixed adapter and smoke settings. |
| `02_prediction/01_package/epss/vulncheck_study/policy_v2.py` | Versioned document policy. |
| `02_prediction/01_package/epss/vulncheck_study/adapter_v2.py` | Separate exports, train-only structured fitting and verification. |
| `02_prediction/01_package/epss/vulncheck_study/classifier_smoke_v2.py` | Small classifiers, metrics and reload checks. |
| `04_data/01_records_and_graphs/vulncheck_snapshot_v2/smoke/` | Prepared 96-CVE integration data. |
| `05_results/05_vulncheck_study/03_classifier_smoke_v2/` | Completed smoke models, predictions and reports. |

From `/home/ayounas/Text_property_Graph`, activate `CodeBERTFusion`. The smoke
already exists; verify it without retraining:

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/03_prepare_and_smoke_v2.py --stage prepare --scale smoke --verify-only
```

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/03_prepare_and_smoke_v2.py --stage smoke --scale smoke --verify-only
```

On a clean checkout with no VK02 artifacts, run the same commands without
`--verify-only` in that order. Existing results cannot be overwritten.

The next preparation-only step is available, but was not run during this task:

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/03_prepare_and_smoke_v2.py --stage prepare --scale full
```

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/03_prepare_and_smoke_v2.py --stage prepare --scale full --verify-only
```

This would export the existing 2,870 training and 1,104 validation CVEs, not train
them. Full classifier training is deliberately disabled. Review full source
coverage, filtering by class and residual shared text before defining the larger
experiment. Prior test exposure, label-acquisition provenance and an appropriate
final evaluation plan remain unresolved; changing targets does not reset them.
