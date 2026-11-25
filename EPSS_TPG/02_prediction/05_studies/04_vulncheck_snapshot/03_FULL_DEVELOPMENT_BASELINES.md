# VK03: Full Development Baselines

**Completed after preparation:** all eight full fits and the prevalence reference
now reproduce. Read the [verified results](04_BASELINE_RESULTS.md). The preparation
record and original frozen settings below are retained for reproducibility.

## 1. Ready for Full Baseline Training

The full VK02 export is prepared and independently reproduced. It contains
2,870 training CVEs and 1,104 validation CVEs, with no subsampling. The VK03
source audit and eight-fit protocol are frozen. No further dataset smoke run
is a prerequisite. Training is implemented and ready for the user to launch;
it has not been run on these development records during preparation.

This is full **baseline** training, not TPG-GNN training. It establishes the
structured and description controls against which graph models can later be
compared. It does not promise improved F1 or a useful classification model.

## 2. Export and Source Checks

| Role | Total CVEs | Listed in VulnCheck | Not listed in this snapshot |
|---|---:|---:|---:|
| Training | 2,870 | 376 | 2,494 |
| Validation | 1,104 | 74 | 1,030 |

The source screen examined 13,260 documents, excluded 452 and retained 12,808.
All 3,974 descriptions remain present. No CVE was dropped by source filtering.
The full structured audit records 22 missing-vector flags and 67 unsupported
or invalid CVSS-version flags; these are flags, not necessarily disjoint CVEs.

### Filtering by Class

| Role and class | CVEs | CVEs with at least one excluded document | Percentage of that class |
|---|---:|---:|---:|
| Training, listed | 376 | 112 | 29.79% |
| Training, not listed | 2,494 | 64 | 2.57% |
| Validation, listed | 74 | 24 | 32.43% |
| Validation, not listed | 1,030 | 23 | 2.23% |

These are CVE counts, not counts of excluded documents. One CVE can have many
documents and one document can trigger several rules. The difference by class
means source availability after filtering can itself carry a shortcut signal.
It is not evidence that all excluded documents were false or that the filter
caused every source-availability difference.

### CVEs Retaining Each Source

| Role and class | Description | GitHub summary | Non-GitHub summary | Social/related posts |
|---|---:|---:|---:|---:|
| Training, listed (376) | 376 | 55 | 302 | 335 |
| Training, not listed (2,494) | 2,494 | 531 | 1,641 | 2,473 |
| Validation, listed (74) | 74 | 17 | 52 | 74 |
| Validation, not listed (1,030) | 1,030 | 184 | 689 | 1,022 |

Each cell counts CVEs with at least one accepted document from that source.
The CSV audit also includes source availability before filtering. For example,
41 listed training CVEs lost all their posts, compared with 19 unlisted training
CVEs. In validation, no listed CVE lost all posts, whereas eight unlisted CVEs
did. A mask-based relationship learned in training may therefore transfer poorly.

Accepted wording crosses the training/validation boundary in seven description
groups and 57 post groups, involving 86 distinct validation CVEs across sources.
Matching uses case folding and whitespace collapse within each source. It does
not establish that the CVEs are duplicates or the same vulnerability. No CVEs
were merged and no roles were changed. These overlaps limit interpretation of
validation independence. Summary/post text is not used in the present classifiers.

## 3. The Frozen Comparison

| Input type | What the classifier receives | Purpose |
|---|---|---|
| Structured only | 29 existing encoded CVSS/reference-count features | Non-text reference. |
| Description only | Training-fitted TF-IDF description features | Simple non-graph text reference. |
| Description + structured | Concatenated TF-IDF and structured vectors | Test whether the two input groups complement one another. |
| Source masks only | Four indicators of accepted-source presence | Diagnose collection/filtering shortcuts, not semantic understanding. |

Each input type has two predeclared logistic fits: ordinary class weights and
balanced weights calculated from training labels alone. This gives **eight
fits**, plus a constant predicting the training listed fraction (376/2,870).
The ordinary-weight fits are primary controls; balancing is a fixed sensitivity
comparison, not a parameter search.

Balanced weights are 0.5753809142 for not listed and 3.8164893617 for listed.
The formula is `number of training CVEs / (2 * training CVEs in that class)`.
These weights change the objective, not the number of CVEs. No validation
information determines weights. Weighted scores are not automatically calibrated
probabilities of listing, let alone probabilities of exploitation.

Fixed settings: logistic regression with `liblinear`, `C=1`, maximum 300 solver
iterations, tolerance 0.0001, seed 17 and two CPU threads. TF-IDF uses unigrams
and bigrams, sublinear term frequency and at most 4,096 vocabulary features.
The vocabulary is fitted on training descriptions only; structured preprocessing
was already fitted on training records during export. Convergence warnings stop
the run rather than silently accepting an unfinished optimization.

No neural epochs, early stopping, seed search or additional graph processing is
involved. The decision threshold is fixed at 0.5. There is no validation threshold
tuning, probability recalibration or automatic winner selection. The current
decision to freeze these settings follows prior development; it is not an
untouched, preregistered evaluation.

Report average precision, ROC-AUC, F1, precision, recall, accuracy, balanced
accuracy, confusion matrices, log loss, Brier score and calibration bins. Undefined
precision stays null. All models score the same 1,104 validation CVEs. Prediction
tables retain each actual label and score. The figures show precision-recall,
ROC, average precision and calibration. No statistical-significance claim or
independent-test claim is produced by this runner.

## 4. Data and Evaluation Boundaries

- Target: listed/not listed in the saved VulnCheck KEV snapshot, not CISA KEV,
  never/ever exploited or future inclusion within a horizon.
- EPSS values and catalogue-derived structured fields are not predictors.
- Exclusion reasons/counts remain audit-only. Source masks enter only their
  explicitly separate diagnostic classifier.
- Description text is used; summaries and post text are screened/exported but
  not passed into these classifiers. No TPG cache or embeddings are loaded.
- Current CVSS, reference counts and descriptions are retrospective, not
  authenticated historical features.
- Previous test and excluded roles stay outside this export and runner. The
  earlier test set has already been observed; changing the target cannot reset it.
- Independent semantic annotation and upstream label-acquisition validation
  remain incomplete. The user-requested comparison is exploratory development,
  not approval of final evaluation or proof that the sources are leakage-free.

## 5. Run One Command at a Time

The full export and frozen protocol already exist. From the repository root:

```bash
conda activate CodeBERTFusion
```

Verify the frozen audit and settings:

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/04_run_development_baselines_v1.py --stage freeze --verify-only
```

Check the full training prerequisites without fitting:

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/04_run_development_baselines_v1.py --stage train --check-only
```

**Start full baseline training on all 2,870 training CVEs:**

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/04_run_development_baselines_v1.py --stage train
```

After completion, independently reload and verify every fitted model:

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/04_run_development_baselines_v1.py --stage train --verify-only
```

The progress bars cover input checks, vocabulary fitting, model fits, reports
and checkpoint reproduction. Each fit prints feature dimensions, class weights,
solver iterations, training loss and validation AP/F1. Failed runs do not publish
partial results; completed results cannot be overwritten. The eight-fit runner
does not provide resume or arbitrary setting overrides.

## 6. Files and Verification

Paths below are relative to `EPSS_TPG`:

| Location | Purpose |
|---|---|
| `04_data/01_records_and_graphs/vulncheck_snapshot_v2/full/` | Separate inputs, labels, structured encoder and document decisions for all development CVEs. |
| `05_results/05_vulncheck_study/04_baseline_plan_v1/` | Frozen protocol, class-wise filtering JSON, source-coverage CSV, shared-text audit and readable summary. |
| `05_results/05_vulncheck_study/05_baseline_development_v1/` | Created by training: eight checkpoints, prediction CSV/JSON, metric/diagnostic JSON, summary TXT/JSON and figure. |
| `02_prediction/04_configs/02_vulncheck/03_development_baselines_v1.json` | Fixed settings and interpretation. |
| `02_prediction/01_package/epss/vulncheck_study/development_v1.py` | Audit, protocol freeze, full fitting and verification implementation. |
| `02_prediction/03_analysis/03_vulncheck_study/04_run_development_baselines_v1.py` | CLI entry point. |

All 42 VK01/VK02/VK03 tests pass. New synthetic tests cover every fit type,
checkpoint reloads, validation-label isolation, training-only weights/vocabulary,
predictor/label joins, duplicate audit decisions, class-level counts and changed
output detection and CSV export/reproduction. Synthetic fits are not results
for the real CVE cohort. Full protocol reproduction and training preflight passed.

Next, analyze this full baseline comparison rather than running another small
smoke. It will establish the reference performance for a separately frozen
TPG-GNN comparison on the same development CVEs. Final evaluation still needs
an explicit plan that acknowledges earlier test exposure.
