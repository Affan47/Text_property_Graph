# CT06: Within-Version Fold and Support Preflight

Implemented and reproduced without training. CT05, all checkpoints, graph caches,
and official train/validation/test identities remain unchanged. This stage has
only `plan` and `audit` commands; it does not unlock training automatically.

## Question and Scope

CT05 is chronological, mixed-version development evidence. The new question is:
does adding accepted post text help estimate Day-60 EPSS when optimization,
checkpoint selection and assessment all use targets from the same EPSS version?

The primary dataset is Reddit with v2 Day-60 target dates, from 2022-02-04
inclusive to 2023-03-07 exclusive. The secondary dataset is the older collection
with v3 target dates, from 2023-03-07 inclusive to 2025-03-17 exclusive, limited
to its original training role. A CVE's version is assigned from its **target
date**, not the CVE ID year or the version range across its 91-day history.
The publishing schedule is recorded in the config; daily archive headers are
not rechecked by this preflight.

| Dataset | Original training CVEs | Within-version pool | Other versions retained outside this study |
|---|---:|---:|---:|
| Reddit v2 | 24,213 | 13,403 | 10,810 |
| Older v3 | 2,546 | 1,511 | 1,035 |

No records are deleted or assigned to a different official role. Other-version
records remain available for separately specified transfer studies. Neither the
official validation nor test cohort is used for selecting this new plan.

The source/model decision follows already inspected results. This is a new
exploratory study, not independent confirmation. One version removes a release
transition but does not remove changes in CVE populations, wording, post coverage
or score distributions over time. Text and CVSS remain retrospective snapshots.

## Roles and Evidence Separation

```text
Original training IDs only
    -> select one version by observed Day-60 target date
    -> freeze assessment starts at date-ordered 40%, 60%, 80% positions
    -> for each fold, retain the next assessment block
    -> remove earlier candidates linked to assessment evidence
    -> reserve latest 20% of remaining earlier dates for checkpoint selection
    -> remove optimization candidates linked to selection evidence
    -> publish role IDs and hashes BEFORE reading their score support
    -> audit counts without moving dates or enabling training
```

- **Optimization:** fits neural weights and structured preprocessing.
- **Selection:** selects the earliest strict best RMSE checkpoint, not the final result.
- **Assessment:** measures inner-fold performance after fitting; never selects weights or settings.
- **Purged:** excluded from that fold's earlier roles because they share an evidence group with a later role.
- **Future:** not yet eligible for that fold. These records may enter later folds.

All equal target dates stay together. Every optimization target strictly predates
every selection target, and every selection target strictly predates assessment.
This concerns target maturity, not proof that current text existed on those dates.

Evidence groups are the exact-normalized-document components already frozen by
CT05 across ALL original training records, including D/M/G/N documents. A path
through an out-of-version CVE still connects its within-version neighbors. This
prevents filtering from silently breaking a previously identified link. These
groups are conservative shared-evidence groups, not assertions of duplicate CVEs;
URLs alone do not create a link. No component crosses roles within a fold.

This is expanding chronological evaluation, not random k-fold cross-validation.
An earlier assessment record can become a fitting candidate in a later fold,
provided the later fold's chronology and component checks permit it. Assessment
IDs themselves appear only once in the combined out-of-fold results. Folds are
not independent replications.

## Verified Roles

| Dataset | Fold | First assessment target date | Optimization CVEs | Selection CVEs | Assessment CVEs | Assessment scores >=0.1 | Assessment CVEs with posts |
|---|---:|---|---:|---:|---:|---:|---:|
| Reddit v2 | 1 | 2022-05-31 | 4,228 | 1,072 | 2,680 | 27 | 305 |
| Reddit v2 | 2 | 2022-07-17 | 6,278 | 1,608 | 2,678 | 14 | 111 |
| Reddit v2 | 3 | 2022-09-06 | 8,371 | 2,116 | 2,695 | 21 | 418 |
| Older v3 | 1 | 2024-11-16 | 325 | 87 | 304 | 9 | 304 |
| Older v3 | 2 | 2025-01-16 | 553 | 174 | 272 | 3 | 272 |
| Older v3 | 3 | 2025-02-08 | 767 | 208 | 333 | 2 | 333 |

Each row is one fold, not an additional dataset. Earlier fitting sets expand,
so optimization/selection counts cannot be summed as unique CVEs. Assessment
counts can be summed because their IDs do not repeat.

| Potential assessment coverage | Reddit v2 | Older v3 |
|---|---:|---:|
| Unique assessment CVEs | 8,053 | 909 |
| Scores >=0.1 | 62 | 14 |
| CVEs with accepted posts | 834 | 909 |
| Scores >=0.1 with posts | 26 | 14 |
| CVEs without accepted posts | 7,219 | 0 |
| Evidence groups represented | 7,183 | 783 |
| Initial warmup CVEs, not assessed | 5,350 | 602 |

These are potential assessment counts; no predictions have been produced. The
starting pools' 213 Reddit and 87 older high scores are **not** the assessment
support. Many occur during warmup. Higher assessment coverage must not be
obtained by training on later records to predict earlier ones.

## Support Checks and Decision

The fixed minimum total counts are 200 optimization, 50 selection and 100
assessment CVEs per fold. Both datasets pass these checks in all three folds.
Optimization and assessment are additionally flagged for review when they have
fewer than 20 high or lower scores, 50 post-bearing CVEs, 20 post-absent CVEs for
the availability comparison, or 10 high-score CVEs with posts. These are review
thresholds, not statistical power calculations. Selection support is also shown
explicitly; a passed total-count gate does not imply robust checkpoint selection.

- Reddit fold 2 has 14 high-score assessment CVEs and only 5 with both a high
  score and posts. It fails the corresponding support review thresholds.
- Reddit selection folds contain 10, 14 and 11 high-score CVEs. Rare-event
  selection uncertainty remains even though their total counts pass.
- All older assessment CVEs have posts. Its flag-only arm cannot identify a
  post-present versus post-absent assessment effect; this limitation is recorded,
  not hidden by changing folds.
- Older assessment has only 9, 3 and 2 high-score CVEs by fold. The selection
  folds contain 7, 1 and 0 high-score CVEs. Treat it as a small secondary study,
  not equal-strength evidence to Reddit or a reason to pick a favorable fold.

Both reports therefore say `SUPPORT_REVIEW_REQUIRED`, while recording that the
minimum total-count gates passed. No fold is removed or redrawn. After these
counts are accepted, the next implementation is a new predictor/label adapter
and bounded smoke gate, not automatic full training. TPG training remains deferred.

## Proposed Matched Models

```text
D:  standalone D mean + [D,0,0,0] flags ---+
DF: standalone D mean + [D,M,0,0] flags ---+-> same pooled network
DM: combined DM mean + [D,M,0,0] flags ----+       + fold-fitted structure
                                                 -> one Day-60 EPSS estimate
```

All arms include the same CVSS/reference-feature branch and the same CVEs.
DF uses no post text. G/N summaries and all EPSS/VulnCheck answer fields are
excluded. Frozen SecureBERT means can be reused; standalone D must use its own
context, not D tokens extracted from a combined-context embedding. Structured
transforms must be fitted on each new optimization subset, never copied from
CT05's differently fitted transforms or full-training vectors.

The proposed specification retains CT05's raw MSE, width 32, batch size 8,
AdamW learning rate 0.001, weight decay 0.0001, seeds 17/29/43, at most 100 epochs,
patience 15, identical arm-wise shuffle orders, no refit and best-only checkpoints.
Optimization mean and median are required controls. A wider ridge grid is a
separate future specification, not enabled here.

Reddit-v2 DM-minus-D OOF RMSE is the primary contrast. DF contrasts and older-v3
results are secondary. Report all folds, all seeds, M-present/M-absent and score
tails, MAE, correlations, AUROC/AP and top-k high-score hits. Average per-seed
metrics; do not silently switch to ensemble predictions. Use 20,000 paired
bootstrap resamples at seed 6029 with both CVE and evidence-component units.
These conditional intervals do not include retraining uncertainty or undo the
previous research use of these records.

## Files and Outputs

- Config: `02_prediction/04_configs/07_fresh_studies/07_within_version_audit_v1.json`.
- Package: `02_prediction/01_package/epss/within_version_audit_v1/`.
- CLI: `02_prediction/03_analysis/07_fresh_studies/14_audit_within_version.py`.
- Tests: `02_prediction/03_analysis/07_fresh_studies/test_within_version_audit.py`.
- Results: `05_results/08_fresh_studies/12_within_version_audit_v1/01_reddit/v2/`
  and `02_previous_multisource/v3/`.
- `plan/fold_manifest.json`: all original training IDs, selected/excluded versions,
  exact role IDs, evidence groups, date cutoffs and input/code hashes. No scores.
- `plan/comparison_spec.json`: proposed matched models and evaluation rules;
  `training_enabled` remains false.
- `plan/fold_membership.csv`: one row per eligible CVE and fold, including purge
  and future roles, so every CVE is accounted for.
- `audit/findings.json` and `audit/REPORT.txt`: counts, warnings and explanations.
- `audit/fold_support.csv`, `source_strata.csv`, `support_gates.csv`: inspectable tables.
- `audit/fold_support.png`: horizontal position is chronological fold; bar height
  is the number of CVEs. Colors separate optimization, selection and assessment.
  Panels count all CVEs, scores >=0.1, and accepted-post CVEs; these are overlapping
  measures, not groups to add together.
- Each stage has `artifact_lock.json`; existing outputs are never overwritten.

Verified manifest SHA-256 values:

- Reddit: `568196721e6bfee5efd0e2577dab80dc452db2b155be8c958e53bf7d95eec241`.
- Older: `8d8042eee27cd9bc0c86a2c0f1abb518abcdb1d8eead49b005b0e6b51ca5121f`.

Sixteen focused tests cover boundary dates, ties, transitive evidence groups,
outcome-independent planning, held-out label exclusion, empty/weak roles,
missing/invalid targets, and synthetic end-to-end publication/reproduction.
Real plans and audits for both datasets also passed fresh-process verification.

## CPU Commands, One at a Time

These outputs already exist from implementation verification. Run verification
commands; omitting `--verify-only` would attempt to create a new output and
correctly refuse to overwrite it. No GPU is needed and no training is launched.

```bash
cd /home/ayounas/Text_property_Graph
```
```bash
conda activate CodeBERTFusion
```
```bash
RUNNER=EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/14_audit_within_version.py
```
```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/07_fresh_studies -p 'test_within_version_audit.py'
```
```bash
python "$RUNNER" --dataset both --stage plan --verify-only
```
```bash
python "$RUNNER" --dataset both --stage audit --verify-only
```

On an independent checkout with no CT06 outputs, create them using `--stage plan`
first and `--stage audit` second, without `--verify-only`, then verify both stages.
The parent CT05 data/config/code locks must match. This stage does not depend on
the external probe/checkpoint files in `/tmp`.
