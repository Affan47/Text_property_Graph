# CT04: Training-Only Chronological Fold Feasibility

## What This Step Does

This is an audit, not a new training run. It freezes inner folds inside each
dataset's existing training role, then checks whether those folds have enough
examples for the proposed pooled-description versus description-plus-post
comparison. Official validation and test roles remain unchanged.

The plan and the support audit were executed and independently reproduced for
both datasets. No new predictions, embeddings, graphs or checkpoints were
created. No test records were adapted. Shared development containers are read
for integrity, but only training CVE dates, accepted text and targets enter
the analysis. Official validation text, features and scores are not used.

## Frozen Rules

1. Order the training CVEs by the **date of the Day-60 EPSS score being
   estimated**. This is neither the year in the CVE identifier nor a Reddit
   post date. It is a retrospective dated-score task, not Day-0 forecasting.
2. Take date cut points at the fixed 40%, 60% and 80% positions in that ordered
   list. Keep every equal date on the same side. These fractions and all
   support rules were written before fold outcomes were counted.
3. Use three expanding windows. Each fitting set has target dates strictly
   before its assessment block. Assessment intervals include their start and
   exclude their end; the final end is one day after the latest training target.
4. Record two policies: chronological only, and chronological with conservative
   shared-evidence exclusions. The latter removes fitting CVEs whose accepted
   text component also touches the assessment set. Assessment IDs do not move.
5. Connect documents only when their accepted text matches after Unicode NFC
   and whitespace normalization. Preserve case, punctuation and CVE numbers.
   Group across D/M/G/N sources. Do not create links from shared URLs, CVE
   mentions or a vendor name alone. This is an evidence-overlap constraint,
   not a declaration that the vulnerabilities are duplicates.
6. Do not change dates, drop an assessment fold, or balance outcomes after
   seeing its counts. Empty or weak folds are reported explicitly.

Grouping uses connected components across the original training cohort. A
chain through another training record can therefore conservatively exclude a
fitting CVE even without a direct matching document in that fold. This does
not detect every paraphrase or vulnerability family. It is not a blanket
guarantee of independence or historical feature availability.

| Dataset | Assessment fold 1 starts | Fold 2 starts | Fold 3 starts | Final exclusive end |
|---|---|---|---|---|
| Older multisource | 2023-01-30 | 2024-09-07 | 2025-01-27 | 2025-03-02 |
| Reddit | 2022-01-09 | 2022-04-26 | 2022-08-01 | 2024-01-01 |

The unequal time spans reflect the frozen training cohort's date distribution.
No EPSS magnitude or M-presence count was used to choose these dates.

## Support Rules and Verified Counts

Operational review thresholds are 200 fitting CVEs, 100 assessment CVEs,
20 high-score and 20 lower-score CVEs in each role, and 50 accepted-post CVEs
in each role. U comparisons separately require 50 summary-bearing CVEs per
role. These are review thresholds, not a power calculation or a promise that
a confidence interval will exclude zero. All source and target intersections
are also reported, even where they have no separate pass/fail threshold.

High-score means Day-60 EPSS >= 0.1. M means accepted prose under the existing
filter, not every raw social-media attachment. U means an accepted GitHub or
non-GitHub webpage summary.

The following table uses the grouped policy. Each assessment remains a subset
of the original training role, not the official validation set.

| Dataset | Fold | Fitting CVEs after exclusion | Excluded fitting CVEs | Assessment CVEs | Assessment high-score CVEs | Assessment M-present CVEs |
|---|---:|---:|---:|---:|---:|---:|
| Older | 1 | 942 | 73 | 512 | 63 | 508 |
| Older | 2 | 1,283 | 244 | 508 | 22 | 507 |
| Older | 3 | 1,772 | 263 | 511 | 2 | 511 |
| Reddit | 1 | 9,534 | 131 | 4,832 | 324 | 550 |
| Reddit | 2 | 14,360 | 137 | 4,781 | 48 | 430 |
| Reddit | 3 | 18,903 | 375 | 4,935 | 41 | 669 |

- Older fold 3 fails the declared high-score support threshold. It still has
  enough records for a descriptive regression evaluation. Retain and flag it;
  do not silently combine it with another fold to improve counts.
- All three Reddit folds pass the overall regression, high-score and M-count
  gates. This is feasibility evidence, not automatic training approval.
- Joint support is smaller: Reddit fitting fold 1 has only eight CVEs that
  are both high-score and M-present. Assessment intersections are 41, 18 and
  25. Do not infer well-supported subgroup effects from the marginal counts.
- Reddit summary coverage fails the separate U gate in folds 1 and 3. This
  does not block a D-versus-DM comparison, but it does not justify expanding
  that comparison to URL-summary arms.
- Group exclusions can remove substantial high-score fitting evidence: the
  older dataset loses 17, 52 and 51 high-score fitting CVEs in folds 1, 2 and 3.
  Those exclusions follow text relationships, not target values.

## Out-of-Fold Coverage

No out-of-fold predictions have been produced yet. A future runner must save
one prediction per covered assessment CVE per seed, then report both pooled
and per-fold results. Pooling must not hide changes over time or use an
assessment fold for epoch selection.

| Dataset | Original training CVEs | Potential assessment-covered CVEs | High-score CVEs covered | Warmup-only CVEs | High-score warmup CVEs |
|---|---:|---:|---:|---:|---:|
| Older | 2,546 | 1,531 | 87 | 1,015 | 45 |
| Reddit | 24,213 | 14,548 | 413 | 9,665 | 310 |

The older plan therefore does **not** assess all 132 high-score training CVEs.
The initial fitting period cannot receive honest forward-chain predictions
from a model fitted on later records. These counts describe coverage if all
three folds are retained, not a selected subset of successful folds.

## Required Next Training Contract

No training CLI is enabled by this audit. The intended next implementation is
a bounded, pooled MSE comparison using standalone-context D versus DM on these
fixed folds, before any GNN expansion. It must:

- Read D vectors from D graphs. D tokens extracted from DM embeddings are not
  a standalone-description control. The plan verifies the source spans of
  the existing D and DM graph entries; full tensor/embedding replay remains a
  prerequisite of the future training adapter.
- Refit structured numeric scaling and categorical vocabularies inside every
  fitting fold. Existing vectors fitted on all outer-training CVEs cannot be
  treated as fold-fitted preprocessing. Frozen, label-free transformer weights
  may be reused.
- Keep inner assessment labels out of model selection. Use a further
  chronological selection split inside the fitting data, or a separately
  frozen epoch budget. Do not call best-assessment-epoch predictions unbiased
  out-of-fold predictions.
- Keep the fixed ridge result visible. Any new ridge penalty selection must
  use fitting-only folds and be recorded as a separate control.
- Use identical assessment CVEs for all compared arms, retain missing-source
  masks, and publish M-present/M-absent results alongside overall results.
- Use at least 10,000 CVE resamples; this specification fixes 20,000 and seed
  6029. Report grouped sensitivity where shared evidence induces dependence.
- Preserve official validation/test roles and all previous results. These
  inner folds reuse previously trained-on records and do not create a new,
  untouched test set.

## Numerical Reproducibility

The new CLI pins BLAS and OpenMP environment variables to four threads before
numerical imports. It also enforces BLAS limits and records actual thread-pool
libraries, versions, thread counts, Python and NumPy in the plan identity.
Fresh-process verification refuses changed code, inputs or runtime identity.
No environment export is needed for this CLI. The old CT03 runner remains
unchanged and still needs its documented thread exports.

## Files and Commands

All paths below are relative to `EPSS_TPG`:

- `02_prediction/04_configs/07_fresh_studies/04_training_fold_audit_v1.json`:
  frozen rules and minimum support; no model-training mode.
- `02_prediction/01_package/epss/training_fold_audit_v1/`: pure fold logic and
  training-only data projection, audit, plots and immutable publication.
- `02_prediction/03_analysis/07_fresh_studies/08_audit_training_folds.py`:
  CPU CLI, `plan` then `audit`; `--verify-only` reproduces completed artifacts.
- `05_results/08_fresh_studies/10_training_fold_audit_v1/`: separate
  `01_reddit` and `02_previous_multisource` folders. Each has `plan/` and
  `audit/`. Read `audit/REPORT.txt` first, then `findings.json` and
  `fold_support.csv`. `source_strata.csv` gives post-present/absent support;
  `fold_membership.csv` accounts for every training CVE in every fold and policy.
- `plan/fold_manifest.json`: exact inner IDs, dates, text components, source
  masks and hashes; no EPSS magnitudes or raw text. `fold_support.png` plots
  CVE counts on y versus inner fold on x, separating fitting and assessment.

Manifest SHA-256 values:

```text
Older: 49f85947d98911130fd903b4451d395704c9acebb62b272faab1c41e75a55767
Reddit: 9d782e64a1183b90b5976ab92d90b81d3162a41e50694cfd54a14d3bbbbf22e6
```

Run individually, from the project root with `CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/07_fresh_studies -p 'test_training_fold_audit.py'
```

```bash
python EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/08_audit_training_folds.py --dataset both --stage plan --verify-only
```

```bash
python EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/08_audit_training_folds.py --dataset both --stage audit --verify-only
```

For a genuinely missing output, omit `--verify-only`, running `plan` before
`audit`. Existing outputs are never overwritten. These commands use CPU only;
they neither require a working GPU nor start training.
