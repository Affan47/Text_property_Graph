# CT08: Is a New VulnCheck Target Ready?

## Decision

The next step is a **training-only label and source audit**, not another fit.
CT07 remains frozen. It did not establish a post-text benefit under its reporting
rules. Changing to a catalogue target is a different research question, not a fix
for the EPSS regression loss.

The implemented audit ran and reproduced on both datasets. Seventeen focused
tests passed. No network requests, embeddings, training, split changes, official
validation scoring or test scoring were performed.

New training is **blocked pending label-time and source review**. This does not
mean that every saved flag is wrong. It means that the evidence inspected here
does not justify releasing a newly verified classification dataset.

## What the Target Would Mean

| Proposed task | Positive label would mean | What it would not mean |
|---|---|---|
| Snapshot classification | The CVE is included in one defined VulnCheck KEV snapshot | Probability, ease or frequency of exploitation; automatic CISA KEV inclusion |
| Future catalogue inclusion | A CVE not already listed at the prediction time enters VulnCheck during the following 30, 60 or 90 days | First exploitation or first public disclosure |

VulnCheck provides exploitation-related catalogue evidence. It is not an
exhaustive truth table for every exploited and unexploited vulnerability.
A verified absence means **not listed at that check**, not **never exploited**.
The saved labels remain in the audit alongside, not replaced by, verification
states. Future-horizon labels are null, not zero, while the requirements remain
unresolved.

## Training-Role Scope

| Dataset | Existing EPSS training CVEs | Catalogue-audit candidates | Additional candidates | Saved listed | Saved false |
|---|---:|---:|---:|---:|---:|
| Reddit | 24,213 | 24,599 | 386 | 544 | 24,055 |
| Older multisource | 2,546 | 2,870 | 324 | 376 | 2,494 |

Each row describes only the corresponding dataset's training-role candidates.
The last two columns count saved flags among **all audit candidates**, not
authenticated new labels. In the original EPSS subsets, the saved listed counts
are still 537 and 303, respectively.

A catalogue classifier does not need a Day-60 EPSS target. The older scope uses
the already saved 2,870 classifier-training IDs. Reddit applies the frozen RD01
publication intervals, evidence groups and prior-role restrictions **before**
the EPSS observation and Day-60 target-maturity requirements. All prior-test
connections and cross-role groups remain excluded. This audit does not approve
new splits or expand Reddit to publication years outside the frozen envelope.

Of the 386 additional Reddit candidates, 317 actually have a saved Day-60 score:
missing scores alone do not explain their absence from the EPSS task. The older
324 additional candidates have no observed Day-60 score. These are scope facts,
not reasons to change completed experiments.

The tool reads full-file bytes for integrity and parses the saved JSON container.
Only IDs/publication metadata are used for role checks outside the selected
training rows. It does not inspect or export held-out text, outcomes or metrics.
The older dataset's previously completed test evaluation remains part of its
history; nothing here resets that cohort to unseen.

## Why Saved False Is Not Yet a Verified Negative

The configured Reddit receipt directory does not exist. No receipt directory
was supplied for the older dataset. Consequently, **zero candidates have a
usable receipt in the inspected locations**. This is not a filesystem-wide
claim that no receipt exists anywhere.

The audit supports an existing receipt directory as an explicit override. It
checks the source/collector identity, response hashes, requested CVE, catalogue
index, timestamps and latest attempt. Failed, incomplete, wrong-CVE or malformed
responses stay unknown. A negative requires a successful, explicitly complete
empty result. Hashes establish integrity relative to recorded files; they are
not independent proof that a server response or historical record is authentic.

The saved Reddit collector previously conflated request failures with absence.
That is why simply reusing every `false` flag is unsafe. The audit does not assume
that the same defect explains every old-dataset negative.

Each output includes a deterministic, **20-CVE training-only request plan**:
10 saved-listed and 10 saved-false cases. No requests are sent. This is a future
collection-path check, not a prevalence sample or whole-cohort verification.
A fresh response describes its own check time and must never silently overwrite
the older snapshot. A prospective collection step must explicitly enforce this
training allowlist; the generic old collector's default sample is not approved
for this study.

## Publication and Addition Times

The delta is the saved VulnCheck addition timestamp minus saved NVD publication.
Both timestamps describe catalogue/database events, not first exploitation.

| Population | Earlier day | Earlier time on publication day | Nonnegative stored delay through 60 days | Dates only: 0 through 60 days |
|---|---:|---:|---:|---:|
| Reddit, all 24,599 candidates | 119 | 22 | 44 | 66 |
| Reddit, original 24,213 EPSS training CVEs | 117 | 21 | 42 | 63 |
| Older, all 2,870 candidates | 83 | 54 | 82 | 136 |
| Older, original 2,546 EPSS training CVEs | 70 | 48 | 69 | 117 |

The first two columns retain negative differences; they are not automatically
errors. The final column ignores time of day, so it includes same-day cases
whose stored timestamp difference is negative. Neither of the final two
columns is an approved training target count.

For example, Reddit's `CVE-2021-31199` has publication
`2021-06-08T23:15:08.360` and addition `2021-06-08T00:00:00Z`. Its stored delta is
about **-0.969 days**. The entries refer to the same date, but their exact order
cannot be established by assuming that midnight is a genuine recorded event
time. In the older data, `CVE-2010-3081` has publication
`2010-09-24T20:00:02.183` and addition `2010-09-15T00:00:00Z`: about **-9.833 days**.
Missing time of day alone cannot explain a difference that large.

The audit assumes UTC for timezone-less NVD timestamps and records that
assumption. Date-only or midnight addition values are conservatively represented
as uncertain times within that day. It evaluates stored-time scenarios at
publication and 30 days later, with horizons of 30, 60 and 90 days. A future
event must be strictly after the anchor and at or before the horizon end.
Already-listed, boundary-uncertain, missing and unverified-absence cases remain
separate. Follow-up completeness and the event-time semantics still require
review; no scenario is silently released as a label.

## Text and Structured Features

| Raw saved source | Reddit candidates with text | Older candidates with text |
|---|---:|---:|
| D: CVE description | 24,599 | 2,870 |
| G: generated GitHub webpage summary | 0 | 590 |
| N: generated non-GitHub webpage summary | 0 | 1,995 |
| M: social posts, including bare IDs | 24,599 | 2,868 |

These are raw-source counts, **not accepted model-input counts**. Separately
joined Reddit summary overlays are outside this audit; the zero entries do not
mean that the earlier summary-joining work never existed.

For M, 24,289 Reddit candidates and 1,787 older candidates have at least one
stored date at or before Day 30. None has an audited exact-text historical
lineage under the inspected contract. A post's date does not prove that its
currently saved wording existed then. ID, capture and edit-history field
presence is inventoried but is not itself verification. D/G/N likewise lack
an audited archive/content-version link for the precise saved text.

The reused lexical screen flags direct catalogue/EPSS disclosures and explicit
wrong-CVE summary cues. It is not a complete content or relevance review.
For example, 2,075 Reddit CVEs in the original EPSS training subset retain
non-bare M passing this screen, compared with the previous 2,077 prose-source
count under a different policy. The saved EPSS masks are not changed.

All 11 catalogue fields (the label plus ten VulnCheck/CISA metadata fields)
are excluded from future predictor candidates, as are their missingness flags,
counts, graph nodes and graph edges. CVSS and non-catalogue reference counts
can be reviewed as structured candidates. Present-day values are not
automatically valid historical inputs. EPSS scores/metadata are excluded from
the initial proposed catalogue baseline so that it is a distinct task.

## Artifacts and Replay

The output root is
`05_results/08_fresh_studies/15_catalogue_readiness_v1/`.
Dataset subdirectories are `01_reddit/audit/` and
`02_previous_multisource/audit/`.

- `REPORT.txt`, `findings.json`: conclusions and exact counts.
- `cve_readiness.csv`: one training-candidate CVE per row; saved flag, receipt
  state, dates, delta and blocking reasons. **Audit only, not model input.**
- `cve_horizon_scenarios_audit_only.csv`: per-CVE anchor/horizon scenarios;
  released targets remain empty. `horizon_scenarios.csv` contains their totals.
- `source_documents_audit_only.csv`: exact-text hashes, source/platform and
  provenance/screen flags. Raw text is not exported.
- `cohort_support.csv`, `timing_categories.csv`, `source_coverage.csv`: tables.
- `real_timing_examples.json`, `future_feature_policy.json`,
  `receipt_request_plan.json`: review aids, not a runnable experiment.
- `training_scope_ids.json`, `identity.json`, `artifact_lock.json`: candidate
  IDs, SHA-256 source/code pins and output hashes.

CSV rows are canonically sorted with LF line endings. Fresh-process verification
recomputes the outputs and checks every output hash and pinned input. Existing
outputs cannot be overwritten. The focused tests cover event boundaries,
timezone handling, malformed/failing receipts, response tampering, later failed
attempts, role exclusion, missing EPSS independence, text-date limitations,
deterministic CSV, end-to-end replay and overwrite refusal.

Run these commands individually. They use **CPU only** and do not train:

```bash
cd /home/ayounas/Text_property_Graph
```
```bash
conda activate CodeBERTFusion
```
```bash
RUNNER=EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/19_audit_catalogue_readiness.py
```
```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/07_fresh_studies -p 'test_catalogue_readiness.py'
```
```bash
python "$RUNNER" --dataset both --check-only
```
```bash
python "$RUNNER" --dataset both --verify-only
```

The reports already exist. On a clean checkout without these reports, omit
`--verify-only` once to create them. To inspect additional receipts later, choose
one dataset, an explicit `--receipt-dir`, and a **new** `--output-dir`; preserve
this original audit. Do not run another classifier until the label/source
decision and a separate experiment protocol are approved.
