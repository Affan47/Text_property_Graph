# VK01 Full Readiness Findings

## 1. Verification and Scope

The full VK01 audit reproduced from the pinned source. Its 13 saved artifacts
passed their hash checks. A separate interpretation recomputed quarantine,
source coverage, label-associated filtering and before/after shared-text counts.
No training, predictions, split changes or edits to the original audit occurred.

The numerical analysis covers all 5,968 CVEs and 20,605 deduplicated source
documents. The saved bounded packet contains 25 review entries, not a random
sample for estimating extraction accuracy. An automated review examined these local
excerpts and checked selected descriptions against the raw file. This is not
independent annotation or an online validation of their claims; the original
review statuses remain pending.

## 2. Labels Are Complete, but Classes Differ Across Roles

| Previous role | CVEs | Listed in VulnCheck | Not listed | Listed percentage |
|---|---:|---:|---:|---:|
| Training | 2,870 | 376 | 2,494 | 13.10% |
| Validation | 1,104 | 74 | 1,030 | 6.70% |
| Previously observed test | 651 | 36 | 615 | 5.53% |
| Previously excluded | 1,343 | 62 | 1,281 | 4.62% |
| Total | 5,968 | 548 | 5,420 | 9.18% |

Every stored label is boolean. Not listed means absent from this snapshot,
not never exploited. These are existing roles, not a newly approved evaluation
split. The lower proportion of positives in later roles matters for precision,
F1, average precision and threshold selection; it is not a model result.

The 324 training-role CVEs without Day-60 EPSS labels have valid VulnCheck
labels. A binary catalogue study therefore does not need to discard them for
missing EPSS. The 1,343 previously excluded CVEs are not automatically restored:
their role and a new evaluation policy need an explicit decision.

## 3. Most Text Survives the Candidate Screen

| Source | Documents examined | Candidate flagged documents | CVEs with raw text | CVEs retaining text |
|---|---:|---:|---:|---:|
| Description | 5,968 | 1 | 5,968 | 5,967 |
| GitHub summary | 1,270 | 4 | 1,270 | 1,266 |
| Non-GitHub summary | 4,180 | 68 | 4,180 | 4,112 |
| Posts | 9,187 | 708 | 5,966 | 5,879 |

There are 781 candidate flagged documents, 3.79% of the 20,605 documents,
across 316 CVEs. After candidate filtering, 19,824 documents remain and no CVE
loses every text source. Several posts can belong to one CVE, so document losses
and CVE losses are different counts. Missing GitHub summaries are primarily
a source-availability issue, not caused by this filter.

The 494 KEV mentions, 258 known-exploited-catalogue phrases, 86 numeric EPSS
cues, 30 VulnCheck mentions and one CISA catalogue URL overlap within documents;
do not add these counts to estimate distinct removed documents. A flagged cue
is not proof that the whole document leaks the target.

## 4. Concrete Source Problems and Valid Context

- **Direct disclosure:** CVE-2024-9644's saved non-GitHub summary explicitly
  says it is listed in the VulnCheck KEV database. This supplies the answer to
  the proposed classifier rather than independent vulnerability evidence.
- **A lexical false-positive mechanism:** the CVE-2023-23333 post excerpt
  contains `def vulncheck(ipaddress, port)`. The shown occurrence is a function
  name, not a catalogue assertion. This does not clear the entire document,
  but shows why a bare keyword match should not be called verified leakage.
- **Summary alignment failure:** CVE-2019-1367's saved description concerns
  the Internet Explorer scripting engine. Its non-GitHub summary instead
  describes an HTTP protocol stack and explicitly names `CVE-2023-12345`.
  This remains a relevance problem even without a catalogue keyword.
- **Legitimate multi-CVE context:** CVE-2006-2450's description explains that
  it is a different issue from CVE-2006-2369. Removing all text that mentions
  another CVE would discard useful evidence.
- **Separate catalogue claims:** CVE-2015-2291's summary mentions CISA KEV
  and an EPSS percentage. These are disclosure/proxy candidates, but CISA is
  not VulnCheck and this inspection does not verify the summary's external claims.

The 3,227 documents mentioning other CVEs and 1,079 containing exploitation
phrases are review flags, not counts of false documents or verified exploits.

## 5. Filtering Is Associated with the Label

Within the existing training role, 111 of 376 listed CVEs (29.52%) have at least
one flagged document, compared with 64 of 2,494 unlisted CVEs (2.57%). Removing
flagged documents makes 41 listed and 19 unlisted training CVEs lose all posts.

This is not classifier performance and does not establish causality. However,
it is a reason to inspect source masks carefully: after filtering, missing
sources could become a label-associated shortcut. Do not expose quarantine
reasons or removed-document counts as predictors. A later missingness-only
control would help distinguish content benefit from source-availability effects.

## 6. Shared Evidence Remains After Filtering

The raw audit found 828 shared-text groups, including 249 crossing previous
roles and involving 725 CVEs. Recomputing overlap only on candidate accepted
documents leaves 188 cross-role groups involving 594 CVEs.

| Pair of previous roles | Remaining shared-text groups | CVEs in first role | CVEs in second role |
|---|---:|---:|---:|
| Training / validation | 64 | 139 | 86 |
| Training / old test | 35 | 61 | 61 |
| Validation / old test | 24 | 28 | 47 |

These pairwise counts overlap and do not sum to the overall totals, which also
include the previously excluded role. Sharing text does not make different
CVEs the same vulnerability. It does limit claims of evidence-independent
evaluation, and ordinary CVE-level intervals may understate that dependence.

## 7. Dates and Structured Features

All 548 listed CVEs have parseable addition timestamps; no unlisted record
has a usable addition timestamp. There are still 219 negative publication-to-
addition differences, including 104 on the same day. Retain these records for
snapshot classification; do not turn them into ordinary future-inclusion
labels. Exact capture provenance and event meaning remain unresolved.

Seventy-seven CVEs lack the current CVSS vector, score and version. All 5,968
have reference-count values. These are presence checks, not a claim that all
nonempty CVSS vectors pass the supported encoder or existed at an earlier date.
The adapter must validate version/vector agreement and preserve missingness.

## 8. Recommended Next Step

The data support a **retrospective binary-label study**, not an immediate claim
of future exploitation prediction. Do not launch the 48 proposed neural fits yet.

The next bounded implementation should version and test the source policy,
resolve the demonstrated keyword ambiguity, retain an explicit relevance-review
path for mismatched summaries, and prepare separate labels/predictor inputs.
Keep the current audit unchanged. No blanket removal of other-CVE context or
automatic graph-cache reuse is justified by these results.

Then, under an explicitly approved retrospective development protocol, run a
small structured/description classifier smoke before the complete source
comparison. The previous test is already observed and cannot be advertised as
a new untouched final test. Future independent evaluation is a separate
requirement; current date anomalies need not block honest snapshot classification.

## Saved Interpretation

TXT/JSON findings are stored separately under:

`EPSS_TPG/05_results/05_vulncheck_study/02_readiness_review_v1/`

Reproduce this interpretation from the repository root:

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/02_review_readiness.py --verify-only
```

The original full audit, candidate policy and pending annotation entries remain
unchanged in `01_readiness_v1/full/`.
