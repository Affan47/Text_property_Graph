# CT09: Catalogue Evidence Recovered and Outcome Verification

## What Was Checked

CT08 reproduced on both datasets. Its lack of receipts in the configured
locations was correct, but it was not a claim that no useful export existed
elsewhere. A broader search of `SummTPGVul`, its tracked branch trees and Git
history found a useful **saved VulnCheck export** on another branch.

Live `git ls-remote` checks confirmed these GitHub heads on 14 November 2026:

| Branch | Commit |
|---|---|
| `new_data_collection` | `70615ef948b6063ea47dc5479174e1b2e6a1eb54` |
| `tpg-paper-revision` | `8787107701c62aa803895b400020997924f31986` |
| `modify-summ-generation` | `778a5a51048b438f5affc57e7b00ae5cda4e81f1` |

The first two dataset branches contain the same collector implementation
(SHA-256 `268d13362b8bede6a58c88e84c24d94259ea3df142e0e594a2aac00c1f647a5f`).
It can return `None` for both a failed lookup and an empty successful lookup,
then write `in_vckev = False` for either case. It does not preserve per-CVE
response receipts. This is a demonstrated code risk, not proof that every
saved false value resulted from failure or that this exact code produced every
historical file. The generic `log.txt` on the Reddit branch contained no
VulnCheck/receipt markers.

A receipt means a saved response associated with the exact requested CVE,
lookup time and outcome. The saved export found below is useful evidence, but
it is not that kind of per-request record.

## Recovered Export

On `modify-summ-generation`, the file
`vulncheck_known_exploited_vulnerabilities.json` is an **85,280,274-byte Git LFS
object containing 5,251 records**. Only that object was downloaded; there was
no branch checkout, pull, dataset replacement or training.

Its SHA-256 matches the repository pointer:

```text
3a1e4fbaf9f982ac4df927b5dbb252098edec2e5c902878963e86b2d11969a39
```

The preserved local copy is
`04_data/04_catalogue_snapshots/vulncheck_3a1e4fbaf9f9.json`.
Provenance is pinned in
`02_prediction/04_configs/07_fresh_studies/11_catalogue_snapshot_v1.json`.

The JSON is a list of VulnCheck-shaped records, including `cve`, `date_added`,
VulnCheck exploitation references, CISA-specific dates and update fields.
It has **no enclosing original retrieval-time or completeness statement**.
The file's Git commit is from 2 October 2026; a commit timestamp is not a
catalogue snapshot timestamp. The new local retrieval time is not the original
collection time either. Hash agreement verifies repository bytes, not independent
API authenticity or historical completeness.

## Training-Only Comparison

| Saved dataset flags versus recovered export | Reddit training candidates | Older training candidates |
|---|---:|---:|
| Saved listed, also present in export | 528 | 376 |
| Saved listed, absent from export | 16 | 0 |
| Saved false, present in export | 0 | 1 |
| Saved false, absent from export | 24,055 | 2,493 |
| Total candidates inspected | 24,599 | 2,870 |

Each row compares the stored flag with **presence in this saved export**, not
with a fresh API response. All 528 and 376 overlapping saved-listed CVEs also
have exactly matching addition-date strings. No target CVE has duplicate
matching export records.

Absence from this file does **not** release a negative training label. Likewise,
presence corroborates a saved listing but does not establish the current state,
the original dataset's snapshot state, or first exploitation time.

The older disagreement is **CVE-2017-5753**. The older dataset saves `false`;
the recovered export includes it with addition `2026-09-08T00:00:00Z`.
This needs review, not an automatic overwrite. It could reflect collection
timing, a previous failure, catalogue revision or a different source history.

Seven of the 16 absent Reddit saved-listed CVEs have stored addition dates
after the export's Git commit. Later catalogue changes are a plausible
explanation for those seven. Nine have earlier dates, including
`CVE-2021-25036` with a stored addition of `2021-12-14T00:00:00Z` and
`CVE-2022-31675` with `2022-08-10T00:00:00Z`. Backfilled dates, export coverage
and the unknown original retrieval time prevent interpreting these nine as
proof of faulty labels. Every discrepancy is listed in the generated report.

Only training candidate IDs were joined to the export. Other catalogue records
were parsed to validate the file and locate those IDs; no held-out outcomes,
text or performance metrics were exported or analyzed.

## Implemented Follow-Up

`20_verify_catalogue_outcomes.py` is a new opt-in collector/verifier. It leaves
CT08, the recovered-export comparison, source datasets and previous experiments
unchanged. Its default request list is the original 20-CVE training-only pilot
per dataset. With `--include-disagreements`, the immutable export-review
discrepancies are added and deduplicated:

| Dataset | Original pilot | Additional distinct discrepancies | Total requests |
|---|---:|---:|---:|
| Reddit | 20 | 15 | 35 |
| Older | 20 | 1 | 21 |

One Reddit discrepancy already belongs to the original pilot. This is a
diagnostic selection, not a prevalence or population-error-rate sample.
Neither mode permits arbitrary CVEs, a full cohort, or test/validation queries.

The collector follows the [official index API](https://docs.vulncheck.com/api/v3/indice)
with a CVE filter and bearer authentication. Its interpretation is conservative:

- A successful response containing the requested CVE in `vulncheck-kev` supports
  `LISTED` at the check time.
- An empty response supports `NOT_LISTED` only with explicit complete-zero
  metadata and no pagination/truncation warning.
- HTTP failures, network failures, incomplete empty responses, wrong CVE/index
  responses and malformed data stay `UNKNOWN`.
- Authentication/rate-limit errors stop the run and any remaining dataset
  queries. Three consecutive transport failures also stop collection.
- Requests are spaced at least one second apart. There are no automatic retry
  loops or followed redirects. A later explicit retry preserves every previous
  receipt and records a new attempt.
- Credentials and request headers are not written to artifacts. Successful
  response bytes, hashes, request parameters and check timestamps are retained.
- Offline verification reclassifies the stored responses and reproduces reports.
  Resume validates previous evidence; it does not silently replace it.

**No live VulnCheck checks have been run here:** `VC_API_KEY` is not available
to the verification process. The saved-export reviews and bounded plans passed.
Fifteen new synthetic tests passed; together with CT08's 17 tests, 32 focused
tests cover the audit and verification helpers. Synthetic HTTP tests are not
evidence of real catalogue outcomes.

## Recommended Next Decision

1. Run the bounded authenticated checks below and inspect unresolved responses
   and all saved-flag discrepancies.
2. If the collector behaves correctly, agree on one new catalogue snapshot or
   an explicitly bounded collection interval and verify the entire approved
   cohort before fitting. A 56-request pilot does not verify all 27,469 candidate
   rows, and dataset overlap means these are not necessarily distinct CVEs.
3. Use **retrospective snapshot classification** as the first feasible target.
   Label it as catalogue inclusion, not future exploitation. Review D/M/G/N
   disclosures and catalogue-derived features before choosing predictors.
4. Then freeze matched structured, pooled-text and TPG comparisons, with source
   ablations and a threshold chosen only in development data. Do not reuse an
   earlier inspected test cohort as an untouched final test.

The future-30/60/90-day inclusion task still needs supported event-time
semantics, at-risk status, complete follow-up and cutoff-valid text. A dated
post is not exact-text history. Catalogue absence is not proof of no exploitation.
No graph expansion or new training is justified by this verification alone.

## Commands, One at a Time

These use CPU and network only; no GPU or training is involved. The offline
review already exists. An API key must be entered locally, never in chat.

```bash
cd /home/ayounas/Text_property_Graph
```
```bash
conda activate CodeBERTFusion
```
```bash
python EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/21_review_saved_catalogue.py --dataset both --verify-only
```
```bash
RUNNER=EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/20_verify_catalogue_outcomes.py
```
```bash
python "$RUNNER" --dataset both --stage plan --include-disagreements
```
```bash
read -r -s -p "VulnCheck API key: " VC_API_KEY
```
```bash
export VC_API_KEY
```
```bash
python "$RUNNER" --dataset both --stage collect --include-disagreements --allow-network --resume
```
```bash
python "$RUNNER" --dataset both --stage verify --include-disagreements
```
```bash
unset VC_API_KEY
```

After a stopped or interrupted collection, correct the access/rate-limit issue
first. Only then explicitly retry unknowns with the same collection command
plus `--retry-unknown`. Ordinary resume skips already attempted unknowns; it
does not automatically spend more requests. If collection stopped on the first
dataset, the second may not exist yet; complete collection before verifying both.

Run the offline tests with:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/07_fresh_studies -p 'test_catalogue*.py'
```

All new results live under
`05_results/08_fresh_studies/16_catalogue_checks_v1/{01_reddit,02_previous_multisource}/`:
`saved_export_review/` contains the completed snapshot comparison;
`bounded_with_disagreements/` will contain real API receipts and outcome reports
after the authenticated run. Do not confuse the two evidence types.
