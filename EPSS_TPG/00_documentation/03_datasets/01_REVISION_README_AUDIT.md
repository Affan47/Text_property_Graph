# Paper-Revision Dataset Audit

## Scope and Conclusion

On 18 September 2026, I read the complete upstream README, inspected the tracked file tree and the relevant preparation scripts, and compared the downloaded datasets. GitHub's `tpg-paper-revision` branch and the local submodule both pointed to `a1e32ca3f16052b987767c013a49e2441b972e55`.

The initial three-file checkout contained the final feature and KEV snapshots, but omitted three files from the README's current-dataset table. Those files are now included. The checkout contains all six documented revision/lineage files without restoring the retired social-media datasets, Megavul datasets or trained runs.

The downloaded files match their Git LFS hashes and agree across their corresponding formats and enrichment stages. However, the README is not a complete, accurate reproduction recipe. There are serialization problems, missing upstream code and defects in the available generation scripts. Download completeness must not be mistaken for training readiness.

## Which Files We Need

All six files below are under `SummTPGVul/Sec4AI4Sec-EPSS/Data_Files/`.

| File | Verified size of dataset | Role |
|---|---|---|
| `cves_unique_with_source_links_clean.csv` | 5,692 unique CVEs, 15 columns | Base extraction; added for provenance checks |
| `cves_unique_with_source_links_clean.json` | 5,692 unique CVEs, 15 fields | Same base data with native URL lists; added for provenance checks |
| `cves_merged_refetched.csv` | 5,968 unique CVEs, 26 columns | Refreshed feature table |
| `cves_merged_refetched.json` | 5,968 unique CVEs, 26 fields | Refreshed feature records with nested URL lists |
| `cves_merged_with_url_dates.json` | 5,968 unique CVEs, 28 fields | Reference-date intermediate; newly included |
| `cves_merged_with_url_dates_vc_kev.json` | 5,968 unique CVEs; 29 fields on 5,420 rows and 39 on 548 rows | Most complete snapshot, including VulnCheck membership and exploitation metadata |

These are not six independent training datasets. The CSV/JSON pairs encode the same records, and the enrichment files contain successive versions of the same 5,968-CVE cohort. Do not concatenate them into extra training examples or put different versions of the same CVE into different splits.

For a future VulnCheck-membership experiment, the enriched JSON is the relevant starting point, with a strict separation between features and target/evidence fields. For a future EPSS-window experiment, the refreshed features and reference dates are present, but the required target series is not. The base pair is for lineage, not the final experiment population.

## Verified Agreements

- Each CSV agrees with its JSON counterpart after decoding URL lists, comparing numeric values and normalizing missing values in memory.
- From the base extraction to the refreshed cohort, 283 CVEs are added and 7 are removed: `5692 + 283 - 7 = 5968`.
- Reference-date enrichment retains every original field and CVE ID. KEV enrichment also retains every field from the date-only intermediate.
- The refreshed data contains 26,221 reference slots representing 14,975 distinct URL strings globally. GitHub and non-GitHub lists partition the reference lists, including repeated entries.
- Reference history contains 18,297 per-CVE distinct-URL entries: 17,878 exact matches and 419 case-insensitive matches. All have an earliest-addition date in this snapshot, and every CVE's reference set is covered. These are stored-data consistency checks, not a re-fetch of the underlying event histories.
- The enriched file contains 548 true and 5,420 false VulnCheck membership flags. Of the 548 listed CVEs, 342 have a CISA addition date and 206 do not.
- CVSS vectors are present for 5,891 records: 5,667 v3.1, 157 v3.0 and 67 v2. There are 77 missing vectors, all on `Deferred` records.
- Publication dates range from 1997-01-01 to 2026-06-18. NVD statuses are 2,389 `Analyzed`, 2,059 `Modified` and 1,520 `Deferred`.

The README mentions 65 removals during its wider construction process. The direct difference between the supplied base and final cohorts is seven removals. The other number may describe an earlier intermediate stage; the six selected files alone do not establish it. Likewise, the stated reason for all missing CVSS vectors being v4-only requires upstream responses or the missing metric-selection implementation to verify independently.

## Problems and Missing Pieces

### The JSON snapshots contain non-standard values

Each of `cves_merged_refetched.json`, `cves_merged_with_url_dates.json` and `cves_merged_with_url_dates_vc_kev.json` contains 2,445 literal `NaN` values. Strict JSON readers reject these files. Python's default JSON loader accepts them, which is why the earlier count-only check did not flag the issue.

The audit converts these constants to `None` only while comparing records and records how many it encountered. It does not modify the files. A future ingestion adapter must explicitly validate and normalize missing fields, rather than silently using zeros or allowing non-finite model inputs. Any normalized derivative should be written outside the source submodule.

### The intermediate field count is wrong in the README

`cves_merged_with_url_dates.json` has 28 fields, not 29: the 26 refreshed fields plus `url_dates_status` and `url_change_dates`. The 29th field, `in_vckev`, appears in the subsequent KEV file.

### Two documented implementations are absent upstream

The README refers to `Sec4AI4Sec-EPSS/Data_Files/fetch_vulncheck_data.py`, but that file is absent from the full tracked branch tree, not merely hidden by sparse checkout. The named `fetch_vckev_data` implementation was not found in the branch's tracked Python files either.

It also attributes refreshed metrics to `refetch_all_metrics` in [nvd_repo_scraper.py](../../../SummTPGVul/SummVul/Scrapers/nvd_repo_scraper.py). That function is absent from the named file and was not found in other tracked Python files. The available script's publication-date helper is unfinished.

The stored enriched dataset is available, but a complete regeneration path cannot be recovered just by adding more sparse-checkout entries. The missing code must be supplied upstream or deliberately reimplemented and tested. No replacement was fabricated in this audit.

### The available NVD reference collector has a confirmed defect

In `nvd_repo_scraper.py:96`, `get_query_response` reads `response["vulnerabilities"][0]["cve"][""]` instead of the `references` field. A synthetic successful response containing a valid reference returned an empty list in the offline test. The exception is caught and reported as no references, so rerunning this path could silently produce empty reference lists.

This reproduces a defect in the available script. It does not establish that this defective version produced the stored snapshots, whose reference lists are populated.

### The summary-generation loop has a confirmed defect

In [gen_llm_summ_web_parsing_bs.py](../../../SummTPGVul/Sec4AI4Sec-EPSS/LLM_summaries_gen/gen_llm_summ_web_parsing_bs.py), `summarize_open_sources` requires `api_key`. The loop in `get_summarizations_for_model` calls it without that argument. Signature binding reproduces the missing-argument error without sending an API request. Setting `NEBULA_TOKEN` alone does not fix the omitted argument at the call site.

Both GitHub and non-GitHub content prompts are already present, so the README's instruction to update the non-GitHub prompt is partly stale. The script can load the revision JSON layout, but that does not make the generation loop runnable as written.

The generator fetches current web content. Its proposed time-window URL selection is only commented-out code. Therefore, even after fixing the argument, a retrospective forecasting experiment would need explicit temporal filtering and a policy for the historical availability of page content.

### Reference-history pagination is not implemented

[fetch_url_dates.py](../../../SummTPGVul/Sec4AI4Sec-EPSS/Data_Files/fetch_url_dates.py) prints a warning if the first response is partial, but immediately returns the received changes without requesting later pages. The recorded `ok` status does not certify pagination completeness. The downloaded files have complete stored URL coverage; this does not prove that every earlier add/remove event was retrieved.

## What Is Still Needed for Experiments

| Proposed experiment | Available | Not yet supplied or validated |
|---|---|---|
| Predict VulnCheck catalogue membership | Descriptions, CVSS, references, reference dates and membership flags | A normalized dataset adapter, target/feature separation, evaluation cutoff and split protocol |
| Predict future KEV appearance | Catalogue dates and reference-date histories | A defensible observation time, future event window, negative censoring rules and time-appropriate features |
| Predict an EPSS minimum/maximum window | Refreshed feature snapshots | The relevant EPSS time series, window targets and version/coverage policy |
| Add revision-specific summaries | URL lists and content-fetching generation code | Fixed generator, generated outputs, failure/coverage checks and temporal controls |

No generated revision-summary columns or EPSS-window target values are present in the six selected files. The branch contains older summary and EPSS artifacts, but those are not substitutes for newly generated values aligned with this cohort and its chosen observation dates. They remain excluded locally.

The README describes VulnCheck membership as a snapshot taken on 8 September 2026. This audit checked the supplied snapshot, not the live catalogue or credentials. Missing catalogue membership is not proof of no exploitation, and `last_modified_timestamp` is not an explicit data-fetch timestamp.

## Reproduce the Audit

From `EPSS_TPG/`, with Git LFS available for the setup step:

```bash
bash 03_scripts/03_maintenance/setup_revision_data.sh
python 03_scripts/03_maintenance/audit_revision_data.py --output 00_documentation/03_datasets/revision_data_audit.json
python 03_scripts/03_maintenance/verify_current_layout.py --expect-empty
python -m unittest discover -s 03_scripts/03_maintenance -p test_revision_audit.py
```

[The machine-readable report](revision_data_audit.json) contains exact file sizes, hashes, counts, comparison results and warnings. An exit status of zero means the downloaded snapshot matches the pinned expectations; it does not mean the upstream problems are resolved. No NVD, VulnCheck, web-content or model-generation requests were made during the audit.

Only the parent setup, verification tools and documentation were changed. The upstream submodule files and its pinned commit remain unchanged.
