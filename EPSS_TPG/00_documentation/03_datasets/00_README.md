# Current Dataset

The source of truth is the `SummTPGVul` Git submodule on branch `tpg-paper-revision`, checked out at `a1e32ca3f16052b987767c013a49e2441b972e55`. The parent repository records a commit, not a moving copy of the branch. Subsequent updates require committing a new submodule pointer.

## Verified Files

These counts were read from the downloaded files on 18 September 2026.

| File under `SummTPGVul/Sec4AI4Sec-EPSS/Data_Files/` | Contents |
|---|---|
| `cves_unique_with_source_links_clean.csv` | Base extraction: 5,692 rows, 15 columns; retained for lineage, not the final training cohort |
| `cves_unique_with_source_links_clean.json` | Same base extraction with native URL lists; 5,692 unique CVEs |
| `cves_merged_refetched.csv` | 5,968 rows and 26 columns; URL lists are JSON strings |
| `cves_merged_refetched.json` | 5,968 records, 5,968 unique CVEs, 26 fields |
| `cves_merged_with_url_dates.json` | Date-enriched intermediate: 5,968 unique CVEs and 28 fields, not the 29 stated in the upstream README |
| `cves_merged_with_url_dates_vc_kev.json` | 5,968 unique CVEs; 29 fields on every record, plus 10 fields on the 548 VulnCheck KEV-listed records |

All records have nonempty descriptions. The enriched JSON has 548 `in_vckev=true` and 5,420 `in_vckev=false` records. These are observed catalogue membership flags, not proof that nonmembers are unexploited. VulnCheck KEV must not be silently treated as the old CISA KEV target.

Fields cover CVE identity, refreshed description, publication/modification timestamps, CVSS vector/score/severity, reference URLs and their GitHub/non-GitHub partitions. The `orig_` fields preserve earlier values. The enriched JSON adds reference-appearance history and VulnCheck KEV metadata.

Neither refreshed dataset has `epss_score`, social-media posts or the old summary columns. No replacement targets or summaries were invented during cleanup.

The two CSV/JSON pairs agree after decoding URL lists and normalizing missing values for comparison. Enrichment preserves all existing fields and CVE IDs. All six downloaded files match the size and SHA-256 in their Git LFS pointers.

Important: the refreshed, date-enriched and KEV-enriched JSON files each contain 2,445 literal `NaN` values. They are not strict JSON despite their extensions. Python's permissive loader accepts them, but an adapter must handle missing values explicitly. The upstream files have not been rewritten.

See [the README and source audit](01_REVISION_README_AUDIT.md) for verified counts, missing upstream code and regeneration defects. Snapshot integrity is not equivalent to a reproducible or training-ready pipeline.

## Training Compatibility

The [CSV adapter](../../02_prediction/01_package/epss/csv_adapter.py) expects `description` and `epss_score` in its social-media route. The refreshed CSV is not compatible with that route. The graph dataset expects a normalized CVE-keyed dictionary; the enriched source is a JSON array.

A new adapter and an explicit target definition are required before training. A VulnCheck-membership experiment must exclude membership fields and exploitation evidence from its inputs. An EPSS-range experiment first needs time-bounded EPSS targets. This cleanup implements neither experiment and does not establish a leakage-free evaluation protocol.

## Storage

- `04_data/02_source_repository/` links to the submodule, without copying it.
- `04_data/01_records_and_graphs/` is empty, reserved for regenerated artifacts and ignored by Git.
- A local sparse checkout exposes all six files listed above, the upstream README and selected preparation scripts. Older social-media/Megavul datasets remain excluded locally.
- See [the transition commands](../07_maintenance/04_DATASET_TRANSITION.md) to reproduce this checkout.
- Previous dataset reports are in [the retirement archive](../../99_archive/03_previous_dataset_work/README.md).
