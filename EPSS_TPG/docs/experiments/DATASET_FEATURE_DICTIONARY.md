# Dataset Feature Dictionary

This document explains the unique dataset features used in the EPSS-TPG
experiments. It is based on the dataset reports in
[../../Datasets_information](../../Datasets_information/README.md), the raw CSV
profiles, [csv_adapter.py](../../epss/csv_adapter.py), and
[tabular_features.py](../../epss/tabular_features.py).

## 1. Datasets Covered

All four main summary datasets share the same 9,218-row CVE corpus and the same
EPSS/CVSS/source structure. They differ mainly in the LLM-generated summary
column.

| Dataset report | Raw CSV | Rows | Primary summary column | Primary summary missingness |
|---|---|---:|---|---:|
| [GPT](../../Datasets_information/gpt_combined_summ/README.md) | `gpt_combined_summ.csv` | 9,218 | `summ_all_sources` | 20.0% |
| [Gemma](../../Datasets_information/gemma_combined_summ/README.md) | `gemma_combined_summ.csv` | 9,218 | `summ_all_sources` | 17.2% |
| [Llama](../../Datasets_information/final_dataset_with_llama_summ/README.md) | `final_dataset_with_llama_summ.csv` | 9,218 | `summ_llama3.1_8b` | 74.2% |
| [DeepSeek](../../Datasets_information/deepseek_combined_summ/README.md) | `deepseek_combined_summ.csv` | 9,218 | `summ_all_sources` | 18.8% |

The adapter now accepts `summary`, `llm_summary`, `summ_all_sources`,
`summ_llama3.1_8b`, `summ_github_urls`, and other `summ*` columns. If the
primary all-source summary is empty but a GitHub-only summary exists, the
adapter can use the GitHub-only summary as fallback.

## 2. Raw CSV Features

These are the unique raw columns found across the four datasets.

| Feature | Type | Present in | Meaning | Pipeline use |
|---|---|---|---|---|
| `cve` | string | all datasets | Row-level CVE identifier. In this corpus it often includes a suffix such as `-1`, `-2`, etc. | Mapped to `cve_id`. Used as graph/sample identifier, not as a predictive feature. |
| `source` | categorical string | all datasets | Source collection that produced the row, e.g. Mastodon, Telegram, Reddit, Hacker News, ExploitDB, BleepingComputer. | Stored as `source_platform`. Not directly encoded by the current model. |
| `date` | date string | all datasets | Source/social-media observation date. It is the best available timestamp in these CSVs. | Mapped to `published`; used to compute normalized vulnerability age in the tabular branch. |
| `time` | time string | all datasets | Time component of the source observation. | Not currently used by `csv_adapter.py` or the model. |
| `text` | string | all datasets | Original social/source text mentioning the CVE. | Not currently fed to TPG training. Current TPG text comes from `description` and optionally `llm_summary`. |
| `epss_score` | float | all datasets | FIRST EPSS exploitation probability. | Target when `--label-mode soft`. Also a leaky tabular input unless `--no-epss-feature` is used. |
| `epss_status` | categorical | all datasets | Whether the EPSS value is `original` or `enriched`. | Not used by `csv_adapter.py` in the current training path. Used in older ablation/preparation analyses. |
| `description` | string | all datasets | NVD CVE description. | Main TPG text for baseline runs. Removed from the graph only in `--summary-only-tpg`. |
| `cvss_version` | float/category | all datasets | CVSS version, usually `3.1` or `3.0`. | Not encoded directly. CVSS semantics are encoded through `cvss_score` and the component columns. |
| `cvss_score` | float | all datasets | CVSS base score from 0 to 10. | Mapped to `cvss3_score`; normalized to 0-1 in tabular features. Strong non-text risk signal. |
| `attack_vector` | categorical | all datasets | CVSS AV. Values include `NETWORK`, `LOCAL`, `ADJACENT_NETWORK`, `PHYSICAL`. | Reconstructed into `cvss3_vector`; one-hot encoded as `AV:N/A/L/P`. |
| `attack_complexity` | categorical | all datasets | CVSS AC. Values: `LOW`, `HIGH`. | Reconstructed into `cvss3_vector`; one-hot encoded as `AC:L/H`. |
| `privileges_required` | categorical | all datasets | CVSS PR. Values: `NONE`, `LOW`, `HIGH`. | Reconstructed into `cvss3_vector`; one-hot encoded as `PR:N/L/H`. |
| `user_interaction` | categorical | all datasets | CVSS UI. Values: `NONE`, `REQUIRED`. | Reconstructed into `cvss3_vector`; one-hot encoded as `UI:N/R`. |
| `scope` | categorical | all datasets | CVSS S. Values: `UNCHANGED`, `CHANGED`. | Reconstructed into `cvss3_vector`; one-hot encoded as `S:U/C`. |
| `confidentiality_impact` | categorical | all datasets | CVSS C impact. Values: `NONE`, `LOW`, `HIGH`. | Reconstructed into `cvss3_vector`; one-hot encoded as `C:N/L/H`. |
| `integrity_impact` | categorical | all datasets | CVSS I impact. Values: `NONE`, `LOW`, `HIGH`. | Reconstructed into `cvss3_vector`; one-hot encoded as `I:N/L/H`. |
| `availability_impact` | categorical | all datasets | CVSS A impact. Values: `NONE`, `LOW`, `HIGH`. | Reconstructed into `cvss3_vector`; one-hot encoded as `A:N/L/H`. |
| `usable` | boolean-like string | all datasets | Dataset readiness marker. One row is effectively not marked cleanly usable. | Not used as a filter by `csv_adapter.py`; profiles report it for readiness checks. |
| `source_count` | integer | all datasets | Number of source/social records associated with the CVE row. | Mapped to `social_source_count`; used as `num_references` fallback and `num_exploits` proxy in tabular features. |
| `sources_available` | boolean | all datasets | Whether source links/text were available. | Not directly encoded by the current model. |
| `code_available` | boolean | all datasets | Whether public code or proof-of-concept material was available. | Mapped to `has_public_exploit`; encoded as a tabular binary feature. |
| `delta_days_max` | float | all datasets | Maximum time delta associated with the source/EPSS collection. Missing for about 79.4% of rows. | Not used by the current adapter/model. |
| `delta_days_min` | float | all datasets | Minimum time delta associated with the source/EPSS collection. Missing for about 79.4% of rows. | Not used by the current adapter/model. |
| `source_links` | string/list-like | all datasets | URLs or source links used to collect evidence and generate summaries. | Not directly encoded. It is upstream evidence for the summary columns. |
| `summ_all_sources` | string | GPT, Gemma, DeepSeek | LLM-generated summary over all available source links. | Preferred summary input for `llm_summary` when present. Used by summary-in-TPG, summary-only, and two-view modes. |
| `summ_github_urls` | string | GPT, Gemma, DeepSeek | LLM-generated summary from GitHub URLs only. Much sparser than all-source summaries. | Adapter fallback if higher-priority summary columns are empty. |
| `summ_llama3.1_8b` | string | Llama | Llama-generated source summary. Sparse: about 74.2% empty. | Mapped to `llm_summary` for Llama runs. |

## 3. Features Created In `labeled_cves.json`

`csv_adapter.py` converts the raw CSV into the format consumed by
`CVEGraphDataset`.

| Converted field | Source field(s) | Meaning | Used by model? |
|---|---|---|---|
| `cve_id` | `cve` | Sample identifier. | Metadata only. |
| `description` | `description` | NVD text used to build TPG graphs. | Yes, except in `--summary-only-tpg`. |
| `published` | `date` | Publication/observation date used as a proxy timestamp. | Yes, tabular age feature. |
| `cvss3_score` | `cvss_score` | CVSS score normalized to 0-1 later. | Yes, when `--hybrid` is enabled. |
| `cvss3_vector` | CVSS component columns | Reconstructed CVSS vector such as `CVSS:3.1/AV:N/...`. | Yes, one-hot encoded when `--hybrid` is enabled. |
| `cwe_ids` | not provided | CWE list. Empty for these CSVs. | Encoded as all-zero CWE features. |
| `references` | not provided | NVD references list. Empty for these CSVs. | Falls back to `social_source_count`. |
| `binary_label` | derived from `epss_score >= 0.1` | Binary high-risk label for metrics and stratification. | Used for stratified splits and classification metrics. |
| `epss_score` | `epss_score` | Soft target. | Label in `--label-mode soft`; leaky feature unless `--no-epss-feature`. |
| `epss_percentile` | default `0.0` | EPSS percentile unavailable in these CSVs. | Constant zero if EPSS features are enabled. |
| `high_epss` | derived from `epss_score >= 0.1` | Convenience binary high-EPSS flag. | Not directly encoded. |
| `in_kev` | default `False` | CISA KEV status unavailable in these CSVs. | Not used as label for these soft-label runs. |
| `has_public_exploit` | `code_available` | Whether public exploit/PoC code exists. | Yes, tabular binary feature. |
| `num_exploits` | `source_count` | Proxy exploit/source count. | Yes, log-normalized tabular feature. |
| `social_source_count` | `source_count` | Number of source mentions. | Yes, fallback for reference count and exploit count. |
| `code_available` | `code_available` | Original boolean retained for transparency. | Used as fallback for `has_public_exploit`. |
| `llm_summary` | summary-like columns | Unified summary field. | Yes, when summary TPG flags are enabled. |
| `source_platform` | `source` | Original source platform. | Metadata only in current model. |

## 4. Text/Graph Features Used By The TPG Branch

The graph branch does not use raw CSV columns directly. It converts text into a
Text Property Graph.

| Graph feature | Created from | Meaning | Notes |
|---|---|---|---|
| Node type one-hot | TPG schema | 13-dimensional one-hot for node labels such as `DOCUMENT`, `SENTENCE`, `TOKEN`, `ENTITY`, `PREDICATE`, `ARGUMENT`, and phrase/clause nodes. | Always present. |
| SecBERT/text embedding | TPG frontend | 768-dimensional contextual embedding where the frontend/model provides one. | Zero-filled for nodes without embeddings. |
| `edge_index` | TPG graph edges | Directed graph connectivity. | Used by PyG message passing. |
| `edge_type` | TPG edge labels | Integer relation type. Base vocabulary has 13 types. | Used by edge-aware GNNs and MultiView. |
| `edge_attr` | `edge_type` | One-hot edge-type vector. | Supports relation-aware layers. |
| Security edge types | `--include-security-edges` | Adds first-class `SEC_*` relations such as `SEC_AFFECTS`, `SEC_EXPLOITED_BY`, and `SEC_HAS_SEVERITY`. | Expands edge vocabulary from 13 to 23. |
| `node_source_type` | summary flags | Marks nodes as `description`, `summary`, or `mixed`. | Added for source-aware analysis and two-view fusion. |
| `edge_source_type` | endpoint node sources | Marks edges as description-only, summary-only, or mixed/cross-source. | Used for graph diagnostics. |
| Source-label features | `--add-source-labels` | Adds 3 node-feature dimensions for `description`, `summary`, and `mixed`. | Lets the GNN see where text came from. |
| Summary pooling node | `--summary-pooling-node` | Adds one pooled summary node linked to the document node. | Uses mean summary sentence embedding when available. |

## 5. Tabular Feature Vector Used When `--hybrid` Is Enabled

The hybrid model concatenates the graph embedding with a tabular feature vector.
With EPSS included, this vector has 57 dimensions. With `--no-epss-feature`, it
has 55 dimensions and removes the direct EPSS leakage path.

| Tabular feature group | Dimensions | Source | Meaning |
|---|---:|---|---|
| CVSS score + has-CVSS flag | 2 | `cvss3_score` | Normalized CVSS score and indicator that CVSS exists. |
| CVSS vector one-hot | 22 | `cvss3_vector` | One-hot representation of AV, AC, PR, UI, S, C, I, A. |
| CWE multi-hot + other | 26 | `cwe_ids` | Top-25 CWE IDs plus an `other` bucket. In these CSVs this is all zero because CWE is not provided. |
| CWE count | 1 | `cwe_ids` | Normalized count of CWE IDs. Zero for these CSVs. |
| Reference/source count | 1 | `references` or `social_source_count` | Log-normalized number of references or social/source mentions. |
| Vulnerability age | 1 | `published` | Log-normalized days between `published` and the fixed reference date. |
| EPSS score + percentile | 2 | `epss_score`, `epss_percentile` | Direct EPSS input. This is leaky when EPSS is the label. Removed by `--no-epss-feature`. |
| Public exploit flag | 1 | `has_public_exploit` / `code_available` | Binary indicator of public exploit or PoC availability. |
| Exploit/source count | 1 | `num_exploits` or `social_source_count` | Log-normalized count proxy. |

## 6. Which Features Are Used In The Main Experiment Types

| Experiment type | Description text | LLM summary text | Security edges | Source labels | Tabular features | EPSS as input |
|---|---:|---:|---:|---:|---:|---:|
| Clean baseline `B` | yes | no | no | no | yes | no |
| `B_S` | yes | yes, concatenated | no | no unless extra flag used | yes | no |
| `B_E` | yes | no | yes | no | yes | no |
| `B_SE` | yes | yes, concatenated | yes | no unless extra flag used | yes | no |
| Summary-only TPG | no | yes | optional | yes if `--add-source-labels` | yes if `--hybrid` | no if `--no-epss-feature` |
| Two-view TPG | yes, separate graph | yes, separate graph | optional | required for view fusion | yes if `--hybrid` | no if `--no-epss-feature` |
| Pure TPG-only ablations | depends on flags | depends on flags | optional | optional | no, because `--hybrid` is omitted | no tabular EPSS path |

## 7. Leakage-Relevant Features

These fields deserve special attention because they can act as target proxies.

| Feature | Risk | Why |
|---|---|---|
| `epss_score` as tabular input | direct leakage | It is also the soft label in `--label-mode soft`. Use `--no-epss-feature`. |
| `epss_percentile` | direct/near-direct leakage | Constant zero in these CSVs, but should still be excluded in leakage-free settings. |
| CVSS score/components | possible target proxy | CVSS severity correlates with exploitation risk and contributes strong tabular signal. |
| `code_available` | possible target proxy | Public code/PoC availability can correlate with exploitation likelihood. |
| `source_count` | possible target proxy | High social/source mention count can encode attention or exploit interest. |
| LLM summary text | possible semantic proxy | Summaries may contain explicit exploit-likelihood phrasing. Summary-in-TPG experiments found no average gain, and summary-only performance was much weaker than description baseline. |

## 8. Practical Reading Guide

- Use `description` for the main TPG baseline.
- Use `llm_summary` only when testing summary-specific hypotheses.
- Use `--no-epss-feature` for any honest EPSS-prediction experiment.
- Omit `--hybrid` when you want to isolate text/TPG signal from tabular signal.
- Use `--graph-diagnostics` when comparing graph sizes, source-node counts, and summary-vs-description graph structure.
