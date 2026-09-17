# Dataset Reading Guide

Start with [the feature dictionary](01_DATASET_FEATURE_DICTIONARY.md), then read [current findings on targets, dates and duplicate CVEs](../01_start_here/01_PROJECT_SCHEMA_AND_FINDINGS.md). The dictionary and older profiles describe particular source versions; keep those versions separate from later datasets.

| Folder | Dataset report |
|---|---|
| [01_gpt](01_gpt/01_README.md) | Original GPT summary dataset |
| [02_gemma](02_gemma/01_README.md) | Original Gemma summary dataset |
| [03_llama](03_llama/01_README.md) | Original Llama summary dataset |
| [04_deepseek](04_deepseek/01_README.md) | Original DeepSeek summary dataset |
| [09_graph_profiles](09_graph_profiles/per_llm_graph_dims.md) | Graph dimensions and security profiles |
| [10_full_profiles](10_full_profiles/per_llm_full_profile.md) | Extended profiling |

Normalized records and graph caches live in `04_data/01_records_and_graphs/`. The source repository is accessible through `04_data/02_source_repository/`; this is a link to the existing Git submodule. Profile JSON artifacts live in `05_results/03_dataset_analysis/`.

For NVD/KEV and current MegaVul label behavior, read the current project findings and the adapter/collector files listed in the [code guide](../../../../00_documentation/01_start_here/02_CODE_READING_ORDER.md). For manuscript tables, see [dataset_features.tex](../06_papers/01_sources/dataset_features.tex).

