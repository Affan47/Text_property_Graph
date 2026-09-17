# Experiment Reading Guide

Read [current results and limitations](../01_start_here/01_PROJECT_SCHEMA_AND_FINDINGS.md) first. Then use the specific historical report that matches the run family:

1. [Summary and security-edge matrix](05_summary_ablation/02_RESULTS.md): the original 16-run program.
2. [Graph ablations](06_graph_ablation/tpg_ablation_results.md).
3. [CVSS ablations](07_cvss_ablation/cvss_ablation_results.md).
4. [LaTeX experiment tables](../06_papers/01_sources/experiment_results_35_runs.tex).

## Where Results Live

| Path | Contents |
|---|---|
| `05_results/01_training/01_social_media/` | Social-media training results |
| `05_results/01_training/02_megavul/` | MegaVul training results |
| `05_results/01_training/03_nvd_kev/` | NVD/KEV and temporal runs |
| `05_results/01_training/04_security_ablation/` | Security-frontend ablations |
| `05_results/01_training/99_legacy/` | Older runs retained for reference |
| `05_results/02_evaluation/` | Separate inference/evaluation outputs |
| `05_results/03_dataset_analysis/` | Profile JSONs and analysis artifacts |

Experiment subfolders preserve their IDs so configurations, checkpoints and predictions remain associated. Original `experiment_config.json` files were not rewritten; their older paths are supported through compatibility links.

Read `experiment_config.json` before interpreting `test_results.json`, and check prediction tables, cohort sizes and observed dates before comparing runs. Reorganizing a folder does not revalidate its scientific conclusions.

Execution order: [script guide](../07_maintenance/02_SCRIPT_GUIDE.md).

