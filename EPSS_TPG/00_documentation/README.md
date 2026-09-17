# Start Here

This is the main reading path. Read the first two documents before choosing a topic. Older research reports are preserved, but their claims and paths should be interpreted using the current project findings.

| Read | Document | What you learn |
|---|---|---|
| 1 | [Project schema and findings](01_start_here/01_PROJECT_SCHEMA_AND_FINDINGS.md) | The question, architecture, datasets, labels, verified results and unresolved issues |
| 2 | [Code reading order](01_start_here/02_CODE_READING_ORDER.md) | Which Python files to read first and how they connect |
| 3 | [TPG guide](02_tpg/00_README.md) | Graph schema, parsing, enrichment, security relations and examples |
| 4 | [Dataset guide](03_datasets/00_README.md) | Inputs, feature dictionary, targets and source-specific reports |
| 5 | [Experiment guide](04_experiments/00_README.md) | Run families, saved results and historical ablations |
| 6 | [Prediction guide](05_prediction/00_README.md) | Dataset tensors, model, training and evaluation |
| 7 | [Paper sources](06_papers/00_README.md) | LaTeX chapters, formulas, dataset tables and results |
| 8 | [Script guide](07_maintenance/02_SCRIPT_GUIDE.md) | Commands to inspect, train, evaluate and maintain the project |

[Running the project](01_start_here/03_RUNNING_THE_PROJECT.md) preserves the practical installation and command reference.
[Layout and compatibility](07_maintenance/01_LAYOUT_AND_COMPATIBILITY.md) explains the numbered folders.

## Navigation Rules

- Numbered folders describe a topic or processing stage.
- Real Markdown and LaTeX documentation is centralized here. README entries beside code are navigation links.
- Python packages keep valid import names such as `tpg` and `epss`. Their functions and classes were not renamed.
- Dataset snapshots, experiment IDs and cache-internal names remain stable identifiers.
- `99_historical_indexes/` preserves earlier indexes and parallel document versions. Use the topic guides above first.
- Research documents were relocated, not scientifically revalidated by the layout change. Some contain historical findings or references to artifacts no longer available.

The Git submodule `SummTPGVul` retains its upstream layout. It is accessible through `04_data/02_source_repository`.
