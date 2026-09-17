# Layout and Compatibility

The numbered layout remains unchanged. The retirement changed its contents, not the Python package names:

- `00_documentation/`: current guides and general TPG documentation.
- `01_tpg/`: graph engine, illustrative examples and document application.
- `02_prediction/`: adapters, GNN implementation and reusable analysis code.
- `03_scripts/`: maintenance tools; previous batch wrappers are archived.
- `04_data/`: empty derived-data directory and source-submodule link.
- `05_results/`: empty working directories for future experiments.
- `06_runtime/`: logs.
- `99_archive/03_previous_dataset_work/`: previous analyses and workflows.

Compatibility names such as `tpg`, `epss`, `data` and `outputs` remain links. `datasets_info` points to the empty active analysis directory, not the archive. Earlier document aliases may point to historical reports.

The original [layout inventory](../../99_archive/03_previous_dataset_work/01_documentation/07_maintenance/layout_manifest.json) is archived. Its preservation check is no longer a valid current test: the user explicitly requested deletion of the previous datasets and runs. The retirement inventory records these deletions separately.

Generic example graphs, the NIST chatbot example and the document application's SQLite workspace are retained; these are not training corpus copies. No unrelated home-directory files, global model caches or Git history were deleted.
