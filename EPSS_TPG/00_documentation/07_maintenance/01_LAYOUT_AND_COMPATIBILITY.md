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

Compatibility names such as `epss`, `data` and `outputs` remain links.
The root-level TPG aliases (`tpg`, `tpg_app`, `tpg_chatbot`, `TPG_examples`,
`examples`, `output`, `output.json`, `security_output.json`, `tpg_uploads`,
and `tpg_workspace.db` with its WAL/SHM links) have been removed. Their targets
under `01_tpg/` were retained. Install `01_tpg/` in editable mode in each Python
environment, as described in [the running guide](../01_start_here/03_RUNNING_THE_PROJECT.md).
`datasets_info` points to the empty active analysis directory, not the archive.
Earlier document aliases may point to historical reports. Archived commands are
historical and may still name the removed paths.

The original [layout inventory](../../99_archive/03_previous_dataset_work/01_documentation/07_maintenance/layout_manifest.json) is archived. Its preservation check is no longer a valid current test: the user explicitly requested deletion of the previous datasets and runs. The retirement inventory records these deletions separately.

Generic example graphs, the NIST chatbot example and the document application's SQLite workspace are retained; these are not training corpus copies. No unrelated home-directory files, global model caches or Git history were deleted.
