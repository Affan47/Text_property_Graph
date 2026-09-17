# Layout and Compatibility

The project was reorganized on 17 September 2026. The numbered directories are the canonical storage locations. Existing unnumbered names are relative symlinks, allowing prior commands and saved paths to resolve to the same files.

## Folder Map

| Old path | Canonical path |
|---|---|
| `tpg/` | `01_tpg/01_core/tpg/` |
| `examples/` | `01_tpg/02_examples/01_scripts/` |
| `TPG_examples/` | `01_tpg/02_examples/02_graphson/` |
| `output/` | `01_tpg/02_examples/03_generated/` |
| `tpg_app/` | `01_tpg/03_application/tpg_app/` |
| `tpg_chatbot/` | `01_tpg/04_chatbot/tpg_chatbot/` |
| `epss/` | `02_prediction/01_package/epss/` |
| `inference/` | `02_prediction/02_inference/` |
| `analysis/` | `02_prediction/03_analysis/` |
| `scripts/training/` | `03_scripts/01_training/` |
| `scripts/inference/` | `03_scripts/02_evaluation/` |
| `scripts/analysis/` | `03_scripts/03_maintenance/` |
| `data/` | `04_data/01_records_and_graphs/` |
| `outputs/` | `05_results/01_training/` |
| `inference_results/` | `05_results/02_evaluation/` |
| `datasets_info/` | `05_results/03_dataset_analysis/` |
| `logs/` | `06_runtime/01_logs/` |
| `docs/` | Compatibility tree linking to `00_documentation/` |

## Naming Rules

Number the organizational layers in reading or workflow order. Keep valid Python package names, dependency-owned directories, stable experiment IDs and cache-internal names unchanged. This prevents directory numbering from changing imports or dataset identity. The source Git submodule remains at its registered path and is linked into the numbered data section.

README files outside the central documentation area are navigation entry points or links. Older parallel documentation copies are preserved under `00_documentation/99_historical_indexes/`, without declaring one automatically newer or more authoritative.

## Preservation and Verification

The migration inventoried 1,937 non-cache-bytecode files totaling 324,820,285,054 bytes. Relocations used renames on the same filesystem. Before code bootstrap edits, all non-document files were checked for preserved size and inode; files below 2 MB also had their SHA-256 checked. Markdown links were rebased to their new locations.

[layout_manifest.json](layout_manifest.json) records old and new names and pre-migration metadata. A local `documentation_before_reorganization.json` snapshot preserves the original Markdown text. No model was retrained, and no dataset, checkpoint, predictions file or result metric was intentionally rewritten.

The marker `.tpg-project-root` lets direct Python entry points locate the project after relocation. Compatibility package links keep `python -m epss.run_pipeline` and `from tpg.pipeline import TPGPipeline` working from the project root. The Dockerfile copies the real package locations into its container.

Existing historical documents may refer to unavailable old artifacts. Those are separate from broken migration links; the link audit records them explicitly rather than inventing replacements.

## Git Checkout Contents

The repository includes the numbered source folders, documentation, saved result summaries and previously tracked datasets. Newly generated per-experiment records, tensor caches, checkpoints, runtime logs, database contents and the TeX installer remain local. Empty runtime directories are tracked so their directory aliases resolve in a fresh checkout. The `tpg_workspace.db` link is an intentional exception: its database target is created when the application first uses it.

Initialize the `SummTPGVul` submodule to populate the source-repository link. The submodule's own local changes are not included in the parent repository's layout commit.

The inventory audit below is for the original working directory, which contains the full experiment artifacts. A fresh checkout does not contain those ignored artifacts and cannot pass that full preservation audit until they are restored.

## Entry-Point Checks

The reorganization was checked with Python syntax parsing, shell syntax checks, CLI help through package and direct-file paths, dry runs of all four training wrappers, and a generic TPG parse/export followed by a CPU hybrid multiview forward pass. The old social-media batch path also passed its dry run. These checks did not retrain any model.

Run the artifact/link audit again with:

```bash
python 03_scripts/03_maintenance/verify_layout.py --write-report
```

See [the verification report](03_VERIFICATION_REPORT.md) for counts and the historical-reference exceptions. The Dockerfile paths were updated but a container image was not built. Existing LaTeX PDFs were not recompiled.
