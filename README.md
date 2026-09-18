# Text Property Graph

This repository contains a Text Property Graph (TPG) implementation, examples of
graphs built from ordinary and security-related text, document-processing
applications, and a separate vulnerability-prediction pipeline that consumes
TPG graphs.

**Current status:** the previous training datasets and saved runs have been
retired. The replacement data comes from the `SummTPGVul` submodule on
`tpg-paper-revision`. It still needs a task-specific adapter and an explicit
target definition before training. Historical results are not results for this
replacement dataset.

## Start Here

| Order | Read | Purpose |
|---|---|---|
| 1 | [Documentation index](EPSS_TPG/00_documentation/README.md) | Navigate the project documentation |
| 2 | [Project status and findings](EPSS_TPG/00_documentation/01_start_here/01_PROJECT_SCHEMA_AND_FINDINGS.md) | Understand the current data transition and limitations |
| 3 | [TPG reading guide](EPSS_TPG/00_documentation/02_tpg/00_README.md) | Learn the graph representation and its applications |
| 4 | [Code reading order](EPSS_TPG/00_documentation/01_start_here/02_CODE_READING_ORDER.md) | Follow the implementation from pipeline to individual stages |
| 5 | [Running the project](EPSS_TPG/00_documentation/01_start_here/03_RUNNING_THE_PROJECT.md) | Set up imports and inspect the command-line entry points |
| 6 | [Current dataset guide](EPSS_TPG/00_documentation/03_datasets/00_README.md) | Check input files and training compatibility before planning a run |

All links below are relative to this repository root. Shell commands explicitly
change into `EPSS_TPG/` when they require that working directory.

## Repository Layout

```text
Text_property_Graph/
|-- README.md                       This map of the repository
|-- .gitmodules                     Source-data submodule configuration
|-- SummTPGVul/                     Upstream dataset repository
`-- EPSS_TPG/
    |-- README.md                   Project-level entry point
    |-- .tpg-project-root           Marker used to locate this checkout
    |-- 00_documentation/           Markdown guides, LaTeX sources and PDFs
    |-- 01_tpg/                     TPG code, examples, applications and workspace
    |-- 02_prediction/              Dataset adapters, GNNs, training and inference
    |-- 03_scripts/                 Maintenance tools and reserved workflow folders
    |-- 04_data/                    Derived-data location and source-repository link
    |-- 05_results/                 Training, evaluation and dataset-analysis outputs
    |-- 06_runtime/01_logs/         Runtime logs
    `-- 99_archive/                 Historical analyses, workflows and legacy links
```

**Storage distinction:** TPG code and artifacts are physically under `01_tpg/`.
The actual Markdown documentation remains under `00_documentation/02_tpg/`, and
the active LaTeX sources remain under `00_documentation/06_papers/`. The README
links beside the code point to those documents; they are not additional copies.
There is currently no separate `01_tpg/00_documentation/` or `01_tpg/06_papers/`.

## TPG Structure

The main code-reading path is:

```text
input text -> frontend -> typed graph -> enrichment passes -> GraphSON / PyG
```

The frontend extracts linguistic structure and, depending on the selected
pipeline, domain-specific information. Passes add relations and other graph
information. Exporters turn the graph into a saved representation or tensors.
The document applications and prediction models are consumers of this graph
engine, not alternative definitions of the TPG itself.

### Core Package

The actual Python package is [01_tpg/01_core/tpg/](EPSS_TPG/01_tpg/01_core/tpg).
Its import name is still `tpg`.

```text
EPSS_TPG/01_tpg/
|-- README.md -> ../00_documentation/02_tpg/00_README.md
|-- pyproject.toml                  Editable installation of the three TPG packages
|-- 01_core/tpg/
|   |-- __init__.py                 Package entry point
|   |-- pipeline.py                 Frontend selection, passes and export methods
|   |-- paths.py                    Canonical input, output and workspace paths
|   |-- schema/
|   |   |-- types.py                Node/edge types, properties and schemas
|   |   |-- graph.py                Graph storage, IDs and graph operations
|   |   `-- domain.py               Declarative domain specifications
|   |-- frontends/
|   |   |-- base.py                 Frontend interface
|   |   |-- spacy_frontend.py       Linguistic parsing and initial graph construction
|   |   |-- security_frontend.py    Rule-based security extraction
|   |   |-- model_security_frontend.py   Transformer-based security extraction
|   |   |-- hybrid_security_frontend.py  Rule/model combination
|   |   `-- pattern_frontend.py     Domain-pattern frontend
|   |-- passes/
|   |   |-- enrichment.py          Coreference, discourse, entity and topic passes
|   |   |-- security_relations.py  Optional dedicated security relations
|   |   `-- cross_modal.py         TPG/CPG alignment code
|   |-- exporters/exporters.py     GraphSON and PyG exporters
|   `-- utils/__init__.py          Utility package initializer
|-- 02_examples/                   Scripts, reference graphs, exports and inputs
|-- 03_application/tpg_app/         Document ingestion, storage and web application
|-- 04_chatbot/tpg_chatbot/          Earlier document question-answering application
|-- 05_workspace/                  Local application state
`-- 08_tests/test_layout.py         Import, path, override and reference-graph checks
```

Subpackages also contain `__init__.py` files. These initialize Python packages;
they are not additional processing stages. Start with
[pipeline.py](EPSS_TPG/01_tpg/01_core/tpg/pipeline.py), then read
[types.py](EPSS_TPG/01_tpg/01_core/tpg/schema/types.py),
[graph.py](EPSS_TPG/01_tpg/01_core/tpg/schema/graph.py), the selected frontend,
its passes, and finally the exporter.

### Examples and Graph Artifacts

All paths in this table are under `EPSS_TPG/01_tpg/02_examples/`.

| Location | Purpose |
|---|---|
| [01_scripts/demo.py](EPSS_TPG/01_tpg/02_examples/01_scripts/demo.py) | Demonstrate generic and security graph construction and export |
| [01_scripts/experiment.py](EPSS_TPG/01_tpg/02_examples/01_scripts/experiment.py) | Process inline text, text files, PDFs or batches |
| [01_scripts/compare_frontends.py](EPSS_TPG/01_tpg/02_examples/01_scripts/compare_frontends.py) | Compare frontend outputs |
| [01_scripts/TPG_sample_output.json](EPSS_TPG/01_tpg/02_examples/01_scripts/TPG_sample_output.json) | Additional saved sample output |
| [02_graphson/](EPSS_TPG/01_tpg/02_examples/02_graphson) | Four reference graphs described below; its README links to the graph guide |
| [03_generated/](EPSS_TPG/01_tpg/02_examples/03_generated) | Generated `output.json`, `security_output.json`, comparisons, GraphSON and PyG exports |
| [04_inputs/01_text/](EPSS_TPG/01_tpg/02_examples/04_inputs/01_text) | General, medical and security text examples |
| [04_inputs/02_pdf/](EPSS_TPG/01_tpg/02_examples/04_inputs/02_pdf) | Example PDF inputs and `generate_test_pdf.py` |

The four reference files are stored in `02_graphson/`:

| File | What it illustrates |
|---|---|
| [tpg_1_simple_paragraph.json](EPSS_TPG/01_tpg/02_examples/02_graphson/tpg_1_simple_paragraph.json) | Ordinary, non-security prose |
| [tpg_2_description_only.json](EPSS_TPG/01_tpg/02_examples/02_graphson/tpg_2_description_only.json) | A vulnerability description |
| [tpg_3_description_plus_summary.json](EPSS_TPG/01_tpg/02_examples/02_graphson/tpg_3_description_plus_summary.json) | The description combined with a summary |
| [tpg_4_description_plus_summary_plus_secedges.json](EPSS_TPG/01_tpg/02_examples/02_graphson/tpg_4_description_plus_summary_plus_secedges.json) | The combined text with explicit security relations |

These are illustrative graph artifacts, not training datasets. The ordinary-text
artifact also records security/hybrid frontend passes; its filename describes
the input text, not proof of a generic-only frontend run. See the
[graph comparison guide](EPSS_TPG/00_documentation/02_tpg/04_GRAPH_EXAMPLES.md)
for the input paragraphs, nodes, edges and embeddings.

### Applications and Runtime State

| File or folder under `EPSS_TPG/01_tpg/` | Purpose |
|---|---|
| [03_application/tpg_app/engine.py](EPSS_TPG/01_tpg/03_application/tpg_app/engine.py) | Connect document extraction, TPG processing, storage and queries |
| [03_application/tpg_app/extractors.py](EPSS_TPG/01_tpg/03_application/tpg_app/extractors.py) | Extract and chunk text from supported input formats |
| [03_application/tpg_app/store.py](EPSS_TPG/01_tpg/03_application/tpg_app/store.py) | SQLite storage and retrieval |
| [03_application/tpg_app/cli.py](EPSS_TPG/01_tpg/03_application/tpg_app/cli.py) | Command-line application interface |
| [03_application/tpg_app/server.py](EPSS_TPG/01_tpg/03_application/tpg_app/server.py) | HTTP API and web application entry point |
| [03_application/tpg_app/static/](EPSS_TPG/01_tpg/03_application/tpg_app/static) | `index.html` and the graph-visualization JavaScript asset |
| [03_application/tpg_app/requirements.txt](EPSS_TPG/01_tpg/03_application/tpg_app/requirements.txt) | Application dependency list |
| [03_application/tpg_app/Dockerfile](EPSS_TPG/01_tpg/03_application/tpg_app/Dockerfile) | Container build definition |
| [04_chatbot/tpg_chatbot/ingest.py](EPSS_TPG/01_tpg/04_chatbot/tpg_chatbot/ingest.py) | Extract passages and collect TPG entities and predicates |
| [04_chatbot/tpg_chatbot/graph_store.py](EPSS_TPG/01_tpg/04_chatbot/tpg_chatbot/graph_store.py) | JSON passage store and lookup indexes |
| [04_chatbot/tpg_chatbot/retriever.py](EPSS_TPG/01_tpg/04_chatbot/tpg_chatbot/retriever.py) | Match questions to stored passages |
| [04_chatbot/tpg_chatbot/chatbot.py](EPSS_TPG/01_tpg/04_chatbot/tpg_chatbot/chatbot.py) | Question-answering command-line interface |
| [04_chatbot/tpg_chatbot/store_nist.json](EPSS_TPG/01_tpg/04_chatbot/tpg_chatbot/store_nist.json) | Saved NIST-document passages and indexes, not a full GraphSON graph |
| [05_workspace/01_database/](EPSS_TPG/01_tpg/05_workspace/01_database) | Local `tpg_workspace.db` and SQLite runtime files |
| [05_workspace/02_uploads/](EPSS_TPG/01_tpg/05_workspace/02_uploads) | Locally uploaded documents, not canonical paper sources |

New chatbot stores default to `05_workspace/03_chatbot_stores/store.json`; that
directory is created when ingestion saves a store. Runtime contents are ignored
by Git, so a fresh checkout need not contain the local database or uploaded PDFs.
[paths.py](EPSS_TPG/01_tpg/01_core/tpg/paths.py) anchors defaults to this checkout.
Explicit CLI paths and the server's `TPG_DB` / `TPG_UPLOADS` overrides remain
available.

## Documentation Map

The actual documents below are under `EPSS_TPG/00_documentation/`.

| Document | Purpose |
|---|---|
| [02_tpg/00_README.md](EPSS_TPG/00_documentation/02_tpg/00_README.md) | TPG reading guide and folder map |
| [02_tpg/01_README.md](EPSS_TPG/00_documentation/02_tpg/01_README.md) | Architecture-document index |
| [02_tpg/02_TPG_COMPLETE_GUIDE.md](EPSS_TPG/00_documentation/02_tpg/02_TPG_COMPLETE_GUIDE.md) | TPG concepts, construction and representation |
| [02_tpg/03_SECURITY_TPG_REFERENCE.md](EPSS_TPG/00_documentation/02_tpg/03_SECURITY_TPG_REFERENCE.md) | Security-specific extraction and relations |
| [02_tpg/04_GRAPH_EXAMPLES.md](EPSS_TPG/00_documentation/02_tpg/04_GRAPH_EXAMPLES.md) | Four input texts and their exported graphs |
| [02_tpg/05_APPLICATION_GUIDE.md](EPSS_TPG/00_documentation/02_tpg/05_APPLICATION_GUIDE.md) | Document application workflows |
| [02_tpg/06_CHATBOT_README.md](EPSS_TPG/00_documentation/02_tpg/06_CHATBOT_README.md) | Chatbot documentation entry point |
| [02_tpg/07_CHATBOT_TECHNICAL_REPORT.md](EPSS_TPG/00_documentation/02_tpg/07_CHATBOT_TECHNICAL_REPORT.md) | Chatbot ingestion, indexes, retrieval and answering |
| [02_tpg/08_DOMAIN_EXAMPLES_README.md](EPSS_TPG/00_documentation/02_tpg/08_DOMAIN_EXAMPLES_README.md) | Domain-example documentation index |
| [02_tpg/09_WHO_ANALYSIS.md](EPSS_TPG/00_documentation/02_tpg/09_WHO_ANALYSIS.md) | WHO/medical-domain example analysis |
| [03_datasets/00_README.md](EPSS_TPG/00_documentation/03_datasets/00_README.md) | Current source files, fields and adapter limitations |
| [03_datasets/01_REVISION_README_AUDIT.md](EPSS_TPG/00_documentation/03_datasets/01_REVISION_README_AUDIT.md) | Verification of upstream documentation, data and preparation code |
| [03_datasets/revision_data_audit.json](EPSS_TPG/00_documentation/03_datasets/revision_data_audit.json) | Machine-readable snapshot audit |
| [04_experiments/00_README.md](EPSS_TPG/00_documentation/04_experiments/00_README.md) | Current experiment status and historical-result boundary |
| [05_prediction/00_README.md](EPSS_TPG/00_documentation/05_prediction/00_README.md) | Prediction package reading guide |
| [05_prediction/01_README.md](EPSS_TPG/00_documentation/05_prediction/01_README.md) | Additional prediction-document navigation |
| [06_papers/00_README.md](EPSS_TPG/00_documentation/06_papers/00_README.md) | Manuscripts, PDFs and build instructions |
| [07_maintenance/01_LAYOUT_AND_COMPATIBILITY.md](EPSS_TPG/00_documentation/07_maintenance/01_LAYOUT_AND_COMPATIBILITY.md) | Canonical paths, removed aliases and retained links |
| [07_maintenance/02_SCRIPT_GUIDE.md](EPSS_TPG/00_documentation/07_maintenance/02_SCRIPT_GUIDE.md) | Maintenance script reading/execution order |
| [07_maintenance/04_DATASET_TRANSITION.md](EPSS_TPG/00_documentation/07_maintenance/04_DATASET_TRANSITION.md) | Dataset checkout, retirement and transition commands |

### LaTeX and PDFs

[06_papers/01_sources/](EPSS_TPG/00_documentation/06_papers/01_sources) contains:

| Source | Subject |
|---|---|
| `tpg_chapter.tex`, `tpg_chapter_grounded.tex` | TPG chapter manuscripts |
| `security_frontend_rationale.tex` | Security frontend design |
| `CVE-graph-example.tex`, `CVE-graph-example-security.tex` | CVE graph illustrations |
| `multiview_tpg_formula.tex` | Multiview model formulas and TPG construction outline |
| `model_architecture.tex`, `model_architecture_abstract.tex` | Model architecture figures |
| `training_testing_context.tex` | Training and testing context |

Existing PDFs are in [06_papers/02_pdf/](EPSS_TPG/00_documentation/06_papers/02_pdf).
Auxiliary compilation files belong in `06_papers/03_build_files/`. PDFs can
predate their sources; a layout change does not regenerate them. Dataset-feature
and experiment-result manuscripts for retired data are in the archive, not this
active source directory.

## Prediction, Data and Results

The prediction code is separate from the reusable TPG engine:

| Location under `EPSS_TPG/` | Purpose |
|---|---|
| [02_prediction/01_package/epss/run_pipeline.py](EPSS_TPG/02_prediction/01_package/epss/run_pipeline.py) | Training command-line entry point |
| [02_prediction/01_package/epss/csv_adapter.py](EPSS_TPG/02_prediction/01_package/epss/csv_adapter.py) | Existing CSV normalization and target preparation |
| [02_prediction/01_package/epss/cve_dataset.py](EPSS_TPG/02_prediction/01_package/epss/cve_dataset.py) | CVE text to TPG to PyG dataset |
| [02_prediction/01_package/epss/tabular_features.py](EPSS_TPG/02_prediction/01_package/epss/tabular_features.py) | Structured feature encoding |
| [02_prediction/01_package/epss/gnn_model.py](EPSS_TPG/02_prediction/01_package/epss/gnn_model.py) | GNN backbones and graph/tabular fusion |
| [02_prediction/01_package/epss/edge_aware_layers.py](EPSS_TPG/02_prediction/01_package/epss/edge_aware_layers.py) | Relation-aware layers and multiview attention |
| [02_prediction/01_package/epss/train.py](EPSS_TPG/02_prediction/01_package/epss/train.py) | Training and evaluation loop |
| [02_prediction/01_package/epss/make_temporal_splits.py](EPSS_TPG/02_prediction/01_package/epss/make_temporal_splits.py) | Publication-date split preparation |
| [02_prediction/01_package/epss/test_only.py](EPSS_TPG/02_prediction/01_package/epss/test_only.py) | Saved-checkpoint evaluation |
| [02_prediction/02_inference/infer.py](EPSS_TPG/02_prediction/02_inference/infer.py) | Fresh-CVE inference entry point |
| [02_prediction/03_analysis/](EPSS_TPG/02_prediction/03_analysis) | Dataset analysis, feature verification and visualization scripts |
| [04_data/01_records_and_graphs/](EPSS_TPG/04_data/01_records_and_graphs) | Reserved local normalized records and graph caches |
| [04_data/02_source_repository/](EPSS_TPG/04_data/02_source_repository) | Symlink to the root `SummTPGVul/` submodule |
| [05_results/01_training/](EPSS_TPG/05_results/01_training) | Training outputs and checkpoints |
| [05_results/02_evaluation/](EPSS_TPG/05_results/02_evaluation) | Evaluation outputs |
| [05_results/03_dataset_analysis/](EPSS_TPG/05_results/03_dataset_analysis) | Dataset profiling outputs |

Source files live in
[SummTPGVul/Sec4AI4Sec-EPSS/Data_Files/](SummTPGVul/Sec4AI4Sec-EPSS/Data_Files).
The selected branch is recorded in [.gitmodules](.gitmodules), while the parent
repository pins a specific submodule commit. A branch setting does not update
that commit automatically. The [dataset guide](EPSS_TPG/00_documentation/03_datasets/00_README.md)
lists the six selected CSV/JSON files and their lineage.

The refreshed data does not contain the old EPSS targets or generated summaries.
VulnCheck KEV membership is not interchangeable with the earlier CISA KEV target.
Read the audit before adapting these fields. The active records and results
folders were cleared during retirement; their generated contents are ignored by
Git. Historical metrics must not be presented as new-dataset results.

## Setup and Checks

Use Python 3.10 or newer and activate the environment containing the project's
dependencies, such as the existing `CodeBERTFusion` environment. From the
repository root:

```bash
cd EPSS_TPG
python -m pip install --no-deps --no-build-isolation -e ./01_tpg
```

This requires setuptools 64 or newer and registers `tpg`, `tpg_app` and
`tpg_chatbot` from their numbered source folders. It does not install NLP,
application or training dependencies. Install once per environment and repeat
after moving the checkout. This is an editable checkout setup, not a standalone
wheel deployment; retain `.tpg-project-root`.

The following commands inspect entry points or run checks; they do not train a
model or regenerate the example graphs:

```bash
python -m tpg_app.cli --help
python -m tpg_chatbot.ingest --help
python 01_tpg/02_examples/01_scripts/experiment.py --help
python -m epss.run_pipeline --help
python 01_tpg/08_tests/test_layout.py
python 03_scripts/03_maintenance/verify_current_layout.py
```

The layout/data check requires the selected dataset files to be downloaded. It
reports known data-readiness warnings separately from integrity errors. Use
`--expect-empty` only when checking the post-retirement empty data/results/log
folders, not after creating new experiment outputs.

### Maintenance Scripts

These files live in [EPSS_TPG/03_scripts/03_maintenance/](EPSS_TPG/03_scripts/03_maintenance).

| Script | Purpose |
|---|---|
| `setup_revision_data.sh` | Set up the pinned sparse submodule checkout and download the selected Git LFS files |
| `audit_revision_data.py` | Audit downloaded file integrity, lineage and upstream readiness gaps without network calls |
| `test_revision_audit.py` | Regression tests for the dataset audit |
| `verify_current_layout.py` | Check the selected data snapshot, links and optional empty working folders |
| `build_paper.sh` | Compile one active LaTeX source and keep auxiliary files separate |

`03_scripts/01_training/` and `03_scripts/02_evaluation/` currently contain
placeholders, not a new experiment batch. The old batch wrappers are archived.
To compile the active TPG chapter, with `pdflatex` available on `PATH`, run from
`EPSS_TPG/`:

```bash
bash 03_scripts/03_maintenance/build_paper.sh tpg_chapter
```

## Removed Aliases and Historical Material

The root-level TPG aliases inside `EPSS_TPG/` have been removed: `tpg`, `tpg_app`,
`tpg_chatbot`, `TPG_examples`, `examples`, `output`, `output.json`,
`security_output.json`, `tpg_uploads`, and the `tpg_workspace.db` / WAL / SHM
links. Use the numbered filesystem paths and the editable package installation.
Unrelated compatibility links, including `epss`, `data`, `outputs` and `docs`,
remain; they do not contain duplicate copies.

| Archive location under `EPSS_TPG/99_archive/` | Contents |
|---|---|
| `01_texlive_installer/` | Local TeX Live installer material; not application code |
| [02_legacy_document_paths/](EPSS_TPG/99_archive/02_legacy_document_paths) | Compatibility links to active and historical documents, including `tpg_architecture/` |
| [03_previous_dataset_work/](EPSS_TPG/99_archive/03_previous_dataset_work/README.md) | Retired dataset analyses, result documents, workflows and cleanup inventories |

The previous-work archive separates `01_documentation/`, `02_dataset_analysis/`,
`03_submodule_analysis/`, `04_workflows/` and `05_audit/`. Its scripts and quoted
paths are historical references, not current execution instructions. General
TPG examples, the NIST chatbot store and local application workspace were
retained; they are not the retired training corpus or checkpoints.
