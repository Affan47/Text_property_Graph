# Code Reading Order

Read the control flow first, then the implementation of each stage. You do not need to start with every helper script.

## 1. Understand One Text Property Graph

All paths below are relative to the project root.

| Order | File | Why read it |
|---|---|---|
| 1 | [tpg/pipeline.py](../../01_tpg/01_core/tpg/pipeline.py) | Frontend selection and the pass sequence |
| 2 | [schema/types.py](../../01_tpg/01_core/tpg/schema/types.py) | Node types, edge types and property definitions |
| 3 | [schema/graph.py](../../01_tpg/01_core/tpg/schema/graph.py) | Graph storage, IDs, edges and queries |
| 4 | [frontends/spacy_frontend.py](../../01_tpg/01_core/tpg/frontends/spacy_frontend.py) | Ordinary text becomes graph structure |
| 5 | [frontends/security_frontend.py](../../01_tpg/01_core/tpg/frontends/security_frontend.py) | Rule-based security extraction |
| 6 | [frontends/hybrid_security_frontend.py](../../01_tpg/01_core/tpg/frontends/hybrid_security_frontend.py) | SecBERT embeddings and model-assisted extraction |
| 7 | [passes/enrichment.py](../../01_tpg/01_core/tpg/passes/enrichment.py) | Coreference, discourse, entity relations and topics |
| 8 | [passes/security_relations.py](../../01_tpg/01_core/tpg/passes/security_relations.py) | Optional dedicated security edges |
| 9 | [exporters/exporters.py](../../01_tpg/01_core/tpg/exporters/exporters.py) | GraphSON and PyG representations |
| 10 | [examples/experiment.py](../../01_tpg/02_examples/01_scripts/experiment.py) | A runnable text/PDF workflow |

The module names inside `01_tpg/01_core/tpg/` are conventional Python package names. Their reading order is documented here rather than encoded into import names.

## 2. Follow One Training Experiment

| Order | File | Responsibility |
|---|---|---|
| 1 | [run_pipeline.py](../../02_prediction/01_package/epss/run_pipeline.py) | CLI flags and end-to-end orchestration |
| 2 | [csv_adapter.py](../../02_prediction/01_package/epss/csv_adapter.py) | CSV columns become normalized records and targets |
| 3 | [data_collector.py](../../02_prediction/01_package/epss/data_collector.py) | Alternative NVD/KEV collection route |
| 4 | [cve_dataset.py](../../02_prediction/01_package/epss/cve_dataset.py) | Text selection, TPG construction, tensor conversion and caching |
| 5 | [tabular_features.py](../../02_prediction/01_package/epss/tabular_features.py) | Structured feature encoding |
| 6 | [gnn_model.py](../../02_prediction/01_package/epss/gnn_model.py) | Backbone factory and graph/tabular fusion |
| 7 | [edge_aware_layers.py](../../02_prediction/01_package/epss/edge_aware_layers.py) | Relation-aware layers and multiview attention |
| 8 | [train.py](../../02_prediction/01_package/epss/train.py) | Splits, optimization, validation and metrics |
| 9 | [visualize.py](../../02_prediction/01_package/epss/visualize.py) | Prediction tables and plots |
| 10 | [make_temporal_splits.py](../../02_prediction/01_package/epss/make_temporal_splits.py) | Publication-date partitions for temporal experiments |

Then read [test_only.py](../../02_prediction/01_package/epss/test_only.py) for checkpoint evaluation and [inference/infer.py](../../02_prediction/02_inference/infer.py) for the user-facing fresh-CVE CLI.

## 3. Follow the TPG Application

Read [engine.py](../../01_tpg/03_application/tpg_app/engine.py), then [extractors.py](../../01_tpg/03_application/tpg_app/extractors.py), [store.py](../../01_tpg/03_application/tpg_app/store.py), and [server.py](../../01_tpg/03_application/tpg_app/server.py). This is a separate document-processing application; the EPSS trainer is not its entry point.

Next: [TPG documentation](../02_tpg/00_README.md) or [script execution order](../07_maintenance/02_SCRIPT_GUIDE.md).

