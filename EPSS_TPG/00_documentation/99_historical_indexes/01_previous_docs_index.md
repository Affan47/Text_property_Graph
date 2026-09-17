# Documentation Index

This folder is the main entry point for project documentation. Runtime data,
trained model outputs, experiment logs, and generated JSON examples remain in
their original folders so existing commands keep working.

## Start Here

| Topic | Folder | Use it for |
|---|---|---|
| EPSS model and training | [epss_model](../05_prediction/01_README.md) | GNN architecture, training pipeline, leakage notes, inference |
| TPG architecture | [tpg_architecture](../02_tpg/01_README.md) | Text Property Graph schema, security frontend, node/edge design |
| Experiment reports | [experiments](../04_experiments/01_EXPERIMENTS_README.md) | Dataset reports, ablations, clean 16-run matrix, summary-only notes |
| Chatbot | [chatbot](../02_tpg/06_CHATBOT_README.md) | TPG document-intelligence chatbot architecture |
| Domain examples | [domain_examples](../02_tpg/08_DOMAIN_EXAMPLES_README.md) | Non-security TPG examples such as WHO document analysis |

## Important Folders Outside `docs/`

| Folder | Why it stays there |
|---|---|
| [../Datasets_information](../../Datasets_information/README.md) | Contains experiment reports plus batch scripts/logs used by the ablation workflow |
| [../TPG_examples](../02_tpg/04_GRAPH_EXAMPLES.md) | Contains generated GraphSON JSON examples next to the explanation |
| [../output](../../01_tpg/02_examples/03_generated) | Training outputs, checkpoints, metrics, predictions |
| [../data](../../04_data/01_records_and_graphs) | Converted `labeled_cves.json` files and PyG graph caches |
