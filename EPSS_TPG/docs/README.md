# Documentation Index

This folder is the main entry point for project documentation. Runtime data,
trained model outputs, experiment logs, and generated JSON examples remain in
their original folders so existing commands keep working.

## Start Here

| Topic | Folder | Use it for |
|---|---|---|
| EPSS model and training | [epss_model](epss_model/README.md) | GNN architecture, training pipeline, leakage notes, inference |
| TPG architecture | [tpg_architecture](tpg_architecture/README.md) | Text Property Graph schema, security frontend, node/edge design |
| Experiment reports | [experiments](experiments/README.md) | Dataset reports, ablations, clean 16-run matrix, summary-only notes |
| Chatbot | [chatbot](chatbot/README.md) | TPG document-intelligence chatbot architecture |
| Domain examples | [domain_examples](domain_examples/README.md) | Non-security TPG examples such as WHO document analysis |

## Important Folders Outside `docs/`

| Folder | Why it stays there |
|---|---|
| [../Datasets_information](../Datasets_information/README.md) | Contains experiment reports plus batch scripts/logs used by the ablation workflow |
| [../TPG_examples](../TPG_examples/README.md) | Contains generated GraphSON JSON examples next to the explanation |
| [../output](../output) | Training outputs, checkpoints, metrics, predictions |
| [../data](../data) | Converted `labeled_cves.json` files and PyG graph caches |
