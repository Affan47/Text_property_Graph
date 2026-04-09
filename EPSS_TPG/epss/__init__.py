"""
EPSS-GNN — Exploit Prediction via Graph Neural Networks on Text Property Graphs
================================================================================
Predicts probability of CVE exploitation within 30 days using GNN-based
graph-level classification on TPG representations of CVE descriptions.

Modules:
    data_collector  — Fetch CVE descriptions (NVD), labels (CISA KEV), scores (EPSS)
    cve_dataset     — PyG InMemoryDataset: CVE text → TPG → PyG Data objects
    gnn_model       — Graph-level GNN classifier (GCN / GAT / GraphSAGE)
    train           — Training loop with PR-AUC, F1, Brier score evaluation
    run_pipeline    — End-to-end orchestration
"""

__version__ = "0.1.0"
