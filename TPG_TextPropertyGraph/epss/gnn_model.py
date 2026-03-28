"""
GNN Model — Graph-Level Binary Classifier for CVE Exploitation Prediction
==========================================================================
Two model architectures:

1. EPSSGraphClassifier (Phase 2 — text-only baseline):
    Input: CVE TPG graph → GNN → Global Pooling → MLP → P(exploitation)

2. HybridEPSSClassifier (Phase 3 — graph + tabular):
    Input: CVE TPG graph + tabular features (CVSS, CWE, age, refs)
      ↓
    [GNN Backbone] → graph embedding (256-dim)
    [Tabular MLP]  → tabular embedding (64-dim)
      ↓
    [Concat] → fusion embedding (320-dim)
      ↓
    [Classifier MLP] → P(exploitation in 30 days)

Three GNN backbones:
    1. GCN  (Graph Convolutional Network)  — Kipf & Welling 2017
    2. GAT  (Graph Attention Network)      — Veličković et al. 2018
    3. SAGE (GraphSAGE)                    — Hamilton et al. 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv,
    global_mean_pool, global_max_pool,
    BatchNorm,
)
from torch_geometric.data import Data
from typing import Optional


class EPSSGraphClassifier(nn.Module):
    """Graph-level binary classifier for CVE exploitation prediction.

    Args:
        in_channels: Input node feature dimension (num_node_types + embedding_dim).
                     With SECURITY_SCHEMA (13 types) + SecBERT (768): in_channels=781.
        hidden_channels: Hidden dimension for GNN layers.
        num_layers: Number of GNN message-passing layers.
        backbone: GNN backbone type ('gcn', 'gat', 'sage').
        dropout: Dropout rate for regularization.
        num_heads: Number of attention heads (GAT only).
        pool: Global pooling strategy ('mean', 'max', 'mean_max').
        num_classes: 1 for binary (sigmoid) or 2 for binary (softmax).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        backbone: str = "gat",
        dropout: float = 0.3,
        num_heads: int = 4,
        pool: str = "mean_max",
        num_classes: int = 1,
    ):
        super().__init__()
        self.backbone_type = backbone
        self.dropout = dropout
        self.pool_type = pool
        self.num_classes = num_classes

        # ─── GNN Layers ──────────────────────────────────────────

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                in_dim = in_channels
            else:
                in_dim = hidden_channels

            if backbone == "gcn":
                self.convs.append(GCNConv(in_dim, hidden_channels))
            elif backbone == "gat":
                # GAT: multi-head attention. Output = hidden_channels (heads concat then project)
                if i == 0:
                    self.convs.append(
                        GATConv(in_dim, hidden_channels // num_heads,
                                heads=num_heads, concat=True, dropout=dropout)
                    )
                else:
                    self.convs.append(
                        GATConv(hidden_channels, hidden_channels // num_heads,
                                heads=num_heads, concat=True, dropout=dropout)
                    )
            elif backbone == "sage":
                self.convs.append(SAGEConv(in_dim, hidden_channels))
            else:
                raise ValueError(f"Unknown backbone: {backbone}")

            self.bns.append(BatchNorm(hidden_channels))

        # ─── Pooling → Graph Embedding ───────────────────────────

        if pool == "mean_max":
            graph_emb_dim = hidden_channels * 2  # concat mean + max
        else:
            graph_emb_dim = hidden_channels

        # ─── MLP Classifier ──────────────────────────────────────

        self.classifier = nn.Sequential(
            nn.Linear(graph_emb_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass: graph → exploitation probability.

        Args:
            data: PyG Data/Batch object with x, edge_index, batch.

        Returns:
            logits: [batch_size, num_classes] or [batch_size, 1] predictions.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Message passing layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling: node embeddings → graph embedding
        if self.pool_type == "mean_max":
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            graph_emb = torch.cat([x_mean, x_max], dim=-1)
        elif self.pool_type == "max":
            graph_emb = global_max_pool(x, batch)
        else:
            graph_emb = global_mean_pool(x, batch)

        # Classify
        logits = self.classifier(graph_emb)
        return logits

    def predict_proba(self, data: Data) -> torch.Tensor:
        """Get exploitation probability (0-1) for each graph."""
        logits = self.forward(data)
        if self.num_classes == 1:
            return torch.sigmoid(logits).squeeze(-1)
        else:
            return F.softmax(logits, dim=-1)[:, 1]

    def get_graph_embedding(self, data: Data) -> torch.Tensor:
        """Extract graph-level embedding (before classifier head).

        Useful for Phase 3 hybrid model: concatenate GNN embedding
        with tabular EPSS features.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        if self.pool_type == "mean_max":
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            return torch.cat([x_mean, x_max], dim=-1)
        elif self.pool_type == "max":
            return global_max_pool(x, batch)
        else:
            return global_mean_pool(x, batch)


class HybridEPSSClassifier(nn.Module):
    """Phase 3: Hybrid GNN + Tabular classifier.

    Combines graph-level embeddings from TPG (structural text understanding)
    with tabular features (CVSS, CWE, references, vulnerability age) that
    EPSS v3 uses but the text-only GNN misses.

    Architecture:
        GNN branch:     TPG graph → EPSSGraphClassifier backbone → 256-dim
        Tabular branch: [53-dim] → Linear(128) → ReLU → Dropout → 64-dim
        Fusion:         concat(256, 64) = 320-dim → MLP → 1 (logit)

    Args:
        in_channels: Node feature dimension for GNN (781 for SecBERT + types).
        tabular_dim: Dimension of tabular feature vector (53 by default).
        hidden_channels: GNN hidden dimension.
        num_layers: Number of GNN message-passing layers.
        backbone: GNN backbone ('gcn', 'gat', 'sage').
        dropout: Dropout rate.
        num_heads: GAT attention heads.
        tabular_hidden: Hidden dim for tabular encoder.
    """

    def __init__(
        self,
        in_channels: int,
        tabular_dim: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        backbone: str = "gat",
        dropout: float = 0.3,
        num_heads: int = 4,
        tabular_hidden: int = 64,
    ):
        super().__init__()
        self.dropout = dropout

        # GNN branch (reuse existing backbone, without its classifier head)
        self.gnn = EPSSGraphClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            backbone=backbone,
            dropout=dropout,
            num_heads=num_heads,
            pool="mean_max",
            num_classes=1,  # won't use its classifier
        )

        # Graph embedding dimension from GNN pooling
        gnn_emb_dim = hidden_channels * 2  # mean_max pooling

        # Tabular branch
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, tabular_hidden * 2),
            nn.BatchNorm1d(tabular_hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tabular_hidden * 2, tabular_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Fusion classifier
        fusion_dim = gnn_emb_dim + tabular_hidden
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass: graph + tabular → exploitation logit.

        Args:
            data: PyG Data/Batch with x, edge_index, batch, tabular.

        Returns:
            logits: [batch_size, 1] predictions.
        """
        # GNN branch: graph → embedding
        graph_emb = self.gnn.get_graph_embedding(data)

        # Tabular branch
        tabular_emb = self.tabular_encoder(data.tabular)

        # Fusion
        fused = torch.cat([graph_emb, tabular_emb], dim=-1)
        logits = self.classifier(fused)
        return logits

    def predict_proba(self, data) -> torch.Tensor:
        logits = self.forward(data)
        return torch.sigmoid(logits).squeeze(-1)

    def get_graph_embedding(self, data) -> torch.Tensor:
        """Get the GNN branch embedding only."""
        return self.gnn.get_graph_embedding(data)


def build_model(
    in_channels: int,
    backbone: str = "gat",
    hidden_channels: int = 128,
    num_layers: int = 3,
    dropout: float = 0.3,
    num_heads: int = 4,
    tabular_dim: int = 0,
) -> nn.Module:
    """Factory function to create GNN or Hybrid model.

    Args:
        in_channels: Node feature dimension (781 for SecBERT + 13 types).
        backbone: GNN backbone ('gcn', 'gat', 'sage').
        hidden_channels: GNN hidden dimension.
        num_layers: Number of GNN layers.
        dropout: Dropout rate.
        num_heads: GAT attention heads.
        tabular_dim: If > 0, build HybridEPSSClassifier. If 0, text-only.

    Returns:
        EPSSGraphClassifier (text-only) or HybridEPSSClassifier (hybrid).
    """
    if tabular_dim > 0:
        return HybridEPSSClassifier(
            in_channels=in_channels,
            tabular_dim=tabular_dim,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            backbone=backbone,
            dropout=dropout,
            num_heads=num_heads,
        )
    return EPSSGraphClassifier(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        backbone=backbone,
        dropout=dropout,
        num_heads=num_heads,
        pool="mean_max",
        num_classes=1,
    )
