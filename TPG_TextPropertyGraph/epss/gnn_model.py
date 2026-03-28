"""
GNN Model — Graph-Level Binary Classifier for CVE Exploitation Prediction
==========================================================================
Three GNN backbones for graph-level classification:
    1. GCN  (Graph Convolutional Network)  — Kipf & Welling 2017
    2. GAT  (Graph Attention Network)      — Veličković et al. 2018
    3. SAGE (GraphSAGE)                    — Hamilton et al. 2017

Architecture:
    Input: CVE TPG graph (nodes with SecBERT embeddings + type encoding)
      ↓
    [GNN Backbone] × L layers (message passing on graph structure)
      ↓
    [Global Pooling] (mean + max concatenation → graph-level embedding)
      ↓
    [MLP Classifier] (graph embedding → P(exploitation in 30 days))

This is the core research contribution: using graph structure from TPG
(coreference chains, discourse relations, entity relationships) instead of
EPSS's flat 147 binary keyword features.
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


def build_model(
    in_channels: int,
    backbone: str = "gat",
    hidden_channels: int = 128,
    num_layers: int = 3,
    dropout: float = 0.3,
    num_heads: int = 4,
) -> EPSSGraphClassifier:
    """Factory function to create the GNN model.

    Default configuration designed for CVE TPG graphs:
        - in_channels: 781 (13 node types + 768 SecBERT embedding)
        - GAT backbone with 4 attention heads
        - 3 message-passing layers
        - mean+max global pooling
        - Binary classification (sigmoid output)
    """
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
