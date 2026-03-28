"""
GNN Model — Graph-Level Binary Classifier for CVE Exploitation Prediction
==========================================================================
Model architectures:

1. EPSSGraphClassifier — edge-agnostic baselines (GCN, GAT, SAGE)
2. EdgeTypeEPSSClassifier — edge-type embedding + MLP messages (from SemVul)
3. RGATEPSSClassifier — relation-specific attention (from SemVul RGAT)
4. MultiViewEPSSClassifier — separate GNN per edge group (from SemVul MultiView)
5. HybridEPSSClassifier — any GNN backbone + tabular features

Six GNN backbones:
    Edge-agnostic:   gcn, gat, sage
    Edge-type-aware: edge_type, rgat, multiview  (ported from SemVul)
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

from epss.edge_aware_layers import (
    EdgeTypeEPSSClassifier,
    RGATEPSSClassifier,
    MultiViewEPSSClassifier,
    build_view_config_from_vocab,
)


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
    """Hybrid GNN + Tabular classifier.

    Combines graph-level embeddings from any GNN backbone (including
    edge-type-aware models from SemVul) with tabular features.

    Args:
        in_channels: Node feature dimension for GNN.
        tabular_dim: Dimension of tabular feature vector.
        hidden_channels: GNN hidden dimension.
        num_layers: Number of GNN message-passing layers.
        backbone: GNN backbone ('gcn','gat','sage','edge_type','rgat','multiview').
        dropout: Dropout rate.
        num_heads: GAT/RGAT attention heads.
        num_edge_types: Number of edge relation types (for edge-aware backbones).
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
        num_edge_types: int = 13,
        tabular_hidden: int = 64,
    ):
        super().__init__()
        self.dropout = dropout

        # GNN branch — supports both edge-agnostic and edge-aware backbones
        self.gnn = _build_gnn_backbone(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            backbone=backbone,
            dropout=dropout,
            num_heads=num_heads,
            num_edge_types=num_edge_types,
        )

        # Graph embedding dimension from mean+max pooling
        gnn_emb_dim = hidden_channels * 2

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
        graph_emb = self.gnn.get_graph_embedding(data)
        tabular_emb = self.tabular_encoder(data.tabular)
        fused = torch.cat([graph_emb, tabular_emb], dim=-1)
        return self.classifier(fused)

    def predict_proba(self, data) -> torch.Tensor:
        return torch.sigmoid(self.forward(data)).squeeze(-1)

    def get_graph_embedding(self, data) -> torch.Tensor:
        return self.gnn.get_graph_embedding(data)


def _build_gnn_backbone(
    in_channels: int,
    hidden_channels: int,
    num_layers: int,
    backbone: str,
    dropout: float,
    num_heads: int,
    num_edge_types: int,
    edge_type_vocab: Optional[dict] = None,
) -> nn.Module:
    """Internal: build a GNN backbone (with get_graph_embedding method)."""
    if backbone == "edge_type":
        return EdgeTypeEPSSClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_edge_types=num_edge_types,
            dropout=dropout,
        )
    elif backbone == "rgat":
        return RGATEPSSClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_edge_types=num_edge_types,
            num_heads=num_heads,
            dropout=dropout,
        )
    elif backbone == "multiview":
        # Resolve view config from vocab (name→index), not hardcoded indices
        view_config = None
        if edge_type_vocab is not None:
            view_config = build_view_config_from_vocab(edge_type_vocab)
        return MultiViewEPSSClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            edge_view_config=view_config,
        )
    else:
        # Edge-agnostic: gcn, gat, sage
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


def build_model(
    in_channels: int,
    backbone: str = "gat",
    hidden_channels: int = 128,
    num_layers: int = 3,
    dropout: float = 0.3,
    num_heads: int = 4,
    tabular_dim: int = 0,
    num_edge_types: int = 13,
    edge_type_vocab: Optional[dict] = None,
) -> nn.Module:
    """Factory function to create GNN or Hybrid model.

    Args:
        in_channels: Node feature dimension (781 for SecBERT + 13 types).
        backbone: GNN backbone. Edge-agnostic: 'gcn','gat','sage'.
                  Edge-aware (from SemVul): 'edge_type','rgat','multiview'.
        hidden_channels: GNN hidden dimension.
        num_layers: Number of GNN layers.
        dropout: Dropout rate.
        num_heads: GAT/RGAT attention heads.
        tabular_dim: If > 0, build HybridEPSSClassifier. If 0, GNN-only.
        num_edge_types: Number of TPG edge relation types (13 base, 23 with security).
        edge_type_vocab: Dict mapping edge type name → index (from edge_type_vocab.json).
                        Used by multiview backbone to resolve view config by name.

    Returns:
        GNN classifier (text-only) or HybridEPSSClassifier (hybrid).
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
            num_edge_types=num_edge_types,
        )
    return _build_gnn_backbone(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        backbone=backbone,
        dropout=dropout,
        num_heads=num_heads,
        num_edge_types=num_edge_types,
        edge_type_vocab=edge_type_vocab,
    )
