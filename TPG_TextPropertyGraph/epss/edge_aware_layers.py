"""
Edge-Type-Aware GNN Layers — Ported from SemVul for TPG graphs
================================================================
Three edge-type-aware message passing mechanisms:

1. EdgeTypeGNN: Learns edge type embeddings, uses MLP(h_i, h_j, e_r) messages
   - Parameter-efficient for many edge types
   - From SemVul: vuln_detection/models/gnn_layers.py

2. RGATConv: Relation-specific weight matrices + edge-type-conditioned attention
   - Most expressive, GATv2-style attention per relation
   - From SemVul: vuln_detection/models/advanced/rgat.py

3. MultiViewEncoder: Separate GNN per semantic edge group + attention fusion
   - Interpretable (attention weights show which view matters)
   - From SemVul: vuln_detection/models/advanced/multiview.py

TPG Edge Types (Level 1, 13 types):
    0: DEP           1: NEXT_TOKEN    2: NEXT_SENT     3: NEXT_PARA
    4: COREF         5: SRL_ARG       6: AMR_EDGE      7: RST_RELATION
    8: DISCOURSE     9: CONTAINS     10: BELONGS_TO    11: ENTITY_REL
   12: SIMILARITY

Multi-View Groups (analogous to SemVul's AST/CFG/DFG):
    syntactic:  DEP, CONTAINS, BELONGS_TO          (like AST)
    sequential: NEXT_TOKEN, NEXT_SENT, NEXT_PARA   (like CFG)
    semantic:   COREF, SRL_ARG, AMR_EDGE            (like DFG)
    discourse:  RST_RELATION, DISCOURSE, ENTITY_REL, SIMILARITY
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GatedGraphConv, global_mean_pool, global_max_pool
from torch_geometric.utils import softmax, sort_edge_index


# ─── Layer 1: EdgeTypeGNN (from SemVul gnn_layers.py) ──────────────────

class EdgeTypeGNNLayer(MessagePassing):
    """Edge-type-aware GNN layer using learned edge embeddings.

    Message function: MLP([h_i, h_j, embed(edge_type)])
    Update function:  MLP([h_node, aggr_messages]) + residual

    More parameter-efficient than per-type weight matrices when
    num_edge_types is large (13+ for TPG).
    """

    def __init__(self, hidden_dim: int, num_edge_types: int = 13, dropout: float = 0.2):
        super().__init__(aggr="add")
        self.hidden_dim = hidden_dim

        self.edge_embed = nn.Embedding(num_edge_types, hidden_dim)

        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type=None):
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)

        edge_attr = self.edge_embed(edge_type)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.norm(x + self.dropout(out))

    def message(self, x_i, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def update(self, aggr_out, x):
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))


# ─── Layer 2: RGAT (from SemVul rgat.py) ───────────────────────────────

class RGATConv(MessagePassing):
    """Memory-Efficient Relational Graph Attention Convolution.

    For node i receiving messages from neighbors j via relation r:
        h_i' = sum_r sum_{j in N_r(i)} alpha_{ij}^{(r)} * W_r * h_j + W_0 * h_i

    Attention (GATv2-style):
        alpha_{ij}^{(r)} = softmax_j( LeakyReLU( a^T [W_r h_i || W_r h_j || e_r] ) )

    Memory: O(N * R * D) instead of O(E * H * D^2).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        heads: int = 4,
        dropout: float = 0.2,
        negative_slope: float = 0.2,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.heads = heads
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.head_dim = out_channels // heads
        assert self.head_dim * heads == out_channels

        # Per-relation weight matrices: W_r
        self.weight = nn.Parameter(torch.empty(num_relations, in_channels, out_channels))
        # Self-loop transform: W_0
        self.root_weight = nn.Parameter(torch.empty(in_channels, out_channels))
        # Edge type embeddings for attention
        self.edge_type_emb = nn.Embedding(num_relations, out_channels)

        # GATv2-style attention parameters
        self.att_src = nn.Parameter(torch.empty(1, heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, self.head_dim))
        self.att_edge = nn.Parameter(torch.empty(1, heads, self.head_dim))

        self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.root_weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.att_edge)
        nn.init.normal_(self.edge_type_emb.weight, std=0.01)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_type):
        N = x.size(0)
        H, D = self.heads, self.head_dim

        # Pre-compute per-relation transforms: [R, N, H, D]
        x_r = torch.einsum("ni,rio->rno", x, self.weight)
        x_r = x_r.view(self.num_relations, N, H, D)

        src, dst = edge_index

        # Select per-edge transforms
        x_src = x_r[edge_type, src]  # [E, H, D]
        x_dst = x_r[edge_type, dst]  # [E, H, D]
        edge_emb = self.edge_type_emb(edge_type).view(-1, H, D)  # [E, H, D]

        # Attention scores
        alpha = (x_src * self.att_src).sum(-1)  # [E, H]
        alpha = alpha + (x_dst * self.att_dst).sum(-1)
        alpha = alpha + (edge_emb * self.att_edge).sum(-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, dst, num_nodes=N)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Weighted messages
        msg = x_src * alpha.unsqueeze(-1)  # [E, H, D]
        out = torch.zeros(N, H, D, device=x.device, dtype=x.dtype)
        dst_expanded = dst.view(-1, 1, 1).expand(-1, H, D)
        out.scatter_add_(0, dst_expanded, msg)
        out = out.view(N, -1)  # [N, out_channels]

        # Self-loop + bias
        out = out + x @ self.root_weight + self.bias
        return out


class RGATBlock(nn.Module):
    """RGAT layer with pre-norm, residual, FFN (transformer-style block)."""

    def __init__(self, hidden_dim: int, num_relations: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv = RGATConv(hidden_dim, hidden_dim, num_relations, heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, edge_index, edge_type):
        h = self.norm1(x)
        h = self.conv(h, edge_index, edge_type)
        h = F.elu(h)
        h = self.dropout1(h)
        x = x + h

        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h
        return x


# ─── Layer 3: Multi-View Encoder (from SemVul multiview.py) ────────────

# ─── Multi-View Edge Grouping ──────────────────────────────────────────
#
# Maps semantic groups to TPG edge type NAMES (not indices).
# Indices are resolved at runtime from edge_type_vocab.json.

TPG_VIEW_NAMES = {
    "syntactic":  ["DEP", "CONTAINS", "BELONGS_TO"],
    "sequential": ["NEXT_TOKEN", "NEXT_SENT", "NEXT_PARA"],
    "semantic":   ["COREF", "SRL_ARG", "AMR_EDGE"],
    "discourse":  ["RST_RELATION", "DISCOURSE", "ENTITY_REL", "SIMILARITY"],
}

# Default fallback indices (Level 1 enum order) if no vocab file is available
TPG_EDGE_VIEWS = {
    "syntactic":  [0, 9, 10],       # DEP, CONTAINS, BELONGS_TO
    "sequential": [1, 2, 3],        # NEXT_TOKEN, NEXT_SENT, NEXT_PARA
    "semantic":   [4, 5, 6],        # COREF, SRL_ARG, AMR_EDGE
    "discourse":  [7, 8, 11, 12],   # RST_RELATION, DISCOURSE, ENTITY_REL, SIMILARITY
}


def build_view_config_from_vocab(edge_type_vocab: Dict[str, int]) -> Dict[str, List[int]]:
    """Build multi-view edge config from edge_type_vocab.json.

    Resolves edge type names to their actual integer indices,
    instead of relying on hardcoded enum ordering.

    Args:
        edge_type_vocab: dict mapping edge type name → integer index
                        (loaded from edge_type_vocab.json)

    Returns:
        Dict mapping view name → list of edge type indices.
    """
    config = {}
    for view_name, type_names in TPG_VIEW_NAMES.items():
        indices = []
        for name in type_names:
            if name in edge_type_vocab:
                indices.append(edge_type_vocab[name])
        if indices:
            config[view_name] = indices
    return config if config else TPG_EDGE_VIEWS


class ViewEncoder(nn.Module):
    """Processes a single edge-type view using GatedGraphConv."""

    def __init__(self, hidden_dim: int, num_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        self.ggnn = GatedGraphConv(out_channels=hidden_dim, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        if edge_index.size(1) == 0:
            return self.norm(x)
        edge_index = sort_edge_index(edge_index)
        h = self.ggnn(x, edge_index)
        h = self.norm(h)
        h = self.dropout(h)
        return h + x  # residual


class ViewAttentionFusion(nn.Module):
    """Fuses multi-view representations using node-conditioned attention."""

    def __init__(self, hidden_dim: int, num_views: int):
        super().__init__()
        self.query_net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, view_embeddings, node_context):
        """
        Args:
            view_embeddings: [N, num_views, hidden_dim]
            node_context: [N, hidden_dim]
        Returns:
            fused: [N, hidden_dim]
            attention: [N, num_views]
        """
        query = self.query_net(node_context)
        keys = self.key_proj(view_embeddings)
        attn = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1) * self.scale
        attention = F.softmax(attn, dim=-1)
        fused = (view_embeddings * attention.unsqueeze(-1)).sum(dim=1)
        return fused, attention


# ─── Graph-Level Classifiers ───────────────────────────────────────────

class EdgeTypeEPSSClassifier(nn.Module):
    """EPSS classifier using EdgeTypeGNN layers (edge-type-aware).

    Architecture: Input proj → [EdgeTypeGNN × L] → mean+max pool → MLP → logit
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        num_edge_types: int = 13,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.layers = nn.ModuleList([
            EdgeTypeGNNLayer(hidden_channels, num_edge_types, dropout)
            for _ in range(num_layers)
        ])

        graph_emb_dim = hidden_channels * 2
        self.classifier = nn.Sequential(
            nn.Linear(graph_emb_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, data):
        x = self.input_proj(data.x)
        edge_index = data.edge_index
        edge_type = data.edge_type if hasattr(data, "edge_type") else None

        for layer in self.layers:
            x = layer(x, edge_index, edge_type)

        x_mean = global_mean_pool(x, data.batch)
        x_max = global_max_pool(x, data.batch)
        graph_emb = torch.cat([x_mean, x_max], dim=-1)
        return self.classifier(graph_emb)

    def get_graph_embedding(self, data):
        x = self.input_proj(data.x)
        edge_index = data.edge_index
        edge_type = data.edge_type if hasattr(data, "edge_type") else None

        for layer in self.layers:
            x = layer(x, edge_index, edge_type)

        x_mean = global_mean_pool(x, data.batch)
        x_max = global_max_pool(x, data.batch)
        return torch.cat([x_mean, x_max], dim=-1)

    def predict_proba(self, data):
        return torch.sigmoid(self.forward(data)).squeeze(-1)


class RGATEPSSClassifier(nn.Module):
    """EPSS classifier using RGAT blocks (relation-specific attention).

    Architecture: Input proj → [RGATBlock × L] → mean+max pool → MLP → logit
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        num_edge_types: int = 13,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList([
            RGATBlock(hidden_channels, num_edge_types, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_channels)

        graph_emb_dim = hidden_channels * 2
        self.classifier = nn.Sequential(
            nn.Linear(graph_emb_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, data):
        x = self.input_proj(data.x)
        edge_index = data.edge_index
        edge_type = data.edge_type if hasattr(data, "edge_type") else None
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)

        for block in self.blocks:
            x = block(x, edge_index, edge_type)
        x = self.final_norm(x)

        x_mean = global_mean_pool(x, data.batch)
        x_max = global_max_pool(x, data.batch)
        graph_emb = torch.cat([x_mean, x_max], dim=-1)
        return self.classifier(graph_emb)

    def get_graph_embedding(self, data):
        x = self.input_proj(data.x)
        edge_index = data.edge_index
        edge_type = data.edge_type if hasattr(data, "edge_type") else None
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)

        for block in self.blocks:
            x = block(x, edge_index, edge_type)
        x = self.final_norm(x)

        x_mean = global_mean_pool(x, data.batch)
        x_max = global_max_pool(x, data.batch)
        return torch.cat([x_mean, x_max], dim=-1)

    def predict_proba(self, data):
        return torch.sigmoid(self.forward(data)).squeeze(-1)


class MultiViewEPSSClassifier(nn.Module):
    """EPSS classifier using multi-view processing (separate GNN per edge group).

    Splits TPG edges into semantic views:
        syntactic:  DEP, CONTAINS, BELONGS_TO        (like CPG AST)
        sequential: NEXT_TOKEN, NEXT_SENT, NEXT_PARA (like CPG CFG)
        semantic:   COREF, SRL_ARG, AMR_EDGE          (like CPG DFG)
        discourse:  RST_RELATION, DISCOURSE, ENTITY_REL, SIMILARITY

    Architecture: Input proj → [ViewEncoders] → Attention Fusion → pool → MLP → logit
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 4,
        dropout: float = 0.3,
        edge_view_config: Optional[Dict[str, List[int]]] = None,
    ):
        super().__init__()
        self.edge_view_config = edge_view_config or TPG_EDGE_VIEWS
        self.num_views = len(self.edge_view_config)

        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.view_encoders = nn.ModuleDict({
            name: ViewEncoder(hidden_channels, num_layers, dropout)
            for name in self.edge_view_config
        })

        self.fusion = ViewAttentionFusion(hidden_channels, self.num_views)

        graph_emb_dim = hidden_channels * 2
        self.classifier = nn.Sequential(
            nn.Linear(graph_emb_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def _get_view_mask(self, edge_type, view_types):
        mask = torch.zeros_like(edge_type, dtype=torch.bool)
        for t in view_types:
            mask = mask | (edge_type == t)
        return mask

    def forward(self, data):
        x = self.input_proj(data.x)
        edge_index = data.edge_index
        edge_type = data.edge_type if hasattr(data, "edge_type") else None
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)

        h0 = x
        view_outputs = []
        for view_name, view_types in self.edge_view_config.items():
            mask = self._get_view_mask(edge_type, view_types)
            edge_index_view = edge_index[:, mask]
            h_view = self.view_encoders[view_name](h0, edge_index_view)
            view_outputs.append(h_view)

        h_stacked = torch.stack(view_outputs, dim=1)  # [N, V, D]
        h_fused, _ = self.fusion(h_stacked, node_context=h0)

        x_mean = global_mean_pool(h_fused, data.batch)
        x_max = global_max_pool(h_fused, data.batch)
        graph_emb = torch.cat([x_mean, x_max], dim=-1)
        return self.classifier(graph_emb)

    def get_graph_embedding(self, data):
        x = self.input_proj(data.x)
        edge_index = data.edge_index
        edge_type = data.edge_type if hasattr(data, "edge_type") else None
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)

        h0 = x
        view_outputs = []
        for view_name, view_types in self.edge_view_config.items():
            mask = self._get_view_mask(edge_type, view_types)
            edge_index_view = edge_index[:, mask]
            h_view = self.view_encoders[view_name](h0, edge_index_view)
            view_outputs.append(h_view)

        h_stacked = torch.stack(view_outputs, dim=1)
        h_fused, _ = self.fusion(h_stacked, node_context=h0)

        x_mean = global_mean_pool(h_fused, data.batch)
        x_max = global_max_pool(h_fused, data.batch)
        return torch.cat([x_mean, x_max], dim=-1)

    def predict_proba(self, data):
        return torch.sigmoid(self.forward(data)).squeeze(-1)
