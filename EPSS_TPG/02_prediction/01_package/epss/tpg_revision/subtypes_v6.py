"""Append-only SEVERITY channel; base model and v5 initialization stay fixed."""
import torch
from torch import nn
from torch.nn import functional as F

from . import subtypes_v5 as v5
from .model_v1 import Classifier as ReadoutClassifier, type_balanced_mean
from ..vulncheck_study.adapter_v2 import require

VERSION = 'v6_subtypes'
NAMES = v5.NAMES + ('SEVERITY',)
VOCAB = {name: i for i, name in enumerate(NAMES)}


def encode_nodes(nodes):
    values = v5.encode_nodes(nodes)
    for i, n in enumerate(nodes):
        if n['label'] == 'ENTITY' and n['properties'].get('ENTITY_TYPE') == 'SEVERITY':
            values[i] = VOCAB['SEVERITY']
    return values


def one_hot(indices):
    require(indices.dtype == torch.long and indices.ndim == 1 and
            bool(((indices >= 0) & (indices < len(NAMES))).all()), 'Invalid v6 subtype indices')
    return F.one_hot(indices, len(NAMES)).float() * (indices != 0)[:, None]


class Classifier(ReadoutClassifier):
    def __init__(self, resources, structured_dim, settings, readout, use_subtypes):
        super().__init__(resources, structured_dim, settings, readout)
        self.use_subtypes = use_subtypes
        # Common modules are initialized first, exactly as in v5.
        self.encoder.subtype_project = nn.Linear(len(NAMES), settings['hidden'], bias=False)
        nn.init.zeros_(self.encoder.subtype_project.weight)

    def encode(self, graph, representation):
        from torch_geometric.nn import global_mean_pool
        require(representation == 'security_tpg', 'v6 comparison is graph-only')
        require(graph.subtype.shape == (graph.num_nodes,), 'Subtype/node alignment differs')
        z = self.encoder.graph_project(graph.x)
        if self.use_subtypes:
            z = z + self.encoder.subtype_project(one_hot(graph.subtype).to(graph.x))
        for layer in self.encoder.layers:
            z = layer(z, graph.edge_index, graph.edge_type, graph.edge_features)
        return (global_mean_pool(z, graph.batch) if self.readout == 'node_mean' else
                type_balanced_mean(z, graph.x[:, :self.encoder.node_type_dim], graph.batch))
