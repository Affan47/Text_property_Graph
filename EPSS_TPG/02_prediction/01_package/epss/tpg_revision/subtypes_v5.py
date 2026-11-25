"""Fixed, label-independent subtype channels alongside the unchanged base features."""
import torch
from torch import nn
from torch.nn import functional as F
from ..vulncheck_study.adapter_v2 import require
from .model_v1 import Classifier as ReadoutClassifier

VERSION = 'v5_subtypes'
# Static schema, not learned from validation/test categories. Novel types -> UNKNOWN.
NAMES = ('NOT_ENTITY', 'MISSING', 'UNKNOWN', 'CVE_ID', 'CWE_ID', 'SOFTWARE', 'VERSION',
         'HARDWARE_VERSION', 'VULN_TYPE', 'IMPACT', 'ATTACK_VECTOR', 'CODE_ELEMENT',
         'REMEDIATION', 'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT',
         'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT',
         'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL')
VOCAB = {name: i for i, name in enumerate(NAMES)}


def encode_nodes(nodes):
    values = []
    for node in nodes:
        if node['label'] != 'ENTITY':
            values.append(VOCAB['NOT_ENTITY'])
        else:
            kind = node['properties'].get('ENTITY_TYPE')
            values.append(VOCAB['MISSING'] if not kind else VOCAB.get(kind, VOCAB['UNKNOWN']))
    return torch.tensor(values, dtype=torch.long)


def one_hot(indices):
    require(indices.dtype == torch.long and indices.ndim == 1 and
            bool(((indices >= 0) & (indices < len(NAMES))).all()), 'Invalid subtype indices')
    return F.one_hot(indices, len(NAMES)).float() * (indices != 0)[:, None]


class Classifier(ReadoutClassifier):
    def __init__(self, resources, structured_dim, settings, readout, use_subtypes):
        super().__init__(resources, structured_dim, settings, readout)
        self.use_subtypes = use_subtypes
        # Keep existing parameter initialization and equal initial predictions.
        self.encoder.subtype_project = nn.Linear(len(NAMES), settings['hidden'], bias=False)
        nn.init.zeros_(self.encoder.subtype_project.weight)

    def encode(self, graph, representation):
        from torch_geometric.nn import global_mean_pool
        from .model_v1 import type_balanced_mean
        require(representation == 'security_tpg', 'v5 pilot is a graph comparison')
        require(graph.subtype.shape == (graph.num_nodes,), 'Subtype/node alignment differs')
        features = one_hot(graph.subtype).to(graph.x)
        z = self.encoder.graph_project(graph.x)
        if self.use_subtypes:
            z = z + self.encoder.subtype_project(features)
        for layer in self.encoder.layers:
            z = layer(z, graph.edge_index, graph.edge_type, graph.edge_features)
        return (global_mean_pool(z, graph.batch) if self.readout == 'node_mean' else
                type_balanced_mean(z, graph.x[:, :self.encoder.node_type_dim], graph.batch))
