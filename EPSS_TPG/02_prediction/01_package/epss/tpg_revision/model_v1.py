"""Matched readout change; original nodes, edges, layers and parameters are retained."""
import torch
from ..vulncheck_study.graph_model_v1 import Classifier as OriginalClassifier
from ..vulncheck_study.adapter_v2 import require


def type_balanced_mean(z, node_types, batch):
    """Mean within each present type, then equal mean over present types per graph."""
    require(z.ndim == 2 and len(z) > 0 and node_types.ndim == 2 and len(z) == len(node_types)
            and batch.shape == (len(z),), 'Invalid readout shapes')
    require(bool(torch.isfinite(z).all()) and bool(torch.isin(node_types, node_types.new_tensor([0., 1.])).all())
            and bool((node_types.sum(1) == 1).all()), 'Need exactly one saved node type per node')
    require(batch.dtype == torch.long and int(batch.min()) == 0
            and len(batch.unique()) == int(batch.max()) + 1, 'Graph batch IDs must be contiguous')
    types, graphs = node_types.shape[1], int(batch.max()) + 1
    group = batch * types + node_types.argmax(1)
    counts = z.new_zeros(graphs * types).index_add(0, group, z.new_ones(len(z)))
    sums = z.new_zeros(graphs * types, z.shape[1]).index_add(0, group, z)
    means = (sums / counts.clamp_min(1)[:, None]).reshape(graphs, types, -1)
    present = counts.reshape(graphs, types) > 0
    return means.sum(1) / present.sum(1)[:, None]


class Classifier(OriginalClassifier):
    def __init__(self, resources, structured_dim, settings, readout='type_balanced'):
        require(readout in ('node_mean', 'type_balanced'), 'Unknown readout')
        super().__init__(resources, structured_dim, settings)
        self.readout = readout

    def encode(self, graph, representation):
        if self.readout == 'node_mean' or representation == 'pooled_text':
            return super().encode(graph, representation)
        require(representation == 'security_tpg', 'Unknown representation')
        z = self.encoder.graph_project(graph.x)
        for layer in self.encoder.layers:
            z = layer(z, graph.edge_index, graph.edge_type, graph.edge_features)
        return type_balanced_mean(z, graph.x[:, :self.encoder.node_type_dim], graph.batch)
