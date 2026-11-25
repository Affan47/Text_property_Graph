"""Bounded source-aware TPG encoding and matched persistence residual fusion."""
from collections import OrderedDict
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool

from . import forecast_identity_graphs_v2 as cache
from .edge_aware_layers import EdgeTypeGNNLayer, EDGE_FEATURE_DIM

require = cache.require
SOURCES = tuple(cache.base.old.SOURCES)
VARIANTS = ('epss_only', 'description_epss', 'all_sources_epss')


def enabled_sources(variant):
    require(variant in VARIANTS, 'Unknown model variant')
    return () if variant == 'epss_only' else SOURCES[:1] if variant == 'description_epss' else SOURCES


class GraphStore:
    """Read-only, byte-bounded tensor LRU. Raw text/audit fields never reach a model."""
    def __init__(self, root, index, vocab, dim, allowed_ids, tensor_hashes, cache_gib=12):
        self.root, self.index, self.vocab = Path(root), index, vocab
        self.dim, self.allowed_ids, self.hashes = dim, set(allowed_ids), tensor_hashes
        self.limit = int(cache_gib * 1024**3)
        require(self.limit >= 0, 'Negative cache budget')
        self.items, self.bytes, self.verified = OrderedDict(), 0, set()

    def load(self, name):
        if name in self.items:
            self.items.move_to_end(name)
            return self.items[name][0]
        path = (self.root / name).resolve()
        require(self.root.resolve() in path.parents and name in self.hashes, 'Unpinned or unsafe tensor path')
        if name not in self.verified:
            require(cache.sha256(path) == self.hashes[name], 'Changed graph tensor')
            self.verified.add(name)
        bundle = torch.load(path, weights_only=True, map_location='cpu')
        material = cache.base.old.shared.materialize(bundle, 'no_coref')
        result = {k: material[k] for k in ('x', 'edge_index', 'edge_type', 'edge_features')}
        x, ei, et, ef = (result[k] for k in result)
        require(x.dtype == torch.float32 and x.ndim == 2 and len(x) > 0 and
                x.shape[1] == self.vocab['num_node_types'] + self.dim + 1, 'Wrong node tensor')
        require(ei.dtype == et.dtype == torch.long and ei.shape == (2, len(et)) and
                ef.dtype == torch.float32 and ef.shape == (len(et), EDGE_FEATURE_DIM), 'Wrong edge tensors')
        require(bool(torch.isfinite(x).all()) and bool(torch.isfinite(ef).all()) and
                bool(((x[:, -1] == 0) | (x[:, -1] == 1)).all()), 'Nonfinite features/invalid embedding mask')
        require(not len(et) or (int(ei.min()) >= 0 and int(ei.max()) < len(x) and
                int(et.min()) >= 0 and int(et.max()) < self.vocab['num_edge_types']), 'Invalid node/relation indices')
        size = sum(v.numel() * v.element_size() for v in result.values())
        while self.items and self.bytes + size > self.limit:
            _, (_, old_size) = self.items.popitem(last=False)
            self.bytes -= old_size
        if size <= self.limit:
            self.items[name] = (result, size)
            self.bytes += size
        return result

    def documents(self, cve, enabled):
        require(cve in self.allowed_ids and cve in self.index and set(enabled) <= set(SOURCES), 'Forbidden CVE/source')
        require(self.index[cve]['split'] in ('train', 'validation'), 'Test graphs forbidden')
        result = []
        for doc in self.index[cve]['documents']:
            if doc['source_text_type'] in enabled:
                parts = [p for p in doc['chunks'] if p['has_graph']]
                require(bool(parts), 'Present source has no graph chunks')
                result.append((SOURCES.index(doc['source_text_type']), [p['tensor'] for p in parts]))
        return result


def graph_batches(store, ids, enabled, max_nodes, max_edges):
    """Yield bounded graph microbatches plus document/source ownership, not CVE text."""
    pending, owners, nodes, edges = [], [], 0, 0
    for i, cve in enumerate(ids):
        for d, (source, paths) in enumerate(store.documents(cve, enabled)):
            for path in paths:
                g = store.load(path)
                n, e = len(g['x']), len(g['edge_type'])
                require(n <= max_nodes and e <= max_edges, 'Single chunk exceeds fixed memory budget; no truncation permitted')
                if pending and (nodes + n > max_nodes or edges + e > max_edges):
                    yield Batch.from_data_list(pending), owners
                    pending, owners, nodes, edges = [], [], 0, 0
                pending.append(Data(**g, num_nodes=n))
                owners.append((i, source, d))
                nodes, edges = nodes + n, edges + e
    if pending:
        yield Batch.from_data_list(pending), owners


def hierarchical_pool(vectors, owners, batch_size, width, reference):
    """Equal chunks within each document; equal documents within each source."""
    docs = {}
    for value, key in zip(vectors, owners):
        docs.setdefault(key, []).append(value)
    sources = {}
    for (i, source, _), values in docs.items():
        sources.setdefault((i, source), []).append(torch.stack(values).mean(0))
    rows, masks = [], []
    for i in range(batch_size):
        rows.append(torch.stack([torch.stack(sources[i, s]).mean(0) if (i, s) in sources
                                 else reference.new_zeros(width) for s in range(len(SOURCES))]))
        masks.append([(i, s) in sources for s in range(len(SOURCES))])
    return torch.stack(rows), torch.tensor(masks, dtype=torch.bool, device=reference.device)


class MatchedRegressor(nn.Module):
    def __init__(self, variant, config, vocab, dim, gain):
        super().__init__()
        require(variant in VARIANTS and math.isfinite(gain) and .001 <= gain <= .05, 'Invalid model specification')
        require(sorted(vocab['edge_types'].values()) == list(range(vocab['num_edge_types'])), 'Non-contiguous edge vocabulary')
        self.variant, self.config, self.gain = variant, config, gain
        h, t = config['hidden_size'], config['temporal_hidden_size']
        # Always construct the same parameter layout so all variants share initial weights.
        self.temporal = nn.GRU(5, t, batch_first=True)
        self.node_project = nn.Sequential(nn.Linear(vocab['num_node_types'] + dim + 1, h), nn.GELU())
        self.layers = nn.ModuleList([EdgeTypeGNNLayer(h, vocab['num_edge_types'], config['dropout'])
                                     for _ in range(config['graph_layers'])])
        self.source_embeddings = nn.Embedding(len(SOURCES), h)
        self.source_attention = nn.Sequential(nn.Linear(h + t, h), nn.Tanh(), nn.Linear(h, 1))
        self.head = nn.Sequential(nn.Linear(t + h + len(SOURCES), h), nn.GELU(), nn.Linear(h, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def encode_graphs(self, x, edge_index, edge_type, edge_features, batch):
        x = self.node_project(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_type, edge_features)
        return global_mean_pool(x, batch)

    def details(self, sequence, store, ids):
        require(sequence.ndim == 3 and sequence.shape[1:] == (31, 5) and len(sequence) == len(ids), 'Wrong history tensor')
        observed = sequence[:, :, 1] == 1
        require(bool(observed.any(dim=1).all()), 'No observed past EPSS')
        last = torch.arange(31, device=sequence.device).expand_as(observed).masked_fill(~observed, -1).max(1).values
        persistence = sequence[:, :, 0].gather(1, last[:, None]).squeeze(1)
        _, hidden = self.temporal(sequence)
        temporal = hidden[-1]
        enabled = enabled_sources(self.variant)
        vectors, owners = [], []
        if enabled:
            for batch, ownership in graph_batches(store, ids, enabled, self.config['graph_batch_nodes'], self.config['graph_batch_edges']):
                args = (batch.x, batch.edge_index, batch.edge_type, batch.edge_features, batch.batch)
                if self.training and torch.is_grad_enabled() and self.config['checkpoint_graph_activations']:
                    value = checkpoint(self.encode_graphs, *args, use_reentrant=False, preserve_rng_state=True)
                else:
                    value = self.encode_graphs(*args)
                vectors.extend(value.unbind(0))
                owners.extend(ownership)
        sources, mask = hierarchical_pool(vectors, owners, len(ids), self.config['hidden_size'], temporal)
        if enabled:
            representations = sources + self.source_embeddings.weight[None]
            scores = self.source_attention(torch.cat((representations, temporal[:, None].expand(-1, len(SOURCES), -1)), -1)).squeeze(-1)
            # A finite sentinel plus explicit renormalization also handles all-missing rows.
            weights = torch.softmax(scores.masked_fill(~mask, -1e9), dim=1) * mask
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-12)
            fused = (weights[:, :, None] * representations).sum(1)
        else:
            weights = temporal.new_zeros((len(ids), len(SOURCES)))
            fused = temporal.new_zeros((len(ids), self.config['hidden_size']))
        correction = torch.tanh(self.gain * self.head(torch.cat((temporal, fused, mask.to(temporal.dtype)), 1))).squeeze(-1)
        raw = persistence + correction
        return persistence, correction, raw, raw.clamp(0, 1), weights, mask

    def forward(self, sequence, store, ids):
        return self.details(sequence, store, ids)[3]
