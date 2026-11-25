"""Bounded graph loading and exact deterministic chunk-gradient replay."""
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import Batch, Data

from ..primary_study import common
from ..primary_study.model import PrimaryModel, mean_groups
from ..primary_study.data import validate_input
from ..consolidated.contracts import StructuredEncoder
from ..forecast_coreference_cache import materialize

require = common.require


class GraphStore:
    def __init__(self, directory, resources, cache_mib=256):
        require(cache_mib >= 0, 'Negative cache budget')
        self.directory = Path(directory).resolve()
        self.resources = resources
        self.limit = int(cache_mib * 1024**2)
        self.items = OrderedDict()
        self.bytes = 0
        self.loads = self.hits = 0

    def load(self, relative, view):
        require(view in ('no_coref', 'repeated_only', 'all_coref', 'security_edges_removed'), 'Unknown edge view')
        key = (relative, view)
        if key in self.items:
            self.hits += 1
            self.items.move_to_end(key)
            return self.items[key][0]
        path = (self.directory / relative).resolve()
        require(self.directory in path.parents, 'Unsafe graph path')
        bundle = torch.load(path, map_location='cpu', weights_only=True)
        item = materialize(bundle, 'no_coref' if view == 'security_edges_removed' else view)
        if view == 'security_edges_removed':
            vocab = self.resources['type_vocabulary']['edge_types']
            sec = torch.tensor([v for k, v in vocab.items() if k.startswith('SEC_')], dtype=torch.long)
            keep = ~torch.isin(item['edge_type'], sec)
            item = {**item, 'edge_index': item['edge_index'][:, keep],
                    'edge_type': item['edge_type'][keep], 'edge_features': item['edge_features'][keep]}
        # Clone the selected tensors so cached views do not retain hidden full bundles.
        fields = {k: item[k].clone() for k in ('x', 'edge_index', 'edge_type', 'edge_features')}
        token = self.resources['type_vocabulary']['node_types']['TOKEN']
        fields['token_mask'] = fields['x'][:, token] == 1
        graph = Data(**fields)
        size = sum(t.numel() * t.element_size() for t in fields.values())
        self.loads += 1
        while self.items and self.bytes + size > self.limit:
            _, (_, old_size) = self.items.popitem(last=False)
            self.bytes -= old_size
        if size <= self.limit:
            self.items[key] = (graph, size)
            self.bytes += size
        return graph


class Batches:
    def __init__(self, directory, rows, state, resources, settings, cache_mib=256):
        for row in rows:
            validate_input(row)
        self.rows = {r['cve']: r for r in rows}
        self.index = common.read(Path(directory) / 'cve_index.json')
        require(set(self.rows) == set(self.index), 'Cache/predictor cohort mismatch')
        self.state, self.resources, self.settings = state, resources, settings
        self.store = GraphStore(directory, resources, cache_mib)

    def plan(self, ids, variant):
        require(ids and len(ids) == len(set(ids)), 'Empty/repeated batch CVEs')
        paths, chunk_docs, doc_sources, masks, numeric = [], [], [], [], []
        for i, cve in enumerate(ids):
            row = self.rows[cve]
            numeric.append(StructuredEncoder.transform(row['structured'], self.state, variant['structured_blocks'])[0])
            mask = [int(s in variant['sources'] and row['source_mask'][j]) for j, s in enumerate(common.cs.SOURCES)]
            seen = [0] * 4
            for doc in self.index[cve]['documents']:
                source = doc['source_text_type']
                if source not in variant['sources']:
                    continue
                chunks = [c for c in doc['chunks'] if c['has_graph']]
                require(chunks, 'Accepted source document has no graph')
                sid = common.cs.SOURCES.index(source)
                docid = len(doc_sources)
                paths.extend(c['tensor'] for c in chunks)
                chunk_docs.extend([docid] * len(chunks))
                doc_sources.append(i * 4 + sid)
                seen[sid] = 1
            require(seen == mask, 'Source masks do not match cache')
            masks.append(mask)
        return {'paths': paths, 'chunk_document': torch.tensor(chunk_docs, dtype=torch.long),
                'document_source': torch.tensor(doc_sources, dtype=torch.long),
                'source_mask': torch.tensor(masks, dtype=torch.bool),
                'structured': torch.tensor(numeric, dtype=torch.float32)}

    def packets(self, plan, variant):
        items, nodes, edges, start = [], 0, 0, 0
        for relative in plan['paths']:
            graph = self.store.load(relative, variant['edge_view'] or 'no_coref')
            n, e = graph.num_nodes, graph.num_edges
            require(n <= self.settings['graph_node_budget'] and e <= self.settings['graph_edge_budget'],
                    'One chunk exceeds the frozen graph budget; refuse truncation')
            if items and (nodes+n > self.settings['graph_node_budget'] or edges+e > self.settings['graph_edge_budget']
                          or len(items) >= self.settings['graph_chunk_budget']):
                yield start, Batch.from_data_list(items)
                start += len(items)
                items, nodes, edges = [], 0, 0
            items.append(graph)
            nodes += n
            edges += e
        if items:
            yield start, Batch.from_data_list(items)


def make_model(resources, structured_dim, variant, settings):
    return PrimaryModel(resources, structured_dim, {**variant, 'structured': bool(variant['structured_blocks'])},
                        hidden=settings['hidden'], layers=settings['layers'])


def encode(model, graph, variant):
    return model.encoder(graph.x, graph.edge_index, graph.edge_type, graph.edge_features,
                         graph.batch, graph.token_mask, variant['representation'])


def fuse(model, chunks, plan):
    require(set(plan) == {'paths','chunk_document','document_source','source_mask','structured'},
            'Unexpected model-plan fields, possibly labels')
    size = len(plan['structured'])
    if len(chunks):
        docs = mean_groups(chunks, plan['chunk_document'], len(plan['document_source']))
        sources = mean_groups(docs, plan['document_source'], size * 4).reshape(size, 4, model.hidden)
    else:
        require(not bool(plan['source_mask'].any()), 'Missing graphs for enabled sources')
        sources = plan['structured'].new_zeros((size, 4, model.hidden))
    return model.fusion(sources, plan['source_mask'], plan['structured'])


def representations(model, batches, plan, variant):
    parts = []
    for _, graph in batches.packets(plan, variant):
        parts.append(encode(model, graph, variant))
    return torch.cat(parts) if parts else plan['structured'].new_empty((0, model.hidden))


def replay_backward(model, batches, plan, variant, target):
    require(model.training, 'Replay backward requires train mode')
    require(all(not isinstance(m, nn.Dropout) or m.p == 0 for m in model.modules()), 'Replay forbids stochastic dropout')
    require(not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()), 'Replay forbids batchnorm state')
    with torch.no_grad():
        values = representations(model, batches, plan, variant)
    proxy = values.detach().requires_grad_(True)
    prediction, weights, mask = fuse(model, proxy, plan)
    require(target.shape == prediction.shape, 'Target/prediction shape mismatch')
    loss = (prediction - target).abs().mean()
    require(bool(torch.isfinite(loss)), 'Nonfinite loss')
    loss.backward()
    if len(values):
        require(proxy.grad is not None, 'Disconnected graph branch')
        for start, graph in batches.packets(plan, variant):
            z = encode(model, graph, variant)
            require(torch.allclose(z.detach(), values[start:start+len(z)], rtol=0, atol=1e-7),
                    'Chunk replay changed; refuse incorrect gradients')
            z.backward(proxy.grad[start:start+len(z)])
    return prediction.detach(), weights.detach(), mask.detach()


def predict(model, batches, ids, variant, batch_size, progress=False):
    model.eval()
    values, weights, masks = [], [], []
    offsets = range(0, len(ids), batch_size)
    if progress:
        offsets = common.tqdm(offsets, desc='Validation CVE batches', unit='batch', leave=False)
    with torch.no_grad():
        for offset in offsets:
            plan = batches.plan(ids[offset:offset+batch_size], variant)
            p, w, m = fuse(model, representations(model, batches, plan, variant), plan)
            values.extend(p.tolist()); weights.extend(w.tolist()); masks.extend(m.tolist())
    return values, weights, masks
