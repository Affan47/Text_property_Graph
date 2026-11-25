"""TRV04 two-source wrapper; the original single-source modules stay unchanged."""
import torch
from torch import nn

from ..vulncheck_study import graph_model_v1 as base
from . import subtypes_v5

SOURCES = ('description', 'github_summary')
require = base.require


class Batches(base.Batches):
    def plan(self, ids, mask_g=False):
        require(ids and len(ids) == len(set(ids)), 'Empty/repeated CVEs')
        paths, owners, sources, masks = [], [], [], []
        for i, c in enumerate(ids):
            seen = [0, 0]
            for doc in self.index[c]['documents']:
                source = doc['source_text_type']
                require(source in SOURCES, 'N/M must never enter the DG adapter')
                if mask_g and source == SOURCES[1]:
                    continue
                parts = [p for p in doc['chunks'] if p['has_graph']]
                require(parts, 'Accepted document without a graph')
                owners.extend([len(sources)] * len(parts))
                paths.extend(p['tensor'] for p in parts)
                sid = SOURCES.index(source)
                sources.append(2*i+sid)
                seen[sid] = 1
            require(seen == [1, int(not mask_g and self.rows[c]['source_mask'][1])], 'G mask changed')
            masks.append(seen)
        return {'paths': paths, 'chunk_document': torch.tensor(owners, dtype=torch.long),
                'document_source': torch.tensor(sources, dtype=torch.long),
                'source_mask': torch.tensor(masks, dtype=torch.bool),
                'structured': torch.tensor([self.rows[c]['structured'] for c in ids], dtype=torch.float32)}


def pool(chunks, plan):
    docs = base.mean_groups(chunks, plan['chunk_document'], len(plan['document_source']))
    return base.mean_groups(docs, plan['document_source'], 2*len(plan['structured'])).reshape(len(plan['structured']), 2, -1)


class Wrapper:
    def add_attention(self, hidden):
        # Appended after all original modules: common initialization is unchanged.
        self.source_attention = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.source_bias = nn.Parameter(torch.zeros(2))

    def from_sources(self, values, mask, structured):
        require(values.ndim == 3 and values.shape[1] == 2 and mask.shape == values.shape[:2]
                and bool(mask[:, 0].all()), 'Invalid D/G representations')
        require(bool(torch.isfinite(values).all()), 'Nonfinite source representation')
        logits = (self.source_attention(values).squeeze(-1)+self.source_bias).masked_fill(~mask, -torch.inf)
        weights = logits.softmax(1)
        # Explicit row-wise path preserves D-only outputs even in a mixed batch.
        mixed = (values * weights[:, :, None]).sum(1)
        text = torch.where(mask[:, 1, None], mixed, values[:, 0])
        return self.head(torch.cat((text, self.tabular(structured)), 1)).squeeze(1), weights

    def fuse(self, chunks, plan):
        require(set(plan) == {'paths', 'chunk_document', 'document_source', 'source_mask', 'structured'},
                'Unexpected inputs, possibly targets')
        if not bool(plan['source_mask'][:, 1].any()):
            owner = plan['document_source'][plan['chunk_document']] // 2
            return base.Classifier.fuse(self, chunks, {'paths': plan['paths'], 'chunk_cve': owner,
                                                      'structured': plan['structured']})
        return self.from_sources(pool(chunks, plan), plan['source_mask'], plan['structured'])[0]


class Pooled(Wrapper, base.Classifier):
    def __init__(self, resources, dim, settings):
        super().__init__(resources, dim, settings)
        self.add_attention(settings['hidden'])


class Graph(Wrapper, subtypes_v5.Classifier):
    def __init__(self, resources, dim, settings):
        super().__init__(resources, dim, settings, 'node_mean', False)
        self.add_attention(settings['hidden'])


def representation(name):
    require(name in ('pooled_text', 'v5_node_mean'), 'Unknown DG representation')
    return 'pooled_text' if name == 'pooled_text' else 'security_tpg'


def make(resources, dim, settings, seed, name, wrapper=True):
    from ..vulncheck_study.graph_training_v1 import seed_everything
    representation(name)
    seed_everything(seed)
    if wrapper:
        return (Pooled if name == 'pooled_text' else Graph)(resources, dim, settings)
    return (base.Classifier(resources, dim, settings) if name == 'pooled_text' else
            subtypes_v5.Classifier(resources, dim, settings, 'node_mean', False))


def load_reference(model, state):
    result = model.load_state_dict(state, strict=False)
    allowed = {k for k in model.state_dict() if k.startswith('source_attention.') or k == 'source_bias'}
    require(set(result.missing_keys) == allowed and not result.unexpected_keys, 'D reference architecture mismatch')
