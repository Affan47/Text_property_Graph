"""SF02: matched, explicitly retrospective CVSS late-fusion comparison."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from . import forecast_matched_multisource_model as base
from . import forecast_structured_features as features

VARIANTS = ('all_sources_without_cvss', 'all_sources_with_current_cvss')


class StructuredStore:
    """Only the fixed numeric CVSS view reaches the model, never the audit records."""
    def __init__(self, graphs, rows, ids):
        base.require(set(rows) == set(ids), 'Structured IDs differ from the matched cohort')
        self.graphs = graphs
        self.rows = {c: torch.tensor(features.check_input(r), dtype=torch.float32) for c,r in rows.items()}

    def documents(self, cve, enabled):
        base.require(cve in self.rows, 'Unselected CVE')
        return self.graphs.documents(cve, enabled)

    def load(self, name):
        return self.graphs.load(name)

    def structured(self, ids, reference):
        base.require(all(c in self.rows for c in ids), 'Unselected structured CVE')
        return torch.stack([self.rows[c] for c in ids]).to(reference)


class StructuredRegressor(base.MatchedRegressor):
    def __init__(self, variant, config, vocab, dim, gain):
        base.require(variant in VARIANTS, 'Wrong structured variant')
        super().__init__('all_sources_epss', config, vocab, dim, gain)
        self.structured_variant = variant
        h, t, s = config['hidden_size'], config['temporal_hidden_size'], 8
        # The control has identical parameters and initialization; its CVSS path is zeroed.
        self.structured_encoder = nn.Sequential(nn.Linear(len(features.FEATURES), s), nn.GELU())
        self.head = nn.Sequential(nn.Linear(t+h+len(base.SOURCES)+s+2, h), nn.GELU(), nn.Linear(h, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def details(self, sequence, store, ids):
        # Preserve the frozen graph/source pooling, without editing its locked implementation.
        base.require(sequence.ndim == 3 and sequence.shape[1:] == (31,5) and len(sequence) == len(ids), 'Wrong history tensor')
        observed = sequence[:,:,1] == 1
        base.require(bool(observed.any(1).all()), 'No observed past EPSS')
        last = torch.arange(31, device=sequence.device).expand_as(observed).masked_fill(~observed,-1).max(1).values
        persistence = sequence[:,:,0].gather(1,last[:,None]).squeeze(1)
        _, hidden = self.temporal(sequence)
        temporal = hidden[-1]
        vectors, owners = [], []
        for batch, ownership in base.graph_batches(store, ids, base.SOURCES, self.config['graph_batch_nodes'], self.config['graph_batch_edges']):
            args = (batch.x, batch.edge_index, batch.edge_type, batch.edge_features, batch.batch)
            if self.training and torch.is_grad_enabled() and self.config['checkpoint_graph_activations']:
                value = checkpoint(self.encode_graphs, *args, use_reentrant=False, preserve_rng_state=True)
            else:
                value = self.encode_graphs(*args)
            vectors.extend(value.unbind(0))
            owners.extend(ownership)
        sources, mask = base.hierarchical_pool(vectors, owners, len(ids), self.config['hidden_size'], temporal)
        representations = sources+self.source_embeddings.weight[None]
        scores = self.source_attention(torch.cat((representations, temporal[:,None].expand(-1,len(base.SOURCES),-1)), -1)).squeeze(-1)
        weights = torch.softmax(scores.masked_fill(~mask,-1e9),1)*mask
        weights = weights/weights.sum(1,keepdim=True).clamp_min(1e-12)
        fused = (weights[:,:,None]*representations).sum(1)
        if self.structured_variant == VARIANTS[1]:
            cvss = store.structured(ids, temporal)
            structured = self.structured_encoder(cvss)*cvss[:,-1:]
            presence = cvss[:,-2:]
        else:
            structured, presence = temporal.new_zeros((len(ids),8)), temporal.new_zeros((len(ids),2))
        combined = torch.cat((temporal, fused, mask.to(temporal.dtype), structured, presence),1)
        correction = torch.tanh(self.gain*self.head(combined)).squeeze(-1)
        raw = persistence+correction
        return persistence, correction, raw, raw.clamp(0,1), weights, mask
