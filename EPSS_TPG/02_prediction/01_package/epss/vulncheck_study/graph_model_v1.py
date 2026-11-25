"""VK04 shared embedding controls, typed-edge GNN, structured late fusion and BCE replay."""
import torch
from torch import nn
from .adapter_v2 import require
from ..primary_training.batching import GraphStore, Batches as OldBatches
from ..primary_study.model import mean_groups
from ..consolidated.models import SourceEncoder


class Batches:
    packets = OldBatches.packets

    def __init__(self, directory, rows, index, resources, settings):
        from .classifier_smoke_v2 import check_rows
        check_rows(rows)
        self.rows = {r['cve']: r for r in rows}
        require(len(self.rows) == len(rows) and set(self.rows) == set(index), 'Cache/input cohort differs')
        self.index, self.settings = index, settings
        self.store = GraphStore(directory, resources, settings['cache_mib'])

    def plan(self, ids):
        require(ids and len(ids) == len(set(ids)), 'Empty/repeated batch')
        paths, chunk_cve = [], []
        for i, cve in enumerate(ids):
            docs = self.index[cve]['documents']
            require(len(docs) == 1 and docs[0]['source_text_type'] == 'description', 'Unexpected non-description input')
            parts = [p for p in docs[0]['chunks'] if p['has_graph']]
            require(parts, 'Missing description graph')
            paths.extend(p['tensor'] for p in parts)
            chunk_cve.extend([i] * len(parts))
        return {'paths': paths, 'chunk_cve': torch.tensor(chunk_cve, dtype=torch.long),
                'structured': torch.tensor([self.rows[c]['structured'] for c in ids], dtype=torch.float32)}


class Classifier(nn.Module):
    def __init__(self, resources, structured_dim, settings):
        super().__init__()
        vocab = resources['type_vocabulary']
        dim, hidden = resources['embedding_dim'], settings['hidden']
        self.encoder = SourceEncoder(vocab['num_node_types'] + dim + 1, dim, vocab['num_node_types'],
                                     vocab['edge_types'], hidden, settings['layers'])
        self.tabular = nn.Sequential(nn.Linear(structured_dim, hidden), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def encode(self, graph, representation):
        return self.encoder(graph.x, graph.edge_index, graph.edge_type, graph.edge_features,
                            graph.batch, graph.token_mask, representation)

    def fuse(self, chunks, plan):
        require(set(plan) == {'paths', 'chunk_cve', 'structured'}, 'Unexpected model inputs, possibly labels')
        text = mean_groups(chunks, plan['chunk_cve'], len(plan['structured']))
        return self.head(torch.cat((text, self.tabular(plan['structured'])), dim=1)).squeeze(1)


def representations(model, batches, plan, representation):
    return torch.cat([model.encode(graph, representation) for _, graph in batches.packets(plan, {'edge_view': 'no_coref'})])


def loss_for(logits, targets, class_weights):
    require(logits.shape == targets.shape and bool(torch.isfinite(logits).all()) and
            bool(torch.isin(targets, targets.new_tensor([0., 1.])).all()), 'Invalid classifier labels/logits')
    weights = logits.new_tensor([class_weights[0], class_weights[1]])[targets.long()]
    return (nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none') * weights).mean()


def replay_backward(model, batches, plan, representation, targets, class_weights):
    require(model.training, 'Training mode required')
    require(all(not isinstance(m, nn.Dropout) or m.p == 0 for m in model.modules()), 'Stochastic replay forbidden')
    require(not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()), 'Batchnorm replay forbidden')
    with torch.no_grad():
        values = representations(model, batches, plan, representation)
    proxy = values.detach().requires_grad_(True)
    logits = model.fuse(proxy, plan)
    loss = loss_for(logits, targets, class_weights)
    require(bool(torch.isfinite(loss)), 'Nonfinite loss')
    loss.backward()
    require(proxy.grad is not None, 'Text branch disconnected')
    for start, graph in batches.packets(plan, {'edge_view': 'no_coref'}):
        z = model.encode(graph, representation)
        require(torch.allclose(z.detach(), values[start:start+len(z)], atol=1e-7, rtol=0), 'Replay changed embeddings')
        z.backward(proxy.grad[start:start+len(z)])
    return float(loss.detach()), logits.detach().sigmoid()


def predict(model, batches, ids, representation, batch_size):
    from tqdm import tqdm
    model.eval()
    values = []
    with torch.no_grad():
        for offset in tqdm(range(0, len(ids), batch_size), desc='Score validation CVE batches', unit='batch', leave=False):
            plan = batches.plan(ids[offset:offset+batch_size])
            logits = model.fuse(representations(model, batches, plan, representation), plan)
            values.extend(logits.sigmoid().tolist())
    return values
