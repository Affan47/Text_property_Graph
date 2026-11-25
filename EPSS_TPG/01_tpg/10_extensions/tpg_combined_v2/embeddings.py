"""Frozen SecureBERT after TPG extraction; pool only declared phrase members."""
import numpy as np
import torch
from tpg_combined_v1.embeddings import Encoder as BaseEncoder, discarded_fragment
from .common import require


def pool_spans(offsets, hidden, spans):
    """A node has a union of member spans; count overlapping subwords once."""
    offsets, hidden = np.asarray(offsets, dtype=np.int64), np.asarray(hidden, dtype=np.float64)
    active = offsets[:, 1] > offsets[:, 0]
    offsets, hidden = offsets[active], hidden[active]
    result, counts = np.zeros((len(spans), hidden.shape[1])), np.zeros(len(spans), dtype=np.int64)
    if not len(offsets):
        return result, counts
    require(np.all(np.diff(offsets[:, 0]) >= 0) and np.all(np.diff(offsets[:, 1]) >= 0), 'Non-monotonic offsets')
    prefix = np.concatenate((np.zeros((1, hidden.shape[1])), hidden.cumsum(axis=0)))
    for i, members in enumerate(spans):
        intervals = []
        for a, b in members:
            if b <= a or a < 0:
                continue
            left = int(np.searchsorted(offsets[:, 1], a, side='right'))
            right = int(np.searchsorted(offsets[:, 0], b, side='left'))
            if right > left:
                intervals.append((left, right))
        merged = []
        for a, b in sorted(intervals):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
            else:
                merged.append((a, b))
        for a, b in merged:
            result[i] += prefix[b] - prefix[a]
            counts[i] += b - a
    return result, counts


def node_spans(graph):
    spans = []
    for node in graph['vertices']:
        p = node['properties']
        if node['label'] in ('CLAUSE', 'VERB_PHRASE'):
            require(p.get('EXTRA', {}).get('member_spans'), 'Phrase membership missing')
            spans.append(p['EXTRA']['member_spans'])
        else:
            spans.append([[p['CHAR_START'], p['CHAR_END']]])
    return spans


class Encoder(BaseEncoder):
    def __init__(self, device='cpu', threads=2):
        super().__init__(device, threads)
        self.identity.update(pooling='v2 union of phrase member subwords; bounding span for other spanned nodes',
                             text_context='combined original text, consecutive 1024-subword windows')

    def encode(self, text, graph):
        require(graph['metadata']['source_text'] == text, 'Embedding source differs from graph')
        spans = node_spans(graph)
        encoded = self.tokenizer(text, max_length=1024, truncation=True, stride=0,
                                 return_overflowing_tokens=True, return_offsets_mapping=True)
        sums, counts = np.zeros((len(spans), 768)), np.zeros(len(spans), dtype=np.int64)
        for k, offsets in enumerate(encoded['offset_mapping']):
            batch = {name: torch.tensor([encoded[name][k]], device=self.device) for name in ('input_ids', 'attention_mask')}
            with torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state[0].cpu().numpy()
            values, n = pool_spans(offsets, hidden, spans)
            sums += values
            counts += n
        mask, exceptions = counts > 0, []
        embeddings = torch.from_numpy((sums / np.maximum(counts, 1)[:, None]).astype(np.float32))
        for i, node in enumerate(graph['vertices']):
            if node['label'] == 'TOKEN' and not mask[i]:
                a, b = spans[i][0]
                require(discarded_fragment(text[a:b], self.tokenizer), f'Missing token embedding: {node["id"]}/{text[a:b]!r}')
                exceptions.append({'node_id': node['id'], 'span': [a, b], 'text': text[a:b],
                                   'reason': 'pinned_tokenizer_discards_format_only_fragment'})
        require(torch.isfinite(embeddings).all(), 'Nonfinite embedding')
        return embeddings, torch.from_numpy(mask), {'windows': len(encoded['input_ids']), 'exceptions': exceptions}


def model_inputs(tensor):
    from tpg_combined_v1.embeddings import model_inputs as old_inputs
    inputs = old_inputs(tensor)
    inputs['x'] = torch.cat((inputs['x'], tensor['target_cve'].float()[:, None]), dim=1)
    inputs['document_index'] = tensor['document_index']
    return inputs
