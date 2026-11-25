import importlib.metadata
import os
from pathlib import Path
import unicodedata
import numpy as np
import torch
from .common import MODEL, REVISION, require, sha

FILES = ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']


def model_path(download=False):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(MODEL, revision=REVISION, allow_patterns=FILES,
                                  local_files_only=not download))


def discarded_fragment(text, tokenizer):
    """No blanket Unicode exemption: require the pinned tokenizer to discard it."""
    return (bool(text) and all(c.isspace() or unicodedata.category(c) in ('Cc', 'Cf', 'Mn') or c == '\ufffd'
                              for c in text) and not tokenizer(text, add_special_tokens=False)['input_ids'])


def pool_spans(offsets, hidden, spans):
    """Global character offsets, including duplicate byte-token spans, stay aligned."""
    offsets = np.asarray(offsets, dtype=np.int64)
    active = offsets[:, 1] > offsets[:, 0]
    offsets, hidden = offsets[active], np.asarray(hidden, dtype=np.float64)[active]
    n, dim = len(spans), hidden.shape[1]
    if not len(offsets):
        return np.zeros((n, dim)), np.zeros(n, dtype=np.int64)
    require(np.all(np.diff(offsets[:, 0]) >= 0) and np.all(np.diff(offsets[:, 1]) >= 0), 'Non-monotonic subword offsets')
    starts, ends = np.asarray(spans, dtype=np.int64).T
    left = np.searchsorted(offsets[:, 1], starts, side='right')
    right = np.searchsorted(offsets[:, 0], ends, side='left')
    valid = (ends > starts) & (starts >= 0)
    right = np.where(valid, np.maximum(left, right), left)
    sums = np.concatenate((np.zeros((1, dim)), hidden.cumsum(axis=0)))
    return sums[right] - sums[left], right - left


class Encoder:
    def __init__(self, device='cpu', threads=2):
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        require(device in ('cpu', 'cuda'), 'Device must be cpu or cuda')
        require(device != 'cuda' or torch.cuda.is_available(), 'CUDA is not available')
        torch.set_num_threads(threads)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
        self.path = model_path()
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True, use_fast=True, trust_remote_code=False)
        require(self.tokenizer.is_fast, 'Character-aligned fast tokenizer required')
        config = AutoConfig.from_pretrained(self.path, local_files_only=True, trust_remote_code=False)
        require(config.model_type == 'modernbert' and config.hidden_size == 768 and config.max_position_embeddings >= 1024,
                'Unexpected SecureBERT 2.0 configuration')
        config.reference_compile = False
        self.model = AutoModel.from_pretrained(self.path, config=config, local_files_only=True,
                   trust_remote_code=False, attn_implementation='eager', torch_dtype=torch.float32).to(device).eval()
        self.model.requires_grad_(False)
        self.device = device
        self.identity = {'model': MODEL, 'revision': REVISION, 'files': {f: sha(self.path / f) for f in FILES},
                         'torch': torch.__version__, 'transformers': importlib.metadata.version('transformers'),
                         'tokenizers': importlib.metadata.version('tokenizers'), 'device': device, 'threads': threads,
                         'window_tokens': 1024, 'attention': 'eager', 'dtype': 'float32', 'trainable': False,
                         'config_position_limit': config.max_position_embeddings}

    def encode(self, text, graph):
        nodes = graph['vertices']
        spans = [(n['properties']['CHAR_START'], n['properties']['CHAR_END']) for n in nodes]
        encoded = self.tokenizer(text, max_length=1024, truncation=True, stride=0,
                                 return_overflowing_tokens=True, return_offsets_mapping=True)
        sums, counts = np.zeros((len(nodes), 768)), np.zeros(len(nodes), dtype=np.int64)
        for k, offsets in enumerate(encoded['offset_mapping']):
            batch = {name: torch.tensor([encoded[name][k]], device=self.device) for name in
                     ('input_ids', 'attention_mask')}
            with torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state[0].cpu().numpy()
            values, n = pool_spans(offsets, hidden, spans)
            sums += values
            counts += n
        mask = counts > 0
        embeddings = torch.from_numpy((sums / np.maximum(counts, 1)[:, None]).astype(np.float32))
        exceptions = []
        for i, node in enumerate(nodes):
            if node['label'] == 'TOKEN' and not mask[i]:
                fragment = text[slice(*spans[i])]
                require(discarded_fragment(fragment, self.tokenizer),
                        f'Missing ordinary token embedding: node={node["id"]} span={spans[i]} text={fragment!r}')
                exceptions.append({'node_id': node['id'], 'text': fragment, 'span': list(spans[i]),
                                   'reason': 'pinned_tokenizer_discards_format_only_fragment'})
        require(torch.isfinite(embeddings).all(), 'Nonfinite embeddings')
        return embeddings, torch.from_numpy(mask), {'windows': len(encoded['input_ids']), 'exceptions': exceptions}


def model_inputs(tensor):
    """Construct the train-time feature matrix without duplicating stored vectors."""
    from torch.nn.functional import one_hot
    features = torch.cat((one_hot(tensor['node_type'], tensor['vocabulary']['num_node_types']).float(),
                          tensor['node_embeddings'], tensor['embedding_mask'].float()[:, None]), dim=1)
    return {'x': features, **{k: tensor[k] for k in ('edge_index', 'edge_type', 'edge_features', 'subtype', 'token_mask')}}
