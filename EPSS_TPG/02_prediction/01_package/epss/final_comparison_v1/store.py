"""All-or-fail per-fit RAM residency, with no eviction and no target inputs.

Optimization and selection share one preloaded working set. After checkpoint
selection, release it before loading assessment. The byte budget covers resident
input tensors, not Python overhead, model activations or the operating system.
"""
import hashlib
from pathlib import Path
import resource
import time

import numpy as np
import torch
from tqdm import tqdm

from ..combined_training_v1 import data as graph_data
from .model import MODELS

c = graph_data.c
require = graph_data.require


def _float_vector(value, length, name):
    if isinstance(value, torch.Tensor):
        require(not value.requires_grad, 'Cached predictors must be frozen: ' + name)
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    require(array.shape == (length,) and bool(np.isfinite(array).all()), 'Invalid ' + name)
    return torch.from_numpy(array.copy())


def _record(ctx, cid, model):
    """Read only explicitly requested predictor rows, never ctx['labels']."""
    require(model in MODELS and cid in ctx['index'], 'Unknown model/CVE input')
    representation, view = MODELS[model]
    contract = ctx['contract']
    if view == 'S':
        key, flags = None, dict.fromkeys(contract['source_order'], False)
    else:
        item = ctx['index'][cid]['views'][view]
        key, flags = item['object_key'], item['source_mask']
        require(set(flags) == set(contract['source_order']) and
                all(type(x) is bool for x in flags.values()), 'Invalid source mask: ' + cid)
        require(not flags['G'] and not flags['N'] and (view == 'DM' or not flags['M']),
                'Unexpected text source in ' + view)
        require(bool(key) == any(flags.values()), 'Source-mask/graph mismatch: ' + cid)
        if key:
            require(key in ctx['receipts'], 'Missing graph receipt: ' + str(key))
    inputs = {
        'graph': None,
        'structured': _float_vector(ctx['vectors'][cid], len(contract['structured_names']), 'structured vector'),
        'source_mask': torch.tensor([flags[s] for s in contract['source_order']], dtype=torch.float32),
    }
    if representation == 'pooled' and key:
        inputs['graph'] = {
            'pooled_vector': _float_vector(ctx['pooled'][view][cid], contract['embedding_dim'], 'pooled vector'),
            'pooled_observed': True,
        }
    return key, inputs


def predictor_digest(ctx, model, ids):
    """Hash exactly the selected predictors and receipts, without expanding TPGs."""
    digest = hashlib.sha256()
    for cid in sorted(ids):
        key, inputs = _record(ctx, cid, model)
        digest.update(c.encoded({'cve': cid, 'key': key,
                                 'receipts': ctx['receipts'][key] if key else None}))
        for name in ('structured', 'source_mask'):
            digest.update(inputs[name].numpy().tobytes())
        if inputs['graph'] is not None:
            digest.update(inputs['graph']['pooled_vector'].numpy().tobytes())
    return digest.hexdigest()


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    return 0


class ResidentStore:
    """Only preloaded rows may be read; input access cannot trigger disk loads."""

    def __init__(self, ctx, model, budget_gib=32):
        require(model in MODELS, 'Unknown final-comparison model')
        require(type(budget_gib) in (int, float) and np.isfinite(budget_gib) and budget_gib > 0,
                'Positive finite RAM budget required')
        self.ctx, self.model = ctx, model
        self.representation, self.view = MODELS[model]
        self.limit = int(budget_gib * 2**30)
        self.items, self.rows = {}, {}
        self.used = self.loads = self.hits = self.peak = 0
        self.phases = []

    def _estimate(self, key):
        obj = Path(self.ctx['directory']) / 'objects' / key
        require([c.sha(obj / name) for name in ('graph_receipt.json', 'embedding_receipt.json')]
                == self.ctx['receipts'][key], 'Cache receipt changed before preloading')
        statistics = c.read(obj / 'graph_receipt.json')['statistics']
        n, e = statistics['nodes'], statistics['edges']
        require(type(n) is int and n > 0 and type(e) is int and e >= 0, 'Invalid graph size receipt')
        contract = self.ctx['contract']
        return n * (4 * contract['node_feature_dim'] + 18) + e * (24 + 4 * contract['edge_feature_dim'])

    def _loader(self):
        if self.ctx.get('is_test', False):
            # Test authorization and separate predictor construction belong to the evaluator.
            from . import test_cache
            return lambda key: test_cache.load_graph(self.ctx, key)
        # The legacy loader gets no labels, role assignments or structured values.
        safe = {k: self.ctx[k] for k in ('directory', 'contract', 'receipts')}
        return graph_data.GraphStore(safe, cache_mib=0).load

    def preload(self, ids, phase):
        require(len(ids) == len(set(ids)), 'Repeated CVE in RAM working set')
        self.clear()
        started = time.perf_counter()
        rows, by_cve = {}, {}
        for cid in sorted(ids):
            key, rows[cid] = _record(self.ctx, cid, self.model)
            by_cve[cid] = key
        small_bytes = sum(tensor_bytes(row) for row in rows.values())
        keys = sorted({key for key in by_cve.values() if key}) if self.representation == 'tpg' else []
        estimates = {key: self._estimate(key) for key in
                     tqdm(keys, desc=f'Check complete {phase} RAM budget', unit='graph', leave=False)}
        planned = small_bytes + sum(estimates.values())
        require(planned <= self.limit,
                f'Complete {phase} working set needs {planned / 2**30:.3f} GiB of tensors; '
                f'budget is {self.limit / 2**30:.3f} GiB. No CVEs were dropped; increase the per-fit budget.')
        self.used = small_bytes
        load = self._loader() if keys else None
        load_seconds = 0.
        try:
            for key in tqdm(keys, desc=f'Preload all {phase} TPGs into RAM', unit='graph'):
                tick = time.perf_counter()
                graph = load(key)
                load_seconds += time.perf_counter() - tick
                require(all(isinstance(v, torch.Tensor) and v.device.type == 'cpu' and
                            not v.requires_grad for v in graph.values()), 'Graph must contain frozen CPU tensors')
                size = tensor_bytes(graph)
                require(size == estimates[key], 'Expanded graph size differs from the receipt estimate')
                require(self.used + size <= self.limit, 'Complete RAM budget exceeded; no truncation permitted')
                self.items[key] = graph
                self.used += size
                self.loads += 1
            if self.representation == 'tpg':
                for cid, key in by_cve.items():
                    rows[cid]['graph'] = self.items[key] if key else None
            require(set(self.items) == set(keys) and set(rows) == set(ids), 'Incomplete preload; refuse training')
            self.rows = rows
        except BaseException:
            self.clear()
            raise
        self.peak = max(self.peak, self.used)
        self.phases.append({'phase': phase, 'cves': len(ids), 'unique_graphs': len(keys),
                            'resident_tensor_bytes': self.used, 'load_seconds': load_seconds,
                            'preload_seconds': time.perf_counter() - started})
        return self

    def inputs(self, cid):
        require(cid in self.rows, 'CVE is not in the complete preloaded working set: ' + str(cid))
        self.hits += 1
        return self.rows[cid]

    def clear(self):
        self.rows.clear()
        self.items.clear()
        self.used = 0

    def profile(self):
        return {'resident_tensor_bytes': self.used, 'peak_resident_tensor_bytes': self.peak,
                'budget_tensor_bytes': self.limit, 'graph_loads': self.loads, 'resident_accesses': self.hits,
                'phases': list(self.phases),
                'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'budget_scope': 'Resident input tensors only; excludes activation/model/Python memory.'}
