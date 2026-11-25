"""Exact frozen TOKEN means, computed once rather than from graphs every epoch."""
from pathlib import Path
import time
import numpy as np
import torch
from tqdm import tqdm

from ..combined_training_v1 import data, model
from ..continuous_forecasting import data as publication

c = data.c
require = data.require
ROOT = data.ROOT/'01_tpg/05_workspace/04_training_acceleration_v1'


def location(dataset, scale, threads):
    return ROOT/c.DATASETS[dataset]/'01_securebert2'/f'cpu_threads_{threads}'/scale


def reduce_graph(graph, contract):
    active = graph['token_mask'] & graph['embedding_mask']
    start = len(contract['node_type_vocab'])
    vector = (graph['x'][active, start:start+contract['embedding_dim']].mean(0)
              if bool(active.any()) else graph['x'].new_zeros(contract['embedding_dim']))
    return vector, bool(active.any())


def prepare(ctx, directory, threads, verify=False):
    torch.set_num_threads(threads)
    identity = {'schema':'frozen_token_means_v1', 'contract':ctx['contract'],
                'provenance':ctx['provenance'], 'threads':threads,
                'implementation_sha256':c.sha(Path(__file__)), 'test_access':False}
    if (directory/'artifact_lock.json').exists():
        publication.check_lock(directory)
        require(c.read(directory/'identity.json') == identity, 'Pooled cache identity changed')
        return load(ctx, directory, threads)
    require(not verify, 'Pooled cache missing; run --stage vectors first')
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'identity.json').exists():
        require(c.read(directory/'identity.json') == identity, 'Partial pooled cache identity changed')
    else:
        data.write(directory/'identity.json', identity)
    store = data.GraphStore(ctx, 0)
    rows = directory/'rows'
    rows.mkdir(exist_ok=True)
    for key in tqdm(sorted(ctx['receipts']), desc='Precompute exact frozen TOKEN means', unit='graph'):
        path = rows/(key+'.json')
        expected = {'key':key, 'receipts':ctx['receipts'][key]}
        if path.exists():
            record = c.read(path)
            require(all(record[k] == v for k,v in expected.items()), 'Pooled row source changed')
            require(record['sha256'] == c.digest({k:v for k,v in record.items() if k!='sha256'}), 'Pooled row corrupted')
        else:
            vector, observed = reduce_graph(store.load(key), ctx['contract'])
            record = {**expected, 'observed':observed, 'vector':vector.tolist()}
            record['sha256'] = c.digest(record)
            temporary = path.with_suffix('.writing')
            data.write(temporary, record)
            temporary.replace(path)
        require(len(record['vector']) == ctx['contract']['embedding_dim'] and
                np.isfinite(record['vector']).all() and type(record['observed']) is bool,
                'Invalid pooled row')
    publication.check_pins(identity['provenance']['inputs'])
    publication.lock(directory)
    return load(ctx, directory, threads)


def load(ctx, directory, threads):
    publication.check_lock(directory)
    identity = c.read(directory/'identity.json')
    require(identity['contract'] == ctx['contract'] and identity['provenance'] == ctx['provenance']
            and identity['threads'] == threads and identity['implementation_sha256'] == c.sha(Path(__file__)),
            'Pooled cache contract/code/CPU reduction identity differs')
    result = {}
    for key in sorted(ctx['receipts']):
        row = c.read(directory/'rows'/(key+'.json'))
        require(row['key'] == key and row['receipts'] == ctx['receipts'][key], 'Wrong pooled graph identity')
        vector = torch.tensor(row['vector'], dtype=torch.float32)
        require(vector.shape == (ctx['contract']['embedding_dim'],) and bool(torch.isfinite(vector).all()), 'Invalid mean vector')
        result[key] = {'pooled_vector':vector, 'pooled_observed':row['observed']}
    return result


class Regressor(model.CombinedRegressor):
    def encode(self, graph):
        if self.representation == 'pooled':
            require(set(graph) == {'pooled_vector','pooled_observed'}, 'Pooled model needs a verified sufficient statistic')
            if not graph['pooled_observed']:
                return graph['pooled_vector'].new_zeros(self.hidden)
            return self.text_project(graph['pooled_vector'])
        require('pooled_vector' not in graph, 'A pooled vector is not a TPG')
        return super().encode(graph)


class Store(data.GraphStore):
    def __init__(self, ctx, representation, vectors, cache_mib):
        super().__init__(ctx, cache_mib)
        self.representation = representation
        self.vectors = vectors
        self.load_seconds = 0.

    def load(self, key):
        if self.representation == 'pooled':
            require(key in self.vectors, 'Missing precomputed pooled vector')
            self.hits += 1
            # GraphStore.inputs moves only tensor values to the requested device.
            return self.vectors[key]
        start = time.monotonic()
        value = super().load(key)
        self.load_seconds += time.monotonic()-start
        return value

    def inputs(self, cid, view, device):
        if self.representation != 'pooled':
            return super().inputs(cid, view, device)
        require(cid in self.ctx['index'] and view in data.VIEWS, 'Unknown pooled input')
        item = self.ctx['index'][cid]['views'][view]
        graph = self.load(item['object_key']) if item['object_key'] else None
        if graph is not None:
            graph = {'pooled_vector':graph['pooled_vector'].to(device), 'pooled_observed':graph['pooled_observed']}
        return {'graph':graph,
                'source_mask':torch.tensor([item['source_mask'][s] for s in data.SOURCE_ORDER], dtype=torch.float32, device=device),
                'structured':torch.tensor(self.ctx['vectors'][cid], dtype=torch.float32, device=device)}

    def profile(self):
        return {'graph_load_seconds':self.load_seconds, 'graph_loads':self.loads,
                'cache_hits':self.hits, 'resident_graph_bytes':self.used,
                'cache_limit_bytes':self.limit, 'representation':self.representation}
