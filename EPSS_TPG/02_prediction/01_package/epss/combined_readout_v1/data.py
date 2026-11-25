"""Lossless compact-cache TOKEN readout; no parser, encoder, edges or test tensors."""
from concurrent.futures import ProcessPoolExecutor
import io
import json
import multiprocessing
from pathlib import Path
import platform
import zipfile

import numpy as np
import torch
from tqdm import tqdm

from ..combined_training_v1 import data as base
from ..combined_training_v2 import runner as execution, pooled
from ..continuous_forecasting import data as publication

c, require, ROOT = base.c, base.require, base.ROOT
write = base.write
CONFIG = ROOT/'02_prediction/04_configs/07_fresh_studies/03_source_readout_v1.json'
CLI = ROOT/'02_prediction/03_analysis/07_fresh_studies/05_run_source_readout.py'
CACHE = ROOT/'01_tpg/05_workspace/05_source_readout_v1/02_previous_multisource/01_securebert2'
RESULTS = ROOT/'05_results/08_fresh_studies/09_source_readout_v1/02_previous_multisource/01_securebert2'


def settings():
    p = c.read(CONFIG)
    require(p['schema'] == 'ct03_source_readout_v1' and p['dataset'] == 'previous' and
            p['view'] == 'DM' and not p['test_enabled'], 'Wrong development-only protocol')
    require(p['arms'] == ['merged', 'source'] and p['objectives'] == ['mae', 'mse'] and
            p['seeds'] == [17, 29, 43] and p['device'] == 'cpu' and p['threads'] == 4,
            'Changed bounded comparison; version the protocol')
    require(p['hidden'] == p['text_total_width'] == 32 and p['source_widths'] == [16, 16], 'Parameter matching changed')
    return p


def pins():
    return {**execution.pins(), **{str(p): c.sha(p) for p in sorted([CONFIG, CLI, *Path(__file__).parent.glob('*.py')])}}


def identity(ctx, scale):
    import sklearn
    return {'schema': 'ct03_source_readout_v1', 'settings': settings(), 'scale': scale,
            'provenance': ctx['provenance'], 'contract': ctx['contract'], 'source_pins': pins(),
            'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                        'numpy': np.__version__, 'sklearn': sklearn.__version__}, 'test_access': False}


def archive_arrays(path, names):
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read('metadata.json'))
        arrays = {name: np.load(io.BytesIO(z.read(name+'.npy')), allow_pickle=False) for name in names}
    return arrays, meta


def reduce_tokens(rows, observed, index, token, sources, documents, span_sources):
    require(rows.dtype == np.float32 and rows.ndim == 2 and rows.shape[1] == 768 and
            np.isfinite(rows).all(), 'Invalid compact embedding rows')
    require(observed.dtype == np.bool_ and observed.shape == (len(rows),), 'Invalid observed mask')
    require(index.ndim == 1 and index.dtype.kind in 'iu' and np.all((index >= 0) & (index < len(rows))), 'Invalid pooling index')
    n = len(index)
    require(token.shape == documents.shape == (n,) and token.dtype == np.bool_ and
            sources.dtype == np.bool_ and sources.shape == (n, 4), 'Invalid node source layout')
    active = token & observed[index]
    require(active.any() and np.all(documents[token] >= 0) and np.all(documents[token] < len(span_sources)),
            'TOKEN assigned to global/unknown document')
    require(np.all(sources[token].sum(axis=1) == 1) and not sources[token, 2:].any(), 'DM TOKEN source is ambiguous or G/N')
    expected = np.array([[s == name for s in 'DMGN'] for name in span_sources], dtype=bool)
    require(np.array_equal(sources[token], expected[documents[token]]), 'Document/source identity mismatch')
    # Preserve the node order and torch float32 reduction used by CT02; never expand edges.
    values = torch.from_numpy(rows[index[active]])
    source_active = sources[active, :2]
    counts = source_active.sum(axis=0)
    separate = [values[torch.from_numpy(source_active[:, j])].mean(0) if counts[j] else torch.zeros(768)
                for j in range(2)]
    merged = values.mean(0)
    reconstructed = sum(separate[j]*int(counts[j]) for j in range(2))/int(counts.sum())
    require(torch.allclose(merged, reconstructed, atol=2e-6, rtol=2e-6), 'Weighted source mean does not recover merged mean')
    return {'merged': merged.tolist(), 'source': [v.tolist() for v in separate], 'counts': counts.tolist(),
            'observed': (counts > 0).tolist(), 'tokens': int(token.sum()), 'observed_tokens': int(active.sum()),
            'weighted_reconstruction_max_abs': float((merged-reconstructed).abs().max())}


def extract(job):
    directory, cid, entry, sources, receipt_hashes, encoder_sha, threads = job
    torch.set_num_threads(threads)
    key = entry['object_key']; obj = Path(directory)/'objects'/key
    require([c.sha(obj/'graph_receipt.json'), c.sha(obj/'embedding_receipt.json')] == receipt_hashes, 'Changed source receipts')
    gr, er = c.read(obj/'graph_receipt.json'), c.read(obj/'embedding_receipt.json')
    c.check_files(obj, gr['files']); c.check_files(obj, er['files'])
    a, meta = archive_arrays(obj/'structure.npz', ['pool_index', 't_token_mask', 't_source_mask', 't_document_index'])
    b, emeta = archive_arrays(obj/'embeddings.npz', ['rows', 'observed'])
    require(emeta['encoder_sha256'] == er['encoder_sha256'] == encoder_sha and
            er['structure_sha256'] == gr['files']['structure.npz'], 'Embedding lineage mismatch')
    require(meta['vocabulary']['node_types'].get('TOKEN') is not None, 'Missing TOKEN vocabulary')
    row = {'cve': cid, 'key': key, 'receipts': receipt_hashes,
           'mask': [entry['source_mask'][s] for s in 'DMGN'],
           **reduce_tokens(b['rows'], b['observed'], a['pool_index'], a['t_token_mask'],
                           a['t_source_mask'], a['t_document_index'], sources)}
    require(all(not flag or row['mask'][j] for j, flag in enumerate(row['observed'])), 'Observed source absent in plan')
    row['sha256'] = c.digest(row)
    return row


def validate_row(row, cid, ctx):
    item = ctx['index'][cid]['views']['DM']
    require(row['cve'] == cid and row['key'] == item['object_key'] and
            row['receipts'] == ctx['receipts'][row['key']] and
            row['sha256'] == c.digest({k: v for k, v in row.items() if k != 'sha256'}), 'Invalid readout row identity')
    require(np.asarray(row['merged']).shape == (768,) and np.asarray(row['source']).shape == (2, 768) and
            np.isfinite(row['merged']).all() and np.isfinite(row['source']).all(), 'Invalid readout shape')


def prepare(ctx, ident, workers=1, resume=False, verify=False):
    require(1 <= workers <= 16, 'Use 1-16 bounded extraction workers')
    import os
    require(workers*4 <= len(os.sched_getaffinity(0)), 'Extraction workers exceed CPU affinity')
    out = CACHE/ident['scale']; plan = c.read(ctx['directory']/'plan.json.gz')
    ids = ctx['ids']['train'] + ctx['ids']['validation']
    with execution.exclusive(CACHE/(ident['scale']+'.writer.lock')):
        if (out/'artifact_lock.json').exists():
            publication.check_lock(out)
            require(c.read(out/'identity.json') == ident, 'Readout identity changed')
            if not verify:
                require(resume, 'Readouts exist; use --resume or --verify-only')
                return
        else:
            require(not verify, 'Readout cache missing')
            if (out/'identity.json').exists():
                require(resume and c.read(out/'identity.json') == ident, 'Partial cache differs; use matching --resume')
            else:
                write(out/'identity.json', ident)
        jobs = []
        for cid in ids:
            path = out/'rows'/(cid+'.json')
            if path.exists() and not verify:
                validate_row(c.read(path), cid, ctx)
                continue
            item = ctx['index'][cid]['views']['DM']; key = item['object_key']
            require(key is not None, 'Expected description in every DM record')
            jobs.append((str(ctx['directory']), cid, item, [s['source'] for s in plan['entries'][key]['spans']],
                         ctx['receipts'][key], c.digest(ctx['contract']['encoder_identity']), 4))
        def publish(row):
            validate_row(row, row['cve'], ctx)
            path = out/'rows'/(row['cve']+'.json')
            if verify:
                require(c.read(path) == row, 'Recomputed readout differs')
            else:
                write(path, row)
        if workers == 1:
            for job in tqdm(jobs, desc='Read compact DM tokens and source means', unit='CVE'):
                publish(extract(job))
        else:
            with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                for row in tqdm(pool.map(extract, jobs, chunksize=1), total=len(jobs), desc='Read compact DM tokens and source means', unit='CVE'):
                    publish(row)
        rows = [c.read(out/'rows'/(cid+'.json')) for cid in ids]
        for cid, row in zip(ids, rows): validate_row(row, cid, ctx)
        reference = pooled.location('previous', ident['scale'], 4)
        require((reference/'artifact_lock.json').exists(), 'CT02 pooled cache required for exact merged-vector parity')
        reference_vectors = pooled.load(ctx, reference, 4)
        for row in tqdm(rows, desc='Require exact CT02 merged TOKEN-mean parity', unit='CVE'):
            require(torch.equal(torch.tensor(row['merged'], dtype=torch.float32),
                                reference_vectors[row['key']]['pooled_vector']), 'Merged-vector parity failed')
        audit = {'cohort': {s: len(v) for s, v in ctx['ids'].items()},
                 'reference_lock': {str(reference/'artifact_lock.json'): c.sha(reference/'artifact_lock.json')},
                 'nodes_used': 'Observed TOKENs only; GLOBAL excluded in both pooled arms',
                 'merged_exact_parity': True, 'edge_arrays_loaded': False,
                 'source_observation_counts': {s: [sum(c.read(out/'rows'/(cid+'.json'))['observed'][j] for cid in cs)
                                                  for j in range(2)] for s, cs in ctx['ids'].items()}}
        if verify:
            require(c.read(out/'audit.json') == audit, 'Readout audit differs')
        else:
            write(out/'audit.json', audit); publication.lock(out)
        publication.check_pins(ident['source_pins']); publication.check_pins(ident['provenance']['inputs'])
    print(f'CT03 {ident["scale"]} source readouts '+('reproduced' if verify else 'saved')+f': {out}. No training or test access.')


def load(ctx, ident):
    out = CACHE/ident['scale']; publication.check_lock(out)
    require(c.read(out/'identity.json') == ident, 'Readout cache identity mismatch')
    publication.check_pins(c.read(out/'audit.json')['reference_lock'])
    inputs = {}
    for cid in ctx['ids']['train']+ctx['ids']['validation']:
        row = c.read(out/'rows'/(cid+'.json')); validate_row(row, cid, ctx)
        inputs[cid] = {k: torch.tensor(row[k], dtype=torch.float32) for k in ('merged', 'source', 'observed', 'mask')}
        inputs[cid]['structured'] = torch.tensor(ctx['vectors'][cid], dtype=torch.float32)
    return inputs, c.sha(out/'artifact_lock.json')
