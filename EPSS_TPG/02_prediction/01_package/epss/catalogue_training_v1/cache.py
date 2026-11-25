"""Screened, training-role-only D/DM cache using the unchanged v3 graph machinery."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import errno
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
from tqdm import tqdm

from tpg_combined_v3 import cache as old, compact
from tpg_combined_v2 import graphs
from tpg_combined_v1.sources import combine
from ..combined_training_v1 import data as graph_data
from ..combined_training_v2.pooled import reduce_graph
from . import common as c


def _source_documents(entry):
    result = []
    for span in entry['spans']:
        text = entry['text'][span['start']:span['end']]
        result.append({'source': span['source'], 'text': text, 'document_id': span['document_id'],
                       'text_sha256': span['text_sha256'], 'is_prose': True})
    return result


def matching_documents(documents, entry):
    expected = Counter((d['source'], d['text']) for d in documents)
    return expected == Counter((d['source'], d['text']) for d in _source_documents(entry))


def plan(cfg, dataset, scale, verify=False):
    base, all_roles = c.input_scope(cfg)
    labels = c.labels_at(base, dataset)
    roles = c.roles_for(all_roles, dataset, scale, labels, cfg)
    allowed = sorted(set().union(*map(set, roles.values())))
    screened = c.read(base/'freeze'/(dataset+'_screened.json'))
    source_root = old.directory(dataset, 'full')
    source = old.load_plan(source_root)
    parent = old.c.load_plan(source['parent_directory'])
    parser = old.c.read(Path(source['parent_directory'])/'parser_identity.json')
    encoder = c.read(source_root/'encoder_identity.json')
    entries, index, reused = {}, {}, {}
    for cid in tqdm(allowed, desc='Match screened D/DM against exact cached text', unit='CVE'):
        views = {}
        for view in cfg['views']:
            docs = [d for d in screened[cid]['documents'] if d['source'] in view]
            previous = source['index'].get(cid, {}).get('views', {}).get(view, {}).get('object_key')
            if previous and matching_documents(docs, source['entries'][previous]):
                key, entry = previous, source['entries'][previous]
                c.require(entry['cve'] == cid and entry['split'] == 'train', 'Reuse outside original training')
                reused[key] = str(source_root/'objects'/key)
            else:
                # Recover original document identities when possible, preserving their order.
                known = {}
                old_dm = source['index'].get(cid, {}).get('views', {}).get('DM', {}).get('object_key')
                if old_dm:
                    known = {(d['source'], d['text']): d for d in _source_documents(source['entries'][old_dm])}
                fixed = [known.get((d['source'], d['text']), {
                    'source': d['source'], 'text': d['text'], 'document_id': d['source']+'-'+d['sha256'],
                    'text_sha256': d['sha256'], 'is_prose': True}) for d in docs]
                text, spans = combine(fixed, view, 'prose')
                c.require(len(text) <= parent['processing']['max_characters'], 'Text too long; never truncate silently')
                entry = {'cve': cid, 'split': 'train', 'text': text, 'spans': spans,
                         'processing_identity': old.c.digest(source['processing'])}
                key = old.c.digest(entry) if spans else None
            if key:
                c.require(entry['processing_identity'] == old.c.digest(source['processing']), 'Graph processing differs')
                entries[key] = entry
            views[view] = {'object_key': key, 'source_mask': {
                s: any(d['source'] == s for d in docs) for s in 'DMGN'}}
        if not views['DM']['source_mask']['M']:
            c.require(views['D'] == views['DM'], 'D/DM masked wrapper mismatch')
        index[cid] = {'views': views}
    pins = c.code_pins()
    pins[str(base/'freeze/artifact_lock.json')] = c.sha(base/'freeze/artifact_lock.json')
    for name in ('plan.json.gz', 'plan_lock.json', 'encoder_identity.json'):
        pins[str(source_root/name)] = c.sha(source_root/name)
    value = {'schema': cfg['schema'], 'dataset': dataset, 'scale': scale, 'roles': roles,
             'entries': entries, 'index': index, 'reuse': reused, 'source_root': str(source_root),
             'processing': source['processing'], 'parent_processing': parent['processing'],
             'parser': parser, 'encoder': encoder, 'source_pins': pins, 'test_access_enabled': False}
    summary = {'dataset': dataset, 'scale': scale, 'roles': {r: len(v) for r, v in roles.items()},
               'unique_graphs': len(entries), 'exact_reuse_candidates': len(reused),
               'new_graphs': len(entries)-len(reused), 'embeddings': 'Frozen SecureBERT 2.0, unchanged revision',
               'training': False, 'official_holdout_access': False}
    root = c.cache_location(cfg, dataset, scale)
    c.publish(root/'plan', {'plan.json': value, 'findings.json': summary}, verify=verify)
    print(c.frozen.dump(summary))
    return value


def load_plan(root):
    c.frozen.check_lock(root/'plan')
    value = c.read(root/'plan/plan.json')
    c.frozen.check_pins(value['source_pins'])
    c.require(not value['test_access_enabled'], 'Official holdout access forbidden')
    return value


def receipt(root, key, plan, embedded=False):
    obj = root/'objects'/key
    if not (obj/'graph_receipt.json').exists():
        c.require(not obj.exists(), 'Unreceipted graph; inspect without overwriting: '+str(obj))
        return None
    record = c.read(obj/'graph_receipt.json')
    c.require(record['entry_sha256'] == key and record['processing_sha256'] == old.c.digest(plan['processing']),
              'Wrong graph content/processing identity')
    old.c.check_files(obj, record['files'])
    if 'parent_relative_path' in record:
        parent = root/record['parent_relative_path']
        c.require(root.resolve() in parent.resolve().parents and c.sha(parent) == record['parent_graph_sha256'],
                  'Original screened graph lineage changed')
    if embedded:
        c.require(old.embedding_receipt(obj, record, plan['encoder']) is not None, 'Missing verified embeddings')
    return record


def _reuse(root, key, plan):
    source = Path(plan['reuse'][key])
    record = c.read(source/'graph_receipt.json')
    old.c.check_files(source, record['files'])
    embedding = old.embedding_receipt(source, record, plan['encoder'])
    c.require(embedding is not None, 'Reusable object has incomplete embeddings')
    c.require(record['entry_sha256'] == key and record['processing_sha256'] == old.c.digest(plan['processing']),
              'Reusable object has different processing')
    names = ['graph_receipt.json', 'embedding_receipt.json', *record['files'], *embedding['files']]
    (root/'objects').mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.reuse-', dir=root) as temp:
        stage = Path(temp)/'object'
        stage.mkdir()
        for name in names:
            # Byte-preserving hardlinks; this pipeline never edits a published object.
            link_verified(source/name, stage/name)
        stage.rename(root/'objects'/key)


def link_verified(source, target):
    try:
        os.link(source, target)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copyfile(source, target)
    c.require(c.sha(source) == c.sha(target), 'Byte-preserving reuse verification failed')


_builder = None


def initialize():
    global _builder
    from tpg_combined_recovery_v1.boundary import Builder
    torch.set_num_threads(1)
    _builder = Builder()


def build_one(job):
    root, key, entry, metadata = job
    root = Path(root)
    c.require(_builder.identity == metadata['parser'], 'Parser no longer matches frozen cache')
    source_entry = {**entry, 'processing_identity': old.c.digest(metadata['parent_processing'])}
    parent_key = old.c.digest(source_entry)
    original, boundary = _builder.build(parent_key, source_entry)
    parent = root/'parents'/parent_key/'graph.json.gz'
    if parent.exists():
        c.require(old.c.read(parent) == original, 'Original screened extraction changed')
    else:
        old.c.create(parent, original)
    original_sha = c.sha(parent)
    graph = old.transform(original, entry, source_entry, parent_key, original_sha)
    graphs.validate(entry, graph)
    tensors = graphs.tensors(graph)
    arrays, header = compact.pack(tensors, graph)
    restored = compact.unpack(arrays, header)
    for name, tensor in tensors.items():
        c.require(torch.equal(tensor, restored[name]) if isinstance(tensor, torch.Tensor)
                  else tensor == restored[name], 'Structural round trip differs: '+name)
    record = {'entry_sha256': key, 'processing_sha256': old.c.digest(metadata['processing']),
              'parent_key': parent_key, 'parent_graph_sha256': original_sha,
              'parent_relative_path': str(parent.relative_to(root)),
              'statistics': graphs.statistics(graph), 'unique_pooling_rows': len(arrays['pool_ptr'])-1,
              'boundary_repair': boundary, 'parser': metadata['parser'], 'scope': 'CC02 original-training inputs only'}
    old.publish(root, key, graph, arrays, header, record)
    return key


def graph_stage(root, plan, workers):
    from tpg_combined_parallel_v1.runner import thread_environment, stop_executor, available_memory_gib, scheduled_keys
    c.require(1 <= workers <= 32, 'Use 1..32 graph workers')
    c.require(available_memory_gib() > 6*workers+2, 'Insufficient graph-worker RAM')
    c.require(shutil.disk_usage(root).free > 20*2**30, 'Keep 20 GiB disk reserve')
    missing = {}
    for key in tqdm(sorted(plan['entries']), desc='Verify/reuse immutable screened graph objects', unit='graph'):
        if receipt(root, key, plan):
            continue
        if key in plan['reuse']:
            _reuse(root, key, plan)
        else:
            missing[key] = plan['entries'][key]
    metadata = {k: plan[k] for k in ('processing', 'parent_processing', 'parser')}
    with thread_environment():
        pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'), initializer=initialize)
        try:
            keys, pending = iter(scheduled_keys(missing)), set()
            with tqdm(total=len(missing), desc='Build unchanged typed TPGs for screened text', unit='graph') as bar:
                while True:
                    while len(pending) < 2*workers:
                        key = next(keys, None)
                        if key is None:
                            break
                        pending.add(pool.submit(build_one, (str(root), key, missing[key], metadata)))
                    if not pending:
                        break
                    done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                    if not done:
                        tqdm.write(f'[CC03 CACHE] {bar.n}/{len(missing)} built; {len(pending)} active/queued')
                    for future in done:
                        future.result()
                        bar.update()
        finally:
            stop_executor(pool)


def embed_stage(root, plan):
    from tpg_combined_v3.embeddings import Encoder
    encoder = None
    c.immutable_json(root/'encoder_identity.json', plan['encoder'])
    for key, entry in tqdm(sorted(plan['entries'].items()), desc='Embed screened graph spans; reuse exact vectors', unit='graph'):
        obj = root/'objects'/key
        record = receipt(root, key, plan)
        c.require(record is not None, 'Finish graph stage before embeddings')
        if old.embedding_receipt(obj, record, plan['encoder']):
            continue
        if encoder is None:
            encoder = Encoder(plan['encoder']['device'], plan['encoder']['threads'])
            c.require(encoder.identity == plan['encoder'], 'Frozen SecureBERT identity differs')
        c.require(shutil.disk_usage(root).free > 20*2**30 + record['unique_pooling_rows']*768*4,
                  'Insufficient embedding disk reserve')
        graph = old.c.read(obj/'graph.json.gz')
        layout, _ = compact.load(obj/'structure.npz')
        rows, observed, coverage = encoder.encode_compact(entry['text'], graph, layout)
        with tempfile.TemporaryDirectory(prefix='.embed-', dir=root) as tmp:
            stage = Path(tmp)
            compact.save(stage/'embeddings.npz', {'rows': rows, 'observed': observed},
                         {'format': compact.FORMAT, 'encoder_sha256': old.c.digest(plan['encoder'])})
            er = {'graph_sha256': record['files']['graph.json.gz'], 'structure_sha256': record['files']['structure.npz'],
                  'encoder_sha256': old.c.digest(plan['encoder']), 'coverage': coverage,
                  'files': {'embeddings.npz': c.sha(stage/'embeddings.npz')}}
            c.require(not (obj/'embedding_receipt.json').exists(), 'Never replace completed embeddings')
            # An interrupted unreceipted payload is preserved, not silently deleted.
            if (obj/'embeddings.npz').exists():
                target = root/'incomplete_embeddings'/key
                target.mkdir(parents=True, exist_ok=False)
                (obj/'embeddings.npz').rename(target/'embeddings.npz')
            (stage/'embeddings.npz').rename(obj/'embeddings.npz')
            c.immutable_json(obj/'embedding_receipt.json', er)


def verify_cache(root, plan, verify=False):
    torch.set_num_threads(4)
    c.require(c.read(root/'encoder_identity.json') == plan['encoder'], 'Cache encoder mismatch')
    rows, receipts, statistics = {}, {}, {}
    resources = None
    for key, entry in tqdm(sorted(plan['entries'].items()), desc='Verify graph text/types/spans and frozen embedding means', unit='graph'):
        obj = root/'objects'/key
        record = receipt(root, key, plan, embedded=True)
        c.require(record is not None, 'Graph is missing')
        graph = old.c.read(obj/'graph.json.gz')
        graphs.validate(entry, graph)
        c.require(graph['doc_id'] == 'CTPG3-'+key
                  and graph['metadata']['combined_text']['processing_identity'] == entry['processing_identity'],
                  'Graph owner/processing changed')
        tensors = old.load_tensors(root, key, True)
        contract = graph_data.contract(graph_data.header(obj/'structure.npz'), plan['encoder'], [])
        if resources is None:
            resources = contract
        c.require(resources == contract, 'Mixed vocabularies or edge-feature schemas')
        graph_data.validate_tensors(tensors, contract)
        from tpg_combined_v2.embeddings import model_inputs
        expanded = {'x': model_inputs(tensors)['x'], 'token_mask': tensors['node_type'] == contract['node_type_vocab']['TOKEN'],
                    'embedding_mask': tensors['embedding_mask']}
        vector, observed = reduce_graph(expanded, contract)
        rows[key] = {'vector': vector.tolist(), 'observed': observed}
        receipts[key] = [c.sha(obj/name) for name in ('graph_receipt.json', 'embedding_receipt.json')]
        statistics[key] = record['statistics']
    c.frozen.check_pins(plan['source_pins'])
    c.publish(root/'verified', {'means.json': rows, 'receipts.json': receipts,
              'contract.json': resources, 'statistics.json': statistics,
              'identity.json': {'plan_sha256': c.sha(root/'plan/artifact_lock.json'), 'test_access': False}}, verify=verify)
    print(f'CC03 {plan["dataset"]} {plan["scale"]} screened cache verified; no training or official holdouts.')
