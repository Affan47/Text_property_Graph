"""Held-out predictor cache and explicit, immutable evaluation release.

No function runs on import. ``prepare`` requires a release made with
``freeze_release`` after all 25 official fits complete. Labels are accessible
only inside ``evaluation_release(..., release_test=True)``. The old development
cache loaders remain unchanged: test role checks and publication live here.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import csv
from datetime import date, timedelta
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
import torch
from tqdm import tqdm

from . import common as c
from ..reddit_primary import structured
from tpg_combined_v1.sources import combine
from tpg_combined_v2 import common as graph_common, graphs
from tpg_combined_v3 import cache as compact_cache, compact

SCHEMA = 'final_comparison_test_cache_v1'
RELEASE_SCHEMA = 'final_comparison_test_release_v1'
SEEDS = (17, 29, 43, 59, 71)
VIEWS = ('D', 'DM')
_evaluation = ContextVar('final_comparison_test_evaluation', default=None)


def source_paths():
    """Metadata only; caller must pin these paths before training starts."""
    audit = c.ROOT/'05_results/03_dataset_analysis/02_reddit_dataset'
    lock = audit/'08_rd01_source_lock_v1'
    sanitized = audit/'04_premodeling_gates_v1/stage1_sanitization'
    return {
        'manifest': lock/'rd01_frozen_manifest.json',
        'policy': lock/'source_policy.json',
        'decisions': lock/'source_document_decisions.csv.gz',
        'approval': audit/'10_rd02_overlap_review_v1/source_approval.json',
        'candidates': sanitized/'predictor_candidates.jsonl.gz',
        'outcomes': sanitized/'outcomes_restricted.jsonl.gz',
    }


def _paths(protocol):
    return {k: Path(v) for k, v in protocol.get('test_source_paths', source_paths()).items()}


def _pins(protocol):
    result = {}
    for key in ('input_pins', 'code_pins'):
        for path, value in protocol[key].items():
            name = str(Path(path).resolve())
            c.require(name not in result or result[name] == value, 'Conflicting frozen pins')
            result[name] = value
    return result


def _pinned(protocol, path):
    path = Path(path).resolve()
    expected = _pins(protocol).get(str(path))
    c.require(expected is not None and path.is_file() and c.sha(path) == expected,
              'Unpinned or changed prerequisite: '+str(path))
    return path


def _official_ids(protocol):
    ids = protocol['official_ids']
    c.require(set(ids) == {'train', 'validation', 'test'}, 'Expected official train/validation/test IDs')
    c.require(all(isinstance(cs, list) and len(cs) == len(set(cs)) for cs in ids.values()),
              'Repeated official IDs')
    all_ids = [cid for cs in ids.values() for cid in cs]
    c.require(len(all_ids) == len(set(all_ids)), 'Official roles overlap')
    c.require(tuple(protocol['settings']['seeds']) == SEEDS, 'Five frozen seeds required')
    return ids


def _state_path():
    return c.OUT/'prepared/run2/fold_0/structured_state.json'


def _check_fit(record):
    checkpoint = Path(record['path']).resolve()
    lock = Path(record['completion_lock']).resolve()
    c.require(lock.is_file() and c.sha(lock) == record['completion_lock_sha256'],
              'Missing or changed completed-fit lock')
    files = c.read(lock)['files']
    graph_common.check_files(lock.parent, files)
    c.require(checkpoint.is_relative_to(lock.parent), 'Checkpoint outside its completed fit')
    relative = str(checkpoint.relative_to(lock.parent))
    c.require(relative in files and files[relative] == record['sha256'] == c.sha(checkpoint),
              'Best checkpoint is not covered by its completed-fit lock')


def freeze_release(protocol, fit_records, constants_path, *, output=None, comparison_files=None):
    """Publish the runner's completed official checkpoint set, without test reads.

    Each record is ``{model, seed, path, completion_lock}``. The runner must
    first verify Run-2 identity/checkpoint replay; this helper independently
    checks the full Cartesian product and completion-lock bytes. The helper
    never deserializes checkpoints or fits preprocessing.
    """
    ids = _official_ids(protocol)
    c.require(all(str(path.resolve()) in _pins(protocol) for path in _paths(protocol).values()),
              'Freeze every test-source path before any training, including restricted outcomes')
    wanted = {(model, seed) for model in c.MODELS for seed in SEEDS}
    c.require(len(fit_records) == 25 and {(r['model'], r['seed']) for r in fit_records} == wanted,
              'Release requires all 25 official checkpoints, without duplicates')
    records = []
    for r in sorted(fit_records, key=lambda r: (r['model'], r['seed'])):
        path, lock = Path(r['path']).resolve(), Path(r['completion_lock']).resolve()
        record = {'model': r['model'], 'seed': r['seed'], 'path': str(path),
                  'sha256': c.sha(path), 'completion_lock': str(lock), 'completion_lock_sha256': c.sha(lock)}
        _check_fit(record)
        records.append(record)
    c.require(len({r['path'] for r in records}) == 25, 'A checkpoint was assigned to multiple fits')
    state_path = _state_path()
    state = c.read(state_path)
    c.require(sorted(state['fit_ids']) == sorted(ids['train']), 'Structured state is not official-training-only')
    c.require(state['feature_names'] == protocol['contract_template']['structured_names'],
              'Official structured columns differ from the frozen contract')
    constants_path = Path(constants_path).resolve()
    c.require(constants_path.is_file(), 'Missing official-training constants')
    locked = {str(state_path.resolve()): c.sha(state_path), str(constants_path): c.sha(constants_path)}
    for path, expected in (comparison_files or {}).items():
        path = Path(path).resolve()
        c.require(c.sha(path) == expected, 'Changed pre-release comparison report')
        locked[str(path)] = expected
    # Recheck code/predictor metadata. Do not open the outcomes stream here.
    for path, expected in _pins(protocol).items():
        if Path(path) == _paths(protocol)['outcomes'].resolve():
            continue
        c.require(c.sha(path) == expected, 'Changed frozen input/code: '+path)
    release = {'schema': RELEASE_SCHEMA, 'protocol_sha256': c.digest(protocol),
               'official_ids_sha256': c.digest(ids), 'checkpoints': records, 'files': locked,
               'constants_path': str(constants_path), 'structured_state_path': str(state_path.resolve()),
               'test_predictors_authorized': True, 'test_labels_released': False}
    output = Path(output or c.OUT/'release_lock.json')
    with c.exclusive(output.with_suffix('.writer.lock')):
        if output.exists():
            c.require(c.read(output) == release, 'A different test release is already frozen')
        else:
            c.write(output, release)
    return release


def verify_release(protocol, release_lock):
    """Fail before opening test sources unless every selected fit remains locked."""
    release = c.read(release_lock) if isinstance(release_lock, (str, Path)) else release_lock
    ids = _official_ids(protocol)
    c.require(isinstance(release, dict) and release.get('schema') == RELEASE_SCHEMA
              and release.get('protocol_sha256') == c.digest(protocol)
              and release.get('official_ids_sha256') == c.digest(ids)
              and release.get('test_predictors_authorized') is True
              and release.get('test_labels_released') is False, 'Missing or changed official release')
    records = release['checkpoints']
    c.require(len(records) == 25 and {(r['model'], r['seed']) for r in records}
              == {(m, s) for m in c.MODELS for s in SEEDS}
              and len({r['path'] for r in records}) == 25, 'Incomplete 25-checkpoint release')
    for r in records:
        _check_fit(r)
    for path, expected in release['files'].items():
        c.require(c.sha(path) == expected, 'Changed locked transform/constants: '+path)
    state_path = Path(release['structured_state_path'])
    c.require(str(state_path) in release['files'] and str(release['constants_path']) in release['files'],
              'Unpinned structured state/constants')
    c.require(sorted(c.read(state_path)['fit_ids']) == sorted(ids['train']), 'Wrong structured fitting IDs')
    for path, expected in protocol['code_pins'].items():
        c.require(c.sha(path) == expected, 'Changed frozen evaluator/code: '+path)
    for path, expected in protocol['input_pins'].items():
        if Path(path).resolve() == _paths(protocol)['outcomes'].resolve():
            continue
        c.require(c.sha(path) == expected, 'Changed frozen predictor prerequisite: '+path)
    return release


def _manifest(protocol):
    paths = _paths(protocol)
    manifest = c.read(_pinned(protocol, paths['manifest']))
    task = manifest['tasks']['RD02']
    c.require(manifest['role_state'] == 'FROZEN' and task['ids'] == _official_ids(protocol),
              'Test IDs differ from frozen RD01 roles')
    c.require(manifest['prior_test_connected_excluded'] == 4987, 'Prior-test exclusions changed')
    approval = c.read(_pinned(protocol, paths['approval']))
    c.require(approval['source_gate'] == 'ACCEPTED_WITH_DECLARED_DESCRIPTION_OVERLAP'
              and approval['original_manifest_sha256'] == c.sha(paths['manifest'])
              and approval['role_ids_sha256'] == task['role_ids_sha256'], 'Source approval differs')
    _pinned(protocol, paths['policy'])
    return manifest


def _fingerprint(post):
    # Use the original sanitizer's exact serializer, not a new deduplication rule.
    from ..reddit_primary.data import fingerprint
    return fingerprint(post)


def screen_predictor(raw, decisions):
    """Same document selection as RD02; no label/catalogue fields copied."""
    cid = raw['cve']
    candidates = [('D', raw['description'], '')] + [
        ('M', post['post'], _fingerprint(post)) for post in raw['social_media_posts']]
    docs = []
    for source, text, fingerprint in candidates:
        if not isinstance(text, str) or not text.strip():
            continue
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        key = (cid, source, text_hash, fingerprint)
        c.require(key in decisions, 'Missing frozen source decision: '+cid)
        decision = decisions[key]
        for field in ('quarantine', 'bare_id'):
            c.require(str(decision[field]).lower() in ('true', 'false'), 'Ambiguous source decision')
        if str(decision['quarantine']).lower() == 'true':
            continue
        prose = source == 'M' and str(decision['bare_id']).lower() != 'true'
        if source == 'M' and not prose:
            continue
        docs.append({'document_id': source+'-'+(fingerprint or text_hash), 'source': source,
                     'text': text, 'text_sha256': text_hash, 'is_prose': prose})
    c.require(len({d['document_id'] for d in docs}) == len(docs), 'Repeated screened document: '+cid)
    # Missing sources remain masked; never silently drop a held-out CVE.
    return {'cve': cid, 'split': 'test', 'documents': docs,
            'structured': {k: raw.get(k) for k in structured.FIELDS}}


def document_groups(rows):
    """Label-blind connected components of NFC/whitespace-exact D/M documents."""
    from ..training_fold_audit_v1 import core
    inputs = {row['cve']: {'document_fingerprints': [core.fingerprint(d['text'])
              for d in row['documents']]} for row in rows}
    c.require(len(inputs) == len(rows), 'Repeated grouping CVE')
    return core.components(inputs)[0]


def _jsonl(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def _root():
    return c.OUT/'test_cache'


def _development_metadata(protocol):
    directory = Path(protocol['graph_directory'])
    # This only reads the already-frozen development plan, never test payloads.
    plan = compact_cache.load_plan(directory)
    c.require(plan['processing'] == protocol['cache_processing'], 'Development processing identity changed')
    parent = graph_common.load_plan(plan['parent_directory'])
    c.require(parent['processing']['smp_policy'] == 'prose', 'Expected the existing prose-post policy')
    encoder = c.read(directory/'encoder_identity.json')
    c.require(encoder == protocol['cache_encoder'], 'Development encoder changed')
    return parent['processing'], c.read(Path(plan['parent_directory'])/'parser_identity.json')


def _make_plan(rows, protocol, parent_processing):
    processing = protocol['cache_processing']
    entries, index = {}, {}
    limit = parent_processing['max_characters']
    for row in rows:
        views = {}
        for view in VIEWS:
            text, spans = combine(row['documents'], view, 'prose')
            c.require(len(text) <= limit, 'Held-out text exceeds the existing limit; no silent truncation')
            entry = {'cve': row['cve'], 'split': 'test', 'text': text, 'spans': spans,
                     'processing_identity': c.digest(processing)}
            key = c.digest(entry) if spans else None
            if key:
                entries[key] = entry
            mask = {source: any(s['source'] == source for s in spans) for source in 'DMGN'}
            views[view] = {'object_key': key, 'source_mask': mask,
                           'all_requested_sources_present': all(mask[source] for source in view)}
        index[row['cve']] = {'split': 'test', 'views': views}
    return {'schema': SCHEMA, 'test_adapted': True, 'labels_included': False, 'variants': list(VIEWS),
            'processing': deepcopy(processing), 'parent_processing': parent_processing,
            'entries': entries, 'index': index}


def _prepare_inputs(protocol, release, verify_only=False):
    destination = _root()/'inputs'
    c.require(not verify_only or destination.exists(), 'No test inputs to verify')
    manifest = _manifest(protocol)
    paths = _paths(protocol)
    decisions = {}
    selected = set(protocol['official_ids']['test'])
    with gzip.open(_pinned(protocol, paths['decisions']), 'rt', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row['task'] != 'RD02' or row['cve'] not in selected:
                continue
            c.require(row['role'] == 'test', 'Wrong test source-decision role')
            key = (row['cve'], row['source'], row['text_sha256'], row['post_fingerprint'])
            c.require(key not in decisions, 'Duplicate frozen source decision')
            decisions[key] = row
    rows, dates = [], {}
    for raw in tqdm(_jsonl(_pinned(protocol, paths['candidates'])), desc='Prepare locked test predictors only', unit='CVE'):
        cid = raw['cve']
        if cid not in selected:
            continue
        c.require(cid not in dates, 'Repeated test CVE')
        rows.append(screen_predictor(raw, decisions))
        dates[cid] = raw['published_date']
    c.require(set(dates) == selected, 'Missing test predictors; no CVEs may be silently dropped')
    rows.sort(key=lambda r: r['cve'])
    state = c.read(release['structured_state_path'])
    vectors = {r['cve']: structured.transform(r['structured'], state)[0] for r in rows}
    parent, parser = _development_metadata(protocol)
    plan = _make_plan(rows, protocol, parent)
    plan['release_sha256'] = c.digest(release)
    plan['protocol_sha256'] = c.digest(protocol)
    plan['groups'] = document_groups(rows)
    plan['publication_dates'] = dates
    plan['target_version_schedule'] = manifest['release_schedule']
    if destination.exists():
        c.publication.check_lock(destination)
        c.require(c.read(destination/'plan.json.gz') == plan
                  and c.read(destination/'structured_vectors.json') == vectors, 'Test input replay differs')
        return
    with c.publication.transaction(destination) as stage:
        graph_common.create(stage/'plan.json.gz', plan)
        graph_common.create(stage/'structured_vectors.json', vectors)
        graph_common.create(stage/'parser_identity.json', parser)
        graph_common.create(stage/'encoder_identity.json', protocol['cache_encoder'])
        graph_common.create(stage/'contract.json', protocol['contract_template'])
        graph_common.create(stage/'summary.json', {'cves': len(rows), 'graphs': len(plan['entries']),
            'cves_with_prose_M': sum(plan['index'][r['cve']]['views']['DM']['source_mask']['M'] for r in rows),
            'labels_included': False, 'grouping': 'NFC + whitespace exact D/M document components; no case folding'})


def _load_plan(protocol, release):
    directory = _root()/'inputs'
    c.publication.check_lock(directory)
    plan = c.read(directory/'plan.json.gz')
    c.require(plan['schema'] == SCHEMA and plan['test_adapted'] is True and not plan['labels_included']
              and plan['protocol_sha256'] == c.digest(protocol) and plan['release_sha256'] == c.digest(release),
              'Test plan/release identity mismatch')
    c.require(set(plan['index']) == set(protocol['official_ids']['test'])
              and all(row['split'] == 'test' for row in plan['index'].values())
              and plan['processing'] == protocol['cache_processing'], 'Changed held-out role/processing')
    c.require(c.read(directory/'encoder_identity.json') == protocol['cache_encoder']
              and c.read(directory/'contract.json') == protocol['contract_template'], 'Changed cache contract')
    for key, entry in plan['entries'].items():
        c.require(entry['split'] == 'test' and entry['cve'] in plan['index'] and c.digest(entry) == key,
                  'Invalid test entry identity')
    c.require(set(plan['publication_dates']) == set(plan['groups']) == set(plan['index']),
              'Test grouping/publication IDs differ')
    used = set()
    for cid, row in plan['index'].items():
        c.require(set(row['views']) == set(VIEWS), 'Unexpected test source views')
        for view in VIEWS:
            item = row['views'][view]
            flags, key = item['source_mask'], item['object_key']
            c.require(set(flags) == set('DMGN') and all(type(x) is bool for x in flags.values())
                      and bool(key) == any(flags.values()), 'Invalid test source masks')
            if key:
                c.require(key in plan['entries'] and plan['entries'][key]['cve'] == cid, 'Wrong test graph owner')
                sources = {s['source'] for s in plan['entries'][key]['spans']}
                c.require(sources <= set(view) and sources == {s for s, active in flags.items() if active},
                          'Test source/graph mapping differs')
                used.add(key)
        if not row['views']['DM']['source_mask']['M']:
            c.require(row['views']['D']['object_key'] == row['views']['DM']['object_key'], 'Masked DM differs from D')
    c.require(used == set(plan['entries']), 'Unreferenced test graph objects')
    return plan


def _content(entry):
    """Role-free extraction arguments, not a relabelled development record."""
    return {k: entry[k] for k in ('cve', 'text', 'spans', 'processing_identity')}


def _object_record(root, key, plan):
    obj = Path(root)/'objects'/key
    receipt = obj/'graph_receipt.json'
    if not receipt.exists():
        c.require(not obj.exists(), 'Unreceipted test graph exists; inspect without overwriting')
        return None
    record = c.read(receipt)
    source_entry = {**_content(plan['entries'][key]), 'processing_identity': c.digest(plan['parent_processing'])}
    c.require(record['entry_sha256'] == key and record['processing_sha256'] == c.digest(plan['processing'])
              and record['split'] == 'test' and record['release_sha256'] == plan['release_sha256']
              and record['parent_key'] == c.digest(source_entry)
              and record['parser'] == c.read(Path(root)/'inputs/parser_identity.json')
              and set(record['files']) == {'graph.json.gz', 'structure.npz'}, 'Test graph receipt mismatch')
    graph_common.check_files(obj, record['files'])
    parent = Path(root)/'parents'/record['parent_key']/'object'
    c.require(c.sha(parent/'graph.json.gz') == record['parent_graph_sha256']
              and c.sha(parent/'artifact_lock.json') == record['parent_lock_sha256'], 'Changed test parent graph')
    c.publication.check_lock(parent)
    return record


def _build_one(root, key, entry, plan, builder, parser):
    c.require(builder.identity == parser, 'Parser differs from development')
    source_entry = {**_content(entry), 'processing_identity': c.digest(plan['parent_processing'])}
    parent_key = c.digest(source_entry)
    # The publisher locks its parent directory. Isolate that lock per object
    # so independent graph workers cannot contend on one global parents lock.
    parent = Path(root)/'parents'/parent_key/'object'
    if parent.exists():
        c.publication.check_lock(parent)
        origin = c.read(parent/'identity.json')
        c.require(origin['entry'] == source_entry and origin['parser'] == parser
                  and origin['split'] == 'test', 'Changed original test graph identity')
        original, boundary_repair = c.read(parent/'graph.json.gz'), origin['boundary_repair']
    else:
        original, boundary_repair = builder.build(parent_key, source_entry)
        with c.publication.transaction(parent) as stage:
            graph_common.create(stage/'graph.json.gz', original)
            graph_common.create(stage/'identity.json', {'entry': source_entry, 'split': 'test',
                                'parser': parser, 'boundary_repair': boundary_repair})
    parent_sha = c.sha(parent/'graph.json.gz')
    fixed = compact_cache.transform(original, _content(entry), source_entry, parent_key, parent_sha)
    fixed['doc_id'] = 'CTPG3-'+key
    fixed['metadata']['evaluation_role'] = 'test'
    fixed['metadata']['annotation_repair']['scope'] = 'Locked held-out cache; unchanged development correction rules'
    graphs.validate(_content(entry), fixed)
    tensors = graphs.tensors(fixed)
    arrays, metadata = compact.pack(tensors, fixed)
    round_trip = compact.unpack(arrays, metadata)
    for name, expected in tensors.items():
        c.require(torch.equal(expected, round_trip[name]) if isinstance(expected, torch.Tensor)
                  else expected == round_trip[name], 'Compact structural round trip differs: '+name)
    record = {'entry_sha256': key, 'split': 'test', 'processing_sha256': c.digest(plan['processing']),
              'release_sha256': plan['release_sha256'], 'parser': parser,
              'statistics': graphs.statistics(fixed), 'unique_pooling_rows': len(arrays['pool_ptr'])-1,
              'boundary_repair': boundary_repair, 'parent_key': parent_key, 'parent_graph_sha256': parent_sha,
              'parent_lock_sha256': c.sha(parent/'artifact_lock.json')}
    return compact_cache.publish(root, key, fixed, arrays, metadata, record)


_worker_builder = None


def _initialize_graph_worker():
    global _worker_builder
    from tpg_combined_recovery_v1.boundary import Builder
    torch.set_num_threads(1)
    _worker_builder = Builder()


def _graph_job(job):
    root, key, entry, metadata, parser = job
    return _build_one(root, key, entry, metadata, _worker_builder, parser)


def _build_graphs(protocol, plan, workers):
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    import multiprocessing
    from tpg_combined_parallel_v1.runner import thread_environment, stop_executor, scheduled_keys
    root = _root()
    c.require(type(workers) is int and 1 <= workers <= 32, 'Use 1..32 graph workers')
    c.require(c.available_memory() > (6*workers+2)*2**30, 'Insufficient RAM for graph workers')
    c.require(shutil.disk_usage(root).free > 20*2**30, 'Preserve the 20 GiB disk reserve')
    parser = c.read(root/'inputs/parser_identity.json')
    pending = {k: e for k, e in plan['entries'].items() if not _object_record(root, k, plan)}
    # Bound queued payloads; never serialize the complete test plan per graph.
    metadata = {k: plan[k] for k in ('processing', 'parent_processing', 'release_sha256')}
    with thread_environment():
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                       initializer=_initialize_graph_worker)
        try:
            keys, futures = iter(scheduled_keys(pending)), {}
            with tqdm(total=len(pending), desc='Build locked test D/DM v7 corrected graphs', unit='graph') as bar:
                while True:
                    while len(futures) < 2*workers:
                        key = next(keys, None)
                        if key is None:
                            break
                        job = (str(root), key, pending[key], metadata, parser)
                        futures[executor.submit(_graph_job, job)] = key
                    if not futures:
                        break
                    done, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
                    if not done:
                        tqdm.write(f'[TEST CACHE] {bar.n}/{len(pending)} complete; {len(futures)} pending')
                    for future in done:
                        key = futures.pop(future)
                        c.require(future.result()['entry_sha256'] == key, 'Wrong graph worker result')
                        bar.update()
                    c.require(shutil.disk_usage(root).free > 20*2**30, 'Disk reserve reached')
        finally:
            stop_executor(executor)


def _embed_graphs(protocol, plan):
    from tpg_combined_v3.embeddings import Encoder
    root = _root()
    identity = protocol['cache_encoder']
    encoder = Encoder(identity['device'], identity['threads'])
    c.require(encoder.identity == identity, 'SecureBERT/runtime identity differs from development')
    ip = root/'encoder_identity.json'
    if ip.exists():
        c.require(c.read(ip) == identity, 'Mixed test encoder identities')
    else:
        graph_common.create(ip, identity)
    for key, entry in tqdm(sorted(plan['entries'].items()), desc='Embed locked test graph nodes', unit='graph'):
        obj = root/'objects'/key
        record = _object_record(root, key, plan)
        c.require(record is not None, 'Finish all test graphs before embeddings')
        if compact_cache.embedding_receipt(obj, record, identity):
            continue
        required = record['unique_pooling_rows']*768*4
        c.require(shutil.disk_usage(root).free > required*1.1+20*2**30, 'Insufficient disk reserve')
        graph = c.read(obj/'graph.json.gz')
        layout, _ = compact.load(obj/'structure.npz')
        rows, observed, coverage = encoder.encode_compact(entry['text'], graph, layout)
        staging = root/'.embedding_staging'/(key+'-'+uuid4().hex)
        staging.mkdir(parents=True)
        compact.save(staging/'embeddings.npz', {'rows': rows, 'observed': observed},
                     {'format': compact.FORMAT, 'encoder_sha256': c.digest(identity)})
        receipt = {'graph_sha256': record['files']['graph.json.gz'],
                   'structure_sha256': record['files']['structure.npz'], 'encoder_sha256': c.digest(identity),
                   'coverage': coverage, 'files': {'embeddings.npz': c.sha(staging/'embeddings.npz')}}
        graph_common.create(staging/'embedding_receipt.json', receipt)
        if (obj/'embeddings.npz').exists():
            preserved = root/'.embedding_staging'/('unreceipted-'+key+'-'+uuid4().hex)
            preserved.mkdir()
            (obj/'embeddings.npz').rename(preserved/'embeddings.npz')
        for name in ('embeddings.npz', 'embedding_receipt.json'):
            (staging/name).rename(obj/name)
        staging.rmdir()


def _context(protocol, plan, *, complete=False, require_graphs=False):
    root = _root()
    contract = protocol['contract_template']
    vectors = c.read(root/'inputs/structured_vectors.json')
    c.require(set(vectors) == set(plan['index']) and all(len(v) == len(contract['structured_names'])
              and np.isfinite(v).all() for v in vectors.values()), 'Invalid test structured vectors')
    receipts, records, means = {}, {}, {}
    for key, entry in tqdm(sorted(plan['entries'].items()), desc='Verify test cache identities', unit='graph'):
        record = _object_record(root, key, plan)
        if record is None:
            c.require(not (complete or require_graphs), 'Test graph cache incomplete')
            continue
        obj = root/'objects'/key
        er = compact_cache.embedding_receipt(obj, record, protocol['cache_encoder'])
        if er is None:
            c.require(not complete, 'Test embedding cache incomplete')
            continue
        graph = c.read(obj/'graph.json.gz')
        graphs.validate(_content(entry), graph)
        c.require(graph['doc_id'] == 'CTPG3-'+key and graph['metadata'].get('evaluation_role') == 'test'
                  and graph['metadata']['combined_text']['processing_identity'] == entry['processing_identity'],
                  'Incorrect graph role/processing/identity')
        tensors = compact_cache.load_tensors(root, key, True)
        c.base.validate_tensors(tensors, contract)
        expected = graphs.tensors(graph)
        for name, value in expected.items():
            c.require(torch.equal(value, tensors[name]) if isinstance(value, torch.Tensor)
                      else value == tensors[name], 'Graph/tensor mismatch: '+name)
        layout, _ = compact.load(obj/'structure.npz')
        wanted = compact.pooling_layout(graph)
        c.require(all(np.array_equal(layout[k], v) for k, v in wanted.items()), 'Wrong embedding-to-node mapping')
        c.require(record['statistics'] == graphs.statistics(graph)
                  and er['coverage']['expanded_nodes'] == len(graph['vertices'])
                  and er['coverage']['stored_rows'] == record['unique_pooling_rows'] == len(layout['pool_ptr'])-1,
                  'Test graph/embedding coverage counts differ')
        missing = tensors['node_ids'][tensors['token_mask'] & ~tensors['embedding_mask']].tolist()
        c.require(set(missing) == {r['node_id'] for r in er['coverage']['exceptions']}, 'Unexplained token coverage gap')
        # Use the same reducer and fixed thread count as the training adapter.
        from ..training_only_pooled_v1.data import vector_job
        row = vector_job((str(root), key, [s['source'] for s in entry['spans']],
                          c.digest(protocol['cache_encoder']), None))
        means[key] = np.asarray(row['vector'], dtype=np.float32)
        receipts[key] = [c.sha(obj/'graph_receipt.json'), c.sha(obj/'embedding_receipt.json')]
        records[key] = record
    ids = sorted(plan['index'])
    pooled = {view: {cid: means.get(plan['index'][cid]['views'][view]['object_key']) for cid in ids} for view in VIEWS}
    if complete:
        for view in VIEWS:
            for cid in ids:
                if plan['index'][cid]['views'][view]['object_key'] is None:
                    pooled[view][cid] = np.zeros(contract['embedding_dim'], dtype=np.float32)
                else:
                    c.require(pooled[view][cid] is not None, 'Incomplete test pooled vector')
    metadata = {}
    for cid in ids:
        target = (date.fromisoformat(plan['publication_dates'][cid])+timedelta(days=60)).isoformat()
        versions = [(start, name) for name, start in plan['target_version_schedule'].items() if start <= target]
        c.require(versions, 'Test target predates the EPSS schedule')
        metadata[cid] = {'group': plan['groups'][cid], 'has_posts': plan['index'][cid]['views']['DM']['source_mask']['M'],
                         'target_date': target, 'target_version': max(versions)[1]}
    return {'directory': root, 'index': plan['index'], 'receipts': receipts, 'records': records,
            'contract': contract, 'vectors': vectors, 'pooledD': pooled['D'], 'pooledDM': pooled['DM'],
            'pooled': pooled, 'groups': plan['groups'], 'publication_dates': plan['publication_dates'],
            'ids': {'train': [], 'validation': ids, 'assessment': []},
            'metadata': metadata, 'is_test': True, 'evaluation_role': 'test',
            'release_sha256': plan['release_sha256'], 'processing_sha256': c.digest(plan['processing']),
            'labels_included': False, 'cache_complete': complete}


def load_graph(ctx, key):
    """Existing model-ready tensor layout, with an explicit test-only receipt check."""
    c.require(ctx.get('is_test') is True and ctx.get('evaluation_role') == 'test'
              and ctx.get('cache_complete') is True and key in ctx['receipts'], 'Unverified test graph context')
    root = Path(ctx['directory'])
    obj = root/'objects'/key
    c.require([c.sha(obj/name) for name in ('graph_receipt.json', 'embedding_receipt.json')] == ctx['receipts'][key],
              'Test cache receipt changed during evaluation')
    record = c.read(obj/'graph_receipt.json')
    c.require(record['split'] == 'test' and record['release_sha256'] == ctx['release_sha256']
              and record['processing_sha256'] == ctx['processing_sha256'], 'Wrong test graph release/processing')
    tensors = compact_cache.load_tensors(root, key, True)
    c.base.validate_tensors(tensors, ctx['contract'])
    from tpg_combined_v2.embeddings import model_inputs
    graph = {k: tensors[k] for k in ('edge_index', 'edge_type', 'edge_features', 'node_type',
                                    'subtype', 'token_mask', 'embedding_mask')}
    graph['x'] = model_inputs(tensors)['x']
    c.require(graph['x'].shape[1] == ctx['contract']['node_feature_dim'], 'Wrong test node feature dimensions')
    return graph


def prepare(protocol, release_lock, stage='inputs', *, workers=4, verify_only=False, resume=False):
    """Gated stages; no legacy development loader is told a test CVE is training.

    The returned context's ``validation`` slot is the shared prediction API's
    iteration slot only; its index entries and evaluation_role stay ``test``.
    Encoder device/thread settings are inherited exactly, not CLI overrides.
    """
    c.require(stage in ('inputs', 'graphs', 'embeddings', 'verify'), 'Unknown test cache stage')
    release = verify_release(protocol, release_lock)
    c.require(not verify_only or (_root()/'inputs/artifact_lock.json').exists(), 'No test cache to verify')
    with c.exclusive(c.OUT/'test_cache.writer.lock'):
        if stage == 'inputs':
            _prepare_inputs(protocol, release, verify_only)
        plan = _load_plan(protocol, release)
        if stage == 'graphs' and not verify_only:
            _build_graphs(protocol, plan, workers)
        if stage == 'embeddings' and not verify_only:
            _embed_graphs(protocol, plan)
        result = _context(protocol, plan, complete=stage in ('embeddings', 'verify'), require_graphs=stage == 'graphs')
        verify_release(protocol, release)
        if result['cache_complete']:
            receipt = {'schema': SCHEMA, 'release_sha256': c.digest(release),
                       'protocol_sha256': c.digest(protocol),
                       'input_lock_sha256': c.sha(_root()/'inputs/artifact_lock.json'),
                       'receipts': result['receipts'], 'test_ids': result['ids']['validation'],
                       'labels_included': False}
            path = _root()/'cache_verification.json'
            if path.exists():
                c.require(c.read(path) == receipt, 'Verified test cache changed')
            else:
                c.require(not verify_only, 'No published test cache verification to reproduce')
                c.write(path, receipt)
        return result


@contextmanager
def evaluation_release(protocol, release_lock, *, release_test=False, stage='evaluate', verify_only=False):
    """One scientific release, allowing only identical replay/crash recovery."""
    c.require(stage == 'evaluate' and release_test is True, 'Test labels require evaluate --release-test')
    release = verify_release(protocol, release_lock)
    root = _root()
    verification = c.read(root/'cache_verification.json')
    c.require(verification['release_sha256'] == c.digest(release)
              and verification['protocol_sha256'] == c.digest(protocol)
              and verification['test_ids'] == sorted(protocol['official_ids']['test'])
              and verification['labels_included'] is False, 'Test cache has not passed verification')
    c.require(verification['input_lock_sha256'] == c.sha(root/'inputs/artifact_lock.json'), 'Changed test input lock')
    plan = _load_plan(protocol, release)
    c.require(set(verification['receipts']) == set(plan['entries']), 'Incomplete verified test receipt inventory')
    for key, expected in verification['receipts'].items():
        obj = root/'objects'/key
        c.require([c.sha(obj/name) for name in ('graph_receipt.json', 'embedding_receipt.json')] == expected,
                  'Test cache changed since verification')
        record = _object_record(root, key, plan)
        c.require(compact_cache.embedding_receipt(obj, record, protocol['cache_encoder']), 'Missing verified test embeddings')
    marker = {'schema': RELEASE_SCHEMA, 'protocol_sha256': c.digest(protocol),
              'release_sha256': c.digest(release), 'cache_verification_sha256': c.sha(root/'cache_verification.json')}
    with c.exclusive(c.OUT/'test_evaluation.writer.lock'):
        path = c.OUT/'test_evaluation_release.json'
        if path.exists():
            c.require(c.read(path) == marker, 'A different scientific test evaluation was already released')
        else:
            c.require(not verify_only, 'No existing scientific test release to verify')
            c.write(path, marker)
        token = _evaluation.set((c.digest(protocol), release))
        try:
            yield marker
        finally:
            _evaluation.reset(token)


def load_test_labels(protocol):
    """Separate outcome read, usable only within the explicit release context."""
    authorization = _evaluation.get()
    c.require(authorization is not None and authorization[0] == c.digest(protocol),
              'Test labels are closed; use evaluate --release-test')
    release = verify_release(protocol, authorization[1])
    plan = _load_plan(protocol, release)
    manifest = _manifest(protocol)
    lo, hi = manifest['publication_intervals']['test']
    labels = {}
    for raw in tqdm(_jsonl(_pinned(protocol, _paths(protocol)['outcomes'])), desc='Load separately released Day-60 targets', unit='CVE'):
        cid = raw['cve']
        if cid not in plan['index']:
            continue
        c.require(cid not in labels, 'Duplicate test label')
        published = plan['publication_dates'][cid]
        target = (date.fromisoformat(published)+timedelta(days=60)).isoformat()
        c.require(lo <= published <= target <= min(hi, manifest['cutoff']), 'Test target outside frozen boundaries')
        values = raw['epss_outcomes']
        value = values['epss_scores'].get(target)
        c.require(values['epss_score_status'].get(target) == 'OBSERVED' and type(value) in (int, float)
                  and np.isfinite(value) and 0 <= value <= 1, 'Missing/invalid test target; never impute or drop it')
        labels[cid] = {'value': float(value), 'date': target}
    c.require(set(labels) == set(protocol['official_ids']['test']), 'Incomplete test labels')
    return labels
