"""VK04 description-only cache; immutable exact-identity reuse and frozen builders."""
from collections import Counter
import hashlib
from pathlib import Path
import shutil

from tqdm import tqdm
from . import adapter_v2 as data, development_v1 as development

CONFIG = data.ROOT / '02_prediction/04_configs/02_vulncheck/04_description_graph_v1.json'


def config():
    p = data.read(CONFIG)
    data.require(p['schema'] == 'vk04_description_graph_v1' and p['edge_view'] == 'no_coref', 'Wrong graph protocol')
    return p


def pins():
    files = [CONFIG, Path(__file__), Path(__file__).with_name('graph_model_v1.py'),
             Path(__file__).with_name('graph_training_v1.py'),
             data.SCRIPTS / '06_run_description_graph_v1.py', data.SCRIPTS / 'test_vulncheck_graph_v1.py']
    from .. import forecast_primary_cache_v2 as old
    from ..primary_training import batching
    from ..consolidated import models
    files += [Path(old.__file__), Path(old.coverage.__file__), Path(batching.__file__), Path(models.__file__),
              data.ROOT / '02_prediction/01_package/epss/edge_aware_layers.py']
    return {**development.code_pins(), **{str(f): data.sha(f) for f in files}}


def select(rows, labels, scale):
    data.require(scale in ('smoke', 'full'), 'Invalid scale')
    if scale == 'full':
        return {s: sorted(r['cve'] for r in rows if r['split'] == s) for s in ('train', 'validation')}
    targets = {r['cve']: r['target'] for r in labels}
    counts = config()['smoke']['counts']
    selected = {}
    for s in ('train', 'validation'):
        per_class = counts[s] // 2
        selected[s] = sorted(c for y in (0, 1) for c in
                             sorted(r['cve'] for r in rows if r['split'] == s and targets[r['cve']] == y)[:per_class])
        data.require(len(selected[s]) == counts[s], 'Insufficient smoke classes')
    return selected


def description_rows(rows, ids):
    from .classifier_smoke_v2 import check_rows
    check_rows(rows)
    selected = {c for values in ids.values() for c in values}
    result = []
    for row in rows:
        if row['cve'] not in selected:
            continue
        docs = [d for d in row['documents'] if d['source_text_type'] == 'description']
        data.require(len(docs) == 1 and docs[0]['text'].strip(), 'Need exactly one accepted description per CVE')
        text = docs[0]['text']
        # Match the old text/source identity convention, never approximate-match descriptions.
        canonical = 'description:' + hashlib.sha256(('\0' + text).encode()).hexdigest()
        result.append({'cve': row['cve'], 'split': row['split'], 'source_mask': [1, 0, 0, 0],
                       'documents': [{'document_id': canonical, 'source_text_type': 'description', 'text': text}]})
    return result


def context(scale):
    from .. import forecast_primary_cache_v2 as old
    p, directory, rows, labels, state, export_identity, source_pins = development.load()
    cfg = config()
    source = data.ROOT / cfg['reuse_root']
    lock = data.read(source / 'artifact_lock.json')
    for name in ('resources.json', 'run_identity.json', 'cve_index.json'):
        data.require(data.sha(source / name) == lock['files'][name], 'Changed reuse metadata: ' + name)
    resources = data.read(source / 'resources.json')
    old_identity = data.read(source / 'run_identity.json')
    data.check_pins(resources['source_pins'])
    data.check_pins(old_identity['source_pins'])
    data.require(old_identity['coverage_policy'] == old.coverage.POLICY, 'Embedding coverage policy differs')
    ids = select(rows, labels, scale)
    graph_rows = description_rows(rows, ids)
    plan = old.old.base.plan(graph_rows, ids, resources['protocol']['max_chunk_characters'])
    reference = data.ROOT / cfg['reference_root']
    data.check_lock(reference)
    baseline_identity = data.read(reference / 'run_identity.json')
    data.check_pins(baseline_identity['pins'])
    data.require(baseline_identity['ids'] == export_identity['ids'], 'Logistic reference cohort differs')
    identity = {'schema': cfg['schema'], 'scale': scale, 'config': cfg, 'ids': ids,
                'source_pins': {**source_pins, **pins(),
                    str(source / 'artifact_lock.json'): data.sha(source / 'artifact_lock.json'),
                    str(reference / 'artifact_lock.json'): data.sha(reference / 'artifact_lock.json')},
                'resources_sha256': data.digest(resources), 'plan_sha256': data.digest(plan),
                'test_graphs': False, 'training': False}
    return cfg, plan, identity, resources, source, lock


def reusable(source, lock, entry):
    from .. import forecast_primary_cache_v2 as old
    name = 'receipts/' + entry['key'] + '.json'
    if name not in lock['files']:
        return False
    data.require(data.sha(source / name) == lock['files'][name], 'Changed reuse receipt')
    receipt = data.read(source / name)
    if not old.old.reusable(receipt, entry):
        return False
    for relative, digest in receipt['files'].items():
        data.require(lock['files'].get(relative) == digest, 'Receipt not authenticated by old cache lock')
    return True


def security_stats(directory, entry, row):
    graph = data.read(directory / ('tpg_json/no_coref/' + entry['stem'] + '.json'))
    edges = [e for e in graph['edges'] if e['label'].startswith('SEC_')]
    for edge in edges:
        extra = edge['properties'].get('EXTRA', {})
        data.require(extra.get('status') == 'candidate' and extra.get('source_text_type') == 'description', 'Lost security provenance')
    return {'cve': entry['cve'], 'key': entry['key'], 'nodes': len(graph['vertices']), 'edges': len(graph['edges']),
            'security_edges': dict(Counter(e['label'] for e in edges)),
            'polarities': dict(Counter(e['properties']['EXTRA'].get('polarity', 'not_recorded') for e in edges)),
            'modalities': dict(Counter(e['properties']['EXTRA'].get('modality', 'not_recorded') for e in edges)),
            'methods': dict(Counter(e['properties']['EXTRA'].get('method', 'not_recorded') for e in edges)),
            'example_security_edges': edges[:2], 'coverage_statistics': row}


def prepare(scale, workers=2, resume=False, verify=False, check_only=False):
    from .. import forecast_primary_cache_v2 as old
    from ..forecast_matched_multisource import writer_lock
    old.torch.set_num_threads(1)
    cfg, plan, identity, resources, source, source_lock = context(scale)
    output = data.ROOT / cfg['cache_root'] / scale
    data.require(1 <= workers <= 16, 'Workers must be 1..16')
    eligible = {k for k, entry in tqdm(plan['entries'].items(), desc='Check exact description reuse identities', unit='chunk')
                if reusable(source, source_lock, entry)}
    print(f'[VK04 CACHE] CVEs={ {s:len(v) for s,v in identity["ids"].items()} }; '
          f'chunks={len(plan["entries"])}; exact reuse candidates={len(eligible)}; missing={len(plan["entries"])-len(eligible)}', flush=True)
    if check_only:
        print('VK04 cache preflight passed. Candidate counts only; tensor checks occur during preparation. No graphs or training.')
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(output.parent / '.writer.lock'):
        tokenizer = old.tokenizer_for(resources)
        if verify:
            data.check_lock(output)
            data.require(data.read(output / 'run_identity.json') == identity, 'Cache identity differs')
            data.require(data.read(output / 'cve_index.json') == plan['index'] and data.read(output / 'resources.json') == resources, 'Cache index/resources differ')
            for name, doc in plan['documents'].items():
                data.require(data.read(output / name) == doc, 'Description text/offsets changed')
        elif output.exists():
            data.require(resume and not (output / 'artifact_lock.json').exists(), 'Cache exists: verify completed cache or resume incomplete cache')
            data.require(data.read(output / 'run_identity.json') == identity, 'Resume identity differs')
        else:
            output.mkdir(parents=True)
            old.atomic_json(output / 'run_identity.json', identity)
            old.atomic_json(output / 'resources.json', resources)
        provenance_path = output / 'chunk_provenance.json'
        provenance = data.read(provenance_path) if provenance_path.exists() else {}
        if not verify:
            missing = []
            for key, entry in tqdm(plan['entries'].items(), desc='Validate or copy description chunks', unit='chunk'):
                receipt_name = 'receipts/' + key + '.json'
                if (output / receipt_name).exists():
                    old.check_entry(output, entry, resources, tokenizer)
                    provenance.setdefault(key, 'verified_resumed_chunk')
                elif key in eligible:
                    old.check_entry(source, entry, resources, tokenizer)
                    receipt = data.read(source / receipt_name)
                    for relative in [*receipt['files'], receipt_name]:
                        dest = output / relative
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source / relative, dest)
                    provenance[key] = 'exact_primary_v2_reuse'
                else:
                    missing.append(entry)
            old.atomic_json(provenance_path, provenance)
            if missing:
                old.bounded_build(output, missing, resources, workers, provenance)
            for name, doc in plan['documents'].items():
                old.atomic_json(output / name, doc)
            old.atomic_json(output / 'cve_index.json', plan['index'])
        audit = []
        for entry in tqdm(plan['entries'].values(), desc='Verify embeddings, edges and description spans', unit='chunk'):
            _, stats = old.check_entry(output, entry, resources, tokenizer)
            row = security_stats(output, entry, stats)
            data.require(row['nodes'] <= cfg['graph_node_budget'] and row['edges'] <= cfg['graph_edge_budget'], 'Chunk exceeds budget; refuse truncation')
            audit.append(row)
        summary = {'schema': cfg['schema'], 'scale': scale, 'cves': {s: len(v) for s, v in identity['ids'].items()},
                   'chunks': len(audit), 'provenance_counts': dict(Counter(provenance.values())),
                   'security_edges': dict(sum((Counter(r['security_edges']) for r in audit), Counter())),
                   'extractor': 'Frozen security v4, text-dependent rule-based candidate extraction; semantic accuracy not established.',
                   'text_changed': False, 'test_graphs': False, 'training': False}
        data.require(set(provenance) == set(plan['entries']), 'Incomplete provenance')
        if verify:
            data.require(data.read(output / 'security_audit.json') == audit and data.read(output / 'summary.json') == summary, 'Cache audit differs')
        else:
            old.atomic_json(output / 'security_audit.json', audit)
            old.atomic_json(output / 'summary.json', summary)
            (output / 'summary.txt').write_text('VK04 description cache\n' + data.dump(summary) +
                'TPGs: tpg_json/no_coref/. Shared frozen node embeddings: tensors/. Text and offsets: documents/.\n')
            old.common.lock(output)
        data.check_pins(identity['source_pins'])
    print(f'VK04 {scale} description cache ' + ('verified.' if verify else f'saved to {output}.') + ' No training or test graphs.')
