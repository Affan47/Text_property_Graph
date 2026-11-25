"""VK10 explicitly released evaluation data; original development guards stay closed."""
from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path
import re

from tqdm import tqdm
from . import adapter_v2 as data, policy_v2 as policy
from .multisource_data_v1 import SOURCES
from ..consolidated.contracts import StructuredEncoder
from .. import forecast_identity_graphs as chunks

FREEZE = data.ROOT/'05_results/05_vulncheck_study/18_final_protocol_v1'
DATA = data.ROOT/'04_data/01_records_and_graphs/vulncheck_final_evaluation_v1'
CACHE = data.ROOT/'01_tpg/05_workspace/vulncheck_final_evaluation_v1'
RESULTS = data.ROOT/'05_results/05_vulncheck_study/19_final_evaluation_v1'
DEVELOPMENT = data.ROOT/'04_data/01_records_and_graphs/vulncheck_multisource_v1/full'
BASE_DATA = data.ROOT/'04_data/01_records_and_graphs/vulncheck_snapshot_v2/full'
DEVELOPMENT_CACHE = data.ROOT/'01_tpg/05_workspace/vulncheck_multisource_v1/full'
SCHEMA = 'vk10_final_evaluation_v1'


def access(scale, release):
    data.require(scale in ('smoke', 'full'), 'Unknown evaluation scale')
    data.require(scale != 'full' or release, 'Full evaluation requires explicit --release-test')


def frozen(scale, release=False):
    access(scale, release)
    data.check_lock(FREEZE)
    identity = data.read(FREEZE/'run_identity.json')
    data.check_pins(identity['pins'])
    protocol = data.read(FREEZE/'final_protocol.json')
    roles = data.read(FREEZE/'cohort_ids.json')
    data.require(protocol['schema'] == 'vk09_final_protocol_v1' and
                 protocol['primary_family'] == 'DGNM__security_tpg', 'Wrong frozen protocol')
    data.require(len(protocol['checkpoint_inventory']) == 36, 'Incomplete model inventory')
    data.require(data.digest(roles['test']) == protocol['evaluation_ids_sha256'], 'Test IDs changed')
    for role, count in protocol['configuration']['cohort']['counts'].items():
        data.require(len(roles[role]) == len(set(roles[role])) == count, 'Role count mismatch')
    data.require(len(set(c for ids in roles.values() for c in ids)) == 5968, 'Roles overlap')
    return protocol, roles


def pins(protocol):
    paths = [FREEZE/'artifact_lock.json', DEVELOPMENT/'artifact_lock.json',
             BASE_DATA/'artifact_lock.json', DEVELOPMENT_CACHE/'artifact_lock.json',
             data.SCRIPTS/'17_run_final_evaluation.py', data.SCRIPTS/'test_vulncheck_final_evaluation_v1.py']
    paths += sorted(Path(__file__).parent.glob('final_evaluation*_v1.py'))
    return {str(p): data.sha(p) for p in paths}


def validate(row, role, dimension=29):
    data.require(set(row) == {'cve', 'split', 'documents', 'source_mask', 'structured'}, 'Unexpected predictor fields')
    data.require(role in ('test', 'validation') and row['split'] == role, 'Evaluation role mismatch')
    data.require(re.fullmatch(r'CVE-\d{4}-\d{4,}', row['cve']), 'Unsafe CVE identifier')
    data.require(len(row['structured']) == dimension and all(type(v) in (int, float) and
                 math.isfinite(v) for v in row['structured']), 'Invalid structured vector')
    data.require(len(row['source_mask']) == 4 and all(type(v) is int and v in (0, 1)
                 for v in row['source_mask']), 'Invalid source mask')
    seen = set()
    for doc in row['documents']:
        data.require(set(doc) == {'document_id', 'source_text_type', 'text'}, 'Unexpected document fields')
        data.require(doc['source_text_type'] in SOURCES and isinstance(doc['text'], str) and doc['text'].strip(), 'Invalid document')
        data.require(doc['document_id'] not in seen, 'Duplicate document identity')
        seen.add(doc['document_id'])
        data.require(policy.inspect(doc['text'], row['cve'], doc['source_text_type'])['decision'] == 'accept', 'Quarantined text in predictors')
    data.require(sum(d['source_text_type'] == 'description' for d in row['documents']) == 1, 'Need exactly one accepted description')
    data.require(row['source_mask'] == [int(any(d['source_text_type'] == s for d in row['documents'])) for s in SOURCES], 'Mask mismatch')


def convert(raw, role, state, config):
    """Apply the unchanged source policy and saved transform; never fit preprocessing."""
    cve = raw['cve']
    data.require(type(raw.get('in_vckev')) is bool, 'Missing/nonboolean label: '+cve)
    mapping = {}
    for source in SOURCES:
        value = raw.get(config['source_fields'][source])
        items = [(p.get('post'), p.get('platform')) for p in value or []] if source == 'smp' else [(value, None)]
        for text, platform in items:
            if isinstance(text, str) and text.strip():
                key = source+':'+data.digest({'source': source, 'text': text, 'platform': platform})
                mapping[key] = source+':'+hashlib.sha256(((platform or '')+'\0'+text).encode()).hexdigest()
    docs, decisions = [], []
    for doc in data.documents(raw, config):
        decision = policy.inspect(doc['text'], cve, doc['source_text_type'])
        decisions.append({'cve': cve, 'split': role, 'document_id': mapping[doc['document_id']],
                          'source_text_type': doc['source_text_type'], **decision})
        if decision['decision'] == 'accept':
            docs.append({**doc, 'document_id': mapping[doc['document_id']]})
    vector, flags = StructuredEncoder.transform({k: raw.get(k) for k in config['structured_fields']},
                                               state, ['cvss', 'reference_count'])
    row = {'cve': cve, 'split': role, 'documents': docs, 'structured': vector,
           'source_mask': [int(any(d['source_text_type'] == s for d in docs)) for s in SOURCES]}
    validate(row, role)
    return row, {'cve': cve, 'split': role, 'target': int(raw['in_vckev'])}, decisions, flags


def overlap(rows, development):
    registry = defaultdict(lambda: defaultdict(set))
    def key(doc):
        return doc['source_text_type'], data.digest(' '.join(doc['text'].casefold().split()))
    for row in development:
        for doc in row['documents']:
            registry[key(doc)][row['split']].add(row['cve'])
    matches = []
    for row in rows:
        for doc in row['documents']:
            hits = {role: sorted(ids-{row['cve']}) for role, ids in registry[key(doc)].items()}
            hits = {role: ids for role, ids in hits.items() if ids}
            if hits:
                matches.append({'cve': row['cve'], 'document_id': doc['document_id'],
                                'source': doc['source_text_type'], 'matching_cves_by_role': hits})
    return {'normalization': 'Within-source Unicode casefold and whitespace collapse; not CVE duplication.',
            'matches': matches, 'shared_with_training': sorted({m['cve'] for m in matches if 'train' in m['matching_cves_by_role']}),
            'shared_with_development': sorted({m['cve'] for m in matches})}


def prepare(scale, release=False, verify=False):
    protocol, roles = frozen(scale, release)
    data.check_lock(DEVELOPMENT); data.check_lock(BASE_DATA)
    config = data.config()
    data.require(config['source_fields'] == protocol['source_fields'] and policy.VERSION == protocol['source_policy'], 'Source policy changed')
    state = data.read(BASE_DATA/'structured_encoder.json')
    data.require(sorted(state['fit_ids']) == roles['train'] and state['feature_names'] == protocol['structured_feature_names'], 'Preprocessor changed')
    development = data.read(DEVELOPMENT/'inputs.json')['rows']
    role = 'test' if scale == 'full' else 'validation'
    # First 64 validation IDs preserve the original eight-CVE batch boundaries.
    ids = roles['test'] if scale == 'full' else roles['validation'][:64]
    source = data.ROOT.parent/protocol['source_dataset']
    data.require(data.sha(source) == protocol['source_dataset_sha256'], 'Raw snapshot changed')
    records = data.read(source); raw = {r['cve']: r for r in records}
    data.require(len(raw) == len(records) == 5968 and set(ids) <= set(raw), 'Missing/duplicate raw CVE')
    rows, labels, decisions, flags = [], [], [], []
    for cve in tqdm(ids, desc='Apply frozen evaluation source policy', unit='CVE'):
        try:
            row, label, audit, quality = convert(raw[cve], role, state, config)
        except Exception as exc:
            raise ValueError(f'Evaluation aborted; failed CVE {cve}: {exc}') from exc
        rows.append(row); labels.append(label); decisions.extend(audit)
        flags.append({'cve': cve, 'flags': quality})
    data.require([r['cve'] for r in rows] == ids, 'Cohort changed')
    if scale == 'smoke':
        expected = {r['cve']: r for r in development}
        data.require(all(row == expected[row['cve']] for row in rows), 'New adapter differs from development')
    coverage = []
    for target in (0, 1):
        selected = {r['cve'] for r in labels if r['target'] == target}
        for j, name in enumerate(SOURCES):
            before = {r['cve'] for r in decisions if r['cve'] in selected and r['source_text_type'] == name}
            after = {r['cve'] for r in rows if r['cve'] in selected and r['source_mask'][j]}
            coverage.append({'listed': target, 'source': name, 'cves_in_class': len(selected),
                             'with_source_before': len(before), 'with_source_after': len(after),
                             'losing_entire_source': len(before-after)})
    dependencies = pins(protocol)
    identity = {'schema': SCHEMA, 'stage': 'data', 'scale': scale, 'role': role, 'ids': ids,
                'protocol_sha256': data.sha(FREEZE/'final_protocol.json'), 'pins': dependencies,
                'test_released': scale == 'full', 'training': False, 'preprocessor_refitted': False}
    summary = {'cves': len(rows), 'listed': sum(r['target'] for r in labels), 'role': role,
               'source_coverage_by_class': coverage, 'documents': len(decisions),
               'quarantined_documents': sum(d['decision'] == 'quarantine' for d in decisions),
               'interpretation': 'Reused-cohort retrospective evaluation' if scale == 'full' else 'Validation integration check'}
    data.publish(DATA/scale, {'inputs.json': {'source_order': list(SOURCES), 'rows': rows}, 'labels.json': labels,
                 'decisions.json': decisions, 'structured_audit.json': flags, 'structured_encoder.json': state,
                 'overlap_audit.json': overlap(rows, development), 'summary.json': summary,
                 'summary.txt': data.dump(summary), 'run_identity.json': identity}, dependencies, verify)
    print(f'[VK10 DATA] {len(rows)} {role} CVEs; {summary["listed"]} listed; no fitting or scoring.', flush=True)


def load(scale, release=False):
    protocol, roles = frozen(scale, release)
    directory = DATA/scale; data.check_lock(directory)
    identity = data.read(directory/'run_identity.json'); data.check_pins(identity['pins'])
    data.require(identity['pins'] == pins(protocol), 'Evaluator code changed after data preparation')
    expected = roles['test'] if scale == 'full' else roles['validation'][:64]
    role = 'test' if scale == 'full' else 'validation'
    rows = data.read(directory/'inputs.json')['rows']
    labels = data.read(directory/'labels.json')
    data.require(identity['ids'] == expected == [r['cve'] for r in rows] == [r['cve'] for r in labels], 'Cohort/input/label join differs')
    for row in rows:
        validate(row, role)
    data.require(all(r['split'] == role and type(r['target']) is int and r['target'] in (0, 1) for r in labels), 'Invalid labels')
    return protocol, identity, rows, labels


def graph_plan(rows, role, limit):
    """Exact original chunk keys, with explicit evaluation role and source offsets."""
    entries, documents, index = {}, {}, {}
    digest = chunks.old.shared.base.text_digest
    for row in rows:
        validate(row, role); cve = row['cve']
        data.require(cve not in index, 'Repeated CVE')
        index[cve] = {'cve': cve, 'split': role, 'source_mask': row['source_mask'], 'documents': []}
        for doc in row['documents']:
            stem = f"{cve}/{doc['source_text_type']}/{digest(doc['document_id'])}"
            parent_hash = digest(doc['text']); parts = chunks.segments(doc['text'], limit)
            data.require(any(p['has_graph'] for p in parts), 'Unprocessable document: '+cve)
            path = 'documents/'+stem+'.json'
            document = {'cve': cve, 'split': role, **doc, 'text_sha256': parent_hash, 'chunks': []}
            for segment in parts:
                part = dict(segment)
                if segment['has_graph']:
                    text = doc['text'][segment['start']:segment['end']]
                    key = digest(cve+'\0'+doc['document_id']+'\0'+str(segment['index']))
                    data.require(key not in entries, 'Chunk collision')
                    entry = {'key': key, 'graph_id': 'CD03-'+key, 'cve': cve, 'split': role,
                             'document_id': doc['document_id'], 'source_text_type': doc['source_text_type'],
                             'text': text, 'text_sha256': digest(text), 'parent_text_sha256': parent_hash,
                             'chunk': dict(segment), 'stem': stem+f"/chunk_{segment['index']:05d}"}
                    entries[key] = entry
                    part.update(key=key, graph='tpg_json/no_coref/'+entry['stem']+'.json', tensor='tensors/'+entry['stem']+'.pt')
                document['chunks'].append(part)
            documents[path] = document
            index[cve]['documents'].append({'document_id': doc['document_id'], 'source_text_type': doc['source_text_type'],
                'source_file': path, 'characters': len(doc['text']), 'chunks': document['chunks']})
    return {'entries': entries, 'documents': documents, 'index': index}
