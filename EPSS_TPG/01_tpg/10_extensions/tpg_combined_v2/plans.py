"""Version an already verified development text plan, never reselect CVEs."""
from copy import deepcopy
from tqdm import tqdm
from . import common as c


def prepare(dataset, scale, resume=False):
    old_dir, destination = c.directory(dataset, scale, 1), c.directory(dataset, scale)
    lock = c.read(old_dir / 'plan_lock.json')
    c.check_files(old_dir, lock['files'])
    old = c.read(old_dir / 'plan.json.gz')
    c.require(not old['test_adapted'] and all(e['split'] in ('train', 'validation') for e in old['entries'].values()), 'Test entries forbidden')
    # This is an explicit version transition, not a bypass for loading old graphs.
    frozen = c.read(c.ROOT / '06_runtime/03_tpg_repair/reference_code_v1/snapshot_lock.json')
    c.require(old['processing']['pins'] == frozen['pins'], 'Unrecognized legacy plan')
    for path, expected in old['input_hashes'].items():
        c.require(c.sha(path) == expected, f'Changed original input: {path}')
    processing = {**old['processing'], 'schema': c.SCHEMA, 'pins': c.pins(), 'graph_version': c.GRAPH_VERSION,
                  'graph_policy': 'v7 document-bounded extraction; one combined graph; marked concatenation links; typed SEC only; no COREF',
                  'embedding_policy': 'union of member-token subwords for phrases, exact spans otherwise',
                  'target_cve_policy': 'explicit binary feature, retain other mentions; no invented CVE nodes',
                  'edge_feature_dim': 88}
    remap, entries = {}, {}
    for key, entry in tqdm(old['entries'].items(), desc=f'Version {dataset} {scale} text plan', unit='graph'):
        current = {**entry, 'processing_identity': c.digest(processing)}
        new_key = c.digest(current)
        remap[key] = new_key
        entries[new_key] = current
    index = deepcopy(old['index'])
    for row in index.values():
        for view in row['views'].values():
            if view['object_key']:
                view['object_key'] = remap[view['object_key']]
    plan = {**old, 'schema': c.SCHEMA, 'dataset': dataset, 'processing': processing, 'entries': entries, 'index': index,
            'input_hashes': {**old['input_hashes'], str(old_dir / 'plan.json.gz'): c.sha(old_dir / 'plan.json.gz'),
                            str(old_dir / 'plan_lock.json'): c.sha(old_dir / 'plan_lock.json')},
            'version_transition': {'parent': str(old_dir), 'old_graph_payloads_used': False, 'cohort_and_text_unchanged': True}}
    if (destination / 'plan_lock.json').exists():
        c.require(resume, 'Plan exists; use --resume for verification')
        c.require(c.load_plan(destination) == plan, 'Versioned plan differs')
    else:
        c.require(not destination.exists() or not any(destination.iterdir()), 'Nonempty output; use a new version')
        payloads = {'plan.json.gz': plan, 'cve_index.json': index, 'coverage.json': old['coverage']}
        for number, variant in enumerate(plan['variants'], 1):
            payloads[f'variants/{number:02d}_{variant}/index.json'] = {cid: {'split': row['split'], **row['views'][variant]} for cid, row in index.items()}
        for name, value in payloads.items():
            c.create(destination / name, value)
        c.create(destination / 'plan_lock.json', {'files': {name: c.sha(destination / name) for name in payloads}})
    print(f'v2 plan: {len(index)} development CVEs, {len(entries)} graph objects; {destination}')

