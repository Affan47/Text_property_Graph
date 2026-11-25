"""Atomic CC02 preparation; network only in the explicit EPSS collection stage."""
from collections import Counter
from pathlib import Path
import sys
import tempfile

import requests
from tqdm import tqdm

from ..catalogue_classification_v1 import snapshot as frozen
from . import core

ROOT = frozen.ROOT
CONFIG = ROOT / '02_prediction/04_configs/08_catalogue_classification/02_inputs_protocol.json'
CLI = ROOT / '02_prediction/03_analysis/08_catalogue_classification/03_prepare_classification_inputs.py'
TEST = CLI.with_name('test_classification_inputs.py')


def code_pins():
    paths = [CONFIG, CLI, TEST, frozen.CONFIG, Path(frozen.__file__),
             ROOT / '02_prediction/01_package/epss/training_fold_audit_v1/core.py',
             Path(core.policy_v2.__file__), *Path(__file__).parent.glob('*.py')]
    return {str(p): frozen.sha(p) for p in sorted(set(paths))}


def context():
    cfg = frozen.read(CONFIG)
    frozen.require(not cfg['training_enabled'] and not cfg['test_access_enabled'], 'Preparation only')
    frozen.require(cfg['role_basis_points'] == {'fitting': 7000, 'selection': 1500, 'assessment': 1500}, 'Role rule changed')
    source = ROOT / cfg['snapshot']
    frozen.check_lock(source / 'plan')
    plan = frozen.read(source / 'plan/plan.json')
    frozen.require(plan['source_pins'] == frozen.pins(), 'CC01 code changed')
    frozen.check_pins(plan['prerequisite_pins'])
    frozen.verify_snapshot(source, plan)
    receipt = frozen.read(source / 'snapshot/receipt.json')
    frozen.require(receipt['snapshot_at'][:10] == cfg['epss_date'], 'Catalogue/EPSS dates differ')
    provenance = code_pins()
    for p in (source / 'plan/plan.json', source / 'snapshot/artifact_lock.json',
              source / 'snapshot/receipt.json', ROOT / cfg['readiness_config']):
        provenance[str(p)] = frozen.sha(p)
    return cfg, plan, source, provenance


def publish(output, objects, texts, verify, resume):
    if verify:
        frozen.require(output.exists(), 'No saved stage to verify: ' + str(output))
    if output.exists():
        frozen.require(verify or resume, 'Existing stage; use --verify-only or --resume')
    with tempfile.TemporaryDirectory(prefix='.cc02-', dir=output.parent) as tmp:
        stage = Path(tmp) / 'ready'
        stage.mkdir()
        for name, value in objects.items():
            frozen.write(stage / name, value)
        for name, value in texts.items():
            (stage / name).write_text(value)
        frozen.lock(stage)
        if output.exists():
            frozen.check_lock(output)
            frozen.require(frozen.read(stage / 'artifact_lock.json') == frozen.read(output / 'artifact_lock.json'),
                           'CC02 reproduction mismatch; do not overwrite saved evidence')
        else:
            stage.rename(output)


def load_screened(cfg, plan, pins):
    readiness = frozen.read(ROOT / cfg['readiness_config'])
    datasets, audits = {}, {}
    for name, spec in tqdm(readiness['datasets'].items(), desc='Verify and screen training-only D/M inputs', unit='dataset'):
        path = ROOT / spec['raw']
        frozen.require(frozen.sha(path) == spec['raw_sha256'], 'Raw dataset identity changed')
        pins[str(path)] = spec['raw_sha256']
        allowed = set(plan['training_ids'][name])
        datasets[name], audits[name] = {}, {}
        # Only CVE identity is inspected on records outside the training allowlist.
        for record in frozen.read(path):
            cve = record.get('cve')
            if cve not in allowed:
                continue
            frozen.require(cve not in datasets[name], 'Duplicate raw training CVE')
            datasets[name][cve], audits[name][cve] = core.screen(record)
        frozen.require(set(datasets[name]) == allowed, 'Missing training CVEs; no silent exclusions')
    spec = readiness['datasets']['reddit']
    path = ROOT / spec['components']
    frozen.require(frozen.sha(path) == spec['components_sha256'], 'Reddit evidence identity changed')
    pins[str(path)] = spec['components_sha256']
    prior = {}
    allowed = set(plan['training_ids']['reddit'])
    for group in frozen.read(path):
        members = sorted(allowed.intersection(group['current_cves']))
        if members:
            frozen.require(not group['connected_to_prior_test'], 'Prior-test-linked group in scope')
            prior[group['component_id']] = members
    frozen.require(set(c for members in prior.values() for c in members) == allowed, 'Incomplete original evidence grouping')
    return datasets, audits, prior


def support(datasets, audits, roles, labels, cfg):
    rows = []
    for name in sorted(datasets):
        for role, ids in roles['datasets'][name].items():
            counts = Counter(labels[name][c]['status'] for c in ids)
            docs = [d for c in ids for d in audits[name][c]['documents']]
            rows.append({'dataset': name, 'role': role, 'cves': len(ids),
                         'listed': counts['LISTED'], 'not_listed': counts['NOT_LISTED'],
                         'unknown': counts['UNKNOWN'], 'groups': len({roles['by_cve'][c]['group'] for c in ids}),
                         'description_cves': sum(datasets[name][c]['mask_D'] for c in ids),
                         'post_cves': sum(datasets[name][c]['mask_M'] for c in ids),
                         'quarantined_documents': sum(d['decision'] == 'quarantine' for d in docs),
                         'bare_id_posts': sum(d['decision'] == 'bare_id' for d in docs),
                         'quarantine_reasons': dict(sorted(Counter(r for d in docs if d['decision'] == 'quarantine' for r in d['reasons']).items()))})
    sufficient = all(min(r['listed'], r['not_listed']) >= cfg['minimum_each_class_per_role'] and r['unknown'] == 0 for r in rows)
    return rows, sufficient


def freeze(cfg, plan, source, pins, output, verify, resume):
    datasets, audits, prior = load_screened(cfg, plan, pins)
    roles = core.freeze_roles(datasets, prior, cfg['role_seed'])
    # Outcomes are joined only after outcome-blind assignments have been made.
    labels = {}
    for name in sorted(datasets):
        path = source / 'snapshot' / (name + '_labels.json')
        labels[name] = {r['cve']: r for r in frozen.read(path)}
        frozen.require(set(labels[name]) == set(datasets[name]), 'Label/input scope mismatch')
        pins[str(path)] = frozen.sha(path)
    rows, sufficient = support(datasets, audits, roles, labels, cfg)
    largest = sorted((len(g['members']) for g in roles['components'].values()), reverse=True)[:10]
    findings = {'schema': cfg['schema'], 'stage': 'freeze', 'support': rows,
                'role_support_sufficient': sufficient, 'unique_cves': len(roles['by_cve']),
                'shared_training_cves': len(set(datasets['reddit']) & set(datasets['previous'])),
                'components': len(roles['components']), 'largest_component_sizes': largest,
                'roles_sha256': core.digest(frozen.dump(roles)), 'epss_date': cfg['epss_date'],
                'epss_verified_at_this_stage': False, 'cache_gate': 'EXACT_SCREENED_LINEAGE_CHECK_REQUIRED',
                'training_enabled': False, 'test_access_enabled': False, 'meaning': cfg['interpretation']}
    objects = {'protocol.json': cfg, 'identity.json': {'source_pins': pins}, 'roles.json': roles, 'findings.json': findings}
    for name in sorted(datasets):
        objects[name + '_screened.json'] = datasets[name]
        objects[name + '_source_audit.json'] = audits[name]
        objects[name + '_labels.json'] = labels[name]
    report = ['CC02: screened training-role inputs and frozen grouped inner roles', '', cfg['interpretation'],
              'Nominal role fractions 70/15/15 use a label-blind hash of each evidence component.',
              'Actual counts need not equal the fractions. No strata, outcome search or split redraw.',
              'Shared references do not identify the same vulnerability. Grouping controls evidence exposure.',
              '', 'Dataset | Role | CVEs | Listed | Not listed | D available | M prose available']
    report += [' | '.join(str(r[k]) for k in ('dataset', 'role', 'cves', 'listed', 'not_listed', 'description_cves', 'post_cves')) for r in rows]
    report += ['', 'Roles SHA-256: ' + findings['roles_sha256'],
               'EPSS is not yet verified by this stage; run collect-epss, then prepare.',
               'No old cache is approved automatically; exact screened-text and processing identity are required.',
               'Catalogue fields, audit decisions, listing dates and original EPSS metadata are not model inputs.',
               'General exploitation wording can remain; lexical screening does not prove absence of answer proxies.',
               'Descriptions/CVSS/posts are retrospective. Same UTC date does not establish within-day ordering.',
               'No training, official validation adaptation or test joins.']
    frozen.check_pins(pins)
    publish(output, objects, {'REPORT.txt': '\n'.join(report) + '\n'}, verify, resume)
    print(frozen.dump(findings))


def check_frozen(base, cfg):
    frozen.check_lock(base / 'freeze')
    identity = frozen.read(base / 'freeze/identity.json')['source_pins']
    frozen.check_pins(identity)
    frozen.require(frozen.read(base / 'freeze/protocol.json') == cfg, 'Changed CC02 protocol')


def epss_verified(base, cfg, plan):
    directory = base / 'epss'
    frozen.check_lock(directory)
    receipt = frozen.read(directory / 'receipt.json')
    archive = directory / 'epss_scores-2026-11-17.csv.gz'
    frozen.require(receipt['requested_url'] == cfg['epss_url'] and receipt['http_status'] == 200
                   and receipt['sha256'] == frozen.sha(archive), 'Changed EPSS receipt/source identity')
    ids = set().union(*map(set, plan['training_ids'].values()))
    scores, inventory = core.parse_epss(archive, cfg['epss_date'], ids, cfg['epss_minimum_rows'], cfg['uncompressed_max_bytes'])
    frozen.require(receipt['inventory'] == inventory and receipt['bytes'] == archive.stat().st_size, 'EPSS receipt does not reproduce')
    return scores, inventory


def collect(cfg, plan, base, allow_network, verify, resume):
    output = base / 'epss'
    if output.exists():
        frozen.require(verify or resume, 'EPSS already saved; use --verify-only or --resume')
        _, inventory = epss_verified(base, cfg, plan)
        print('Dated EPSS archive reproduced: ' + frozen.dump(inventory))
        return
    frozen.require(not verify, 'No EPSS archive to verify')
    frozen.require(allow_network, 'EPSS collection needs explicit --allow-network; no API key is needed')
    with tempfile.TemporaryDirectory(prefix='.cc02-download-', dir=base) as tmp:
        stage = Path(tmp) / 'ready'
        stage.mkdir()
        archive = stage / 'epss_scores-2026-11-17.csv.gz'
        started = frozen.utc()
        # One explicit dated source, no current-file fallback or credential reuse.
        with requests.get(cfg['epss_url'], stream=True, timeout=(15, 90), allow_redirects=False) as response:
            frozen.require(response.status_code == 200, 'EPSS HTTP ' + str(response.status_code) + '; no missing scores inferred')
            total = int(response.headers.get('Content-Length', 0))
            frozen.require(total <= cfg['download_max_bytes'], 'Download exceeds frozen budget')
            size = 0
            with archive.open('xb') as stream, tqdm(total=total or None, desc='Download dated EPSS archive', unit='B', unit_scale=True) as bar:
                for chunk in response.iter_content(1024 * 1024):
                    size += len(chunk)
                    frozen.require(size <= cfg['download_max_bytes'], 'Download exceeds frozen budget')
                    stream.write(chunk)
                    bar.update(len(chunk))
            frozen.require(not total or total == size, 'Incomplete HTTP body')
        ids = set().union(*map(set, plan['training_ids'].values()))
        _, inventory = core.parse_epss(archive, cfg['epss_date'], ids, cfg['epss_minimum_rows'], cfg['uncompressed_max_bytes'])
        frozen.write(stage / 'receipt.json', {'requested_url': cfg['epss_url'], 'http_status': 200,
                     'started_at': started, 'ended_at': frozen.utc(), 'bytes': size, 'sha256': frozen.sha(archive),
                     'inventory': inventory, 'official_source_documentation': 'https://www.first.org/epss/data',
                     'timing_limit': 'Same UTC day only, not verified prior to the catalogue backup time.'})
        frozen.lock(stage)
        stage.rename(output)
    epss_verified(base, cfg, plan)
    print('Dated EPSS archive saved; no training or test joins.')


def prepare(cfg, plan, base, pins, verify, resume):
    check_frozen(base, cfg)
    frozen.require((base / 'epss').exists(), 'Missing dated EPSS archive; run --stage collect-epss --allow-network')
    scores, inventory = epss_verified(base, cfg, plan)
    from ..reddit_primary import structured
    roles = frozen.read(base / 'freeze/roles.json')
    identity = frozen.read(base / 'freeze/identity.json')['source_pins']
    pins.update(identity)
    for p in (base / 'freeze/artifact_lock.json', base / 'epss/artifact_lock.json', base / 'epss/receipt.json'):
        pins[str(p)] = frozen.sha(p)
    # Include imported local preprocessing dependencies, not just the top-level helper.
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith('epss.') and path and Path(path).suffix == '.py':
            pins[path] = frozen.sha(Path(path))
    objects, counts = {}, []
    for name in tqdm(sorted(plan['training_ids']), desc='Fit preprocessing on fitting roles; export matched inputs', unit='dataset'):
        rows = frozen.read(base / 'freeze' / (name + '_screened.json'))
        ids = sorted(rows)
        fitted = roles['datasets'][name]['fitting']
        state = structured.fit([{'cve': c, 'split': 'train', 'structured': rows[c]['structured']} for c in fitted])
        observed = sorted(scores[c] for c in fitted if c in scores)
        frozen.require(observed, 'No fitting EPSS observations; cannot define numerical preprocessing')
        n = len(observed)
        median = (observed[(n-1)//2] + observed[n//2]) / 2
        state['epss'] = {'fitting_ids': fitted, 'observed_count': n, 'median': median,
                         'feature_names': ['epss_score_or_fitting_median', 'epss_observed']}
        joined = core.join_epss(ids, scores)
        predictors = {}
        for c in ids:
            vector, issues = structured.transform(rows[c]['structured'], state)
            predictors[c] = {'structured': vector, 'epss': joined[c],
                             'epss_features': [scores.get(c, median), joined[c]['observed']],
                             'documents': rows[c]['documents'], 'mask_D': rows[c]['mask_D'], 'mask_M': rows[c]['mask_M']}
        frozen.require(state['fit_ids'] == fitted, 'Preprocessing fit scope differs')
        objects[name + '_predictors.json'] = predictors
        objects[name + '_preprocessing.json'] = state
        objects[name + '_labels.json'] = frozen.read(base / 'freeze' / (name + '_labels.json'))
        for role, members in roles['datasets'][name].items():
            counts.append({'dataset': name, 'role': role, 'cves': len(members),
                           'epss_observed': sum(c in scores for c in members), 'epss_missing': sum(c not in scores for c in members)})
    findings = {'schema': cfg['schema'], 'stage': 'prepare', 'epss_date': cfg['epss_date'], 'epss_inventory': inventory,
                'support': counts, 'roles_sha256': frozen.sha(base / 'freeze/roles.json'),
                'gate': 'READY_MATCHED_INPUTS' if frozen.read(base / 'freeze/findings.json')['role_support_sufficient'] else 'BLOCKED_CLASS_SUPPORT',
                'cache_gate': cfg['cache_policy'], 'training_enabled': False, 'test_access_enabled': False,
                'meaning': cfg['interpretation']}
    objects.update({'roles.json': roles, 'protocol.json': cfg, 'identity.json': {'source_pins': pins}, 'findings.json': findings})
    frozen.check_pins(pins)
    publish(base / 'prepared', objects, {'REPORT.txt': frozen.dump(findings)}, verify, resume)
    print(frozen.dump(findings))


def run(stage, verify=False, resume=False, allow_network=False):
    frozen.require(not (verify and allow_network), 'Verification is offline')
    frozen.require(not allow_network or stage == 'collect-epss', 'Network is restricted to EPSS collection')
    cfg, plan, source, pins = context()
    base = ROOT / cfg['output']
    with frozen.exclusive(base):
        if stage == 'freeze':
            freeze(cfg, plan, source, pins, base / 'freeze', verify, resume)
        elif stage == 'collect-epss':
            check_frozen(base, cfg)
            collect(cfg, plan, base, allow_network, verify, resume)
        else:
            prepare(cfg, plan, base, pins, verify, resume)
    print('CC02 ' + stage + (' reproduced.' if verify else ' complete.') + ' No training or official holdout joins.')
