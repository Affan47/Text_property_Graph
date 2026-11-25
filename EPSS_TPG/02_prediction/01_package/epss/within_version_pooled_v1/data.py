"""Reuse immutable text means; refit structured features only inside each fold."""
from collections import Counter
from datetime import date, timedelta
import gzip
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..training_only_pooled_v1 import data as old
from ..within_version_audit_v1 import core, runner as audit

c, require, ROOT = old.c, old.require, old.ROOT
CONFIG = ROOT/'02_prediction/04_configs/07_fresh_studies/08_within_version_pooled_v1.json'
CLI = ROOT/'02_prediction/03_analysis/07_fresh_studies/15_run_within_version_pooled.py'
OUTPUT = ROOT/'05_results/08_fresh_studies/13_within_version_pooled_v1/01_reddit/v2'
SAN = ROOT/'05_results/03_dataset_analysis/02_reddit_dataset/04_premodeling_gates_v1/stage1_sanitization'
ROLES = ('optimization', 'selection', 'assessment')


def settings():
    p = c.read(CONFIG)
    require(p['schema'] == 'ct07_reddit_v2_adapter_smoke_v1' and p['arms'] == ['D', 'DF', 'DM']
            and p['seeds'] == [17, 29, 43] and p['dataset'] == 'reddit'
            and not any(p[k] for k in ('full_training_enabled', 'older_v3_training_enabled',
                                       'outer_validation_access', 'test_access')), 'Wrong CT07 scope')
    return p


def pins():
    paths = [CONFIG, CLI, *Path(__file__).parent.glob('*.py')]
    return {**audit.pins(), **{str(p): c.sha(p) for p in sorted(paths)}}


def publication_day(value, target):
    core.canonical_day(value)
    require((date.fromisoformat(value)+timedelta(days=60)).isoformat() == target,
            'Publication + 60 does not match the frozen target date')
    return value


def restrict(roles, dates, start):
    core.canonical_day(start)
    return {r: [cid for cid in ids if r != 'optimization' or dates[cid] >= start]
            for r, ids in roles.items()}


def choose(ids, rows, n):
    ordered = sorted(ids, key=lambda cid: (c.digest(cid), cid))
    require(len(ordered) >= n >= 2, 'Insufficient smoke support; do not change dates')
    chosen = []
    for flag in (True, False):
        match = next((cid for cid in ordered if rows[cid]['source_mask']['M'] == flag), None)
        if match is not None:
            chosen.append(match)
    return sorted((chosen+[cid for cid in ordered if cid not in chosen])[:n])


def definitions(manifest, dates, p):
    result = []
    for f in manifest['folds']:
        core.verify_fold(f, manifest['rows'], manifest['component_by_cve'])
        full = {r: f[r] for r in ROLES}
        after = restrict(full, dates, p['publication_start_inclusive'])
        before = sorted(set(full['optimization'])-set(after['optimization']))
        s = p['smoke']; rows = manifest['rows']
        smoke = {r: choose(full[r], rows, s[r]) for r in ('selection', 'assessment')}
        smoke['optimization'] = sorted(choose(before, rows, s['optimization_before_release'])+
                                       choose(after['optimization'], rows, s['optimization_since_release']))
        reduced = restrict(smoke, dates, p['publication_start_inclusive'])
        result.append({'fold': f['fold'], 'main': {'full': full, 'smoke': smoke},
                       'published_since_v2': {'full': after, 'smoke': reduced}})
    return result


def plan(verify=False):
    p = settings(); previous, identity = audit.parent('reddit')
    parent = audit.location('reddit', audit.settings())
    for part in ('plan', 'audit'):
        old.publication.check_lock(parent/part)
    manifest = c.read(parent/'plan/fold_manifest.json')
    require(manifest['identity'] == identity, 'CT06 parent identity changed')
    reproduced = core.make_plan(previous['rows'], previous['component_by_cve'], audit.settings(), 'reddit')
    require(all(manifest[k] == v for k, v in reproduced.items()), 'Frozen CT06 roles differ')
    inputs = dict(identity['input_hashes'])
    for part, name in [('plan', 'fold_manifest.json'), ('plan', 'artifact_lock.json'),
                       ('audit', 'findings.json'), ('audit', 'artifact_lock.json')]:
        inputs[str(parent/part/name)] = c.sha(parent/part/name)
    lock = c.read(SAN/'artifact_lock.json')['files']
    source = SAN/'predictor_candidates.jsonl.gz'
    require(c.sha(source) == lock[source.name], 'Publication source changed')
    inputs[str(source)] = lock[source.name]
    inputs[str(SAN/'artifact_lock.json')] = c.sha(SAN/'artifact_lock.json')
    dates = {}
    with gzip.open(source, 'rt') as stream:
        for line in tqdm(stream, desc='Verify original-training publication dates', unit='CVE'):
            row = json.loads(line); cid = row['cve']
            if cid not in manifest['rows']:
                continue
            require(cid not in dates, 'Duplicate publication record')
            dates[cid] = publication_day(row['published_date'], manifest['rows'][cid]['anchor_date'])
    require(set(dates) == set(manifest['rows']), 'Missing original-training publication dates')
    # Roles and smoke IDs are finalized before reading any score values.
    folds = definitions(manifest, dates, p)
    labels = core.validate_labels(manifest['rows'], old.folds.label_map(previous['label_source']))
    support = []
    for f in folds:
        for policy in p['policies']:
            for role, ids in f[policy]['full'].items():
                support.append({'fold': f['fold'], 'policy': policy, 'role': role,
                                **core.support(ids, manifest['rows'], labels, manifest['component_by_cve'], .1)})
    groups = Counter(manifest['component_by_cve'][cid] for cid in manifest['oof_ids']
                     if manifest['rows'][cid]['source_mask']['M'])
    week = [cid for cid, day in dates.items() if '2021-12-06' <= day <= '2021-12-14']
    purge = []
    for f in manifest['folds']:
        a, s = (sum(labels[cid] >= .1 for cid in f[r]) for r in ('purged_to_assessment', 'purged_to_selection'))
        purge.append({'fold': f['fold'], 'assessment_purge_high': a, 'selection_purge_high': s,
                      'total_purged_high': a+s,
                      'december_week_optimization_high': sum(cid in f['optimization'] and labels[cid] >= .1 for cid in week)})
    value = {'settings': p, 'input_hashes': inputs, 'source_pins': pins(), 'runtime': old.runtime(),
             'rows': manifest['rows'], 'component_by_cve': manifest['component_by_cve'], 'folds': folds,
             'publication_dates_not_model_input': dates, 'export': previous['export'],
             'label_source': previous['label_source'], 'directory': previous['directory'], 'encoder': previous['encoder']}
    findings = {'support': support, 'purge_high_scores': purge,
                'december_week': {'cves': len(week), 'high_score': sum(labels[cid] >= .1 for cid in week)},
                'post_bearing_assessment': {'cves': sum(groups.values()), 'groups': len(groups),
                                            'top_five_groups': groups.most_common(5)},
                'older_v3': 'Audit retained unchanged; no adapter or training enabled.',
                'archive_claims': 'No new archive-header or same-CVE daily-score verification in CT07.',
                'full_training_enabled': False, 'official_validation_or_test_adapted': False}
    with old.publication.transaction(OUTPUT/'protocol', verify=verify) as temp:
        old.base.write(temp/'protocol.json', value)
        old.base.write(temp/'findings.json', findings)
        (temp/'support.csv').write_text(old.reporting.csv_text(support))
        lines = ['CT07: Reddit v2 training-only adapter and smoke specification',
                 'All original CT06 role IDs, evidence groups and saved artifacts remain unchanged.',
                 'Main keeps all CT06 optimization records. The sensitivity keeps publication >=2022-02-04 only.',
                 'Both policies use identical selection and assessment IDs and labels; structured transforms are fitted separately.',
                 'D = description; DF = description plus post-presence flag; DM = description plus accepted post text.',
                 'Every arm has the same structured branch. G/N summaries and EPSS/KEV predictors are excluded.',
                 'Full training is disabled. Older v3 is audit-only, not declared mathematically uninterpretable.',
                 'Reddit high-score purges after BOTH stages: '+str(purge),
                 'Post-bearing OOF group concentration: '+str(findings['post_bearing_assessment']),
                 'No new archival header verification, external collection or power calculation was performed.',
                 '', *p['outcome_rules'], '', 'Fold | policy | role | CVEs | high score | with posts']
        lines += [f"{r['fold']} | {r['policy']} | {r['role']} | {r['cves']} | {r['high_score']} | {r['M_present']}" for r in support]
        (temp/'REPORT.txt').write_text('\n'.join(lines)+'\n')
        old.publication.check_pins(inputs); old.publication.check_pins(value['source_pins'])
    print(f'CT07 protocol {"reproduced" if verify else "saved"}: {OUTPUT/"protocol"}. No training or official holdout access.')


def load_protocol():
    old.publication.check_lock(OUTPUT/'protocol')
    p = c.read(OUTPUT/'protocol/protocol.json')
    require(p['settings'] == settings() and p['source_pins'] == pins() and p['runtime'] == old.runtime(),
            'CT07 protocol/code/runtime changed')
    old.publication.check_pins(p['input_hashes'])
    return p


def assert_predictors(arrays, roles):
    expected = {r+s for r in ROLES for s in ('_structured', '_D', '_DM', '_D_mask', '_DM_mask')}
    require(set(arrays) == expected and set(roles) == set(ROLES), 'Unexpected predictor keys/roles')
    for role, ids in roles.items():
        a = {k[len(role):]: v for k, v in arrays.items() if k.startswith(role+'_')}
        require(all(len(v) == len(ids) and np.isfinite(v).all() for v in a.values()), 'Bad predictor rows')
        require(a['_D'].shape == a['_DM'].shape == (len(ids), 768), 'Wrong text vector dimension')
        require(a['_D_mask'].shape == a['_DM_mask'].shape == (len(ids), 4), 'Wrong mask dimension')
        require(np.array_equal(a['_D_mask'], np.tile([1, 0, 0, 0], (len(ids), 1))), 'D mask contains non-D sources')
        mask = a['_DM_mask']
        require(np.isin(mask, [0, 1]).all() and (mask[:, 0] == 1).all() and not mask[:, 2:].any(), 'G/N or invalid mask')
        absent = mask[:, 1] == 0
        require(np.array_equal(a['_D'][absent], a['_DM'][absent]), 'Absent-post vector parity failed')


def prepare(scale, verify=False):
    require(scale in ('smoke', 'full'), 'Unknown scale')
    p = load_protocol()
    selected = sorted({cid for f in p['folds'] for r in ROLES for cid in f['main'][scale][r]})
    require(set(selected) <= set(p['rows']) and all(p['rows'][cid]['outer_role'] == 'train' for cid in selected),
            'Non-training record requested')
    cached = old.CACHE/'01_reddit/full'
    locks = c.read(cached/'artifact_lock.json')['files']
    row_pins = {str(cached/'artifact_lock.json'): c.sha(cached/'artifact_lock.json')}
    keys = sorted({p['rows'][cid][view+'_key'] for cid in selected for view in ('D', 'DM')})
    vectors = {}
    for key in tqdm(keys, desc='Reuse hash-verified standalone D/combined DM means', unit='graph'):
        name = 'rows/'+key+'.json'
        require(name in locks, 'Required CT05 pooled row missing; do not silently re-embed')
        path = cached/name
        row = old.vector_job((p['directory'], key, None, c.digest(p['encoder']), (str(path), locks[name])))
        vectors[key] = row['vector']; row_pins[str(path)] = locks[name]
        obj = Path(p['directory'])/'objects'/key
        for receipt in ('graph_receipt.json', 'embedding_receipt.json'):
            row_pins[str(obj/receipt)] = c.sha(obj/receipt)
    if scale == 'smoke':
        gp = old.base.cache.load_plan(p['directory'])
        for key in tqdm(keys, desc='Fresh compact reduction parity (no encoder rerun)', unit='graph'):
            entry = gp['entries'][key]
            require(entry['split'] == 'train' and entry['cve'] in selected, 'Held-out graph requested')
            fresh = old.vector_job((p['directory'], key, [s['source'] for s in entry['spans']], c.digest(p['encoder']), None))
            require(np.array_equal(fresh['vector'], vectors[key]), 'Compact/reused mean differs')
        del gp
    raw = c.read(Path(p['export'])/'predictor_inputs.json.gz')
    structured = {r['cve']: r for r in raw if r['cve'] in selected}; del raw
    require(set(structured) == set(selected) and all(r['split'] == 'train' for r in structured.values()), 'Wrong structured IDs')
    labels = core.validate_labels({cid: p['rows'][cid] for cid in selected}, old.folds.label_map(p['label_source']))
    identity = {'protocol_sha256': c.sha(OUTPUT/'protocol/protocol.json'), 'scale': scale,
                'source_pins': pins(), 'runtime': old.runtime(), 'row_pins': row_pins}
    with old.publication.transaction(OUTPUT/scale/'prepared', verify=verify) as temp:
        old.base.write(temp/'identity.json', identity)
        for policy in p['settings']['policies']:
            for f in tqdm(p['folds'], desc=f'{policy}: optimization-only structured encoding', unit='fold'):
                roles = f[policy][scale]
                state = old.structured.fit([structured[cid] for cid in roles['optimization']])
                require(state['fit_ids'] == sorted(roles['optimization']), 'Preprocessor uses non-optimization records')
                arrays, ys = {}, {}
                for role, ids in roles.items():
                    ys[role] = [labels[cid] for cid in ids]
                    arrays[role+'_structured'] = np.asarray([old.structured.transform(structured[cid]['structured'], state)[0] for cid in ids], dtype=np.float32)
                    for view in ('D', 'DM'):
                        arrays[role+'_'+view] = np.asarray([vectors[p['rows'][cid][view+'_key']] for cid in ids], dtype=np.float32)
                        arrays[role+'_'+view+'_mask'] = np.asarray([[1, int(view == 'DM' and p['rows'][cid]['source_mask']['M']), 0, 0] for cid in ids], dtype=np.float32)
                assert_predictors(arrays, roles)
                folder = temp/policy/f"fold_{f['fold']}"; folder.mkdir(parents=True)
                old.base.write(folder/'ids.json', roles); old.base.write(folder/'labels.json', ys)
                old.base.write(folder/'structured_state.json', state)
                old.npz_write(folder/'predictors.npz', arrays)
        old.publication.check_pins(row_pins); old.publication.check_pins(p['input_hashes']); old.publication.check_pins(pins())
    print(f'CT07 {scale} adapter {"reproduced" if verify else "saved"}. No training or new embeddings.')


def load_fold(p, scale, policy, fold):
    folder = OUTPUT/scale/'prepared'/policy/f'fold_{fold}'
    ids, ys = c.read(folder/'ids.json'), c.read(folder/'labels.json')
    expected = next(f[policy][scale] for f in p['folds'] if f['fold'] == fold)
    require(ids == expected, 'Prepared roles differ from frozen protocol')
    require(c.read(folder/'structured_state.json')['fit_ids'] == ids['optimization'], 'Wrong preprocessing fit IDs')
    require(set(ys) == set(ROLES) and all(len(ys[r]) == len(ids[r]) for r in ROLES), 'Label alignment differs')
    with np.load(folder/'predictors.npz', allow_pickle=False) as loaded:
        arrays = dict(loaded)
    assert_predictors(arrays, ids)
    return ids, arrays, ys
