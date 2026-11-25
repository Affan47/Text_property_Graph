"""Small deterministic integration fits only; full training is a later gate."""
import numpy as np
import torch
from tqdm import tqdm

from ..training_only_pooled_v1 import fitting, model, report as metrics
from . import data


def inputs(arrays, role, arm):
    data.require(arm in ('D', 'DF', 'DM'), 'Unknown arm')
    return {k: torch.from_numpy(np.asarray(arrays[role+s], dtype=np.float32)) for k, s in
            [('text', '_DM' if arm == 'DM' else '_D'), ('structured', '_structured'),
             ('source_mask', '_D_mask' if arm == 'D' else '_DM_mask')]}


def scope(scale):
    data.require(scale == 'smoke', 'CT07 full training is disabled; review the adapter and smoke gate first')


def parity(arrays, p):
    results = []
    for seed in p['seeds']:
        fitting.seed_all(seed)
        net = model.Regressor(arrays['assessment_structured'].shape[1], p['hidden'])
        for role in data.ROLES:
            absent = arrays[role+'_DM_mask'][:, 1] == 0
            x = {arm: inputs(arrays, role, arm) for arm in p['arms']}
            data.require(torch.equal(x['D']['text'], x['DF']['text']), 'DF must reuse standalone D exactly')
            predictions = {arm: model.predict(net, a) for arm, a in x.items()}
            data.require(all(np.array_equal(predictions['D'][absent], predictions[a][absent]) for a in ('DF', 'DM')),
                         'Same-weight masked-source parity failed')
            results.append({'seed': seed, 'role': role, 'absent_M': int(absent.sum()),
                            'max_absent_prediction_difference': 0.0,
                            'parameters_all_arms': sum(t.numel() for t in net.parameters())})
    return results


def train(scale='smoke', verify=False, resume=False):
    scope(scale)
    p = data.load_protocol(); settings = {**p['settings'], **p['settings']['smoke']}
    root = data.OUTPUT/scale; prepared = root/'prepared'
    data.old.publication.check_lock(prepared)
    ident = data.c.read(prepared/'identity.json')
    data.require(ident['source_pins'] == data.pins() and ident['runtime'] == data.old.runtime()
                 and ident['protocol_sha256'] == data.c.sha(data.OUTPUT/'protocol/protocol.json'), 'Prepared identity differs')
    data.old.publication.check_pins(ident['row_pins'])
    rows, parities, hashes = [], [], {}
    total = len(settings['policies'])*len(p['folds'])*len(settings['arms'])*len(settings['seeds'])
    with data.old.execution.exclusive(root/'training.lock'):
        with tqdm(total=total, desc='CT07 smoke fits/checkpoint replays', unit='fit') as bar:
            for policy in settings['policies']:
                for f in p['folds']:
                    fold = f['fold']; ids, arrays, ys = data.load_fold(p, scale, policy, fold)
                    parities += [{'policy': policy, 'fold': fold, **r} for r in parity(arrays, settings)]
                    for name, value in [('optimization_mean', np.mean(ys['optimization'])),
                                        ('optimization_median', np.median(ys['optimization']))]:
                        rows += prediction_rows(p, policy, fold, name, 0, ids['assessment'], ys['assessment'],
                                                np.full(len(ids['assessment']), value))
                    for arm in settings['arms']:
                        for seed in settings['seeds']:
                            folder = root/'fits'/policy/f'fold_{fold}'/arm/f'seed_{seed}'
                            identity = {'dataset': 'reddit_v2_'+policy, 'arm': arm, 'seed': seed, 'fold': fold,
                                        'prepared_sha256': data.c.sha(prepared/'artifact_lock.json'),
                                        'settings': settings, 'source_pins': data.pins(), 'runtime': data.old.runtime()}
                            xx = {r: inputs(arrays, r, arm) for r in data.ROLES}
                            complete = (folder/'artifact_lock.json').exists()
                            if verify or complete:
                                data.require(complete, 'Missing completed fit; run smoke without --verify-only first')
                                data.old.publication.check_lock(folder)
                                net, cp = fitting.reload(folder, identity, xx['optimization']['structured'].shape[1], settings,
                                                         xx['selection'], ys['selection'])
                            else:
                                tqdm.write(f'[CT07] {policy} fold={fold} arm={arm} seed={seed}; using pinned CT05 fitting engine')
                                net, cp = fitting.fit(folder, identity, xx['optimization'], ys['optimization'],
                                                      xx['selection'], ys['selection'], settings, seed, fold, resume)
                            predictions = model.predict(net, xx['assessment'])
                            again, _ = fitting.reload(folder, identity, xx['optimization']['structured'].shape[1], settings,
                                                      xx['selection'], ys['selection'])
                            data.require(np.array_equal(predictions, model.predict(again, xx['assessment'])), 'Checkpoint replay differs')
                            rs = prediction_rows(p, policy, fold, arm, seed, ids['assessment'], ys['assessment'], predictions)
                            payload = data.old.reporting.csv_text(rs)
                            if complete:
                                data.require((folder/'predictions.csv').read_bytes() == payload.encode(), 'Saved predictions differ')
                            else:
                                (folder/'predictions.csv').write_text(payload)
                                data.old.publication.check_pins(identity['source_pins'])
                                data.old.publication.lock(folder)
                            hashes[str(folder/'artifact_lock.json')] = data.c.sha(folder/'artifact_lock.json')
                            rows += rs; bar.update()
        publish(p, rows, parities, hashes, verify or (root/'report').exists())
    print(f'CT07 smoke {"reproduced" if verify else "saved"}: {root/"report"}. 54 bounded fits; no full training or official validation/test scoring.')


def prediction_rows(p, policy, fold, arm, seed, ids, ys, predictions):
    data.require(len(ids) == len(ys) == len(predictions) and np.isfinite(predictions).all(), 'Invalid prediction alignment')
    return [{'policy': policy, 'fold': fold, 'arm': arm, 'seed': seed, 'cve': cid,
             'actual': float(y), 'prediction': float(pred), 'M_present': p['rows'][cid]['source_mask']['M'],
             'component': p['component_by_cve'][cid]} for cid, y, pred in zip(ids, ys, predictions)]


def publish(p, rows, parities, hashes, verify):
    data.require(len(rows) == len({(r['policy'], r['arm'], r['seed'], r['cve']) for r in rows}), 'Repeated OOF prediction')
    summaries, intervals = [], []
    parts = ('all', 'fold_1', 'fold_2', 'fold_3', 'M_present', 'M_absent', 'score_ge_0.1', 'score_lt_0.1')
    def subset(rs, part):
        return sorted([r for r in rs if part == 'all' or part == f"fold_{r['fold']}" or
                       part == ('M_present' if r['M_present'] else 'M_absent') or
                       part == ('score_ge_0.1' if r['actual'] >= .1 else 'score_lt_0.1')], key=lambda r: r['cve'])
    for policy in p['settings']['policies']:
        expected = {cid for f in p['folds'] for cid in f[policy]['smoke']['assessment']}
        variants = [(a, s) for a in p['settings']['arms'] for s in p['settings']['seeds']]+[(a, 0) for a in p['settings']['controls']]
        for arm, seed in variants:
            rs = [r for r in rows if (r['policy'], r['arm'], r['seed']) == (policy, arm, seed)]
            data.require({r['cve'] for r in rs} == expected, 'Incomplete OOF coverage')
            for part in parts:
                ss = subset(rs, part)
                value = metrics.metrics([r['actual'] for r in ss], [r['prediction'] for r in ss], [r['cve'] for r in ss]) if ss else {'n': 0}
                summaries.append({'policy': policy, 'arm': arm, 'seed': seed, 'partition': part, **value})
        for part in tqdm(parts[:6], desc=f'{policy}: smoke interval wiring', unit='partition'):
            rs = subset([r for r in rows if (r['policy'], r['arm'], r['seed']) == (policy, 'D', 17)], part)
            ids = [r['cve'] for r in rs]
            if not ids:
                continue
            lookup = {(r['arm'], r['seed'], r['cve']): r for r in rows if r['policy'] == policy}
            errors = {arm: np.array([[(lookup[arm, seed, cid]['prediction']-lookup[arm, seed, cid]['actual'])**2 for cid in ids]
                                    for seed in p['settings']['seeds']]) for arm in p['settings']['arms']}
            for a, b in [('DM', 'D'), ('DM', 'DF'), ('DF', 'D')]:
                for unit in p['settings']['bootstrap_units']:
                    groups = ids if unit == 'CVE' else [p['component_by_cve'][cid] for cid in ids]
                    intervals.append({'policy': policy, 'contrast': a+'_minus_'+b, 'partition': part, 'unit': unit,
                                      **metrics.paired_ci(errors[a], errors[b], groups, p['settings']['smoke']['bootstrap_repetitions'],
                                                         p['settings']['bootstrap_seed'])})
    with data.old.publication.transaction(data.OUTPUT/'smoke/report', verify=verify) as temp:
        data.old.base.write(temp/'identity.json', {'fit_locks': hashes, 'source_pins': data.pins(),
                                                  'prepared_sha256': data.c.sha(data.OUTPUT/'smoke/prepared/artifact_lock.json')})
        findings = {'scope': 'smoke wiring only; not scientific performance evidence', 'fits': len(hashes),
                    'policies': p['settings']['policies'], 'same_weight_parity': parities,
                    'metrics': summaries, 'paired_intervals': intervals,
                    'checkpoint_and_prediction_replay': True, 'full_training_enabled': False,
                    'official_validation_or_test_scored': False}
        data.old.base.write(temp/'findings.json', findings)
        for name, value in [('predictions', rows), ('metrics', summaries), ('paired_intervals', intervals), ('parity', parities)]:
            (temp/(name+'.csv')).write_text(data.old.reporting.csv_text(value))
        (temp/'REPORT.txt').write_text('CT07 Reddit v2 smoke gate\n54 fits: two policies x three folds x three arms x three seeds.\n'
            'Optimization fits preprocessing/weights, selection chooses the best checkpoint, assessment is scored afterwards.\n'
            'D: description. DF: identical description vector plus post flag. DM: combined description and post vector.\n'
            'All arms include identical structured inputs within a policy and fold. No GNN is trained.\n'
            'Masked-source same-weight predictions and checkpoint reloads are exact. CSVs are canonically sorted.\n'
            'Main uses 16 optimization CVEs/fold; publication sensitivity uses its 8 post-release CVEs.\n'
            'Both policies use the same 8 selection and 8 assessment CVEs per fold. No score-based smoke sampling.\n'
            'Two epochs, not full training. Smoke intervals have 20 resamples to test the reporting path only.\n'
            'A later approved full protocol uses 20,000 CVE and evidence-component resamples.\n'
            'No performance conclusions, threshold tuning, official validation or test scoring.\n')
        data.old.publication.check_pins(hashes); data.old.publication.check_pins(data.pins())
