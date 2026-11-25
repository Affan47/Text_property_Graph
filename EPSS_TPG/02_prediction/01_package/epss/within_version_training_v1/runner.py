"""Parallel independent fits; immutable inputs, best-only checkpoints and replay."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from ..training_only_pooled_v1 import fitting, model
from ..within_version_pooled_v1 import data, runner as adapter
from . import protocol, reporting

_STATE = None


def parameters(scale):
    p = data.settings().copy()
    if scale == 'smoke':
        p.update({k: p['smoke'][k] for k in ('epochs', 'patience')})
    return p


def init_worker(frozen, p, scale):
    global _STATE
    torch.set_num_threads(4)
    _STATE = frozen, p, scale


def assert_frozen_identity(frozen):
    data.require(protocol.pins() == frozen['source_pins'] and data.old.runtime() == frozen['runtime'],
                 'Frozen code/runtime changed between fits; do not continue a mixed-identity run')
    data.old.publication.check_pins(frozen['input_hashes'])


def one_fit(job):
    policy, fold, arm, seed, verify, resume = job
    frozen, p, scale = _STATE
    settings = parameters(scale)
    with threadpool_limits(limits=4, user_api='blas'):
        assert_frozen_identity(frozen)
        ids, arrays, ys = data.load_fold(p, scale, policy, fold)
        inputs = {r: adapter.inputs(arrays, r, arm) for r in data.ROLES}
        prepared_lock = data.OUTPUT/scale/'prepared/artifact_lock.json'
        identity = {'dataset': 'reddit_v2_'+policy, 'scale': scale, 'policy': policy, 'fold': fold,
                    'arm': arm, 'seed': seed, 'settings': settings,
                    'execution_settings': frozen['settings'], 'source_pins': frozen['source_pins'], 'runtime': frozen['runtime'],
                    'prepared_sha256': data.c.sha(prepared_lock),
                    'protocol_sha256': data.c.sha(protocol.OUTPUT/'protocol/protocol.json')}
        root = protocol.OUTPUT/scale/'fits'/policy/f'fold_{fold}'/arm/f'seed_{seed}'
        with data.old.execution.exclusive(root.parent/(root.name+'.lock')):
            complete = (root/'artifact_lock.json').exists()
            data.require(not verify or complete, 'Missing fit: verification never trains')
            if complete:
                data.old.publication.check_lock(root)
                net, checkpoint = fitting.reload(root, identity, inputs['optimization']['structured'].shape[1], settings,
                                                 inputs['selection'], ys['selection'])
            else:
                tqdm.write(f'[CT07-FULL] scale={scale} policy={policy} fold={fold} {arm} seed={seed}; shared CT05 fit engine')
                net, checkpoint = fitting.fit(root, identity, inputs['optimization'], ys['optimization'],
                                              inputs['selection'], ys['selection'], settings, seed, fold, resume)
            pred = model.predict(net, inputs['assessment'])
            reload_net, _ = fitting.reload(root, identity, inputs['optimization']['structured'].shape[1], settings,
                                           inputs['selection'], ys['selection'])
            data.require(np.array_equal(pred, model.predict(reload_net, inputs['assessment'])), 'Fresh checkpoint reload differs')
            rows = [protocol.row(p, policy, fold, arm, seed, cid, y, value)
                    for cid, y, value in zip(ids['assessment'], ys['assessment'], pred)]
            smoke_parity = None
            if scale == 'smoke':
                old_root = data.OUTPUT/'smoke/fits'/policy/f'fold_{fold}'/arm/f'seed_{seed}'
                data.old.publication.check_lock(old_root)
                cp = torch.load(old_root/'best_checkpoint.pt', map_location='cpu', weights_only=False)
                data.require(data.c.read(root/'history.json') == data.c.read(old_root/'history.json')
                             and checkpoint['epoch'] == cp['epoch']
                             and all(torch.equal(v, cp['model'][k]) for k, v in checkpoint['model'].items()),
                             'New execution does not match frozen CT07 smoke trajectory')
                old_pred = adapter.prediction_rows(p, policy, fold, arm, seed, ids['assessment'], ys['assessment'], pred)
                data.require((old_root/'predictions.csv').read_bytes() == data.old.reporting.csv_text(old_pred).encode(), 'Old/new smoke predictions differ')
                smoke_parity = {'old_lock_sha256': data.c.sha(old_root/'artifact_lock.json'), 'history_weights_predictions_exact': True}
            result = {'checkpoint_sha256': data.c.sha(root/'best_checkpoint.pt'), 'best_epoch': checkpoint['epoch'],
                      'selection_rmse': checkpoint['selection_rmse'], 'history_epochs': len(data.c.read(root/'history.json')),
                      'assessment': reporting.old_report.metrics(ys['assessment'], pred, ids['assessment']),
                      'smoke_parity': smoke_parity}
            payload = data.old.reporting.csv_text(rows)
            if complete:
                data.require(data.c.read(root/'evaluation.json') == result and (root/'predictions.csv').read_bytes() == payload.encode(),
                             'Saved model evaluation does not reproduce')
            else:
                data.old.base.write(root/'evaluation.json', result); (root/'predictions.csv').write_text(payload)
                data.old.publication.check_pins(identity['source_pins'])
                data.old.publication.lock(root)
            return rows, {str(root/'artifact_lock.json'): data.c.sha(root/'artifact_lock.json')}


def check(scale, jobs):
    data.require(scale in ('smoke', 'full'), 'Unknown scale')
    frozen, p = protocol.load(); s = frozen['settings']
    data.require(1 <= jobs <= s['max_jobs'] and jobs*4 <= len(os.sched_getaffinity(0)), 'CPU job budget exceeded')
    with open('/proc/meminfo') as stream:
        available = int(next(line.split()[1] for line in stream if line.startswith('MemAvailable:')))*1024
    data.require(jobs*4*2**30 < .7*available, 'RAM budget exceeded; reduce --jobs')
    prepared = data.OUTPUT/scale/'prepared'
    data.old.publication.check_lock(prepared)
    ident = data.c.read(prepared/'identity.json')
    data.require(ident['source_pins'] == data.pins() and ident['protocol_sha256'] == frozen['ct07_protocol_sha256'], 'Prepared inputs differ')
    for policy in s['policies']:
        for f in p['folds']:
            data.load_fold(p, scale, policy, f['fold'])
    if scale == 'full':
        smoke = protocol.OUTPUT/'smoke/report'; data.old.publication.check_lock(smoke)
        receipt = data.c.read(smoke/'identity.json')
        data.require(receipt['source_pins'] == protocol.pins()
                     and receipt['protocol_sha256'] == data.c.sha(protocol.OUTPUT/'protocol/protocol.json'), 'Run matching new smoke first')
        data.old.publication.check_pins(receipt['fit_locks'])
    return frozen, p


def publish(rows, p, frozen, fit_locks, parities, scale, verify):
    settings = frozen['settings']; root = protocol.OUTPUT/scale/'report'
    reps = settings['smoke']['bootstrap_repetitions'] if scale == 'smoke' else settings['bootstrap_repetitions']
    for policy in settings['policies']:
        counts = {f['fold']: len(f[policy][scale]['assessment']) for f in p['folds']}
        for arm, seed in [(a, s) for a in settings['arms'] for s in settings['seeds']]+[(c, 0) for c in settings['controls']]:
            subset = [r for r in rows if (r['policy'], r['model'], r['seed']) == (policy, arm, seed)]
            data.require({f: sum(r['fold'] == f for r in subset) for f in counts} == counts, 'Incomplete OOF coverage')
    metrics, intervals = reporting.analyze(rows, p, settings, reps)
    gates = reporting.outcome_gates(intervals, settings)
    with data.old.publication.transaction(root, verify=verify) as temp:
        data.old.base.write(temp/'identity.json', {'protocol_sha256': data.c.sha(protocol.OUTPUT/'protocol/protocol.json'),
                                                  'fit_locks': fit_locks, 'source_pins': protocol.pins()})
        data.old.base.write(temp/'findings.json', {'scale': scale, 'metrics': metrics, 'paired_intervals': intervals,
                                                 'reporting_gates_not_confirmatory_tests': gates, 'same_weight_parity': parities,
                                                 'smoke_only': scale == 'smoke', 'official_validation_or_test_scored': False})
        for name, rs in [('oof_predictions', rows), ('metrics', metrics), ('paired_intervals', intervals), ('reporting_gates', gates), ('parity', parities)]:
            (temp/(name+'.csv')).write_text(data.old.reporting.csv_text(rs))
        lines = [f'CT07 {scale}: fixed within-v2 pooled comparison with policy-specific constants',
                 'D: standalone description. DF: D plus post-presence flag. DM: combined description and accepted posts.',
                 'All arms include matched optimization-fitted structured features; no GNN is trained.',
                 '27 main and 27 separately reported sensitivity fits. The assessment CVEs and labels match between policies.',
                 'Controls use each policy/fold optimization role only. Assessment scores never fit constants or choose checkpoints.',
                 'RMSE primary, mean of seed-wise estimates. All, post-present and post-absent paired contrasts use both CVE and evidence-component resampling.',
                 f'{reps} resamples per comparison, seed 6029. Intervals are nominal and exploratory, not multiplicity-adjusted confirmation.',
                 'Earlier CT05 numbers are frozen in protocol/: reference only, not new data or a matched version-only intervention.',
                 'The same data have been inspected repeatedly. No official validation/test scoring, threshold tuning, or automatic promotion.',
                 'Smoke values test execution only; never use them as performance evidence.' if scale == 'smoke' else 'Full results retain all fixed folds, seeds and declared comparisons.',
                 '', *settings['outcome_rules'], '', 'Policy | model | partition | mean seed-wise RMSE']
        for policy in settings['policies']:
            for name in settings['controls']+settings['arms']:
                for part in settings['interval_partitions']:
                    values = [r['rmse'] for r in metrics if r['policy'] == policy and r['model'] == name and r['partition'] == part and r['n']]
                    if values:
                        lines.append(f'{policy} | {name} | {part} | {np.mean(values):.9f}')
        (temp/'REPORT.txt').write_text('\n'.join(lines)+'\n')
        reporting.figure(temp/'oof_rmse.png', metrics, settings['policies'], settings['arms'], settings['controls'],
                         'CT07 smoke: wiring only, not performance evidence' if scale == 'smoke' else 'CT07 full: paired inner-assessment score estimation')
        data.old.publication.check_pins(fit_locks); assert_frozen_identity(frozen)


def run(scale, jobs=3, verify=False, resume=False, check_only=False):
    frozen, p = check(scale, jobs)
    if check_only:
        print(f'CT07 {scale} full-execution preflight passed: 54 fits, CPU {jobs} jobs x 4 threads; no training.')
        return
    root = protocol.OUTPUT/scale; s = frozen['settings']; tasks = []
    with data.old.execution.exclusive(protocol.OUTPUT/(scale+'.run.lock')):
        rows, states, parities, hashes = [], [], [], {}
        for policy in s['policies']:
            r, st = protocol.control_rows(p, scale, policy); rows += r; states += st
            for f in tqdm(p['folds'], desc=policy+': same-weight source-mask parity', unit='fold'):
                _, a, _ = data.load_fold(p, scale, policy, f['fold'])
                parities += [{'policy': policy, 'fold': f['fold'], **r} for r in adapter.parity(a, p['settings'])]
        # Full controls must match the pre-training publication byte for byte.
        if scale == 'full':
            data.require(data.c.read(protocol.OUTPUT/'protocol/constant_states.json') == states
                         and (protocol.OUTPUT/'protocol/constant_predictions.csv').read_bytes() == data.old.reporting.csv_text(rows).encode(),
                         'Constants differ from frozen pre-training controls')
        for policy in s['policies']:
            for f in p['folds']:
                tasks += [(policy, f['fold'], arm, seed, verify, resume) for arm in s['arms'] for seed in s['seeds']]
        if jobs == 1:
            init_worker(frozen, p, scale)
            for task in tqdm(tasks, desc='CT07 matched fits/replays', unit='fit'):
                r, h = one_fit(task); rows += r; hashes.update(h)
        else:
            with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context('spawn'), initializer=init_worker,
                                     initargs=(frozen, p, scale)) as pool:
                futures = [pool.submit(one_fit, task) for task in tasks]
                for future in tqdm(as_completed(futures), total=len(futures), desc='CT07 matched fits/replays', unit='fit'):
                    r, h = future.result(); rows += r; hashes.update(h)
        destination = root/'report'
        data.require(not verify or destination.exists(), 'Missing report cannot be verified')
        publish(rows, p, frozen, hashes, parities, scale, destination.exists())
    print(f'CT07 {scale} full-execution {"checkpoints, controls and OOF reports verified" if verify else "saved"}: {root}. No official validation/test scoring.')
