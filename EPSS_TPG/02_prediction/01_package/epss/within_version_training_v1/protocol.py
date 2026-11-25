"""Freeze additions around immutable CT07 data; preserve historical OOF evidence."""
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..within_version_pooled_v1 import data, runner as adapter_runner
from . import reporting

CONFIG = data.ROOT/'02_prediction/04_configs/07_fresh_studies/09_within_version_training_v1.json'
CLI = data.ROOT/'02_prediction/03_analysis/07_fresh_studies/17_run_within_version_training.py'
OUTPUT = data.ROOT/'05_results/08_fresh_studies/14_within_version_training_v1/01_reddit/v2'


def settings():
    p = data.c.read(CONFIG); old = data.settings()
    data.require(p['schema'] == 'ct07_within_version_full_execution_v1' and p['full_training_enabled']
                 and not p['official_validation_access'] and not p['test_access'], 'Wrong execution scope')
    for key in ('policies', 'arms', 'seeds', 'controls', 'hidden', 'batch_size', 'learning_rate',
                'weight_decay', 'epochs', 'patience', 'bootstrap_repetitions', 'bootstrap_seed', 'bootstrap_units'):
        data.require(p[key] == old[key], 'Unexpected change to frozen setting: '+key)
    return p


def pins():
    return {**data.pins(), **{str(p): data.c.sha(p) for p in [CONFIG, CLI, *sorted(Path(__file__).parent.glob('*.py'))]}}


def row(p, policy, fold, name, seed, cid, y, prediction):
    return {'policy': policy, 'fold': fold, 'model': name, 'seed': seed, 'cve': cid, 'actual': float(y),
            'prediction': float(prediction), 'M_present': p['rows'][cid]['source_mask']['M'],
            'target_date': p['rows'][cid]['anchor_date']}


def control_rows(p, scale, policy):
    rows, states = [], []
    for f in p['folds']:
        ids, _, y = data.load_fold(p, scale, policy, f['fold'])
        values = reporting.constants(y['optimization'])
        states.append({'policy': policy, 'fold': f['fold'], 'optimization_ids_sha256': data.c.digest(ids['optimization']),
                       'optimization_cves': len(ids['optimization']), 'values': values})
        for name, value in values.items():
            rows += [row(p, policy, f['fold'], name, 0, cid, actual, value) for cid, actual in zip(ids['assessment'], y['assessment'])]
    return rows, states


def historical(p):
    old = data.old; base = old.OUT/'01_reddit'; spec = old.load_protocol('reddit')
    for folder in (base/'full/report', base/'full/prepared', base/'flag_checkpoint_review_v1'):
        old.publication.check_lock(folder)
    report_identity = data.c.read(base/'full/report/identity.json')
    old.publication.check_pins(report_identity['fit_locks'])
    for path in report_identity['fit_locks']:
        old.publication.check_lock(Path(path).parent)
    flag_identity = data.c.read(base/'flag_checkpoint_review_v1/identity.json')['input_and_code_hashes']
    old.publication.check_pins(flag_identity)
    refs = {str(base/q): data.c.sha(base/q) for q in ('full/report/artifact_lock.json', 'full/prepared/artifact_lock.json',
                                                    'flag_checkpoint_review_v1/artifact_lock.json', 'protocol/protocol.json')}
    assignments = {cid: f['fold'] for f in p['folds'] for cid in f['main']['full']['assessment']}
    target_lookup = data.core.validate_labels({cid: p['rows'][cid] for cid in assignments}, old.folds.label_map(p['label_source']))
    with (base/'full/report/oof_predictions.csv').open(newline='') as stream:
        official = {(r['model'], int(r['seed']), r['cve']): r for r in csv.DictReader(stream)
                    if r['model'] in ('D', 'DM') and r['cve'] in assignments}
    rows, crosswalk = [], []
    for f in tqdm(spec['folds'], desc='Verify historical OOF roles and frozen reference predictions', unit='fold'):
        fold = f['fold']; prepared = old.load_fold('reddit', 'full', fold)
        ids = prepared['ids']; pos = {cid: i for i, cid in enumerate(ids['assessment'])}
        selected = sorted(set(pos) & set(assignments))
        training_components = {spec['component_by_cve'][cid] for role in ('optimization', 'selection') for cid in ids[role]}
        data.require(not training_components & {spec['component_by_cve'][cid] for cid in selected}, 'Reference evidence overlaps fitting')
        paths = [Path(path) for path in flag_identity if path.endswith(f'ct05_flag_01_reddit_fold{fold}.npz')]
        data.require(len(paths) == 1, 'Ambiguous DF historical source')
        source = paths[0]; refs[str(source)] = flag_identity[str(source)]
        with np.load(source, allow_pickle=False) as z:
            data.require(z['ids'].tolist() == ids['assessment'] and np.array_equal(z['y'], prepared['labels']['assessment']), 'DF rows/labels differ')
            for cid in selected:
                i = pos[cid]; actual = target_lookup[cid]
                data.require(actual == z['y'][i] and bool(z['mflag'][i]) == p['rows'][cid]['source_mask']['M'], 'Historical target/mask mismatch')
                crosswalk.append({'cve': cid, 'ct07_fold': assignments[cid], 'ct05_fold': fold,
                                  'ct05_role': 'assessment', 'ct07_role': 'assessment'})
                for arm in ('D', 'DF', 'DM'):
                    for seed in p['settings']['seeds']:
                        pred = float(z[f'{arm}|{seed}'][i])
                        if arm != 'DF':
                            r = official[arm, seed, cid]
                            data.require(float(r['prediction']) == pred and float(r['actual']) == actual and int(r['fold']) == fold,
                                         'Historical predictions do not match official CT05 output')
                        data.require(np.isfinite(pred) and 0 <= pred <= 1, 'Invalid historical prediction')
                        rows.append(row(p, 'main', assignments[cid], arm, seed, cid, actual, pred))
    data.require(len(crosswalk) == len(assignments) and {r['cve'] for r in crosswalk} == set(assignments), 'Incomplete/duplicate historical OOF mapping')
    return rows, crosswalk, {**refs, **flag_identity}


def freeze(verify=False):
    p = data.load_protocol(); s = settings(); source_hashes = {}
    for part in ('full/prepared', 'smoke/prepared', 'smoke/report'):
        data.old.publication.check_lock(data.OUTPUT/part)
        path = data.OUTPUT/part/'artifact_lock.json'; source_hashes[str(path)] = data.c.sha(path)
    full_identity = data.c.read(data.OUTPUT/'full/prepared/identity.json')
    data.require(full_identity['source_pins'] == data.pins() and full_identity['protocol_sha256'] == data.c.sha(data.OUTPUT/'protocol/protocol.json'), 'Wrong full preparation identity')
    data.old.publication.check_pins(full_identity['row_pins'])
    smoke = data.c.read(data.OUTPUT/'smoke/report/identity.json')
    data.require(smoke['source_pins'] == data.pins(), 'Old smoke identity changed')
    data.old.publication.check_pins(smoke['fit_locks'])
    control_predictions, states = [], []
    for policy in s['policies']:
        rs, st = control_rows(p, 'full', policy); control_predictions += rs; states += st
    refs, crosswalk, origin_hashes = historical(p)
    reference_settings = {**s, 'policies': ['main']}
    ref_metrics, ref_intervals = reporting.analyze(refs+[r for r in control_predictions if r['policy'] == 'main'], p, reference_settings, s['bootstrap_repetitions'])
    constant_metrics = []
    for policy in s['policies']:
        for name in s['controls']:
            for part in s['metric_partitions']:
                rs = sorted([r for r in control_predictions if r['policy'] == policy and r['model'] == name and reporting.in_partition(r, part)], key=lambda r:r['cve'])
                if rs:
                    constant_metrics.append({'policy': policy, 'model': name, 'partition': part,
                                             **reporting.old_report.metrics([r['actual'] for r in rs], [r['prediction'] for r in rs], [r['cve'] for r in rs])})
    artifact = {'settings': s, 'ct07_protocol_sha256': data.c.sha(data.OUTPUT/'protocol/protocol.json'),
                'source_pins': pins(), 'runtime': data.old.runtime(), 'input_hashes': source_hashes,
                'historical_origin_hashes_for_record_not_runtime_dependency': origin_hashes,
                'old_df_origin': 'External exploratory flag arm; nine checkpoints replayed and hash-locked in CT05 flag_checkpoint_review_v1',
                'interpretation': 'Exploratory re-estimation on previously assessed CVEs; frozen constants and historical references do not select new checkpoints.'}
    with data.old.publication.transaction(OUTPUT/'protocol', verify=verify) as temp:
        data.old.base.write(temp/'protocol.json', artifact)
        data.old.base.write(temp/'constant_states.json', states)
        data.old.base.write(temp/'findings.json', {'constant_metrics': constant_metrics, 'ct05_reference_metrics': ref_metrics,
                                                  'ct05_reference_intervals': ref_intervals, 'reference_cves': len(crosswalk),
                                                  'historical_reference_only': True, 'full_neural_training': False})
        for name, rs in [('constant_predictions', control_predictions), ('constant_metrics', constant_metrics),
                         ('ct05_reference_predictions', refs), ('ct05_reference_metrics', ref_metrics),
                         ('ct05_reference_intervals', ref_intervals), ('historical_role_crosswalk', crosswalk)]:
            (temp/(name+'.csv')).write_text(data.old.reporting.csv_text(rs))
        lines = ['CT07 full-execution specification and frozen pre-training controls',
                 'Constants use each policy and fold\'s optimization scores only. Selection/assessment never estimate their values.',
                 'Earlier CT05 predictions were OOF for each referenced CVE, not fitted or checkpoint-selected on those CVEs.',
                 'CT05 had different folds, fitting cohorts and preprocessing: reference contrasts do not isolate a version effect.',
                 'DF was an external exploratory arm with replayed checkpoints; it was not an original preregistered CT05 arm.',
                 'The same CVEs have already been assessed. This is exploratory re-estimation, not new independent evidence.',
                 'Historical source values and hashes are recorded here durably; /tmp is not needed during subsequent training.',
                 '', 'Policy | constant | partition | CVEs | RMSE']
        lines += [f"{r['policy']} | {r['model']} | {r['partition']} | {r['n']} | {r['rmse']:.9f}" for r in constant_metrics if r['partition'] in s['interval_partitions']]
        lines += ['', 'CT05 reference | partition | three-seed mean RMSE']
        for name in s['arms']:
            for part in s['interval_partitions']:
                values = [r['rmse'] for r in ref_metrics if r['model'] == name and r['partition'] == part]
                lines.append(f'{name} | {part} | {np.mean(values):.9f}')
        lines += ['', *s['outcome_rules'], '',
                  'Full run: 27 main + 27 sensitivity neural fits, CPU, 4 threads/fit, three fixed seeds, no GNN/encoder retraining.',
                  'Same pooled architecture, MSE, AdamW, 100 maximum epochs, patience 15, best checkpoint only.',
                  '20,000 paired resamples, seed 6029, CVE and evidence-component units, all/post-present/post-absent contrasts.',
                  'Fresh smoke must reproduce CT07 smoke histories and predictions before full training. No full fits were launched by this freeze.']
        (temp/'REPORT.txt').write_text('\n'.join(lines)+'\n')
        reporting.figure(temp/'ct05_reference_rmse.png', ref_metrics, ['main'], s['arms'], s['controls'], 'Historical CT05 predictions on CT07 assessment CVEs (not new fits)')
        data.old.publication.check_pins(origin_hashes); data.old.publication.check_pins(source_hashes); data.old.publication.check_pins(pins())
    print(f'CT07 full specification {"reproduced" if verify else "frozen"}: {OUTPUT/"protocol"}. No full training or official holdout access.')


def load():
    data.old.publication.check_lock(OUTPUT/'protocol')
    frozen = data.c.read(OUTPUT/'protocol/protocol.json')
    data.require(frozen['settings'] == settings() and frozen['source_pins'] == pins()
                 and frozen['runtime'] == data.old.runtime()
                 and frozen['ct07_protocol_sha256'] == data.c.sha(data.OUTPUT/'protocol/protocol.json'), 'Frozen execution identity changed')
    data.old.publication.check_pins(frozen['input_hashes'])
    return frozen, data.load_protocol()
