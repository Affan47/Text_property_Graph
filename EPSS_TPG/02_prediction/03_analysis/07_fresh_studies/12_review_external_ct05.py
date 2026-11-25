"""Read saved CT05/probe predictions; audit target-date versions without fitting."""
import argparse
import csv
from datetime import date
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
             'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '4'
ROOT = Path(__file__).resolve().parents[3]
for rel in ('01_tpg/01_core', '01_tpg/10_extensions', '02_prediction/01_package',
            '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT / rel))

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from tqdm import tqdm
from epss.training_only_pooled_v1 import data, report

SEEDS = (17, 29, 43)
# FIRST publishing dates, not model-name dates or whole-91-day-window categories.
RELEASES = (('2021-04-14', 'v1'), ('2022-02-04', 'v2'),
            ('2023-03-07', 'v3'), ('2025-03-17', 'v4'), ('2026-06-15', 'v5'))
FIRST_SOURCE = 'https://www.first.org/epss/data'
DEFAULT_PROBES = Path('external_probes')


def target_version(day):
    parsed = date.fromisoformat(day)
    return next((version for start, version in reversed(RELEASES)
                 if parsed >= date.fromisoformat(start)), 'before_EPSS')


def support(rows, role):
    result = []
    for version in sorted({target_version(r['date']) for r in rows}):
        group = [r for r in rows if target_version(r['date']) == version]
        y = np.array([r['y'] for r in group])
        result.append({'role': role, 'version_at_target_date': version, 'cves': len(group),
                       'first_target_date': min(r['date'] for r in group),
                       'last_target_date': max(r['date'] for r in group),
                       'minimum': float(y.min()), 'median': float(np.median(y)),
                       'mean': float(y.mean()), 'high_score_cves': int((y >= .1).sum()),
                       'at_0_00885': int((y == .00885).sum())})
    return result


def collect(dataset, probe_dir):
    parent = data.OUT / data.NAMES[dataset]
    root = parent / 'full'
    pins = {str(Path(__file__)): data.c.sha(__file__), **data.pins()}

    def pin(path):
        path = Path(path)
        pins[str(path)] = data.c.sha(path)
        return path

    def locked_file(path):
        path = Path(path)
        lock = next((p / 'artifact_lock.json' for p in (path.parent, path.parent.parent)
                     if (p / 'artifact_lock.json').is_file()), None)
        data.require(lock is not None, 'Missing input artifact lock: ' + str(path))
        expected = data.c.read(pin(lock))['files'][str(path.relative_to(lock.parent))]
        data.require(data.c.sha(pin(path)) == expected, 'Changed saved input: ' + str(path))
        return data.c.read(path)

    protocol = locked_file(parent / 'protocol/protocol.json')
    data.require(protocol['source_pins'] == data.pins() and not protocol['test_access'],
                 'CT05 protocol identity differs')
    data.publication.check_lock(root / 'report')
    pin(root / 'report/artifact_lock.json')
    official = data.c.read(pin(root / 'report/findings.json'))
    with pin(root / 'report/oof_predictions.csv').open(newline='') as stream:
        saved = list(csv.DictReader(stream))
    lookup = {(r['model'], int(r['seed']), r['cve']): r for r in saved}
    data.require(len(lookup) == len(saved), 'Duplicate official predictions')

    labels = locked_file(Path(protocol['label_source']))
    if isinstance(labels, list):
        records = {r['cve']: {'y': r['value'], 'date': r['target_date'], 'role': r['split']}
                   for r in labels}
        data.require(len(records) == len(labels), 'Duplicate labels')
    else:
        summary = locked_file(Path(protocol['export']) / 'summary.json')
        data.require(not summary['test_adapted'] and
                     summary['selected_cves'] == {'train': 24213, 'validation': 972},
                     'Not the frozen development-only export')
        records = {c: {'y': r['target'], 'date': r['target_date'],
                       'role': 'train' if c in protocol['rows'] else 'validation'}
                   for c, r in labels.items()}
    data.require({c for c, r in records.items() if r['role'] == 'train'} == set(protocol['rows']),
                 'Outer training identities differ')
    expected = {'train': 2546, 'validation': 1104} if dataset == 'previous' else {'train': 24213, 'validation': 972}
    data.require({role: sum(r['role'] == role for r in records.values()) for role in expected} == expected
                 and all(r['role'] in expected for r in records.values()), 'Unexpected outer roles')
    outer = [row for role in expected for row in support([r for r in records.values() if r['role'] == role], role)]
    fold_support, ridge, probe_replay = [], [], []
    arrays = {}
    for fold in tqdm(protocol['folds'], desc='Verify saved probe IDs and official predictions', unit='fold'):
        n = fold['fold']
        ids = locked_file(root / f'prepared/fold_{n}/ids.json')
        lab = locked_file(root / f'prepared/fold_{n}/labels.json')
        for role in ('optimization', 'selection', 'assessment'):
            data.require(ids[role] == fold['full'][role], 'Fold identities differ')
            data.require([records[c]['y'] for c in ids[role]] == lab[role], 'Fold labels differ')
            fold_support.extend({'fold': n, **row} for row in support([records[c] for c in ids[role]], role))
        states = locked_file(root / f'controls/fold_{n}/states.json')
        for arm in ('D', 'DM'):
            state = states[arm + '_ridge_selected']
            ridge.append({'fold': n, 'arm': arm, 'alpha': state['alpha'], 'selection_grid': state['selection_grid']})
        with np.load(pin(probe_dir / f'ct05_flag_{data.NAMES[dataset]}_fold{n}.npz'), allow_pickle=False) as z:
            data.require(z['ids'].tolist() == ids['assessment'] and np.array_equal(z['y'], lab['assessment']),
                         'External probe cohort or labels differ')
            data.require(np.array_equal(z['mflag'], [protocol['rows'][c]['source_mask']['M'] for c in ids['assessment']]),
                         'External post mask differs')
            for arm in ('D', 'DM', 'DF'):
                for seed in SEEDS:
                    pred = z[f'{arm}|{seed}']
                    data.require(pred.shape == z['y'].shape and np.isfinite(pred).all()
                                 and ((pred >= 0) & (pred <= 1)).all(), 'Bad probe predictions')
                    if arm != 'DF':
                        ref = np.array([float(lookup[arm, seed, c]['prediction']) for c in ids['assessment']])
                        error = float(np.max(np.abs(ref - pred)))
                        ev = locked_file(root / f'fits/fold_{n}/{arm}/seed_{seed}/evaluation.json')
                        data.require(error <= 1e-12 and int(z[f'{arm}|{seed}|epoch']) == ev['best_epoch'],
                                     'External D/DM replay or selected epoch differs')
                        probe_replay.append({'fold': n, 'arm': arm, 'seed': seed, 'max_abs_difference': error})
            for key in ('ids', 'y', 'mflag', *[f'{a}|{s}' for a in ('D', 'DM', 'DF') for s in SEEDS]):
                arrays.setdefault(key, []).append(z[key])
    z = {k: np.concatenate(v) for k, v in arrays.items()}
    order = np.argsort(z['ids'])
    z = {k: v[order] for k, v in z.items()}
    data.require(len(set(z['ids'])) == len(z['ids']), 'Repeated probe OOF CVE')
    metrics, intervals = [], []
    versions = np.array([target_version(records[c]['date']) for c in z['ids']])
    partitions = {'all': np.ones(len(z['y']), bool), 'M_present': z['mflag'] == 1, 'M_absent': z['mflag'] == 0,
                  **{v: versions == v for v in sorted(set(versions))}}
    for part, take in partitions.items():
        for arm in ('D', 'DM', 'DF'):
            e = np.stack([(z[f'{arm}|{s}'][take] - z['y'][take])**2 for s in SEEDS])
            metrics.append({'partition': part, 'model': arm, 'cves': int(take.sum()),
                            'rmse_mean': float(np.sqrt(e.mean(1)).mean()),
                            'rmse_by_seed': dict(zip(map(str, SEEDS), np.sqrt(e.mean(1)).tolist()))})
    for part in tqdm(('all', 'M_present', 'M_absent'), desc='Post-hoc DM-minus-flag paired intervals', unit='partition'):
        take = partitions[part]
        errors = {arm: np.stack([(z[f'{arm}|{s}'][take] - z['y'][take])**2 for s in SEEDS]) for arm in ('DM', 'DF')}
        ids = z['ids'][take].tolist()
        for unit in ('CVE', 'evidence_component'):
            groups = ids if unit == 'CVE' else [protocol['component_by_cve'][c] for c in ids]
            intervals.append({'partition': part, 'unit': unit,
                              **report.paired_ci(errors['DM'], errors['DF'], groups, 20000, 6029)})
    for name in ('ct05_flag_probe.py', 'era_probe.py'):
        pin(probe_dir / name)
    finding = {'dataset': dataset, 'release_schedule': RELEASES, 'release_source': FIRST_SOURCE,
               'version_assignment': 'Inferred from Day-60 target date and publishing schedule; not reverified against daily archive headers.',
               'outer_role_target_version_counts': outer, 'inner_fold_target_version_counts': fold_support,
               'official_primary_intervals': [r for r in official['paired_intervals'] if r['partition'] == 'all'],
               'external_probe_replay': probe_replay, 'external_probe_metrics': metrics,
               'external_probe_DM_minus_DF_intervals': intervals, 'ridge_selection': ridge,
               'training': False, 'new_predictions': False, 'test_access': False,
               'outer_validation_usage': 'Saved target dates and distribution audit only; no predictions or model selection.',
               'external_probe_scope': '27 probe fits per dataset: 18 D/DM repeats, 9 novel DF fits. Prediction files matched; no DF checkpoints supplied.',
               'limitations': [
                   'D/DM prediction and selected-epoch parity does not authenticate the missing DF checkpoints or training histories.',
                   'DF is an external post-hoc training experiment, not part of the frozen CT05 primary contrast.',
                   'New diagnostic intervals use 20,000 resamples, seed 6029; no seed search. They condition on saved models.',
                   'Version, CVE cohort, post availability and time change together. Association does not identify a cause.',
                   'The observed v2 minimum 0.00885 describes these saved targets, not a proven universal EPSS-v2 floor.',
                   'Ridge was tuned on its frozen grid; choosing its endpoint is not absence of tuning.',
                   'The era probe reuses full-optimization structured preprocessing and changes update counts when subsetting.',
                   'Different EPSS versions retain the probability-of-exploitation meaning; they do change the score generator.',
                   'No additional training, thresholds, split changes or held-out test analysis performed.']}
    return finding, pins


def text_report(f):
    lines = ['CT05 external-analysis verification: ' + f['dataset'],
             'This is a new read-only review. Original CT05 artifacts and primary conclusions are unchanged.',
             'D: description plus structured features. DM: description and posts plus structured features.',
             'DF: description plus post-presence flags and structured features, without post text.',
             '', '1. Day-60 target version (not the version range across the full 91-day history)',
             f['version_assignment'], 'Role | version | CVEs | minimum target | median target']
    lines += [f"{r['role']} | {r['version_at_target_date']} | {r['cves']} | {r['minimum']:.6f} | {r['median']:.6f}"
              for r in f['outer_role_target_version_counts']]
    lines += ['', '2. External control RMSE, lower is better; average of three seed metrics',
              'Partition | model | CVEs | RMSE']
    lines += [f"{r['partition']} | {r['model']} | {r['cves']} | {r['rmse_mean']:.8f}" for r in f['external_probe_metrics']]
    lines += ['', '3. Exploratory DM minus DF RMSE; below zero favors the post-text arm',
              '20,000 resamples, seed 6029. Neither interval includes model-refitting uncertainty.']
    lines += [f"{r['partition']} | {r['unit']} | {r['difference']:+.8f} [{r['low']:+.8f}, {r['high']:+.8f}]"
              for r in f['external_probe_DM_minus_DF_intervals']]
    lines += ['', '4. Qualifications', f['external_probe_scope'], *f['limitations'], '',
              'Decision: retain the CT05 primary result as not established on either dataset.',
              'Post-bearing Reddit CVEs show exploratory benefit beyond a presence flag, not proof of causality or graph superiority.',
              'Do not expand TPG training or reopen test evaluation yet. Specify version-aware interpretation before further fits.',
              'A same-version restriction is one design option, not the only valid scientific task.',
              'Keep retrospective single-score estimation separate from leakage-safe temporal forecasting.']
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('previous', 'reddit', 'both'), default='both')
    p.add_argument('--probe-dir', type=Path, default=DEFAULT_PROBES)
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4, user_api='blas'):
        for dataset in ('previous', 'reddit') if a.dataset == 'both' else (a.dataset,):
            f, pins = collect(dataset, a.probe_dir)
            output = data.OUT / data.NAMES[dataset] / 'external_review_v1'
            with data.publication.transaction(output, verify=a.verify_only) as temp:
                data.base.write(temp / 'findings.json', f)
                data.base.write(temp / 'identity.json', {'input_and_code_hashes': pins, 'runtime': data.runtime()})
                (temp / 'REPORT.txt').write_text(text_report(f))
                for key in ('outer_role_target_version_counts', 'inner_fold_target_version_counts',
                            'external_probe_metrics', 'external_probe_DM_minus_DF_intervals'):
                    (temp / (key + '.csv')).write_text(data.reporting.csv_text(f[key]))
                data.publication.check_pins(pins)
            print(f'CT05 {dataset} external review {"reproduced" if a.verify_only else "saved"}: {output}. No training or test scoring.')


if __name__ == '__main__':
    main()
