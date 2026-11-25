"""Replay external DF checkpoints and audit within-version study support; never fit."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys

for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
            'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
ROOT = Path(__file__).resolve().parents[3]
for rel in ('01_tpg/01_core', '01_tpg/10_extensions', '02_prediction/01_package',
            '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT / rel))

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from tqdm import tqdm
from epss.training_only_pooled_v1 import data, model

REVIEW_PATH = Path(__file__).with_name('12_review_external_ct05.py')
spec = importlib.util.spec_from_file_location('ct05_external_review', REVIEW_PATH)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
SEEDS = (17, 29, 43)


def check_history(history, selected, max_epochs=100, patience=15):
    data.require(0 < len(history) <= max_epochs, 'Empty/overlong history')
    best, epoch_best, stale = float('inf'), None, 0
    for i, row in enumerate(history, 1):
        data.require(row['epoch'] == i and np.isfinite(row['optimization_mse'])
                     and row['optimization_mse'] >= 0 and np.isfinite(row['selection_rmse'])
                     and row['selection_rmse'] >= 0, 'Malformed history')
        if row['selection_rmse'] < best:
            best, epoch_best, stale = row['selection_rmse'], i, 0
        else:
            stale += 1
        data.require(row['patience'] == stale and (stale < patience or i == len(history)),
                     'History violates strict-improvement/early-stop rule')
    data.require(selected == epoch_best and (stale == patience or len(history) == max_epochs),
                 'Wrong selected checkpoint or incomplete history')
    return best


def collect(dataset, probe_dir):
    protocol = data.load_protocol(dataset)
    parent = data.OUT / data.NAMES[dataset]
    prior = parent / 'external_review_v1'
    data.publication.check_lock(prior)
    prior_identity = data.c.read(prior / 'identity.json')
    data.publication.check_pins(prior_identity['input_and_code_hashes'])
    pins = {**data.pins(), **prior_identity['input_and_code_hashes']}
    for p in (Path(__file__), REVIEW_PATH, prior / 'artifact_lock.json',
              parent / 'protocol/artifact_lock.json', probe_dir / 'ct05_flag_provenance.py'):
        pins[str(p)] = data.c.sha(p)
    replay, labels, arrays = [], {}, {}
    for fold in tqdm(protocol['folds'], desc='Replay frozen DF selection and assessment', unit='fold'):
        n = fold['fold']; prepared = data.load_fold(dataset, 'full', n)
        pins[str(parent / 'full/prepared/artifact_lock.json')] = prepared['prepared_sha256']
        ids, a, y = (prepared[k] for k in ('ids', 'arrays', 'labels'))
        for role in ids:
            for c, value in zip(ids[role], y[role]):
                data.require(c not in labels or labels[c] == value, 'Inconsistent repeated label')
                labels[c] = value
        x = {}
        for role in ('selection', 'assessment'):
            x[role] = {'text': torch.from_numpy(a[role + '_D']),
                       'structured': torch.from_numpy(a[role + '_structured']),
                       'source_mask': torch.from_numpy(a[role + '_DM_mask'])}
        path = probe_dir / f'ct05_flag_checkpoints_{data.NAMES[dataset]}_fold{n}.pt'
        pins[str(path)] = data.c.sha(path)
        checkpoints = torch.load(path, map_location='cpu', weights_only=True)
        data.require(set(checkpoints) == set(SEEDS), 'Unexpected checkpoint seeds')
        probe = probe_dir / f'ct05_flag_{data.NAMES[dataset]}_fold{n}.npz'
        pins[str(probe)] = data.c.sha(probe)
        with np.load(probe, allow_pickle=False) as saved:
            data.require(saved['ids'].tolist() == ids['assessment']
                         and np.array_equal(saved['y'], y['assessment']), 'Probe cohort/labels changed')
            for seed in SEEDS:
                cp = checkpoints[seed]
                selection_expected = check_history(cp['history'], cp['epoch'])
                net = model.Regressor(a['assessment_structured'].shape[1], protocol['settings']['hidden'])
                net.load_state_dict(cp['model'], strict=True)
                sel = model.predict(net, x['selection'])
                pred = model.predict(net, x['assessment'])
                selection_actual = float(np.sqrt(np.mean((sel - np.array(y['selection']))**2)))
                data.require(selection_actual == selection_expected, 'Selection score does not replay')
                data.require(np.array_equal(pred, saved[f'DF|{seed}'])
                             and cp['epoch'] == int(saved[f'DF|{seed}|epoch']),
                             'Checkpoint predictions/epoch differ from first probe')
                data.require(all(torch.equal(value, cp['model'][key]) for key, value in net.state_dict().items()),
                             'Replay mutated model state')
                replay.append({'fold': n, 'seed': seed, 'best_epoch': cp['epoch'],
                               'history_epochs': len(cp['history']), 'selection_rmse': selection_actual,
                               'assessment_rmse': float(np.sqrt(np.mean((pred - saved['y'])**2))),
                               'max_prediction_difference': float(np.max(np.abs(pred - saved[f'DF|{seed}']))),
                               'checkpoint_sha256': pins[str(path)]})
            for key in ('ids', 'y', 'mflag', *[f'{arm}|{s}' for arm in ('D', 'DM', 'DF') for s in SEEDS]):
                arrays.setdefault(key, []).append(saved[key])
    # Read only already-exported labels for the original training IDs; no test input is opened.
    raw = data.c.read(protocol['label_source'])
    lookup = {r['cve']: r['value'] for r in raw} if isinstance(raw, list) else {c: r['target'] for c, r in raw.items()}
    support = []
    for version in sorted({review.target_version(r['anchor_date']) for r in protocol['rows'].values()}):
        ids = sorted(c for c, r in protocol['rows'].items() if review.target_version(r['anchor_date']) == version)
        present = [c for c in ids if protocol['rows'][c]['source_mask']['M']]
        support.append({'target_version': version, 'training_pool_cves': len(ids),
                        'scores_ge_0_1': sum(lookup[c] >= .1 for c in ids),
                        'with_posts': len(present), 'with_posts_and_score_ge_0_1': sum(lookup[c] >= .1 for c in present)})
    z = {k: np.concatenate(v) for k, v in arrays.items()}
    order = np.argsort(z['ids']); z = {k: v[order] for k, v in z.items()}
    data.require(len(set(z['ids'])) == len(z['ids']), 'Repeated assessment CVE')
    versions = np.array([review.target_version(protocol['rows'][c]['anchor_date']) for c in z['ids']])
    present = z['mflag'] == 1
    error = {arm: np.stack([(z[f'{arm}|{s}'] - z['y'])**2 for s in SEEDS]) for arm in ('D', 'DM', 'DF')}
    savings = (error['DF'] - error['DM']).mean(0)
    total = float(savings[present].sum())
    metrics = []
    for v in sorted(set(versions)):
        take = present & (versions == v)
        if not take.any():
            continue
        rmse = {arm: np.sqrt(e[:, take].mean(1)) for arm, e in error.items()}
        gain = float(savings[take].sum())
        metrics.append({'target_version': v, 'post_bearing_oof_cves': int(take.sum()),
                        'scores_ge_0_1': int((z['y'][take] >= .1).sum()),
                        **{arm + '_rmse': float(vals.mean()) for arm, vals in rmse.items()},
                        'DM_minus_DF_by_seed': dict(zip(map(str, SEEDS), (rmse['DM'] - rmse['DF']).tolist())),
                        'mean_seed_squared_error_saving_sum': gain,
                        'share_of_post_bearing_net_squared_error_saving_percent': 100 * gain / total if total else None})
    result = {'dataset': dataset, 'checkpoint_replays': replay, 'original_training_pool_support': support,
              'post_bearing_version_metrics': metrics,
              'net_post_bearing_squared_error_saving': total,
              'status': 'Nine external DF checkpoints per dataset replayed; no retraining.',
              'limitations': [
                  'Selected weights, selection-score replay and full logged early-stop histories are checked; gradients are not re-executed.',
                  'These externally trained arms remain exploratory and do not replace CT05 or its preregistered primary contrast.',
                  'Squared-error contributions are additive; RMSE differences are not. Net shares may exceed 100 percent.',
                  'Training-pool support is not the future assessment sample size: warmup, selection and component exclusions reduce it.',
                  'Within-version designs remove a release transition, not cohort/time shifts or retrospective feature availability.',
                  'A design chosen after these results is a new exploratory study, not independent confirmation.',
                  'The official train/validation/test identities and every completed fit remain unchanged.'],
              'training': False, 'new_experiment': False, 'test_access': False}
    return result, pins


def text_report(f):
    lines = ['CT05 flag-checkpoint provenance and within-version support: ' + f['dataset'],
             'D = description. DF = description and post-presence flag. DM = description and posts.',
             'All three arms also use the same structured features. No GNN is tested here.',
             f['status'], '', '1. Original training pool, before any new split or evidence-group purge',
             'EPSS target version | CVEs | score >=0.1 | with posts | with posts and score >=0.1']
    for r in f['original_training_pool_support']:
        lines.append(' | '.join(str(r[k]) for k in ('target_version', 'training_pool_cves', 'scores_ge_0_1',
                                                   'with_posts', 'with_posts_and_score_ge_0_1')))
    lines += ['', '2. Existing CT05 assessment CVEs with posts, separated by Day-60 target version',
              'Version | CVEs | D RMSE | DF RMSE | DM RMSE | share of net squared-error saving (%)']
    for r in f['post_bearing_version_metrics']:
        lines.append(f"{r['target_version']} | {r['post_bearing_oof_cves']} | {r['D_rmse']:.8f} | "
                     f"{r['DF_rmse']:.8f} | {r['DM_rmse']:.8f} | "
                     f"{r['share_of_post_bearing_net_squared_error_saving_percent']:.3f}")
    lines += ['', '3. Interpretation and boundaries', *f['limitations'], '',
              'Recommended next design: Reddit-v2 within-version pooled D/DF/DM, older-v3 secondary.',
              'Freeze date/group rules and support gates before evaluating new folds; never move cutoffs for a favorable outcome.',
              'Refit structured transforms on each new optimization role; reuse only genuinely frozen encoder vectors.',
              'Keep CT05 as mixed-version chronological evidence; retain official outer roles for a separately specified transfer evaluation.',
              'Use RMSE primary; report MAE, ranking, tails, post coverage and evidence-component intervals.',
              'Keep TPG training and test evaluation closed until that bounded comparison is interpreted.']
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('previous', 'reddit', 'both'), default='both')
    p.add_argument('--probe-dir', type=Path, default=review.DEFAULT_PROBES)
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=4, user_api='blas'):
        for dataset in ('previous', 'reddit') if a.dataset == 'both' else (a.dataset,):
            f, pins = collect(dataset, a.probe_dir)
            output = data.OUT / data.NAMES[dataset] / 'flag_checkpoint_review_v1'
            with data.publication.transaction(output, verify=a.verify_only) as temp:
                data.base.write(temp / 'findings.json', f)
                data.base.write(temp / 'identity.json', {'input_and_code_hashes': pins, 'runtime': data.runtime()})
                (temp / 'REPORT.txt').write_text(text_report(f))
                for key in ('checkpoint_replays', 'original_training_pool_support', 'post_bearing_version_metrics'):
                    (temp / (key + '.csv')).write_text(data.reporting.csv_text(f[key]))
                data.publication.check_pins(pins)
            print(f'CT05 {dataset} flag-checkpoint review {"reproduced" if a.verify_only else "saved"}: {output}. No training or test scoring.')


if __name__ == '__main__':
    main()
