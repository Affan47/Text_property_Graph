"""Verify the FC01 loader-only correction against preserved inputs/checkpoints.

This replays existing selection predictions, never fits a model or scores an
assessment/test cohort. Observational timing is separate from deterministic locks.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '4'
import argparse
import gc
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
for rel in ('01_tpg/01_core', '01_tpg/10_extensions', '02_prediction/01_package', '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT / rel))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from tqdm import tqdm
from epss.final_comparison_v1 import data, fitting, protocol
from epss.final_comparison_v1.common import OUT, ROOT, read, write, sha, require, publication
from epss.final_comparison_v1.store import ResidentStore, predictor_digest

PRESERVED = OUT / 'preserved/pre_npz_loading_fix_20261116'
REPORT = ROOT / '05_results/08_fresh_studies/29_fc01_loader_fix_v1/review'


def historical_pin(path):
    path = Path(path)
    relative = path.relative_to(ROOT)
    if path.parent == ROOT / '02_prediction/01_package/epss/final_comparison_v1':
        return PRESERVED / 'code/final_comparison_v1' / path.name
    if path.name.startswith('test_final_comparison') or path.name in (
            '32_run_final_comparison.py', '11_final_comparison_v1.json'):
        return PRESERVED / 'code' / path.name
    if path.is_relative_to(OUT):
        suffix = path.relative_to(OUT)
        if suffix.parts[0] in ('prepared', 'protocol', 'integration'):
            return PRESERVED / suffix
    return ROOT / relative


def review(verify=False):
    p = protocol.load()
    old = read(PRESERVED / 'protocol/protocol.json')
    publication.check_lock(PRESERVED / 'protocol')
    publication.check_lock(PRESERVED / 'prepared/run1')
    old_pins = {str(historical_pin(path)): value for path, value in old['code_pins'].items()}
    publication.check_pins(old_pins)
    require({k: v for k, v in p.items() if k != 'code_pins'} ==
            {k: v for k, v in old.items() if k != 'code_pins'}, 'Non-code protocol field changed')
    changed = sorted(k for k in old['code_pins'] if p['code_pins'].get(k) != old['code_pins'][k])
    added = sorted(set(p['code_pins']) - set(old['code_pins']))
    require(changed == [str(Path(data.__file__))] and
            [Path(k).name for k in added] == ['test_final_comparison_loader.py'], 'Unexpected code revision')
    # The earlier startup snapshot stays immutable, with old paths resolved to its preserved predecessor.
    snapshot = ROOT / '05_results/08_fresh_studies/28_fc01_startup_review_v1/review'
    publication.check_lock(snapshot)
    for path, value in read(snapshot / 'findings.json')['input_pins'].items():
        require(sha(historical_pin(path)) == value, 'Startup snapshot source changed: ' + path)
    old_files = publication.check_lock(PRESERVED / 'prepared/run1')
    new_files = publication.check_lock(OUT / 'prepared/run1')
    require(set(old_files) == set(new_files), 'Prepared file set changed')
    require([k for k in old_files if old_files[k] != new_files[k]] == ['identity.json'],
            'Prepared predictor/label/role/graph bytes changed')
    before = read(PRESERVED / 'prepared/run1/identity.json')
    after = read(OUT / 'prepared/run1/identity.json')
    require({k: v for k, v in before.items() if k != 'protocol_sha256'} ==
            {k: v for k, v in after.items() if k != 'protocol_sha256'}, 'Preparation lineage changed')
    timings, folds, replays = [], [], []
    for fold in tqdm(sorted(p['folds']), desc='Verify unchanged full-fold predictor rows', unit='fold'):
        started = time.perf_counter()
        ctx = data.load_context(p, 'run1', fold)
        timings.append({'fold': int(fold), 'load_context_seconds': time.perf_counter() - started})
        path = OUT / f'prepared/run1/fold_{fold}/predictors.npz'
        with np.load(path, allow_pickle=False) as arrays:
            for role, cs in ctx['ids'].items():
                for field in ('structured', 'D', 'DM'):
                    rows = ctx['vectors'] if field == 'structured' else ctx['pooled'][field]
                    expected = arrays[role + '_' + field]
                    actual = np.asarray([rows[cid] for cid in cs], dtype=expected.dtype).reshape(expected.shape)
                    require(actual.tobytes() == expected.tobytes(), 'Predictor byte/row parity failed')
        folds.append({'fold': int(fold), 'cves': sum(map(len, ctx['ids'].values())),
                      'arrays_decoded': 9, 'predictor_archive_sha256': sha(path), 'exact_row_bytes': True})
        if fold == '1':
            roles = {**ctx['ids'], 'assessment': []}
            ctx = {**ctx, 'ids': roles}
            for lock in tqdm(sorted((PRESERVED / 'integration/fits').rglob('artifact_lock.json')),
                             desc='Replay preserved integration checkpoints; no fitting', unit='model'):
                folder = lock.parent
                publication.check_lock(folder)
                identity = read(folder / 'run_identity.json')
                fit = read(folder / 'fit_contract.json')
                model, seed = fit['model'], fit['seed']
                require(fit['roles'] == roles and fit['settings']['epochs'] == 1, 'Not a one-epoch integration fit')
                ids = roles['train'] + roles['validation']
                require(predictor_digest(ctx, model, ids) == fit['development_predictors_sha256'],
                        'Frozen fit predictor digest changed')
                for source, value in identity['source_pins'].items():
                    require(sha(historical_pin(source)) == value, 'Historical source pin changed: ' + source)
                with fitting.cpu_runtime():
                    frozen = fitting._fit_contract(ctx, identity, model, seed, fit['settings'], roles)
                    require(frozen == fit, 'Historical fit contract no longer reproduces')
                    net, checkpoint = fitting._checkpoint(folder / 'best_checkpoint.pt', identity, fit, model, seed)
                    fitting._history_check(read(folder / 'history.json'), checkpoint, fit['settings'], complete=True)
                    store = ResidentStore(ctx, model, budget_gib=8)
                    store.preload(roles['validation'], 'preserved selection replay')
                    actual = fitting.evaluate(net, store, ctx, roles['validation'], 'preserved selection')
                    expected = read(folder / 'evaluation.json')
                    require(actual['predictions'] == expected['predictions']['validation'] and
                            actual['metrics'] == expected['metrics']['validation'], 'Checkpoint replay differs')
                    csv_text = fitting._csv({'validation': actual['predictions'], 'assessment': []})
                    require(csv_text.encode() == (folder / 'predictions.csv').read_bytes(), 'Prediction CSV changed')
                    store.clear()
                    del net, store, checkpoint
                publication.check_lock(folder)
                replays.append({'model': model, 'seed': seed, 'predictions': len(actual['predictions']),
                                'checkpoint_sha256': sha(folder / 'best_checkpoint.pt'),
                                'prediction_csv_sha256': sha(folder / 'predictions.csv'),
                                'fit_lock_sha256': sha(lock), 'exact_replay': True})
        del ctx
        gc.collect()
    require(len(replays) == 4, 'Expected four preserved completed integration fits')
    evidence = [Path(__file__), OUT / 'protocol/artifact_lock.json', OUT / 'protocol/protocol.json',
                OUT / 'prepared/run1/artifact_lock.json', PRESERVED / 'protocol/artifact_lock.json',
                PRESERVED / 'prepared/run1/artifact_lock.json', snapshot / 'artifact_lock.json']
    evidence += sorted((PRESERVED / 'integration/fits').rglob('artifact_lock.json'))
    findings = {'schema': 'fc01_npz_loader_correction_v1', 'status': 'VERIFIED_LOADER_ONLY_CORRECTION',
                'old_protocol_sha256': sha(PRESERVED / 'protocol/protocol.json'),
                'new_protocol_sha256': sha(OUT / 'protocol/protocol.json'),
                'old_loader_sha256': sha(PRESERVED / 'code/final_comparison_v1/data.py'),
                'new_loader_sha256': sha(Path(data.__file__)),
                'scientific_settings_unchanged': True, 'prepared_payloads_unchanged': True,
                'folds': folds, 'preserved_checkpoint_replays': replays,
                'gate_status': 'FIVE_FIT_INTEGRATION_RESTART_REQUIRED_UNDER_NEW_CODE_IDENTITY',
                'training_performed': False, 'assessment_scoring': False, 'test_access': False,
                'input_pins': {str(path): sha(path) for path in evidence}}
    with publication.transaction(REPORT, verify=verify) as tmp:
        write(tmp / 'findings.json', findings)
        (tmp / 'REPORT.txt').write_text(
            'FC01 loader-only correction\n'
            'Authorized integration process stopped; all four completed checkpoints preserved.\n'
            'Each of nine predictor arrays is now decoded once, not once per CVE.\n'
            'All three folds have exact row-byte parity; prepared payloads and scientific settings are unchanged.\n'
            'Four preserved checkpoints reproduce every selection prediction and canonical CSV exactly.\n'
            'Old code, locks and preparation are retained under preserved/pre_npz_loading_fix_20261116.\n'
            'The new protocol changes code identity only. Restart the five-row one-epoch integration gate.\n'
            'No scientific fit, parameter update, assessment scoring or test scoring was performed.\n')
        publication.check_pins(findings['input_pins'])
        publication.check_pins(p['code_pins'])
    if not verify:
        write(REPORT.parent / 'timing_observations.json', {'folds': timings, 'not_a_training_ETA': True})
    print('FC01 loader correction ' + ('reproduced' if verify else 'verified and saved') + ': ' + str(REPORT))
    print(timings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4, user_api='blas'):
        review(args.verify_only)
