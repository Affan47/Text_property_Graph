"""R05: bounded matched Adam weight-decay comparison, frozen graph cache."""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import statistics

from . import forecast_coreference_residual_development as prior

torch, task, residual = prior.torch, prior.task, prior.residual
read, write, sha256, require = prior.read_json, prior.write_json, prior.sha256, prior.require
tqdm = prior.cache.tqdm
VERSION = 'r05_residual_decay_comparison_v1'
ROOT = prior.cache.base.EPSS_ROOT
PROTOCOL = ROOT/'02_prediction/04_configs/residual_decay_comparison_v1.json'
SCRIPT = ROOT/'02_prediction/03_analysis/01_revision_dataset/run_forecast_decay_comparison.py'
TEST = SCRIPT.with_name('test_forecast_decay_comparison.py')
OUTPUT = task.RESULT_ROOT/'residual_decay_comparison_v1'
SOURCE_LOCK = 'a7309b871ec471d481584597917509c76df7ca2858b09e8c08c6e221c026591b'
DQ19_LOCK = 'f0a622f9ecc5e8edccd25ffe79d419103563c316501c88da691612346e6ed969'
MODELS = ('epss_only', 'no_coref')
CONDITIONS = {'original_decay': .0001, 'zero_decay': 0.0}
FIT_FILES = {'checkpoint.pt', 'history.json', 'gradients.json', 'predictions.json', 'metrics.json'}


def protocol():
    p = read(PROTOCOL)
    require(p['schema'] == VERSION and p['models'] == list(MODELS) and
            p['conditions'] == CONDITIONS and p['cohort']['train'] == 128 and
            p['cohort']['validation'] == 64, 'Fixed comparison protocol differs')
    original = read(prior.PROTOCOL)['config']
    expected = {**original, 'epochs': 30, 'patience': 30, 'correction_gain': .00885}
    require(p['config'] == expected, 'Fixed training configuration differs')
    return p


def select_cohort(ids, splits, sizes):
    require(set(ids) == set(sizes) == {'train', 'validation'}, 'Only development splits allowed')
    result = {}
    for split, n in sizes.items():
        require(len(set(ids[split])) == len(ids[split]) and len(ids[split]) >= n > 0,
                'Invalid source cohort')
        require(all(splits[c]['split'] == split for c in ids[split]), 'Changed split role')
        result[split] = sorted(ids[split], key=lambda c: hashlib.sha256(
            ('r05:subset:' + split + ':' + c).encode()).digest())[:n]
    require(not set(result['train']).intersection(result['validation']), 'Overlapping IDs')
    groups = {s: {splits[c]['group_id'] for c in values} for s, values in result.items()}
    require(not groups['train'].intersection(groups['validation']), 'Related group crosses splits')
    return result


def verify_hashes(pins, description):
    for name, digest in tqdm(sorted(pins.items()), desc=description, unit='file'):
        require(sha256(Path(name)) == digest, 'Changed source/artifact: ' + name)


def context():
    p = protocol()
    source = prior.DEFAULT_OUTPUT
    require(sha256(source/'evaluation_lock.json') == SOURCE_LOCK, 'Wrong R04 source')
    lock = read(source/'evaluation_lock.json')
    source_pins = {str(source/n): h for n, h in lock['files'].items()}
    verify_hashes(source_pins, 'Verify frozen R04 artifacts')
    saved = read(source/'run_identity.json')
    require(saved['torch_version'] == str(torch.__version__) and
            saved['config']['correction_gain'] == p['config']['correction_gain'], 'Environment/gain differs')
    prepared = prior.cache.base.PREPARED
    splits = read(prepared/'split_manifest.json')
    ids = select_cohort(saved['ids'], splits, {s: p['cohort'][s] for s in ('train', 'validation')})
    directory = Path(saved['cache_directory'])
    require(sha256(directory/'cache_lock.json') == saved['cache_lock_sha256'], 'Changed cache lock')
    cache_lock = read(directory/'cache_lock.json')
    pins = {**saved['pinned_files'], **source_pins,
            str(source/'evaluation_lock.json'): SOURCE_LOCK,
            str(directory/'cache_lock.json'): saved['cache_lock_sha256'],
            str(directory/'manifest.json'): cache_lock['files']['manifest.json']}
    dq = task.RESULT_ROOT/'coreference_residual_gradient_audit_v1'
    require(sha256(dq/'diagnostic_lock.json') == DQ19_LOCK, 'Wrong DQ19 prerequisite')
    pins[str(dq/'diagnostic_lock.json')] = DQ19_LOCK
    pins.update({str(dq/n): h for n, h in read(dq/'diagnostic_lock.json')['files'].items()})
    for c in ids['train'] + ids['validation']:
        name = f'graphs/{c}.pt'
        pins[str(directory/name)] = cache_lock['files'][name]
    for path in (Path(__file__).resolve(), SCRIPT, TEST, PROTOCOL):
        pins[str(path)] = sha256(path)
    verify_hashes(pins, 'Verify sources and subset graphs')
    manifest = read(directory/'manifest.json')
    require(manifest['ids'] == saved['ids'], 'Cache/source split differs')
    manifest = {**manifest, 'ids': ids}
    names = ('inputs.json', 'labels.json', 'split_manifest.json', 'task_cohorts.json')
    data = prior.smoke.select_data(manifest, {n: read(prepared/n) for n in names})
    bank = prior.GraphBank(directory, ids['train'] + ids['validation'], capacity=64)
    views = {v: bank.view('no_coref') for v in MODELS}
    identity = {'schema': VERSION, 'protocol': p, 'ids': ids, 'pinned_files': pins,
                'source_r04_lock_sha256': SOURCE_LOCK, 'dq19_lock_sha256': DQ19_LOCK,
                'cache_lock_sha256': saved['cache_lock_sha256'], 'torch_version': str(torch.__version__),
                'test_performance_evaluated': False, 'new_embeddings': False}
    return manifest, views, data, identity


def settings(p, seed, condition):
    require(seed in p['config']['seeds'] and condition in CONDITIONS, 'Unknown seed/condition')
    return {**deepcopy(p['config']), 'seeds': [seed], 'weight_decay': CONDITIONS[condition]}


def initialize(variant, manifest, graph, cfg):
    return residual.initialize('epss_only' if variant == 'epss_only' else 'tpg_epss', manifest, graph, cfg)


def parameter_norms(state):
    grouped = {}
    for name, v in state.items():
        key = name if name.startswith('head.') else name.split('.')[0]
        grouped.setdefault(key, []).append(v.reshape(-1).double())
    return {k: float(torch.cat(v).norm()) for k, v in grouped.items()}


def scalar_control(train_ids, inputs, labels):
    require(bool(train_ids) and len(set(train_ids)) == len(train_ids), 'Invalid scalar fit IDs')
    residuals = [labels[c]['y']['endpoint'] - task.predict_persistence(inputs[c])['prediction']
                 for c in train_ids]
    return {'fit_ids': list(train_ids), 'offset': statistics.median(residuals),
            'rule': 'median(training target minus persistence), then clip forecasts to [0,1]',
            'objective': 'unclipped training residual MAE; separate from neural Smooth L1'}


def baselines(manifest, data):
    fit = scalar_control(manifest['ids']['train'], data['inputs.json'], data['labels.json'])
    ids = manifest['ids']['validation']
    baseline = [task.predict_persistence(data['inputs.json'][c])['prediction'] for c in ids]
    corrected = [min(1.0, max(0.0, x + fit['offset'])) for x in baseline]
    return {'scalar_fit': fit, 'persistence': {'predictions': baseline,
            **prior.score_predictions(baseline, manifest, data)},
            'scalar_correction': {'predictions': corrected, **prior.score_predictions(corrected, manifest, data)}}


def fit_one(variant, condition, seed, manifest, graph, data, p, progress=True):
    cfg = settings(p, seed, condition)
    initial = initialize(variant, manifest, graph, cfg)
    orders = prior.epoch_orders(manifest['ids']['train'], cfg)
    # Reuse the frozen fixed-epoch R04 algorithm, not its early-stopping full runner.
    saved, history, gradients = residual.fit(variant, initial, orders, manifest, graph, data, cfg, progress)
    saved['matched_experiment'] = VERSION
    saved['decay_condition'] = condition
    saved['weight_decay'] = cfg['weight_decay']
    return saved, history, gradients


def assess(saved, initial, manifest, graph, data):
    values = residual.infer_details(residual.restore(saved), manifest['ids']['validation'],
                                    graph, data['inputs.json'])
    result = prior.score_predictions(values['prediction'], manifest, data)
    result.update(selected_epoch=saved['selected_epoch'], clipping=residual.clipping(values),
                  correction_mean=statistics.mean(values['correction']),
                  correction_std=statistics.pstdev(values['correction']),
                  correction_min=min(values['correction']), correction_max=max(values['correction']),
                  initial_parameter_norms=parameter_norms(initial['state_dict']),
                  selected_parameter_norms=parameter_norms(saved['state_dict']))
    return values, result


def check_fit(directory, variant, condition, seed, manifest, graph, data, p):
    lock = read(directory/'fit_lock.json')
    require(set(lock['files']) == FIT_FILES, 'Incomplete fit lock')
    verify_hashes({str(directory/n): h for n, h in lock['files'].items()}, 'Verify completed fit')
    cfg = settings(p, seed, condition)
    initial = initialize(variant, manifest, graph, cfg)
    saved = torch.load(directory/'checkpoint.pt', weights_only=True, map_location='cpu')
    require(saved['matched_experiment'] == VERSION and saved['decay_condition'] == condition and
            saved['seed'] == seed and saved['edge_configuration'] == variant and
            saved['weight_decay'] == cfg['weight_decay'] and saved['spec'] == initial['spec'] and
            saved['initial_state_sha256'] == initial['initial_state_sha256'] and
            saved['fit_cve_ids'] == manifest['ids']['train'] and
            saved['selection_cve_ids'] == manifest['ids']['validation'], 'Wrong checkpoint identity')
    history = read(directory/'history.json')
    require(len(history) == cfg['epochs'] and
            [r['epoch'] for r in history] == list(range(1, cfg['epochs'] + 1)) and
            saved['selected_epoch'] == min(history, key=lambda r: r['validation_mae'])['epoch'],
            'Wrong budget or checkpoint selection')
    require(all(r['training_examples'] == len(manifest['ids']['train']) for r in history),
            'Wrong training population')
    values, metrics = assess(saved, initial, manifest, graph, data)
    require(values == read(directory/'predictions.json') and metrics == read(directory/'metrics.json'),
            'Saved predictions/metrics differ')
    return metrics


def fit_name(variant, condition, seed):
    return f'{variant}__{condition}__{seed}'


def preflight(manifest, views, data, p):
    for seed in tqdm(p['config']['seeds'], desc='Preflight paired states/orders', unit='seed'):
        orders = [prior.epoch_orders(manifest['ids']['train'], settings(p, seed, c)) for c in CONDITIONS]
        require(orders[0] == orders[1], 'Unmatched training order')
        for variant in MODELS:
            states = [initialize(variant, manifest, views[variant], settings(p, seed, c)) for c in CONDITIONS]
            require(states[0]['initial_state_sha256'] == states[1]['initial_state_sha256'], 'Unmatched weights')
            with torch.random.fork_rng(devices=[]):
                net = residual.ResidualTPGRegressor(**states[0]['spec'])
            net.load_state_dict(states[0]['state_dict'])
            net.eval()
            with torch.no_grad():
                ids = manifest['ids']['train'][:p['config']['batch_size']]
                batch = task.make_batch(ids, views[variant], data['inputs.json'])
                require(not {'y', 'target_date', 'cve'}.intersection(batch.keys()), 'Target in input')
                base, delta, _, prediction = net.details(batch)
                require(torch.equal(base, prediction) and bool((delta == 0).all()), 'Not persistence initialization')


def aggregate(results, seeds):
    paired, summaries = [], []
    for variant in MODELS:
        for seed in seeds:
            a = results[fit_name(variant, 'original_decay', seed)]['metrics']
            b = results[fit_name(variant, 'zero_decay', seed)]['metrics']
            paired.append({'model': variant, 'seed': seed,
                           'zero_minus_original_mae': b['mae'] - a['mae'],
                           'zero_minus_original_rmse': b['rmse'] - a['rmse']})
        for condition in CONDITIONS:
            values = [results[fit_name(variant, condition, s)]['metrics']['mae'] for s in seeds]
            summaries.append({'model': variant, 'condition': condition,
                              'mean_mae': statistics.mean(values),
                              'sample_std_mae': statistics.stdev(values) if len(values) > 1 else None})
    return paired, summaries


def report(summary):
    lines = ['R05: Controlled weight-decay comparison', '',
             'Bounded development subsets only. No test scoring or new embeddings.',
             'Original and zero decay use identical CVEs, initial weights and epoch orders.',
             'All 30 epochs run; the earliest best validation checkpoint is selected.',
             'Primary comparison: zero minus original validation MAE. Negative favors zero decay.',
             'Do not compare this subset MAE directly with full R04 cohort MAE.', '',
             'Baselines (same validation CVEs):']
    for name in ('persistence', 'scalar_correction'):
        lines.append(f"{name}: MAE={summary['baselines'][name]['metrics']['mae']:.10g}")
    lines += ['', 'Fit (model__condition__seed) | Selected epoch | Validation MAE | Correction SD']
    for name, row in sorted(summary['fits'].items()):
        lines.append(f"{name} | {row['selected_epoch']} | {row['metrics']['mae']:.10g} | {row['correction_std']:.10g}")
    lines += ['', 'Paired comparison (zero minus original):']
    for row in summary['paired']:
        lines.append(f"{row['model']}, seed {row['seed']}: MAE {row['zero_minus_original_mae']:+.10g}; "
                     f"RMSE {row['zero_minus_original_rmse']:+.10g}")
    lines += ['', 'Three-seed MAE mean and sample standard deviation:']
    for row in summary['aggregates']:
        lines.append(f"{row['model']}, {row['condition']}: {row['mean_mae']:.10g}, SD={row['sample_std_mae']}")
    lines += ['', 'Interpretation'] + ['- ' + x for x in summary['limitations']]
    lines += ['- Compare encoder norms, gradient counters, correction variation and baselines before expanding.',
              '- An improvement is not guaranteed. This experiment does not authorize a full run.']
    return '\n'.join(lines) + '\n'


def expected_files(p):
    names = {'run_identity.json', 'baselines.json', 'summary.json', 'summary.txt'}
    names.update(f'orders_seed_{s}.json' for s in p['config']['seeds'])
    for v in MODELS:
        for c in CONDITIONS:
            for s in p['config']['seeds']:
                names.update(f'fits/{fit_name(v,c,s)}/{n}' for n in FIT_FILES | {'fit_lock.json'})
    return names


def execute(check_only=False, verify_only=False, resume=False, output=OUTPUT):
    require(sum((check_only, verify_only, resume)) <= 1, 'Choose one mode')
    output = Path(output)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest, views, data, identity = context()
    p = identity['protocol']
    if check_only:
        preflight(manifest, views, data, p)
        verify_hashes(identity['pinned_files'], 'Confirm preflight source hashes')
        print('R05 bounded preflight passed. No dataset training, optimizer steps or result writes.')
        return
    if verify_only or resume:
        require(output.is_dir() and read(output/'run_identity.json') == identity, 'Changed run identity')
        if verify_only:
            lock = read(output/'evaluation_lock.json')
            require(set(lock['files']) == expected_files(p), 'Incomplete output lock')
            verify_hashes({str(output/n): h for n, h in lock['files'].items()}, 'Verify result hashes')
        else:
            require(not (output/'evaluation_lock.json').exists(), 'Completed output: use --verify-only')
    else:
        require(not output.exists(), 'Output exists: use --resume for unfinished or --verify-only for completed')
        preflight(manifest, views, data, p)
        output.mkdir(parents=True, exist_ok=False)
        write(output/'run_identity.json', identity)
    for seed in p['config']['seeds']:
        path = output/f'orders_seed_{seed}.json'
        orders = prior.epoch_orders(manifest['ids']['train'], settings(p, seed, 'original_decay'))
        if path.exists():
            require(read(path) == orders, 'Changed saved batch orders')
        else:
            require(not verify_only, 'Missing saved batch orders')
            write(path, orders)
    refs = baselines(manifest, data)
    if (output/'baselines.json').exists():
        require(read(output/'baselines.json') == refs, 'Changed scalar fit/baselines')
    else:
        require(not verify_only, 'Missing baselines')
        write(output/'baselines.json', refs)
    combinations = [(v, c, s) for s in p['config']['seeds'] for v in MODELS for c in CONDITIONS]
    results = {}
    for variant, condition, seed in tqdm(combinations, desc='Matched decay comparison', unit='fit'):
        name = fit_name(variant, condition, seed)
        directory = output/'fits'/name
        if not (directory/'fit_lock.json').exists():
            require(not verify_only, 'Missing completed fit')
            tqdm.write('Starting matched fit: ' + name)
            cfg = settings(p, seed, condition)
            saved, history, gradients = fit_one(variant, condition, seed, manifest, views[variant], data, p)
            initial = initialize(variant, manifest, views[variant], cfg)
            values, metrics = assess(saved, initial, manifest, views[variant], data)
            directory.mkdir(parents=True, exist_ok=True)
            prior.atomic_torch_save(directory/'checkpoint.pt', saved)
            for n, obj in (('history.json', history), ('gradients.json', gradients),
                           ('predictions.json', values), ('metrics.json', metrics)):
                write(directory/n, obj)
            write(directory/'fit_lock.json', {'schema': VERSION,
                                              'files': {n: sha256(directory/n) for n in sorted(FIT_FILES)}})
        results[name] = check_fit(directory, variant, condition, seed, manifest, views[variant], data, p)
    paired, aggregates = aggregate(results, p['config']['seeds'])
    summary = {'schema': VERSION, 'cohort_sizes': {s: len(v) for s,v in manifest['ids'].items()},
               'test_performance_evaluated': False, 'baselines': refs, 'fits': results,
               'paired': paired, 'aggregates': aggregates, 'limitations': p['limitations']}
    verify_hashes(identity['pinned_files'], 'Confirm frozen sources unchanged')
    if verify_only:
        require(read(output/'summary.json') == summary and
                (output/'summary.txt').read_text() == report(summary), 'Summary reproduction differs')
        print('R05 bounded comparison checkpoints and validation predictions verified. No retraining or test scoring.')
        return
    write(output/'summary.json', summary)
    (output/'summary.txt').write_text(report(summary), encoding='utf-8')
    write(output/'evaluation_lock.json', {'schema': VERSION,
                                         'files': {n: sha256(output/n) for n in sorted(expected_files(p))}})
    print(f'R05 bounded weight-decay comparison saved to {output}. No full training or test scoring.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-only', action='store_true')
    modes.add_argument('--verify-only', action='store_true')
    modes.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    execute(args.check_only, args.verify_only, args.resume)
