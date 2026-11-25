"""R02: bounded, matched COREF training smoke test; validation only."""
import argparse
from datetime import date
import hashlib
import math
from pathlib import Path
import statistics

import torch
from tqdm.auto import tqdm

from . import forecast_coreference_cache as cache
from . import forecast_tpg as task

require, read_json, write_json, sha256 = cache.require, cache.read_json, cache.write_json, cache.sha256
VERSION = 'r02_matched_coreference_smoke_v1'
VARIANTS = (*cache.MODES, 'epss_only')
DEFAULT_OUTPUT = task.RESULT_ROOT/'coreference_training_smoke_v1'
SCRIPT = cache.base.EPSS_ROOT/'02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_smoke.py'
TEST = SCRIPT.with_name('test_forecast_coreference_smoke.py')


def config():
    result = task.config_for('smoke')
    result['dropout'] = 0.0
    return result


def digest_state(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def select_data(manifest, data):
    ids = manifest['ids']
    require(set(ids) == {'train', 'validation'}, 'Only train/validation roles permitted')
    rows = [{'cve': c, 'split': split} for split in ids for c in ids[split]]
    cache.selected_ids(rows, data['split_manifest.json'], data['task_cohorts.json'])
    selected = [r['cve'] for r in rows]
    result = {name: {c: data[name][c] for c in selected}
              for name in ('inputs.json', 'labels.json', 'split_manifest.json')}
    for c in selected:
        x, y = result['inputs.json'][c], result['labels.json'][c]
        cache.base.contract.predictor_view(x)
        require(y['eligible']['endpoint'] and y['y']['endpoint'] is not None, 'Missing/ineligible endpoint')
        cache.base.contract.score(y['y']['endpoint'])
        require((date.fromisoformat(y['target_date']) - date.fromisoformat(x['input_end_date'])).days == 30,
                'Expected Day-30 anchor and Day-60 target')
    return result


def context(directory):
    directory = Path(directory).resolve()
    manifest, bundles = cache.verify(directory)
    require(manifest['scale'] == 'smoke' and {s: len(v) for s, v in manifest['ids'].items()} ==
            {'train': 32, 'validation': 24}, 'Only the fixed DQ16 smoke cohort is allowed')
    # Source checksums were verified by DQ16. Only selected development records
    # enter the training/evaluation functions, never the reserved-test cohort.
    names = ('inputs.json', 'labels.json', 'split_manifest.json', 'task_cohorts.json')
    data = select_data(manifest, {n: read_json(cache.base.PREPARED/n) for n in names})
    graphs = {mode: {c: cache.materialize(b, mode) for c, b in bundles.items()} for mode in cache.MODES}
    graphs['epss_only'] = graphs['all_coref']
    pins = dict(manifest['source_pins'])
    pins.update({str(directory/n): h for n, h in read_json(directory/'cache_lock.json')['files'].items()})
    paths = [directory/'cache_lock.json', Path(__file__), Path(task.__file__),
             Path(task.sequence.__file__), Path(task.__file__).with_name('forecast_baseline.py'),
             Path(task.__file__).with_name('edge_aware_layers.py'), SCRIPT, TEST]
    paths += [cache.base.PREPARED/n for n in names]
    for path in paths:
        key, digest = str(path.resolve()), sha256(path)
        require(key not in pins or pins[key] == digest, 'Changed dependency while collecting identity')
        pins[key] = digest
    cache.base.check_pins(pins)
    identity = {'schema': VERSION, 'cache_directory': str(directory), 'pinned_files': pins,
                'config': config(), 'ids': manifest['ids'], 'torch_version': str(torch.__version__),
                'limitation': cache.base.LIMITATION, 'test_performance_evaluated': False}
    return manifest, graphs, data, identity


def initialize(mode, manifest, graphs, data, settings):
    spec = task.model_spec(mode, manifest, graphs, settings)
    mean = statistics.mean(data['labels.json'][c]['y']['endpoint'] for c in manifest['ids']['train'])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(settings['seeds'][0])
        net = task.TPGRegressor(**spec)
        bounded = min(1-1e-5, max(1e-5, mean))
        with torch.no_grad():
            net.head[-1].bias.fill_(math.log(bounded/(1-bounded)))
    state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    return {'spec': spec, 'state_dict': state, 'initial_state_sha256': digest_state(state),
            'training_target_mean': mean}


def epoch_orders(ids, settings):
    generator = torch.Generator().manual_seed(settings['seeds'][0])
    return [[ids[i] for i in torch.randperm(len(ids), generator=generator).tolist()]
            for _ in range(settings['epochs'])]


def fit(variant, initial, orders, manifest, graphs, data, settings, progress=True):
    require(variant in VARIANTS, 'Unknown variant')
    train, val = (manifest['ids'][s] for s in ('train', 'validation'))
    require(len(orders) == settings['epochs'] and all(sorted(o) == sorted(train) for o in orders),
            'Training batch schedule differs')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(settings['seeds'][0])
        net = task.TPGRegressor(**initial['spec'])
        net.load_state_dict(initial['state_dict'], strict=True)
        require(digest_state(net.state_dict()) == initial['initial_state_sha256'], 'Initialization differs')
        optimizer = torch.optim.Adam(net.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
        criterion = torch.nn.SmoothL1Loss(beta=settings['huber_beta'])
        history, best, best_mae = [], None, math.inf
        gradients = {prefix: {'steps_with_gradient': 0, 'steps_with_nonzero_gradient': 0, 'absolute_gradient_sum': 0.0}
                     for prefix in ('temporal', 'node_project', 'layers', 'head') if hasattr(net, prefix)}
        for epoch, order in enumerate(tqdm(orders, desc=variant, unit='epoch', disable=not progress), 1):
            net.train()
            total = 0.0
            for start in tqdm(range(0, len(order), settings['batch_size']), desc='Training batches', unit='batch',
                              leave=False, disable=not progress):
                ids = order[start:start+settings['batch_size']]
                batch = task.make_batch(ids, graphs, data['inputs.json'])
                y = torch.tensor([data['labels.json'][c]['y']['endpoint'] for c in ids], dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(net(batch), y)
                require(bool(torch.isfinite(loss)), 'Non-finite loss')
                loss.backward()
                for prefix, stats in gradients.items():
                    values = [p.grad for n, p in net.named_parameters() if n.startswith(prefix+'.') and p.grad is not None]
                    require(all(bool(torch.isfinite(v).all()) for v in values), 'Non-finite branch gradient')
                    magnitude = sum(float(v.abs().sum()) for v in values)
                    stats['steps_with_gradient'] += int(bool(values))
                    stats['steps_with_nonzero_gradient'] += int(magnitude > 0)
                    stats['absolute_gradient_sum'] += magnitude
                torch.nn.utils.clip_grad_norm_(net.parameters(), settings['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach()) * len(ids)
            errors = {}
            for split in ('train', 'validation'):
                ids = manifest['ids'][split]
                pred = task.infer(net, ids, graphs, data['inputs.json'], settings['batch_size'], split+' metrics', progress)
                actual = [data['labels.json'][c]['y']['endpoint'] for c in ids]
                errors[split] = task.regression_metrics(actual, pred)['mae']
            history.append({'epoch': epoch, 'training_huber_loss': total/len(train),
                            'training_mae': errors['train'], 'validation_mae': errors['validation']})
            if errors['validation'] < best_mae:
                best_mae = errors['validation']
                best = {'schema': task.VERSION, 'spec': initial['spec'], 'seed': settings['seeds'][0],
                    'selected_epoch': epoch, 'fit_cve_ids': train, 'selection_cve_ids': val,
                    'state_dict': {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                    'test_performance_evaluated': False, 'limitation': cache.base.LIMITATION,
                    'initial_head_bias_training_target_mean': initial['training_target_mean'],
                    'matched_experiment': VERSION, 'edge_configuration': variant,
                    'initial_state_sha256': initial['initial_state_sha256']}
            if progress:
                tqdm.write(f"{variant}, epoch {epoch}: training MAE={errors['train']:.8f}, validation MAE={errors['validation']:.8f}")
    require(best is not None, 'No checkpoint selected')
    return best, history, gradients


def report(summary):
    lines = ['R02: Matched COREF Training Smoke Test', '',
        'Engineering smoke test only: 32 training CVEs, 24 validation CVEs, seed 17, three epochs.',
        'This small reviewed sample is not a final performance benchmark. No test scoring.',
        'Inputs: current-description TPG and EPSS Days 0-30. Target: EPSS on Day 60.',
        'The horizon is 30 days after the Day-30 prediction anchor, not exploitation probability estimation.', '',
        'Model | Selected epoch | Validation MAE | Validation RMSE | Equal-group MAE',
        '------|----------------|----------------|-----------------|----------------']
    for row in summary['results']:
        m = row['metrics']
        lines.append(f"{row['variant']} | {row['selected_epoch']} | {m['mae']:.8f} | {m['rmse']:.8f} | {m['equal_group_mae']:.8f}")
    lines += ['', 'MAE averages absolute errors; RMSE emphasizes larger errors. Lower is better.',
        'An EPSS error of 0.01 is one percentage point. Equal-group MAE gives each related-CVE group equal weight.',
        'no_coref, repeated_only and all_coref use identical initial weights, batch orders and settings.',
        'Only the selected COREF edges differ. Nodes, other edges and missingness masks remain shared.',
        'The GNN has two 32-unit relation-aware layers and mean node pooling. EPSS uses a 16-unit GRU.',
        'Their representations are concatenated into a sigmoid direct-score regression head.',
        'Dropout is zero in this smoke test to avoid edge-count-dependent random dropout differences.',
        'The head bias uses only the training-target mean. Adam, Smooth L1 loss and gradient clipping are fixed.',
        'All three epochs run; the earliest minimum validation-MAE checkpoint is selected.',
        'epss_only is a separately initialized/trained temporal reference on exactly the same CVEs.',
        'persistence uses the last observed input EPSS, which may be earlier than Day 30; it is not fitted.',
        'Gradient diagnostics show whether branches received gradients, not whether they add predictive value.',
        'The comparison uses the existing edge-type-aware GNN, not the separate multiview classifier.',
        'Frozen SecBERT embeddings are not updated. No parser rerun or new extraction occurs.', '',
        cache.base.LIMITATION,
        'Validation has been inspected repeatedly. Neither these metrics nor a smoke-run ranking is a final estimate.',
        'No full expansion, test scoring, hyperparameter search or automatic winning-model promotion.', '',
        'summary.json/txt: settings, metrics and branch-gradient diagnostics.',
        'initialization/*.pt and batch_orders.json: fixed initial states and actual shared training schedules.',
        'checkpoints/*.pt: selected weights and the DQ16 cache-lock checksum.',
        'predictions.json: validation CVEs, actual Day-60 scores and predictions from all five comparisons.',
        'training_history.json and training_curves.png: per-epoch training and validation errors.',
        'evaluation_lock.json: output checksums. No existing results are overwritten.', '']
    return '\n'.join(lines)


def plot(histories, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, variant in zip(axes.flat, VARIANTS):
        h = histories[variant]
        for field, style in (('training_mae', '-'), ('validation_mae', '--')):
            ax.plot([r['epoch'] for r in h], [r[field] for r in h], style, marker='o', label=field.replace('_', ' '))
        ax.set(title=variant, xlabel='Training epoch', ylabel='EPSS mean absolute error')
        ax.set_xticks([r['epoch'] for r in h]); ax.legend(fontsize=8)
    fig.suptitle('Smoke test: 32 training / 24 validation CVEs; seed 17. Lower error is better.')
    fig.savefig(output, dpi=140); plt.close(fig)


def evaluate(saved_models, manifest, graphs, data, settings):
    ids = manifest['ids']['validation']
    actual = [data['labels.json'][c]['y']['endpoint'] for c in ids]
    persistence = [task.predict_persistence(data['inputs.json'][c])['prediction'] for c in ids]
    rows = [{'cve': c, 'actual_day60_epss': y, 'persistence': p,
             'prediction_anchor_date': data['inputs.json'][c]['input_end_date'],
             'target_date': data['labels.json'][c]['target_date']} for c, y, p in zip(ids, actual, persistence)]
    results = [{'variant': 'persistence', 'selected_epoch': None,
                'metrics': task.metrics(actual, persistence, ids, data)}]
    for variant in VARIANTS:
        saved = saved_models[variant]
        require(saved['matched_experiment'] == VERSION and saved['edge_configuration'] == variant,
                'Wrong checkpoint configuration')
        require(saved['fit_cve_ids'] == manifest['ids']['train'] and saved['selection_cve_ids'] == ids,
                'Checkpoint cohorts differ')
        predictions = task.infer(task.restore(saved), ids, graphs[variant], data['inputs.json'],
                                 settings['batch_size'], variant+' validation')
        for row, p in zip(rows, predictions):
            row[variant] = p
        results.append({'variant': variant, 'selected_epoch': saved['selected_epoch'],
                        'metrics': task.metrics(actual, predictions, ids, data)})
    return rows, results


def verify_output(output, manifest, graphs, data, identity):
    lock = read_json(output/'evaluation_lock.json')
    expected = {'summary.json', 'summary.txt', 'predictions.json', 'training_history.json', 'training_curves.png',
                'batch_orders.json', 'run_status.json'} | {'checkpoints/'+v+'.pt' for v in VARIANTS} | {
                'initialization/'+m+'.pt' for m in ('tpg_epss', 'epss_only')}
    require(lock['schema'] == VERSION and set(lock['files']) == expected, 'Unexpected output contract')
    require({str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()} == expected | {'evaluation_lock.json'},
            'Unexpected output files')
    for name, digest in lock['files'].items():
        require(sha256(output/name) == digest, f'Changed result: {name}')
    summary = read_json(output/'summary.json')
    require(summary['identity'] == identity and summary['schema'] == VERSION and
            summary['test_performance_evaluated'] is False, 'Run identity differs')
    require(read_json(output/'run_status.json')['status'] == 'complete', 'Run is not complete')
    settings = identity['config']
    orders = read_json(output/'batch_orders.json')
    require(orders == epoch_orders(manifest['ids']['train'], settings), 'Batch orders differ')
    initial = {}
    for mode in ('tpg_epss', 'epss_only'):
        saved = torch.load(output/'initialization'/(mode+'.pt'), weights_only=True, map_location='cpu')
        expected_init = initialize(mode, manifest, graphs['all_coref'], data, settings)
        require(saved['spec'] == expected_init['spec'] and saved['training_target_mean'] == expected_init['training_target_mean']
                and digest_state(saved['state_dict']) == saved['initial_state_sha256'] == expected_init['initial_state_sha256'],
                'Initial state does not reproduce')
        initial[mode] = saved
    models = {v: torch.load(output/'checkpoints'/(v+'.pt'), weights_only=True, map_location='cpu') for v in VARIANTS}
    history = read_json(output/'training_history.json')
    require(set(history) == set(VARIANTS), 'Missing training history')
    for v, saved in models.items():
        mode = 'epss_only' if v == 'epss_only' else 'tpg_epss'
        require(saved['spec'] == initial[mode]['spec'] and saved['seed'] == settings['seeds'][0] and
                saved['initial_state_sha256'] == initial[mode]['initial_state_sha256'], 'Checkpoint initialization differs')
        require(saved['cache_lock_sha256'] == identity['pinned_files'][str(Path(identity['cache_directory'])/'cache_lock.json')],
                'Checkpoint cache identity differs')
        h = history[v]
        require([r['epoch'] for r in h] == list(range(1, settings['epochs']+1)), 'Training budget differs')
        require(saved['selected_epoch'] == min(h, key=lambda r:r['validation_mae'])['epoch'], 'Selection differs')
    rows, results = evaluate(models, manifest, graphs, data, settings)
    require(read_json(output/'predictions.json') == rows and summary['results'] == results, 'Predictions/metrics do not reproduce')
    for result in results[1:]:
        h = history[result['variant']][result['selected_epoch']-1]
        require(h['validation_mae'] == result['metrics']['mae'], 'Selected checkpoint/history mismatch')
    require((output/'summary.txt').read_text(encoding='utf-8') == report(summary), 'Text report differs')
    cache.base.check_pins(identity['pinned_files'])


def run(directory, output, check_only=False, verify_only=False):
    output = Path(output).resolve()
    manifest, graphs, data, identity = context(directory)
    if verify_only:
        verify_output(output, manifest, graphs, data, identity)
        print('R02 smoke results and validation predictions verified. No retraining or test scoring.'); return
    cache.base.check_destination(output, [cache.DEFAULT_OUTPUT.parent, cache.base.PREPARED,
        cache.TPG, cache.base.EPSS_ROOT/'02_prediction', cache.base.EPSS_ROOT.parent/'SummTPGVul'])
    require(task.RESULT_ROOT.resolve() in output.parents, 'Output must be inside the evaluation results directory')
    settings = identity['config']
    initial = {m: initialize(m, manifest, graphs['all_coref'], data, settings) for m in ('tpg_epss', 'epss_only')}
    orders = epoch_orders(manifest['ids']['train'], settings)
    for variant in tqdm(VARIANTS, desc='Check small training-role forward batches', unit='model'):
        mode = 'epss_only' if variant == 'epss_only' else 'tpg_epss'
        with torch.random.fork_rng(devices=[]):
            net = task.TPGRegressor(**initial[mode]['spec']).eval()
            net.load_state_dict(initial[mode]['state_dict'])
        task.infer(net, manifest['ids']['train'][:2], graphs[variant], data['inputs.json'], 2, progress=False)
    cache.base.check_pins(identity['pinned_files'])
    if check_only:
        print('R02 preflight passed. No optimizer steps, validation scoring, output writes or test scoring.'); return
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'run_status.json', {'status': 'running', 'identity': identity})
    try:
        (output/'checkpoints').mkdir(); (output/'initialization').mkdir()
        write_json(output/'batch_orders.json', orders)
        for mode, init in initial.items():
            torch.save(init, output/'initialization'/(mode+'.pt'))
        models, histories, gradients = {}, {}, {}
        for variant in tqdm(VARIANTS, desc='Matched smoke experiments', unit='model'):
            mode = 'epss_only' if variant == 'epss_only' else 'tpg_epss'
            model, history, grad = fit(variant, initial[mode], orders, manifest, graphs[variant], data, settings)
            model['cache_lock_sha256'] = identity['pinned_files'][str(Path(identity['cache_directory'])/'cache_lock.json')]
            torch.save(model, output/'checkpoints'/(variant+'.pt'))
            models[variant], histories[variant], gradients[variant] = model, history, grad
        rows, results = evaluate(models, manifest, graphs, data, settings)
        summary = {'schema': VERSION, 'identity': identity, 'results': results, 'gradient_diagnostics': gradients,
                   'test_performance_evaluated': False, 'smoke_only': True, 'full_expansion_authorized': False}
        with tqdm(total=3, desc='Save reports and verify provenance', unit='stage') as bar:
            write_json(output/'summary.json', summary); write_json(output/'predictions.json', rows)
            write_json(output/'training_history.json', histories)
            (output/'summary.txt').write_text(report(summary), encoding='utf-8'); bar.update()
            plot(histories, output/'training_curves.png'); bar.update()
            cache.base.check_pins(identity['pinned_files'])
            write_json(output/'run_status.json', {'status': 'complete', 'identity': identity})
            write_json(output/'evaluation_lock.json', {'schema': VERSION, 'files': {
                str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}}); bar.update()
        verify_output(output, manifest, graphs, data, identity)
    except Exception as exc:
        write_json(output/'run_status.json', {'status': 'failed', 'identity': identity, 'error': str(exc)})
        raise
    print(f'R02 matched COREF smoke saved to {output}. Validation only; no test scoring.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, default=cache.DEFAULT_OUTPUT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument('--check-only', action='store_true')
    flags.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        run(args.cache_dir, args.output_dir, args.check_only, args.verify_only)
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


if __name__ == '__main__':
    main()
