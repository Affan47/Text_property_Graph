"""R03: fixed multi-seed matched COREF development regression, validation only."""
import argparse
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
import math
from pathlib import Path
import statistics

import torch
from tqdm.auto import tqdm

from . import forecast_coreference_development_v2 as cache
from . import forecast_coreference_smoke as smoke
from . import forecast_tpg as task

require, read_json, write_json, sha256 = cache.require, cache.read_json, cache.write_json, cache.sha256
initialize, epoch_orders, digest_state = smoke.initialize, smoke.epoch_orders, smoke.digest_state
VERSION = 'r03_matched_coreference_development_v1'
VARIANTS = smoke.VARIANTS
DEFAULT_OUTPUT = task.RESULT_ROOT/'coreference_training_development_v1'
SCRIPT = cache.base.EPSS_ROOT/'02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_development.py'
TEST = SCRIPT.with_name('test_forecast_coreference_development.py')
FIT_FILES = {'checkpoint.pt', 'history.json', 'gradients.json', 'predictions.json', 'metrics.json'}
METRICS = ('mae', 'rmse', 'mean_signed_error', 'spearman_rho', 'equal_group_mae')


def settings_for(protocol, seed=None):
    settings = deepcopy(protocol['config'])
    expected = smoke.config()
    expected.update(epochs=100, patience=15, seeds=[17, 29, 43])
    require(settings == expected and protocol['execution'] ==
            {'torch_threads': 1, 'deterministic_algorithms': True}, 'Fixed R03 protocol differs')
    require(protocol['variants'] == list(VARIANTS) and protocol['cohort'] ==
            {'task': 'endpoint', 'train': 1023, 'validation': 197, 'split_changes': False, 'test_scoring': False},
            'Fixed experiment scope differs')
    if seed is not None:
        require(seed in settings['seeds'], 'Unregistered seed')
        settings['seeds'] = [seed]
    return settings


class GraphBank:
    """Bounded read-only tensor cache; views share nodes without tripling storage."""
    def __init__(self, directory, ids, capacity=128):
        self.directory, self.ids = Path(directory), frozenset(ids)
        self.capacity, self.items = capacity, OrderedDict()
        require(capacity > 0, 'Invalid cache capacity')

    def bundle(self, cve):
        require(cve in self.ids, 'CVE outside development cohort')
        if cve not in self.items:
            self.items[cve] = torch.load(self.directory/'graphs'/(cve+'.pt'),
                                         map_location='cpu', weights_only=True)
            if len(self.items) > self.capacity:
                self.items.popitem(last=False)
        self.items.move_to_end(cve)
        return self.items[cve]

    def view(self, mode):
        bank = self
        require(mode in cache.shared.MODES, 'Unknown graph view')
        class View(Mapping):
            def __len__(self):
                return len(bank.ids)
            def __iter__(self):
                return iter(sorted(bank.ids))
            def __getitem__(self, cve):
                return cache.shared.materialize(bank.bundle(cve), mode)
        return View()


def context(directory):
    directory = Path(directory).resolve()
    manifest = cache.verify(directory, expected_scale='full')
    protocol = read_json(cache.PROTOCOL)
    settings_for(protocol)
    require(manifest['protocol'] == protocol and {s: len(v) for s,v in manifest['ids'].items()} ==
            {'train': 1023, 'validation': 197}, 'Full cache/protocol mismatch')
    names = ('inputs.json', 'labels.json', 'split_manifest.json', 'task_cohorts.json')
    data = smoke.select_data(manifest, {n: read_json(cache.base.PREPARED/n) for n in names})
    ids = [c for s in cache.ROLES for c in manifest['ids'][s]]
    bank = GraphBank(directory, ids)
    graphs = {mode: bank.view(mode) for mode in cache.shared.MODES}
    graphs['epss_only'] = graphs['all_coref']
    pins = dict(manifest['source_pins'])
    pins.update({str(directory/n): h for n,h in read_json(directory/'cache_lock.json')['files'].items()})
    paths = [directory/'cache_lock.json', Path(__file__), Path(smoke.__file__), Path(task.__file__),
             Path(task.sequence.__file__), Path(task.__file__).with_name('forecast_baseline.py'),
             Path(task.__file__).with_name('edge_aware_layers.py'), SCRIPT, TEST]
    for path in paths:
        key, digest = str(path.resolve()), sha256(path)
        require(key not in pins or pins[key] == digest, 'Source changed while collecting identity')
        pins[key] = digest
    cache.base.check_pins(pins)
    identity = {'schema': VERSION, 'cache_directory': str(directory), 'ids': manifest['ids'],
                'protocol': protocol, 'pinned_files': pins, 'torch_version': str(torch.__version__),
                'cache_lock_sha256': sha256(directory/'cache_lock.json'),
                'test_performance_evaluated': False, 'limitation': cache.base.LIMITATION}
    return manifest, graphs, data, identity


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
        history, best, best_mae, stale = [], None, math.inf, 0
        gradients = {prefix: {'steps_with_gradient': 0, 'steps_with_nonzero_gradient': 0, 'absolute_gradient_sum': 0.0}
                     for prefix in ('temporal', 'node_project', 'layers', 'head') if hasattr(net, prefix)}
        for epoch, order in enumerate(tqdm(orders, desc=f'{variant}, seed {settings["seeds"][0]}', unit='epoch', disable=not progress), 1):
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
                best_mae, stale = errors['validation'], 0
                best = {'schema': task.VERSION, 'spec': initial['spec'], 'seed': settings['seeds'][0],
                    'selected_epoch': epoch, 'fit_cve_ids': train, 'selection_cve_ids': val,
                    'state_dict': {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                    'test_performance_evaluated': False, 'limitation': cache.base.LIMITATION,
                    'initial_head_bias_training_target_mean': initial['training_target_mean'],
                    'matched_experiment': VERSION, 'edge_configuration': variant,
                    'initial_state_sha256': initial['initial_state_sha256']}
            else:
                stale += 1
            if progress:
                tqdm.write(f"{variant}, seed {settings['seeds'][0]}, epoch {epoch}: training MAE={errors['train']:.8f}, validation MAE={errors['validation']:.8f}")
            if stale >= settings['patience']:
                break
    require(best is not None, 'No checkpoint selected')
    return best, history, gradients


def history_check(history, settings, selected_epoch):
    require(history and len(history) <= settings['epochs'] and
            [r['epoch'] for r in history] == list(range(1, len(history)+1)), 'Invalid epoch history')
    best, stale = math.inf, 0
    for i, row in enumerate(history):
        require(all(math.isfinite(row[k]) and row[k] >= 0 for k in
                    ('training_mae', 'validation_mae', 'training_huber_loss')), 'Invalid history values')
        if row['validation_mae'] < best:
            best, stale = row['validation_mae'], 0
        else:
            stale += 1
        require(stale < settings['patience'] or i == len(history)-1, 'Training continued past stopping rule')
    require(len(history) == settings['epochs'] or stale >= settings['patience'], 'Training stopped too early')
    require(selected_epoch == min(history, key=lambda r: r['validation_mae'])['epoch'], 'Checkpoint selection differs')


def score_predictions(predictions, manifest, data):
    ids = manifest['ids']['validation']
    require(len(predictions) == len(ids), 'Prediction length differs')
    actual = [data['labels.json'][c]['y']['endpoint'] for c in ids]
    groups = {'31_observed': [], '30_observed': [], 'at_most_29_observed': []}
    for i, cve in enumerate(ids):
        count = sum(data['inputs.json'][cve]['X']['observed_mask'])
        groups['31_observed' if count == 31 else '30_observed' if count == 30 else 'at_most_29_observed'].append(i)
    missingness = {}
    for name, indexes in groups.items():
        selected = [ids[i] for i in indexes]
        missingness[name] = {'n': len(selected), 'small_group_below_20': len(selected) < 20,
            'metrics': task.metrics([actual[i] for i in indexes], [predictions[i] for i in indexes], selected, data)
                       if selected else None}
    return {'metrics': task.metrics(actual, predictions, ids, data), 'missingness': missingness,
            'prediction_range': {'min': min(predictions), 'max': max(predictions),
                                 'mean': statistics.mean(predictions), 'std': statistics.pstdev(predictions)}}


def atomic_torch_save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temp)
    temp.replace(path)


def seed_artifacts(output, seed, manifest, graphs, data, settings, create=False):
    initial = {}
    for mode in ('tpg_epss', 'epss_only'):
        expected = initialize(mode, manifest, graphs['all_coref'], data, settings)
        path = output/'initialization'/f'seed_{seed}'/(mode+'.pt')
        if create and not path.exists():
            atomic_torch_save(path, expected)
        saved = torch.load(path, weights_only=True, map_location='cpu')
        require(saved['spec'] == expected['spec'] and saved['training_target_mean'] == expected['training_target_mean'] and
                digest_state(saved['state_dict']) == saved['initial_state_sha256'] == expected['initial_state_sha256'],
                'Saved initialization differs')
        initial[mode] = saved
    orders = epoch_orders(manifest['ids']['train'], settings)
    path = output/f'orders_seed_{seed}.json'
    if create and not path.exists():
        write_json(path, orders)
    require(read_json(path) == orders, 'Saved batch orders differ')
    return initial, orders


def check_fit(directory, variant, seed, initial, manifest, graphs, data, settings, identity):
    lock = read_json(directory/'fit_lock.json')
    require(lock['schema'] == VERSION and set(lock['files']) == FIT_FILES, 'Wrong fit artifact contract')
    require({p.name for p in directory.iterdir()} == FIT_FILES | {'fit_lock.json'}, 'Unexpected fit files')
    for name, digest in lock['files'].items():
        require(sha256(directory/name) == digest, f'Changed fit artifact: {directory.name}/{name}')
    saved = torch.load(directory/'checkpoint.pt', weights_only=True, map_location='cpu')
    require(saved['matched_experiment'] == VERSION and saved['edge_configuration'] == variant and
            saved['seed'] == seed and saved['spec'] == initial['spec'] and
            saved['initial_state_sha256'] == initial['initial_state_sha256'] and
            saved['cache_lock_sha256'] == identity['cache_lock_sha256'] and
            saved['fit_cve_ids'] == manifest['ids']['train'] and
            saved['selection_cve_ids'] == manifest['ids']['validation'], 'Checkpoint identity differs')
    history = read_json(directory/'history.json')
    history_check(history, settings, saved['selected_epoch'])
    gradients = read_json(directory/'gradients.json')
    net = task.restore(saved)
    prefixes = {p for p in ('temporal', 'node_project', 'layers', 'head') if hasattr(net, p)}
    require(set(gradients) == prefixes, 'Missing branch-gradient diagnostic')
    steps = len(history)*math.ceil(len(manifest['ids']['train'])/settings['batch_size'])
    for row in gradients.values():
        require(0 <= row['steps_with_nonzero_gradient'] <= row['steps_with_gradient'] <= steps and
                math.isfinite(row['absolute_gradient_sum']) and row['absolute_gradient_sum'] >= 0,
                'Invalid gradient diagnostic')
    predictions = task.infer(net, manifest['ids']['validation'], graphs, data['inputs.json'],
                             settings['batch_size'], f'Verify {variant} seed {seed}')
    expected_predictions = [{'cve': c, 'prediction': p} for c,p in zip(manifest['ids']['validation'], predictions)]
    require(read_json(directory/'predictions.json') == expected_predictions, 'Saved predictions do not reproduce')
    metrics = score_predictions(predictions, manifest, data)
    require(read_json(directory/'metrics.json') == metrics and
            history[saved['selected_epoch']-1]['validation_mae'] == metrics['metrics']['mae'],
            'Selected-checkpoint metrics differ')
    return {'variant': variant, 'seed': seed, 'selected_epoch': saved['selected_epoch'],
            'epochs_completed': len(history), 'budget_reached': len(history) == settings['epochs'],
            'selected_at_budget_boundary': saved['selected_epoch'] == settings['epochs'], **metrics}, predictions, history


def fit_directory(output, seed, variant):
    return output/'fits'/f'{variant}__{seed}'


def save_fit(directory, saved, history, gradients, predictions, metrics):
    directory.mkdir(parents=True, exist_ok=True)
    require(not (directory/'fit_lock.json').exists(), 'Completed fit must not be overwritten')
    require({p.name for p in directory.iterdir()} <= FIT_FILES | {n+'.tmp' for n in FIT_FILES},
            'Unexpected unfinished fit files')
    atomic_torch_save(directory/'checkpoint.pt', saved)
    for name, value in (('history', history), ('gradients', gradients), ('predictions', predictions), ('metrics', metrics)):
        write_json(directory/(name+'.json'), value)
    write_json(directory/'fit_lock.json', {'schema': VERSION,
               'files': {n: sha256(directory/n) for n in sorted(FIT_FILES)}})


def aggregate(results, seeds):
    aggregates, pairs = {}, []
    lookup = {(r['variant'], r['seed']): r for r in results}
    for variant in VARIANTS:
        rows = [lookup[(variant, seed)] for seed in seeds]
        aggregates[variant] = {}
        for metric in METRICS:
            values = [r['metrics'][metric] for r in rows if r['metrics'][metric] is not None]
            aggregates[variant][metric] = {'defined_seeds': len(values),
                'mean': statistics.mean(values) if values else None,
                'sample_std': statistics.stdev(values) if len(values) > 1 else None}
        for seed in seeds:
            for reference in ('persistence', 'epss_only', 'no_coref'):
                if reference == variant:
                    continue
                ref = lookup[(reference, None if reference == 'persistence' else seed)]
                pairs.append({'variant': variant, 'seed': seed, 'reference': reference,
                    'mae_difference': lookup[(variant, seed)]['metrics']['mae']-ref['metrics']['mae']})
    return aggregates, pairs


def collect(output, manifest, graphs, data, identity):
    ids = manifest['ids']['validation']
    persistence = [task.predict_persistence(data['inputs.json'][c])['prediction'] for c in ids]
    results = [{'variant': 'persistence', 'seed': None, 'selected_epoch': None,
                'epochs_completed': 0, **score_predictions(persistence, manifest, data)}]
    rows = [{'cve': c, 'actual_day60_epss': data['labels.json'][c]['y']['endpoint'],
             'prediction_anchor_date': data['inputs.json'][c]['input_end_date'],
             'target_date': data['labels.json'][c]['target_date'], 'persistence': p,
             'predictions': {}} for c,p in zip(ids, persistence)]
    histories = {}
    seeds = identity['protocol']['config']['seeds']
    for seed in tqdm(seeds, desc='Verify seed results', unit='seed'):
        settings = deepcopy(identity['protocol']['config']); settings['seeds'] = [seed]
        initial, _ = seed_artifacts(output, seed, manifest, graphs, data, settings)
        for variant in VARIANTS:
            mode = 'epss_only' if variant == 'epss_only' else 'tpg_epss'
            result, pred, history = check_fit(fit_directory(output, seed, variant), variant, seed,
                initial[mode], manifest, graphs[variant], data, settings, identity)
            results.append(result)
            key = f'{variant}__{seed}'; histories[key] = history
            for row, value in zip(rows, pred):
                row['predictions'][key] = value
    aggregated, pairs = aggregate(results, seeds)
    summary = {'schema': VERSION, 'identity': identity, 'results': results,
               'seed_aggregates': aggregated, 'paired_mae_differences': pairs,
               'test_performance_evaluated': False}
    return summary, rows, histories


def report(summary):
    lines = ['R03: Matched COREF Development Experiment', '',
        'Current-description graphs and EPSS Days 0-30 predict the stored Day-60 EPSS score.',
        'This is retrospective validation, not exploitation prediction or test performance.',
        f"CVEs by role: { {s: len(v) for s,v in summary['identity']['ids'].items()} }.",
        'Four trained variants, three fixed seeds; persistence is one deterministic reference.',
        'Each seed shares GNN initialization and batch orders across the three graph modes.',
        'Up to 100 epochs; patience 15; earliest strictly lowest validation MAE is selected.', '',
        'Model | Seed | Selected epoch | Epochs run | Validation MAE | RMSE | Equal-group MAE']
    for r in summary['results']:
        m = r['metrics']
        lines.append(f"{r['variant']} | {r['seed']} | {r['selected_epoch']} | {r['epochs_completed']} | "
                     f"{m['mae']:.8f} | {m['rmse']:.8f} | {m['equal_group_mae']:.8f}")
    lines += ['', 'Model | Mean validation MAE across seeds | Sample standard deviation']
    for mode, values in summary['seed_aggregates'].items():
        m = values['mae']; lines.append(f"{mode} | {m['mean']:.8f} | {m['sample_std']:.8f}")
    lines += ['', 'Lower MAE/RMSE is better; 0.01 is one EPSS percentage point.',
        'Mean signed error is predicted minus actual. Spearman is null when undefined.',
        'Equal-group MAE gives each related-CVE group equal weight. No best-seed selection.',
        'Paired MAE differences in JSON compare identical CVEs; a negative value favors the named variant.',
        'Missingness groups use past scores only: 31 observed, 30 observed, or at most 29 observed.',
        'Groups below 20 CVEs are flagged; empty groups have null metrics. No group is dropped from the main results.',
        'Gradient diagnostics indicate gradient flow, not branch importance or causal contribution.',
        'Prediction ranges and budget-boundary flags help identify constant outputs or incomplete convergence.',
        'No new parsing, embeddings, split changes, test scoring or hyperparameter search.', '',
        *summary['identity']['protocol']['limitations'], '',
        'Files: initialization/ stores initial states; orders_seed_*.json stores complete planned schedules.',
        'fits/ stores selected checkpoints, epoch histories, gradients, predictions, metrics and per-fit locks.',
        'summary.json/txt stores all seeds, aggregates and matched comparisons; predictions.json stores CVE-level outputs.',
        'training_history.json and training_curves.png show training/validation errors over epochs.',
        'evaluation_lock.json hashes completed outputs. Completed fits are never overwritten.', '']
    return '\n'.join(lines)


def plot(histories, output, seeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(VARIANTS), len(seeds), figsize=(15, 12), squeeze=False, constrained_layout=True)
    for i, variant in enumerate(VARIANTS):
        for j, seed in enumerate(seeds):
            rows = histories[f'{variant}__{seed}']; ax = axes[i,j]
            for key, style in (('training_mae', '-'), ('validation_mae', '--')):
                ax.plot([r['epoch'] for r in rows], [r[key] for r in rows], style, label=key.replace('_', ' '))
            ax.set(title=f'{variant}, seed {seed}', xlabel='Epoch', ylabel='EPSS mean absolute error')
            ax.legend(fontsize=8)
    fig.suptitle('Matched development runs: solid = training; dashed = validation; lower error is better')
    fig.savefig(output, dpi=140); plt.close(fig)


def expected_files(seeds):
    names = {'run_identity.json', 'run_status.json', 'summary.json', 'summary.txt',
             'predictions.json', 'training_history.json', 'training_curves.png'}
    for seed in seeds:
        names.add(f'orders_seed_{seed}.json')
        names.update(f'initialization/seed_{seed}/{mode}.pt' for mode in ('tpg_epss', 'epss_only'))
        names.update(f'fits/{v}__{seed}/{n}' for v in VARIANTS for n in FIT_FILES | {'fit_lock.json'})
    return names


def verify_output(output, manifest, graphs, data, identity):
    lock = read_json(output/'evaluation_lock.json')
    expected = expected_files(identity['protocol']['config']['seeds'])
    require(lock['schema'] == VERSION and set(lock['files']) == expected, 'Wrong result file contract')
    require({str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()} ==
            expected | {'evaluation_lock.json'}, 'Unexpected result files')
    for name, digest in tqdm(lock['files'].items(), desc='Verify result checksums', unit='file'):
        require(sha256(output/name) == digest, f'Changed result: {name}')
    require(read_json(output/'run_identity.json') == identity and
            read_json(output/'run_status.json')['status'] == 'complete', 'Run identity/status differs')
    summary, predictions, histories = collect(output, manifest, graphs, data, identity)
    require(read_json(output/'summary.json') == summary and read_json(output/'predictions.json') == predictions and
            read_json(output/'training_history.json') == histories, 'Results do not reproduce')
    require((output/'summary.txt').read_text() == report(summary), 'Text report differs')
    cache.base.check_pins(identity['pinned_files'])


def preflight(manifest, graphs, data, settings):
    initial = {m: initialize(m, manifest, graphs['all_coref'], data, settings) for m in ('tpg_epss', 'epss_only')}
    batches = [('small_train', manifest['ids']['train'][:2])]
    for split in cache.ROLES:
        # Graph size only; validation targets do not enter the preflight.
        sizes = {r['cve']: r['nodes'] for r in manifest['graphs']} if 'graphs' in manifest else {
            c: len(graphs['all_coref'][c]['x']) for c in manifest['ids'][split]}
        batches.append(('largest_'+split, sorted(manifest['ids'][split], key=lambda c: (-sizes[c], c))[:settings['batch_size']]))
    with tqdm(total=len(VARIANTS)*len(batches), desc='Check small and large graph forward batches', unit='batch') as bar:
        for variant in VARIANTS:
            mode = 'epss_only' if variant == 'epss_only' else 'tpg_epss'
            with torch.random.fork_rng(devices=[]):
                net = task.TPGRegressor(**initial[mode]['spec']).eval()
                net.load_state_dict(initial[mode]['state_dict'])
            for name, ids in batches:
                task.infer(net, ids, graphs[variant], data['inputs.json'], settings['batch_size'], progress=False)
                bar.update()
    return initial


def run(directory, output, check_only=False, verify_only=False, resume=False):
    output = Path(output).resolve()
    manifest, graphs, data, identity = context(directory)
    seeds = identity['protocol']['config']['seeds']
    require(task.RESULT_ROOT.resolve() in output.parents, 'Use a new evaluation-results folder')
    if verify_only:
        verify_output(output, manifest, graphs, data, identity)
        print('R03 all seeds, checkpoints and validation predictions verified. No retraining or test scoring.'); return
    if resume:
        require(output.is_dir() and not (output/'evaluation_lock.json').exists(),
                'Resume needs unfinished output; use --verify-only for completed results')
        require(read_json(output/'run_identity.json') == identity, 'Resume sources/settings changed')
        allowed = expected_files(seeds)
        actual = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}
        require(actual <= allowed | {n+'.tmp' for n in allowed}, 'Unexpected unfinished result files')
    else:
        cache.base.check_destination(output, [cache.ROOT, cache.base.PREPARED, cache.base.EPSS_ROOT/'02_prediction'])
    settings = deepcopy(identity['protocol']['config']); settings['seeds'] = [seeds[0]]
    preflight(manifest, graphs, data, settings)
    cache.base.check_pins(identity['pinned_files'])
    if check_only:
        print('R03 preflight passed: small and largest development graph batches; no fitting, scoring or output writes.'); return
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/'run_identity.json', identity)
    write_json(output/'run_status.json', {'status': 'running'})
    try:
        with tqdm(total=len(seeds)*len(VARIANTS), desc='Matched fixed-seed fits', unit='fit') as progress:
            for seed in seeds:
                settings = deepcopy(identity['protocol']['config']); settings['seeds'] = [seed]
                initial, orders = seed_artifacts(output, seed, manifest, graphs, data, settings, create=True)
                for variant in VARIANTS:
                    path = fit_directory(output, seed, variant)
                    mode = 'epss_only' if variant == 'epss_only' else 'tpg_epss'
                    if not (path/'fit_lock.json').exists():
                        saved, history, gradients = fit(variant, initial[mode], orders, manifest,
                                                       graphs[variant], data, settings)
                        saved['cache_lock_sha256'] = identity['cache_lock_sha256']
                        predictions = task.infer(task.restore(saved), manifest['ids']['validation'],
                            graphs[variant], data['inputs.json'], settings['batch_size'], 'Selected checkpoint')
                        rows = [{'cve': c, 'prediction': p} for c,p in zip(manifest['ids']['validation'], predictions)]
                        save_fit(path, saved, history, gradients, rows, score_predictions(predictions, manifest, data))
                    check_fit(path, variant, seed, initial[mode], manifest, graphs[variant], data, settings, identity)
                    progress.update()
        summary, rows, histories = collect(output, manifest, graphs, data, identity)
        with tqdm(total=3, desc='Save reports and lock matched experiment', unit='stage') as bar:
            for name, value in (('summary', summary), ('predictions', rows), ('training_history', histories)):
                write_json(output/(name+'.json'), value)
            (output/'summary.txt').write_text(report(summary), encoding='utf-8'); bar.update()
            plot(histories, output/'training_curves.png', seeds); bar.update()
            cache.base.check_pins(identity['pinned_files'])
            write_json(output/'run_status.json', {'status': 'complete'})
            write_json(output/'evaluation_lock.json', {'schema': VERSION, 'files': {
                str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}})
            bar.update()
        verify_output(output, manifest, graphs, data, identity)
    except Exception as exc:
        write_json(output/'run_status.json', {'status': 'failed', 'error': str(exc)})
        raise
    print(f'R03 matched development experiment saved to {output}. All fixed seeds; validation only, no test scoring.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, default=cache.ROOT/'full')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument('--check-only', action='store_true')
    flags.add_argument('--verify-only', action='store_true')
    flags.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        run(args.cache_dir, args.output_dir, args.check_only, args.verify_only, args.resume)
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


if __name__ == '__main__':
    main()
