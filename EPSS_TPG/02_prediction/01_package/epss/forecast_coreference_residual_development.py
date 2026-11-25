"""R04 v2 full development: frozen scaled residual, 12 matched fits, no test scoring."""
import argparse
from copy import deepcopy
import math
from pathlib import Path

import torch
from tqdm.auto import tqdm

from . import forecast_coreference_development_v2 as cache
from . import forecast_coreference_smoke as smoke
from . import forecast_tpg as task
from . import forecast_coreference_training as prior
from . import forecast_coreference_residual_v2 as residual

require, read_json, write_json, sha256 = cache.require, cache.read_json, cache.write_json, cache.sha256
initialize, epoch_orders, digest_state = residual.initialize, smoke.epoch_orders, smoke.digest_state
VERSION = 'r04_coreference_residual_development_v2'
VARIANTS = smoke.VARIANTS
DEFAULT_OUTPUT = task.RESULT_ROOT/'coreference_residual_development_v2'
SCRIPT = cache.base.EPSS_ROOT/'02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_residual_development.py'
TEST = SCRIPT.with_name('test_forecast_coreference_residual_development.py')
FIT_FILES = {'checkpoint.pt', 'history.json', 'gradients.json', 'predictions.json', 'metrics.json'}
METRICS = ('mae', 'rmse', 'mean_signed_error', 'spearman_rho', 'equal_group_mae')


PROTOCOL = residual.FULL_PROTOCOL
GraphBank = prior.GraphBank
history_check = prior.history_check
score_predictions = prior.score_predictions
atomic_torch_save = prior.atomic_torch_save
aggregate = prior.aggregate
plot = prior.plot


def settings_for(protocol, seed=None, gain=None):
    # The planned JSON is already pinned by smoke. Do not edit its historical status.
    require(protocol == residual.planned_full_protocol(), 'Fixed full protocol differs')
    settings = deepcopy(protocol['config'])
    if seed is not None:
        require(seed in settings['seeds'], 'Unregistered seed')
        settings['seeds'] = [seed]
    if gain is not None:
        require(type(gain) in (int, float) and math.isfinite(gain) and .001 <= gain <= .05, 'Invalid gain')
        settings['correction_gain'] = gain
    return settings


def context(directory):
    require(Path(directory).resolve() == (cache.ROOT/'full').resolve(), 'Use the fixed DQ17 v2 full cache')
    sm, graphs, sd, si = residual.context()
    residual.verify_output(residual.DEFAULT_OUTPUT, sm, graphs, sd, si)
    manifest = read_json(Path(directory)/'manifest.json')
    require({s: len(v) for s,v in manifest['ids'].items()} == {'train': 1023, 'validation': 197},
            'Full cohort differs')
    names = ('inputs.json', 'labels.json', 'split_manifest.json', 'task_cohorts.json')
    data = smoke.select_data(manifest, {n: read_json(cache.base.PREPARED/n) for n in names})
    protocol = read_json(PROTOCOL)
    settings_for(protocol)
    gain = residual.training_gain(manifest['ids']['train'], data['inputs.json'])
    config = settings_for(protocol, gain=gain['gain'])
    tqdm.write(f"Full training-input gain={gain['gain']:.10g}; CVEs={gain['n']}")
    # R03 comparisons must use the same IDs, targets and verified saved predictions.
    old_identity = read_json(prior.DEFAULT_OUTPUT/'run_identity.json')
    require(old_identity['ids'] == manifest['ids'] and
            old_identity['cache_lock_sha256'] == si['cache_lock_sha256'], 'R03 cohort/cache differs')
    cache.base.check_pins(old_identity['pinned_files'])
    prior.verify_output(prior.DEFAULT_OUTPUT, manifest, graphs, data, old_identity)
    pins = dict(si['pinned_files'])
    for directory_, lockname in ((residual.DEFAULT_OUTPUT, 'evaluation_lock.json'),
                                  (prior.DEFAULT_OUTPUT, 'evaluation_lock.json')):
        for name, digest in read_json(directory_/lockname)['files'].items():
            path = str((directory_/name).resolve())
            require(path not in pins or pins[path] == digest, 'Conflicting reference provenance')
            pins[path] = digest
        pins[str((directory_/lockname).resolve())] = sha256(directory_/lockname)
    for path in (Path(__file__), SCRIPT, TEST):
        pins[str(path.resolve())] = sha256(path)
    cache.base.check_pins(pins)
    identity = {'schema': VERSION, 'protocol': protocol, 'config': config, 'gain_fit': gain,
                'ids': manifest['ids'], 'cache_directory': str(Path(directory).resolve()),
                'cache_lock_sha256': si['cache_lock_sha256'], 'pinned_files': pins,
                'torch_version': str(torch.__version__), 'test_performance_evaluated': False,
                'limitations': si['limitations'], 'smoke_lock_sha256': sha256(residual.DEFAULT_OUTPUT/'evaluation_lock.json'),
                'r03_reference_results': read_json(prior.DEFAULT_OUTPUT/'summary.json')['results'],
                'implementation_status': 'full_runner_after_verified_smoke_review'}
    return manifest, graphs, data, identity


def fit(variant, initial, orders, manifest, graphs, data, config, progress=True):
    require(variant in VARIANTS and len(orders) == config['epochs'] and
            all(sorted(o) == sorted(manifest['ids']['train']) for o in orders), 'Invalid training schedule')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seeds'][0])
        net = residual.ResidualTPGRegressor(**initial['spec'])
        net.load_state_dict(initial['state_dict'], strict=True)
        require(prior.digest_state(net.state_dict()) == initial['initial_state_sha256'], 'Initialization differs')
        optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        criterion = torch.nn.SmoothL1Loss(beta=config['huber_beta'])
        history, best, best_mae, stale = [], None, math.inf, 0
        gradients = {k: {'steps_with_gradient': 0, 'steps_with_nonzero_gradient': 0, 'absolute_gradient_sum': 0.0}
                     for k in ('temporal', 'node_project', 'layers', 'head') if hasattr(net, k)}
        first_step = None
        for epoch, order in enumerate(tqdm(orders, desc=f'{variant}, seed {config["seeds"][0]}', unit='epoch', disable=not progress), 1):
            net.train()
            total, clipped_low, clipped_high = 0.0, 0, 0
            for start in tqdm(range(0, len(order), config['batch_size']), desc='Training batches', unit='batch', leave=False, disable=not progress):
                ids = order[start:start+config['batch_size']]
                batch = task.make_batch(ids, graphs, data['inputs.json'])
                y = torch.tensor([data['labels.json'][c]['y']['endpoint'] for c in ids], dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                _, _, raw, prediction = net.details(batch)
                loss = criterion(prediction, y)
                require(bool(torch.isfinite(loss)), 'Non-finite loss')
                loss.backward()
                snapshot = residual.gradient_snapshot(net)
                if first_step is None:
                    first_step = snapshot
                for k,v in snapshot.items():
                    gradients[k]['steps_with_gradient'] += int(v['has_gradient'])
                    gradients[k]['steps_with_nonzero_gradient'] += int(v['absolute_sum'] > 0)
                    gradients[k]['absolute_gradient_sum'] += v['absolute_sum']
                torch.nn.utils.clip_grad_norm_(net.parameters(), config['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach())*len(ids)
                clipped_low += int((raw.detach() < 0).sum()); clipped_high += int((raw.detach() > 1).sum())
            errors = {}
            for split in ('train', 'validation'):
                ids = manifest['ids'][split]
                pred = task.infer(net, ids, graphs, data['inputs.json'], config['batch_size'], split+' metrics', progress)
                errors[split] = task.regression_metrics([data['labels.json'][c]['y']['endpoint'] for c in ids], pred)['mae']
            history.append({'epoch': epoch, 'training_huber_loss': total/len(order), 'training_mae': errors['train'],
                'validation_mae': errors['validation'], 'training_examples': len(order),
                'training_clipped_below_zero': clipped_low, 'training_clipped_above_one': clipped_high})
            if errors['validation'] < best_mae:
                best_mae, stale = errors['validation'], 0
                best = {'schema': residual.VERSION, 'matched_experiment': VERSION, 'head_rule': residual.HEAD_RULE, 'spec': initial['spec'],
                    'seed': config['seeds'][0], 'edge_configuration': variant, 'selected_epoch': epoch,
                    'epoch_zero_eligible_for_selection': False, 'test_performance_evaluated': False,
                    'fit_cve_ids': manifest['ids']['train'], 'selection_cve_ids': manifest['ids']['validation'],
                    'initial_state_sha256': initial['initial_state_sha256'],
                    'state_dict': {k: v.detach().cpu().clone() for k,v in net.state_dict().items()}}
            else:
                stale += 1
            if progress:
                tqdm.write(f"{variant}, seed {config['seeds'][0]}, epoch {epoch}: training MAE={errors['train']:.8f}, validation MAE={errors['validation']:.8f}, clipped={clipped_low+clipped_high}/{len(order)}")
            if stale >= config['patience']:
                break
    require(best is not None, 'No selected checkpoint')
    return best, history, {'first_step_data_loss_gradients': first_step, 'all_steps': gradients}



def seed_artifacts(output, seed, manifest, graphs, data, settings, create=False):
    initial = {}
    for mode in ('tpg_epss', 'epss_only'):
        expected = initialize(mode, manifest, graphs['all_coref'], settings)
        path = output/'initialization'/f'seed_{seed}'/(mode+'.pt')
        if create and not path.exists():
            atomic_torch_save(path, expected)
        saved = torch.load(path, weights_only=True, map_location='cpu')
        require(saved['schema'] == residual.VERSION and saved['head_rule'] == residual.HEAD_RULE and
                saved['uses_target_statistics'] is False and saved['spec'] == expected['spec'] and
                digest_state(saved['state_dict']) == saved['initial_state_sha256'] == expected['initial_state_sha256'],
                'Saved initialization differs')
        initial[mode] = saved
    orders = epoch_orders(manifest['ids']['train'], settings)
    path = output/f'orders_seed_{seed}.json'
    if create and not path.exists():
        write_json(path, orders)
    require(read_json(path) == orders, 'Saved batch orders differ')
    checks = initialization_checks(initial, manifest, graphs, data)
    path = output/f'epoch_zero_seed_{seed}.json'
    if create and not path.exists():
        write_json(path, checks)
    require(read_json(path) == checks, 'Epoch-zero checks differ')
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
    net = residual.restore(saved)
    prefixes = {p for p in ('temporal', 'node_project', 'layers', 'head') if hasattr(net, p)}
    require(set(gradients) == {'first_step_data_loss_gradients', 'all_steps'} and
            set(gradients['all_steps']) == set(gradients['first_step_data_loss_gradients']) == prefixes,
            'Missing branch-gradient diagnostic')
    steps = len(history)*math.ceil(len(manifest['ids']['train'])/settings['batch_size'])
    for row in gradients['all_steps'].values():
        require(0 <= row['steps_with_nonzero_gradient'] <= row['steps_with_gradient'] <= steps and
                math.isfinite(row['absolute_gradient_sum']) and row['absolute_gradient_sum'] >= 0,
                'Invalid gradient diagnostic')
    for g in gradients['first_step_data_loss_gradients'].values():
        require(type(g['has_gradient']) is bool and math.isfinite(g['absolute_sum']) and g['absolute_sum'] >= 0,
                'Invalid first-step gradients')
    for row in history:
        require(row['training_examples'] == len(manifest['ids']['train']) and
                0 <= row['training_clipped_below_zero'] and 0 <= row['training_clipped_above_one'] and
                row['training_clipped_below_zero'] + row['training_clipped_above_one'] <= row['training_examples'],
                'Invalid training clipping')
    values = residual.infer_details(net, manifest['ids']['validation'], graphs, data['inputs.json'])
    expected_predictions = detail_rows(manifest, values)
    require(read_json(directory/'predictions.json') == expected_predictions, 'Saved predictions do not reproduce')
    metrics = {**score_predictions(values['prediction'], manifest, data), 'clipping': residual.clipping(values)}
    require(read_json(directory/'metrics.json') == metrics and
            history[saved['selected_epoch']-1]['validation_mae'] == metrics['metrics']['mae'],
            'Selected-checkpoint metrics differ')
    return {'variant': variant, 'seed': seed, 'selected_epoch': saved['selected_epoch'],
            'epochs_completed': len(history), 'budget_reached': len(history) == settings['epochs'],
            'selected_at_budget_boundary': saved['selected_epoch'] == settings['epochs'], **metrics}, values, history


def detail_rows(manifest, values):
    return [{'cve': c, **{k:v[i] for k,v in values.items()}}
            for i,c in enumerate(manifest['ids']['validation'])]


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


def collect(output, manifest, graphs, data, identity):
    ids = manifest['ids']['validation']
    persistence = [task.predict_persistence(data['inputs.json'][c])['prediction'] for c in ids]
    results = [{'variant': 'persistence', 'seed': None, 'selected_epoch': None,
                'epochs_completed': 0, **score_predictions(persistence, manifest, data)}]
    rows = [{'cve': c, 'actual_day60_epss': data['labels.json'][c]['y']['endpoint'],
             'prediction_anchor_date': data['inputs.json'][c]['input_end_date'],
             'target_date': data['labels.json'][c]['target_date'], 'persistence': p,
             'predictions': {}, 'residual_details': {}} for c,p in zip(ids, persistence)]
    histories = {}
    seeds = identity['config']['seeds']
    for seed in tqdm(seeds, desc='Verify seed results', unit='seed'):
        settings = deepcopy(identity['config']); settings['seeds'] = [seed]
        initial, _ = seed_artifacts(output, seed, manifest, graphs, data, settings)
        for variant in VARIANTS:
            mode = 'epss_only' if variant == 'epss_only' else 'tpg_epss'
            result, pred, history = check_fit(fit_directory(output, seed, variant), variant, seed,
                initial[mode], manifest, graphs[variant], data, settings, identity)
            results.append(result)
            key = f'{variant}__{seed}'; histories[key] = history
            for i,row in enumerate(rows):
                row['predictions'][key] = pred['prediction'][i]
                row['residual_details'][key] = {k:v[i] for k,v in pred.items()}
    aggregated, pairs = aggregate(results, seeds)
    matched_r03 = compare_r03(results, identity['r03_reference_results'], seeds)
    summary = {'schema': VERSION, 'identity': identity, 'results': results,
               'matched_r03_differences': matched_r03,
               'seed_aggregates': aggregated, 'paired_mae_differences': pairs,
               'test_performance_evaluated': False}
    return summary, rows, histories


def compare_r03(results, reference, seeds):
    lookup = {(r['variant'], r['seed']): r for r in reference}
    expected = {(v,s) for v in VARIANTS for s in seeds} | {('persistence', None)}
    require(set(lookup) == expected and len(reference) == len(expected), 'Incomplete R03 reference')
    require(len(results) == len(expected) and {(r['variant'],r['seed']) for r in results} == expected,
            'Incomplete residual comparison')
    return [{'variant': r['variant'], 'seed': r['seed'], 'reference': 'R03_same_variant_same_seed',
             'mae_difference': r['metrics']['mae']-lookup[(r['variant'],r['seed'])]['metrics']['mae'],
             'rmse_difference': r['metrics']['rmse']-lookup[(r['variant'],r['seed'])]['metrics']['rmse']}
            for r in results if r['variant'] != 'persistence']


def report(summary):
    lines = ['R04 v2: Scaled-Residual Full Development Experiment', '',
        'EPSS Days 0-30 and current-description TPGs predict stored Day-60 EPSS.',
        'This is retrospective validation, not exploitation prediction or test performance.',
        f"CVEs by role: { {s:len(v) for s,v in summary['identity']['ids'].items()} }.",
        f"Training-input correction gain: {summary['identity']['gain_fit']['gain']:.10g}.",
        'Gain uses only the median latest observed past training score, bounded to [0.001,0.05].',
        'Prediction = clip(latest observed input EPSS + tanh(gain * correction head), 0, 1).',
        'The gain scales initial response, not a permanent small cap. Clipping remains possible.',
        'Four variants, seeds 17/29/43, up to 100 epochs, patience 15; earliest best trained epoch.',
        'Epoch zero equals float32 persistence but cannot be selected. No warm start from smoke.',
        'All graph variants share seed-specific initialization and batch orders.',
        'Full means the existing eligible development cohort, not all 5,968 source CVEs.', '',
        'Model | Seed | Selected epoch | Epochs run | Validation MAE | RMSE | Clipped below zero | Clipped above one']
    for r in summary['results']:
        m, c = r['metrics'], r.get('clipping', {})
        lines.append(f"{r['variant']} | {r['seed']} | {r['selected_epoch']} | {r['epochs_completed']} | "
                     f"{m['mae']:.8f} | {m['rmse']:.8f} | {c.get('below_zero','-')} | {c.get('above_one','-')}")
    lines += ['', 'Model | Three-seed mean validation MAE | Sample standard deviation']
    for mode, values in summary['seed_aggregates'].items():
        m = values['mae']
        lines.append(f"{mode} | {m['mean']:.8f} | {m['sample_std']}")
    lines += ['', 'MAE is average absolute error; RMSE weights larger errors more. Lower is better.',
        'Clipping counts refer to selected validation raw outputs before clipping to [0,1].',
        'Training clipping counts are recorded for each epoch before optimizer updates.',
        'Every CVE prediction retains persistence, correction, raw score and final score.',
        'First-step upstream data-loss gradients are zero by design; later gradient flow is recorded.',
        'Gradients do not establish useful or causal branch contribution.',
        'Missingness strata and equal-group metrics remain available; small/empty groups are explicit.',
        'All 12 fits are reported, not the best seed. Paired differences use identical validation CVEs.',
        'matched_r03_differences compares the same variant and seed with frozen R03; negative favors R04.',
        'Do not compare full metrics directly with the 24-CVE smoke metrics.',
        'Full scale is fitted once on training inputs only, not validation scores.',
        'Completed fits are never overwritten. Resume skips verified fits; an interrupted fit restarts.',
        'No test scoring, new embeddings, new graphs, split changes or hyperparameter search.', '',
        *summary['identity']['limitations'], '']
    return '\n'.join(lines)


def expected_files(seeds):
    names = {'run_identity.json', 'run_status.json', 'summary.json', 'summary.txt',
             'predictions.json', 'training_history.json', 'training_curves.png'}
    for seed in seeds:
        names.add(f'orders_seed_{seed}.json')
        names.add(f'epoch_zero_seed_{seed}.json')
        names.update(f'initialization/seed_{seed}/{mode}.pt' for mode in ('tpg_epss', 'epss_only'))
        names.update(f'fits/{v}__{seed}/{n}' for v in VARIANTS for n in FIT_FILES | {'fit_lock.json'})
    return names


def verify_output(output, manifest, graphs, data, identity):
    lock = read_json(output/'evaluation_lock.json')
    expected = expected_files(identity['config']['seeds'])
    require(lock['schema'] == VERSION and set(lock['files']) == expected, 'Wrong result file contract')
    require({str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()} ==
            expected | {'evaluation_lock.json'}, 'Unexpected result files')
    for name, digest in tqdm(lock['files'].items(), desc='Verify result checksums', unit='file'):
        require(sha256(output/name) == digest, f'Changed result: {name}')
    require(read_json(output/'run_identity.json') == identity and
            read_json(output/'run_status.json')['status'] == 'complete', 'Run identity/status differs')
    require(identity['gain_fit'] == residual.training_gain(manifest['ids']['train'], data['inputs.json']) and
            identity['config']['correction_gain'] == identity['gain_fit']['gain'], 'Full gain differs')
    summary, predictions, histories = collect(output, manifest, graphs, data, identity)
    require(read_json(output/'summary.json') == summary and read_json(output/'predictions.json') == predictions and
            read_json(output/'training_history.json') == histories, 'Results do not reproduce')
    require((output/'summary.txt').read_text() == report(summary), 'Text report differs')
    cache.base.check_pins(identity['pinned_files'])


def initialization_checks(initial, manifest, graphs, data):
    # Check small and largest graphs without reading target values.
    selected = {}
    sizes = {r['cve']: r['nodes'] for r in manifest.get('graphs', [])}
    for split in ('train', 'validation'):
        ids = manifest['ids'][split]
        ranked = sorted(ids, key=lambda c: (-(sizes[c] if c in sizes else len(graphs['all_coref'][c]['x'])), c))
        selected[split] = list(dict.fromkeys(ids[:2]+ranked[:initial['epss_only']['spec']['config']['batch_size']]))
    return residual.initial_checks(initial, dict(manifest, ids=selected), graphs, data)


def preflight(manifest, graphs, data, settings):
    initial = {m: initialize(m, manifest, graphs['all_coref'], settings) for m in ('tpg_epss', 'epss_only')}
    initialization_checks(initial, manifest, graphs, data)
    return initial


def run(directory, output, check_only=False, verify_only=False, resume=False):
    output = Path(output).resolve()
    manifest, graphs, data, identity = context(directory)
    seeds = identity['config']['seeds']
    require(output.parent == task.RESULT_ROOT.resolve(), 'Use a new direct child of the evaluation-results directory')
    if verify_only:
        verify_output(output, manifest, graphs, data, identity)
        print('R04 v2 full all seeds, checkpoints and validation predictions verified. No retraining or test scoring.'); return
    if resume:
        require(output.is_dir() and not (output/'evaluation_lock.json').exists(),
                'Resume needs unfinished output; use --verify-only for completed results')
        require(read_json(output/'run_identity.json') == identity, 'Resume sources/settings changed')
        allowed = expected_files(seeds)
        actual = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}
        require(actual <= allowed | {n+'.tmp' for n in allowed}, 'Unexpected unfinished result files')
    else:
        cache.base.check_destination(output, [cache.ROOT, cache.base.PREPARED, cache.base.EPSS_ROOT/'02_prediction'])
    settings = deepcopy(identity['config']); settings['seeds'] = [seeds[0]]
    preflight(manifest, graphs, data, settings)
    cache.base.check_pins(identity['pinned_files'])
    if check_only:
        print('R04 v2 full preflight passed: persistence on small/largest graphs; no new fitting or output writes. Frozen validation references verified.'); return
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/'run_identity.json', identity)
    write_json(output/'run_status.json', {'status': 'running'})
    try:
        with tqdm(total=len(seeds)*len(VARIANTS), desc='Matched fixed-seed fits', unit='fit') as progress:
            for seed in seeds:
                settings = deepcopy(identity['config']); settings['seeds'] = [seed]
                initial, orders = seed_artifacts(output, seed, manifest, graphs, data, settings, create=True)
                for variant in VARIANTS:
                    path = fit_directory(output, seed, variant)
                    mode = 'epss_only' if variant == 'epss_only' else 'tpg_epss'
                    if not (path/'fit_lock.json').exists():
                        saved, history, gradients = fit(variant, initial[mode], orders, manifest,
                                                       graphs[variant], data, settings)
                        saved['cache_lock_sha256'] = identity['cache_lock_sha256']
                        values = residual.infer_details(residual.restore(saved), manifest['ids']['validation'],
                            graphs[variant], data['inputs.json'])
                        rows = detail_rows(manifest, values)
                        metrics = {**score_predictions(values['prediction'], manifest, data),
                                   'clipping': residual.clipping(values)}
                        save_fit(path, saved, history, gradients, rows, metrics)
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
            require({str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()} == expected_files(seeds),
                    'Unexpected files before final lock')
            write_json(output/'evaluation_lock.json', {'schema': VERSION, 'files': {
                str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}})
            bar.update()
        verify_output(output, manifest, graphs, data, identity)
    except Exception as exc:
        if not (output/'evaluation_lock.json').exists():
            write_json(output/'run_status.json', {'status': 'failed', 'error': str(exc)})
        raise
    print(f'R04 v2 full development experiment saved to {output}. All fixed seeds; validation only, no test scoring.')


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
