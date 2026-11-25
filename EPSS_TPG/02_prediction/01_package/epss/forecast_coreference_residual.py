"""R04 smoke: matched TPG/EPSS persistence residuals using frozen graph caches."""
from copy import deepcopy
import argparse
import math
from pathlib import Path

import torch
from torch_geometric.nn import global_mean_pool
from tqdm.auto import tqdm

from . import forecast_coreference_training as prior

task, cache, smoke = prior.task, prior.cache, prior.smoke
require, read, write, sha256 = prior.require, prior.read_json, prior.write_json, prior.sha256
VERSION = 'r04_coreference_residual_smoke_v1'
HEAD_RULE = 'clip(last_observed_input_score + tanh(zero_initialized_final_linear(concatenated_branch_MLP)), 0, 1)'
PROTOCOL = cache.base.EPSS_ROOT/'02_prediction/04_configs/coreference_residual_smoke_v1.json'
SCRIPT = cache.base.EPSS_ROOT/'02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_residual.py'
TEST = SCRIPT.with_name('test_forecast_coreference_residual.py')
DEFAULT_OUTPUT = task.RESULT_ROOT/'coreference_residual_smoke_v1'
DETAILS = ('persistence_float32', 'correction', 'raw_prediction', 'prediction')
VARIANTS = prior.VARIANTS


def settings():
    protocol = read(PROTOCOL)
    require(protocol['schema'] == VERSION and protocol['scope'] == 'smoke_only' and
            protocol['head_rule'] == HEAD_RULE and protocol['epoch_zero_eligible_for_selection'] is False,
            'Wrong residual protocol')
    require(protocol['config'] == smoke.config() and protocol['variants'] == list(VARIANTS) and
            protocol['execution'] == {'torch_threads': 1, 'deterministic_algorithms': True}, 'Fixed settings differ')
    require(protocol['cohort'] == {'train': 32, 'validation': 24,
            'selection': 'unchanged_DQ16_ids_from_verified_DQ17_v2_cache'}, 'Fixed cohort differs')
    return protocol


def last_observed(sequence):
    require(sequence.ndim == 3 and tuple(sequence.shape[1:]) == (31, len(task.sequence.CHANNELS)), 'Invalid sequence shape')
    observed = sequence[:, :, 1] == 1
    require(bool(observed.any(dim=1).all()), 'No observed past EPSS')
    days = torch.arange(31, device=sequence.device).expand(len(sequence), -1)
    indexes = days.masked_fill(~observed, -1).max(dim=1).values
    base = sequence[:, :, 0].gather(1, indexes[:, None]).squeeze(1)
    require(bool(torch.isfinite(base).all()) and bool(((base >= 0) & (base <= 1)).all()), 'Invalid persistence score')
    return base


class ResidualTPGRegressor(task.TPGRegressor):
    def __init__(self, **spec):
        require(spec['mode'] in ('tpg_epss', 'epss_only'), 'Unsupported residual mode')
        super().__init__(**spec)
        torch.nn.init.zeros_(self.head[-1].weight)
        torch.nn.init.zeros_(self.head[-1].bias)

    def representation(self, batch):
        # Keep the frozen R03 encoders and pooling; only the output rule changes.
        _, hidden = self.temporal(batch.epss_sequence)
        parts = [hidden[-1]]
        if self.mode == 'tpg_epss':
            x = self.node_project(batch.x)
            for layer in self.layers:
                x = layer(x, batch.edge_index, batch.edge_type, batch.edge_features)
            parts.append(global_mean_pool(x, batch.batch))
        return torch.cat(parts, dim=1)

    def details(self, batch):
        base = last_observed(batch.epss_sequence)
        correction = torch.tanh(self.head(self.representation(batch))).squeeze(-1)
        raw = base + correction
        return base, correction, raw, raw.clamp(0, 1)

    def forward(self, batch):
        return self.details(batch)[-1]


def initialize(mode, manifest, graphs, config):
    spec = task.model_spec(mode, manifest, graphs, config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seeds'][0])
        net = ResidualTPGRegressor(**spec)
    state = {k: v.detach().cpu().clone() for k,v in net.state_dict().items()}
    return {'schema': VERSION, 'spec': spec, 'head_rule': HEAD_RULE, 'state_dict': state,
            'initial_state_sha256': prior.digest_state(state), 'uses_target_statistics': False}


def restore(saved):
    require(saved['schema'] == VERSION and saved['head_rule'] == HEAD_RULE and
            saved['test_performance_evaluated'] is False and saved['epoch_zero_eligible_for_selection'] is False,
            'Wrong residual checkpoint contract')
    require(type(saved['selected_epoch']) is int and saved['selected_epoch'] >= 1, 'Epoch zero cannot be selected')
    with torch.random.fork_rng(devices=[]):
        net = ResidualTPGRegressor(**saved['spec'])
    require(all(bool(torch.isfinite(v).all()) for v in saved['state_dict'].values()), 'Non-finite checkpoint')
    net.load_state_dict(saved['state_dict'], strict=True)
    return net.eval()


def infer_details(net, ids, graphs, inputs, progress=True):
    net.eval()
    values = {name: [] for name in DETAILS}
    with torch.no_grad():
        for start in tqdm(range(0, len(ids), net.config['batch_size']), desc='Residual prediction batches',
                          unit='batch', leave=False, disable=not progress):
            batch = task.make_batch(ids[start:start+net.config['batch_size']], graphs, inputs)
            output = net.details(batch)
            require(all(bool(torch.isfinite(v).all()) for v in output), 'Non-finite prediction')
            require(bool(((output[-1] >= 0) & (output[-1] <= 1)).all()), 'Unbounded prediction')
            for name, tensor in zip(DETAILS, output):
                values[name].extend(tensor.tolist())
    return values


def gradient_snapshot(net):
    result = {}
    for prefix in ('temporal', 'node_project', 'layers', 'head'):
        if not hasattr(net, prefix):
            continue
        values = [p.grad for n,p in net.named_parameters() if n.startswith(prefix+'.') and p.grad is not None]
        require(all(bool(torch.isfinite(v).all()) for v in values), 'Non-finite gradient')
        result[prefix] = {'has_gradient': bool(values), 'absolute_sum': sum(float(v.abs().sum()) for v in values)}
    return result


def fit(variant, initial, orders, manifest, graphs, data, config, progress=True):
    require(variant in VARIANTS and len(orders) == config['epochs'] and
            all(sorted(o) == sorted(manifest['ids']['train']) for o in orders), 'Invalid training schedule')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seeds'][0])
        net = ResidualTPGRegressor(**initial['spec'])
        net.load_state_dict(initial['state_dict'], strict=True)
        require(prior.digest_state(net.state_dict()) == initial['initial_state_sha256'], 'Initialization differs')
        optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        criterion = torch.nn.SmoothL1Loss(beta=config['huber_beta'])
        history, best, best_mae = [], None, math.inf
        gradients = {k: {'steps_with_gradient': 0, 'steps_with_nonzero_gradient': 0, 'absolute_gradient_sum': 0.0}
                     for k in ('temporal', 'node_project', 'layers', 'head') if hasattr(net, k)}
        first_step = None
        for epoch, order in enumerate(tqdm(orders, desc=variant, unit='epoch', disable=not progress), 1):
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
                snapshot = gradient_snapshot(net)
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
                best_mae = errors['validation']
                best = {'schema': VERSION, 'head_rule': HEAD_RULE, 'spec': initial['spec'],
                    'seed': config['seeds'][0], 'edge_configuration': variant, 'selected_epoch': epoch,
                    'epoch_zero_eligible_for_selection': False, 'test_performance_evaluated': False,
                    'fit_cve_ids': manifest['ids']['train'], 'selection_cve_ids': manifest['ids']['validation'],
                    'initial_state_sha256': initial['initial_state_sha256'],
                    'state_dict': {k: v.detach().cpu().clone() for k,v in net.state_dict().items()}}
            if progress:
                tqdm.write(f"{variant}, epoch {epoch}: training MAE={errors['train']:.8f}, validation MAE={errors['validation']:.8f}, clipped={clipped_low+clipped_high}/{len(order)}")
    require(best is not None, 'No selected checkpoint')
    return best, history, {'first_step_data_loss_gradients': first_step, 'all_steps': gradients}


def clipping(values):
    return {'below_zero': sum(v < 0 for v in values['raw_prediction']),
            'above_one': sum(v > 1 for v in values['raw_prediction']),
            'mean_absolute_correction': sum(abs(v) for v in values['correction'])/len(values['correction'])}


def evaluate(models, manifest, graphs, data):
    ids = manifest['ids']['validation']
    actual = [data['labels.json'][c]['y']['endpoint'] for c in ids]
    baseline = [task.predict_persistence(data['inputs.json'][c])['prediction'] for c in ids]
    results = [{'variant': 'persistence', 'selected_epoch': None, **prior.score_predictions(baseline, manifest, data)}]
    rows = [{'cve': c, 'actual_day60_epss': y, 'persistence': p,
             'prediction_anchor_date': data['inputs.json'][c]['input_end_date'],
             'target_date': data['labels.json'][c]['target_date'], 'models': {}} for c,y,p in zip(ids, actual, baseline)]
    for variant in tqdm(VARIANTS, desc='Reload and evaluate selected checkpoints', unit='model'):
        saved = models[variant]
        require(saved['edge_configuration'] == variant and saved['fit_cve_ids'] == manifest['ids']['train'] and
                saved['selection_cve_ids'] == ids, 'Checkpoint cohort differs')
        values = infer_details(restore(saved), ids, graphs[variant], data['inputs.json'])
        results.append({'variant': variant, 'selected_epoch': saved['selected_epoch'],
                        **prior.score_predictions(values['prediction'], manifest, data), 'clipping': clipping(values)})
        for i,row in enumerate(rows):
            row['models'][variant] = {k:v[i] for k,v in values.items()}
    return rows, results


def initial_checks(initial, manifest, graphs, data, progress=True):
    checks = {}
    for variant in tqdm(VARIANTS, desc='Check exact persistence initialization', unit='model', disable=not progress):
        mode = 'epss_only' if variant == 'epss_only' else 'tpg_epss'
        with torch.random.fork_rng(devices=[]):
            net = ResidualTPGRegressor(**initial[mode]['spec'])
        net.load_state_dict(initial[mode]['state_dict'])
        checks[variant] = {}
        for split in ('train', 'validation'):
            ids = manifest['ids'][split]
            values = infer_details(net, ids, graphs[variant], data['inputs.json'], progress)
            require(values['prediction'] == values['persistence_float32'] and all(v == 0 for v in values['correction']),
                    'Initial predictions must equal float32 persistence exactly')
            baseline = [task.predict_persistence(data['inputs.json'][c])['prediction'] for c in ids]
            error = max(abs(a-b) for a,b in zip(baseline, values['prediction']))
            require(error <= 6e-8, 'Initial baseline differs beyond float32 rounding')
            checks[variant][split] = {'n': len(ids), 'exact_float32_persistence': True,
                                      'maximum_source_precision_difference': error}
    return checks


def context():
    protocol = settings()
    manifest, graphs, data, parent = prior.context(cache.ROOT/'full')
    original = read(cache.shared.DEFAULT_OUTPUT/'manifest.json')
    ids = original['ids']
    require({s: len(v) for s,v in ids.items()} == {'train': 32, 'validation': 24} and
            all(set(ids[s]) <= set(manifest['ids'][s]) for s in ids), 'Smoke IDs outside fixed development split')
    manifest = dict(manifest, ids=ids)
    data = smoke.select_data(manifest, {**data, 'task_cohorts.json': {'endpoint': parent['ids']}})
    pins = dict(parent['pinned_files'])
    # Pin completed DQ18 evidence without rerunning interventions or training.
    dq18 = task.RESULT_ROOT/'coreference_branch_diagnostic_v1/full'
    d_identity = read(dq18/'identity.json')
    require(d_identity['ids'] == parent['ids']['validation'] and d_identity['schema'] == 'dq18_r03_branch_response_v1', 'Wrong DQ18 cohort')
    for path, digest in d_identity['pinned_files'].items():
        require(path not in pins or pins[path] == digest, 'Conflicting DQ18 and cache source pins')
        pins[path] = digest
    lock = read(dq18/'diagnostic_lock.json')
    require(lock['schema'] == 'dq18_r03_branch_response_v1', 'Wrong DQ18 lock')
    pins.update({str(dq18/n): h for n,h in lock['files'].items()})
    for path in (dq18/'diagnostic_lock.json', PROTOCOL, Path(__file__), SCRIPT, TEST):
        pins[str(path.resolve())] = sha256(path)
    cache.base.check_pins(pins)
    identity = {'schema': VERSION, 'protocol': protocol, 'config': protocol['config'], 'ids': ids,
        'cache_directory': parent['cache_directory'], 'cache_lock_sha256': parent['cache_lock_sha256'],
        'pinned_files': pins, 'torch_version': str(torch.__version__), 'test_performance_evaluated': False,
        'limitations': parent['protocol']['limitations'], 'full_expansion_authorized': False}
    return manifest, graphs, data, identity


def report(summary):
    lines = ['R04: Persistence-Residual TPG Smoke', '',
        '32 training / 24 validation CVEs; seed 17; three trained epochs; four matched models.',
        'This is an engineering smoke check, not an estimate from the full 197-CVE validation cohort.',
        'Inputs: current-description graph and observed EPSS Days 0-30, masks and observation age.',
        'Target: stored Day-60 EPSS. No exploitation labels, new embeddings, graph expansion or test scoring.',
        'Output = clip(latest observed input score + tanh(correction head), 0, 1).',
        'The final linear layer starts at zero. Epoch zero equals float32 persistence exactly.',
        'Source-score precision differs from float32 only by rounding; the source-precision reference stays separate.',
        'Earliest lowest validation MAE among epochs 1-3 selects the checkpoint. Epoch zero is diagnostic only.', '',
        'Model | Selected epoch | Validation MAE | RMSE | Clipped below 0 | Clipped above 1']
    for r in summary['results']:
        counts = r.get('clipping', {})
        lines.append(f"{r['variant']} | {r['selected_epoch']} | {r['metrics']['mae']:.8f} | {r['metrics']['rmse']:.8f} | "
                     f"{counts.get('below_zero', '-')} | {counts.get('above_one', '-')}")
    lines += ['', 'MAE is average absolute error; 0.01 is one EPSS percentage point. Lower MAE/RMSE is better.',
        'Clipping counts refer to raw selected-checkpoint outputs outside [0,1], before final clipping.',
        'Raw predictions, corrections and float32 persistence are retained per CVE in predictions.json.',
        'history.json records every epoch and training-time clipping before each optimizer update.',
        'gradients.json records the first data-loss gradient and accumulated gradient counts.',
        'At initialization the zero output weights block upstream data-loss gradients on the first batch.',
        'The final layer can learn first, then upstream gradients can flow. This is expected, not a disconnected encoder.',
        'Adam weight decay can still update parameters even when their data-loss gradient is zero.',
        'Hard clipping has zero gradient outside [0,1]; clipping diagnostics must be reviewed.',
        'The graph modes share identical initial states and batch orders. EPSS-only is separately initialized.',
        'Epoch-zero checks use no labels. Actual scores enter the loss and evaluation only.',
        'No guarantee of beating persistence: earlier temporal-only residual models did not.',
        'Do not select a graph winner from this short, small run or compare it directly with full-cohort R03 metrics.', '',
        *summary['identity']['limitations'], '']
    return '\n'.join(lines)


def expected_files():
    return {'summary.json', 'summary.txt', 'predictions.json', 'history.json', 'gradients.json',
            'epoch_zero.json', 'batch_orders.json', 'training_curves.png', 'run_identity.json', 'run_status.json'} | {
            f'checkpoints/{v}.pt' for v in VARIANTS} | {f'initialization/{m}.pt' for m in ('tpg_epss', 'epss_only')}


def verify_output(output, manifest, graphs, data, identity):
    lock = read(output/'evaluation_lock.json')
    require(lock['schema'] == VERSION and set(lock['files']) == expected_files(), 'Wrong output contract')
    require({str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()} == expected_files() | {'evaluation_lock.json'}, 'Unexpected output files')
    for name, digest in tqdm(lock['files'].items(), desc='Verify R04 artifacts', unit='file'):
        require(sha256(output/name) == digest, 'Changed artifact: '+name)
    require(read(output/'run_identity.json') == identity and read(output/'run_status.json')['status'] == 'complete', 'Run identity/status differs')
    config = identity['config']
    require(read(output/'batch_orders.json') == smoke.epoch_orders(manifest['ids']['train'], config), 'Changed batch orders')
    initial = {}
    for mode in ('tpg_epss', 'epss_only'):
        saved = torch.load(output/'initialization'/(mode+'.pt'), map_location='cpu', weights_only=True)
        expected = initialize(mode, manifest, graphs['all_coref'], config)
        require(saved['schema'] == VERSION and saved['head_rule'] == HEAD_RULE and saved['uses_target_statistics'] is False and
                saved['spec'] == expected['spec'] and prior.digest_state(saved['state_dict']) ==
                saved['initial_state_sha256'] == expected['initial_state_sha256'], 'Initial state differs')
        initial[mode] = saved
    require(read(output/'epoch_zero.json') == initial_checks(initial, manifest, graphs, data), 'Epoch zero differs')
    models, histories, grads = {}, read(output/'history.json'), read(output/'gradients.json')
    require(set(histories) == set(grads) == set(VARIANTS), 'Missing model diagnostics')
    for variant in VARIANTS:
        saved = torch.load(output/'checkpoints'/(variant+'.pt'), map_location='cpu', weights_only=True)
        mode = 'epss_only' if variant == 'epss_only' else 'tpg_epss'
        require(saved['spec'] == initial[mode]['spec'] and saved['seed'] == config['seeds'][0] and
                saved['initial_state_sha256'] == initial[mode]['initial_state_sha256'] and
                saved['cache_lock_sha256'] == identity['cache_lock_sha256'], 'Checkpoint source differs')
        h = histories[variant]
        require([r['epoch'] for r in h] == list(range(1, config['epochs']+1)), 'Wrong epoch count')
        require(all(math.isfinite(r[k]) and r[k] >= 0 for r in h for k in ('training_mae','validation_mae','training_huber_loss')), 'Invalid history')
        require(saved['selected_epoch'] == min(h, key=lambda r:r['validation_mae'])['epoch'], 'Selection differs')
        steps = config['epochs']*math.ceil(len(manifest['ids']['train'])/config['batch_size'])
        for g in grads[variant]['all_steps'].values():
            require(0 <= g['steps_with_nonzero_gradient'] <= g['steps_with_gradient'] <= steps and
                    math.isfinite(g['absolute_gradient_sum']) and g['absolute_gradient_sum'] >= 0, 'Invalid gradient counters')
        for r in h:
            require(r['training_examples'] == len(manifest['ids']['train']) and
                    0 <= r['training_clipped_below_zero']+r['training_clipped_above_one'] <= r['training_examples'], 'Invalid clipping count')
        models[variant] = saved
    rows, results = evaluate(models, manifest, graphs, data)
    summary = {'schema': VERSION, 'identity': identity, 'results': results,
               'smoke_only': True, 'test_performance_evaluated': False}
    require(read(output/'predictions.json') == rows and read(output/'summary.json') == summary and
            (output/'summary.txt').read_text() == report(summary), 'Saved outputs do not reproduce')
    for r in results[1:]:
        require(histories[r['variant']][r['selected_epoch']-1]['validation_mae'] == r['metrics']['mae'], 'Checkpoint/history mismatch')
    cache.base.check_pins(identity['pinned_files'])


def execute(output, check_only=False, verify_only=False):
    manifest, graphs, data, identity = context()
    output = Path(output).resolve()
    require(output.parent == task.RESULT_ROOT.resolve(), 'Use a new direct child of the evaluation-results directory')
    if verify_only:
        verify_output(output, manifest, graphs, data, identity)
        print('R04 residual smoke checkpoints and predictions verified. No retraining or test scoring.'); return
    require(not output.exists(), 'Output exists; use --verify-only')
    config = identity['config']
    initial = {m: initialize(m, manifest, graphs['all_coref'], config) for m in ('tpg_epss', 'epss_only')}
    zero = initial_checks(initial, manifest, graphs, data)
    cache.base.check_pins(identity['pinned_files'])
    if check_only:
        print('R04 persistence initialization preflight passed. No optimizer steps, label scoring or output writes.'); return
    output.mkdir(parents=True, exist_ok=False)
    write(output/'run_identity.json', identity)
    write(output/'run_status.json', {'status': 'running'})
    try:
        orders = smoke.epoch_orders(manifest['ids']['train'], config)
        write(output/'batch_orders.json', orders); write(output/'epoch_zero.json', zero)
        for mode, state in initial.items():
            prior.atomic_torch_save(output/'initialization'/(mode+'.pt'), state)
        models, histories, grads = {}, {}, {}
        for variant in tqdm(VARIANTS, desc='Matched residual smoke fits', unit='model'):
            mode = 'epss_only' if variant == 'epss_only' else 'tpg_epss'
            saved, history, gradient = fit(variant, initial[mode], orders, manifest, graphs[variant], data, config)
            saved['cache_lock_sha256'] = identity['cache_lock_sha256']
            prior.atomic_torch_save(output/'checkpoints'/(variant+'.pt'), saved)
            models[variant], histories[variant], grads[variant] = saved, history, gradient
        rows, results = evaluate(models, manifest, graphs, data)
        summary = {'schema': VERSION, 'identity': identity, 'results': results, 'smoke_only': True, 'test_performance_evaluated': False}
        with tqdm(total=4, desc='Save reports and provenance', unit='stage') as bar:
            for name, values in (('summary', summary), ('predictions', rows), ('history', histories), ('gradients', grads)):
                write(output/(name+'.json'), values)
            (output/'summary.txt').write_text(report(summary)); bar.update()
            smoke.plot(histories, output/'training_curves.png'); bar.update()
            cache.base.check_pins(identity['pinned_files']); bar.update()
            write(output/'run_status.json', {'status': 'complete'})
            write(output/'evaluation_lock.json', {'schema': VERSION, 'files': {n: sha256(output/n) for n in sorted(expected_files())}}); bar.update()
        verify_output(output, manifest, graphs, data, identity)
    except Exception as exc:
        # Preserve completed locked artifacts if post-save verification fails.
        if not (output/'evaluation_lock.json').exists():
            write(output/'run_status.json', {'status': 'failed', 'error': str(exc)})
        raise
    print(f'R04 residual smoke saved to {output}. No full training expansion or test scoring.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument('--check-only', action='store_true')
    flags.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    threads = torch.get_num_threads()
    deterministic, warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        execute(args.output_dir, args.check_only, args.verify_only)
    finally:
        torch.set_num_threads(threads); torch.use_deterministic_algorithms(deterministic, warn_only=warn)


if __name__ == '__main__':
    main()
