"""R04 v2: train-input-scaled residual smoke; frozen v1 remains unchanged.

The versioned orchestration preserves v1 contracts; encoder and diagnostic helpers
are reused without changing or monkey-patching the frozen implementation.
"""
import statistics
import argparse
import math
from pathlib import Path

import torch
from tqdm.auto import tqdm

from . import forecast_coreference_training as prior
from . import forecast_coreference_residual as v1

task, cache, smoke = prior.task, prior.cache, prior.smoke
require, read, write, sha256 = prior.require, prior.read_json, prior.write_json, prior.sha256
VERSION = 'r04_coreference_residual_smoke_v2'
HEAD_RULE = 'clip(last_observed_input_score + tanh(training_input_gain * zero_initialized_final_linear(concatenated_branch_MLP)), 0, 1)'
GAIN_RULE = 'clip(median(latest_observed_EPSS_per_training_CVE), 0.001, 0.05)'
FULL_PROTOCOL = cache.base.EPSS_ROOT/'02_prediction/04_configs/coreference_residual_development_v2.json'
PROTOCOL = cache.base.EPSS_ROOT/'02_prediction/04_configs/coreference_residual_smoke_v2.json'
SCRIPT = cache.base.EPSS_ROOT/'02_prediction/03_analysis/01_revision_dataset/run_forecast_coreference_residual_v2.py'
TEST = SCRIPT.with_name('test_forecast_coreference_residual_v2.py')
DEFAULT_OUTPUT = task.RESULT_ROOT/'coreference_residual_smoke_v2'
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
    require(protocol['gain_rule'] == GAIN_RULE, 'Wrong gain rule')
    require(protocol['future_development_plan']['protocol'] == FULL_PROTOCOL.name, 'Wrong full plan')
    return protocol


last_observed = v1.last_observed
infer_details = v1.infer_details
gradient_snapshot = v1.gradient_snapshot
clipping = v1.clipping


def training_gain(ids, inputs):
    """One vote per training CVE; no labels, validation or test values."""
    require(bool(ids) and len(set(ids)) == len(ids), 'Empty or duplicate training IDs')
    values = []
    for cve in tqdm(ids, desc='Derive gain from training history', unit='CVE', leave=False):
        cache.base.contract.predictor_view(inputs[cve])
        value = task.predict_persistence(inputs[cve])['prediction']
        require(math.isfinite(value) and 0 <= value <= 1, 'Invalid past EPSS')
        values.append(value)
    median = statistics.median(values)
    return {'rule': GAIN_RULE, 'fit_cve_ids': list(ids), 'n': len(ids),
            'median_latest_observed_epss': median, 'gain': min(.05, max(.001, median)),
            'uses_target_statistics': False, 'uses_validation_inputs': False}


class ResidualTPGRegressor(v1.ResidualTPGRegressor):
    def __init__(self, **spec):
        gain = spec['config'].get('correction_gain')
        require(type(gain) in (int, float) and math.isfinite(gain) and .001 <= gain <= .05,
                'Invalid or missing correction gain')
        super().__init__(**spec)
        self.correction_gain = float(gain)

    def details(self, batch):
        base = last_observed(batch.epss_sequence)
        # Scale inside tanh: smaller local response, not a small hard output cap.
        correction = torch.tanh(self.correction_gain * self.head(self.representation(batch))).squeeze(-1)
        raw = base + correction
        return base, correction, raw, raw.clamp(0, 1)


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
    full_plan = read(FULL_PROTOCOL)
    require(full_plan == planned_full_protocol(), 'Full protocol changed')
    manifest, graphs, data, parent = v1.context()
    # Verify the completed reference before fitting a new version.
    v1.verify_output(v1.DEFAULT_OUTPUT, manifest, graphs, data, parent)
    gain = training_gain(manifest['ids']['train'], data['inputs.json'])
    tqdm.write(f"Training-only gain={gain['gain']:.10g}; median latest past EPSS="
               f"{gain['median_latest_observed_epss']:.10g}; CVEs={gain['n']}")
    config = dict(protocol['config'], correction_gain=gain['gain'])
    pins = dict(parent['pinned_files'])
    lock = read(v1.DEFAULT_OUTPUT/'evaluation_lock.json')
    pins.update({str((v1.DEFAULT_OUTPUT/n).resolve()): h for n,h in lock['files'].items()})
    for path in (v1.DEFAULT_OUTPUT/'evaluation_lock.json', PROTOCOL, FULL_PROTOCOL, Path(__file__), SCRIPT, TEST):
        pins[str(path.resolve())] = sha256(path)
    cache.base.check_pins(pins)
    identity = {'schema': VERSION, 'protocol': protocol, 'config': config, 'ids': manifest['ids'],
        'gain_fit': gain, 'planned_full_protocol': full_plan,
        'cache_directory': parent['cache_directory'], 'cache_lock_sha256': parent['cache_lock_sha256'],
        'reference_lock_sha256': sha256(v1.DEFAULT_OUTPUT/'evaluation_lock.json'),
        'pinned_files': pins, 'torch_version': str(torch.__version__), 'test_performance_evaluated': False,
        'limitations': parent['limitations'], 'full_expansion_authorized': False}
    return manifest, graphs, data, identity


def planned_full_protocol():
    return {
        'schema': 'r04_coreference_residual_development_v2',
        'status': 'planned_not_runnable_pending_smoke_verification_and_review',
        'prerequisite': VERSION,
        'cohort': {'train': 1023, 'validation': 197, 'source': 'unchanged_DQ17_v2_development_ids'},
        'variants': list(VARIANTS), 'head_rule': HEAD_RULE, 'gain_rule': GAIN_RULE,
        'gain_fit': 'fit_once_on_all_1023_training_input_histories_then_freeze_across_seeds_and_models',
        'config': dict(smoke.config(), epochs=100, patience=15, seeds=[17, 29, 43]),
        'execution': {'torch_threads': 1, 'deterministic_algorithms': True},
        'checkpoint_selection': 'earliest strictly lowest validation MAE; stop after 15 non-improving epochs',
        'epoch_zero_eligible_for_selection': False,
        'initialization': 'fresh_per_seed; shared_fusion_weights_and_batch_orders; no_smoke_warm_start',
        'comparators': ['persistence', 'matched_R03_all_seeds'],
        'reporting': 'all_12_fits_and_unweighted_three_seed_mean_and_sample_standard_deviation; no_best_seed',
        'test_performance_evaluated': False,
        'gate': ['v2_smoke_artifact_and_prediction_verification',
                 'review_initialization_gradients_clipping_and_corrections',
                 'explicit_review_before_implementing_or_starting_full_runner'],
        'performance_gate': 'not_required_to_beat_persistence_on_the_24_CVE_smoke',
        'exclusions': ['test_scoring', 'new_graphs', 'new_embeddings', 'SMP', 'summaries',
                       'CVSS', 'VulnCheck_inputs', 'hyperparameter_search', 'split_changes']
    }


def report(summary):
    lines = ['R04 v2: Training-History-Scaled Residual Smoke', '',
        '32 training / 24 validation CVEs; seed 17; three trained epochs; four matched models.',
        'This is an engineering smoke check, not an estimate from the full 197-CVE validation cohort.',
        'Inputs: current-description graph and observed EPSS Days 0-30, masks and observation age.',
        'Target: stored Day-60 EPSS. No exploitation labels, new embeddings, graph expansion or test scoring.',
        'Output = clip(latest observed input score + tanh(gain * correction head), 0, 1).',
        'Gain = median latest observed EPSS across training CVEs, bounded to [0.001,0.05].',
        f"Training-input gain: {summary['identity']['gain_fit']['gain']:.10g}.",
        'No target, validation or test value determines the gain. True observed zeros remain included.',
        'Scaling is inside tanh: it reduces the initial response but does not impose a small correction cap.',
        'Large learned corrections and clipping remain possible. There is no guarantee of improved MAE.',
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
    for name, digest in tqdm(lock['files'].items(), desc='Verify R04 v2 artifacts', unit='file'):
        require(sha256(output/name) == digest, 'Changed artifact: '+name)
    require(read(output/'run_identity.json') == identity and read(output/'run_status.json')['status'] == 'complete', 'Run identity/status differs')
    config = identity['config']
    require(identity['gain_fit'] == training_gain(manifest['ids']['train'], data['inputs.json']) and
            config['correction_gain'] == identity['gain_fit']['gain'], 'Gain does not reproduce')
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
        print('R04 v2 scaled residual smoke checkpoints and predictions verified. No retraining or test scoring.'); return
    require(not output.exists(), 'Output exists; use --verify-only')
    config = identity['config']
    initial = {m: initialize(m, manifest, graphs['all_coref'], config) for m in ('tpg_epss', 'epss_only')}
    zero = initial_checks(initial, manifest, graphs, data)
    cache.base.check_pins(identity['pinned_files'])
    if check_only:
        print('R04 v2 persistence initialization preflight passed. No optimizer steps or output writes; existing v1 validation predictions were verified.'); return
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
    print(f'R04 v2 scaled residual smoke saved to {output}. No full training expansion or test scoring.')


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
