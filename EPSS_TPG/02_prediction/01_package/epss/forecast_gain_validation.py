"""GV01: transfer GD01's unit gain to fixed MS06 development cohorts."""
import argparse
from copy import deepcopy
import math
from pathlib import Path
import statistics

import torch

from . import forecast_day60_training as old

prior, controls = old.prior, old.controls
read, write, require, sha256, tqdm = old.read, old.write, old.require, old.sha256, old.tqdm
SCHEMA = 'gv01_gain_validation_v1'
VARIANTS = ('epss_only', 'all_sources_epss')
CONFIG = old.prep.ROOT/'02_prediction/04_configs/gain_validation_v1.json'
OUTPUT = old.prep.ROOT/'05_results/02_evaluation/epss_day30_to_day60/gain_validation_v1'
SCRIPT = prior.SCRIPTS/'run_forecast_gain_validation.py'
TEST = prior.SCRIPTS/'test_forecast_gain_validation.py'
FIT_FILES = old.FIT_FILES | {'comparison.json'}


def context(scale, threads, cache_gib):
    print('[CHECK] Frozen MS06 histories, source masks, graph receipts and controls.', flush=True)
    identity, tensors, labels, store = old.context(scale, threads, cache_gib)
    directory = old.OUTPUT/scale
    pins, _ = prior.check_files(directory, 'evaluation_lock.json', old.SCHEMA, prior.expected(identity))
    saved = read(directory/'run_identity.json')
    for key in ('config', 'ids', 'gain', 'vocab', 'embedding_dim', 'graph_cache_lock_sha256', 'source_coverage'):
        require(saved[key] == identity[key], 'MS06 control mismatch: '+key)
    protocol = read(CONFIG)
    require(protocol['schema'] == SCHEMA and protocol['variants'] == list(VARIANTS)
            and protocol['gain'] == 1.0 and not protocol['test_scoring']
            and (protocol['anchor_day'], protocol['target_day']) == (30, 60), 'Wrong GV01 protocol')
    if scale == 'full':
        require({s: len(v) for s, v in identity['ids'].items()} == protocol['cohort'], 'Changed cohort')
    control_results = {}
    for seed in identity['config']['seeds']:
        require(read(directory/f'orders_seed_{seed}.json') == controls.orders(identity['ids']['train'], identity['config'], seed),
                'Unmatched control batch schedule')
        for variant in VARIANTS:
            name = f'{variant}__{seed}'
            old.check_fit(directory/'fits'/name, variant, seed, identity, tensors, labels, store)
            control_results[name] = {
                'rows': read(directory/'fits'/name/'predictions.json'),
                'metrics': read(directory/'fits'/name/'metrics.json'),
                'initial_state_sha256': torch.load(directory/'fits'/name/'checkpoint.pt', weights_only=True,
                                                  map_location='cpu')['initial_state_sha256']}
    identity = deepcopy(identity)
    identity.update(schema=SCHEMA, original_gain=identity['gain'], gain=1.0, comparison_protocol=protocol)
    identity['pinned_files'].update(pins)
    for path in (CONFIG, SCRIPT, TEST, Path(__file__).resolve()):
        identity['pinned_files'][str(path)] = sha256(path)
    print(f'[COHORT] {len(identity["ids"]["train"])} train / {len(identity["ids"]["validation"])} validation; '
          f'gain {identity["original_gain"]} -> 1. No test access.', flush=True)
    return identity, tensors, labels, store, control_results


def initialize(identity, variant, seed):
    require(variant in VARIANTS and identity['gain'] == 1.0, 'Wrong comparison model')
    base = dict(identity, gain=identity['original_gain'])
    net = prior.initialize(base, variant, seed)
    net.gain = identity['gain']
    return net


def diagnostics(rows):
    require(bool(rows), 'Empty diagnostic rows')
    n = len(rows)
    raw = [r['raw_prediction'] for r in rows]
    corrections = [r['correction'] for r in rows]
    require(all(math.isfinite(v) for r in rows for v in
                (r['raw_prediction'], r['prediction'], r['correction'], r['actual_day60_epss'])), 'Nonfinite diagnostics')
    result = {
        'n': n, 'raw_mae': statistics.mean(abs(r['raw_prediction']-r['actual_day60_epss']) for r in rows),
        'clipped_mae': statistics.mean(abs(r['prediction']-r['actual_day60_epss']) for r in rows),
        'clipped_below_zero': sum(v < 0 for v in raw), 'clipped_above_one': sum(v > 1 for v in raw),
        'predicted_exactly_zero': sum(r['prediction'] == 0 for r in rows),
        'predicted_exactly_one': sum(r['prediction'] == 1 for r in rows),
        'raw_min': min(raw), 'raw_max': max(raw), 'correction_std': statistics.pstdev(corrections),
        'saturated_tanh': sum(abs(v) > .99 for v in corrections)}
    result['clipped_fraction'] = (result['clipped_below_zero']+result['clipped_above_one'])/n
    return result


def evaluate(saved, identity, tensors, labels, store, progress=True):
    net = initialize(identity, saved['variant'], saved['seed'])
    net.load_state_dict(saved['state_dict'], strict=True)
    rows = prior.infer(net, identity['ids']['validation'], tensors['validation'], store,
                       identity['config']['batch_size'], progress)
    metrics = old.scored(rows, labels)
    return rows, {'selected_epoch': saved['selected_epoch'], 'metrics': metrics, 'diagnostics': diagnostics(rows)}


def compare(rows, control):
    require([r['cve'] for r in rows] == [r['cve'] for r in control['rows']], 'Unmatched control CVEs')
    paired = []
    for r, c in zip(rows, control['rows']):
        require(r['actual_day60_epss'] == c['actual_day60_epss'] and
                r['persistence_float32'] == c['persistence_float32'], 'Unmatched target/baseline')
        y = r['actual_day60_epss']
        error = abs(r['prediction']-y)
        paired.append({'cve': r['cve'], 'actual_day60_epss': y, 'prediction': r['prediction'],
            'original_prediction': c['prediction'], 'persistence': r['persistence_float32'],
            'absolute_error': error, 'error_minus_original': error-abs(c['prediction']-y),
            'error_minus_persistence': error-abs(r['persistence_float32']-y)})
    return {'rows': paired, 'mae_minus_original': statistics.mean(r['error_minus_original'] for r in paired),
            'mae_minus_persistence': statistics.mean(r['error_minus_persistence'] for r in paired),
            'definition': 'Negative differences favor gain 1; paired validation CVEs, not independent test evidence.'}


def fit(variant, seed, identity, tensors, labels, store, progress=True):
    cfg, ids = identity['config'], identity['ids']
    net = initialize(identity, variant, seed)
    initial = controls.state_digest(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    lookup = {c: i for i, c in enumerate(ids['train'])}
    target = torch.tensor([labels[c]['y']['endpoint'] for c in ids['train']], dtype=torch.float32)
    best, best_mae, stale, history = None, math.inf, 0, []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for epoch, order in enumerate(tqdm(controls.orders(ids['train'], cfg, seed),
                desc=f'{variant}/seed{seed}', unit='epoch', disable=not progress), 1):
            net.train()
            total, raw_total, low, high, saturated, clipped_grad = 0., 0., 0, 0, 0, 0
            gradients = {k: 0. for k in ('temporal', 'node_project', 'layers', 'source_embeddings', 'source_attention', 'head')}
            for start in tqdm(range(0, len(order), cfg['batch_size']), desc='Training batches', unit='batch',
                              leave=False, disable=not progress):
                chosen = order[start:start+cfg['batch_size']]
                ix = [lookup[c] for c in chosen]
                optimizer.zero_grad(set_to_none=True)
                parts = net.details(tensors['train'][ix], store, chosen)
                require(all(bool(torch.isfinite(v).all()) for v in parts[:5]), 'Nonfinite training output')
                loss = torch.nn.functional.l1_loss(parts[3], target[ix])
                loss.backward()
                for name, parameter in net.named_parameters():
                    if parameter.grad is not None:
                        gradients[name.split('.')[0]] += float(parameter.grad.abs().sum())
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach())*len(chosen)
                raw_total += float((parts[2].detach()-target[ix]).abs().sum())
                low += int((parts[2].detach() < 0).sum())
                high += int((parts[2].detach() > 1).sum())
                saturated += int((parts[1].detach().abs() > .99).sum())
                clipped_grad += int(float(norm) > cfg['gradient_clip_norm'])
            rows = prior.infer(net, ids['validation'], tensors['validation'], store, cfg['batch_size'], progress)
            error = old.scored(rows, labels)['mae']
            diag = diagnostics(rows)
            history.append({'epoch': epoch, 'training_objective_mae': total/len(order), 'validation_mae': error,
                'training_examples': len(order), 'data_gradient_abs_sums': gradients,
                'training_raw_mae': raw_total/len(order), 'training_clipped_below_zero': low,
                'training_clipped_above_one': high, 'training_saturated_tanh': saturated,
                'gradient_clipped_batches': clipped_grad, 'validation_diagnostics': diag})
            if error < best_mae:
                best_mae, stale = error, 0
                best = {'schema': SCHEMA, 'variant': variant, 'seed': seed, 'selected_epoch': epoch,
                    **{k: identity[k] for k in ('config', 'gain', 'original_gain', 'ids', 'vocab', 'embedding_dim', 'graph_cache_lock_sha256')},
                    'initial_state_sha256': initial,
                    'state_dict': {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, 'test_scoring': False}
            else:
                stale += 1
            if progress:
                tqdm.write(f'[EPOCH] {variant}/{seed}/{epoch}: train MAE={total/len(order):.8f}, '
                    f'validation MAE={error:.8f}, raw MAE={diag["raw_mae"]:.8f}, '
                    f'clipped={diag["clipped_fraction"]:.1%}, stale={stale}/{cfg["patience"]}')
            if stale >= cfg['patience']:
                break
    require(best is not None, 'No selected checkpoint')
    return best, history


def check_fit(directory, variant, seed, identity, tensors, labels, store, control):
    prior.check_files(directory, 'fit_lock.json', SCHEMA, FIT_FILES)
    saved = torch.load(directory/'checkpoint.pt', weights_only=True, map_location='cpu')
    require(saved['schema'] == SCHEMA and saved['variant'] == variant and saved['seed'] == seed
            and not saved['test_scoring'], 'Wrong checkpoint')
    require(all(saved[k] == identity[k] for k in ('config', 'gain', 'original_gain', 'ids', 'vocab', 'embedding_dim',
                                                  'graph_cache_lock_sha256')), 'Checkpoint provenance differs')
    require(saved['initial_state_sha256'] == control['initial_state_sha256'] ==
            controls.state_digest(initialize(identity, variant, seed).state_dict()), 'Unmatched initial weights')
    require(all(bool(torch.isfinite(v).all()) for v in saved['state_dict'].values()), 'Nonfinite weights')
    history, cfg = read(directory/'history.json'), identity['config']
    require(history and len(history) <= cfg['epochs'], 'Wrong epoch count')
    best, stale = math.inf, 0
    for i, r in enumerate(history, 1):
        require(r['epoch'] == i and r['training_examples'] == len(identity['ids']['train'])
                and math.isfinite(r['validation_mae']) and r['validation_mae'] >= 0, 'Invalid history')
        if r['validation_mae'] < best:
            best, stale = r['validation_mae'], 0
        else:
            stale += 1
        require(stale < cfg['patience'] or i == len(history), 'Continued after stopping')
    require(len(history) == cfg['epochs'] or stale >= cfg['patience'], 'Premature stop')
    require(saved['selected_epoch'] == min(history, key=lambda r: r['validation_mae'])['epoch'], 'Wrong selected epoch')
    rows, metrics = evaluate(saved, identity, tensors, labels, store)
    comparison = compare(rows, control)
    require(rows == read(directory/'predictions.json') and metrics == read(directory/'metrics.json')
            and comparison == read(directory/'comparison.json')
            and metrics['metrics']['mae'] == history[saved['selected_epoch']-1]['validation_mae']
            and metrics['diagnostics'] == history[saved['selected_epoch']-1]['validation_diagnostics'], 'Reproduction differs')
    return {**metrics, 'epochs_run': len(history),
            'mae_minus_original': comparison['mae_minus_original'], 'mae_minus_persistence': comparison['mae_minus_persistence']}


def summarize(results, refs, identity):
    seeds = identity['config']['seeds']
    aggregates = {}
    for variant in VARIANTS:
        values = [results[f'{variant}__{s}']['metrics']['mae'] for s in seeds]
        aggregates[variant] = {'mean_mae': statistics.mean(values),
                              'sample_std_mae': statistics.stdev(values) if len(values) > 1 else None}
    return {'schema': SCHEMA, 'scale': identity['scale'], 'cohort': {s: len(v) for s, v in identity['ids'].items()},
        'gain': identity['gain'], 'original_gain': identity['original_gain'], 'fits': results, 'aggregates': aggregates,
        'persistence': refs['metrics'], 'source_coverage': identity['source_coverage'],
        'paired_all_sources_minus_epss_mae': {str(s): results[f'all_sources_epss__{s}']['metrics']['mae']-
                                            results[f'epss_only__{s}']['metrics']['mae'] for s in seeds},
        'test_scoring': False}


def report(s):
    lines = ['GV01: fixed Day-60 validation of gain 1', '', f'CVEs: {s["cohort"]}; scale: {s["scale"]}',
        'Day 0-30 histories -> observed Day-60 EPSS. No new graphs, embeddings or test scoring.',
        'Only gain differs from locked MS06 controls. All fixed seeds reported, not a selected winner.',
        'Gain is inside tanh(gain * head_output); it is not a hard correction bound.',
        'Variant/seed | Selected epoch | MAE | RMSE | Minus original MAE | Minus persistence MAE | Clipped %']
    for name, r in sorted(s['fits'].items()):
        lines.append(f'{name} | {r["selected_epoch"]} | {r["metrics"]["mae"]:.9g} | {r["metrics"]["rmse"]:.9g} | '
                     f'{r["mae_minus_original"]:+.9g} | {r["mae_minus_persistence"]:+.9g} | '
                     f'{100*r["diagnostics"]["clipped_fraction"]:.2f}')
    lines += ['', f'Across seeds: {s["aggregates"]}', f'Persistence: {s["persistence"]}',
        f'Paired all-source minus EPSS MAE: {s["paired_all_sources_minus_epss_mae"]}',
        'Negative paired differences favor gain 1. MAE/RMSE are errors on the 0-1 score scale.',
        'history.json records online training raw/clipped MAE, gradient sums and validation clipping every epoch.',
        'Training counts are online batch exposures; validation counts describe one fixed evaluation per epoch.',
        'predictions.json gives actual, predicted, raw, persistence and source attention for each validation CVE.',
        'comparison.json gives paired CVE errors against original MS06 and persistence.',
        'Raw error exposes excursions hidden by clipping; clipping outside [0,1] can suppress loss gradients.',
        'Descriptions/summaries are retrospective; source dates do not authenticate historical text availability.',
        'Validation informed model development; this is not independent final performance or exploitation prediction.',
        'Partial inputs retain missingness masks; observed targets are not imputed. Test remains closed.',
        'Smoke is an integration check, not evidence for selecting settings.']
    return '\n'.join(lines)+'\n'


def expected(identity):
    names = {'run_identity.json', 'baselines.json', 'summary.json', 'summary.txt'}
    for seed in identity['config']['seeds']:
        names.add(f'orders_seed_{seed}.json')
        for variant in VARIANTS:
            names.update(f'fits/{variant}__{seed}/{n}' for n in FIT_FILES | {'fit_lock.json'})
    return names


def verify_run(directory, identity, tensors, labels, store, control_results):
    prior.check_files(directory, 'evaluation_lock.json', SCHEMA, expected(identity))
    require(read(directory/'run_identity.json') == identity, 'Run identity differs')
    results = {}
    for seed in identity['config']['seeds']:
        require(read(directory/f'orders_seed_{seed}.json') == controls.orders(identity['ids']['train'], identity['config'], seed),
                'Changed batch schedule')
        for variant in VARIANTS:
            name = f'{variant}__{seed}'
            results[name] = check_fit(directory/'fits'/name, variant, seed, identity, tensors, labels, store, control_results[name])
    refs = old.references(identity, tensors, labels)
    summary = summarize(results, refs, identity)
    require(read(directory/'baselines.json') == refs and read(directory/'summary.json') == summary
            and (directory/'summary.txt').read_text() == report(summary), 'Summary differs')


def preflight(identity, tensors, store, controls_by_fit):
    for seed in identity['config']['seeds']:
        for variant in VARIANTS:
            net = initialize(identity, variant, seed).eval()
            require(controls.state_digest(net.state_dict()) == controls_by_fit[f'{variant}__{seed}']['initial_state_sha256'],
                    'Changed initial weights')
            with torch.no_grad():
                parts = net.details(tensors['train'][:2], store, identity['ids']['train'][:2])
            require(torch.equal(parts[0], parts[3]) and bool((parts[1] == 0).all()), 'Not persistence initialization')


def execute(scale, threads, cache_gib, check_only=False, verify_only=False, resume=False):
    args = context(scale, threads, cache_gib)
    identity, tensors, labels, store, control_results = args
    if check_only:
        preflight(identity, tensors, store, control_results)
        print(f'GV01 {scale} preflight passed. No writes or training.')
        return
    if scale == 'full':
        small = context('smoke', threads, cache_gib)
        verify_run(OUTPUT/'smoke', *small)
        identity['pinned_files'][str(OUTPUT/'smoke'/'evaluation_lock.json')] = sha256(OUTPUT/'smoke'/'evaluation_lock.json')
        del small
    directory = OUTPUT/scale
    if verify_only:
        verify_run(directory, *args)
        old.prep.sources.check_pins(identity['pinned_files'])
        print(f'GV01 {scale} checkpoints, clipping diagnostics and paired predictions verified. No retraining or test scoring.')
        return
    preflight(identity, tensors, store, control_results)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with prior.writer_lock(OUTPUT/'.writer.lock'):
        if resume:
            require(directory.exists() and not (directory/'evaluation_lock.json').exists()
                    and read(directory/'run_identity.json') == identity, 'Wrong/complete resume directory')
        else:
            require(not directory.exists(), 'Run exists; use --resume or --verify-only')
            directory.mkdir()
            write(directory/'run_identity.json', identity)
        for seed in identity['config']['seeds']:
            value = controls.orders(identity['ids']['train'], identity['config'], seed)
            path = directory/f'orders_seed_{seed}.json'
            if path.exists():
                require(read(path) == value, 'Changed batch order')
            else:
                write(path, value)
        refs = old.references(identity, tensors, labels)
        if (directory/'baselines.json').exists():
            require(read(directory/'baselines.json') == refs, 'Changed persistence')
        else:
            write(directory/'baselines.json', refs)
        results = {}
        for seed, variant in tqdm([(s, v) for s in identity['config']['seeds'] for v in VARIANTS], desc='GV01 matched fits', unit='fit'):
            name = f'{variant}__{seed}'
            fitdir = directory/'fits'/name
            if not (fitdir/'fit_lock.json').exists():
                print(f'[FIT] {name}; fresh initialization, gain 1; original control remains unchanged.', flush=True)
                saved, history = fit(variant, seed, identity, tensors, labels, store)
                rows, metrics = evaluate(saved, identity, tensors, labels, store)
                fitdir.mkdir(parents=True, exist_ok=True)
                controls.save_checkpoint(fitdir/'checkpoint.pt', saved)
                for n, value in (('history.json', history), ('predictions.json', rows), ('metrics.json', metrics),
                                 ('comparison.json', compare(rows, control_results[name]))):
                    write(fitdir/n, value)
                write(fitdir/'fit_lock.json', {'schema': SCHEMA, 'files': {n: sha256(fitdir/n) for n in sorted(FIT_FILES)}})
            results[name] = check_fit(fitdir, variant, seed, identity, tensors, labels, store, control_results[name])
        summary = summarize(results, refs, identity)
        old.prep.sources.check_pins(identity['pinned_files'])
        write(directory/'summary.json', summary)
        (directory/'summary.txt').write_text(report(summary), encoding='utf-8')
        write(directory/'evaluation_lock.json', {'schema': SCHEMA, 'files': {n: sha256(directory/n) for n in sorted(expected(identity))}})
    print(f'GV01 {scale} saved to {directory}. All configured fits; validation only, no test scoring.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--cache-gib', type=float, default=12)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    require(args.threads > 0 and math.isfinite(args.cache_gib) and args.cache_gib >= 0, 'Invalid runtime settings')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    execute(args.scale, args.threads, args.cache_gib, args.check_only, args.verify_only, args.resume)


if __name__ == '__main__':
    main()
