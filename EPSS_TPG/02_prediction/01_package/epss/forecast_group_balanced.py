"""GB01: training-only group weights; unchanged GV01 data, model and selection."""
import argparse
from collections import Counter
from copy import deepcopy
import math
from pathlib import Path
import statistics

import torch

from . import forecast_gain_validation as gv

old, prior, controls = gv.old, gv.prior, gv.controls
read, write, require, sha256, tqdm = gv.read, gv.write, gv.require, gv.sha256, gv.tqdm
SCHEMA = 'gb01_group_balanced_validation_v1'
VARIANTS, GROUPS = gv.VARIANTS, ('unchanged', 'increased', 'decreased')
CONFIG = old.prep.ROOT/'02_prediction/04_configs/group_balanced_validation_v1.json'
OUTPUT = old.prep.ROOT/'05_results/02_evaluation/epss_day30_to_day60/group_balanced_validation_v1'
SCRIPT = prior.SCRIPTS/'run_forecast_group_balanced.py'
TEST = prior.SCRIPTS/'test_forecast_group_balanced.py'
FIT_FILES = gv.FIT_FILES
initialize = gv.initialize


def movement_groups(ids, inputs, labels):
    require(ids and len(ids) == len(set(ids)), 'Empty/duplicate grouping IDs')
    result = {}
    for c in ids:
        label = labels[c]
        require(label['cve'] == c and label['target_day'] == 60 and label['horizon_days'] == 30
                and label['target_observed'] is True, 'Wrong grouping label')
        history = old.prep.history_view(inputs[c])['values']
        observed = [v for v in history if v is not None]
        require(bool(observed), 'No observed input for grouping')
        past, target = observed[-1], label['y']['endpoint']
        old.prep.contract.score(past)
        old.prep.contract.score(target)
        result[c] = 'unchanged' if target == past else 'increased' if target > past else 'decreased'
    return result


def training_weights(groups, require_all=True):
    require(bool(groups) and set(groups.values()) <= set(GROUPS), 'Invalid training groups')
    counts = {g: sum(v == g for v in groups.values()) for g in GROUPS}
    k, n = sum(v > 0 for v in counts.values()), len(groups)
    require(not require_all or k == 3, 'Full training requires all three groups')
    weights = {g: n/(k*count) if count else None for g, count in counts.items()}
    by_cve = {c: weights[g] for c, g in groups.items()}
    require(math.isclose(sum(by_cve.values()), n, rel_tol=1e-12), 'Weights do not average to one')
    return {'counts': counts, 'nonempty_groups': k, 'weights_by_group': weights, 'weights_by_cve': by_cve,
            'total_weight_by_group': {g: counts[g]*weights[g] if counts[g] else 0. for g in GROUPS}}


def weighted_mae(prediction, target, weight):
    require(prediction.ndim == 1 and prediction.shape == target.shape == weight.shape, 'Wrong weighted loss shapes')
    require(bool(torch.isfinite(weight).all()) and bool((weight > 0).all()), 'Invalid loss weights')
    return (weight*(prediction-target).abs()).mean()


def context(scale, threads, cache_gib):
    args = gv.context(scale, threads, cache_gib)
    identity, tensors, labels, store, _ = args
    if scale == 'full':
        path = gv.OUTPUT/'smoke'/'evaluation_lock.json'
        identity['pinned_files'][str(path)] = sha256(path)
    print('[CONTROL] Reproduce saved GV01 checkpoints; no control retraining.', flush=True)
    gv.verify_run(gv.OUTPUT/scale, *args)
    pins, _ = prior.check_files(gv.OUTPUT/scale, 'evaluation_lock.json', gv.SCHEMA, gv.expected(identity))
    controls_by_fit = {}
    for seed in identity['config']['seeds']:
        for variant in VARIANTS:
            name = f'{variant}__{seed}'
            directory = gv.OUTPUT/scale/'fits'/name
            controls_by_fit[name] = {'rows': read(directory/'predictions.json'),
                'initial_state_sha256': torch.load(directory/'checkpoint.pt', weights_only=True,
                                                  map_location='cpu')['initial_state_sha256']}
    protocol = read(CONFIG)
    require(protocol['schema'] == SCHEMA and protocol['variants'] == list(VARIANTS)
            and protocol['groups'] == list(GROUPS) and protocol['gain'] == 1.
            and (protocol['anchor_day'], protocol['target_day']) == (30, 60) and not protocol['test_scoring'],
            'Wrong GB01 protocol')
    if scale == 'full':
        require({s: len(v) for s, v in identity['ids'].items()} == protocol['cohort'], 'Changed cohort')
    source = old.prep.OUTPUT/scale/'inputs.json'
    require(sha256(source) == identity['pinned_files'][str(source)], 'Changed grouping inputs')
    inputs = read(source)
    groups = {s: movement_groups(ids, inputs, labels) for s, ids in identity['ids'].items()}
    weights = training_weights(groups['train'], scale == 'full')
    identity = deepcopy(identity)
    identity.update(schema=SCHEMA, balanced_protocol=protocol, movement_groups=groups, training_weighting=weights)
    identity['pinned_files'].update(pins)
    for path in (CONFIG, SCRIPT, TEST, Path(__file__).resolve()):
        identity['pinned_files'][str(path)] = sha256(path)
    print(f'[WEIGHTS] Training counts: {weights["counts"]}; weights: {weights["weights_by_group"]}', flush=True)
    print('[POLICY] Same batches; weights affect loss only. Validation selection remains unweighted.', flush=True)
    return identity, tensors, labels, store, controls_by_fit


def grouped_metrics(rows, groups):
    require(set(groups) == {r['cve'] for r in rows}, 'Wrong metric group IDs')
    result = {}
    for g in GROUPS:
        chosen = [r for r in rows if groups[r['cve']] == g]
        if not chosen:
            result[g] = {'n': 0, 'metrics': None, 'persistence_metrics': None}
            continue
        y = [r['actual_day60_epss'] for r in chosen]
        result[g] = {'n': len(chosen),
            'metrics': controls.baseline.regression_metrics(y, [r['prediction'] for r in chosen]),
            'persistence_metrics': controls.baseline.regression_metrics(y, [r['persistence_float32'] for r in chosen])}
    return result


def evaluate(saved, identity, tensors, labels, store, progress=True):
    rows, metrics = gv.evaluate(saved, identity, tensors, labels, store, progress)
    metrics['by_movement'] = grouped_metrics(rows, identity['movement_groups']['validation'])
    return rows, metrics


def compare(rows, control, groups):
    comparison = gv.compare(rows, control)
    comparison['mae_minus_gv01'] = comparison.pop('mae_minus_original')
    for r in comparison['rows']:
        r['gv01_prediction'] = r.pop('original_prediction')
        r['error_minus_gv01'] = r.pop('error_minus_original')
    comparison['gv01_by_movement'] = grouped_metrics(control['rows'], groups)
    comparison['definition'] = 'Negative differences favor GB01 over unweighted GV01 or persistence; validation only.'
    return comparison


def fit(variant, seed, identity, tensors, labels, store, progress=True):
    cfg, ids = identity['config'], identity['ids']
    net = initialize(identity, variant, seed)
    initial = controls.state_digest(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    lookup = {c: i for i, c in enumerate(ids['train'])}
    target = torch.tensor([labels[c]['y']['endpoint'] for c in ids['train']], dtype=torch.float32)
    weights = torch.tensor([identity['training_weighting']['weights_by_cve'][c] for c in ids['train']], dtype=torch.float32)
    best, best_mae, stale, history = None, math.inf, 0, []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for epoch, order in enumerate(tqdm(controls.orders(ids['train'], cfg, seed),
                desc=f'{variant}/seed{seed}', unit='epoch', disable=not progress), 1):
            net.train()
            total, unweighted, raw_total, low, high, saturated, clipped_grad = 0., 0., 0., 0, 0, 0, 0
            group_errors, group_counts = Counter(), Counter()
            gradients = {k: 0. for k in ('temporal', 'node_project', 'layers', 'source_embeddings', 'source_attention', 'head')}
            for start in tqdm(range(0, len(order), cfg['batch_size']), desc='Weighted training batches',
                              unit='batch', leave=False, disable=not progress):
                chosen = order[start:start+cfg['batch_size']]
                ix = [lookup[c] for c in chosen]
                optimizer.zero_grad(set_to_none=True)
                parts = net.details(tensors['train'][ix], store, chosen)
                require(all(bool(torch.isfinite(v).all()) for v in parts[:5]), 'Nonfinite training output')
                loss = weighted_mae(parts[3], target[ix], weights[ix])
                require(bool(torch.isfinite(loss)), 'Nonfinite loss')
                loss.backward()
                for name, parameter in net.named_parameters():
                    if parameter.grad is not None:
                        gradients[name.split('.')[0]] += float(parameter.grad.abs().sum())
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                errors = (parts[3].detach()-target[ix]).abs()
                total += float(loss.detach())*len(chosen)
                unweighted += float(errors.sum())
                raw_total += float((parts[2].detach()-target[ix]).abs().sum())
                for c, error in zip(chosen, errors.tolist()):
                    group = identity['movement_groups']['train'][c]
                    group_counts[group] += 1
                    group_errors[group] += error
                low += int((parts[2].detach() < 0).sum())
                high += int((parts[2].detach() > 1).sum())
                saturated += int((parts[1].detach().abs() > .99).sum())
                clipped_grad += int(float(norm) > cfg['gradient_clip_norm'])
            rows = prior.infer(net, ids['validation'], tensors['validation'], store, cfg['batch_size'], progress)
            error = old.scored(rows, labels)['mae']
            diag = gv.diagnostics(rows)
            history.append({'epoch': epoch, 'training_weighted_mae': total/len(order),
                'training_unweighted_mae': unweighted/len(order), 'validation_mae': error,
                'training_examples': len(order), 'data_gradient_abs_sums': gradients,
                'training_raw_mae': raw_total/len(order), 'training_clipped_below_zero': low,
                'training_clipped_above_one': high, 'training_saturated_tanh': saturated,
                'gradient_clipped_batches': clipped_grad, 'validation_diagnostics': diag,
                'validation_by_movement': grouped_metrics(rows, identity['movement_groups']['validation']),
                'training_by_movement': {g: {'n': group_counts[g],
                    'mae': group_errors[g]/group_counts[g] if group_counts[g] else None} for g in GROUPS}})
            if error < best_mae:
                best_mae, stale = error, 0
                best = {'schema': SCHEMA, 'variant': variant, 'seed': seed, 'selected_epoch': epoch,
                    **{k: identity[k] for k in ('config', 'gain', 'original_gain', 'ids', 'vocab', 'embedding_dim',
                                               'graph_cache_lock_sha256', 'training_weighting')},
                    'initial_state_sha256': initial,
                    'state_dict': {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, 'test_scoring': False}
            else:
                stale += 1
            if progress:
                tqdm.write(f'[EPOCH] {variant}/{seed}/{epoch}: train weighted MAE={total/len(order):.8f}, '
                    f'train ordinary MAE={unweighted/len(order):.8f}, validation MAE={error:.8f}, '
                    f'clipped={diag["clipped_fraction"]:.1%}, correction std={diag["correction_std"]:.3g}, '
                    f'stale={stale}/{cfg["patience"]}')
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
                                                'graph_cache_lock_sha256', 'training_weighting')), 'Checkpoint provenance differs')
    require(saved['initial_state_sha256'] == control['initial_state_sha256'] ==
            controls.state_digest(initialize(identity, variant, seed).state_dict()), 'Unmatched initial weights')
    require(all(bool(torch.isfinite(v).all()) for v in saved['state_dict'].values()), 'Nonfinite weights')
    history, cfg = read(directory/'history.json'), identity['config']
    require(history and len(history) <= cfg['epochs'], 'Wrong epoch count')
    best, stale = math.inf, 0
    for i, r in enumerate(history, 1):
        require(r['epoch'] == i and r['training_examples'] == len(identity['ids']['train'])
                and all(math.isfinite(r[k]) and r[k] >= 0 for k in
                        ('validation_mae', 'training_weighted_mae', 'training_unweighted_mae', 'training_raw_mae')), 'Invalid history')
        require({g: r['training_by_movement'][g]['n'] for g in GROUPS} == identity['training_weighting']['counts'],
                'Group exposures differ')
        macro = statistics.mean(r['training_by_movement'][g]['mae'] for g in GROUPS if r['training_by_movement'][g]['n'])
        require(math.isclose(macro, r['training_weighted_mae'], rel_tol=2e-6, abs_tol=1e-9), 'Weighted loss differs from macro group MAE')
        if r['validation_mae'] < best:
            best, stale = r['validation_mae'], 0
        else:
            stale += 1
        require(stale < cfg['patience'] or i == len(history), 'Continued after stopping')
    require(len(history) == cfg['epochs'] or stale >= cfg['patience'], 'Premature stop')
    require(saved['selected_epoch'] == min(history, key=lambda r: r['validation_mae'])['epoch'], 'Wrong selected epoch')
    rows, metrics = evaluate(saved, identity, tensors, labels, store)
    comparison = compare(rows, control, identity['movement_groups']['validation'])
    selected = history[saved['selected_epoch']-1]
    require(rows == read(directory/'predictions.json') and metrics == read(directory/'metrics.json')
            and comparison == read(directory/'comparison.json') and metrics['metrics']['mae'] == selected['validation_mae']
            and metrics['diagnostics'] == selected['validation_diagnostics']
            and metrics['by_movement'] == selected['validation_by_movement'], 'Reproduction differs')
    return {**metrics, 'epochs_run': len(history), 'mae_minus_gv01': comparison['mae_minus_gv01'],
            'mae_minus_persistence': comparison['mae_minus_persistence']}


def summarize(results, refs, identity):
    summary = gv.summarize(results, refs, identity)
    summary.update(schema=SCHEMA, weighting=identity['training_weighting']['weights_by_group'],
                   movement_counts={s: dict(Counter(g.values())) for s, g in identity['movement_groups'].items()})
    return summary


def report(s):
    lines = ['GB01: group-balanced MAE on the fixed Day-60 cohort', '', f'CVEs: {s["cohort"]}; scale: {s["scale"]}',
        'Only the training loss weights change from GV01. Gain 1, model, optimizer and batches are unchanged.',
        f'Training weights: {s["weighting"]}', f'Movement counts: {s["movement_counts"]}',
        'Weights N/(K*n_group) give each nonempty training group equal total loss weight; mean weight is one.',
        'Full requires all three groups. Smoke may omit groups and is not evidence of effectiveness.',
        'Group means are compared using original stored values, not float32 equality or a fitted threshold.',
        'Groups and weights are supervision/evaluation metadata, never model inputs. No target-day scores enter encoders.',
        'Checkpoint selection remains earliest strictly best ordinary validation MAE from epoch 1.',
        'Variant/seed | Selected epoch | Ordinary MAE | RMSE | Minus GV01 MAE | Minus persistence MAE | Clipped %']
    for name, r in sorted(s['fits'].items()):
        lines.append(f'{name} | {r["selected_epoch"]} | {r["metrics"]["mae"]:.9g} | {r["metrics"]["rmse"]:.9g} | '
            f'{r["mae_minus_gv01"]:+.9g} | {r["mae_minus_persistence"]:+.9g} | {100*r["diagnostics"]["clipped_fraction"]:.2f}')
        for g, row in r['by_movement'].items():
            lines.append(f'  {g}: n={row["n"]}; MAE={row["metrics"]["mae"] if row["metrics"] else None}; '
                         f'persistence MAE={row["persistence_metrics"]["mae"] if row["persistence_metrics"] else None}')
    lines += ['', f'Across seeds: {s["aggregates"]}', f'Persistence: {s["persistence"]}',
        f'Paired all-source minus EPSS MAE: {s["paired_all_sources_minus_epss_mae"]}',
        'Negative paired differences favor GB01. MAE/RMSE are errors on the 0-1 score scale.',
        'history.json separates weighted loss from ordinary training MAE and records group errors, clipping and gradients.',
        'metrics.json includes ordinary validation/group errors, correction variability and raw/clipped diagnostics.',
        'comparison.json retains paired CVE errors against GV01/persistence and GV01 errors by movement group.',
        'weighting.json documents exact group assignments and training-only weights; absent group metrics are null, not zero.',
        'Reweighting changes the training objective and may worsen overall accuracy. It does not repair temporal shift.',
        'Descriptions/summaries remain retrospective where historical availability is unverified.',
        'Validation informed development. This is not independent final performance or exploitation classification.',
        'Partial histories remain masked, targets are never imputed, no new graphs/embeddings or test scoring.']
    return '\n'.join(lines)+'\n'


def expected(identity):
    return gv.expected(identity) | {'weighting.json'}


def audit(identity):
    return {'training_weighting': identity['training_weighting'], 'movement_groups': identity['movement_groups'],
            'policy': identity['balanced_protocol'], 'not_model_inputs': True}


def verify_run(directory, identity, tensors, labels, store, control_results):
    prior.check_files(directory, 'evaluation_lock.json', SCHEMA, expected(identity))
    require(read(directory/'run_identity.json') == identity and read(directory/'weighting.json') == audit(identity), 'Run identity/weights differ')
    results = {}
    for seed in identity['config']['seeds']:
        require(read(directory/f'orders_seed_{seed}.json') == controls.orders(identity['ids']['train'], identity['config'], seed), 'Changed batch schedule')
        for variant in VARIANTS:
            name = f'{variant}__{seed}'
            results[name] = check_fit(directory/'fits'/name, variant, seed, identity, tensors, labels, store, control_results[name])
    refs = old.references(identity, tensors, labels)
    summary = summarize(results, refs, identity)
    require(read(directory/'baselines.json') == refs and read(directory/'summary.json') == summary
            and (directory/'summary.txt').read_text() == report(summary), 'Summary differs')


def execute(scale, threads, cache_gib, check_only=False, verify_only=False, resume=False):
    args = context(scale, threads, cache_gib)
    identity, tensors, labels, store, control_results = args
    if check_only:
        gv.preflight(identity, tensors, store, control_results)
        print(f'GB01 {scale} preflight passed. No writes or training.')
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
        print(f'GB01 {scale} checkpoints, training-only weights and paired predictions verified. No retraining or test scoring.')
        return
    gv.preflight(identity, tensors, store, control_results)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with prior.writer_lock(OUTPUT/'.writer.lock'):
        if resume:
            require(directory.exists() and not (directory/'evaluation_lock.json').exists()
                    and read(directory/'run_identity.json') == identity, 'Wrong/complete resume directory')
        else:
            require(not directory.exists(), 'Run exists; use --resume or --verify-only')
            directory.mkdir()
            write(directory/'run_identity.json', identity)
        for name, value in [('weighting.json', audit(identity)), ('baselines.json', old.references(identity, tensors, labels))] + [
                (f'orders_seed_{seed}.json', controls.orders(identity['ids']['train'], identity['config'], seed))
                for seed in identity['config']['seeds']]:
            path = directory/name
            if path.exists():
                require(read(path) == value, 'Changed '+name)
            else:
                write(path, value)
        results = {}
        for seed, variant in tqdm([(s, v) for s in identity['config']['seeds'] for v in VARIANTS], desc='GB01 matched fits', unit='fit'):
            name = f'{variant}__{seed}'
            fitdir = directory/'fits'/name
            if not (fitdir/'fit_lock.json').exists():
                print(f'[FIT] {name}: fresh initialization, training-only balanced weights.', flush=True)
                saved, history = fit(variant, seed, identity, tensors, labels, store)
                rows, metrics = evaluate(saved, identity, tensors, labels, store)
                fitdir.mkdir(parents=True, exist_ok=True)
                controls.save_checkpoint(fitdir/'checkpoint.pt', saved)
                for n, value in (('history.json', history), ('predictions.json', rows), ('metrics.json', metrics),
                        ('comparison.json', compare(rows, control_results[name], identity['movement_groups']['validation']))):
                    write(fitdir/n, value)
                write(fitdir/'fit_lock.json', {'schema': SCHEMA, 'files': {n: sha256(fitdir/n) for n in sorted(FIT_FILES)}})
            results[name] = check_fit(fitdir, variant, seed, identity, tensors, labels, store, control_results[name])
        summary = summarize(results, read(directory/'baselines.json'), identity)
        old.prep.sources.check_pins(identity['pinned_files'])
        write(directory/'summary.json', summary)
        (directory/'summary.txt').write_text(report(summary), encoding='utf-8')
        write(directory/'evaluation_lock.json', {'schema': SCHEMA, 'files': {n: sha256(directory/n) for n in sorted(expected(identity))}})
    print(f'GB01 {scale} saved to {directory}. All configured fits; validation only, no test scoring.')


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
