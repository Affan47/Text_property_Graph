"""MS03: matched EPSS/description/all-source regression on the CD02 cohort."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import math
from pathlib import Path
import statistics

import torch

from . import forecast_matched_multisource_model as model
from . import forecast_later_controls as controls
from . import edge_aware_layers

cache = model.cache
read, write, require, sha256, tqdm = cache.read, cache.write, cache.require, cache.sha256, cache.tqdm
ROOT = cache.base.ROOT
SCHEMA = 'ms03_matched_multisource_training_v1'
CONFIG = ROOT / '02_prediction/04_configs/matched_multisource_training_v1.json'
SCRIPTS = ROOT / '02_prediction/03_analysis/01_revision_dataset'
SCRIPT = SCRIPTS / 'run_forecast_matched_multisource.py'
TEST = SCRIPTS / 'test_forecast_matched_multisource.py'
OUTPUT = ROOT / '05_results/02_evaluation/epss_day60_to_day90/matched_multisource_v1'
FIT_FILES = {'checkpoint.pt', 'history.json', 'predictions.json', 'metrics.json'}


def configuration(scale):
    p = read(CONFIG)
    require(p['schema'] == SCHEMA and p['variants'] == list(model.VARIANTS) and
            p['source_order'] == list(model.SOURCES) and p['history'] == 'recent_30_60' and
            p['anchor_day'] == 60 and p['target_day'] == 90 and p['loss'] == 'mae' and
            p['edge_configuration'] == 'no_coref' and p['test_scoring'] is False, 'Wrong training protocol')
    require(scale in ('smoke', 'full'), 'Unknown scale')
    cfg = deepcopy(p['config'])
    if scale == 'smoke':
        cfg.update(p['smoke'])
    return p, cfg


def tensor_inputs(records):
    for record in records:
        require(set(record) == {'cve', 'publication_date', 'input_end_date', 'anchor_day', 'time_policy', 'histories'},
                'Unexpected input metadata/target field')
    return controls.input_tensor(records, 'recent_30_60')


def check_files(directory, lock_name, schema, expected=None):
    lock = read(directory / lock_name)
    require(lock['schema'] == schema and (expected is None or set(lock['files']) == expected), 'Wrong artifact lock')
    pins = {str(directory / lock_name): sha256(directory / lock_name)}
    for name, digest in tqdm(lock['files'].items(), desc='Verify locked files', unit='file', leave=False):
        path = (directory / name).resolve()
        require(directory.resolve() in path.parents and sha256(path) == digest, 'Changed/unsafe artifact: ' + name)
        pins[str(path)] = digest
    return pins, lock


def context(scale, threads, cache_gib):
    p, cfg = configuration(scale)
    ctx = cache.context('full')
    root = cache.OUTPUT / 'full'
    pins, lock = check_files(root, 'cache_lock.json', cache.SCHEMA, cache.expected(ctx))
    require(read(root / 'run_identity.json') == ctx['identity'] and read(root / 'run_status.json') == {'status': 'complete'},
            'Wrong/incomplete graph cache')
    index = read(root / 'cve_index.json')
    require(index == ctx['index'], 'Source routing changed')
    pins.update(ctx['identity']['source_pins'])
    prepared = cache.base.preparation.OUTPUT / 'full'
    names = ('run_identity.json', 'inputs.json', 'labels.json', 'split_manifest.json', 'source_inputs.json', 'training_text_registry.json')
    a = {name: read(prepared / name) for name in names}
    cache.base.preparation.validate(a)
    require(a['run_identity.json']['ids'] == ctx['identity']['ids'] and
            {s: len(v) for s, v in ctx['identity']['ids'].items()} == p['cohort'], 'Wrong full cohort')
    small = cache.base.preparation.OUTPUT / 'smoke'
    small_identity = read(small / 'run_identity.json')
    small_lock = read(small / 'preparation_lock.json')
    require(sha256(small / 'run_identity.json') == small_lock['files']['run_identity.json'], 'Changed smoke selection')
    pins[str(small / 'run_identity.json')] = sha256(small / 'run_identity.json')
    pins[str(small / 'preparation_lock.json')] = sha256(small / 'preparation_lock.json')
    ids = small_identity['ids'] if scale == 'smoke' else ctx['identity']['ids']
    require(set(ids) == {'train', 'validation'} and all(set(ids[s]) <= set(ctx['identity']['ids'][s]) for s in ids), 'Wrong smoke roles')
    inputs = {c: a['inputs.json'][c] for chosen in ids.values() for c in chosen}
    labels = {c: a['labels.json'][c] for c in inputs}
    for c, label in labels.items():
        require(label['cve'] == c and label['target_observed'] is True and label['target_day'] == 90 and
                label['horizon_days'] == 30 and set(label['y']) == {'endpoint'} and label['y']['endpoint'] is not None,
                'Missing/wrong target; target imputation forbidden')
    tensors = {s: tensor_inputs([inputs[c] for c in chosen]) for s, chosen in ids.items()}
    graph_report = read(root / 'analysis.json')
    require(all(r['nodes'] <= cfg['graph_batch_nodes'] and r['edges_no_coref'] <= cfg['graph_batch_edges']
                for r in graph_report['chunks']), 'Graph budget would exclude a chunk')
    for path in (CONFIG, SCRIPT, TEST, Path(__file__).resolve(), Path(model.__file__).resolve(),
                 Path(controls.__file__).resolve(), Path(controls.data.__file__).resolve(),
                 Path(controls.baseline.__file__).resolve(), Path(edge_aware_layers.__file__).resolve()):
        pins[str(path)] = sha256(path)
    identity = {'schema': SCHEMA, 'scale': scale, 'protocol': p, 'config': cfg, 'ids': ids,
                'gain': controls.training_gain(ids['train'], inputs), 'gain_fit_ids': ids['train'],
                'vocab': ctx['identity']['type_vocabulary'], 'embedding_dim': ctx['identity']['embedding_dim'],
                'execution': {'device': 'cpu', 'threads': threads, 'cache_gib': cache_gib, 'deterministic_algorithms': True},
                'torch_version': str(torch.__version__), 'pinned_files': pins,
                'graph_cache_lock_sha256': sha256(root / 'cache_lock.json'),
                'source_coverage': {s: {source: sum(index[c]['source_mask'][i] for c in chosen)
                                    for i, source in enumerate(model.SOURCES)} for s, chosen in ids.items()},
                'test_scoring': False}
    store = model.GraphStore(root, index, identity['vocab'], identity['embedding_dim'], inputs,
                             {n: h for n, h in lock['files'].items() if n.startswith('tensors/')}, cache_gib)
    return identity, tensors, labels, store


def initialize(identity, variant, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return model.MatchedRegressor(variant, identity['config'], identity['vocab'], identity['embedding_dim'], identity['gain'])


def infer(net, ids, tensor, store, batch_size, progress=True):
    net.eval()
    result = []
    with torch.no_grad():
        for start in tqdm(range(0, len(ids), batch_size), desc='Validation batches', unit='batch', leave=False, disable=not progress):
            chosen = ids[start:start + batch_size]
            parts = net.details(tensor[start:start + batch_size], store, chosen)
            require(all(bool(torch.isfinite(v).all()) for v in parts[:5]), 'Nonfinite predictions')
            for i, c in enumerate(chosen):
                result.append({'cve': c, **{name: float(parts[k][i]) for k, name in enumerate(
                    ('persistence_float32', 'correction', 'raw_prediction', 'prediction'))},
                    'source_attention': parts[4][i].tolist(), 'source_mask': parts[5][i].tolist()})
    return result


def scored(rows, labels):
    for row in rows:
        row['actual_day90_epss'] = labels[row['cve']]['y']['endpoint']
    return controls.baseline.regression_metrics([r['actual_day90_epss'] for r in rows], [r['prediction'] for r in rows])


def references(identity, tensors, labels):
    x = tensors['validation']
    ids = identity['ids']['validation']
    observed = x[:, :, 1] == 1
    last = torch.arange(31).expand_as(observed).masked_fill(~observed, -1).max(1).values
    values = x[:, :, 0].gather(1, last[:, None]).squeeze(1).tolist()
    rows = [{'cve': c, 'prediction': v} for c, v in zip(ids, values)]
    metrics = scored(rows, labels)
    return {'definition': 'Latest observed Day-30-to-Day-60 EPSS, in the same float32 representation as all model bases.',
            'rows': rows, 'metrics': metrics}


def evaluate(saved, identity, tensors, labels, store, progress=True):
    net = initialize(identity, saved['variant'], saved['seed'])
    net.load_state_dict(saved['state_dict'], strict=True)
    rows = infer(net, identity['ids']['validation'], tensors['validation'], store, identity['config']['batch_size'], progress)
    metrics = scored(rows, labels)
    return rows, {'selected_epoch': saved['selected_epoch'], 'metrics': metrics,
                  'correction_std': statistics.pstdev(r['correction'] for r in rows),
                  'clipped_below_zero': sum(r['raw_prediction'] < 0 for r in rows),
                  'clipped_above_one': sum(r['raw_prediction'] > 1 for r in rows)}


def preflight(identity, tensors, store):
    cfg = identity['config']
    ids = identity['ids']['train'][:2]
    for seed in tqdm(cfg['seeds'], desc='Check matched initialization', unit='seed'):
        hashes = []
        for variant in model.VARIANTS:
            net = initialize(identity, variant, seed).eval()
            hashes.append(controls.state_digest(net.state_dict()))
            with torch.no_grad():
                base, correction, _, pred, weights, masks = net.details(tensors['train'][:len(ids)], store, ids)
            require(torch.equal(base, pred) and bool((correction == 0).all()), 'Not persistence initialization')
            require(bool((weights[~masks] == 0).all()), 'Absent source received attention')
        require(len(set(hashes)) == 1, 'Unmatched initialization')


def fit(variant, seed, identity, tensors, labels, store, progress=True):
    cfg, ids = identity['config'], identity['ids']
    net = initialize(identity, variant, seed)
    initial = controls.state_digest(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    lookup = {c: i for i, c in enumerate(ids['train'])}
    target = torch.tensor([labels[c]['y']['endpoint'] for c in ids['train']], dtype=torch.float32)
    best, best_mae, stale, history = None, math.inf, 0, []
    schedule = controls.orders(ids['train'], cfg, seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for epoch, order in enumerate(tqdm(schedule, desc=f'{variant}/seed{seed}', unit='epoch', disable=not progress), 1):
            net.train()
            total, low, high = 0., 0, 0
            gradients = {k: 0. for k in ('temporal', 'node_project', 'layers', 'source_embeddings', 'source_attention', 'head')}
            for start in tqdm(range(0, len(order), cfg['batch_size']), desc='Training CVE batches', unit='batch', leave=False, disable=not progress):
                chosen = order[start:start + cfg['batch_size']]
                ix = [lookup[c] for c in chosen]
                optimizer.zero_grad(set_to_none=True)
                parts = net.details(tensors['train'][ix], store, chosen)
                loss = torch.nn.functional.l1_loss(parts[3], target[ix])
                require(bool(torch.isfinite(loss)), 'Nonfinite loss')
                loss.backward()
                for name, parameter in net.named_parameters():
                    if parameter.grad is not None:
                        gradients[name.split('.')[0]] += float(parameter.grad.abs().sum())
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach()) * len(chosen)
                low += int((parts[2].detach() < 0).sum())
                high += int((parts[2].detach() > 1).sum())
            rows = infer(net, ids['validation'], tensors['validation'], store, cfg['batch_size'], progress)
            error = scored(rows, labels)['mae']
            history.append({'epoch': epoch, 'training_objective_mae': total / len(order), 'validation_mae': error,
                            'training_examples': len(order), 'data_gradient_abs_sums': gradients,
                            'clipped_below_zero': low, 'clipped_above_one': high})
            if error < best_mae:
                best_mae, stale = error, 0
                best = {'schema': SCHEMA, 'variant': variant, 'seed': seed, 'selected_epoch': epoch,
                        'config': cfg, 'gain': identity['gain'], 'ids': ids, 'vocab': identity['vocab'],
                        'embedding_dim': identity['embedding_dim'], 'graph_cache_lock_sha256': identity['graph_cache_lock_sha256'],
                        'initial_state_sha256': initial, 'state_dict': {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                        'test_scoring': False}
            else:
                stale += 1
            if progress:
                tqdm.write(f'{variant}, seed {seed}, epoch {epoch}: train MAE={total/len(order):.8f}, validation MAE={error:.8f}, stale={stale}/{cfg["patience"]}')
            if stale >= cfg['patience']:
                break
    require(best is not None, 'No checkpoint selected')
    return best, history


def check_fit(directory, variant, seed, identity, tensors, labels, store):
    check_files(directory, 'fit_lock.json', SCHEMA, FIT_FILES)
    saved = torch.load(directory / 'checkpoint.pt', weights_only=True, map_location='cpu')
    require(saved['schema'] == SCHEMA and saved['variant'] == variant and saved['seed'] == seed and not saved['test_scoring'], 'Wrong checkpoint')
    require(all(saved[k] == identity[k] for k in ('config', 'gain', 'ids', 'vocab', 'embedding_dim', 'graph_cache_lock_sha256')), 'Checkpoint provenance differs')
    require(saved['initial_state_sha256'] == controls.state_digest(initialize(identity, variant, seed).state_dict()), 'Initialization differs')
    require(all(bool(torch.isfinite(v).all()) for v in saved['state_dict'].values()), 'Nonfinite weights')
    history = read(directory / 'history.json')
    cfg = identity['config']
    require(history and len(history) <= cfg['epochs'], 'Invalid epoch count')
    best, stale = math.inf, 0
    for i, row in enumerate(history, 1):
        require(row['epoch'] == i and row['training_examples'] == len(identity['ids']['train']) and
                all(math.isfinite(row[k]) and row[k] >= 0 for k in ('training_objective_mae', 'validation_mae')) and
                all(math.isfinite(v) and v >= 0 for v in row['data_gradient_abs_sums'].values()), 'Invalid history')
        if row['validation_mae'] < best:
            best, stale = row['validation_mae'], 0
        else:
            stale += 1
        require(stale < cfg['patience'] or i == len(history), 'Continued after stopping')
    require(len(history) == cfg['epochs'] or stale >= cfg['patience'], 'Premature stop')
    require(saved['selected_epoch'] == min(history, key=lambda r: r['validation_mae'])['epoch'], 'Wrong selected epoch')
    rows, metrics = evaluate(saved, identity, tensors, labels, store)
    require(rows == read(directory / 'predictions.json') and metrics == read(directory / 'metrics.json') and
            metrics['metrics']['mae'] == history[saved['selected_epoch'] - 1]['validation_mae'], 'Checkpoint predictions do not reproduce')
    return {**metrics, 'epochs_run': len(history)}


def summarize(results, refs, identity):
    seeds = identity['config']['seeds']
    aggregates = {}
    for v in model.VARIANTS:
        values = [results[f'{v}__{s}']['metrics']['mae'] for s in seeds]
        aggregates[v] = {'mean_mae': statistics.mean(values), 'sample_std_mae': statistics.stdev(values) if len(values) > 1 else None}
    return {'schema': SCHEMA, 'scale': identity['scale'], 'cohort': {s: len(v) for s, v in identity['ids'].items()},
            'persistence': refs['metrics'], 'fits': results, 'aggregates': aggregates,
            'paired': [{'seed': s, 'description_minus_epss_mae': results[f'description_epss__{s}']['metrics']['mae'] - results[f'epss_only__{s}']['metrics']['mae'],
                        'all_sources_minus_epss_mae': results[f'all_sources_epss__{s}']['metrics']['mae'] - results[f'epss_only__{s}']['metrics']['mae'],
                        'all_sources_minus_description_mae': results[f'all_sources_epss__{s}']['metrics']['mae'] - results[f'description_epss__{s}']['metrics']['mae']} for s in seeds],
            'source_coverage': identity['source_coverage'], 'limitations': identity['protocol']['limitations'], 'test_scoring': False}


def report(s):
    lines = ['MS03 matched multi-source forecasting', '', f"Scale: {s['scale']}; CVEs: {s['cohort']}",
             'Day 30-60 EPSS history predicts the observed Day-90 score. Validation only; test set closed.',
             'Smoke scores check integration, not which settings to use. All fixed seeds are reported.',
             'Negative paired MAE differences favor the added text sources; they do not establish causality.', '',
             'Variant/seed | Best epoch | Epochs run | Validation MAE | RMSE | MAE minus persistence']
    for name, row in sorted(s['fits'].items()):
        lines.append(f"{name} | {row['selected_epoch']} | {row['epochs_run']} | {row['metrics']['mae']:.10g} | {row['metrics']['rmse']:.10g} | {row['metrics']['mae']-s['persistence']['mae']:+.10g}")
    lines += ['', f"Persistence: {s['persistence']}", f"Aggregates: {s['aggregates']}", f"Paired: {s['paired']}",
              f"Source coverage: {s['source_coverage']}", '', 'Limitations'] + s['limitations']
    return '\n'.join(lines) + '\n'


def expected(identity):
    names = {'run_identity.json', 'baselines.json', 'summary.json', 'summary.txt'}
    for seed in identity['config']['seeds']:
        names.add(f'orders_seed_{seed}.json')
        for variant in model.VARIANTS:
            names.update(f'fits/{variant}__{seed}/{n}' for n in FIT_FILES | {'fit_lock.json'})
    return names


def verify_run(directory, identity, tensors, labels, store):
    check_files(directory, 'evaluation_lock.json', SCHEMA, expected(identity))
    require(read(directory / 'run_identity.json') == identity, 'Run identity differs')
    results = {}
    for seed in identity['config']['seeds']:
        require(read(directory / f'orders_seed_{seed}.json') == controls.orders(identity['ids']['train'], identity['config'], seed), 'Batch schedule differs')
        for variant in model.VARIANTS:
            name = f'{variant}__{seed}'
            results[name] = check_fit(directory / 'fits' / name, variant, seed, identity, tensors, labels, store)
    refs = references(identity, tensors, labels)
    result = summarize(results, refs, identity)
    require(read(directory / 'baselines.json') == refs and read(directory / 'summary.json') == result and
            (directory / 'summary.txt').read_text() == report(result), 'Summary differs')


@contextmanager
def writer_lock(path):
    """Refuse concurrent training/resume processes targeting the same result tree."""
    with path.open('a') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another MS03 process holds the result lock') from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def execute(scale, threads, cache_gib, check_only=False, verify_only=False, resume=False):
    if check_only:
        return _execute(scale, threads, cache_gib, check_only=True)
    if verify_only:
        require(OUTPUT.exists(), 'No MS03 runs to verify')
    else:
        OUTPUT.mkdir(parents=True, exist_ok=True)
    with writer_lock(OUTPUT / '.writer.lock'):
        return _execute(scale, threads, cache_gib, verify_only=verify_only, resume=resume)


def _execute(scale, threads, cache_gib, check_only=False, verify_only=False, resume=False):
    identity, tensors, labels, store = context(scale, threads, cache_gib)
    if scale == 'full' and not check_only:
        # Reuse the already checked graph resources, but reproduce all smoke fits.
        small = read(cache.base.preparation.OUTPUT / 'smoke' / 'run_identity.json')['ids']
        smoke = deepcopy(identity)
        smoke.update(scale='smoke', ids=small, config=configuration('smoke')[1], gain_fit_ids=small['train'])
        prepared_inputs = read(cache.base.preparation.OUTPUT / 'full' / 'inputs.json')
        smoke['gain'] = controls.training_gain(small['train'], prepared_inputs)
        smoke['source_coverage'] = {s: {source: sum(store.index[c]['source_mask'][i] for c in chosen)
                                       for i, source in enumerate(model.SOURCES)} for s, chosen in small.items()}
        small_tensors = {s: tensor_inputs([prepared_inputs[c] for c in chosen]) for s, chosen in small.items()}
        verify_run(OUTPUT / 'smoke', smoke, small_tensors, labels, store)
        identity['pinned_files'][str(OUTPUT / 'smoke' / 'evaluation_lock.json')] = sha256(OUTPUT / 'smoke' / 'evaluation_lock.json')
    directory = OUTPUT / scale
    if verify_only:
        verify_run(directory, identity, tensors, labels, store)
        cache.base.preparation.sources.check_pins(identity['pinned_files'])
        print(f'MS03 {scale} all checkpoints and validation predictions verified. No retraining or test scoring.')
        return
    preflight(identity, tensors, store)
    if check_only:
        print(f'MS03 {scale} preflight passed. No training, output writes or test scoring.')
        return
    if resume:
        require(directory.exists() and not (directory / 'evaluation_lock.json').exists() and
                read(directory / 'run_identity.json') == identity, 'Wrong/complete resume directory')
    else:
        require(not directory.exists(), 'Run exists; use --resume or --verify-only')
        directory.mkdir(parents=True)
        write(directory / 'run_identity.json', identity)
    for seed in identity['config']['seeds']:
        path = directory / f'orders_seed_{seed}.json'
        value = controls.orders(identity['ids']['train'], identity['config'], seed)
        if path.exists():
            require(read(path) == value, 'Changed batch schedule')
        else:
            write(path, value)
    refs = references(identity, tensors, labels)
    if (directory / 'baselines.json').exists():
        require(read(directory / 'baselines.json') == refs, 'Changed baseline')
    else:
        write(directory / 'baselines.json', refs)
    results = {}
    for seed, variant in tqdm([(s, v) for s in identity['config']['seeds'] for v in model.VARIANTS], desc='Matched model fits', unit='fit'):
        name = f'{variant}__{seed}'
        fitdir = directory / 'fits' / name
        if not (fitdir / 'fit_lock.json').exists():
            if fitdir.exists():
                tqdm.write('Restarting incomplete fit from original seed: ' + name)
            saved, history = fit(variant, seed, identity, tensors, labels, store)
            rows, metrics = evaluate(saved, identity, tensors, labels, store)
            fitdir.mkdir(parents=True, exist_ok=True)
            controls.save_checkpoint(fitdir / 'checkpoint.pt', saved)
            for n, value in (('history.json', history), ('predictions.json', rows), ('metrics.json', metrics)):
                write(fitdir / n, value)
            write(fitdir / 'fit_lock.json', {'schema': SCHEMA, 'files': {n: sha256(fitdir / n) for n in sorted(FIT_FILES)}})
        results[name] = check_fit(fitdir, variant, seed, identity, tensors, labels, store)
    result = summarize(results, refs, identity)
    cache.base.preparation.sources.check_pins(identity['pinned_files'])
    write(directory / 'summary.json', result)
    (directory / 'summary.txt').write_text(report(result), encoding='utf-8')
    write(directory / 'evaluation_lock.json', {'schema': SCHEMA, 'files': {n: sha256(directory / n) for n in sorted(expected(identity))}})
    print(f'MS03 {scale} matched training saved to {directory}. All configured fits; validation only, no test scoring.')


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
    require(args.threads > 0 and math.isfinite(args.cache_gib) and args.cache_gib >= 0, 'Invalid runtime resources')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    execute(args.scale, args.threads, args.cache_gib, args.check_only, args.verify_only, args.resume)


if __name__ == '__main__':
    main()
