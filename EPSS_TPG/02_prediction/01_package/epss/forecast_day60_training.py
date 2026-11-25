"""MS06: matched Day-60 regression, with immutable MS05 graph locations."""
import argparse
from copy import deepcopy
import math
from pathlib import Path
import statistics

import torch

from . import forecast_day30_graphs as cache
from . import forecast_matched_multisource as prior

prep, model, controls = cache.preparation, prior.model, prior.controls
read, write, require, sha256, tqdm = cache.read, cache.write, cache.require, cache.sha256, cache.tqdm
SCHEMA = 'ms06_day60_matched_training_v1'
CONFIG = prep.ROOT/'02_prediction/04_configs/day60_matched_training_v1.json'
OUTPUT = prep.ROOT/'05_results/02_evaluation/epss_day30_to_day60/matched_multisource_v1'
SCRIPT = prior.SCRIPTS/'run_forecast_day60_multisource.py'
TEST = prior.SCRIPTS/'test_forecast_day60_pipeline.py'
FIT_FILES = prior.FIT_FILES


def configuration(scale):
    p = read(CONFIG)
    require(p['schema'] == SCHEMA and p['variants'] == list(model.VARIANTS)
            and (p['anchor_day'], p['target_day'], p['horizon_days']) == (30, 60, 30)
            and p['history'] == prep.VIEW and p['loss'] == 'mae'
            and p['edge_configuration'] == 'no_coref' and not p['test_scoring'], 'Wrong Day-60 training protocol')
    old = read(prior.CONFIG)
    require(p['config'] == old['config'] and p['smoke'] == old['smoke'], 'Settings differ from fixed matched comparison')
    require(scale in ('smoke', 'full'), 'Unknown training scale')
    cfg = deepcopy(p['config'])
    if scale == 'smoke':
        cfg.update(p['smoke'])
    return p, cfg


def tensor_inputs(rows):
    require(bool(rows), 'Empty history batch')
    return torch.tensor([prep.sequence_channels(r) for r in rows], dtype=torch.float32)


def training_gain(ids, inputs):
    require(bool(ids) and len(ids) == len(set(ids)), 'Empty/duplicate training gain IDs')
    values = [next(v for v in reversed(prep.history_view(inputs[c])['values']) if v is not None) for c in ids]
    return min(.05, max(.001, statistics.median(values)))


class GraphStore:
    """Dispatch only selected tensors to pinned, read-only cache roots."""
    documents = model.GraphStore.documents

    def __init__(self, ctx, loc, allowed_ids, cache_gib):
        self.index, self.allowed_ids = ctx['index'], set(allowed_ids)
        require(self.allowed_ids <= set(self.index), 'Unknown graph CVE')
        self.routes, hashes = {}, {}
        for key, e in ctx['entries'].items():
            if e['cve'] not in self.allowed_ids:
                continue
            name = 'tensors/'+e['stem']+'.pt'
            root = loc[key]['root']
            require(root in (str(cache.REUSE), str(cache.OUTPUT/ctx['identity']['scale'])), 'Unapproved tensor root')
            require(name not in self.routes, 'Duplicate tensor route')
            self.routes[name] = root
            hashes.setdefault(root, {})[name] = loc[key]['files'][name]
        identity = ctx['identity']
        self.stores = {root: model.GraphStore(root, self.index, identity['type_vocabulary'], identity['embedding_dim'],
            self.allowed_ids, h, cache_gib/max(1, len(hashes))) for root, h in hashes.items()}

    def load(self, name):
        require(name in self.routes, 'Unselected or unpinned tensor')
        return self.stores[self.routes[name]].load(name)


def context(scale, threads, cache_gib):
    p, cfg = configuration(scale)
    ctx = cache.context(scale)
    directory = cache.OUTPUT/scale
    loc = cache.verify(directory, ctx)
    pins, _ = prior.check_files(directory, 'cache_lock.json', cache.SCHEMA, cache.expected(ctx))
    pins.update(ctx['identity']['source_pins'])
    data = prep.OUTPUT/scale
    lock = read(data/'preparation_lock.json')
    a = {n: read(data/n) for n in lock['files'] if n.endswith('.json')}
    prep.validate(a)
    ids = a['run_identity.json']['ids']
    require(ids == ctx['identity']['ids'], 'Graph and history cohorts differ')
    expected = p['cohort'] if scale == 'full' else a['protocol.json']['smoke_counts']
    require({s: len(v) for s, v in ids.items()} == expected, 'Wrong matched cohort')
    tensors = {s: tensor_inputs([a['inputs.json'][c] for c in chosen]) for s, chosen in ids.items()}
    require(all(r['statistics']['nodes'] <= cfg['graph_batch_nodes'] and r['statistics']['edges_no_coref'] <= cfg['graph_batch_edges']
                for r in loc.values()), 'Graph budget would exclude a chunk')
    frozen_run = read(prep.MS03/'run_identity.json')
    for path in (Path(model.__file__).resolve(), Path(prior.__file__).resolve(), Path(controls.__file__).resolve(),
                 Path(prior.edge_aware_layers.__file__).resolve(), prior.CONFIG):
        require(sha256(path) == frozen_run['pinned_files'][str(path)], 'Frozen model/training helper changed')
        pins[str(path)] = sha256(path)
    for path in (CONFIG, SCRIPT, TEST, Path(__file__).resolve()):
        pins[str(path)] = sha256(path)
    identity = {'schema': SCHEMA, 'scale': scale, 'protocol': p, 'config': cfg, 'ids': ids,
        'gain': training_gain(ids['train'], a['inputs.json']), 'gain_fit_ids': ids['train'],
        'vocab': ctx['identity']['type_vocabulary'], 'embedding_dim': ctx['identity']['embedding_dim'],
        'execution': {'device': 'cpu', 'threads': threads, 'cache_gib': cache_gib, 'deterministic_algorithms': True},
        'torch_version': str(torch.__version__), 'pinned_files': pins,
        'graph_cache_lock_sha256': sha256(directory/'cache_lock.json'),
        'source_coverage': {s: {source: sum(ctx['index'][c]['source_mask'][i] for c in chosen)
                            for i, source in enumerate(model.SOURCES)} for s, chosen in ids.items()}, 'test_scoring': False}
    store = GraphStore(ctx, loc, a['inputs.json'], cache_gib)
    return identity, tensors, a['labels.json'], store


def scored(rows, labels):
    for r in rows:
        y = labels[r['cve']]
        require(y['target_day'] == 60 and y['horizon_days'] == 30 and y['target_observed'] is True
                and y['y']['endpoint'] is not None, 'Wrong/missing Day-60 target')
        prep.contract.score(y['y']['endpoint'])
        r['actual_day60_epss'] = y['y']['endpoint']
    return controls.baseline.regression_metrics([r['actual_day60_epss'] for r in rows], [r['prediction'] for r in rows])


def references(identity, tensors, labels):
    x = tensors['validation']
    observed = x[:, :, 1] == 1
    last = torch.arange(31).expand_as(observed).masked_fill(~observed, -1).max(1).values
    values = x[:, :, 0].gather(1, last[:, None]).squeeze(1).tolist()
    rows = [{'cve': c, 'prediction': v} for c, v in zip(identity['ids']['validation'], values)]
    return {'definition': identity['protocol']['reference'], 'metrics': scored(rows, labels), 'rows': rows}


def evaluate(saved, identity, tensors, labels, store, progress=True):
    net = prior.initialize(identity, saved['variant'], saved['seed'])
    net.load_state_dict(saved['state_dict'], strict=True)
    rows = prior.infer(net, identity['ids']['validation'], tensors['validation'], store, identity['config']['batch_size'], progress)
    metrics = scored(rows, labels)
    return rows, {'selected_epoch': saved['selected_epoch'], 'metrics': metrics,
        'correction_std': statistics.pstdev(r['correction'] for r in rows),
        'clipped_below_zero': sum(r['raw_prediction'] < 0 for r in rows), 'clipped_above_one': sum(r['raw_prediction'] > 1 for r in rows)}


def fit(variant, seed, identity, tensors, labels, store, progress=True):
    cfg, ids = identity['config'], identity['ids']
    net = prior.initialize(identity, variant, seed)
    initial = controls.state_digest(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    lookup = {c: i for i, c in enumerate(ids['train'])}
    target = torch.tensor([labels[c]['y']['endpoint'] for c in ids['train']], dtype=torch.float32)
    best, best_mae, stale, history = None, math.inf, 0, []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for epoch, order in enumerate(tqdm(controls.orders(ids['train'], cfg, seed), desc=f'{variant}/seed{seed}', unit='epoch', disable=not progress), 1):
            net.train()
            total, low, high = 0., 0, 0
            gradients = {k: 0. for k in ('temporal', 'node_project', 'layers', 'source_embeddings', 'source_attention', 'head')}
            for start in tqdm(range(0, len(order), cfg['batch_size']), desc='Training CVE batches', unit='batch', leave=False, disable=not progress):
                chosen = order[start:start+cfg['batch_size']]
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
                total += float(loss.detach())*len(chosen)
                low += int((parts[2].detach() < 0).sum())
                high += int((parts[2].detach() > 1).sum())
            rows = prior.infer(net, ids['validation'], tensors['validation'], store, cfg['batch_size'], progress)
            error = scored(rows, labels)['mae']
            history.append({'epoch': epoch, 'training_objective_mae': total/len(order), 'validation_mae': error,
                'training_examples': len(order), 'data_gradient_abs_sums': gradients,
                'clipped_below_zero': low, 'clipped_above_one': high})
            if error < best_mae:
                best_mae, stale = error, 0
                best = {'schema': SCHEMA, 'variant': variant, 'seed': seed, 'selected_epoch': epoch,
                    **{k: identity[k] for k in ('config','gain','ids','vocab','embedding_dim','graph_cache_lock_sha256')},
                    'initial_state_sha256': initial, 'state_dict': {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                    'test_scoring': False}
            else:
                stale += 1
            if progress:
                tqdm.write(f'{variant}, seed {seed}, epoch {epoch}: train MAE={total/len(order):.8f}, validation MAE={error:.8f}, stale={stale}/{cfg["patience"]}')
            if stale >= cfg['patience']:
                break
    require(best is not None, 'No selected checkpoint')
    return best, history


def check_fit(directory, variant, seed, identity, tensors, labels, store):
    prior.check_files(directory, 'fit_lock.json', SCHEMA, FIT_FILES)
    saved = torch.load(directory/'checkpoint.pt', weights_only=True, map_location='cpu')
    require(saved['schema'] == SCHEMA and saved['variant'] == variant and saved['seed'] == seed and not saved['test_scoring'], 'Wrong checkpoint')
    require(all(saved[k] == identity[k] for k in ('config','gain','ids','vocab','embedding_dim','graph_cache_lock_sha256')), 'Checkpoint provenance differs')
    require(saved['initial_state_sha256'] == controls.state_digest(prior.initialize(identity, variant, seed).state_dict()), 'Initialization differs')
    require(all(bool(torch.isfinite(v).all()) for v in saved['state_dict'].values()), 'Nonfinite weights')
    history = read(directory/'history.json')
    cfg = identity['config']
    require(history and len(history) <= cfg['epochs'], 'Wrong epoch count')
    best, stale = math.inf, 0
    for i, r in enumerate(history, 1):
        require(r['epoch'] == i and r['training_examples'] == len(identity['ids']['train'])
                and all(math.isfinite(r[k]) and r[k] >= 0 for k in ('training_objective_mae','validation_mae'))
                and all(math.isfinite(v) and v >= 0 for v in r['data_gradient_abs_sums'].values()), 'Invalid history')
        if r['validation_mae'] < best:
            best, stale = r['validation_mae'], 0
        else:
            stale += 1
        require(stale < cfg['patience'] or i == len(history), 'Continued after stopping')
    require(len(history) == cfg['epochs'] or stale >= cfg['patience'], 'Premature stop')
    require(saved['selected_epoch'] == min(history, key=lambda r: r['validation_mae'])['epoch'], 'Wrong selected epoch')
    rows, metrics = evaluate(saved, identity, tensors, labels, store)
    require(rows == read(directory/'predictions.json') and metrics == read(directory/'metrics.json')
            and metrics['metrics']['mae'] == history[saved['selected_epoch']-1]['validation_mae'], 'Predictions do not reproduce')
    return {**metrics, 'epochs_run': len(history)}


def summarize(results, refs, identity):
    s = prior.summarize(results, refs, identity)
    s.update(schema=SCHEMA, anchor_day=30, target_day=60)
    return s


def report(s):
    lines = ['MS06: matched Day-60 regression', '', f"CVEs: {s['cohort']}; scale: {s['scale']}",
        'Days 0-30 EPSS -> observed Day-60 EPSS. Validation only; no test scoring.',
        'MAE and RMSE measure error on the 0-1 EPSS scale; lower is better.',
        'This is score regression, not exploitation classification. No F1/accuracy threshold is invented.',
        'All three variants use the same CVEs, batches, initial weights per seed and training settings.', '',
        'Model/seed | Selected epoch | Epochs run | MAE | RMSE | MAE minus persistence']
    for name, r in sorted(s['fits'].items()):
        lines.append(f"{name} | {r['selected_epoch']} | {r['epochs_run']} | {r['metrics']['mae']:.10g} | {r['metrics']['rmse']:.10g} | {r['metrics']['mae']-s['persistence']['mae']:+.10g}")
    return '\n'.join(lines + ['', f"Persistence: {s['persistence']}", f"Across seeds: {s['aggregates']}",
        f"Paired differences: {s['paired']}", f"Source coverage: {s['source_coverage']}", '',
        'fits/<variant>__<seed>/predictions.json contains actual_day60_epss, prediction and correction for each validation CVE.',
        'Negative MAE differences favor the learned model; small changes do not establish practical benefit.',
        *s['limitations'], ''])


def verify_run(directory, identity, tensors, labels, store):
    prior.check_files(directory, 'evaluation_lock.json', SCHEMA, prior.expected(identity))
    require(read(directory/'run_identity.json') == identity, 'Run identity differs')
    results = {}
    for seed in identity['config']['seeds']:
        require(read(directory/f'orders_seed_{seed}.json') == controls.orders(identity['ids']['train'], identity['config'], seed), 'Batch schedule differs')
        for variant in model.VARIANTS:
            name = f'{variant}__{seed}'
            results[name] = check_fit(directory/'fits'/name, variant, seed, identity, tensors, labels, store)
    refs = references(identity, tensors, labels)
    s = summarize(results, refs, identity)
    require(read(directory/'baselines.json') == refs and read(directory/'summary.json') == s
            and (directory/'summary.txt').read_text() == report(s), 'Summary differs')


def execute(scale, threads, cache_gib, check_only=False, verify_only=False, resume=False):
    identity, tensors, labels, store = context(scale, threads, cache_gib)
    if check_only:
        prior.preflight(identity, tensors, store)
        print(f'MS06 {scale} preflight passed. No training or writes.')
        return
    if scale == 'full':
        small = context('smoke', threads, cache_gib)
        verify_run(OUTPUT/'smoke', *small)
        identity['pinned_files'][str(OUTPUT/'smoke'/'evaluation_lock.json')] = sha256(OUTPUT/'smoke'/'evaluation_lock.json')
        del small
    directory = OUTPUT/scale
    if verify_only:
        verify_run(directory, identity, tensors, labels, store)
        prep.sources.check_pins(identity['pinned_files'])
        print(f'MS06 {scale} checkpoints and Day-60 predictions verified. No retraining or test scoring.')
        return
    prior.preflight(identity, tensors, store)
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
            path = directory/f'orders_seed_{seed}.json'
            value = controls.orders(identity['ids']['train'], identity['config'], seed)
            if path.exists():
                require(read(path) == value, 'Changed batch order')
            else:
                write(path, value)
        refs = references(identity, tensors, labels)
        if (directory/'baselines.json').exists():
            require(read(directory/'baselines.json') == refs, 'Changed persistence reference')
        else:
            write(directory/'baselines.json', refs)
        results = {}
        for seed, variant in tqdm([(s, v) for s in identity['config']['seeds'] for v in model.VARIANTS], desc='Matched Day-60 fits', unit='fit'):
            name = f'{variant}__{seed}'
            fitdir = directory/'fits'/name
            if not (fitdir/'fit_lock.json').exists():
                saved, history = fit(variant, seed, identity, tensors, labels, store)
                rows, metrics = evaluate(saved, identity, tensors, labels, store)
                fitdir.mkdir(parents=True, exist_ok=True)
                controls.save_checkpoint(fitdir/'checkpoint.pt', saved)
                for n, value in (('history.json',history), ('predictions.json',rows), ('metrics.json',metrics)):
                    write(fitdir/n, value)
                write(fitdir/'fit_lock.json', {'schema': SCHEMA, 'files': {n: sha256(fitdir/n) for n in sorted(FIT_FILES)}})
            results[name] = check_fit(fitdir, variant, seed, identity, tensors, labels, store)
        s = summarize(results, refs, identity)
        prep.sources.check_pins(identity['pinned_files'])
        write(directory/'summary.json', s)
        (directory/'summary.txt').write_text(report(s), encoding='utf-8')
        write(directory/'evaluation_lock.json', {'schema': SCHEMA, 'files': {n: sha256(directory/n) for n in sorted(prior.expected(identity))}})
    print(f'MS06 {scale} matched Day-60 run saved to {directory}. All fixed fits; no test scoring.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', choices=('smoke','full'), default='smoke')
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
