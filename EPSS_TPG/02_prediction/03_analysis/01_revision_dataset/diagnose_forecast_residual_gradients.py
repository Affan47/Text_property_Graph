"""DQ19: bounded training-batch gradient audit; no optimizer or model updates."""
import argparse
from collections import defaultdict
from datetime import date
import hashlib
import importlib
import inspect
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss import forecast_coreference_residual_development as run

torch, task = run.torch, run.task
require, read, write, sha256 = run.require, run.read_json, run.write_json, run.sha256
VERSION = 'dq19_training_gradient_decay_v1'
OUTPUT = task.RESULT_ROOT / 'coreference_residual_gradient_audit_v1'
SOURCE_LOCK = 'a7309b871ec471d481584597917509c76df7ca2858b09e8c08c6e221c026591b'
PROTOCOL = {
    'schema': VERSION, 'split': 'train', 'sample_size': 32, 'batch_size': 8,
    'selection': 'ascending sha256(dq19:training: + CVE); independent of labels',
    'seeds': [17, 29, 43], 'variants': list(run.VARIANTS),
    'states': ['initialization', 'selected_checkpoint'],
    'optimizer_steps': 0, 'validation_scoring': False, 'test_scoring': False,
    'new_embeddings': False, 'threads': 1, 'device': 'cpu',
}
FILES = {'identity.json', 'analysis.json', 'analysis.txt', 'gradient_details.json'}
LIMITATIONS = [
    'Only 32 hash-selected training CVEs in four batches; not a full-training gradient survey.',
    'Initialization and saved checkpoints only, not a reconstruction of the intervening trajectory.',
    'Checkpoints were previously selected using validation; this diagnostic uses training inputs and labels only.',
    'Shared JSON files are parsed, then restricted to selected training IDs; no validation/test values are analyzed.',
    'Adam moment buffers were not saved. Norm ratios describe gradient terms, not actual parameter-update ratios.',
    'Ordinary Adam adds L2 decay after external gradient clipping; this is not AdamW.',
    'Zero final-layer initialization blocks upstream data gradients initially; that alone is not a defect.',
    'Batch-gradient cancellation, stable targets, loss scaling and parameter scale can all affect these ratios.',
    'This audit cannot establish that weight decay caused the observed prediction behavior.',
]


def select_ids(train_ids, splits, count=32):
    require(len(set(train_ids)) == len(train_ids) and len(train_ids) >= count > 0,
            'Invalid training cohort or sample size')
    require(all(splits[c]['split'] == 'train' for c in train_ids), 'Non-training CVE in cohort')
    return sorted(train_ids, key=lambda c: hashlib.sha256(
        ('dq19:training:' + c).encode()).digest())[:count]


def group_name(name):
    if name.startswith('head.'):
        return name
    return name.split('.')[0]


def term_statistics(parts):
    """Float64 summaries of the float32 terms Adam would receive, before moments."""
    theta = torch.cat([p['theta'].reshape(-1).double() for p in parts])
    active = [p for p in parts if p['raw'] is not None]
    out = {'parameters': theta.numel(), 'active_parameters': sum(p['theta'].numel() for p in active),
           'tensors_without_gradient': len(parts) - len(active),
           'parameter_l2': float(theta.norm())}
    if active:
        raw, data, decay = [torch.cat([p[k].reshape(-1).double() for p in active])
                            for k in ('raw', 'data', 'decay')]
    else:
        raw = data = decay = torch.zeros(1, dtype=torch.float64)
    r, g, d = float(raw.norm()), float(data.norm()), float(decay.norm())
    out.update(raw_data_gradient_l2=r, clipped_data_gradient_l2=g, decay_l2=d,
               combined_gradient_l2=float((data + decay).norm()),
               decay_to_data_ratio=d/g if g else None,
               data_decay_cosine=float(torch.dot(data, decay)/(g*d)) if g and d else None,
               data_gradient_nonzero_elements=int(torch.count_nonzero(data)),
               comparison=('no_active_gradient' if not active else
                           'both_zero' if g == d == 0 else 'decay_only' if g == 0 else
                           'decay_larger' if d > g else 'data_larger' if g > d else 'equal'))
    return out


def measure_gradients(net, loss, decay, clip):
    """Only gradient buffers may change; deliberately never instantiate an optimizer."""
    require(decay >= 0 and clip > 0 and bool(torch.isfinite(loss)), 'Invalid loss/settings')
    require(all(p.grad is None for p in net.parameters()), 'Stale gradients')
    before = run.digest_state(net.state_dict())
    loss.backward()
    parts = {}
    for name, p in net.named_parameters():
        require(p.grad is None or bool(torch.isfinite(p.grad).all()), 'Non-finite gradient')
        parts[name] = {'theta': p.detach().clone(),
                       'raw': None if p.grad is None else p.grad.detach().clone()}
    total = float(torch.nn.utils.clip_grad_norm_(
        net.parameters(), clip, error_if_nonfinite=True))
    for name, p in net.named_parameters():
        parts[name]['data'] = None if p.grad is None else p.grad.detach().clone()
        # Adam skips grad=None, but still applies decay to an explicit zero gradient.
        parts[name]['decay'] = None if p.grad is None else p.detach().mul(decay)
    groups = defaultdict(list)
    for name, values in parts.items():
        groups[group_name(name)].append(values)
    result = {
        'global_raw_gradient_l2': total,
        'clip_applied': total > clip,
        'parameters': {n: term_statistics([p]) for n, p in parts.items()},
        'groups': {n: term_statistics(p) for n, p in groups.items()},
        'all_parameters': term_statistics(list(parts.values())),
    }
    net.zero_grad(set_to_none=True)
    require(run.digest_state(net.state_dict()) == before, 'Parameter/buffer mutation')
    return result


def verify_paths(pins, description):
    for name, expected in run.cache.tqdm(sorted(pins.items()), desc=description, unit='file'):
        require(sha256(Path(name)) == expected, 'Changed source: ' + name)


def context():
    source = run.DEFAULT_OUTPUT
    require(sha256(source/'evaluation_lock.json') == SOURCE_LOCK, 'Wrong frozen R04 run')
    lock = read(source/'evaluation_lock.json')
    verify_paths({str(source/n): h for n, h in lock['files'].items()}, 'Verify R04 hashes')
    saved = read(source/'run_identity.json')
    require(saved['schema'] == run.VERSION and saved['torch_version'] == str(torch.__version__),
            'Wrong run or PyTorch version')
    require(saved['config']['seeds'] == PROTOCOL['seeds'] and saved['config']['dropout'] == 0
            and saved['config']['batch_size'] == 8, 'Training settings changed')
    splits = read(run.cache.base.PREPARED/'split_manifest.json')
    ids = select_ids(saved['ids']['train'], splits)
    require(not set(ids).intersection(saved['ids']['validation']), 'Validation ID selected')
    directory = Path(saved['cache_directory'])
    require(sha256(directory/'cache_lock.json') == saved['cache_lock_sha256'], 'Changed cache lock')
    cache_lock = read(directory/'cache_lock.json')
    pins = dict(saved['pinned_files'])
    pins[str(source/'evaluation_lock.json')] = SOURCE_LOCK
    for cve in ids:
        name = 'graphs/' + cve + '.pt'
        pins[str(directory/name)] = cache_lock['files'][name]
    for name in ('cache_lock.json', 'manifest.json'):
        pins[str(directory/name)] = (saved['cache_lock_sha256'] if name == 'cache_lock.json'
                                     else cache_lock['files'][name])
    optimizer_module = importlib.import_module('torch.optim.adam')
    adam_source = inspect.getsource(optimizer_module._single_tensor_adam)
    require('grad = grad.add(param, alpha=weight_decay)' in adam_source,
            'Inspect changed Adam implementation before proceeding')
    for path in (Path(__file__).resolve(), Path(__file__).with_name('test_forecast_residual_gradients.py'),
                 Path(inspect.getfile(optimizer_module)), Path(inspect.getfile(torch.nn.utils.clip_grad_norm_))):
        pins[str(path)] = sha256(path)
    verify_paths(pins, 'Verify inputs, code and selected graphs')
    manifest = read(directory/'manifest.json')
    require(manifest['ids'] == saved['ids'], 'Cache cohort differs')
    data = {name: {c: value for c, value in read(run.cache.base.PREPARED/name).items() if c in ids}
            for name in ('inputs.json', 'labels.json')}
    for c in ids:
        x, y = data['inputs.json'][c], data['labels.json'][c]
        run.cache.base.contract.predictor_view(x)
        require(y['eligible']['endpoint'], 'Ineligible training target')
        run.cache.base.contract.score(y['y']['endpoint'])
        require((date.fromisoformat(y['target_date']) - date.fromisoformat(x['input_end_date'])).days == 30,
                'Wrong prediction horizon')
    identity = {'protocol': PROTOCOL, 'selected_train_ids': ids,
                'batches': [ids[i:i+8] for i in range(0, len(ids), 8)],
                'source_lock_sha256': SOURCE_LOCK, 'pinned_files': pins,
                'config': saved['config'], 'torch_version': str(torch.__version__),
                'limitations': LIMITATIONS}
    return saved, data, run.GraphBank(directory, ids, capacity=32), identity


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        for name, value in row['groups'].items():
            grouped[(row['state'], row['seed'], row['variant'], name)].append(value)
    result = []
    for (state, seed, variant, name), values in sorted(grouped.items()):
        ratios = [v['decay_to_data_ratio'] for v in values if v['decay_to_data_ratio'] is not None]
        result.append({'state': state, 'seed': seed, 'variant': variant, 'group': name,
                       'batches': len(values),
                       'decay_larger_batches': sum(v['comparison'] in ('decay_only', 'decay_larger') for v in values),
                       'zero_data_gradient_batches': sum(v['clipped_data_gradient_l2'] == 0 for v in values),
                       'median_decay_to_data_ratio': statistics.median(ratios) if ratios else None,
                       'median_data_gradient_l2': statistics.median(v['clipped_data_gradient_l2'] for v in values),
                       'parameter_l2': values[0]['parameter_l2'],
                       'decay_l2': values[0]['decay_l2']})
    return result


def diagnose(saved, data, bank, identity):
    rows = []
    combinations = [(s, v, state) for s in PROTOCOL['seeds'] for v in run.VARIANTS
                    for state in PROTOCOL['states']]
    for seed, variant, state in run.cache.tqdm(combinations, desc='Training-only gradient checks', unit='model'):
        mode = 'epss_only' if variant == 'epss_only' else 'tpg_epss'
        initial = torch.load(run.DEFAULT_OUTPUT/f'initialization/seed_{seed}/{mode}.pt',
                             weights_only=True, map_location='cpu')
        if state == 'initialization':
            with torch.random.fork_rng(devices=[]):
                net = run.residual.ResidualTPGRegressor(**initial['spec'])
            net.load_state_dict(initial['state_dict'], strict=True)
            require(run.digest_state(net.state_dict()) == initial['initial_state_sha256'], 'Changed initialization')
            epoch = 0
        else:
            checkpoint = torch.load(run.DEFAULT_OUTPUT/f'fits/{variant}__{seed}/checkpoint.pt',
                                    weights_only=True, map_location='cpu')
            require(checkpoint['matched_experiment'] == run.VERSION and checkpoint['seed'] == seed
                    and checkpoint['edge_configuration'] == variant
                    and checkpoint['fit_cve_ids'] == saved['ids']['train']
                    and checkpoint['initial_state_sha256'] == initial['initial_state_sha256'],
                    'Checkpoint provenance differs')
            net = run.residual.restore(checkpoint)
            epoch = checkpoint['selected_epoch']
        require(net.config['correction_gain'] == saved['config']['correction_gain'], 'Wrong residual gain')
        net.train()
        before = run.digest_state(net.state_dict())
        graphs = bank.view('no_coref' if variant == 'epss_only' else variant)
        for index, ids in enumerate(identity['batches']):
            batch = task.make_batch(ids, graphs, data['inputs.json'])
            require(not any(k in batch for k in ('y', 'target_date', 'cve')), 'Target in model inputs')
            y = torch.tensor([data['labels.json'][c]['y']['endpoint'] for c in ids], dtype=torch.float32)
            base, correction, raw, predicted = net.details(batch)
            loss = torch.nn.functional.smooth_l1_loss(predicted, y, beta=saved['config']['huber_beta'])
            stats = measure_gradients(net, loss, saved['config']['weight_decay'],
                                      saved['config']['gradient_clip_norm'])
            rows.append({'seed': seed, 'variant': variant, 'state': state, 'selected_epoch': epoch,
                         'batch': index, 'ids': ids, 'loss': float(loss.detach()),
                         'targets_equal_persistence_float32': int((base.detach() == y).sum()),
                         'clipped_outputs': int(((raw.detach() < 0) | (raw.detach() > 1)).sum()),
                         **stats})
        require(run.digest_state(net.state_dict()) == before, 'Forward changed weights or buffers')
    return rows


def report(summary):
    lines = ['DQ19: Data-loss gradients versus weight decay', '',
             'Scope: 32 training CVEs, four fixed batches, 12 fits, initialization and selected checkpoint.',
             '96 backward passes; no optimizer steps, new embeddings, validation or test scoring.',
             'The same batches are used for every seed, graph variant and checkpoint state.', '',
             'Data-loss gradient: the learning signal from the training targets.',
             'Decay term: 0.0001 times each parameter, added by Adam after external gradient clipping.',
             'Ratio: decay L2 norm divided by clipped data-gradient L2 norm within the named module.',
             'A ratio above 1 means decay has the larger local norm, not necessarily the larger Adam update.',
             'NA means the data-gradient norm is zero in every sampled batch; see zero-gradient counts.',
             'Decay>data counts include a nonzero decay term with a zero data gradient.',
             'Group ratios summarize four separate batches, not one gradient averaged over 32 CVEs.', '',
             f"Clipped gradient batches: {summary['gradient_clipped_batches']}/96.",
             f"Outputs outside [0,1] before output clipping: {summary['clipped_output_evaluations']}.", '',
             'State | Seed | Model | Parameter group | Median ratio | Decay>data batches | Zero-data batches']
    for r in summary['groups']:
        ratio = r['median_decay_to_data_ratio']
        lines.append(f"{r['state']} | {r['seed']} | {r['variant']} | {r['group']} | "
                     f"{'NA' if ratio is None else format(ratio, '.6g')} | "
                     f"{r['decay_larger_batches']}/4 | {r['zero_data_gradient_batches']}/4")
    lines += ['', 'Interpretation limits'] + ['- ' + s for s in LIMITATIONS]
    return '\n'.join(lines) + '\n'


def execute(verify_only=False):
    require(verify_only or not OUTPUT.exists(), 'Output exists; use --verify-only to reproduce without overwriting')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    saved, data, bank, identity = context()
    rows = diagnose(saved, data, bank, identity)
    summary = {'schema': VERSION, 'training_cves': 32, 'backward_passes': len(rows),
               'optimizer_steps': 0, 'validation_scoring': False, 'test_scoring': False,
               'gradient_clipped_batches': sum(r['clip_applied'] for r in rows),
               'clipped_output_evaluations': sum(r['clipped_outputs'] for r in rows),
               'groups': aggregate(rows), 'limitations': LIMITATIONS}
    verify_paths(identity['pinned_files'], 'Confirm source files unchanged')
    require(sha256(run.DEFAULT_OUTPUT/'evaluation_lock.json') == SOURCE_LOCK, 'Changed R04 lock')
    source_files = read(run.DEFAULT_OUTPUT/'evaluation_lock.json')['files']
    verify_paths({str(run.DEFAULT_OUTPUT/n): h for n, h in source_files.items()}, 'Confirm saved runs unchanged')
    values = {'identity.json': identity, 'analysis.json': summary, 'gradient_details.json': rows}
    if verify_only:
        lock = read(OUTPUT/'diagnostic_lock.json')
        require(set(lock['files']) == FILES, 'Incomplete diagnostic lock')
        verify_paths({str(OUTPUT/n): h for n, h in lock['files'].items()}, 'Verify diagnostic hashes')
        for name, value in values.items():
            require(read(OUTPUT/name) == value, 'Reproduction differs: ' + name)
        require((OUTPUT/'analysis.txt').read_text() == report(summary), 'Text report differs')
        print('DQ19 training-only gradient audit reproduced. No optimizer steps or validation/test scoring.')
        return
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for name, value in values.items():
        write(OUTPUT/name, value)
    (OUTPUT/'analysis.txt').write_text(report(summary), encoding='utf-8')
    write(OUTPUT/'diagnostic_lock.json', {'schema': VERSION,
                                        'files': {n: sha256(OUTPUT/n) for n in sorted(FILES)}})
    print(f'DQ19 training-only gradient audit saved to {OUTPUT}. No optimizer steps or validation/test scoring.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true',
                        help='Repeat the bounded backward checks and compare without overwriting reports')
    execute(parser.parse_args().verify_only)


if __name__ == '__main__':
    main()
