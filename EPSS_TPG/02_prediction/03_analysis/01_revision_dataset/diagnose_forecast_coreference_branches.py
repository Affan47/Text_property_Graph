"""DQ18: fixed R03 branch-response diagnostics, smoke first, validation only."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_coreference_training as run

torch, task = run.torch, run.task
require, read, write, sha256 = run.require, run.read_json, run.write_json, run.sha256
VERSION = 'dq18_r03_branch_response_v1'
ROOT = task.RESULT_ROOT/'coreference_branch_diagnostic_v1'
PROTOCOL = {
    'schema': VERSION, 'model_seeds': [17, 29, 43], 'variants': list(run.VARIANTS),
    'donor_seeds': [101, 211, 307], 'smoke_cves': 16,
    'observed_score_shift': .01, 'response_threshold': 1e-7,
    'prediction_atol': 1e-7, 'prediction_rtol': 1e-6,
    'probes': ['original', 'swap_epss', 'swap_graph', 'zero_epss', 'zero_graph',
               'observed_plus', 'observed_minus'],
    'split': 'validation', 'training': False, 'test_scoring': False,
    'node_embeddings_regenerated': False, 'threads': 1, 'device': 'cpu',
}
FILES = {'identity.json', 'analysis.json', 'analysis.txt', 'predictions.json'}


def select_ids(ids, scale):
    require(scale in ('smoke', 'full'), 'Unknown scale')
    require(len(ids) == len(set(ids)), 'Duplicate CVE IDs')
    if scale == 'full':
        return list(ids)
    chosen = set(sorted(ids, key=lambda c: hashlib.sha256(('dq18:smoke:'+c).encode()).digest())[:PROTOCOL['smoke_cves']])
    return [c for c in ids if c in chosen]


def donor_map(ids, groups, seed):
    """Rotate grouped blocks, as in DQ09; every donor is from a different group."""
    sizes = Counter(groups[c] for c in ids)
    require(len(ids) == len(set(ids)) and len(ids) >= 2, 'Need distinct CVEs')
    shift = max(sizes.values())
    require(shift*2 <= len(ids), 'No different-group bijection; do not relax grouping')
    key = lambda x: hashlib.sha256(f'dq18:{seed}:{x}'.encode()).digest()
    ordered = [c for g in sorted(sizes, key=key) for c in sorted(ids, key=key) if groups[c] == g]
    donors = ordered[shift:]+ordered[:shift]
    require(all(groups[c] != groups[d] for c,d in zip(ordered, donors)), 'Related donor')
    return dict(zip(ordered, donors))


def close(a, b):
    return len(a) == len(b) and torch.allclose(torch.tensor(a, dtype=torch.float64),
        torch.tensor(b, dtype=torch.float64), atol=PROTOCOL['prediction_atol'], rtol=PROTOCOL['prediction_rtol'])


def shifted_sequence(sequence, amount):
    changed = sequence.clone()
    observed = sequence[:, :, 1] == 1
    changed[:, :, 0] = torch.where(observed, (sequence[:, :, 0]+amount).clamp(0, 1), sequence[:, :, 0])
    require(torch.equal(changed[:, :, 1:], sequence[:, :, 1:]) and
            torch.equal(changed[:, :, 0][~observed], sequence[:, :, 0][~observed]), 'Changed missingness/age')
    return changed


def encode(net, ids, graphs, inputs, shift=0.0, progress=True):
    """Capture the actual head input, never reconstruct GNN or GRU formulas."""
    require(net.mode in ('tpg_epss', 'epss_only'), 'Unsupported branch layout')
    representations, predictions = [], []

    def capture(module, args):
        representations.append(args[0].detach().clone())

    handle = net.head.register_forward_pre_hook(capture)
    net.eval()
    try:
        with torch.no_grad():
            starts = range(0, len(ids), net.config['batch_size'])
            for start in run.cache.tqdm(starts, desc='Encode validation batches', unit='batch', leave=False, disable=not progress):
                selected = ids[start:start+net.config['batch_size']]
                batch = task.make_batch(selected, graphs, inputs)
                if shift:
                    batch.epss_sequence = shifted_sequence(batch.epss_sequence, shift)
                values = net(batch)
                require(bool(torch.isfinite(values).all()) and bool(((values >= 0) & (values <= 1)).all()), 'Invalid output')
                predictions.extend(values.tolist())
    finally:
        handle.remove()
    result = torch.cat(representations)
    width = net.config['temporal_hidden_size'] + (net.config['hidden_size'] if net.mode == 'tpg_epss' else 0)
    require(result.shape == (len(ids), width) and bool(torch.isfinite(result).all()), 'Head-input shape differs')
    return predictions, result


def intervene(representation, ids, temporal_width, probe, donors=None):
    require(probe in ('original', 'swap_epss', 'swap_graph', 'zero_epss', 'zero_graph'), 'Unknown latent probe')
    require(len(ids) == len(representation) and 0 < temporal_width <= representation.shape[1], 'Branch alignment differs')
    changed = representation.clone()
    if probe.startswith('swap_'):
        require(donors is not None and set(donors) == set(ids) and set(donors.values()) == set(ids), 'Invalid donor bijection')
        position = {c:i for i,c in enumerate(ids)}
        indexes = [position[donors[c]] for c in ids]
        columns = slice(0, temporal_width) if probe == 'swap_epss' else slice(temporal_width, None)
        changed[:, columns] = representation[indexes, columns]
    elif probe == 'zero_epss':
        changed[:, :temporal_width] = 0
    elif probe == 'zero_graph':
        changed[:, temporal_width:] = 0
    return changed


def head_predictions(net, representations):
    net.eval()
    with torch.no_grad():
        values = torch.cat([torch.sigmoid(net.head(x)).squeeze(-1)
                            for x in representations.split(net.config['batch_size'])])
    require(bool(torch.isfinite(values).all()) and bool(((values >= 0) & (values <= 1)).all()), 'Invalid head output')
    return values.tolist()


def response(original, predicted, actual):
    require(len(original) == len(predicted) == len(actual), 'Response alignment differs')
    if not actual:
        return {'n': 0, 'mean_absolute_prediction_change': None, 'mean_signed_prediction_change': None,
                'max_absolute_prediction_change': None, 'changed_cves': 0, 'original_mae': None,
                'probe_mae': None, 'mae_difference': None}
    change = [b-a for a,b in zip(original, predicted)]
    base_mae = statistics.mean(abs(p-y) for p,y in zip(original, actual))
    probe_mae = statistics.mean(abs(p-y) for p,y in zip(predicted, actual))
    return {'n': len(actual), 'mean_absolute_prediction_change': statistics.mean(map(abs, change)),
        'mean_signed_prediction_change': statistics.mean(change), 'max_absolute_prediction_change': max(map(abs, change)),
        'changed_cves': sum(abs(x) > PROTOCOL['response_threshold'] for x in change),
        'original_mae': base_mae, 'probe_mae': probe_mae, 'mae_difference': probe_mae-base_mae}


def subsets(ids, data, manifest):
    graph = {r['cve']: r['edges_by_mode'] for r in manifest['graphs']}
    observed = {c: sum(data['inputs.json'][c]['X']['observed_mask']) for c in ids}
    return {'all': list(range(len(ids))),
        '31_observed': [i for i,c in enumerate(ids) if observed[c] == 31],
        '30_observed': [i for i,c in enumerate(ids) if observed[c] == 30],
        'at_most_29_observed': [i for i,c in enumerate(ids) if observed[c] <= 29],
        'with_coref': [i for i,c in enumerate(ids) if graph[c]['all_coref'] > graph[c]['no_coref']],
        'without_coref': [i for i,c in enumerate(ids) if graph[c]['all_coref'] == graph[c]['no_coref']]}


def diagnose(manifest, views, data, identity, models, baseline, progress=True):
    ids = identity['ids']
    require(len(ids) == len(set(ids)) and set(ids) <= set(manifest['ids']['validation']), 'Only selected validation CVEs allowed')
    expected = {f'{m}__{s}' for m in PROTOCOL['variants'] for s in PROTOCOL['model_seeds']}
    require(set(models) == expected, 'All fixed models required')
    group_ids = {c: data['split_manifest.json'][c]['group_id'] for c in ids}
    require(identity['donor_maps'] == {str(s): donor_map(ids, group_ids, s) for s in PROTOCOL['donor_seeds']}, 'Donor maps differ')
    actual = [data['labels.json'][c]['y']['endpoint'] for c in ids]
    require(actual == [baseline[c]['actual_day60_epss'] for c in ids], 'Recipient targets differ')
    groups = subsets(ids, data, manifest)
    responses, rows, checks = [], [], []
    planned = [('original', None)] + [(p, s) for p in ('swap_epss', 'swap_graph') for s in PROTOCOL['donor_seeds']]
    planned += [(p, None) for p in ('zero_epss', 'zero_graph', 'observed_plus', 'observed_minus')]
    for key, saved in run.cache.tqdm(models.items(), desc='Diagnose fixed checkpoints', unit='model', disable=not progress):
        mode, seed = saved['edge_configuration'], saved['seed']
        net = task.restore(saved)
        before = run.digest_state(net.state_dict())
        original, representation = encode(net, ids, views[mode], data['inputs.json'], progress=progress)
        require(close(original, [baseline[c]['predictions'][key] for c in ids]), 'Saved predictions differ')
        require(close(original, head_predictions(net, representation)), 'Captured head does not reproduce output')
        temporal_width = net.config['temporal_hidden_size']
        for probe, permutation in run.cache.tqdm(planned, desc=key+' interventions', unit='probe', leave=False, disable=not progress):
            donors = identity['donor_maps'][str(permutation)] if permutation is not None else None
            if probe.startswith('observed_'):
                amount = PROTOCOL['observed_score_shift']*(1 if probe == 'observed_plus' else -1)
                predictions, shifted = encode(net, ids, views[mode], data['inputs.json'], amount, progress)
                require(torch.equal(shifted[:, temporal_width:], representation[:, temporal_width:]), 'EPSS shift changed graph branch')
            else:
                predictions = head_predictions(net, intervene(representation, ids, temporal_width, probe, donors))
            if mode == 'epss_only' and probe in ('swap_graph', 'zero_graph'):
                require(predictions == original, 'Graph intervention changed EPSS-only control')
            for subset, indexes in groups.items():
                responses.append({'variant': mode, 'seed': seed, 'probe': probe, 'donor_seed': permutation,
                    'subset': subset, 'small_group_below_20': len(indexes) < 20,
                    **response([original[i] for i in indexes], [predictions[i] for i in indexes], [actual[i] for i in indexes])})
            rows.extend({'cve': c, 'variant': mode, 'seed': seed, 'probe': probe,
                'donor_seed': permutation, 'donor_cve': donors[c] if donors else None,
                'original_prediction': original[i], 'probe_prediction': predictions[i],
                'actual_day60_epss': actual[i]} for i,c in enumerate(ids))
        require(run.digest_state(net.state_dict()) == before and all(p.grad is None for p in net.parameters()), 'Model mutated')
        checks.append({'variant': mode, 'seed': seed, 'state_unchanged': True, 'gradients_created': False,
            'original_predictions_reproduced': True,
            'temporal_latent_mean_norm': float(representation[:, :temporal_width].norm(dim=1).mean()),
            'graph_latent_mean_norm': float(representation[:, temporal_width:].norm(dim=1).mean()) if mode != 'epss_only' else None})
    # Collapse donor repetitions within each seed before describing between-seed variation.
    aggregates = []
    for mode in PROTOCOL['variants']:
        for probe in PROTOCOL['probes']:
            metrics = {}
            for metric in ('mean_absolute_prediction_change', 'mae_difference'):
                values = [statistics.mean(r[metric] for r in responses if r['variant'] == mode and
                    r['seed'] == seed and r['probe'] == probe and r['subset'] == 'all') for seed in PROTOCOL['model_seeds']]
                metrics[metric] = {'per_seed': dict(zip(map(str, PROTOCOL['model_seeds']), values)),
                                  'mean': statistics.mean(values), 'sample_std': statistics.stdev(values)}
            aggregates.append({'variant': mode, 'probe': probe, **metrics})
    return {'schema': VERSION, 'scale': identity['scale'], 'cves': len(ids), 'responses': responses,
            'seed_aggregates': aggregates, 'checks': checks, 'test_scoring': False, 'training': False}, rows


def report(result):
    lines = ['DQ18: Saved R03 Branch Responses', '', f"Scale: {result['scale']}; validation CVEs: {result['cves']}.",
        'No training, new embeddings, graph generation or test scoring. All 12 saved models are included.',
        'Original predictions must reproduce. Checkpoint states must remain unchanged.', '',
        'What is changed?',
        '- swap_epss: use another unrelated validation CVE\'s encoded EPSS history, including its masks and age information.',
        '- swap_graph: use another unrelated validation CVE\'s graph embedding. The recipient EPSS branch stays fixed.',
        '- zero_epss / zero_graph: replace one encoded branch with zeros immediately before the existing head.',
        '- observed_plus / observed_minus: add/subtract 0.01 from observed past EPSS scores, clipped to [0,1].',
        '  Missing-score padding, observed masks, ages and day indexes stay unchanged.',
        '  Clipping can reduce the applied change near 0 or 1.',
        'Swaps use three fixed different-group bijections, shared across model variants and seeds.',
        'Separate encoders make latent swaps equivalent to swapping their source inputs in evaluation mode.',
        'Donors are not matched on publication date or EPSS level; swaps deliberately disrupt the CVE-history pairing.',
        'Graph probes on EPSS-only models must be exact no-ops. Zeroing and swaps can create unfamiliar inputs.', '',
        'How to read the results',
        'Mean absolute prediction change measures response strength, not improvement.',
        'MAE difference = intervened error minus original error on the recipient CVE targets.',
        'Positive MAE difference means worse predictions. Negative means lower error under that artificial intervention.',
        'Neither direction proves causality, reliable attribution, or improvement from retraining without that branch.',
        'Changes above 0.0000001 are counted as numerical responses, not scientifically meaningful effects.',
        'Latent norms are diagnostics of scale only, not branch importance.', '',
        'Variant | Probe | Mean absolute prediction change | MAE difference | Seed SD of prediction change']
    for r in result['seed_aggregates']:
        lines.append(f"{r['variant']} | {r['probe']} | {r['mean_absolute_prediction_change']['mean']:.8f} | "
            f"{r['mae_difference']['mean']:.8f} | {r['mean_absolute_prediction_change']['sample_std']:.8f}")
    lines += ['', 'Each average first averages the three donor maps within a model seed, then all three model seeds.',
        'The sample standard deviation describes three model seeds, not independent CVEs or a confidence interval.',
        'analysis.json retains every seed, donor map result, completeness group and with/without-COREF subset.',
        'predictions.json retains every recipient, donor and intervened prediction. Empty subsets have null errors.',
        'COREF groups describe edge presence in the all-COREF reference graph, even when the evaluated model removes those edges.',
        'Subgroups overlap; groups below 20 CVEs are marked. Smoke is a software check, not the main result.', '',
        'Decision boundary',
        'Inspect whether narrow-output seeds respond to either branch, and whether graph sensitivity accompanies lower error.',
        'Do not select a winning seed, remove hard CVEs or rewrite R03 results from this diagnostic.',
        'Only after the full diagnostic should a separate persistence-plus-correction fusion experiment be specified,',
        'with a matched EPSS-only residual control and unchanged development cohorts.',
        'Historical descriptions remain unverified; these are retrospective score forecasts, not exploitation probabilities.',
        'Repeated validation inspection limits confirmatory claims. No causal conclusions or test results.', '']
    return '\n'.join(lines)


def check_files(output, identity):
    lock = read(output/'diagnostic_lock.json')
    require(lock['schema'] == VERSION and set(lock['files']) == FILES, 'Wrong diagnostic file contract')
    require({p.name for p in output.iterdir()} == FILES | {'diagnostic_lock.json'}, 'Unexpected diagnostic files')
    require(read(output/'identity.json') == identity, 'Diagnostic identity differs')
    for name, digest in lock['files'].items():
        require(sha256(output/name) == digest, 'Changed diagnostic: '+name)


def context(scale):
    source = run.DEFAULT_OUTPUT
    source_identity = read(source/'run_identity.json')
    manifest, views, data, current = run.context(source_identity['cache_directory'])
    run.verify_output(source, manifest, views, data, current)
    ids = select_ids(manifest['ids']['validation'], scale)
    groups = {c: data['split_manifest.json'][c]['group_id'] for c in ids}
    pins = dict(current['pinned_files'])
    pins.update({str(source/n): h for n,h in read(source/'evaluation_lock.json')['files'].items()})
    for path in (source/'evaluation_lock.json', Path(__file__), Path(__file__).with_name('test_forecast_coreference_branches.py')):
        pins[str(path.resolve())] = sha256(path)
    identity = {'schema': VERSION, 'scale': scale, 'protocol': deepcopy(PROTOCOL), 'ids': ids,
        'donor_maps': {str(s): donor_map(ids, groups, s) for s in PROTOCOL['donor_seeds']},
        'source_directory': str(source), 'source_lock_sha256': sha256(source/'evaluation_lock.json'),
        'pinned_files': pins, 'torch_version': str(torch.__version__)}
    baseline = {r['cve']: r for r in read(source/'predictions.json')}
    models = {}
    for mode in PROTOCOL['variants']:
        for seed in PROTOCOL['model_seeds']:
            key = f'{mode}__{seed}'
            models[key] = torch.load(source/'fits'/key/'checkpoint.pt', map_location='cpu', weights_only=True)
    return manifest, views, data, identity, models, baseline


def execute(scale, check_only=False, verify_only=False):
    manifest, views, data, identity, models, baseline = context(scale)
    output = ROOT/scale
    if check_only:
        for key, saved in run.cache.tqdm(models.items(), desc='Preflight original predictions', unit='model'):
            pred, representation = encode(task.restore(saved), identity['ids'], views[saved['edge_configuration']], data['inputs.json'])
            require(close(pred, [baseline[c]['predictions'][key] for c in identity['ids']]), 'Baseline differs')
        run.cache.base.check_pins(identity['pinned_files'])
        print('DQ18 preflight passed. Original predictions only; no interventions, fitting or output writes.')
        return
    if scale == 'full':
        smoke = deepcopy(identity)
        smoke['scale'] = 'smoke'
        smoke['ids'] = select_ids(manifest['ids']['validation'], 'smoke')
        group = {c: data['split_manifest.json'][c]['group_id'] for c in smoke['ids']}
        smoke['donor_maps'] = {str(s): donor_map(smoke['ids'], group, s) for s in PROTOCOL['donor_seeds']}
        check_files(ROOT/'smoke', smoke)
    if verify_only:
        check_files(output, identity)
    else:
        require(not output.exists(), 'Output already exists; use --verify-only')
    result, rows = diagnose(manifest, views, data, identity, models, baseline)
    run.cache.base.check_pins(identity['pinned_files'])
    text = report(result)
    if verify_only:
        require(read(output/'analysis.json') == result and read(output/'predictions.json') == rows and
                (output/'analysis.txt').read_text() == text, 'Diagnostic does not reproduce')
        print(f'DQ18 {scale} diagnostic reproduced. No training or test scoring.')
        return
    output.mkdir(parents=True, exist_ok=False)
    with run.cache.tqdm(total=5, desc='Save separate diagnostic and lock', unit='file') as bar:
        for name, value in (('identity.json', identity), ('analysis.json', result), ('predictions.json', rows)):
            write(output/name, value); bar.update()
        (output/'analysis.txt').write_text(text); bar.update()
        write(output/'diagnostic_lock.json', {'schema': VERSION, 'files': {n: sha256(output/n) for n in sorted(FILES)}}); bar.update()
    check_files(output, identity)
    print(f'DQ18 {scale} saved to {output}. No training, new embeddings or test scoring.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument('--check-only', action='store_true')
    flags.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    threads = torch.get_num_threads()
    deterministic, warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        execute(args.scale, args.check_only, args.verify_only)
    finally:
        torch.set_num_threads(threads); torch.use_deterministic_algorithms(deterministic, warn_only=warn)


if __name__ == '__main__':
    main()
