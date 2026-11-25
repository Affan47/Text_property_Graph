"""Matched validation-only current-description TPG regression; small smoke run first."""
import argparse
from collections import defaultdict
import copy
import csv
import math
from pathlib import Path
import statistics

import torch
from torch import nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
from tqdm.auto import tqdm

if __package__:
    from . import forecast_tpg_data as cache
    from . import forecast_sequence as sequence
    from .forecast_baseline import regression_metrics, predict_persistence
    from .edge_aware_layers import EdgeTypeGNNLayer
else:
    import forecast_tpg_data as cache
    import forecast_sequence as sequence
    from forecast_baseline import regression_metrics, predict_persistence
    from edge_aware_layers import EdgeTypeGNNLayer

read_json, write_json, require, sha256 = cache.read_json, cache.write_json, cache.require, cache.sha256
VERSION = 'epss_retrospective_tpg_regression_v1'
MODES = ('epss_only', 'text_epss', 'tpg_only', 'tpg_epss', 'tpg_no_edges_epss')
RESULT_ROOT = cache.EPSS_ROOT/'05_results/02_evaluation/epss_day30_to_day60'
CONFIG = {'hidden_size': 32, 'temporal_hidden_size': 16, 'graph_layers': 2, 'batch_size': 8,
    'learning_rate': .001, 'weight_decay': .0001, 'huber_beta': .01, 'gradient_clip_norm': 1.0,
    'dropout': .1, 'epochs': 80, 'patience': 15, 'seeds': [17, 29, 43], 'device': 'cpu'}


def config_for(scale):
    require(scale in cache.SCALES, 'Unknown run scale')
    config = copy.deepcopy(CONFIG)
    if scale == 'smoke':
        config.update(epochs=3, patience=3, seeds=[17])
    return config


class TPGRegressor(nn.Module):
    def __init__(self, mode, node_dim, embedding_dim, edge_vocab, config):
        super().__init__()
        require(mode in MODES, 'Unknown experiment mode')
        require(sorted(edge_vocab.values()) == list(range(len(edge_vocab))), 'Invalid saved edge vocabulary')
        self.mode = mode
        self.config = config
        self.edge_vocab = dict(edge_vocab)
        hidden = config['hidden_size']
        size = 0
        if mode != 'tpg_only':
            self.temporal = nn.GRU(len(sequence.CHANNELS), config['temporal_hidden_size'], batch_first=True)
            size += config['temporal_hidden_size']
        if mode == 'text_epss':
            self.text = nn.Sequential(nn.Linear(embedding_dim, hidden), nn.GELU())
            size += hidden
        if mode.startswith('tpg'):
            self.node_project = nn.Sequential(nn.Linear(node_dim, hidden), nn.GELU())
            self.layers = nn.ModuleList([EdgeTypeGNNLayer(hidden, len(edge_vocab), dropout=config['dropout'])
                                        for _ in range(config['graph_layers'])])
            size += hidden
        self.head = nn.Sequential(nn.Linear(size, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, batch):
        representations = []
        if self.mode != 'tpg_only':
            _, h = self.temporal(batch.epss_sequence)
            representations.append(h[-1])
        if self.mode == 'text_epss':
            representations.append(self.text(batch.text_embedding))
        if self.mode.startswith('tpg'):
            x = self.node_project(batch.x)
            ei, et, ef = batch.edge_index, batch.edge_type, batch.edge_features
            if self.mode == 'tpg_no_edges_epss':
                # Same nodes, layers, pooling and initialization; only inter-node messages removed.
                ei, et, ef = ei[:, :0], et[:0], ef[:0]
            for layer in self.layers:
                x = layer(x, ei, et, ef)
            representations.append(global_mean_pool(x, batch.batch))
        return torch.sigmoid(self.head(torch.cat(representations, dim=1))).squeeze(-1)


def make_batch(ids, graphs, inputs):
    seq = sequence.input_tensor([inputs[c] for c in ids])
    rows = []
    for i, cve in enumerate(ids):
        g = graphs[cve]
        rows.append(Data(x=g['x'], edge_index=g['edge_index'], edge_type=g['edge_type'],
                         edge_features=g['edge_features'], text_embedding=g['text_embedding'][None],
                         epss_sequence=seq[i:i+1], num_nodes=len(g['x'])))
    # Targets, CVE IDs and dates are not attributes of the forward-pass input.
    return Batch.from_data_list(rows)


def infer(net, ids, graphs, inputs, batch_size, description='Validation', progress=True):
    net.eval()
    predictions = []
    starts = range(0, len(ids), batch_size)
    with torch.no_grad():
        for start in tqdm(starts, desc=description, unit='batch', leave=False, disable=not progress):
            pred = net(make_batch(ids[start:start+batch_size], graphs, inputs))
            require(torch.isfinite(pred).all().item() and ((pred >= 0) & (pred <= 1)).all().item(),
                    'Invalid model predictions')
            predictions.extend(pred.tolist())
    return predictions


def model_spec(mode, manifest, graphs, config):
    first = graphs[manifest['ids']['train'][0]]
    return {'mode': mode, 'node_dim': first['x'].shape[1], 'embedding_dim': manifest['embedding_dim'],
            'edge_vocab': manifest['type_vocabulary']['edge_types'], 'config': copy.deepcopy(config)}


def restore(saved):
    require(saved['schema'] == VERSION and saved['test_performance_evaluated'] is False, 'Wrong model contract')
    with torch.random.fork_rng(devices=[]):
        net = TPGRegressor(**saved['spec'])
    require(all(torch.isfinite(v).all().item() for v in saved['state_dict'].values()), 'Invalid checkpoint weights')
    net.load_state_dict(saved['state_dict'], strict=True)
    return net.eval()


def fit(mode, seed, manifest, graphs, data, config, progress=True):
    train, val = (manifest['ids'][s] for s in ('train', 'validation'))
    inputs, labels = data['inputs.json'], data['labels.json']
    actual = {s: [labels[c]['y']['endpoint'] for c in manifest['ids'][s]] for s in ('train', 'validation')}
    spec = model_spec(mode, manifest, graphs, config)
    history, best, best_mae, stale = [], None, math.inf, 0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = TPGRegressor(**spec)
        mean = statistics.mean(actual['train'])
        bounded = min(1-1e-5, max(1e-5, mean))
        with torch.no_grad():
            net.head[-1].bias.fill_(math.log(bounded/(1-bounded)))
        optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        criterion = nn.SmoothL1Loss(beta=config['huber_beta'])
        generator = torch.Generator().manual_seed(seed)
        epochs = tqdm(range(1, config['epochs']+1), desc=f'{mode}, seed {seed}', unit='epoch', disable=not progress)
        for epoch in epochs:
            net.train()
            total_loss = 0.0
            order = torch.randperm(len(train), generator=generator).tolist()
            for start in tqdm(range(0, len(train), config['batch_size']), desc='Training', unit='batch',
                              leave=False, disable=not progress):
                ids = [train[i] for i in order[start:start+config['batch_size']]]
                batch = make_batch(ids, graphs, inputs)
                y = torch.tensor([labels[c]['y']['endpoint'] for c in ids], dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(net(batch), y)
                require(torch.isfinite(loss).item(), 'Non-finite training loss')
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), config['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                total_loss += float(loss.detach())*len(ids)
            train_pred = infer(net, train, graphs, inputs, config['batch_size'], 'Training metrics', progress)
            val_pred = infer(net, val, graphs, inputs, config['batch_size'], progress=progress)
            train_mae = regression_metrics(actual['train'], train_pred)['mae']
            val_mae = regression_metrics(actual['validation'], val_pred)['mae']
            history.append({'epoch': epoch, 'training_huber_loss': total_loss/len(train),
                            'training_mae': train_mae, 'validation_mae': val_mae})
            epochs.set_postfix(train_mae=f'{train_mae:.6f}', val_mae=f'{val_mae:.6f}')
            if val_mae < best_mae:
                best_mae, stale = val_mae, 0
                best = {'schema': VERSION, 'spec': spec, 'seed': seed, 'selected_epoch': epoch,
                    'fit_cve_ids': train, 'selection_cve_ids': val, 'initial_head_bias_training_target_mean': mean,
                    'state_dict': {k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
                    'test_performance_evaluated': False, 'limitation': cache.LIMITATION}
            else:
                stale += 1
            if stale >= config['patience']:
                break
        epochs.close()
    require(best is not None, 'No trained checkpoint selected')
    return best, history


def metrics(actual, predicted, ids, data):
    result = regression_metrics(actual, predicted)
    errors = defaultdict(list)
    for c, y, p in zip(ids, actual, predicted):
        errors[data['split_manifest.json'][c]['group_id']].append(abs(y-p))
    result['equal_group_mae'] = statistics.mean(statistics.mean(v) for v in errors.values())
    return result


def report(result):
    lines = ['# Current-Description TPG Regression', '', cache.LIMITATION, '',
        f"Scale: {result['scale']}. Cohorts: {result['cohort_counts']}. Fixed settings: {result['config']}.",
        'Smoke runs are engineering checks, not performance evidence. No test scoring.', '',
        '## Comparisons', '',
        '| Model | Seed | Selected epoch | Validation MAE | Validation RMSE | Equal-group MAE |',
        '|---|---:|---:|---:|---:|---:|']
    for row in result['results']:
        m = row['metrics']
        lines.append(f"| {row['mode']} | {row['seed']} | {row['selected_epoch']} | {m['mae']:.8f} | {m['rmse']:.8f} | {m['equal_group_mae']:.8f} |")
    lines += ['', 'Every row uses the same validation CVEs and Day-60 target. Lower errors are better.',
        'MAE averages absolute score errors; RMSE gives greater weight to large errors.',
        'Equal-group MAE averages within related groups, then gives each group equal weight.',
        'An error of 0.01 is one percentage point of EPSS. This is not exploitation accuracy.', '',
        'epss_only: mask-aware 16-unit GRU on Days 0-30; no text.',
        'text_epss: mean frozen token embedding plus the same EPSS encoder; no graph edges.',
        'tpg_only: two 32-unit relation-aware layers and mean node pooling; no EPSS input.',
        'tpg_epss: the graph representation concatenated with the EPSS representation.',
        'tpg_no_edges_epss: same graph nodes, weights at initialization and layers as tpg_epss,',
        'but no inter-node messages. This helps isolate connections from extra node representations.',
        'persistence: last observed EPSS during Days 0-30, not necessarily Day 30.', '',
        'All neural modes use a sigmoid regression head, train-only mean bias initialization,',
        'Smooth L1 loss, Adam, gradient clipping and earliest best validation-MAE checkpoint.',
        'This is a direct-score head, not the earlier persistence-residual head.',
        'Relation IDs and 64-dimensional relation-property features enter GNN messages.',
        'Missing node embeddings have an explicit mask. Missing EPSS inputs retain masks and ages.',
        'No target is imputed. Learned relation extraction is not trained in this stage.',
        'The frozen pretrained encoder and current descriptions may contain later knowledge.',
        'Chronological splits do not remove that historical-information limitation.',
        'No CVE IDs, dates, graph labels or structured catalogue fields enter the forward pass.',
        'Every fixed seed is reported; there is no best-seed selection or tuning search.',
        'Validation has been inspected repeatedly. Its results are not an untouched final estimate.', '',
        '## Files', '',
        'summary.json/txt and README.md: settings, matched metrics and limitations.',
        'predictions.json/csv: per-CVE actual and predicted scores for validation only.',
        'training_history.json: training loss and training/validation MAE for every epoch.',
        'checkpoints/*.pt: tensor-only-loadable selected model states and settings.',
        'training_curves.png: epoch on x, score MAE on y; solid training, dashed validation.',
        'evaluation_lock.json: hashes of all successful outputs; run_status.json records failures.', '']
    return '\n'.join(lines)


def plot(histories, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(MODES), 1, figsize=(9, 14), constrained_layout=True)
    for ax, mode in zip(axes, MODES):
        for key, rows in histories.items():
            if key.startswith(mode+'__'):
                for metric, style in (('training_mae', '-'), ('validation_mae', '--')):
                    ax.plot([r['epoch'] for r in rows], [r[metric] for r in rows], style,
                            label=key.split('__')[1]+' '+metric.replace('_mae', ''))
        ax.set(title=mode, xlabel='Training epoch', ylabel='EPSS mean absolute error')
        ax.legend(fontsize=8)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def run(cache_dir, output=None, check_only=False):
    cache_dir = Path(cache_dir).resolve()
    manifest, graphs, data = cache.load_cache(cache_dir)
    output = Path(output or RESULT_ROOT/('tpg_description_'+manifest['scale']+'_v1')).resolve()
    cache.check_destination(output, [cache_dir, Path(manifest['prepared_directory']),
                                     Path(data['preparation_summary.json']['source_path']).parent])
    config = config_for(manifest['scale'])
    train, val = (manifest['ids'][s] for s in ('train', 'validation'))
    config['device'] = 'cpu'
    pins = dict(manifest['source_pins'])
    for p in Path(__file__).parent.glob('forecast_*.py'):
        pins[str(p.resolve())] = sha256(p)
    pins[str(Path(__file__).with_name('edge_aware_layers.py'))] = sha256(Path(__file__).with_name('edge_aware_layers.py'))
    pins.update({str(cache_dir/n):h for n,h in read_json(cache_dir/'cache_lock.json')['files'].items()})
    pins[str(cache_dir/'cache_lock.json')] = sha256(cache_dir/'cache_lock.json')
    identity = {'cache_directory': str(cache_dir), 'pinned_files': pins,
                'schema': VERSION, 'scale': manifest['scale'], 'config': config,
                'torch_version': str(torch.__version__), 'limitation': cache.LIMITATION}
    for mode in tqdm(MODES, desc='Check all model forward passes', unit='model'):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            net = TPGRegressor(**model_spec(mode, manifest, graphs, config)).eval()
        require(len(infer(net, train[:2], graphs, data['inputs.json'], 2, progress=False)) == min(2,len(train)),
                'Forward batch size differs')
    if check_only:
        cache.check_pins(pins)
        print('All five model preflights passed; no training, output writes or test scoring.')
        return
    output.mkdir(parents=True, exist_ok=False)
    (output/'checkpoints').mkdir()
    write_json(output/'run_status.json', {'status': 'running', 'identity': identity})
    try:
        actual = [data['labels.json'][c]['y']['endpoint'] for c in val]
        persistence = [predict_persistence(data['inputs.json'][c])['prediction'] for c in val]
        rows = [{'cve':c, 'group_id':data['split_manifest.json'][c]['group_id'],
                 'prediction_anchor_date':data['inputs.json'][c]['input_end_date'],
                 'target_date':data['labels.json'][c]['target_date'], 'actual_day60_epss':y,
                 'persistence':p} for c,y,p in zip(val,actual,persistence)]
        results = [{'mode':'persistence', 'seed':None, 'selected_epoch':None,
                    'metrics':metrics(actual, persistence, val, data)}]
        histories = {}
        for mode in tqdm(MODES, desc='Matched experiments', unit='model'):
            for seed in tqdm(config['seeds'], desc=mode+' seeds', unit='seed', leave=False):
                model, history = fit(mode, seed, manifest, graphs, data, config)
                key = mode+'__seed_'+str(seed)
                path = output/'checkpoints'/(key+'.pt')
                model['cache_lock_sha256'] = pins[str(cache_dir/'cache_lock.json')]
                torch.save(model, path)
                pred = infer(restore(model), val, graphs, data['inputs.json'], config['batch_size'])
                reloaded = torch.load(path, weights_only=True, map_location='cpu')
                require(pred == infer(restore(reloaded), val, graphs, data['inputs.json'], config['batch_size'], progress=False),
                        'Saved checkpoint predictions differ')
                histories[key] = history
                for row, p in zip(rows, pred):
                    row[key] = p
                result = {'mode':mode, 'seed':seed, 'selected_epoch':model['selected_epoch'],
                          'metrics':metrics(actual, pred, val, data)}
                results.append(result)
                tqdm.write(f"{mode}, seed {seed}: validation MAE={result['metrics']['mae']:.8f}; epoch {model['selected_epoch']}")
        summary = {'schema':VERSION, 'scale':manifest['scale'], 'config':config, 'identity':identity,
            'cohort_counts':{s:len(v) for s,v in manifest['ids'].items()}, 'cohort_ids':manifest['ids'],
            'results':results, 'limitation':cache.LIMITATION, 'test_performance_evaluated':False,
            'split_changes_performed':False, 'best_seed_selection_performed':False,
            'seed_mae_summary':{mode:{'mean':statistics.mean(r['metrics']['mae'] for r in results if r['mode']==mode),
                'sample_sd':statistics.stdev(r['metrics']['mae'] for r in results if r['mode']==mode) if len(config['seeds'])>1 else None}
                for mode in MODES}}
        with tqdm(total=4, desc='Save reports and verify provenance', unit='stage') as bar:
            write_json(output/'summary.json', summary)
            write_json(output/'predictions.json', rows)
            write_json(output/'training_history.json', histories)
            with (output/'predictions.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            for name in ('summary.txt', 'README.md'):
                (output/name).write_text(report(summary), encoding='utf-8')
            bar.update()
            plot(histories, output/'training_curves.png')
            bar.update()
            cache.check_pins(pins)
            bar.update()
            write_json(output/'run_status.json', {'status':'complete', 'identity':identity})
            write_json(output/'evaluation_lock.json', {'schema':VERSION,
                'test_performance_evaluated':False,
                'files':{str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}})
            bar.update()
        print(f'Retrospective TPG {manifest["scale"]} run saved to {output}. Validation only; no test scoring.')
    except Exception as exc:
        write_json(output/'run_status.json', {'status':'failed', 'error':str(exc), 'identity':identity})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, default=cache.CACHE_ROOT/'description_smoke_v1')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        run(args.cache_dir, args.output_dir, args.check_only)
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


if __name__ == '__main__':
    main()
