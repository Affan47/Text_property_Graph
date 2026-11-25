"""Replay the supplied external pilot without importing its training loops."""
import argparse
import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for rel in ('01_tpg/01_core', '01_tpg/10_extensions', '02_prediction/01_package', '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT/rel))

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from tqdm import tqdm
from epss.combined_readout_v1 import data
from epss.continuous_forecasting import data as publication
from epss.reddit_development_v1.common import compare_ci

c, require = data.c, data.require
DEFAULT = Path('external_probes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-dir', type=Path, default=DEFAULT)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(4)
    names = ['split_extract.py', 'fusion_control.py', 'fusion_control.json', 'fusion_control_predictions.npz',
             'fusion_control_checkpoints.pt', 'pooled_views.npz', 'pooled_DM_split.npz']
    pins = {str(args.probe_dir/n): c.sha(args.probe_dir/n) for n in names}
    pins[str(Path(__file__).resolve())] = c.sha(__file__)
    z = np.load(args.probe_dir/'pooled_views.npz', allow_pickle=False)
    split = np.load(args.probe_dir/'pooled_DM_split.npz', allow_pickle=False)
    saved = np.load(args.probe_dir/'fusion_control_predictions.npz', allow_pickle=False)
    report = c.read(args.probe_dir/'fusion_control.json')
    checkpoints = torch.load(args.probe_dir/'fusion_control_checkpoints.pt', map_location='cpu', weights_only=True)
    ctx = data.base.context('previous', 'full'); train = z['train']; validation = ~train
    ids = ctx['ids']['train']+ctx['ids']['validation']
    require(z['ids'].tolist() == split['ids'].tolist() == ids and saved['ids'].tolist() == ctx['ids']['validation'], 'Wrong pilot IDs')
    require(train.tolist() == [True]*len(ctx['ids']['train'])+[False]*len(ctx['ids']['validation']), 'Pilot roles changed')
    expected_y = np.array([ctx['labels'][cid]['value'] for cid in ids], dtype=np.float32)
    require(np.array_equal(z['y'], expected_y) and np.array_equal(saved['actual'], expected_y[validation]), 'Pilot targets differ')
    # Only the reviewed network class is compiled. The module-level fitting loops never execute.
    tree = ast.parse((args.probe_dir/'fusion_control.py').read_text())
    definitions = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Net']
    require(len(definitions) == 1, 'Unexpected pilot class definition')
    namespace = {'nn': nn, 'torch': torch}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), 'reviewed_pilot_class', 'exec'), namespace)
    Net = namespace['Net']
    specs = {'A': (['DM'], 32), 'B': (['DM'], 66), 'C': (['Dsplit', 'Msplit'], 16),
             'D': (['Dsplit', 'Msplit'], 32), 'E': (['Dsep', 'Msep'], 16), 'F': (['Dsep', 'Msep'], 32)}
    arrays = {'DM': z['x_DM'], 'Dsep': z['x_D'], 'Msep': z['x_M'], 'Dsplit': split['x_D'], 'Msplit': split['x_M']}
    observed = {'DM': z['has_DM'], 'Dsep': z['has_D'], 'Msep': z['has_M'],
                'Dsplit': (split['n_D'] > 0).astype(np.float32), 'Msplit': (split['n_M'] > 0).astype(np.float32)}
    y = saved['actual']; high = y >= .1; replay = []
    def metrics(p):
        return {'mae': float(abs(p-y).mean()), 'rmse': float(np.sqrt(((p-y)**2).mean())),
            'spearman': float(spearmanr(y, p).statistic), 'auroc': float(roc_auc_score(high, p)),
            'ap': float(average_precision_score(high, p)), 'top50': int(high[np.argsort(-p)[:50]].sum()),
            'mae_hi': float(abs(p-y)[high].mean()), 'mae_lo': float(abs(p-y)[~high].mean()),
            'found': int((p[high] >= .1).sum()), 'pred_ge': int((p >= .1).sum())}
    for key, cp in tqdm(sorted(checkpoints.items()), desc='Replay 36 supplied frozen neural checkpoints', unit='fit'):
        arm, loss, seed = key.split('|'); views, width = specs[arm]; net = Net(len(views), width)
        net.load_state_dict(cp['model'], strict=True); net.eval()
        with torch.inference_mode():
            pred = torch.sigmoid(net([torch.tensor(arrays[v][validation]) for v in views],
                                      [torch.tensor(observed[v][validation]) for v in views],
                                      torch.tensor(z['mask_DM'][validation]), torch.tensor(z['struct'][validation]))).numpy()
        difference = float(abs(pred-saved[key]).max())
        require(difference <= 1e-7, 'Pilot checkpoint replay differs')
        record = next(v for k, v in report.items() if k.startswith(arm+' ') and k.endswith('/'+loss))
        expected = record['seeds'][[17, 29, 43].index(int(seed))]
        require(record['parameters'] == sum(p.numel() for p in net.parameters()), 'Pilot capacity differs')
        require(expected['epoch'] == cp['epoch'], 'Pilot epoch mismatch')
        for metric, value in metrics(saved[key]).items():
            require(abs(value-expected[metric]) < 1e-7, 'Pilot saved metric differs: '+metric)
        replay.append({'fit': key, 'max_abs_prediction_difference': difference, 'parameters': record['parameters'], 'best_epoch': cp['epoch']})
    require(len(replay) == 36, 'Incomplete pilot checkpoint collection')
    for name, expected in report['logistic'].items():
        p = saved['logistic|'+name]
        actual = metrics(p)
        require(all(abs(actual[k]-v) < 1e-10 for k, v in expected.items()), 'Logistic prediction metrics differ')
    comparisons = []
    for objective in ('mae', 'mse'):
        a, b = [np.array([saved[f'{arm}|{objective}|{seed}'] for seed in (17, 29, 43)]) for arm in ('C', 'A')]
        for metric in ('mae', 'rmse'):
            error_a, error_b = a.astype(float)-y, b.astype(float)-y
            errors = (abs(error_a), abs(error_b)) if metric == 'mae' else (error_a**2, error_b**2)
            comparisons.append({'objective': objective, 'metric': metric, 'contrast': 'C minus A', **compare_ci(*errors, metric)})
    shares = split['n_D'][validation]/(split['n_D'][validation]+split['n_M'][validation])
    plan = c.read(ctx['directory']/'plan.json.gz'); checked = []
    for role in ('train', 'validation'):
        ordered = sorted(ctx['ids'][role], key=lambda cid: (ctx['records'][ctx['index'][cid]['views']['DM']['object_key']]['statistics']['nodes'], cid))
        for cid in (ordered[0], ordered[len(ordered)//2]):
            item = ctx['index'][cid]['views']['DM']; key = item['object_key']
            row = data.extract((str(ctx['directory']), cid, item, [s['source'] for s in plan['entries'][key]['spans']],
                                ctx['receipts'][key], c.digest(ctx['contract']['encoder_identity']), 4))
            i = ids.index(cid)
            require(np.array_equal(np.array(row['source'], dtype=np.float32), np.stack([split['x_D'][i], split['x_M'][i]]))
                    and row['counts'] == [int(split['n_D'][i]), int(split['n_M'][i])], 'Pilot source extraction mismatch')
            checked.append(cid)
    findings = {'schema': 'external_fusion_control_review_v1', 'training': False, 'test_access': False,
        'checkpoint_replay': replay, 'logistic_metrics_from_saved_scores': report['logistic'],
        'logistic_fit_replay': False, 'logistic_fit_replay_reason': 'Logistic estimator/scaler states not supplied',
        'paired_error_comparisons': comparisons, 'bounded_source_extraction_checked': checked,
        'median_description_token_share': {'high_epss': float(np.median(shares[high])), 'lower_epss': float(np.median(shares[~high]))},
        'limitations': ['Pilot remains exploratory and batched; not official CT02 replay.',
            'Equal parameter count is not identical architecture: projection width and LayerNorm groups differ.',
            'C and E are not identical under MSE; supplied RMSE is approximately 0.0988 versus 0.1010.',
            'No universal claim that capacity or contextual embeddings never matter.',
            'No full retraining or independent historical/test evaluation.'], 'input_hashes': pins}
    out = data.RESULTS/'pilot_review_v1'
    with publication.transaction(out, verify=args.verify_only) as stage:
        data.write(stage/'findings.json', findings)
        (stage/'REPORT.txt').write_text('External controlled pilot verification; no new fits.\n'+publication.dump(findings))
        for name in names[:5]: (stage/('supplied_'+name)).write_bytes((args.probe_dir/name).read_bytes())
        publication.check_pins(pins)
    z.close(); split.close(); saved.close()
    print(f'External pilot '+('reproduced' if args.verify_only else 'verified')+f': {out}; no new training or test access.')


if __name__ == '__main__': main()
