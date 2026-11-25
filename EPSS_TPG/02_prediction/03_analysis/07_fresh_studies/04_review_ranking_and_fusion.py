"""Post-hoc CT02 ranking/contribution review; no model fitting or test access."""
import argparse
from collections import Counter
import io
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
for rel in ('01_tpg/01_core', '01_tpg/10_extensions', '02_prediction/01_package',
            '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT / rel))

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
from epss.combined_training_v1 import data, reporting
from epss.combined_training_v2 import pooled, runner
from epss.continuous_forecasting import data as publication

c, require = data.c, data.require
PROBE = Path('external_probes')


def ranking(ids, actual, prediction):
    actual, prediction = np.asarray(actual), np.asarray(prediction)
    require(actual.shape == prediction.shape == (len(ids),), 'Ranking length mismatch')
    require(np.isfinite(actual).all() and np.isfinite(prediction).all(), 'Nonfinite ranking input')
    positive = actual >= .1
    require(0 < positive.sum() < len(ids), 'Ranking needs both outcome groups')
    order = np.lexsort((np.asarray(ids), -prediction))
    row = {'auroc': float(roc_auc_score(positive, prediction)),
           'average_precision': float(average_precision_score(positive, prediction)),
           'mae': float(abs(prediction-actual).mean()),
           'mae_below_0_1': float(abs(prediction-actual)[~positive].mean()),
           'mae_at_least_0_1': float(abs(prediction-actual)[positive].mean())}
    for k in (10, 25, 50, 100):
        if k > len(ids):
            continue
        score = prediction[order[k-1]]
        strict = prediction > score
        ties = prediction == score
        slots = k-int(strict.sum())
        hits = int(positive[order[:k]].sum())
        row.update({f'top{k}_hits': hits, f'precision_at_{k}': hits/k,
                    f'recall_at_{k}': hits/int(positive.sum()),
                    f'top{k}_boundary_ties': int(ties.sum()),
                    f'top{k}_tie_min_hits': int(positive[strict].sum()) + max(0, slots-int((ties & ~positive).sum())),
                    f'top{k}_tie_max_hits': int(positive[strict].sum()) + min(slots, int((ties & positive).sum()))})
    return row


def contributions(ids, actual, predictions, baseline):
    errors = abs(predictions-actual)
    gains = abs(baseline-actual) - errors.mean(axis=0)
    order = np.lexsort((np.asarray(ids), -gains))
    total = float(gains.sum())
    rows = [{'cve': ids[i], 'actual': float(actual[i]), 'mean_prediction': float(predictions[:, i].mean()),
             'baseline_absolute_error': float(abs(baseline[i]-actual[i])),
             'mean_model_absolute_error': float(errors[:, i].mean()),
             'absolute_error_reduction': float(gains[i]),
             'share_of_net_gain_percent': float(100*gains[i]/total) if total != 0 else None}
            for i in order]
    kept = order[4:]
    return rows, {'net_mae_gain': total/len(ids),
        'top_one_share_percent': 100*float(gains[order[0]])/total if total != 0 else None,
        'top_four_share_percent': 100*float(gains[order[:4]].sum())/total if total != 0 else None,
        'excluding_top_four_cves': [ids[i] for i in order[:4]],
        'excluding_top_four_model_minus_baseline_mae': float(-gains[kept].mean()),
        'exclusion_is_post_hoc_sensitivity_not_a_new_cohort': True}


def load_results(ctx, run, pins):
    review = run.parent/'full_review_v1'
    publication.check_lock(review)
    prior = c.read(review/'findings.json')
    publication.check_pins(prior['input_hashes'])
    pins.update(prior['input_hashes'])
    pins[str(review/'artifact_lock.json')] = c.sha(review/'artifact_lock.json')
    identity = c.read(run/'report/run_identity.json')
    require(identity['provenance'] == ctx['provenance'] and not identity['test_access'], 'Cohort changed')
    result = c.read(run/'controls/evaluation.json')
    epochs = {}
    for rep, view, seed in runner.fit_list(identity['variants'], identity['settings']['seeds']):
        folder = runner.fit_folder(run, rep, view, seed)
        publication.check_lock(folder)
        name = f'{rep}/{view}/seed_{seed}'
        result[name] = c.read(folder/'evaluation.json')
        require([r['cve'] for r in result[name]['predictions']] == ctx['ids']['validation'], 'Unmatched rows')
        require(result[name] == reporting.evaluate(ctx, view, [r['raw_prediction'] for r in result[name]['predictions']]),
                'Saved metric/target mismatch')
    for row in prior['histories']:
        epochs[row['model']+'/seed_'+str(row['seed'])] = row
    return result, epochs, identity


def source_mapping(ctx, pins):
    plan = c.read(ctx['directory']/'plan.json.gz')
    selected = []
    # Outcome-blind bounded sample: shortest and middle-size graphs for each split/view.
    for split in ('train', 'validation'):
        for view in ('D', 'M', 'U', 'DM', 'DU', 'DMU'):
            keys = {ctx['index'][cid]['views'][view]['object_key'] for cid in ctx['ids'][split]} - {None}
            keys = sorted(keys, key=lambda k: (ctx['records'][k]['statistics']['nodes'], k))
            for k in (keys[0], keys[len(keys)//2]):
                if k not in selected:
                    selected.append(k)
    rows = []
    for key in tqdm(selected, desc='Check bounded node/document/source mapping', unit='graph'):
        entry = plan['entries'][key]
        obj = ctx['directory']/'objects'/key
        receipt = c.read(obj/'graph_receipt.json')
        for name, expected in receipt['files'].items():
            require(c.sha(obj/name) == expected, 'Changed graph payload')
            pins[str(obj/name)] = expected
        graph = c.read(obj/'graph.json.gz')
        require(graph['metadata']['combined_text']['spans'] == entry['spans'], 'Source spans changed')
        with zipfile.ZipFile(obj/'structure.npz') as z:
            arrays = {name: np.load(io.BytesIO(z.read(('' if name == 'pool_index' else 't_')+name+'.npy')), allow_pickle=False)
                      for name in ('node_ids', 'document_index', 'token_mask', 'pool_index')}
        require(arrays['node_ids'].tolist() == [n['id'] for n in graph['vertices']], 'Node row order changed')
        sources, token_sources = Counter(), Counter()
        for i, node in enumerate(graph['vertices']):
            prop = node['properties']; doc = int(arrays['document_index'][i])
            require(doc == prop['DOC_IDX'], 'Tensor/JSON document mapping mismatch')
            if doc == -1:
                require(node['label'] == 'DOCUMENT' and prop['EXTRA']['source_text_type'] == 'combined', 'Unassigned ordinary node')
                source = 'GLOBAL'
            else:
                require(0 <= doc < len(entry['spans']), 'Invalid document index')
                span = entry['spans'][doc]; source = span['source']
                require(prop['EXTRA']['document_id'] == span['document_id'] and
                        prop['EXTRA']['source_text_type'] == source, 'Source identity mismatch')
            require(bool(arrays['token_mask'][i]) == (node['label'] == 'TOKEN'), 'Wrong token-row mapping')
            sources[source] += 1
            if node['label'] == 'TOKEN':
                token_sources[source] += 1
        require(len(arrays['pool_index']) == len(graph['vertices']), 'Embedding row map length mismatch')
        rows.append({'cve': entry['cve'], 'split': entry['split'], 'key': key,
                     'nodes_by_source': dict(sources), 'tokens_by_source': dict(token_sources)})
    return {'sample_rule': 'Shortest and median node count per split/view; no outcome selection',
            'graphs_checked': len(rows), 'rows': rows, 'full_mapping_audit_completed': False,
            'tensor_field': 'document_index', 'json_source_field': 'vertices[].properties.EXTRA.source_text_type',
            'current_training_loader_preserves_document_index': False,
            'global_document_policy_required': True,
            'cross_source_messages_remain_with_source_readout': True,
            'combined_context_differs_from_separate_source_encoder_context': True}


def probe_review(ctx, probe, pins):
    paths = [probe/name for name in ('fusion_probe.py', 'fusion_probe.json', 'pooled_views.npz')]
    for path in paths:
        require(path.is_file(), 'Missing supplied probe evidence: '+str(path))
        pins[str(path)] = c.sha(path)
    values = c.read(probe/'fusion_probe.json')
    for arm in values.values():
        require(len(arm['seeds']) == 3, 'Unexpected probe seed count')
        for metric, value in arm['mean'].items():
            require(abs(value-float(np.mean([s[metric] for s in arm['seeds']]))) <= .0000051,
                    'Probe aggregate differs from its supplied seed records')
    ids = ctx['ids']['train'] + ctx['ids']['validation']
    vectors = pooled.load(ctx, pooled.location('previous', 'full', 4), 4)
    pooled_root = pooled.location('previous', 'full', 4)
    pins[str(pooled_root/'artifact_lock.json')] = c.sha(pooled_root/'artifact_lock.json')
    with np.load(probe/'pooled_views.npz', allow_pickle=False) as z:
        require(z['ids'].tolist() == ids and z['train'].tolist() == [True]*len(ctx['ids']['train'])+[False]*len(ctx['ids']['validation']),
                'Probe CVE identities/roles mismatch')
        require(np.array_equal(z['y'], np.array([ctx['labels'][cid]['value'] for cid in ids], dtype=np.float32)) and
                np.array_equal(z['struct'], np.array([ctx['vectors'][cid] for cid in ids], dtype=np.float32)),
                'Probe target/structured inputs mismatch')
        for view in tqdm(('D', 'M', 'U', 'DM', 'DMU'), desc='Verify probe vectors against frozen cache', unit='view'):
            for i, cid in enumerate(ids):
                item = ctx['index'][cid]['views'][view]; key = item['object_key']
                observed = bool(key and vectors[key]['pooled_observed'])
                expected = vectors[key]['pooled_vector'].numpy() if observed else np.zeros(768, dtype=np.float32)
                require(np.array_equal(z['x_'+view][i], expected) and bool(z['has_'+view][i]) == observed and
                        np.array_equal(z['mask_'+view][i], [item['source_mask'][s] for s in 'DMGN']),
                        'Probe vectors/source masks differ from cache')
    return {'input_arrays_verified': True, 'aggregate_arithmetic_verified': True,
            'checkpoint_prediction_replay_possible': False,
            'reason': 'No saved fusion checkpoints or per-CVE predictions in supplied artifacts; logistic results are printed but not saved.',
            'new_probe_training_run': False, 'neural_fits_in_probe_code': 30, 'logistic_fits_in_probe_code': 6,
            'parameter_counts_from_architecture': {'one_branch': 28705, 'two_branches': 54401, 'three_branches': 80097},
            'confounds': ['Separate-source embedding context versus combined 1024-subword context',
                         'Source-specific projections and wider concatenated head increase parameter count',
                         'Probe initializes text projections before head; official runner initializes head first',
                         'Batched computation rather than official per-CVE gradient accumulation',
                         'Thirty additional neural and six logistic fits inspected on the same validation cohort',
                         'Logistic fits predict EPSS >= 0.1, not EPSS regression or exploitation/KEV'],
            'supplied_metrics_not_independently_replayed': values}


def render(document):
    rows = document['ranking_seed_means']; concentration = document['concentration']
    cols = ['model', 'mae', 'auroc', 'average_precision', 'top50_hits']
    lines = ['CT02: POST-HOC RANKING AND SOURCE-FUSION REVIEW', '',
        '1. Scope', 'Saved validation predictions only. No fitting, inference, threshold optimization or test scoring.',
        'High EPSS means saved Day-60 score >= 0.1, not exploitation or VulnCheck inclusion.',
        'This threshold and all new tables are post-hoc diagnostics, not frozen confirmatory endpoints.', '',
        '2. Ranking', f"There are {document['positive_cves']} high-score CVEs among {document['validation_cves']}; random AP is {document['positive_prevalence']:.6f}.",
        'AUROC measures pairwise high/low-score ordering; AP summarizes precision and recall of that ranking.',
        'Top-50 hits counts high-score CVEs among the 50 largest predictions. Values average seed metrics, not predictions.',
        'Ties are resolved by CVE ID; per-seed tables include attainable tie bounds. A constant ranking has no meaningful top-k ordering.',
        ' | '.join(cols)]
    lines += [' | '.join(str(row[k]) if k == 'model' else f'{row[k]:.6f}' for k in cols) for row in rows]
    lines += ['', '3. Concentration of D-pooled improvement',
        f"Overall MAE gain versus training median: {concentration['net_mae_gain']:.9f}.",
        f"Largest contributor supplies {concentration['top_one_share_percent']:.2f}% of net gain; top four supply {concentration['top_four_share_percent']:.2f}%.",
        f"Without the four post-hoc selected contributors, model-minus-baseline MAE is {concentration['excluding_top_four_model_minus_baseline_mae']:+.9f}.",
        'More than 100% means other CVEs offset the gain. Do not remove these CVEs from the official cohort.',
        'This is a sensitivity explanation, not evidence that the contributing records are incorrect.', '',
        '4. What the source-fusion probe establishes and does not establish',
        'Its cached input vectors, masks, roles and targets match the official development inputs.',
        'Its JSON mean values match its supplied seed records. Saved fusion predictions/checkpoints are absent, so those outcomes were not independently replayed.',
        'The logistic AUROC/top-k claim cannot be reproduced from the supplied JSON: that code only prints logistic metrics.',
        'Per-source probe inputs use standalone D/M/U embeddings. Combined embeddings use a different text/window context.',
        'One/two/three-source probe heads have 28,705 / 54,401 / 80,097 trainable parameters.',
        'Initialization order and batching also differ from the official implementation.',
        'The result motivates a controlled source-readout experiment; it does not prove merging caused all of the losses.', '',
        '5. Cache feasibility',
        f"Checked {document['mapping']['graphs_checked']} outcome-blind sampled graphs against JSON nodes and compact tensor row identities.",
        'document_index maps tensor rows to the plan spans; spans and node metadata identify D/M/G/N.',
        'The current training loader omits document_index. A new adapter must retain it.',
        'The global DOCUMENT node has index -1 and covers all sources; it needs an explicit GLOBAL or exclusion policy.',
        'Source-specific readout can reuse existing graphs/embeddings, but a full mapping audit is still required.',
        'Cross-source edges and combined-context embeddings remain: source readout does not isolate D from other sources.', '',
        '6. Decision',
        'Retain the completed MAE study. Treat high-score ranking as exploratory evidence, not exploitation prediction.',
        'An M-model also uses structured features and masks: its ranking is not attributable solely to social-media text.',
        'Early best epochs do not prove the model is untrained; epoch 2 already follows hundreds of optimizer updates.',
        'Before expensive MSE TPG fits: freeze a bounded pooled merged-versus-source-readout comparison, with MAE/MSE kept separate.',
        'Use the same combined-cache embeddings first; share projection parameters or add a parameter-matched control, and declare the global-node policy.',
        'Primary regression objective: raw-score MSE with validation RMSE selection; report MAE, ranking and tail errors as declared secondary metrics.',
        'Optimize graph loading in a separate parity-qualified version before new TPG training.',
        'For Reddit, predeclare the method and comparisons before further development; verify the held-out test lock. A different dataset alone does not prove general superiority.',
        'Reddit has no accepted validation webpage summaries, so its supported text arms remain D, M and DM.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--probe-dir', type=Path, default=PROBE)
    args = parser.parse_args()
    ctx = data.context('previous', 'full')
    run = runner.destination('previous', 'mae_fast_v1', 'full')
    pins = {str(Path(__file__).resolve()): c.sha(Path(__file__))}
    results, epochs, identity = load_results(ctx, run, pins)
    ids = ctx['ids']['validation']; y = np.array([ctx['labels'][cid]['value'] for cid in ids])
    seeds = identity['settings']['seeds']
    predictions = {name: np.array([r['prediction'] for r in result['predictions']]) for name, result in results.items()}
    rows = []
    for name, pred in tqdm(sorted(predictions.items()), desc='Compute saved-prediction ranking metrics', unit='fit'):
        rows.append({'model': name, **ranking(ids, y, pred), 'best_epoch': epochs.get(name, {}).get('best_epoch'),
                     'training_mae_at_best': epochs.get(name, {}).get('training_mae_at_best')})
    means = []
    for model in sorted({r['model'].split('/seed_')[0] for r in rows}):
        group = [r for r in rows if r['model'].split('/seed_')[0] == model]
        means.append({'model': model, **{k: float(np.mean([r[k] for r in group])) for k in group[0]
                    if k not in ('model', 'best_epoch', 'training_mae_at_best')},
                      'seeds': len(group) if '/seed_' in group[0]['model'] else 0})
    d = np.array([predictions[f'pooled/D/seed_{seed}'] for seed in seeds])
    contribution_rows, concentration = contributions(ids, y, d, predictions['training_median'])
    selected = {r['cve'] for r in contribution_rows[:4]}
    examples = [{'cve': cid, 'actual': float(y[i]),
                 **{name: float(np.mean([predictions[f'{name}/seed_{s}'][i] for s in seeds]))
                    for name in sorted({r['model'].split('/seed_')[0] for r in rows if '/seed_' in r['model']})}}
                for i, cid in enumerate(ids) if cid in selected]
    mapping = source_mapping(ctx, pins)
    probe = probe_review(ctx, args.probe_dir, pins)
    document = {'schema': 'ct02_posthoc_ranking_fusion_v1', 'training': False, 'test_access': False,
        'threshold_selected_posthoc': .1, 'validation_cves': len(ids), 'positive_cves': int((y >= .1).sum()),
        'positive_prevalence': float((y >= .1).mean()), 'ranking_seed_means': means,
        'concentration': concentration, 'mapping': mapping, 'probe': probe, 'input_hashes': pins}
    output = run.parent/'ranking_fusion_review_v1'
    with publication.transaction(output, verify=args.verify_only) as stage:
        data.write(stage/'findings.json', document)
        (stage/'REPORT.txt').write_text(render(document))
        for name, values in [('ranking_per_seed', rows), ('ranking_seed_means', means),
                             ('d_gain_contributions', contribution_rows), ('top_contributor_predictions', examples)]:
            (stage/(name+'.csv')).write_text(reporting.csv_text(values))
        for name in ('fusion_probe.py', 'fusion_probe.json'):
            (stage/('supplied_'+name)).write_bytes((args.probe_dir/name).read_bytes())
        publication.check_pins(pins)
    print(f"CT02 ranking/fusion review {'reproduced' if args.verify_only else 'saved'}: {output}. No fitting, inference or test access.")


if __name__ == '__main__':
    main()
