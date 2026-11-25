"""F01-Tree: one fixed shallow tree on frozen EPSS statistics; no test scoring."""
import argparse
import copy
import csv
import math
from pathlib import Path
import statistics
from collections import defaultdict

import numpy as np
from sklearn.tree import DecisionTreeRegressor

if __package__:
    from . import forecast_huber as reference
    from .forecast_data import (STAT_FEATURES, fit_training_preprocessor, read_json,
        require, score, sha256, transform_statistics, write_json)
    from .forecast_baseline import regression_metrics
else:
    import forecast_huber as reference
    from forecast_data import (STAT_FEATURES, fit_training_preprocessor, read_json,
        require, score, sha256, transform_statistics, write_json)
    from forecast_baseline import regression_metrics

VERSION = 'epss_fixed_tree_v1'
CONFIG = {'criterion': 'squared_error', 'splitter': 'best', 'max_depth': 3,
          'min_samples_split': 2, 'min_samples_leaf': 20, 'max_features': None,
          'random_state': 17, 'ccp_alpha': 0.0}
OUTPUT_FILES = {'model.json', 'predictions.json', 'predictions.csv', 'summary.json',
    'summary.txt', 'README.md', 'tree_nodes.csv', 'validation_comparison.png',
    'run_status.json', 'evaluation_lock.json'}


def fit_model(data):
    ids = reference.ridge.cohort(data, 'train')
    scaler = data['training_preprocessor.json']
    require(scaler['fit_cve_ids'] == ids and scaler == fit_training_preprocessor(
        data['inputs.json'], data['labels.json'], data['split_manifest.json'],
        policy=data['split_policy.json']), 'Scaler is not training-only')
    x = np.asarray([transform_statistics(data['inputs.json'][c], scaler) for c in ids], dtype=np.float32)
    y = [score(data['labels.json'][c]['y']['endpoint']) for c in ids]
    require(all(v is not None for v in y), 'Missing training target')
    estimator = DecisionTreeRegressor(**CONFIG).fit(x, y)
    tree = estimator.tree_
    nodes = [{'node_id': i, 'left': int(tree.children_left[i]), 'right': int(tree.children_right[i]),
        'feature': int(tree.feature[i]), 'threshold': float(tree.threshold[i]),
        'value': float(tree.value[i, 0, 0]), 'training_cves': int(tree.n_node_samples[i])}
        for i in range(tree.node_count)]
    model = {'schema': VERSION, 'task': 'endpoint', 'fit_split': 'train', 'config': dict(CONFIG),
        'fit_cve_ids': list(ids), 'feature_names': list(STAT_FEATURES),
        'preprocessor': copy.deepcopy(scaler), 'nodes': nodes, 'input_dtype': 'float32',
        'output_rule': 'clip_to_0_1', 'selection_policy': 'fixed_config_no_search_no_validation_selection',
        'training_weight_policy': 'equal_weight_no_removed_cves'}
    validate_model(model)
    portable = [predict(model, data['inputs.json'][c])['tree_raw_prediction'] for c in ids]
    require(np.array_equal(portable, estimator.predict(x)), 'Exported tree disagrees with sklearn')
    return model


def validate_model(model):
    require(model['schema'] == VERSION and model['task'] == 'endpoint' and model['fit_split'] == 'train', 'Wrong tree model')
    require(model['config'] == CONFIG and model['feature_names'] == list(STAT_FEATURES), 'Tree design differs')
    require(model['input_dtype'] == 'float32' and model['output_rule'] == 'clip_to_0_1', 'Wrong inference rule')
    ids = model['fit_cve_ids']
    require(ids and ids == sorted(set(ids)) and ids == model['preprocessor']['fit_cve_ids'], 'Wrong training cohort')
    require(model['preprocessor']['task'] == 'endpoint', 'Wrong scaler task')
    nodes = model['nodes']
    require(1 <= len(nodes) <= 2**(CONFIG['max_depth']+1)-1, 'Invalid tree size')
    seen = set()
    pending = [(0, 0)]
    while pending:
        i, depth = pending.pop()
        require(type(i) is int and 0 <= i < len(nodes) and i not in seen, 'Cycle/shared/invalid child')
        seen.add(i)
        n = nodes[i]
        require(n['node_id'] == i and depth <= CONFIG['max_depth'], 'Invalid node/depth')
        require(type(n['training_cves']) is int and n['training_cves'] > 0, 'Invalid node sample count')
        require(all(type(n[k]) in (int, float) and math.isfinite(n[k]) for k in ('threshold', 'value'))
                and 0 <= n['value'] <= 1, 'Invalid node values')
        leaf = n['left'] == n['right'] == -1
        if leaf:
            require(n['feature'] == -2 and (i == 0 or n['training_cves'] >= CONFIG['min_samples_leaf']), 'Invalid leaf')
        else:
            require(type(n['feature']) is int and 0 <= n['feature'] < len(STAT_FEATURES), 'Invalid split feature')
            for child in (n['left'], n['right']):
                require(type(child) is int and 0 <= child < len(nodes), 'Invalid child')
            require(nodes[n['left']]['training_cves']+nodes[n['right']]['training_cves'] == n['training_cves'], 'Sample counts disagree')
            pending.extend([(n['left'], depth+1), (n['right'], depth+1)])
    require(len(seen) == len(nodes) and nodes[0]['training_cves'] == len(ids), 'Unreachable nodes or wrong root count')


def predict(model, record):
    validate_model(model)
    # sklearn converts predictors to float32 before comparing to float64 thresholds.
    x = np.asarray(transform_statistics(record, model['preprocessor']), dtype=np.float32)
    require(np.isfinite(x).all(), 'Non-finite tree input')
    node = model['nodes'][0]
    while node['left'] != -1:
        node = model['nodes'][node['left'] if float(x[node['feature']]) <= node['threshold'] else node['right']]
    raw = node['value']
    return {'tree_raw_prediction': raw, 'tree_prediction': min(1.0, max(0.0, raw)), 'tree_leaf_id': node['node_id']}


def evaluate(data, model, references, split='validation'):
    require(split == 'validation', 'Test scoring is disabled')
    require(model['fit_cve_ids'] == reference.ridge.cohort(data, 'train') and
            model['preprocessor'] == data['training_preprocessor.json'], 'Different training cohort/scaler')
    ids = reference.ridge.cohort(data, 'validation')
    require([r['cve'] for r in references] == ids, 'Comparator cohort differs')
    rows = []
    for r in references:
        c = r['cve']
        require(r['split'] == 'validation' and r['actual_day60_epss'] == data['labels.json'][c]['y']['endpoint']
            and r['target_date'] == data['labels.json'][c]['target_date']
            and r['prediction_anchor_date'] == data['inputs.json'][c]['input_end_date']
            and r['group_id'] == data['split_manifest.json'][c]['group_id'], 'Comparator alignment differs')
        row = {**r, **predict(model, data['inputs.json'][c])}
        error = row['tree_prediction']-row['actual_day60_epss']
        row.update(tree_signed_error=error, tree_absolute_error=abs(error),
            absolute_error_difference_tree_minus_persistence=abs(error)-row['persistence_absolute_error'])
        rows.append(row)
    metrics, group_mae = {}, {}
    for name in ('persistence', 'ridge', 'huber', 'tree'):
        metrics[name] = regression_metrics([r['actual_day60_epss'] for r in rows], [r[name+'_prediction'] for r in rows])
        groups = defaultdict(list)
        for r in rows:
            groups[r['group_id']].append(abs(r[name+'_prediction']-r['actual_day60_epss']))
        group_mae[name] = statistics.mean(statistics.mean(v) for v in groups.values())
    p, t = metrics['persistence']['mae'], metrics['tree']['mae']
    return rows, {'schema': VERSION, 'experiment': 'F01-Tree', 'config': dict(CONFIG),
        'training_cves': len(model['fit_cve_ids']), 'validation_cves': len(rows),
        'metrics': metrics, 'equal_group_mae': group_mae, 'node_count': len(model['nodes']),
        'leaf_count': sum(n['left'] == -1 for n in model['nodes']),
        'mae_difference_tree_minus_persistence': t-p, 'mae_skill_vs_persistence': 1-t/p if p else None,
        'test_performance_evaluated': False, 'hyperparameter_search_performed': False,
        'primary_cohort_changed': False, 'split_policy': data['split_policy.json']}


def preflight(prepared, freeze_dir, ridge_dir, huber_dir):
    data, ridge_rows, provenance = reference.preflight(prepared, freeze_dir, ridge_dir)
    lock = reference.checked_files(huber_dir, 'evaluation_lock.json', reference.VERSION,
                                  reference.OUTPUT_FILES-{'evaluation_lock.json'})
    require(not lock['test_performance_evaluated'] and lock['preparation_lock_sha256'] == provenance['preparation_lock_sha256'], 'Wrong Huber run')
    status, model = read_json(huber_dir/'run_status.json'), read_json(huber_dir/'model.json')
    require(status['status'] == 'complete' and status['provenance'] == model['provenance'], 'Incomplete Huber run')
    require(model['provenance']['source_sha256'] == provenance['source_sha256'], 'Huber source differs')
    for p, digest in model['provenance']['pinned_files'].items():
        require(sha256(p) == digest, f'Changed Huber input/code: {p}')
    rows, _ = reference.evaluate(data, model, ridge_rows)
    require(rows == read_json(huber_dir/'predictions.json'), 'Huber predictions do not reproduce')
    provenance.update(schema=VERSION, config=dict(CONFIG), huber_directory=str(huber_dir))
    provenance['pinned_files'].update({str(p): sha256(p) for p in (Path(__file__).resolve(), huber_dir/'evaluation_lock.json')})
    return data, rows, provenance


def report(summary):
    def fmt(value):
        return 'Not defined' if value is None else f'{value:.8f}'
    lines = ['# F01-Tree: Fixed Shallow EPSS Baseline', '',
        'Predict Day-60 EPSS using seven statistics of observed Days 0-30 scores.',
        'Day 0 is NVD publication; the prediction anchor is Day 30. This is not',
        'exploitation prediction. No text, TPG, SMP, CVSS or KEV inputs are used.', '',
        f"Training CVEs: {summary['training_cves']}; validation CVEs: {summary['validation_cves']}.",
        f'Fixed settings: {CONFIG}.',
        'At most three decisions along a path and eight leaves; each non-root leaf',
        'contains at least 20 training CVEs. Each leaf predicts its training-target mean.',
        'Splits minimize squared error. The seed resolves randomness, not a seed search.',
        'The seven features are latest score, mean, population standard deviation,',
        'slope, observed-day count, latest-score age, and whether slope is available.',
        'The existing training-only scaler and missing-value policy are unchanged.',
        'No new imputation, future inputs, target filling or cohort filtering.', '',
        '## Matched Validation Results', '',
        '| Model | Average absolute error (MAE) | Large-error-sensitive RMSE | Rank correlation | Equal-group MAE |',
        '|---|---:|---:|---:|---:|']
    for name, m in summary['metrics'].items():
        lines.append(f"| {name} | {fmt(m['mae'])} | {fmt(m['rmse'])} | {fmt(m['spearman_rho'])} | {fmt(summary['equal_group_mae'][name])} |")
    lines += ['', 'Every row uses the same validation CVEs and Day-60 target. Lower MAE/RMSE',
        'is better. EPSS and errors use 0-1 units; 0.01 is one percentage point.',
        'Rank correlation describes ordering, not accuracy. Equal-group MAE first',
        'averages errors in each related-CVE group, then weights groups equally.',
        f"MAE difference (tree minus persistence): {fmt(summary['mae_difference_tree_minus_persistence'])}.",
        f"Tree nodes: {summary['node_count']}; leaves: {summary['leaf_count']}.", '',
        '## Limits and Interpretation', '',
        'This is a small nonlinear comparator, not an isolated loss-only ablation.',
        'Settings were declared after earlier validation results were inspected.',
        'There is no parameter search, validation-based stopping or test scoring.',
        'This is development evidence, not independent confirmation or a promised gain.',
        'Leaf averages cannot extrapolate beyond training target values. A shallow tree',
        'can miss rare jumps or underfit; no settings are automatically changed afterward.',
        'Do not interpret split rules as causal or use them to justify dropping CVEs.', '',
        '## Artifacts and Figure', '',
        'model.json stores the complete tree and training-only scaler; no pickle is needed.',
        'tree_nodes.csv lists each node, children, feature index, threshold, leaf value and',
        'training count. Feature indices follow model.feature_names; -2 means leaf,',
        'child -1 means no child. Thresholds use standardized features and float32 inputs.',
        'predictions.csv/json includes each validation CVE and the reached leaf.',
        'The figure x-axis is actual Day-60 EPSS; y-axis is predicted EPSS.',
        'The dashed diagonal is exact agreement. Each panel uses the same validation CVEs.',
        'summary.txt/json and README.md explain the run; evaluation_lock.json hashes it.',
        'Keep persistence as reference and review results before any further experiment.',
        'Estimator reference: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html']
    return '\n'.join(lines)+'\n'


def plot(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    for ax, name in zip(axes, ('persistence', 'ridge', 'huber', 'tree')):
        ax.scatter([r['actual_day60_epss'] for r in rows], [r[name+'_prediction'] for r in rows], s=15, alpha=.6)
        ax.plot([0, 1], [0, 1], '--', color='#555555')
        ax.set(title=name.title(), xlabel='Actual Day-60 EPSS', ylabel='Predicted Day-60 EPSS', xlim=(0, 1), ylim=(0, 1))
    fig.suptitle(f'F01-Tree: same validation CVEs in every panel (n={len(rows)})')
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(prepared, freeze_dir, ridge_dir, huber_dir, output, check_only=False):
    prepared, freeze_dir, ridge_dir, huber_dir, output = map(lambda p: Path(p).resolve(),
        (prepared, freeze_dir, ridge_dir, huber_dir, output))
    data, references, provenance = preflight(prepared, freeze_dir, ridge_dir, huber_dir)
    source = Path(data['preparation_summary.json']['source_path']).resolve()
    for protected in (prepared, freeze_dir, ridge_dir, huber_dir, source.parent):
        require(output != protected and protected not in output.parents and output not in protected.parents, 'Output overlaps protected inputs')
    require(not output.exists(), 'Output exists; preserve it and choose a new --output-dir')
    if check_only:
        print(f"Preflight passed: {len(reference.ridge.cohort(data, 'train'))} training / {len(references)} validation CVEs; config={CONFIG}.")
        print('No fitting, writes or test scoring. Saved comparators reproduced without refitting.')
        return
    output.mkdir(parents=True, exist_ok=False)
    try:
        write_json(output/'run_status.json', {'status': 'running', 'provenance': provenance})
        print('Fitting one fixed shallow tree on training CVEs only...', flush=True)
        model = fit_model(data)
        rows, summary = evaluate(data, model, references)
        model['provenance'] = copy.deepcopy(provenance)
        summary['provenance'] = provenance
        for p, digest in provenance['pinned_files'].items():
            require(sha256(p) == digest, f'Changed input/code during run: {p}')
        for name, value in [('model', model), ('predictions', rows), ('summary', summary)]:
            write_json(output/f'{name}.json', value)
        for name, values in [('predictions', rows), ('tree_nodes', model['nodes'])]:
            with (output/f'{name}.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
        for name in ('summary.txt', 'README.md'):
            (output/name).write_text(report(summary), encoding='utf-8')
        plot(rows, output/'validation_comparison.png')
        write_json(output/'run_status.json', {'status': 'complete', 'provenance': provenance})
        write_json(output/'evaluation_lock.json', {'schema': VERSION,
            'files': {n: sha256(output/n) for n in sorted(OUTPUT_FILES-{'evaluation_lock.json'})},
            'preparation_lock_sha256': provenance['preparation_lock_sha256'], 'test_performance_evaluated': False})
        print(f'F01-Tree saved to {output}. Validation only; no test scoring.')
        return summary
    except Exception as exc:
        write_json(output/'run_status.json', {'status': 'failed', 'error_type': type(exc).__name__, 'message': str(exc), 'provenance': provenance})
        (output/'failure.txt').write_text(f'{type(exc).__name__}: {exc}\nNo successful result certified. Preserve this run.\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, default=reference.PREPARED)
    parser.add_argument('--freeze-dir', type=Path, default=reference.ROOT/'missingness_sensitivity_v1')
    parser.add_argument('--ridge-dir', type=Path, default=reference.ROOT/'ridge_validation')
    parser.add_argument('--huber-dir', type=Path, default=reference.ROOT/'huber_validation')
    parser.add_argument('--output-dir', type=Path, default=reference.ROOT/'tree_validation')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    run(args.prepared_dir, args.freeze_dir, args.ridge_dir, args.huber_dir, args.output_dir, args.check_only)


if __name__ == '__main__':
    main()
