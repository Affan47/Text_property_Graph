"""Verify the saved shallow tree, leaf means and validation metrics; never fit."""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import statistics
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss import forecast_tree as tree
from epss.forecast_data import read_json, require, sha256, transform_statistics, write_json
from review_forecast_evidence import check_csv
from review_forecast_missingness import independent_metrics


def verify_nodes(model, data):
    tree.validate_model(model)
    targets = defaultdict(list)
    for c in model['fit_cve_ids']:
        x = np.asarray(transform_statistics(data['inputs.json'][c], model['preprocessor']), dtype=np.float32)
        i = 0
        while True:
            node = model['nodes'][i]
            targets[i].append(data['labels.json'][c]['y']['endpoint'])
            if node['left'] == -1:
                break
            i = node['left'] if float(x[node['feature']]) <= node['threshold'] else node['right']
    for node in model['nodes']:
        values = targets[node['node_id']]
        require(len(values) == node['training_cves'], 'Training node count differs')
        require(math.isclose(statistics.mean(values), node['value'], rel_tol=1e-12, abs_tol=1e-14), 'Node mean differs from training targets')
    return targets


def describe(data, model, rows, targets):
    def measured(chosen, name):
        return independent_metrics([r['actual_day60_epss'] for r in chosen],
            [r[name+'_prediction'] for r in chosen], [r['group_id'] for r in chosen])
    leaves = []
    for node in model['nodes']:
        if node['left'] != -1:
            continue
        selected = [r for r in rows if r['tree_leaf_id'] == node['node_id']]
        values = targets[node['node_id']]
        leaves.append({'leaf_id': node['node_id'], 'training_cves': len(values),
            'training_target_min': min(values), 'training_target_max': max(values),
            'training_target_median': statistics.median(values), 'forecast': node['value'],
            'validation_cves': len(selected), 'metrics': {m: measured(selected, m) for m in ('persistence', 'tree')}})
    subsets = []
    for name, low, high in [('All 31 scores present', 31, 31), ('One score missing', 30, 30),
                          ('Two or more scores missing', 1, 29)]:
        chosen = [r for r in rows if low <= sum(v is not None for v in data['inputs.json'][r['cve']]['X']['values']) <= high]
        subsets.append({'input_group': name, 'cves': len(chosen),
            'metrics': {m: measured(chosen, m) for m in ('persistence', 'tree')}})
    rules = []
    for n in model['nodes']:
        if n['left'] == -1:
            continue
        f = n['feature']
        rules.append({'node_id': n['node_id'], 'feature': model['feature_names'][f],
            'standardized_threshold': n['threshold'],
            'approximate_original_units_threshold': n['threshold']*model['preprocessor']['scales'][f]+model['preprocessor']['means'][f],
            'left_if_less_or_equal': n['left'], 'right_if_greater': n['right']})
    changes = [r['tree_absolute_error']-r['persistence_absolute_error'] for r in rows]
    unchanged = [r for r in rows if r['actual_day60_epss'] == r['persistence_prediction']]
    keys = ('cve', 'actual_day60_epss', 'persistence_prediction', 'tree_prediction',
            'persistence_absolute_error', 'tree_absolute_error', 'tree_leaf_id')
    return {'leaves': leaves, 'split_rules': rules, 'input_coverage_subsets': subsets,
        'tree_lower_error_cves': sum(v < 0 for v in changes),
        'tree_higher_error_cves': sum(v > 0 for v in changes), 'equal_error_cves': sum(v == 0 for v in changes),
        'unchanged_endpoint_cves': len(unchanged), 'unchanged_endpoint_tree_metrics': measured(unchanged, 'tree'),
        'largest_persistence_errors': [{k: r[k] for k in keys} for r in sorted(rows, key=lambda r: (-r['persistence_absolute_error'], r['cve']))[:3]],
        'largest_tree_errors': [{k: r[k] for k in keys} for r in sorted(rows, key=lambda r: (-r['tree_absolute_error'], r['cve']))[:5]]}


def review(directory):
    lock = tree.reference.checked_files(directory, 'evaluation_lock.json', tree.VERSION,
                                         tree.OUTPUT_FILES-{'evaluation_lock.json'})
    require(set(lock['files']) == tree.OUTPUT_FILES-{'evaluation_lock.json'} and
            lock['test_performance_evaluated'] is False, 'Wrong artifact set/scope')
    model, status = read_json(directory/'model.json'), read_json(directory/'run_status.json')
    provenance = model['provenance']
    require(status['status'] == 'complete' and status['provenance'] == provenance, 'Incomplete run/provenance mismatch')
    for path, digest in provenance['pinned_files'].items():
        require(sha256(path) == digest, f'Changed input/code: {path}')
    data, references, current = tree.preflight(*[Path(provenance[k]) for k in
        ('prepared_directory', 'freeze_directory', 'ridge_directory', 'huber_directory')])
    require(current == provenance and lock['preparation_lock_sha256'] == provenance['preparation_lock_sha256'], 'Preflight provenance differs')
    rows, summary = tree.evaluate(data, model, references)
    require(rows == read_json(directory/'predictions.json'), 'Saved predictions do not reproduce')
    check_csv(directory/'predictions.csv', rows)
    check_csv(directory/'tree_nodes.csv', model['nodes'])
    targets = verify_nodes(model, data)
    summary['provenance'] = provenance
    require(summary == read_json(directory/'summary.json'), 'Summary does not reproduce')
    for name in ('summary.txt', 'README.md'):
        require((directory/name).read_text() == tree.report(summary), 'Report text differs')
    for name in ('persistence', 'ridge', 'huber', 'tree'):
        metrics = independent_metrics([r['actual_day60_epss'] for r in rows],
            [r[name+'_prediction'] for r in rows], [r['group_id'] for r in rows])
        for key in ('mae', 'rmse', 'equal_group_mae'):
            saved = summary['equal_group_mae'][name] if key == 'equal_group_mae' else summary['metrics'][name][key]
            require(math.isclose(saved, metrics[key], rel_tol=1e-11, abs_tol=1e-13), 'Independent metric differs')
    result = {'schema': 'epss_tree_post_run_review_v1', 'checks_passed': True,
        'run_directory': str(directory), 'run_lock_sha256': sha256(directory/'evaluation_lock.json'),
        'review_script_sha256': sha256(__file__), 'training_cves': len(model['fit_cve_ids']),
        'validation_cves': len(rows), 'metrics': summary['metrics'], 'equal_group_mae': summary['equal_group_mae'],
        'mae_skill_vs_persistence': summary['mae_skill_vs_persistence'], **describe(data, model, rows, targets),
        'training_performed': False, 'test_scoring_performed': False, 'raw_or_completed_run_modified': False,
        'verification_scope': 'Frozen source/code/output hashes, cohort/scaler alignment, saved-tree inference, all training node counts and means, validation predictions, CSV/JSON/text, independent decimal-based MAE/RMSE/group MAE. No refitting, split optimization or test scoring. Figure hash checked; not pixel reproduction.'}
    require(lock == tree.reference.checked_files(directory, 'evaluation_lock.json', tree.VERSION,
        tree.OUTPUT_FILES-{'evaluation_lock.json'}), 'Run changed during review')
    return result


def explanation(r):
    def fmt(v):
        return 'Not available' if v is None else f'{v:.8f}'
    lines = ['# F01-Tree: Verified Validation Findings', '',
        'The task is to predict Day-60 EPSS using scores available through Day 30.',
        'Day 0 means NVD publication. This is not exploitation prediction.',
        f"Training CVEs: {r['training_cves']}; validation CVEs: {r['validation_cves']}. No test scoring.", '',
        '## Matched Results', '',
        '| Model | Average absolute error (MAE) | Large-error-sensitive RMSE | Rank correlation | Equal-group MAE |',
        '|---|---:|---:|---:|---:|']
    for name, m in r['metrics'].items():
        lines.append(f"| {name} | {fmt(m['mae'])} | {fmt(m['rmse'])} | {fmt(m['spearman_rho'])} | {fmt(r['equal_group_mae'][name])} |")
    lines += ['', 'Every row uses the same validation CVEs and targets. Lower MAE/RMSE is better.',
        'EPSS and errors use 0-1 units; 0.01 is one percentage point. Equal-group MAE',
        'weights related-CVE groups equally; rank correlation measures ordering, not accuracy.',
        'Relative MAE increase versus persistence (%): '+fmt(None if r['mae_skill_vs_persistence'] is None else -100*r['mae_skill_vs_persistence'])+'.',
        f"Tree errors: {r['tree_lower_error_cves']} smaller, {r['tree_higher_error_cves']} larger, {r['equal_error_cves']} equal to persistence.",
        'These are descriptive validation comparisons, not significance tests.', '',
        '## What the Tree Actually Predicts', '',
        'Each leaf is a group formed by the learned decisions. Every CVE reaching it',
        'receives the same forecast: the mean Day-60 target of its training CVEs.',
        '| Leaf ID | Training CVEs | Median training target | Forecast for every CVE in this leaf | Validation CVEs | Persistence MAE | Tree MAE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for leaf in r['leaves']:
        lines.append(f"| {leaf['leaf_id']} | {leaf['training_cves']} | {fmt(leaf['training_target_median'])} | {fmt(leaf['forecast'])} | {leaf['validation_cves']} | {fmt(leaf['metrics']['persistence']['mae'])} | {fmt(leaf['metrics']['tree']['mae'])} |")
    lines += ['', 'The saved training counts and mean targets were recomputed for every node.',
        'Means can be above typical small scores when a group also contains large targets.',
        'Replacing individual latest scores with only a few group averages can therefore',
        'lose useful distinctions. This describes the fitted model, not a causal explanation',
        'or proof that every tree model will fail. Split rules appear in analysis.json;',
        'original-unit thresholds are approximate because inference uses float32 scaled inputs.',
        f"For {r['unchanged_endpoint_cves']} validation CVEs the target exactly equals the latest input score.",
        'Persistence has zero error on these CVEs; tree MAE on the same diagnostic subset is '+fmt(r['unchanged_endpoint_tree_metrics']['mae'])+'.', '',
        '## The Previously Identified Large Changes', '',
        '| CVE | Latest input score | Actual Day-60 score | Tree forecast | Tree absolute error |',
        '|---|---:|---:|---:|---:|']
    for row in r['largest_persistence_errors']:
        lines.append(f"| {row['cve']} | {row['persistence_prediction']:.5f} | {row['actual_day60_epss']:.5f} | {row['tree_prediction']:.8f} | {row['tree_absolute_error']:.8f} |")
    lines += ['', 'Examples are chosen by the largest persistence errors, not to filter the cohort.',
        'The review JSON also contains the five largest tree errors.', '',
        '## Same Missing-Data Groups', '',
        '| Scores available in Days 0-30 | Validation CVEs | Persistence MAE | Tree MAE |', '|---|---:|---:|---:|']
    for row in r['input_coverage_subsets']:
        lines.append(f"| {row['input_group']} | {row['cves']} | {fmt(row['metrics']['persistence']['mae'])} | {fmt(row['metrics']['tree']['mae'])} |")
    lines += ['', 'These are the same groups used in DQ04; all remain in the main experiment.',
        'Different groups contain different CVEs. This is not evidence that missing scores',
        'caused the error difference, and no group should be removed to improve results.', '',
        '## Decision and Limits', '',
        'Keep this fixed tree as a completed comparison, not a replacement for persistence.',
        'Do not silently deepen it, change leaf size, drop difficult CVEs, or open the test set.',
        'The small Huber difference is not a meaningful forecasting breakthrough either.',
        'Further model selection on this validation set is development, not independent evidence.',
        'The current comparisons provide no result about TPG or SMP because those inputs were absent.',
        'Before another endpoint model, define a new research question and fixed protocol.',
        'A future full-window min/max task (F04) is a different target, not a way to improve',
        'these endpoint numbers. Historical text/SMP inputs still require their availability checks.', '',
        '## Verification', '', r['verification_scope']]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=tree.reference.ROOT/'tree_validation')
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    result = review(directory)
    output = directory.with_name(directory.name+'_review')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'analysis.json', result)
    for name in ('analysis.txt', 'README.md'):
        (output/name).write_text(explanation(result), encoding='utf-8')
    print(f'Tree verification saved to {output}. No fitting or test scoring performed.')


if __name__ == '__main__':
    main()
