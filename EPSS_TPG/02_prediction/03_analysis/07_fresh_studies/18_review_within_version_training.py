"""Review completed CT07 full artifacts without fitting or accessing official holdouts."""
import argparse
from collections import defaultdict
import csv
import os
from pathlib import Path
import sys

for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
ROOT = Path(__file__).resolve().parents[3]
for rel in ('01_tpg/01_core', '01_tpg/10_extensions', '02_prediction/01_package', '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT/rel))

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from tqdm import tqdm
from epss.within_version_training_v1 import protocol, reporting
from epss.within_version_pooled_v1 import data


def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def predictions(path):
    rows = read_rows(path)
    for row in rows:
        data.require(row['M_present'] in ('True', 'False'), 'Invalid source flag')
        row['M_present'] = row['M_present'] == 'True'
        for key in ('actual', 'prediction'):
            row[key] = float(row[key])
            data.require(np.isfinite(row[key]) and 0 <= row[key] <= 1, 'Invalid EPSS value')
        for key in ('fold', 'seed'):
            row[key] = int(row[key])
    keys = [(r['policy'], r['model'], r['seed'], r['cve']) for r in rows]
    data.require(len(keys) == len(set(keys)), 'Duplicate assessment row')
    return rows


def mean_metrics(metrics):
    groups = defaultdict(list)
    for row in metrics:
        groups[row['policy'], row['partition'], row['model']].append(row)
    result = []
    keys = ('rmse', 'mae', 'r2', 'pearson', 'spearman', 'signed_bias', 'auroc_high_score',
            'ap_high_score', 'top_10_hits', 'top_25_hits', 'top_50_hits', 'top_100_hits')
    for (policy, part, model), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r['seed'])
        data.require(len({(r['n'], r['high_score_cves']) for r in rows}) == 1, 'Unmatched metric support')
        result.append({'policy': policy, 'partition': part, 'model': model,
                       'cves': rows[0]['n'], 'high_score_cves': rows[0]['high_score_cves'],
                       'seeds': len(rows),
                       **{k: float(np.mean([r[k] for r in rows])) if all(r[k] is not None for r in rows)
                          else None for k in keys}})
    return result


def seed_differences(metrics):
    lookup = {(r['policy'], r['partition'], r['model'], r['seed']): r['rmse'] for r in metrics}
    rows = []
    for policy in ('main', 'published_since_v2'):
        for part in ('all', 'fold_1', 'fold_2', 'fold_3', 'M_present', 'M_absent'):
            for seed in (17, 29, 43):
                rows.append({'policy': policy, 'partition': part, 'seed': seed,
                             **{a+'_minus_'+b: lookup[policy, part, a, seed]-lookup[policy, part, b, seed]
                                for a, b in [('DM', 'D'), ('DM', 'DF'), ('DF', 'D')]}})
    return rows


def inspect_history(history, evaluation):
    data.require([r['epoch'] for r in history] == list(range(1, len(history)+1)), 'Incomplete epoch history')
    best = min(history, key=lambda r: r['selection_rmse'])
    data.require(best['epoch'] == evaluation['best_epoch']
                 and best['selection_rmse'] == evaluation['selection_rmse']
                 and len(history) == evaluation['history_epochs'], 'Not the earliest strict best selection checkpoint')
    data.require(history[-1]['patience'] == 15 or len(history) == 100, 'Incomplete full fit')
    return {'best_epoch': best['epoch'], 'completed_epochs': len(history),
            'selection_rmse_best': best['selection_rmse'],
            'selection_rmse_final': history[-1]['selection_rmse'],
            'optimization_mse_first': history[0]['optimization_mse'],
            'optimization_mse_best_epoch': best['optimization_mse'],
            'optimization_mse_final': history[-1]['optimization_mse']}


def check_metrics(rows, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[row['policy'], row['model'], row['seed']].append(row)
    for record in tqdm(metrics, desc='Recalculate saved assessment error tables', unit='row'):
        selected = sorted((r for r in groups[record['policy'], record['model'], record['seed']]
                           if reporting.in_partition(r, record['partition'])), key=lambda r: r['cve'])
        y = np.array([r['actual'] for r in selected]); pred = np.array([r['prediction'] for r in selected])
        data.require(len(y) == record['n'] and int((y >= .1).sum()) == record['high_score_cves'], 'Support changed')
        error = pred-y
        for key, value in [('rmse', np.sqrt(np.mean(error**2))), ('mae', np.mean(abs(error))), ('signed_bias', np.mean(error))]:
            data.require(np.isclose(value, record[key], rtol=0, atol=1e-14), 'Saved metric differs: '+key)
    return groups


def error_analysis(groups, settings):
    result, examples = [], []
    for policy in settings['policies']:
        for model in settings['arms']:
            rs = [sorted(groups[policy, model, seed], key=lambda r: r['cve']) for seed in settings['seeds']]
            y = np.array([r['actual'] for r in rs[0]])
            ps = np.array([[r['prediction'] for r in rows] for rows in rs])
            high = y >= .1
            mse = ((ps-y)**2).mean(axis=0)
            result.append({'policy': policy, 'model': model, 'cves': len(y), 'high_score_cves': int(high.sum()),
                           'high_score_squared_error_share': float(mse[high].sum()/mse.sum()),
                           'mean_predicted_score_on_high_cases': float(ps[:, high].mean()),
                           'mean_actual_score_on_high_cases': float(y[high].mean()),
                           'mean_high_score_true_positives_at_0_1': float((ps[:, high] >= .1).sum(axis=1).mean()),
                           'mean_false_positives_at_0_1': float((ps[:, ~high] >= .1).sum(axis=1).mean()),
                           'largest_five_squared_error_share': float(np.sort(mse)[-5:].sum()/mse.sum())})
            if model == 'DM':
                # Descriptive examples only: every CVE remains in all reported metrics.
                indices = sorted(range(len(y)), key=lambda i: (-mse[i], rs[0][i]['cve']))[:10]
                # Constants have seed zero, not 17.
                lookups = {a: {r['cve']: r for r in groups[policy, a, 0]} for a in settings['controls']}
                for rank, i in enumerate(indices, 1):
                    row = rs[0][i]; cid = row['cve']
                    example = {'policy': policy, 'rank_by_DM_mean_squared_error': rank,
                               'cve': cid, 'fold': row['fold'], 'M_present': row['M_present'],
                               'actual': float(y[i]), 'DM_mean_squared_error': float(mse[i])}
                    for a in settings['arms']:
                        vals = [next(r['prediction'] for r in groups[policy, a, seed] if r['cve'] == cid)
                                for seed in settings['seeds']]
                        example[a+'_prediction_mean'] = float(np.mean(vals))
                        example[a+'_prediction_min'] = float(min(vals))
                        example[a+'_prediction_max'] = float(max(vals))
                    example.update({a: lookups[a][cid]['prediction'] for a in settings['controls']})
                    examples.append(example)
    return result, examples


def report_text(summary, intervals, histories, errors, examples, historical, differences):
    def metric(policy, part, model, key='rmse'):
        return next(r[key] for r in summary if (r['policy'], r['partition'], r['model']) == (policy, part, model))
    def ci(policy, part, contrast, unit='evidence_component'):
        r = next(r for r in intervals if (r['policy'], r['partition'], r['contrast'], r['unit']) == (policy, part, contrast, unit))
        return f"{r['difference']:+.9f} [{r['low']:+.9f}, {r['high']:+.9f}]"
    lines = [
        'CT07 completed full study: independent saved-result review', '',
        '1. What was evaluated',
        '54 completed pooled neural fits: two policies x three chronological folds x D/DF/DM x three seeds.',
        '8,053 distinct inner-assessment CVEs, including 834 with accepted posts and 7,219 without.',
        '62 have Day-60 EPSS >= 0.1; 26 of those have posts. These are not exploitation labels.',
        'D uses descriptions; DF uses the same descriptions plus a post-presence flag; DM adds post text.',
        'All three also use the same types of optimization-fitted structured inputs. No GNN was trained.',
        'Each assessment prediction came from a checkpoint selected on a separate earlier selection role.',
        'This is retrospective within-v2 score estimation, not forecasting, new data or final test evaluation.',
        'Official validation/test roles were not scored. Current text/CVSS can postdate the target.', '',
        '2. Overall results',
        'Each model row averages three seed-wise metrics, not ensemble predictions. Constants differ by fold.',
        'RMSE emphasizes large score errors; MAE is mean absolute score error. Lower is better for both.',
        'AUROC/AP rank EPSS >=0.1 cases, not exploited CVEs. AP is average precision. Higher is better.',
        'Policy | Model | RMSE | MAE | R2 | Spearman | High-score AUROC | High-score AP | Top-50 hits']
    for r in summary:
        if r['partition'] == 'all':
            lines.append(' | '.join([r['policy'], r['model']]+[f"{r[k]:.6f}" if r[k] is not None else 'undefined'
                         for k in ('rmse', 'mae', 'r2', 'spearman', 'auroc_high_score', 'ap_high_score', 'top_50_hits')]))
    lines += ['', '3. Primary result and reporting gates',
              'Differences below are first model minus reference RMSE; negative favors the first model.',
              'Intervals use 20,000 paired draws at seed 6029, conditional on the saved fitted models.',
              'They are nominal exploratory intervals, not adjusted for multiple comparisons or repeated data use.',
              'Main DM minus D, grouped: '+ci('main', 'all', 'DM_minus_D'),
              'Main DM minus D, CVE resampling: '+ci('main', 'all', 'DM_minus_D', 'CVE'),
              'The primary addition-of-post-text improvement is not established; both intervals include zero.',
              'Main DM minus DF, grouped: '+ci('main', 'all', 'DM_minus_DF'),
              'Main DM minus DF, CVE resampling: '+ci('main', 'all', 'DM_minus_DF', 'CVE'),
              'DM is worse than DF under the CVE interval, but not decisively under grouped resampling.',
              'Main DM minus optimization mean: '+ci('main', 'all', 'DM_minus_optimization_mean'),
              'Main DM minus optimization median: '+ci('main', 'all', 'DM_minus_optimization_median'),
              'DF has the lowest main point-estimate RMSE and beats the mean control, but its interval against the median crosses zero.',
              'No overall neural arm establishes superiority over both constants under both interval units.',
              'None of the six frozen DM useful-estimation/content reporting gates passes. This is not proof that text has no information.', '',
              '4. Post-present and post-absent results',
              'Policy | Model | RMSE on 834 with posts | RMSE on 7,219 without posts']
    for policy in ('main', 'published_since_v2'):
        for a in ('optimization_mean', 'optimization_median', 'D', 'DF', 'DM'):
            lines.append(f"{policy} | {a} | {metric(policy, 'M_present', a):.9f} | {metric(policy, 'M_absent', a):.9f}")
    lines += ['Main post-present DM minus D: '+ci('main', 'M_present', 'DM_minus_D'),
              'Main post-present DM minus DF: '+ci('main', 'M_present', 'DM_minus_DF'),
              'D beats both constants on post-present CVEs despite receiving no post text; cohort selection is not content benefit.',
              'Main post-absent DM is worse than the median under both intervals.', '',
              '5. Publication-restricted sensitivity is not a replacement primary result',
              'Only optimization is restricted to publication >=2022-02-04; selection/assessment IDs are unchanged.',
              'This also changes sample size, recency, composition, preprocessing and optimization updates.',
              'Restricted overall DM minus D: '+ci('published_since_v2', 'all', 'DM_minus_D'),
              'Restricted overall DM minus DF: '+ci('published_since_v2', 'all', 'DM_minus_DF'),
              'Restricted overall DM minus own mean: '+ci('published_since_v2', 'all', 'DM_minus_optimization_mean'),
              'Restricted overall DM minus own median: '+ci('published_since_v2', 'all', 'DM_minus_optimization_median'),
              'Restricted post-present DM minus D: '+ci('published_since_v2', 'M_present', 'DM_minus_D'),
              'Restricted post-absent DM minus D: '+ci('published_since_v2', 'M_absent', 'DM_minus_D'),
              'The relative overall DM-versus-D gain is supported but concentrated in post-absent cases.',
              'D and DM have identical inputs there under equal weights; independently fitted weights/checkpoints differ.',
              'An effect on these cases is not a direct inference-time benefit from feeding post text.', '',
              'Policy | Partition | DM-minus-D RMSE for seeds 17, 29, 43 (negative favors DM)']
    for policy in ('main', 'published_since_v2'):
        for part in ('all', 'fold_1', 'fold_2', 'fold_3', 'M_present', 'M_absent'):
            ds = [r['DM_minus_D'] for r in differences if r['policy'] == policy and r['partition'] == part]
            lines.append(f"{policy} | {part} | "+', '.join(f'{v:+.9f}' for v in ds))
    lines += ['',
              '6. Ranking, high-score errors and training history',
              'High-score prevalence is 62/8,053 (0.77%). Good AUROC does not imply accurate scores or high top-rank precision.',
              'Main DM has higher AUROC than D/DF but lower AP than either; there is no universal ranking winner.',
              'Ranking pooled over folds compares separately trained score scales. Fold-wise constants can rank across folds.',
              'Top-k constant hits depend on deterministic tie-breaking and are not evidence of useful within-fold ranking.',
              'All 54 selected epochs equal the earliest minimum selection RMSE. All runs stopped by patience, not the epoch cap.',
              'Training-history CSV includes every best epoch and final epoch; lower training error alone is not assessment improvement.']
    for policy in ('main', 'published_since_v2'):
        for a in ('D', 'DF', 'DM'):
            hs = [r for r in histories if r['policy'] == policy and r['model'] == a]
            er = next(r for r in errors if r['policy'] == policy and r['model'] == a)
            lines.append(f"{policy} {a}: best epochs by fold/seed {[r['best_epoch'] for r in hs]}; "
                         f"high-case MAE={metric(policy, 'score_ge_0.1', a, 'mae'):.6f}; "
                         f"high cases account for {100*er['high_score_squared_error_share']:.2f}% of squared error.")
    lines += ['', '7. Real examples: five largest main-DM errors, no exclusions',
              'Predictions here are seed means for illustration only; official metrics average seed-wise errors.',
              'CVE | Posts present | Actual Day-60 EPSS | D prediction | DF prediction | DM prediction']
    for r in examples:
        if r['policy'] == 'main' and r['rank_by_DM_mean_squared_error'] <= 5:
            lines.append(f"{r['cve']} | {r['M_present']} | {r['actual']:.6f} | {r['D_prediction_mean']:.6f} | "
                         f"{r['DF_prediction_mean']:.6f} | {r['DM_prediction_mean']:.6f}")
    lines += ['', '8. Comparison with the frozen CT05 reference',
              'Same 8,053 assessment CVEs, different earlier fitting/selection roles. DF was an external exploratory reference.',
              'CT05 versus CT07 is not a matched version-only intervention or independent new evidence.',
              'Model | CT05 overall RMSE | CT07 main overall RMSE | CT05 post-present RMSE | CT07 main post-present RMSE']
    for a in ('D', 'DF', 'DM'):
        old = lambda part: np.mean([float(r['rmse']) for r in historical if r['model'] == a and r['partition'] == part])
        lines.append(f"{a} | {old('all'):.9f} | {metric('main', 'all', a):.9f} | {old('M_present'):.9f} | {metric('main', 'M_present', a):.9f}")
    lines += ['The earlier post-text advantage is not reproduced in the current main D/DF/DM comparison.', '',
              '9. Decision',
              'Keep CT07 complete and frozen. The primary content-addition result is inconclusive, not a passing GNN-expansion gate.',
              'Do not launch a large TPG sweep, retune the loss, redraw folds or promote a sensitivity/subgroup winner from these results.',
              'Consolidate CT05-CT07 as exploratory evidence; define the scientific target and independent evaluation before more fitting.',
              'A ranking/exploitation task would require its own labels, controls and protocol; EPSS>=0.1 is not VulnCheck listing.',
              'Future confirmatory source/graph work needs adequate post-bearing high-score support and protected evaluation data.',
              'This review recomputes saved errors and verifies hashes/history. The original --verify-only runner replays checkpoint predictions.',
              'All saved experiment artifacts, inputs and models remain unchanged. No training or new holdout scoring.']
    return '\n'.join(lines)+'\n'


def figure(path, intervals):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pairs = [('DM_minus_D', 'DM minus D'), ('DM_minus_DF', 'DM minus DF'),
             ('DM_minus_optimization_mean', 'DM minus fit mean'), ('DM_minus_optimization_median', 'DM minus fit median')]
    for ax, policy, title in zip(axes, ('main', 'published_since_v2'), ('Main policy', 'Publication-restricted sensitivity')):
        rows = [next(r for r in intervals if r['policy'] == policy and r['partition'] == 'all'
                     and r['unit'] == 'evidence_component' and r['contrast'] == pair) for pair, _ in pairs]
        for i, r in enumerate(rows):
            ax.plot([r['low'], r['high']], [i, i], color='#277b91', linewidth=2)
            ax.plot(r['difference'], i, 'o', color='#a64c60')
        ax.axvline(0, color='#555555', linestyle='--', linewidth=1)
        ax.set(title=title, yticks=range(4), yticklabels=[label for _, label in pairs],
               xlabel='RMSE difference: negative favors DM', xlim=(-.0021, .0026))
        ax.invert_yaxis(); ax.grid(axis='x', alpha=.2); ax.tick_params(labelsize=9)
    fig.suptitle('CT07: adding post text does not clear the constant-control gate')
    fig.text(.5, .035, 'Each row compares the same 8,053 CVEs. D = description; DF = description + post flag; DM = description + posts.\n'
             'Dots: mean seed-wise differences. Lines: nominal 95% paired evidence-group intervals, 20,000 draws.\n'
             'Crossing the dashed zero line means the comparison is inconclusive. These are not test-set results.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .16, 1, .94))
    fig.savefig(path, dpi=150, metadata={'Software': 'CT07 full results review v1'}); plt.close(fig)


def run(verify=False):
    frozen, p = protocol.load(); s = frozen['settings']; root = protocol.OUTPUT/'full'; report = root/'report'
    data.old.publication.check_lock(report)
    ident = data.c.read(report/'identity.json')
    data.require(ident['source_pins'] == protocol.pins()
                 and ident['protocol_sha256'] == data.c.sha(protocol.OUTPUT/'protocol/protocol.json'), 'Report identity changed')
    findings = data.c.read(report/'findings.json')
    data.require(findings['scale'] == 'full' and not findings['official_validation_or_test_scored'], 'Wrong evaluation scope')
    data.require(len(ident['fit_locks']) == 54, 'Missing completed fits')
    histories = []
    for lock, digest in tqdm(sorted(ident['fit_locks'].items()), desc='Check 54 fit locks and selection histories', unit='fit'):
        folder = Path(lock).parent; data.require(data.c.sha(Path(lock)) == digest, 'Fit lock changed')
        data.old.publication.check_lock(folder)
        identity = data.c.read(folder/'identity.json'); evaluation = data.c.read(folder/'evaluation.json')
        data.require(evaluation['checkpoint_sha256'] == data.c.sha(folder/'best_checkpoint.pt'), 'Checkpoint hash differs')
        histories.append({'policy': identity['policy'], 'fold': identity['fold'], 'model': identity['arm'], 'seed': identity['seed'],
                          **inspect_history(data.c.read(folder/'history.json'), evaluation)})
    rows = predictions(report/'oof_predictions.csv')
    groups = check_metrics(rows, findings['metrics'])
    reference = {r['cve']: (r['actual'], r['fold'], r['M_present'], r['target_date']) for r in groups['main', 'D', 17]}
    data.require(len(reference) == 8053 and sum(v[2] for v in reference.values()) == 834
                 and sum(v[0] >= .1 for v in reference.values()) == 62 and len(groups) == 22, 'Unexpected assessment cohort')
    for key, rs in groups.items():
        data.require({r['cve']: (r['actual'], r['fold'], r['M_present'], r['target_date']) for r in rs} == reference,
                     'Unmatched labels, masks or assessment roles: '+str(key))
    summary = mean_metrics(findings['metrics']); intervals = findings['paired_intervals']
    data.require(len(intervals) == 108 and all(r['repetitions'] == 20000 and r['seed'] == 6029 for r in intervals), 'Interval protocol changed')
    gates = reporting.outcome_gates(intervals, s)
    data.require(gates == findings['reporting_gates_not_confirmatory_tests'], 'Reporting gates differ')
    errors, examples = error_analysis(groups, s)
    differences = seed_differences(findings['metrics'])
    historical = read_rows(protocol.OUTPUT/'protocol/ct05_reference_metrics.csv')
    pins = {str(report/'artifact_lock.json'): data.c.sha(report/'artifact_lock.json'),
            str(protocol.OUTPUT/'protocol/artifact_lock.json'): data.c.sha(protocol.OUTPUT/'protocol/artifact_lock.json'),
            str(Path(__file__).resolve()): data.c.sha(Path(__file__)), **ident['fit_locks']}
    review = {'schema': 'ct07_full_review_v1', 'metrics': summary, 'paired_intervals': intervals,
              'reporting_gates': gates, 'training_history': histories, 'error_concentration': errors,
              'examples_selection': 'Ten largest mean-seed DM squared errors per policy; no records excluded',
              'examples': examples, 'completed_fits': 54, 'assessment_cves': 8053,
              'seed_differences': differences,
              'with_posts': 834, 'without_posts': 7219, 'high_score_cves': 62,
              'checkpoint_replay_command': '17_run_within_version_training.py --stage train --scale full --jobs 3 --verify-only',
              'review_trains_or_scores_holdouts': False}
    destination = protocol.OUTPUT/'full_review_v1'
    with data.old.publication.transaction(destination, verify=verify) as temp:
        data.old.base.write(temp/'identity.json', {'input_hashes': pins, 'source_pins': protocol.pins(), 'runtime': data.old.runtime()})
        data.old.base.write(temp/'findings.json', review)
        for name, rs in [('mean_metrics', summary), ('paired_intervals', intervals), ('training_history', histories),
                         ('error_concentration', errors), ('real_examples', examples), ('reporting_gates', gates)]:
            (temp/(name+'.csv')).write_text(data.old.reporting.csv_text(rs))
        (temp/'seed_differences.csv').write_text(data.old.reporting.csv_text(differences))
        (temp/'REPORT.txt').write_text(report_text(summary, intervals, histories, errors, examples, historical, differences))
        figure(temp/'paired_rmse_intervals.png', intervals)
        data.old.publication.check_pins(pins)
    print(f'CT07 full results review {"reproduced" if verify else "saved"}: {destination}. No training or official holdout scoring.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4, user_api='blas'):
        run(args.verify_only)
