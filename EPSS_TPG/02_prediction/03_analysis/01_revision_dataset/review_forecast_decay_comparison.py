"""Review locked R05 outputs without training or new predictions."""
import argparse
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_decay_comparison as run

ROOT = run.OUTPUT
OUTPUT = ROOT.with_name(ROOT.name + '_review')


def analyze():
    lock = run.read(ROOT/'evaluation_lock.json')
    run.require(set(lock['files']) == run.expected_files(run.protocol()), 'Incomplete R05 artifacts')
    run.verify_hashes({str(ROOT/n): h for n,h in lock['files'].items()}, 'Verify R05 outputs')
    identity = run.read(ROOT/'run_identity.json')
    run.verify_hashes(identity['pinned_files'], 'Verify unchanged R05 sources')
    summary = run.read(ROOT/'summary.json')
    ids = identity['ids']
    inputs = run.read(run.prior.cache.base.PREPARED/'inputs.json')
    labels = run.read(run.prior.cache.base.PREPARED/'labels.json')
    cohorts = {}
    for split, selected in ids.items():
        rows = [{'cve': c, 'persistence': run.task.predict_persistence(inputs[c])['prediction'],
                 'actual_day60': labels[c]['y']['endpoint']} for c in selected]
        for row in rows:
            row['change'] = row['actual_day60'] - row['persistence']
            row['baseline_absolute_error'] = abs(row['change'])
        cohorts[split] = {'n': len(rows), 'unchanged': sum(r['change'] == 0 for r in rows),
                          'increased': sum(r['change'] > 0 for r in rows),
                          'decreased': sum(r['change'] < 0 for r in rows),
                          'persistence_mae': statistics.mean(r['baseline_absolute_error'] for r in rows),
                          'rows': rows}
    val = cohorts['validation']['rows']
    run.require([r['persistence'] for r in val] == summary['baselines']['persistence']['predictions'],
                'Baseline ordering/values differ')
    baseline_errors = [r['baseline_absolute_error'] for r in val]
    fits = {}
    for name, metrics in sorted(summary['fits'].items()):
        directory = ROOT/'fits'/name
        pred = run.read(directory/'predictions.json')
        history = run.read(directory/'history.json')
        gradients = run.read(directory/'gradients.json')
        run.require(len(history) == 30 and len(pred['prediction']) == len(val), 'Wrong budget or predictions')
        calculated_mae = statistics.mean(abs(r['actual_day60'] - p) for r,p in zip(val,pred['prediction']))
        run.require(abs(calculated_mae - metrics['metrics']['mae']) < 1e-14, 'MAE does not reproduce')
        errors = [abs(r['actual_day60'] - p) for r,p in zip(val,pred['prediction'])]
        norms = {g: (metrics['selected_parameter_norms'][g]/initial if initial else None)
                 for g,initial in metrics['initial_parameter_norms'].items()}
        fits[name] = {
            'metrics': metrics,
            'mae_minus_persistence': calculated_mae - cohorts['validation']['persistence_mae'],
            'improved_cves': sum(a < b for a,b in zip(errors,baseline_errors)),
            'worsened_cves': sum(a > b for a,b in zip(errors,baseline_errors)),
            'unchanged_error_cves': sum(a == b for a,b in zip(errors,baseline_errors)),
            'positive_corrections': sum(v > 0 for v in pred['correction']),
            'negative_corrections': sum(v < 0 for v in pred['correction']),
            'selected_to_initial_norm_ratios': norms,
            'history': history, 'gradient_counters': gradients,
            'training_clipped_evaluations': sum(h['training_clipped_below_zero'] +
                                               h['training_clipped_above_one'] for h in history),
            'validation_rows': [{**r, 'prediction': p, 'absolute_error': e,
                                 'correction': d} for r,p,e,d in zip(val,pred['prediction'],errors,pred['correction'])],
        }
    total = sum(baseline_errors)
    top = sorted(val, key=lambda r: r['baseline_absolute_error'], reverse=True)[:7]
    run.require(sorted(v['metrics']['selected_epoch'] for v in fits.values()) == [1]*10 + [2]*2,
                'Checkpoint pattern changed; review interpretation')
    run.require(all(v['metrics']['clipping']['below_zero'] == v['metrics']['clipping']['above_one'] == 0
                    for v in fits.values()), 'Selected clipping changed; review interpretation')
    output = {
        'schema': 'r05_decay_comparison_review_v1',
        'source_lock_sha256': run.sha256(ROOT/'evaluation_lock.json'),
        'review_script_sha256': run.sha256(Path(__file__)),
        'training_performed': False, 'new_predictions': False, 'test_scoring': False,
        'cohorts': cohorts, 'fits': fits, 'paired': summary['paired'],
        'aggregates': summary['aggregates'], 'scalar_offset': summary['baselines']['scalar_fit']['offset'],
        'top_persistence_errors': top,
        'top_case_fraction_of_total_absolute_error': top[0]['baseline_absolute_error']/total if total else None,
        'zero_decay_mae_improved_pairs': sum(p['zero_minus_original_mae'] < 0 for p in summary['paired']),
        'zero_decay_mae_worsened_pairs': sum(p['zero_minus_original_mae'] > 0 for p in summary['paired']),
        'all_fits_worse_mae_than_persistence': all(v['mae_minus_persistence'] > 0 for v in fits.values()),
        'all_selected_rmse_better_than_persistence': all(
            v['metrics']['metrics']['rmse'] < summary['baselines']['persistence']['metrics']['rmse'] for v in fits.values()),
        'limitations': summary['limitations'] + [
            'Only selected checkpoint weights were saved; epoch-30 weight norms cannot be recovered.',
            'Training clipping counts are repeated sample evaluations, not distinct CVEs.',
            'Nonzero gradient counters show a computational path, not useful predictive information.',
            'Differences across three seeds are descriptive; the same small validation sample is reused.',
        ],
        'conclusion': 'Zero decay preserves selected-checkpoint encoder norms but is not a reliable MAE improvement. Do not expand zero-decay training on this evidence.',
        'next_candidate_not_implemented': 'A fixed loss-only comparison of MAE versus current Smooth L1, using original decay and matched EPSS-only/graph controls. Persistence remains a valid reference; do not force a learned checkpoint to replace it.',
    }
    run.verify_hashes({str(ROOT/n): h for n,h in lock['files'].items()}, 'Confirm original outputs unchanged')
    return output


def text(a):
    lines = [
        'R05: Verified interpretation of the controlled weight-decay comparison', '',
        '1. Decision',
        a['conclusion'],
        'The earlier DQ19 finding was a local gradient imbalance, not proof that removing decay would improve forecasting.',
        'This controlled run shows that retaining weights is not sufficient for better held-out predictions.', '',
        '2. What was verified',
        'All 79 source artifacts and 12 checkpoint predictions were independently verified using the R05 verifier.',
        'This review then recomputes MAE, case counts and history summaries from the locked saved outputs.',
        'No retraining, new forecasts, graph changes or test scoring. All fits ran 30 epochs (480 updates).',
        'Ten fits select epoch 1; the two seed-43 graph fits select epoch 2.',
        'Within each seed/model comparison, the selected epochs match across decay settings.', '',
        '3. Validation results',
        'MAE is the mean absolute EPSS prediction error; lower is better.',
        f"Persistence MAE: {a['cohorts']['validation']['persistence_mae']:.8f}.",
        f"The training-fitted scalar correction is {a['scalar_offset']}; it therefore equals persistence.",
        'Model | Decay | Mean MAE across three seeds | Sample standard deviation',
    ]
    for r in a['aggregates']:
        lines.append(f"{r['model']} | {r['condition']} | {r['mean_mae']:.8f} | {r['sample_std_mae']:.8f}")
    lines += [
        'Zero decay worsens MAE in five of six paired comparisons. Only graph seed 29 improves.',
        'All 12 learned fits have worse MAE than persistence. Graph fusion has no consistent advantage.',
        f"All selected RMSEs below persistence: {a['all_selected_rmse_better_than_persistence']}.",
        'RMSE emphasizes larger errors; a small reduction there does not establish an overall MAE benefit.', '',
        '4. Why these cases are difficult',
    ]
    for split in ('train', 'validation'):
        c = a['cohorts'][split]
        lines.append(f"{split}: {c['n']} CVEs; {c['unchanged']} unchanged targets, "
                     f"{c['increased']} increases and {c['decreased']} decreases relative to persistence.")
    top = a['top_persistence_errors'][0]
    lines += [
        f"{top['cve']} rises from {top['persistence']:.5f} to {top['actual_day60']:.5f}.",
        f"It accounts for {100*a['top_case_fraction_of_total_absolute_error']:.2f}% of persistence's total absolute validation error.",
        'All selected models still predict roughly 0.00226-0.00253 for this case, far below 0.95432.',
        'The models improve only three or four CVEs each and worsen the other 60 or 61.',
        'Most predictions receive a small positive correction, adding error to the many unchanged targets.',
        'This is not a reason to remove the large-change CVE or to modify splits based on its outcome.', '',
        '5. Encoder shrinkage versus prediction',
        'The following ratios are selected-checkpoint parameter norm divided by initial norm.',
        'A ratio near one means similar total weight magnitude, not proof of useful learned information.',
        'Fit | EPSS encoder ratio | Graph input projection ratio',
    ]
    for name,r in a['fits'].items():
        norms = r['selected_to_initial_norm_ratios']
        lines.append(f"{name} | {norms['temporal']:.6f} | {norms.get('node_project', 'not present')}")
    lines += [
        'Removing decay retains EPSS encoder norms near 100% and graph projection norms near 101-104%.',
        'Original decay leaves EPSS norms near 82-91% and graph projection norms near 18-45%.',
        'Nevertheless the zero-decay EPSS-only corrections remain almost constant across CVEs.',
        'Graph corrections vary more, but that variation does not reliably improve validation error.',
        'Encoder data gradients are nonzero in 478 or 479 of 480 updates for all conditions.',
        'This rules out an entirely disconnected training path, not a lack of useful input signal.', '',
        '6. Training curves and clipping',
        'For graph models, training error decreases substantially while validation error later worsens.',
        'This is consistent with overfitting on the small training subset.',
        'The table uses epoch-30 history, not saved epoch-30 checkpoints (those weights were not saved).',
        'Fit | Epoch-30 training MAE | Epoch-30 validation MAE | Training clipped evaluations',
    ]
    for name,r in a['fits'].items():
        last = r['history'][-1]
        lines.append(f"{name} | {last['training_mae']:.8f} | {last['validation_mae']:.8f} | {r['training_clipped_evaluations']}")
    lines += [
        'All selected validation predictions are unclipped. Later graph training has some clipping.',
        'These counts are repeated training evaluations across epochs, not numbers of distinct affected CVEs.',
        'EPSS-only training Smooth L1 loss falls slightly while its training MAE generally rises.',
        'Smooth L1 and MAE optimize different error tradeoffs; better loss need not mean better MAE.',
        'The graph runs additionally show a generalization gap, so loss choice is not an established fix.', '',
        '7. Next step',
        a['next_candidate_not_implemented'],
        'Do not simultaneously change loss, gain, graph structure, split or optimizer.',
        'No new experiment is implemented by this review; keep the held-out test set closed.', '',
        '8. Limitations',
    ] + ['- ' + v for v in a['limitations']]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    run.require(args.verify_only or not OUTPUT.exists(), 'Review exists; use --verify-only')
    result = analyze()
    report = text(result)
    if args.verify_only:
        run.require(run.read(OUTPUT/'analysis.json') == result and
                    (OUTPUT/'analysis.txt').read_text() == report, 'Review reproduction differs')
        print('R05 review reproduced. No retraining, new forecasts or test scoring.')
        return
    OUTPUT.mkdir(parents=True, exist_ok=False)
    run.write(OUTPUT/'analysis.json', result)
    (OUTPUT/'analysis.txt').write_text(report, encoding='utf-8')
    print(f'R05 review saved to {OUTPUT}. No retraining, new forecasts or test scoring.')


if __name__ == '__main__':
    main()
