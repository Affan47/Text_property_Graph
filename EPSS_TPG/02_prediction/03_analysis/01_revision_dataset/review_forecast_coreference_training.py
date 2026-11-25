"""Review frozen R03 validation outputs; no fitting, tuning or test scoring."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

EPSS = Path(__file__).resolve().parents[3]
ROOT = EPSS/'05_results/02_evaluation/epss_day30_to_day60'
SOURCE = ROOT/'coreference_training_development_v1'
OUTPUT = ROOT/'coreference_training_development_v1_review'
MODES = ('no_coref', 'repeated_only', 'all_coref', 'epss_only')
SEEDS = (17, 29, 43)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _hash(stream)


def _hash(stream):
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024*1024), b''):
        result.update(chunk)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def errors(actual, predictions):
    require(len(actual) == len(predictions) and len(actual) > 0, 'Invalid prediction alignment')
    values = [p-y for y, p in zip(actual, predictions)]
    require(all(math.isfinite(v) for v in values), 'Non-finite error')
    absolute = sorted(map(abs, values), reverse=True)
    total = sum(absolute)
    return {'n': len(values), 'mae': statistics.mean(absolute),
            'rmse': math.sqrt(statistics.mean(v*v for v in values)),
            'mean_signed_error': statistics.mean(values),
            'top_3_absolute_error_share': sum(absolute[:3])/total if total else None}


def summarize(source):
    summary, rows = read(source/'summary.json'), read(source/'predictions.json')
    identity = summary['identity']
    require([r['cve'] for r in rows] == identity['ids']['validation'], 'CVE order differs')
    require(len(rows) == 197 and len(identity['ids']['train']) == 1023, 'Cohort differs')
    expected = {f'{m}__{s}' for m in MODES for s in SEEDS}
    require(all(set(r['predictions']) == expected for r in rows), 'Missing or extra fit predictions')
    require(summary['test_performance_evaluated'] is False, 'Unexpected evaluation scope')
    manifest = read(Path(identity['cache_directory'])/'manifest.json')
    graphs = {g['cve']: g['edges_by_mode'] for g in manifest['graphs']}
    ids = [r['cve'] for r in rows]
    actual = [r['actual_day60_epss'] for r in rows]
    predictions = {'persistence': [r['persistence'] for r in rows],
                   **{k: [r['predictions'][k] for r in rows] for k in sorted(expected)}}
    groups = {
        'unchanged_endpoint': [i for i,r in enumerate(rows) if r['actual_day60_epss'] == r['persistence']],
        'changed_endpoint': [i for i,r in enumerate(rows) if r['actual_day60_epss'] != r['persistence']],
        'with_coref': [i for i,c in enumerate(ids) if graphs[c]['all_coref'] > graphs[c]['no_coref']],
        'without_coref': [i for i,c in enumerate(ids) if graphs[c]['all_coref'] == graphs[c]['no_coref']],
        'with_pronoun_coref': [i for i,c in enumerate(ids) if graphs[c]['all_coref'] > graphs[c]['repeated_only']],
    }
    subgroup_results = {name: {'n': len(indexes), 'small_group_below_20': len(indexes) < 20,
        'models': {k: errors([actual[i] for i in indexes], [values[i] for i in indexes])
                   for k,values in predictions.items()} if indexes else {}}
        for name,indexes in groups.items()}
    fits = []
    for saved in summary['results']:
        mode, seed = saved['variant'], saved['seed']
        key = 'persistence' if seed is None else f'{mode}__{seed}'
        stats = errors(actual, predictions[key])
        require(all(math.isclose(stats[k], saved['metrics'][k], abs_tol=1e-12)
                    for k in ('mae', 'rmse', 'mean_signed_error')), 'Saved metrics differ')
        entry = dict(saved, error_concentration=stats['top_3_absolute_error_share'])
        if seed is not None:
            path = source/'fits'/key
            history, gradients = read(path/'history.json'), read(path/'gradients.json')
            selected = history[saved['selected_epoch']-1]
            entry.update(selected_history=selected, final_history=history[-1],
                train_validation_mae_gap=selected['validation_mae']-selected['training_mae'],
                gradients=gradients,
                nonzero_gradients_at_every_step=all(g['steps_with_nonzero_gradient'] ==
                    saved['epochs_completed']*math.ceil(1023/8) for g in gradients.values()))
        fits.append(entry)
    paired = []
    for mode in MODES:
        for seed in SEEDS:
            values = predictions[f'{mode}__{seed}']
            for ref in ('persistence', 'epss_only', 'no_coref'):
                if ref == mode:
                    continue
                other = predictions['persistence' if ref == 'persistence' else f'{ref}__{seed}']
                differences = [abs(p-y)-abs(q-y) for p,q,y in zip(values, other, actual)]
                paired.append({'variant': mode, 'seed': seed, 'reference': ref,
                    'mae_difference': statistics.mean(differences),
                    'lower_error_cves': sum(d < -1e-12 for d in differences),
                    'equal_error_cves': sum(abs(d) <= 1e-12 for d in differences),
                    'higher_error_cves': sum(d > 1e-12 for d in differences)})
    total = sum(abs(r['actual_day60_epss']-r['persistence']) for r in rows)
    extremes = []
    for r in sorted(rows, key=lambda r: -abs(r['actual_day60_epss']-r['persistence']))[:10]:
        extremes.append(dict(r, persistence_absolute_error=abs(r['actual_day60_epss']-r['persistence']),
            share_of_persistence_absolute_error=abs(r['actual_day60_epss']-r['persistence'])/total))
    return {'schema': 'r03_saved_validation_review_v1',
        'cohort': {k: len(v) for k,v in identity['ids'].items()},
        'actual_scores': {'min': min(actual), 'max': max(actual), 'median': statistics.median(actual),
                          'mean': statistics.mean(actual)},
        'fits': fits, 'seed_aggregates': summary['seed_aggregates'],
        'paired_comparisons': paired, 'subgroups': subgroup_results,
        'largest_persistence_errors': extremes, 'no_training': True, 'test_scoring': False,
        'limitations': identity['protocol']['limitations'] + [
            'Changed/unchanged endpoint groups use outcomes for post-hoc description only, not sample selection.',
            'Equal endpoints do not imply an unchanged intervening EPSS curve.',
            'COREF subgroups overlap; all trained weights can differ even on a graph without COREF edges.',
            'Three seeds describe optimization variability, not a confidence interval over future CVEs.',
            'Gradient flow does not establish effective branch use; no new branch interventions were performed.']}


def report(result):
    lines = ['R03: Verified Validation Findings', '',
        'Task: Days 0-30 EPSS and current CVE description predict the stored Day-60 EPSS score.',
        '1,023 training CVEs; the same 197 validation CVEs for every comparison. No test scoring.',
        'Frozen verification reloads every checkpoint and reproduces every saved validation prediction.', '',
        '1. Overall comparison',
        'MAE is average absolute score error; lower is better. An error of 0.01 is one percentage point.',
        'Variant | Mean MAE | Sample standard deviation across three seeds']
    base = result['fits'][0]['metrics']['mae']
    lines.append(f'Persistence (latest observed score) | {base:.8f} | not applicable')
    for mode, stats in result['seed_aggregates'].items():
        lines.append(f"{mode} | {stats['mae']['mean']:.8f} | {stats['mae']['sample_std']:.8f}")
    lines += ['', 'Every trained fit has higher MAE than persistence. Repeated-only has the lowest mean',
        'MAE among graph variants, but not consistently across seeds. EPSS-only has lower mean MAE',
        'than every graph variant. This run does not demonstrate added forecasting value from TPG.',
        'EPSS-only seed 29 has slightly lower RMSE than persistence; that does not reverse the MAE result.', '',
        '2. Every seed and selected checkpoint',
        'Variant | Seed | Selected epoch | Epochs run | Training MAE at selection | Validation MAE | Prediction min..max']
    for fit in result['fits'][1:]:
        r = fit['prediction_range']
        lines.append(f"{fit['variant']} | {fit['seed']} | {fit['selected_epoch']} | {fit['epochs_completed']} | "
            f"{fit['selected_history']['training_mae']:.8f} | {fit['metrics']['mae']:.8f} | {r['min']:.8f}..{r['max']:.8f}")
    lines += ['', 'All fits stop by the fixed patience rule, not by reaching 100 epochs.',
        'No-COREF predictions remain in a narrow low-score range in all seeds. Repeated-only seed 29,',
        'all-COREF seed 43 and especially EPSS-only seed 43 show similar compressed outputs.',
        'Other seeds produce a much wider range. This is seed-sensitive fitting, not a uniform failure.',
        'Training/validation gaps and worsening later validation errors are consistent with poor generalization.',
        'These observations do not prove that graph relations are wrong or identify one causal mechanism.',
        'Nonzero branch gradients occur at every recorded training step, but do not prove useful branch contributions.', '',
        '3. Why persistence is strong',
        f"{result['subgroups']['unchanged_endpoint']['n']}/197 endpoints equal the latest observed input score.",
        'Persistence is exactly correct for these cases. This does not mean every intervening score was unchanged.',
        f"The top three CVEs contribute {100*result['fits'][0]['error_concentration']:.2f}% of persistence absolute error.",
        'CVE | Latest observed input EPSS | Actual Day-60 EPSS | Persistence absolute error']
    for r in result['largest_persistence_errors'][:3]:
        lines.append(f"{r['cve']} | {r['persistence']:.5f} | {r['actual_day60_epss']:.5f} | {r['persistence_absolute_error']:.5f}")
    lines += ['', 'Post-hoc endpoint group | CVEs | Persistence MAE | No-COREF mean MAE | Repeated-only mean MAE | All-COREF mean MAE | EPSS-only mean MAE']
    for name in ('unchanged_endpoint', 'changed_endpoint'):
        group = result['subgroups'][name]
        means = [statistics.mean(group['models'][f'{m}__{s}']['mae'] for s in SEEDS) for m in MODES]
        lines.append(f"{name} | {group['n']} | {group['models']['persistence']['mae']:.8f} | " + ' | '.join(f'{v:.8f}' for v in means))
    lines += ['', 'Keep these difficult CVEs in the main evaluation; do not discard them to improve metrics.', '',
        '4. Input completeness',
        'Observed past scores | CVEs | Persistence MAE | No-COREF mean MAE | Repeated-only mean MAE | All-COREF mean MAE | EPSS-only mean MAE']
    for name, group in result['fits'][0]['missingness'].items():
        means = [statistics.mean(f['missingness'][name]['metrics']['mae'] for f in result['fits'] if f['variant'] == m) for m in MODES]
        lines.append(f"{name} | {group['n']} | {group['metrics']['mae']:.8f} | " + ' | '.join(f'{v:.8f}' for v in means))
    lines += ['', 'Persistence also has lower MAE in all three input-completeness groups for every trained fit.',
        'Performance problems are therefore not confined to CVEs with incomplete past scores.', '',
        '5. COREF interpretation',
        f"Validation graphs with any COREF: {result['subgroups']['with_coref']['n']}; without: {result['subgroups']['without_coref']['n']}; with pronoun COREF: {result['subgroups']['with_pronoun_coref']['n']}.",
        'Repeated-only improves over no-COREF in seeds 17 and 43 but not 29.',
        'All-COREF improves over repeated-only only in seed 29. Adding pronoun edges is not consistently helpful.',
        'These are separately trained models: changing training edges can affect predictions even where a validation graph has no COREF.',
        'Only explicit COREF edges change; this is not an ablation of every coreference-derived node or feature.', '',
        '6. Recommended next step',
        'Keep R03 fixed. First diagnose the saved R03 models with matched EPSS/text interventions to check branch response,',
        'especially wider-range versus compressed-output seeds. Reuse the established diagnostic design where applicable.',
        'Then predefine a small persistence-residual fusion experiment: latest observed EPSS plus a learned correction.',
        'Include a matched EPSS-only residual control, the same cohort, seeds and baseline. Prior temporal-only',
        'residual runs improved over direct prediction but did not beat persistence, so improvement is not assured.',
        'Do not combine graph expansion, new loss, extra text sources and a new head in one unexplained change.',
        'Do not open the test set or select a best seed from these exploratory validation results.', '',
        '7. Limitations', *result['limitations'], '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Reproduce the separate review without writing')
    args = parser.parse_args()
    sys.path.insert(0, str(EPSS/'02_prediction/01_package'))
    from epss import forecast_coreference_training as runner
    runner.torch.set_num_threads(1)
    runner.torch.use_deterministic_algorithms(True)
    identity = read(SOURCE/'run_identity.json')
    context = runner.context(identity['cache_directory'])
    runner.verify_output(SOURCE, *context)
    lock_hash = digest(SOURCE/'evaluation_lock.json')
    result = summarize(SOURCE)
    result['verification'] = {'checkpoint_predictions_reproduced': True,
        'source_evaluation_lock_sha256': lock_hash, 'review_script_sha256': digest(Path(__file__))}
    text = report(result)
    require(digest(SOURCE/'evaluation_lock.json') == lock_hash, 'Source lock changed')
    if args.verify_only:
        require(read(OUTPUT/'analysis.json') == result, 'Review JSON differs')
        require((OUTPUT/'analysis.txt').read_text() == text, 'Review text differs')
        for name, saved in read(OUTPUT/'review_lock.json')['files'].items():
            require(digest(OUTPUT/name) == saved, 'Review artifact changed')
        print('R03 review reproduced; source checkpoints and validation predictions verified. No training or test scoring.')
        return
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (OUTPUT/'analysis.txt').write_text(text)
    (OUTPUT/'review_lock.json').write_text(json.dumps({'files': {
        name: digest(OUTPUT/name) for name in ('analysis.json', 'analysis.txt')}}, indent=2)+'\n')
    print(f'R03 review saved to {OUTPUT}. Original artifacts unchanged; no training or test scoring.')


if __name__ == '__main__':
    main()
