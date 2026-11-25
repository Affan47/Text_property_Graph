"""Review locked MS03 development artifacts; no training or new predictions."""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss import forecast_matched_multisource as run

OUTPUT = run.ROOT / '05_results/02_evaluation/epss_day60_to_day90/matched_multisource_review_v1'
PREFIXES = ('temporal', 'node_project', 'layers', 'source_embeddings', 'source_attention', 'head')


def movement(rows):
    delta = [r['target'] - r['last_observed_score'] for r in rows]
    errors = sorted(map(abs, delta), reverse=True)
    total = sum(errors)
    return {'n': len(rows), 'unchanged': sum(d == 0 for d in delta),
            'increased': sum(d > 0 for d in delta), 'decreased': sum(d < 0 for d in delta),
            'unchanged_percent': 100 * sum(d == 0 for d in delta)/len(rows),
            'median_change': statistics.median(delta), 'median_absolute_change': statistics.median(errors),
            'raw_persistence_mae': statistics.mean(errors),
            'top_error_shares': {str(n): sum(errors[:n])/total if total else 0. for n in (3, 10, 20)},
            'partial_recent_histories': sum(r['observed_days'] < 31 for r in rows)}


def corrections(rows):
    values = [r['correction'] for r in rows]
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r['source_mask'])].append(r['correction'])
    uniform_error = 0.
    for r in rows:
        mask = run.torch.tensor(r['source_mask'], dtype=run.torch.float32)
        uniform = mask / mask.sum().clamp_min(1)
        actual = run.torch.tensor(r['source_attention'])
        uniform_error = max(uniform_error, float((actual - uniform).abs().max()))
    return {'min': min(values), 'max': max(values), 'std': statistics.pstdev(values),
            'distinct_values': len(set(values)), 'mask_groups': len(groups),
            'max_within_mask_group_range': max(max(v)-min(v) for v in groups.values()),
            'max_attention_difference_from_uniform_present_sources': uniform_error}


def norm(state, prefix):
    return math.sqrt(sum(float(v.double().square().sum()) for k, v in state.items() if k.startswith(prefix+'.')))


def analyze():
    directory = run.OUTPUT / 'full'
    identity = run.read(directory / 'run_identity.json')
    run.require(identity['scale'] == 'full' and identity['schema'] == run.SCHEMA and not identity['test_scoring'], 'Wrong run')
    pins, _ = run.check_files(directory, 'evaluation_lock.json', run.SCHEMA, run.expected(identity))
    small = run.OUTPUT / 'smoke'
    small_identity = run.read(small / 'run_identity.json')
    small_pins, _ = run.check_files(small, 'evaluation_lock.json', run.SCHEMA, run.expected(small_identity))
    pins.update(small_pins)
    pins[str(Path(__file__).resolve())] = run.sha256(Path(__file__).resolve())
    def pinned_read(path):
        name = str(path)
        run.require(name in identity['pinned_files'] and run.sha256(path) == identity['pinned_files'][name], 'Unpinned/changed input: '+name)
        pins[name] = identity['pinned_files'][name]
        return run.read(path)
    prepared = run.cache.base.preparation.OUTPUT / 'full'
    inputs = pinned_read(prepared/'inputs.json')
    labels = pinned_read(prepared/'labels.json')
    index = pinned_read(run.cache.OUTPUT/'full/cve_index.json')
    summary = run.read(directory/'summary.json')
    refs = run.read(directory/'baselines.json')
    ids = identity['ids']['validation']
    run.require([r['cve'] for r in refs['rows']] == ids, 'Unmatched baseline IDs')
    tensors = {'validation': run.tensor_inputs([inputs[c] for c in ids])}
    run.require(run.references(identity, tensors, labels) == refs, 'Persistence does not reproduce')
    cases, distributions = {}, {}
    for split, chosen in identity['ids'].items():
        rows = [{'cve': c, 'anchor_date': inputs[c]['input_end_date'], 'target_date': labels[c]['target_date'],
                 'last_observed_score': run.controls.persistence(inputs[c]), 'target': labels[c]['y']['endpoint'],
                 'observed_days': sum(inputs[c]['histories']['recent_30_60']['observed_mask'])} for c in chosen]
        distributions[split] = movement(rows)
        cases[split] = rows
    top = sorted(cases['validation'], key=lambda r: (-abs(r['target']-r['last_observed_score']), r['cve']))[:10]
    top = [{**r, 'absolute_change': abs(r['target']-r['last_observed_score'])} for r in top]
    fits, predicted = {}, {}
    for seed in run.tqdm(identity['config']['seeds'], desc='Review saved seeds', unit='seed'):
        for variant in run.model.VARIANTS:
            name = f'{variant}__{seed}'
            fitdir = directory/'fits'/name
            rows = run.read(fitdir/'predictions.json')
            run.require([r['cve'] for r in rows] == ids, 'Unmatched prediction IDs')
            for row, base in zip(rows, refs['rows']):
                run.require(row['actual_day90_epss'] == labels[row['cve']]['y']['endpoint'] and
                            row['persistence_float32'] == base['prediction'], 'Changed target/base')
                expected = [bool(v) and source in run.model.enabled_sources(variant)
                            for source, v in zip(run.model.SOURCES, index[row['cve']]['source_mask'])]
                run.require(row['source_mask'] == expected and all(w == 0 for w, available in zip(row['source_attention'], expected) if not available), 'Wrong source mask/attention')
            metrics = run.scored(rows, labels)
            run.require(metrics == summary['fits'][name]['metrics'], 'Saved metrics differ')
            history = run.read(fitdir/'history.json')
            saved = run.torch.load(fitdir/'checkpoint.pt', weights_only=True, map_location='cpu')
            initial = run.initialize(identity, variant, seed).state_dict()
            run.require(saved['initial_state_sha256'] == run.controls.state_digest(initial), 'Initial weights differ')
            selected = saved['selected_epoch']
            run.require(selected == min(history, key=lambda h: h['validation_mae'])['epoch'] and
                        metrics['mae'] == history[selected-1]['validation_mae'], 'Wrong selected checkpoint')
            sizes = {p: {'initial_l2': norm(initial, p), 'saved_l2': norm(saved['state_dict'], p)} for p in PREFIXES}
            groups = {}
            for j, source in enumerate(run.model.SOURCES):
                for present in (False, True):
                    chosen = [r for r in rows if bool(index[r['cve']]['source_mask'][j]) == present]
                    if chosen:
                        groups[source+('_present' if present else '_absent')] = {
                            'n': len(chosen), 'mae': statistics.mean(abs(r['prediction']-r['actual_day90_epss']) for r in chosen),
                            'persistence_mae': statistics.mean(abs(r['persistence_float32']-r['actual_day90_epss']) for r in chosen)}
            fits[name] = {'metrics': metrics, 'best_epoch': selected, 'epochs_run': len(history),
                          'mae_minus_persistence': metrics['mae']-refs['metrics']['mae'],
                          'relative_mae_reduction_percent': 100*(refs['metrics']['mae']-metrics['mae'])/refs['metrics']['mae'],
                          'corrections': corrections(rows), 'parameter_norms': sizes,
                          'first_epoch_gradient_sums': history[0]['data_gradient_abs_sums'],
                          'selected_epoch_gradient_sums': history[selected-1]['data_gradient_abs_sums'],
                          'source_availability_subgroups': groups, 'history': history}
            predicted[name] = rows
    paired = []
    for seed in identity['config']['seeds']:
        for a, b in [('description_epss', 'epss_only'), ('all_sources_epss', 'epss_only'), ('all_sources_epss', 'description_epss')]:
            left, right = predicted[f'{a}__{seed}'], predicted[f'{b}__{seed}']
            diff = [abs(x['prediction']-x['actual_day90_epss'])-abs(y['prediction']-y['actual_day90_epss']) for x, y in zip(left, right)]
            paired.append({'seed': seed, 'comparison': a+' minus '+b, 'mean_absolute_error_difference': statistics.mean(diff),
                           'cves_lower_error': sum(d < 0 for d in diff), 'cves_equal_error': sum(d == 0 for d in diff),
                           'cves_higher_error': sum(d > 0 for d in diff)})
    aggregate = {}
    for variant in run.model.VARIANTS:
        values = [fits[f'{variant}__{s}']['metrics']['mae'] for s in identity['config']['seeds']]
        aggregate[variant] = {'mean_mae': statistics.mean(values), 'sample_std_mae': statistics.stdev(values)}
    run.require(aggregate == summary['aggregates'], 'Aggregate differs')
    return {'schema': 'ms03_completed_review_v1', 'original_artifact_pins': pins,
            'verification_scope': 'Locked saved artifacts, metrics, input/label pairing and initial-state checks. This review does not rerun inference; use the original --scale full --verify-only command for checkpoint replay.',
            'cohort': {s: len(v) for s, v in identity['ids'].items()}, 'gain': identity['gain'],
            'persistence': refs['metrics'], 'aggregates': aggregate, 'fits': fits, 'paired': paired,
            'endpoint_changes_from_original_scores': distributions, 'largest_validation_changes': top,
            'source_coverage': identity['source_coverage'], 'limitations': identity['protocol']['limitations'],
            'test_scoring': False, 'training': False, 'new_predictions': False}


def report(a):
    lines = ['MS03: completed matched comparison', '', '1. What was compared',
             '2,548 training / 1,104 validation CVEs; Day 30-60 history predicts Day-90 EPSS.',
             'Nine fresh fits: three configurations x seeds 17, 29, 43. Test set remains closed.',
             'Source masks, partial histories, targets, batch schedules and settings are fixed across modes.',
             '', '2. Results (smaller MAE is better; scores are probabilities on a 0-1 scale)',
             f"Persistence MAE: {a['persistence']['mae']:.12f}",
             'Model | Mean MAE over three seeds | Sample standard deviation']
    for name, r in a['aggregates'].items():
        lines.append(f"{name} | {r['mean_mae']:.12f} | {r['sample_std_mae']:.12g}")
    lines += ['', 'Model/seed | Selected epoch | Epochs run | MAE | RMSE | Relative MAE reduction vs persistence (%)']
    for name, r in sorted(a['fits'].items()):
        lines.append(f"{name} | {r['best_epoch']} | {r['epochs_run']} | {r['metrics']['mae']:.12f} | {r['metrics']['rmse']:.12f} | {r['relative_mae_reduction_percent']:.6f}")
    lines += ['', 'All fits beat persistence numerically, but only by a very small amount.',
              'EPSS-only has the lowest mean MAE. Adding description or all text does not consistently help across seeds.',
              'No significance or causal benefit is claimed from three seeds and reused validation data.', '',
              '3. What the learned correction actually does',
              'The prediction equals the latest observed EPSS plus a small correction, clipped to [0,1].',
              'Model/seed | Distinct corrections | Minimum | Maximum | Largest range within the same source mask | Attention difference from uniform']
    for name, r in sorted(a['fits'].items()):
        c = r['corrections']
        lines.append(f"{name} | {c['distinct_values']} | {c['min']:.12g} | {c['max']:.12g} | {c['max_within_mask_group_range']:.12g} | {c['max_attention_difference_from_uniform_present_sources']:.12g}")
    lines += ['', 'Seven fits have one correction for every validation CVE. Two all-source fits have eight.',
              'Those two fits have zero correction variation within each source-availability pattern.',
              'All-source attention is uniform across available sources at stored float32 precision.',
              'Different average attention values across sources therefore reflect coverage, not learned importance.',
              'These are saved-output observations, not a counterfactual text-removal or causal test.', '',
              '4. Encoder weights and gradients',
              'Model/seed | GRU weight norm | Node projection weight norm | Selected-epoch GRU data-gradient sum | GNN data-gradient sum']
    for name, r in sorted(a['fits'].items()):
        p, g = r['parameter_norms'], r['selected_epoch_gradient_sums']
        lines.append(f"{name} | {p['temporal']['saved_l2']:.6g} | {p['node_project']['saved_l2']:.6g} | {g['temporal']:.6g} | {g['layers']:.6g}")
    lines += ['', 'The GRU and active node projections shrink close to zero. Data gradients at selected epochs are tiny.',
              'Unused graph weights in EPSS-only stay at initialization; they are not a functioning text branch.',
              'Weight decay, residual scaling and a mostly unchanged training target are plausible contributors.',
              'These observations do not isolate which mechanism caused collapse; optimizer settings are not changed here.', '',
              '5. Training and validation differ',
              'Split | CVEs | Same endpoint as latest input | Higher endpoint | Lower endpoint | Median absolute change | Partial input histories']
    for split, r in a['endpoint_changes_from_original_scores'].items():
        lines.append(f"{split} | {r['n']} | {r['unchanged']} | {r['increased']} | {r['decreased']} | {r['median_absolute_change']:.8g} | {r['partial_recent_histories']}")
    lines += ['', 'These change counts use original decimal scores; float32 conversion is not counted as a real change.',
              'The official performance table retains the original float32 persistence reference.',
              'An unchanged endpoint does not mean every score between Day 60 and Day 90 was constant.',
              'Most training CVEs favor persistence, whereas validation has far more endpoint changes.',
              'All validation histories are complete, so missing validation input days cannot explain this result.', '',
              '6. Largest observed validation changes',
              'CVE | Latest input EPSS | Actual Day-90 EPSS | Absolute difference']
    for r in a['largest_validation_changes']:
        lines.append(f"{r['cve']} | {r['last_observed_score']:.8g} | {r['target']:.8g} | {r['absolute_change']:.8g}")
    shares = a['endpoint_changes_from_original_scores']['validation']['top_error_shares']
    lines += [f"The largest 3/10/20 errors contribute {100*shares['3']:.2f}% / {100*shares['10']:.2f}% / {100*shares['20']:.2f}% of total raw persistence absolute error.",
              '', '7. Interpretation and next decision',
              'The multi-source pipeline completed, but useful text-content learning is not demonstrated.',
              'This is not evidence that TPGs or summaries are inherently useless, nor proof of a graph-extraction bug.',
              'Do not run another unchanged full comparison or open the test set to choose settings.',
              'A useful next bounded step is a training-only capacity check on changing-target CVEs, with matched EPSS-only/text models and persistence.',
              'Check that they can learn nonconstant corrections and inspect data gradients against regularization before freezing any new full protocol.',
              'Such a diagnostic subset must not replace the primary cohort or be presented as unbiased performance.',
              '', '8. Provenance and limits', a['verification_scope'],
              'Original run artifacts are read-only. Review outputs are separate. No new predictions, training or test scoring.']
    lines += a['limitations']
    return '\n'.join(lines)+'\n'


def figures(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for name, r in sorted(a['fits'].items()):
        axes[0].plot([h['epoch'] for h in r['history']], [h['validation_mae'] for h in r['history']], label=name)
    axes[0].axhline(a['persistence']['mae'], color='black', linestyle='--', label='Persistence')
    axes[0].set(xlabel='Completed training epoch', ylabel='Validation MAE (0-1 EPSS scale)', title='All nine validation curves (1,104 CVEs)')
    axes[0].legend(fontsize=6)
    kinds = ['unchanged', 'increased', 'decreased']
    for offset, split in [(-.2, 'train'), (.2, 'validation')]:
        r = a['endpoint_changes_from_original_scores'][split]
        axes[1].bar([j+offset for j in range(3)], [100*r[k]/r['n'] for k in kinds], width=.4, label=f"{split} (n={r['n']})")
    axes[1].set_xticks(range(3), ['Same endpoint', 'Higher endpoint', 'Lower endpoint'])
    axes[1].set(ylabel='CVEs in split (%)', title='Day-90 score compared with latest input score')
    axes[1].legend()
    fig.text(.5, .015, 'Left: smaller forecast error is better. Right: original scores, not float32 rounding; endpoint equality does not imply an unchanged path.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(OUTPUT/'learning_and_target_shift.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    run.torch.set_num_threads(1)
    a = analyze()
    if args.verify_only:
        run.check_files(OUTPUT, 'review_lock.json', a['schema'])
        run.require(run.read(OUTPUT/'analysis.json') == a and (OUTPUT/'analysis.txt').read_text() == report(a), 'Review differs')
    else:
        run.require(not OUTPUT.exists(), 'Review exists; use --verify-only')
        OUTPUT.mkdir(parents=True)
        run.write(OUTPUT/'analysis.json', a)
        (OUTPUT/'analysis.txt').write_text(report(a), encoding='utf-8')
        figures(a)
        run.write(OUTPUT/'review_lock.json', {'schema': a['schema'], 'files': {n: run.sha256(OUTPUT/n) for n in ('analysis.json', 'analysis.txt', 'learning_and_target_shift.png')}})
    run.cache.base.preparation.sources.check_pins(a['original_artifact_pins'])
    print(f'MS03 review {"reproduced" if args.verify_only else "saved"}: {OUTPUT}. No training, new predictions or test scoring.')


if __name__ == '__main__':
    main()
