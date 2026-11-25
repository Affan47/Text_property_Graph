"""Review locked R07 development results without fitting or new predictions."""
import argparse
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_mae_development as run

SOURCE = run.OUTPUT
OUTPUT = SOURCE.with_name(SOURCE.name + '_review')


def errors(actual, prediction):
    run.require(len(actual) == len(prediction) > 0, 'Prediction length differs')
    run.require(all(math.isfinite(v) for v in actual + prediction), 'Non-finite value')
    absolute = [abs(a-p) for a,p in zip(actual,prediction)]
    return {'mae': statistics.mean(absolute),
            'rmse': math.sqrt(statistics.mean(e*e for e in absolute)),
            'absolute': absolute}


def directions(actual, base):
    errors(actual, base)
    return {'unchanged': sum(a == b for a,b in zip(actual,base)),
            'increased': sum(a > b for a,b in zip(actual,base)),
            'decreased': sum(a < b for a,b in zip(actual,base))}


def near(a, b, message):
    run.require(math.isclose(a, b, abs_tol=1e-14, rel_tol=1e-12), message)


def analyze():
    lock = run.read(SOURCE/'evaluation_lock.json')
    identity = run.read(SOURCE/'run_identity.json')
    p = identity['protocol']
    run.require(p == run.protocol() and set(lock['files']) == run.expected_files(p),
                'Wrong protocol or incomplete outputs')
    hashes = {str(SOURCE/n): h for n,h in lock['files'].items()}
    run.verify_hashes(hashes, 'Verify R07 result files')
    run.verify_hashes(identity['pinned_files'], 'Verify frozen inputs and prior runs')
    s = run.read(SOURCE/'summary.json')
    ids = identity['ids']
    run.require(set(ids) == {'train','validation'} and
                {k:len(v) for k,v in ids.items()} == {'train':1023,'validation':197},
                'Unexpected development cohorts')
    selected = ids['validation']
    names = ('inputs.json','labels.json','split_manifest.json','task_cohorts.json')
    data = run.prior.smoke.select_data(
        {'ids':ids}, {n:run.read(run.prior.cache.base.PREPARED/n) for n in names})
    inputs,labels = data['inputs.json'],data['labels.json']
    actual = [labels[c]['y']['endpoint'] for c in selected]
    base = [run.task.predict_persistence(inputs[c])['prediction'] for c in selected]
    refs = run.r06.r05.baselines({'ids':ids}, data)
    run.require(refs == s['baselines'], 'Baselines differ')
    base_metrics = errors(actual, base)
    near(base_metrics['mae'],refs['persistence']['metrics']['mae'], 'Baseline MAE differs')
    total_error = sum(base_metrics['absolute'])
    fits = {}
    for name,m in run.tqdm(sorted(s['fits'].items()), desc='Review saved fits', unit='fit'):
        directory = SOURCE/'fits'/name
        pred = run.read(directory/'predictions.json')
        history = run.read(directory/'history.json')
        gradients = run.read(directory/'gradients.json')
        run.require(all(len(v) == len(selected) for v in pred.values()), 'Prediction arrays differ')
        value = errors(actual,pred['prediction'])
        near(value['mae'],m['metrics']['mae'], 'MAE differs')
        near(value['rmse'],m['metrics']['rmse'], 'RMSE differs')
        near(statistics.pstdev(pred['correction']),m['correction_std'], 'Correction SD differs')
        run.history_check(history,p['config'],m['selected_epoch'])
        run.require(len(history) == m['epochs_run'], 'Epoch count differs')
        initial, final = m['initial_parameter_norms'],m['selected_parameter_norms']
        float_base = errors(actual,pred['persistence_float32'])
        base_errors = base_metrics['absolute']
        fits[name] = {
            'metrics':m, 'history':history, 'gradients':gradients,
            'mae_excess_over_persistence':value['mae']-base_metrics['mae'],
            'float32_persistence_mae':float_base['mae'],
            'mae_excess_over_float32_persistence':value['mae']-float_base['mae'],
            'improved_cases':sum(a < b for a,b in zip(value['absolute'],base_errors)),
            'worsened_cases':sum(a > b for a,b in zip(value['absolute'],base_errors)),
            'equal_error_cases':sum(a == b for a,b in zip(value['absolute'],base_errors)),
            'distinct_corrections':len(set(pred['correction'])),
            'correction_span':max(pred['correction'])-min(pred['correction']),
            'selected_to_initial_norm_ratios':{
                k:final[k]/v if v else None for k,v in initial.items()},
            'training_clipping_count':sum(h['training_clipped_below_zero']+
                                         h['training_clipped_above_one'] for h in history),
            'selected_training_mae':history[m['selected_epoch']-1]['training_mae'],
            'last_training_mae':history[-1]['training_mae'],
            'validation_cases':[
                {'cve':c, 'input_end_date':inputs[c]['input_end_date'],
                 'target_date':labels[c]['target_date'],
                 'observed_input_days':sum(inputs[c]['X']['observed_mask']),
                 'actual_day60':y, 'persistence':b, 'prediction':v,
                 'correction':d, 'absolute_error':e}
                for c,y,b,v,d,e in zip(selected,actual,base,pred['prediction'],
                                       pred['correction'],value['absolute'])],
        }
    run.require(s == run.summarize(
        {k:v['metrics'] for k,v in fits.items()},refs,identity), 'Summary differs')
    order = sorted(range(len(selected)), key=lambda i:(-base_metrics['absolute'][i],selected[i]))
    top = [
        {'cve':selected[i], 'persistence':base[i], 'actual_day60':actual[i],
         'absolute_error':base_metrics['absolute'][i],
         'fraction_of_total_absolute_error':base_metrics['absolute'][i]/total_error if total_error else 0,
         'model_predictions':{k:v['validation_cases'][i]['prediction'] for k,v in fits.items()}}
        for i in order[:10]]
    train_actual = [labels[c]['y']['endpoint'] for c in ids['train']]
    train_base = [run.task.predict_persistence(inputs[c])['prediction'] for c in ids['train']]
    missingness = []
    for group,b in refs['persistence']['missingness'].items():
        missingness.append({'input_history':group,'count':b['n'],
                           'persistence_mae':b['metrics']['mae'],
                           'models':{model:statistics.mean(
                               fits[run.fit_name(model,seed)]['metrics']['missingness'][group]['metrics']['mae']
                               for seed in p['config']['seeds']) for model in run.MODELS}})
    paired_graph = [
        {'seed':seed,
         'graph_minus_epss_only_mae':fits[run.fit_name('no_coref',seed)]['metrics']['metrics']['mae']-
                                   fits[run.fit_name('epss_only',seed)]['metrics']['metrics']['mae']}
        for seed in p['config']['seeds']]
    old_mean = statistics.mean(v['metrics']['mae'] for v in identity['r04_references'].values())
    new_mean = statistics.mean(v['metrics']['metrics']['mae'] for v in fits.values())
    result = {
        'schema':'r07_full_mae_review_v1',
        'source_lock_sha256':run.sha256(SOURCE/'evaluation_lock.json'),
        'script_sha256':run.sha256(Path(__file__)),
        'verified_artifact_count':len(lock['files']),
        'training_performed':False,'new_predictions':False,'test_scoring':False,
        'cohort_sizes':{k:len(v) for k,v in ids.items()},
        'baseline_metrics':refs['persistence']['metrics'],
        'scalar_offset':refs['scalar_fit']['offset'],
        'training_persistence_metrics':{k:v for k,v in errors(train_actual,train_base).items() if k!='absolute'},
        'training_endpoint_changes':directions(train_actual,train_base),
        'validation_endpoint_changes':directions(actual,base),
        'fits':fits,'aggregates':s['aggregates'],'r04_paired':s['paired'],
        'graph_minus_epss_only':paired_graph,
        'r04_macro_mae':old_mean,'r07_macro_mae':new_mean,
        'relative_macro_mae_reduction_vs_r04':1-new_mean/old_mean,
        'missingness':missingness,'largest_error_cases':top,
        'top_two_fraction_of_total_absolute_error':sum(x['fraction_of_total_absolute_error'] for x in top[:2]),
        'all_fits_worse_than_persistence':all(v['mae_excess_over_persistence']>0 for v in fits.values()),
        'all_fits_worse_than_float32_persistence':all(v['mae_excess_over_float32_persistence']>0 for v in fits.values()),
        'limitations':s['limitations']+[
            'Constant corrections describe observed validation outputs, not a proof of invariance on every possible input.',
            'Saved gradients aggregate all epochs; they do not locate exactly when encoder gradients became zero.',
            'Parameter shrinkage is observed; this review alone does not establish its cause.',
            'Missing-history groups differ in CVEs; their errors do not estimate the causal effect of missing scores.',
            'EPSS score prediction is not observed exploitation prediction or VulnCheck catalogue prediction.',
        ],
    }
    run.verify_hashes(hashes, 'Confirm original R07 outputs unchanged')
    return result


def report(a):
    lines = [
        'R07: Full eligible-cohort MAE results', '',
        '1. Main finding',
        'The six fits completed correctly, but none beats retaining the latest observed EPSS score.',
        'TPG+EPSS does not show a useful improvement over EPSS-only on these validation CVEs.',
        'MAE removes most of the harmful corrections seen in R04; it does not forecast the large changes.',
        '', '2. Scope and verification',
        f"{a['verified_artifact_count']} locked files; 1023 training and 197 validation CVEs; seeds 17, 29, 43.",
        'The separate R07 verifier reloaded all six checkpoints and reproduced validation predictions.',
        'This review recomputes saved errors, early stopping, references and case-level summaries.',
        'No training, new model forecasts, new embeddings, split changes or test scoring.',
        '', '3. Results',
        'MAE is average absolute score error (lower is better); scores are on the 0-1 scale.',
        'RMSE gives more weight to large errors. Seed SD is run-to-run spread, not a confidence interval.',
        'Model | Mean validation MAE over three seeds | Sample SD',
    ]
    for row in a['aggregates']:
        lines.append(f"{row['model']} | {row['mean_mae']:.11f} | {row['sample_std_mae']:.11f}")
    lines += [
        f"Persistence MAE={a['baseline_metrics']['mae']:.11f}; RMSE={a['baseline_metrics']['rmse']:.11f}.",
        f"Training-fitted scalar correction={a['scalar_offset']}; it therefore equals persistence.",
        f"All fits worse than persistence: {a['all_fits_worse_than_persistence']}.",
        f"All fits also worse than float32 persistence: {a['all_fits_worse_than_float32_persistence']}.",
        'The float32 check separates learned correction errors from baseline numerical representation.',
        '', 'Model/seed | Selected epoch | Epochs run | Validation MAE | RMSE | Mean correction',
    ]
    for name,row in a['fits'].items():
        m = row['metrics']
        lines.append(f"{name} | {m['selected_epoch']} | {m['epochs_run']} | {m['metrics']['mae']:.11f} | "
                     f"{m['metrics']['rmse']:.11f} | {m['correction_mean']:.5g}")
    lines += [
        '', '4. Comparison with full R04',
        f"Mean MAE across both models/seeds: R04={a['r04_macro_mae']:.11f}, R07={a['r07_macro_mae']:.11f}.",
        f"Relative reduction={100*a['relative_macro_mae_reduction_vs_r04']:.3f}%. All six paired MAEs improve.",
        'The cohort, seeds, initialization, schedule, optimizer and architecture are matched.',
        'Loss differs; selected epochs differ as a consequence. This is not a same-step weight comparison.',
        'R06 used a smaller cohort; do not interpret its different headline MAE as a full-cohort improvement.',
        'Graph minus EPSS-only MAE at each seed (negative favors graph):',
    ]
    for row in a['graph_minus_epss_only']:
        lines.append(f"Seed {row['seed']}: {row['graph_minus_epss_only_mae']:+.6g}")
    lines += ['', '5. What the models learned',
              'Five fits have identical corrections on all validation CVEs; the sixth is almost constant.',
              'Predictions still vary because they inherit each CVE last observed score.',
              'The learned text/history correction has no practically useful CVE-specific variation here.',
              'This is evidence about these fitted models and this task, not a general verdict on TPG.',
              'Model/seed | Distinct corrections | Correction range width | Temporal norm retained | Graph projector norm retained']
    for name,row in a['fits'].items():
        ratios = row['selected_to_initial_norm_ratios']
        graph = f"{ratios['node_project']:.4g}" if 'node_project' in ratios else 'not applicable'
        lines.append(f"{name} | {row['distinct_corrections']} | {row['correction_span']:.4g} | "
                     f"{ratios['temporal']:.4g} | {graph}")
    lines += ['Norm retained means selected checkpoint norm divided by its initialization norm.',
              'Encoder and first head-layer weights shrink strongly. Graph-layer aggregate norms can retain',
              'other parameters, so a nonzero aggregate does not establish effective text use.',
              'All fits stop by the fixed patience rule, before the 100-epoch cap.',
              'Gradient counts below cover the entire training run, including epochs after the selected checkpoint.',
              'Model/seed | Branch | Steps with nonzero data-loss gradient / total optimizer steps']
    for name,row in a['fits'].items():
        for branch,g in row['gradients']['all_steps'].items():
            lines.append(f"{name} | {branch} | {g['steps_with_nonzero_gradient']}/{row['metrics']['optimizer_steps']}")
    lines += [
        'A gradient existing at some steps is not proof that the final model uses that branch effectively.',
        f"Recorded training clipping count over all fits={sum(v['training_clipping_count'] for v in a['fits'].values())}.",
        f"Selected validation clipping count={sum(v['metrics']['clipping']['below_zero']+v['metrics']['clipping']['above_one'] for v in a['fits'].values())}.",
        '', '6. Stable scores and large changes',
        f"Training endpoint changes versus last observed input: {a['training_endpoint_changes']}.",
        f"Validation endpoint changes versus last observed input: {a['validation_endpoint_changes']}.",
        f"Training persistence MAE={a['training_persistence_metrics']['mae']:.11f}.",
        'Unchanged means equal endpoint values, not necessarily a flat intervening EPSS trajectory.',
        'MAE can favor small corrections when many endpoints are unchanged. That alone does not prove',
        'the remaining changes are unpredictable from other features or under a different objective.',
        f"The two largest errors contribute {100*a['top_two_fraction_of_total_absolute_error']:.2f}% of total baseline absolute error.",
        'CVE | Last observed input score | Actual Day-60 score | Share of baseline absolute error | Model prediction range',
    ]
    for row in a['largest_error_cases'][:5]:
        values = list(row['model_predictions'].values())
        lines.append(f"{row['cve']} | {row['persistence']:.5f} | {row['actual_day60']:.5f} | "
                     f"{100*row['fraction_of_total_absolute_error']:.2f}% | {min(values):.9f} to {max(values):.9f}")
    lines += ['', '7. Missing input histories',
              'Rows group the same validation CVEs by how many of Days 0-30 have observed scores.',
              'Every model is compared with persistence on identical CVEs in each row.',
              'Observed input scores | CVEs | Persistence MAE | EPSS-only mean MAE | TPG+EPSS mean MAE']
    for row in a['missingness']:
        lines.append(f"{row['input_history']} | {row['count']} | {row['persistence_mae']:.11f} | "
                     f"{row['models']['epss_only']:.11f} | {row['models']['no_coref']:.11f}")
    lines += [
        'Partial histories are included; do not discard them based on this outcome comparison.',
        '', '8. Recommendation',
        'The planned full training is finished. Do not extend epochs or reopen bounded tuning to obtain a win.',
        'Record persistence as the best validation-MAE reference. R07 is a negative result for added learned correction.',
        'Freeze a final test protocol that reports persistence, the scalar reference and all six learned fits',
        'on identical eligible test CVEs, including paired errors and missingness groups. Do not choose a best seed.',
        'Prepare any required test graph cache only under that frozen evaluation protocol.',
        'Final test evaluation must be separately authorized; it was not performed here.',
        'Any later redesign of objectives, cohorts, features or regularization is a new study, not a repair of these results.',
        '', '9. Limitations',
    ] + ['- '+v for v in a['limitations']]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    run.require(args.verify_only or not OUTPUT.exists(), 'Review exists; use --verify-only')
    result = analyze()
    text = report(result)
    if args.verify_only:
        run.require(run.read(OUTPUT/'analysis.json') == result and
                    (OUTPUT/'analysis.txt').read_text() == text, 'Review differs')
        print('R07 review reproduced. No training, new forecasts or test scoring.')
    else:
        OUTPUT.mkdir(parents=True, exist_ok=False)
        run.write(OUTPUT/'analysis.json', result)
        (OUTPUT/'analysis.txt').write_text(text, encoding='utf-8')
        print(f'R07 review saved to {OUTPUT}. No training, new forecasts or test scoring.')


if __name__ == '__main__':
    main()
