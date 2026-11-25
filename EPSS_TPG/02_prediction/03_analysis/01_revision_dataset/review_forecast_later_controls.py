"""Review completed H02 artifacts without updates, new fits or test scoring."""
import argparse
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_later_controls as run

data = run.data
read, write, require, sha256 = run.read, run.write, run.require, run.sha256
OUTPUT = data.ROOT/'05_results/02_evaluation/epss_day60_to_day90/history_controls_review_v1'
FILES = {'analysis.json', 'analysis.txt', 'learning_curves.png', 'error_concentration.png'}


def error_summary(rows):
    require(bool(rows), 'Empty review group')
    errors = [abs(r['actual_day90_epss']-r['persistence']) for r in rows]
    return {'n':len(rows), 'same_endpoint_as_last_input':sum(e == 0 for e in errors),
            'increased':sum(r['actual_day90_epss'] > r['persistence'] for r in rows),
            'decreased':sum(r['actual_day90_epss'] < r['persistence'] for r in rows),
            'persistence_mae':statistics.mean(errors), 'median_absolute_error':statistics.median(errors),
            'total_absolute_error':sum(errors)}


def paired_predictions(a, b):
    require([r['cve'] for r in a] == [r['cve'] for r in b], 'Unpaired predictions')
    require([r['actual_day90_epss'] for r in a] == [r['actual_day90_epss'] for r in b], 'Different targets')
    return {'identical_predictions':sum(x['prediction'] == y['prediction'] for x,y in zip(a,b)),
            'n':len(a), 'max_absolute_prediction_difference':max(abs(x['prediction']-y['prediction']) for x,y in zip(a,b)),
            'max_absolute_correction_difference':max(abs(x['correction']-y['correction']) for x,y in zip(a,b))}


def norm(state, prefix):
    return math.sqrt(sum(float(v.double().square().sum()) for k,v in state.items() if k.startswith(prefix)))


def analyze():
    run.torch.set_num_threads(1)
    run.torch.use_deterministic_algorithms(True)
    a = data.load()
    original_pins = {}
    summaries = {}
    for scale in ('smoke','full'):
        identity,tensors = run.make_context(a,scale)
        if scale == 'full':
            p = run.OUTPUT/'smoke/evaluation_lock.json'
            identity['pinned_files'][str(p)] = sha256(p)
        directory = run.OUTPUT/scale
        run.verify_run(directory,identity,tensors,a)
        lock = read(directory/'evaluation_lock.json')
        original_pins.update(identity['pinned_files'])
        original_pins[str(directory/'evaluation_lock.json')] = sha256(directory/'evaluation_lock.json')
        original_pins.update({str(directory/n):h for n,h in lock['files'].items()})
        summaries[scale] = read(directory/'summary.json')
    # 'identity' and 'tensors' now refer to the full matched development run.
    ids = identity['ids']
    input_checks = {}
    for s in ids:
        recent,longer = tensors['recent_30_60'][s], tensors['long_0_60'][s]
        require(tuple(recent.shape) == (len(ids[s]),31,5) and tuple(longer.shape) == (len(ids[s]),61,5), 'Wrong tensor shapes')
        require(run.torch.equal(recent[:,:,[0,1,4]],longer[:,30:,[0,1,4]]), 'Shared observations/time differ')
        input_checks[s] = {'recent_shape':list(recent.shape), 'long_shape':list(longer.shape),
                           'distinct_storage':recent.data_ptr() != longer.data_ptr(),
                           'cves_with_extra_older_observations':int((longer[:,:30,1].sum(dim=1) > 0).sum())}
    endpoint_counts, cases = {}, {}
    for s,chosen in ids.items():
        rows = []
        for c in chosen:
            x = a['inputs.json'][c]
            rows.append({'cve':c, 'persistence':run.persistence(x),
                         'actual_day90_epss':a['labels.json'][c]['y']['endpoint'],
                         'anchor_date':x['input_end_date'], 'target_date':a['labels.json'][c]['target_date'],
                         'recent_observed_days':sum(x['histories']['recent_30_60']['observed_mask']),
                         'long_observed_days':sum(x['histories']['long_0_60']['observed_mask'])})
        endpoint_counts[s] = error_summary(rows)
        cases[s] = rows
    total = endpoint_counts['validation']['total_absolute_error']
    top = sorted(cases['validation'],key=lambda r:(-abs(r['actual_day90_epss']-r['persistence']),r['cve']))[:10]
    top = [{**r, 'absolute_error':abs(r['actual_day90_epss']-r['persistence']),
            'share_of_total_absolute_error':abs(r['actual_day90_epss']-r['persistence'])/total if total else 0.}
           for r in top]
    rows_by_fit, fits, pairs = {}, {}, {}
    for seed in run.tqdm(identity['config']['seeds'],desc='Review matched seeds',unit='seed'):
        states = {}
        for variant in data.VARIANTS:
            name = f'{variant}__{seed}'
            directory = run.OUTPUT/'full/fits'/name
            saved = run.torch.load(directory/'checkpoint.pt',weights_only=True,map_location='cpu')
            states[variant] = saved['state_dict']
            predictions = read(directory/'predictions.json')
            rows_by_fit[name] = predictions
            h = read(directory/'history.json')
            correction = [r['correction'] for r in predictions]
            initial = run.initialize(identity['config'],identity['gain'],seed).state_dict()
            fits[name] = {'metrics':read(directory/'metrics.json'), 'history':h,
                          'correction_min':min(correction), 'correction_max':max(correction),
                          'max_absolute_correction':max(map(abs,correction)),
                          'distinct_corrections':len(set(correction)),
                          'training_clips_all_epochs':sum(r['clipped_below_zero']+r['clipped_above_one'] for r in h),
                          'epochs_with_nonzero_temporal_data_gradients':sum(r['temporal_data_gradient_abs_sum'] > 0 for r in h),
                          'initial_temporal_parameter_l2':norm(initial,'temporal.'),
                          'selected_temporal_parameter_l2':norm(saved['state_dict'],'temporal.'),
                          'state_changed_from_initial':run.state_digest(initial) != run.state_digest(saved['state_dict'])}
        first,second = (rows_by_fit[f'{v}__{seed}'] for v in data.VARIANTS)
        pairs[str(seed)] = {**paired_predictions(first,second),
                           'identical_selected_weights':run.state_digest(states[data.VARIANTS[0]]) == run.state_digest(states[data.VARIANTS[1]]),
                           'max_selected_weight_difference':max(float((states[data.VARIANTS[0]][k]-states[data.VARIANTS[1]][k]).abs().max()) for k in states[data.VARIANTS[0]])}
    groups = {}
    for field,length in (('recent_observed_days',31),('long_observed_days',61)):
        for complete in (True,False):
            selected = [r for r in cases['validation'] if (r[field] == length) == complete]
            if not selected:
                continue
            chosen = {r['cve'] for r in selected}
            metrics = {}
            for name,predictions in rows_by_fit.items():
                subset = [r for r in predictions if r['cve'] in chosen]
                metrics[name] = run.baseline.regression_metrics([r['actual_day90_epss'] for r in subset], [r['prediction'] for r in subset])
            groups[field+('_complete' if complete else '_partial')] = {'persistence':error_summary(selected),'fits':metrics}
    result = {'schema':'h02_results_review_v1', 'review_script_sha256':sha256(Path(__file__)),
              'original_pins':original_pins, 'training':False, 'test_scoring':False,
              'summaries':summaries, 'input_checks':input_checks, 'endpoint_counts':endpoint_counts,
              'paired':pairs, 'fits':fits, 'top_validation_cases':top,
              'top_two_error_share':sum(r['share_of_total_absolute_error'] for r in top[:2]),
              'validation_missingness_groups':groups,
              'source_coverage':a['source_coverage.json'],
              'limitations':identity['protocol']['limitations']+[
                  'Matching endpoints do not establish a flat intervening EPSS trajectory.',
                  'Very small corrections and shrinking weights are descriptive; this review does not isolate a causal optimizer mechanism.',
                  'No useful longer-history effect in these models does not establish that older history contains no predictive information.',
                  'High rank correlation can coexist with large errors on the few changing CVEs.',
                  'Outlier contributions are retained; no CVEs or errors are removed from the primary results.']}
    data.h01.sources.check_pins(original_pins)
    return result


def report(a):
    s = a['summaries']['full']
    lines = ['H02 completed results: recent versus longer EPSS history', '',
             'All smoke/full checkpoint predictions were reproduced. No retraining or test scoring.',
             '1029 training / 173 validation CVEs; Day-60 anchor, observed Day-90 target.', '',
             '1. Does longer history improve prediction?',
             'No improvement was demonstrated in these fixed models. Every seed has identical recent/long validation predictions.',
             f"Persistence MAE={s['persistence']['mae']:.12g}; RMSE={s['persistence']['rmse']:.12g}.",
             'History/seed | Selected epoch | Epochs run | Validation MAE | RMSE | MAE minus persistence']
    for name,r in s['fits'].items():
        lines.append(f"{name} | {r['selected_epoch']} | {r['epochs_run']} | {r['metrics']['mae']:.12g} | "
                     f"{r['metrics']['rmse']:.12g} | {r['metrics']['mae']-s['persistence']['mae']:+.12g}")
    lines += ['', 'Differences from persistence are tiny; they are not a meaningful performance gain.',
              'The smoke used a different small cohort and is only an execution check. Its MAE is not compared with full-run MAE.', '',
              '2. Why can persistence be difficult to improve?',
              'Split | CVEs | Day-90 equals last observed input | Day-90 higher | Day-90 lower | Persistence MAE']
    for split,r in a['endpoint_counts'].items():
        lines.append(f"{split} | {r['n']} | {r['same_endpoint_as_last_input']} | {r['increased']} | {r['decreased']} | {r['persistence_mae']:.12g}")
    lines += ['Equality compares two endpoint values, not all days between them.',
              'With MAE, an unnecessary correction on many unchanged endpoints can outweigh gains on a few changed ones.',
              'This explains the strength of the reference; it does not prove that collapse was unavoidable.', '',
              '3. Where does the validation error come from?',
              'CVE | Last observed input | Actual Day-90 EPSS | Absolute persistence error | Share of total absolute error']
    for r in a['top_validation_cases'][:5]:
        lines.append(f"{r['cve']} | {r['persistence']} | {r['actual_day90_epss']} | {r['absolute_error']:.8g} | {100*r['share_of_total_absolute_error']:.2f}%")
    lines += [f"The two largest errors account for {100*a['top_two_error_share']:.2f}% of total absolute validation error.",
              'These CVEs remain in every reported metric. High rank correlation does not mean their increases were forecast.', '',
              '4. Did the models receive different histories and learn anything?',
              f"Input checks: {a['input_checks']}",
              'Actual loaders produced separate 31-day and 61-day tensors with common recent scores/masks/time coordinates.',
              'History/seed | Distinct corrections | Minimum correction | Maximum correction | Initial temporal weight L2 | Selected temporal weight L2']
    for name,r in a['fits'].items():
        lines.append(f"{name} | {r['distinct_corrections']} | {r['correction_min']:.10g} | {r['correction_max']:.10g} | "
                     f"{r['initial_temporal_parameter_l2']:.8g} | {r['selected_temporal_parameter_l2']:.8g}")
    lines += [f"Paired checkpoint comparisons: {a['paired']}",
              'Weights changed and temporal data gradients were recorded. This is not evidence of an entirely disconnected temporal branch.',
              'Nevertheless, seeds 17/29 give constant corrections and seed 43 gives negligible input-dependent corrections.',
              'Some corrections differ below the resolution of the final float32 sum, so final predictions can be identical.',
              'Recorded shrinking parameters and gradients are consistent with the near-persistence solution; they do not establish weight decay as its sole cause.', '',
              '5. Missing-input sensitivity on identical validation subsets',
              'Subset | CVEs | Persistence MAE | Recent mean MAE across seeds | Longer mean MAE across seeds']
    for name,g in a['validation_missingness_groups'].items():
        means = [statistics.mean(r['mae'] for k,r in g['fits'].items() if k.startswith(v)) for v in data.VARIANTS]
        lines.append(f"{name} | {g['persistence']['n']} | {g['persistence']['persistence_mae']:.10g} | {means[0]:.10g} | {means[1]:.10g}")
    lines += ['Complete means every input day in that named window has a score; partial means at least one is missing.',
              'These subsets differ in CVEs and error concentration. Their MAEs do not isolate an effect of missingness.', '',
              '6. Source coverage for the next stage',
              'Split | Source | CVEs with text | CVEs passing candidate-view rules']
    for r in a['source_coverage']['coverage']:
        lines.append(f"{r['split']} | {r['source']} | {r['cves_with_text']} | {r['cves_in_date_screened_view']}")
    lines += ['These sources were NOT used in H02. Source screening is not historical authentication.', '',
              '7. Decision',
              'Keep persistence as the reference. Longer history has not earned a performance advantage.',
              'Do not tune another EPSS-only loop or remove the difficult cases based on these results.',
              'Next implement a small source-aware TPG graph/cache preview using the fixed Day-60 source inputs and masks.',
              'Then freeze matched fusion/source-removal comparisons on these same CVEs, with an EPSS-only control.',
              'Recent history is the simpler provisional control; choosing it is a simplicity decision, not a statistically established win.',
              'Retrospective text comparisons and historically verified forecasting must remain separate. Final test stays closed.', '',
              '8. Figures',
              'learning_curves.png: x is epoch; y is validation MAE minus persistence MAE. Zero equals persistence; negative would be better.',
              'error_concentration.png: x is absolute Day-90 persistence error in EPSS score units; y names the largest-error CVEs. Percentage labels give each share of total error.', '',
              '9. Limitations']+['- '+v for v in a['limitations']]
    return '\n'.join(lines)+'\n'


def figures(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,3,figsize=(12,3.8),sharey=True)
    base = a['summaries']['full']['persistence']['mae']
    for ax,seed in zip(axes,(17,29,43)):
        for v,style in zip(data.VARIANTS,('-', '--')):
            h = a['fits'][f'{v}__{seed}']['history']
            ax.plot([r['epoch'] for r in h],[r['validation_mae']-base for r in h],style,label=v)
        ax.axhline(0,color='black',linewidth=.7)
        ax.set(title=f'Seed {seed}',xlabel='Epoch')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Validation MAE minus persistence MAE')
    fig.suptitle('H02 validation learning curves: 173 matched CVEs')
    fig.tight_layout()
    fig.savefig(OUTPUT/'learning_curves.png',dpi=180)
    plt.close(fig)
    rows = a['top_validation_cases'][:5]
    fig,ax = plt.subplots(figsize=(9,4))
    ax.barh([r['cve'] for r in rows],[r['absolute_error'] for r in rows],color='#327ba8')
    ax.invert_yaxis()
    for i,r in enumerate(rows):
        ax.text(r['absolute_error']+.01,i,f"{100*r['share_of_total_absolute_error']:.2f}%",va='center')
    ax.set_xlim(0,max(r['absolute_error'] for r in rows)*1.22)
    ax.set(xlabel='Absolute Day-90 persistence error (EPSS score units)',ylabel='CVE',
           title='Largest errors among 173 validation CVEs; labels show share of total error')
    fig.tight_layout()
    fig.savefig(OUTPUT/'error_concentration.png',dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    args = parser.parse_args()
    require(args.verify_only or not OUTPUT.exists(),'Review exists; use --verify-only')
    a = analyze()
    text = report(a)
    if args.verify_only:
        lock = read(OUTPUT/'review_lock.json')
        require(set(lock['files']) == FILES,'Incomplete review')
        for n,h in lock['files'].items():
            require(sha256(OUTPUT/n) == h,'Changed review: '+n)
        require(read(OUTPUT/'analysis.json') == a and (OUTPUT/'analysis.txt').read_text() == text,'Review differs')
        print('H02 results review reproduced. No training or test scoring.')
        return
    OUTPUT.mkdir(parents=True,exist_ok=False)
    write(OUTPUT/'analysis.json',a)
    (OUTPUT/'analysis.txt').write_text(text,encoding='utf-8')
    figures(a)
    write(OUTPUT/'review_lock.json',{'files':{n:sha256(OUTPUT/n) for n in sorted(FILES)}})
    print(f'H02 results review saved to {OUTPUT}. Original runs unchanged; no training or test scoring.')


if __name__ == '__main__':
    main()
