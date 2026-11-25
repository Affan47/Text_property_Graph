"""Read-only PS04 result review: reproduce saved metrics, effects and concentration."""
from collections import defaultdict
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss.primary_study import common
from epss.primary_source_comparison import runner, reporting
from epss.primary_training import runner as original

SOURCE = runner.OUTPUT / 'full'
OUTPUT = common.RESULTS / 'source_comparison_review_v1'
read, require = common.read, common.require


def concentration(ids, actual, errors, reference):
    gain = reference - errors
    top = sorted(range(len(ids)), key=lambda i: (-gain[i], ids[i]))
    high = actual >= .1
    keep = np.ones(len(ids), dtype=bool); keep[top[0]] = False
    total_gain = float(gain.sum())
    return {'high_score_cves': int(high.sum()),
            'high_score_mae': float(errors[high].mean()),
            'high_score_fraction_of_absolute_error': float(errors[high].sum()/errors.sum()),
            'relative_mae_gain_over_median_percent': float(100*gain.mean()/reference.mean()),
            'better_than_median_cves': int((gain > 0).sum()), 'worse_than_median_cves': int((gain < 0).sum()),
            'largest_gain_cve': ids[top[0]], 'largest_gain_fraction_of_net_gain': float(gain[top[0]]/total_gain) if total_gain>0 else None,
            'mae_difference_without_largest_gain_cve_descriptive_only': float((errors-reference)[keep].mean()),
            'top_gain_cases': [{'cve': ids[i], 'actual': float(actual[i]), 'mean_seed_absolute_error': float(errors[i]),
                                'gain_over_median': float(gain[i])} for i in top[:10]],
            'interpretation': 'Descriptive influence check; no CVEs are removed. A net-gain fraction can exceed one when other cases offset the gain.'}


def run():
    identity = read(SOURCE/'run_identity.json')
    require(identity['scale']=='full' and not identity['test_scoring'], 'Wrong study scope')
    require({s:len(v) for s,v in identity['ids'].items()}=={'train':2546,'validation':1104}, 'Cohort changed')
    common.check_pins(identity['source_pins'])
    plan = runner.plan('full')
    require(plan==identity['comparison_plan'], 'Comparison plan changed')
    locations = {r['key']:runner.fit_directory(SOURCE,original.OUTPUT/'full',r) for r in plan}
    base = read(original.OUTPUT/'full/run_identity.json')
    histories, groups, predictions, epochs = [], defaultdict(list), {}, 0
    for item in common.tqdm(plan, desc='Verify completed reused/new fit artifacts', unit='fit'):
        path = locations[item['key']]
        common.verify_lock(path)
        expected = base if item['origin']=='ps02_reused' else identity
        require(read(path/'fit_identity.json')=={'run_digest':common.digest(expected),'variant':item['variant'],'seed':item['seed']},
                'Fit protocol mismatch')
        rows = read(path/'predictions.json'); predictions[item['key']] = rows
        groups[item['variant']['id']].append(item['key'])
        if item['seed'] is not None:
            history = read(path/'history.json'); diagnostic = read(path/'diagnostics.json')
            require([h['epoch'] for h in history]==list(range(1,len(history)+1)), 'Epoch sequence differs')
            best = min(history,key=lambda h:h['validation_mae'])
            require(best['epoch']==diagnostic['best_epoch'] and len(history)==diagnostic['epochs'], 'Selection/history mismatch')
            mae = np.mean([abs(r['actual']-r['predicted']) for r in rows])
            require(abs(mae-best['validation_mae'])<1e-7, 'Prediction/history mismatch')
            details = {'fit':item['key'],'origin':item['origin'],'epochs':len(history),'best_epoch':best['epoch'],
                       'early_stopped':diagnostic['early_stopped'],
                       'zero_gradient_batches':sum(h['zero_gradient_batches'] for h in history)}
            histories.append(details)
            if item['origin']=='ps04_new': epochs += len(history)
    common.verify_lock(SOURCE/'reports')
    summary = reporting.aggregate(locations, plan, identity)
    require(summary==read(SOURCE/'reports/analysis.json'), 'Saved matched report differs')
    ids = identity['ids']['validation']
    first = predictions[plan[0]['key']]
    y = np.array([r['actual'] for r in first])
    errors = {name: np.mean([[abs(r['actual']-r['predicted']) for r in predictions[k]] for k in keys],axis=0)
              for name,keys in groups.items()}
    reference = errors['P01_NONE_training_median']
    models = {}
    for name,details in summary['configurations'].items():
        if name.startswith('P01_NONE_'): continue
        category = []
        for f in details['per_fit']:
            category.append({'seed':f['seed'],**f['metrics']['threshold_categories']['0.1']})
        models[name] = {**details, 'error_concentration':concentration(ids,y,errors[name],reference),
                        'epss_at_least_0_1_by_seed':category}
    examples=[]
    # These same three diagnostic cases were already inspected in PS03.
    for cve in ('CVE-2025-0108','CVE-2024-48248','CVE-2025-1974'):
        i=ids.index(cve)
        for name in ('P01_DM_pooled_text_S1','P01_DM_security_tpg_S1','P01_DGNM_security_tpg_S1'):
            examples.append({'cve':cve,'target_date':first[i]['target_date'],'actual':float(y[i]),'configuration':name,
                **{f'seed_{seed}_prediction':predictions[name+f'__seed_{seed}'][i]['predicted'] for seed in (17,29,43)}})
    source_pairs = summary['paired_comparisons'][:12]
    require(all(p['ci95_lower']<=0<=p['ci95_upper'] for p in source_pairs),'Update interpretation: source/graph interval excludes zero')
    findings = [
        'All 27 fit artifact locks and saved matched metrics/paired intervals reproduce. Checkpoint inference was verified by the original user-run CLI, not rerun by this review.',
        'Description + SMP pooled text has the lowest three-seed mean MAE: 0.022487723, versus training median 0.023603505 (4.73% lower). All models retain CVSS and reference count.',
        'Adding SMP to description improves pooled-text MAE in all three seeds, but the paired 95% interval includes zero. TPG DM is worse than TPG D on average and in two of three seeds.',
        'Adding URL summaries to pooled-text DM worsens MAE in all three seeds. Adding the same summaries to TPG DM improves MAE in all three seeds. Neither paired interval excludes zero.',
        'All twelve source-addition/graph-versus-text intervals include zero. No reliable TPG advantage or general source benefit is established.',
        'Two unadjusted median-baseline intervals narrowly exclude zero: DM TPG and all-source pooled text. These are validation-conditional comparisons, not multiplicity-adjusted or independent final-test findings.',
        'The best pooled DM model still misses most scores >=0.1 at that fixed threshold: recalls are 10%, 8%, 4%. Approximately 95.5% accuracy is dominated by the 1,054 lower-score CVEs.',
        'For pooled DM, 50 high-score CVEs account for 85.60% of absolute error. CVE-2025-0108 accounts for 74.63% of net gain over the median reference; improvement excluding it remains small and positive descriptively.',
        'Recommendation: close this bounded development phase. Freeze the eight S1 configurations, three baselines, all three seeds, metrics and reporting rules before implementing one held-out test evaluation. Designate pooled DM as the validation-selected leader and all-source TPG as the lowest-MAE graph comparator; do not select a best seed.',
        'Do not claim security-edge benefit or causal SMP effects. More source/architecture searches are not required before an honest held-out assessment of the existing models.'
    ]
    result={'schema':'ps04_source_review_v1','source_report_lock_sha256':common.sha256(SOURCE/'reports/artifact_lock.json'),
        'review_script_sha256':common.sha256(__file__), 'fits':len(plan),'new_fits':12,'reused_fits':15,
        'new_training_epochs':epochs,'histories':histories,'models':models,
        'training_median_mae':float(reference.mean()),'validation_high_score_count':int((y>=.1).sum()),
        'paired_comparisons':summary['paired_comparisons'],'descriptive_interactions':summary['descriptive_interactions'],
        'examples':examples,'example_selection':'Three previously inspected PS03 cases; not a representative performance sample.',
        'findings':findings,'limitations':summary['limitations'],'training':False,'new_predictions':False,'test_scoring':False}
    lines=['PS04 full source-comparison interpretation','',*findings,'',
        'Configuration | Mean MAE | Mean RMSE | Mean R2 | Gain over median (%) | High-score share of absolute error (%)']
    lines += [f"{n} | {m['metrics']['mae']['mean']:.9f} | {m['metrics']['rmse']['mean']:.9f} | {m['metrics']['r2']['mean']:.6f} | "
              f"{m['error_concentration']['relative_mae_gain_over_median_percent']:.3f} | {100*m['error_concentration']['high_score_fraction_of_absolute_error']:.2f}"
              for n,m in models.items()]
    lines += ['', f'New fits completed {epochs} epochs; no zero-total-data-gradient batches: '+str(all(h['zero_gradient_batches']==0 for h in histories)),
              '', 'Limitations',*summary['limitations']]
    OUTPUT.mkdir(parents=True,exist_ok=True)
    common.write_report(OUTPUT,result,lines)
    with (OUTPUT/'real_prediction_examples.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(examples[0]));writer.writeheader();writer.writerows(examples)
    common.lock(OUTPUT)
    print(f'PS04 review saved to {OUTPUT}. Saved arithmetic reproduced; no retraining, new predictions or test scoring.')


if __name__=='__main__':
    run()
