"""Verify and explain a completed residual run, without fitting or test scoring."""
import argparse
import csv
import statistics
from pathlib import Path

import numpy as np
import torch

from review_forecast_sequence import PROJECT, PACKAGE, check_lock, close, metrics, verify_history
from epss.forecast_data import read_json, require, sha256, write_json
from epss.forecast_residual import (CONFIG, HEAD_RULE, OUTPUT_FILES, ResidualGRU,
                                    evaluate, input_tensor, predict, preflight)


def review(run_dir, prepared):
    lock = check_lock(run_dir)
    require(set(lock['files']) == OUTPUT_FILES-{'evaluation_lock.json'}, 'Incomplete residual run')
    summary = read_json(run_dir/'summary.json')
    saved = read_json(run_dir/'models.json')
    history = read_json(run_dir/'training_history.json')
    initial = read_json(run_dir/'initialization.json')
    rows = read_json(run_dir/'predictions.json')
    provenance = summary['provenance']
    require(provenance == saved['provenance'], 'Model/summary provenance differs')
    reference_dir = Path(provenance['reference_directory'])
    reference_hash = sha256(reference_dir/'evaluation_lock.json')
    require(reference_hash == lock['reference_lock_sha256'] == provenance['reference_lock_sha256'],
            'Reference lock changed')
    data, reference, preparation_hash = preflight(prepared, reference_dir)
    require(preparation_hash == lock['preparation_lock_sha256'] == provenance['preparation_lock_sha256'],
            'Preparation lock differs')
    require(provenance['source_sha256'] == data['preparation_summary.json']['source_sha256'], 'Source mismatch')
    require(saved['config'] == summary['config'] == CONFIG and saved['head_rule'] == HEAD_RULE, 'Configuration mismatch')
    for name, digest in provenance['code_sha256'].items():
        require(Path(name).name == name and sha256(PACKAGE/'epss'/name) == digest, 'Changed training implementation')
    predicted_rows, recomputed = evaluate(data, saved['models'], reference)
    require(predicted_rows == rows, 'Saved predictions do not match models or reference')
    require(recomputed == {k:v for k,v in summary.items() if k!='provenance'}, 'Recomputed summary differs')
    with (run_dir/'predictions.csv').open() as handle:
        table = list(csv.DictReader(handle))
    require(len(table)==len(rows) and all(all(a[k]==str(v) for k,v in b.items()) for a,b in zip(table,rows)),
            'CSV/JSON mismatch')
    require([s['seed'] for s in initial]==CONFIG['seeds'], 'Missing initialization diagnostics')
    ids = data['task_cohorts.json']['endpoint']
    train_records = [data['inputs.json'][c] for c in ids['train']]
    val_records = [data['inputs.json'][c] for c in ids['validation']]
    train_y = torch.tensor([data['labels.json'][c]['y']['endpoint'] for c in ids['train']],dtype=torch.float32).tolist()
    actual = [r['actual_day60_epss'] for r in rows]
    base = np.asarray([r['persistence_prediction'] for r in rows])
    base_errors = np.abs(base-actual)
    exact = base_errors==0
    per_seed = []
    for model, init in zip(saved['models'], initial):
        seed=model['seed']
        curve=verify_history(history[str(seed)],model['selected_epoch'],CONFIG)
        details=predict(model,val_records)
        training=predict(model,train_records)
        measured=metrics(actual,details['prediction'])
        original=next(s for s in summary['per_seed'] if s['seed']==seed)
        for key,value in measured.items():
            close(value,original['metrics'][key],'Independent metric mismatch')
        selected=history[str(seed)][model['selected_epoch']-1]
        close(metrics(train_y,training['prediction'])['mae'],selected['training_mae'],'Training MAE mismatch')
        close(measured['mae'],selected['validation_mae'],'Selected validation MAE mismatch')
        for split,values in [('training',training),('validation',details)]:
            count=sum(v<0 or v>1 for v in values['raw_prediction'])
            require(count==selected[split+'_clipped_count'],'Selected clipping count mismatch')
        close(float(torch.tensor(details['learned_correction']).abs().mean()),
              selected['validation_mean_absolute_correction'],'Correction magnitude mismatch')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            net=ResidualGRU()
            with torch.no_grad():
                baseline,correction,raw,pred=net.details(input_tensor(val_records))
        require(torch.equal(baseline,pred) and torch.count_nonzero(correction).item()==0,
                'Epoch-zero model is not persistence')
        require(init['epoch']==0 and init['eligible_for_selection'] is False and init['zero_correction_verified'] is True,
                'Invalid epoch-zero selection status')
        for key,value in metrics(actual,pred.tolist()).items():
            close(value,init['validation_metrics'][key],'Initialization metric mismatch')
        errors=np.abs(np.asarray(details['prediction'])-actual)
        corrections=np.asarray(details['learned_correction'])
        worst=sorted(range(len(rows)),key=lambda i:(-errors[i],rows[i]['cve']))[:5]
        per_seed.append({'seed':seed,**curve,'metrics':measured,
            'mae_percent_higher_than_persistence':100*(measured['mae']/summary['persistence_metrics']['mae']-1)
                if summary['persistence_metrics']['mae'] else None,
            'direct_gru_mae':original['direct_gru_metrics_same_seed']['mae'],
            'mean_absolute_correction':float(abs(corrections).mean()),
            'maximum_absolute_correction':float(abs(corrections).max()),
            'mean_signed_correction':float(corrections.mean()),
            'lower_error_cves':int((errors<base_errors).sum()),'equal_error_cves':int((errors==base_errors).sum()),
            'higher_error_cves':int((errors>base_errors).sum()),
            'mae_on_persistence_exact_cves':float(errors[exact].mean()) if exact.any() else None,
            'mae_on_other_cves':float(errors[~exact].mean()) if (~exact).any() else None,
            'selected_training_clipped':selected['training_clipped_count'],
            'selected_validation_clipped':selected['validation_clipped_count'],
            'maximum_epoch_end_training_clipped':max(r['training_clipped_count'] for r in history[str(seed)]),
            'maximum_epoch_end_validation_clipped':max(r['validation_clipped_count'] for r in history[str(seed)]),
            'validation_epochs_with_clipping':[r['epoch'] for r in history[str(seed)] if r['validation_clipped_count']],
            'largest_errors':[{'cve':rows[i]['cve'],'actual':actual[i],'persistence':float(base[i]),
                'residual':details['prediction'][i],'correction':details['learned_correction'][i],
                'absolute_error':float(errors[i])} for i in worst]})
    return {'schema':'epss_residual_post_run_review_v1','checks_passed':True,
        'model_fitting_performed':False,'test_predictions_calculated':False,
        'run_directory':str(run_dir),'run_lock_sha256':sha256(run_dir/'evaluation_lock.json'),
        'review_script_sha256':sha256(__file__),
        'review_helper_sha256':sha256(Path(__file__).with_name('review_forecast_sequence.py')),
        'training_cves':len(ids['train']),'validation_cves':len(rows),
        'persistence_exact_endpoint_cves':int(exact.sum()),'other_endpoint_cves':int((~exact).sum()),
        'persistence_mae_on_other_cves':float(base_errors[~exact].mean()) if (~exact).any() else None,
        'persistence_metrics':summary['persistence_metrics'],'ridge_metrics':summary['ridge_metrics'],
        'across_seed_metrics':summary['across_seed_metrics'],'per_seed':per_seed}


def explanation(result):
    def fmt(v): return 'Not defined' if v is None else f'{v:.8f}'
    lines=['# Residual GRU: Verified Results and Interpretation','',
        'This review checks the completed user run. It does not train another model',
        'or calculate test predictions. Saved run files remain unchanged.','',
        '## What Passed Verification','',
        'Source/preparation/reference/code hashes; every saved validation prediction',
        'reproduced from its model; CSV/JSON agreement; independent metric calculation;',
        'checkpoint choices and early stopping; selected-checkpoint training errors;',
        'epoch-zero persistence initialization; selected clipping and correction counts.',
        'Earlier-epoch clipping counts are reported from the saved logs. Without each',
        'earlier checkpoint, those counts cannot be independently regenerated.',
        f"Training used {result['training_cves']} CVEs and validation used {result['validation_cves']}.",'',
        '## Matched Validation Comparison','',
        'Every row uses the same CVEs and target dates. MAE is average absolute error;',
        'RMSE emphasizes large errors; lower is better for both. Spearman measures',
        'ordering rather than score accuracy. Errors use the 0-1 scale, where 0.01',
        'is one percentage point. Relative MAE increases below are percentages of',
        'the baseline error, not percentage points of EPSS.','',
        '| Model | Validation MAE | Validation RMSE | Spearman |','|---|---:|---:|---:|']
    for name,m in [('Persistence',result['persistence_metrics']),('Ridge',result['ridge_metrics'])]+[
        (f"Residual seed {s['seed']}",s['metrics']) for s in result['per_seed']]:
        lines.append(f"| {name} | {fmt(m['mae'])} | {fmt(m['rmse'])} | {fmt(m['spearman_rho'])} |")
    lines+=['','| Seed | Direct GRU MAE | Residual GRU MAE | Residual MAE above persistence (%) |',
        '|---|---:|---:|---:|']
    for s in result['per_seed']:
        lines.append(f"| {s['seed']} | {fmt(s['direct_gru_mae'])} | {fmt(s['metrics']['mae'])} | {fmt(s['mae_percent_higher_than_persistence'])} |")
    lines+=['','The residual head substantially improves on the direct head in these runs,',
        'but every seed still has slightly higher MAE than persistence. RMSE is',
        'slightly lower than persistence; this does not establish a useful or',
        'statistically significant improvement. Do not round both MAEs to four',
        'decimals and conclude they are identical.','',
        'Across-seed MAE: mean '+fmt(result['across_seed_metrics']['mae']['mean'])+
        ', sample standard deviation '+fmt(result['across_seed_metrics']['mae']['sample_sd'])+'.',
        'This averages metrics, not predictions. It is not an ensemble score or a',
        'confidence interval. Same-split validation was used to propose the head and',
        'select checkpoints, so this remains development evidence.','',
        '## Checkpoints and Corrections','',
        '| Seed | Epochs completed | Saved epoch | Mean absolute correction | Largest absolute correction |',
        '|---|---:|---:|---:|---:|']
    for s in result['per_seed']:
        lines.append(f"| {s['seed']} | {s['epochs_completed']} | {s['selected_epoch']} | {fmt(s['mean_absolute_correction'])} | {fmt(s['maximum_absolute_correction'])} |")
    lines+=['','Correction means the adjustment to the most recent observed score. These',
        'small selected-checkpoint adjustments explain why predictions remain close',
        'to persistence; they do not prove that the model learned a new predictive',
        'signal. Epoch zero was excluded from selection, as specified before the run.',
        'All seeds stopped under the recorded patience rule.','',
        '## Clipping During Training','',
        '| Seed | Selected checkpoint: training CVEs clipped | Selected checkpoint: validation CVEs clipped | Maximum training count at any epoch end | Maximum validation count at any epoch end |',
        '|---|---:|---:|---:|---:|']
    for s in result['per_seed']:
        lines.append(f"| {s['seed']} | {s['selected_training_clipped']} | {s['selected_validation_clipped']} | {s['maximum_epoch_end_training_clipped']} | {s['maximum_epoch_end_validation_clipped']} |")
    lines+=['','Clipped means the raw prediction fell outside 0-1 and was moved to the',
        'nearest bound. The maxima are the largest count at one recorded epoch end,',
        'not distinct CVEs across training or a sum across epochs. Minibatch peaks',
        'were not logged. No selected-checkpoint predictions were clipped, but some',
        'earlier epochs were. Clipping can suppress gradients outside the bounds;',
        'these observations alone do not show that it caused the result.','',
        '## Which CVEs Benefit?','',
        f"Persistence exactly matches the saved endpoint for {result['persistence_exact_endpoint_cves']} CVEs;",
        f"{result['other_endpoint_cves']} endpoints differ. Intervening days need not have been unchanged.",
        'These outcome-defined subsets are diagnostics only, not prediction inputs.','',
        '| Seed | Lower error than persistence: CVEs | Equal error: CVEs | Higher error: CVEs | MAE on persistence-exact CVEs | MAE on other CVEs |',
        '|---|---:|---:|---:|---:|---:|']
    for s in result['per_seed']:
        lines.append(f"| {s['seed']} | {s['lower_error_cves']} | {s['equal_error_cves']} | {s['higher_error_cves']} | {fmt(s['mae_on_persistence_exact_cves'])} | {fmt(s['mae_on_other_cves'])} |")
    lines+=['',f"Persistence MAE on the differing endpoints is {fmt(result['persistence_mae_on_other_cves'])}.",
        'Count comparisons describe numerical errors at stored precision; tiny',
        'differences need not be operationally meaningful.','', '## Real Large-Error Examples']
    for s in result['per_seed']:
        lines+=['',f"### Seed {s['seed']}",'',
            '| CVE | Recorded Day-60 score | Persistence | Residual prediction | Learned correction |',
            '|---|---:|---:|---:|---:|']
        for r in s['largest_errors'][:3]:
            lines.append(f"| {r['cve']} | {r['actual']:.5f} | {r['persistence']:.5f} | {r['residual']:.8f} | {r['correction']:.8f} |")
    lines+=['','## Interpretation and Next Decision','',
        'No inconsistency was found in the verified predictions, labels, metrics or',
        'selection rules. That is not proof that the entire dataset or code is bug-free.',
        'The residual design preserves the useful last score instead of learning',
        'it from scratch, but has not beaten persistence on MAE in this study.',
        'Large future jumps remain difficult. Do not discard the seeds or change',
        'the split to obtain an improvement. No TPG or SMP contribution was tested.',
        'Keep persistence as the primary MAE reference, and keep the test set sealed.',
        'Before another training configuration, review the large-change CVEs and',
        'which additional text or structured evidence really existed at Day 30.',
        'If temporal evidence is unavailable, label any current-text TPG study as',
        'retrospective instead of claiming leakage-safe historical forecasting.',
        'A later model/loss change needs a separate experiment and user-run command.','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=PROJECT/'05_results/02_evaluation/epss_day30_to_day60/gru_residual_user_run')
    parser.add_argument('--prepared-dir',type=Path,default=PROJECT/'04_data/01_records_and_graphs/epss_day30_to_day60')
    args=parser.parse_args()
    run_dir=args.run_dir.resolve()
    output=run_dir.with_name(run_dir.name+'_review')
    require(not output.exists() or all(p.is_file() and not p.is_symlink() and p.name in
        ('analysis.json','analysis.txt','README.md') for p in output.iterdir()),'Unrelated review files')
    threads=torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        result=review(run_dir,args.prepared_dir)
    finally:
        torch.set_num_threads(threads)
    write_json(output/'analysis.json',result)
    for name in ('analysis.txt','README.md'):
        (output/name).write_text(explanation(result),encoding='utf-8')
    print(f"Verified {result['validation_cves']} validation CVEs, all three residual seeds. Review: {output}")
    print('No fitting or test scoring performed.')


if __name__=='__main__':
    main()
