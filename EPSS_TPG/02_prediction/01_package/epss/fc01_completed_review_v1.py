"""Read-only review of the frozen FC01 Run-1 predictions; no new model fits."""
import csv
import io
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from .final_comparison_v1 import reporting
from .final_comparison_v1.common import ROOT, OUT, publication, read, write, sha, require

RUN = OUT / 'runs/run1'
OUTPUT = ROOT / '05_results/08_fresh_studies/30_fc01_run1_review_v1/review'
CLI = ROOT / '02_prediction/03_analysis/07_fresh_studies/35_review_final_run1.py'


def close(actual, expected):
    require(actual is None and expected is None or actual is not None and expected is not None and
            np.isclose(actual, expected, rtol=0, atol=2e-14), f'Metric mismatch: {actual} / {expected}')


def metrics(y, p):
    binary = y >= .1
    order = np.argsort(-p, kind='stable')
    return {'rmse': float(np.sqrt(np.mean((p-y)**2))), 'mae': float(np.mean(np.abs(p-y))),
            'spearman': None if np.ptp(p)==0 or np.ptp(y)==0 else float(spearmanr(y,p).statistic),
            'auroc_high_score': float(roc_auc_score(binary,p)) if len(set(binary))==2 else None,
            'ap_high_score': float(average_precision_score(binary,p)) if len(set(binary))==2 else None,
            'top_50_hits': float(binary[order[:50]].sum()), 'top_100_hits': float(binary[order[:100]].sum())}


def interval_decisions(intervals):
    rows = [r for r in intervals if r['partition']=='all' and r['unit']=='evidence_group_and_seed']
    require({r['contrast'] for r in rows}=={a+'_minus_'+b for a,b in reporting.CONTRASTS}
            and len(rows)==6, 'Incomplete primary family')
    return [{**r, 'favorable_seeds': sum(s['difference']<0 for s in r['seed_differences']),
             'improvement_supported': r['difference']<0 and r['family_high']<0,
             'nominal_contains_zero': r['nominal_low']<=0<=r['nominal_high'],
             'family_contains_zero': r['family_low']<=0<=r['family_high']} for r in rows]


def summarize_histories(histories):
    summary = []
    for model in reporting.LEARNED:
        rs = [r for r in histories if r['model']==model]
        require(len(rs)==15, 'Missing model/fold/seed fit')
        summary.append({'model': model, 'fits': len(rs),
                        'best_epoch_min': min(r['best_epoch'] for r in rs),
                        'best_epoch_median': float(np.median([r['best_epoch'] for r in rs])),
                        'best_epoch_max': max(r['best_epoch'] for r in rs),
                        'epochs_run_min': min(r['epochs_run'] for r in rs),
                        'epochs_run_max': max(r['epochs_run'] for r in rs),
                        'parameter_count': sorted({r['parameter_count'] for r in rs})})
    return summary


def csv_text(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue()


def run(verify=False):
    report = RUN / 'report'
    publication.check_lock(report)
    identity = read(report/'identity.json'); f=read(report/'findings.json')
    p = read(OUT/'protocol/protocol.json')
    publication.check_lock(OUT/'protocol')
    publication.check_pins(p['code_pins']); publication.check_pins(p['input_pins'])
    require(identity['protocol_sha256']==sha(OUT/'protocol/protocol.json') and identity['run']=='run1'
            and not identity['official_test_scored'] and len(identity['fit_locks'])==75, 'Wrong completed study')
    pins = {str(path):sha(path) for path in report.iterdir() if path.is_file()}
    pins.update({str(path):sha(path) for path in (Path(__file__), CLI,
                 CLI.with_name('test_fc01_completed_review.py'), OUT/'protocol/protocol.json', OUT/'protocol/artifact_lock.json')})
    histories = []
    for name, value in tqdm(sorted(identity['fit_locks'].items()), desc='Check 75 completed fit locks', unit='fit'):
        lock = Path(name); require(sha(lock)==value, 'Changed fit lock')
        publication.check_lock(lock.parent); pins[name]=value
        fit = read(lock.parent/'fit_contract.json'); h=read(lock.parent/'history.json')
        best = min(h, key=lambda row: row['validation']['rmse'])
        require(h[-1]['best_epoch']==best['epoch'] and
                (len(h)==100 or h[-1]['patience_used']==15), 'Invalid completed stopping rule')
        require(len(list(lock.parent.glob('*.pt')))==1, 'Expected best-only checkpoint')
        fold = int(lock.parent.parent.parent.name.split('_')[-1])
        require(fit['roles']==p['folds'][str(fold)], 'Role change')
        histories.append({'model':fit['model'], 'fold':fold, 'seed':fit['seed'], 'epochs_run':len(h),
                          'best_epoch':best['epoch'], 'selection_rmse':best['validation']['rmse'],
                          'first_train_mse':h[0]['train_mse'], 'selected_train_mse':best['train_mse'],
                          'last_train_mse':h[-1]['train_mse'],
                          'parameter_count':fit['architecture']['parameter_count'],
                          'checkpoint_sha256':sha(lock.parent/'best_checkpoint.pt')})
    require({(r['fold'],r['model'],r['seed']) for r in histories} ==
            {(fold,model,seed) for fold in (1,2,3) for model in reporting.LEARNED for seed in f['settings']['seeds']},
            'Incomplete 75-fit grid')
    with (report/'predictions.csv').open(newline='') as stream:
        records = list(csv.DictReader(stream))
    for row in records:
        row.update(fold=int(row['fold']), seed=int(row['seed']), target=float(row['target']),
                   prediction=float(row['prediction']), has_posts=row['has_posts']=='True')
    metadata, predictions = reporting._prepare(records, f['settings'])
    require(len(metadata)==8053 and {r['target_version'] for r in metadata}=={'v2'}, 'Wrong assessment support')
    require({r['cve'] for r in metadata} <= set(p['official_ids']['train']), 'Official holdout in Run 1')
    y=np.array([r['target'] for r in metadata]); partitions=dict(reporting._partitions(metadata))
    mi={name:i for i,name in enumerate(reporting.MODELS)}; si={seed:i for i,seed in enumerate(f['settings']['seeds'])}
    measured={}
    for row in tqdm(f['seed_metrics'], desc='Recompute saved per-seed metrics', unit='row'):
        mask=partitions[row['partition']]
        current=metrics(y[mask],predictions[mi[row['model']],si[row['seed']],mask])
        for key,value in current.items():close(value,row[key])
        measured[row['partition'],row['model'],row['seed']]=current
    for row in f['metrics']:
        rs=[measured[row['partition'],row['model'],s] for s in f['settings']['seeds']]
        for key in rs[0]:
            value=None if any(r[key] is None for r in rs) else float(np.mean([r[key] for r in rs]))
            close(value,row[key])
    decisions=interval_decisions(f['intervals'])
    for row in decisions:
        for seedrow in row['seed_differences']:
            seed=seedrow['seed']
            close(measured['all',row['candidate'],seed]['rmse']-measured['all',row['reference'],seed]['rmse'],
                  seedrow['difference'])
    strata=[]
    for name,mask in [('below_0.1',y<.1),('at_least_0.1',y>=.1)]:
        for model in reporting.MODELS:
            delta=predictions[mi[model]][:,mask]-y[mask]
            strata.append({'partition':name,'model':model,'n':int(mask.sum()),
                           'rmse':float(np.sqrt(np.mean(delta**2,axis=1)).mean()),
                           'mae':float(np.abs(delta).mean(axis=1).mean()),
                           'signed_bias':float(delta.mean()),
                           'squared_error_share':float(np.mean(np.sum(delta**2,axis=1)/
                               np.sum((predictions[mi[model]]-y)**2,axis=1)))})
    overview={r['model']:r for r in f['metrics'] if r['partition']=='all'}
    point_changes=[{'model':model,'rmse_min_seed':min(measured['all',model,s]['rmse'] for s in si),
                   'rmse_max_seed':max(measured['all',model,s]['rmse'] for s in si),
                   'rmse_reduction_vs_mean_percent':100*(1-overview[model]['rmse']/overview['training_mean']['rmse']),
                   'rmse_reduction_vs_median_percent':100*(1-overview[model]['rmse']/overview['training_median']['rmse']),
                   'mae_increase_vs_median_percent':100*(overview[model]['mae']/overview['training_median']['mae']-1)}
                  for model in reporting.LEARNED]
    error=np.mean((predictions[mi['tpg_DM']]-y)**2,axis=0)
    chosen=list(dict.fromkeys([*np.argsort(-y,kind='stable')[:3],*np.argsort(-error,kind='stable')[:3]]))
    examples=[{'cve':metadata[i]['cve'],'fold':metadata[i]['fold'],'has_posts':metadata[i]['has_posts'],
               'actual':float(y[i]),**{model+'_mean_prediction':float(predictions[mi[model],:,i].mean())
                                      for model in reporting.MODELS}}
              for i in chosen]
    result={'schema':'fc01_completed_run1_review_v1', 'fits':75, 'assessment_cves':8053,
            'seeds':f['settings']['seeds'],'metrics':f['metrics'],'seed_metrics':f['seed_metrics'],
            'supports':f['supports'],'primary_decisions':decisions,
            'primary_improvements_supported':sum(r['improvement_supported'] for r in decisions),
            'history_summary':summarize_histories(histories),'history_rows':histories,
            'target_strata_descriptive':strata,'point_changes_descriptive':point_changes,
            'examples_descriptive':examples,
            'example_selection':'Union of three largest actual scores and three largest TPG-DM mean seed-wise squared errors; no CVEs removed.',
            'source_pins':pins,'new_training':False,'test_scoring':False,
            'metrics_recomputed':True,'bootstrap_recomputed_here':False,
            'checkpoint_inference_performed_here':False,
            'decision':'KEEP_RUN1_FROZEN_PROCEED_TO_UNCHANGED_RUN2',
            'limitations':['Retrospective Day-60 estimation, not historical forecasting or observed exploitation.',
                           'Run-1 assessment was previously inspected; this is development evidence.',
                           'Six full-cohort RMSE intervals are primary; slice/ranking comparisons are descriptive.',
                           'Five seeds and saved-fit resampling do not include all training-cohort uncertainty.',
                           'No flag-only arm: adding posts does not isolate wording from availability or training drift.',
                           'Lower point error than constants is not a separately established baseline-superiority claim.',
                           'Within-version targets do not remove calendar, cohort or source-coverage changes.']}
    lines=['FC01 Run 1: completed, frozen development comparison',
           '75 fits; three chronological folds; five seeds; 8,053 assessment CVEs; EPSS v2 Day-60 targets.',
           'No primary improvement is established. Every nominal and family-adjusted primary interval contains zero.',
           'No new fitting, threshold search or test access. Completed artifacts remain unchanged.', '',
           'Model | RMSE | MAE | AUROC(high>=0.1) | AP | top-100 hits (mean seed-wise)']
    lines += [f"{m} | {r['rmse']:.8f} | {r['mae']:.8f} | {r['auroc_high_score']:.6f} | {r['ap_high_score']:.6f} | {r['top_100_hits']:.1f}"
              for m,r in overview.items()]
    lines += ['', 'Primary paired RMSE contrasts; negative favors candidate. Family coverage 99.1667%.']
    lines += [f"{r['contrast']}: {r['difference']:+.8f} [{r['family_low']:+.8f}, {r['family_high']:+.8f}]; favorable seeds {r['favorable_seeds']}/5"
              for r in decisions]
    lines += ['', 'Interpretation:',
              'TPG-D is worse than pooled D by point estimate in all three folds and four of five seeds.',
              'TPG-DM has the lowest overall point RMSE, but its 0.00003471 advantage over pooling is inconclusive.',
              'For DM, graph RMSE improves in fold 2 but worsens in folds 1 and 3; it improves in only two seeds.',
              'On 834 post-bearing CVEs, pooled DM has lower point RMSE than graph DM; no subgroup claim is prespecified.',
              'All learned models have lower overall point RMSE than both constants, but higher MAE than the median.',
              'Pooled D leads overall AP; TPG-DM leads top-100 hits; no model dominates all ranking measures.',
              'Only 62/8053 targets are >=0.1. An AUROC near 0.9 does not mean 90% accuracy or exploitation detection.',
              'Constant ranking metrics pooled across folds can differ from 0.5; each fold uses a different training constant.',
              'Proceed with all 25 Run-2 fits under the already frozen protocol; do not promote a Run-1 winner.',
              *result['limitations']]
    with publication.transaction(OUTPUT,verify=verify) as tmp:
        write(tmp/'findings.json',result)
        for name,rows in [('training_histories',histories),('target_strata',strata),('point_changes',point_changes),('real_examples',examples)]:
            (tmp/(name+'.csv')).write_text(csv_text(rows))
        (tmp/'REPORT.txt').write_text('\n'.join(lines)+'\n')
        publication.check_pins(pins)
    print('FC01 Run-1 review '+('reproduced' if verify else 'saved')+f': {OUTPUT}. No training or test scoring.')

