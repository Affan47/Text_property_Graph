"""Read-only interpretation of DS01 and the bounded 24-graph loading benchmark."""
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from .description_surrogate_v1 import data, reporting

a=data.a
BASE=data.ROOT/'05_results/08_fresh_studies'
RUN=data.OUTPUT/'full'
BENCH=BASE/'20_graph_loading_benchmark_v1/01_reddit/dm24_v1'
OUTPUT=BASE/'26_description_surrogate_review_v1/01_reddit/review'
CLI=data.CLI.with_name('31_review_description_surrogate.py')


def rows(path):
    with path.open(newline='') as stream:return list(csv.DictReader(stream))


def equal(actual,expected):
    a.require(actual is None and expected is None or actual is not None and expected is not None and
              bool(np.isclose(actual,expected,rtol=0,atol=2e-14)),f'Recomputed value differs: {actual} / {expected}')


def regression_metrics(y,p):
    diff=np.asarray(p,dtype=float)-np.asarray(y,dtype=float)
    return {'n':len(diff),'mae':float(np.abs(diff).mean()),'rmse':float(np.sqrt(np.mean(diff**2))),
            'signed_bias':float(diff.mean())}


def ranking_metrics(y,p):
    if len(np.unique(y))<2:return {'auroc':None,'average_precision':None}
    return {'auroc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p))}


def summarize_benchmark(b):
    a.require(b['all_parameters_unchanged'] and b['training_fits']==0 and not b['labels_loaded'] and not b['test_access'],
              'Benchmark scope changed')
    a.require(b['config']['limit']==24 and b['config']['repetitions']==3 and b['config']['threads']==4,'Wrong benchmark')
    times=[]
    for store in ('uncached_verified_loader','resident_RAM'):
        rs=[r for r in b['timings'] if r['store']==store]
        a.require(len(rs)==3 and {r['repeat'] for r in rs}=={1,2,3} and all(r['graphs']==24 for r in rs),'Incomplete timing repeats')
        times.append({'store':store,**{k:float(np.mean([r[k] for r in rs]))
                     for k in ('load_seconds','forward_backward_seconds','total_seconds')}})
    u,r=times
    return {'sample_graphs':len(b['sample']),'candidate_graphs':b['candidate_graphs'],
            'nodes_min':min(x['nodes'] for x in b['sample'].values()),'nodes_max':max(x['nodes'] for x in b['sample'].values()),
            'resident_tensor_bytes':b['resident_tensor_bytes'],'resident_mib':b['resident_tensor_bytes']/2**20,
            'peak_process_rss_gib':b['peak_process_rss_kib']/2**20,'preload_seconds':b['preload_seconds'],
            'timing_means':times,'warm_measured_segment_speedup':u['total_seconds']/r['total_seconds'],
            'three_pass_speedup_including_preload':3*u['total_seconds']/(3*r['total_seconds']+b['preload_seconds']),
            'uncached_load_fraction':u['load_seconds']/u['total_seconds'],
            'all_parameters_unchanged':True,'full_epoch_speedup_established':False,
            'full_cohort_memory_established':False,'timings_recomputed':False}


def run(verify=False):
    p=data.protocol();report=RUN/'report';a.check_lock(report);a.check_lock(BENCH)
    identity=a.read(report/'identity.json');f=a.read(report/'findings.json');b=a.read(BENCH/'findings.json')
    a.require(identity['source_pins']==p['source_pins'] and not identity['test_scoring'] and not identity['network_requests'],
              'Report identity or scope differs')
    a.check_pins(identity['source_pins']);a.check_pins(b['input_hashes'])
    pins={**identity['source_pins'],**b['input_hashes']}
    for directory in (report,BENCH,data.OUTPUT/'protocol'):
        pins.update({str(path):a.sha(path) for path in directory.iterdir() if path.is_file()})
    for path in (Path(__file__),CLI,CLI.with_name('test_description_surrogate_review.py')):pins[str(path)]=a.sha(path)
    a.require(len(identity['fit_locks'])==30,'Expected exactly 30 fits')
    history=[]
    for lock,digest in tqdm(sorted(identity['fit_locks'].items()),desc='Verify all best-checkpoint records',unit='fit'):
        path=Path(lock);a.require(a.sha(path)==digest,'Fit lock changed');a.check_lock(path.parent)
        pins[str(path)]=digest
        ev=a.read(path.parent/'evaluation.json');h=a.read(path.parent/'history.json');ident=a.read(path.parent/'identity.json')
        best=min(h,key=lambda r:r['selection_rmse'])
        a.require(best['epoch']==ev['best_epoch'] and a.sha(path.parent/'best_checkpoint.pt')==ev['checkpoint_sha256'],
                  'Best checkpoint selection/hash differs')
        a.require(ev['parameters']==25761 and ev['predictors']==['text'] and len(list(path.parent.glob('*.pt')))==1,
                  'Unexpected model inputs/parameters/checkpoints')
        a.require(ident['source_pins']==p['source_pins'] and ident['settings']==p['settings'],'Fit settings changed')
        history.append({'fold':ident['fold'],'seed':ident['seed'],'best_epoch':ev['best_epoch'],'epochs_completed':len(h),
                        'selection_rmse':ev['selection_rmse'],'first_optimization_mse':h[0]['optimization_mse'],
                        'last_optimization_mse':h[-1]['optimization_mse'],
                        'checkpoint_sha256':ev['checkpoint_sha256']})
    a.require({(r['fold'],r['seed']) for r in history}=={(fold,seed) for fold in (1,2,3) for seed in p['settings']['seeds']},
              'Incomplete fold/seed grid')
    oof=rows(report/'oof_predictions.csv');ranking=rows(report/'early_warning_predictions.csv')
    for r in oof:
        r.update(seed=int(r['seed']),fold=int(r['fold']),actual=float(r['actual']),prediction=float(r['prediction']))
    for r in tqdm(f['day60_metrics'],desc='Recompute Day60 metrics from saved predictions',unit='row'):
        rr=[x for x in oof if x['model']==r['model'] and x['seed']==r['seed'] and (r['fold']==0 or x['fold']==r['fold'])]
        rec=regression_metrics([x['actual'] for x in rr],[x['prediction'] for x in rr])
        for k,v in rec.items():equal(v,r[k])
    for r in tqdm(f['early_warning_metrics'],desc='Recompute ranking metrics from saved scores',unit='row'):
        rr=[x for x in ranking if x['policy']==r['policy'] and (r['fold']==0 or int(x['fold'])==r['fold'])]
        y=np.array([int(x['label']) for x in rr]);scores=np.array([float(x[r['model']]) for x in rr])
        a.require(len(y)==r['cves'] and int(y.sum())==r['risers'],'Ranking support changed')
        for k,v in ranking_metrics(y,scores).items():equal(v,r[k])
    a.require(reporting.decision(f['paired_intervals'],False)==f['decision']=='STOP_THIS_SURROGATE_BRANCH_INCONCLUSIVE_OR_NO_GAIN',
              'Frozen progression decision changed')
    day60=[]
    for name in sorted({r['model'] for r in f['day60_metrics']}):
        for fold in (0,1,2,3):
            rs=[r for r in f['day60_metrics'] if r['model']==name and r['fold']==fold]
            day60.append({'model':name,'fold':fold,'fits_or_controls':len(rs),'cves':rs[0]['n'],
                          **{k+'_'+stat:float(fn([r[k] for r in rs])) for k in ('mae','rmse','signed_bias')
                             for stat,fn in (('mean',np.mean),('min',np.min),('max',np.max))}})
    lookup={(r['policy'],r['model'],r['fold']):r for r in f['early_warning_metrics']}
    folds=[];seeds=[]
    for policy in ('isolated_day_assumption','strict'):
        for fold in (0,1,2,3):
            base=lookup[policy,'past_epss',fold];add=lookup[policy,'description_plus_epss',fold]
            folds.append({'policy':policy,'fold':fold,'cves':base['cves'],'risers':base['risers'],
                          **{k+'_change':None if base[k] is None else add[k]-base[k] for k in ('auroc','average_precision')}})
        for seed in p['settings']['seeds']:
            base=lookup[policy,'past_epss',0];add=lookup[policy,f'description_plus_epss_seed_{seed}',0]
            seeds.append({'policy':policy,'seed':seed,**{k:add[k] for k in ('auroc','average_precision')},
                          **{k+'_change':add[k]-base[k] for k in ('auroc','average_precision')}})
    primary=[r for r in seeds if r['policy']=='isolated_day_assumption']
    result={'schema':'ds01_completed_review_v1','fits':len(history),'assessment_cves':8053,
            'decision':f['decision'],'support':f['supports'],'day60_summary':day60,
            'ranking_summary':[r for r in f['early_warning_metrics'] if r['model'] in ('past_epss','description_only','description_plus_epss')],
            'paired_intervals':f['paired_intervals'],'fold_changes':folds,'seed_results':seeds,'history_summary':history,
            'primary_seeds_positive_ap':sum(r['average_precision_change']>0 for r in primary),
            'primary_seeds_positive_auroc':sum(r['auroc_change']>0 for r in primary),
            'benchmark':summarize_benchmark(b),'new_training':False,'new_network_requests':0,'test_scoring':False,
            'interpretation':'Primary gate failed; strict-label AP sensitivity does not supersede the primary rule. Stop this surrogate branch.',
            'next_step':'Separate bounded within-version Day60 MSE graph-versus-pooled specification, resource inventory and input/readout parity before training.',
            'source_pins':pins}
    lines=['DS01: completed description-only surrogate and graph-loading review','',result['decision'],
           '30 best-only fits, 3 CT07 main folds x 10 seeds, original training roles only. No new fits or holdout scoring.',
           'All saved Day60 and early-warning point metrics independently recomputed; original fits/reports unchanged.',
           'Primary AP increases, but its group-and-seed interval includes zero; AUROC decreases slightly.',
           'Strict labels retain the same positives and remove 4,281 assumed negatives. Positive AP there is sensitivity evidence, not the primary pass.',
           'Nine of ten individual seeds improve primary AP; none improves primary AUROC. Ensemble ranking differs from mean seed-wise metrics.',
           'Day60 RMSE has a small point advantage over constants; MAE is worse than the median control. No new uncertainty claim for these errors.',
           'The older structured-D reference differs in inputs, head capacity and seed count; this result cannot assign the prior gain to CVSS or prove leakage caused it.',
           'Current descriptions and model-fitting dates remain retrospective. Stopping the surrogate is not a universal rejection of text forecasting.',
           'The benchmark confirms tensor/representation parity with unchanged parameters in its saved run; verification hashes tensors, not wall-clock timing.',
           'Warm segment speed excludes preload. Three-pass inclusive speed includes preload but not all process overhead.',
           'Do not extrapolate 24 graphs to full-fold RAM or training epoch time. Full-cohort memory and matched-input/readout checks still precede graph training.',
           'No automatic Reddit v1 extension, further histories, threshold tuning or new epoch search.',
           '', 'Policy | Model | AUROC | AP']
    lines += [f'{r["policy"]} | {r["model"]} | {r["auroc"]} | {r["average_precision"]}'
              for r in result['ranking_summary'] if r['fold']==0]
    lines += ['', 'Policy | Unit | Metric | Change | Low | High']
    lines += [f'{r["policy"]} | {r["unit"]} | {r["metric"]} | {r["difference"]} | {r["low"]} | {r["high"]}' for r in result['paired_intervals']]
    lines += ['', 'Benchmark: '+a.dump(result['benchmark'])]
    a.publish(OUTPUT,{'findings.json':result,'REPORT.txt':'\n'.join(lines)+'\n',
                     'day60_summary.csv':reporting.csv_text(day60),'fold_changes.csv':reporting.csv_text(folds),
                     'seed_results.csv':reporting.csv_text(seeds),'selected_checkpoints.csv':reporting.csv_text(history),
                     'ranking_summary.csv':reporting.csv_text(result['ranking_summary'])},pins,verify)
    print('DS01 completed review '+('reproduced: ' if verify else 'saved: ')+str(OUTPUT)+'. No training or test scoring.')
