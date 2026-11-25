"""Bounded execution, complete-fit gates, and one final scientific release."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import io
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tqdm import tqdm

from .common import OUT, MODELS, read, write, sha, digest, require, publication, exclusive, completed, available_memory
from . import protocol, data, fitting, reporting, test_cache


def fit_path(run,fold,model,seed,integration=False):
    root=OUT/'integration/fits' if integration else OUT/'runs'/run/'fits'
    return root/f'fold_{fold}'/model/f'seed_{seed}'


def specs(p,run,integration=False):
    folds=[1] if integration else [int(f) for f in data.selected_folds(p,run)]
    seeds=[p['settings']['integration_seed']] if integration else p['settings']['seeds']
    return [(fold,model,seed) for fold in folds for model in MODELS for seed in seeds]


def fit_identity(p,ctx,model,seed,integration=False):
    return {'schema':'fc01_fit_v1','settings':p['settings'],'contract':ctx['contract'],
            'run':ctx['run'],'fold':ctx['fold'],'model':model,'seed':seed,'integration':integration,
            'source_pins':p['code_pins'],'protocol_sha256':sha(OUT/'protocol/protocol.json'),
            'prepared_lock_sha256':sha(OUT/'prepared'/ctx['run']/'artifact_lock.json'),
            'runtime':p['runtime'],'test_access':False}


def guard(p,run,integration=False):
    require(not (OUT/'test_evaluation_release.json').exists(), 'Final test has been released: new fitting is permanently closed')
    if not integration:
        publication.check_lock(OUT/'integration/gate')
        gate=read(OUT/'integration/gate/identity.json')
        require(gate['protocol_sha256']==sha(OUT/'protocol/protocol.json') and gate['fits']==5,
                'Missing matching one-epoch integration gate')
        publication.check_pins(gate['fit_locks'])
    if run=='run2':publication.check_lock(OUT/'runs/run1/report')


def resources(p,run,jobs,budget_gib,integration=False):
    require(type(jobs) is int and jobs>=1 and jobs*4<=len(os.sched_getaffinity(0)), 'Invalid CPU worker budget')
    require(np.isfinite(budget_gib) and budget_gib>0, 'Positive per-fit tensor RAM budget required')
    inventory=read(OUT/'prepared'/run/'resource_inventory.json')
    rows=[r for r in inventory if not integration or r['fold']==1]
    needed=max(r['resident_tensor_bytes'] for r in rows)
    # Inventory conservatively includes assessment graphs together with train/selection.
    require(needed+2**26<=budget_gib*2**30,
            f'Largest complete working-set upper bound is {needed/2**30:.3f} GiB; increase --budget-gib, never drop CVEs')
    largest=max(r['largest_graph_tensor_bytes'] for r in rows)
    per_worker=budget_gib*2**30+max(4*2**30,largest*8)
    available=available_memory()
    require(jobs*per_worker<=available*.70, 'RAM safety gate: reduce --jobs or request a smaller sufficient per-fit budget')
    result={'jobs':jobs,'threads_per_fit':4,'tensor_budget_gib':budget_gib,
            'working_set_upper_bound_gib':needed/2**30,'estimated_worker_peak_gib':per_worker/2**30,
            'available_gib':available/2**30,'inventory':rows}
    print('[RAM] '+json.dumps({k:v for k,v in result.items() if k!='inventory'},sort_keys=True),flush=True)
    return result


def preflight(run,jobs,budget_gib):
    p=protocol.load();publication.check_lock(OUT/'prepared'/run)
    resources(p,run,jobs,budget_gib)
    print(f'[PLAN] {run}: {len(specs(p,run))} fits; seeds={p["settings"]["seeds"]}; test closed. No fitting.')


def worker_fit(job):
    run,fold,model,seed,verify,resume,integration,budget=job
    p=protocol.load()
    if not verify:guard(p,run,integration)
    ctx=data.load_context(p,run,fold)
    if integration:
        ctx={**ctx,'ids':{**ctx['ids'],'assessment':[]}}
    identity=fit_identity(p,ctx,model,seed,integration)
    destination=fit_path(run,fold,model,seed,integration)
    with exclusive(OUT/'locks'/('integration' if integration else run)/f'{fold}_{model}_{seed}.lock'):
        fitting.fit(destination,ctx,identity,model,seed,verify=verify,resume=resume,
                    epochs_override=1 if integration else None,budget_gib=budget)
    return str(destination)


def csv_text(rows):
    if not rows:return ''
    keys=sorted({k for row in rows for k in row});stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=keys,lineterminator='\n');writer.writeheader()
    for row in rows:
        writer.writerow({k:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
                         if isinstance(v,(list,dict)) else v for k,v in row.items()})
    return stream.getvalue()


def checks(p,run,integration=False):
    result={}
    for fold,model,seed in specs(p,run,integration):
        folder=fit_path(run,fold,model,seed,integration)
        publication.check_lock(folder)
        ident=read(folder/'run_identity.json')
        require(ident['protocol_sha256']==sha(OUT/'protocol/protocol.json') and ident['model']==model
                and ident['seed']==seed and ident['run']==run and ident['fold']==fold
                and ident['integration']==integration, 'Fit identity differs')
        result[str(folder/'artifact_lock.json')]=sha(folder/'artifact_lock.json')
    return result


def train(run,jobs=2,budget_gib=32,verify=False,resume=False,integration=False):
    require(not (verify and resume),'Choose --verify-only or --resume, not both')
    p=protocol.load()
    require(not integration or run=='run1','Integration uses only Run 1 fold 1')
    if not verify:guard(p,run,integration)
    publication.check_lock(OUT/'prepared'/run)
    resources(p,run,jobs,budget_gib,integration)
    jobspec=[(run,f,m,s,verify,resume,integration,budget_gib) for f,m,s in specs(p,run,integration)]
    with exclusive(OUT/'locks'/('integration' if integration else run)/'scheduler.lock'):
        if jobs==1:
            for job in tqdm(jobspec,desc='Final matched fits / replays',unit='fit'):worker_fit(job)
        else:
            with ProcessPoolExecutor(max_workers=jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures=[pool.submit(worker_fit,job) for job in jobspec]
                for future in tqdm(as_completed(futures),total=len(futures),desc='Final matched fits / replays',unit='fit'):
                    future.result()
        locks=checks(p,run,integration)
        if integration:
            destination=OUT/'integration/gate';is_verify=completed(destination)
            with publication.transaction(destination,verify=is_verify) as tmp:
                write(tmp/'identity.json',{'protocol_sha256':sha(OUT/'protocol/protocol.json'),
                     'fits':5,'fold':1,'seed':17,'epochs':1,'fit_locks':locks,
                     'test_access':False,'assessment_scored':False,'weights_reused_in_scientific_fits':False})
                (tmp/'REPORT.txt').write_text('All five one-epoch fold-1 integration fits/replays passed.\n'
                    'Checkpoint selection/reload and complete RAM preload are required by each fit.\n'
                    'Scientific runs initialize fresh; no settings are chosen from integration outcomes.\n')
            print('FC01 single integration gate passed; scientific training remains user-run.')
        else:
            report(run,verify=verify or completed(OUT/'runs'/run/'report'))
            if verify and run=='run2':
                attestation={'protocol_sha256':sha(OUT/'protocol/protocol.json'),'fit_locks':locks}
                target=OUT/'run2_replay.json'
                if target.exists():require(read(target)==attestation,'Official replay identity changed')
                else:write(target,attestation)


def prediction_rows(ctx,evaluation,model,seed,role):
    rows=[];predictions=evaluation['predictions'][role]
    require([r['cve'] for r in predictions]==ctx['ids'][role],'Prediction order/IDs differ')
    for row in predictions:
        cid=row['cve'];meta=ctx['metadata'][cid]
        require(float(row['target'])==ctx['labels'][cid]['value'],'Saved target alignment changed')
        rows.append({'cve':cid,'fold':ctx['fold'],'model':model,'seed':seed,
                     'target':ctx['labels'][cid]['value'],'prediction':float(row['prediction']),
                     'group':meta['group'],'has_posts':meta['has_posts'],'target_version':meta['target_version'],
                     'overlap_partition':meta.get('overlap_partition','not_applicable')})
    return rows


def constant_rows(ctx,ids,fold):
    return [{'cve':cid,'fold':fold,'model':model,'seed':0,'target':ctx['labels'][cid]['value'],
             'prediction':value,'group':ctx['metadata'][cid]['group'],'has_posts':ctx['metadata'][cid]['has_posts'],
             'target_version':ctx['metadata'][cid]['target_version'],
             'overlap_partition':ctx['metadata'][cid].get('overlap_partition','not_applicable')}
            for model,value in ctx['constants'].items() for cid in ids]


def publish_report(destination,rows,p,identity,verify=False,comparison=None):
    rows=sorted(rows,key=lambda r:(r['fold'],r['model'],r['seed'],r['cve']))
    values=reporting.generate(rows,{**p['settings'],'progress':True})
    with publication.transaction(destination,verify=verify) as tmp:
        write(tmp/'identity.json',identity);write(tmp/'findings.json',values)
        if comparison is not None:write(tmp/'cross_run_decisions.json',comparison(values))
        for name in ('metrics','seed_metrics','supports','intervals','baseline_differences'):
            (tmp/(name+'.csv')).write_text(csv_text(values[name]))
        (tmp/'predictions.csv').write_text(csv_text(rows))
        lines=['FC01 fixed final comparison',
               'Raw-score MSE; earliest best selection RMSE; no ensemble substitution.',
               'Group+seed intervals primary, CVE+seed sensitivity; six family-adjusted contrasts.',
               'Retrospective Day-60 score estimation, not exploitation forecasting.',
               'Baseline differences are descriptive; no unadjusted baseline-superiority claim.',
               '', 'Model | CVEs | mean seed-wise RMSE | MAE | Spearman | AP(high>=0.1)']
        lines += [f"{r['model']} | {r['n']} | {r['rmse']} | {r['mae']} | {r['spearman']} | {r['ap_high_score']}"
                  for r in values['metrics'] if r['partition']=='all']
        lines+=['',*values['notes']]
        (tmp/'REPORT.txt').write_text('\n'.join(lines)+'\n')
        figures(tmp,values)
        publication.check_pins(p['code_pins']);publication.check_pins(identity['fit_locks'])


def figures(directory,values):
    rows=[r for r in values['metrics'] if r['partition']=='all']
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.barh([r['model'] for r in rows],[r['rmse'] for r in rows],color='#287b80')
    ax.set(xlabel='Mean of seed-wise RMSE (lower is better)',title='Fixed matched predictors, same CVEs')
    ax.invert_yaxis();fig.tight_layout();fig.savefig(directory/'rmse.png',dpi=150);plt.close(fig)
    rows=primary_intervals(values).values()
    rows=list(rows)
    fig,ax=plt.subplots(figsize=(10,4.8))
    for i,row in enumerate(rows):
        ax.plot([row['family_low'],row['family_high']],[i,i],color='#287b80',linewidth=2)
        ax.plot(row['difference'],i,'o',color='#aa4939')
    ax.axvline(0,color='#555555',linestyle='--',linewidth=1)
    ax.set(yticks=range(len(rows)),yticklabels=[r['contrast'] for r in rows],
           xlabel='Candidate minus reference RMSE (negative favors candidate)',
           title='Evidence-group + seed intervals: six-contrast family adjustment')
    ax.invert_yaxis();fig.tight_layout();fig.savefig(directory/'paired_rmse.png',dpi=150);plt.close(fig)


def report(run,verify=False):
    p=protocol.load();locks=checks(p,run);rows=[]
    for fold in data.selected_folds(p,run):
        ctx=data.load_context(p,run,fold);role='assessment' if run=='run1' else 'validation'
        rows+=constant_rows(ctx,ctx['ids'][role],int(fold))
        for model in MODELS:
            for seed in p['settings']['seeds']:
                folder=fit_path(run,int(fold),model,seed)
                rows+=prediction_rows(ctx,read(folder/'evaluation.json'),model,seed,role)
    identity={'protocol_sha256':sha(OUT/'protocol/protocol.json'),'fit_locks':locks,
              'run':run,'official_test_scored':False,'assessment_previously_inspected':True}
    publish_report(OUT/'runs'/run/'report',rows,p,identity,verify)
    print(f'FC01 {run} checkpoints and matched report '+('reproduced' if verify else 'saved')+'. No test scoring.')


def lock_test():
    p=protocol.load();publication.check_lock(OUT/'runs/run1/report');publication.check_lock(OUT/'runs/run2/report')
    locks=checks(p,'run2');records=[]
    # A fresh-process replay attestation is mandatory before exposing test predictors.
    attestation=OUT/'run2_replay.json'
    require(attestation.exists() and read(attestation)=={'protocol_sha256':sha(OUT/'protocol/protocol.json'),'fit_locks':locks},
            'Run train --run run2 --verify-only first; no test predictors are available yet')
    for fold,model,seed in specs(p,'run2'):
        path=fit_path('run2',fold,model,seed)
        records.append({'model':model,'seed':seed,'path':str(path/'best_checkpoint.pt'),
                        'completion_lock':str(path/'artifact_lock.json')})
    comparison_files={str(OUT/'runs'/run/'report'/name):sha(OUT/'runs'/run/'report'/name)
                      for run in ('run1','run2') for name in ('artifact_lock.json','findings.json')}
    test_cache.freeze_release(p,records,OUT/'prepared/run2/fold_0/constants.json',comparison_files=comparison_files)
    print('FC01 all 25 checkpoints/transforms frozen for test predictors. No test scores read.')


def primary_intervals(values):
    rows=[r for r in values['intervals'] if r['partition']=='all' and r['unit']=='evidence_group_and_seed']
    expected={a+'_minus_'+b for a,b in reporting.CONTRASTS}
    require(len(rows)==len(expected) and {r['contrast'] for r in rows}==expected,
            'All six primary evidence-group/seed intervals are required')
    require(all(r['family_low'] is not None and r['family_high'] is not None and r['difference'] is not None
                for r in rows),'Undefined primary interval; refuse a final claim')
    return {r['contrast']:r for r in rows}


def cross_run_decisions(test):
    publication.check_lock(OUT/'runs/run1/report')
    dev=read(OUT/'runs/run1/report/findings.json')
    a,b=primary_intervals(dev),primary_intervals(test);rows=[]
    for name in sorted(a):
        support=a[name]['family_high']<0 and a[name]['difference']<0
        direction=b[name]['difference']<0;confirmed=b[name]['family_high']<0 and direction
        rows.append({'contrast':name,'run1_adjusted_support':support,'test_direction_favorable':direction,
                     'test_adjusted_support':confirmed,
                     'claim':'supported_in_both_prespecified_comparisons' if support and confirmed else
                       ('development_support_directionally_replicated_only' if support and direction else 'improvement_not_established_by_frozen_rule')})
    return {'decisions':rows,'baseline_superiority_established_by_this_rule':False,'further_training_allowed':False}


def test_resources(ctx,budget_gib):
    require(np.isfinite(budget_gib) and budget_gib>0,'Positive test RAM budget required before release')
    inventory=[data.estimate_view_bytes(ctx['index'],ctx['records'],ctx['contract'],ctx['ids']['validation'],v)
               for v in ('D','DM')]
    needed=max(r['resident_tensor_bytes'] for r in inventory)
    largest=max(r['largest_graph_tensor_bytes'] for r in inventory)
    require(needed+2**26<=budget_gib*2**30,
            f'Test working set upper bound is {needed/2**30:.3f} GiB; increase --budget-gib before release')
    require(budget_gib*2**30+max(4*2**30,largest*8)<=available_memory()*.70,
            'Test RAM safety gate failed before release')
    print(f'[TEST RAM] {needed/2**30:.3f} GiB tensor bound; {budget_gib} GiB allowance; labels still closed.',flush=True)


def evaluate(release_test=False,verify=False,resume=False,budget_gib=32):
    require(release_test,'Held-out outcomes require explicit --release-test')
    require(not (verify and resume),'Choose --verify-only or --resume, not both')
    require(np.isfinite(budget_gib) and budget_gib>0,'Positive test RAM budget required before release')
    p=protocol.load();release=OUT/'release_lock.json';output=OUT/'test_report'
    require(not output.exists() or verify or resume,'Test report exists; only identical --verify-only/--resume is permitted')
    require(not verify or completed(output),'No completed test report to verify')
    ctx=test_cache.prepare(p,release,stage='verify',verify_only=verify)
    test_resources(ctx,budget_gib)
    with test_cache.evaluation_release(p,release,release_test=True,verify_only=verify):
        ctx['labels']=test_cache.load_test_labels(p)
        ctx['constants']=read(OUT/'prepared/run2/fold_0/constants.json')
        records=constant_rows(ctx,ctx['ids']['validation'],0);locks=checks(p,'run2')
        official=data.load_context(p,'run2',0)
        for model in tqdm(MODELS,desc='Evaluate all frozen model rows',unit='model'):
            for seed in p['settings']['seeds']:
                folder=fit_path('run2',0,model,seed)
                ident=fit_identity(p,official,model,seed)
                result=fitting.predict_checkpoint(folder,ctx,ident,model,seed,ctx['ids']['validation'],budget_gib=budget_gib)
                for row in result:
                    cid=row['cve'];m=ctx['metadata'][cid]
                    records.append({'cve':cid,'fold':0,'model':model,'seed':seed,'target':ctx['labels'][cid]['value'],
                                    'prediction':row['prediction'],'group':m['group'],'has_posts':m['has_posts'],'target_version':m['target_version'],
                                    'overlap_partition':m.get('overlap_partition','not_applicable')})
        identity={'protocol_sha256':sha(OUT/'protocol/protocol.json'),'fit_locks':locks,
                  'run1_report_lock_sha256':sha(OUT/'runs/run1/report/artifact_lock.json'),
                  'release_lock_sha256':sha(release),'test_release_sha256':sha(OUT/'test_evaluation_release.json'),
                  'official_test_scored':True,'training':False}
        publish_report(output,records,p,identity,verify=verify or completed(output),comparison=cross_run_decisions)
    print('FC01 single final test report '+('reproduced' if verify else 'published')+'. No fitting or retuning; experimental series closed.')
