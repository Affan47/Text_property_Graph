"""Bounded text-only surrogate; no official holdouts or network operations."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from . import data, fitting, model, reporting

_STATE=None


def parameters(p,scale):
    s=dict(p['settings'])
    if scale=='smoke':s.update(s['smoke'])
    return s


def init_worker(p,scale):
    global _STATE
    torch.set_num_threads(4);_STATE=p,scale


def one_fit(job):
    fold,seed,verify,resume=job;p,scale=_STATE;s=parameters(p,scale)
    with threadpool_limits(limits=4,user_api='blas'):
        data.a.require(data.pins()==p['source_pins'] and data.parent.old.runtime()==p['runtime'],'Code/runtime changed')
        ids,arrays,ys=data.load_fold(p,scale,fold);inputs={r:torch.from_numpy(arrays[r]) for r in data.ROLES}
        identity={'fold':fold,'seed':seed,'scale':scale,'settings':s,'source_pins':p['source_pins'],'runtime':p['runtime'],
                  'protocol_sha256':data.a.sha(data.OUTPUT/'protocol/protocol.json'),
                  'prepared_sha256':data.a.sha(data.OUTPUT/scale/'prepared/artifact_lock.json')}
        root=data.OUTPUT/scale/'fits'/f'fold_{fold}'/f'seed_{seed}'
        with data.parent.old.execution.exclusive(root.parent/(root.name+'.lock')):
            complete=(root/'artifact_lock.json').exists()
            data.a.require(not verify or complete,'Missing fit; verification never trains')
            if complete:
                data.a.check_lock(root);net,cp=fitting.reload(root,identity,s,inputs['selection'],ys['selection'])
            else:
                net,cp=fitting.fit(root,identity,s,inputs['optimization'],ys['optimization'],inputs['selection'],ys['selection'],resume)
            pred=model.predict(net,inputs['assessment'])
            again,_=fitting.reload(root,identity,s,inputs['selection'],ys['selection'])
            data.a.require(np.array_equal(pred,model.predict(again,inputs['assessment'])),'Fresh checkpoint replay differs')
            rows=[dict(fold=fold,seed=seed,model='description_only',cve=c,actual=float(y),prediction=float(v))
                  for c,y,v in zip(ids['assessment'],ys['assessment'],pred)]
            result={'best_epoch':cp['epoch'],'selection_rmse':cp['selection_rmse'],
                    'parameters':sum(v.numel() for v in net.parameters()),'checkpoint_sha256':data.a.sha(root/'best_checkpoint.pt'),
                    'predictors':['text'],'assessment':reporting.regression(ys['assessment'],pred)}
            payload=reporting.csv_text(rows)
            if complete:
                data.a.require(data.a.read(root/'evaluation.json')==result and (root/'predictions.csv').read_bytes()==payload.encode(),
                               'Saved predictions/report differ')
            else:
                fitting.write(root/'evaluation.json',result);(root/'predictions.csv').write_text(payload)
                data.a.check_pins(p['source_pins']);data.a.check_pins(p['input_hashes'])
                data.parent.old.publication.lock(root)
            return rows,{str(root/'artifact_lock.json'):data.a.sha(root/'artifact_lock.json')}


def check(scale,jobs):
    p=data.protocol();data.a.require(1<=jobs<=p['settings']['max_jobs'] and jobs*4<=len(os.sched_getaffinity(0)),
                                   'CPU budget exceeded')
    with open('/proc/meminfo') as stream:
        available=int(next(line.split()[1] for line in stream if line.startswith('MemAvailable:')))*1024
    data.a.require(jobs*4*2**30 < available*.7,'RAM budget exceeded; reduce --jobs')
    root=data.OUTPUT/scale/'prepared';data.a.check_lock(root);identity=data.a.read(root/'identity.json')
    data.a.require(identity['source_pins']==data.pins() and identity['protocol_sha256']==data.a.sha(data.OUTPUT/'protocol/protocol.json'),
                   'Prepared inputs have different identity')
    for fold in p['roles'][scale]:data.load_fold(p,scale,int(fold))
    if scale=='full':
        smoke=data.OUTPUT/'smoke/report';data.a.check_lock(smoke);gate=data.a.read(smoke/'identity.json')
        data.a.require(gate['source_pins']==p['source_pins'] and gate['protocol_sha256']==data.a.sha(data.OUTPUT/'protocol/protocol.json'),
                       'Run matching smoke before full training')
        data.a.check_pins(gate['fit_locks'])
    return p


def controls(p,scale):
    rows=[]
    for fold in p['roles'][scale]:
        ids,_,ys=data.load_fold(p,scale,int(fold))
        for name,fn in (('optimization_mean',np.mean),('optimization_median',np.median)):
            val=float(fn(ys['optimization']))
            rows += [dict(fold=int(fold),seed=0,model=name,cve=c,actual=float(y),prediction=val)
                     for c,y in zip(ids['assessment'],ys['assessment'])]
    # Frozen CT07 is a reference, not a matched ten-seed treatment contrast.
    import csv
    selected={r['cve']:r['fold'] for r in rows}
    with open(data.evidence.REPORT/'oof_predictions.csv',newline='') as stream:
        for r in csv.DictReader(stream):
            if r['policy']=='main' and r['model']=='D' and r['cve'] in selected:
                data.a.require(int(r['fold'])==selected[r['cve']],'Reference fold differs')
                rows.append(dict(fold=int(r['fold']),seed=int(r['seed']),model='frozen_CT07_D_plus_structured',
                                 cve=r['cve'],actual=float(r['actual']),prediction=float(r['prediction'])))
    return rows


def publish(rows,p,scale,locks,verify):
    s=parameters(p,scale)
    expected={c:int(f) for f,r in p['roles'][scale].items() for c in r['assessment']}
    for seed in s['seeds']:
        rr=[r for r in rows if r['model']=='description_only' and r['seed']==seed]
        data.a.require(len(rr)==len(expected) and {r['cve']:r['fold'] for r in rr}==expected,'Incomplete/duplicate OOF coverage')
    data.a.check_pins(locks)
    findings=reporting.analyze(rows,p,scale)
    identity={'source_pins':p['source_pins'],'protocol_sha256':data.a.sha(data.OUTPUT/'protocol/protocol.json'),
              'fit_locks':locks,'test_scoring':False,'network_requests':False}
    with data.parent.old.publication.transaction(data.OUTPUT/scale/'report',verify=verify) as temp:
        fitting.write(temp/'identity.json',identity);fitting.write(temp/'findings.json',findings)
        (temp/'oof_predictions.csv').write_text(reporting.csv_text(rows))
        for key in ('day60_metrics','early_warning_metrics','early_warning_predictions','paired_intervals','supports'):
            (temp/(key+'.csv')).write_text(reporting.csv_text(findings[key]))
        (temp/'REPORT.txt').write_text('\n'.join([
            'DS01 description-only surrogate: '+scale,findings['decision'],
            'Day60 MSE fitting; earliest strict selection-RMSE best checkpoint. No crossing-label fitting or threshold tuning.',
            'Inputs: cached current-description vector only. No CVSS, references, source masks or EPSS in the neural model.',
            'This smaller head is not a parameter-count-matched causal feature ablation.',
            'Primary post-fit diagnostic: logit(mean-seed description score) + logit(past EPSS), versus raw past EPSS.',
            'The logit sum is a fixed ranking rule, not a calibrated probability or independence assumption.',
            'Intervals resample evidence groups and globally paired seed identities; fixed-fit intervals are also retained.',
            'Single-internal-day negative assumption is primary; fully observed negatives are the strict sensitivity.',
            'Fold-specific constants can rank across folds; pooled ranking is not the same as within-fold ranking.',
            'CT07 D+structured is a frozen three-seed retrospective reference, not a new ten-seed matched control.',
            'Current wording/encoder timing and late model-fitting dates remain unresolved. A positive gate cannot establish a forecast.',
            'An inconclusive gate stops this surrogate branch, not all possible text forecasting.',
            'No official validation/test scoring, Reddit v1 extension, new embeddings or graph training.'])+'\n')
        data.a.check_pins(locks);data.a.check_pins(p['source_pins']);data.a.check_pins(p['input_hashes'])


def run(scale,jobs=3,verify=False,resume=False,check_only=False):
    p=check(scale,jobs);s=parameters(p,scale)
    if check_only:
        print(f'DS01 {scale} preflight: {3*len(s["seeds"])} fits; CPU {jobs} x 4 threads; no training.');return
    with data.parent.old.execution.exclusive(data.OUTPUT/(scale+'.run.lock')):
        rows=controls(p,scale);locks={}
        jobspec=[(int(f),seed,verify,resume) for f in p['roles'][scale] for seed in s['seeds']]
        if jobs==1:
            init_worker(p,scale)
            for job in tqdm(jobspec,desc='Description-only fits/replays',unit='fit'):
                r,h=one_fit(job);rows+=r;locks.update(h)
        else:
            with ProcessPoolExecutor(jobs,mp_context=multiprocessing.get_context('spawn'),initializer=init_worker,initargs=(p,scale)) as pool:
                pending=[pool.submit(one_fit,j) for j in jobspec]
                for future in tqdm(as_completed(pending),total=len(pending),desc='Description-only fits/replays',unit='fit'):
                    r,h=future.result();rows+=r;locks.update(h)
        output=data.OUTPUT/scale/'report'
        data.a.require(not verify or output.exists(),'Missing report cannot be verified')
        publish(rows,p,scale,locks,output.exists())
    print(f'DS01 {scale} '+('checkpoints and reports reproduced' if verify else 'saved')+f': {output}. No official validation/test scoring.')
