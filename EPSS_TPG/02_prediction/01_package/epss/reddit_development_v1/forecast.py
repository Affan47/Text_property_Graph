"""RD04 full development: daily lazy histories, fixed controls and three GRUs."""
from contextlib import contextmanager
import csv
import gzip
import io
import time

import numpy as np
from tqdm import tqdm
from . import common as c
from ..reddit_forecasting import data, evaluation as reval
from ..continuous_forecasting import controls, evaluation as ev, runner as base


@contextmanager
def prediction_writer(path):
    if path is None:
        yield None
        return
    with path.open('wb') as raw, gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0,compresslevel=1) as z:
        with io.TextIOWrapper(z,encoding='utf-8',newline='') as text:
            yield csv.writer(text,lineterminator='\n')


def summarize(records):
    summary={}
    for s in reval.SLICES:
        chosen=[r['metrics'][s] for r in records if r['metrics'][s] is not None]
        means={f:float(np.mean([r[f] for r in chosen])) if chosen else None for f in ev.FIELDS}
        summary[s]=dict(cves=len(chosen),cells=sum(r['cells'] for r in chosen),**means,
                       rmse=float(np.sqrt(means['mse'])) if chosen else None,
                       raw_rmse=float(np.sqrt(means['raw_mse'])) if chosen else None)
    return dict(per_cve=records,summary=summary)


def row_metrics(row,y,raw,mask):
    out=ev.cve_metrics(y,raw,mask)
    for name,chosen in data.partitions(row).items():
        out.update({f'{s}/{name}':m for s,m in ev.cve_metrics(y,raw,chosen).items()})
    return out


def evaluate(rows,p,net=None,correction=None,path=None):
    if net is not None:
        net.eval()
    records=[]; previous=None
    with prediction_writer(path) as writer:
        if writer:
            writer.writerow(['cve','anchor_day','target_day','lead_days','actual_epss','raw_prediction',
                             'clipped_prediction','anchor_target_crossing','history_target_crossing'])
        for row in tqdm(sorted(rows,key=lambda r:r['cve']),desc='RD04 matched daily validation',unit='CVE',leave=False):
            c.require(row['split'] in ('train','validation'),'Forbidden evaluation role')
            y,raw,mask=ev.predict_row(row,p['anchor_packet'],net,correction)
            records.append(dict(cve=row['cve'],metrics=row_metrics(row,y,raw,mask)))
            if writer:
                parts=data.partitions(row); clipped=np.clip(raw,0,1)
                for t,j in np.argwhere(mask):
                    key=(row['cve'],int(t),int(t+j+1))
                    c.require(previous is None or key>previous,'Noncanonical or duplicate forecast cell');previous=key
                    writer.writerow([*key,int(j+1),float(y[t,j]),float(raw[t,j]),float(clipped[t,j]),
                        int(parts['anchor_target_crossing'][t,j]),int(parts['history_target_crossing'][t,j])])
    return summarize(records)


def context(scale):
    c.gate(); rows,parent=data.load(scale); p=parent['settings']
    required=dict(hidden=16,head_hidden=32,learning_rate=.001,weight_decay=.0001,batch_cves=8,
                  epochs=100,patience=15,seeds=list(c.SEEDS),threads=2,device='cpu')
    c.require(all(p[k]==v for k,v in required.items()),'Frozen RD04 settings differ')
    counts={s:dict(cves=len(rs),cells=sum(int(data.eligible(r).sum()) for r in rs),
                  primary_cves=sum(bool(data.eligible(r)[30,29]) for r in rs)) for s,rs in rows.items()}
    if scale=='full':
        c.require(counts==dict(train=dict(cves=24548,cells=96289074,primary_cves=24179),
                               validation=dict(cves=1106,cells=3692438,primary_cves=972)), 'RD04 frozen counts differ')
    ident=c.identity('RD04',scale,dict(preflight=parent,preflight_lock=c.sha(data.OUTPUT/'preflight'/scale/'artifact_lock.json')),p)
    ident['counts']=counts
    return rows,ident


def control_stage(directory,rows,ident,verify,resume):
    started=time.monotonic()
    if directory.exists():
        c.checked(directory,ident)
        c.require(verify or resume, 'Controls exist; use --resume or --verify-only')
        if resume and not verify:
            return c.read(directory/'evaluation.json')
    if verify:
        state=c.read(directory/'lead_correction.json')
        c.require(state['training_ids']==[r['cve'] for r in rows['train']] and
                  state['training_series_sha256']==c.digest(rows['train']),'Correction provenance differs')
    else:
        state=controls.fit(rows['train'])
    with c.core.transaction(directory,verify) as stage:
        c.put(stage,'run_identity.json',ident);c.put(stage,'lead_correction.json',state)
        results={name:evaluate(rows['validation'],ident['settings'],correction=correction,
                 path=stage/f'{name}_predictions.csv.gz') for name,correction in [('persistence',None),('lead_correction',state)]}
        c.put(stage,'evaluation.json',results)
        c.put(stage,'runtime.json',c.read(directory/'runtime.json') if verify else
              {'fit_and_evaluation_seconds':time.monotonic()-started})
    return results


def destination(scale):
    return data.OUTPUT/('full' if scale=='full' else 'development_smoke_v1')


def run(scale,verify=False,resume=False):
    rows,ident=context(scale); p=ident['settings']; output=destination(scale)
    if (output/'artifact_lock.json').exists():
        c.require(verify,'Completed output is immutable; use --verify-only');c.core.check_lock(output)
    c.require(not verify or (output/'artifact_lock.json').exists(),'Cannot verify missing completed outputs')
    output.mkdir(parents=True,exist_ok=True)
    results=control_stage(output/'controls',rows,ident,verify,resume);runtimes={}
    for seed in tqdm(c.SEEDS,desc='RD04 three fixed GRU fits',unit='seed'):
        key=f'seed_{seed}'
        results[key]=c.fit_unit(output/key,output/'recovery'/f'{key}.pt',ident,key,
            lambda:base.model_for(seed,p),
            lambda net,opt,epoch:base.train_epoch(net,rows['train'],opt,seed,epoch,p),
            lambda net,stage:evaluate(rows['validation'],p,net=net,path=stage/'predictions.csv.gz' if stage else None),
            lambda r:r['summary'],lambda m:m['day30_to_day60']['mae'],
            p['epochs'] if scale=='full' else p['smoke']['epochs'],p['patience'] if scale=='full' else p['smoke']['patience'],
            verify,resume,persistence=results['persistence'])
        runtimes[key]=c.read(output/key/'runtime.json')
    with c.core.transaction(output/'report',c.report_replay(output/'report',verify,resume)) as stage:
        intervals=reval.intervals(results,p['bootstrap_repetitions'],p['bootstrap_seed'])
        tables=[dict(model=n,slice=s,**m) for n,r in results.items() for s,m in r['summary'].items()]
        means={s:c.mean_metrics([results[f'seed_{seed}']['summary'][s] for seed in c.SEEDS]) for s in reval.SLICES}
        tables += [dict(model='GRU_seed_mean',slice=s,**m) for s,m in means.items()]
        findings=dict(task='RD04',scale=scale,test_scoring=False,counts=ident['counts'],seed_means=means,
            models={n:r['summary'] for n,r in results.items()},runtimes=runtimes,
            controls_runtime=c.read(output/'controls/runtime.json'),
            training_release_coverage=c.read(data.OUTPUT/'preflight/full/summary.json')['full_preflight']['train'],
            checkpoint_hashes={f'seed_{s}':c.sha(output/f'seed_{s}/checkpoint.pt') for s in c.SEEDS},
            upstream=ident['upstream'],limitations=['No text or catalogue predictors. Missing targets are never imputed.',
             'Epoch 0 is exact persistence and is not a selection candidate.',
             'Training release diagnostics are coverage counts, not additional training-set prediction evaluations.',
             'Empty release-crossing validation slices are null, not zero error. Seed means are metric means, not ensembles.'])
        c.put(stage,'run_identity.json',ident);c.put(stage,'findings.json',findings);c.put(stage,'paired_intervals.json',intervals)
        c.csv_table(stage/'rd04_full_metrics.csv',tables)
        interval_rows=[dict(comparison=r['comparison'],slice=r['slice'],metric=k,**v)
                       for r in intervals for k,v in r.get('metrics',{}).items()]
        c.csv_table(stage/'paired_intervals.csv',interval_rows)
        (stage/'REPORT.txt').write_text('RD04 daily-updated EPSS development forecasts\nNo test scoring.\n'+c.dump(findings))
        c.core.check_pins(ident['source_pins'])
    c.finish_lock(output,verify)
    print(f'RD04 {scale} '+('checkpoints, epoch zero and predictions verified' if verify else 'development comparison saved')+f' at {output}. No test scoring.')
