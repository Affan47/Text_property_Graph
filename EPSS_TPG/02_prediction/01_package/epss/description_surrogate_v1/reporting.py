"""Post-fit diagnostics only; crossing labels never enter fitting or selection."""
import csv
import io
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .. import early_warning_gap_review_v1 as gap
from . import data


def csv_text(rows):
    if not rows:
        return ''
    stream=io.StringIO(newline='')
    fields=sorted({k for r in rows for k in r})
    writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
    for r in sorted(rows,key=lambda x:tuple(str(x.get(k,'')) for k in fields)):
        writer.writerow(r)
    return stream.getvalue()


def regression(y,p):
    d=np.asarray(p)-np.asarray(y)
    return {'n':len(y),'mae':float(np.abs(d).mean()),'rmse':float(np.sqrt(np.mean(d*d))),
            'signed_bias':float(d.mean())}


def seed_mean(predictions,counts):
    # One seed draw spans ALL folds, preserving the paired seed identity.
    return np.asarray(predictions) @ (np.asarray(counts,dtype=float)/np.sum(counts))


def intervals(y,anchor,pred,groups,repetitions,seed):
    if len(y)==0 or len(np.unique(y))<2:
        return []
    _,g=np.unique(groups,return_inverse=True);ng=int(g.max())+1;ns=pred.shape[1]
    rg,rs=[np.random.default_rng(x) for x in np.random.SeedSequence(seed).spawn(2)]
    mean=pred.mean(axis=1);fixed=gap.logit(mean)+gap.logit(anchor)
    point_c=gap.old.metrics(y,fixed);point_r=gap.old.metrics(y,anchor)
    samples={'group_fixed_fits':[],'group_and_seed':[]}
    for start in tqdm(range(0,repetitions,32),desc='Paired group and seed resampling',unit='batch',leave=False):
        size=min(32,repetitions-start)
        weights=rg.multinomial(ng,np.full(ng,1/ng),size=size)[:,g].astype(float)
        draws=rs.multinomial(ns,np.full(ns,1/ns),size=size)
        ref=np.stack(gap.weighted_metrics(y,anchor,weights),axis=1)
        conditional=np.stack(gap.weighted_metrics(y,fixed,weights),axis=1)
        joint=[]
        for i,counts in enumerate(draws):
            score=gap.logit(seed_mean(pred,counts))+gap.logit(anchor)
            joint.append(np.array(gap.weighted_metrics(y,score,weights[i:i+1]))[:,0]-ref[i])
        samples['group_fixed_fits'].append(conditional-ref)
        samples['group_and_seed'].append(np.array(joint))
    rows=[]
    for unit,chunks in samples.items():
        values=np.concatenate(chunks)
        for j,metric in enumerate(('auroc','average_precision')):
            valid=values[:,j][np.isfinite(values[:,j])]
            lo,hi=np.quantile(valid,[.025,.975]) if len(valid) else (None,None)
            rows.append({'unit':unit,'metric':metric,'candidate':'description_plus_epss',
                         'reference':'past_epss','difference':point_c[metric]-point_r[metric],
                         'low':None if lo is None else float(lo),'high':None if hi is None else float(hi),
                         'draws':repetitions,'valid_draws':len(valid),'groups':ng,'seeds':ns})
    return rows


def decision(rows,smoke):
    if smoke:return 'SMOKE_ONLY_NO_SCIENTIFIC_DECISION'
    r=next(r for r in rows if r['policy']=='isolated_day_assumption' and
           r['unit']=='group_and_seed' and r['metric']=='average_precision')
    return ('EXPLORATORY_SIGNAL_REQUIRES_NEW_HISTORICAL_PROTOCOL' if
            r['difference']>0 and r['low'] is not None and r['low']>0 else
            'STOP_THIS_SURROGATE_BRANCH_INCONCLUSIVE_OR_NO_GAIN')


def analyze(rows,p,scale):
    s=p['settings'];seeds=s['smoke']['seeds'] if scale=='smoke' else s['seeds']
    models=sorted({r['model'] for r in rows});metrics=[]
    for name in models:
        for seed in sorted({r['seed'] for r in rows if r['model']==name}):
            rr=sorted([r for r in rows if r['model']==name and r['seed']==seed],key=lambda r:r['cve'])
            for fold in (0,1,2,3):
                block=rr if fold==0 else [r for r in rr if r['fold']==fold]
                if block:metrics.append({'model':name,'seed':seed,'fold':fold,
                                          **regression([r['actual'] for r in block],[r['prediction'] for r in block])})
    # The Day7 plan is first parsed here, AFTER every required fit is locked.
    plan_path=next(k for k in p['input_hashes'] if k.endswith('/plan/plan.json'))
    plan=data.a.read(Path(plan_path))
    lookup={(r['cve'],r['seed']):r for r in rows if r['model']=='description_only'}
    all_ids=sorted({c for c,_ in lookup})
    requested=set(plan['assessment_ids']);ids=[c for c in all_ids if c in requested]
    if scale=='full':
        data.a.require(len(all_ids)==8053 and len(ids)==7957,'Unexpected assessment coverage')
    ew=[];paired=[];predictions=[];supports=[]
    for policy in ('isolated_day_assumption','strict'):
        key='label' if policy=='isolated_day_assumption' else 'strict_label'
        chosen=[c for c in ids if plan['records'][c]['label_not_predictor'][key] is not None]
        labels=np.array([plan['records'][c]['label_not_predictor'][key] for c in chosen],dtype=int)
        anchor=np.array([plan['records'][c]['label_not_predictor']['anchor_score'] for c in chosen])
        pp=np.array([[lookup[c,seed]['prediction'] for seed in seeds] for c in chosen],dtype=float).reshape(len(chosen),len(seeds))
        folds=np.array([lookup[c,seeds[0]]['fold'] for c in chosen])
        groups=np.array([p['component_by_cve'][c] for c in chosen])
        supports.append({'policy':policy,'cves':len(chosen),'risers':int(labels.sum()),'unresolved_or_ineligible':len(ids)-len(chosen)})
        if scale=='full':
            data.a.require((len(chosen),int(labels.sum()))==((7797,62) if policy=='isolated_day_assumption' else (3516,62)),
                           'Locked label policy counts changed')
        if not len(chosen):continue
        scores={'past_epss':anchor,'description_only':pp.mean(axis=1),
                'description_plus_epss':gap.logit(pp.mean(axis=1))+gap.logit(anchor)}
        for j,seed in enumerate(seeds):
            scores[f'description_plus_epss_seed_{seed}']=gap.logit(pp[:,j])+gap.logit(anchor)
        for name,values in scores.items():
            for fold in (0,1,2,3):
                mask=np.ones(len(chosen),dtype=bool) if fold==0 else folds==fold
                if mask.any():ew.append({'policy':policy,'model':name,'fold':fold,'cves':int(mask.sum()),
                                         'risers':int(labels[mask].sum()),**gap.old.metrics(labels[mask],values[mask])})
        for i,c in enumerate(chosen):
            predictions.append({'policy':policy,'cve':c,'fold':int(folds[i]),'component':str(groups[i]),
                                'label':int(labels[i]),**{k:float(v[i]) for k,v in scores.items()}})
        reps=s['smoke']['bootstrap_repetitions'] if scale=='smoke' else s['bootstrap_repetitions']
        paired += [dict(policy=policy,**r) for r in intervals(labels,anchor,pp,groups,reps,s['bootstrap_seed'])]
    return {'day60_metrics':metrics,'early_warning_metrics':ew,'paired_intervals':paired,'supports':supports,
            'decision':decision(paired,scale=='smoke'),'early_warning_predictions':predictions,
            'historical_forecast':False,'new_independent_evidence':False}
