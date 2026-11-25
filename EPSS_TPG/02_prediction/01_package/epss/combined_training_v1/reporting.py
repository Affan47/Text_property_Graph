"""Matched regression metrics, predefined source slices and development figures."""
import csv
import io
import numpy as np
from scipy.stats import spearmanr
from ..reddit_development_v1.common import compare_ci
from .data import require, c


def metrics(y, prediction):
    y=np.asarray(y,dtype=np.float64);raw=np.asarray(prediction,dtype=np.float64)
    require(y.shape==raw.shape and np.isfinite(y).all() and np.isfinite(raw).all(),'Invalid evaluation vectors')
    names=('mae','rmse','r2','pearson','spearman','signed_bias','raw_mae','raw_rmse','clipped_fraction')
    if not len(y):return {'n':0,**dict.fromkeys(names)}
    p=raw.clip(0,1);e=p-y;variable=len(y)>1 and np.ptp(y)>0 and np.ptp(p)>0
    return {'n':len(y),'mae':float(abs(e).mean()),'rmse':float(np.sqrt(np.mean(e**2))),
        'r2':float(1-np.sum(e**2)/np.sum((y-y.mean())**2)) if len(y)>1 and np.ptp(y)>0 else None,
        'pearson':float(np.corrcoef(y,p)[0,1]) if variable else None,
        'spearman':float(spearmanr(y,p).statistic) if variable else None,
        'signed_bias':float(e.mean()),'raw_mae':float(abs(raw-y).mean()),
        'raw_rmse':float(np.sqrt(np.mean((raw-y)**2))),'clipped_fraction':float((raw!=p).mean())}


def slices(ctx,view):
    ids=ctx['ids']['validation'];parts={'all':ids}
    if view!='S':
        lookup={cid:ctx['index'][cid]['views'][view] for cid in ids}
        for name,condition in (
            ('text_present',lambda r:bool(r['object_key'])),
            ('text_absent',lambda r:not r['object_key']),
            ('M_present',lambda r:r['source_mask']['M']),
            ('M_absent',lambda r:not r['source_mask']['M']),
            ('U_present',lambda r:r['source_mask']['G'] or r['source_mask']['N']),
            ('U_absent',lambda r:not(r['source_mask']['G'] or r['source_mask']['N']))):
            parts[name]=[cid for cid in ids if condition(lookup[cid])]
    for name,value in (('training_description_overlap','overlap_with_train'),('no_training_description_overlap','no_detected_overlap_with_train')):
        if ctx['audit']:parts[name]=[cid for cid in ids if ctx['audit'][cid]['overlap_partition']==value]
    if ctx['audit']:
        for flag in ('local_release_crossing','whole91day_version_changed'):
            for value in (True,False):parts[flag+'_'+str(value)]=[cid for cid in ids if ctx['audit'][cid][flag]==value]
    parts['target_below_0.01']=[cid for cid in ids if ctx['labels'][cid]['value']<.01]
    parts['target_at_least_0.1']=[cid for cid in ids if ctx['labels'][cid]['value']>=.1]
    return parts


def evaluate(ctx,view,values):
    ids=ctx['ids']['validation'];require(len(values)==len(ids),'Prediction count mismatch')
    rows=[{'cve':cid,'split':'validation','target_date':ctx['labels'][cid]['date'],
           'actual':ctx['labels'][cid]['value'],'raw_prediction':float(p),'prediction':float(np.clip(p,0,1))}
          for cid,p in zip(ids,values)]
    lookup={r['cve']:r for r in rows}
    result={name:metrics([lookup[c]['actual'] for c in cs],[lookup[c]['raw_prediction'] for c in cs])
            for name,cs in slices(ctx,view).items()}
    return {'predictions':rows,'metrics':result}


def csv_text(rows):
    stream=io.StringIO(newline='');columns=sorted({k for r in rows for k in r})
    writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n');writer.writeheader()
    for row in sorted(rows,key=lambda r:c.encoded(r)):writer.writerow(row)
    return stream.getvalue()


def summary(ctx,results,settings,variants,scale):
    means={}
    for representation,view in [('structured','S')]+[(r,v) for v in variants for r in ('pooled','tpg')]:
        key=representation+'/'+view
        measures=[results[f'{key}/seed_{s}']['metrics']['all'] for s in settings['seeds']]
        means[key]={k:None if any(m[k] is None for m in measures) else float(np.mean([m[k] for m in measures])) for k in measures[0]}
    pairs=[]
    if scale=='full':
        for view in variants:
            for partition,ids in slices(ctx,view).items():
                if not ids:continue
                for reference in ('pooled/'+view,'structured/S','training_median'):
                    for metric in ('mae','rmse'):
                        errors=[]
                        for name in ('tpg/'+view,reference):
                            rows=[]
                            for seed in settings['seeds']:
                                key=name if name=='training_median' else name+f'/seed_{seed}'
                                lookup={r['cve']:r for r in results[key]['predictions']}
                                err=np.array([lookup[c]['prediction']-lookup[c]['actual'] for c in ids])
                                rows.append(abs(err) if metric=='mae' else err**2)
                            errors.append(rows)
                        pairs.append({'view':view,'partition':partition,'reference':reference,'metric':metric,
                            **compare_ci(*errors,metric,repetitions=settings['bootstrap_repetitions'],seed=settings['bootstrap_seed'])})
    return {'dataset':ctx['provenance']['dataset'],'scale':scale,'task':'Retrospective Day-60 EPSS regression',
        'test_scoring':False,'cohort':{s:len(v) for s,v in ctx['ids'].items()},'seed_mean_metrics':means,
        'controls':{k:results[k]['metrics']['all'] for k in ('training_mean','training_median','structured_ridge')},
        'coverage':ctx['coverage'],'paired_comparisons':pairs,
        'limitations':['Smoke results test software only, not predictive quality.',
            'Current text and CVSS are retrospective, not verified historical Day-60 inputs.',
            'Zero validation summaries in Reddit blocks its full U/DU/DMU comparison.',
            'Source absence keeps the CVE with an explicit mask; no substitute source or invented target.',
            'Repeated validation use is development, not independent final evidence.',
            'Regression metrics are not classification F1/accuracy; VulnCheck labels are not used.']}


def figures(directory,ctx,findings):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(findings['controls'])+list(findings['seed_mean_metrics'])
    rows={**findings['controls'],**findings['seed_mean_metrics']}
    fig,ax=plt.subplots(figsize=(10,max(4,len(names)*.3)))
    ax.barh(names,[rows[k]['mae'] for k in names],color=['#666666' if k in findings['controls'] else '#216b88' for k in names])
    ax.invert_yaxis();ax.set_xlabel('Validation MAE in EPSS score units (lower is better)')
    ax.set_title('Matched Day-60 estimation: neural bars average seed metrics')
    fig.tight_layout();fig.savefig(directory/'validation_mae.png',dpi=150);plt.close(fig)
    views=list(ctx['coverage']['train']);matrix=np.array([[ctx['coverage'][s][v]['nonempty']/ctx['coverage'][s][v]['cves'] for v in views] for s in ('train','validation')])
    fig,ax=plt.subplots(figsize=(9,3));im=ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGnBu')
    ax.set_xticks(range(len(views)),views);ax.set_yticks([0,1],['Training','Validation'])
    for i in range(2):
        for j in range(len(views)):ax.text(j,i,f'{100*matrix[i,j]:.1f}%',ha='center',va='center',color='white' if matrix[i,j]>.65 else 'black')
    ax.set_title('CVEs with any usable text in each combined variant')
    fig.colorbar(im,ax=ax,label='Fraction with a graph; this does not mean every requested source is present')
    fig.tight_layout();fig.savefig(directory/'source_coverage.png',dpi=150);plt.close(fig)
