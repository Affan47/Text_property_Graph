"""Strict allowlist, validated v3 categories and training-fitted preprocessing."""
import math
import numpy as np
from ..forecast_structured_features import CVSS_FIELDS, METRICS, encode_cvss
from ..continuous_forecasting.data import require

URLS=('all_sources_urls','github_only_urls','non_github_urls')
FIELDS=(*CVSS_FIELDS,*URLS)


def parse(fields):
    require(set(fields)==set(FIELDS),'Unexpected structured fields, possibly outcomes')
    encoded,issues=encode_cvss({k:fields[k] for k in CVSS_FIELDS})
    valid=bool(encoded[-1]); vector=fields['cvss_vector']
    pairs=dict(x.split(':') for x in vector.split('/')[1:]) if valid else {}
    cats={k:pairs.get(k,'MISSING') for k in METRICS}
    cats['version']=vector.split('/')[0][5:] if valid else 'MISSING'
    score=fields['cvss_vector_score']
    nums={'cvss_score':float(score) if valid and type(score) in (int,float) and math.isfinite(score) and 0<=score<=10 else None}
    for name in URLS:
        values=fields[name]
        require(values is None or isinstance(values,list) and all(isinstance(v,str) for v in values),'Invalid reference list')
        nums['log1p_'+name]=math.log1p(len(set(values))) if values is not None else None
    return cats,nums,issues


def fit(rows):
    require(rows and all(r['split']=='train' for r in rows),'Preprocessing must use training only')
    parsed=[parse(r['structured']) for r in rows]; categories={}; numeric={}; names=[]
    for key in sorted(parsed[0][0]):
        categories[key]=sorted({p[0][key] for p in parsed}-{'MISSING'})+['MISSING','UNKNOWN']
        names += [f'{key}={v}' for v in categories[key]]
    for key in sorted(parsed[0][1]):
        observed=[p[1][key] for p in parsed if p[1][key] is not None]
        median=float(np.median(observed)) if observed else 0.
        values=np.asarray([p[1][key] if p[1][key] is not None else median for p in parsed])
        numeric[key]={'median':median,'mean':float(values.mean()),'scale':float(values.std()) or 1.}
        names += [key+'_standardized',key+'_observed']
    names+=['cvss_vector_observed','github_url_present','github_url_presence_observed']
    return {'categories':categories,'numeric':numeric,'feature_names':names,'fit_ids':sorted(r['cve'] for r in rows)}


def transform(fields,state):
    cats,nums,issues=parse(fields); result=[]
    for key in sorted(state['categories']):
        vocab=state['categories'][key]
        value=cats[key] if cats[key] in vocab else 'UNKNOWN'
        result += [float(value==v) for v in vocab]
    for key in sorted(state['numeric']):
        p=state['numeric'][key]
        value=nums[key]; result += [((value if value is not None else p['median'])-p['mean'])/p['scale'],float(value is not None)]
    github=fields['github_only_urls']
    result += [float(cats['version']!='MISSING'),float(bool(github)),float(github is not None)]
    require(len(result)==len(state['feature_names']) and np.isfinite(result).all(),'Bad structured vector')
    return result,issues
