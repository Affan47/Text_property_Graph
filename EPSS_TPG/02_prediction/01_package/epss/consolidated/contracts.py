"""Pure input/target, source-mask and training-only encoding contracts."""
from datetime import date, timedelta
import math
import re

import numpy as np

from .protocol import SOURCES, require, parent
from .. import forecast_structured_features as cvss


def primary_label(record, day=60):
    require(day == 60, 'Only the agreed Day-60 primary target is registered')
    pub = parent.contract.timestamp(record['published_timestamp']).date()
    require(record['published_date'] == pub.isoformat(), 'Publication fields disagree')
    stamp = (pub+timedelta(days=day)).isoformat()
    require(stamp in record['epss_scores'], 'Missing date key, not an observed zero')
    value = parent.contract.score(record['epss_scores'][stamp])
    return {'cve':record['cve'],'target_date':stamp,'target_day':day,'value':value,'observed':value is not None}


def text_documents(record, fields):
    docs = []
    for source in SOURCES:
        values = record.get(fields[source])
        if source == 'smp':
            require(isinstance(values,list),'Expected SMP list')
            items = [(v.get('post'),v.get('platform')) for v in values if isinstance(v,dict)]
        else:
            items = [(values,None)]
        seen = set()
        for text,platform in items:
            require(platform is None or isinstance(platform,str),'Platform must be text or null')
            if not isinstance(text,str) or not text.strip() or (text,platform) in seen:
                continue
            seen.add((text,platform))
            # Conservative disclosure screen, not a claim of exhaustive leakage detection.
            explicit = bool(re.search(r'\bEPSS\b.{0,80}\d|\d.{0,40}\bEPSS\b',text,re.I|re.S))
            docs.append({'source':source,'text':text,'platform':platform,
                         'quarantined_epss_disclosure':explicit,'historically_verified':False})
    return docs


def source_mask(documents, enabled):
    require(len(enabled) == len(set(enabled)) and set(enabled) <= set(SOURCES),'Unknown/repeated source')
    return [int(s in enabled and any(d['source']==s and not d['quarantined_epss_disclosure'] for d in documents)) for s in SOURCES]


def _count(value):
    return math.log1p(value) if type(value) in (float,int) and math.isfinite(value) and value>=0 and int(value)==value else None


class StructuredEncoder:
    """No label fields, train-only numeric scaling, fixed CVSS categories."""
    @staticmethod
    def fit(records):
        require(records and all(r['split']=='train' for r in records),'Fit preprocessing on training only')
        require(len({r['cve'] for r in records})==len(records),'Repeated training CVEs')
        require(all(set(r)<=set(cvss.CVSS_FIELDS)|{'cve','split','n_references_nvd'} for r in records),'Unexpected preprocessing fields')
        values = [v for r in records if (v := _count(r.get('n_references_nvd'))) is not None]
        mean = sum(values)/len(values) if values else 0.
        sd = math.sqrt(sum((v-mean)**2 for v in values)/len(values)) if values else 1.
        return {'schema':'cs01_structured_v1','reference_mean':mean,'reference_scale':sd or 1.,
                'fit_ids':[r['cve'] for r in records], 'feature_names':cvss.FEATURES+['reference_log1p_standardized','reference_observed']}

    @staticmethod
    def transform(fields, state, blocks):
        require(set(fields)==set(cvss.CVSS_FIELDS)|{'n_references_nvd'},'Unexpected fields, possibly labels/catalogue metadata')
        require(set(blocks)<= {'cvss','reference_count'},'Unknown structured block')
        vector, issues = cvss.encode_cvss({k:fields[k] for k in cvss.CVSS_FIELDS})
        if 'cvss' not in blocks:vector=[0.]*len(vector)
        value = _count(fields['n_references_nvd']) if 'reference_count' in blocks else None
        return vector+[(value-state['reference_mean'])/state['reference_scale'] if value is not None else 0.,float(value is not None)],issues


def fit_multihot(values, split='train'):
    require(split=='train','Fit vocabulary on training only')
    require(all(v is None or isinstance(v,list) and all(isinstance(s,str) for s in v) for v in values),'Expected lists of categories')
    return sorted({s for v in values if v for s in v})


def encode_multihot(value, vocabulary):
    require(vocabulary == sorted(set(vocabulary)), 'Invalid vocabulary')
    require(value is None or isinstance(value,list) and all(isinstance(s,str) for s in value),'Expected category list or null')
    chosen = set(value or [])
    return [int(s in chosen) for s in vocabulary]+[int(value is None),int(bool(chosen-set(vocabulary)))]


def past_channels(values, anchor):
    require(0 <= anchor <= 89 and len(values)>=anchor+1,'Invalid prefix length')
    result, last = [], None
    for j in range(anchor+1):
        v = parent.contract.score(values[j])
        if v is not None:last=j
        result.append([v if v is not None else 0.,int(v is not None),
                       (j-last)/90 if last is not None else 0.,int(last is not None),j/90])
    require(last is not None,'No observed past score')
    return result


def temporal_coverage(record, split, policy):
    """Count all date-eligible t/h pairs; never inspect numerical values beyond the role boundary."""
    require(split in ('train','validation'),'Test series forbidden in development')
    parent.contract.validated_split_policy(policy)
    pub = parent.contract.timestamp(record['published_timestamp']).date()
    require(record['published_date'] == pub.isoformat(),'Publication mismatch')
    dates = [(pub+timedelta(days=d)).isoformat() for d in range(91)]
    require(set(record['epss_scores'])==set(dates),'Expected exactly Day 0-90 keys')
    upper = policy['validation_start'] if split=='train' else policy['test_start']
    lower = None if split=='train' else policy['validation_start']
    values = [parent.contract.score(record['epss_scores'][stamp]) if stamp < upper and
              (policy['observation_end'] is None or stamp<=policy['observation_end']) else None for stamp in dates]
    observed = np.array([v is not None for v in values],dtype=bool)
    t = np.arange(90)[:,None];h=np.arange(1,91)[None,:];end=t+h
    inside = end<=90
    end_clamped = np.minimum(end,90)
    date_ok = np.array([stamp < upper and (lower is None or stamp>=lower) and
                       (policy['observation_end'] is None or stamp<=policy['observation_end']) for stamp in dates])
    admissible = inside & date_ok[t] & date_ok[end_clamped]
    history_ok = np.cumsum(observed)[:90,None]>0
    eligible = admissible & history_ok & observed[end_clamped]
    no_history = admissible & ~history_ok
    missing_target = admissible & history_ok & ~observed[end_clamped]
    return {'eligible':eligible,'date_admissible':admissible,'missing_target':missing_target,'no_history':no_history,
            'anchor_count':int(eligible.any(axis=1).sum()),'pair_count':int(eligible.sum()),'values':values}
