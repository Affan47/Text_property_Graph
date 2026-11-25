"""Compact RD04 development series; reuse tested CF01 past-only tensor primitives."""
from datetime import date, timedelta
import gzip
import io
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..continuous_forecasting import data as core
from ..continuous_forecasting import controls, evaluation, runner
from ..consolidated import models, contracts

ROOT=core.ROOT
CONFIG=ROOT/'02_prediction/04_configs/05_reddit_studies/RD04_stage1_v1.json'
AUDIT=ROOT/'05_results/03_dataset_analysis/02_reddit_dataset'
LOCK=AUDIT/'08_rd01_source_lock_v1'
SAN=AUDIT/'04_premodeling_gates_v1/stage1_sanitization'
OUTPUT=ROOT/'05_results/07_reddit_studies/01_rd04_daily_v1'
read,dump,sha,digest,require=core.read,core.dump,core.sha,core.digest,core.require
eligible,prefix,requests,packets,batch,forward=core.eligible,core.prefix,core.requests,core.packets,core.batch,core.forward


def checked(directory):
    core.check_lock(directory)
    core.check_pins(read(directory/'artifact_lock.json').get('inputs',{}))


def json_rows(path):
    import json
    with gzip.open(path,'rt') as f:
        for line in f: yield json.loads(line)


def settings():
    p=read(CONFIG)
    require(p['schema']=='rd04_daily_stage1_v1' and p['seeds']==[17,29,43]
            and not p['test_access_enabled'] and not p['full_training_enabled'],'Unexpected RD04 stage contract')
    return p


def adapt(cve,publication,values,statuses,role,manifest):
    require(role in ('train','validation'),'Held-out adaptation is forbidden')
    lo,hi=manifest['publication_intervals'][role]
    require(lo<=publication<=hi,'Publication outside frozen role')
    pub=date.fromisoformat(publication)
    dates=[(pub+timedelta(days=d)).isoformat() for d in range(91)]
    require(set(values)==set(statuses)==set(dates),'Wrong 91-day date keys')
    allowed=[lo<=d<=min(hi,manifest['cutoff']) for d in dates]
    clean=[]; missing=[]
    for d,ok in zip(dates,allowed):
        if not ok:
            clean.append(None); missing.append('ROLE_DATE_EXCLUDED'); continue
        v=values[d]
        require(v is None or (type(v) in (int,float) and np.isfinite(v) and 0<=v<=1),'Invalid source score')
        require((v is not None)==(statuses[d]=='OBSERVED'),'Score/status disagreement')
        clean.append(v); missing.append(statuses[d])
    return {'cve':cve,'split':role,'publication_date':publication,'values':clean,
            'date_allowed':allowed,'availability_status':missing,
            'release_dates':sorted(manifest['release_schedule'].values())[1:]}


def partitions(row):
    mask=eligible(row); pub=date.fromisoformat(row['publication_date'])
    release_indices=np.asarray([(date.fromisoformat(d)-pub).days for d in row['release_dates']],dtype=int)
    t=np.arange(90)[:,None]; u=t+np.arange(1,91)[None,:]
    first=next((i for i,v in enumerate(row['values']) if v is not None),91)
    cross=np.zeros((90,90),dtype=bool); history=np.zeros_like(cross)
    for release in release_indices:
        cross|=(t<release)&(release<=u)
        history|=(first<release)&(release<=u)
    return {'anchor_target_crossing':mask&cross,'anchor_target_single_version':mask&~cross,
            'history_target_crossing':mask&history,'history_target_single_version':mask&~history}


def coverage(row):
    t=np.arange(90)[:,None]; u=t+np.arange(1,91)[None,:]; inside=u<=90
    obs=np.asarray([v is not None for v in row['values']]); allowed=np.asarray(row['date_allowed'])
    dates=inside&allowed[t]&allowed[np.minimum(u,90)]
    history=np.cumsum(obs)[:90,None]>0
    mats={'eligible':eligible(row),'date_excluded':inside&~dates,
          'no_history':dates&~history,'missing_target':dates&history&~obs[np.minimum(u,90)]}
    require(np.array_equal(sum(m.astype(int) for m in mats.values()),inside.astype(int)),'Coverage partition does not reconcile')
    return mats


def pins():
    paths=[CONFIG,LOCK/'artifact_lock.json',LOCK/'rd01_frozen_manifest.json',SAN/'artifact_lock.json']
    paths+=list(Path(__file__).parent.glob('*.py'))
    paths+=[Path(m.__file__) for m in (core,controls,evaluation,runner,models,contracts)]
    return {str(p):sha(p) for p in sorted(set(paths))}


def prepare(scale,verify=False):
    require(scale in ('full','smoke'),'Unknown scale')
    checked(LOCK); checked(SAN); manifest=read(LOCK/'rd01_frozen_manifest.json'); p=settings()
    require(not manifest['test_release_enabled'] and manifest['role_state']=='FROZEN','Manifest is not locked')
    task=manifest['tasks']['RD04']; ids={s:task['ids'][s] for s in ('train','validation')}
    require(digest(task['ids'])==task['role_ids_sha256'],'Frozen task IDs changed')
    roles={c:s for s,cs in ids.items() for c in cs}
    require(len(roles)==sum(map(len,ids.values())) and not set(roles)&set(task['ids']['test']),'Overlapping/forbidden roles')
    publications={r['cve']:r['published_date'] for r in tqdm(json_rows(SAN/'predictor_candidates.jsonl.gz'),total=44985,
        desc='Select frozen development publication dates',unit='CVE') if r['cve'] in roles}
    rows={s:[] for s in ids}; counts={s:{'cves':0,'primary_cves':0,'max_CVE_weight_error':0.,
        **{k:0 for k in ('eligible','date_excluded','no_history','missing_target','anchor_target_crossing',
                        'anchor_target_single_version','history_target_crossing','history_target_single_version')}} for s in ids}
    matrices={s:np.zeros((90,90),dtype=np.int64) for s in ids}
    seen=set()
    for raw in tqdm(json_rows(SAN/'outcomes_restricted.jsonl.gz'),total=44985,desc='Adapt development only; verify masks and boundaries',unit='CVE'):
        c=raw['cve']
        if c not in roles: continue
        require(c not in seen,'Duplicate CVE'); seen.add(c)
        s=roles[c]; epss=raw['epss_outcomes']
        row=adapt(c,publications[c],epss['epss_scores'],epss['epss_score_status'],s,manifest)
        mats=coverage(row); m=mats['eligible']; require(m.any(),'Frozen CVE has no eligible pairs')
        part=partitions(row)
        require(np.array_equal(part['anchor_target_crossing']|part['anchor_target_single_version'],m),'Version partitions differ')
        n_anchors=int(m.any(1).sum())
        mass=sum((1/n_anchors)/int(m[t].sum())*int(m[t].sum()) for t in np.flatnonzero(m.any(1)))
        require(abs(mass-1)<1e-12,'CVE does not have unit hierarchical loss mass')
        counts[s]['max_CVE_weight_error']=max(counts[s]['max_CVE_weight_error'],abs(mass-1))
        counts[s]['cves']+=1; counts[s]['primary_cves']+=int(m[30,29]); matrices[s]+=m
        for k,v in {**mats,**part}.items(): counts[s][k]+=int(v.sum())
        rows[s].append(row)
    require(seen==set(roles),'Missing development CVEs')
    for s in rows:
        rows[s].sort(key=lambda r:r['cve'])
        require(counts[s]['cves']==task['counts'][s] and counts[s]['eligible']==task['eligible_cells'][s],'Accepted counts differ')
    if scale=='smoke':
        for s in rows:
            chosen=set(core.choose(ids[s],p['smoke'][s+'_cves'])); rows[s]=[r for r in rows[s] if r['cve'] in chosen]
    source_pins=pins()
    identity={'schema':p['schema'],'scale':scale,'settings':p,'source_pins':source_pins,
        'ids':{s:[r['cve'] for r in rs] for s,rs in rows.items()},'manifest_sha256':sha(LOCK/'rd01_frozen_manifest.json')}
    summary={'full_preflight':counts,'selected':{s:{'cves':len(rs),'cells':sum(int(eligible(r).sum()) for r in rs),
        'primary_cves':sum(bool(eligible(r)[30,29]) for r in rs)} for s,rs in rows.items()},
        'test_adapted':False,'training':False,'rows_are_restricted_development_series':True,
        'input_contract':'Only prefix through t plus lead; labels separately assembled by batch(); out-of-role values are null.',
        'version_definition':p['version_partition'],'storage':'91 values per CVE, lazy 90x90 matrices per CVE; no global expanded window table.'}
    with core.transaction(OUTPUT/'preflight'/scale,verify) as stage:
        for name,value in [('run_identity.json',identity),('summary.json',summary),('coverage.json',{s:m.tolist() for s,m in matrices.items()})]:
            (stage/name).write_text(dump(value))
        with (stage/'series.json.gz').open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as f:
            f.write(dump(rows).encode())
        (stage/'REPORT.txt').write_text('RD04 development preflight\n'+dump(summary)+'\nNo training or test adaptation. Full-cohort counts are metadata, not model performance.\n')
        core.check_pins(source_pins)
    print('RD04 '+scale+' preflight '+('reproduced' if verify else 'saved')+'. No fitting or test adaptation.')


def load(scale):
    directory=OUTPUT/'preflight'/scale; core.check_lock(directory)
    identity=read(directory/'run_identity.json'); core.check_pins(identity['source_pins'])
    require(identity['settings']==settings() and identity['scale']==scale,'Preflight settings changed')
    import json
    with gzip.open(directory/'series.json.gz','rt') as f: rows=json.load(f)
    require(set(rows)=={'train','validation'},'Forbidden adapted roles')
    for s,rs in rows.items():
        require([r['cve'] for r in rs]==identity['ids'][s] and all(r['split']==s for r in rs),'Wrong row identities')
    return rows,identity
