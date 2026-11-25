"""Screened RD02 development export: separate predictors, labels and audit metadata."""
import csv
from datetime import date,timedelta
import gzip
import hashlib
import json
from pathlib import Path
import sys
from tqdm import tqdm
from ..reddit_forecasting import data as rd
from . import structured

ROOT=rd.ROOT; read,dump,sha,digest,require=rd.read,rd.dump,rd.sha,rd.digest,rd.require
OUT=ROOT/'05_results/07_reddit_studies/02_rd02_primary_v1'
REVIEW=rd.AUDIT/'10_rd02_overlap_review_v1'
CONFIG=ROOT/'02_prediction/04_configs/05_reddit_studies/RD02_registered_v2.json'
SPEC=CONFIG.with_name('RD02_spec.json')
SCRIPTS=ROOT/'02_prediction/03_analysis/06_reddit_dataset'
sys.path.insert(0,str(SCRIPTS))
from audit_reddit import fingerprint


def write_gz(path,value):
    with path.open('wb') as f,gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0) as z:
        z.write(dump(value).encode())


def read_gz(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def validation_partitions():
    with (REVIEW/'validation_overlap_partitions.csv').open() as f:
        result={r['cve']:r['partition'] for r in csv.DictReader(f)}
    require(sum(v=='overlap_with_train' for v in result.values())==29 and len(result)==972,'Overlap partition changed')
    return result


def choose(rows,partitions,n):
    ordered=sorted(rows,key=lambda r:r['cve']); selected=[]
    predicates=[lambda r:not r['mask_M_prose'],lambda r:r['mask_M_prose'],
                lambda r:partitions.get(r['cve'])=='overlap_with_train',
                lambda r:partitions.get(r['cve'])=='no_detected_overlap_with_train']
    for pred in predicates:
        found=next((r for r in ordered if pred(r) and r['cve'] not in selected),None)
        if found:selected.append(found['cve'])
    selected+= [r['cve'] for r in ordered if r['cve'] not in selected][:n-len(selected)]
    return sorted(selected)


def prepare(scale,verify=False):
    require(scale in ('gate','smoke','full'),'Unknown export scale')
    rd.checked(rd.LOCK);rd.checked(rd.SAN);rd.core.check_lock(REVIEW)
    approval=read(REVIEW/'source_approval.json');rd.core.check_pins(approval['source_pins'])
    manifest=read(rd.LOCK/'rd01_frozen_manifest.json');task=manifest['tasks']['RD02']
    require(approval['source_gate']=='ACCEPTED_WITH_DECLARED_DESCRIPTION_OVERLAP' and
            approval['original_manifest_sha256']==sha(rd.LOCK/'rd01_frozen_manifest.json') and
            approval['role_ids_sha256']==digest(task['ids'])==task['role_ids_sha256'] and
            not manifest['test_release_enabled'],'Unapproved/changed RD02 roles')
    roles={c:s for s in ('train','validation') for c in task['ids'][s]}; decisions={}
    with gzip.open(rd.LOCK/'source_document_decisions.csv.gz','rt') as f:
        for r in csv.DictReader(f):
            if r['task']=='RD02' and r['cve'] in roles:
                decisions[(r['cve'],r['source'],r['text_sha256'],r['post_fingerprint'])]=r
    rows=[]; dates={};counts={s:{'D':0,'M_all':0,'M_prose':0,'cves_prose':0} for s in ('train','validation')}
    for raw in tqdm(rd.json_rows(rd.SAN/'predictor_candidates.jsonl.gz'),total=44985,desc='Export screened development predictors',unit='CVE'):
        c=raw['cve']
        if c not in roles:continue
        s=roles[c];docs=[];dates[c]=raw['published_date']
        candidates=[('D',raw['description'],'')]+[('M',p['post'],fingerprint(p)) for p in raw['social_media_posts']]
        for source,text,fp in candidates:
            if not isinstance(text,str) or not text.strip():continue
            h=hashlib.sha256(text.encode()).hexdigest();d=decisions[(c,source,h,fp)]
            if d['quarantine'].lower()=='true':continue
            prose=source=='M' and d['bare_id'].lower()!='true'
            docs.append({'document_id':source+'-'+(fp or h),'source':source,'text':text,'text_sha256':h,'is_prose':prose})
            counts[s]['D' if source=='D' else 'M_all']+=1
            counts[s]['M_prose']+=int(prose)
        prose=any(d['is_prose'] for d in docs);counts[s]['cves_prose']+=int(prose)
        row={'cve':c,'split':s,'structured':{k:raw.get(k) for k in structured.FIELDS},'documents':docs,
             'mask_D':any(d['source']=='D' for d in docs),'mask_M_all':any(d['source']=='M' for d in docs),'mask_M_prose':prose}
        require(row['mask_D'] and len({d['document_id'] for d in docs})==len(docs),'Missing D/duplicate attachment')
        rows.append(row)
    require(counts=={'train':{'D':24213,'M_all':26578,'M_prose':3309,'cves_prose':2077},
                     'validation':{'D':972,'M_all':1289,'M_prose':1283,'cves_prose':968}},'Frozen source counts changed')
    require({r['cve'] for r in rows}==set(roles) and len(rows)==len(roles),'Missing/repeated development CVE')
    partitions=validation_partitions(); full_ids={s:sorted(c for c in roles if roles[c]==s) for s in ('train','validation')}
    ids=full_ids if scale=='full' else {s:choose([r for r in rows if r['split']==s],partitions,
                    32 if scale=='gate' and s=='train' else 12 if s=='train' else 6) for s in full_ids}
    if scale=='gate':ids['validation']=[]
    selected={c for cs in ids.values() for c in cs};rows=sorted([r for r in rows if r['cve'] in selected],key=lambda r:r['cve'])
    targets={}; audit={};releases=sorted(manifest['release_schedule'].values())[1:]
    for raw in tqdm(rd.json_rows(rd.SAN/'outcomes_restricted.jsonl.gz'),total=44985,desc='Export observed development Day-60 labels separately',unit='CVE'):
        c=raw['cve']
        if c not in selected:continue
        pub=dates[c];target=(date.fromisoformat(pub)+timedelta(days=60)).isoformat();s=roles[c]
        require(manifest['publication_intervals'][s][0]<=pub<=target<=manifest['publication_intervals'][s][1],'Target crosses role boundary')
        e=raw['epss_outcomes'];v=e['epss_scores'][target]
        require(type(v) in (float,int) and 0<=v<=1 and e['epss_score_status'][target]=='OBSERVED','Missing/invalid target')
        targets[c]={'target':v,'target_date':target,'target_day':60}
        audit[c]={'split':s,'overlap_partition':partitions.get(c),'local_release_crossing':any(pub<d<=target for d in releases),
                  'whole91day_version_changed':e['epss_version_changed']}
    require(set(targets)==selected,'Incomplete labels')
    state=structured.fit([r for r in rows if r['split']=='train'])
    vectors={r['cve']:structured.transform(r['structured'],state)[0] for r in rows}
    reloaded=json.loads(dump(state))
    require(all(structured.transform(r['structured'],reloaded)[0]==vectors[r['cve']] for r in rows),
            'Structured state JSON round-trip changes column order')
    pins={str(p):sha(p) for p in [CONFIG,SPEC,REVIEW/'artifact_lock.json',rd.LOCK/'artifact_lock.json',rd.SAN/'artifact_lock.json',
                                Path(__file__),Path(structured.__file__),SCRIPTS/'audit_reddit.py']}
    identity={'schema':'rd02_development_export_v1','scale':scale,'ids':ids,'source_pins':pins,
              'role_ids_sha256':task['role_ids_sha256'],'approval_sha256':sha(REVIEW/'source_approval.json'),
              'interpretation':'Retrospective current D/M/CVSS; no claim of availability by Day 60','test_adapted':False}
    summary={'full_source_counts':counts,'selected_cves':{s:len(cs) for s,cs in ids.items()},
             'structured_features':len(state['feature_names']),'validation_full_partitions':{'overlap_with_train':29,'no_detected_overlap_with_train':943},
             'source_policy':'M_prose selects from M_all; no additional CVE or URL removals','test_adapted':False,'training':False}
    with rd.core.transaction(OUT/'export'/scale,verify) as stage:
        write_gz(stage/'predictor_inputs.json.gz',rows)
        for name,value in [('labels.json',targets),('audit_not_model_input.json',audit),('structured_state.json',state),
                           ('structured_vectors.json',vectors),('run_identity.json',identity),('summary.json',summary)]:
            (stage/name).write_text(dump(value))
        (stage/'REPORT.txt').write_text('RD02 screened development export\n'+dump(summary)+'\nNo test records exported. Labels/audit are not model inputs.\n')
        rd.core.check_pins(pins)
    print('RD02 '+scale+' export '+('reproduced' if verify else 'saved')+'. No training or test adaptation.')


def load(scale,labels=False):
    directory=OUT/'export'/scale;rd.core.check_lock(directory);identity=read(directory/'run_identity.json')
    rd.core.check_pins(identity['source_pins']);rows=read_gz(directory/'predictor_inputs.json.gz')
    require({r['cve'] for r in rows}=={c for v in identity['ids'].values() for c in v} and
            all(r['split'] in ('train','validation') for r in rows),'Wrong development cohort')
    for r in rows:require(set(r)=={'cve','split','structured','documents','mask_D','mask_M_all','mask_M_prose'},'Unexpected predictor fields')
    return rows,identity,read(directory/'labels.json') if labels else None
