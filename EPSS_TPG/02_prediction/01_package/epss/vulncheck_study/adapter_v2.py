"""Separate source-filtered predictors and binary labels; development roles only."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import tempfile

from tqdm import tqdm
from . import policy_v2 as policy

ROOT=Path(__file__).resolve().parents[4]
CONFIG=ROOT/'02_prediction/04_configs/02_vulncheck/02_adapter_smoke_v2.json'
SCRIPTS=ROOT/'02_prediction/03_analysis/03_vulncheck_study'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(path.read_text())
def dump(value):return json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
def digest(value):return hashlib.sha256(dump(value).encode()).hexdigest()


def check_lock(directory):
    lock=read(directory/'artifact_lock.json')
    require({str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}==set(lock['files'])|{'artifact_lock.json'},'Unexpected artifact set: '+str(directory))
    for name,h in lock['files'].items():
        path=(directory/name).resolve()
        require(directory.resolve() in path.parents and sha(path)==h,'Changed/unsafe artifact: '+name)
    return lock


def check_pins(pins):
    for name,h in pins.items():require(sha(Path(name))==h,'Changed input: '+name)


def config():
    p=read(CONFIG)
    require(p['schema']=='vk02_snapshot_adapter_smoke_v2' and p['source_policy']==policy.VERSION,'Wrong policy/configuration')
    require(not p['evaluation']['full_training_enabled'],'Full training is not enabled')
    return p


def code_pins():
    files=[CONFIG,Path(__file__),Path(policy.__file__),Path(__file__).with_name('classifier_smoke_v2.py'),
           SCRIPTS/'03_prepare_and_smoke_v2.py',SCRIPTS/'test_vulncheck_adapter_v2.py',
           ROOT/'02_prediction/01_package/epss/consolidated/contracts.py',
           ROOT/'02_prediction/01_package/epss/forecast_structured_features.py']
    return {str(p):sha(p) for p in files}


def select_ids(records,manifest,p,scale):
    require(scale in ('smoke','full'),'Unknown scale')
    by_id={r['cve']:r for r in records}
    require(len(by_id)==len(records) and set(by_id)==set(manifest),'Duplicate or unmatched CVE IDs')
    selected={}
    for role in ('train','validation'):
        eligible=sorted(c for c,m in manifest.items() if m['split']==role and type(by_id[c].get('in_vckev')) is bool)
        if scale=='full':selected[role]=eligible;continue
        n=p['smoke']['counts'][role]
        positives=[c for c in eligible if by_id[c]['in_vckev']]
        negatives=[c for c in eligible if not by_id[c]['in_vckev']]
        require(len(positives)>=2 and len(negatives)>=2 and len(eligible)>=n,'Insufficient two-class smoke cohort')
        npos=min(len(positives),n-2,max(2,round(n*len(positives)/len(eligible))))
        npos=max(npos,n-len(negatives))
        require(2<=npos<=n-2,'Insufficient smoke classes')
        rng=random.Random(p['smoke']['seed']+(0 if role=='train' else 1))
        selected[role]=sorted(rng.sample(positives,npos)+rng.sample(negatives,n-npos))
    require(not set(selected['train'])&set(selected['validation']),'Roles overlap')
    return selected


def documents(record,p):
    for source in p['source_order']:
        value=record.get(p['source_fields'][source])
        if source=='smp':
            require(value is None or isinstance(value,list),'SMP must be list/null')
            items=[]
            for row in value or []:
                require(isinstance(row,dict),'SMP item must be an object')
                items.append((row.get('post'),row.get('platform')))
        else:items=[(value,None)]
        seen=set()
        for text,platform in items:
            require(text is None or isinstance(text,str),'Nontext document')
            require(platform is None or isinstance(platform,str),'Invalid platform')
            if not text or not text.strip() or (text,platform) in seen:continue
            seen.add((text,platform))
            identity={'source':source,'text':text,'platform':platform}
            yield {'document_id':source+':'+digest(identity),'source_text_type':source,'text':text}


def validate_predictor(row,p,dim):
    require(set(row)=={'cve','split','documents','source_mask','structured'},'Unexpected predictor keys, possibly target metadata')
    require(row['split'] in ('train','validation'),'Test/excluded predictor export forbidden')
    require(len(row['structured'])==dim and all(type(v) in (int,float) and math.isfinite(v) for v in row['structured']),'Invalid structured vector')
    require(len(row['source_mask'])==4 and all(type(v) is int and v in (0,1) for v in row['source_mask']),'Invalid source mask')
    seen=set()
    for doc in row['documents']:
        require(set(doc)=={'document_id','source_text_type','text'},'Unexpected document attributes')
        require(doc['source_text_type'] in p['source_order'] and isinstance(doc['text'],str) and doc['text'].strip(),'Invalid text/source')
        require(doc['document_id'] not in seen,'Duplicate document');seen.add(doc['document_id'])
        require(policy.inspect(doc['text'],row['cve'],doc['source_text_type'])['decision']=='accept','Quarantined text leaked into predictors')
    require(row['source_mask']==[int(any(d['source_text_type']==s for d in row['documents'])) for s in p['source_order']],'Source mask disagrees with accepted documents')


def publish(directory,artifacts,pins,verify=False):
    check_pins(pins)
    if verify:
        check_lock(directory)
        require(set(artifacts)==set(read(directory/'artifact_lock.json')['files']),'Artifact names changed')
        for name,v in artifacts.items():
            expected=v if isinstance(v,str) else dump(v)
            require((directory/name).read_text()==expected,'Export differs: '+name)
        return
    require(not directory.exists(),'Output exists; use --verify-only')
    directory.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.vk02-',dir=directory.parent))
    try:
        for name,v in tqdm(artifacts.items(),desc='Save separate predictors, labels and audit decisions',unit='file'):
            (stage/name).write_text(v if isinstance(v,str) else dump(v))
        (stage/'artifact_lock.json').write_text(dump({'files':{name:sha(stage/name) for name in artifacts}}))
        check_pins(pins);require(not directory.exists(),'Output appeared; refuse overwrite');stage.rename(directory)
    finally:
        if stage.exists():shutil.rmtree(stage)


def prepare(scale='smoke',verify=False):
    from ..consolidated.contracts import StructuredEncoder
    p=config();audit=ROOT/p['audit'];review=ROOT/p['review'];source=ROOT.parent/p['source']
    with tqdm(total=3,desc='Verify frozen VK01 inputs',unit='stage') as bar:
        check_lock(audit);check_lock(review);bar.update()
        old=read(audit/'run_identity.json')
        for name,h in old['pins'].items():require(sha(ROOT.parent/name)==h,'Changed original audit dependency')
        require(sha(source)==p['source_sha256']==old['source_sha256'],'Source identity mismatch');bar.update()
        records=read(source);manifest=read(ROOT/p['manifest']);ids=select_ids(records,manifest,p,scale);bar.update()
    selected=set(ids['train']+ids['validation']);raw={r['cve']:r for r in records if r['cve'] in selected}
    pins={**code_pins(),str(source):sha(source),str(ROOT/p['manifest']):sha(ROOT/p['manifest']),
          str(audit/'artifact_lock.json'):sha(audit/'artifact_lock.json'),str(review/'artifact_lock.json'):sha(review/'artifact_lock.json')}
    staged=[];labels=[];decisions=[];issues=[]
    for split in ('train','validation'):
        for cve in tqdm(ids[split],desc='Apply v2 policy to '+split+' sources',unit='CVE'):
            r=raw[cve];require(type(r.get('in_vckev')) is bool,'Missing/nonboolean label')
            accepted=[]
            for doc in documents(r,p):
                decision=policy.inspect(doc['text'],cve,doc['source_text_type'])
                decisions.append({'cve':cve,'split':split,'document_id':doc['document_id'],
                                  'source_text_type':doc['source_text_type'],'text_sha256':hashlib.sha256(doc['text'].encode()).hexdigest(),**decision})
                if decision['decision']=='accept':accepted.append(doc)
            require(accepted,'No remaining sources; do not silently drop a CVE')
            staged.append({'cve':cve,'split':split,'documents':accepted,
                           'source_mask':[int(any(d['source_text_type']==s for d in accepted)) for s in p['source_order']],
                           'structured_raw':{k:r.get(k) for k in p['structured_fields']}})
            labels.append({'cve':cve,'split':split,'target':int(r['in_vckev'])})
    fit=[{'cve':r['cve'],'split':'train',**r['structured_raw']} for r in staged if r['split']=='train']
    state=StructuredEncoder.fit(fit);rows=[]
    for row in tqdm(staged,desc='Validate and encode CVSS; apply training-only scaling',unit='CVE'):
        vector,flags=StructuredEncoder.transform(row['structured_raw'],state,['cvss','reference_count'])
        result={k:v for k,v in row.items() if k!='structured_raw'};result['structured']=vector
        validate_predictor(result,p,len(state['feature_names']));rows.append(result)
        issues.append({'cve':row['cve'],'split':row['split'],'cvss_quality_flags':flags})
    case_results=[]
    for case in policy.review_cases():
        actual=policy.inspect(case['text'],case['cve'],case['source'])['decision']
        require(actual==case['expected'],'Reviewed policy case failed: '+case['case'])
        case_results.append({**case,'actual':actual,'review_type':'Automated engineering review; regression fixture, not independent annotation'})
    counts={s:dict(Counter(r['target'] for r in labels if r['split']==s)) for s in ids}
    summary={'schema':p['schema'],'scale':scale,'cves':{s:len(v) for s,v in ids.items()},'class_counts':counts,
             'source_coverage':{s:{name:sum(r['source_mask'][i] for r in rows if r['split']==s)
                  for i,name in enumerate(p['source_order'])} for s in ids},
             'documents_examined':len(decisions),'documents_accepted':sum(d['decision']=='accept' for d in decisions),
             'quarantine_reasons':dict(Counter(reason for d in decisions for reason in d['reasons'])),
             'ignored_function_identifier_spans':sum(len(d['ignored_spans']) for d in decisions),
             'cvss_quality_flags':dict(Counter(flag for r in issues for flag in r['cvss_quality_flags'])),
             'structured_dimension':len(state['feature_names']),
             'source_policy_review':'Engineering-reviewed/tested; not independently annotated or approved for final claims.',
             'training':False,'test_export':False,'test_scoring':False,'cache_reused':False,'encoder_rerun':False}
    lines=['VK02 separate predictor/label export',str(summary),
           'inputs.json has accepted text, source masks and structured vectors only, plus CVE/role join metadata.',
           'labels.json is supervision only. Never pass it or decisions.json into a text encoder or classifier feature builder.',
           'CVE/role are not numerical predictors. EPSS and catalogue fields are not structured inputs.',
           'Current text/CVSS are retrospective. Existing test/excluded roles are not exported or reassigned.',
           'No graph/embedding cache is loaded. Full export is not permission for full training.',
           'Small stratified sample supports integration checks only; no threshold/model selection.']
    artifacts={'inputs.json':{'source_order':p['source_order'],'rows':rows},'labels.json':labels,
               'structured_encoder.json':state,'decisions.json':decisions,'structured_audit.json':issues,
               'policy_review.json':{'version':policy.VERSION,'cases':case_results,'independent_annotation':False},
               'summary.json':summary,'summary.txt':'\n'.join(lines)+'\n',
               'run_identity.json':{'schema':p['schema'],'scale':scale,'ids':ids,'config':p,'pins':pins,
                                    'evaluation_status':p['evaluation'],'test_export':False}}
    directory=ROOT/p['data_root']/scale;publish(directory,artifacts,pins,verify)
    print(f'[VK02 COHORT] {summary["cves"]}; class counts={counts}; structured={summary["structured_dimension"]} columns')
    print(f'VK02 {scale} predictor/label export '+('reproduced.' if verify else 'saved to '+str(directory)+'.')+' No training or test export.')
    return directory


def load_smoke():
    p=config();directory=ROOT/p['data_root']/'smoke';check_lock(directory)
    identity=read(directory/'run_identity.json');check_pins(identity['pins'])
    require(identity['config']==p and identity['scale']=='smoke','Wrong smoke identity')
    inputs=read(directory/'inputs.json');rows=inputs['rows'];labels=read(directory/'labels.json')
    state=read(directory/'structured_encoder.json')
    for row in rows:validate_predictor(row,p,len(state['feature_names']))
    require(sorted(state['fit_ids'])==identity['ids']['train'],'Scaler not fitted only to smoke training CVEs')
    require(len({r['cve'] for r in rows})==len(rows)==len(labels),'Repeated/missing predictor or label rows')
    require([(r['cve'],r['split']) for r in rows]==[(r['cve'],r['split']) for r in labels],'Input/label join mismatch')
    require(all(type(r['target']) is int and r['target'] in (0,1) for r in labels),'Invalid binary target')
    require({s:sorted(r['cve'] for r in rows if r['split']==s) for s in identity['ids']}==identity['ids'],'Cohort changed')
    return rows,labels,state,identity,directory
