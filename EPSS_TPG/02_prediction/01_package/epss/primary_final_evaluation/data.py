"""Held-out labels stay separate from predictor-only records and graph planning."""
from collections import Counter, defaultdict
from datetime import date, timedelta
import re

from . import protocol as p
from ..primary_study import data as development
from ..consolidated import contracts
from .. import forecast_identity_graphs as chunks

read,write,require,tqdm=p.read,p.write,p.require,p.tqdm


def validate_input(row,role):
    require(set(row)=={'cve','split','documents','source_mask','structured'} and row['split']==role,
            'Wrong role or unexpected predictor fields')
    require(role in ('test','validation'), 'Evaluation roles only')
    require(re.fullmatch(r'CVE-\d{4}-\d{4,}',row['cve']), 'Unsafe CVE ID')
    require(set(row['structured'])==set(contracts.cvss.CVSS_FIELDS)|{'n_references_nvd'}, 'Unexpected structured fields')
    seen=set()
    for doc in row['documents']:
        require(set(doc)=={'document_id','source_text_type','text'} and doc['source_text_type'] in p.common.cs.SOURCES
                and isinstance(doc['text'],str) and doc['text'].strip(), 'Invalid source document')
        require(doc['document_id'] not in seen, 'Repeated source document');seen.add(doc['document_id'])
    require(row['source_mask']==[int(any(d['source_text_type']==s for d in row['documents'])) for s in p.common.cs.SOURCES], 'Source masks differ')


def heldout_record(record,frozen):
    c=record['cve']
    require(c in frozen['reserved_test_ids'], 'Record outside reserved test IDs')
    pub=p.common.cs.parent.contract.timestamp(record['published_timestamp']).date()
    require(record['published_date']==pub.isoformat(), 'Publication fields disagree')
    anchor=(pub+timedelta(days=30)).isoformat();end=(pub+timedelta(days=60)).isoformat()
    role,reason=p.common.cs.parent.contract.candidate_split(anchor,end,frozen['split_policy'])
    if role!='test' or reason:
        return None,None,{'cve':c,'reason':reason or 'not_test_date_admissible','target_date':end}
    # Keep the strict original parser; malformed data fails rather than changing eligibility silently.
    label=contracts.primary_label(record)
    if not label['observed']:
        return None,None,{'cve':c,'reason':'missing_day60_target','target_date':end}
    docs=contracts.text_documents(record,frozen['source_fields'])
    accepted=[{'document_id':development.document_id(d),'source_text_type':d['source'],'text':d['text']}
              for d in docs if not d['quarantined_epss_disclosure']]
    row={'cve':c,'split':'test','documents':accepted,'source_mask':contracts.source_mask(docs,p.common.cs.SOURCES),
         'structured':{k:record.get(k) for k in (*contracts.cvss.CVSS_FIELDS,'n_references_nvd')}}
    validate_input(row,'test')
    return row,{'split':'test',**label},{'cve':c,'reason':'included','target_date':end,
        'quarantined_documents':sum(d['quarantined_epss_disclosure'] for d in docs)}


def graph_plan(rows,role,limit):
    """Same chunk identity and boundaries as development, with explicit test provenance."""
    entries,documents,index={},{},{}
    for row in rows:
        validate_input(row,role);c=row['cve']
        require(c not in index,'Duplicate CVE')
        index[c]={'cve':c,'split':role,'source_mask':row['source_mask'],'documents':[]}
        for d in row['documents']:
            digest=chunks.old.shared.base.text_digest(d['document_id'])
            stem=f"{c}/{d['source_text_type']}/{digest}"
            parent_hash=chunks.old.shared.base.text_digest(d['text'])
            parts=chunks.segments(d['text'],limit)
            require(any(s['has_graph'] for s in parts), 'Source has no embeddable text; no silent exclusion')
            path='documents/'+stem+'.json'
            doc={'cve':c,'split':role,**d,'text_sha256':parent_hash,'chunks':[]}
            for s in parts:
                part=dict(s)
                if s['has_graph']:
                    text=d['text'][s['start']:s['end']]
                    key=chunks.old.shared.base.text_digest(c+'\0'+d['document_id']+'\0'+str(s['index']))
                    require(key not in entries,'Chunk identity collision')
                    entry={'key':key,'graph_id':'CD03-'+key,'cve':c,'split':role,'document_id':d['document_id'],
                           'source_text_type':d['source_text_type'],'text':text,'text_sha256':chunks.old.shared.base.text_digest(text),
                           'parent_text_sha256':parent_hash,'chunk':dict(s),'stem':stem+f"/chunk_{s['index']:05d}"}
                    entries[key]=entry
                    part.update(key=key,graph=f"tpg_json/no_coref/{entry['stem']}.json",tensor=f"tensors/{entry['stem']}.pt")
                doc['chunks'].append(part)
            documents[path]=doc
            index[c]['documents'].append({'document_id':d['document_id'],'source_text_type':d['source_text_type'],
                'source_file':path,'characters':len(d['text']),'chunks':doc['chunks']})
    return {'entries':entries,'documents':documents,'index':index}


def overlap_audit(rows):
    registry=defaultdict(set)
    for r in read(p.common.DATA/'full/inputs.json')['rows']:
        for d in r['documents']:
            registry[chunks.preparation.text_key(d['text'])].add(r['split'])
    result=[]
    for r in rows:
        for d in r['documents']:
            key=chunks.preparation.text_key(d['text']);roles=sorted(registry.get(key,set()))
            if roles:result.append({'cve':r['cve'],'document_id':d['document_id'],'source':d['source_text_type'],
                                   'normalized_text_sha256':key,'shared_with_development_roles':roles})
    return {'matching_documents':result,'interpretation':'Shared text is reported, not removed. Distinct CVEs are not declared duplicates. Smoke intentionally uses development records.'}


def prepare(scale,release_test=False,verify=False):
    p.access(scale,release_test);frozen=p.load();identity=p.stage_identity(frozen,scale)
    role=identity['role'];rows=[];labels=[];audit=[]
    if scale=='smoke':
        rows=[r for r in read(p.common.DATA/'smoke/inputs.json')['rows'] if r['split']=='validation']
        ids={r['cve'] for r in rows}
        labels=[r for r in read(p.common.DATA/'smoke/labels.json') if r['cve'] in ids]
        require(ids<=set(frozen['development_ids']['validation']), 'Smoke must use development validation only')
        audit=[{'cve':r['cve'],'reason':'development_smoke'} for r in rows]
    else:
        # This is the first permitted record-level test access, after explicit release and freeze.
        records=read(frozen['source_path']);by_id={r['cve']:r for r in records}
        require(len(by_id)==len(records) and set(frozen['reserved_test_ids'])<=set(by_id), 'Duplicate/missing raw CVE')
        for c in tqdm(frozen['reserved_test_ids'],desc='Apply frozen test eligibility and source policy',unit='CVE'):
            row,label,decision=heldout_record(by_id[c],frozen);audit.append(decision)
            if row is not None:rows.append(row);labels.append(label)
    rows=sorted(rows,key=lambda r:r['cve']);labels=sorted(labels,key=lambda r:r['cve'])
    require(rows and [r['cve'] for r in rows]==[r['cve'] for r in labels], 'Empty/unmatched evaluation cohort')
    for row in rows:validate_input(row,role)
    identity['ids']=[r['cve'] for r in rows]
    resources=read(frozen['resources_path']);state=read(frozen['structured_encoder_path'])
    require(sorted(state['fit_ids'])==frozen['development_ids']['train'],'Changed preprocessing cohort')
    summary={'scale':scale,'role':role,'eligible_cves':len(rows),'reason_counts':dict(Counter(r['reason'] for r in audit)),
             'source_coverage':{s:sum(r['source_mask'][i] for r in rows) for i,s in enumerate(p.common.cs.SOURCES)},
             'test_released':scale=='full','test_scoring':False,'preprocessor_refitted':False}
    artifacts={'inputs.json':{'source_order':list(p.common.cs.SOURCES),'rows':rows},'labels.json':labels,
               'eligibility_audit.json':audit,'overlap_audit.json':overlap_audit(rows),
               'structured_encoder.json':state,'resources.json':resources,'run_identity.json':identity,'analysis.json':summary}
    directory=p.DATA/scale
    if verify:
        p.common.verify_lock(directory)
        for name,value in artifacts.items():require(read(directory/name)==value,'Changed final data: '+name)
    else:
        require(not directory.exists(),'Data exists; use --verify-only');directory.mkdir(parents=True)
        for name,value in artifacts.items():write(directory/name,value)
        (directory/'analysis.txt').write_text('PS05 separate evaluation predictors and labels\n'+str(summary)+'\nNo preprocessing fit, training or scoring. Text is retrospective.\n')
        p.common.lock(directory)
    print(f'PS05 {scale} data '+('verified.' if verify else 'prepared.')+' No training or scoring.')


def load(scale,release_test=False):
    p.access(scale,release_test);frozen=p.load();directory=p.DATA/scale
    p.common.verify_lock(directory);identity=read(directory/'run_identity.json')
    for k,v in p.stage_identity(frozen,scale).items():require(identity[k]==v,'Data/freeze mismatch')
    rows=read(directory/'inputs.json')['rows']
    for row in rows:validate_input(row,identity['role'])
    require([r['cve'] for r in rows]==identity['ids'],'Changed predictor cohort')
    return frozen,identity,rows,read(directory/'resources.json'),read(directory/'structured_encoder.json')
