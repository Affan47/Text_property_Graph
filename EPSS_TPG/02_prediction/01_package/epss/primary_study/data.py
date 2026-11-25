"""Separate predictor/label export, using the already verified CS01 CVE roles."""
from collections import Counter
import hashlib

from .common import *
from ..consolidated import contracts
from .. import forecast_identity_graphs as graph_plan


def document_id(doc):
    return doc['source']+':'+hashlib.sha256(((doc['platform'] or '')+'\0'+doc['text']).encode()).hexdigest()


def smoke_ids(candidates,masks,count):
    remaining=sorted(set(candidates));chosen=[];covered=set()
    while remaining and len(chosen)<count:
        best=min(remaining,key=lambda c:(-sum(bool(v) and i not in covered for i,v in enumerate(masks[c])),c))
        chosen.append(best);remaining.remove(best)
        covered.update(i for i,v in enumerate(masks[best]) if v)
    return sorted(chosen)


def validate_input(row):
    require(set(row)=={'cve','split','documents','source_mask','structured'}, 'Unexpected predictor fields')
    require(row['split'] in ('train','validation'), 'Test predictor export forbidden')
    require(set(row['structured'])==set(contracts.cvss.CVSS_FIELDS)|{'n_references_nvd'}, 'Unexpected structured predictor')
    ids=set()
    for doc in row['documents']:
        require(set(doc)=={'document_id','source_text_type','text'}, 'Unexpected text predictor')
        require(doc['source_text_type'] in cs.SOURCES and isinstance(doc['text'],str) and doc['text'].strip(), 'Invalid source document')
        require(doc['document_id'] not in ids, 'Duplicate document');ids.add(doc['document_id'])
    require(row['source_mask']==[int(any(d['source_text_type']==s for d in row['documents'])) for s in cs.SOURCES], 'Source mask mismatch')


def feature_review(records):
    """Inspect development fields only; recommendations are not predictor additions."""
    names = ('n_references_nvd','vuln_status','cvss_vector_source','cvss_vector_severity',
             'github_only_urls','non_github_urls','all_sources_urls','url_change_dates',
             'url_dates_status','published_timestamp','last_modified_timestamp','social_media_posts',
             'vckev_cwes','vendor_project','epss_version_changed')
    coverage = {}
    for name in names:
        values=[r.get(name) for r in records]
        present=[v for v in values if v is not None and v!='' and v!=[]]
        coverage[name]={'records':len(values),'nonempty':len(present),
                       'python_types':dict(Counter(type(v).__name__ for v in values))}
        if name in ('vuln_status','cvss_vector_source','cvss_vector_severity','url_dates_status'):
            coverage[name]['value_counts']=dict(Counter(str(v) for v in values))
    return {'coverage':coverage, 'enabled_beyond_cvss':['n_references_nvd'],
        'retrospective_candidates_not_enabled':{
            'vuln_status':'Train-only one-hot plus missing/unseen flags; current NVD workflow status may encode curation effects.',
            'github_only_urls/non_github_urls':'Deduplicated host-validated counts or domain diversity; source-coupled ablations required, current links only.',
            'social_media_posts':'Counts/platform multi-hot only from accepted posts, gated off with SMP; timestamps do not automatically establish availability.',
            'text_length/graph_size':'Source-coupled diagnostics first; may mostly measure scraping coverage rather than vulnerability properties.'},
        'not_independent_or_excluded':{
            'cvss_vector_severity':'Redundant with base score; not an independent non-CVSS feature.',
            'cvss_vector_source/url_dates_status':'Curation/scraping metadata; audit first, not enabled.',
            'published_timestamp':'Join/date/split metadata; calendar trends can become shortcuts.',
            'last_modified_timestamp':'Current update date, not historical feature availability.',
            'vckev_cwes/vendor_project/other_vckev_fields':'Catalogue enrichment excluded, including derived counts/missingness.',
            'epss_version_changed/epss_scores':'Future-window/target information excluded from primary predictors.'},
        'interpretation':'Coverage is not a claim of predictive benefit or historical validity.'}


def assemble(scale):
    p=config()
    for s in ('full','smoke') if scale=='smoke' else ('full',):
        verify_lock(DESIGN/s,'design_lock.json')
        check_pins(read(DESIGN/s/'run_identity.json')['source_pins'])
    design=read(DESIGN/'full'/'run_identity.json')
    audit=read(DESIGN/'full'/'cohort_audit.json')
    ids={s:sorted(r['cve'] for r in audit if r['split']==s and r['primary_eligible']) for s in ('train','validation')}
    if scale=='smoke':
        smoke=read(DESIGN/'smoke'/'run_identity.json')['ids']
        masks={r['cve']:r['source_mask'] for r in audit}
        ids={s:smoke_ids([c for c in ids[s] if c in smoke[s]],masks,p['smoke_counts'][s]) for s in ids}
        require({s:len(v) for s,v in ids.items()}==p['smoke_counts'], 'Insufficient bounded smoke IDs')
    require(not set(ids['train']) & set(ids['validation']), 'CVE crosses roles')
    locator=cs.parent.sources.PREPARED/'preparation_summary.json'
    source=Path(read(locator)['source_path'])
    require(sha256(source)==design['source_pins'][str(source)], 'Dataset changed')
    selected=set(ids['train']+ids['validation'])
    raw={r['cve']:r for r in read(source) if r['cve'] in selected}
    require(set(raw)==selected, 'Missing selected records')
    rows=[];labels=[];quarantine=[]
    for split, chosen in ids.items():
        for cve in tqdm(chosen,desc='Export separate '+split+' predictors/targets',unit='CVE'):
            r=raw[cve];label=contracts.primary_label(r)
            require(label['observed'], 'Primary target missing')
            docs=contracts.text_documents(r,cs.load()['source_fields'])
            accepted=[]
            for doc in docs:
                if doc['quarantined_epss_disclosure']:
                    quarantine.append({'cve':cve,'source':doc['source'],'document_id':document_id(doc)});continue
                accepted.append({'document_id':document_id(doc),'source_text_type':doc['source'],'text':doc['text']})
            row={'cve':cve,'split':split,'documents':accepted,'source_mask':contracts.source_mask(docs,cs.SOURCES),
                 'structured':{k:r.get(k) for k in (*contracts.cvss.CVSS_FIELDS,'n_references_nvd')}}
            validate_input(row);rows.append(row);labels.append({'split':split,**label})
    fitted=contracts.StructuredEncoder.fit([{'cve':r['cve'],'split':'train',**r['structured']} for r in rows if r['split']=='train'])
    old_lock=read(OLD/'cache_lock.json')
    require(sha256(OLD/'run_identity.json')==old_lock['files']['run_identity.json'], 'Old resource identity changed')
    resource=read(OLD/'run_identity.json')
    # Verify the extraction/embedding helpers that actually produced candidate chunks.
    frozen_code={n:h for n,h in resource['source_pins'].items() if n.endswith('.py')}
    check_pins(frozen_code)
    resources={k:resource[k] for k in ('encoder','embedding_dim','type_vocabulary','spacy_pipeline_sha256')}
    resources['protocol']=resource['protocol'];resources['source_pins']=frozen_code
    planned=graph_plan.plan([{k:v for k,v in r.items() if k!='structured'} for r in rows],ids,resources['protocol']['max_chunk_characters'])
    pins={**code_pins(),**design['source_pins']}
    for f in (DESIGN/'full'/'design_lock.json',OLD/'cache_lock.json',OLD/'run_identity.json'):
        pins[str(f)]=sha256(f)
    if scale=='smoke':pins[str(DESIGN/'smoke'/'design_lock.json')]=sha256(DESIGN/'smoke'/'design_lock.json')
    return {'inputs.json':{'source_order':list(cs.SOURCES),'rows':rows},'labels.json':labels,
        'structured_encoder.json':fitted,'resources.json':resources,'quarantine.json':quarantine,
        'feature_review.json':feature_review(list(raw.values())),
        'run_identity.json':{'schema':p['schema'],'scale':scale,'ids':ids,'source_pins':pins,
                            'config':p,'retrospective':True,'test_scoring':False},
        'analysis.json':{'scale':scale,'cves':{s:len(v) for s,v in ids.items()},'documents':len(planned['documents']),
                         'chunks':len(planned['entries']),'quarantined_documents':len(quarantine),
                         'source_coverage':{s:{source:sum(r['source_mask'][i] for r in rows if r['split']==s)
                          for i,source in enumerate(cs.SOURCES)} for s in ids},'training':False,'test_export':False}}


def prepare(scale,verify=False):
    directory=DATA/scale;artifacts=assemble(scale)
    summary=artifacts['analysis.json']
    report=('PS01 primary data export\nRetrospective Day-60 EPSS; no EPSS predictor fields.\n'
            f"CVEs: {summary['cves']}; documents: {summary['documents']}; chunks: {summary['chunks']}\n"
            'inputs.json is predictor-only; labels.json is supervision, never graph input.\n'
            'CVSS plus current reference count are the only structured blocks.\n'
            'feature_review.json audits additional candidates without enabling them.\n'
            'No training or test export.\n')
    if verify:
        verify_lock(directory)
        for n,v in artifacts.items():require(read(directory/n)==v,'Export does not reproduce: '+n)
        require((directory/'analysis.txt').read_text()==report,'Report differs')
    else:
        require(not directory.exists(),'Data exists; use --verify-only');directory.mkdir(parents=True)
        for n,v in artifacts.items():write(directory/n,v)
        (directory/'analysis.txt').write_text(report,encoding='utf-8')
        check_pins(artifacts['run_identity.json']['source_pins']);lock(directory)
    print(f'PS01 {scale} predictor/label export '+('reproduced.' if verify else f'saved to {directory}.'))


def load_inputs(scale):
    directory=DATA/scale;verify_lock(directory)
    identity=read(directory/'run_identity.json');check_pins(identity['source_pins'])
    rows=read(directory/'inputs.json')['rows']
    for row in rows:validate_input(row)
    require({s:sorted(r['cve'] for r in rows if r['split']==s) for s in identity['ids']}==identity['ids'], 'Changed roles')
    return rows,identity,read(directory/'resources.json')
