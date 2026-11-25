"""Exact-identity cache reuse and bounded security extraction diagnostics."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import shutil

from .common import *
from .data import load_inputs
from .. import forecast_identity_graphs_v2 as builder_api

base=builder_api.base
torch=builder_api.torch
_worker=None


def graph_plan(scale):
    rows,identity,resources=load_inputs(scale)
    plan=base.plan([{k:v for k,v in r.items() if k!='structured'} for r in rows],
                   identity['ids'],resources['protocol']['max_chunk_characters'])
    return plan,identity,resources


def reusable(receipt,entry):
    return receipt.get('entry')=={k:v for k,v in entry.items() if k!='text'} and set(receipt.get('files',{}))==base.files(entry)


def copy_verified(directory,entry,resources,old_lock):
    relative='receipts/'+entry['key']+'.json'
    if relative not in old_lock['files']:return False
    require(sha256(OLD/relative)==old_lock['files'][relative], 'Changed old receipt')
    receipt=read(OLD/relative)
    if not reusable(receipt,entry):return False
    for name,value in receipt['files'].items():
        require(old_lock['files'].get(name)==value, 'Receipt not authenticated by old cache lock')
    builder_api.check_entry(OLD,entry,resources)
    for name in [*receipt['files'],relative]:
        target=directory/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(OLD/name,target)
    return True


def worker_init(resources):
    global _worker
    torch.set_num_threads(1)
    _worker=(resources,builder_api.Builder(resources,False))


def worker_build(payload):
    directory,entry=payload
    resources,builder=_worker
    builder_api.save_entry(Path(directory),entry,resources,builder)
    return entry['key']


def dynamic_checks():
    """Synthetic extraction checks, never model inputs or measured CVE outcomes."""
    from tpg.schema.types import SecurityEdgeType
    pipeline=base.old.extension.ProductVersionSecurityPipeline()
    texts={
        'affirmed':'CVE-2024-12345 affects Apache Tomcat version 9.0.1.',
        'negated':'CVE-2024-12345 does not affect Apache Tomcat version 9.0.1.',
        'possible':'CVE-2024-12345 may affect Apache Tomcat version 9.0.1.',
        'neutral':'The library opens at nine and closes at five.'}
    output={}
    for name,text in tqdm(texts.items(),desc='Check text-dependent security extraction',unit='case'):
        graph=pipeline.run(text,doc_id='PS01-SYNTHETIC-'+name)
        require(not graph.validate(),'Invalid synthetic graph')
        edges=[e for e in graph.edges() if isinstance(e.edge_type,SecurityEdgeType)]
        output[name]={'text':text,'security_edges':[{'relation':'SEC_'+e.edge_type.name,
            'source':graph.get_node(e.source).properties.text,'target':graph.get_node(e.target).properties.text,
            'evidence':e.properties.extra} for e in edges]}
    for name,polarity,modality in [('affirmed','affirmed','asserted'),('negated','negated','asserted'),('possible','affirmed','possible')]:
        affects=[e for e in output[name]['security_edges'] if e['relation']=='SEC_AFFECTS']
        require(affects and any(e['evidence'].get('polarity')==polarity and e['evidence'].get('modality')==modality for e in affects),
                'Security assertion qualifier failed: '+name)
    require(not output['neutral']['security_edges'],'Neutral text gained security relations')
    return {'synthetic_checks':output,'extractor':'fixed v4 rule/entity/dependency policy, text-dependent; not a trained relation predictor',
        'interpretation':'These checks establish execution and qualifier preservation, not extraction accuracy on arbitrary text.'}


def inspect_edges(plan,directory,resources):
    counts=Counter();methods=Counter();polarity=Counter();modalities=Counter();examples={};rows=[]
    for entry in tqdm(plan['entries'].values(),desc='Verify tensors, security evidence and source spans',unit='chunk'):
        _,stats=builder_api.check_entry(directory,entry,resources)
        g=read(directory/'tpg_json/no_coref'/(entry['stem']+'.json'))
        nodes={n['id']:n for n in g['vertices']}
        for edge in g['edges']:
            if not edge['label'].startswith('SEC_'):continue
            extra=edge['properties'].get('EXTRA',{})
            require(extra.get('source_text_type')==entry['source_text_type'],'Lost source provenance')
            require(extra.get('status')=='candidate','Security edge lacks candidate status')
            counts[edge['label']]+=1;methods[extra.get('method','not_recorded')]+=1
            polarity[extra.get('polarity','not_recorded')]+=1;modalities[extra.get('modality','not_recorded')]+=1
            samples=examples.setdefault(edge['label'],[])
            if len(samples)<4:samples.append({'cve':entry['cve'],'source':entry['source_text_type'],
                'chunk':entry['chunk']['index'],'source_node':nodes[edge['outV']], 'target_node':nodes[edge['inV']],
                'edge':edge,'graph_file':'tpg_json/no_coref/'+entry['stem']+'.json'})
        rows.append(stats)
    return {'chunks':rows,'security_edge_counts':dict(counts),'methods':dict(methods),'polarities':dict(polarity),
        'modalities':dict(modalities),'real_examples':examples,'edge_feature_width':64,
        'feature_contract':'Saved relation vocabulary plus signed-hash dependency/SRL/relation/polarity/modality features. Hash collisions are possible.',
        'learned_extraction_active':False,'semantic_accuracy_measured':False,
        'limitations':['Rule inventory and entity patterns are fixed; edge instances depend on input text.',
                      'No cross-chunk or cross-source semantic relations are invented.',
                      'GNN gradients update the predictor, not this frozen extraction policy.']}


def verify(scale):
    directory=CACHE/scale
    verify_lock(directory)
    plan,identity,resources=graph_plan(scale)
    require(read(directory/'run_identity.json')==identity,'Changed cache identity')
    require(read(directory/'resources.json')==resources,'Changed graph resources')
    require(read(directory/'cve_index.json')==plan['index'],'Index differs')
    for name,doc in plan['documents'].items():require(read(directory/name)==doc,'Document coverage differs')
    audit=inspect_edges(plan,directory,resources)
    require(audit==read(directory/'security_audit.json'),'Security audit does not reproduce')
    require(read(directory/'dynamic_checks.json')==dynamic_checks(),'Dynamic extraction checks differ')
    provenance=read(directory/'chunk_provenance.json')
    require(set(provenance)==set(plan['entries']),'Missing chunk provenance')
    summary=read(directory/'analysis.json')
    require(summary['chunks']==len(provenance) and summary['reused']==sum(v=='verified_old_cache' for v in provenance.values()),'Cache counts differ')
    print(f'PS01 {scale} cache verified. No encoder rerun, training or test graphs.')


def prepare(scale,workers=1,resume=False):
    require(1<=workers<=16,'Workers must be 1..16')
    plan,identity,resources=graph_plan(scale);directory=CACHE/scale
    if directory.exists():
        require(resume and not (directory/'artifact_lock.json').exists(),'Cache exists; use --verify-only, or --resume for unfinished cache')
        require(read(directory/'run_identity.json')==identity and read(directory/'resources.json')==resources,'Resume identity changed')
    else:
        directory.mkdir(parents=True);write(directory/'run_identity.json',identity);write(directory/'resources.json',resources)
    old_lock=read(OLD/'cache_lock.json')
    provenance=read(directory/'chunk_provenance.json') if (directory/'chunk_provenance.json').exists() else {}
    missing=[]
    for entry in tqdm(plan['entries'].values(),desc='Check exact-identity chunk reuse',unit='chunk'):
        receipt=directory/'receipts'/(entry['key']+'.json')
        if receipt.exists():
            builder_api.check_entry(directory,entry,resources)
            provenance.setdefault(entry['key'],'new_build')
        elif copy_verified(directory,entry,resources,old_lock):provenance[entry['key']]='verified_old_cache'
        else:missing.append(entry)
    write(directory/'chunk_provenance.json',provenance)
    print(f"[CACHE] {len(provenance)} verified/reused chunks; {len(missing)} to build; workers={workers}",flush=True)
    if missing:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                initializer=worker_init,initargs=(resources,)) as pool:
            for key in tqdm(pool.map(worker_build,[(str(directory),e) for e in missing],chunksize=1),
                            total=len(missing),desc='Build missing TPGs and frozen embeddings',unit='chunk'):
                provenance[key]='new_build'
    write(directory/'chunk_provenance.json',provenance)
    for name,doc in plan['documents'].items():write(directory/name,doc)
    write(directory/'cve_index.json',plan['index'])
    audit=inspect_edges(plan,directory,resources);write(directory/'security_audit.json',audit)
    write(directory/'dynamic_checks.json',dynamic_checks())
    summary={'scale':scale,'cves':{s:len(v) for s,v in identity['ids'].items()},'documents':len(plan['documents']),
             'chunks':len(plan['entries']),'reused':sum(v=='verified_old_cache' for v in provenance.values()),
             'built':sum(v=='new_build' for v in provenance.values()),'security_edges':audit['security_edge_counts'],
             'training':False,'test_graphs':False}
    write_report(directory,summary,['PS01 source-aware primary cache',str(summary),
        'Actual graphs: tpg_json/<edge-view>/<CVE>/<source>/<document>/chunk_*.json',
        'Embeddings and edge tensors: tensors/<CVE>/<source>/<document>/chunk_*.pt',
        'Full text and offsets: documents/; cve_index.json joins CVEs, documents and chunks.',
        'security_audit.json contains real edge examples, counts, evidence methods and qualifiers.',
        'dynamic_checks.json contains labelled synthetic extraction checks, not dataset evidence.',
        'Rule-based candidate extraction is not self-learning. No semantic accuracy is inferred from passing validation.',
        'Old caches and raw source files are unchanged.'])
    check_pins(identity['source_pins']);lock(directory)
    print(f'PS01 {scale} cache saved to {directory}. No training or test graphs.')
