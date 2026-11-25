"""Immutable corrected graph cache, atomic publication, resumable compact vectors."""
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import signal
from uuid import uuid4
import numpy as np
import torch
from tqdm import tqdm

from tpg_combined_v2 import common as c, graphs, cache as parent_cache
from tpg_combined_quality_v1 import reviewed_correction as repair
from tpg_combined_recovery_v1 import pool
from tpg_combined_parallel_v1.runner import writer_lock, thread_environment, available_memory_gib, scheduled_keys
from . import compact

SCHEMA='combined_text_securebert2_v3'
RUNTIME=c.ROOT/'06_runtime/02_fresh_studies/compact_v3'


def directory(dataset,scale):return c.directory(dataset,scale,3)


def pins():
    files=list(Path(__file__).parent.glob('*.py'))
    files += [Path(repair.__file__),Path(repair.base.__file__),Path(pool.__file__)]
    files += [c.ROOT/'01_tpg/10_extensions/tpg_combined_parallel_v1/runner.py',
              c.ROOT/'01_tpg/02_examples/01_scripts/build_dataset_tpgs_v3.py']
    return {str(p.relative_to(c.ROOT)):c.sha(p) for p in sorted(files)}


def load_plan(root):
    root=Path(root);c.check_files(root,c.read(root/'plan_lock.json')['files'])
    plan=c.read(root/'plan.json.gz')
    c.require(plan['schema']==SCHEMA and plan['processing']['v3_pins']==pins(),'Changed v3 implementation; use a new version, never rewrite pins')
    c.require(not plan['test_adapted'] and all(e['split'] in ('train','validation') for e in plan['entries'].values()),'Test adaptation forbidden')
    for name,expected in plan['parent_files'].items():c.require(c.sha(name)==expected,f'Parent plan changed: {name}')
    # Parent remains authoritative for its code and all frozen predictor-source identities.
    parent=c.load_plan(plan['parent_directory'])
    c.require(c.digest(parent['processing'])==plan['processing']['parent_processing_sha256'],'Parent processing changed')
    return plan


def prepare(dataset,scale,resume=False):
    root=directory(dataset,scale);source=c.directory(dataset,scale)
    parent=c.load_plan(source)
    processing={'schema':SCHEMA,'parent_processing_sha256':c.digest(parent['processing']),
        'v3_pins':pins(),'annotation_policy':repair.POLICY,'citation_extension':'same_document_cis_link_numbered_heading_v1',
        'format':compact.FORMAT,'model':c.MODEL,'revision':c.REVISION,
        'text_context':'unchanged combined text; same 1024-subword windows; no cross-text vector reuse',
        'source_policy':'unchanged; no dropping/relevance filtering/truncation',
        'graph_format_version':c.GRAPH_VERSION,'dtype':'float32; no quantization'}
    entries,parents,remap={},{},{}
    for old,e in tqdm(parent['entries'].items(),desc=f'Plan corrected {dataset} {scale} graph identities',unit='graph'):
        item={**e,'processing_identity':c.digest(processing)};key=c.digest(item)
        entries[key]=item;parents[key]=old;remap[old]=key
    index=deepcopy(parent['index'])
    for row in index.values():
        for view in row['views'].values():
            if view['object_key']:view['object_key']=remap[view['object_key']]
    plan={**parent,'schema':SCHEMA,'processing':processing,'entries':entries,'index':index,
          'parent_directory':str(source),'parent_keys':parents,
          'parent_files':{str(source/name):c.sha(source/name) for name in ('plan.json.gz','plan_lock.json','parser_identity.json')},
          'annotation_promotion':'Reviewed bounded v8 corrections; not a general claim of semantic correctness',
          'test_adapted':False}
    root.parent.mkdir(parents=True,exist_ok=True)
    with writer_lock(root):
        if (root/'plan_lock.json').exists():
            c.require(resume,'Plan exists; pass --resume to verify')
            c.require(load_plan(root)==plan,'Changed plan');print(f'Plan reproduced: {root}');return
        payload={'plan.json.gz':plan,'cve_index.json':index,'coverage.json':parent['coverage']}
        for i,v in enumerate(parent['variants'],1):
            payload[f'variants/{i:02d}_{v}/index.json']={cid:{'split':r['split'],**r['views'][v]} for cid,r in index.items()}
        for name,value in payload.items():
            if (root/name).exists():c.require(c.read(root/name)==value,'Conflicting partial plan')
            else:c.create(root/name,value)
        c.create(root/'plan_lock.json',{'files':{name:c.sha(root/name) for name in payload}})
    print(f'v3 plan: {len(index):,} development CVEs, {len(entries):,} objects; {root}; no source changes.')


def graph_receipt(root,key,plan):
    obj=Path(root)/'objects'/key;p=obj/'graph_receipt.json'
    if not p.exists():
        c.require(not obj.exists(),f'Unpublished target exists: {obj}; do not overwrite it')
        return None
    r=c.read(p)
    c.require(r['entry_sha256']==key and r['processing_sha256']==c.digest(plan['processing']) and
              r['parent_key']==plan['parent_keys'][key],'Receipt identity differs')
    c.require(set(r['files'])=={'graph.json.gz','structure.npz'},'Unexpected graph object files')
    c.check_files(obj,r['files'])
    source=Path(plan['parent_directory'])/'objects'/r['parent_key']
    c.require(c.sha(source/'graph.json.gz')==r['parent_graph_sha256'] and
              c.sha(source/'graph_receipt.json')==r['parent_receipt_sha256'],'Original graph/receipt changed')
    return r


def transform(original,entry,parent_entry,parent_key,parent_sha):
    fixed=repair.correct(original)
    repair.validate_preview(parent_entry,original,fixed)
    c.require(repair.correct(fixed)==fixed,'Correction is not idempotent')
    fixed['doc_id']='CTPG3-'+c.digest(entry)
    fixed['metadata']['combined_text']['processing_identity']=entry['processing_identity']
    audit=fixed['metadata'].pop('annotation_repair_preview')
    audit['scope']='Versioned development-cache annotations; original text and syntax retained'
    fixed['metadata']['annotation_repair']=audit
    fixed['metadata']['cache_lineage']={'cache_schema':SCHEMA,'parent_object_key':parent_key,
        'parent_graph_file_sha256':parent_sha,'correction_policy':repair.POLICY}
    graphs.validate(entry,fixed)
    return fixed


def publish(root,key,graph,arrays,metadata,record):
    root=Path(root);target=root/'objects'/key
    stage=root/'.staging'/(key+'-'+uuid4().hex);stage.mkdir(parents=True)
    c.create(stage/'graph.json.gz',graph);compact.save(stage/'structure.npz',arrays,metadata)
    record={**record,'files':{name:c.sha(stage/name) for name in ('graph.json.gz','structure.npz')}}
    c.create(stage/'graph_receipt.json',record)
    for p in stage.iterdir():
        with p.open('rb') as f:os.fsync(f.fileno())
    target.parent.mkdir(parents=True,exist_ok=True)
    c.require(not target.exists(),'Refuse overwrite of completed graph object')
    stage.rename(target)
    return record


def initialize(context):
    torch.set_num_threads(1)
    c.require(context['pins']==pins(),'Worker code identity changed')
    return context


def build_one(state,job):
    root=Path(state['root']);source=Path(state['source']);old=job['parent_key']
    c.require(shutil.disk_usage(root).free>20*2**30,'Less than 20 GiB reserve; stop before filling disk')
    original_entry={**job['entry'],'processing_identity':state['parent_processing']}
    r=parent_cache.receipt(source,old,original_entry);c.require(r,'Missing original graph')
    path=source/'objects'/old/'graph.json.gz';sha=c.sha(path)
    original=c.read(path)
    fixed=transform(original,job['entry'],original_entry,old,sha)
    tensors=graphs.tensors(fixed);arrays,metadata=compact.pack(tensors,fixed)
    rebuilt=compact.unpack(arrays,metadata)
    for name,value in tensors.items():
        c.require(torch.equal(value,rebuilt[name]) if isinstance(value,torch.Tensor) else value==rebuilt[name],f'Compact round trip differs: {name}')
    record={'entry_sha256':job['key'],'processing_sha256':state['processing_sha256'],'parent_key':old,
        'parent_graph_sha256':sha,'parent_receipt_sha256':c.sha(source/'objects'/old/'graph_receipt.json'),
        'statistics':graphs.statistics(fixed),'unique_pooling_rows':len(arrays['pool_ptr'])-1,
        'dense_node_embedding_bytes':len(fixed['vertices'])*768*4,
        'compact_embedding_bytes':(len(arrays['pool_ptr'])-1)*768*4,
        'dense_edge_feature_bytes':tensors['edge_features'].numel()*4,
        'sparse_edge_feature_bytes':sum(arrays[k].nbytes for k in ('edge_ptr','edge_col','edge_value')),
        'correction_counts':fixed['metadata']['annotation_repair']['rule_counts']}
    saved=publish(root,job['key'],fixed,arrays,metadata,record)
    c.require(c.sha(path)==sha,'Original graph changed during correction')
    return {'key':job['key'],'cve':job['entry']['cve'],'statistics':saved['statistics']}


def receipts(root,plan):
    saved={}
    for key in tqdm(plan['entries'],desc='Check completed compact objects and original hashes',unit='graph'):
        r=graph_receipt(root,key,plan)
        if r:saved[key]=r
    return saved


def summarize(root,plan,records):
    names=('dense_node_embedding_bytes','compact_embedding_bytes','dense_edge_feature_bytes','sparse_edge_feature_bytes')
    stats={name:sum(r[name] for r in records.values()) for name in names}
    stats.update(graphs=len(records),planned_graphs=len(plan['entries']),nodes=sum(r['statistics']['nodes'] for r in records.values()),
        edges=sum(r['statistics']['edges'] for r in records.values()),unique_pooling_rows=sum(r['unique_pooling_rows'] for r in records.values()),
        free_disk_gib=shutil.disk_usage(root).free/2**30)
    print('[STORAGE] '+str({k:round(v/2**30,3) for k,v in stats.items() if k.endswith('_bytes')}),flush=True)
    return stats


def build(dataset,scale,workers=4,limit=None):
    root=directory(dataset,scale);plan=load_plan(root)
    c.require(limit is None or limit>0,'Limit must be positive')
    c.require(1<=workers<=min(32,len(os.sched_getaffinity(0))),'Invalid worker count')
    c.require(available_memory_gib()>6*workers+2,'Reduce workers: reserve 6 GiB per worker plus 2 GiB')
    with writer_lock(root),thread_environment():
        saved=receipts(root,plan)
        pending={k:e for k,e in plan['entries'].items() if k not in saved}
        keys=scheduled_keys(pending);keys=keys if limit is None else keys[:limit]
        if keys:
            state={'root':str(root),'source':plan['parent_directory'],'pins':pins(),
                'parent_processing':plan['processing']['parent_processing_sha256'],'processing_sha256':c.digest(plan['processing'])}
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
            log=RUNTIME/dataset/scale/(stamp+'.jsonl');log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('x',buffering=1) as stream,tqdm(total=len(keys),desc=f'Correct and compact {dataset} (no parser/encoder)',unit='graph') as bar:
                def event(kind,data):
                    import json
                    safe={k:v for k,v in data.items() if k not in ('job','active')}
                    if isinstance(data.get('job'),dict):safe['key']=data['job']['key']
                    stream.write(json.dumps({'kind':kind,**safe},sort_keys=True)+'\n')
                    if kind=='result':bar.update()
                    if kind=='error':tqdm.write(data['traceback'])
                    if kind=='heartbeat':tqdm.write(f'[ACTIVE] {data["completed"]}/{len(keys)} corrected; {len(data["active"])} workers; {shutil.disk_usage(root).free/2**30:.1f} GiB free')
                old=signal.getsignal(signal.SIGTERM)
                def interrupt(*_):raise KeyboardInterrupt('SIGTERM: stopping workers safely')
                signal.signal(signal.SIGTERM,interrupt)
                try:
                    pool.run(({'key':k,'entry':pending[k],'parent_key':plan['parent_keys'][k]} for k in keys),
                        initialize,build_one,state,min(workers,len(keys)),event,task_timeout=3600)
                finally:signal.signal(signal.SIGTERM,old)
        saved=receipts(root,plan);stats=summarize(root,plan,saved)
        if len(saved)==len(plan['entries']):
            stable={k:v for k,v in stats.items() if k!='free_disk_gib'}
            p=root/'graph_storage_summary.json'
            if p.exists():c.require(c.read(p)==stable,'Graph storage summary changed')
            else:c.create(p,stable)
        print(f'v3 graph stage: {len(saved)}/{len(plan["entries"])} complete: {root}; no embeddings or training.')


def embedding_receipt(obj,graph_record,identity):
    p=obj/'embedding_receipt.json'
    if not p.exists():return None
    r=c.read(p)
    c.require(r['graph_sha256']==graph_record['files']['graph.json.gz'] and
        r['structure_sha256']==graph_record['files']['structure.npz'] and r['encoder_sha256']==c.digest(identity),'Embedding identity differs')
    c.require(set(r['files'])=={'embeddings.npz'},'Unexpected embedding files')
    c.check_files(obj,r['files']);return r


def preflight(dataset,scale):
    root=directory(dataset,scale);plan=load_plan(root);saved=receipts(root,plan)
    stats=summarize(root,plan,saved)
    c.require(len(saved)==len(plan['entries']),'Finish graph stage before embedding preflight')
    identity=c.read(root/'encoder_identity.json') if (root/'encoder_identity.json').exists() else None
    remaining=0
    for k,r in saved.items():
        obj=root/'objects'/k
        if (obj/'embedding_receipt.json').exists():
            c.require(identity is not None,'Embedding receipt without encoder identity')
            embedding_receipt(obj,r,identity)
        else:remaining+=r['compact_embedding_bytes']
    needed=remaining*1.10+20*2**30
    print(f'[PREFLIGHT] remaining row payload {remaining/2**30:.2f} GiB; required including 10% margin + 20 GiB reserve {needed/2**30:.2f} GiB; free {shutil.disk_usage(root).free/2**30:.2f} GiB')
    c.require(shutil.disk_usage(root).free>needed,'Insufficient storage; use durable storage or review cleanup, not /tmp')
    return plan,saved


def embed(dataset,scale,device='cpu',threads=8,limit=None,parity=False):
    from .embeddings import Encoder
    c.require(threads>0 and (limit is None or limit>0),'Threads/limit must be positive')
    root=directory(dataset,scale)
    with writer_lock(root):
        plan,saved=preflight(dataset,scale)
        encoder=Encoder(device,threads);ip=root/'encoder_identity.json'
        if ip.exists():c.require(c.read(ip)==encoder.identity,'Different encoder/device/thread identity; do not mix caches')
        else:c.create(ip,encoder.identity)
        done=0
        for key in tqdm(sorted(saved),desc=f'Embed compact {dataset} pooling rows ({device})',unit='graph'):
            obj=root/'objects'/key;r=saved[key]
            if embedding_receipt(obj,r,encoder.identity):continue
            if limit is not None and done>=limit:break
            c.require(shutil.disk_usage(root).free>r['compact_embedding_bytes']*1.1+20*2**30,'Disk reserve reached')
            graph=c.read(obj/'graph.json.gz');arrays,metadata=compact.load(obj/'structure.npz')
            vectors,mask,coverage=encoder.encode_compact(plan['entries'][key]['text'],graph,arrays)
            if parity:
                reference,reference_mask,_=encoder.encode(plan['entries'][key]['text'],graph)
                c.require(torch.equal(reference,torch.from_numpy(vectors[arrays['pool_index']])) and
                          torch.equal(reference_mask,torch.from_numpy(mask[arrays['pool_index']])),'Dense/compact embedding parity failed')
            tmp=root/'.embedding_staging'/(key+'-'+uuid4().hex);tmp.mkdir(parents=True)
            compact.save(tmp/'embeddings.npz',{'rows':vectors,'observed':mask},{'format':compact.FORMAT,'encoder_sha256':c.digest(encoder.identity)})
            er={'graph_sha256':r['files']['graph.json.gz'],'structure_sha256':r['files']['structure.npz'],
                'encoder_sha256':c.digest(encoder.identity),'coverage':coverage,'dense_parity_verified':parity,
                'files':{'embeddings.npz':c.sha(tmp/'embeddings.npz')}}
            c.create(tmp/'embedding_receipt.json',er)
            # Resume preserves an unreceipted payload left between the two renames.
            if (obj/'embeddings.npz').exists():
                q=root/'.embedding_staging'/('unreceipted-'+key+'-'+uuid4().hex);q.mkdir()
                (obj/'embeddings.npz').rename(q/'embeddings.npz')
            for name in ('embeddings.npz','embedding_receipt.json'):
                with (tmp/name).open('rb') as f:os.fsync(f.fileno())
                (tmp/name).rename(obj/name)
            tmp.rmdir();done+=1
        print(f'v3 compact embedding stage complete: {root}; no training.')


def load_tensors(root,key,require_embeddings=True):
    """Train-time adapter: expand only the requested graph, not the whole corpus."""
    root=Path(root);obj=root/'objects'/key;record=c.read(obj/'graph_receipt.json')
    c.require(record['entry_sha256']==key,'Wrong graph object identity')
    c.check_files(obj,record['files'])
    arrays,metadata=compact.load(obj/'structure.npz');tensors=compact.unpack(arrays,metadata)
    if require_embeddings:
        identity=c.read(root/'encoder_identity.json');er=embedding_receipt(obj,record,identity)
        c.require(er,'Embeddings not complete')
        emb,meta=compact.load(obj/'embeddings.npz')
        c.require(meta['format']==compact.FORMAT and meta['encoder_sha256']==c.digest(identity),'Embedding payload identity mismatch')
        tensors=compact.add_embeddings(tensors,arrays,emb['rows'],emb['observed'])
    return tensors


def verify(dataset,scale,require_embeddings=False,replay=False):
    root=directory(dataset,scale);plan=load_plan(root);saved=receipts(root,plan)
    c.require(len(saved)==len(plan['entries']),'Graph cache incomplete')
    for key,r in tqdm(saved.items(),desc='Verify corrected graphs, compact round trips and source identity',unit='graph'):
        obj=root/'objects'/key;g=c.read(obj/'graph.json.gz');graphs.validate(plan['entries'][key],g)
        expected=graphs.tensors(g);actual=load_tensors(root,key,require_embeddings)
        for name,value in expected.items():
            c.require(torch.equal(value,actual[name]) if isinstance(value,torch.Tensor) else value==actual[name],f'Tensor differs: {name}')
        layout,_=compact.load(obj/'structure.npz')
        wanted=compact.pooling_layout(g)
        c.require(all(np.array_equal(layout[k],v) for k,v in wanted.items()),'Pooling layout differs from graph')
        if require_embeddings:
            missing=set(actual['node_ids'][actual['token_mask'] & ~actual['embedding_mask']].tolist())
            er=c.read(obj/'embedding_receipt.json')
            c.require(missing=={x['node_id'] for x in er['coverage']['exceptions']},'Token-coverage exceptions differ')
        if replay:
            original=c.read(Path(plan['parent_directory'])/'objects'/r['parent_key']/'graph.json.gz')
            old_entry={**plan['entries'][key],'processing_identity':plan['processing']['parent_processing_sha256']}
            c.require(transform(original,plan['entries'][key],old_entry,r['parent_key'],r['parent_graph_sha256'])==g,'Correction replay differs')
    stats=summarize(root,plan,saved)
    print(f'v3 {dataset} {scale} verified: {stats["graphs"]} graphs; embeddings={require_embeddings}; no parser rerun/training.')
    return stats
