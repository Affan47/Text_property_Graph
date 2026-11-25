"""Operational full-cache builder around the frozen RD02 extraction contract."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing
import shutil

import torch
from tqdm import tqdm
from . import common as c
from ..reddit_primary import cache, data

RESERVE = 8 * 1024**3


def check_plan(scale):
    entries,index,res,identity=cache.plan(scale)
    c.require(res['chunk_characters']==1200 and res['embedding_dim']==768,'Frozen chunk/encoder contract differs')
    vocab=res['type_vocabulary']
    c.require(vocab['num_node_types']==13 and vocab['num_edge_types']==23 and
              vocab['edge_types']['SEC_HAS_VERSION']==14 and len(res['subtypes'])==32 and res['subtypes']['SEVERITY']==31,
              'Frozen node/edge/subtype vocabulary differs')
    c.require(res['encoder']['revision']=='7c603df5bc4c5ba9c731bcc2ea0ab2db36e104cb','Encoder revision differs')
    counts=dict(cves=len(index),documents=sum(len(r['documents']) for r in index.values()),chunks=len(entries))
    if scale=='full':
        c.require(counts==dict(cves=25185,documents=53052,chunks=66800),'Approved graph counts differ')
    return entries,index,res,identity,counts


def check_chunk(directory,e,res):
    stats=cache.check(directory,e,res)
    graph=c.read(directory/'graphs'/f'{e["key"]}.json')
    tensor=torch.load(directory/'tensors'/f'{e["key"]}.pt',map_location='cpu',weights_only=True)
    c.require(tensor['x'].shape==(len(graph['vertices']),782),'Wrong node tensor width')
    c.require(tensor['subtype'].shape==(len(graph['vertices']),) and
              bool(((tensor['subtype']>=0)&(tensor['subtype']<32)).all()),'Subtype channel out of range')
    for node in graph['vertices']:
        p=node['properties'];extra=p.get('EXTRA',{});s,t=p['CHAR_START'],p['CHAR_END']
        valid=bool(extra.get('span_available',True) and 0<=s<t<=len(e['text']))
        c.require(extra.get('source_text_type')==e['source_text_type'] and extra.get('source_span_available')==valid,
                  'Source type/span provenance differs')
        c.require(extra.get('source_document_char_start')==(e['chunk']['start']+s if valid else None) and
                  extra.get('source_document_char_end')==(e['chunk']['start']+t if valid else None),'Parent source offsets differ')
        if valid and node['label']=='TOKEN':
            c.require(p['TEXT']==e['text'][s:t],'Token source text mismatch')
    actual=dict(nodes=len(graph['vertices']),edges=len(graph['edges']),
                security_edges=dict(Counter(x['label'] for x in graph['edges'] if x['label'].startswith('SEC_'))),
                severity_entities=int((tensor['subtype']==31).sum()))
    c.require(all(stats[k]==v for k,v in actual.items()),'Receipt counts differ from saved graph')
    return stats


def prepare(scale,workers=4,resume=False,verify=False):
    c.gate();entries,index,res,identity,counts=check_plan(scale);directory=cache.DEFAULT/scale
    c.require(1<=workers<=16,'workers must be 1..16')
    completed=(directory/'artifact_lock.json').exists()
    if completed:
        c.require(verify,'Completed cache is immutable; use --verify-only');c.core.check_lock(directory)
    elif verify:
        raise ValueError('No completed cache to verify')
    elif directory.exists():
        c.require(resume,'Partial cache exists; use --resume')
    else:
        c.require(shutil.disk_usage(directory.parent).free>max(512*1024**2,len(entries)*700000)+RESERVE,
                  'Need estimated remaining cache space plus 8 GiB operating reserve')
        directory.mkdir(parents=True)
        cache.atomic(directory/'run_identity.json',identity);cache.atomic(directory/'resources.json',res);cache.atomic(directory/'cve_index.json',index)
    c.require(c.read(directory/'run_identity.json')==identity and c.read(directory/'resources.json')==res
              and c.read(directory/'cve_index.json')==index,'Existing cache identity differs')
    missing=[]
    reuse=[cache.DEFAULT/'gate',cache.DEFAULT/'smoke']
    for key,e in tqdm(entries.items(),desc='Check or reuse exact RD02 chunks',unit='chunk'):
        if (directory/'receipts'/f'{key}.json').exists():
            continue
        c.require(not verify,'Missing completed receipt')
        source=next((p for p in reuse if p!=directory and (p/'receipts'/f'{key}.json').exists()),None)
        if source:
            check_chunk(source,e,res)
            for rel in (f'graphs/{key}.json',f'tensors/{key}.pt',f'receipts/{key}.json'):
                dest=directory/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                tmp=dest.with_suffix(dest.suffix+'.copying');shutil.copyfile(source/rel,tmp);tmp.replace(dest)
        else:missing.append(e)
    if missing:
        c.require(shutil.disk_usage(directory).free>RESERVE,'Below operating reserve; move cache to durable larger storage')
        iterator=iter(missing);pending=set()
        pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=cache.worker_init,initargs=(res,))
        def submit():
            e=next(iterator,None)
            if e is not None:pending.add(pool.submit(cache.worker_build,(str(directory),e)))
        try:
            for _ in range(min(len(missing),workers*2)):submit()
            with tqdm(total=len(missing),desc='Build frozen v6/SecBERT chunks',unit='chunk') as bar:
                while pending:
                    done,_=wait(pending,return_when=FIRST_COMPLETED)
                    for f in done:
                        pending.remove(f);f.result();bar.update()
                        c.require(shutil.disk_usage(directory).free>RESERVE,'Operating reserve reached; cache remains resumable')
                        submit()
        finally:
            for f in pending:f.cancel()
            pool.shutdown(wait=True,cancel_futures=True)
    statistics=[check_chunk(directory,e,res) for e in tqdm(entries.values(),desc='Verify spans, embeddings, edges and subtypes',unit='chunk')]
    summary=dict(cves={s:len(ids) for s,ids in identity['ids'].items()},documents=counts['documents'],chunks=counts['chunks'],
        nodes=sum(r['nodes'] for r in statistics),edges=sum(r['edges'] for r in statistics),
        severity_entities=sum(r['severity_entities'] for r in statistics),test_graphs=False,training=False,
        security_edge_counts=dict(sum((Counter(r['security_edges']) for r in statistics),Counter())),
        interpretation='Frozen rule-based candidate extraction; structural validity is not semantic accuracy.')
    if verify:c.require(c.read(directory/'summary.json')==summary,'Cache summary differs')
    else:
        cache.atomic(directory/'summary.json',summary);c.core.check_pins(identity['source_pins']);c.core.lock(directory)
    pointer=data.OUT/f'cache_{scale}_location.json';location=dict(directory=str(directory),artifact_lock_sha256=c.sha(directory/'artifact_lock.json'))
    if pointer.exists():c.require(c.read(pointer)==location,'Existing cache pointer differs')
    else:cache.atomic(pointer,location)
    print(f'RD02 {scale} cache '+('verified' if verify else 'saved')+f': {summary}; SHA256={location["artifact_lock_sha256"]}. No test graphs or training.')
