"""PS01 cache v2: tokenizer-proven missing nonspacing marks, immutable v1 reuse."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
import multiprocessing
from pathlib import Path
import shutil
import unicodedata

from .primary_study import cache as old, common
from . import forecast_primary_mark_coverage as coverage
from .forecast_matched_multisource import writer_lock

read,write,require,sha256,tqdm=common.read,common.write,common.require,common.sha256,common.tqdm
torch=old.torch
OUTPUT=common.ROOT/'01_tpg/05_workspace/primary_day60_v2'
SCRIPT=common.SCRIPTS/'04_prepare_primary_cache_v2.py'
TEST=common.SCRIPTS/'test_primary_mark_coverage.py'
SCHEMA='ps01_primary_cache_v2'
FAILURE='752194e850b7527602c7f6d30cb173daa696dcd211ddf533411f11a36fcd0056'
_worker=None


def atomic_json(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    write(temp,value);temp.replace(path)


def context(scale):
    plan,data_identity,resources=old.graph_plan(scale)
    pins={str(p):sha256(p) for p in (Path(__file__),Path(coverage.__file__),SCRIPT,TEST)}
    identity={'schema':SCHEMA,'scale':scale,'data_identity':data_identity,'source_pins':pins,
        'coverage_policy':coverage.POLICY,'unicode_database':unicodedata.unidata_version,
        'policy':'Only missing Mn-only tokens proven removed by the pinned tokenizer and uncovered by contextual offsets; retain source text, nodes, spans, zero vectors and zero masks.',
        'training':False,'test_graphs':False}
    return plan,identity,resources


def tokenizer_for(resources):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(resources['encoder']['model_name'],
        revision=resources['encoder']['revision'],use_fast=True,local_files_only=True,trust_remote_code=False)
    coverage.check_tokenizer(tokenizer,resources['encoder'])
    return tokenizer


class Builder(old.builder_api.Builder):
    def __init__(self,resources):
        super().__init__(resources,False)
        coverage.check_tokenizer(self.encoder.tokenizer,resources['encoder'])

    def build(self,entry):
        graph=old.base.annotate(self.pipeline.run(entry['text'],doc_id=entry['graph_id']),entry)
        saved=json.loads(old.base.old.shared.GraphSONExporter().export_string(graph))
        docs,_=old.base.old.shared.preview.variants(saved)
        old.base.old.validate_documents(docs,entry)
        bundle,vocab,marks=coverage.build_bundle(docs,self.encoder)
        return docs,bundle,vocab,marks


def statistics(entry,docs,bundle,vocab,marks):
    row=old.builder_api.statistics(entry,docs,bundle,vocab)
    row['unembedded_whitespace_tokens']-=len(marks)
    require(row['unembedded_whitespace_tokens']>=0,'Wrong missing-token accounting')
    row['unembedded_nonspacing_tokens']=marks
    return row


def save_entry(directory,entry,resources,builder):
    docs,bundle,vocab,marks=builder.build(entry)
    require(vocab==resources['type_vocabulary'],'Vocabulary changed')
    for mode,graph in docs.items():write(directory/'tpg_json'/mode/(entry['stem']+'.json'),graph)
    path=directory/'tensors'/(entry['stem']+'.pt');path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.pt.tmp');torch.save(bundle,temporary);temporary.replace(path)
    atomic_json(directory/'receipts'/(entry['key']+'.json'),{
        'entry':{k:v for k,v in entry.items() if k!='text'},'embedding_coverage_policy':coverage.POLICY,
        'statistics':statistics(entry,docs,bundle,vocab,marks),
        'files':{name:sha256(directory/name) for name in sorted(old.base.files(entry))}})


def check_entry(directory,entry,resources,tokenizer):
    receipt=read(directory/'receipts'/(entry['key']+'.json'))
    if receipt.get('embedding_coverage_policy')!=coverage.POLICY:
        # Legacy receipts must still pass their original, stricter validator.
        bundle,row=old.builder_api.check_entry(directory,entry,resources)
        return bundle,{**row,'unembedded_nonspacing_tokens':[]}
    require(old.reusable(receipt,entry),'Wrong chunk identity/file contract')
    for name,h in receipt['files'].items():require(sha256(directory/name)==h,'Changed chunk: '+name)
    docs={m:read(directory/'tpg_json'/m/(entry['stem']+'.json')) for m in coverage.legacy.MODES}
    old.base.old.validate_documents(docs,entry)
    for graph in docs.values():
        require(graph['metadata']['source_chunk']=={'parent_text_sha256':entry['parent_text_sha256'],**entry['chunk']},'Changed chunk offsets')
        for node in graph['vertices']:
            props=node['properties'];extra=props['EXTRA']
            available=bool(extra.get('span_available',True) and 0<=props['CHAR_START']<props['CHAR_END']<=len(entry['text']))
            require(extra['source_span_available']==available and
                extra['source_document_char_start']==(entry['chunk']['start']+props['CHAR_START'] if available else None) and
                extra['source_document_char_end']==(entry['chunk']['start']+props['CHAR_END'] if available else None),
                'Changed parent-document offsets')
    bundle=torch.load(directory/'tensors'/(entry['stem']+'.pt'),weights_only=True,map_location='cpu')
    marks=coverage.validate_bundle(bundle,docs,resources['type_vocabulary'],resources['embedding_dim'],tokenizer)
    row=statistics(entry,docs,bundle,resources['type_vocabulary'],marks)
    require(row==receipt['statistics'],'Nonspacing audit/statistics differ')
    return bundle,row


def reuse_partial(source,directory,entry,resources,data_identity):
    receipt=source/'receipts'/(entry['key']+'.json')
    if not receipt.exists():return False
    require(read(source/'run_identity.json')==data_identity and read(source/'resources.json')==resources,
            'V1 partial-cache source identity differs')
    saved=read(receipt)
    require(old.reusable(saved,entry),'V1 partial receipt does not match current text/processing identity')
    old.builder_api.check_entry(source,entry,resources)
    for name in [*saved['files'],'receipts/'+entry['key']+'.json']:
        target=directory/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,target)
    return True


def worker_init(resources):
    global _worker
    torch.set_num_threads(1);_worker=(resources,Builder(resources))


def worker_build(payload):
    directory,entry=payload;resources,builder=_worker
    save_entry(Path(directory),entry,resources,builder)
    return entry['key']


def bounded_build(directory,entries,resources,workers,provenance):
    """At most two tasks per worker are queued; failures do not drain a full corpus queue."""
    iterator=iter(entries)
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=worker_init,initargs=(resources,))
    pending=set()
    def submit():
        entry=next(iterator,None)
        if entry is not None:pending.add(pool.submit(worker_build,(str(directory),entry)))
    try:
        for _ in range(min(len(entries),workers*2)):submit()
        with tqdm(total=len(entries),desc='Build missing v2 chunks',unit='chunk') as bar:
            while pending:
                done,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    pending.remove(future);key=future.result();provenance[key]='new_v2_build';bar.update()
                    submit()
    finally:
        for future in pending:future.cancel()
        pool.shutdown(wait=True,cancel_futures=True)
        atomic_json(directory/'chunk_provenance.json',provenance)


def summarize(plan,directory,resources,tokenizer):
    rows=[];marks=[];edges=Counter()
    for entry in tqdm(plan['entries'].values(),desc='Verify graph/tensor/span coverage',unit='chunk'):
        _,row=check_entry(directory,entry,resources,tokenizer);rows.append(row)
        edges.update({k:v for k,v in row['edge_type_counts'].items() if k.startswith('SEC_')})
        marks.extend({'key':entry['key'],'cve':entry['cve'],'source':entry['source_text_type'],
                      'parent_chunk_start':entry['chunk']['start'],**m} for m in row['unembedded_nonspacing_tokens'])
    return {'chunks':rows,'removed_nonspacing_tokens':marks,'removed_nonspacing_count':len(marks),
            'security_edge_counts':dict(edges),'training':False,'test_graphs':False}


def prepare(scale,workers,resume=False,verify=False):
    torch.set_num_threads(1)
    plan,identity,resources=context(scale);directory=OUTPUT/scale;tokenizer=tokenizer_for(resources)
    if verify:
        common.verify_lock(directory)
        require(read(directory/'run_identity.json')==identity and read(directory/'resources.json')==resources,'Changed v2 cache identity')
        require(read(directory/'cve_index.json')==plan['index'],'Changed CVE index')
        for name,doc in plan['documents'].items():require(read(directory/name)==doc,'Changed source document')
        require(summarize(plan,directory,resources,tokenizer)==read(directory/'coverage_audit.json'),'Audit does not reproduce')
        require(set(read(directory/'chunk_provenance.json'))==set(plan['entries']),'Wrong completed chunk set')
        common.check_pins(identity['source_pins'])
        print(f'PS01 v2 {scale} cache verified. No encoder rerun or training.');return
    require(1<=workers<=16,'Workers must be 1..16')
    if directory.exists():
        require(resume and not (directory/'artifact_lock.json').exists(),'Output exists; use --verify-only, or --resume for incomplete v2')
        require(read(directory/'run_identity.json')==identity and read(directory/'resources.json')==resources,'Resume contract changed')
    else:
        directory.mkdir(parents=True);write(directory/'run_identity.json',identity);write(directory/'resources.json',resources)
    provenance=read(directory/'chunk_provenance.json') if (directory/'chunk_provenance.json').exists() else {}
    old_lock=read(common.OLD/'cache_lock.json');missing=[]
    source=common.CACHE/scale
    for entry in tqdm(plan['entries'].values(),desc='Preserve verified v1/v2 completed chunks',unit='chunk'):
        receipt=directory/'receipts'/(entry['key']+'.json')
        if receipt.exists():
            check_entry(directory,entry,resources,tokenizer)
            provenance.setdefault(entry['key'],'new_v2_build' if read(receipt).get('embedding_coverage_policy')==coverage.POLICY else 'legacy_reuse_origin_unrecorded')
        elif reuse_partial(source,directory,entry,resources,identity['data_identity']):provenance[entry['key']]='verified_ps01_v1'
        elif old.copy_verified(directory,entry,resources,old_lock):provenance[entry['key']]='verified_cd03'
        else:missing.append(entry)
    atomic_json(directory/'chunk_provenance.json',provenance)
    print(f'[V2] {len(provenance)} verified completed chunks; {len(missing)} missing. Original cache unchanged.',flush=True)
    if missing:bounded_build(directory,missing,resources,workers,provenance)
    for name,doc in plan['documents'].items():write(directory/name,doc)
    write(directory/'cve_index.json',plan['index'])
    audit=summarize(plan,directory,resources,tokenizer);write(directory/'coverage_audit.json',audit)
    summary={'schema':SCHEMA,'scale':scale,'chunks':len(plan['entries']),'provenance_counts':dict(Counter(provenance.values())),
             'nonspacing_tokens_without_embeddings':audit['removed_nonspacing_count'],
             'coverage_policy':coverage.POLICY,'source_text_changed':False,'training':False,'test_graphs':False}
    common.write_report(directory,summary,['PS01 v2 primary cache',str(summary),
        'Only Mn-only missing tokens with empty pinned normalization and no contextual tokenizer overlap are exempt.',
        'These nodes retain source text/spans and explicit zero embedding masks; ordinary missing word embeddings fail.',
        'Legacy receipts pass their original policy. Original v1 cache/data/results remain unchanged.',
        'coverage_audit.json lists affected CVEs, codepoints, spans and source context.',
        'This addresses tokenizer coverage, not the quality of corrupted source text.'])
    common.check_pins(identity['source_pins']);common.check_pins(identity['data_identity']['source_pins']);common.lock(directory)
    print(f'PS01 v2 {scale} cache saved to {directory}. No training or test graphs.')


def check_failure():
    torch.set_num_threads(1)
    plan,identity,resources=context('full');entry=plan['entries'][FAILURE]
    directory=OUTPUT/'checks/failing_chunk';tokenizer=tokenizer_for(resources)
    if directory.exists():
        common.verify_lock(directory)
        require(read(directory/'run_identity.json')==identity,'Changed failure-check identity')
        _,stats=check_entry(directory,entry,resources,tokenizer)
        require(stats==read(directory/'checked_statistics.json'),'Failure check changed')
        print('PS01 v2 failing-chunk check verified; no encoder rerun.');return
    builder=Builder(resources)
    # Keep the original validator unchanged and explicitly reproduce its rejection.
    docs,bundle,vocab,marks=builder.build(entry)
    original_error=None
    try:old.builder_api.coverage.validate_bundle(bundle,docs,vocab,resources['embedding_dim'])
    except ValueError as exc:original_error=str(exc)
    require(original_error and 'Missing ordinary text-token embeddings' in original_error,'Original failure was not reproduced')
    require(any('U+06E7' in m['codepoints'] for m in marks),'Reported failing character not recovered')
    directory.mkdir(parents=True);write(directory/'run_identity.json',identity)
    # Reuse this bounded build in memory; no second encoder pass.
    class Ready:
        def build(self,e):
            require(e==entry,'Wrong bounded entry');return docs,bundle,vocab,marks
    save_entry(directory,entry,resources,Ready())
    _,stats=check_entry(directory,entry,resources,tokenizer)
    write(directory/'checked_statistics.json',stats)
    common.write_report(directory,{'cve':entry['cve'],'source':entry['source_text_type'],'key':FAILURE,
        'original_error':original_error,'verified_removed_marks':marks,'nodes':stats['nodes'],
        'embedded_nodes':stats['embedded_nodes'],'ordinary_word_coverage_verified':True,'text_changed':False},
        ['PS01 v2 real failing-chunk regression',entry['cve']+' / '+entry['source_text_type'],original_error,
         'U+06E7 is a nonspacing mark removed by the pinned SecBERT normalizer, not an ordinary word.',
         'Original text, nodes, spans and edges are retained. The missing mark stays zero/masked and is excluded from token pooling.',
         'All ordinary lexical embeddings and all graph/tensor invariants passed. No full expansion or training.'])
    common.lock(directory);print(f'PS01 v2 real failing chunk checked and saved to {directory}.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale',choices=('smoke','full'),default='smoke')
    parser.add_argument('--workers',type=int,default=2)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--resume',action='store_true')
    mode.add_argument('--verify-only',action='store_true');mode.add_argument('--check-failure',action='store_true')
    args=parser.parse_args();OUTPUT.mkdir(parents=True,exist_ok=True)
    with writer_lock(OUTPUT/'.writer.lock'):
        if args.check_failure:check_failure()
        else:prepare(args.scale,args.workers,args.resume,args.verify_only)


if __name__=='__main__':main()
