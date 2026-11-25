"""VK07 exact-identity reuse of frozen TPG chunks with explicit source provenance."""
from collections import Counter
from pathlib import Path
import shutil

from tqdm import tqdm
from . import multisource_data_v1 as prep, graph_cache_v1 as description
from .. import forecast_primary_cache_v2 as builder

data = prep.data


def reference_sources(directory,plan,sources,hash_files=True):
    """Resolve only exact, pinned source roots and entry contracts; never arbitrary paths."""
    references=data.read(directory/'chunk_references.json')
    allowed={str(s.resolve()):(s,lock) for s,lock in sources}
    data.require(set(references)<=set(plan['entries']), 'Unknown referenced chunks')
    tensor_roots={}
    for key,path in tqdm(references.items(),desc='Verify externally stored chunk files',unit='chunk'):
        data.require(path in allowed,'Unpinned external cache')
        source,lock=allowed[path];entry=plan['entries'][key]
        data.require(description.reusable(source,lock,entry),'Referenced entry identity differs')
        receipt=data.read(source/('receipts/'+key+'.json'))
        for name,h in receipt['files'].items():
            resolved=(source/name).resolve()
            data.require(source.resolve() in resolved.parents and lock['files'].get(name)==h, 'Unsafe/unlocked external file')
            if hash_files:data.require(data.sha(resolved)==h,'External chunk changed: '+name)
        tensor_roots['tensors/'+entry['stem']+'.pt']=str(source.resolve())
    return references,tensor_roots


def context(scale):
    p,directory,rows,labels,prepared = prep.load(scale)
    source_paths = [data.ROOT/description.config()['cache_root']/'full', data.ROOT/description.config()['reuse_root']]
    sources,resources = [],None
    for source in source_paths:
        lock = data.read(source/'artifact_lock.json')
        for name in ('run_identity.json','resources.json'):
            data.require(data.sha(source/name)==lock['files'][name], 'Changed cache resource identity')
        old_identity = data.read(source/'run_identity.json')
        candidate = data.read(source/'resources.json')
        data.check_pins(old_identity['source_pins'])
        data.check_pins(candidate['source_pins'])
        if resources is None:
            resources = candidate
        data.require(candidate==resources, 'Cache resource policies differ')
        sources.append((source,lock))
    plan = builder.old.base.plan([{k:v for k,v in r.items() if k!='structured'} for r in rows],
                                prepared['ids'],resources['protocol']['max_chunk_characters'])
    identity = {'schema':p['schema']+'_cache', 'scale':scale, 'ids':prepared['ids'],
        'resources_sha256':data.digest(resources),'plan_sha256':data.digest(plan),
        'pins':{**prepared['pins'],str(directory/'artifact_lock.json'):data.sha(directory/'artifact_lock.json'),
                **{str(s/'artifact_lock.json'):data.sha(s/'artifact_lock.json') for s,_ in sources}},
        'training':False,'test_graphs':False}
    return p,plan,identity,resources,sources


def prepare(scale, workers=2, resume=False, verify=False, check_only=False):
    from ..forecast_matched_multisource import writer_lock
    builder.torch.set_num_threads(1)
    data.require(1<=workers<=16, 'Workers must be 1..16')
    p,plan,identity,resources,sources = context(scale)
    output = data.ROOT/p['cache_root']/scale
    eligible, referenced_bytes = {}, 0
    for key,entry in tqdm(plan['entries'].items(), desc='Check exact multi-source reuse identities', unit='chunk'):
        for source,lock in sources:
            if description.reusable(source,lock,entry):
                eligible[key]=source
                receipt=data.read(source/('receipts/'+key+'.json'))
                referenced_bytes+=sum((source/name).stat().st_size for name in receipt['files'])
                break
    print(f'[VK07 CACHE] {len(plan["entries"])} chunks; {len(eligible)} exact reuse candidates; '
          f'{len(plan["entries"])-len(eligible)} missing; workers={workers}.',flush=True)
    print(f'[VK07 STORAGE] {referenced_bytes/1024**3:.2f} GiB reusable without copying; '
          f'{shutil.disk_usage(data.ROOT).free/1024**3:.2f} GiB free. Preserve original referenced caches.',flush=True)
    if check_only:
        print('VK07 cache preflight passed. No graph generation or training.'); return
    output.parent.mkdir(parents=True,exist_ok=True)
    with writer_lock(output.parent/'.writer.lock'):
        tokenizer = builder.tokenizer_for(resources)
        if verify:
            data.check_lock(output)
            data.require(data.read(output/'run_identity.json')==identity and data.read(output/'resources.json')==resources,'Cache identity differs')
            data.require(data.read(output/'cve_index.json')==plan['index'],'Index differs')
            for name,doc in plan['documents'].items():
                data.require(data.read(output/name)==doc,'Source document differs')
        elif output.exists():
            data.require(resume and not (output/'artifact_lock.json').exists(),'Use --verify-only for completed cache or --resume for incomplete cache')
            data.require(data.read(output/'run_identity.json')==identity,'Resume identity differs')
        else:
            output.mkdir(); builder.atomic_json(output/'run_identity.json',identity)
            builder.atomic_json(output/'resources.json',resources)
        provenance_path=output/'chunk_provenance.json'
        provenance=data.read(provenance_path) if provenance_path.exists() else {}
        reference_path=output/'chunk_references.json'
        references=data.read(reference_path) if reference_path.exists() else {}
        if not verify:
            missing=[]
            for key,entry in tqdm(plan['entries'].items(),desc='Reuse or validate accepted source chunks',unit='chunk'):
                receipt='receipts/'+key+'.json'
                if key in references:
                    data.require(key in eligible and references[key]==str(eligible[key].resolve()),'Resumed reference changed')
                    builder.check_entry(eligible[key],entry,resources,tokenizer)
                    provenance.setdefault(key,'external_exact_reuse:'+str(eligible[key].relative_to(data.ROOT)))
                elif (output/receipt).exists():
                    builder.check_entry(output,entry,resources,tokenizer)
                    provenance.setdefault(key,'verified_resumed_chunk')
                elif key in eligible:
                    source=eligible[key]
                    builder.check_entry(source,entry,resources,tokenizer)
                    references[key]=str(source.resolve())
                    provenance[key]='external_exact_reuse:'+str(source.relative_to(data.ROOT))
                    builder.atomic_json(reference_path,references)
                    builder.atomic_json(provenance_path,provenance)
                else:
                    missing.append(entry)
            builder.atomic_json(provenance_path,provenance)
            builder.atomic_json(reference_path,references)
            if missing:
                builder.bounded_build(output,missing,resources,workers,provenance)
            for name,doc in plan['documents'].items():builder.atomic_json(output/name,doc)
            builder.atomic_json(output/'cve_index.json',plan['index'])
        reference_sources(output,plan,sources,hash_files=False)
        audit=[]
        for entry in tqdm(plan['entries'].values(),desc='Verify source spans, embeddings and security evidence',unit='chunk'):
            location=Path(references[entry['key']]) if entry['key'] in references else output
            _,stats=builder.check_entry(location,entry,resources,tokenizer)
            graph=data.read(location/('tpg_json/no_coref/'+entry['stem']+'.json'))
            edges=[e for e in graph['edges'] if e['label'].startswith('SEC_')]
            for e in edges:
                extra=e['properties'].get('EXTRA',{})
                data.require(extra.get('status')=='candidate' and extra.get('source_text_type')==entry['source_text_type'],'Security provenance changed')
            limits=description.config()
            data.require(len(graph['vertices'])<=limits['graph_node_budget'] and len(graph['edges'])<=limits['graph_edge_budget'],'Oversized chunk; refuse truncation')
            audit.append({'cve':entry['cve'],'source':entry['source_text_type'],'key':entry['key'],
                          'statistics':stats,'security_edges':dict(Counter(e['label'] for e in edges)),
                          'example_security_edges':edges[:2]})
        summary={'schema':identity['schema'],'scale':scale,'cves':{s:len(v) for s,v in identity['ids'].items()},
                 'chunks':len(audit),'chunks_by_source':dict(Counter(r['source'] for r in audit)),
                 'provenance_counts':dict(Counter(provenance.values())),
                 'externally_reused_chunks':len(references),'externally_reused_bytes':referenced_bytes,
                 'training':False,'test_graphs':False,'text_changed':False,
                 'interpretation':'Separate source documents; no synthetic cross-source edges. Fixed candidate extractor, not learned relation discovery.'}
        data.require(set(provenance)==set(plan['entries']),'Missing chunk provenance')
        if verify:
            data.require(data.read(output/'security_audit.json')==audit and data.read(output/'summary.json')==summary,'Cache audit differs')
        else:
            builder.atomic_json(output/'security_audit.json',audit)
            builder.atomic_json(output/'summary.json',summary)
            (output/'summary.txt').write_text('VK07 accepted multi-source cache\n'+data.dump(summary)+
                'chunk_references.json maps reused chunk keys to their original cache roots. Preserve these roots.\n'+
                'Within each root: graphs in tpg_json/no_coref/, embeddings in tensors/. Newly built chunks are local.\n'+
                'Accepted source text and spans are local in documents/. No copies, hard links or symlinks are created for reused chunks.\n')
            builder.common.lock(output)
        data.check_pins(identity['pins'])
    print(f'VK07 {scale} source-aware cache '+('verified.' if verify else f'saved to {output}.')+' No training or test graphs.')
