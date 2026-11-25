"""Evaluation graph cache using the frozen v2 extraction/embedding implementation."""
from collections import Counter
import shutil

from . import protocol as p, data
from .. import forecast_primary_cache_v2 as v2

read,write,require,tqdm=p.read,p.write,p.require,p.tqdm


def audit(directory,plan,resources,tokenizer):
    rows=[]
    for entry in tqdm(plan['entries'].values(),desc='Verify evaluation graph tensors and spans',unit='chunk'):
        _,stats=v2.check_entry(directory,entry,resources,tokenizer);rows.append(stats)
    return rows


def prepare(scale,release_test=False,workers=2,resume=False,verify=False):
    p.access(scale,release_test)
    frozen,identity,rows,resources,state=data.load(scale,release_test)
    require(1<=workers<=16,'Workers must be 1..16');v2.torch.set_num_threads(1)
    plan=data.graph_plan(rows,identity['role'],resources['protocol']['max_chunk_characters'])
    identity={**identity,'data_lock_sha256':p.common.sha256(p.DATA/scale/'artifact_lock.json'),
              'coverage_policy':v2.coverage.POLICY}
    directory=p.CACHE/scale;tokenizer=v2.tokenizer_for(resources)
    if verify:
        p.common.verify_lock(directory)
        require(read(directory/'run_identity.json')==identity and read(directory/'resources.json')==resources,'Cache identity changed')
        require(read(directory/'cve_index.json')==plan['index'],'Cache index changed')
        for name,doc in plan['documents'].items():require(read(directory/name)==doc,'Source document changed')
        require(audit(directory,plan,resources,tokenizer)==read(directory/'coverage_audit.json'),'Cache coverage changed')
        require(set(read(directory/'chunk_provenance.json'))==set(plan['entries']),'Missing cache provenance')
        print(f'PS05 {scale} cache verified. No encoder rerun or scoring.');return
    if directory.exists():
        require(resume and not (directory/'artifact_lock.json').exists(),'Cache exists; use --verify-only or --resume for incomplete cache')
        require(read(directory/'run_identity.json')==identity and read(directory/'resources.json')==resources,'Resume cache mismatch')
    else:
        directory.mkdir(parents=True);write(directory/'run_identity.json',identity);write(directory/'resources.json',resources)
    provenance=read(directory/'chunk_provenance.json') if (directory/'chunk_provenance.json').exists() else {}
    missing=[];old=v2.OUTPUT/'full'
    # Exact-CVE chunk keys mean reserved test records cannot reuse development chunks.
    # Smoke deliberately reuses the identical development documents and tensors.
    old_lock=read(old/'artifact_lock.json') if scale=='smoke' else None
    for entry in tqdm(plan['entries'].values(),desc='Reuse verified evaluation chunks or queue builds',unit='chunk'):
        name='receipts/'+entry['key']+'.json'
        if (directory/name).exists():
            v2.check_entry(directory,entry,resources,tokenizer)
            provenance.setdefault(entry['key'],'resumed_verified_chunk')
        elif old_lock is not None and name in old_lock['files']:
            require(p.common.sha256(old/name)==old_lock['files'][name], 'Changed source receipt')
            receipt=read(old/name)
            require(v2.old.reusable(receipt,entry),'Smoke source/processing identity differs')
            require(read(old/'resources.json')==resources,'Smoke embedding resources differ')
            for path,h in receipt['files'].items():require(old_lock['files'].get(path)==h,'Unauthenticated source chunk')
            v2.check_entry(old,entry,resources,tokenizer)
            for path in [name,*receipt['files']]:
                target=directory/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/path,target)
            provenance[entry['key']]='verified_development_smoke_reuse'
        else:missing.append(entry)
    v2.atomic_json(directory/'chunk_provenance.json',provenance)
    print(f'[PS05 CACHE] {len(provenance)} verified chunks; {len(missing)} to build with workers={workers}. Frozen extractor and encoder.',flush=True)
    if missing:v2.bounded_build(directory,missing,resources,workers,provenance)
    for name,doc in plan['documents'].items():write(directory/name,doc)
    write(directory/'cve_index.json',plan['index'])
    checked=audit(directory,plan,resources,tokenizer);write(directory/'coverage_audit.json',checked)
    require(set(provenance)==set(plan['entries']),'Incomplete graph cache')
    summary={'scale':scale,'role':identity['role'],'cves':len(rows),'documents':len(plan['documents']),
             'chunks':len(plan['entries']),'provenance_counts':dict(Counter(provenance.values())),
             'training':False,'test_graphs':scale=='full','test_scoring':False}
    p.common.write_report(directory,summary,['PS05 evaluation-only source cache',str(summary),
        'Same v2 security extractor, node embeddings, source labels and graph views as development.',
        'Test roles remain test in documents, index and receipts; no target scores enter graph construction.'])
    p.common.check_pins(frozen['source_pins']);p.common.lock(directory)
    print(f'PS05 {scale} cache prepared. No model training or scoring.')
