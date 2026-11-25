"""Verify the bounded v3 smoke and publish storage/readiness TXT and JSON reports."""
import argparse
from collections import Counter
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
for rel in ('01_tpg/01_core','01_tpg/10_extensions','02_prediction/01_package'):
    sys.path.insert(0,str(ROOT/rel))
from tpg_combined_v3 import cache
from tpg_combined_v2 import common as c

BASE=ROOT/'05_results/08_fresh_studies/06_storage_and_compact_cache_v1'
OUTPUT=BASE/'smoke_verification'


def findings():
    datasets,inputs={},{}
    for dataset in ('reddit','previous'):
        root=cache.directory(dataset,'smoke')
        stats=cache.verify(dataset,'smoke',require_embeddings=True,replay=True)
        stats.pop('free_disk_gib')
        plan=cache.load_plan(root);changes=Counter();exceptions=0;stored=Counter()
        for key in plan['entries']:
            obj=root/'objects'/key;r=c.read(obj/'graph_receipt.json');e=c.read(obj/'embedding_receipt.json')
            c.require(e['dense_parity_verified'],'Real dense embedding comparison was not performed')
            changes.update(r['correction_counts']);exceptions+=len(e['coverage']['exceptions'])
            for name in ('graph.json.gz','structure.npz','embeddings.npz','graph_receipt.json','embedding_receipt.json'):
                inputs[str(obj/name)]=c.sha(obj/name);stored[name]+=(obj/name).stat().st_size
            parent=Path(plan['parent_directory'])/'objects'/r['parent_key']
            for name in ('graph.json.gz','graph_receipt.json'):inputs[str(parent/name)]=c.sha(parent/name)
        for scale in ('smoke','full'):
            other=cache.directory(dataset,scale)
            for name in ('plan.json.gz','plan_lock.json'):inputs[str(other/name)]=c.sha(other/name)
        inputs[str(root/'encoder_identity.json')]=c.sha(root/'encoder_identity.json')
        full=cache.load_plan(cache.directory(dataset,'full'))
        datasets[dataset]={**stats,'smoke_cves':len(plan['index']),'full_planned_cves':len(full['index']),
            'full_planned_objects':len(full['entries']), 'full_split_counts':dict(Counter(r['split'] for r in full['index'].values())),
            'annotation_findings':dict(changes),'tokenizer_discarded_exceptions':exceptions,'stored_file_bytes':dict(stored),
            'all_smoke_embeddings_equal_dense':True,'test_adapted':False}
    for name in cache.pins():inputs[str(ROOT/name)]=c.sha(ROOT/name)
    inputs[str(Path(__file__).resolve())]=c.sha(__file__)
    inputs[str(BASE/'storage/storage_inventory.json')]=c.sha(BASE/'storage/storage_inventory.json')
    return {'schema':'combined_v3_readiness_v1','datasets':datasets,'input_sha256':inputs,
            'full_cache_built':False,'training_run':False,'files_deleted':0,
            'limits':['Smoke savings are not a full-corpus storage measurement.',
                      'Reviewed annotation fixes do not establish complete semantic accuracy.',
                      'Sparse edges are expanded per graph at runtime; full-graph RAM requirements remain.',
                      'Original v2 objects are required for provenance verification; do not delete them.',
                      'No new source filtering, summaries, split assignments or labels were created.']}


def report(d):
    storage=d['storage_observation'];rows=['STORAGE AND CORRECTED COMPACT-CACHE READINESS',
        'No raw data, original graphs, checkpoints or experimental results were deleted.',
        f'Free project-filesystem space when this report was saved: {storage["free_bytes"]/2**30:.2f} GiB.',
        f'Conda package-download archives, dry-run only: {storage["conda_tarballs_bytes"]/2**30:.2f} GiB.',
        'These archives could be removed without removing installed environments; cleanup was NOT run.',
        'Pip download cache is about 647 MiB; retain the roughly 9 GiB Hugging Face model cache.',
        'Project raw data (4.93 GiB), Git (2.54 GiB) and completed graphs should be preserved.',
        'Only about 79 MiB is archived and 8.69 MiB is Python/test cache: neither is a useful space target.',
        'The storage audit does not account for every filesystem user, container layer or open deleted file.',
        '',f'Combined-graph regression tests: {d["regression_tests"]} passed.',
        'Fresh verification reconstructs tensors and replays annotation corrections; it does not rerun the encoder.',
        'Dense-reference embedding parity was checked during the real CPU smoke generation.', '']
    for name,r in d['datasets'].items():
        rows += [name.upper(),f'Smoke: {r["smoke_cves"]} CVEs, {r["graphs"]} distinct graphs, {r["nodes"]} nodes, {r["edges"]} edges.',
            f'Full plan only: {r["full_planned_cves"]} CVEs, {r["full_planned_objects"]} objects; roles {r["full_split_counts"]}.',
            f'Node embedding payload: {r["dense_node_embedding_bytes"]/2**20:.3f} MiB dense -> {r["compact_embedding_bytes"]/2**20:.3f} MiB shared rows.',
            f'Edge feature payload: {r["dense_edge_feature_bytes"]/2**20:.3f} MiB dense -> {r["sparse_edge_feature_bytes"]/2**20:.3f} MiB sparse.',
            f'Annotation findings: {r["annotation_findings"]}.',
            f'Tokenizer-discarded token exceptions: {r["tokenizer_discarded_exceptions"]}. No ordinary-token gaps allowed.',
            'All smoke vectors, masks and structural tensors reconstruct exactly.','']
    rows += ['WHAT CHANGES',
        'The reviewed correction pass updates a separate graph copy: passive-agent objects, narrowly mislabelled code words,',
        'generic duplicate CVE/CWE labels, identifier-only topics and context-supported CIS Safeguard headings.',
        'Original text, document boundaries, linguistic syntax, source masks and split roles stay unchanged.',
        'Compact storage shares exact pooling spans only within one graph/context. It does not merge nodes.',
        'Edge features retain all nonzero float32 values and restore the original 88-column tensor when loaded.',
        'SecureBERT still runs AFTER graph creation. No float16 quantization or cross-context embedding reuse.',
        '', 'NEXT RUN',
        'Run the full graph correction/compact export, then the storage preflight, then frozen embeddings.',
        'The preflight checks remaining raw vector payload plus 10% margin and a 20 GiB disk reserve.',
        'Use one embedding device/thread configuration per cache. Full training has not started.',
        'Commands: 00_documentation/02_tpg/12_COMPACT_CORRECTED_CACHE.md',
        '', 'LIMITATIONS', *['- '+x for x in d['limits']]]
    return '\n'.join(rows)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify-only',action='store_true');args=p.parse_args()
    if args.verify_only:
        c.check_files(OUTPUT,c.read(OUTPUT/'report_lock.json')['files'])
        old=c.read(OUTPUT/'findings.json')
        for name,sha in old['input_sha256'].items():c.require(c.sha(name)==sha,f'Changed report input: {name}')
        current=findings()
        current.update(storage_observation=old['storage_observation'],regression_tests=old['regression_tests'])
        c.require(current==old,'Readiness findings do not reproduce')
        c.require((OUTPUT/'REPORT.txt').read_text()==report(current),'Report text differs')
        print(f'v3 smoke storage/readiness report reproduced: {OUTPUT}; no encoder rerun or training.');return
    c.require(not OUTPUT.exists(),'Report exists; use --verify-only')
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'01_tpg/08_tests'),'-p','test_combined*.py'],
        cwd=ROOT.parent,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    c.require(result.returncode==0,result.stdout)
    count=int(re.search(r'Ran (\d+) tests',result.stdout).group(1))
    conda=Path.home()/'.miniconda3/bin/conda'
    dry=subprocess.run([str(conda),'clean','--tarballs','--dry-run','--json'],text=True,capture_output=True,check=True)
    import json
    sizes=json.loads(dry.stdout)
    current=findings();current.update(regression_tests=count,storage_observation={
        'free_bytes':shutil.disk_usage(ROOT).free,'conda_tarballs_bytes':sizes['tarballs']['total_size']})
    OUTPUT.mkdir(parents=True)
    c.create(OUTPUT/'findings.json',current);c.create(OUTPUT/'conda_tarball_dry_run.json',sizes)
    for name,text in (('REPORT.txt',report(current)),('regression_tests.txt',result.stdout)):
        with (OUTPUT/name).open('x') as f:f.write(text)
    c.create(OUTPUT/'report_lock.json',{'files':{p.name:c.sha(p) for p in sorted(OUTPUT.iterdir())}})
    print(f'v3 smoke storage/readiness report saved: {OUTPUT}; no cleanup or training.')


if __name__=='__main__':main()
