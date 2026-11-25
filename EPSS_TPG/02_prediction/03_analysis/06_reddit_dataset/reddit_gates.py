"""RD00 storage and pre-modeling gates. No training or split assignment."""
import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile

from tqdm import tqdm
from audit_reddit import ROOT, DEFAULT_DATA, FINAL, OLD, OLD_ROLES, CVE, read, sha, dumps, require, fingerprint
from repair_provenance import atomic, writer_lock, verify_receipt

DATA = ROOT/'04_data/03_reddit_collection'
SOURCE = DATA/'01_source_70615ef'
DERIVED = DATA/'02_derived_v2'
RESULTS = ROOT/'05_results/03_dataset_analysis/02_reddit_dataset/04_premodeling_gates_v1'
AUDIT = RESULTS.parent/'01_readiness_v1'
FINAL_HASH = 'b74b501609d68046cb3941c63e511cf80702fdce50bfaa064d0bc343c9be7309'
CATALOGUE = ('vendor_project','vckev_cwes','vckev_date_added','vckev_updated_at',
    'cisa_date_added','due_date','vckev_known_ransomware_campaign_use',
    'vckev_reported_exploited_by_canaries','vckev_reported_exploitation','vckev_xdb')
URLS = ('all_sources_urls','github_only_urls','non_github_urls')
ALLOWLIST = ('cve','description','published_date','published_timestamp','last_modified_timestamp',
    'vuln_status','cvss_vector','cvss_vector_version','cvss_vector_score',
    'cvss_vector_severity','cvss_vector_source','n_references_nvd',*URLS,'social_media_posts')


def lock_output(folder, inputs):
    files={str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*'))
           if p.is_file() and p.name!='artifact_lock.json'}
    atomic(folder/'artifact_lock.json',{'files':files,'inputs':inputs})


def verify_output(folder):
    lock=read(folder/'artifact_lock.json')
    for name,h in tqdm(lock['files'].items(),desc='Verify locked outputs',unit='file'):
        require(sha(folder/name)==h,'Changed output: '+str(folder/name))
    for name,h in lock.get('inputs',{}).items():
        require(sha(Path(name))==h,'Changed input: '+name)
    return lock


def preserve(verify=False):
    manifest_path=DEFAULT_DATA/'download_manifest.json'
    expected=read(AUDIT/'findings.json')['pins'][str(manifest_path)]
    # Once preserved, verification needs no /tmp source files.
    manifest=read((SOURCE/'download_manifest.json') if verify else manifest_path)
    if verify:
        require(sha(SOURCE/'download_manifest.json')==expected,'Changed source manifest')
        verify_output(RESULTS/'stage0')
        for name,h in tqdm(manifest['files'].items(),desc='Verify durable raw bundle',unit='file'):
            require(sha(SOURCE/name)==h,'Durable source mismatch: '+name)
        print('Stage 0 durable source hashes verified; /tmp is not needed for this check.'); return
    require(sha(manifest_path)==expected,'Unexpected download manifest')
    SOURCE.mkdir(parents=True,exist_ok=True)
    missing_bytes=sum((DEFAULT_DATA/name).stat().st_size for name in manifest['files'] if not (SOURCE/name).exists())
    require(shutil.disk_usage(SOURCE).free>missing_bytes+300*1024**2,'Insufficient disk space for durable bundle and reserve')
    rows=[]
    for name,h in tqdm(manifest['files'].items(),desc='Copy and hash durable sources',unit='file'):
        src,dst=DEFAULT_DATA/name,SOURCE/name
        require(sha(src)==h,'Original source mismatch: '+name)
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():
            tmp=dst.with_name(dst.name+'.copying')
            shutil.copyfile(src,tmp)
            require(sha(tmp)==h,'Copy mismatch: '+name)
            with tmp.open('rb') as f: os.fsync(f.fileno())
            os.replace(tmp,dst)
        require(sha(dst)==h,'Existing destination mismatch: '+name)
        rows.append({'relative_path':name,'bytes':dst.stat().st_size,'sha256':h})
    if not (SOURCE/'download_manifest.json').exists(): shutil.copyfile(manifest_path,SOURCE/'download_manifest.json')
    require(sha(SOURCE/'download_manifest.json')==expected,'Manifest copy mismatch')
    report={'schema':'reddit_stage0_v1','commit':manifest['commit'],'files':rows,
            'total_bytes':sum(r['bytes'] for r in rows),'final_dataset_sha256':sha(SOURCE/'extracted'/FINAL),
            'copy_not_move':True,'original_tmp_preserved':True,'training':False}
    require(report['final_dataset_sha256']==FINAL_HASH,'Final dataset identity differs')
    out=RESULTS/'stage0'; out.mkdir(parents=True,exist_ok=True)
    if (out/'artifact_lock.json').exists():
        require(read(out/'report.json')==report,'Preservation report changed'); verify_output(out)
    else:
        atomic(out/'report.json',report)
        (out/'report.txt').write_text('Stage 0 durable preservation\n'+dumps(report))
        lock_output(out,{str(SOURCE/'download_manifest.json'):expected,str(Path(__file__)):sha(Path(__file__))})
    print(f'Stage 0 copied and verified {len(rows)} files, {report["total_bytes"]:,} bytes: {SOURCE}')


class UnionFind:
    def __init__(self,ids): self.parent={x:x for x in ids}
    def find(self,x):
        self.parent.setdefault(x,x)
        root=x
        while self.parent[root]!=root: root=self.parent[root]
        while self.parent[x]!=x:
            x,self.parent[x]=self.parent[x],root
        return root
    def join(self,ids):
        ids=sorted(set(ids))
        if not ids: return
        root=self.find(ids[0])
        for x in ids[1:]:
            other=self.find(x)
            if other!=root:
                low,high=sorted((root,other)); self.parent[high]=low; root=low
    def groups(self,ids):
        groups=defaultdict(list)
        for c in sorted(ids): groups[self.find(c)].append(c)
        return dict(groups)


def dedup_record(record):
    """Exact string dedup only; no URL rewriting or semantic post deletion."""
    out={k:record[k] for k in ALLOWLIST}
    counts={}
    for k in URLS:
        require(all(isinstance(x,str) for x in out[k]),'Non-string URL')
        out[k]=list(dict.fromkeys(out[k])); counts[k]=len(record[k])-len(out[k])
    out['distinct_reference_counts']={k:len(out[k]) for k in URLS}
    seen=set(); posts=[]; audit=[]
    for index,p in enumerate(record['social_media_posts']):
        key=fingerprint(p)
        if key in seen: continue
        seen.add(key); posts.append(p)
        mentioned=sorted(set(x.upper() for x in CVE.findall(p['post'])))
        audit.append({'fingerprint':key,'original_post_index':index,'bare_cve_id':bool(CVE.fullmatch(p['post'].strip())),
                      'mentioned_cves':mentioned,'multi_cve_mention':len(mentioned)>1})
    out['social_media_posts']=posts
    require(not set(CATALOGUE+('in_vckev',))&out.keys(),'Catalogue leakage in predictor candidates')
    require(not any(k.startswith('epss') for k in out),'Outcomes leaked into predictor candidates')
    return out,counts,len(record['social_media_posts'])-len(posts),audit


def write_jsonl(path,rows):
    with path.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as z:
        with io.TextIOWrapper(z,encoding='utf-8',newline='\n') as f:
            for row in rows: f.write(json.dumps(row,sort_keys=True,ensure_ascii=True,allow_nan=False)+'\n')


def receipt_audit(folder,source_records):
    legacy={r['cve']:r['in_vckev'] for r in source_records}
    if not folder or not folder.exists():
        return {'state':'BLOCKED_NO_SAVED_RECEIPTS','checked_cves':0,'verified_listed':0,
                'verified_not_listed':0,'unknown':len(legacy),'training_labels_ready':False},{}
    protocol=read(folder/'protocol.json'); require(protocol['source_sha256']==FINAL_HASH,'Receipts refer to another dataset')
    require(protocol['script_sha256']==sha(Path(__file__).with_name('repair_provenance.py')),'Receipt tool changed; review required')
    ids=protocol['ids']; require(len(ids)==len(set(ids)) and set(ids)<=set(legacy),'Receipt CVEs invalid')
    latest={}; states=Counter()
    for c in ids:
        directory=folder/'receipts'/c
        for path in sorted(directory.glob('*.receipt.json')):
            r=read(path); verify_receipt(r,directory)
            require(r['cve']==c and r['legacy_in_vckev']==legacy[c],'Receipt identity mismatch')
            latest[c]=r
        states[latest[c]['status'] if c in latest else 'NOT_CHECKED']+=1
    pos=sum(r['verified_in_vckev'] is True for r in latest.values())
    neg=sum(r['verified_in_vckev'] is False for r in latest.values())
    return {'state':'FRESH_RECEIPTS_AUDITED','checked_cves':len(latest),'verified_listed':pos,
            'verified_not_listed':neg,'unknown':len(legacy)-pos-neg,'statuses':dict(states),
            'training_labels_ready':False,'interpretation':'Fresh lookup times; never reconstructed old snapshot'},latest


def sanitize(output, receipts=None, verify=False):
    v2=DERIVED/'reddit_posts_vc_kev_epss_v2.json'
    require(v2.exists(),'Run targeted recovery first; it must preserve unresolved scores explicitly')
    verify_output(RESULTS/'stage1_epss')
    source_hash=sha(v2); rows=read(v2); ids={r['cve'] for r in rows}
    roles=read(OLD_ROLES); prior={c:role for role,cs in roles.items() for c in cs}
    old=read(OLD); old_texts=defaultdict(set)
    for r in old:
        for p in r.get('social_media_posts',[]):
            if p.get('platform')=='reddit' and isinstance(p.get('post'),str):
                old_texts[p['post']].add(r['cve'])
    exact=UnionFind(ids); conservative=UnionFind(ids|set(prior)); posts={}; counters=Counter(); prepared=[]; post_audits=[]
    byid={r['cve']:r for r in rows}
    for r in tqdm(rows,desc='Sanitize URLs and posts; isolate outcomes',unit='CVE'):
        clean,removed,duplicate,pa=dedup_record(r); prepared.append(clean)
        counters.update({'removed_'+k:v for k,v in removed.items()}); counters['removed_duplicate_post_entries']+=duplicate
        for p,meta in zip(clean['social_media_posts'],pa):
            key=meta['fingerprint']; item=posts.setdefault(key,{'cves':set(),'mentioned_cves':set(meta['mentioned_cves']),
                'bare_cve_id':meta['bare_cve_id'],'date':p['date'],'text':p['post'],'platform':p['platform']})
            item['cves'].add(r['cve'])
            counters['retained_post_entries']+=1
            counters['bare_id_entries']+=meta['bare_cve_id']; counters['multi_cve_entries_after_dedup']+=meta['multi_cve_mention']
            post_audits.append({'cve':r['cve'],**meta})
    for key,p in tqdm(sorted(posts.items()),desc='Connect shared posts and CVE mentions',unit='post'):
        exact.join(p['cves'])
        # Include external mentioned IDs and prior exact text reuse as conservative exposure links.
        conservative.join(p['cves']|p['mentioned_cves']|old_texts.get(p['text'],set()))
    groups=exact.groups(ids); all_groups=conservative.groups(set(conservative.parent))
    comp_rows=[]; cve_component={}
    for root,members in sorted(all_groups.items()):
        current=sorted(set(members)&ids)
        if not current: continue
        component='CC-'+hashlib.sha256(dumps(members).encode()).hexdigest()[:20]
        exposure=Counter(prior[c] for c in members if c in prior)
        pubs=sorted(byid[c]['published_date'] for c in current)
        result={'component_id':component,'current_cves':current,'external_or_old_only_cves':sorted(set(members)-ids),
                'current_count':len(current),'prior_roles':dict(exposure),'publication_min':pubs[0],'publication_max':pubs[-1],
                'ineligible_for_untouched_test':bool(exposure), 'connected_to_prior_test':bool(exposure.get('test',0)),
                'interpretation':'Conservative shared-text/advisory group, not identical vulnerabilities'}
        comp_rows.append(result)
        for c in current: cve_component[c]=component
    protected_test={c for g in comp_rows if g['connected_to_prior_test'] for c in g['current_cves']}
    protected_all={c for g in comp_rows if g['ineligible_for_untouched_test'] for c in g['current_cves']}
    status,verified=receipt_audit(receipts,rows)
    era=[r for r in rows if r['published_date']>='2021-04-14']
    counts={'cves':len(rows),'post_epss_launch_cves':len(era),
        'saved_listed_all_years':sum(r['in_vckev'] for r in rows),
        'saved_listed_post_launch':sum(r['in_vckev'] for r in era),
        'any_epss_all_years':sum(any(v is not None for v in r['epss_scores'].values()) for r in rows),
        'any_epss_post_launch':sum(any(v is not None for v in r['epss_scores'].values()) for r in era),
        **dict(counters),'unique_post_fingerprints':len(posts),
        'shared_post_fingerprints':sum(len(p['cves'])>1 for p in posts.values()),
        'multi_mention_unique_fingerprints':sum(len(p['mentioned_cves'])>1 for p in posts.values()),
        'exact_shared_post_components':len(groups),'largest_exact_component':max(map(len,groups.values())),
        'conservative_components':len(comp_rows),'largest_conservative_current_component':max(g['current_count'] for g in comp_rows),
        'prior_test_connected_current_cves':len(protected_test),'new_ids_connected_to_prior_test':len(protected_test-set(prior)),
        'any_prior_exposure_connected_current_cves':len(protected_all),
        'post_launch_not_connected_to_prior_exposure':sum(r['cve'] not in protected_all for r in era)}
    require(counts['removed_duplicate_post_entries']==154 and counts['shared_post_fingerprints']==3915,'Baseline dedup counts changed')
    yearly={}
    for r in era:
        y=r['published_date'][:4]; bucket=yearly.setdefault(y,Counter()); bucket['cves']+=1
        bucket['saved_listed_unverified']+=r['in_vckev']; bucket['protected_from_new_test']+=r['cve'] in protected_all
        dt=(date.fromisoformat(r['published_date'])+timedelta(days=60)).isoformat()
        bucket['day60_observed']+=r['epss_scores'].get(dt) is not None
        bucket['day60_mature_by_cutoff']+=dt<='2026-11-03'
    report={'schema':'reddit_stage1_sanitization_v1','source_sha256':source_hash,'counts':counts,
        'vulncheck_receipts':status,'yearly':{y:dict(v) for y,v in sorted(yearly.items())},
        'quarantined_catalogue_fields':list(CATALOGUE),'excluded_from_predictors':['in_vckev','all epss fields',*CATALOGUE],
        'predictor_allowlist':list(ALLOWLIST)+['distinct_reference_counts'],
        'split_frozen':False,'training':False,'target_values_examined_for_split_selection':False,
        'historical_text_verified':False,'rules':{'url_dedup':'exact string preserving first occurrence',
        'post_dedup':'exact platform/text/date fingerprint within CVE',
        'bulk_candidate':'shared by multiple CVEs OR multiple CVE mentions; tag only, not delete',
        'component_edges':'shared fingerprint + mentioned CVE IDs + exact old Reddit text reuse',
        'prior_protection':'all previous roles excluded from future untouched test through conservative closure',
        'component_members_are_not_duplicate_cves':True}}
    inputs={str(v2):source_hash,str(OLD):sha(OLD),str(OLD_ROLES):sha(OLD_ROLES),str(Path(__file__)):sha(Path(__file__))}
    if receipts and receipts.exists():
        inputs.update({str(p):sha(p) for p in receipts.rglob('*.json')})
    with tempfile.TemporaryDirectory(prefix='reddit-sanitization-',dir='/tmp') as td:
        stage=Path(td)
        write_jsonl(stage/'predictor_candidates.jsonl.gz',prepared)
        def outcome_rows():
            for r in rows:
                yield {'cve':r['cve'],'legacy_in_vckev':r['in_vckev'],
                    'verified_in_vckev':verified.get(r['cve'],{}).get('verified_in_vckev'),
                    'verified_at_utc':verified.get(r['cve'],{}).get('checked_at_utc'),
                    'catalogue_fields':{k:r[k] for k in CATALOGUE if k in r},
                    'epss_outcomes':{k:v for k,v in r.items() if k.startswith('epss')}}
        write_jsonl(stage/'outcomes_restricted.jsonl.gz',outcome_rows())
        write_jsonl(stage/'post_audit_not_model_input.jsonl.gz',post_audits)
        write_jsonl(stage/'shared_post_graph.jsonl.gz',({'fingerprint':k,'cves':sorted(p['cves']),
            'mentioned_cves':sorted(p['mentioned_cves']),'bare_cve_id':p['bare_cve_id'],
            'bulk_advisory_candidate':len(p['cves'])>1 or len(p['mentioned_cves'])>1,
            'date':p['date'],'text_excerpt':p['text'][:350]} for k,p in sorted(posts.items())))
        atomic(stage/'connected_components.json',comp_rows)
        atomic(stage/'exact_shared_post_components.json',list(groups.values()))
        atomic(stage/'cve_component_map.json',cve_component)
        atomic(stage/'report.json',report)
        (stage/'report.txt').write_text('Stage 1 sanitization and exposure gates\n'+dumps(report)+
            '\nCandidates are retrospective, not approved historical inputs. All unknown VulnCheck outcomes remain null.\n'
            'Bulk tags identify review candidates, not confirmed irrelevant posts. No CVEs were merged or dropped.\n')
        lock_output(stage,inputs)
        if verify:
            verify_output(output)
            require(read(stage/'artifact_lock.json')==read(output/'artifact_lock.json'),'Sanitization replay differs')
        else:
            require(not output.exists(),'Output exists; use --verify-only')
            shutil.copytree(stage,output)
    print(dumps(report))
    print('Stage 1 '+('reproduced' if verify else 'saved')+f': {output}. No split assignment or training.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['preserve','sanitize'],required=True)
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--receipt-dir',type=Path)
    p.add_argument('--output-dir',type=Path,default=RESULTS/'stage1_sanitization')
    args=p.parse_args()
    with writer_lock(RESULTS/'pipeline'):
        if args.stage=='preserve': preserve(args.verify_only)
        else: sanitize(args.output_dir,args.receipt_dir,args.verify_only)
