"""Freeze accepted CVE roles and audit source policy. No fitting or score analysis."""
import argparse
from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata

from tqdm import tqdm

from audit_reddit import ROOT, read, sha, dumps, require, fingerprint
from reddit_gates import RESULTS, lock_output, verify_output
from audit_rd01_splits import (OUT as DIAG, CONFIG as RD01_CONFIG, SAN, ROLES, jsonl,
                              load, groups_for, qualify, csv_file, post_group)

CONFIG=ROOT/'02_prediction/04_configs/05_reddit_studies/RD01_source_lock_v1.json'
OUT=DIAG.parent/'08_rd01_source_lock_v1'
TOPIC=re.compile(r'\bepss(?:[_ -]?(?:score|scores|percentile))?\b|\bexploit\s+prediction\s+scoring\s+system\b',re.I)
NUMBER=re.compile(r'(?<!\w)(?:0?\.\d+|\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?)(?!\w)')
CATALOGUE=re.compile(r'\b(?:vuln\s*check|(?:vc[-_ ]?)?kev)\b|known[ -]+exploited[ -]+vulnerabilit',re.I)


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())


def disclosure(text):
    hits=list(TOPIC.finditer(text)); numeric=False
    # Remove identifiers/URLs only from numeric context; topic screening still sees the full text.
    for hit in hits:
        context=text[max(0,hit.start()-40):min(len(text),hit.end()+80)]
        context=re.sub(r'https?://\S+|\bCVE-\d{4}-\d+\b',' ',context,flags=re.I)
        numeric |= bool(NUMBER.search(context))
    return {'quarantine':bool(hits),'epss_numeric_context':numeric,
            'catalogue_cue':bool(CATALOGUE.search(text)),
            'spans':[list(m.span()) for m in hits]}


def near_pairs(texts,threshold=.9,min_tokens=20,shingle_words=5):
    """Complete prefix-join candidates for set Jaccard, followed by exact validation."""
    sets={}
    for k,text in texts.items():
        words=re.findall(r'\w+',text)
        if len(words)>=min_tokens:
            sets[k]={tuple(words[i:i+shingle_words]) for i in range(len(words)-shingle_words+1)}
    frequency=Counter(s for ss in sets.values() for s in ss)
    inverted=defaultdict(list)
    for key in tqdm(sorted(sets,key=lambda k:(len(sets[k]),k)),desc='Exact high-similarity prefix join',unit='text'):
        current=sets[key]; count=len(current)
        ordered=sorted(current,key=lambda s:(frequency[s],s))
        prefix=ordered[:count-math.ceil(threshold*count)+1]
        candidates=set()
        for s in prefix:
            candidates.update(k for k in inverted[s] if len(sets[k])>=threshold*count)
        for other in sorted(candidates):
            intersection=len(current&sets[other]); union=count+len(sets[other])-intersection
            if intersection>=threshold*union:
                yield min(key,other),max(key,other),intersection/union
        for s in prefix:
            inverted[s].append(key)


def source_audit(tasks,protocol,stage):
    selected={c for task in tasks.values() for ids in task.values() for c in ids}
    associations=[]; docs={}; normalized={}; normgroups=defaultdict(set)
    for row in tqdm(jsonl(SAN/'predictor_candidates.jsonl.gz'),total=44985,desc='Screen D/M text without target values',unit='CVE'):
        c=row['cve']
        if c not in selected: continue
        documents=[('D',row['description'],None)]+[('M',p['post'],fingerprint(p)) for p in row['social_media_posts']]
        for source,text,post in documents:
            if not isinstance(text,str) or not text.strip(): continue
            key=hashlib.sha256(text.encode()).hexdigest()
            if key not in docs:
                norm=normalize(text); nk=hashlib.sha256(norm.encode()).hexdigest()
                docs[key]={'key':key,'normalized_key':nk,'flags':disclosure(text),
                           'bare_id':source=='M' and post_group(text)=='bare_id_only'}
                normalized[nk]=norm; normgroups[nk].add(key)
            for task,roles in tasks.items():
                for role,ids in roles.items():
                    if c in ids:
                        associations.append({'task':task,'role':role,'cve':c,'source':source,
                            'text_sha256':key,'post_fingerprint':post or '',
                            'bare_id':source=='M' and post_group(text)=='bare_id_only',
                            **{k:docs[key]['flags'][k] for k in ('quarantine','epss_numeric_context','catalogue_cue')}})
    # Aggregate by normalized document; inspect all screened and rejected documents separately.
    norm_roles=defaultdict(lambda:defaultdict(set)); norm_counts=defaultdict(Counter)
    for a in associations:
        if not a['quarantine']:
            nk=docs[a['text_sha256']]['normalized_key']
            norm_roles[nk][a['task']].add(a['role']); norm_counts[nk][a['source']]+=1
    duplicate_rows=[]; flagged=set(); task_pairs=Counter()
    for nk in sorted(norm_roles):
        for task,roles in sorted(norm_roles[nk].items()):
            if len(roles)>1:
                flagged.add(nk); task_pairs[task,'exact']+=1
                duplicate_rows.append({'kind':'normalized_exact','task':task,'left_normalized_sha256':nk,
                    'right_normalized_sha256':nk,'jaccard':1.,'roles':','.join(sorted(roles))})
    near=protocol['near_duplicate']
    eligible={k:normalized[k] for k in norm_roles}
    for left,right,score in near_pairs(eligible,near['jaccard_threshold'],near['min_tokens'],near['shingle_words']):
        for task in tasks:
            l=norm_roles[left].get(task,set()); r=norm_roles[right].get(task,set())
            if l and r and any(a!=b for a in l for b in r):
                flagged.update((left,right)); task_pairs[task,'near']+=1
                duplicate_rows.append({'kind':'five_word_jaccard','task':task,'left_normalized_sha256':left,
                    'right_normalized_sha256':right,'jaccard':score,'roles':','.join(sorted(l|r))})
    duplicate_rows.sort(key=lambda r:(r['task'],r['kind'],r['left_normalized_sha256'],r['right_normalized_sha256']))
    csv_file(stage/'cross_role_duplicate_signatures.csv',duplicate_rows,
             ['kind','task','left_normalized_sha256','right_normalized_sha256','jaccard','roles'])
    policies=[]; coverage=[]
    for task,roles in tasks.items():
        for role,ids in roles.items():
            selected_rows=[a for a in associations if a['task']==task and a['role']==role]
            for source in ('D','M_all','M_prose'):
                rows=[a for a in selected_rows if a['source']==('D' if source=='D' else 'M')]
                accepted=[a for a in rows if not a['quarantine'] and not (source=='M_prose' and a['bare_id'])]
                coverage.append({'task':task,'role':role,'source_view':source,'cves':len(ids),
                    'attachments_before_screen':len(rows),'quarantined_disclosure_attachments':sum(a['quarantine'] for a in rows),
                    'bare_id_attachments_before_screen':sum(a['bare_id'] for a in rows),
                    'accepted_attachments':len(accepted),'cves_with_source':len({a['cve'] for a in accepted}),
                    'cves_without_source':len(ids)-len({a['cve'] for a in accepted})})
            byc=defaultdict(list)
            for a in selected_rows: byc[a['cve']].append(a)
            for c in sorted(ids):
                aa=byc[c]; d=[a for a in aa if a['source']=='D' and not a['quarantine']]
                m=[a for a in aa if a['source']=='M' and not a['quarantine']]
                prose=[a for a in m if not a['bare_id']]
                policies.append({'task':task,'role':role,'cve':c,'mask_D':bool(d),'mask_M_all':bool(m),
                    'mask_M_prose':bool(prose),'M_all_attachments':len(m),'M_prose_attachments':len(prose),
                    'cross_role_overlap_requires_review':any(docs[a['text_sha256']]['normalized_key'] in flagged for a in aa if not a['quarantine'])})
    csv_file(stage/'source_coverage.csv',coverage,list(coverage[0]))
    csv_file(stage/'source_masks_audit_only.csv.gz',policies,list(policies[0]))
    for a in associations:
        a['normalized_sha256']=docs[a['text_sha256']]['normalized_key']
        a['overlap_requires_review']=a['normalized_sha256'] in flagged and not a['quarantine']
    associations.sort(key=lambda a:(a['task'],a['role'],a['cve'],a['source'],a['text_sha256'],a['post_fingerprint']))
    csv_file(stage/'source_document_decisions.csv.gz',associations,list(associations[0]))
    decisions={task:{kind:task_pairs[task,kind] for kind in ('exact','near')} for task in tasks}
    return coverage,decisions,{'unique_raw_texts':len(docs),'unique_normalized_texts':len(normalized),
        'text_values_exported':False,'near_duplicate_definition':near,
        'limitation':'Finite lexical screen, not proof of semantic identity, historical availability or complete disclosure detection. Distinct CVEs are not merged.'}


def report(manifest,coverage,dup,scan):
    return '\n'.join(['RD01 SOURCE AUDIT AND ROLE LOCK','',
        'No model training, smoke training, numerical test-score export or test scoring.',
        'Accepted source design: retain_all plus compatible prior train/validation roles. All original prior-test-linked CVEs remain excluded.',
        'The manifest fixes CVE identities and date boundaries. RD02 source approval remains a separate gate; a role lock is not proof that texts are independent.',
        'M_all retains disclosure-screened attachments. M_prose also excludes identifier-only posts. A CVE with no remaining source has mask_M=0; it is not dropped.',
        'EPSS topic mentions are conservatively quarantined as whole documents. Numeric context is recorded separately. Not every generic EPSS mention is a confirmed score disclosure.',
        'Catalogue mentions are audit cues, not verified outcomes and not EPSS targets. Catalogue-derived structured fields remain prohibited.',
        'Cross-role comparisons retain identifiers and version numbers. Exact normalized and five-word Jaccard >=0.90 candidates are review findings, not assertions that CVEs are duplicates.',
        'No raw held-out text snippets are exported. No test EPSS values enter source decisions.',
        '', 'Role counts and readiness',dumps({k:{x:v for x,v in task.items() if x!='ids'} for k,task in manifest['tasks'].items()}),
        'Cross-role overlap signatures',dumps(dup),'Source coverage (attachment and CVE counts are different)',dumps(coverage),
        'Limits',dumps(scan),
        'RD04 needs the new development-only adapter and loader preflight before user-run smoke fits. The old CF01 CLI must not be presented as a runner for this new cohort.',
        'RD02 caches must use screened D/M text and distinct M_all/M_prose masks. No cache/encoder or GNN training was run.',
        'Manifest hashes are in manifest_hashes.json. --verify-only replays the source checks; it does not train.'])+'\n'


def run(output,verify=False):
    verify_output(DIAG); protocol=read(CONFIG); oldprotocol=read(RD01_CONFIG)
    records,posts,attachments,exposure,prior,baseline,releases,schedule=load(oldprotocol)
    task,flow=qualify(groups_for(records,posts),records,exposure,'retain_matching_prior_roles',oldprotocol)
    tasks={k:{r:task[k][r] for r in ROLES} for k in ('RD02','RD04')}
    require([[len(tasks[k][r]) for r in ROLES] for k in tasks]==[[24213,972,3977],[24548,1106,3978]],'Accepted cohort counts changed')
    require([flow['daily_cells'][r] for r in ROLES]==[96289074,3692438,16060201],'Accepted daily cells changed')
    inputs={str(p):sha(p) for p in [CONFIG,RD01_CONFIG,DIAG/'artifact_lock.json',SAN/'artifact_lock.json',
        Path(__file__),Path(__file__).with_name('audit_rd01_splits.py')]}
    with tempfile.TemporaryDirectory(prefix='rd01-source-lock-',dir='/tmp') as tmp:
        stage=Path(tmp); (stage/'source_policy.json').write_text(dumps(protocol))
        coverage,dup,scan=source_audit(tasks,protocol,stage)
        manifest={'schema':'rd01_role_manifest_v1','role_state':'FROZEN','source_policy_sha256':sha(CONFIG),
            'post_policy':'retain_all','exposure_policy':'retain_matching_prior_roles',
            'publication_intervals':oldprotocol['publication_intervals'],'cutoff':oldprotocol['cutoff'],
            'prior_test_connected_excluded':4987,'test_release_enabled':False,'training_executed':False,
            'tasks':{},'release_schedule':schedule}
        hashes={}
        for name,roles in tasks.items():
            ids={r:sorted(cs) for r,cs in roles.items()}
            counts={r:len(cs) for r,cs in roles.items()}
            ready=name=='RD04' or not any(dup[name].values())
            manifest['tasks'][name]={'ids':ids,'counts':counts,'role_ids_sha256':hashlib.sha256(dumps(ids).encode()).hexdigest(),
                'source_gate':'NOT_APPLICABLE_EPSS_ONLY' if name=='RD04' else ('PASSED_LEXICAL_SCREEN_ONLY' if ready else 'BLOCKED_CROSS_ROLE_TEXT_REVIEW'),
                'smoke_readiness':'ADAPTER_AND_PREFLIGHT_NOT_IMPLEMENTED' if name=='RD04' else ('CACHE_AND_ADAPTER_NOT_IMPLEMENTED' if ready else 'BLOCKED_SOURCE_REVIEW_AND_CACHE'),
                'training_enabled':False}
            if name=='RD04': manifest['tasks'][name]['eligible_cells']=dict(flow['daily_cells'])
            path=stage/(name.lower()+'_role_ids.json'); path.write_text(dumps(ids)); hashes[path.name]=sha(path)
        (stage/'rd01_frozen_manifest.json').write_text(dumps(manifest))
        hashes['rd01_frozen_manifest.json']=sha(stage/'rd01_frozen_manifest.json')
        (stage/'manifest_hashes.json').write_text(dumps(hashes))
        (stage/'findings.json').write_text(dumps({'schema':'rd01_source_audit_v1','coverage':coverage,'duplicates':dup,'scan':scan,
            'readiness':{k:{a:b for a,b in t.items() if a!='ids'} for k,t in manifest['tasks'].items()}}))
        (stage/'REPORT.txt').write_text(report(manifest,coverage,dup,scan))
        for p,h in inputs.items(): require(sha(Path(p))==h,'Input changed during source audit')
        lock_output(stage,inputs)
        if verify:
            verify_output(output); require(read(stage/'artifact_lock.json')==read(output/'artifact_lock.json'),'Source lock replay differs')
        else:
            require(not output.exists(),'Use --verify-only for existing output'); shutil.copytree(stage,output)
    print('RD01 roles and source audit '+('reproduced' if verify else 'saved')+' at '+str(output)+'. No training; check task-specific source gates.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--verify-only',action='store_true')
    p.add_argument('--output-dir',type=Path,default=OUT); a=p.parse_args(); run(a.output_dir,a.verify_only)
