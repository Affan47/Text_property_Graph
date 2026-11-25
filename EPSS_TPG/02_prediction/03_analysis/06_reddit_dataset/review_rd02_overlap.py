"""RD02 metadata-only overlap disposition; retain CVE identities, never read outcomes."""
import argparse
from collections import Counter, defaultdict
import csv
import difflib
import gzip
import hashlib
from pathlib import Path
import re
import sys
from tqdm import tqdm
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.reddit_forecasting import data
from lock_rd01_sources import normalize

OUTPUT=data.AUDIT/'10_rd02_overlap_review_v1'
LONG_SHARED='6051be23228ba509132d17b956c99a1ad412c0b2b9255b75ae3bb59708e428b2'


def table(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def structure(left,right,kind):
    words=[re.findall(r'\w+',t) for t in (left,right)]
    changes=[]
    for op,a,b,c,d in difflib.SequenceMatcher(None,words[0],words[1],autojunk=False).get_opcodes():
        if op!='equal':
            changed=words[0][a:b]+words[1][c:d]
            changes.append({'operation':op,'left_tokens':b-a,'right_tokens':d-c,
                            'all_changed_tokens_contain_digit':all(any(x.isdigit() for x in t) for t in changed)})
    return {'left_characters':len(left),'right_characters':len(right),
            'left_word_tokens':len(words[0]),'right_word_tokens':len(words[1]),
            'change_blocks':changes,'normalized_identical':left==right,
            'classification':('short_product_vulnerability_title' if kind=='normalized_exact' and len(words[0])<=10
                              else 'shared_mechanism_description' if kind=='normalized_exact'
                              else 'high_overlap_description_pair')}


def run(verify=False):
    data.checked(data.LOCK); data.checked(data.SAN)
    manifest=data.read(data.LOCK/'rd01_frozen_manifest.json'); task=manifest['tasks']['RD02']
    data.require(data.digest(task['ids'])==task['role_ids_sha256'],'RD02 IDs changed')
    with (data.LOCK/'cross_role_duplicate_signatures.csv').open() as f:
        signatures=[r for r in csv.DictReader(f) if r['task']=='RD02']
    data.require(Counter(r['kind'] for r in signatures)=={'normalized_exact':23,'five_word_jaccard':1},'Review scope changed')
    wanted={r[k] for r in signatures for k in ('left_normalized_sha256','right_normalized_sha256')}
    associations=defaultdict(lambda:defaultdict(set)); source_types=defaultdict(set)
    with gzip.open(data.LOCK/'source_document_decisions.csv.gz','rt') as stream:
        for r in csv.DictReader(stream):
            if r['task']!='RD02' or r['normalized_sha256'] not in wanted or r['quarantine'].lower()=='true': continue
            associations[r['normalized_sha256']][r['role']].add(r['cve'])
            source_types[r['normalized_sha256']].add(r['source'])
    data.require(all(v=={'D'} for v in source_types.values()),'Unexpected non-description overlap')
    texts={}
    # Only metadata derived from test-linked descriptions is published; no outcome file is opened.
    for row in tqdm(data.json_rows(data.SAN/'predictor_candidates.jsonl.gz'),total=44985,
                    desc='Review overlap structure; no outcome access',unit='CVE'):
        text=normalize(row['description'] or ''); key=hashlib.sha256(text.encode()).hexdigest()
        if key in wanted: texts[key]=text
    data.require(set(texts)==wanted,'Missing reviewed description signatures')
    reports=[]; structural=[]; flagged={role:{kind:set() for kind in ('exact','near')} for role in task['ids']}
    validation_train={'exact':set(),'near':set()}
    for s in signatures:
        left,right=s['left_normalized_sha256'],s['right_normalized_sha256']
        meta=structure(texts[left],texts[right],s['kind']); kind='exact' if s['kind']=='normalized_exact' else 'near'
        if kind=='exact' and meta['classification']=='shared_mechanism_description':
            data.require(left==LONG_SHARED,'Unreviewed long exact group')
        roles={r:associations[left].get(r,set())|associations[right].get(r,set()) for r in task['ids']}
        for role,ids in roles.items(): flagged[role][kind].update(ids)
        if kind=='exact' and roles['train']:
            validation_train[kind].update(roles['validation'])
        elif kind=='near':
            if associations[left].get('train'): validation_train[kind].update(associations[right].get('validation',set()))
            if associations[right].get('train'): validation_train[kind].update(associations[left].get('validation',set()))
        structural.append({'left_sha256':left,'right_sha256':right,**meta})
        reports.append({'kind':kind,'left_sha256':left,'right_sha256':right,'jaccard':float(s['jaccard']),
            'source':'D','roles':s['roles'],'train_cves':len(roles['train']),
            'validation_cves':len(roles['validation']),'test_cves':len(roles['test']),
            **{k:meta[k] for k in ('classification','left_characters','right_characters','left_word_tokens','right_word_tokens')},
            'test_linked_text_withheld':'test' in s['roles'].split(','),
            'disposition':'RETAIN_DISTINCT_CVES_WITH_OVERLAP_SENSITIVITY'})
    counts={role:{'exact':len(v['exact']),'near':len(v['near']),
                  'union':len(v['exact']|v['near'])} for role,v in flagged.items()}
    val_union=validation_train['exact']|validation_train['near']
    val_partition=[{'cve':c,'exact_description_overlap_with_train':c in validation_train['exact'],
        'near_description_overlap_with_train':c in validation_train['near'],
        'partition':'overlap_with_train' if c in val_union else 'no_detected_overlap_with_train'}
                   for c in sorted(task['ids']['validation'])]
    config=data.ROOT/'02_prediction/04_configs/05_reddit_studies/RD02_registered_v2.json'
    paths=[Path(__file__),Path(__file__).with_name('lock_rd01_sources.py'),
           data.LOCK/'artifact_lock.json',data.LOCK/'rd01_frozen_manifest.json',
           data.LOCK/'source_policy.json',data.SAN/'artifact_lock.json',config]
    pins={str(p):data.sha(p) for p in paths}
    approval={'schema':'rd02_source_overlap_approval_v1','source_gate':'ACCEPTED_WITH_DECLARED_DESCRIPTION_OVERLAP',
        'supersedes_gate_only':'BLOCKED_CROSS_ROLE_TEXT_REVIEW','original_manifest_sha256':pins[str(data.LOCK/'rd01_frozen_manifest.json')],
        'role_ids_sha256':task['role_ids_sha256'],'source_policy_sha256':pins[str(data.LOCK/'source_policy.json')],
        'registered_spec_sha256':pins[str(config)],'source_pins':pins,
        'role_ids_changed':False,'sources_removed_by_this_review':0,'training_enabled':False,'test_release_enabled':False,
        'cache_readiness':'SOURCE_REVIEW_ACCEPTED_ADAPTER_AND_CACHE_NOT_IMPLEMENTED',
        'decision_basis':'User-approved retention of distinct CVE IDs with shared descriptions; structural review is not independent vulnerability adjudication.',
        'conditions':['Primary analysis retains all assigned CVEs; do not select a winner on overlap subsets.',
                      'Report validation overlap-with-training and no-detected-overlap sensitivity metrics.',
                      'Retain existing disclosure screening and M_all/M_prose masks without changes.',
                      'Keep test text snippets and numerical test targets out of development outputs.',
                      'Do not claim text-independent evaluation, biological independence, or distinct root causes from unique IDs alone.']}
    approval['conditions'][-1]='Do not claim text-independent evaluation or distinct root causes from unique IDs alone.'
    findings={'signature_counts':{'exact':23,'near':1},'structural_classes':dict(Counter(r['classification'] for r in reports)),
              'affected_cves_by_any_cross_role_overlap':counts,
              'validation_train_overlap':{k:len(v) for k,v in validation_train.items()},
              'validation_train_overlap_union':len(val_union),'validation_without_detected_train_overlap':972-len(val_union),
              'raw_text_exported':False,'outcome_file_opened':False,'test_scoring':False,
              'interpretation':'Repeated descriptions are repeated inputs, not proof that the CVEs are the same vulnerability. Shared references are retained across CVEs.'}
    with data.core.transaction(OUTPUT,verify) as stage:
        table(stage/'signature_review.csv',reports); table(stage/'validation_overlap_partitions.csv',val_partition)
        for name,value in [('source_approval.json',approval),('findings.json',findings),('structural_differences.json',structural)]:
            (stage/name).write_text(data.dump(value))
        (stage/'REPORT.txt').write_text('RD02: review of cross-role description overlap\n\n'+data.dump(findings)+
            '\nWhat was reviewed\n22 short product/vulnerability titles and one longer shared mechanism description are repeated exactly after normalization. '
            'One additional pair has five-word-shingle Jaccard similarity above 0.90. The near pair links training and test, not validation. '
            'The signature table reports lengths, counts and roles, not held-out wording. Structural differences report token counts, not tokens.\n\n'
            'Decision\nRetain the distinct CVE IDs and descriptions. A common title may describe separate vulnerabilities; '
            'the longer repeated mechanism text also remains a real shared-input risk. This review does not verify distinct root causes. '
            'The primary cohort stays unchanged, with an additional validation-overlap sensitivity analysis defined before training.\n\n'
            'Limits\nNo outcome file was opened and no test snippet was published. This is a bounded structural review under an accepted policy, '
            'not an independent semantic annotation study or proof that all leakage is absent. The old blocked manifest remains unchanged. '
            'source_approval.json is its supplemental, hash-linked source disposition, not permission to score test data. '
            'RD02 graph/cache preparation and smoke training remain a later step.\n')
        data.core.check_pins(pins)
    print('RD02 overlap review '+('reproduced' if verify else 'saved')+'. No training, graphs, raw test text or outcome access.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--verify-only',action='store_true')
    run(parser.parse_args().verify_only)
