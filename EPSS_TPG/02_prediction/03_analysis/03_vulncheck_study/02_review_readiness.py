"""Interpret the frozen VK01 audit without editing its labels, policy or review judgments."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT/'05_results/05_vulncheck_study/01_readiness_v1/full'
OUTPUT = ROOT/'05_results/05_vulncheck_study/02_readiness_review_v1'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(ok, message):
    if not ok:raise ValueError(message)


def dump(value):
    return json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'


def overlap(rows):
    groups = defaultdict(list)
    for row in rows:groups[row['normalized_text_sha256']].append(row)
    cross = [g for g in groups.values() if len({r['cve'] for r in g})>1 and len({r['previous_role'] for r in g})>1]
    pairs = {}
    for left,right in (('train','validation'),('train','test'),('validation','test')):
        matching=[g for g in cross if {left,right}<={r['previous_role'] for r in g}]
        pairs[left+'__'+right]={'text_groups':len(matching),
            left+'_cves':len({r['cve'] for g in matching for r in g if r['previous_role']==left}),
            right+'_cves':len({r['cve'] for g in matching for r in g if r['previous_role']==right})}
    return {'cross_role_groups':len(cross),'cves':len({r['cve'] for g in cross for r in g}),
            'role_pairs':pairs,'interpretation':'Distinct CVEs sharing normalized text; counts across pairs overlap.'}


def build():
    with tqdm(total=5,desc='Review frozen label/source evidence',unit='stage') as bar:
        lock=read(AUDIT/'artifact_lock.json')
        for name,digest in lock['files'].items():require(sha(AUDIT/name)==digest,'Changed audit file: '+name)
        identity=read(AUDIT/'run_identity.json')
        for name,digest in identity['pins'].items():require(sha(ROOT.parent/name)==digest,'Changed audit dependency: '+name)
        policy=read(AUDIT/'protocol.json');source=ROOT.parent/policy['source']
        require(sha(source)==identity['source_sha256'],'Source dataset changed')
        summary=read(AUDIT/'summary.json');docs=read(AUDIT/'source_audit.json')
        labels={r['cve']:r for r in read(AUDIT/'labels_audit.json')}
        masks=read(AUDIT/'source_masks_audit.json');examples=read(AUDIT/'review_examples.json');bar.update()
        require(len(labels)==summary['labels']['records'] and len(docs)==summary['text_scope']['documents'],'Coverage mismatch')
        flagged=[r for r in docs if r['candidate_quarantine']]
        accepted=[r for r in docs if not r['candidate_quarantine']]
        affected={r['cve'] for r in flagged}
        require(len(flagged)==summary['candidate_quarantined_documents'],'Quarantine mismatch')
        per_source={s:{'documents':sum(r['source']==s for r in docs),
                       'flagged_documents':sum(r['source']==s for r in flagged),
                       'raw_text_cves':sum(r['raw_source_mask'][s] for r in masks),
                       'candidate_text_cves':sum(r['candidate_source_mask'][s] for r in masks)}
                    for s in policy['source_fields']}
        by_class=[]
        for role in ('train','validation','test','excluded'):
            for target in (0,1):
                ids={c for c,r in labels.items() if r['previous_role']==role and r['target']==target}
                matching=[r for r in masks if r['cve'] in ids]
                by_class.append({'previous_role':role,'target':target,'cves':len(ids),
                    'cves_with_flagged_document':len(ids&affected),
                    'flagged_percent':100*len(ids&affected)/len(ids) if ids else None,
                    'raw_smp_cves':sum(r['raw_source_mask']['smp'] for r in matching),
                    'candidate_smp_cves':sum(r['candidate_source_mask']['smp'] for r in matching),
                    'lost_all_smp':sum(r['raw_source_mask']['smp'] and not r['candidate_source_mask']['smp'] for r in matching)})
        bar.update()
        raw_overlap=overlap(docs);candidate_overlap=overlap(accepted)
        require(raw_overlap['cross_role_groups']==summary['shared_text']['cross_previous_role_groups'],'Overlap mismatch')
        bar.update()
        records=read(source)
        raw_examples={r['cve']:r for r in records if r['cve'] in ('CVE-2019-1367','CVE-2024-9644','CVE-2023-23333')}
        require(len(raw_examples)==3,'Missing reviewed examples')
        notes=[
            {'cve':'CVE-2024-9644','source':'non_github_summary',
             'observation':'Saved summary explicitly says the CVE is listed in the VulnCheck KEV database.',
             'interpretation':'Direct label-disclosure example; do not let the classifier read the answer from this document.'},
            {'cve':'CVE-2023-23333','source':'smp',
             'observation':'The flagged excerpt contains def vulncheck(ipaddress, port).',
             'interpretation':'The shown occurrence is a function identifier, not a catalogue assertion. This is a lexical false-positive mechanism; it is not blanket clearance of the complete document.'},
            {'cve':'CVE-2019-1367','source':'non_github_summary',
             'description':raw_examples['CVE-2019-1367']['description'],
             'summary':raw_examples['CVE-2019-1367'][policy['source_fields']['non_github_summary']],
             'interpretation':'Saved description concerns the Internet Explorer scripting engine; saved summary instead names an HTTP protocol stack and CVE-2023-12345. This is a concrete source-alignment concern that the catalogue screen does not solve.'},
            {'cve':'CVE-2006-2450','source':'description',
             'observation':'The description says it is a different issue than CVE-2006-2369.',
             'interpretation':'Another CVE mention can legitimately distinguish vulnerabilities. Do not automatically remove every multi-CVE document.'},
            {'cve':'CVE-2015-2291','source':'non_github_summary',
             'observation':'The saved text includes a CISA Known Exploited Vulnerabilities listing claim and an EPSS percentage.',
             'interpretation':'Catalogue/score disclosure candidate. CISA is not VulnCheck; the summary claim is not independently verified here.'}]
        for note in notes:note['review_provenance']='Automated qualitative review of saved local evidence; not independent annotation or online fact-checking.'
        bar.update()
        result={'schema':'vk01_full_review_v1','source_audit_lock_sha256':sha(AUDIT/'artifact_lock.json'),
                'review_script_sha256':sha(Path(__file__)),'source_sha256':sha(source),
                'labels':summary['labels'],'labels_by_previous_role':summary['labels_by_previous_role'],
                'documents':len(docs),'flagged_documents':len(flagged),'flagged_document_percent':100*len(flagged)/len(docs),
                'affected_cves':len(affected),'accepted_documents':len(accepted),'by_source':per_source,
                'filter_association_by_class':by_class,'raw_overlap':raw_overlap,'candidate_overlap':candidate_overlap,
                'no_remaining_text_cves':[r['cve'] for r in masks if not any(r['candidate_source_mask'].values())],
                'bounded_examples':len(examples),'date_audit':summary['date_audit'],
                'field_coverage':summary['field_coverage'],'qualitative_notes':notes,
                'training':False,'new_predictions':False,'split_changes':False,'original_review_statuses_changed':False,
                'recommendation':'Freeze a reviewed, versioned source policy and explicitly retrospective development evaluation before a classifier/cache adapter and small baseline smoke. Do not launch 48 neural fits yet.',
                'limitations':['Lexical flags are not measured annotation precision.',
                    'Quarantine is more common among listed CVEs; retained-source masks may carry label-associated missingness.',
                    'Current text and structured values are not verified historical inputs.',
                    'Existing test is already observed; a new independent final evaluation has not been established.',
                    'Structured field presence is not CVSS semantic-validation coverage.']}
        for name,digest in lock['files'].items():require(sha(AUDIT/name)==digest,'Original audit changed')
        bar.update()
    return result


def report(r):
    lines=['VK01 full readiness: verified interpretation','No training, new predictions, split changes or edits to the original audit.',
           '',f"Labels: {r['labels']}",f"Documents: {r['documents']}; flagged: {r['flagged_documents']} ({r['flagged_document_percent']:.2f}%); affected CVEs: {r['affected_cves']}.",
           f"Candidate accepted documents: {r['accepted_documents']}; CVEs losing every source: {len(r['no_remaining_text_cves'])}.",
           '', 'Source | Documents | Flagged | Raw text CVEs | Candidate text CVEs']
    for source,v in r['by_source'].items():lines.append(f"{source} | {v['documents']} | {v['flagged_documents']} | {v['raw_text_cves']} | {v['candidate_text_cves']}")
    lines+=['','Previous role | Listed label | CVEs | CVEs with flagged text | Percent | CVEs losing all posts']
    for v in r['filter_association_by_class']:
        lines.append(f"{v['previous_role']} | {v['target']} | {v['cves']} | {v['cves_with_flagged_document']} | {v['flagged_percent']:.2f}% | {v['lost_all_smp']}")
    lines+=['','This association is descriptive, not classifier performance. Missing-source masks could become a shortcut.',
            'Raw text overlap: '+str(r['raw_overlap']),'After candidate filtering: '+str(r['candidate_overlap']),
            'Shared text does not make distinct CVEs identical vulnerabilities.', '', 'Bounded qualitative observations:']
    for note in r['qualitative_notes']:
        lines.extend([note['cve']+' / '+note['source'],note.get('observation',note.get('summary','')),note['interpretation']])
    lines+=['','The original 25 review entries remain pending; these notes are not independent annotations.',
            '77 CVEs have no current CVSS vector/score/version. Nonempty CVSS still requires encoder validation.',
            'All 5968 have reference-count values; current values are not authenticated historical features.',
            '',r['recommendation'],*r['limitations']]
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();result=build();artifacts={'analysis.json':dump(result),'analysis.txt':report(result)}
    if args.verify_only:
        lock=read(OUTPUT/'artifact_lock.json')
        require(set(lock['files'])==set(artifacts),'Unexpected saved review artifacts')
        for name,contents in artifacts.items():
            require(sha(OUTPUT/name)==lock['files'][name] and (OUTPUT/name).read_text()==contents,'Review differs: '+name)
    else:
        require(not OUTPUT.exists(),'Review exists; use --verify-only')
        OUTPUT.mkdir(parents=True)
        for name,contents in artifacts.items():(OUTPUT/name).write_text(contents)
        (OUTPUT/'artifact_lock.json').write_text(dump({'files':{name:sha(OUTPUT/name) for name in artifacts}}))
    print(f'VK01 full interpretation {"reproduced" if args.verify_only else "saved to "+str(OUTPUT)}. No training or changes to the original audit.')


if __name__=='__main__':main()
