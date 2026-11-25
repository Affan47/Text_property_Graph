"""VK01: audit labels, source disclosures and evaluation provenance, without fitting."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT/'02_prediction/04_configs/02_vulncheck/01_readiness.json'
SCRIPT = ROOT/'02_prediction/03_analysis/03_vulncheck_study/01_audit_readiness.py'
TEST = SCRIPT.parent/'test_vulncheck_readiness.py'
ROLES = ('train', 'validation', 'test', 'excluded')
CATALOGUE_PATTERNS = {
    'vulncheck_mention': r'\bvuln\s*check\b',
    'kev_catalogue_mention': r'\b(?:vc[\s_-]*)?kev\b',
    'known_exploited_catalogue_phrase': r'\bknown[\s-]+exploited[\s-]+vulnerabilit(?:y|ies)\b',
    'cisa_catalogue_url': r'cisa\.gov/[^\s]*known-exploited',
}
COMPILED = {k: re.compile(v, re.I) for k, v in CATALOGUE_PATTERNS.items()}
EPSS = re.compile(r'\bEPSS\b.{0,80}\d|\d.{0,40}\bEPSS\b', re.I | re.S)
CVE = re.compile(r'\bCVE-\d{4}-\d{4,}\b', re.I)
EXPLOIT = re.compile(r'\b(?:actively exploited|exploited in the wild|exploitation observed)\b', re.I)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def dump(value):
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)+'\n'


def label(value):
    return int(value) if type(value) is bool else None


def timestamp(value):
    if not isinstance(value, str) or not value.strip():
        return None, 'missing_or_nontext'
    try:
        stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None, 'unparseable'
    assumed = stamp.tzinfo is None
    stamp = stamp.replace(tzinfo=timezone.utc) if assumed else stamp.astimezone(timezone.utc)
    return stamp, 'assumed_utc' if assumed else 'explicit_timezone'


def temporal(record):
    pub, pub_flag = timestamp(record.get('published_timestamp'))
    added, add_flag = timestamp(record.get('vckev_date_added'))
    listed = label(record.get('in_vckev'))
    delta = (added-pub).total_seconds()/86400 if pub and added else None
    return {'publication_year': str(pub.year) if pub else 'unavailable',
            'publication_parse': pub_flag, 'addition_parse': add_flag,
            'publication_fields_agree': bool(pub and record.get('published_date') == pub.date().isoformat()),
            'delta_days_stored': delta, 'listed_without_usable_date': listed == 1 and added is None,
            'not_listed_with_addition_date': listed == 0 and added is not None,
            'negative_delta': bool(delta is not None and delta < 0),
            'same_day_negative': bool(pub and added and added < pub and pub.date() == added.date()),
            'interpretation': 'Stored timestamp comparison only; not a training eligibility rule for snapshot labels.'}


def check_predictor_keys(keys, policy):
    allowed = set(policy['structured_allowlist']) | set(policy['source_fields'].values())
    require(set(keys) <= allowed, 'Unapproved predictor fields, including possible label-derived data')


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def inspect_text(text, cve):
    hits = {k: m.start() for k, pattern in COMPILED.items() if (m := pattern.search(text))}
    epss = EPSS.search(text)
    if epss:
        hits['numeric_epss_disclosure_candidate'] = epss.start()
    mentioned = sorted(set(m.upper() for m in CVE.findall(text)))
    return {'disclosure_cues': sorted(hits), 'candidate_quarantine': bool(hits),
            'other_cve_ids': [c for c in mentioned if c != cve],
            'target_cve_mentioned': cve in mentioned, 'exploitation_statement': bool(EXPLOIT.search(text)),
            'first_cue_offset': min(hits.values()) if hits else None}


def documents(record, policy, issues):
    for source, field in policy['source_fields'].items():
        value = record.get(field)
        if source == 'smp':
            if value is None:
                continue
            if not isinstance(value, list):
                issues['smp_not_list'] += 1
                continue
            items = []
            for post in value:
                if not isinstance(post, dict):
                    issues['smp_item_not_object'] += 1
                    continue
                items.append((post.get('post'), post.get('platform')))
        else:
            items = [(value, None)]
        seen = set()
        for text, platform in items:
            if text is None:
                continue
            if not isinstance(text, str):
                issues[source+'_text_not_string'] += 1
                continue
            if not text.strip():
                continue
            if platform is not None and not isinstance(platform, str):
                issues['platform_not_string'] += 1
                platform = None
            key = (text, platform)
            if key in seen:
                issues['within_cve_source_duplicate'] += 1
                continue
            seen.add(key)
            yield source, text, platform


def overlap_groups(rows):
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row['normalized_text_sha256']].append(row)
    out = []
    for digest, values in sorted(by_hash.items()):
        cves = sorted({v['cve'] for v in values})
        if len(cves) < 2:
            continue
        out.append({'normalized_text_sha256': digest, 'cves': cves,
                    'previous_roles': sorted({v['previous_role'] for v in values}),
                    'sources': sorted({v['source'] for v in values}), 'document_occurrences': len(values)})
    return out


def matrix(policy):
    rows = [{'id': 'VK_PRIOR', 'kind': 'baseline', 'inputs': [], 'target': 'in_vckev'},
            {'id': 'VK_STRUCT', 'kind': 'baseline', 'inputs': ['cvss', 'reference_count'], 'target': 'in_vckev'}]
    for sources in policy['proposed_training']['source_sets']:
        for representation in policy['proposed_training']['representations']:
            for structured in policy['proposed_training']['structured']:
                rows.append({'id': f'VK_{sources}_{representation}_S{int(structured)}',
                             'kind': 'neural', 'sources': sources, 'representation': representation,
                             'structured': structured, 'seeds': policy['proposed_training']['seeds'],
                             'target': 'in_vckev'})
    for row in rows:
        row.update(status='proposed_not_trained', output='VulnCheck snapshot classification score; calibration not assumed',
                   horizon='None: saved-snapshot target, not future inclusion', epss_input=False)
    return rows


def summarize_labels(labels):
    counts = Counter(row['target'] for row in labels)
    return {'records': len(labels), 'listed': counts[1], 'not_listed': counts[0],
            'invalid_or_missing_label': counts[None],
            'listed_percent_of_valid': 100*counts[1]/(counts[0]+counts[1]) if counts[0]+counts[1] else None}


def select_text_ids(records, manifest, policy, scale):
    if scale == 'full':
        return {r['cve'] for r in records}
    masks = {}
    for record in records:
        present = set()
        for source, field in policy['source_fields'].items():
            value = record.get(field)
            if source == 'smp':
                available = isinstance(value,list) and any(isinstance(p,dict) and
                    isinstance(p.get('post'),str) and p['post'].strip() for p in value)
            else:
                available = isinstance(value,str) and bool(value.strip())
            if available:present.add(source)
        masks[record['cve']] = present
    selected = set()
    for role, count in policy['smoke_text_counts'].items():
        remaining = {c for c in masks if manifest[c]['split']==role}
        covered = set()
        for _ in range(min(count,len(remaining))):
            chosen = min(remaining,key=lambda c:(-len(masks[c]-covered),c))
            selected.add(chosen);covered.update(masks[chosen]);remaining.remove(chosen)
    return selected


def audit(records, manifest, policy, scale, previous_audit):
    require(scale in ('smoke', 'full'), 'Unknown scale')
    require(bool(records), 'Empty source dataset')
    ids = [r.get('cve') for r in records]
    require(all(isinstance(c, str) and CVE.fullmatch(c) for c in ids), 'Invalid CVE identifier')
    require(len(set(ids)) == len(ids), 'Duplicate CVE IDs; refuse implicit merging')
    require(set(manifest) == set(ids), 'Source/previous manifest CVE coverage differs')
    require(all(v.get('split') in ROLES for v in manifest.values()), 'Unexpected previous role')
    labels = []
    for record in tqdm(sorted(records, key=lambda r:r['cve']), desc='Audit every snapshot label and stored date', unit='CVE'):
        cve = record['cve']
        labels.append({'cve': cve, 'target': label(record.get('in_vckev')), 'previous_role': manifest[cve]['split'],
                       'label_status': 'valid_boolean' if label(record.get('in_vckev')) is not None else 'invalid_or_missing',
                       'date_audit': temporal(record)})
    selected = select_text_ids(records,manifest,policy,scale)
    doc_rows, examples, source_rows = [], [], []
    issues, example_counts = Counter(), Counter()
    field_coverage = {k: Counter() for k in policy['structured_allowlist']}
    for record in tqdm(sorted(records,key=lambda r:r['cve']), desc='Inspect allowlisted fields', unit='CVE'):
        for name in field_coverage:
            value = record.get(name)
            field_coverage[name]['nonempty' if value is not None and value != '' else 'missing'] += 1
            field_coverage[name]['type_'+type(value).__name__] += 1
    for record in tqdm(sorted((r for r in records if r['cve'] in selected), key=lambda r:r['cve']),
                       desc='Screen source documents and disclosure cues', unit='CVE'):
        cve = record['cve']; role = manifest[cve]['split']
        raw_mask = {s:0 for s in policy['source_fields']}; candidate_mask = dict(raw_mask)
        for source, text, platform in documents(record, policy, issues):
            raw_mask[source] = 1
            flags = inspect_text(text, cve)
            candidate_mask[source] |= int(not flags['candidate_quarantine'])
            doc = {'cve': cve, 'previous_role': role, 'source': source, 'platform': platform,
                   'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                   'normalized_text_sha256': hashlib.sha256(normalize(text).encode()).hexdigest(),
                   'characters':len(text), **flags}
            doc_rows.append(doc)
            review_cues = flags['disclosure_cues'] or (['other_cve_context'] if flags['other_cve_ids'] else [])
            for cue in review_cues:
                if role not in ('train', 'validation') or example_counts[(source,cue)] >= 2:
                    continue
                start = max(0, (flags['first_cue_offset'] or 0)-100)
                examples.append({**doc, 'review_cue':cue, 'excerpt':text[start:start+450], 'excerpt_start':start,
                                 'review_status':'pending', 'interpretation':'A cue is not proof of target leakage or incorrect text.'})
                example_counts[(source,cue)] += 1
        source_rows.append({'cve':cve, 'previous_role':role, 'raw_source_mask':raw_mask,
                            'candidate_source_mask':candidate_mask})
    with tqdm(total=3, desc='Summarize coverage, overlaps and protocol', unit='stage') as bar:
        groups = overlap_groups(doc_rows); bar.update()
        coverage = {role:{source:{'cves_examined':sum(r['previous_role']==role for r in source_rows),
                            'raw_text_cves':sum(r['raw_source_mask'][source] for r in source_rows if r['previous_role']==role),
                            'candidate_text_cves':sum(r['candidate_source_mask'][source] for r in source_rows if r['previous_role']==role)}
                          for source in policy['source_fields']} for role in ROLES}
        old_missing = {r['cve'] for r in previous_audit if r['split']=='train' and not r['primary_eligible']}
        cross = [g for g in groups if len(g['previous_roles'])>1]
        all_labels = summarize_labels(labels)
        invalid = all_labels['invalid_or_missing_label']
        summary = {'schema':policy['schema'], 'scale':scale, 'labels':all_labels,
                   'labels_by_previous_role':{s:summarize_labels([r for r in labels if r['previous_role']==s]) for s in ROLES},
                   'labels_by_publication_year':{y:summarize_labels([r for r in labels if r['date_audit']['publication_year']==y])
                                                for y in sorted({r['date_audit']['publication_year'] for r in labels})},
                   'date_audit':{key:sum(r['date_audit'][key] for r in labels) for key in
                                 ('negative_delta','same_day_negative','listed_without_usable_date','not_listed_with_addition_date')},
                   'previous_training_without_day60_now_label_eligible':sum(r['cve'] in old_missing and r['target'] is not None for r in labels),
                   'text_scope':{'cves_examined':len(selected), 'documents':len(doc_rows),
                                'selection':'Full dataset, or deterministic source-availability-covering train/validation smoke; no label-based selection.',
                                'interpretation':'Full labels/field coverage; smoke text counts are not representative population estimates.'},
                   'source_coverage':coverage,
                   'disclosure_cues':dict(Counter(c for row in doc_rows for c in row['disclosure_cues'])),
                   'candidate_quarantined_documents':sum(r['candidate_quarantine'] for r in doc_rows),
                   'document_context':{'other_cve_mentions':sum(bool(r['other_cve_ids']) for r in doc_rows),
                                       'exploitation_statements':sum(r['exploitation_statement'] for r in doc_rows)},
                   'shared_text':{'groups':len(groups), 'cross_previous_role_groups':len(cross),
                                  'cves_in_cross_role_groups':len({c for g in cross for c in g['cves']}),
                                  'interpretation':'Same normalized wording, not proof of duplicate vulnerabilities; no reassignment.'},
                   'field_coverage':{k:dict(v) for k,v in field_coverage.items()}, 'source_format_issues':dict(issues),
                   'pending_review_examples':len(examples), 'eligible_label_count':all_labels['records']-invalid,
                   'training_ready':False, 'independent_final_evaluation_ready':False,
                   'blocking_decisions':['Review candidate text disclosures and freeze the source policy.',
                       'Approve an evaluation plan acknowledging prior test exposure; fresh independent evaluation is not established.',
                       'Document snapshot acquisition/label mapping provenance; stored booleans alone do not independently validate the enrichment.'],
                   'future_inclusion_status':'Blocked pending event-date interpretation, complete snapshot observation and historical inputs.',
                   'training':False, 'predictions_created':False, 'test_scoring':False, 'split_changes':False}
        bar.update()
        allowed = policy['structured_allowlist']+list(policy['source_fields'].values())
        check_predictor_keys(allowed, policy)
        require(not set(allowed)&set(policy['excluded_predictor_fields']), 'Target fields overlap input allowlist')
        feature_policy = {'predictor_allowlist':allowed, 'excluded_fields':policy['excluded_predictor_fields'],
                          'derived_exclusions':'Also exclude counts, masks, nodes, edges and embeddings derived from excluded fields.',
                          'text_gate':'Candidate document quarantine, not a verified leakage-free input export.',
                          'historical_availability':'Not verified; current text and CVSS can be used only with a retrospective interpretation.',
                          'existing_cache_policy':'No cache admitted by this audit. Reuse later requires exact accepted text, source and processing identities.',
                          'gnn':'Existing EdgeTypeGNNLayer is a candidate; do not claim multiview or learned relation extraction.',
                          'encoding':'CVSS fixed one-hot v3 groups, normalized base score and observed flags; log1p reference count with training-only scaling. No catalogue multi-hot.'}
        bar.update()
    return {'labels_audit.json':labels, 'source_audit.json':doc_rows, 'source_masks_audit.json':source_rows,
            'shared_text_groups.json':groups, 'review_examples.json':examples, 'feature_policy.json':feature_policy,
            'experiment_matrix.json':matrix(policy), 'protocol.json':policy, 'summary.json':summary}


def report(a):
    s=a['summary.json']; lines=['VK01: VulnCheck snapshot-classification readiness',
        'Audit only. No training, graph building, predictions, split changes or test scoring.',
        'Target: listed (1) / not listed (0) in this saved VulnCheck snapshot.',
        'Not listed does not mean never exploited. This is not CISA KEV or future-inclusion prediction.',
        '', 'All-source label coverage: '+str(s['labels']),
        '', 'Previous role | CVEs | Listed | Not listed | Invalid labels | Listed percent']
    for role,r in s['labels_by_previous_role'].items():
        lines.append(f"{role} | {r['records']} | {r['listed']} | {r['not_listed']} | {r['invalid_or_missing_label']} | {r['listed_percent_of_valid']}")
    lines += ['', 'These are existing roles, NOT a new approved train/validation/test split.',
              'The previous test has already been inspected; it is not a fresh final test.',
              'Date checks: '+str(s['date_audit']),
              'Nonstandard JSON constants mapped to null for this audit only: '+str(s.get('raw_nonstandard_constants_mapped_to_null_in_audit_only',{})),
              'Previously excluded training Day-60 labels now eligible as binary labels: '+str(s['previous_training_without_day60_now_label_eligible']),
              'A missing EPSS score does not make a valid VulnCheck boolean label missing.',
              '', 'Text audit scope: '+str(s['text_scope']),
              'Previous role | Source | CVEs examined | Raw text | Candidate accepted text']
    for role, sources in s['source_coverage'].items():
        for source, row in sources.items():
            lines.append(f"{role} | {source} | {row['cves_examined']} | {row['raw_text_cves']} | {row['candidate_text_cves']}")
    lines += ['', 'Disclosure cues: '+str(s['disclosure_cues']),
              'Candidate quarantined documents: '+str(s['candidate_quarantined_documents']),
              'These are heuristic review candidates, not proven leaks. Full document quarantine is conservative.',
              'Other CVE or exploitation statements alone do not trigger quarantine.',
              'Shared text: '+str(s['shared_text']), 'Pending review examples: '+str(s['pending_review_examples']),
              '', 'Proposed comparison: D / DM / DGN / DGNM x pooled / TPG x structured off / on; 16 neural configurations.',
              'Three proposed seeds per neural configuration; prevalence and structured-logistic controls. No fits exist.',
              'D=description; G=GitHub summary; N=non-GitHub summary; M=social/related posts.',
              'Select on validation AP. Report fixed 0.5 and frozen validation-selected F1 threshold.',
              'Report PR/AP, ROC-AUC, precision, recall, F1, balanced accuracy, confusion matrices and calibration.',
              'First isolate text/structured effects without EPSS. A later EPSS addition needs a justified observation date.',
              '', 'Before implementation of training:', *['- '+x for x in s['blocking_decisions']],
              'Future-inclusion status: '+s['future_inclusion_status'],
              '', 'Legacy clarification:',
              'Old CSV adapter -> one scalar epss_score per record, not a 91-day input sequence.',
              'Soft mode retains the continuous scalar as supervision; weighted BCE is the loss.',
              'Legacy reporting binarizes actual EPSS at 0.1 and predictions at 0.5 by default.',
              'One scalar target is compatible with regression. It does not by itself make a study classification.',
              'The new completed primary study also has a scalar Day-60 target, MAE loss, and no EPSS history predictor.',
              'Legacy scalar-score, binary KEV and time-series forecasting tasks must not be conflated.']
    return '\n'.join(lines)+'\n'


def figures(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outputs={}
    roles=list(ROLES)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    not_listed=[summary['labels_by_previous_role'][s]['not_listed'] for s in roles]
    listed=[summary['labels_by_previous_role'][s]['listed'] for s in roles]
    ax.bar(roles,not_listed,label='Not listed in snapshot',color='#608f9a')
    ax.bar(roles,listed,bottom=not_listed,label='Listed in VulnCheck',color='#b7604c')
    ax.set(xlabel='Previous EPSS study role (not a newly approved split)',ylabel='Number of CVEs',
           title='VulnCheck labels across all existing CVE roles')
    ax.legend(); buffer=BytesIO();fig.savefig(buffer,format='png',dpi=160);plt.close(fig)
    outputs['figures/previous_role_labels.png']=buffer.getvalue()
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    sources=list(summary['source_coverage']['train'])
    data=[[summary['source_coverage'][role][source]['candidate_text_cves'] for source in sources] for role in roles]
    image=ax.imshow(data,aspect='auto',cmap='YlGnBu')
    ax.set(xticks=range(4),xticklabels=['Description','GitHub summary','Other summary','Posts'],yticks=range(4),yticklabels=roles,
           xlabel='Text source after candidate disclosure screen',ylabel='Previous CVE role',
           title=f"{summary['scale'].title()} text audit: CVEs retaining each source")
    for i,row in enumerate(data):
        for j,value in enumerate(row):ax.text(j,i,str(value),ha='center',va='center',color='white' if value>max(map(max,data))/2 else 'black')
    fig.colorbar(image,ax=ax,label='CVEs with candidate accepted text (not model accuracy)')
    buffer=BytesIO();fig.savefig(buffer,format='png',dpi=160);plt.close(fig)
    outputs['figures/candidate_source_coverage.png']=buffer.getvalue()
    return outputs


def execute(scale, check_only=False, verify_only=False):
    policy=read(CONFIG); source=ROOT.parent/policy['source']; manifest_path=ROOT/policy['previous_manifest']
    require(not any(policy[k] for k in ('training_enabled','test_scoring_enabled','graph_building_enabled')),
            'Readiness stage must not enable training, scoring or graph building')
    identity_path=ROOT/policy['previous_design_identity']; identity=read(identity_path)
    pins={str(p.relative_to(ROOT.parent)):sha(p) for p in (CONFIG,Path(__file__),SCRIPT,TEST,identity_path,
             manifest_path,ROOT/policy['previous_primary_audit'])}
    with tqdm(total=3,desc='Verify source identity and load audit inputs',unit='stage') as bar:
        require(sha(source)==policy['source_sha256'],'Dataset changed; review and version the audit specification');bar.update()
        require(identity['source_pins'].get(str(manifest_path))==sha(manifest_path),'Previous split manifest changed')
        require(identity['source_pins'].get(str(source))==policy['source_sha256'],'Design and dataset identity disagree');bar.update()
        constants=Counter()
        def constant(value):
            constants[value]+=1
            return None
        records=json.loads(source.read_text(),parse_constant=constant)
        manifest=read(manifest_path);old_audit=read(ROOT/policy['previous_primary_audit']);bar.update()
    a=audit(records,manifest,policy,scale,old_audit)
    a['summary.json']['raw_nonstandard_constants_mapped_to_null_in_audit_only']=dict(constants)
    a['run_identity.json']={'schema':policy['schema'],'scale':scale,'source_sha256':policy['source_sha256'],
                             'pins':pins,'new_training':False,'test_scoring':False,
                             'snapshot_capture_timestamp':'not independently authenticated by this audit'}
    a['summary.txt']=report(a)
    directory=ROOT/policy['output']/scale
    if check_only:
        print(a['summary.txt']);print('VK01 preflight passed. No artifacts written.');return
    def unchanged():
        require(sha(source)==policy['source_sha256'],'Source changed during audit')
        for name,digest in pins.items():require(sha(ROOT.parent/name)==digest,'Changed audit input: '+name)
    if verify_only:
        lock=read(directory/'artifact_lock.json')
        expected=set(a)|{'figures/previous_role_labels.png','figures/candidate_source_coverage.png'}
        require(set(lock['files'])==expected,'Unexpected saved artifact set')
        for name,digest in tqdm(lock['files'].items(),desc='Verify saved artifact hashes',unit='file'):
            require(sha(directory/name)==digest,'Saved artifact changed: '+name)
        for name,value in a.items():
            saved=(directory/name).read_text() if name.endswith('.txt') else read(directory/name)
            require(saved==value,'Audit did not reproduce: '+name)
        unchanged()
        print(f'VK01 {scale} readiness reproduced. No training, split changes or test scoring.');return
    require(not directory.exists(),'Audit exists; use --verify-only. Do not overwrite a reviewed protocol.')
    directory.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.vk01-',dir=directory.parent))
    try:
        images=figures(a['summary.json'])
        for name,value in tqdm(a.items(),desc='Save label/source audits and proposed protocol',unit='file'):
            (staging/name).write_text(value if isinstance(value,str) else dump(value))
        for name,value in images.items():
            (staging/name).parent.mkdir(exist_ok=True);(staging/name).write_bytes(value)
        files={str(p.relative_to(staging)):sha(p) for p in staging.rglob('*') if p.is_file()}
        (staging/'artifact_lock.json').write_text(dump({'schema':policy['schema'],'files':files}))
        unchanged();require(not directory.exists(),'Output appeared during audit; refusing overwrite')
        staging.rename(directory)
    finally:
        if staging.exists():shutil.rmtree(staging)
    print(a['summary.txt'])
    print(f'VK01 {scale} readiness saved to {directory}. No training, split changes or test scoring.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale',choices=('smoke','full'),default='smoke')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only',action='store_true')
    mode.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();execute(args.scale,args.check_only,args.verify_only)


if __name__=='__main__':main()
