"""H01: independent Day-60 -> Day-90 preparation, bounded development export."""
import argparse
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from tqdm.auto import tqdm

from . import forecast_data as contract
from . import forecast_multisource as sources

read,write,require,sha256 = contract.read_json,contract.write_json,contract.require,contract.sha256
ROOT = sources.ROOT
SCHEMA = 'day60_to_day90_preview_v1'
CONFIG = ROOT/'02_prediction/04_configs/day60_to_day90_preview_v1.json'
OUTPUT = ROOT/'04_data/01_records_and_graphs/epss_day60_to_day90_v1/preview'
SCRIPTS = sources.SCRIPTS
FILES = {'protocol.json','run_identity.json','split_manifest.json','cohort_audit.json',
         'inputs.json','labels.json','source_audit.json','source_inputs.json','analysis.json','analysis.txt'}


def protocol():
    p = read(CONFIG)
    require(p['schema']==SCHEMA and p['anchor_day']==60 and p['target_day']==90 and
            p['histories']=={'recent_30_60':[30,60],'long_0_60':[0,60]} and
            not any(p[k] for k in ('training_enabled','test_score_export','test_text_export',
                                   'full_development_export_enabled')), 'Wrong preview contract')
    contract.validated_split_policy(p['split_policy'])
    return p


def build_assignment(publications, groups, old_manifest, policy):
    """Use only identities/dates/groups; never inspect test scores or text here."""
    require(set(publications)==set(groups)==set(old_manifest), 'Assignment IDs differ')
    inputs, labels = {}, {}
    for c, day in publications.items():
        publication = date.fromisoformat(day)
        inputs[c] = {'input_end_date':(publication+timedelta(days=60)).isoformat()}
        labels[c] = {'target_date':(publication+timedelta(days=90)).isoformat()}
    manifest, conflicts = contract.split_manifest(inputs,labels,groups,policy)
    for c,r in manifest.items():
        if r['split'] in ('train','validation'):
            require(old_manifest[c]['split']!='test', 'Old reserved test record entered development')
    role_groups = {s:{r['group_id'] for r in manifest.values() if r['split']==s}
                   for s in ('train','validation','test')}
    require(all(role_groups[a].isdisjoint(role_groups[b]) for a,b in
                (('train','validation'),('train','test'),('validation','test'))),'Related group crosses roles')
    return manifest, conflicts


def history(dates, values, start, end):
    selected = values[start:end+1]
    mask = [int(v is not None) for v in selected]
    age, known, last = [], [], None
    for i,v in enumerate(selected):
        if v is not None:
            last = i
        known.append(int(last is not None))
        age.append(i-last if last is not None else None)
    return {'dates':dates[start:end+1], 'day_offsets':list(range(start,end+1)),
            'values':selected,'observed_mask':mask,
            'age_since_observation_days':age,'age_known_mask':known}


def derive(record):
    dates, values = contract.score_days(record)
    windows = {name:history(dates,values,*bounds) for name,bounds in
               (('recent_30_60',(30,60)),('long_0_60',(0,60)))}
    observed = {name:sum(h['observed_mask']) for name,h in windows.items()}
    has_target = values[90] is not None
    eligible = {name:has_target and count>0 for name,count in observed.items()}
    row = {'cve':record['cve'], 'observed_input_days':observed,'target_observed':has_target,
           'eligible_by_history':eligible,'matched_eligible':all(eligible.values()),
           'exclusion_reasons':([ 'missing_day90_target' ] if not has_target else [])+
                               [f'no_observed_{name}' for name,count in observed.items() if count==0]}
    inputs = {'cve':record['cve'],'publication_date':record['published_date'],
              'input_end_date':dates[60],'anchor_day':60,
              'time_policy':'score_date_only_release_time_unverified','histories':windows}
    label = {'cve':record['cve'],'target_date':dates[90],'target_day':90,
             'horizon_days':30,'target_observed':has_target,'y':{'endpoint':values[90]}}
    return inputs,label,row


def predict_input(inputs, name):
    """Future models consume one history, not the audit, target, or full record."""
    require(name in ('recent_30_60','long_0_60'), 'Unknown history')
    item = inputs['histories'][name]
    require(max(item['day_offsets'])==60 and all(d<=inputs['input_end_date'] for d in item['dates']),
            'Future data in input history')
    return {k:item[k] for k in ('values','observed_mask','age_since_observation_days','age_known_mask','day_offsets')}


def development_rows(raw, manifest):
    inputs, labels, audit = {}, {}, {}
    eligible = {'train':[], 'validation':[]}
    ids = sorted(c for c,r in manifest.items() if r['split'] in eligible)
    for c in tqdm(ids,desc='Check new development eligibility',unit='CVE'):
        require(raw[c]['cve']==c, 'Source CVE/key mismatch')
        x,y,row = derive(raw[c])
        require(x['input_end_date']==manifest[c]['input_end_date'] and
                y['target_date']==manifest[c]['target_date'], 'Derived dates differ from assignment')
        split = manifest[c]['split']
        audit[c] = {'split':split, **row}
        if row['matched_eligible']:
            inputs[c],labels[c] = x,y
            eligible[split].append(c)
    return inputs,labels,audit,eligible


def context():
    p = protocol()
    with tqdm(total=3,desc='Verify frozen source and new task contract',unit='stage') as bar:
        data = contract.load_prepared(sources.PREPARED)
        require(data['split_policy.json']==p['split_policy'], 'Calendar boundaries changed')
        bar.update()
        lock = read(sources.R07/'evaluation_lock.json')
        require(sha256(sources.R07/'run_identity.json')==lock['files']['run_identity.json'], 'Changed R07 identity')
        old_identity = read(sources.R07/'run_identity.json')
        source = Path(data['preparation_summary.json']['source_path'])
        rows = read(source)
        raw = {r['cve']:r for r in rows}
        require(len(raw)==len(rows)==p['expected_source_cves'], 'Unexpected source count/duplicates')
        bar.update()
        paths = [sources.PREPARED/n for n in (*contract.PREPARED_FILES,'preparation_lock.json')]
        pins = {str(path):sha256(path) for path in paths}
        pins[str(source)] = data['preparation_summary.json']['source_sha256']
        require(all(old_identity['pinned_files'].get(path)==h for path,h in pins.items()),
                'R07 source/preparation changed')
        for path in (CONFIG,Path(__file__).resolve(),Path(contract.__file__).resolve(),
                     Path(sources.__file__).resolve(),sources.CONFIG,
                     SCRIPTS/'prepare_forecast_later_anchor.py',SCRIPTS/'test_forecast_later_anchor.py',
                     sources.R07/'evaluation_lock.json',sources.R07/'run_identity.json'):
            pins[str(path)] = sha256(path)
        sources.check_pins(pins)
        bar.update()
    # Frozen publication dates and groups, not test contents, determine assignment.
    old_manifest = data['split_manifest.json']
    publications = {c:r['publication_date'] for c,r in data['inputs.json'].items()}
    groups = {c:r['group_id'] for c,r in old_manifest.items()}
    members = {c:g for g,ids in data['duplicate_groups.json'].items() for c in ids}
    require(groups==members, 'Frozen group membership differs')
    manifest, conflicts = build_assignment(publications,groups,old_manifest,p['split_policy'])
    inputs,labels,audit,full_ids = development_rows(raw,manifest)
    ids = sources.select_preview(full_ids,p['preview_counts'],p['selection_salt'])
    records = []
    fields = read(sources.CONFIG)['source_fields']
    for split, chosen in ids.items():
        for c in tqdm(chosen,desc=f'Apply Day-60 {split} source masks',unit='CVE'):
            records.append({'split':split,**sources.adapt_record(raw[c],inputs[c]['input_end_date'],fields)})
    identity = {'schema':SCHEMA,'protocol':p,'source_pins':pins,'ids':ids,
                'full_development_ids':full_ids, 'source_order':list(sources.SOURCES),
                'training':False,'test_score_export':False,'test_text_export':False}
    transition = Counter(old_manifest[c]['split']+' -> '+r['split'] for c,r in manifest.items())
    counts = {}
    for split in ('train','validation'):
        rows = [r for r in audit.values() if r['split']==split]
        counts[split] = {
            'assigned':len(rows), 'matched_eligible':len(full_ids[split]),
            'recent_history_eligible':sum(r['eligible_by_history']['recent_30_60'] for r in rows),
            'long_history_eligible':sum(r['eligible_by_history']['long_0_60'] for r in rows),
            'long_history_eligible_but_no_recent_input':sum(r['eligible_by_history']['long_0_60'] and
                                                         not r['eligible_by_history']['recent_30_60'] for r in rows),
            'missing_day90_target':sum(not r['target_observed'] for r in rows),
            'missing_recent_history':sum(r['observed_input_days']['recent_30_60']==0 for r in rows),
            'overlap_with_r07_eligible_ids':len(set(full_ids[split]).intersection(old_identity['ids'][split]))}
    overview = {
        'source_cves':len(manifest),'assignment_counts':dict(Counter(r['split'] for r in manifest.values())),
        'split_exclusion_reasons':dict(Counter(r['exclusion_reason'] for r in manifest.values() if r['split']=='excluded')),
        'conflicting_group_count':len(conflicts),'old_to_new_assignment_counts':dict(transition),
        'development':counts, 'test_eligibility':'not_evaluated_or_exported',
        'preview_counts':{s:len(v) for s,v in ids.items()},
        'source_coverage':sources.summarize(records,{'ids':ids,'protocol':read(sources.CONFIG)})['coverage']}
    selected = ids['train']+ids['validation']
    preview_x = {c:inputs[c] for c in selected}
    preview_y = {c:labels[c] for c in selected}
    validate_preview(preview_x,preview_y,manifest,ids)
    artifacts = {
        'protocol.json':p,'run_identity.json':identity,'split_manifest.json':manifest,
        'cohort_audit.json':audit,'inputs.json':preview_x,'labels.json':preview_y,
        'source_audit.json':records,
        'source_inputs.json':{'view':'retrospective_date_screened','anchor_day':60,
                              'source_order':list(sources.SOURCES),
                              'rows':[{'cve':r['cve'],'split':r['split'],
                                       **sources.model_input(r,list(sources.SOURCES))} for r in records]},
        'analysis.json':overview}
    return artifacts


def validate_preview(inputs, labels, manifest, ids):
    selected = set(ids['train']+ids['validation'])
    require(set(inputs)==set(labels)==selected, 'Unmatched input/target IDs')
    require(not set(ids['train']).intersection(ids['validation']), 'CVE crosses roles')
    for split, chosen in ids.items():
        for c in chosen:
            x,y = inputs[c],labels[c]
            require(manifest[c]['split']==split and split in ('train','validation'), 'Test or incorrect role in preview')
            require(date.fromisoformat(y['target_date'])-date.fromisoformat(x['input_end_date'])==timedelta(days=30),
                    'Wrong forecast horizon')
            require(y['target_observed'] and y['y']['endpoint'] is not None, 'Missing target')
            bases = []
            for name,length in (('recent_30_60',31),('long_0_60',61)):
                h = x['histories'][name]
                require(len(h['values'])==length and sum(h['observed_mask'])>0, 'Wrong input length or empty history')
                predict_input(x,name)
                bases.append(next(v for v in reversed(h['values']) if v is not None))
            require(bases[0]==bases[1], 'Matched histories have different persistence baseline')
            require(x['histories']['recent_30_60']['values']==x['histories']['long_0_60']['values'][30:],
                    'Overlapping histories differ')


def report(a):
    p = a['protocol.json']
    s = a['analysis.json']
    lines = ['H01: Day-60 to Day-90 matched preparation preview', '',
             'This is a new task. R07 Day-30 to Day-60 results and MS01 are unchanged.',
             'No training, graph generation, embeddings or test scoring.',
             '', '1. Inputs and target',
             'Recent history: Days 30-60 inclusive (31 daily values).',
             'Longer history: Days 0-60 inclusive (61 daily values).',
             'Both predict the observed Day-90 EPSS score, 30 days after the Day-60 cutoff.',
             'Both use identical CVEs. A later target is never filled or inferred.',
             'Missing past scores have masks and age indicators; zeros remain observed scores.',
             'The recent history cannot use observations before Day 30, including for age indicators.',
             'Future persistence references are equal on the matched cohort; no reference errors are scored here.',
             '', '2. Recomputed chronological assignment',
             'Train: Day-60 anchor and Day-90 target before 2024-01-01.',
             'Validation: Day-60 anchor in 2024, Day-90 target before 2025-01-01.',
             'Reserved test: Day-60 anchor in 2025, target within the frozen observation cutoff.',
             'Boundary-crossing labels and groups connecting active periods are excluded.',
             'Group IDs and calendar boundaries are unchanged; assignment is recomputed, not copied.',
             f"All source CVEs: {s['source_cves']}. New roles: {s['assignment_counts']}.",
             f"Exclusion reasons: {s['split_exclusion_reasons']}.",
             f"Conflicting related groups: {s['conflicting_group_count']}.",
             'Test rows in split_manifest.json contain IDs, group IDs and dates only. No test targets/text/histories exported.',
             '', '3. Development eligibility',
             'Split | Assigned | Recent-history eligible | Longer-history eligible | Matched eligible | Retained R07 eligible CVEs']
    for split,r in s['development'].items():
        lines.append(f"{split} | {r['assigned']} | {r['recent_history_eligible']} | {r['long_history_eligible']} | "
                     f"{r['matched_eligible']} | {r['overlap_with_r07_eligible_ids']}")
    lines += ['Recent/longer eligibility each requires the observed target; missingness reasons may overlap.',
              'cohort_audit.json contains only development-role availability, not numerical targets.',
              'The comparison uses the intersection, not different CVEs for each history.',
              f"Bounded preview: {s['preview_counts']}. Selected by fixed ID hash, not target values.",
              '', '4. Text at the new anchor',
              'The same source adapter now screens stored post dates against Day 60.',
              'Descriptions/summaries remain retrospective; passing the post-date screen does not authenticate publication.',
              'These are newly selected preview CVEs; coverage differences from MS01 are not a paired anchor comparison.',
              'Split | Source | CVEs with text | CVEs passing candidate-view rules']
    for r in s['source_coverage']:
        lines.append(f"{r['split']} | {r['source']} | {r['cves_with_text']} | {r['cves_in_date_screened_view']}")
    lines += ['', '5. Files and safety',
              'inputs.json contains only past EPSS histories for preview train/validation CVEs.',
              'labels.json separately contains their Day-90 targets. Never pass it into the encoder.',
              'source_inputs.json contains allowlisted text and source masks; source_audit.json is not model input.',
              'split_manifest.json accounts for all CVEs without exporting test scores or test text.',
              'cohort_audit.json and run_identity.json retain eligibility and matched development IDs.',
              'protocol.json defines this stage; preparation_lock.json protects all artifacts.',
              'Full development export and training are deliberately not CLI modes yet.',
              '', '6. Next step',
              'Review this preview and new cohort counts first. Then prepare the matched development histories',
              'and freeze an EPSS-only comparison of recent versus longer history before adding TPG fusion.',
              'Do not reuse the old fixed-31-day loader for the 61-day history without a tested new contract.',
              'A subsequent graph stage must use Day-60 source masks and the same matched CVEs.',
              'Final test remains closed. A later prediction anchor does not establish an earlier warning.',
              '', '7. Limitations']+['- '+x for x in p['limitations']]
    return '\n'.join(lines)+'\n'


def prepare(check_only=False,verify_only=False):
    require(not (check_only and verify_only),'Choose one mode')
    if not check_only and not verify_only:
        require(not OUTPUT.exists(),'Preview exists; use --verify-only')
    artifacts = context()
    text = report(artifacts)
    sources.check_pins(artifacts['run_identity.json']['source_pins'])
    if check_only:
        print(text)
        print('H01 Day-60 to Day-90 preflight passed. No files written or training.')
        return
    if verify_only:
        lock = read(OUTPUT/'preparation_lock.json')
        require(set(lock['files'])==FILES and lock['schema']==SCHEMA,'Wrong lock')
        for n,h in lock['files'].items():
            require(sha256(OUTPUT/n)==h,'Changed preview artifact: '+n)
        for n,value in artifacts.items():
            require(read(OUTPUT/n)==value,'Reproduction differs: '+n)
        require((OUTPUT/'analysis.txt').read_text()==text,'Report differs')
        print('H01 Day-60 to Day-90 preview reproduced. No training, graphs or test scoring.')
        return
    OUTPUT.mkdir(parents=True,exist_ok=False)
    for n,value in tqdm(artifacts.items(),desc='Save separate later-anchor preview',unit='file'):
        write(OUTPUT/n,value)
    (OUTPUT/'analysis.txt').write_text(text,encoding='utf-8')
    write(OUTPUT/'preparation_lock.json',{'schema':SCHEMA,'files':{n:sha256(OUTPUT/n) for n in sorted(FILES)}})
    print(f'H01 Day-60 to Day-90 preview saved to {OUTPUT}. No training, graphs or test scoring.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-only',action='store_true')
    modes.add_argument('--verify-only',action='store_true')
    args = parser.parse_args()
    prepare(args.check_only,args.verify_only)


if __name__ == '__main__':
    main()
