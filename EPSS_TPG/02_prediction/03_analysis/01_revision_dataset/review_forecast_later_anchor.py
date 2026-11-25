"""Review all H01 preview artifacts without fitting or scoring predictions."""
import argparse
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_later_anchor as prep

read,write,require,sha256 = prep.read,prep.write,prep.require,prep.sha256
OUTPUT = prep.ROOT/'05_results/02_evaluation/epss_day60_to_day90/preview_review_v1'


def eligibility_partition(rows):
    counts = Counter()
    for row in rows:
        missing_target = not row['target_observed']
        missing_recent = row['observed_input_days']['recent_30_60']==0
        expected = not missing_target and not missing_recent
        require(row['matched_eligible']==expected, 'Eligibility differs from observed availability')
        counts['eligible' if expected else 'both_missing' if missing_target and missing_recent
               else 'target_only_missing' if missing_target else 'recent_input_only_missing'] += 1
    return {k:counts[k] for k in ('eligible','both_missing','target_only_missing','recent_input_only_missing')}


def count_summary(values):
    return {'n':len(values),'minimum':min(values) if values else None,
            'median':statistics.median(values) if values else None,
            'maximum':max(values) if values else None,
            'histogram':dict(sorted(Counter(str(v) for v in values).items(),key=lambda p:int(p[0])))}


def check_history(x,name):
    h = x['histories'][name]
    start = 30 if name=='recent_30_60' else 0
    offsets = list(range(start,61))
    require(h['day_offsets']==offsets, 'Unexpected history offsets')
    dates = [(date.fromisoformat(x['publication_date'])+timedelta(days=i)).isoformat() for i in offsets]
    require(h['dates']==dates and dates[-1]==x['input_end_date'], 'History dates differ')
    require(len(h['values'])==len(dates), 'Sequence length differs')
    last = None
    masks,ages,known = [],[],[]
    for i,v in enumerate(h['values']):
        prep.contract.score(v)
        masks.append(int(v is not None))
        if v is not None:
            last = i
        ages.append(i-last if last is not None else None)
        known.append(int(last is not None))
    require(h['observed_mask']==masks and h['age_since_observation_days']==ages and
            h['age_known_mask']==known, 'History masks/ages differ')
    return {'observed_days':sum(masks), 'latest_score_age_days':ages[-1],
            'missing_at_anchor':masks[-1]==0}


def analyze():
    directory = prep.OUTPUT
    lock = read(directory/'preparation_lock.json')
    require(lock['schema']==prep.SCHEMA and set(lock['files'])==prep.FILES, 'Wrong H01 output lock')
    for name,h in prep.tqdm(lock['files'].items(),desc='Verify all H01 artifacts',unit='file'):
        require(sha256(directory/name)==h, 'Changed artifact: '+name)
    a = {n:read(directory/n) for n in prep.FILES if n.endswith('.json')}
    require(read(prep.CONFIG)==a['protocol.json'], 'Protocol changed')
    identity = a['run_identity.json']
    prep.sources.check_pins(identity['source_pins'])
    require((directory/'analysis.txt').read_text()==prep.report(a), 'Original report differs')
    manifest,audit = a['split_manifest.json'],a['cohort_audit.json']
    ids = identity['ids']
    inputs,labels = a['inputs.json'],a['labels.json']
    prep.validate_preview(inputs,labels,manifest,ids)
    full = identity['full_development_ids']
    require(set(audit)=={c for c,r in manifest.items() if r['split'] in ('train','validation')},
            'Test or missing role in development audit')
    groups = {s:{r['group_id'] for r in manifest.values() if r['split']==s}
              for s in ('train','validation','test')}
    require(all(groups[s].isdisjoint(groups[t]) for s,t in
                (('train','validation'),('train','test'),('validation','test'))), 'Groups cross roles')
    old = read(prep.sources.R07/'run_identity.json')['ids']
    old_manifest = read(prep.sources.PREPARED/'split_manifest.json')
    development = {}
    for split in ('train','validation'):
        rows = [r for r in audit.values() if r['split']==split]
        eligible = [r for r in rows if r['matched_eligible']]
        require(sorted(r['cve'] for r in eligible)==full[split], 'Full eligibility IDs differ')
        partition = eligibility_partition(rows)
        declared = a['analysis.json']['development'][split]
        require(sum(partition.values())==declared['assigned'] and
                partition['eligible']==declared['matched_eligible'], 'Eligibility counts differ')
        histories = {}
        for name,length in (('recent_30_60',31),('long_0_60',61)):
            observed = [r['observed_input_days'][name] for r in eligible]
            require(all(0<n<=length for n in observed), 'Invalid observation count')
            histories[name] = {**count_summary(observed),
                               'complete':sum(n==length for n in observed),
                               'partial':sum(n<length for n in observed)}
        previous,current = set(old[split]),set(full[split])
        lost = sorted(previous-current)
        gained = sorted(current-previous)
        lost_reasons = Counter(
            manifest[c]['exclusion_reason'] if manifest[c]['split']=='excluded'
            else '+'.join(audit[c]['exclusion_reasons']) if c in audit
            else 'new_role_'+manifest[c]['split'] for c in lost)
        extra = [r['observed_input_days']['long_0_60']-r['observed_input_days']['recent_30_60'] for r in eligible]
        development[split] = {
            'eligibility_partition':partition,'histories':histories,
            'extra_observed_days_0_29':{**count_summary(extra),'none':sum(n==0 for n in extra)},
            'old_eligible':len(previous),'new_eligible':len(current),'retained':len(previous&current),
            'gained_ids':gained,'lost_ids':lost,'lost_reasons':dict(lost_reasons),
            'gained_old_roles':dict(Counter(old_manifest[c]['split'] for c in gained))}
    cases = []
    for split,chosen in ids.items():
        for c in chosen:
            require(date.fromisoformat(labels[c]['target_date'])==
                    date.fromisoformat(inputs[c]['publication_date'])+timedelta(days=90), 'Wrong target date')
            prep.contract.score(labels[c]['y']['endpoint'])
            cases.append({'cve':c,'split':split,
                          'anchor_date':inputs[c]['input_end_date'],'target_date':labels[c]['target_date'],
                          'histories':{name:check_history(inputs[c],name) for name in inputs[c]['histories']}})
    source_rows = a['source_audit.json']
    expected_payload = [{'cve':r['cve'],'split':r['split'],
                         **prep.sources.model_input(r,list(prep.sources.SOURCES))} for r in source_rows]
    require(a['source_inputs.json']['rows']==expected_payload and
            sorted(r['cve'] for r in source_rows)==sorted(inputs), 'Source routing or masks differ')
    posts = [d for r in source_rows for d in r['documents'] if d['source_text_type']=='smp']
    cues = ('loading message','css error','no technical details')
    flagged = [{'cve':r['cve'],'source':d['source_text_type'],'text':d['text']}
               for r in source_rows for d in r['documents'] if 'summary' in d['source_text_type']
               and any(cue in d['text'].lower() for cue in cues)]
    conflicts = [{'cve':r['cve'],'platform':d['platform'],
                  'dates':[e['value'] for e in d['stored_date_evidence']],
                  'candidate_eligible':d['view_eligible']['retrospective_date_screened']}
                 for r in source_rows for d in r['documents']
                 if 'same_platform_text_has_conflicting_dates' in d['quality_flags']]
    result = {
        'schema':'h01_preview_review_v1','source_lock_sha256':sha256(directory/'preparation_lock.json'),
        'script_sha256':sha256(Path(__file__)), 'verified_artifacts':len(lock['files']),
        'training':False,'model_predictions':False,'test_scoring':False,'source_modified':False,
        'assignment':a['analysis.json'],'development':development,'preview_cases':cases,
        'preview_source_review':{
            'post_date_counts':dict(Counter(e['relation'] for d in posts for e in d['stored_date_evidence'])),
            'post_timezone_counts':dict(Counter(e['timezone'] for d in posts for e in d['stored_date_evidence'])),
            'post_conflicts':conflicts,'summary_heuristic_flags':flagged,
            'training_max_candidate_characters':{
                s:max((len(d['text']) for r in source_rows if r['split']=='train' for d in r['documents']
                       if d['source_text_type']==s and d['view_eligible']['retrospective_date_screened']),default=0)
                for s in prep.sources.SOURCES}},
        'limitations':a['protocol.json']['limitations']+[
            'Preview selection is fixed and small; do not infer full-cohort text coverage from it.',
            'No keyword warning in a summary is not evidence of factual accuracy or historical availability.',
            'The zero-GitHub validation preview cannot evaluate incremental GitHub-summary benefit.',
            'Missing-history eligibility does not guarantee that a sparsely observed CVE is easy to forecast.',
            'No forecasting metric is computed in this preparation review.',
        ]}
    for name,h in lock['files'].items():
        require(sha256(directory/name)==h,'Original artifact changed during review')
    return result


def report(a):
    lines = ['H01: verified Day-60 to Day-90 data-readiness review', '',
             '1. Status',
             f"All {a['verified_artifacts']} preparation artifacts verified; the H01 verifier also reproduced them from frozen source.",
             'This is a completed data preview, not a completed training experiment.',
             'No models, predictions, performance metrics, graph embeddings or test scoring.',
             '', '2. New matched cohorts',
             'Split | Assigned | Eligible for both histories | Both input and target missing | Target only missing | Recent input only missing']
    for split,r in a['development'].items():
        p = r['eligibility_partition']
        lines.append(f"{split} | {sum(p.values())} | {p['eligible']} | {p['both_missing']} | "
                     f"{p['target_only_missing']} | {p['recent_input_only_missing']}")
    lines += ['These exclusion columns are mutually exclusive; the original missing-target/missing-input totals overlap.',
              'All 5968 CVEs remain in metadata accounting; 2985 are assigned to reserved test, not verified test-eligible.',
              'No related group crosses active train/validation/test roles. Source group definitions stay conservative.',
              '', '3. Changes from R07',
              'Split | Old eligible | New eligible | Retained | Gained | Lost']
    for split,r in a['development'].items():
        lines.append(f"{split} | {r['old_eligible']} | {r['new_eligible']} | {r['retained']} | "
                     f"{len(r['gained_ids'])} | {len(r['lost_ids'])}")
        lines.append(f"Lost reasons: {r['lost_reasons']}; gained records' old roles: {r['gained_old_roles']}.")
    lines += ['Moving the cutoff changes eligibility and split maturity; R07/H01 headline errors would not be a matched comparison.',
              '', '4. Missing observations on the full matched development cohort',
              'Split | History | Eligible CVEs | Complete input histories | Partial histories | Minimum observed input scores']
    for split,r in a['development'].items():
        for name,h in r['histories'].items():
            lines.append(f"{split} | {name} | {h['n']} | {h['complete']} | {h['partial']} | {h['minimum']}")
        lines.append(f"{split}: {r['extra_observed_days_0_29']['none']} CVEs gain no observed scores when extending the window to Day 0.")
    lines += ['Recent history has 31 days; longer history has 61. Longer histories can be incomplete even when the recent window is complete.',
              'Do not discard partial histories based on this review. The future loader must honor masks and within-window observation ages.',
              '', '5. Preview contract checks',
              'Exactly 16 training and 8 validation CVEs have numeric histories and labels exported.',
              'Inputs stop at Day 60, targets are observed at Day 90, and the forecast horizon is 30 days.',
              'Both history variants use identical IDs and share the same latest observed score.',
              'Recent history does not inherit earlier observations through its age indicators.',
              'Stored masks, dates, sequence lengths and observation ages were independently reconstructed.',
              '', '6. Source coverage in the preview',
              'Split | Source | CVEs with text | CVEs with candidate text']
    for r in a['assignment']['source_coverage']:
        lines.append(f"{r['split']} | {r['source']} | {r['cves_with_text']} | {r['cves_in_date_screened_view']}")
    sr = a['preview_source_review']
    lines += ['No GitHub summary appears in these eight validation CVEs. This tests missing-source behavior, not GitHub benefit.',
              'A future full-development source audit must report coverage before source-ablation claims.',
              'Do not silently change this sample to obtain better results.',
              'The sample differs from MS01; 4/8 versus 1/8 retained SMP is not a paired estimate of the anchor effect.',
              f"Stored post-date entries: {sr['post_date_counts']}. Timezone entries: {sr['post_timezone_counts']}.",
              'Conflicting same-text post timestamps:']
    for r in sr['post_conflicts']:
        lines.append(f"{r['cve']} [{r['platform']}]: {r['dates']}; candidate eligible={r['candidate_eligible']}.")
    lines += [f"Summary keyword warnings: {len(sr['summary_heuristic_flags'])}. Absence of these cues does not establish quality.",
              f"Training-only maximum candidate text lengths, in characters: {sr['training_max_candidate_characters']}.",
              'All text remains historically unverified. Longer evidence windows do not authenticate source dates.',
              '', '7. Next step',
              'The preview is structurally ready for full matched development-data preparation.',
              'Prepare the same 1029 training / 173 validation CVEs for both histories; keep test labels/text closed.',
              'Implement a new tested loader for 31- and 61-day histories; do not reuse a hardcoded 31-day loop.',
              'Freeze matched EPSS-only recent/long controls and persistence before training. No new loss/decay search is needed.',
              'Keep multi-source TPG fusion in the planned study, with a full development source-coverage audit and Day-60 masks.',
              'No full export or training has been performed by this review.',
              '', '8. Limitations']+['- '+v for v in a['limitations']]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    args = parser.parse_args()
    require(args.verify_only or not OUTPUT.exists(),'Review exists; use --verify-only')
    result = analyze()
    text = report(result)
    if args.verify_only:
        require(read(OUTPUT/'analysis.json')==result and (OUTPUT/'analysis.txt').read_text()==text,'Review differs')
        print('H01 readiness review reproduced. No training or test scoring.')
    else:
        OUTPUT.mkdir(parents=True,exist_ok=False)
        write(OUTPUT/'analysis.json',result)
        (OUTPUT/'analysis.txt').write_text(text,encoding='utf-8')
        print(f'H01 readiness review saved to {OUTPUT}. No training or test scoring.')


if __name__ == '__main__':
    main()
