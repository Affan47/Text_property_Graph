"""Review MS01 and compare EPSS windows on frozen training CVEs only."""
import argparse
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_multisource as source

read, write, require, sha256, tqdm = source.read, source.write, source.require, source.sha256, source.tqdm
OUTPUT = source.ROOT/'05_results/02_evaluation/epss_day30_to_day60/multisource_window_review_v1'
WINDOWS = {'days_0_30':(0,30), 'days_30_60':(30,60), 'days_60_90':(60,90)}
MIN_PAIRS = 20


def difference(a, b):
    return float(Decimal(str(a))-Decimal(str(b)))


def window(values, start, end):
    require(0 <= start < end < len(values), 'Invalid window')
    observed = [v for v in values[start:end+1] if v is not None]
    changes = [difference(values[i],values[i-1]) for i in range(start+1,end+1)
               if values[i] is not None and values[i-1] is not None]
    return {'observed_scores':len(observed), 'observed_adjacent_pairs':len(changes),
            'changed_adjacent_pairs':sum(v != 0 for v in changes),
            'mean_absolute_daily_change':statistics.mean(abs(v) for v in changes) if changes else None,
            'observed_range':max(observed)-min(observed) if observed else None,
            'range_is_exact_for_window':len(observed)==end-start+1}


def last_observed(values, start, end):
    found = [(i,values[i]) for i in range(start,end+1) if values[i] is not None]
    require(bool(found), 'No observed input in window')
    return found[-1]


def sign(value):
    return 'up' if value > 0 else 'down' if value < 0 else 'unchanged'


def training_row(record, validation_start):
    publication = date.fromisoformat(record['published_date'])
    day90 = publication+timedelta(days=90)
    if day90 >= date.fromisoformat(validation_start):
        return {'cve':record['cve'], 'excluded':'day90_reaches_validation_period',
                'day90_date':day90.isoformat()}
    dates, values = source.contract.score_days(record)
    windows = {name:window(values,*bounds) for name,bounds in WINDOWS.items()}
    first_day, first = last_observed(values,0,30)
    middle_day, middle = last_observed(values,30,60)
    endpoints = None
    if values[60] is not None and values[90] is not None:
        endpoints = {'day30_last_observed':first,'day30_last_observed_day':first_day,
                     'day60_last_observed':middle,'day60_last_observed_day':middle_day,
                     'day60':values[60],'day90':values[90],
                     'change_30_to_60':difference(values[60],first),
                     'change_60_to_90':difference(values[90],middle),
                     'change_30_to_90':difference(values[90],first)}
    change_date = record.get('epss_version_change_date')
    change = date.fromisoformat(change_date) if change_date else None
    return {'cve':record['cve'], 'publication_date':publication.isoformat(),
            'day90_date':dates[90], 'windows':windows,'endpoints':endpoints,
            'missing_day90':values[90] is None,
            'scheduled_version_change_in_days_0_90':bool(change and publication<=change<=day90),
            'version_change_date':change_date,
            'version_metadata_basis':record.get('epss_version_metadata_basis'),
            'version_transition_windows':[name for name,(start,end) in WINDOWS.items()
                if change and publication+timedelta(days=start)<change<=publication+timedelta(days=end)]}


def distribution(values):
    return {'count':len(values),'mean':statistics.mean(values) if values else None,
            'median':statistics.median(values) if values else None,
            'min':min(values) if values else None,'max':max(values) if values else None}


def matched_windows(rows, minimum):
    chosen = [r for r in rows if all(w['observed_adjacent_pairs'] >= minimum for w in r['windows'].values())]
    summaries = {}
    for name in WINDOWS:
        summaries[name] = {
            'cves':len(chosen),
            'mean_absolute_daily_change':distribution([r['windows'][name]['mean_absolute_daily_change'] for r in chosen]),
            'cves_with_any_observed_adjacent_change':sum(r['windows'][name]['changed_adjacent_pairs']>0 for r in chosen),
            'total_observed_adjacent_pairs':sum(r['windows'][name]['observed_adjacent_pairs'] for r in chosen),
            'total_changed_adjacent_pairs':sum(r['windows'][name]['changed_adjacent_pairs'] for r in chosen)}
    comparison = Counter()
    for r in chosen:
        early = r['windows']['days_0_30']['mean_absolute_daily_change']
        middle = r['windows']['days_30_60']['mean_absolute_daily_change']
        comparison['middle_more_variable' if middle>early else
                   'early_more_variable' if early>middle else 'equal_variability'] += 1
    return {'minimum_adjacent_pairs_each_window':minimum,'ids':[r['cve'] for r in chosen],
            'windows':summaries,'early_middle_comparison':dict(comparison)}


def endpoint_summary(rows):
    chosen = [r for r in rows if r['endpoints'] is not None]
    results = {}
    for name in ('change_30_to_60','change_60_to_90','change_30_to_90'):
        values = [r['endpoints'][name] for r in chosen]
        results[name] = {'count':len(values), 'directions':dict(Counter(sign(v) for v in values)),
                         'absolute_change':distribution([abs(v) for v in values])}
    transitions = Counter(sign(r['endpoints']['change_30_to_60'])+' -> '+
                          sign(r['endpoints']['change_60_to_90']) for r in chosen)
    return {'ids':[r['cve'] for r in chosen],'comparisons':results,'direction_transitions':dict(transitions)}


def preview_review(records):
    cues = ('loading message','css error','no technical details','no useful')
    flagged, conflicts = [], []
    for row in records:
        for d in row['documents']:
            if 'summary' in d['source_text_type']:
                matches = [c for c in cues if c in d['text'].lower()]
                if matches:
                    flagged.append({'cve':row['cve'],'source':d['source_text_type'],
                                    'cues':matches,'text':d['text'],
                                    'interpretation':'Heuristic review cue only; source pages not rechecked; no mask changed'})
            if 'same_platform_text_has_conflicting_dates' in d['quality_flags']:
                conflicts.append({'cve':row['cve'],'platform':d['platform'],
                                  'dates':[e['value'] for e in d['stored_date_evidence']]})
    return {'summary_review_cues':flagged,'conflicting_post_documents':conflicts,
            'lengths_by_source':{s:distribution([len(d['text']) for r in records for d in r['documents']
                                                if d['source_text_type']==s]) for s in source.SOURCES},
            'training_candidate_max_characters_by_source':{
                s:max((len(d['text']) for r in records if r['split']=='train' for d in r['documents']
                       if d['source_text_type']==s and d['view_eligible'][source.VIEWS[0]]), default=0)
                for s in source.SOURCES}}


def analyze():
    raw_preview, anchors, identity = source.context()
    preview = source.OUTPUT
    lock = read(preview/'preview_lock.json')
    require(set(lock['files']) == source.FILES and read(preview/'run_identity.json') == identity,
            'Preview identity differs')
    for name,h in lock['files'].items():
        require(sha256(preview/name)==h, 'Changed preview artifact: '+name)
    records = [{'split':split,**source.adapt_record(raw_preview[c],anchors[c],identity['protocol']['source_fields'])}
               for split,ids in identity['ids'].items() for c in ids]
    require(records == read(preview/'records.json'), 'Preview reconstruction differs')
    coverage = source.summarize(records,identity)
    require(coverage == read(preview/'analysis.json'), 'Preview summary differs')
    train_ids = read(source.R07/'run_identity.json')['ids']['train']
    manifest = read(source.PREPARED/'split_manifest.json')
    require(len(train_ids)==1023 and all(manifest[c]['split']=='train' for c in train_ids), 'Training-only gate failed')
    policy = read(source.PREPARED/'split_policy.json')
    path = Path(read(source.PREPARED/'preparation_summary.json')['source_path'])
    wanted = set(train_ids)
    raw = {r['cve']:r for r in read(path) if r['cve'] in wanted}
    rows = [training_row(raw[c],policy['validation_start']) for c in tqdm(train_ids,desc='Compare training-only EPSS windows',unit='CVE')]
    eligible = [r for r in rows if 'excluded' not in r]
    no_version = [r for r in eligible if not r['scheduled_version_change_in_days_0_90']]
    matched = matched_windows(eligible, MIN_PAIRS)
    endpoints = endpoint_summary(eligible)
    # Illustrative cases chosen only from training, not from validation model errors.
    examples = sorted([r for r in eligible if r['endpoints'] is not None],
                      key=lambda r:(-abs(r['endpoints']['change_30_to_60']),r['cve']))[:5]
    result = {
        'schema':'multisource_window_review_v1','script_sha256':sha256(Path(__file__)),
        'source_preview_lock_sha256':sha256(preview/'preview_lock.json'),
        'source_pins':identity['source_pins'],
        'scope':'Source-preview text analysis covers 16 train/8 validation CVEs; EPSS window study covers original training CVEs only.',
        'training':False,'new_forecasts':False,'test_scoring':False,'splits_changed':False,
        'preview':{'coverage':coverage, **preview_review(records)},
        'window_study':{'original_training_cves':len(train_ids),
                        'excluded_day90_reaches_validation':len(rows)-len(eligible),
                        'day90_before_validation_cves':len(eligible),
                        'missing_day90_among_in_period':sum(r['missing_day90'] for r in eligible),
                        'matched_windows':matched,
                        'complete_all_windows_sensitivity':matched_windows(eligible,30),
                        'no_scheduled_version_change_sensitivity':matched_windows(no_version,MIN_PAIRS),
                        'scheduled_version_change_count':len(eligible)-len(no_version),
                        'endpoints':endpoints,
                        'no_scheduled_version_change_endpoints':endpoint_summary(no_version),
                        'training_examples':examples, 'per_cve':rows},
        'limitations':[
            'Only original R07 training CVEs are considered for the window comparison; no full-dataset inference.',
            'Exclude Day-90 targets reaching the validation period; no replacement split is constructed.',
            'Missing observations are not filled, and changes across gaps are not treated as daily changes.',
            'Main variability table requires at least 20 observed adjacent pairs in each of the three windows.',
            'Matched complete-window and no-scheduled-version-change subsets are sensitivity analyses, not training filters.',
            'Schedule metadata is not independent confirmation of every archive model version.',
            'More past movement is not evidence of easier future prediction; persistence differences here are descriptive, not held-out model evaluation.',
            'Using Days 30-60 to predict Day 90 is a new prediction anchor and task; labels, source timing and chronological splits must be rebuilt.',
            'The source contains only Days 0-90 for this study; later targets require additional observations.',
            'Summary cue checks do not establish factual correctness or identify every bad summary.',
        ]}
    source.check_pins(identity['source_pins'])
    for name,h in lock['files'].items():
        require(sha256(preview/name)==h, 'Preview changed during review')
    return result


def report(a):
    p,w = a['preview'],a['window_study']
    lines = ['MS01 source review and training-only EPSS window feasibility', '',
             '1. Source preview verified',
             '24 CVEs: 16 training, 8 validation. Original preview is unchanged.',
             'Source | Training CVEs with candidate text / 16 | Validation CVEs with candidate text / 8']
    for s in source.SOURCES:
        counts = {r['split']:r['cves_in_date_screened_view'] for r in p['coverage']['coverage'] if r['source']==s}
        lines.append(f"{s} | {counts['train']} | {counts['validation']}")
    lines += ['Only 1/8 validation CVEs has a retained SMP after the stored-date screen; this is a small preview.',
              'No description, summary or post is authenticated as historical Day-30 text.',
              'Two same-text post cases have conflicting supplied dates; keep those associations for review,',
              'since identical wording can also be posted more than once. Conflict is not proof of corrupt data.',
              'Nonempty summaries are not automatically useful: the following heuristic cues need source review.']
    for r in p['summary_review_cues']:
        lines.append(f"{r['cve']} [{r['source']}]: {r['text']}")
    lines += ['No quality mask was changed and no source website was re-fetched.',
              'Training-only maximum candidate document lengths (characters, not tokens):',
              str(p['training_candidate_max_characters_by_source']),
              'Use this to plan graph sizing; do not silently truncate or choose limits from validation outcomes.',
              '', '2. Question: are Days 30-60 more variable than Days 0-30?',
              f"Original training CVEs: {w['original_training_cves']}; excluded because Day 90 reaches validation: {w['excluded_day90_reaches_validation']}.",
              f"In-period CVEs: {w['day90_before_validation_cves']}; missing Day-90 score: {w['missing_day90_among_in_period']}.",
              'Compare the same CVEs across all three windows. Each window has 31 daily values and 30 adjacent pairs.',
              'Only consecutive observed days form a pair. We never bridge a missing day.',
              'Main table requires at least 20 of 30 pairs observed in every window.',
              'Window | Matched CVEs | CVEs with any observed daily change | Mean absolute daily score change | Median per-CVE daily change']
    for name,r in w['matched_windows']['windows'].items():
        d = r['mean_absolute_daily_change']
        lines.append(f"{name} | {r['cves']} | {r['cves_with_any_observed_adjacent_change']} | {d['mean']} | {d['median']}")
    lines += [f"Paired first/middle-window comparison: {w['matched_windows']['early_middle_comparison']}",
              'Mean daily change is first calculated per CVE over its observed adjacent pairs, then averaged across CVEs.',
              'Window endpoints overlap but their daily-change intervals do not.',
              '', '3. Coverage and EPSS-version sensitivity',
              f"Complete all-window subset: {len(w['complete_all_windows_sensitivity']['ids'])} CVEs.",
              f"Scheduled version change somewhere in Days 0-90: {w['scheduled_version_change_count']} of the in-period CVEs.",
              f"No-scheduled-change subset with enough pairs: {len(w['no_scheduled_version_change_sensitivity']['ids'])} CVEs.",
              f"Early/middle comparison on that subset: {w['no_scheduled_version_change_sensitivity']['early_middle_comparison']}",
              'Full numerical sensitivity tables and per-CVE counts are in analysis.json.',
              '', '4. Endpoint changes on a separate matched cohort',
              'This cohort requires observed Day-60/90 targets and usable input histories; it need not satisfy',
              'the adjacent-pair coverage rule above. All three rows use identical CVEs.',
              'Forecast timing | CVEs | Endpoint unchanged/up/down | Mean absolute change from last observed input']
    labels = {'change_30_to_60':'Observe Days 0-30; target Day 60 (30-day horizon)',
              'change_60_to_90':'Observe Days 30-60; target Day 90 (30-day horizon)',
              'change_30_to_90':'Observe Days 0-30; target Day 90 (60-day horizon)'}
    for name,r in w['endpoints']['comparisons'].items():
        lines.append(f"{labels[name]} | {r['count']} | {r['directions']} | {r['absolute_change']['mean']}")
    lines += ['These numbers are persistence reference errors on old training records, not a trained-model comparison.',
              f"Change-direction transitions, first forecast interval -> second: {w['endpoints']['direction_transitions']}",
              'Real training examples:']
    for r in w['training_examples'][:3]:
        e = r['endpoints']
        lines.append(f"{r['cve']}: last observed through Day 30={e['day30_last_observed']}; "
                     f"Day 60={e['day60']}; Day 90={e['day90']}; scheduled version change={r['version_change_date']}.")
    lines += ['', '5. Interpretation and next decision',
              'Yes, moving the prediction anchor to Day 60 is feasible for records with enough observations.',
              'A model at Day 60 can use changes that were unavailable at Day 30. It cannot recover a missed',
              'Day-30 warning retrospectively or demonstrate that the same information was available earlier.',
              'More changing input scores may help or may reflect unpredictable jumps; no guarantee follows.',
              'Compare Days 0-30 -> 60 with Days 30-60 -> 90 using matched CVEs, the same 30-day horizon,',
              'and a persistence reference at each anchor. Adding Days 0-60 -> 90 is a separate longer-history ablation.',
              'Do not use Day-60 data to predict a target at Day 45 or Day 60 as if it were future forecasting.',
              'Keep R07 and MS01 fixed. A Day-60 task needs a new split manifest with anchor/label maturity',
              'and related-group checks, new source-time masks and a preregistered training protocol.',
              'Do not change the task solely because descriptive errors look smaller.',
              'No training, new model forecasts or test analysis has been performed.',
              '', '6. Limitations']+['- '+v for v in a['limitations']]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    args = parser.parse_args()
    require(args.verify_only or not OUTPUT.exists(), 'Review exists; use --verify-only')
    value = analyze()
    text = report(value)
    if args.verify_only:
        require(read(OUTPUT/'analysis.json')==value and (OUTPUT/'analysis.txt').read_text()==text,
                'Review differs')
        print('MS01 and training-window review reproduced. No training or test analysis.')
    else:
        OUTPUT.mkdir(parents=True,exist_ok=False)
        write(OUTPUT/'analysis.json',value)
        (OUTPUT/'analysis.txt').write_text(text,encoding='utf-8')
        print(f'MS01 and training-window review saved to {OUTPUT}. No training or test analysis.')


if __name__ == '__main__':
    main()
