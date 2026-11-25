"""RO01: bounded rolling-origin feasibility audit; no training or test scoring."""
import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
from fractions import Fraction
from pathlib import Path

from . import forecast_trajectory as prior

data, helpers = prior.data, prior.helpers
parent = data.parent
read, write, require, sha256, tqdm = data.read, data.write, data.require, data.sha256, data.tqdm
SCHEMA = 'ro01_rolling_origin_feasibility_v1'
ROOT = data.ROOT
CONFIG = ROOT/'02_prediction/04_configs/rolling_origin_audit_v1.json'
OUTPUT = ROOT/'05_results/02_evaluation/epss_rolling_origin/feasibility_v1'
SCRIPT = helpers.SCRIPTS/'audit_forecast_rolling_origin.py'
TEST = helpers.SCRIPTS/'test_forecast_rolling_origin.py'
FILES = {'protocol.json', 'run_identity.json', 'window_audit.json', 'summary.json', 'summary.txt'}


def protocol():
    p = read(CONFIG)
    require(p['schema'] == SCHEMA and p['anchors'] == [30,45,60] and p['leads'] == [7,14,30]
            and p['history_lengths'] == [10,20,31] and p['large_change_threshold'] == .01
            and not p['training'] and not p['test_scoring'], 'Unexpected RO01 protocol')
    return p


def inspect_window(raw, cve, split, publication, anchor, lead, policy, p):
    require(split in ('train', 'validation'), 'Refuse test CVE inspection')
    require(anchor in p['anchors'] and lead in p['leads'], 'Unregistered window')
    pub = date.fromisoformat(publication)
    anchor_date, target_date = [(pub+timedelta(days=d)).isoformat() for d in (anchor,anchor+lead)]
    candidate, reason = parent.contract.candidate_split(anchor_date, target_date, policy)
    row = {'window_id': f'{cve}__a{anchor}__h{lead}', 'cve': cve, 'split': split,
           'anchor_day': anchor, 'lead_days': lead, 'target_day': anchor+lead,
           'anchor_date': anchor_date, 'target_date': target_date,
           'candidate_calendar_role': candidate, 'temporal_admissible': candidate == split and not reason,
           'eligible': False, 'reasons': [], 'input_observed_days': None, 'latest_observed_input_day': None,
           'input_age_days': None, 'target_observed': None, 'latest_input_score': None, 'target_score': None,
           'change': None, 'direction': None, 'large_absolute_change': None,
           'pooled_weight_denominator': None, 'per_lead_weight_denominator': None,
           'pooled_cve_weight': None, 'per_lead_cve_weight': None}
    # Do not read numeric history/target values for a date-ineligible window.
    if not row['temporal_admissible']:
        row['reasons'] = [reason or 'anchor_leaves_original_cve_role']
        return row
    require(raw['cve'] == cve and raw['published_date'] == publication
            and parent.contract.timestamp(raw['published_timestamp']).date() == pub, 'Changed CVE/publication')
    dates = [(pub+timedelta(days=d)).isoformat() for d in range(91)]
    scores = raw['epss_scores']
    require(isinstance(scores, dict) and set(scores) == set(dates), 'Expected all Day 0-90 keys; missing values must be null')
    windows = {length: [(day, parent.contract.score(scores[dates[day]]))
                       for day in range(anchor-length+1, anchor+1)] for length in p['history_lengths']}
    observed = {length: [(day,v) for day,v in values if v is not None] for length,values in windows.items()}
    row['input_observed_days'] = {str(length): len(values) for length,values in observed.items()}
    target = parent.contract.score(scores[target_date])
    row['target_observed'] = target is not None
    row['target_score'] = target
    if not all(observed.values()):
        row['reasons'].append('no_observed_score_in_every_compared_history')
    if target is None:
        row['reasons'].append('missing_endpoint_score')
    if all(observed.values()):
        latest = [values[-1] for values in observed.values()]
        require(all(item == latest[0] for item in latest), 'Unmatched persistence across histories')
        day, base = latest[0]
        row.update(latest_observed_input_day=day, input_age_days=anchor-day, latest_input_score=base)
    row['eligible'] = not row['reasons']
    if row['eligible']:
        delta = target-row['latest_input_score']
        row.update(change=delta, direction='unchanged' if delta == 0 else 'increase' if delta > 0 else 'decrease',
                   large_absolute_change=abs(delta) >= p['large_change_threshold'])
    return row


def assign_weights(rows):
    pooled = Counter(r['cve'] for r in rows if r['eligible'])
    per_lead = Counter((r['cve'],r['lead_days']) for r in rows if r['eligible'])
    totals, lead_totals = defaultdict(Fraction), defaultdict(Fraction)
    for r in rows:
        if r['eligible']:
            n, m = pooled[r['cve']], per_lead[r['cve'],r['lead_days']]
            r.update(pooled_weight_denominator=n, per_lead_weight_denominator=m,
                     pooled_cve_weight=float(Fraction(1,n)), per_lead_cve_weight=float(Fraction(1,m)))
            totals[r['cve']] += Fraction(1,n)
            lead_totals[r['cve'],r['lead_days']] += Fraction(1,m)
    require(all(v == 1 for v in totals.values()) and all(v == 1 for v in lead_totals.values()), 'Unequal total CVE weights')


def describe(rows):
    accepted = [r for r in rows if r['eligible']]
    return {'candidate_windows': len(rows), 'eligible_windows': len(accepted),
            'temporally_excluded': sum(not r['temporal_admissible'] for r in rows),
            'reason_counts_nonexclusive': dict(sorted(Counter(reason for r in rows for reason in r['reasons']).items())),
            'unchanged': sum(r['direction'] == 'unchanged' for r in accepted),
            'increase': sum(r['direction'] == 'increase' for r in accepted),
            'decrease': sum(r['direction'] == 'decrease' for r in accepted),
            'large_absolute_change': sum(r['large_absolute_change'] for r in accepted),
            'large_change_percent': 100*sum(r['large_absolute_change'] for r in accepted)/len(accepted) if accepted else None,
            'partial_histories': {str(n): sum(r['input_observed_days'][str(n)] < n for r in accepted) for n in (10,20,31)},
            'missing_score_on_anchor_date': sum(r['input_age_days'] > 0 for r in accepted)}


def summarize(rows, ids, p):
    groups, paired, combined = {}, {}, {}
    for split, chosen in ids.items():
        group = [r for r in rows if r['split'] == split]
        groups[split] = {str(a): {str(h): describe([r for r in group if r['anchor_day'] == a and r['lead_days'] == h])
                                 for h in p['leads']} for a in p['anchors']}
        eligible = [r for r in group if r['eligible']]
        by_cve = Counter(r['cve'] for r in eligible)
        combined[split] = {**describe(group), 'cves_with_any_eligible_window': len(by_cve),
            'cves_without_eligible_windows': len(chosen)-len(by_cve),
            'eligible_windows_per_cve': dict(sorted(Counter(str(by_cve[c]) for c in chosen).items())),
            'cves_with_any_large_change': len({r['cve'] for r in eligible if r['large_absolute_change']}),
            'cves_with_any_nonzero_change': len({r['cve'] for r in eligible if r['change'] != 0}),
            'equal_cve_weight_large_change_percent': 100*sum(r['pooled_cve_weight'] for r in eligible if r['large_absolute_change'])/len(by_cve) if by_cve else None}
        paired[split] = {}
        for lead in p['leads']:
            lookup = {a: {r['cve']: r for r in eligible if r['anchor_day'] == a and r['lead_days'] == lead}
                      for a in p['anchors']}
            common = set.intersection(*(set(v) for v in lookup.values()))
            pairs = {}
            for later in (45,60):
                matched = set(lookup[30]) & set(lookup[later])
                before = {c for c in matched if lookup[30][c]['large_absolute_change']}
                after = {c for c in matched if lookup[later][c]['large_absolute_change']}
                pairs[str(later)] = {'matched_cves': len(matched), 'day30_large_change': len(before),
                    'later_large_change': len(after), 'later_only_large_change': len(after-before),
                    'day30_only_large_change': len(before-after), 'both_large_change': len(before & after)}
            paired[split][str(lead)] = {'common_cves_all_three_anchors': len(common),
                'common_cohort_by_anchor': {str(a): describe([lookup[a][c] for c in sorted(common)]) for a in p['anchors']},
                'day30_vs_later': pairs}
    return {'schema': SCHEMA, 'cohort': {s: len(v) for s,v in ids.items()}, 'by_window': groups,
            'paired_anchors': paired, 'combined': combined, 'training': False, 'test_scoring': False,
            'limitations': p['limitations'],
            'decision': 'Feasibility evidence only. No automatic training approval or selection of the anchor with most future changes.'}


def context(scale):
    require(scale in ('smoke','full'), 'Unknown scale')
    p = protocol()
    directory = data.OUTPUT/'full'
    pins, _ = helpers.check_files(directory, 'preparation_lock.json', data.SCHEMA, data.FILES)
    identity = read(directory/'run_identity.json')
    require(identity['schema'] == data.SCHEMA and identity['scale'] == 'full'
            and set(identity['ids']) == {'train','validation'}
            and {s: len(v) for s,v in identity['ids'].items()} == p['expected_cohort'], 'Wrong development cohort')
    parent.sources.check_pins(identity['source_pins'])
    pins.update(identity['source_pins'])
    run_identity = read(prior.OUTPUT/'full/run_identity.json')
    require(run_identity['ids'] == identity['ids'] and run_identity['scale'] == 'full'
            and run_identity['schema'] == data.SCHEMA and not run_identity['test_scoring'], 'Wrong completed TR01 run')
    completed, _ = helpers.check_files(prior.OUTPUT/'full','evaluation_lock.json',data.SCHEMA,prior.expected(run_identity))
    pins.update(completed)
    parent.sources.check_pins(run_identity['pinned_files'])
    pins.update(run_identity['pinned_files'])
    ids = identity['ids']
    if scale == 'smoke':
        small, _ = helpers.check_files(data.OUTPUT/'smoke','preparation_lock.json',data.SCHEMA,data.FILES)
        pins.update(small)
        ids = read(data.OUTPUT/'smoke/run_identity.json')['ids']
    require(set(ids) == {'train','validation'} and all(v and v == sorted(set(v)) and set(v) <= set(identity['ids'][s]) for s,v in ids.items())
            and not set(ids['train']) & set(ids['validation']), 'Repeated or changed CVE roles')
    locator = parent.sources.PREPARED/'preparation_summary.json'
    require(sha256(locator) == pins[str(locator)], 'Changed source locator')
    source = Path(read(locator)['source_path'])
    require(sha256(source) == pins[str(source)], 'Changed source')
    selected = set(ids['train']+ids['validation'])
    records = read(source)
    require(len(records) == len({r['cve'] for r in records}), 'Duplicate source CVE')
    raw = {r['cve']: r for r in records if r['cve'] in selected}
    require(set(raw) == selected, 'Missing development CVE')
    frozen = read(directory/'inputs.json')
    manifest = read(directory/'split_manifest.json')
    for s, chosen in ids.items():
        require(all(manifest[c]['split'] == s for c in chosen), 'Changed frozen roles')
    for path in (CONFIG,SCRIPT,TEST,Path(__file__).resolve(),Path(data.__file__).resolve(),
                 Path(prior.__file__).resolve(),Path(helpers.__file__).resolve()):
        pins[str(path)] = sha256(path)
    return p, ids, raw, frozen, identity['split_policy'], pins


def report(s):
    lines = ['RO01 rolling-origin feasibility: no training or test scoring',
        'All anchors are days after CVE publication. A lead is days after that anchor.',
        'Example: anchor 45, lead 14 uses only past input through Day 45 and checks the Day-59 endpoint.',
        'Every CVE keeps its existing train/validation role. Date-ineligible windows are not numerically inspected.',
        'Only a recorded endpoint and at least one observed score in EACH 10/20/31-day history are required.',
        'Whether the endpoint increases/decreases is never an eligibility criterion.',
        f'Cohort: {s["cohort"]}', '',
        'Split | Anchor day | Days ahead | Candidate CVEs | Eligible CVEs | Date-excluded | Unchanged | Increased | Decreased | Absolute change >=0.01']
    for split, anchors in s['by_window'].items():
        for a, leads in anchors.items():
            for h,r in leads.items():
                lines.append(f'{split} | {a} | {h} | {r["candidate_windows"]} | {r["eligible_windows"]} | '
                    f'{r["temporally_excluded"]} | {r["unchanged"]} | {r["increase"]} | {r["decrease"]} | {r["large_absolute_change"]}')
    lines += ['', 'Counts of changes compare the endpoint to the latest observed score within the history, in original source precision.',
        '0.01 is one EPSS percentage point. Missing endpoints are not zero or unchanged.',
        'Reason counts in JSON can overlap (both input and target missing); the date-excluded count is disjoint.',
        '', 'Matched comparisons: same CVEs, same future lead, different anchor days',
        'Split | Lead | Later anchor | Matched CVEs | Large change at Day-30 anchor | Large change at later anchor | Later only | Day-30 only']
    for split, leads in s['paired_anchors'].items():
        for h,r in leads.items():
            for a,pair in r['day30_vs_later'].items():
                lines.append(f'{split} | {h} | {a} | {pair["matched_cves"]} | {pair["day30_large_change"]} | '
                    f'{pair["later_large_change"]} | {pair["later_only_large_change"]} | {pair["day30_only_large_change"]}')
    lines += ['', 'Later-only counts describe outcomes; they do not select examples or prove earlier warning.',
        'JSON also reports the common eligible cohort across all three anchors for each lead.',
        '', 'CVE weighting plan',
        'For a pooled task, each eligible window gets 1 / all eligible windows for that CVE.',
        'For separate lead-specific tasks, each window gets 1 / eligible anchors for that CVE and lead.',
        'Each denominator is exported; exact rational checks confirm total weight 1 per eligible CVE (or CVE/lead).',
        'These are proposed loss weights, not observed outcome weights. Multiple windows are not independent CVEs.',
        '', 'Files',
        'window_audit.json contains diagnostic rows, including targets where dates are allowed. It is NOT model input.',
        'summary.json / summary.txt explain coverage and paired counts. protocol.json fixes the design.',
        'run_identity.json and audit_lock.json preserve source hashes and unchanged roles.',
        '', s['decision'], *s['limitations']]
    return '\n'.join(lines)+'\n'


def build(scale):
    p, ids, raw, frozen, policy, pins = context(scale)
    rows = []
    for split, chosen in ids.items():
        for c in tqdm(chosen,desc='Audit '+split+' rolling windows',unit='CVE'):
            for anchor in p['anchors']:
                for lead in p['leads']:
                    rows.append(inspect_window(raw[c],c,split,frozen[c]['publication_date'],anchor,lead,policy,p))
    require(len(rows) == len({r['window_id'] for r in rows}) == 9*sum(map(len,ids.values())), 'Missing/duplicate windows')
    assign_weights(rows)
    s = summarize(rows,ids,p)
    return {'protocol.json': p, 'run_identity.json': {'schema':SCHEMA,'scale':scale,'ids':ids,
        'split_policy':policy,'source_pins':pins,'training':False,'test_scoring':False},
        'window_audit.json': {'audit_only':True,'rows':rows}, 'summary.json':s,'summary.txt':report(s)}


def execute(scale, check_only=False, verify_only=False):
    a = build(scale)
    print('[COHORT] '+str(a['summary.json']['cohort'])+'; all CVE roles retained.',flush=True)
    for split,r in a['summary.json']['combined'].items():
        print(f'[WINDOWS] {split}: {r["eligible_windows"]}/{r["candidate_windows"]} eligible; '
              f'{r["temporally_excluded"]} date-excluded; {r["cves_with_any_eligible_window"]} CVEs represented.',flush=True)
    pins = a['run_identity.json']['source_pins']
    parent.sources.check_pins(pins)
    if check_only:
        print(f'RO01 {scale} preflight passed. No files written, training or test scoring.')
        return
    directory = OUTPUT/scale
    if verify_only:
        helpers.check_files(directory,'audit_lock.json',SCHEMA,FILES)
        for n,v in a.items():
            require(((directory/n).read_text() if n.endswith('.txt') else read(directory/n)) == v,'Audit differs: '+n)
        print(f'RO01 {scale} rolling-origin audit reproduced. No training or test scoring.')
        return
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with helpers.writer_lock(OUTPUT/'.writer.lock'):
        require(not directory.exists(),'Audit exists; use --verify-only')
        directory.mkdir()
        for n,v in a.items():
            if n.endswith('.txt'):
                (directory/n).write_text(v,encoding='utf-8')
            else:
                write(directory/n,v)
        parent.sources.check_pins(pins)
        write(directory/'audit_lock.json',{'schema':SCHEMA,'files':{n:sha256(directory/n) for n in sorted(FILES)}})
    print(f'RO01 {scale} rolling-origin audit saved to {directory}. No training or test scoring.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale',choices=('smoke','full'),default='smoke')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only',action='store_true')
    mode.add_argument('--verify-only',action='store_true')
    args = parser.parse_args()
    execute(args.scale,args.check_only,args.verify_only)


if __name__ == '__main__':
    main()
