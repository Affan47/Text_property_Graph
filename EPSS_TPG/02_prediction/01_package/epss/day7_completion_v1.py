"""Close the assessment-history collection using immutable evidence, without fits."""
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from . import day7_followup_v1 as prior
from .day7_availability_v1 import collector, plan

a = prior.a
BASE = prior.prior.old.BASE
FIRST = prior.PASS
COMPLETE = FIRST.with_name('assessment_complete_20261116_v1')
OUTPUT = BASE/'24_day7_completion_review_v1/01_reddit/review'
CLI = prior.CLI.with_name('29_review_day7_completion.py')


def indexed(rows, keys):
    out = {tuple(r[k] for k in keys): r for r in rows}
    a.require(len(out) == len(rows), 'Duplicate comparison rows')
    return out


def ensure_preserved(old_rows, new_rows, keys):
    before, after = indexed(old_rows, keys), indexed(new_rows, keys)
    a.require(set(before) <= set(after), 'Previously present records disappeared')
    a.require(all(after[k] == row for k, row in before.items()), 'Previously frozen row changed')
    return len(before)


def in_snapshot_interval(receipt, start, end):
    finished = plan.timestamp(receipt['finished_at'])
    return plan.timestamp(start) < finished <= plan.timestamp(end)


def state_rates(rows):
    result = []
    for cutoff in ('end_day6', 'end_day7'):
        members = [r for r in rows if r['cutoff_name'] == cutoff]
        for state in sorted({r['cvss_status'] for r in members}):
            received = [r for r in members if r['cvss_status'] == state]
            eligible = [r for r in received if r['label'] != '']
            positive = sum(r['label'] == '1' for r in eligible)
            result.append({'cutoff':cutoff, 'state':state, 'histories':len(received),
                           'eligible':len(eligible), 'risers':positive,
                           'riser_rate':positive/len(eligible) if eligible else None})
    return result


def run(verify=False):
    p = plan.load()
    pins = {str(path):a.sha(path) for path in
            (Path(__file__), Path(prior.__file__), CLI, CLI.with_name('test_day7_completion.py'))}
    data = {}
    for name, directory in [('first', FIRST), ('complete', COMPLETE)]:
        a.check_lock(directory)
        finding = a.read(directory/'findings.json'); a.check_pins(finding['source_pins'])
        pins.update(finding['source_pins'])
        for path in directory.iterdir():
            if path.is_file():pins[str(path)] = a.sha(path)
        data[name] = {'findings':finding, 'snapshot':a.read(directory/'snapshot.json'),
                      'fields':a.read(directory/'field_reconstructions.json'),
                      'availability':prior.csv_rows(directory/'availability.csv'),
                      'metrics':prior.csv_rows(directory/'ranking_metrics.csv'),
                      'intervals':prior.csv_rows(directory/'paired_intervals.csv'),
                      'scores':prior.csv_rows(directory/'ranking_scores.csv')}
    first, full = data['first'], data['complete']
    old, new = first['snapshot'], full['snapshot']
    a.require(new['policy'] == old['policy'], 'Availability or ranking policy changed')
    a.require(new['requested_ids'] == old['requested_ids'] == p['assessment_ids'], 'Assessment identity changed')
    a.require(not new['pending_ids'] and new['received_ids'] == new['requested_ids'], 'Collection not complete')
    recovered = sorted(set(new['received_ids'])-set(old['received_ids']))
    a.require(recovered == old['pending_ids'], 'Recovered set differs from original pending set')
    unchanged_availability = ensure_preserved(first['availability'], full['availability'], ('cve','cutoff_name'))
    unchanged_scores = ensure_preserved(first['scores'], full['scores'], ('cve',))
    a.require(all(full['fields'][c] == v for c,v in first['fields'].items()), 'Historical reconstruction changed')
    requests, late = [], []
    for c in tqdm(new['received_ids'], desc='Verify retry receipts and description events; offline', unit='CVE'):
        events, epins = collector.read_complete(c); a.require(events is not None, 'Missing complete history')
        pins.update(epins)
        for path in sorted((collector.STORE/c/'attempts').glob('*/page_*/receipt.json')):
            info = a.read(path)
            if not in_snapshot_interval(info, old['captured_at_utc'], new['captured_at_utc']):continue
            body = path.with_name('body.json')
            a.require(a.sha(body) == info['body_sha256'] and not info['credentials_sent'], 'Invalid retry receipt')
            a.require(c in recovered, 'A previously complete CVE was requested again')
            pins[str(path)] = a.sha(path); pins[str(body)] = info['body_sha256']
            requests.append({'cve':c, 'receipt':str(path), 'http_status':info['http_status'],
                             'error':info['error'], 'started_at':info['started_at'], 'finished_at':info['finished_at']})
        meta = p['records'][c]
        if any(plan.timestamp(meta['cutoff']) < plan.timestamp(e['created']) <= plan.timestamp(meta['snapshot_asof'])
               and any(d.get('type') == 'Description' for d in e['details']) for e in events):late.append(c)
    rows = full['availability']; end7 = [r for r in rows if r['cutoff_name'] == 'end_day7']
    changed = [c for c,v in full['fields'].items()
               if v['end_day7']['description']['supported'] and not v['end_day7']['description']['same_as_saved']]
    matched_risers = [r for r in end7 if r['label'] == '1' and r['dated_or_assumed_description_cvss_match'] == 'True']
    same7 = [r for r in matched_risers if r['first_cvss_addition_day'] == '7']
    before = indexed(first['findings']['support'], ('cutoff','tier'))
    support_delta = []
    for r in full['findings']['support']:
        oldrow = before[r['cutoff'],r['tier']]
        support_delta.append({'cutoff':r['cutoff'], 'tier':r['tier'],
                              'first_eligible':oldrow['eligible_cves'], 'complete_eligible':r['eligible_cves'],
                              'added_eligible':r['eligible_cves']-oldrow['eligible_cves'], 'risers':r['risers'],
                              'positive_groups':r['positive_groups']})
    pairs = indexed(full['intervals'], ('cutoff','tier','metric'))
    mm = indexed([r for r in full['metrics'] if r['fold']=='0' and not r.get('label_sensitivity')],
                 ('cutoff','tier','method'))
    ranking = []
    for key, r in sorted(pairs.items()):
        cutoff,tier,metric = key
        ranking.append({'cutoff':cutoff,'tier':tier,'metric':metric,
                        'past_epss':mm[cutoff,tier,'past_epss'][metric],
                        'model_plus_epss':mm[cutoff,tier,'frozen_D_plus_past_epss'][metric],
                        **{k:r[k] for k in ('difference','low','high','draws','valid_draws','groups')}})
    subset = {c for c in recovered}
    recovered_rows = [r for r in end7 if r['cve'] in subset]
    remaining = sorted(set(p['cohort_ids'])-set(new['received_ids']))
    extra_complete = []
    for c in remaining:
        marker = collector.STORE/c/'complete.json'
        if marker.is_file() and plan.timestamp(a.read(marker)['completed_at']) <= plan.timestamp(new['captured_at_utc']):
            extra_complete.append(c); pins[str(marker)] = a.sha(marker)
    result = {'schema':'day7_completion_review_v1', 'first_snapshot':str(FIRST), 'complete_snapshot':str(COMPLETE),
              'requested':len(new['requested_ids']), 'completed':len(new['received_ids']), 'pending':0,
              'recovered':len(recovered), 'recovered_cves':recovered,
              'retry_requests':len(requests), 'retry_http_counts':dict(Counter(str(r['http_status']) for r in requests)),
              'recovered_label_counts':dict(Counter(r['label'] for r in recovered_rows)),
              'unchanged_availability_rows':unchanged_availability, 'unchanged_ranking_rows':unchanged_scores,
              'original_complete_field_reconstructions_unchanged':len(first['fields']),
              'planned_cohort':len(p['cohort_ids']), 'nonassessment_records':len(remaining),
              'nonassessment_completed_at_snapshot':len(extra_complete),
              'late_description_event_cves':len(late), 'supported_different_descriptions':len(changed),
              'cvss_rates':state_rates(rows), 'label_status_counts':dict(Counter(r['label_status'] for r in end7)),
              'support':full['findings']['support'], 'support_changes':support_delta,
              'timing':full['findings']['timing'], 'description_no_recorded_event':full['findings']['description_no_recorded_event_through_snapshot'],
              'day7_matched_risers':len(matched_risers), 'matched_first_cvss_on_day7':len(same7),
              'those_crossing_on_day8':sum(p['records'][r['cve']]['label_not_predictor']['first_observed_crossing_day']==8 for r in same7),
              'ranking':ranking, 'source_pins':pins, 'training':False, 'network_requests':0, 'test_scoring':False,
              'decision':'ASSESSMENT_COLLECTION_COMPLETE; HISTORICAL_FORECAST_NOT_ESTABLISHED'}
    report = ['Day-7 assessment collection: completed retry and final analysis','',
              f'{len(new["received_ids"]):,} / {len(new["requested_ids"]):,} valid histories; {len(recovered)} recovered; zero pending.',
              f'{len(requests)} retry requests: {result["retry_http_counts"]}. Requests are not unique CVEs.',
              'Earlier receipt bodies, reconstructed fields, labels and shared ranking-score rows are unchanged.',
              f'{len(remaining):,} other planned original-training CVEs are outside this assessment collection; '
              f'{len(extra_complete)} have completion markers by the frozen snapshot time.',
              'All scores remain frozen CT07 predictions, not a newly trained early-warning classifier.',
              'Broad matched descriptions assume complete history; reference/preprocessing/model timing stays unverified.',
              'No-recorded-assignment is specific to the matched CVSS provider/version, not absence of every severity assessment.',
              'Positive AP or AUROC intervals do not establish historical prediction or description causality.',
              '20,000-draw evidence-group intervals condition on mean-seed scores and the one-day label assumption; unadjusted.',
              'Strict labels and end-of-Day6 sensitivity remain in the full tier report. No threshold/epoch is selected.',
              '', 'Cutoff | Tier | CVEs | Risers | Positive groups']
    report += [f'{r["cutoff"]} | {r["tier"]} | {r["eligible_cves"]} | {r["risers"]} | {r["positive_groups"]}'
               for r in result['support']]
    report += ['', 'Cutoff | Tier | Metric | Past EPSS | D + EPSS | Difference | 95% bounds']
    report += [f'{r["cutoff"]} | {r["tier"]} | {r["metric"]} | {r["past_epss"]} | {r["model_plus_epss"]} | '
               f'{r["difference"]} | {r["low"]} to {r["high"]}' for r in ranking]
    a.publish(OUTPUT, {'findings.json':result, 'retry_receipts.csv':prior.prior.old.csv_text(requests),
                      'recovered_cves.csv':prior.prior.old.csv_text(recovered_rows),
                      'cvss_rates.csv':prior.prior.old.csv_text(result['cvss_rates']),
                      'ranking.csv':prior.prior.old.csv_text(ranking),
                      'support_changes.csv':prior.prior.old.csv_text(support_delta),
                      'late_description_event_cves.json':late,
                      'supported_different_descriptions.json':changed,
                      'REPORT.txt':'\n'.join(report)+'\n'}, pins, verify)
    print('Day-7 completion review '+('reproduced: ' if verify else 'saved: ')+str(OUTPUT)+
          '. No training, requests or test scoring.')
