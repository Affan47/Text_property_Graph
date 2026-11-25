"""Read-only, conditional SMP-date/EPSS association study. No causal inference or fitting."""
from collections import Counter, defaultdict
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from verify_smp_dates import ROOT, OUT, SOURCE as PREVIOUS, CSV_SOURCE, digest
from smp_embedded_dates import score_lookup

SOURCE = PREVIOUS.with_name('epss_recollected_2026-09-30_smp.json')
OFFSETS = (-30, -14, -7, -1, 0, 1, 7, 14, 30)
HORIZONS = (1, 7, 14, 30)
EPSILON = 1e-6
THRESHOLD = .01


def parse_date(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt


def slope(xs, ys):
    if len(xs) < 2 or any(y is None for y in ys):
        return None
    xm, ym = statistics.mean(xs), statistics.mean(ys)
    return sum((x-xm)*(y-ym) for x, y in zip(xs, ys)) / sum((x-xm)**2 for x in xs)


def trend_class(before, after, epsilon=EPSILON):
    if before is None or after is None:
        return None
    pre, post = before > epsilon, after > epsilon
    return ('RISING_BOTH' if pre and post else 'RISING_BEFORE_ONLY' if pre else
            'RISING_AFTER_ONLY' if post else 'NEITHER_PERIOD_RISING')


def describe(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {'n': 0, 'mean': None, 'median': None, 'min': None, 'max': None}
    return {'n': len(values), 'mean': statistics.mean(values),
            'median': statistics.median(values), 'min': values[0], 'max': values[-1]}


def alignment(record, event_day, last_day):
    publication = date.fromisoformat(record['published_date'])
    scores = record['epss_scores']
    lookup = lambda k: score_lookup(scores, event_day + timedelta(days=k), publication, last_day)
    offsets = {str(k): lookup(k) for k in OFFSETS}
    windows = {}
    change = record.get('epss_version_change_date')
    change = date.fromisoformat(change) if change else None
    for h in HORIZONS:
        before = [lookup(k) for k in range(-h, 0)]
        after = [lookup(k) for k in range(1, h+1)]
        pre_complete = all(v['status'] == 'OBSERVED' for v in before)
        post_complete = all(v['status'] == 'OBSERVED' for v in after)
        all_complete = pre_complete and post_complete
        pre_slope = slope(list(range(-h, 0)), [v['score'] for v in before]) if pre_complete else None
        post_slope = slope(list(range(1, h+1)), [v['score'] for v in after]) if post_complete else None
        a, b, at = offsets['-1']['score'], offsets[str(h)]['score'], offsets['0']['score']
        post_values = [v['score'] for v in after]
        missing = Counter(v['status'] for v in before+after if v['status'] != 'OBSERVED')
        windows[str(h)] = {
            'complete_pre': pre_complete, 'complete_post': post_complete,
            'complete_both': all_complete, 'incomplete_reasons': dict(missing),
            'change_from_previous_day': b-a if a is not None and b is not None else None,
            'change_from_same_day': b-at if at is not None and b is not None else None,
            'pre_slope': pre_slope, 'post_slope': post_slope,
            'slope_difference': post_slope-pre_slope if all_complete and h > 1 else None,
            'trend_class': trend_class(pre_slope, post_slope),
            'post_minimum': min(post_values) if post_complete else None,
            'post_maximum': max(post_values) if post_complete else None,
            'first_day_of_post_maximum': post_values.index(max(post_values))+1 if post_complete else None,
            'crosses_0_01_after': (a < THRESHOLD and max(post_values) >= THRESHOLD)
                if a is not None and post_complete else None,
            # This flag follows saved metadata, not a fresh archive-version audit.
            'recorded_version_change_in_window': bool(change and event_day-timedelta(days=h) < change <= event_day+timedelta(days=h)),
        }
    return {'offsets': offsets, 'windows': windows}


def summarize_cohort(rows, end, horizon):
    complete = [r for r in rows if r['alignment'][str(end)]['windows'][str(horizon)]['complete_both']]
    windows = [r['alignment'][str(end)]['windows'][str(horizon)] for r in complete]
    by_cve = defaultdict(list)
    for r, w in zip(complete, windows):
        if w['change_from_previous_day'] is not None:
            by_cve[r['cve']].append(w['change_from_previous_day'])
    changes = [w['change_from_previous_day'] for w in windows]
    reason_counts = Counter()
    for r in rows:
        w = r['alignment'][str(end)]['windows'][str(horizon)]
        reason_counts.update(w['incomplete_reasons'].keys())
    return {
        'candidate_records': len(rows), 'complete_records': len(complete),
        'distinct_cves': len({r['cve'] for r in complete}),
        'excluded_incomplete': len(rows)-len(complete), 'overlapping_exclusion_reasons': dict(reason_counts),
        'trend_counts': dict(Counter(w['trend_class'] for w in windows if w['trend_class'])),
        'change_from_previous_day': describe(changes),
        'cve_equal_weighted_mean_change': describe([statistics.mean(v) for v in by_cve.values()]),
        'increase': sum(v > 1e-12 for v in changes if v is not None),
        'unchanged': sum(abs(v) <= 1e-12 for v in changes if v is not None),
        'decrease': sum(v < -1e-12 for v in changes if v is not None),
        'recorded_version_changes': sum(w['recorded_version_change_in_window'] for w in windows),
        'threshold_crossings': sum(w['crosses_0_01_after'] is True for w in windows),
        'same_text_multiple_dates': sum(r['same_platform_text_multiple_dates'] for r in complete),
        'other_observed_post_within_window': sum(r['other_post_dates_within'][str(horizon)] > 0 for r in complete),
    }


def analyze():
    records = json.loads(SOURCE.read_text())
    old = {r['cve']: r for r in json.loads(PREVIOUS.read_text())}
    prior = {r['association_id']: r for r in json.loads((OUT/'smp_candidate_event_audit.json').read_text())}
    prior_summary = json.loads((OUT/'smp_date_verification.json').read_text())
    assert digest(PREVIOUS) == prior_summary['source_hashes'][str(PREVIOUS)]
    assert digest(CSV_SOURCE) == prior_summary['source_hashes'][str(CSV_SOURCE)]
    assert len({r['cve'] for r in records}) == len(records)
    assert set(old) == {r['cve'] for r in records}
    audit, raw_counts, platform = [], Counter(), defaultdict(Counter)
    seen, text_dates, text_cves, cve_dates = {}, defaultdict(set), defaultdict(set), defaultdict(set)
    for r in records:
        previous = old[r['cve']]
        assert {k:v for k,v in r.items() if k != 'social_media_posts'} == {k:v for k,v in previous.items() if k != 'social_media_posts'}
        assert [{k:v for k,v in p.items() if k != 'date'} for p in r['social_media_posts']] == previous['social_media_posts']
        for i, p in enumerate(r['social_media_posts']):
            raw_counts['associations'] += 1
            platform[p['platform']]['associations'] += 1
            stamp = parse_date(p.get('date'))
            ident = f"{r['cve']}:smp:{i}"
            prior_row = prior[ident]
            raw_counts['dates_equal_previous_source_replay'] += p.get('date') == prior_row['candidate_timestamp']
            raw_counts['parseable_dates'] += stamp is not None
            raw_counts['explicit_timezone'] += bool(stamp and stamp.tzinfo)
            raw_counts['midnight'] += bool(stamp and stamp.time().isoformat() == '00:00:00')
            platform[p['platform']]['midnight'] += bool(stamp and stamp.time().isoformat() == '00:00:00')
            key = (r['cve'], p['platform'], p['post'], p.get('date'))
            if key in seen:
                seen[key]['duplicate_association_ids'].append(ident)
                raw_counts['exact_duplicate_excess'] += 1
                continue
            row = {'association_id': ident, 'duplicate_association_ids': [], 'cve': r['cve'],
                   'platform': p['platform'], 'post_sha256': hashlib.sha256(p['post'].encode()).hexdigest(),
                   'post_excerpt': p['post'][:350], 'stored_post_date': p.get('date'),
                   'event_date': stamp.date().isoformat() if stamp else None,
                   'timezone_status': 'EXPLICIT_OFFSET_TO_UTC' if stamp and stamp.tzinfo else 'NOT_SUPPLIED',
                   'publication_timestamp_independently_verified': False,
                   'date_provenance': 'SOURCE_CSV_DATE_PLUS_TIME; described as post dates by dataset owner',
                   'historical_text_version_verified': False,
                   'source_csv_row': prior_row['ordinal_source_csv_row'],
                   'published_timestamp': r['published_timestamp'], 'published_date': r['published_date'],
                   'date_matches_previous_source_replay': p.get('date') == prior_row['candidate_timestamp'],
                   'date_in_post_review': prior_row['embedded_date_review'],
                   'date_status': 'PARSEABLE' if stamp else 'INVALID_OR_MISSING', 'alignment': {}}
            seen[key] = row
            audit.append(row)
            if not stamp:
                continue
            day = stamp.date()
            delta = (day-date.fromisoformat(r['published_date'])).days
            row['days_after_nvd_publication'] = delta
            row['relative_to_nvd'] = 'BEFORE' if delta < 0 else 'SAME_DAY' if delta == 0 else 'AFTER'
            row['alignment'] = {str(end): alignment(r, day, end) for end in (60,90)}
            textkey = (p['platform'], row['post_sha256'])
            text_dates[textkey].add(row['stored_post_date'])
            text_cves[textkey].add(r['cve'])
            cve_dates[r['cve']].add(row['event_date'])
    for row in audit:
        textkey = (row['platform'], row['post_sha256'])
        row['same_platform_text_multiple_dates'] = len(text_dates[textkey]) > 1
        row['same_platform_text_shared_across_cves'] = len(text_cves[textkey]) > 1
        if not row['event_date']:
            continue
        day = date.fromisoformat(row['event_date'])
        deltas = [(date.fromisoformat(d)-day).days for d in cve_dates[row['cve']]]
        row['other_post_dates_within'] = {str(h): sum(0 < abs(k) <= h for k in deltas) for h in HORIZONS}
        row['observed_post_days_before'] = {str(h): sum(-h <= k < 0 for k in deltas) for h in HORIZONS}
        row['first_observed_date_for_cve'] = row['event_date'] == min(cve_dates[row['cve']])
    valid = [r for r in audit if r['event_date']]
    first = {}
    for row in sorted(valid, key=lambda r: (r['event_date'], r['stored_post_date'], r['association_id'])):
        first.setdefault(row['cve'], row)
    cohorts = {}
    for end in (60,90):
        cohorts[str(end)] = {str(h): summarize_cohort(valid, end, h) for h in HORIZONS}
    sensitivity = {}
    for end in (60,90):
        sensitivity[str(end)] = {
            'first_observed_entry_per_cve': summarize_cohort(list(first.values()), end, 7),
            'no_text_multiple_dates': summarize_cohort([r for r in valid if not r['same_platform_text_multiple_dates']], end, 7),
            'no_recorded_version_transition': summarize_cohort([r for r in valid if not r['alignment'][str(end)]['windows']['7']['recorded_version_change_in_window']], end, 7),
            'no_other_observed_post_day_within_7': summarize_cohort([r for r in valid if not r['other_post_dates_within']['7']], end, 7),
        }
        complete = [r for r in valid if r['alignment'][str(end)]['windows']['7']['complete_both']]
        sensitivity[str(end)]['trend_counts_slope_threshold_0_00001'] = dict(Counter(trend_class(
            r['alignment'][str(end)]['windows']['7']['pre_slope'],
            r['alignment'][str(end)]['windows']['7']['post_slope'], 1e-5) for r in complete))
    shifts = {}
    raw_by_id = {r['cve']:r for r in records}
    for shift in (-1,1):
        shift_windows = [alignment(raw_by_id[r['cve']], date.fromisoformat(r['event_date'])+timedelta(days=shift),60)['windows']['7'] for r in valid]
        complete = [w for w in shift_windows if w['complete_both']]
        shifts[str(shift)] = {'complete_records':len(complete),
                             'trend_counts':dict(Counter(w['trend_class'] for w in complete)),
                             'change':describe([w['change_from_previous_day'] for w in complete])}
    by_platform = {}
    for name, counts in platform.items():
        rows = [r for r in valid if r['platform'] == name]
        by_platform[name] = dict(counts, distinct_records=len(rows),
            relative_to_nvd=dict(Counter(r['relative_to_nvd'] for r in rows)),
            score_on_post_day_60=sum(r['alignment']['60']['offsets']['0']['status']=='OBSERVED' for r in rows),
            seven_day=summarize_cohort(rows,60,7))
    summary = {
        'study_scope':'Conditional daily alignment using supplied post dates. Descriptive association only; no causality, model fitting or significance test.',
        'date_evidence':'All new dates copied from existing source CSV. Owner identifies these as publication dates; platform event identity and timezone are not independently authenticated by this enrichment.',
        'source':str(SOURCE), 'source_sha256':digest(SOURCE), 'previous_sha256':digest(PREVIOUS),
        'source_csv_sha256':digest(CSV_SOURCE),
        'source_commit':subprocess.check_output(['git','-C',str(SOURCE.parents[2]),'rev-parse','HEAD'],text=True).strip(),
        'analysis_script_sha256':digest(Path(__file__)), 'raw_counts':dict(raw_counts),
        'cves':len(records), 'cves_with_posts':len(cve_dates), 'distinct_records':len(audit),
        'non_smp_values_unchanged':True, 'post_text_platform_and_order_unchanged':True,
        'independently_verified_publication_events':0,
        'text_groups_with_multiple_dates':sum(len(v)>1 for v in text_dates.values()),
        'text_groups_shared_across_cves':sum(len(v)>1 for v in text_cves.values()),
        'relative_to_nvd':dict(Counter(r['relative_to_nvd'] for r in valid)),
        'platforms':by_platform, 'cohorts':cohorts, 'sensitivity':sensitivity,
        'one_day_alignment_sensitivity':shifts, 'rising_slope_threshold':EPSILON,
        'crossing_threshold':THRESHOLD, 'missing_values_imputed':False,
        'offset_coverage':{str(end):{str(k):dict(Counter(r['alignment'][str(end)]['offsets'][str(k)]['status'] for r in valid)) for k in OFFSETS} for end in (60,90)},
    }
    return summary, audit


def validate(summary, audit):
    assert summary['raw_counts']['associations'] == len(audit) + summary['raw_counts']['exact_duplicate_excess']
    assert summary['raw_counts']['dates_equal_previous_source_replay'] == summary['raw_counts']['associations']
    for row in audit:
        for end, aligned in row['alignment'].items():
            for off, item in aligned['offsets'].items():
                assert date.fromisoformat(item['date']) == date.fromisoformat(row['event_date'])+timedelta(days=int(off))
                assert (item['score'] is not None) == (item['status']=='OBSERVED')
            for h, w in aligned['windows'].items():
                assert w['complete_both'] == (w['complete_pre'] and w['complete_post'])
                if not w['complete_both']:
                    assert w['trend_class'] is None
                if not w['complete_post']:
                    assert w['post_minimum'] is None and w['post_maximum'] is None
                if w['post_minimum'] is not None:
                    assert 0 <= w['post_minimum'] <= w['post_maximum'] <= 1
                    assert 1 <= w['first_day_of_post_maximum'] <= int(h)
    return {'validated_distinct_records':len(audit), 'raw_unchanged':True,
            'offset_dates_and_masks_validated':True, 'no_partial_window_extrema_or_trends':True}


def independent_check(audit):
    """Recalculate from raw date keys without the alignment/score helpers."""
    records = {r['cve']:r for r in json.loads(SOURCE.read_text())}
    checked_scores, checked_windows = 0, 0
    for event in audit:
        if event['event_date'] is None:
            assert not event['alignment']
            continue
        raw = records[event['cve']]
        publication = date.fromisoformat(raw['published_date'])
        day = date.fromisoformat(event['event_date'])
        for end in (60,90):
            values = {}
            for k in range(-30,31):
                requested = day+timedelta(days=k)
                values[k] = raw['epss_scores'].get(requested.isoformat()) if 0 <= (requested-publication).days <= end else None
            aligned = event['alignment'][str(end)]
            for k in OFFSETS:
                assert values[k] == aligned['offsets'][str(k)]['score']
                checked_scores += 1
            for h in HORIZONS:
                w = aligned['windows'][str(h)]
                complete = all(values[k] is not None for k in range(-h,h+1) if k)
                assert complete == w['complete_both']
                if values[-1] is not None and values[h] is not None:
                    assert math.isclose(values[h]-values[-1],w['change_from_previous_day'],abs_tol=1e-12)
                if complete and h > 1:
                    pre = statistics.linear_regression(list(range(-h,0)),[values[k] for k in range(-h,0)]).slope
                    post = statistics.linear_regression(list(range(1,h+1)),[values[k] for k in range(1,h+1)]).slope
                    assert math.isclose(pre,w['pre_slope'],abs_tol=1e-12)
                    assert math.isclose(post,w['post_slope'],abs_tol=1e-12)
                checked_windows += 1
    return {'independently_recomputed_score_lookups':checked_scores,
            'independently_recomputed_windows':checked_windows}


def figures(summary, audit):
    directory = OUT/'figures'
    directory.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5,3.7),layout='constrained')
    xs = list(OFFSETS)
    for end, color in ((60,'#216d89'),(90,'#a34f36')):
        counts = [summary['offset_coverage'][str(end)][str(k)].get('OBSERVED',0) for k in xs]
        ax.plot(xs, counts, 'o-', color=color, label=f'Publication to Day {end}')
    ax.set(xlabel='Days before (-) or after (+) the supplied post date; 0 = that date',
           ylabel='CVE/post records with a numeric EPSS score',
           title=f'Available scores around supplied post dates (n={len(audit):,} records)')
    ax.set_xticks(xs)
    ax.grid(alpha=.18)
    ax.legend(fontsize=9)
    fig.savefig(directory/'smp_dated_score_coverage.png',dpi=180)
    plt.close(fig)
    complete = [r for r in audit if r.get('alignment',{}).get('60',{}).get('windows',{}).get('7',{}).get('complete_both')]
    order = ['RISING_BEFORE_ONLY','RISING_AFTER_ONLY','RISING_BOTH','NEITHER_PERIOD_RISING']
    labels = ['Rising before only','Rising after only','Rising before and after','Neither period rising']
    counts = [summary['cohorts']['60']['7']['trend_counts'].get(k,0) for k in order]
    fig, ax = plt.subplots(figsize=(8.5,3.5),layout='constrained')
    bars = ax.barh(labels,counts,color=['#356d8d','#af663c','#51816a','#83858b'])
    ax.bar_label(bars,padding=4)
    ax.set(xlabel='CVE/post records with all seven scores on each side',
           title=f'When scores rise around the supplied date (n={len(complete):,})')
    ax.set_xlim(0,max(counts)*1.15)
    ax.invert_yaxis()
    fig.savefig(directory/'smp_dated_trend_groups.png',dpi=180)
    plt.close(fig)
    examples = []
    for platform in ('reddit','telegram','mastodon'):
        candidates = [r for r in complete if r['platform']==platform and not r['same_platform_text_multiple_dates'] and not r['alignment']['60']['windows']['7']['recorded_version_change_in_window']]
        if candidates:
            examples.append(sorted(candidates,key=lambda r:r['association_id'])[0])
    raw = {r['cve']:r for r in json.loads(SOURCE.read_text())}
    fig, axes = plt.subplots(len(examples),1,figsize=(8.5,6.2),layout='constrained',squeeze=False)
    for ax, row in zip(axes[:,0], examples):
        event = date.fromisoformat(row['event_date'])
        record = raw[row['cve']]
        days = list(range(-7,8))
        vals = [score_lookup(record['epss_scores'],event+timedelta(days=k),date.fromisoformat(record['published_date']),60)['score'] for k in days]
        ax.plot(days,[float('nan') if v is None else v for v in vals],'o-',ms=3,color='#216d89')
        ax.axvline(0,color='#a34f36',ls='--',label='Supplied post date')
        ax.set_title(f"{row['cve']} / {row['platform']} / {row['event_date']}",fontsize=10,loc='left')
        ax.set_ylabel('EPSS score')
        ax.grid(alpha=.18)
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    axes[-1,0].set_xlabel('Days relative to supplied post date (negative = before; positive = after)')
    fig.savefig(directory/'smp_dated_real_examples.png',dpi=180)
    plt.close(fig)
    summary['examples'] = [{k:r[k] for k in ('association_id','cve','platform','stored_post_date','post_excerpt','days_after_nvd_publication','alignment')} for r in examples]


def report(summary):
    s = summary
    r = s['raw_counts']
    c = s['cohorts']['60']['7']
    names = {'hackernews':'Hacker News', 'bleepingcomputer':'Bleeping Computer', 'exploitdb':'Exploit-DB'}
    rows = '\n'.join(f"| {names.get(p,p.title())} | {v['associations']:,} | {v['midnight']:,} | {v['distinct_records']:,} | {v['seven_day']['complete_records']:,} / {v['seven_day']['distinct_cves']:,} |" for p,v in s['platforms'].items())
    coverage = '\n'.join(f"| {h} {'day' if h==1 else 'days'} on each side | {s['cohorts']['60'][str(h)]['complete_records']:,} / {s['cohorts']['60'][str(h)]['distinct_cves']:,} | {s['cohorts']['60'][str(h)]['excluded_incomplete']:,} | {s['cohorts']['90'][str(h)]['complete_records']:,} / {s['cohorts']['90'][str(h)]['distinct_cves']:,} |" for h in HORIZONS)
    trend_labels = {'RISING_BEFORE_ONLY':'Rising before only','RISING_AFTER_ONLY':'Rising after only','RISING_BOTH':'Rising before and after','NEITHER_PERIOD_RISING':'Neither period rising'}
    trends = '\n'.join(f"| {label} | {c['trend_counts'].get(k,0):,} | {100*c['trend_counts'].get(k,0)/c['complete_records']:.1f}% |" for k,label in trend_labels.items())
    changes = '\n'.join(f"| {h} {'day' if h==1 else 'days'} after the supplied post date | {v['complete_records']:,} | {v['increase']:,} | {v['unchanged']:,} | {v['decrease']:,} | {v['change_from_previous_day']['median']:.6f} |" for h in HORIZONS if (v:=s['cohorts']['60'][str(h)])['complete_records'])
    sensitivity_names = {'first_observed_entry_per_cve':'First observed entry per CVE',
                         'no_text_multiple_dates':'Exclude text stored with multiple dates',
                         'no_recorded_version_transition':'Exclude recorded EPSS version changes',
                         'no_other_observed_post_day_within_7':'Exclude other observed post dates within seven days'}
    sensitivities = '\n'.join(f"| {sensitivity_names[name]} | {v['complete_records']:,} / {v['distinct_cves']:,} | {v['change_from_previous_day']['median']:.6f} |" for name,v in s['sensitivity']['60'].items() if 'complete_records' in v)
    stricter = '; '.join(f"{label.lower()}: {s['sensitivity']['60']['trend_counts_slope_threshold_0_00001'].get(k,0):,}" for k,label in trend_labels.items())
    examples = []
    for e in s['examples']:
        off = e['alignment']['60']['offsets']
        examples.append(f"- {e['cve']} ({e['platform']}), supplied date {e['stored_post_date']}: EPSS seven days before = {off['-7']['score']}, one day before = {off['-1']['score']}, on that date = {off['0']['score']}, seven days after = {off['7']['score']}. Text excerpt: {e['post_excerpt'][:170]}...")
    return f'''### 10.6 The New Dataset with Dates Attached to Every Post

Update of 14 October 2026. This extends, rather than replaces, the earlier date checks. The new file contains {s['cves']:,} CVEs and {r['associations']:,} CVE/post entries. Every entry now has a parseable date. All {r['dates_equal_previous_source_replay']:,} dates equal the source-row timestamps already recovered in the previous audit. The post texts, their order, all EPSS scores and other CVE fields are unchanged. Thus this update makes dates directly available in the JSON; it does not recover additional EPSS values or independently establish new post times.

The dataset owner identifies the new field as the post publication date. The enrichment code combines the existing source table's date and time. It does not fetch a platform creation timestamp, original post ID or permalink. We therefore analyze the supplied post dates as a clearly stated assumption. This is not evidence that the dates are scraping times, but the enrichment alone does not prove their exact publication meaning either.

All {r['associations']:,} timestamps omit a timezone; {r['midnight']:,} are midnight. In particular, all Mastodon, Hacker News, BleepingComputer and Exploit-DB entries have midnight times. Treat these as day-level records, not proof that every item appeared at precisely midnight. The comparison below uses the stored day without silently assigning UTC. A one-day shift is reported as a sensitivity check, not as a correction. BleepingComputer articles and Exploit-DB entries are related sources, not necessarily social-media posts.

| Source | CVE/post entries | Midnight times | Distinct CVE/text/date records | Complete 7-day windows: records / CVEs |
|---|---|---|---|---|
{rows}

Table 5. Each source row shows the entries in the new file. Midnight times count entries saved at 00:00. Distinct records remove only identical CVE, source, text and timestamp repeats. The final column counts records with all seven EPSS scores before and all seven after the supplied post date, within publication through Day 60; it also gives the distinct CVEs represented. A record is not necessarily a unique platform post.

Removing {r['exact_duplicate_excess']:,} exact repeat entries leaves {s['distinct_records']:,} records. There are {s['text_groups_with_multiple_dates']:,} source/text groups with more than one timestamp and {s['text_groups_shared_across_cves']:,} shared across CVEs. These can be reposts, repeated collection, digests, or reused text; do not treat every row as an independent event. Among distinct records, {s['relative_to_nvd'].get('BEFORE',0):,} dates are before NVD publication, {s['relative_to_nvd'].get('SAME_DAY',0):,} are on that day and {s['relative_to_nvd'].get('AFTER',0):,} are later. A discussion before NVD publication is possible and is not automatically an erroneous date.

### 10.7 Which Before/After Comparisons Are Available?

Here t is the supplied post date, not the earlier date parsed from text. The primary interval remains NVD publication through Day 60. We also check the full saved Day 0-90 history separately. Scores are matched by date; a missing score, a missing date key and a date outside the selected interval remain separate conditions. None is replaced with zero or interpolated.

| Required scores | Day 0-60: records / CVEs | Records excluded within Day 0-60 | Day 0-90: records / CVEs |
|---|---|---|---|
{coverage}

Table 6. Each row requires every daily score on both sides of t. Seven days means t-7 through t-1 and t+1 through t+7, excluding t itself. The second and fourth columns give usable record counts and distinct CVE counts, not verified causal events. The third column is {s['distinct_records']:,} minus the second column's record count. Different rows use different cohorts, so their results are not interchangeable. Within Day 0-60, 30 days on each side is possible only when t is Day 30.

For the seven-day comparison, {c['excluded_incomplete']:,} records lack a complete primary window. In {c['overlapping_exclusion_reasons'].get('OUTSIDE_SELECTED_WINDOW',0):,} records, at least one requested day falls before publication or after Day 60. In {c['overlapping_exclusion_reasons'].get('NULL_SCORE',0):,}, at least one requested in-window date has no numeric EPSS. These reasons can overlap and must not be added as separate excluded populations.

{{{{figure_dated_coverage}}}}

Figure 5. The horizontal axis shows days relative to the supplied post date: negative values are before it, zero is that date, and positive values are after it. The vertical axis counts distinct CVE/text/date records with a numeric score at each requested day. Blue uses publication through Day 60; orange uses the full saved Day 0-90 interval. Different points can represent different records, so this is a coverage plot, not an average EPSS curve.

### 10.8 Were EPSS Scores Already Rising Before the Post Date?

We fit a straight line separately through the seven observed scores before t and the seven after t. This is a descriptive slope calculation, not a predictive model. We call a period rising when its slope exceeds 0.000001 EPSS units per day. This small, stated threshold is a descriptive convention, not a statistically significant increase. The same-day score is excluded because the saved daily EPSS value might have been produced before or after the post.

| Observed pattern | Complete seven-day records | Share of these records |
|---|---|---|
{trends}

Table 7. All rows use the same {c['complete_records']:,} records across {c['distinct_cves']:,} CVEs. Rising before only means the pre-date slope exceeds the threshold and the post-date slope does not; rising after only means the reverse. Both means both slopes exceed it. Neither means neither does, including flat or decreasing periods. These groups do not establish an effect caused by posting, and after only does not prove that all of the score increase occurred after publication of a post.

{{{{figure_dated_trends}}}}

Figure 6. Each horizontal bar counts one of Table 7's non-overlapping patterns. The horizontal axis is the number of complete CVE/post records; the vertical labels describe when scores rise. Multiple records can belong to the same CVE. The chart shows timing patterns under the supplied-date assumption, not causality or independent sample sizes.

### 10.9 How Much Did EPSS Change After the Supplied Date?

The next table subtracts EPSS on t-1 from EPSS on t+h. We use t-1 as the baseline so the same-day score is not assumed to precede the post. Every row also requires a complete h-day window on both sides, for comparability with the corresponding trend analysis. Increases and decreases use the numerical sign, with a 0.000000000001 tolerance for floating-point equality. Changes are in EPSS score units, not percentages or relative percentage increases.

| Comparison endpoint | Complete records | Higher EPSS | Unchanged EPSS | Lower EPSS | Median change from t-1 |
|---|---|---|---|---|---|
{changes}

Table 8. For example, the seven-day row compares the score one day before the supplied post date with the score seven days after it. Higher, unchanged and lower counts sum to complete records. Median change is the middle signed change, not a causal estimate. A rising slope and a positive endpoint change answer different questions; their counts need not agree.

For seven days, the record-weighted mean change is {c['change_from_previous_day']['mean']:.6f}, compared with {c['cve_equal_weighted_mean_change']['mean']:.6f} when changes are first averaged within each CVE and each CVE then receives equal weight. The median is {c['change_from_previous_day']['median']:.6f}. {c['threshold_crossings']:,} complete records start below 0.01 on t-1 and reach at least 0.01 during t+1 through t+7. That threshold is illustrative, not a risk boundary. Post-window minima, maxima, the first day of the maximum, slopes, and same-day-baseline changes are retained in the derived records; exact extrema are left unavailable when any required day is missing.

### 10.10 Repeated Posts, Version Changes and Date Uncertainty

Of the complete seven-day records, {c['recorded_version_changes']:,} cross an EPSS version-change date recorded in the dataset, {c['same_text_multiple_dates']:,} involve text also stored with another date, and {c['other_observed_post_within_window']:,} have another observed post date for the same CVE within seven days. These groups overlap. Model-version flags follow the saved metadata, not a new verification of all historical archive headers.

| Seven-day sensitivity check | Complete records / CVEs | Median change from t-1 to t+7 |
|---|---|---|
{sensitivities}

Table 9. Each row repeats the calculation with one restriction at a time. First observed keeps one earliest available record per CVE, with deterministic tie-breaking; it does not establish the first-ever post. The second row excludes records whose source/text appears at several times. The third excludes a stored EPSS version change within the window. The fourth excludes additional recorded dates within seven days, but cannot prove there was no other online discussion. These subsets are not matched experimental controls.

Moving every supplied date one day earlier leaves {s['one_day_alignment_sensitivity']['-1']['complete_records']:,} complete seven-day records; moving it one day later leaves {s['one_day_alignment_sensitivity']['1']['complete_records']:,}. This checks sensitivity to day assignment and unknown timezones. It does not identify the correct timezone. With a stricter rising threshold of 0.00001 per day, the pattern counts are {stricter}. Conclusions about tiny slopes depend on this definition.

The derived records also count distinct observed post days in the previous 7, 14 and 30 days. They measure the discussion captured in this collection, not total platform activity. Same-day repeats and an incomplete collection must not be interpreted as exact posting intensity. No monitoring record proves that a day with no saved post was truly a day without discussion.

### 10.11 Real Examples Using the New Date Field

{chr(10).join(examples)}

The examples use the lexicographically first eligible association for Reddit, Telegram and Mastodon, with a complete seven-day primary window, no text mapped to multiple dates, and no recorded version transition in that window. This fixed selection rule avoids choosing only visually dramatic increases.

{{{{figure_dated_examples}}}}

Figure 7. Each panel is a real CVE from the new dated file. The horizontal axis is days relative to its supplied post date, unlike Figure 4's publication-relative axis. The dashed line marks that date. The vertical axis is actual daily EPSS, with a separate scale for each CVE. Gaps remain missing. These are observed scores, not forecasts; the line at zero does not establish when the daily EPSS score became available or that the post caused a change.

### 10.12 What This Update Resolves and What It Does Not

The date field removes the need to reconstruct the source-table mapping for each post. It makes the conditional before/after table reproducible. It does not change the available EPSS history or resolve platform identity, timezone, edited-text timing, repeated collection, or causal attribution. The analysis can now describe score changes around supplied dates, with the limitations above. No causal impact estimate or training result is claimed.

For the existing forecasting design, inputs still stop at Day 30. A later post cannot be added to that input. For a future SMP-centered experiment, use a verified event date, EPSS available before that event, and text known to exist then; evaluate future scores on exactly the same CVEs and windows with and without SMP information. Start with seven days, test 14 days next, and use 30 days only with adequate history. Prefer the full Day 0-90 saved interval for a separate event study, without silently changing the existing Day-60 forecast task. Platform-event verification and historical text checks remain necessary before a publication-timed predictive claim. EPSS-only experiments still come first; no model was trained here.
'''


def main():
    before = digest(SOURCE)
    summary, audit = analyze()
    checks = validate(summary,audit)
    checks.update(independent_check(audit))
    figures(summary,audit)
    checks['source_hash_unchanged_after_analysis'] = digest(SOURCE) == before
    assert checks['source_hash_unchanged_after_analysis']
    summary['checks'] = checks
    for suffix in ('json','txt'):
        (OUT/f'smp_dated_epss_analysis.{suffix}').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (OUT/'smp_dated_epss_events.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    with (OUT/'smp_dated_epss_events.csv').open('w',newline='') as handle:
        flat = []
        for r in audit:
            item = {k:r.get(k) for k in (
                'association_id','cve','platform','stored_post_date','event_date',
                'timezone_status','date_provenance','date_status','source_csv_row',
                'publication_timestamp_independently_verified','historical_text_version_verified',
                'days_after_nvd_publication','relative_to_nvd','same_platform_text_multiple_dates',
                'same_platform_text_shared_across_cves','first_observed_date_for_cve')}
            for end, values in r['alignment'].items():
                for k,v in values['offsets'].items():
                    item[f'day0_{end}_epss_t{k}'] = v['score']
                for h, w in values['windows'].items():
                    for k,v in w.items():
                        item[f'day0_{end}_h{h}_{k}'] = json.dumps(v,sort_keys=True) if isinstance(v,dict) else v
            flat.append(item)
        writer = csv.DictWriter(handle,fieldnames=list(dict.fromkeys(k for r in flat for k in r)))
        writer.writeheader()
        writer.writerows(flat)
    text = report(summary)
    for suffix in ('md','txt'):
        (OUT/f'SMP_DATED_EPSS_ANALYSIS.{suffix}').write_text(text)
    print(json.dumps({k:summary[k] for k in ('raw_counts','cohorts','checks')},indent=2))


if __name__ == '__main__':
    main()
