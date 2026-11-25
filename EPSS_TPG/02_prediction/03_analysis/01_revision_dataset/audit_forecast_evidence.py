"""Audit validation errors and dated evidence; no fitting, inference or test analysis."""
import argparse
import csv
import hashlib
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
PACKAGE = PROJECT / '02_prediction/01_package'
sys.path.insert(0, str(PACKAGE))
from epss.forecast_data import load_prepared, read_json, require, sha256, write_json

ROOT = PROJECT / '05_results/02_evaluation/epss_day30_to_day60'
PREPARED = PROJECT / '04_data/01_records_and_graphs/epss_day30_to_day60'
SCHEMA = 'forecast_evidence_audit_v1'
THRESHOLD = Decimal('0.01')
RELATIONS = ('before_anchor', 'on_anchor_date', 'after_anchor', 'missing_date', 'invalid_date')


def date_evidence(value, anchor):
    """Compare stored dates only; never assign a timezone to a naive timestamp."""
    result = {'stored_timestamp': value, 'stored_date': None, 'timezone': None,
              'resolution': None, 'relative_to_anchor': 'missing_date'}
    if value is None or value == '':
        return result
    try:
        if not isinstance(value, str):
            raise ValueError('Not text')
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
            day = date.fromisoformat(value)
            result.update(timezone='not_supplied', resolution='date_only')
        elif re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?', value):
            stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
            day = stamp.date()
            result.update(timezone='explicit_offset' if stamp.tzinfo else 'not_supplied',
                          resolution='midnight_time_semantics_unverified' if
                          (stamp.hour, stamp.minute, stamp.second, stamp.microsecond) == (0, 0, 0, 0)
                          else 'time_recorded_accuracy_unverified')
        else:
            raise ValueError('Unsupported date representation')
        result['stored_date'] = day.isoformat()
        result['relative_to_anchor'] = ('before_anchor' if day < anchor else
                                        'after_anchor' if day > anchor else 'on_anchor_date')
    except (ValueError, OverflowError):
        result['relative_to_anchor'] = 'invalid_date'
    return result


def decimal_difference(a, b):
    return float(Decimal(str(a)) - Decimal(str(b)))


def endpoint_group(actual, base):
    distance = abs(Decimal(str(actual)) - Decimal(str(base)))
    return ('unchanged_endpoint' if distance == 0 else
            'large_endpoint_change' if distance >= THRESHOLD else 'smaller_endpoint_change')


def text_info(value):
    require(value is None or isinstance(value, str), 'Unexpected text field type')
    text = value or ''
    return {'text_present': bool(text.strip()), 'text_characters': len(text),
            'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
            'verbatim_excerpt': text[:1000], 'excerpt_truncated': len(text) > 1000}


def list_field(record, field):
    value = record.get(field, [])
    require(isinstance(value, list), f'Expected a list: {record["cve"]} {field}')
    return value


def collect_evidence(record, anchor):
    """These are audit records, never a predictor whitelist or historical snapshots."""
    rows = []

    def add(kind, path, value=None, timestamp=None, **extra):
        rows.append({'cve': record['cve'], 'kind': kind, 'source_field': path,
                     **date_evidence(timestamp, anchor), **text_info(value),
                     'historical_text_verified': False, **extra})

    for field in ('description', 'orig_description'):
        add('description', field, record.get(field), record.get('last_modified_timestamp'),
            date_meaning='NVD record update, not a timestamp for this description version')
    for field in sorted(record):
        if field.startswith('summ_') or field == 'llm_summary':
            add('generated_summary', field, record[field],
                date_meaning='No summary-generation or source-content snapshot timestamp verified')

    for i, post in enumerate(list_field(record, 'social_media_posts')):
        require(isinstance(post, dict), 'Post must be an object')
        add('post', f'social_media_posts[{i}]', post.get('post'), post.get('date'),
            platform=post.get('platform'),
            date_meaning='Stored post/source date; not independently authenticated publication time')

    urls = defaultdict(list)
    for field in ('all_sources_urls', 'github_only_urls', 'non_github_urls'):
        for url in list_field(record, field):
            require(isinstance(url, str), 'URL must be text')
            if field not in urls[url]:
                urls[url].append(field)
    history = defaultdict(list)
    for i, item in enumerate(list_field(record, 'url_change_dates')):
        require(isinstance(item, dict) and isinstance(item.get('url'), str), 'Invalid reference history')
        history[item['url']].append((i, item))
    for url in sorted(set(urls) | set(history)):
        for i, item in history.get(url, [(None, {})]):
            path = f'url_change_dates[{i}]' if i is not None else ','.join(urls[url])
            add('reference', path, url, item.get('earliest_added_date'), url=url,
                url_list_fields=urls[url], history_match_mode=item.get('match_mode'),
                date_meaning='Earliest recorded addition of reference to NVD; not page publication or page content date')
    return rows


def make_case(record, inputs, labels, prediction):
    anchor = date.fromisoformat(inputs['input_end_date'])
    values, dates = inputs['X']['values'], inputs['X']['dates']
    latest = max(i for i, value in enumerate(values) if value is not None)
    base, actual = values[latest], labels['y']['endpoint']
    require(prediction['cve'] == record['cve'] == inputs['cve'] == labels['cve'], 'CVE mismatch')
    require(base == prediction['persistence_prediction'] and actual == prediction['actual_day60_epss'],
            'Saved run differs from prepared scores')
    future = labels['y']['trajectory']
    scores = values + future
    all_dates = dates + labels['future_dates']
    require(len(values) == 31 and len(future) == 30 and len(all_dates) == 61, 'Wrong day window')
    publication = date.fromisoformat(inputs['publication_date'])
    require(all_dates == [(publication + timedelta(days=i)).isoformat() for i in range(61)],
            'Nonconsecutive score dates')
    changes = [{'day': i, 'date': all_dates[i], 'change': decimal_difference(scores[i], scores[i-1])}
               for i in range(31, 61) if scores[i] is not None and scores[i-1] is not None]
    first = next((i for i in range(31, 61) if scores[i] is not None and scores[i] != base), None)
    evidence = collect_evidence(record, anchor)
    row = {'cve': record['cve'], 'publication_date': inputs['publication_date'],
           'anchor_date': anchor.isoformat(), 'target_date': labels['target_date'],
           'latest_input_score': base, 'latest_input_score_date': dates[latest],
           'latest_input_age_days': 30-latest, 'exact_day30_score': values[30], 'day60_score': actual,
           'change_since_latest_input': decimal_difference(actual, base),
           'change_since_exact_day30': decimal_difference(actual, values[30]) if values[30] is not None else None,
           'absolute_persistence_error': abs(decimal_difference(actual, base)),
           'endpoint_group': endpoint_group(actual, base),
           'future_observed_days': sum(v is not None for v in future),
           'first_observed_future_difference_date': all_dates[first] if first is not None else None,
           'largest_adjacent_future_change': max(changes, key=lambda c: abs(c['change'])) if changes else None,
           'epss_series': [{'day': i, 'date': day, 'score': score,
                            'collection_status': record.get('epss_score_status', {}).get(day, 'not_recorded')}
                           for i, (day, score) in enumerate(zip(all_dates, scores))],
           'epss_version_metadata_posthoc_only': {k: record.get(k) for k in
              ('epss_version_changed', 'epss_from_to_version', 'epss_version_change_day', 'epss_version_change_date')},
           'nvd_update_date_evidence': date_evidence(record.get('last_modified_timestamp'), anchor),
           'cvss_historical_version_verified': False,
           'evidence_counts': dict(Counter(e['kind'] for e in evidence))}
    return row, evidence


def annotate_repeated_posts(evidence):
    by_text = defaultdict(list)
    exact = Counter()
    for row in evidence:
        if row['kind'] != 'post' or not row['text_present']:
            continue
        key = (row['platform'], row['text_sha256'])
        by_text[key].append(row)
        exact[(row['cve'], *key, row['stored_timestamp'])] += 1
    for rows in by_text.values():
        timestamps = {row['stored_timestamp'] for row in rows if row['stored_timestamp']}
        for row in rows:
            row['same_text_distinct_stored_timestamps'] = len(timestamps)
            row['same_text_cves_in_validation'] = len({item['cve'] for item in rows})
            row['identical_associations_in_cve'] = exact[(row['cve'], row['platform'], row['text_sha256'], row['stored_timestamp'])]


def summarize(cases, evidence):
    groups = {r['cve']: r['endpoint_group'] for r in cases}
    coverage = []
    for group in ('all_validation', 'unchanged_endpoint', 'smaller_endpoint_change', 'large_endpoint_change'):
        ids = {c for c, g in groups.items() if group == 'all_validation' or g == group}
        for kind in ('description', 'generated_summary', 'post', 'reference'):
            rows = [r for r in evidence if r['cve'] in ids and r['kind'] == kind]
            coverage.append({'endpoint_group': group, 'cves_in_group': len(ids), 'evidence_kind': kind,
                'records': len(rows), 'cves_with_nonempty_text': len({r['cve'] for r in rows if r['text_present']}),
                **{f'records_{relation}': sum(r['relative_to_anchor'] == relation for r in rows) for relation in RELATIONS},
                'nonempty_cves_with_stored_date_before_anchor': len({r['cve'] for r in rows
                    if r['text_present'] and r['relative_to_anchor'] == 'before_anchor'}),
                'independently_verified_historical_texts': 0})
    posts = [r for r in evidence if r['kind'] == 'post']
    platforms = []
    for platform in sorted({r.get('platform') or 'not_supplied' for r in posts}):
        rows = [r for r in posts if (r.get('platform') or 'not_supplied') == platform]
        platforms.append({'platform': platform, 'post_associations': len(rows),
            'cves': len({r['cve'] for r in rows}),
            **{relation: sum(r['relative_to_anchor'] == relation for r in rows) for relation in RELATIONS},
            'timezone_not_supplied': sum(r['timezone'] == 'not_supplied' for r in rows),
            'text_has_multiple_stored_timestamps': sum(r.get('same_text_distinct_stored_timestamps', 0) > 1 for r in rows),
            'associations_in_exact_repeat_sets': sum(r.get('identical_associations_in_cve', 0) > 1 for r in rows)})
    return {'schema': SCHEMA, 'validation_cves': len(cases), 'training_performed': False,
            'test_predictions_or_error_analysis_performed': False,
            'large_change_threshold_absolute_epss': float(THRESHOLD),
            'group_counts': dict(Counter(r['endpoint_group'] for r in cases)),
            'largest_change_cves': [r['cve'] for r in sorted(cases, key=lambda r: (-r['absolute_persistence_error'], r['cve']))[:10]],
            'coverage': coverage,
            'post_platforms': platforms,
            'post_date_caution': 'Stored dates only. Same-day ordering, timezone and original publication are not established.',
            'predictor_authorization': 'None. Audit output includes future scores and must not be used as model input.',
            'selection_caution': 'Validation outcomes define diagnostic groups after the run; this is not a predictive test.'}


def preflight(prepared, run):
    data = load_prepared(prepared)
    lock = read_json(run / 'evaluation_lock.json')
    require(lock['schema'] == 'epss_gru_residual_v1' and lock['test_performance_evaluated'] is False,
            'Expected validation-only residual run')
    require(lock['preparation_lock_sha256'] == sha256(prepared / 'preparation_lock.json'), 'Preparation changed')
    require({'predictions.json', 'summary.json'} <= set(lock['files']), 'Missing run artifacts')
    for name, digest in lock['files'].items():
        require(Path(name).name == name and sha256(run / name) == digest, f'Changed run artifact: {name}')
    saved = read_json(run / 'summary.json')
    source = data['preparation_summary.json']
    require(saved['provenance']['source_sha256'] == source['source_sha256'], 'Run source mismatch')
    raw = read_json(source['source_path'])
    require(sha256(source['source_path']) == source['source_sha256'], 'Source changed during audit')
    require(len(raw) == len({r['cve'] for r in raw}), 'Duplicate source CVEs')
    predictions = read_json(run / 'predictions.json')
    ids = data['task_cohorts.json']['endpoint']['validation']
    require(len(predictions) == len(ids) and {r['cve'] for r in predictions} == set(ids), 'Run cohort mismatch')
    records = {r['cve']: r for r in raw if r['cve'] in set(ids)}
    require(set(records) == set(ids), 'Missing validation source CVEs')
    return data, records, {r['cve']: r for r in predictions}


def report(summary, cases, evidence):
    lines = ['# Day-30 Evidence Availability Audit', '',
        f"Scope: all {len(cases)} endpoint-eligible validation CVEs, not the reserved test set.",
        'Day 30 is 30 days after NVD publication; Day 60 is 30 days later.',
        'No training, model inference, online verification or source changes were performed.', '',
        '## What the Groups Mean', '',
        'The comparison is Day-60 EPSS minus the latest observed EPSS on or before Day 30.',
        'Large means an absolute difference of at least 0.01 (one percentage point).',
        'Smaller means a nonzero difference below 0.01. Unchanged means equal endpoints,',
        'not necessarily an unchanged curve between them. Missing Day-30 scores are not filled.',
        'These groups use future outcomes for diagnosis, never as input features.', '',
        '| Validation group | CVEs |', '|---|---:|']
    lines += [f'| {k} | {v} |' for k, v in summary['group_counts'].items()]
    lines += ['', '## Largest Endpoint Changes', '',
        '| CVE | Last input EPSS | Age of input at Day 30 (days) | Day-60 EPSS | Change in EPSS (0-1 scale) |',
        '|---|---:|---:|---:|---:|']
    index = {c['cve']: c for c in cases}
    for cve in summary['largest_change_cves']:
        r = index[cve]
        lines.append(f"| {cve} | {r['latest_input_score']:.5f} | {r['latest_input_age_days']} | {r['day60_score']:.5f} | {r['change_since_latest_input']:+.5f} |")
    lines += ['', '## Evidence Dates: What Can Be Concluded', '',
        '| Evidence | Records | CVEs with text | Records dated before Day 30 | On Day 30 | After Day 30 | Missing or invalid date |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary['coverage']:
        if r['endpoint_group'] == 'all_validation':
            lines.append(f"| {r['evidence_kind']} | {r['records']} | {r['cves_with_nonempty_text']} | {r['records_before_anchor']} | {r['records_on_anchor_date']} | {r['records_after_anchor']} | {r['records_missing_date']+r['records_invalid_date']} |")
    lines += ['',
        'Each evidence row is a field, post association, or URL/history entry. Counts are',
        'not independent CVEs; duplicate posts and repeated history entries are retained and flagged.',
        'For descriptions, the date is the NVD record update, not the description publication date.',
        'Even an update before Day 30 does not authenticate a historical description version.',
        'For references, the date means when the link was recorded in NVD, not when the page',
        'was published or when its present content became available. Match modes are retained.',
        'For posts, the date is the stored source claim. Some sources are articles or exploit',
        'database entries, not social posts. Timezones are not invented for undated offsets.',
        'Date comparison uses the date as written. An explicit offset is retained but not',
        'converted to an assumed anchor timezone. On-Day-30 records need an intraday policy.',
        'Generated summaries have no verified generation or underlying source snapshot date.',
        'No evidence record is certified as historically available by this local audit.', '',
        '## Posts by Platform', '',
        '| Platform | CVE/post associations | Unique CVEs | Before Day 30 | On Day 30 | After Day 30 | Missing or invalid date | No timezone supplied |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['post_platforms']:
        lines.append(f"| {r['platform']} | {r['post_associations']} | {r['cves']} | {r['before_anchor']} | {r['on_anchor_date']} | {r['after_anchor']} | {r['missing_date']+r['invalid_date']} | {r['timezone_not_supplied']} |")
    lines += ['',
        'A CVE can appear under several platforms. Post association counts keep repeated records;',
        'the JSON flags exact repeats and the same text paired with different dates within validation.',
        'Shared text on different CVEs is not independent evidence. No missing timezone is assumed UTC.', '',
        '## Real Examples to Review', '']
    for cve in summary['largest_change_cves'][:3]:
        r = index[cve]
        lines += [f'### {cve}', '', f"Prediction anchor: {r['anchor_date']}; target: {r['target_date']}.",
                  f"First observed later score different from the last input: {r['first_observed_future_difference_date']}.",
                  'This is the first observed difference, not proof of the first real change.', '']
        for kind in ('description', 'reference', 'post', 'generated_summary'):
            candidates = [e for e in evidence if e['cve'] == cve and e['kind'] == kind and e['text_present']]
            chosen = sorted(candidates, key=lambda e: (e['stored_date'] or '9999', e['source_field']))[:1]
            if not chosen:
                lines.append(f'- {kind}: no nonempty example.')
            for e in chosen:
                lines += [f"- {kind}, field `{e['source_field']}`: stored date {e['stored_timestamp'] or 'not supplied'}; {e['relative_to_anchor']}.",
                          f"  Dataset excerpt (not a verified factual claim): {e['verbatim_excerpt'][:450]!r}"]
        lines.append('')
    lines += ['## Figures and Tables', '',
        '`largest_changes.png`: each panel is one validation CVE. X is days since NVD',
        'publication; Y is EPSS on its 0-1 scale. The dashed line is Day 30. Gaps stay',
        'missing. The dotted baseline is the latest input score. These are observed curves,',
        'not evidence that any post or advisory caused a score change.',
        '`evidence_dates.png`: X identifies evidence type; Y counts evidence records.',
        'Colored sections show stored dates relative to Day 30, not verified availability.',
        '`cve_cases.csv` has one row per validation CVE; the JSON also holds all daily values.',
        '`evidence_records.csv/json` retains source paths, URLs, excerpts, date caveats and',
        'post repetition flags. Excerpts are verbatim and truncated; hashes identify full text.',
        '`coverage.csv` separates all validation CVEs and the three diagnostic groups.', '',
        '## Decision Before Adding TPG or SMP Inputs', '',
        'Obtain historical description versions, authenticate relevant post timestamps, and',
        'recover time-stamped source content before using it in a historical TPG experiment.',
        'Regenerate summaries only from content demonstrably available by the anchor.',
        'Current CVSS and the original-description field are not automatically historical.',
        'Do not use VulnCheck/CISA enrichment, future EPSS, or these audit groups as predictors.',
        'Full-window EPSS version metadata is post-hoc context only. Missing scores remain unknown.',
        'Any future text comparison needs the same eligible CVEs, dates and a matched persistence',
        'baseline, plus EPSS-only and text-only ablations. Do not change the split after viewing errors.',
        'Validation has already informed model development. These findings do not establish',
        'predictive benefit, causality or final test performance.']
    return '\n'.join(lines) + '\n'


def write_table(path, rows):
    keys = sorted(set().union(*(row.keys() for row in rows)))
    import json
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=True, allow_nan=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def plots(output, cases, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = sorted(cases, key=lambda r: (-r['absolute_persistence_error'], r['cve']))[:3]
    fig, axes = plt.subplots(len(selected), 1, figsize=(9, 3.3*len(selected)), squeeze=False, constrained_layout=True)
    for ax, row in zip(axes[:, 0], selected):
        ax.plot([p['day'] for p in row['epss_series']],
                [p['score'] if p['score'] is not None else math.nan for p in row['epss_series']], color='#216e75')
        ax.axvline(30, linestyle='--', color='#ac4b36', label='Prediction anchor: Day 30')
        ax.axhline(row['latest_input_score'], linestyle=':', color='#555555', label='Last input EPSS')
        ax.set(title=row['cve'], xlabel='Days since NVD publication', ylabel='Observed EPSS (0-1)', ylim=(-.02, 1.02))
        ax.legend(fontsize=8)
    fig.suptitle('Largest validation endpoint changes (selection uses future outcomes)')
    fig.savefig(output/'largest_changes.png', dpi=160)
    plt.close(fig)
    rows = [r for r in summary['coverage'] if r['endpoint_group'] == 'all_validation']
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    bottom = [0] * len(rows)
    for relation, color in zip(RELATIONS, ('#247b70', '#b89130', '#ba5944', '#9a9a9a', '#6c5893')):
        counts = [r['records_'+relation] for r in rows]
        ax.bar([r['evidence_kind'].replace('_', ' ') for r in rows], counts, bottom=bottom,
               label=relation.replace('_', ' '), color=color)
        bottom = [a+b for a, b in zip(bottom, counts)]
    ax.set(title=f"Stored evidence dates for {len(cases)} validation CVEs (not proof of availability)",
           xlabel='Evidence type; descriptions use NVD update dates', ylabel='Evidence records (not unique CVEs)')
    ax.legend(fontsize=8)
    fig.savefig(output/'evidence_dates.png', dpi=160)
    plt.close(fig)


def check_destination(output, protected):
    output = output.resolve()
    for path in protected:
        path = path.resolve()
        require(output != path and path not in output.parents and output not in path.parents,
                f'Output overlaps protected source/run/preparation: {path}')
    require(not output.exists(), 'Output exists; choose a new --output-dir to preserve earlier findings')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, default=PREPARED)
    parser.add_argument('--run-dir', type=Path, default=ROOT/'gru_residual_user_run')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'evidence_availability_audit')
    parser.add_argument('--check-only', action='store_true', help='Validate and construct audit records in memory; do not write reports')
    args = parser.parse_args()
    data, raw, predictions = preflight(args.prepared_dir, args.run_dir)
    cases, evidence = [], []
    for cve in sorted(raw):
        case, records = make_case(raw[cve], data['inputs.json'][cve], data['labels.json'][cve], predictions[cve])
        cases.append(case)
        evidence.extend(records)
    annotate_repeated_posts(evidence)
    summary = summarize(cases, evidence)
    summary['provenance'] = {'source_sha256': data['preparation_summary.json']['source_sha256'],
        'preparation_lock_sha256': sha256(args.prepared_dir/'preparation_lock.json'),
        'run_lock_sha256': sha256(args.run_dir/'evaluation_lock.json'), 'script_sha256': sha256(__file__),
        'adapter_sha256': sha256(PACKAGE/'epss/forecast_data.py')}
    if args.check_only:
        print(f'Validated {len(cases)} validation CVEs and {len(evidence)} evidence records. No files written; no training or test analysis.')
        return
    check_destination(args.output_dir, [args.prepared_dir, args.run_dir,
                                       Path(data['preparation_summary.json']['source_path']).parent])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir/'cve_cases.json', cases)
    write_json(args.output_dir/'evidence_records.json', evidence)
    write_json(args.output_dir/'analysis.json', summary)
    write_table(args.output_dir/'cve_cases.csv', [{k: v for k, v in r.items() if k != 'epss_series'} for r in cases])
    write_table(args.output_dir/'evidence_records.csv', evidence)
    write_table(args.output_dir/'coverage.csv', summary['coverage'])
    write_table(args.output_dir/'post_platforms.csv', summary['post_platforms'])
    narrative = report(summary, cases, evidence)
    for name in ('README.md', 'analysis.txt'):
        (args.output_dir/name).write_text(narrative, encoding='utf-8')
    plots(args.output_dir, cases, summary)
    write_json(args.output_dir/'audit_lock.json', {'schema': SCHEMA, 'provenance': summary['provenance'],
        'files': {p.name: sha256(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}})
    print(f'Audit saved to {args.output_dir}. No training or test analysis performed.')


if __name__ == '__main__':
    main()
