"""Reproduce a completed evidence audit and explain its findings without training."""
import argparse
import csv
import json
from collections import Counter
from datetime import date
from decimal import Decimal
from pathlib import Path

import audit_forecast_evidence as audit
from epss.forecast_data import read_json, require, sha256, write_json

AUDIT_FILES = {'README.md', 'analysis.json', 'analysis.txt', 'coverage.csv', 'cve_cases.csv',
               'cve_cases.json', 'evidence_dates.png', 'evidence_records.csv', 'evidence_records.json',
               'largest_changes.png', 'post_platforms.csv'}
# The human interpretation below describes this completed cohort, not arbitrary audits.
INTERPRETED_PREPARATION = '1520102f56cacbd67ad68b44b875eba77f350f08b2c1bef900fd62d3f45bf4da'


def check_csv(path, expected):
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    keys = sorted(set().union(*(r.keys() for r in expected)))
    normalized = [{key: json.dumps(r[key], ensure_ascii=True, allow_nan=False)
                   if isinstance(r.get(key), (list, dict)) else
                   '' if r.get(key) is None else str(r[key]) for key in keys} for r in expected]
    require(rows == normalized, f'CSV/JSON differs: {path.name}')


def findings(cases, evidence):
    large = [r for r in cases if r['endpoint_group'] == 'large_endpoint_change']
    errors = {r['cve']: abs(Decimal(str(r['day60_score'])) - Decimal(str(r['latest_input_score']))) for r in cases}
    total = sum(errors.values(), Decimal(0))
    large_error = sum((errors[r['cve']] for r in large), Decimal(0))
    details = []
    for row in sorted(large, key=lambda r: (-errors[r['cve']], r['cve'])):
        posts = [e for e in evidence if e['cve'] == row['cve'] and e['kind'] == 'post']
        details.append({k: row[k] for k in ('cve', 'anchor_date', 'target_date', 'latest_input_score',
            'day60_score', 'change_since_latest_input', 'largest_adjacent_future_change',
            'epss_version_metadata_posthoc_only')})
        details[-1]['posts'] = [{k: e[k] for k in ('platform', 'stored_timestamp', 'source_field')} for e in posts]
        details[-1]['posts_after_target'] = sum(e['stored_date'] is not None and
            date.fromisoformat(e['stored_date']) > date.fromisoformat(row['target_date']) for e in posts)
    posts = [r for r in evidence if r['kind'] == 'post']
    summaries = [r for r in evidence if r['kind'] == 'generated_summary']
    return {'validation_cves': len(cases), 'large_change_cves': len(large),
        'large_changes_share_of_total_absolute_error_percent': float(100*large_error/total) if total else None,
        'persistence_mae': float(total/len(cases)) if cases else None,
        'other_cves_mae': float((total-large_error)/(len(cases)-len(large))) if len(cases)>len(large) else None,
        'large_change_details': details,
        'nonempty_summaries': sum(r['text_present'] for r in summaries),
        'summary_fields': len(summaries), 'cves_with_nonempty_summary': len({r['cve'] for r in summaries if r['text_present']}),
        'post_associations': len(posts),
        'distinct_exact_post_associations': len({(r['cve'], r['platform'], r['text_sha256'], r['stored_timestamp']) for r in posts}),
        'score_status_nonmissing_days0_60': dict(Counter(p['collection_status'] for r in cases for p in r['epss_series'] if p['score'] is not None)),
        'score_status_input_days0_30': dict(Counter(p['collection_status'] for r in cases for p in r['epss_series'][:31] if p['score'] is not None)),
        'score_status_target_day60': dict(Counter(r['epss_series'][-1]['collection_status'] for r in cases)),
        'cvss_or_text_historically_authenticated_by_this_review': False}


def review(directory, prepared, run):
    lock = read_json(directory/'audit_lock.json')
    require(lock['schema'] == audit.SCHEMA and set(lock['files']) == AUDIT_FILES, 'Wrong or incomplete audit lock')
    for name, digest in lock['files'].items():
        require(sha256(directory/name) == digest, f'Changed audit file: {name}')
    provenance = lock['provenance']
    require(provenance['script_sha256'] == sha256(audit.__file__), 'Audit implementation changed')
    require(provenance['adapter_sha256'] == sha256(audit.PACKAGE/'epss/forecast_data.py'), 'Adapter changed')
    require(provenance['preparation_lock_sha256'] == sha256(prepared/'preparation_lock.json'), 'Preparation changed')
    require(provenance['run_lock_sha256'] == sha256(run/'evaluation_lock.json'), 'Reference run changed')
    data, raw, predictions = audit.preflight(prepared, run)
    require(provenance['source_sha256'] == data['preparation_summary.json']['source_sha256'], 'Source differs')
    cases, evidence = [], []
    for cve in sorted(raw):
        row, entries = audit.make_case(raw[cve], data['inputs.json'][cve], data['labels.json'][cve], predictions[cve])
        cases.append(row)
        evidence.extend(entries)
    audit.annotate_repeated_posts(evidence)
    summary = audit.summarize(cases, evidence)
    summary['provenance'] = provenance
    for name, rows in [('cve_cases.json', cases), ('evidence_records.json', evidence), ('analysis.json', summary)]:
        require(read_json(directory/name) == rows, f'Reproduced output differs: {name}')
    check_csv(directory/'cve_cases.csv', [{k: v for k, v in r.items() if k != 'epss_series'} for r in cases])
    check_csv(directory/'evidence_records.csv', evidence)
    check_csv(directory/'coverage.csv', summary['coverage'])
    check_csv(directory/'post_platforms.csv', summary['post_platforms'])
    for name in ('README.md', 'analysis.txt'):
        require((directory/name).read_text(encoding='utf-8') == audit.report(summary, cases, evidence), 'Report changed')
    # Independently check endpoint groups against raw date-keyed values.
    for row in cases:
        raw_scores = raw[row['cve']]['epss_scores']
        change = Decimal(str(raw_scores[row['target_date']])) - Decimal(str(raw_scores[row['latest_input_score_date']]))
        require(float(change) == row['change_since_latest_input'], 'Raw delta mismatch')
        group = 'unchanged_endpoint' if change == 0 else 'large_endpoint_change' if abs(change) >= Decimal('.01') else 'smaller_endpoint_change'
        require(row['endpoint_group'] == group, 'Raw group mismatch')
    return {'schema': 'forecast_evidence_review_v1', 'checks_passed': True,
            'audit_directory': str(directory.resolve()), 'audit_lock_sha256': sha256(directory/'audit_lock.json'),
            'review_script_sha256': sha256(__file__), 'provenance': provenance,
            'training_performed': False, 'test_error_analysis_performed': False,
            'group_counts': summary['group_counts'], 'coverage': summary['coverage'],
            'findings': findings(cases, evidence)}


def explanation(result):
    require(result['provenance']['preparation_lock_sha256'] == INTERPRETED_PREPARATION,
            'This interpretation is specific to the completed Day-30/Day-60 cohort; review new cohorts separately')
    f = result['findings']
    lines = ['# Completed Evidence Audit: Findings and Next Decision', '',
        'The source/preparation/run checksums, audit output checksums, all JSON rows,',
        'CSV tables and generated narrative were checked. Endpoint differences and',
        'groups were also recalculated directly from the raw date-keyed scores.',
        'This establishes reproducibility, not external accuracy of the stored scores.',
        'The original audit is unchanged. No training or test error analysis was performed.', '',
        '## 1. A Few Large Changes Dominate the Errors', '',
        f"There are {f['validation_cves']} validation CVEs: {result['group_counts']}.",
        'Unchanged means equal endpoint scores, not necessarily a flat curve.',
        f"The {f['large_change_cves']} large changes account for {f['large_changes_share_of_total_absolute_error_percent']:.2f}% of total absolute persistence error.",
        f"Persistence MAE is {f['persistence_mae']:.8f}; among the other CVEs it is {f['other_cves_mae']:.8f}.",
        'The latter is a diagnostic subset, not a replacement headline result.', '',
        '| CVE | Day-30 EPSS | Day-60 EPSS | Increase (0-1 scale) | Stored posts after Day 60 / all stored posts |',
        '|---|---:|---:|---:|---:|']
    for r in f['large_change_details']:
        lines.append(f"| {r['cve']} | {r['latest_input_score']:.5f} | {r['day60_score']:.5f} | {r['change_since_latest_input']:+.5f} | {r['posts_after_target']} / {len(r['posts'])} |")
    lines += ['', 'An increase of 0.01 is one percentage point. All three have an observed Day-30 score.',
        'All their stored post dates fall after Day 60. Under those dates, these posts',
        'cannot help a prediction made on Day 30. This does not prove that no earlier',
        'public discussion existed; it may be missing or differently dated in this dataset.',
        'All three stored version records say v3 with no version change. That metadata',
        'alone does not authenticate the score series or establish the cause of a jump.', '',
        '## 2. Stored Text Is Not Necessarily Historical Text', '',
        'All 197 CVEs have descriptions. All associated NVD update timestamps are',
        'after Day 30. This is not proof the descriptions were written after Day 30:',
        'the update might concern a different field. Historical versions are still needed.',
        '469 of 570 reference/history entries are dated before Day 30, across 195 CVEs.',
        'Those are NVD reference dates, not timestamps proving the present webpage content existed.',
        f"There are {f['nonempty_summaries']} nonempty summaries among {f['summary_fields']} summary fields, across {f['cves_with_nonempty_summary']} CVEs.",
        'All summary fields lack verified generation/source-snapshot dates. Empty fields',
        'are included in the 394-field table; this is not a claim of 394 usable summaries.', '',
        '## 3. Post Coverage Is Limited at the Prediction Date', '',
        'Of 319 CVE/post associations, 76 are dated before Day 30, two on Day 30,',
        'and 241 after Day 30. The 76 earlier records cover 50 CVEs.',
        f"Exact deduplication leaves {f['distinct_exact_post_associations']} associations; this does not remove semantic duplicates.",
        'All 319 timestamps lack a timezone. Six Telegram associations belong to',
        'exact-repeat sets; these are six rows, not six duplicate excess rows.',
        'Six Telegram associations also have identical platform/text with multiple',
        'stored timestamps within validation. These sets need source-level checking.',
        'Of the 76 earlier associations, 46 are Exploit-DB records, 26 Mastodon,',
        'three Reddit and one Telegram. Exploit-DB entries are not ordinary social posts.', '',
        '## 4. Examples That Need Manual Source Review', '',
        '- CVE-2024-8963: the non-GitHub summary describes a generic Ivanti portal',
        '  error and concludes that reliable exploitation is very unlikely. An error',
        '  page does not support that conclusion. Treat it as unusable vulnerability',
        '  evidence until the source retrieval and summary are reviewed.',
        '- CVE-2023-7028: an Exploit-DB text contains "# Date: 2024-01-12", but its',
        '  stored date is 2024-03-14. The body date is before the 2024-02-11 anchor;',
        '  the stored date is after the 2024-03-12 endpoint. This can change eligibility.',
        '  The body date may describe authorship or another event; do not automatically',
        '  substitute it for verified publication time.',
        '- CVE-2024-23692: the Exploit-DB text contains "# Date: 2024-09-22" while',
        '  the stored date is 2025-03-28. Both are later than the 2024-07-30 endpoint.',
        'These are examples from the locked audit excerpts, not an online fact check.', '',
        '## 5. Verify the Scores Before Increasing Model Complexity', '',
        f"Non-missing Days 0-60 score status counts: {f['score_status_nonmissing_days0_60']}.",
        f"Day-60 target status counts: {f['score_status_target_day60']}.",
        'EXISTING_UNVERIFIED means that the recollection preserved an existing value',
        'without recording a successful match to a downloaded historical snapshot.',
        'It is not proof that the value is false. Filling missing values and verifying',
        'already-present values are different tasks. Every non-missing validation',
        'value in this 61-day window, including all 197 endpoints, has this status.',
        'The collector marks a preserved value OBSERVED only when the downloaded',
        'value matches; conflicts are logged and the old value is preserved.', '',
        '## Recommended Next Step', '',
        'First implement a separate historical EPSS verification study. Compare retained',
        'values with dated official snapshots, keep original values and report differences,',
        'missing snapshots and uncertain provenance. Start with the three large-change',
        'curves as diagnosis, then apply one uniform policy to training and validation,',
        'not only cases where a model made errors. Do not select a correction policy',
        'based on which version makes a model look better. Keep test performance closed.',
        'Use available verified local snapshots first; fetching missing ones is a separate run.',
        'If scores change, publish a versioned derived dataset and rebuild the prepared',
        'bundle; do not edit a locked bundle or compare scores from different cohorts.',
        'After that, resolve historical text and post dates before a leakage-safe TPG/SMP',
        'forecast. Current-text association can be studied separately, but must not be',
        'presented as a deployable prediction made in the past. No new model is justified',
        'by these three selected examples alone.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, default=audit.ROOT/'evidence_availability_audit')
    parser.add_argument('--prepared-dir', type=Path, default=audit.PREPARED)
    parser.add_argument('--run-dir', type=Path, default=audit.ROOT/'gru_residual_user_run')
    args = parser.parse_args()
    result = review(args.audit_dir, args.prepared_dir, args.run_dir)
    collector = audit.PROJECT.parent/'SummTPGVul/Sec4AI4Sec-EPSS/Data_Files/fetch_epss_data.py'
    result['collector_inspected_sha256'] = sha256(collector)
    output = args.audit_dir.with_name(args.audit_dir.name+'_review')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'analysis.json', result)
    text = explanation(result)
    for name in ('README.md', 'analysis.txt'):
        (output/name).write_text(text, encoding='utf-8')
    print(f'Verified audit; review saved to {output}. No training or test analysis.')


if __name__ == '__main__':
    main()
