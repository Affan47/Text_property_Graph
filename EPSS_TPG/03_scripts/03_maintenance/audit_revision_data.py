#!/usr/bin/env python3
"""Offline integrity and readiness audit of the pinned revision dataset files."""
import argparse
import ast
from collections import Counter
import contextlib
import csv
import hashlib
import inspect
import io
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.tpg-project-root').is_file())
SOURCE = ROOT.parent / 'SummTPGVul'
PREFIX = 'Sec4AI4Sec-EPSS/Data_Files/'
BASE = 'cves_unique_with_source_links_clean'
REF = 'cves_merged_refetched'
DATED = 'cves_merged_with_url_dates'
KEV = 'cves_merged_with_url_dates_vc_kev'
REVISION = 'a1e32ca3f16052b987767c013a49e2441b972e55'


def read_json(path):
    constants = Counter()

    def missing(value):
        constants[value] += 1
        return None

    # Normalize only the in-memory comparison, never the upstream files.
    return json.loads(path.read_text(), parse_constant=missing), dict(constants)


def csv_matches(value, expected):
    if expected is None:
        return value == ''
    if isinstance(expected, list):
        return json.loads(value) == expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return float(value) == expected
    return value == expected


def upstream_code_checks():
    files = subprocess.check_output(['git', '-C', str(SOURCE), 'ls-tree', '-r', '--name-only', 'HEAD'], text=True).splitlines()
    scraper = ast.parse((SOURCE / 'SummVul/Scrapers/nvd_repo_scraper.py').read_text())
    generator = ast.parse((SOURCE / 'Sec4AI4Sec-EPSS/LLM_summaries_gen/gen_llm_summ_web_parsing_bs.py').read_text())
    result = {
        'fetch_vulncheck_data_py_tracked': PREFIX + 'fetch_vulncheck_data.py' in files,
        'refetch_all_metrics_in_documented_script': any(isinstance(n, ast.FunctionDef) and n.name == 'refetch_all_metrics' for n in ast.walk(scraper)),
    }
    # Evaluate isolated definitions with synthetic arguments, never API calls.
    definition = next(n for n in generator.body if isinstance(n, ast.FunctionDef) and n.name == 'summarize_open_sources')
    namespace = {}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), '<summary-signature>', 'exec'), namespace)
    calls = [n for n in ast.walk(generator) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == definition.name]
    missing_api_key = False
    for call in calls:
        try:
            inspect.signature(namespace[definition.name]).bind(*([None] * len(call.args)), **{kw.arg: None for kw in call.keywords if kw.arg})
        except TypeError as exc:
            missing_api_key |= 'api_key' in str(exc)
    result['summary_call_missing_api_key'] = missing_api_key

    definition = next(n for n in scraper.body if isinstance(n, ast.FunctionDef) and n.name == 'get_query_response')
    response = SimpleNamespace(status_code=200, json=lambda: {'vulnerabilities': [{'cve': {'references': [{'url': 'https://example.org/advisory'}]}}]})
    namespace = {'CONFIG': {'NVD_API_KEY': 'offline-test', 'NVD_BASE_URL': 'unused'},
                 'MAX_ATTEMPTS': 1, 'REQUEST_TIMEOUT': 1,
                 'requests': SimpleNamespace(get=lambda *a, **kw: response, exceptions=SimpleNamespace(RequestException=RuntimeError))}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), '<mocked-reference-fetch>', 'exec'), namespace)
    with contextlib.redirect_stdout(io.StringIO()):
        result['mocked_valid_reference_response_returns'] = namespace[definition.name]('CVE-2026-12345')
    result['reference_fetch_empty_key_access'] = any(
        isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and n.slice.value == ''
        for n in ast.walk(definition))

    history = ast.parse((SOURCE / PREFIX / 'fetch_url_dates.py').read_text())
    definition = next(n for n in history.body if isinstance(n, ast.FunctionDef) and n.name == 'get_query_reponse')
    requests_seen = []

    def page(*args, **kwargs):
        params = dict(kwargs.get('params', {}))
        requests_seen.append(params)
        if len(requests_seen) > 3:
            raise RuntimeError('History mock exceeded the two-page fixture')
        index = params.get('startIndex', 0)
        return SimpleNamespace(status_code=200, json=lambda: {
            'totalResults': 2, 'resultsPerPage': 1, 'startIndex': index,
            'cveChanges': [{'change': {'id': f'event-{index}'}}] if index < 2 else [],
        })

    namespace = {'CONFIG': {'NVD_API_KEY': 'offline-test', 'CVE_CHANGELOG_URL': 'unused',
                            'MAX_ATTEMPTS': 1, 'REQUEST_TIMEOUT': 1},
                 'req': SimpleNamespace(get=page, exceptions=SimpleNamespace(RequestException=OSError))}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), '<mocked-reference-history>', 'exec'), namespace)
    with contextlib.redirect_stdout(io.StringIO()):
        changes = namespace[definition.name]({'cve': 'CVE-2026-12345'})
    result['reference_history_two_page_fixture'] = {
        'expected_events': 2, 'returned_events': len(changes or []),
        'request_parameters': requests_seen,
        'complete': changes == [{'id': 'event-0'}, {'id': 'event-1'}],
    }
    return result


def audit():
    revision = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError('Audit expectations apply only to the pinned source revision')
    report = {'source_revision': revision, 'files': {}, 'integrity_errors': [], 'readiness_warnings': []}
    errors, warnings = report['integrity_errors'], report['readiness_warnings']
    records = {}
    for stem, expected_rows, fields in [(BASE, 5692, {15}), (REF, 5968, {26}), (DATED, 5968, {28}), (KEV, 5968, {29, 39})]:
        name = stem + '.json'
        rows, constants = read_json(SOURCE / PREFIX / name)
        by_id = {r['cve']: r for r in rows}
        records[stem] = by_id
        info = {'records': len(rows), 'unique_cves': len(by_id), 'field_count_distribution': dict(Counter(len(r) for r in rows)), 'nonstandard_literals': constants}
        report['files'][name] = info
        if len(rows) != expected_rows or len(by_id) != expected_rows or set(map(len, rows)) != fields:
            errors.append(f'{name}: record count, uniqueness or field count differs')
        if constants:
            warnings.append(f'{name}: {sum(constants.values())} non-standard JSON constants; strict JSON parsers reject this file')
        if any(Counter(r['all_sources_urls']) != Counter(r['github_only_urls'] + r['non_github_urls']) for r in rows):
            errors.append(f'{name}: reference URL partitions disagree')
    for stem in [BASE, REF]:
        name = stem + '.csv'
        with (SOURCE / PREFIX / name).open(newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            columns = reader.fieldnames
        differences = 0
        for row in rows:
            expected = records[stem].get(row['cve'])
            if expected is None or set(row) != set(expected):
                differences += 1
                continue
            for key, value in row.items():
                try:
                    differences += not csv_matches(value, expected[key])
                except (ValueError, TypeError):
                    differences += 1
        info = {'records': len(rows), 'columns': len(columns), 'csv_json_differing_cells_or_rows': differences}
        report['files'][name] = info
        if differences or len(rows) != len(records[stem]) or {r['cve'] for r in rows} != records[stem].keys():
            errors.append(f'{name}: CSV/JSON values or CVE cohort disagree')

    for name, info in report['files'].items():
        content = (SOURCE / PREFIX / name).read_bytes()
        pointer = subprocess.check_output(['git', '-C', str(SOURCE), 'show', f'HEAD:{PREFIX}{name}'], text=True)
        expected = dict(line.split(' ', 1) for line in pointer.splitlines() if ' ' in line)
        digest = hashlib.sha256(content).hexdigest()
        info.update(bytes=len(content), sha256=digest)
        if expected.get('oid') != 'sha256:' + digest or expected.get('size') != str(len(content)):
            errors.append(f'{name}: LFS object size/hash mismatch')

    base, ref, dated, kev = (records[k] for k in [BASE, REF, DATED, KEV])
    report['lineage'] = {'base_to_refetched_added': len(ref.keys() - base.keys()), 'base_to_refetched_removed': len(base.keys() - ref.keys())}
    for label, before, after in [('refetched_to_dates', ref, dated), ('dates_to_kev', dated, kev)]:
        differences = sum(r != {k: after[c].get(k) for k in r} for c, r in before.items() if c in after)
        report['lineage'][label] = {'same_cve_ids': before.keys() == after.keys(), 'rows_with_changed_existing_fields': differences}
        if before.keys() != after.keys() or differences:
            errors.append(f'{label}: enrichment changed the cohort or existing fields')
    positive = [r for r in kev.values() if r.get('in_vckev') is True]
    details = [d for r in kev.values() for d in r['url_change_dates']]
    report['metrics'] = {
        'kev_positive': len(positive), 'kev_negative': sum(r.get('in_vckev') is False for r in kev.values()),
        'cisa_members_among_kev': sum(bool(r['cisa_date_added']) for r in positive),
        'missing_cvss_vectors': sum(not r['cvss_vector'] for r in ref.values()),
        'cvss_versions': dict(Counter(str(r['cvss_vector_version']) for r in ref.values())),
        'nvd_statuses': dict(Counter(r['vuln_status'] for r in ref.values())),
        'reference_url_slots': sum(len(r['all_sources_urls']) for r in ref.values()),
        'distinct_url_strings': len({u for r in ref.values() for u in r['all_sources_urls']}),
        'reference_date_entries': len(details), 'reference_match_modes': dict(Counter(d['match_mode'] for d in details)),
        'missing_earliest_reference_dates': sum(d['earliest_added_date'] is None for d in details),
        'url_dates_status': dict(Counter(r['url_dates_status'] for r in kev.values())),
        'published_date_min': min(r['published_date'] for r in ref.values()),
        'published_date_max': max(r['published_date'] for r in ref.values()),
    }
    if len(positive) != 548 or report['metrics']['kev_negative'] != 5420:
        errors.append('Unexpected or non-boolean KEV membership flags')
    if any({d['url'] for d in r['url_change_dates']} != set(r['all_sources_urls']) for r in kev.values()):
        errors.append('Reference-date coverage differs from the reference URLs')
    checks = report['upstream_code_checks'] = upstream_code_checks()
    readme = (SOURCE / 'README.md').read_text()
    row = next((line for line in readme.splitlines() if line.startswith('| ') and f'{DATED}.json`' in line), '')
    documented = re.search(r'(\d+) fields', row)
    actual = len({k for r in dated.values() for k in r})
    report['date_intermediate_schema_check'] = {
        'documented_fields': int(documented.group(1)) if documented else None, 'observed_fields': actual}
    if documented and int(documented.group(1)) != actual:
        warnings.append(f'The README says {documented.group(1)} fields for the date-only intermediate; the file has {actual}')
    keys = {k.lower() for stage in records.values() for r in stage.values() for k in r}
    if not any('epss' in k or 'summary' in k or k.startswith('summ_') for k in keys):
        warnings.append('The six revision files do not provide EPSS-window targets or generated revision summaries')
    if not checks['fetch_vulncheck_data_py_tracked'] or not checks['refetch_all_metrics_in_documented_script']:
        warnings.append('Documented preparation components are missing; see upstream_code_checks for each tracked file and function checked')
    if checks['summary_call_missing_api_key']:
        warnings.append('The upstream summary loop omits the required api_key argument')
    if checks['reference_fetch_empty_key_access'] and checks['mocked_valid_reference_response_returns'] == []:
        warnings.append('The upstream NVD reference fetch indexes cve[""] instead of cve["references"]; a mocked valid response yields an empty list')
    if not checks['reference_history_two_page_fixture']['complete']:
        warnings.append('The reference-history fetch returns incomplete history on an offline two-page fixture; this does not prove that a saved CVE lost history')
    warnings.append('Snapshot consistency does not validate historical availability, negative labels or a leakage-free training split')
    report['audit_scope'] = 'Pinned-snapshot integrity plus observed documentation and isolated offline code checks; no external evidence or historical availability verification.'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Write the audit report as strict JSON; source files remain untouched')
    args = parser.parse_args()
    report = audit()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    return bool(report['integrity_errors'])


if __name__ == '__main__':
    raise SystemExit(main())
