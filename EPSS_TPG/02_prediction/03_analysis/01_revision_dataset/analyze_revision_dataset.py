#!/usr/bin/env python3
"""Read-only profiling of all six revision snapshots; writes strict JSON and text."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import importlib.util
from importlib.metadata import version as package_version
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.tpg-project-root').is_file())
BASE = 'cves_unique_with_source_links_clean'
REF = 'cves_merged_refetched'
DATED = 'cves_merged_with_url_dates'
KEV = 'cves_merged_with_url_dates_vc_kev'
FILES = [BASE + '.csv', BASE + '.json', REF + '.csv', REF + '.json', DATED + '.json', KEV + '.json']
URL_LISTS = {'all_sources_urls', 'github_only_urls', 'non_github_urls'}
NUMERIC = {'cvss_score', 'cvss_version', 'orig_cvss_score', 'orig_cvss_version',
           'cvss_vector_score', 'cvss_vector_version', 'n_references_nvd'}
MISSING = object()


class NonFinite(str):
    """Preserve invalid JSON number tokens separately from explicit nulls."""


MEANINGS = {
    'cve': 'CVE identifier and join key; not a numerical predictor.',
    'description': 'Vulnerability description in this snapshot.',
    'cvss_version': 'Version of the stored CVSS metric; categorical despite numeric storage.',
    'cvss_score': 'Stored CVSS score; compare within a metric version.',
    'attack_vector': 'Stored CVSS attack-vector category.',
    'attack_complexity': 'Stored CVSS attack-complexity category.',
    'privileges_required': 'Stored CVSS privileges-required category.',
    'user_interaction': 'Stored CVSS user-interaction category.',
    'scope': 'Stored CVSS scope category.',
    'confidentiality_impact': 'Stored CVSS confidentiality-impact category.',
    'integrity_impact': 'Stored CVSS integrity-impact category.',
    'availability_impact': 'Stored CVSS availability-impact category.',
    'all_sources_urls': 'NVD reference URL list, including repeated URL slots.',
    'github_only_urls': 'Stored GitHub partition of the reference URL list.',
    'non_github_urls': 'Stored non-GitHub partition of the reference URL list.',
    'vuln_status': 'NVD record processing status at the refreshed snapshot.',
    'n_references_nvd': 'Stored reference-entry count, not necessarily unique URLs.',
    'published_date': 'NVD publication calendar date.',
    'published_timestamp': 'NVD publication timestamp.',
    'last_modified_timestamp': 'NVD last-modification timestamp; not a download or feature-availability timestamp.',
    'cvss_vector': 'Serialized CVSS vector; tokens must remain version-specific.',
    'cvss_vector_version': 'Version of the selected CVSS vector; categorical.',
    'cvss_vector_score': 'Stored score associated with the selected vector; not independently recomputed here.',
    'cvss_vector_severity': 'Stored severity category; absence is audited by version.',
    'cvss_vector_source': 'Author/source identifier of the selected CVSS metric, not a separate acquisition source.',
    'url_change_dates': 'Per-reference history objects from NVD change-history enrichment.',
    'url_dates_status': 'Reference-history enrichment status.',
    'in_vckev': 'Observed VulnCheck KEV membership; possible target only after task definition.',
    'vckev_date_added': 'Recorded VulnCheck catalogue addition date, not necessarily first exploitation.',
    'vckev_updated_at': 'Recorded update timestamp on the VulnCheck enrichment.',
    'cisa_date_added': 'CISA addition timestamp supplied inside the VulnCheck-listed subset.',
    'due_date': 'Due date supplied with that enrichment; not a model prediction horizon.',
    'vendor_project': 'Vendor/project name supplied only with the KEV enrichment.',
    'vckev_cwes': 'CWE identifiers supplied with the KEV enrichment.',
    'vckev_known_ransomware_campaign_use': 'Stored ransomware-use category; Unknown is not proof of no use.',
    'vckev_reported_exploited_by_canaries': 'Boolean supplied by the KEV enrichment; absence is not false.',
    'vckev_reported_exploitation': 'List of reported-exploitation evidence objects.',
    'vckev_xdb': 'List of exploit-database references supplied by VulnCheck.',
    'url': 'URL stored in the enclosing reference/evidence object.',
    'earliest_added_date': 'Earliest matched NVD reference-addition event, not web-page publication.',
    'last_modified_url_date': 'Recorded URL-history modification date, not page modification time.',
    'added_dates': 'Matched reference-addition event timestamps.',
    'removed_dates': 'Matched reference-removal event timestamps.',
    'n_removals': 'Recorded number of reference-removal events.',
    'match_mode': 'How a reference URL was matched to the available history.',
    'date_added': 'Addition timestamp for the enclosing exploitation evidence or exploit reference.',
    'clone_ssh_url': 'SSH clone URL of an exploit repository.',
    'exploit_type': 'Stored exploit type/category.',
    'xdb_id': 'Exploit-database record identifier.',
    'xdb_url': 'Exploit-database record URL.',
}


def missing_kind(value):
    if value is MISSING:
        return 'absent_key'
    if value is None:
        return 'null'
    if isinstance(value, NonFinite) or (isinstance(value, float) and not math.isfinite(value)):
        return 'nonfinite_number'
    if isinstance(value, str) and not value.strip():
        return 'blank_string'
    return None


def clean(value):
    if missing_kind(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def signature(value):
    """Semantic comparison for decoded CSV/JSON pairs only."""
    return json.dumps(clean(value), sort_keys=True, ensure_ascii=True, allow_nan=False)


def exact_signature(value):
    """Preserve type, missingness reason and key presence for structural checks."""
    def tagged(v):
        reason = missing_kind(v)
        if reason:
            return [reason, str(v) if isinstance(v, str) or reason == 'nonfinite_number' else None]
        if isinstance(v, dict):
            return ['dict', [[k, tagged(x)] for k, x in sorted(v.items())]]
        if isinstance(v, list):
            return ['list', [tagged(x) for x in v]]
        return [type(v).__name__, v]
    return json.dumps(tagged(value), ensure_ascii=True, allow_nan=False)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON object key: {key!r}')
        result[key] = value
    return result


def stats(values):
    a = np.asarray([v for v in values if finite_number(v)], dtype=float)
    if not len(a):
        return {'count': 0}
    quantiles = np.quantile(a, [0, .01, .25, .5, .75, .95, .99, 1])
    result = dict(zip(['min', 'p01', 'p25', 'median', 'p75', 'p95', 'p99', 'max'], map(float, quantiles)))
    result.update(count=len(a), mean=float(a.mean()), std_population=float(a.std()),
                  zeros=int((a == 0).sum()), negative=int((a < 0).sum()))
    iqr = quantiles[4] - quantiles[2]
    result['outside_1_5_iqr'] = int(((a < quantiles[2] - 1.5 * iqr) | (a > quantiles[4] + 1.5 * iqr)).sum()) if iqr > 0 else None
    result['iqr_rule_status'] = 'evaluated' if iqr > 0 else 'not_applicable_zero_iqr'
    return result


def frequencies(values, limit=20):
    counter = Counter(values)
    ordered = sorted(counter.items(), key=lambda pair: (-pair[1], str(pair[0])))
    return {'distinct': len(counter), 'total': sum(counter.values()),
            'values': [{'value': str(k), 'count': n} for k, n in ordered[:limit]],
            'omitted_categories': max(0, len(counter) - limit)}


def parse_date(value):
    if not isinstance(value, str) or isinstance(value, NonFinite):
        return None
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?', value.strip()):
        return None
    try:
        result = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


def is_date(path):
    name = path.split('.')[-1].removesuffix('[]')
    return (name.endswith(('_timestamp', '_date', '_dates', '_date_added', '_updated_at')) and name != 'url_change_dates') or name == 'date_added'


def field_definition(path):
    root = path.split('.')[0].removesuffix('[]')
    leaf = path.split('.')[-1].removesuffix('[]')
    meaning = MEANINGS.get(leaf, 'Observed field; no external semantics inferred.')
    if leaf.startswith('orig_'):
        meaning = 'Original snapshot: ' + MEANINGS.get(leaf[5:], 'stored earlier value.')
    if root == 'in_vckev':
        role = 'candidate_target_not_selected'
    elif root.startswith('vckev_') or root in {'vendor_project', 'cisa_date_added', 'due_date'}:
        role = 'exclude_from_membership_predictors_target_linked_enrichment'
    elif root == 'cve':
        role = 'join_and_group_key_not_predictor'
    elif root in URL_LISTS or root in {'url_change_dates', 'url_dates_status', 'n_references_nvd'}:
        role = 'reference_evidence_requires_as_of_filtering'
    elif is_date(root):
        role = 'temporal_audit_or_split_field_not_automatically_predictor'
    else:
        role = 'candidate_input_requires_provenance_and_task_review'
    return {'meaning': meaning, 'membership_task_role': role,
            'semantic_note': 'Version fields are categories. Empty lists are observed empty collections, not missing keys.'}


def load_file(path):
    constants = Counter()
    content = path.read_bytes()
    if content.startswith(b'version https://git-lfs.github.com/spec/'):
        raise ValueError(f'{path}: Git LFS pointer, not downloaded dataset contents')
    fingerprint = {'source_sha256': hashlib.sha256(content).hexdigest(), 'source_bytes': len(content)}

    def nonfinite(token):
        constants[token] += 1
        return NonFinite(token)

    if path.suffix == '.json':
        def parse_float(token):
            value = float(token)
            return value if math.isfinite(value) else nonfinite('overflow:' + token)

        rows = json.loads(content.decode('utf-8-sig'), parse_constant=nonfinite,
                          parse_float=parse_float, object_pairs_hook=unique_object)
        if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
            raise ValueError(f'{path}: expected an array of records')
        return rows, {**fingerprint, 'format': 'JSON',
                      'nonstandard_number_tokens': {k: v for k, v in constants.items() if not k.startswith('overflow:')},
                      'float_overflow_tokens': {k: v for k, v in constants.items() if k.startswith('overflow:')},
                      'strict_json_numbers': not any(not k.startswith('overflow:') for k in constants),
                      'conversion_errors': []}
    if path.suffix != '.csv':
        raise ValueError(f'{path}: only CSV and JSON are supported')
    errors = []
    with io.StringIO(content.decode('utf-8-sig'), newline='') as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if not headers or any(not key.strip() for key in headers) or len(headers) != len(set(headers)):
            raise ValueError(f'{path}: empty or duplicate CSV headers')
        rows = list(reader)
    for i, row in enumerate(rows):
        if None in row:
            raise ValueError(f'{path}: excess columns in CSV record {i + 1}')
        if any(value is None for value in row.values()):
            raise ValueError(f'{path}: too few columns in CSV record {i + 1}')
        for key, value in row.items():
            if not value or not value.strip():
                continue
            try:
                if key in URL_LISTS:
                    parsed = json.loads(value, parse_constant=nonfinite, object_pairs_hook=unique_object)
                    if not isinstance(parsed, list):
                        raise ValueError('not a JSON list')
                    row[key] = parsed
                elif key in NUMERIC:
                    number = float(value)
                    row[key] = (int(number) if key == 'n_references_nvd' and number.is_integer()
                                else number if math.isfinite(number) else NonFinite(value))
            except (ValueError, TypeError) as exc:
                errors.append({'record': i + 1, 'field': key, 'error': str(exc)})
    return rows, {**fingerprint, 'format': 'CSV', 'storage_types': 'All CSV cells are strings before explicit decoding.',
                  'decoded_list_columns': sorted(URL_LISTS.intersection(headers)),
                  'decoded_numeric_columns': sorted(NUMERIC.intersection(headers)),
                  'nonstandard_number_tokens_in_list_cells': dict(constants), 'conversion_errors': errors}


def index_records(rows, name):
    result = {}
    for number, row in enumerate(rows, 1):
        cve = row.get('cve')
        if type(cve) is not str or re.fullmatch(r'CVE-\d{4}-\d{4,}', cve) is None:
            raise ValueError(f'{name}, record {number}: missing or invalid CVE identifier')
        if cve in result:
            raise ValueError(f'{name}: duplicate CVE {cve}; one-row-per-CVE analysis would silently collapse it')
        result[cve] = row
    return result


def validate_records(rows, name, decoding):
    """Reject unsupported shapes before downstream code can coerce or drop them."""
    if not rows:
        raise ValueError(f'{name}: empty dataset; cannot analyze a CVE cohort')
    if decoding.get('conversion_errors'):
        raise ValueError(f'{name}: CSV decoding failed: {decoding["conversion_errors"][:3]}')
    index_records(rows, name)
    object_lists = {'url_change_dates', 'vckev_reported_exploitation', 'vckev_xdb'}
    string_lists = URL_LISTS | {'added_dates', 'removed_dates', 'vckev_cwes'}
    boolean_fields = {'in_vckev', 'vckev_reported_exploited_by_canaries'}

    def validate(value, key, location):
        if key in object_lists | string_lists:
            if not isinstance(value, list):
                raise ValueError(f'{location}: expected list, not {type(value).__name__}')
            for j, item in enumerate(value):
                if key in string_lists and type(item) is not str:
                    raise ValueError(f'{location}[{j}]: expected string element')
                if key in object_lists:
                    if not isinstance(item, dict):
                        raise ValueError(f'{location}[{j}]: expected object element')
                    if key == 'url_change_dates':
                        required = {'url', 'added_dates', 'removed_dates', 'n_removals', 'match_mode'}
                        if not required.issubset(item):
                            raise ValueError(f'{location}[{j}]: incomplete reference-history object')
                    for field, val in item.items():
                        validate(val, field, f'{location}[{j}].{field}')
            return
        if missing_kind(value):
            if key in {'cve', 'description', 'in_vckev', 'url', 'match_mode'}:
                raise ValueError(f'{location}: required value is missing')
            return
        if key in boolean_fields:
            valid = type(value) is bool
        elif key in NUMERIC | {'n_removals'}:
            valid = finite_number(value)
            if key in {'n_removals', 'n_references_nvd'}:
                valid = valid and value >= 0 and float(value).is_integer()
        elif key in MEANINGS or key.startswith('orig_') or is_date(key):
            valid = type(value) is str
        else:
            valid = True  # Extra fields are profiled, not silently discarded.
        if not valid:
            raise ValueError(f'{location}: incompatible value type {type(value).__name__}')

    for number, row in enumerate(rows, 1):
        required = {'cve', 'description'} | URL_LISTS
        if name == KEV + '.json':
            required |= {'in_vckev', 'url_change_dates'}
        if name == DATED + '.json':
            required.add('url_change_dates')
        if not required.issubset(row):
            raise ValueError(f'{name}, record {number}: missing required keys {sorted(required - row.keys())}')
        for key, value in row.items():
            validate(value, key, f'{name}, record {number}, {key}')


def profile_values(path, values, as_of):
    reasons = Counter(missing_kind(v) for v in values if missing_kind(v))
    valid = [v for v in values if not missing_kind(v)]
    types = Counter('nonstandard_number' if isinstance(v, NonFinite) else type(v).__name__
                    for v in values if v is not MISSING)
    result = {'path': path, **field_definition(path), 'denominator': len(values),
              'denominator_unit': 'record' if '.' not in path and '[]' not in path else 'element/object in enclosing collection',
              'observed_types': dict(types), 'missing': dict(reasons), 'missing_count': sum(reasons.values()),
              'missing_percent': 100 * sum(reasons.values()) / len(values) if values else 0,
              'empty_list_count': sum(isinstance(v, list) and not v for v in valid),
              'empty_object_count': sum(isinstance(v, dict) and not v for v in valid),
              'distinct_nonmissing_values': len({exact_signature(v) for v in valid})}
    if not valid:
        result['semantic_type'] = 'no_nonmissing_values'
        return result
    if all(isinstance(v, list) for v in valid):
        result.update(semantic_type='list', list_length=stats([len(v) for v in valid]),
                      element_types=dict(Counter(type(x).__name__ for v in valid for x in v)),
                      duplicate_items_within_lists=sum(len(v) - len({exact_signature(x) for x in v}) for v in valid))
    elif all(isinstance(v, dict) for v in valid):
        result['semantic_type'] = 'object'
    elif all(type(v) is bool for v in valid):
        result.update(semantic_type='boolean', frequencies=frequencies(valid))
    elif all(type(v) in (int, float) for v in valid):
        if 'version' in path:
            result['semantic_type'] = 'categorical_version'
            result['frequencies'] = frequencies(valid)
        else:
            result.update(semantic_type='numeric', numeric=stats(valid))
    elif is_date(path):
        dates = [parse_date(v) for v in valid]
        parsed = [v for v in dates if v is not None]
        result.update(semantic_type='datetime', dates={
            'valid': len(parsed), 'invalid': len(dates) - len(parsed),
            'min_utc': min(parsed).isoformat() if parsed else None,
            'max_utc': max(parsed).isoformat() if parsed else None,
            'after_analysis_as_of': sum(v > as_of for v in parsed),
            'date_only': sum(isinstance(v, str) and len(v) == 10 for v in valid),
            'without_explicit_timezone': sum(isinstance(v, str) and not re.search(r'(Z|[+-]\d\d:\d\d)$', v) for v in valid),
            'year_counts': dict(sorted(Counter(str(v.year) for v in parsed).items())),
            'invalid_examples': [str(v)[:120] for v, d in zip(valid, dates) if d is None][:10],
        })
    elif all(isinstance(v, str) for v in valid):
        leaf = path.split('.')[-1]
        text = 'description' in leaf
        result.update(semantic_type='free_text' if text else 'identifier' if leaf in {'cve', 'xdb_id'} else 'string_category_or_reference',
                      character_length=stats([len(v) for v in valid]))
        if text:
            normalized = Counter(' '.join(v.casefold().split()) for v in valid)
            result.update(word_count=stats([len(v.split()) for v in valid]),
                          non_ascii_rows=sum(not v.isascii() for v in valid),
                          normalized_duplicate_groups=sum(n > 1 for n in normalized.values()),
                          normalized_duplicate_excess=sum(n - 1 for n in normalized.values()))
        else:
            result['frequencies'] = frequencies(valid, 100 if len(set(valid)) <= 100 else 20)
    else:
        result['semantic_type'] = 'mixed_types_requires_review'
    return result


def nested_profiles(path, values, output, as_of):
    output[path] = profile_values(path, values, as_of)
    objects = [v for v in values if isinstance(v, dict)]
    lists = [v for v in values if isinstance(v, list)]
    if objects:
        for key in sorted({k for obj in objects for k in obj}):
            nested_profiles(path + '.' + key, [v.get(key, MISSING) for v in objects], output, as_of)
    if lists:
        items = [item for value in lists for item in value]
        if items:
            nested_profiles(path + '[]', items, output, as_of)


def url_host(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value)
        return parsed.hostname.lower() if parsed.scheme in {'http', 'https', 'ftp'} and parsed.hostname else None
    except ValueError:
        return None


def quality_checks(rows):
    issues = defaultdict(list)
    checks = Counter()
    hosts, shared = Counter(), defaultdict(set)
    schemes, github_hosts = Counter(), Counter()
    descriptions = defaultdict(list)
    reference_slots = []
    histories = []
    vector_components = defaultdict(Counter)
    numeric_lags = defaultdict(list)
    for i, row in enumerate(rows):
        cve = row.get('cve', f'row:{i}')

        def flag(name, condition):
            checks[name] += 1
            if condition:
                issues[name].append(cve)

        flag('invalid_cve_identifier', not isinstance(cve, str) or re.fullmatch(r'CVE-\d{4}-\d{4,}', cve) is None)
        urls = row.get('all_sources_urls', [])
        if not isinstance(urls, list):
            flag('reference_list_wrong_type', True)
            urls = []
        reference_slots.extend(urls)
        github, other = row.get('github_only_urls', []), row.get('non_github_urls', [])
        flag('url_partition_multiset_mismatch', Counter(urls) != Counter(github + other))
        def github_domain(url):
            host = url_host(url) or ''
            return host == 'github.com' or host.endswith('.github.com')

        flag('github_partition_non_exact_host', any(url_host(u) != 'github.com' for u in github))
        flag('github_partition_outside_github_domain', any(not github_domain(u) for u in github))
        flag('non_github_partition_contains_github_domain', any(github_domain(u) for u in other))
        flag('unparseable_reference_url', any(url_host(u) is None for u in urls))
        flag('non_http_reference', any(not str(u).startswith(('http://', 'https://')) for u in urls))
        flag('repeated_url_scheme', any(re.match(r'^https?://https?://', str(u)) for u in urls))
        for url in urls:
            try:
                schemes[urlsplit(url).scheme] += 1
            except ValueError:
                schemes['<unparseable>'] += 1
        github_hosts.update(url_host(u) or '<unparsed>' for u in github)
        flag('duplicate_reference_slots', len(urls) != len(set(urls)))
        if 'n_references_nvd' in row:
            flag('reference_count_mismatch', row['n_references_nvd'] != len(urls))
        for host in {url_host(u) for u in urls} - {None}:
            hosts[host] += 1
        for url in set(urls):
            shared[url].add(cve)
        for name in ('cvss_score', 'orig_cvss_score', 'cvss_vector_score'):
            value = row.get(name)
            if type(value) in (float, int):
                flag(name + '_outside_0_10', not 0 <= value <= 10)
        vector = row.get('cvss_vector')
        version = row.get('cvss_vector_version')
        if isinstance(vector, str) and not missing_kind(vector):
            tokens = vector.split('/')
            explicit = tokens.pop(0)[5:] if tokens[0].startswith('CVSS:') else None
            pairs = [token.split(':', 1) for token in tokens]
            flag('cvss_vector_syntax_error', any(len(pair) != 2 or not all(pair) for pair in pairs))
            flag('cvss_vector_version_mismatch', explicit is not None and str(version) != explicit)
            keys = [pair[0] for pair in pairs]
            flag('cvss_duplicate_metric_keys', len(keys) != len(set(keys)))
            required = {'AV', 'AC', 'Au', 'C', 'I', 'A'} if version == 2.0 else {'AV', 'AC', 'PR', 'UI', 'S', 'C', 'I', 'A'}
            if version in {2.0, 3.0, 3.1}:
                flag('cvss_missing_required_metric_keys', not required.issubset(keys))
            for pair in pairs:
                if len(pair) == 2:
                    vector_components[f'{version}:{pair[0]}'][pair[1]] += 1
        publication = parse_date(row.get('published_timestamp'))
        modified = parse_date(row.get('last_modified_timestamp'))
        if publication and modified:
            lag = (modified - publication).total_seconds() / 86400
            numeric_lags['modification_minus_publication_days'].append(lag)
            flag('modification_before_publication', lag < 0)
        if publication and row.get('published_date'):
            flag('publication_date_timestamp_disagreement', publication.date().isoformat() != row['published_date'])
        for name in ('vckev_date_added', 'cisa_date_added'):
            date = parse_date(row.get(name))
            if publication and date:
                numeric_lags[name + '_minus_publication_days'].append((date - publication).total_seconds() / 86400)
                flag(name + '_calendar_date_before_publication', date.date() < publication.date())
                flag(name + '_timestamp_before_publication_same_calendar_day', date < publication and date.date() == publication.date())
        description = row.get('description')
        if isinstance(description, str) and not missing_kind(description):
            descriptions[' '.join(description.casefold().split())].append((cve, row.get('in_vckev')))
        if 'url_change_dates' in row:
            items = row['url_change_dates']
            histories.extend(items)
            flag('reference_history_url_set_mismatch', set(urls) != {d.get('url') for d in items})
            flag('duplicate_history_urls', len(items) != len({d.get('url') for d in items}))
            for item in items:
                additions = [parse_date(v) for v in item.get('added_dates', [])]
                removals = item.get('removed_dates', [])
                earliest = parse_date(item.get('earliest_added_date'))
                if earliest and publication:
                    numeric_lags['reference_first_added_minus_publication_days'].append((earliest - publication).total_seconds() / 86400)
                flag('history_removal_count_mismatch', item.get('n_removals') != len(removals))
                additions = [v for v in additions if v]
                flag('history_earliest_addition_mismatch', bool(additions) and earliest != min(additions))
    result = {
        'check_evaluation_counts': dict(checks),
        'issues': {name: {'affected_cves': len(set(ids)), 'events': len(ids), 'example_cves': sorted(set(ids))[:20]}
                   for name, ids in sorted(issues.items())},
        'reference_slots': len(reference_slots), 'distinct_reference_strings': len(set(reference_slots)),
        'urls_shared_between_cves': sum(len(ids) > 1 for ids in shared.values()),
        'cves_sharing_at_least_one_url': len(set().union(*(ids for ids in shared.values() if len(ids) > 1))) if shared else 0,
        'top_reference_hosts_by_distinct_cves': dict(hosts.most_common(30)),
        'reference_scheme_counts_by_slot': dict(schemes),
        'github_partition_host_counts_by_slot': dict(github_hosts),
        'url_check_notes': 'Non-exact GitHub hosts include valid subdomains. Archive wrappers have different hosts. FTP is a non-HTTP reference, not automatically corrupt. Repeated schemes are separately flagged.',
        'normalized_description_duplicate_groups': [
            {'text_sha256': hashlib.sha256(text.encode()).hexdigest(), 'cves': [c for c, _ in group],
             'mixed_membership': {flag for _, flag in group} == {True, False}}
            for text, group in sorted(descriptions.items()) if len(group) > 1
        ],
        'cvss_vector_component_counts_by_version': {k: dict(v) for k, v in sorted(vector_components.items())},
        'history_entries': len(histories),
        'history_match_modes': dict(Counter(d.get('match_mode') for d in histories)),
        'temporal_intervals_days': {name: stats(values) for name, values in numeric_lags.items()},
        'checks_scope': 'Local consistency only. Issues include review observations, not exclusively errors. Date-clock differences are not automatically invalid. URLs were not fetched, CVSS scores were not independently recomputed, and catalogue completeness was not established.',
    }
    return result


def profile_file(path, rows, decoding, as_of):
    def contains_nonfinite(value):
        if missing_kind(value) == 'nonfinite_number':
            return True
        if isinstance(value, dict):
            return any(contains_nonfinite(v) for v in value.values())
        if isinstance(value, list):
            return any(contains_nonfinite(v) for v in value)
        return False

    keys = sorted({key for row in rows for key in row})
    fields = {}
    for key in keys:
        nested_profiles(key, [r.get(key, MISSING) for r in rows], fields, as_of)
    ids = Counter(signature(r.get('cve')) for r in rows)
    row_counts = Counter(exact_signature(r) for r in rows)
    return {
        'file': path.name, 'bytes': decoding['source_bytes'], 'sha256': decoding['source_sha256'],
        'rows': len(rows), 'top_level_feature_count': len(keys), 'top_level_features': keys,
        'field_count_per_record': dict(sorted(Counter(str(len(r)) for r in rows).items())),
        'unique_cve_values': len(ids), 'duplicate_cve_excess': sum(n - 1 for n in ids.values()),
        'duplicate_full_record_excess': sum(n - 1 for n in row_counts.values()),
        'records_containing_nonfinite_values': sum(contains_nonfinite(row) for row in rows),
        'decoding': decoding, 'profiled_paths_including_nested': len(fields), 'features': fields,
        'top_level_semantic_type_counts': dict(Counter(fields[key]['semantic_type'] for key in keys)),
        'quality': quality_checks(rows),
    }


def lineage(datasets):
    result = {}
    indexed = {name: index_records(rows, name) for name, rows in datasets.items()}
    for stem in (BASE, REF):
        left, right = datasets[stem + '.csv'], datasets[stem + '.json']
        right_ids = indexed[stem + '.json']
        differences = Counter()
        presence = Counter()
        for row in left:
            counterpart = right_ids.get(row.get('cve'), {})
            for key in set(row) | set(counterpart):
                if (key in row) != (key in counterpart):
                    presence[key] += 1
                if signature(row.get(key, MISSING)) != signature(counterpart.get(key, MISSING)):
                    differences[key] += 1
        result[stem + '_csv_json'] = {'same_cve_set': {r['cve'] for r in left} == set(right_ids),
                                    'same_row_count': len(left) == len(right), 'differing_cells_by_field': dict(differences),
                                    'key_presence_differences': dict(presence),
                                    'comparison': 'Decoded CSV lists/numbers; blank, null and nonfinite normalized to null in memory.'}
    for before, after in ((BASE, REF), (REF, DATED), (DATED, KEV)):
        a = indexed[before + '.json']
        b = indexed[after + '.json']
        common = a.keys() & b.keys()
        old_fields = {k for r in a.values() for k in r}
        new_fields = {k for r in b.values() for k in r}
        fields = old_fields & new_fields
        changes = {k: sum(exact_signature(a[c].get(k, MISSING)) != exact_signature(b[c].get(k, MISSING)) for c in common)
                   for k in sorted(fields)}
        result[before + '_to_' + after] = {'common_cves': len(common), 'added_cves': sorted(b.keys() - a.keys()),
                                         'removed_cves': sorted(a.keys() - b.keys()), 'changed_shared_fields': changes,
                                         'removed_schema_fields': sorted(old_fields - new_fields),
                                         'added_schema_fields': sorted(new_fields - old_fields),
                                         'comparison': 'Exact decoded types, missingness reasons and key presence; CSV/JSON semantic normalization is not used for stage preservation.'}
        if before == BASE:
            original_fields = old_fields - URL_LISTS - {'cve'}
            result[before + '_to_' + after]['base_to_orig_field_differences'] = {
                key: sum(exact_signature(a[c].get(key, MISSING)) != exact_signature(b[c].get('orig_' + key, MISSING)) for c in common)
                for key in sorted(original_fields)
            }
    return result


def membership_analysis(rows):
    groups = {'listed': [r for r in rows if r.get('in_vckev') is True],
              'not_listed': [r for r in rows if r.get('in_vckev') is False]}
    keys = sorted({key for row in rows for key in row})
    availability = {}
    for key in keys:
        availability[key] = {}
        for group, subset in groups.items():
            present = sum(key in r for r in subset)
            nonmissing = sum(not missing_kind(r.get(key, MISSING)) for r in subset)
            availability[key][group] = {'rows': len(subset), 'present': present, 'nonmissing': nonmissing,
                                        'missing_percent': 100 * (len(subset) - nonmissing) / len(subset) if subset else None}
    perfect = [k for k, v in availability.items() if v['listed']['present'] == len(groups['listed'])
               and groups['listed'] and v['not_listed']['present'] == 0 and groups['not_listed']]
    years = defaultdict(lambda: Counter())
    by_version = defaultdict(lambda: Counter())
    for row in rows:
        group = 'listed' if row.get('in_vckev') is True else 'not_listed' if row.get('in_vckev') is False else 'invalid'
        date = parse_date(row.get('published_date'))
        years[str(date.year) if date else 'missing'][group] += 1
        by_version[str(clean(row.get('cvss_vector_version')))][group] += 1
    group_distributions = {}
    for group, subset in groups.items():
        group_distributions[group] = {
            'description_words': stats([len(r.get('description', '').split()) for r in subset]),
            'reference_slots': stats([len(r.get('all_sources_urls', [])) for r in subset]),
            'unique_reference_urls': stats([len(set(r.get('all_sources_urls', []))) for r in subset]),
            'current_scores_by_version': {str(v): stats([r.get('cvss_vector_score') for r in subset
                                                       if r.get('cvss_vector_version') == v]) for v in (2.0, 3.0, 3.1)},
            'categories': {k: frequencies([str(clean(r.get(k))) for r in subset], 100)
                           for k in ('vuln_status', 'cvss_vector_severity', 'orig_attack_vector', 'orig_attack_complexity')},
        }
    return {'meaning': 'Observed membership, not confirmed exploited versus confirmed unexploited ground truth.',
            'counts': {k: len(v) for k, v in groups.items()},
            'invalid_or_missing_flags': len(rows) - sum(map(len, groups.values())),
            'listed_percent': 100 * len(groups['listed']) / sum(map(len, groups.values())) if any(groups.values()) else None,
            'listed_percent_denominator': 'records with a native boolean membership value',
            'presence_alone_identifies_membership': perfect,
            'descriptive_group_distributions': group_distributions,
            'availability_by_membership': availability,
            'publication_year_counts': {k: dict(v) for k, v in sorted(years.items())},
            'cvss_version_counts': {k: dict(v) for k, v in sorted(by_version.items())}}


def drift_analysis(rows):
    descriptions = [(r.get('orig_description'), r.get('description')) for r in rows]
    comparable_text = [(a, b) for a, b in descriptions if not missing_kind(a) and not missing_kind(b)]
    pairs = [(r['orig_cvss_score'], r['cvss_vector_score'], r['orig_cvss_version'], r['cvss_vector_version'])
             for r in rows if all(not missing_kind(r.get(k)) for k in
                                  ('orig_cvss_score', 'cvss_vector_score', 'orig_cvss_version', 'cvss_vector_version'))]
    same = [(b - a) for a, b, av, bv in pairs if av == bv]
    return {'description_comparable_pairs': len(comparable_text),
            'original_description_missing': sum(bool(missing_kind(a)) for a, _ in descriptions),
            'current_description_available_without_original': sum(bool(missing_kind(a)) and not missing_kind(b) for a, b in descriptions),
            'description_exact_changes': sum(a != b for a, b in comparable_text),
            'description_normalized_changes': sum(' '.join(a.casefold().split()) != ' '.join(b.casefold().split()) for a, b in comparable_text),
            'score_comparable_pairs': len(pairs), 'stored_score_differences_all_versions': sum(a != b for a, b, _, _ in pairs),
            'metric_version_changes': sum(av != bv for _, _, av, bv in pairs),
            'same_version_pairs': len(same), 'same_version_score_changes': sum(d != 0 for d in same),
            'same_version_score_delta': stats(same),
            'note': 'Earlier capture dates are not established by the orig_ prefix. Score changes across versions are not severity deltas.'}


def correlation_analysis(rows):
    # Restrict current score comparisons to 3.1; mask original scores of other versions.
    observations = []
    for row in rows:
        if row.get('cvss_vector_version') != 3.1:
            continue
        urls = row.get('all_sources_urls', [])
        published = parse_date(row.get('published_date'))
        observations.append({
            'current_cvss_3_1': valid_cvss_score(row.get('cvss_vector_score')),
            'original_cvss_3_1': valid_cvss_score(row.get('orig_cvss_score')) if row.get('orig_cvss_version') == 3.1 else None,
            'description_words': len(row.get('description', '').split()),
            'reference_slots': len(urls), 'unique_references': len(set(urls)),
            'github_unique_references': len(set(row.get('github_only_urls', []))),
            'publication_year': published.year if published else None,
        })
    frame = pd.DataFrame(observations)
    frame = frame.loc[:, frame.nunique() > 1]
    corr = frame.corr(method='spearman', min_periods=3)
    mask = frame.notna().to_numpy(dtype=int)
    return {'method': 'Spearman, pairwise complete observations; current CVSS version 3.1 only.',
            'cohort_rows': len(frame), 'features': list(frame.columns),
            'matrix': [[float(v) if pd.notna(v) else None for v in line] for line in corr.to_numpy()],
            'pairwise_sample_counts': (mask.T @ mask).tolist(),
            'note': 'Exploratory association only, not causality or approval of a training feature. Labels and target-linked enrichments excluded.'}


def valid_cvss_score(value):
    return value if finite_number(value) and 0 <= value <= 10 else None


def write_pair(base, report, title):
    base.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False)
    base.with_suffix('.json').write_text(payload + '\n')
    # Keep every statistic in the text companion, with a readable orientation first.
    intro = [title, '=' * len(title)]
    if 'rows' in report:
        intro += [f"Records: {report['rows']}; top-level fields: {report['top_level_feature_count']}",
                  f"Unique CVE values: {report['unique_cve_values']}", '', 'FIELD SUMMARY',
                  'Field | semantic type | missing/denominator | observed storage types']
        for key, field in report['features'].items():
            intro.append(f"{key} | {field['semantic_type']} | {field['missing_count']}/{field['denominator']} | {field['observed_types']}")
    if 'findings' in report:
        intro += ['', 'FINDINGS', *[f'- {value}' for value in report['findings']], '', 'RECOMMENDATIONS',
                  *[f'- {value}' for value in report['recommendations']]]
    intro += ['', 'COMPLETE STRUCTURED DETAILS', payload]
    base.with_suffix('.txt').write_text('\n'.join(intro) + '\n')


def existing_audit():
    path = ROOT / '03_scripts/03_maintenance/audit_revision_data.py'
    spec = importlib.util.spec_from_file_location('revision_integrity_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT.parent / 'SummTPGVul/Sec4AI4Sec-EPSS/Data_Files')
    parser.add_argument('--output', type=Path, default=ROOT / '05_results/03_dataset_analysis/01_revision_dataset')
    parser.add_argument('--as-of', help='ISO date/time for future-date checks; defaults to execution time in UTC')
    args = parser.parse_args()
    as_of = parse_date(args.as_of) if args.as_of else datetime.now(timezone.utc)
    if as_of is None:
        parser.error('--as-of must be an ISO date or timestamp')
    source, output = args.source.resolve(), args.output.resolve()
    source_repo = ROOT.parent / 'SummTPGVul'
    if output.is_relative_to(source_repo.resolve()) or output.is_relative_to(source):
        parser.error('Reports must not be written inside the source dataset repository')
    publish_analysis(source, output, as_of)


def publish_analysis(source, output, as_of):
    """Complete and validate a staged run before replacing existing reports."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.revision-analysis-', dir=output.parent) as folder:
        staged = Path(folder)
        run_analysis(source, staged, as_of)
        files = sorted(p for p in staged.rglob('*') if p.is_file())
        # Reject symlinks that could redirect report writes outside the chosen output.
        for path in files:
            target = output / path.relative_to(staged)
            if not target.resolve().is_relative_to(output.resolve()):
                raise ValueError(f'Report target escapes output directory: {target}')
        files.sort(key=lambda p: p.name == '00_analysis.json')
        for path in files:
            target = output / path.relative_to(staged)
            target.parent.mkdir(parents=True, exist_ok=True)
            path.replace(target)
    print(f'Reports and figures saved to {output}', flush=True)


def run_analysis(source, output, as_of):
    source_repo = ROOT.parent / 'SummTPGVul'
    datasets, profiles = {}, {}
    for name in FILES:
        path = source / name
        rows, decoding = load_file(path)
        validate_records(rows, name, decoding)
        print(f'Profiling {name}: {len(rows)} records', flush=True)
        datasets[name] = rows
        profiles[name] = profile_file(path, rows, decoding, as_of)
        write_pair(output / '01_file_profiles' / name.replace('.', '_'), profiles[name], name)
    member = membership_analysis(datasets[KEV + '.json'])
    drift = drift_analysis(datasets[REF + '.json'])
    correlations = correlation_analysis(datasets[KEV + '.json'])
    dictionary = {}
    for name, profile in profiles.items():
        for path, field in profile['features'].items():
            entry = dictionary.setdefault(path, {**field_definition(path), 'files': {}})
            entry['files'][name] = {k: field[k] for k in ('observed_types', 'semantic_type', 'denominator', 'missing', 'missing_percent')}
    write_pair(output / '02_feature_dictionary', dictionary, 'Feature dictionary: observed fields and nested paths')
    source_revision = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    audit = existing_audit() if source == (source_repo / 'Sec4AI4Sec-EPSS/Data_Files').resolve() else {'skipped': 'Custom source; pinned-snapshot audit not applied.'}
    if audit.get('integrity_errors'):
        raise ValueError(f"Source integrity audit failed: {audit['integrity_errors']}")
    write_pair(output / '03_source_integrity_audit', audit, 'Existing source integrity and preparation-code audit')
    from plots import make_figures
    figures = make_figures(output / '04_figures', profiles, datasets, member, drift, correlations,
                           missing_kind=missing_kind, parse_date=parse_date)
    final = profiles[KEV + '.json']
    top_level_keys = {key for p in profiles.values() for key in p['top_level_features']}
    summaries = sorted(key for key in top_level_keys if 'summary' in key.lower() or key.lower().startswith('summ_'))
    epss = sorted(key for key in top_level_keys if 'epss' in key.lower())
    quality = final['quality']
    duplicate_groups = quality['normalized_description_duplicate_groups']
    nonstandard_files = {name: p['decoding']['nonstandard_number_tokens'] for name, p in profiles.items()
                         if p['decoding'].get('nonstandard_number_tokens')}
    evidence_max = final['features'].get('vckev_reported_exploitation', {}).get('list_length', {}).get('max')
    findings = [
        'Six files represent four related snapshots, including two CSV/JSON pairs; do not concatenate these into six independent cohorts.',
        f"Base cohort: {profiles[BASE + '.json']['rows']} CVEs and {profiles[BASE + '.json']['top_level_feature_count']} fields. Refetched/date-enriched/KEV-enriched union field counts: " + '/'.join(str(profiles[stem + '.json']['top_level_feature_count']) for stem in (REF, DATED, KEV)) + '.',
        f"Membership: {member['counts']['listed']} listed and {member['counts']['not_listed']} not listed ({member['listed_percent']:.2f}% listed). Nonmembership is not verified absence of exploitation.",
        f"{len(member['presence_alone_identifies_membership'])} fields identify membership from key presence alone: " + ', '.join(member['presence_alone_identifies_membership']),
        f"{drift['description_exact_changes']} descriptions changed among {drift['description_comparable_pairs']} nonmissing original/current pairs; {drift['stored_score_differences_all_versions']} stored scores differ among {drift['score_comparable_pairs']} pairs.",
        f'JSON files containing non-standard number tokens: {list(nonstandard_files)}. Field profiles distinguish these from explicit nulls and absent keys.',
        f'Columns named for EPSS: {epss}; columns named for summaries: {summaries}. No target or summary was invented.',
        'Reference history dates describe NVD history, not publication of the linked page; last_modified_timestamp is not proof of a feature capture time.',
        f"The final snapshot contains {final['quality']['reference_slots']} reference slots and {final['quality']['distinct_reference_strings']} distinct URL strings; repeated slots and shared URLs are counted separately.",
        f"Normalized descriptions repeat across {len(duplicate_groups)} groups covering {sum(len(g['cves']) for g in duplicate_groups)} CVEs; {sum(g['mixed_membership'] for g in duplicate_groups)} groups contain both membership values. This is text reuse, not proof that distinct CVE identifiers are duplicates.",
        f"{quality['cves_sharing_at_least_one_url']} CVEs share at least one reference URL with another CVE. Review this overlap before splitting.",
        f"Repeated URL schemes occur on {quality['issues'].get('repeated_url_scheme', {}).get('affected_cves', 0)} CVEs; non-HTTP references occur on {quality['issues'].get('non_http_reference', {}).get('affected_cves', 0)} CVEs. Valid FTP and archive wrappers require separate treatment from malformed URLs.",
        f"VulnCheck addition dates precede the NVD publication calendar date for {quality['issues'].get('vckev_date_added_calendar_date_before_publication', {}).get('affected_cves', 0)} CVEs. Another {quality['issues'].get('vckev_date_added_timestamp_before_publication_same_calendar_day', {}).get('affected_cves', 0)} have earlier timestamps on the same day. Different clocks/precision must not be treated as a verified prediction horizon.",
        (f'The largest exploitation-evidence list contains {evidence_max:.0f} objects. These target-linked lists should not be added to a membership-prediction graph.'
         if evidence_max is not None else 'No exploitation-evidence list lengths were available; absence of evidence is not proof of non-exploitation.'),
    ]
    report = {
        'analysis_version': 2, 'generated_utc': datetime.now(timezone.utc).isoformat(), 'as_of_utc': as_of.isoformat(),
        'source_directory': str(source), 'source_git_revision': source_revision.stdout.strip() if source_revision.returncode == 0 else None,
        'analysis_code_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ('analyze_revision_dataset.py', 'plots.py')},
        'source_audit_code_sha256': hashlib.sha256((ROOT / '03_scripts/03_maintenance/audit_revision_data.py').read_bytes()).hexdigest(),
        'environment': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__,
                        'matplotlib': package_version('matplotlib'), 'scipy': package_version('scipy')},
        'methodology': [
            'All records and all observed fields are profiled; no row sampling.',
            'Top-level denominators are CVE records; nested-path denominators are enclosing objects or list elements, not CVEs.',
            'Absent keys, explicit nulls, nonfinite tokens and blank strings are missing states; empty lists/objects are counted separately as observed containers.',
            'CSV numeric and list fields are decoded only by explicit schema; other strings are not silently coerced.',
            'Duplicate JSON keys, malformed CSV rows, duplicate/invalid CVE identifiers and incompatible field/container types stop report publication.',
            'CSV/JSON equivalence normalizes missing cells; within-JSON lineage and duplicate checks preserve key absence, null, nonfinite tokens and value types.',
            'Date-only/naive ISO timestamps are interpreted as UTC for comparison and separately counted; original source strings are preserved.',
            'Numeric outliers use a descriptive 1.5-IQR rule only when IQR is positive; a zero IQR is reported as not applicable, not as automatic outliers. Standard deviations use the population convention.',
            'Class-specific percentages use valid boolean groups; an absent group is undefined, not zero. Plots exclude missing, nonfinite and out-of-range CVSS scores.',
            'Reports are staged until analysis, source-integrity and source-stability checks succeed, then replaced file by file; this is not an atomic whole-directory transaction.',
            'High-cardinality distributions show the top 20 values; all categories are shown when there are at most 100 string values. Counts use the full dataset.',
            'No URLs/APIs are fetched, no predictive model is trained, and source files are never rewritten.',
        ],
        'files': {name: {k: p[k] for k in ('rows', 'top_level_feature_count', 'profiled_paths_including_nested', 'field_count_per_record', 'sha256', 'bytes', 'records_containing_nonfinite_values')}
                  for name, p in profiles.items()},
        'unique_top_level_features_across_files': len(top_level_keys),
        'unique_cves_across_all_files': len({r['cve'] for rows in datasets.values() for r in rows}),
        'unique_profile_paths_across_files': len(dictionary), 'lineage': lineage(datasets),
        'membership': member, 'original_vs_refreshed': drift, 'correlations': correlations,
        'findings': findings,
        'recommendations': [
            'Define the prediction question, observation cutoff and label horizon before choosing features; keep catalogue membership distinct from confirmed exploitation.',
            'Exclude in_vckev and all KEV-only fields, including their missingness indicators, from membership predictors. Vendor and CWE fields need independently acquired coverage before use.',
            'Build a strict normalization adapter: preserve missingness reasons, decode nested lists, validate booleans and retain source hashes.',
            'Use one row per CVE from one chosen snapshot. Use the other snapshots only for lineage/drift comparisons, not random-split augmentation.',
            'Treat current descriptions, references and exploit evidence as potentially post-outcome information; assess snapshot availability before any temporal experiment.',
            'Inspect shared URLs, duplicate descriptions and repository/vendor overlap across proposed splits. Shared generic advisory URLs alone do not prove duplicate vulnerabilities.',
            'Keep CVSS categories version-specific and investigate missingness by version/status; do not invent zero scores or mix original/current components.',
            'Review the long-tailed reference and exploit-evidence lists before building graphs; deduplicate exact per-CVE URLs and measure node/edge growth without discarding provenance.',
            'Evaluate future models with temporal/grouped splits, PR-AUC and class-aware metrics against the cohort prevalence baseline; fit preprocessing on training records only.',
            'Review pinned upstream preparation-code gaps in the source audit before claiming end-to-end reproducibility. Local integrity does not establish sampling representativeness.',
        ],
        'source_audit_readiness_warnings': audit.get('readiness_warnings', []), 'figures': figures,
        'limitations': ['No independent verification of linked evidence, catalogue completeness, historical feature availability or CVSS score arithmetic.',
                        'Descriptive associations are confounded by cohort selection, publication year and enrichment coverage; they are not causal effects.',
                        'Original-source recovery claims cannot all be reconstructed from these six files alone.'],
    }
    changed = [name for name, p in profiles.items() if hashlib.sha256((source / name).read_bytes()).hexdigest() != p['sha256']]
    report['source_files_changed_during_analysis'] = changed
    if changed:
        raise RuntimeError(f'Source files changed during analysis: {changed}')
    write_pair(output / '00_analysis', report, 'Revision dataset analysis')


if __name__ == '__main__':
    main()
