#!/usr/bin/env python3
"""Offline CVE publication-to-VulnCheck study. No fetching, training or raw edits."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
from importlib.metadata import version as package_version
import json
from pathlib import Path
import re
import sys
import tempfile
import csv

import numpy as np
from scipy.stats import spearmanr

from analyze_revision_dataset import (
    ROOT, KEV, MISSING, clean, load_file, missing_kind, parse_date,
    index_records, finite_number,
)

HORIZONS = (30, 60, 90)
DAY = timedelta(days=1)
CATEGORIES = ('NEGATIVE', 'SAME_DAY', '0_30_DAYS', '31_60_DAYS',
              '61_90_DAYS', 'OVER_90_DAYS', 'MISSING_DATE', 'NOT_LISTED')
FORBIDDEN = (
    'in_vckev', 'vckev_date_added', 'vckev_updated_at', 'vendor_project',
    'vckev_cwes', 'vckev_known_ransomware_campaign_use',
    'vckev_reported_exploited_by_canaries', 'vckev_reported_exploitation',
    'vckev_xdb', 'cisa_date_added', 'due_date',
)
COMPONENTS = {
    'AV': 'attack_vector', 'AC': 'attack_complexity', 'PR': 'privileges_required',
    'UI': 'user_interaction', 'S': 'scope', 'C': 'confidentiality_impact',
    'I': 'integrity_impact', 'A': 'availability_impact', 'Au': 'authentication_v2',
}
V3_VALUES = {'AV': 'NALP', 'AC': 'LH', 'PR': 'NLH', 'UI': 'NR', 'S': 'UC',
             'C': 'NLH', 'I': 'NLH', 'A': 'NLH'}
V2_VALUES = {'AV': 'LAN', 'AC': 'HML', 'Au': 'MSN', 'C': 'NPC', 'I': 'NPC', 'A': 'NPC'}
SOURCE_FIELDS = ('cve', 'published_date', 'published_timestamp', 'in_vckev',
                 'vckev_date_added', 'last_modified_timestamp', 'vckev_updated_at',
                 'cisa_date_added', 'cvss_vector', 'cvss_vector_score',
                 'cvss_vector_version', 'cvss_vector_severity', 'cvss_vector_source',
                 'vendor_project', 'vckev_known_ransomware_campaign_use',
                 'vckev_reported_exploited_by_canaries')
DEFAULT_BASELINE = ROOT / '05_results/03_dataset_analysis/01_revision_dataset'


def iso(value):
    return value.isoformat().replace('+00:00', 'Z') if value is not None else None


def date_info(value):
    """Only the explicit ISO formats used by the baseline; never locale guessing."""
    reason = missing_kind(value)
    dt = None if reason else parse_date(value)
    precision = 'missing' if reason else 'invalid'
    zone = None
    if dt is not None:
        text = value.strip()
        if len(text) == 10:
            precision = 'date_only'
        else:
            match = re.search(r'\.(\d+)', text)
            precision = f'fractional_seconds_{len(match[1])}' if match else 'seconds'
            if match and len(match[1]) > 6:
                # datetime truncates extra digits; do not silently claim exactness.
                return {'dt': None, 'status': 'unsupported_submicrosecond_precision',
                        'precision': 'unsupported', 'timezone': None, 'midnight': None}
        zone = 'explicit_offset' if re.search(r'(Z|[+-]\d{2}:\d{2})$', text) else 'assumed_UTC'
    return {'dt': dt, 'status': reason or ('valid' if dt else 'invalid_format'),
            'precision': precision, 'timezone': zone,
            'midnight': dt.time().isoformat() == '00:00:00' if dt else None}


def time_bounds(info, midnight_as_day=False):
    dt = info['dt']
    if dt is None:
        return None, None
    uncertain_day = info['precision'] == 'date_only' or (midnight_as_day and info['midnight'])
    return dt, dt + DAY if uncertain_day else dt


def elapsed_microseconds(start, end):
    delta = end - start
    return (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds


def days(start, end):
    return elapsed_microseconds(start, end) / 86400000000 if start is not None and end is not None else None


def distribution(values):
    values = [float(v) for v in values if finite_number(v)]
    if not values:
        return {'count': 0, **dict.fromkeys(('min', 'max', 'mean', 'median', 'std_population',
                                           'p25', 'p50', 'p75', 'p90', 'p95'))}
    arr = np.asarray(values)
    q = np.quantile(arr, [.25, .5, .75, .9, .95], method='linear')
    return {'count': len(values), 'min': float(arr.min()), 'max': float(arr.max()),
            'mean': float(arr.mean()), 'median': float(q[1]),
            'std_population': float(arr.std(ddof=0)),
            **dict(zip(('p25', 'p50', 'p75', 'p90', 'p95'), map(float, q)))}


def category(member, anchor, event):
    if not member:
        return 'NOT_LISTED'
    if anchor is None or event is None:
        return 'MISSING_DATE'
    delta = event - anchor
    if delta < timedelta(0):
        return 'NEGATIVE'
    if event.date() == anchor.date():
        return 'SAME_DAY'
    for horizon, name in ((30, '0_30_DAYS'), (60, '31_60_DAYS'), (90, '61_90_DAYS')):
        if delta <= timedelta(days=horizon):
            return name
    return 'OVER_90_DAYS'


def horizon_label(member, anchor_info, event_info, horizon, cutoff=None, *, day_sensitivity=True):
    """Labels in (publication, publication+h]; return value plus its evidence state.

    A known later first-addition date resolves a zero without assuming follow-up
    for unrelated unlisted records. Midnight sensitivity is deliberately separate
    from the exact stored timestamp arithmetic.
    """
    anchor, event = anchor_info['dt'], event_info['dt']
    if anchor is None:
        return None, 'MISSING_OR_INVALID_ANCHOR'
    if not member and event is not None:
        return None, 'MEMBERSHIP_EVENT_CONFLICT'
    if member:
        if event is None:
            return None, 'MISSING_OR_INVALID_EVENT'
        if event < anchor:
            return None, 'NEGATIVE_REQUIRES_REVIEW'
        if event == anchor:
            return None, 'ALREADY_AT_ANCHOR_REQUIRES_REVIEW'
        if cutoff is not None and event > cutoff:
            return None, 'EVENT_AFTER_CUTOFF_CONFLICT'
        if day_sensitivity:
            a0, a1 = time_bounds(anchor_info)
            e0, e1 = time_bounds(event_info, midnight_as_day=True)
        else:
            a0 = a1 = anchor
            e0 = e1 = event
        lower, upper = e0 - a1, e1 - a0
        boundary = timedelta(days=horizon)
        if lower > timedelta(0) and upper <= boundary:
            return 1, 'EVENT_WITHIN_HORIZON'
        if lower > boundary:
            return 0, 'EVENT_AFTER_HORIZON'
        return None, 'PRECISION_AMBIGUOUS'
    if cutoff is None:
        return None, 'UNKNOWN_OBSERVATION_CUTOFF'
    if cutoff < anchor:
        return None, 'ANCHOR_AFTER_CUTOFF'
    _, latest_anchor = time_bounds(anchor_info) if day_sensitivity else (anchor, anchor)
    if cutoff >= latest_anchor + timedelta(days=horizon):
        return 0, 'FULL_HORIZON_NO_RECORDED_EVENT'
    return None, 'RIGHT_CENSORED'


def vector_components(vector, version):
    """Extract version-specific base tokens, not a score calculator or orig_ fallback."""
    out = {f'current_{name}': None for name in COMPONENTS.values()}
    if missing_kind(vector) or missing_kind(version):
        return out, 'missing_vector_or_version'
    if not isinstance(vector, str) or version not in (2.0, 3.0, 3.1):
        return out, 'unsupported_vector_or_version'
    tokens = vector.split('/')
    if version in (3.0, 3.1):
        if tokens[0] != f'CVSS:{version}':
            return out, 'version_prefix_mismatch'
        tokens = tokens[1:]
        allowed = V3_VALUES
    else:
        if tokens[0] == 'CVSS:2.0':
            tokens = tokens[1:]
        allowed = V2_VALUES
    pairs = {}
    for token in tokens:
        key, sep, value = token.partition(':')
        if not sep or not value or key in pairs:
            return out, 'malformed_or_duplicate_token'
        pairs[key] = value
    if any(key not in pairs or pairs[key] not in set(values) for key, values in allowed.items()):
        return out, 'invalid_or_missing_base_component'
    for key in allowed:
        out[f'current_{COMPONENTS[key]}'] = pairs[key]
    return out, 'valid_base_tokens_only'


def derive_row(raw, cutoff=None, documented_cutoff=None):
    if type(raw.get('in_vckev')) is not bool:
        raise ValueError(f'{raw.get("cve")}: membership must be a native boolean')
    row = {key: clean(raw.get(key)) for key in SOURCE_FIELDS}
    anchor_info = date_info(raw.get('published_timestamp', MISSING))
    event_info = date_info(raw.get('vckev_date_added', MISSING))
    anchor, event = anchor_info['dt'], event_info['dt']
    member = raw['in_vckev']
    notes = []
    for name, info in (('publication', anchor_info), ('vckev_addition', event_info)):
        row[f'{name}_parse_status'] = info['status']
        row[f'{name}_stored_precision'] = info['precision']
        row[f'{name}_timezone_policy'] = info['timezone']
        if info['timezone'] == 'assumed_UTC':
            notes.append(f'{name}_timezone_assumed_UTC')
        if info['dt'] is None and (member or name == 'publication'):
            notes.append(f'{name}_{info["status"]}')
    if event_info['midnight']:
        notes.append('vckev_midnight_value_time_of_day_unverified')
    date_only = date_info(raw.get('published_date', MISSING))['dt']
    if anchor and date_only and anchor.date() != date_only.date():
        notes.append('published_date_timestamp_calendar_mismatch')
    delta = days(anchor, event)
    row.update(prediction_anchor_timestamp=iso(anchor), prediction_anchor_field='published_timestamp',
               target_event_timestamp=iso(event), target_event_field='vckev_date_added',
               observation_cutoff_timestamp=iso(cutoff), documented_snapshot_cutoff_timestamp=iso(documented_cutoff),
               delta_days_exact=delta, delta_hours_exact=None if delta is None else elapsed_microseconds(anchor,event)/3600000000,
               delta_microseconds=None if delta is None else elapsed_microseconds(anchor, event),
               delta_category=category(member, anchor, event),
               same_calendar_day=anchor.date() == event.date() if anchor and event else None,
               calendar_day_difference=(event.date() - anchor.date()).days if anchor and event else None,
               followup_days=days(anchor, cutoff), documented_snapshot_followup_days=days(anchor, documented_cutoff))
    if delta is not None:
        a0, a1 = time_bounds(anchor_info)
        e0, e1 = time_bounds(event_info, midnight_as_day=True)
        row['day_sensitivity_delta_lower_days'] = days(a1, e0)
        row['day_sensitivity_delta_upper_days'] = days(a0, e1)
    else:
        row['day_sensitivity_delta_lower_days'] = row['day_sensitivity_delta_upper_days'] = None
    if member and delta is not None and delta < 0:
        row['temporal_status'] = 'anomalous_or_requires_review'
        row['negative_subtype'] = 'same_calendar_day_earlier_timestamp' if row['same_calendar_day'] else 'earlier_calendar_day'
        notes.append('negative_delta_preserved_targets_withheld')
    elif not member and event is not None:
        row['temporal_status'] = 'membership_event_conflict'
        row['negative_subtype'] = None
        notes.append('unlisted_record_has_event_date')
    elif anchor is None or (member and event is None):
        row['temporal_status'] = 'missing_or_invalid_required_date'
        row['negative_subtype'] = None
    elif member and delta == 0:
        row['temporal_status'] = 'at_anchor_requires_review'
        row['negative_subtype'] = None
    else:
        row['temporal_status'] = 'observed_post_publication_event' if member else 'not_listed_in_snapshot'
        row['negative_subtype'] = None
    for h in HORIZONS:
        value, state = horizon_label(member, anchor_info, event_info, h, cutoff)
        nominal, nominal_state = horizon_label(member, anchor_info, event_info, h, cutoff, day_sensitivity=False)
        assumed, assumed_state = horizon_label(member, anchor_info, event_info, h, documented_cutoff) if documented_cutoff else (None, 'NO_DOCUMENTED_CUTOFF')
        row[f'target_{h}d'], row[f'target_{h}d_status'] = value, state
        row[f'stored_timestamp_target_{h}d'] = nominal
        row[f'stored_timestamp_target_{h}d_status'] = nominal_state
        row[f'documented_snapshot_target_{h}d'] = assumed
        row[f'documented_snapshot_target_{h}d_status'] = assumed_state
        row[f'horizon_{h}d_end_timestamp'] = iso(anchor + timedelta(days=h)) if anchor else None
        _, last_anchor = time_bounds(anchor_info)
        row[f'full_followup_{h}d'] = cutoff >= last_anchor + timedelta(days=h) if cutoff and last_anchor else None
        row[f'documented_full_followup_{h}d'] = documented_cutoff >= last_anchor + timedelta(days=h) if documented_cutoff and last_anchor else None
        row[f'censoring_status_{h}d'] = state
        if 'CONFLICT' in state or state == 'PRECISION_AMBIGUOUS':
            notes.append(f'{h}d_{state.lower()}')
    states = [row[f'target_{h}d_status'] for h in HORIZONS]
    row['censoring_status'] = '|'.join(f'{h}D:{s}' for h, s in zip(HORIZONS, states))
    v90, _ = horizon_label(member, anchor_info, event_info, 90, cutoff)
    row['target_over_90d'] = 1 - v90 if member and v90 is not None else None
    if row['target_over_90d'] is not None:
        notes.append('over90_label_conditions_on_observed_event_not_complement_for_unlisted')
    cisa_info = date_info(raw.get('cisa_date_added', MISSING))
    row['cisa_date_parse_status'] = cisa_info['status']
    row['cisa_date_available'] = cisa_info['dt'] is not None
    row['publication_to_cisa_days'] = days(anchor, cisa_info['dt'])
    row['vckev_to_cisa_days'] = days(event, cisa_info['dt'])
    history = raw.get('url_change_dates') or []
    ref_values = [date_info(d.get('earliest_added_date', MISSING)) for d in history]
    usable_refs = [x['dt'] for x in ref_values if x['dt'] is not None]
    earliest = min(usable_refs) if usable_refs else None
    row['earliest_reference_added_timestamp'] = iso(earliest)
    row['reference_history_entries'] = len(history)
    row['reference_history_unusable_dates'] = len(history) - len(usable_refs)
    row['publication_to_earliest_reference_days'] = days(anchor, earliest)
    row['earliest_reference_to_vckev_days'] = days(earliest, event)
    evidence_dates = [date_info(e.get('date_added'))['dt'] for e in raw.get('vckev_reported_exploitation', [])]
    evidence_dates = [x for x in evidence_dates if x is not None]
    row['earliest_stored_exploitation_evidence_date'] = iso(min(evidence_dates)) if evidence_dates else None
    row['vckev_equals_earliest_evidence_date'] = event == min(evidence_dates) if event and evidence_dates else None
    updated = date_info(raw.get('vckev_updated_at'))['dt']
    row['vckev_updated_equals_added'] = updated == event if updated and event else None
    if updated and event and updated < event:
        notes.append('vckev_update_precedes_added')
    if row['vckev_updated_equals_added']:
        notes.append('vckev_updated_at_equals_added_not_independent_cutoff')
    if cutoff is None:
        notes.append('exact_snapshot_time_and_complete_capture_unconfirmed')
    row.update(publication_year=anchor.year if anchor else None,
               publication_month=anchor.strftime('%Y-%m') if anchor else None,
               publication_quarter=f'{anchor.year}-Q{(anchor.month-1)//3+1}' if anchor else None,
               vckev_addition_month=event.strftime('%Y-%m') if event and member else None)
    row['cve_identifier_year'] = int(raw['cve'].split('-')[1])
    row['cve_year_differs_from_publication_year'] = row['cve_identifier_year'] != anchor.year if anchor else None
    components, vector_status = vector_components(raw.get('cvss_vector'), raw.get('cvss_vector_version'))
    row.update(components)
    row['cvss_component_parse_status'] = vector_status
    if vector_status != 'valid_base_tokens_only':
        notes.append(vector_status)
    row['notes_or_quality_flag'] = notes
    core = ('cve', 'published_date', 'published_timestamp', 'in_vckev', 'vckev_date_added',
            'delta_days_exact', 'delta_hours_exact', 'delta_category', 'target_30d', 'target_60d',
            'target_90d', 'target_over_90d', 'temporal_status', 'censoring_status', 'notes_or_quality_flag')
    return {**{key: row[key] for key in core}, **row}


def feature_policy():
    return {
        'approved_predictor_features': [],
        'candidate_predictor_fields_requiring_asof_evidence': ['description', 'cvss_vector', 'cvss_vector_score',
            'cvss_vector_version', 'cvss_vector_severity'] + [f'current_{n}' for n in COMPONENTS.values()],
        'forbidden_raw_fields': list(FORBIDDEN),
        'forbidden_derivations': ['missingness/presence indicators', 'nodes', 'edges', 'counts',
                                  'embeddings', 'timestamps', 'any transformation of forbidden fields'],
        'analysis_table_is_not_a_predictor_matrix': True,
        'temporal_and_label_columns_role': 'label construction, joins, filtering and post-hoc audit only',
        'future_epss_join': {'key': 'cve', 'upper_bound': 'prediction_anchor_timestamp',
                            'rule': 'Use the latest score whose actual availability time is <= anchor; keep score date and availability provenance. Never nearest-future or bidirectional imputation.',
                            'daily_score_without_release_time': 'Same-day availability is not guaranteed. Use a documented conservative prior-day rule or verify release time; otherwise leave missing.',
                            'no_historical_value': 'Missing, not replaced with a future score.'},
        'historical_readiness': 'Not established: current descriptions and CVSS may include later information.',
    }


def validate_derived(raw_rows, rows, cutoff=None, documented_cutoff=None):
    raw = index_records(raw_rows, 'raw source')
    derived = index_records(rows, 'derived table')
    if raw.keys() != derived.keys() or len(rows) != len(raw_rows):
        raise AssertionError('One-to-one CVE lineage failed')
    for row in rows:
        if row['delta_category'] not in CATEGORIES:
            raise AssertionError('Invalid exclusive category')
        regenerated = derive_row(raw[row['cve']], cutoff, documented_cutoff)
        if regenerated != row:
            raise AssertionError(f'Nonreproducible derivation: {row["cve"]}')
        for prefix in ('target_', 'stored_timestamp_target_', 'documented_snapshot_target_'):
            values = [row[f'{prefix}{h}d'] for h in HORIZONS]
            if values[0] == 1 and values[1:] != [1, 1]:
                raise AssertionError('30-day cumulative monotonicity failed')
            if values[1] == 1 and values[2] != 1:
                raise AssertionError('60-day cumulative monotonicity failed')
            if row['delta_category'] == 'NEGATIVE' and any(v is not None for v in values):
                raise AssertionError('Negative delta was given a binary target')
            for h, value in zip(HORIZONS, values):
                state = row[f'{prefix}{h}d_status']
                if state in ('RIGHT_CENSORED', 'UNKNOWN_OBSERVATION_CUTOFF', 'PRECISION_AMBIGUOUS') and value is not None:
                    raise AssertionError('Unknown/censored target was coerced to binary')
    policy = feature_policy()
    if set(policy['forbidden_raw_fields']) & set(policy['candidate_predictor_fields_requiring_asof_evidence']):
        raise AssertionError('Target-linked predictor candidate')
    return {'rows_validated': len(rows), 'checks': ['unique CVE lineage', 'exclusive categories',
            'all derivations recomputed', 'cumulative labels', 'negative isolation',
            'censoring and precision unknowns', 'predictor allowlist/denylist separation'], 'passed': True}


def grouped(rows, field, *, version_stratified=False):
    groups = defaultdict(list)
    for row in rows:
        value = row.get(field)
        label = str(value) if value is not None else 'MISSING'
        if version_stratified:
            label = f'CVSS {row.get("cvss_vector_version") or "MISSING"} | {label}'
        groups[label].append(row)
    result = []
    for key, group in sorted(groups.items()):
        listed = [r for r in group if r['in_vckev']]
        pos = [r['delta_days_exact'] for r in listed if r['delta_days_exact'] is not None and r['delta_days_exact'] > 0]
        neg = sum(r['delta_category'] == 'NEGATIVE' for r in listed)
        positive_stats = distribution(pos)
        result.append({'group_field': field, 'group': key, 'total_cves': len(group),
            'listed_cves': len(listed), 'listed_percent': 100 * len(listed) / len(group),
            'negative_delta_count': neg, 'negative_percent_of_listed': 100 * neg / len(listed) if listed else None,
            'positive_delta_count': len(pos), 'median_positive_delta_days': positive_stats['median'],
            'mean_positive_delta_days': positive_stats['mean'],
            'p25_positive_delta_days': positive_stats['p25'], 'p75_positive_delta_days': positive_stats['p75'],
            'p90_positive_delta_days': positive_stats['p90'],
            'median_signed_listed_delta_days': distribution([r['delta_days_exact'] for r in listed])['median'],
            'within_30d_stored_count': sum(0 < d <= 30 for d in pos),
            'within_60d_stored_count': sum(0 < d <= 60 for d in pos),
            'within_90d_stored_count': sum(0 < d <= 90 for d in pos),
            'over_90d_stored_count': sum(d > 90 for d in pos),
            'same_calendar_day_count': sum(r['same_calendar_day'] is True for r in listed),
            'documented_full_30d': sum(r['documented_full_followup_30d'] is True for r in group),
            'documented_full_60d': sum(r['documented_full_followup_60d'] is True for r in group),
            'documented_full_90d': sum(r['documented_full_followup_90d'] is True for r in group),
            'documented_right_censored_90d': sum(r['documented_snapshot_target_90d_status'] == 'RIGHT_CENSORED' for r in group)})
    return result


def label_counts(rows):
    result = []
    for policy, prefix in [('primary_precision_aware', 'target_'),
                           ('stored_timestamp_only', 'stored_timestamp_target_'),
                           ('documented_snapshot_sensitivity', 'documented_snapshot_target_')]:
        for h in HORIZONS:
            states = Counter(r[f'{prefix}{h}d_status'] for r in rows)
            result.append({'policy': policy, 'horizon_days': h, 'total': len(rows),
                'positive': sum(r[f'{prefix}{h}d'] == 1 for r in rows),
                'negative': sum(r[f'{prefix}{h}d'] == 0 for r in rows),
                'unknown': sum(r[f'{prefix}{h}d'] is None for r in rows),
                'right_censored': states['RIGHT_CENSORED'],
                'unknown_cutoff': states['UNKNOWN_OBSERVATION_CUTOFF'],
                'negative_delta_review': states['NEGATIVE_REQUIRES_REVIEW'],
                'precision_ambiguous': states['PRECISION_AMBIGUOUS'], 'status_counts': dict(states)})
    return result


def temporal_fields(raw_rows):
    table = []
    fields = ('published_date', 'published_timestamp', 'vckev_date_added', 'vckev_updated_at',
              'last_modified_timestamp', 'cisa_date_added')
    for subset, rows in [('all_cves', raw_rows), ('vckev_listed', [r for r in raw_rows if r['in_vckev']])]:
        for field in fields:
            values = [date_info(r.get(field, MISSING)) for r in rows]
            usable = [v['dt'] for v in values if v['dt'] is not None]
            table.append({'subset': subset, 'field': field, 'records': len(rows), 'usable': len(usable),
                'status_counts': dict(Counter(v['status'] for v in values)),
                'precision_counts': dict(Counter(v['precision'] for v in values)),
                'timezone_counts': dict(Counter(v['timezone'] or 'unavailable' for v in values)),
                'midnight_values': sum(v['midnight'] is True for v in values),
                'min_utc': iso(min(usable)) if usable else None, 'max_utc': iso(max(usable)) if usable else None})
    nested = [d.get('earliest_added_date', MISSING) for r in raw_rows for d in r.get('url_change_dates', [])]
    values = [date_info(x) for x in nested]
    usable = [v['dt'] for v in values if v['dt'] is not None]
    table.append({'subset': 'reference_history_objects_not_cves', 'field': 'earliest_added_date',
                  'records': len(values), 'usable': len(usable), 'status_counts': dict(Counter(v['status'] for v in values)),
                  'precision_counts': dict(Counter(v['precision'] for v in values)),
                  'timezone_counts': dict(Counter(v['timezone'] or 'unavailable' for v in values)),
                  'midnight_values': sum(v['midnight'] is True for v in values),
                  'min_utc': iso(min(usable)) if usable else None, 'max_utc': iso(max(usable)) if usable else None})
    return table


def schema(rows):
    fields = []
    specific_meanings = {
        'target_event_timestamp': 'UTC-normalized stored VulnCheck addition timestamp, not a binary target or a predictor.',
        'target_event_field': 'Source field used for the event: vckev_date_added.',
        'delta_category': 'One exclusive stored-time bracket. NEGATIVE precedes SAME_DAY; NOT_LISTED has no observed event bracket.',
        'delta_microseconds': 'Exact integer event-minus-anchor microseconds, reproducible without floating-point rounding.',
        'cisa_date_available': 'A usable CISA date was supplied; absence does not establish CISA nonmembership.',
    }
    for key in rows[0]:
        if key in specific_meanings:
            meaning = specific_meanings[key]
        elif key in SOURCE_FIELDS:
            meaning = 'Raw source value, retained without date-string rewriting; raw missing/nonfinite values become null in this derivative.'
        elif 'status' in key or key == 'notes_or_quality_flag':
            meaning = 'Audit state; consult label policy and parsing rules. Never a predictor.'
        elif key.startswith('documented_snapshot_target_'):
            meaning = 'Sensitivity-only label under the documented snapshot-day/complete-capture assumption; not the primary target.'
        elif key.startswith('stored_timestamp_target_'):
            meaning = 'Provisional point-timestamp label, ignoring day-level precision sensitivity; unknown if observation is unresolved.'
        elif key.startswith('target_'):
            meaning = 'Primary provisional precision-aware label: 0/1 or null. Over-90 conditions on an observed event, not eventual membership of unlisted records.'
        elif key.startswith('current_'):
            meaning = 'Base component token parsed from current CVSS vector and version; not from orig_ columns. Null if absent/unsupported.'
        elif 'followup' in key:
            meaning = 'Elapsed coverage or full-horizon flag under the named cutoff. Adequate duration does not prove reliable outcomes or historical predictors.'
        elif 'delta_' in key:
            meaning = 'Recorded event minus publication. Exact quantities are arithmetic on normalized stored timestamps, not verified real-world timing precision.'
        elif 'reference' in key:
            meaning = 'Secondary CVE-level summary of stored NVD reference history, not webpage/exploitation time.'
        elif 'cisa' in key:
            meaning = 'Secondary CISA comparison kept separate from VulnCheck; delta direction is encoded in the column name.'
        elif 'timestamp' in key or 'cutoff' in key:
            meaning = 'UTC-normalized temporal metadata for label construction or an as-of join; never an automatic predictor.'
        else:
            meaning = 'Descriptive grouping or quality-check result; not an approved predictor.'
        fields.append({'column': key, 'observed_types': sorted({type(r[key]).__name__ for r in rows}),
                       'null_count': sum(r[key] is None for r in rows), 'meaning': meaning})
    return {'csv_null': 'NA', 'json_null': None, 'csv_booleans': 'true/false',
            'csv_nested_values': 'JSON-encoded cells (e.g. notes_or_quality_flag)',
            'one_row_unit': 'one unique CVE; all source CVEs retained', 'fields': fields}


def summarize(raw_rows, rows, cutoff_info, validation):
    listed = [r for r in rows if r['in_vckev']]
    signed = [r['delta_days_exact'] for r in listed if r['delta_days_exact'] is not None]
    positive = [d for d in signed if d > 0]
    negative = sorted([r for r in listed if r['delta_category'] == 'NEGATIVE'], key=lambda r: r['delta_microseconds'])
    brackets = [{'category': k, 'count': sum(r['delta_category'] == k for r in rows),
                 'denominator': 'all CVEs' if k == 'NOT_LISTED' else 'VulnCheck-listed CVEs'} for k in CATEGORIES]
    for item in brackets:
        denominator = len(rows) if item['category'] == 'NOT_LISTED' else len(listed)
        item['percent'] = 100 * item['count'] / denominator if denominator else None
    delays = []
    for h in (7, 14, 30, 60, 90, 180, 365):
        n = sum(0 < d <= h for d in signed)
        delays.append({'horizon_days': h, 'positive_delta_events_within': n,
                       'percent_of_positive_deltas': 100*n/len(positive) if positive else None,
                       'percent_of_listed': 100*n/len(listed) if listed else None})
    overall = [{'cohort': name, **distribution(values)} for name, values in (
        ('all_usable_listed_signed_deltas', signed), ('strictly_positive_deltas', positive),
        ('negative_deltas', [r['delta_days_exact'] for r in negative]))]
    cvss = []
    for field in ('cvss_vector_version', 'cvss_vector_severity', 'cvss_vector_score'):
        cvss += grouped(rows, field, version_stratified=field != 'cvss_vector_version')
    for name in COMPONENTS.values():
        cvss += grouped(rows, f'current_{name}', version_stratified=True)
    correlations = []
    for version in (2.0, 3.0, 3.1):
        group = [r for r in listed if r['cvss_vector_version'] == version and finite_number(r['cvss_vector_score'])
                 and 0 <= r['cvss_vector_score'] <= 10 and r['delta_days_exact'] is not None and r['delta_days_exact'] > 0]
        x, y = [r['cvss_vector_score'] for r in group], [r['delta_days_exact'] for r in group]
        rho = float(spearmanr(x, y).statistic) if len(group) >= 3 and len(set(x)) > 1 and len(set(y)) > 1 else None
        correlations.append({'cvss_version': version, 'paired_positive_delta_cves': len(group), 'spearman_score_vs_delay': rho})
    clusters = []
    for field in ('publication_year', 'cvss_vector_version', 'cvss_vector_source', 'vendor_project',
                  'vckev_known_ransomware_campaign_use', 'vckev_reported_exploited_by_canaries',
                  'publication_stored_precision', 'vckev_addition_stored_precision', 'vckev_addition_month',
                  'cisa_date_available', 'vckev_equals_earliest_evidence_date'):
        clusters += grouped(listed, field)
    cisa = [r for r in listed if r['vckev_to_cisa_days'] is not None]
    cisa_counts = {'paired_cves': len(cisa), 'vckev_recorded_earlier': sum(r['vckev_to_cisa_days'] > 0 for r in cisa),
                   'cisa_recorded_earlier': sum(r['vckev_to_cisa_days'] < 0 for r in cisa),
                   'same_stored_timestamp': sum(r['vckev_to_cisa_days'] == 0 for r in cisa)}
    cisa_stats = [{'comparison': k, **distribution([r[k] for r in cisa])}
                  for k in ('delta_days_exact', 'publication_to_cisa_days', 'vckev_to_cisa_days')]
    secondary = [{'comparison': k, **distribution([r[k] for r in rows])}
                 for k in ('publication_to_earliest_reference_days', 'earliest_reference_to_vckev_days')]
    counts = label_counts(rows)
    observation = [{'horizon_days': h,
                    'primary_full_duration': sum(r[f'full_followup_{h}d'] is True for r in rows),
                    'primary_short_duration': sum(r[f'full_followup_{h}d'] is False for r in rows),
                    'primary_unknown_duration': sum(r[f'full_followup_{h}d'] is None for r in rows),
                    'documented_full_duration': sum(r[f'documented_full_followup_{h}d'] is True for r in rows),
                    'documented_short_duration': sum(r[f'documented_full_followup_{h}d'] is False for r in rows),
                    'documented_unknown_duration': sum(r[f'documented_full_followup_{h}d'] is None for r in rows),
                    'primary_resolved_event_labels': sum(r['in_vckev'] and r[f'target_{h}d'] is not None for r in rows)} for h in HORIZONS]
    report = {'study': 'Temporal Relationship Between CVE Publication and VulnCheck KEV Inclusion',
        'analysis_version': 1, 'cves': len(rows), 'vckev_listed': len(listed),
        'vckev_not_listed': len(rows)-len(listed), 'usable_listed_timestamp_pairs': len(signed),
        'same_calendar_day_including_negative': sum(r['same_calendar_day'] is True for r in listed),
        'same_stored_timestamp': sum(d == 0 for d in signed),
        'over_one_year_positive_deltas': sum(d > 365 for d in positive),
        'observation_cutoff': cutoff_info, 'overall_temporal_statistics': overall,
        'temporal_bracket_counts': brackets, 'cumulative_stored_event_counts': delays,
        'horizon_label_counts': counts, 'observation_coverage': observation,
        'negative_delta_analysis': {'count': len(negative), 'percent_of_listed': 100*len(negative)/len(listed) if listed else None,
            'subtype_counts': dict(Counter(r['negative_subtype'] for r in negative)),
            'day_sensitivity_can_cross_zero': sum(r['day_sensitivity_delta_upper_days'] > 0 for r in negative),
            'extreme_cases': [{k: r[k] for k in ('cve', 'published_timestamp', 'vckev_date_added', 'delta_days_exact',
                               'negative_subtype', 'cvss_vector_version', 'cvss_vector_source', 'vendor_project')} for r in negative[:20]],
            'group_profiles': clusters,
            'interpretation': 'Same-day midnight normalization can explain uncertain ordering, not confirm it. Earlier-calendar-day cases survive day-level sensitivity. Backfill, delayed NVD publication, event-definition differences or incorrect mappings are hypotheses requiring source provenance, not established causes.'},
        'publication_year_statistics': grouped(rows, 'publication_year'),
        'publication_month_statistics': grouped(rows, 'publication_month'),
        'publication_quarter_statistics': grouped(rows, 'publication_quarter'),
        'addition_month_statistics': grouped(listed, 'vckev_addition_month'),
        'cvss_group_statistics': cvss, 'cvss_score_associations': correlations,
        'cvss_component_parse_counts': dict(Counter(r['cvss_component_parse_status'] for r in rows)),
        'cisa_comparison': {**cisa_counts, 'statistics': cisa_stats,
                            'scope': 'Only paired dates supplied in the VulnCheck-listed subset; not an independent CISA cohort or operational detection lead time.'},
        'reference_history_secondary': {'statistics': secondary,
            'aggregation': 'Earliest usable stored reference-addition timestamp per CVE, one observation per CVE; missing history entries retained as quality counts.',
            'limitations': 'NVD change-history dates are not page publication, exploit publication or exploitation dates. Incomplete pagination and case-insensitive matching remain baseline limitations.'},
        'missing_temporal_fields': temporal_fields(raw_rows),
        'quality_counts': dict(Counter(note for r in rows for note in r['notes_or_quality_flag'])),
        'vckev_updated_equals_added_count': sum(r['vckev_updated_equals_added'] is True for r in listed),
        'vckev_equals_earliest_evidence_date_count': sum(r['vckev_equals_earliest_evidence_date'] is True for r in listed),
        'validation': validation, 'feature_policy': feature_policy(),
        'ready_for_predictive_modeling': False,
        'limitations': [
            'All labels are provisional encodings of stored catalogue dates, not verified first-exploitation or first-disclosure targets.',
            'The documented snapshot day is not an exact export timestamp or proof of complete historical catalogue coverage. Primary zeros for unlisted rows require an explicit, provenance-backed complete-snapshot assertion.',
            'All-midnight catalogue values suggest day-level precision but do not prove rounding. Primary labels test within-day uncertainty; point-timestamp labels are supplied separately.',
            'Membership at one snapshot does not establish historic first entry, removals, re-additions, completeness of negative labels or the meaning of backfilled dates.',
            'Descriptions and CVSS were refreshed in 2026 and are not verified as publication-time inputs. Current CVSS associations are retrospective and noncausal.',
            'The empirical CDF is conditional on observed positive delays, not a survival estimate or a population exploitation probability.',
            'No source pages or exploitation claims were independently verified; no models, EPSS fetches or parameter optimization were run.'],
    }
    report['metadata_semantics_review'] = {
        'updated_equals_added': report['vckev_updated_equals_added_count'],
        'addition_equals_earliest_evidence_date': report['vckev_equals_earliest_evidence_date_count'],
        'denominator': len(listed),
        'nonmatching_evidence_cases': [{k: r[k] for k in ('cve','vckev_date_added','earliest_stored_exploitation_evidence_date')}
                                     for r in listed if r['vckev_equals_earliest_evidence_date'] is False],
        'interpretation': 'These repeated equalities are a provenance concern, not proof of erroneous data or of a specific upstream mapping. Recover the missing enrichment code and catalogue export logs before interpreting the stored date as first observed catalogue insertion. Do not replace the target event with the evidence date.',
    }
    examples = []
    for target in (20,45,75,150):
        choices = [r for r in listed if r['delta_days_exact'] is not None and r['delta_days_exact']>0]
        if choices:
            examples.append(min(choices, key=lambda r:abs(r['delta_days_exact']-target)))
    if negative:
        examples.append(negative[0])
        same_day = [r for r in negative if r['same_calendar_day']]
        if same_day:
            examples.append(same_day[0])
    examples += [r for r in rows if any(r[f'target_{h}d_status']=='PRECISION_AMBIGUOUS' or
                    r[f'documented_snapshot_target_{h}d_status']=='RIGHT_CENSORED' for h in HORIZONS)]
    columns = ('cve','published_timestamp','vckev_date_added','in_vckev','delta_days_exact','delta_category',
               'target_30d','target_60d','target_90d','documented_snapshot_target_30d',
               'documented_snapshot_target_60d','documented_snapshot_target_90d')
    report['actual_cve_examples'] = [{k:r[k] for k in columns} for r in {r['cve']:r for r in examples}.values()]
    report['descriptive_findings'] = [
        f"Usable publication/addition pairs: {len(signed)} of {len(listed)} listed CVEs. Exact stored deltas retain {len(negative)} negative cases.",
        f"Same-calendar-day cases total {report['same_calendar_day_including_negative']}; their sign determines whether they enter NEGATIVE or SAME_DAY. No negative was rounded to zero.",
        f"The stored addition equals the earliest supplied exploitation-evidence date for {report['vckev_equals_earliest_evidence_date_count']} listed CVEs; update equals addition for {report['vckev_updated_equals_added_count']}. This needs mapping/provenance review before first-insertion forecasting claims.",
        f"Of {len(cisa)} paired CISA/VulnCheck dates, VulnCheck is earlier in {cisa_counts['vckev_recorded_earlier']}, CISA earlier in {cisa_counts['cisa_recorded_earlier']}, and they tie in {cisa_counts['same_stored_timestamp']}. This is a selected paired subset, not a catalogue-wide timeliness comparison.",
        'Current CVSS version and metric author are grouping variables, not evidence of publication-time feature availability. Group sample sizes and negative-case rates accompany medians.',
        'Publication months/quarters and recorded addition months are tabulated separately. Concentrations in recorded dates can reflect cohort selection or backfill; ingestion bursts are not established from this snapshot alone.',
    ]
    return report


def write_table(base, rows, columns=None):
    base.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or list(dict.fromkeys(k for row in rows for k in row))
    if not columns:
        columns = ['cve']
    def cell(value):
        if value is None:
            return 'NA'
        if isinstance(value, (dict, list, bool)):
            return json.dumps(value, ensure_ascii=True, allow_nan=False)
        return value
    for suffix, delimiter in (('.csv', ','), ('.txt', '\t')):
        with base.with_suffix(suffix).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter=delimiter)
            writer.writeheader()
            writer.writerows({key: cell(row.get(key)) for key in columns} for row in rows)
    base.with_suffix('.json').write_text(json.dumps(rows, indent=2, ensure_ascii=True, allow_nan=False)+'\n')


def write_report(base, data, introduction):
    payload = json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False)
    base.with_suffix('.json').write_text(payload+'\n')
    base.with_suffix('.txt').write_text(introduction+'\n\nCOMPLETE STRUCTURED DETAILS\n'+payload+'\n')


def research_answers(report):
    pos = next(x for x in report['overall_temporal_statistics'] if x['cohort'] == 'strictly_positive_deltas')
    brackets = {r['category']: r['count'] for r in report['temporal_bracket_counts']}
    labels = report['horizon_label_counts']
    coverage = report['observation_coverage']
    neg = report['negative_delta_analysis']
    return [
        {'question': '1. How many CVEs?', 'answer': f"{report['cves']} unique CVEs; one output row per input CVE."},
        {'question': '2. How many are in VulnCheck KEV?', 'answer': f"{report['vckev_listed']} listed; {report['vckev_not_listed']} not listed in this snapshot. Nonmembership is not never exploited."},
        {'question': '3. How many listed CVEs have usable timestamps?', 'answer': f"{report['usable_listed_timestamp_pairs']} parseable pairs. Parseability is not proof of true time-of-day precision or first-entry semantics."},
        {'question': '4. What is the delay distribution?', 'answer': f"Positive stored deltas: n={pos['count']}, median={pos['median']} days, mean={pos['mean']}, p25={pos['p25']}, p75={pos['p75']}, p90={pos['p90']}, p95={pos['p95']}, min={pos['min']}, max={pos['max']}, population SD={pos['std_population']}. Signed and negative distributions are retained separately."},
        {'question': '5. How many are in each time bracket?', 'answer': f"{brackets}. Same-calendar-day including negatives: {report['same_calendar_day_including_negative']}. NEGATIVE takes precedence over SAME_DAY, so there is no double counting."},
        {'question': '6. How many negative deltas?', 'answer': f"{neg['count']} ({neg['percent_of_listed']}% of listed CVEs); all horizon targets withheld for these records."},
        {'question': '7. What characterizes negative cases?', 'answer': f"{neg['subtype_counts']}. Within-day uncertainty can cross zero for {neg['day_sensitivity_can_cross_zero']} cases; it cannot explain the earlier-calendar-day cases by itself. Group profiles and extreme cases are provided. Causes are unverified."},
        {'question': '8. How many have enough follow-up?', 'answer': json.dumps(coverage)+'. Full elapsed duration is not a guarantee of reliable labels; primary and documented-day coverage are distinct.'},
        {'question': '9. What are the target class counts?', 'answer': json.dumps(labels)+'. Unknown includes missing cutoff, negative review and precision ambiguity, not only right censoring.'},
        {'question': '10. Which horizon is most sensible?', 'answer': f"No horizon is scientifically validated yet. Compare 30/60/90 as predeclared alternatives: 90 days captures more observed post-publication events but needs longer follow-up; 30 days represents a more immediate operational question. Choose based on the intended decision, not apparent model performance or counts alone. {brackets['OVER_90_DAYS']} of {pos['count']} observed positive-delay events fall outside 90 days."},
        {'question': '11. What could invalidate a horizon?', 'answer': 'Unverified capture cutoff/completeness, backfilled or non-first-entry date semantics, negative chronology, day-resolution boundary ambiguity, uncertain historical feature availability, and selected-cohort bias. Longer horizons do not repair these issues.'},
        {'question': '12. Can the table support the next phase?', 'answer': 'Yes as a reproducible audit and label-construction foundation; no as an automatically training-ready matrix. Resolve the stated provenance and historical-availability blockers, select one policy explicitly, exclude NA/review cases as justified, and prevent outcome-derived predictors. No model is implemented here.'},
    ]


def run(args, output):
    source = args.source.resolve()
    raw_rows, decoding = load_file(source)
    index_records(raw_rows, source.name)
    if not raw_rows:
        raise ValueError('Empty temporal cohort')
    baseline_bytes = args.baseline.read_bytes()
    baseline = json.loads(baseline_bytes)
    expected = baseline.get('files', {}).get(source.name, {}).get('sha256')
    if expected != decoding['source_sha256']:
        raise ValueError('Source does not match the verified baseline manifest. Reverify the baseline first.')
    if len(raw_rows) != baseline['files'][source.name]['rows']:
        raise ValueError('Source record count differs from verified baseline')
    readme_bytes = args.source_readme.read_bytes()
    readme_text = readme_bytes.decode('utf-8')
    match = re.search(r'flag reflects the state of the VulnCheck index on the date of the run \((\d{4}-\d{2}-\d{2})\)', readme_text)
    documented = parse_date(match[1]) if match else None
    cutoff = parse_date(args.snapshot_cutoff) if args.snapshot_cutoff else None
    if args.snapshot_cutoff:
        if cutoff is None or not re.search(r'(Z|[+-]\d{2}:\d{2})$', args.snapshot_cutoff) or len(args.snapshot_cutoff) <= 10:
            raise ValueError('--snapshot-cutoff must be a full ISO timestamp with timezone')
        if not args.cutoff_provenance or not args.confirm_complete_snapshot:
            raise ValueError('An asserted cutoff needs --cutoff-provenance and --confirm-complete-snapshot')
    elif args.cutoff_provenance or args.confirm_complete_snapshot:
        raise ValueError('Cutoff provenance/completeness assertion requires --snapshot-cutoff')
    listed_dates = [date_info(r.get('vckev_date_added'))['dt'] for r in raw_rows if r['in_vckev']]
    if cutoff and any(d is not None and d > cutoff for d in listed_dates):
        raise ValueError('Asserted snapshot precedes a stored addition; do not reinterpret this later snapshot as an earlier export')
    cutoff_info = {'primary_cutoff_utc': iso(cutoff),
        'primary_basis': 'operator_asserted_complete_snapshot_not_independently_verified' if cutoff else 'no_exact_complete_snapshot_cutoff_asserted',
        'operator_provenance': args.cutoff_provenance,
        'documented_run_date': match[1] if match else None,
        'documented_day_start_utc': iso(documented),
        'documentary_evidence': match[0] if match else None,
        'source_readme_sha256': hashlib.sha256(readme_bytes).hexdigest(),
        'documented_assumption': 'Sensitivity scenario only: start of the documented run day in UTC, assuming complete snapshot membership. Timezone and exact capture time are undocumented.',
        'not_used_as_cutoff': ['maximum NVD modification date', 'maximum VulnCheck addition/update date', 'filesystem mtime', 'analysis execution date', 'Git commit date'],
        'no_independent_negative_validation': True}
    rows = [derive_row(r, cutoff, documented) for r in sorted(raw_rows, key=lambda r: r['cve'])]
    validation = validate_derived(raw_rows, rows, cutoff, documented)
    report = summarize(raw_rows, rows, cutoff_info, validation)
    report['source'] = {'file': str(source), 'sha256': decoding['source_sha256'],
        'source_git_revision': baseline.get('source_git_revision'),
        'baseline_manifest': str(args.baseline.resolve()), 'baseline_sha256': hashlib.sha256(baseline_bytes).hexdigest(),
        'baseline_generated_utc': baseline.get('generated_utc'), 'baseline_analysis_version': baseline.get('analysis_version'),
        'decoding': decoding}
    report['external_definition_references'] = [
        {'title': 'VulnCheck KEV introduction', 'url': 'https://docs.vulncheck.com/community/vulncheck-kev',
         'purpose': 'Catalogue scope; not validation of this local cohort.'},
        {'title': 'VulnCheck KEV API/schema', 'url': 'https://docs.vulncheck.com/community/vulncheck-kev/schema',
         'purpose': 'Distinguishes catalogue, CISA and evidence fields; does not establish the missing local enrichment mapping.'},
        {'title': 'FIRST CVSS v3.1 specification', 'url': 'https://www.first.org/cvss/v3.1/specification-document',
         'purpose': 'Version-specific base metric vocabulary, not a score audit.'},
        {'title': 'FIRST EPSS FAQ', 'url': 'https://www.first.org/epss/faq',
         'purpose': 'Background for a future historical EPSS join; no scores fetched in this study.'}]
    report['generated_utc'] = datetime.now(timezone.utc).isoformat()
    report['environment'] = {'python': sys.version, 'numpy': np.__version__,
                             'scipy': package_version('scipy'), 'matplotlib': package_version('matplotlib')}
    report['code_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                            for name in ('temporal_analysis.py', 'temporal_plots.py', 'analyze_revision_dataset.py')}
    report['methodology'] = {
        'anchor': 'published_timestamp, NVD-recorded publication; no silent fallback to published_date',
        'event': 'vckev_date_added, stored VulnCheck addition date according to the dataset mapping; no replacement with CISA or reference dates',
        'delta': 'event - anchor; exact arithmetic relative to normalized stored timestamps, with integer microseconds retained',
        'exclusive_category_priority': 'NOT_LISTED if false; otherwise MISSING_DATE for unusable required dates, NEGATIVE if delta<0, SAME_DAY for nonnegative same UTC day, then (0,30], (30,60], (60,90], >90 days',
        'cumulative_targets': 'Event in (anchor, anchor+horizon]; zero-at-anchor is not incident and requires review; all negative deltas withheld',
        'day_precision_sensitivity': 'A date-only value or a midnight catalogue value is treated as possibly anywhere in [00:00,next 00:00). Upper bounds are conservative. Midnight does not prove truncation; nominal point labels remain separately visible.',
        'primary_labels': 'Precision-aware event-derived labels. Unlisted zeros require an explicitly asserted complete-snapshot cutoff. NA is not zero.',
        'documented_labels': 'Separate sensitivity-only labels using the documented snapshot-day start and complete-capture assumption; never silently substituted into primary targets.',
        'zero_after_late_event': 'A stored first-addition event beyond the horizon supports a provisional zero for that horizon, conditional on first-entry semantics; it does not require complete follow-up for unrelated unlisted CVEs.',
        'unknown_not_equal_censored': 'Negative review, precision ambiguity, missing dates and unknown cutoff are distinguished from known insufficient follow-up.',
        'cvss': 'Current vector base tokens stratified by version; no orig_ substitution. v2 authentication is not mapped to v3 privileges, UI or scope. No CVSS score recomputation.',
        'statistics': 'Population standard deviation; NumPy linear empirical quantiles. CVSS rank associations are descriptive, without causal tests or multiple-comparison claims.',
        'over_90d': 'Defined only when an observed, review-eligible event is definitely after 90 days (1) or within 90 (0); unlisted remains unknown.',
    }
    if documented:
        sensitivity = []
        for name, boundary in [('documented_day_start', documented), ('documented_day_end', documented+DAY-timedelta(microseconds=1))]:
            for h in HORIZONS:
                outcomes = [horizon_label(r['in_vckev'], date_info(r.get('published_timestamp')),
                           date_info(r.get('vckev_date_added')), h, boundary) for r in raw_rows]
                sensitivity.append({'scenario': name, 'cutoff_utc': iso(boundary), 'horizon_days': h,
                    'positive': sum(v == 1 for v, _ in outcomes), 'negative': sum(v == 0 for v, _ in outcomes),
                    'unknown': sum(v is None for v, _ in outcomes),
                    'right_censored': sum(s == 'RIGHT_CENSORED' for _, s in outcomes)})
        report['cutoff_day_sensitivity'] = sensitivity
    else:
        report['cutoff_day_sensitivity'] = []
    report['research_questions'] = research_answers(report)
    output.mkdir(parents=True, exist_ok=True)
    write_table(output/'cve_vckev_temporal_analysis', rows)
    negative = sorted([r for r in rows if r['delta_category']=='NEGATIVE'], key=lambda r:r['delta_microseconds'])
    censored = [r for r in rows if any(r[f'{prefix}{h}d_status']=='RIGHT_CENSORED'
                for prefix in ('target_', 'documented_snapshot_target_') for h in HORIZONS)]
    write_table(output/'negative_delta_cases', negative, list(rows[0]))
    write_table(output/'censored_cases', censored, list(rows[0]))
    windows = [{
        'cve': r['cve'], 'prediction_anchor_timestamp': r['prediction_anchor_timestamp'],
        'prediction_horizon_days': h, 'prediction_window_end': r[f'horizon_{h}d_end_timestamp'],
        'target_event_timestamp': r['target_event_timestamp'], 'target': r[f'target_{h}d'],
        'target_status': r[f'target_{h}d_status'], 'in_vckev': r['in_vckev'],
        'documented_snapshot_target': r[f'documented_snapshot_target_{h}d'],
        'documented_snapshot_target_status': r[f'documented_snapshot_target_{h}d_status'],
        'epss_asof_join_latest_allowed_timestamp': r['prediction_anchor_timestamp'],
    } for r in rows for h in HORIZONS]
    write_table(output/'prediction_windows', windows)
    overview_columns = ['cve','published_timestamp','vckev_date_added','in_vckev','delta_days_exact',
                        'delta_category','target_30d','target_60d','target_90d',
                        'documented_snapshot_target_30d','documented_snapshot_target_60d','documented_snapshot_target_90d']
    write_table(output/'01_tables/cve_temporal_overview', [{k:r[k] for k in overview_columns} for r in rows])
    write_table(output/'01_tables/actual_cve_examples', report['actual_cve_examples'])
    for name in ('overall_temporal_statistics', 'temporal_bracket_counts', 'cumulative_stored_event_counts',
                 'horizon_label_counts', 'observation_coverage', 'publication_year_statistics',
                 'publication_month_statistics', 'publication_quarter_statistics', 'addition_month_statistics',
                 'cvss_group_statistics', 'cvss_score_associations', 'missing_temporal_fields', 'cutoff_day_sensitivity'):
        write_table(output/'01_tables'/name, report[name])
    write_table(output/'01_tables/negative_case_groups', report['negative_delta_analysis']['group_profiles'])
    write_table(output/'01_tables/cisa_timing_comparison', report['cisa_comparison']['statistics'])
    write_table(output/'01_tables/cisa_paired_cves', [{k:r[k] for k in ('cve','published_timestamp','vckev_date_added',
                'cisa_date_added','delta_days_exact','publication_to_cisa_days','vckev_to_cisa_days')}
                for r in rows if r['vckev_to_cisa_days'] is not None])
    write_table(output/'01_tables/reference_timing_comparison', report['reference_history_secondary']['statistics'])
    write_report(output/'temporal_table_schema', schema(rows), 'CVE temporal table: column types, missing values and roles')
    write_report(output/'future_feature_policy', feature_policy(), 'Feature policy for a future experiment; no model or approved predictor set is implemented')
    from temporal_plots import make_temporal_figures
    report['figures'] = make_temporal_figures(output/'02_figures', rows, report)
    for path, expected_hash in ((source, decoding['source_sha256']),
                                (args.baseline, hashlib.sha256(baseline_bytes).hexdigest()),
                                (args.source_readme, hashlib.sha256(readme_bytes).hexdigest())):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f'Input changed during analysis: {path}')
    report['source_and_baseline_unchanged'] = True
    report['output_manifest'] = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(output.rglob('*')) if p.is_file()}
    intro = report['study']+'\n'+'='*len(report['study'])+'\n\n'
    intro += '\n\n'.join(x['question']+'\n'+x['answer'] for x in report['research_questions'])
    intro += '\n\nADDITIONAL DESCRIPTIVE FINDINGS\n'+'\n'.join('- '+x for x in report['descriptive_findings'])
    intro += '\n\nIMPORTANT: Read the policy/status columns. The analysis table contains outcome-linked information and MUST NOT be passed wholesale to a model.'
    write_report(output/'temporal_analysis_summary', report, intro)
    print(json.dumps({'rows': len(rows), 'listed': report['vckev_listed'], 'brackets': report['temporal_bracket_counts'],
                      'labels': report['horizon_label_counts'], 'validation': validation}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT.parent/'SummTPGVul/Sec4AI4Sec-EPSS/Data_Files'/f'{KEV}.json')
    parser.add_argument('--baseline', type=Path, default=DEFAULT_BASELINE/'00_analysis.json')
    parser.add_argument('--source-readme', type=Path, default=ROOT.parent/'SummTPGVul/README.md')
    parser.add_argument('--output', type=Path, default=DEFAULT_BASELINE/'06_temporal_vckev')
    parser.add_argument('--snapshot-cutoff', help='Optional provenance-backed complete-snapshot timestamp; no implicit default')
    parser.add_argument('--cutoff-provenance', help='Evidence for the asserted exact snapshot and complete CVE membership check')
    parser.add_argument('--confirm-complete-snapshot', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to((ROOT.parent/'SummTPGVul').resolve()) or output == args.baseline.parent.resolve():
        parser.error('Use a separate derived-output directory, outside the source repository and baseline report root')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.temporal-study-', dir=output.parent) as tmp:
        staged = Path(tmp)
        run(args, staged)
        files = sorted(p for p in staged.rglob('*') if p.is_file())
        for path in files:
            dest = output/path.relative_to(staged)
            if not dest.resolve().is_relative_to(output):
                raise ValueError('Output symlink escapes report directory')
        files.sort(key=lambda p: p.name=='temporal_analysis_summary.json')
        for path in files:
            dest = output/path.relative_to(staged)
            dest.parent.mkdir(parents=True, exist_ok=True)
            path.replace(dest)
    print(f'Temporal study saved to {output}')


if __name__ == '__main__':
    main()
