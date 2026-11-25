"""Conservative field reconstruction anchored to the preserved NVD snapshot."""
from datetime import timedelta
import re

from .plan import timestamp


def vector(value, version):
    if not isinstance(value, str):
        return None
    found = re.search(r'(?:CVSS:3\.[01]/)?AV:[NALP]/AC:[LH]/PR:[NLH]/UI:[NR]/S:[UC]/C:[NLH]/I:[NLH]/A:[NLH](?![A-Za-z0-9/])', value)
    if not found:
        return None
    result = found.group()
    return result if result.startswith('CVSS:') else 'CVSS:'+version+'/'+result


def cvss_provider(value, event):
    if not isinstance(value, str):
        return None
    prefix = re.split(r'(?:CVSS:3\.[01]/)?AV:', value, maxsplit=1)[0].strip()
    if prefix.casefold() in ('nist', 'nvd', 'nvd@nist.gov'):
        return 'nvd@nist.gov'
    source = event.get('sourceIdentifier')
    # NIST can edit other providers' metrics; an ambiguous provider is not NVD.
    if source and source != 'nvd@nist.gov':
        return source
    return None


def field_events(events, field, record, asof):
    rows, problems = [], []
    for event in sorted(events, key=lambda e: (timestamp(e['created']), e['cveChangeId'])):
        when = timestamp(event['created'])
        if when > asof:
            continue
        for detail in event['details']:
            typ = detail.get('type', '')
            if field == 'description':
                if typ != 'Description':
                    continue
                old, new = detail.get('oldValue'), detail.get('newValue')
                if any(v is not None and not isinstance(v, str) for v in (old, new)):
                    problems.append('NON_STRING_DESCRIPTION'); continue
            else:
                match = re.fullmatch(r'CVSS V(3\.[01])', typ)
                if not match:
                    continue
                version = match[1]
                if version != str(record.get('cvss_vector_version')):
                    continue
                providers = {cvss_provider(detail[k], event) for k in ('oldValue', 'newValue') if k in detail}
                if None in providers or len(providers) != 1:
                    problems.append('AMBIGUOUS_CVSS_PROVIDER'); continue
                if providers != {record.get('cvss_vector_source')}:
                    continue
                old = vector(detail.get('oldValue'), version)
                new = vector(detail.get('newValue'), version)
                if any(detail.get(k) and v is None for k, v in (('oldValue', old), ('newValue', new))):
                    problems.append('UNPARSED_CVSS_VECTOR'); continue
            action = detail.get('action')
            if action not in ('Added', 'Changed', 'Removed'):
                problems.append('UNHANDLED_ACTION'); continue
            if (action == 'Changed' and (old is None or new is None)
                    or action == 'Added' and new is None or action == 'Removed' and old is None):
                problems.append('MISSING_EVENT_VALUE'); continue
            rows.append({'time': when, 'id': event['cveChangeId'], 'action': action,
                         'old': old if action != 'Added' else None, 'new': new if action != 'Removed' else None})
    return rows, sorted(set(problems))


def reconstruct_field(events, field, record, cutoff):
    saved = record.get('description' if field == 'description' else 'cvss_vector')
    asof = timestamp(record['last_modified_timestamp'])
    rows, problems = field_events(events, field, record, asof)
    base = {'status': None, 'value_at_cutoff': None, 'supported': False,
            'same_as_saved': False, 'support_event_ids': [], 'problems': problems,
            'first_recorded_addition': next((r['time'].isoformat() for r in rows if r['action'] == 'Added'), None),
            'first_recorded_saved_value': next((r['time'].isoformat() for r in rows if r['new'] == saved and saved is not None), None)}
    if problems:
        return {**base, 'status': 'UNRESOLVED_UNPARSED_HISTORY'}
    if asof < cutoff:
        return {**base, 'status': 'UNRESOLVED_SAVED_SNAPSHOT_PRECEDES_CUTOFF'}
    # Multiple same-time field mutations cannot be ordered by arbitrary event IDs.
    stamps = [r['time'] for r in rows]
    if len(set(stamps)) != len(stamps):
        return {**base, 'status': 'UNRESOLVED_SIMULTANEOUS_FIELD_EVENTS'}
    current = saved
    after = [r for r in rows if r['time'] > cutoff]
    for r in reversed(after):
        if current != r['new']:
            return {**base, 'status': 'UNRESOLVED_SNAPSHOT_EVENT_CHAIN_MISMATCH'}
        current = r['old']
    before = [r for r in rows if r['time'] <= cutoff]
    base.update(value_at_cutoff=current, same_as_saved=current == saved and current is not None)
    if before:
        last = before[-1]
        if current != last['new']:
            return {**base, 'status': 'UNRESOLVED_CUTOFF_EVENT_CHAIN_MISMATCH'}
        return {**base, 'status': 'SUPPORTED_BY_DATED_FIELD_EVENT' if current is not None else 'REMOVED_BEFORE_CUTOFF',
                'supported': current is not None, 'support_event_ids': [last['id']]+[r['id'] for r in after]}
    if after and after[0]['action'] == 'Added' and current is None:
        return {**base, 'status': 'NO_ASSIGNMENT_RECORDED_BY_CUTOFF', 'support_event_ids': [after[0]['id']]}
    return {**base, 'status': 'HISTORY_CONSISTENT_BUT_ORIGIN_UNDATED' if current is not None else 'UNRESOLVED_NO_FIELD_HISTORY'}


def reconstruct(events, record, meta):
    cutoff = timestamp(meta['cutoff']); alternate = timestamp(meta['cutoff_168h'])
    description = reconstruct_field(events, 'description', record, cutoff)
    cvss = reconstruct_field(events, 'cvss', record, cutoff)
    alt_d = reconstruct_field(events, 'description', record, alternate)
    alt_c = reconstruct_field(events, 'cvss', record, alternate)
    both = all(r['supported'] and r['same_as_saved'] for r in (description, cvss))
    pub = timestamp(record['published_timestamp'])
    added = timestamp(cvss['first_recorded_addition']) if cvss['first_recorded_addition'] else None
    label = meta.get('label_not_predictor', {})
    crossing = label.get('first_observed_crossing_day') if label.get('label') == 1 else None
    difference = ((pub+timedelta(days=crossing)).date()-added.date()).days if added and crossing is not None else None
    return {'cve': record['cve'], 'history_events': len(events), 'cutoff': meta['cutoff'],
            'snapshot_asof': record['last_modified_timestamp'], 'description': description, 'cvss': cvss,
            'cvss_score_at_cutoff': record.get('cvss_vector_score') if cvss['supported'] and cvss['same_as_saved'] else None,
            'score_rule': 'Reuse stored score only for the identical supported vector; changed vectors are not rescored here',
            'description_cvss_match_supported': both,
            'publication_plus_168h_match_supported': all(r['supported'] and r['same_as_saved'] for r in (alt_d, alt_c)),
            'cutoff_168h': meta['cutoff_168h'], 'description_168h_status': alt_d['status'], 'cvss_168h_status': alt_c['status'],
            'first_recorded_cvss_addition_day': (added.date()-pub.date()).days if added else None,
            'crossing_day_minus_cvss_addition_day': difference,
            'reference_counts_verified': False, 'all_saved_D_inputs_verified': False,
            'all_input_gate_reason': 'Reference counts/flags and model/preprocessor availability at Day 7 remain unverified',
            'label_not_predictor': label}
