from datetime import datetime, timedelta, timezone
from pathlib import Path

from .. import early_warning_review_v1 as old
from ..early_warning_gap_review_v1 import gap_label

a = old.artifacts
ROOT = old.BASE/'19_day7_input_availability_v1/01_reddit'
CLI = old.CLI.with_name('24_audit_day7_availability.py')
POLICY = {
    'schema': 'day7_public_history_v1', 'endpoint': 'https://services.nvd.nist.gov/rest/json/cvehistory/2.0',
    'request_interval_seconds': 6.5, 'timeout_seconds': 60, 'attempts_per_page': 3,
    'page_size': 2000, 'max_response_bytes': 64*1024*1024, 'max_pages_per_cve': 100,
    'cutoff': 'End of publication Day 7 in UTC; publication timestamp + 168h also audited',
    'naive_nvd_timestamps': 'Interpreted as UTC; original strings retained',
    'label': 'One isolated missing future day may be an assumed negative; never fill scores',
    'history_anchor': 'Saved record at its last_modified_timestamp, not the current live CVE value',
    'strict_support': 'Exact state linked to a dated field event at/before cutoff; no-event origin stays unknown',
    'reference_count_history': 'Not verified by description/CVSS history; all-input gate stays closed',
    'scientific_status': 'Exploratory frozen-score subgroup diagnostic, not prospective model validation',
    'ranking_intervals': {'unit': 'eligible evidence group', 'draws': 20000, 'seed': 6029},
    'test_access': False, 'training': False, 'api_key_required': False,
}


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def iso(value):
    return value.astimezone(timezone.utc).isoformat()


def pins():
    paths = [*Path(__file__).parent.glob('*.py'), CLI, CLI.with_name('test_day7_availability.py'),
             Path(old.__file__), Path(gap_label.__globals__['__file__']), Path(a.__file__)]
    return {**old.PINS, **{str(p): a.sha(p) for p in paths}}


def prepare(verify=False):
    source_pins = pins(); a.check_pins(source_pins)
    a.check_lock(old.PROTOCOL.parent); a.check_lock(old.REPORT)
    protocol = a.read(old.PROTOCOL)
    a.require(all(r['outer_role'] == 'train' for r in protocol['rows'].values()), 'Non-training role')
    used = {c for f in protocol['folds'] for cs in f['main']['full'].values() for c in cs}
    assessment = {c for f in protocol['folds'] for c in f['main']['full']['assessment']}
    raw = {r['cve']: r for r in a.read(old.SOURCE) if r['cve'] in used}
    a.require(len(used) == 13378 and set(raw) == used, 'Frozen cohort changed')
    first = sorted(c for c in assessment if old.whole_v2(timestamp(raw[c]['published_timestamp']).date()))
    a.require(len(first) == 7957, 'Frozen assessment changed')
    records = {}
    for c in sorted(used):
        r = raw[c]; pub = timestamp(r['published_timestamp'])
        a.require(pub.date().isoformat() == r['published_date'], 'Publication date/timestamp mismatch')
        cutoff = datetime.combine(pub.date()+timedelta(days=8), datetime.min.time(), timezone.utc)-timedelta(microseconds=1)
        records[c] = {'cve': c, 'assessment': c in first, 'publication': r['published_timestamp'],
                      'snapshot_asof': r['last_modified_timestamp'], 'cutoff': iso(cutoff),
                      'cutoff_168h': iso(pub+timedelta(days=7)),
                      'component': protocol['component_by_cve'][c]}
        if c in first:
            values = old.series(r)[1]
            label = gap_label(values, isolated=True)
            records[c]['label_not_predictor'] = label
    value = {'policy': POLICY, 'assessment_ids': first, 'cohort_ids': first+sorted(used-set(first)),
             'excluded_protocol_only_cves': sorted(set(protocol['rows'])-used), 'records': records,
             'input_hashes': source_pins,
             'minimum_single_page_hours': {'assessment': len(first)*6.5/3600, 'cohort_total': len(used)*6.5/3600}}
    a.publish(ROOT/'plan', {'plan.json': value, 'REPORT.txt':
        'Day-7 input-history audit. 7,957 assessment CVEs first; 13,378 total original-training CVEs.\n'
        'No training, raw-score changes, official validation/test export, or network requests.\n'
        'Minimum request spacing 6.5 seconds; stop other NVD collectors on the same public IP.\n'
        'Pagination, backoff and retries increase the 14.4h / 24.2h single-page lower estimates.\n'
        'An HTTP-successful history is not proof that every initial field value is historically dated.\n'}, source_pins, verify)
    print('Day-7 audit plan '+('verified' if verify else 'saved')+': '+str(ROOT/'plan'))


def load():
    a.check_lock(ROOT/'plan')
    value = a.read(ROOT/'plan/plan.json')
    a.require(value['policy'] == POLICY and value['input_hashes'] == pins(), 'Plan/code identity changed')
    a.check_pins(value['input_hashes'])
    return value
