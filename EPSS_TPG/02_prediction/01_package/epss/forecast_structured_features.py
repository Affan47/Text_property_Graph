"""SF01: audit structured inputs before admitting them to Day-30 forecasting."""
import argparse
from collections import Counter
from datetime import date, datetime, timedelta
import math
from pathlib import Path

from . import forecast_day60_multisource as prep

read, write, require, sha256, tqdm = prep.read, prep.write, prep.require, prep.sha256, prep.tqdm
SCHEMA = 'sf01_day30_structured_audit_v1'
OUTPUT = prep.ROOT/'05_results/02_evaluation/epss_day30_to_day60/structured_availability_v1'
SCRIPTS = prep.ROOT/'02_prediction/03_analysis/01_revision_dataset'
SCRIPT = SCRIPTS/'audit_forecast_structured_features.py'
TEST = SCRIPTS/'test_forecast_structured_features.py'
METRICS = {'AV': 'NALP', 'AC': 'LH', 'PR': 'NLH', 'UI': 'NR', 'S': 'UC',
           'C': 'NLH', 'I': 'NLH', 'A': 'NLH'}
OPTIONAL = {'E': 'XUPFH', 'RL': 'XOTWU', 'RC': 'XURC', 'CR': 'XHML', 'IR': 'XHML', 'AR': 'XHML',
            **{'M'+k: 'X'+v for k,v in METRICS.items()}}
FEATURES = ([f'cvss_{k}_{v}' for k,values in METRICS.items() for v in values]
            + ['cvss_version_3_0', 'cvss_version_3_1', 'cvss_base_score_div10',
               'cvss_score_observed', 'cvss_vector_observed'])
CVSS_FIELDS = ('cvss_vector', 'cvss_vector_version', 'cvss_vector_score')
EXCLUDED = {
    'cvss_vector_severity': 'Redundant with base score; no independent historical timestamp.',
    'cvss_vector_source': 'Scoring-source metadata, not evidence of when the score was available.',
    'orig_cvss_score': 'Earlier collected value, not a verified publication-time value.',
    'orig_cvss_version': 'Earlier collected value, without an authenticated observation time.',
    'n_references_nvd': 'Current reference count; reference removals and historical snapshots are incomplete.',
    'url_change_dates': 'Current link history is incomplete for reconstructing the full Day-30 reference set.',
    'last_modified_timestamp': 'Record-level update time, not a CVSS availability time; audit only.',
    'published_timestamp': 'Anchor metadata only; not a standalone score predictor here.',
    'vuln_status': 'Current NVD status, without a historical version.',
    'epss_version_changed': 'May describe a change after Day 30 in the full 90-day window.',
    'epss_from_to_version': 'Full-window metadata, not a past-only version feature.',
    'epss_version_change_day': 'May disclose a future change; not used.',
    'epss_version_change_date': 'May disclose a future change; not used.',
    'in_vckev': 'Catalogue outcome/enrichment, excluded from the structured predictor contract.',
    'vckev_date_added': 'Catalogue event time; audit/label use only.',
    'vckev_updated_at': 'Catalogue update metadata, not a historical input.',
    'vendor_project': 'VulnCheck-derived field, excluded including its missingness.',
    'vckev_cwes': 'VulnCheck-derived CWEs; no independently dated CWE field is used.',
    'vckev_known_ransomware_campaign_use': 'Exploitation enrichment, excluded including its missingness.',
    'vckev_reported_exploited_by_canaries': 'Exploitation enrichment, excluded including its missingness.',
    'vckev_reported_exploitation': 'Exploitation enrichment, excluded including its missingness.',
    'vckev_xdb': 'Exploitation enrichment, excluded including its missingness.',
    'cisa_date_added': 'Separate catalogue event; excluded from predictors.',
    'due_date': 'Catalogue-derived deadline, excluded from predictors.'}
FILES = {'run_identity.json', 'cve_audit.json', 'retrospective_inputs.json', 'summary.json', 'summary.txt'}


def encode_cvss(fields):
    """Fixed CVSS v3 base vocabulary; no labels, catalogue fields or fitted encodings."""
    require(set(fields) == set(CVSS_FIELDS), 'Unexpected structured fields; use the explicit allowlist')
    vector, version, score = (fields[k] for k in CVSS_FIELDS)
    features = [0.] * len(FEATURES)
    issues = []
    if vector is None or vector == '':
        return features, ['missing_vector']
    if not isinstance(vector, str):
        return features, ['invalid_vector_type']
    tokens = vector.split('/')
    if tokens[0] not in ('CVSS:3.0', 'CVSS:3.1'):
        return features, ['unsupported_or_invalid_version']
    v = tokens[0][5:]
    normalized_version = (f'{version:.1f}' if type(version) in (int,float) and version in (3.0,3.1) else version)
    if normalized_version != v:
        return features, ['version_field_disagrees_with_vector']
    pairs = {}
    for token in tokens[1:]:
        parts = token.split(':')
        if len(parts) != 2 or parts[0] in pairs:
            return features, ['invalid_or_duplicate_metric']
        k, value = parts
        if k not in METRICS and k not in OPTIONAL:
            return features, ['unknown_metric']
        if value not in (METRICS.get(k) or OPTIONAL[k]) or len(value) != 1:
            return features, ['invalid_metric_value']
        pairs[k] = value
    if not set(METRICS) <= set(pairs):
        return features, ['incomplete_base_vector']
    for k in METRICS:
        features[FEATURES.index(f'cvss_{k}_{pairs[k]}')] = 1.
    features[FEATURES.index('cvss_version_'+v.replace('.', '_'))] = 1.
    features[-1] = 1.
    if score is None:
        issues.append('missing_base_score')
    elif isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 10:
        issues.append('invalid_base_score')
    else:
        features[-3], features[-2] = score/10., 1.
    if set(pairs)-set(METRICS):
        issues.append('optional_metrics_not_used')
    return features, issues


def input_row(raw):
    values, issues = encode_cvss({k: raw.get(k) for k in CVSS_FIELDS})
    return {'values': values}, issues


def check_input(row):
    require(set(row) == {'values'}, 'Unexpected model input metadata')
    values = row['values']
    require(isinstance(values, list) and len(values) == len(FEATURES)
            and all(type(v) in (int,float) and math.isfinite(v) and 0 <= v <= 1 for v in values),
            'Invalid structured tensor values')
    categorical = values[:-3] + values[-2:]
    require(all(v in (0,1) for v in categorical), 'Invalid structured masks/categories')
    cursor = 0
    for choices in METRICS.values():
        require(sum(values[cursor:cursor+len(choices)]) == values[-1], 'Wrong CVSS one-hot group')
        cursor += len(choices)
    require(sum(values[cursor:cursor+2]) == values[-1] and values[-2] <= values[-1]
            and (values[-2] == 1 or values[-3] == 0), 'Inconsistent structured presence masks')
    return values


def gate(mode, allow_retrospective=False):
    require(mode in ('historical', 'retrospective'), 'Unknown structured availability policy')
    require(mode != 'historical', 'Historical CVSS inputs blocked: no authenticated field-level Day-30 values. '
            'A unique CVE ID links a value to a vulnerability, not to its past availability.')
    require(allow_retrospective, 'Current CVSS needs --allow-retrospective-structured; '
            'this comparison cannot establish leakage-safe Day-30 prediction.')


def timestamp_date(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00')).date()
    except ValueError:
        return None


def inspect(raw, anchor):
    features, issues = input_row(raw)
    modified = timestamp_date(raw.get('last_modified_timestamp'))
    cutoff = date.fromisoformat(anchor)
    history = raw.get('url_change_dates')
    history = history if isinstance(history, list) else []
    return {'cve': raw['cve'], 'day30_anchor_date': anchor,
            'last_modified_timestamp': raw.get('last_modified_timestamp'),
            'nvd_update_relation': ('unparseable_or_missing' if modified is None else
                'after_day30' if modified > cutoff else 'same_day_as_day30' if modified == cutoff else 'before_day30'),
            'cvss_availability': 'not_verified_at_day30',
            'current_cvss': {k: raw.get(k) for k in CVSS_FIELDS},
            'cvss_quality_flags': issues,
            'usable_retrospective_vector': bool(features['values'][-1]),
            'n_references_nvd_current': raw.get('n_references_nvd'),
            'url_dates_status': raw.get('url_dates_status'),
            'url_history_objects': len(history),
            'url_history_keys': sorted({k for x in history if isinstance(x,dict) for k in x}),
            'historical_reference_count_verified': False}, features


def context(scale):
    require(scale in ('smoke','full'), 'Unknown audit scale')
    directory = prep.OUTPUT/scale
    pins = prep.check_lock(directory, 'preparation_lock.json')
    identity = read(directory/'run_identity.json')
    require(identity['schema'] == prep.SCHEMA and identity['scale'] == scale, 'Wrong Day-30 preparation')
    ids = identity['ids']
    require(set(ids) == {'train','validation'} and not set(ids['train']) & set(ids['validation']), 'Wrong development roles')
    inputs = read(directory/'inputs.json')
    require(set(inputs) == {c for chosen in ids.values() for c in chosen}, 'Input/selection mismatch')
    locator = prep.sources.PREPARED/'preparation_summary.json'
    require(sha256(locator) == identity['source_pins'].get(str(locator)), 'Changed source locator')
    source = Path(read(locator)['source_path'])
    require(sha256(source) == identity['source_pins'].get(str(source)), 'Changed raw dataset')
    for path in (source, locator, Path(__file__).resolve(), SCRIPT, TEST):
        pins[str(path)] = sha256(path)
    collector = prep.ROOT.parent/'SummTPGVul/SummVul/Scrapers/nvd_repo_scraper.py'
    pins[str(collector)] = sha256(collector)
    return ids, inputs, source, pins


def build(scale):
    ids, inputs, source, pins = context(scale)
    selected = {c for chosen in ids.values() for c in chosen}
    # Only identifiers are consulted for unselected records. No test values are adapted.
    rows = [r for r in read(source) if r['cve'] in selected]
    require(len(rows) == len(selected) == len({r['cve'] for r in rows}), 'Duplicate/missing development CVE')
    raw = {r['cve']: r for r in rows}
    audit, numeric = {}, {}
    for split, chosen in ids.items():
        for c in tqdm(chosen, desc='Audit '+split+' structured timing', unit='CVE'):
            anchor = inputs[c]['input_end_date']
            require(anchor == (date.fromisoformat(raw[c]['published_date'])+timedelta(days=30)).isoformat(), 'Wrong anchor')
            audit[c], numeric[c] = inspect(raw[c], anchor)
            audit[c]['split'] = split
            check_input(numeric[c])
    summary = {'schema': SCHEMA, 'scale': scale, 'cohort': {s:len(v) for s,v in ids.items()},
        'historically_verified_cvss_cves': 0,
        'nvd_update_relation': dict(Counter(r['nvd_update_relation'] for r in audit.values())),
        'cvss_versions': dict(Counter(str(r['current_cvss']['cvss_vector_version']) for r in audit.values())),
        'cvss_quality_flags': dict(Counter(k for r in audit.values() for k in r['cvss_quality_flags'])),
        'retrospective_coverage': {s: {'usable_vector': sum(audit[c]['usable_retrospective_vector'] for c in chosen),
            'observed_score': sum(int(numeric[c]['values'][-2]) for c in chosen)} for s,chosen in ids.items()},
        'url_history_objects': sum(r['url_history_objects'] for r in audit.values()),
        'url_history_keys': sorted({k for r in audit.values() for k in r['url_history_keys']}),
        'url_dates_status': dict(Counter(str(r['url_dates_status']) for r in audit.values())),
        'fields': [{ 'field':k, 'present_non_null':sum(r.get(k) is not None for r in rows),
                    'decision': ('retrospective_only' if k in CVSS_FIELDS else 'excluded'),
                    'reason': ('Current snapshot, no field-level historical observation evidence.' if k in CVSS_FIELDS else EXCLUDED[k])}
                   for k in (*CVSS_FIELDS, *EXCLUDED)],
        'examples': [audit[c] for c in (['CVE-2013-10001'] if 'CVE-2013-10001' in ids['train'] else ids['train'][:1])],
        'feature_names': FEATURES, 'feature_count': len(FEATURES),
        'historical_training_allowed': False, 'test_inspection': False,
        'references': ['https://www.first.org/cvss/v3.1/specification-document',
                       'https://nvd.nist.gov/developers/vulnerabilities'],
        'limitations': [
            'Unique CVE identity establishes record linkage, not historical feature availability.',
            'A later NVD update does not prove CVSS itself changed. Earlier updates would not prove availability either.',
            'orig_* means an earlier collected value, without a verified collection or metric creation time.',
            'Current link histories cannot reconstruct reference removals or the complete earlier reference set.',
            'Current CVSS can support an explicitly retrospective sensitivity comparison only.',
            'Descriptions/summaries in the existing graph experiments are also retrospective.',
            'No individual held-out test feature values, labels or predictions are analyzed.',
            'No current score is backdated and no raw source or existing run is modified.']}
    identity = {'schema': SCHEMA, 'scale':scale, 'ids':ids, 'pinned_files':pins,
                'feature_names':FEATURES, 'anchor_day':30, 'test_inspection':False}
    return {'run_identity.json':identity, 'cve_audit.json':audit, 'retrospective_inputs.json':numeric,
            'summary.json':summary, 'summary.txt':report(summary)}


def report(s):
    lines = ['SF01: Which structured values can be used at Day 30?', '',
        f"Development CVEs: {s['cohort']}. No held-out feature analysis or training.",
        'A value can belong to the correct CVE and still have been added after the prediction cutoff.',
        f"CVEs with verified historical CVSS values: {s['historically_verified_cvss_cves']}",
        f"NVD update date compared with Day 30: {s['nvd_update_relation']}",
        f"Current CVSS versions: {s['cvss_versions']}", f"Current usable CVSS: {s['retrospective_coverage']}",
        f"CVSS quality flags: {s['cvss_quality_flags']}", '',
        'Field | Non-null development CVEs | Use | Why']
    lines.extend(f"{r['field']} | {r['present_non_null']} | {r['decision']} | {r['reason']}" for r in s['fields'])
    lines += ['', 'Real training example(s):']
    for e in s['examples']:
        lines += [f"{e['cve']}: Day 30 = {e['day30_anchor_date']}; saved NVD update = {e['last_modified_timestamp']}.",
                  f"Saved current CVSS: {e['current_cvss']}",
                  'This does not tell us which CVSS was available at Day 30.']
    lines += ['', 'Model preparation:',
        f"{s['feature_count']} numeric inputs: base-vector categories, version, score/10 and score/vector presence masks.",
        'Missing/invalid vectors have zero features and an explicit absent-vector mask. CVEs are not dropped.',
        'Historical mode is blocked, not silently converted to all-missing CVSS training.',
        'Retrospective mode requires an explicit CLI acknowledgement and has its own matched control.',
        'No CWE, exploit flag, current reference count or full-window EPSS version flag is admitted.', '',
        *s['limitations'], '']
    return '\n'.join(lines)


def verify(directory, artifacts=None):
    from .forecast_matched_multisource import check_files
    check_files(directory, 'audit_lock.json', SCHEMA, FILES)
    identity = read(directory/'run_identity.json')
    prep.sources.check_pins(identity['pinned_files'])
    if artifacts is not None:
        for name,value in artifacts.items():
            stored = (directory/name).read_text() if name.endswith('.txt') else read(directory/name)
            require(stored == value, 'Audit does not reproduce: '+name)
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', choices=('smoke','full'), default='smoke')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    artifacts = build(args.scale)
    directory = OUTPUT/args.scale
    if args.verify_only:
        verify(directory, artifacts)
        print(f'SF01 {args.scale} audit reproduced. No training or test feature analysis.')
        return
    require(not directory.exists(), 'Audit exists; use --verify-only')
    directory.mkdir(parents=True)
    for name,value in artifacts.items():
        if name.endswith('.txt'):
            (directory/name).write_text(value, encoding='utf-8')
        else:
            write(directory/name,value)
    prep.sources.check_pins(artifacts['run_identity.json']['pinned_files'])
    write(directory/'audit_lock.json', {'schema':SCHEMA, 'files':{n:sha256(directory/n) for n in sorted(FILES)}})
    print(f'SF01 {args.scale} audit saved to {directory}. Historical CVSS blocked; no training or test feature analysis.')
