"""Preserve CT08; collect and replay a bounded set of fresh catalogue receipts."""
from collections import Counter
import csv
import hashlib
import os
from pathlib import Path
import re
import time

import requests
from tqdm import tqdm

from ..catalogue_readiness_v1 import audit

ROOT = audit.ROOT
READ, SHA, REQUIRE = audit.READ, audit.SHA, audit.REQUIRE
io = audit.artifact_io
evidence = audit.receipts
CLI = ROOT/'02_prediction/03_analysis/07_fresh_studies/20_verify_catalogue_outcomes.py'
OUTPUT = ROOT/'05_results/08_fresh_studies/16_catalogue_checks_v1'
ATTEMPT_FILE = re.compile(r'^\d{6}\.(receipt|response)\.json$')


def approved_ids(base_ids, additional, scope):
    ids = sorted(set(base_ids) | set(additional))
    REQUIRE(len(ids) <= 40 and set(ids) <= set(scope), 'Bounded training allowlist violated')
    REQUIRE(all(audit.common.CVE.fullmatch(c) for c in ids), 'Invalid request ID')
    return ids


def load_plan(dataset, include_disagreements=False):
    spec = READ(audit.CONFIG); cfg = spec['datasets'][dataset]
    directory = ROOT/spec['output']/cfg['output_name']/'audit'
    with tqdm(total=3, desc='Verify frozen training-only request plan', unit='stage') as bar:
        io.check_lock(directory); bar.update()
        identity = READ(directory/'identity.json')
        io.check_pins(identity['input_hashes']); bar.update()
        plan = READ(directory/'receipt_request_plan.json')
        base_ids = plan['ids']; scope = READ(directory/'training_scope_ids.json')
        REQUIRE(base_ids == sorted(set(base_ids)) and len(base_ids) == 20, 'Expected exactly 20 fixed request IDs')
        extra_pins = {}; additional = []
        if include_disagreements:
            review = OUTPUT/cfg['output_name']/'saved_export_review'
            io.check_lock(review)
            extra_pins = dict(READ(review/'identity.json')['pins']); io.check_pins(extra_pins)
            extra_pins.update({str(p): SHA(p) for p in review.iterdir() if p.is_file()})
            additional = READ(review/'additional_training_request_plan.json')['ids']
        ids = approved_ids(base_ids, additional, scope)
        REQUIRE(hashlib.sha256(audit.common.dumps(scope).encode()).hexdigest() == identity['candidate_ids_sha256'], 'Changed candidate identity')
        REQUIRE(plan['source_sha256'] == cfg['raw_sha256'], 'Request plan/source mismatch')
        with (directory/'cve_readiness.csv').open(newline='') as stream:
            flags = {r['cve']: {'True': True, 'False': False}.get(r['saved_in_vckev'])
                     for r in csv.DictReader(stream) if r['cve'] in ids}
        REQUIRE(set(flags) == set(ids) and all(type(v) is bool for v in flags.values()), 'Invalid saved flags')
        REQUIRE(sum(flags[c] is True for c in base_ids) == 10 and sum(flags[c] is False for c in base_ids) == 10,
                'Expected ten saved true and ten saved false flags in original pilot')
        bar.update()
    pins = dict(identity['input_hashes'])
    pins.update(extra_pins)
    pins.update({str(p): SHA(p) for p in directory.iterdir() if p.is_file()})
    for p in (Path(__file__), Path(__file__).with_name('__init__.py'), CLI,
              CLI.with_name('test_catalogue_checks.py')):
        pins[str(p)] = SHA(p)
    return {'schema': 'ct09_catalogue_checks_v1', 'dataset': dataset, 'ids': ids,
            'include_saved_export_disagreements': include_disagreements,
            'legacy_flags': flags, 'candidate_count': len(scope), 'input_hashes': pins,
            'source_sha256': cfg['raw_sha256'], 'endpoint': evidence.ENDPOINT,
            'query_parameters': {'limit': 100, 'page': 1}, 'timeout_seconds': 30,
            'minimum_interval_seconds': 1, 'max_requests_per_invocation': len(ids),
            'stop_http_codes': [401, 403, 429], 'stop_consecutive_transport_failures': 3,
            'interpretation': 'Fresh check times, not reconstruction of the old snapshot or first exploitation',
            'training_enabled': False, 'holdout_queries_enabled': False,
            'api_documentation': 'https://docs.vulncheck.com/api/v3/indice'}, cfg['output_name']


def validate_files(directory, protocol):
    allowed = {'protocol.json', 'verified_labels.json', 'outcomes.csv', 'findings.json',
               'REPORT.txt', 'artifact_lock.json'}
    for path in directory.rglob('*'):
        REQUIRE(not path.is_symlink(), 'Symlinks are not permitted in receipt outputs')
        if not path.is_file():
            continue
        parts = path.relative_to(directory).parts
        REQUIRE((len(parts) == 1 and parts[0] in allowed)
                or (len(parts) == 3 and parts[0] == 'receipts' and parts[1] in protocol['ids']
                    and ATTEMPT_FILE.fullmatch(parts[2])), 'Unexpected or out-of-scope receipt artifact: '+str(path))


def read_attempts(directory, protocol):
    validate_files(directory, protocol)
    latest = {}
    for cve in protocol['ids']:
        folder = directory/'receipts'/cve
        last_attempt, last_time = 0, None
        for path in sorted(folder.glob('*.receipt.json')):
            r = READ(path)
            state, checked = audit.receipt_state(r, cve, folder)
            REQUIRE(r['legacy_in_vckev'] is protocol['legacy_flags'][cve], 'Changed legacy flag')
            REQUIRE(type(r['attempt']) is int and r['attempt'] > last_attempt
                    and path.name == f"{r['attempt']:06d}.receipt.json", 'Invalid attempt ordering')
            REQUIRE(last_time is None or checked >= last_time, 'Backward receipt chronology')
            REQUIRE(r['request_endpoint'] == protocol['endpoint']
                    and r['request_parameters'] == dict(protocol['query_parameters'], cve=cve), 'Changed request identity')
            REQUIRE(r.get('response_file') in (None, f"{r['attempt']:06d}.response.json"), 'Unexpected response filename')
            latest[cve] = r
            last_attempt, last_time = r['attempt'], checked
    return latest


def next_attempt(folder):
    return max((int(p.name.split('.')[0]) for p in folder.glob('*.json')), default=0)+1


def collect_one(session, token, cve, protocol, folder):
    folder.mkdir(parents=True, exist_ok=True)
    number = next_attempt(folder)
    REQUIRE(number < 1000000, 'Attempt number exceeds the receipt format')
    start = evidence.now(); status = None; body = None; response_file = None; response_sha = None
    params = dict(protocol['query_parameters'], cve=cve)
    try:
        response = session.get(protocol['endpoint'], params=params,
                               headers={'Authorization': 'Bearer '+token, 'Accept': 'application/json'},
                               timeout=protocol['timeout_seconds'], allow_redirects=False)
        try:
            status = response.status_code
            if status == 200:
                path = folder/f'{number:06d}.response.json'
                with path.open('xb') as stream:
                    stream.write(response.content)
                response_file, response_sha = path.name, SHA(path)
                try:
                    body = READ(path)
                except (ValueError, UnicodeError):
                    pass
        finally:
            response.close()
        value, reason, matched = evidence.classify_cve(cve, status, body)
    except requests.RequestException:
        value, reason, matched, status = None, 'NETWORK_ERROR', [], None
    r = {'cve': cve, 'attempt': number, 'legacy_in_vckev': protocol['legacy_flags'][cve],
         'verified_in_vckev': value, 'status': reason, 'http_status': status,
         'request_started_at_utc': start, 'checked_at_utc': evidence.now(),
         'request_endpoint': protocol['endpoint'], 'request_parameters': params,
         'response_file': response_file, 'response_sha256': response_sha, 'matched_records': len(matched),
         'label_date_basis': 'fresh_lookup_not_original_snapshot'}
    audit.receipt_state(r, cve, folder)
    path = folder/f'{number:06d}.receipt.json'
    REQUIRE(not path.exists(), 'Refuse to overwrite an existing receipt')
    evidence.atomic(path, r)
    return r


def reports(protocol, latest):
    rows = []
    for cve in protocol['ids']:
        r = latest.get(cve, {})
        value = r.get('verified_in_vckev')
        rows.append({'cve': cve, 'saved_in_vckev': protocol['legacy_flags'][cve],
                     'verified_state_at_check': 'LISTED' if value is True else 'NOT_LISTED' if value is False else 'UNKNOWN',
                     'verified_target_at_check': value, 'receipt_status': r.get('status', 'NOT_CHECKED'),
                     'checked_at_utc': r.get('checked_at_utc'), 'latest_attempt': r.get('attempt'),
                     'saved_flag_disagreement': value != protocol['legacy_flags'][cve] if value is not None else None})
    counts = dict(Counter(r['verified_state_at_check'] for r in rows))
    for key in ('LISTED', 'NOT_LISTED', 'UNKNOWN'):
        counts.setdefault(key, 0)
    summary = {'schema': protocol['schema'], 'dataset': protocol['dataset'], 'requested_cves': len(rows),
               'training_candidate_cves': protocol['candidate_count'], 'pilot_outcomes': counts,
               'verified_pilot_cves': counts['LISTED']+counts['NOT_LISTED'],
               'candidate_cves_not_verified_by_this_pilot': protocol['candidate_count']-counts['LISTED']-counts['NOT_LISTED'],
               'saved_flag_disagreements': sum(r['saved_flag_disagreement'] is True for r in rows),
               'statuses': dict(Counter(r['receipt_status'] for r in rows)),
               'all_pilot_requests_resolved': counts['UNKNOWN'] == 0,
               'training_enabled': False, 'holdout_queried': False,
               'readiness': 'BOUNDED_COLLECTION_CHECK_ONLY_NOT_A_COHORT_LABEL_RELEASE',
               'limitations': ['Check times describe now, not the original snapshot.',
                               'Not listed is not proof of no exploitation.',
                               'The deterministic balanced pilot is not a prevalence or error-rate sample.',
                               'Disagreements may reflect catalogue changes, not necessarily old collector errors.',
                               'Hashes verify recorded bytes, not independent server authenticity.']}
    lines = [f"CT09 {protocol['dataset']}: bounded training-only VulnCheck checks", '',
             f"Requested CVEs: {len(rows)}; listed: {counts['LISTED']}; not listed: {counts['NOT_LISTED']}; unknown: {counts['UNKNOWN']}.",
             f"Saved-flag disagreements among resolved checks: {summary['saved_flag_disagreements']}.",
             f"Training candidates still unverified by this pilot: {summary['candidate_cves_not_verified_by_this_pilot']}.",
             '', 'CVE | Saved flag | Verified at check | Check time | Response status']
    lines += [f"{r['cve']} | {r['saved_in_vckev']} | {r['verified_state_at_check']} | {r['checked_at_utc']} | {r['receipt_status']}" for r in rows]
    lines += ['', 'Limits and next decision', *summary['limitations'],
              'No old dataset flags, CT08 artifacts, splits or models were changed.',
              'Review all UNKNOWN responses and disagreements before agreeing a full-label collection policy.',
              'For a first new classifier, prefer explicitly retrospective snapshot classification; do not claim future exploitation.',
              'Future-inclusion tasks remain gated by time semantics, follow-up and historical text provenance.',
              'Do not begin training from this bounded pilot.']
    return {'verified_labels.json': rows, 'findings.json': summary,
            'outcomes.csv': audit.csv_text(rows, list(rows[0])), 'REPORT.txt': '\n'.join(lines)+'\n'}


def save_reports(directory, artifacts):
    for name, value in artifacts.items():
        if isinstance(value, str):
            path = directory/name
            temporary = path.with_name('.'+path.name+'.tmp')
            temporary.write_text(value, encoding='utf-8'); temporary.replace(path)
        else:
            evidence.atomic(directory/name, value)
    files = {str(p.relative_to(directory)): SHA(p) for p in sorted(directory.rglob('*'))
             if p.is_file() and p.name != 'artifact_lock.json'}
    evidence.atomic(directory/'artifact_lock.json', {'files': files})


def run(dataset, stage='plan', allow_network=False, resume=False, retry_unknown=False, output_dir=None,
        include_disagreements=False):
    REQUIRE(stage in ('plan', 'collect', 'verify'), 'Unknown stage')
    REQUIRE(not allow_network or stage == 'collect', 'Only collect can access the network')
    REQUIRE(not retry_unknown or (stage == 'collect' and resume), 'Retry requires collect --resume')
    protocol, name = load_plan(dataset, include_disagreements)
    directory = output_dir or OUTPUT/name/('bounded_with_disagreements' if include_disagreements else 'bounded')
    if stage == 'plan':
        print(audit.common.dumps({'dataset': dataset, 'ids': protocol['ids'], 'requested_cves': len(protocol['ids']),
                                 'output': str(directory), 'network_requests': 0, 'training_enabled': False}))
        return
    if stage == 'collect':
        REQUIRE(allow_network, 'Collection requires --allow-network; no requests sent')
        token = os.environ.get('VC_API_KEY')
        REQUIRE(bool(token and token.strip()), 'VC_API_KEY is not set; no requests sent or output created')
    else:
        REQUIRE(directory.exists(), 'No collection to verify; run collect with a local API key first')
    with evidence.writer_lock(directory):
        if directory.exists():
            REQUIRE(stage == 'verify' or resume, 'Existing run requires --resume')
            REQUIRE(READ(directory/'protocol.json') == protocol, 'Changed run identity; use a new output directory')
            if (directory/'artifact_lock.json').exists():
                # A resumed interrupted collection may have new receipts, but cannot change old evidence.
                for path, expected in READ(directory/'artifact_lock.json')['files'].items():
                    if stage == 'collect' and path in ('verified_labels.json', 'outcomes.csv', 'findings.json', 'REPORT.txt'):
                        continue
                    target = directory/path
                    REQUIRE(target.resolve().is_relative_to(directory.resolve()) and SHA(target) == expected,
                            'Changed prior artifact: '+path)
        else:
            directory.mkdir(parents=True)
            evidence.atomic(directory/'protocol.json', protocol)
        latest = read_attempts(directory, protocol)
        sent = 0; stop_reason = None
        if stage == 'collect':
            failures = 0
            with requests.Session() as session:
                for cve in tqdm(protocol['ids'], desc='Check approved training CVEs in VulnCheck', unit='CVE'):
                    if cve in latest and (latest[cve]['verified_in_vckev'] is not None or not retry_unknown):
                        continue
                    REQUIRE(sent < protocol['max_requests_per_invocation'], 'Request limit reached')
                    r = collect_one(session, token, cve, protocol, directory/'receipts'/cve)
                    sent += 1; latest[cve] = r
                    failures = failures+1 if r['status'] == 'NETWORK_ERROR' else 0
                    if r['http_status'] in protocol['stop_http_codes'] or failures >= protocol['stop_consecutive_transport_failures']:
                        stop_reason = r['status']
                        tqdm.write(f"[STOP] {r['status']} HTTP={r['http_status']}; unresolved cases remain UNKNOWN.")
                        break
                    time.sleep(protocol['minimum_interval_seconds'])
        latest = read_attempts(directory, protocol)
        artifacts = reports(protocol, latest)
        io.check_pins(protocol['input_hashes'])
        if stage == 'verify':
            io.check_lock(directory)
            for filename, value in artifacts.items():
                expected = value if isinstance(value, str) else audit.common.dumps(value)
                REQUIRE((directory/filename).read_bytes() == expected.encode(), 'Report does not reproduce: '+filename)
        else:
            save_reports(directory, artifacts)
            io.check_lock(directory)
        counts = artifacts['findings.json']['pilot_outcomes']
        print(f'CT09 {dataset} {stage} complete: {directory}; outcomes={counts}; requests this call={sent}; no training.')
        return {'stop_reason': stop_reason, **artifacts['findings.json']}
