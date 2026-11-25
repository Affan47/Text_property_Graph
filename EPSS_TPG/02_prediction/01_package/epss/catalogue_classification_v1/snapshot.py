"""Immutable, authenticated full-backup collection and training-role labels.

Downloading a global catalogue is not a held-out evaluation: membership is
joined only to the frozen training allowlists. No raw dataset is adapted here.
"""
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from urllib.parse import urlsplit
import uuid
import zipfile

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / '02_prediction/04_configs/08_catalogue_classification/01_snapshot_protocol.json'
CLI = ROOT / '02_prediction/03_analysis/08_catalogue_classification/01_prepare_catalogue_study.py'
TEST = CLI.with_name('test_catalogue_snapshot.py')
CVE = re.compile(r'CVE-\d{4}-\d{4,}\Z')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def dump(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(dump(value))


def utc():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    require(isinstance(value, str), 'Missing dated backup metadata')
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(dt.tzinfo is not None, 'Naive timestamp is not accepted')
    return dt.astimezone(timezone.utc)


def check_lock(directory):
    directory = Path(directory)
    files = read(directory / 'artifact_lock.json')['files']
    require(bool(files), 'Empty artifact lock')
    for relative, expected in files.items():
        path = directory / relative
        require(directory.resolve() in path.resolve().parents and path.is_file()
                and sha(path) == expected, 'Changed/unsafe artifact: ' + str(path))
    return files


def lock(directory):
    write(directory / 'artifact_lock.json', {'files': {
        str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}})


@contextmanager
def exclusive(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.writer.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another CC01 command owns this output lock') from None
        yield


def pins():
    return {str(p): sha(p) for p in [CONFIG, CLI, TEST, *sorted(Path(__file__).parent.glob('*.py'))]}


def check_pins(values):
    for path, expected in tqdm(values.items(), desc='Verify frozen identities', unit='file', leave=False):
        require(Path(path).is_file() and sha(path) == expected, 'Changed prerequisite: ' + path)


def make_plan(config):
    require(not config['training_enabled'] and not config['test_access_enabled'], 'Scope cannot enable training/test')
    scopes, inputs = {}, {}
    for name, specification in tqdm(config['datasets'].items(), desc='Verify training-only candidate scopes', unit='dataset'):
        directory = ROOT / specification['audit']
        check_lock(directory)
        check_pins(read(directory / 'identity.json')['input_hashes'])
        source = directory / 'training_scope_ids.json'
        ids = read(source)
        require(isinstance(ids, list) and ids == sorted(set(ids)) and all(CVE.fullmatch(x) for x in ids),
                'Invalid training allowlist')
        require(len(ids) == specification['training_candidates'], 'Changed training candidate count')
        scopes[name] = ids
        for path in (source, directory / 'artifact_lock.json', directory / 'identity.json'):
            inputs[str(path)] = sha(path)
    return {'schema': config['schema'], 'specification': config, 'training_ids': scopes,
            'source_pins': pins(), 'prerequisite_pins': inputs,
            'training_enabled': False, 'test_access_enabled': False}


def validate_metadata(payload, cfg, received_at):
    require(isinstance(payload, dict) and payload.get('_meta', {}).get('index') == cfg['index'],
            'Backup metadata is for the wrong index')
    candidates = payload.get('data')
    require(isinstance(candidates, list) and len(candidates) == 1, 'Expected one unambiguous complete-backup descriptor')
    record = candidates[0]
    require(isinstance(record, dict), 'Malformed backup descriptor')
    require(isinstance(record.get('sha256'), str) and re.fullmatch('[a-fA-F0-9]{64}', record['sha256']),
            'Missing authoritative archive SHA-256')
    url = urlsplit(record.get('url', ''))
    require(url.scheme == 'https' and url.hostname in cfg['download_hosts'] and url.port in (None, 443)
            and not url.username and not url.password and not url.fragment, 'Unsafe backup download host or URL')
    require(record.get('filename', '').endswith('.zip'), 'Only documented ZIP backups are supported')
    added, received = timestamp(record.get('date_added')), timestamp(received_at)
    require(timedelta(0) <= received - added <= timedelta(days=cfg['max_backup_age_days']),
            'Backup is future-dated or older than the frozen freshness limit')
    return record


def decode_records(archive, cfg, snapshot_at):
    """Fail closed on unsupported packaging; never skip a malformed member."""
    listed, members, count = {}, [], 0
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        require(0 < len(entries) <= cfg['max_members'], 'Empty/oversized archive directory')
        require(len({i.filename for i in entries}) == len(entries), 'Duplicate ZIP member names')
        require(sum(i.file_size for i in entries) <= cfg['max_uncompressed_bytes'], 'Uncompressed archive exceeds budget')
        for info in tqdm(entries, desc='Validate complete catalogue members', unit='member', leave=False):
            path = PurePosixPath(info.filename)
            require(not path.is_absolute() and '..' not in path.parts and '\\' not in info.filename
                    and not info.flag_bits & 1 and (info.external_attr >> 16) & 0o170000 != 0o120000,
                    'Unsafe/encrypted ZIP member')
            if info.is_dir():
                continue
            require(path.suffix == '.json', 'Unsupported archive member; no partial catalogue release')
            body = z.read(info)  # ZIP checks CRC; no extraction onto the filesystem.
            payload = json.loads(body)
            if isinstance(payload, dict):
                payload = payload.get('data')
            require(isinstance(payload, list) and bool(payload), 'Expected nonempty JSON record array')
            members.append({'name': info.filename, 'sha256': hashlib.sha256(body).hexdigest(), 'records': len(payload)})
            for row in payload:
                require(isinstance(row, dict), 'Malformed catalogue record')
                ids = row.get('cve')
                require(isinstance(ids, list) and ids and all(isinstance(c, str) and CVE.fullmatch(c) for c in ids),
                        'Invalid/missing CVE identity in catalogue')
                require(len(ids) == len(set(ids)), 'Repeated CVE within record')
                date = row.get('date_added')
                require(timestamp(date) <= timestamp(snapshot_at), 'Record added after backup date')
                count += 1
                for cve in ids:
                    require(cve not in listed, 'Duplicate CVE records require explicit review')
                    listed[cve] = date
    require(count > 0 and bool(members), 'No catalogue records; cannot infer any negatives')
    return listed, {'records': count, 'unique_cves': len(listed), 'members': members}


def label_rows(ids, listed=None):
    require(len(ids) == len(set(ids)), 'Repeated training candidate')
    return [{'cve': c, 'label': None if listed is None else int(c in listed),
             'status': 'UNKNOWN' if listed is None else ('LISTED' if c in listed else 'NOT_LISTED'),
             'catalogue_date_added': None if listed is None else listed.get(c)} for c in sorted(ids)]


def publish_labels(directory, plan, listed, receipt_sha=None):
    counts = {}
    for dataset, ids in tqdm(plan['training_ids'].items(), desc='Export training-only labels', unit='dataset'):
        rows = label_rows(ids, listed)
        write(directory / (dataset + '_labels.json'), rows)
        counts[dataset] = {s: sum(r['status'] == s for r in rows) for s in ('LISTED', 'NOT_LISTED', 'UNKNOWN')}
    findings = {'schema': plan['schema'], 'counts': counts, 'receipt_sha256': receipt_sha,
                'label_gate': 'READY_SNAPSHOT_LABELS' if listed is not None else 'BLOCKED_NO_VERIFIED_SNAPSHOT',
                'training_enabled': False, 'test_access_enabled': False,
                'remaining_gates': ['Date-aligned EPSS snapshot and missingness policy',
                                    'Catalogue-answer disclosure screen and predictor/cache lineage',
                                    'Frozen evidence-group-disjoint inner roles and fit settings'],
                'meaning': plan['specification']['interpretation']}
    write(directory / 'findings.json', findings)
    (directory / 'REPORT.txt').write_text(
        'CC01: independent retrospective catalogue classification\n\n' + findings['meaning'] + '\n\n'
        + dump(findings) + '\nNo training, new embeddings, official validation or test label joins.\n'
        'Absence means not listed at this verified backup, never proof of no exploitation.\n'
        'A global source archive is retained privately; only training-role CVEs are joined.\n')


def request_bytes(session, url, path, limit, headers, events, event_name):
    event = {'kind': event_name, 'started_at': utc()}
    events.append(event)
    try:
        with session.get(url, headers=headers, timeout=(15, 120), allow_redirects=False, stream=True) as response:
            event['status_code'] = response.status_code
            require(response.status_code == 200, 'HTTP request failed; outcomes remain UNKNOWN')
            size = 0
            with path.open('xb') as target, tqdm(desc='Download ' + event_name, unit='B', unit_scale=True) as bar:
                os.chmod(path, 0o600)
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    size += len(chunk)
                    require(size <= limit, 'Download exceeds frozen size limit')
                    target.write(chunk)
                    bar.update(len(chunk))
            require(size > 0, 'Empty HTTP response')
            event.update(bytes=size, sha256=sha(path))
    finally:
        event['ended_at'] = utc()


def collect(directory, plan, token, session=None):
    require(bool(token) and '\n' not in token and '\r' not in token, 'Set VC_API_KEY locally; never put it on the CLI')
    cfg = plan['specification']
    attempts = directory / 'attempts'
    attempts.mkdir(exist_ok=True, mode=0o700)
    attempt = attempts / uuid.uuid4().hex
    attempt.mkdir(mode=0o700)
    events = []
    owns_session = session is None
    if owns_session:
        session = requests.Session()
        session.trust_env = False  # Do not forward credentials through ambient proxies/netrc.
    try:
        request_bytes(session, cfg['endpoint'], attempt / 'metadata.json', cfg['max_metadata_bytes'],
                      {'Accept': 'application/json', 'Authorization': 'Bearer ' + token}, events, 'metadata')
        record = validate_metadata(read(attempt / 'metadata.json'), cfg, events[-1]['ended_at'])
        # Deliberately no Authorization header on the signed archive URL.
        request_bytes(session, record['url'], attempt / 'catalogue.zip', cfg['max_archive_bytes'],
                      {'Accept': 'application/zip'}, events, 'archive')
        require(sha(attempt / 'catalogue.zip') == record['sha256'].lower(), 'Archive differs from official SHA-256')
        listed, inventory = decode_records(attempt / 'catalogue.zip', cfg, record['date_added'])
        receipt = {'schema': cfg['schema'], 'endpoint': cfg['endpoint'], 'index': cfg['index'],
                   'snapshot_at': record['date_added'], 'request_events': events,
                   'official_archive_sha256': record['sha256'].lower(), 'inventory': inventory,
                   'plan_sha256': sha(directory / 'plan/plan.json'),
                   'provenance': 'Authenticated official backup descriptor + whole archive SHA-256; not digital signature or proof of exhaustive exploitation coverage'}
        write(attempt / 'receipt.json', receipt)
        publish_labels(attempt, plan, listed, sha(attempt / 'receipt.json'))
        lock(attempt)
        attempt.rename(directory / 'snapshot')
    except Exception as error:
        # Never log response URLs, response bodies, tokens or requests exception strings.
        write(attempt / 'failure.json', {'outcomes': 'UNKNOWN', 'error_type': type(error).__name__, 'events': events})
        raise ValueError('Collection not released. Private attempt retained; all unresolved outcomes stay UNKNOWN. '
                         'Check failure.json HTTP codes or run verify for structural diagnostics.') from None
    finally:
        if owns_session:
            session.close()


def verify_snapshot(directory, plan):
    saved = directory / 'snapshot'
    check_lock(saved)
    receipt = read(saved / 'receipt.json')
    require(receipt['plan_sha256'] == sha(directory / 'plan/plan.json'), 'Snapshot belongs to another plan')
    events = receipt['request_events']
    require(len(events) == 2 and [e['kind'] for e in events] == ['metadata', 'archive']
            and all(e['status_code'] == 200 for e in events), 'Incomplete receipt chain')
    for event, name in zip(events, ('metadata.json', 'catalogue.zip')):
        require(sha(saved / name) == event['sha256'] and (saved / name).stat().st_size == event['bytes'], 'Changed raw response')
    record = validate_metadata(read(saved / 'metadata.json'), plan['specification'], events[0]['ended_at'])
    require(sha(saved / 'catalogue.zip') == record['sha256'].lower() == receipt['official_archive_sha256']
            and record['date_added'] == receipt['snapshot_at'], 'Broken authoritative archive/date chain')
    listed, inventory = decode_records(saved / 'catalogue.zip', plan['specification'], receipt['snapshot_at'])
    require(inventory == receipt['inventory'], 'Archive inventory differs')
    with tempfile.TemporaryDirectory(prefix='cc01-verify-') as tmp:
        stage = Path(tmp)
        publish_labels(stage, plan, listed, sha(saved / 'receipt.json'))
        for path in stage.iterdir():
            require(sha(path) == sha(saved / path.name), 'Derived labels/report do not reproduce')
    return read(saved / 'findings.json')


def run(stage, snapshot_id, verify_only=False, allow_network=False, resume=False):
    require(re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}', snapshot_id), 'Unsafe snapshot ID')
    cfg = read(CONFIG)
    directory = ROOT / cfg['output'] / snapshot_id
    with exclusive(directory):
        if stage == 'plan':
            plan = make_plan(cfg)
            if verify_only:
                check_lock(directory / 'plan')
                require(read(directory / 'plan/plan.json') == plan, 'Plan differs')
            else:
                require(not (directory / 'plan').exists(), 'Plan exists; use --verify-only')
                with tempfile.TemporaryDirectory(prefix='.plan-', dir=directory) as tmp:
                    target = Path(tmp) / 'ready'
                    target.mkdir()
                    write(target / 'plan.json', plan)
                    publish_labels(target, plan, None)
                    lock(target)
                    target.rename(directory / 'plan')
            print('CC01 plan verified/frozen. Labels UNKNOWN; no requests, fitting or test joins. ' + str(directory))
            return
        check_lock(directory / 'plan')
        plan = read(directory / 'plan/plan.json')
        require(plan['source_pins'] == pins() and plan['specification'] == cfg, 'CC01 code/config changed since plan')
        check_pins(plan['prerequisite_pins'])
        if stage == 'collect' and not (directory / 'snapshot').exists():
            require(allow_network, 'Collection requires explicit --allow-network')
            require(resume or not (directory / 'attempts').exists(), 'Prior attempt exists; inspect then use --resume')
            collect(directory, plan, os.environ.get('VC_API_KEY', '').strip())
        else:
            require((directory / 'snapshot').exists(), 'No complete snapshot; collect first. No negatives were released.')
            require(stage != 'collect' or resume, 'Complete snapshot exists; use verify or collect --resume')
        findings = verify_snapshot(directory, plan)
        print('CC01 snapshot and training-only labels reproduced. No training or test joins.')
        print(dump(findings))
