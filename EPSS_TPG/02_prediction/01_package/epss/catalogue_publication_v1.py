"""Public EPSS publication diagnostics outside CC02's immutable source pins.

HEAD checks are availability hints, not score verification. Only the original
CC02 collector can release a dated archive, using its unchanged strict parser.
"""
from datetime import date, datetime, timezone
from pathlib import Path
import re
import tempfile
from urllib.parse import urljoin, urlsplit
import uuid

import requests
from tqdm import tqdm

from .catalogue_classification_v1 import snapshot as frozen
from .catalogue_inputs_v1 import runner as inputs

ROOT = frozen.ROOT
OUTPUT = ROOT / '05_results/09_catalogue_classification/04_epss_publication_v1'
CLI = ROOT / '02_prediction/03_analysis/08_catalogue_classification/04_check_epss_publication.py'
TEST = CLI.with_name('test_epss_publication.py')
SCHEMA = 'cc02_epss_publication_v1'
CURRENT = 'https://epss.empiricalsecurity.com/epss_scores-current.csv.gz'


def new_id():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex[:8]


def sources(cfg):
    required = cfg['epss_date']
    frozen.require(date.fromisoformat(required).isoformat() == required, 'Noncanonical required date')
    return {'configured_archive': cfg['epss_url'],
            'publisher_dated': f'https://epss.empiricalsecurity.com/epss_scores-{required}.csv.gz',
            'publisher_current': CURRENT}


def advertised_date(event):
    if event.get('http_status') not in (301, 302, 303, 307, 308):
        return None
    try:
        target = urlsplit(urljoin(CURRENT, event.get('location', '')))
        port = target.port
    except ValueError:
        return None
    if (target.scheme != 'https' or target.hostname != 'epss.empiricalsecurity.com'
            or target.username or target.password or port not in (None, 443)
            or target.query or target.fragment):
        return None
    match = re.fullmatch(r'/epss_scores-(\d{4}-\d{2}-\d{2})\.csv\.gz', target.path)
    if not match:
        return None
    try:
        return date.fromisoformat(match[1]).isoformat()
    except ValueError:
        return None


def summarize(events, required):
    by_source = {e['source']: e for e in events}
    frozen.require(set(by_source) == {'configured_archive', 'publisher_dated', 'publisher_current'}
                   and len(events) == 3, 'Incomplete publication status check')
    github = by_source['configured_archive'].get('http_status')
    publisher = by_source['publisher_dated'].get('http_status')
    current = advertised_date(by_source['publisher_current'])
    if github == 200:
        state = 'CONFIGURED_ARCHIVE_AVAILABLE'
        action = 'Run the unchanged collector; HTTP 200 alone does not verify the internal date or contents.'
    elif publisher == 200:
        state = 'PUBLISHER_AVAILABLE_MIRROR_PENDING'
        action = 'The frozen GitHub source is not ready. Retry later; no unrecorded source substitution.'
    elif github in (403, 404) and publisher in (403, 404) and current and current < required:
        state = 'WAITING_FOR_REQUIRED_DATE'
        action = 'Publisher currently advertises an older date. Retry later; never substitute that older file.'
    else:
        state = 'PUBLICATION_STATUS_UNRESOLVED'
        action = 'Endpoint status is inconclusive. Retry later; no missing CVE scores may be inferred.'
    return {'schema': SCHEMA, 'required_date': required, 'state': state,
            'latest_advertised_date': current, 'configured_archive_http': github,
            'publisher_dated_http': publisher, 'can_attempt_configured_download': github == 200,
            'action': action, 'scores_verified': False, 'scores_downloaded_by_status': False,
            'requests': 3, 'network_method': 'HEAD, redirects disabled',
            'meaning': 'Publication availability only, not an EPSS score or missingness audit.',
            'training': False, 'test_access': False}


def probe(session, name, url):
    event = {'source': name, 'requested_url': url, 'started_at': frozen.utc()}
    try:
        with session.head(url, timeout=(10, 30), allow_redirects=False) as response:
            event['http_status'] = response.status_code
            event['headers'] = {k: response.headers[k] for k in
                                ('Date', 'Content-Type', 'Content-Length', 'Last-Modified', 'ETag')
                                if k in response.headers}
            if name == 'publisher_current' and response.headers.get('Location'):
                candidate = {**event, 'location': response.headers['Location']}
                if advertised_date(candidate):
                    event['location'] = response.headers['Location']
                else:
                    event['unsafe_or_unrecognized_location'] = True
    except requests.RequestException as exc:
        # Do not persist proxy/netrc details or arbitrary exception URLs.
        event.update(http_status=None, error_type=type(exc).__name__)
    event['ended_at'] = frozen.utc()
    return event


def identities(cfg):
    base = ROOT / cfg['output']
    paths = [Path(__file__), CLI, TEST, inputs.CONFIG,
             base / 'freeze/artifact_lock.json', base / 'freeze/roles.json']
    return {str(p): frozen.sha(p) for p in paths}


def check(cfg, name=None):
    name = name or new_id()
    frozen.require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,95}', name), 'Unsafe check name')
    output = OUTPUT / 'checks' / name
    pins = identities(cfg)
    with frozen.exclusive(OUTPUT):
        frozen.require(not output.exists(), 'Status check exists; choose a new name or --verify-only')
        events = []
        with requests.Session() as session:
            session.trust_env = False
            session.headers.update({'User-Agent': 'EPSS-TPG-CC02-publication-check/1', 'Accept-Encoding': 'identity'})
            for source, url in tqdm(sources(cfg).items(), desc='Check dated EPSS publication; no score download', unit='endpoint'):
                events.append(probe(session, source, url))
        findings = summarize(events, cfg['epss_date'])
        findings['checked_at'] = events[-1]['ended_at']
        findings['roles_sha256'] = pins[str(ROOT / cfg['output'] / 'freeze/roles.json')]
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.publication-', dir=output.parent) as tmp:
            stage = Path(tmp) / 'ready'
            stage.mkdir()
            frozen.write(stage / 'receipt.json', {'required_date': cfg['epss_date'], 'events': events})
            frozen.write(stage / 'findings.json', findings)
            frozen.write(stage / 'identity.json', {'source_pins': pins})
            frozen.lock(stage)
            frozen.check_pins(pins)
            stage.rename(output)
    return output, findings


def verify_check(name=None):
    checks = sorted(p for p in (OUTPUT / 'checks').glob('*') if p.is_dir() and (p / 'artifact_lock.json').exists())
    frozen.require(checks, 'No publication check saved. Run --stage status --allow-network first.')
    if name is not None:
        frozen.require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,95}', name), 'Unsafe check name')
    output = OUTPUT / 'checks' / name if name else checks[-1]
    frozen.check_lock(output)
    pins = frozen.read(output / 'identity.json')['source_pins']
    frozen.check_pins(pins)
    receipt = frozen.read(output / 'receipt.json')
    expected = summarize(receipt['events'], receipt['required_date'])
    expected['checked_at'] = receipt['events'][-1]['ended_at']
    cfg = frozen.read(inputs.CONFIG)
    expected['roles_sha256'] = pins[str(ROOT / cfg['output'] / 'freeze/roles.json')]
    frozen.require(expected == frozen.read(output / 'findings.json'), 'Publication diagnostic does not reproduce')
    return output, expected


def collect_checked(cfg, allow_network, verify, resume, name=None):
    base = ROOT / cfg['output']
    if (base / 'epss').exists():
        # Once collected, verification/resume must not issue fresh status requests.
        inputs.run('collect-epss', verify=verify, resume=resume)
        return 0
    if verify:
        print('No EPSS archive to verify. Collection has not succeeded; do not run prepare yet.')
        return 2
    frozen.require(allow_network, 'Publication checking/collection requires --allow-network. No API key is needed.')
    output, findings = check(cfg, name)
    print('Publication receipt: ' + str(output))
    print(frozen.dump(findings))
    if not findings['can_attempt_configured_download']:
        print('No archive released. Re-run this same collect-epss command later; the required date stays fixed.')
        return 2
    # Delegate the download and its content verification without changing any pinned source.
    try:
        inputs.run('collect-epss', allow_network=True, resume=resume)
    except (ValueError, requests.RequestException, OSError, EOFError) as exc:
        record = {'publication_check': str(output), 'finished_at': frozen.utc(), 'collector_succeeded': False,
                  'error_type': type(exc).__name__,
                  'error_message': str(exc) if isinstance(exc, ValueError) else type(exc).__name__,
                  'action': 'Retry the same command; no replacement date or missing scores inferred.'}
        attempt = OUTPUT / 'collector_failures' / new_id()
        with frozen.exclusive(OUTPUT):
            attempt.mkdir(parents=True)
            frozen.write(attempt / 'receipt.json', record)
            frozen.lock(attempt)
        print('CC02 download/content verification did not complete. Failure receipt: ' + str(attempt))
        return 2
    return 0


def run(stage, allow_network=False, verify=False, resume=False, name=None):
    frozen.require(not (verify and allow_network), 'Verification is offline; omit --allow-network')
    cfg, _, _, _ = inputs.context()
    inputs.check_frozen(ROOT / cfg['output'], cfg)
    if stage == 'collect-epss':
        return collect_checked(cfg, allow_network, verify, resume, name)
    if verify:
        output, findings = verify_check(name)
        print('Saved status reproduced, not a new live check: ' + str(output))
    else:
        frozen.require(allow_network, 'Live status checking requires --allow-network')
        output, findings = check(cfg, name)
        print('Publication check saved: ' + str(output))
    print(frozen.dump(findings))
    return 0
