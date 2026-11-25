"""Serial public GETs with immutable raw receipts and fail-closed pagination."""
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from tqdm import tqdm
from . import plan

a = plan.a
STORE = plan.ROOT/'collection'


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.writing')
    with temporary.open('xb') as stream:
        stream.write(value if isinstance(value, bytes) else a.dump(value).encode())
        stream.flush(); os.fsync(stream.fileno())
    # Publish atomically without replacing an existing receipt.
    os.link(temporary, path)
    temporary.unlink()


def validate_page(payload, cve, start, expected_total=None):
    a.require(payload.get('format') == 'NVD_CVEHistory' and payload.get('version') == '2.0', 'Unexpected API format')
    a.require(type(payload.get('startIndex')) is int and payload['startIndex'] == start, 'Wrong page start')
    total = payload.get('totalResults'); count = payload.get('resultsPerPage')
    changes = payload.get('cveChanges')
    a.require(type(total) is int and total >= 0 and type(count) is int and count >= 0, 'Invalid page counts')
    a.require(isinstance(changes, list) and len(changes) == count, 'Truncated page count')
    a.require(start+count <= total and (count > 0 or total == 0), 'Empty/stalled pagination')
    a.require(expected_total is None or total == expected_total, 'History total changed during pagination')
    ids = []
    for row in changes:
        event = row['change']
        a.require(event['cveId'] == cve and isinstance(event['details'], list), 'Wrong CVE/event schema')
        plan.timestamp(event['created'])
        a.require(isinstance(event['cveChangeId'], str) and event['cveChangeId'], 'Missing event ID')
        ids.append(event['cveChangeId'])
    a.require(len(ids) == len(set(ids)), 'Duplicate event ID within page')
    return total, changes


def read_complete(cve):
    directory = STORE/cve
    marker = directory/'complete.json'
    if not marker.exists():
        return None, {}
    receipt = a.read(marker); pins = {str(marker): a.sha(marker)}
    a.require(receipt['cve'] == cve, 'Receipt CVE mismatch')
    events, total, start, ids = [], None, 0, set()
    for row in receipt['pages']:
        body = directory/row['body']; meta = directory/row['receipt']
        a.require(body.resolve().is_relative_to(directory.resolve()) and meta.resolve().is_relative_to(directory.resolve()), 'Unsafe receipt path')
        a.require(a.sha(body) == row['body_sha256'] and a.sha(meta) == row['receipt_sha256'], 'Raw receipt changed')
        info = a.read(meta)
        expected_url = plan.POLICY['endpoint']+'?'+urlencode({'cveId': cve, 'startIndex': start, 'resultsPerPage': plan.POLICY['page_size']})
        a.require(info['http_status'] == 200 and info['url'] == expected_url and info['body_sha256'] == row['body_sha256'], 'Invalid transport receipt')
        total, page = validate_page(a.read(body), cve, start, total)
        for item in page:
            event = item['change']
            a.require(event['cveChangeId'] not in ids, 'Duplicate event across pages')
            ids.add(event['cveChangeId']); events.append(event)
        start += len(page)
        pins[str(body)] = row['body_sha256']; pins[str(meta)] = row['receipt_sha256']
    a.require(total is not None and start == total == receipt['total_events'], 'Incomplete receipt pagination')
    return events, pins


@contextmanager
def exclusive():
    STORE.mkdir(parents=True, exist_ok=True)
    with (STORE/'.collector.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('A collector/report already owns this collection') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def retry_delay(headers, attempt):
    value = headers.get('Retry-After') or headers.get('retry-after')
    try:
        seconds = float(value)
    except (ValueError, TypeError):
        try:
            seconds = (parsedate_to_datetime(value)-datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError):
            seconds = 0
    return max(30*2**attempt, seconds)


def cooldown(seconds):
    clock = STORE/'.last_request.json'
    value = a.read(clock) if clock.exists() else {'epoch': time.time()}
    value['not_before_epoch'] = max(value.get('not_before_epoch', 0), time.time()+seconds)
    temp = clock.with_suffix('.writing'); temp.write_text(a.dump(value)); temp.replace(clock)


class PublicClient:
    def __init__(self, max_requests=None):
        self.used = 0; self.maximum = max_requests

    def get(self, url, directory):
        if self.maximum is not None and self.used >= self.maximum:
            raise StopIteration('Requested network budget exhausted; rerun with --resume')
        clock = STORE/'.last_request.json'
        if clock.exists():
            previous = a.read(clock)
            delay = max(previous['epoch']+plan.POLICY['request_interval_seconds'], previous.get('not_before_epoch', 0))-time.time()
            if delay > 600:
                raise StopIteration(f'Server cooldown still active for {delay:.0f}s; resume after it expires')
            time.sleep(max(0, delay))
        # Persist the rate clock before the request, including across process restarts.
        temp = clock.with_suffix('.writing')
        temp.write_text(a.dump({'epoch': time.time()})); temp.replace(clock)
        self.used += 1
        started = datetime.now(timezone.utc).isoformat()
        body, status, headers, error = b'', None, {}, None
        try:
            request = Request(url, headers={'User-Agent': 'EPSS-TPG-day7-history-audit/1.0', 'Accept': 'application/json'})
            with urlopen(request, timeout=plan.POLICY['timeout_seconds']) as response:
                status = response.status; headers = dict(response.headers)
                body = response.read(plan.POLICY['max_response_bytes']+1)
        except HTTPError as exc:
            status = exc.code; headers = dict(exc.headers)
            error = type(exc).__name__
            try:
                body = exc.read(plan.POLICY['max_response_bytes']+1)
            except (TimeoutError, OSError) as read_error:
                error += '; body read failed: '+type(read_error).__name__
        except (URLError, TimeoutError, OSError) as exc:
            error = type(exc).__name__+': '+str(exc)
        metadata = {'url': url, 'started_at': started, 'finished_at': datetime.now(timezone.utc).isoformat(),
                    'http_status': status, 'headers': headers, 'error': error, 'body_sha256': sha_bytes(body),
                    'bytes': len(body), 'credentials_sent': False}
        write_new(directory/'body.json', body); write_new(directory/'receipt.json', metadata)
        a.require(len(body) <= plan.POLICY['max_response_bytes'], 'Oversized response retained, not accepted')
        return metadata, body


def fetch(cve, client):
    directory = STORE/cve; directory.mkdir(parents=True, exist_ok=True)
    attempts = directory/'attempts'; attempts.mkdir(exist_ok=True)
    number = max([int(p.name) for p in attempts.iterdir() if p.name.isdigit()]+[0])+1
    run = attempts/f'{number:06d}'
    start, total, pages, ids = 0, None, [], set()
    try:
        for page_number in range(plan.POLICY['max_pages_per_cve']):
            url = plan.POLICY['endpoint']+'?'+urlencode({'cveId': cve, 'startIndex': start, 'resultsPerPage': plan.POLICY['page_size']})
            for retry in range(plan.POLICY['attempts_per_page']):
                folder = run/f'page_{page_number:04d}_try_{retry:02d}'
                info, body = client.get(url, folder)
                if info['http_status'] == 200:
                    total, entries = validate_page(json.loads(body), cve, start, total)
                    for item in entries:
                        identifier = item['change']['cveChangeId']
                        a.require(identifier not in ids, 'Repeated event across pages')
                        ids.add(identifier)
                    pages.append({'body': str((folder/'body.json').relative_to(directory)),
                                  'receipt': str((folder/'receipt.json').relative_to(directory)),
                                  'body_sha256': info['body_sha256'], 'receipt_sha256': a.sha(folder/'receipt.json')})
                    start += len(entries)
                    break
                if info['http_status'] in (400, 401, 404):
                    raise ValueError('Network request failed: '+str(info['http_status'])+' '+str(info['error']))
                delay = retry_delay(info['headers'], retry)
                cooldown(delay)
                if delay > 600:
                    raise StopIteration(f'Server requests {delay:.0f}s cooldown; persisted before pausing')
                if retry+1 == plan.POLICY['attempts_per_page']:
                    raise ValueError('Network retries exhausted; cooldown applies before the next CVE')
                tqdm.write(f'[BACKOFF] {cve}: HTTP {info["http_status"]}; waiting {delay:.1f}s')
                time.sleep(delay)
            if start == total:
                write_new(directory/'complete.json', {'cve': cve, 'total_events': total, 'pages': pages,
                          'completed_at': datetime.now(timezone.utc).isoformat(), 'transport': 'public HTTPS; no API key'})
                read_complete(cve)
                return True
        raise ValueError('Pagination limit exceeded; no complete receipt published')
    except (ValueError, KeyError, TypeError) as exc:
        write_new(run/'failure.json', {'cve': cve, 'error': str(exc), 'pages_received': len(pages), 'events_received': start})
        tqdm.write('[UNRESOLVED] '+cve+': '+str(exc))
        return False


def collect(scope, download=False, resume=False, limit=None, max_requests=None):
    p = plan.load()
    a.require(download, 'Network is opt-in: use --download')
    a.require(limit is None or limit > 0, 'limit must be positive')
    a.require(max_requests is None or max_requests > 0, 'max-requests must be positive')
    ids = p['assessment_ids'] if scope == 'assessment' else p['cohort_ids']
    a.require(resume or not any(STORE.glob('CVE-*/attempts')), 'Collection exists; use --resume')
    print('[PUBLIC API] No key. Stop other NVD collectors sharing this IP first. One request per >=6.5s.', flush=True)
    with exclusive():
        pending = []
        for cve in tqdm(ids, desc='Verify completed history receipts', unit='CVE'):
            if read_complete(cve)[0] is None:
                pending.append(cve)
        if limit is not None:
            pending = pending[:limit]
        client = PublicClient(max_requests); failures = 0
        try:
            for cve in tqdm(pending, desc='Fetch public NVD history; no training', unit='CVE'):
                failures = 0 if fetch(cve, client) else failures+1
                a.require(failures < 5, 'Five consecutive unresolved CVEs: circuit breaker stopped requests')
        except StopIteration as exc:
            print('[PAUSED] '+str(exc))
        print(f'Collection pass ended after {client.used} requests. Completed CVEs are immutable; failed CVEs remain unknown.')


def status(scope):
    p = plan.load(); ids = p['assessment_ids'] if scope == 'assessment' else p['cohort_ids']
    counts = {'complete_nonempty': 0, 'complete_empty_history': 0, 'unresolved_or_not_requested': 0}
    for cve in tqdm(ids, desc='Check saved NVD receipt integrity', unit='CVE'):
        events, _ = read_complete(cve)
        key = 'unresolved_or_not_requested' if events is None else ('complete_nonempty' if events else 'complete_empty_history')
        counts[key] += 1
    print(a.dump({'scope': scope, 'cves': len(ids), **counts,
                  'minimum_remaining_request_hours': counts['unresolved_or_not_requested']*6.5/3600,
                  'note': 'Transport coverage only, not historical field availability.'}))
