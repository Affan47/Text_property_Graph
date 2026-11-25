"""Read-only, CPU Reddit dataset audit. Never imports training or upstream collectors."""
import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import date, timedelta
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from urllib.parse import urlparse

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
COMMIT = '70615ef948b6063ea47dc5479174e1b2e6a1eb54'
BRANCH = 'new_data_collection'
REMOTE = 'https://github.com/observatio/SummTPGVul.git'
DEFAULT_DATA = Path('/tmp/tpg_reddit_audit_70615ef')
DEFAULT_OUTPUT = ROOT / '05_results/03_dataset_analysis/02_reddit_dataset/01_readiness_v1'
REL = 'Sec4AI4Sec-EPSS/Data_Files/reddit_data'
FINAL = 'reddit_posts_vc_kev_epss.json'
COLLECTOR = 'Sec4AI4Sec-EPSS/Data_Files/fetch_epss_data.py'
KEV = 'Sec4AI4Sec-EPSS/Data_Files/fetch_vulncheck_data.py'
SCRAPER = 'SummVul/Scrapers/reddit_scraper_new.py'
SELECTED = ['README.md', '.gitattributes', SCRAPER, COLLECTOR, KEV,
            *[REL + '/' + n for n in ('reddit_posts.csv', 'reddit_posts.json',
               'reddit_posts_vc_kev.json', 'reddit_posts_vc_kev_epss_1.zip')]]
OLD = ROOT.parent / 'SummTPGVul/Sec4AI4Sec-EPSS/Data_Files/epss_recollected_2026-09-30_smp.json'
OLD_ROLES = ROOT / '05_results/05_vulncheck_study/18_final_protocol_v1/cohort_ids.json'
CF_IDS = ROOT / '05_results/04_consolidated_studies/02_continuous_epss/stage1_v1/preflight/full/run_identity.json'
DAY = re.compile(r'\d{4}-\d{2}-\d{2}\Z')
CVE = re.compile(r'\bCVE-\d{4}-\d{4,}(?!\d)', re.I)
OFFSETS = (-30, -14, -7, -1, 0, 1, 7, 14, 30)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def dumps(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + '\n'


def read(path):
    def bad(value):
        raise ValueError('Nonstandard JSON constant: ' + value)
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, 'Duplicate JSON key: ' + k)
            out[k] = v
        return out
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f, parse_constant=bad, object_pairs_hook=pairs)


def iso_day(value):
    if not isinstance(value, str) or not DAY.fullmatch(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def score_ok(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def post_key(p):
    return (p.get('platform'), p.get('post'), p.get('date'))


def fingerprint(p):
    return digest(post_key(p))


def align(scores, posted):
    """Date-level coverage only; a daily EPSS release is not a known intraday event."""
    return {str(i): scores.get((posted + timedelta(days=i)).isoformat()) for i in OFFSETS}


def dump_rows(path, rows):
    rows = sorted(rows, key=lambda r: dumps(r))
    with path.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as zipped:
        with io.TextIOWrapper(zipped, encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=sorted(rows[0]) if rows else ['empty'], lineterminator='\n')
            w.writeheader()
            w.writerows(rows)


def fetch_lfs_http(repo):
    """Use the standard LFS batch API; keep credentials and signed URLs out of logs."""
    import requests
    pending = []
    for rel in SELECTED:
        p = repo / rel
        with p.open('rb') as f:
            prefix = f.read(200)
        if prefix.startswith(b'version https://git-lfs'):
            fields = dict(line.split(' ', 1) for line in prefix.decode().splitlines())
            pending.append((p, fields['oid'].split(':')[1], int(fields['size'])))
    if not pending:
        return
    result = subprocess.run(['git','credential','fill'], input='protocol=https\nhost=github.com\n\n',
                            capture_output=True, text=True, check=True, timeout=30)
    credential = dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    response = requests.post(REMOTE + '/info/lfs/objects/batch',
        auth=(credential.get('username',''), credential.get('password','')),
        json={'operation':'download','transfers':['basic'],
              'objects':[{'oid':oid,'size':size} for _,oid,size in pending]},
        headers={'Accept':'application/vnd.git-lfs+json'}, timeout=60)
    require(response.status_code == 200, 'LFS batch request failed: HTTP ' + str(response.status_code))
    objects = {obj['oid']:obj for obj in response.json()['objects']}
    for path, oid, size in pending:
        obj = objects[oid]
        require('error' not in obj and obj['size'] == size, 'LFS object unavailable: ' + path.name)
        action = obj['actions']['download']
        require(urlparse(action['href']).scheme == 'https', 'Insecure LFS download URL')
        # Do not reuse GitHub Authorization on a redirected object-storage host.
        with requests.get(action['href'], headers=action.get('header',{}), stream=True, timeout=(30,120)) as r:
            require(r.status_code == 200, 'LFS download failed for ' + path.name + ': HTTP ' + str(r.status_code))
            temp = path.with_suffix(path.suffix + '.partial')
            with temp.open('wb') as f, tqdm(total=size, desc=path.name, unit='B', unit_scale=True) as progress:
                for chunk in r.iter_content(1024*1024):
                    f.write(chunk); progress.update(len(chunk))
            require(temp.stat().st_size == size and sha(temp) == oid, 'LFS size/hash mismatch: ' + path.name)
            temp.replace(path)


def fetch(base):
    """Pin a separate sparse set of files; never switch or pull the project submodule."""
    base.mkdir(parents=True, exist_ok=True)
    repo = base / 'repository'
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1', GIT_TERMINAL_PROMPT='0')
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    def git(*args, capture=False):
        cmd = ['git'] + (['-C', str(repo)] if repo.exists() else []) + list(args)
        return subprocess.run(cmd, env=env, check=True, timeout=600,
                              stdout=subprocess.PIPE if capture else None, text=True).stdout
    if not repo.exists():
        subprocess.run(['git', 'clone', '--depth', '1', '--filter=blob:none', '--no-checkout',
                        '--single-branch', '--branch', BRANCH, REMOTE, str(repo)],
                       env=env, check=True, timeout=600)
    require(git('rev-parse', 'HEAD', capture=True).strip() == COMMIT,
            'Branch HEAD differs from reviewed commit; use a new versioned audit')
    # Only this isolated checkout is modified. Do not overwrite a locally changed source.
    for rel in SELECTED:
        p = repo / rel
        if p.exists():
            with p.open('rb') as f:
                if not f.read(80).startswith(b'version https://git-lfs'):
                    continue
        git('-c', 'filter.lfs.process=', '-c', 'filter.lfs.smudge=', '-c', 'filter.lfs.required=false',
            'checkout', COMMIT, '--', rel)
    fetch_lfs_http(repo)
    for rel in tqdm(SELECTED, desc='Verify Git/LFS source identity', unit='file'):
        stored = subprocess.check_output(['git', '-C', str(repo), 'show', COMMIT + ':' + rel], env=env)
        p = repo / rel
        if stored.startswith(b'version https://git-lfs'):
            fields = dict(line.split(' ', 1) for line in stored.decode().splitlines())
            require(p.stat().st_size == int(fields['size']) and sha(p) == fields['oid'].split(':')[1],
                    'Git LFS identity mismatch: ' + rel)
        else:
            require(sha(p) == hashlib.sha256(stored).hexdigest(), 'Git blob mismatch: ' + rel)
    extracted = base / 'extracted'
    extracted.mkdir(exist_ok=True)
    with zipfile.ZipFile(repo / REL / 'reddit_posts_vc_kev_epss_1.zip') as z:
        expected = {FINAL, FINAL + '.epss-report.json', FINAL + '.epss-report.txt', FINAL + '.epss-state.json'}
        require(set(z.namelist()) == expected, 'Unexpected archive contents')
        for name in tqdm(sorted(expected), desc='Extract or verify archive members', unit='file'):
            p = extracted / name
            h = hashlib.sha256()
            with z.open(name) as src:
                if p.exists():
                    for block in iter(lambda: src.read(1024 * 1024), b''):
                        h.update(block)
                    require(sha(p) == h.hexdigest(), 'Changed extracted member: ' + name)
                else:
                    temp = p.with_suffix(p.suffix + '.partial')
                    with temp.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
                    temp.replace(p)
    manifest = {'schema': 'reddit_source_v1', 'commit': COMMIT, 'branch': BRANCH, 'remote': REMOTE,
                'files': {str(p.relative_to(base)): sha(p) for p in
                          [*[repo / r for r in SELECTED], *sorted(extracted.iterdir())]}}
    target = base / 'download_manifest.json'
    if target.exists():
        require(read(target) == manifest, 'Download manifest changed')
    else:
        target.write_text(dumps(manifest))
    print('Pinned Reddit files ready:', base)


def source_checks(repo):
    """Execute only AST-extracted collector functions with offline fake responses."""
    tree = ast.parse((repo / KEV).read_text())
    names = {'query_kev_entry', 'fetch_vckev_data'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    class ApiError(Exception):
        def __init__(self, status):
            self.status, self.reason = status, 'offline simulated failure'
    class Client:
        def __init__(self, code):
            self.code = code
        def index_vulncheck_kev_get(self, **kwargs):
            raise ApiError(self.code)
    from types import SimpleNamespace
    import contextlib
    result = {}
    for code in (401, 403, 429, 500):
        env = dict(ApiException=ApiError, get_vc_client=lambda: Client(code), _retry_after=lambda e: None,
                   urllib3=SimpleNamespace(exceptions=SimpleNamespace(HTTPError=OSError)),
                   time=SimpleNamespace(sleep=lambda _: None), json=json, tqdm=lambda x, **kw: x,
                   save=lambda entries, path: None,
                   CONFIG=dict(VC_API_TOKEN='offline-test', MAX_ATTEMPTS=1, MAX_RATE_LIMIT_WAITS=0,
                               REQUEST_TIMEOUT=1, RATE_LIMIT_SLEEP=0, CHECKPOINT_EVERY=50))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(repo / KEV), 'exec'), env)
        with tempfile.TemporaryDirectory(prefix='reddit-offline-', dir='/tmp') as tmp:
            p = Path(tmp) / 'cve.json'
            p.write_text('[{"cve":"CVE-2022-46908"}]')
            with contextlib.redirect_stdout(io.StringIO()):
                out = env['fetch_vckev_data'](str(p))
        result[str(code)] = {'saved_in_vckev': out[0]['in_vckev'], 'request_succeeded': False}
    scrape = ast.parse((repo / SCRAPER).read_text())
    return {'offline_error_to_false_tests': result,
            'collector_function_lines': {n.name: n.lineno for n in nodes},
            'scraper_function_lines': {n.name: n.lineno for n in scrape.body if isinstance(n, ast.FunctionDef)},
            'interpretation': 'Reproduces current code behavior, not proof these failures occurred during collection.'}


def old_ids_and_posts():
    old = read(OLD)
    ids = {r['cve'] for r in old}
    texts = defaultdict(set)
    descriptions = defaultdict(set)
    for r in old:
        if r.get('description'):
            descriptions[digest(r['description'])].add(r['cve'])
        for p in r.get('social_media_posts', []):
            if p.get('platform', '').lower() == 'reddit' and p.get('post'):
                texts[digest(p['post'])].add(r['cve'])
    roles = read(OLD_ROLES)
    require(isinstance(roles, dict), 'Unknown old cohort ID layout')
    # Only IDs/roles are used; no old evaluation predictions or metrics are opened.
    return ids, texts, descriptions, roles


def lineage(repo, records):
    final = {r['cve']: r for r in records}
    result = {}
    for name in ('reddit_posts.json', 'reddit_posts_vc_kev.json'):
        rows = read(repo / REL / name)
        require(len(rows) == len({r['cve'] for r in rows}), 'Duplicate intermediate CVE')
        changed, example = 0, []
        for r in tqdm(rows, desc='Check source preservation: ' + name, unit='CVE'):
            different = [k for k, v in r.items() if r['cve'] not in final or final[r['cve']].get(k) != v]
            if different:
                changed += 1
                if len(example) < 5:
                    example.append({'cve': r['cve'], 'fields': different})
        result[name] = {'records': len(rows), 'same_ids': set(final) == {r['cve'] for r in rows},
                        'changed_existing_records': changed, 'examples': example}
        del rows
    expected = Counter((r['cve'], *post_key(p)) for r in records for p in r['social_media_posts'])
    actual = Counter()
    fields, invalid = [], Counter()
    csv.field_size_limit(16 * 1024 * 1024)
    with (repo / REL / 'reddit_posts.csv').open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        for r in tqdm(reader, desc='Match CSV post/date associations', unit='row'):
            actual[(r['cve'], r['source'].lower(), r['text'], r['created_at'])] += 1
            invalid[r.get('vuln_status') or 'missing'] += 1
    result['reddit_posts.csv'] = {'rows': sum(actual.values()), 'columns': fields, 'vuln_status': invalid,
        'associations_missing_from_final': sum((actual - expected).values()),
        'associations_added_in_final': sum((expected - actual).values())}
    return result


def audit(base, output, verify=False):
    require(output.exists() if verify else not output.exists(),
            'Use --verify-only for an existing audit; do not overwrite saved reports')
    manifest = read(base / 'download_manifest.json')
    require(manifest['commit'] == COMMIT, 'Unexpected dataset commit')
    for rel, expected in tqdm(manifest['files'].items(), desc='Verify downloaded source hashes', unit='file'):
        require(sha(base / rel) == expected, 'Changed source: ' + rel)
    repo, ext = base / 'repository', base / 'extracted'
    pins = {str(p): sha(p) for p in (OLD, OLD_ROLES, CF_IDS, Path(__file__),
            Path(__file__).with_name('test_audit_reddit.py'), base / 'download_manifest.json',
            ROOT / '00_documentation/06_papers/02_pdf/Reddit dataset report.pdf')}
    print('[READ] Strict JSON parsing; raw inputs remain read-only')
    records = read(ext / FINAL)
    require(len(records) == len({r['cve'] for r in records}), 'Duplicate CVE IDs')
    report = read(ext / (FINAL + '.epss-report.json'))
    state = read(ext / (FINAL + '.epss-state.json'))
    require(state['identity']['cves'] == [r['cve'] for r in records], 'State order differs')
    require(state['identity']['days'] == 91, 'Unexpected history window')
    sources = lineage(repo, records)
    old_ids, old_posts, old_desc, roles = old_ids_and_posts()
    counts, status_counts, relation, epss_counts = Counter(), Counter(), Counter(), Counter()
    field_types, field_presence, post_fields = defaultdict(Counter), Counter(), Counter()
    posts, text_index, description_index = {}, defaultdict(set), defaultdict(set)
    score_coverage, years, windows = Counter(), defaultdict(Counter), defaultdict(Counter)
    dates, post_dates, cve_rows, event_rows = [], [], [], []
    daily_coverage = defaultdict(Counter)
    missing_day0_only = 0
    failed_dates, date_issue_examples = Counter(), []
    matched_old_text_cves, matched_old_description_cves = set(), set()
    for idx, r in enumerate(tqdm(records, desc='Audit CVEs, posts and daily EPSS alignment', unit='CVE')):
        cve = r['cve']
        require(CVE.fullmatch(cve), 'Invalid CVE ID: ' + cve)
        pub = iso_day(r['published_date'])
        require(pub is not None, 'Invalid publication day: ' + cve)
        dates.append(pub.isoformat())
        require(r.get('in_vckev') in (True, False) and type(r['in_vckev']) is bool, 'Nonbinary saved KEV flag')
        counts['listed'] += r['in_vckev']
        counts['publication_on_or_after_epss_launch'] += pub >= date(2021, 4, 14)
        counts['publication_strictly_after_epss_launch'] += pub > date(2021, 4, 14)
        year = years[str(pub.year)]
        year['cves'] += 1
        year['listed'] += r['in_vckev']
        for k, v in r.items():
            field_types[k][type(v).__name__] += 1
            field_presence[k] += v is not None
        scores, flags = r['epss_scores'], r['epss_score_status']
        expected_dates = [(pub + timedelta(days=i)).isoformat() for i in range(91)]
        require(set(scores) == set(flags) == set(expected_dates), 'EPSS date window mismatch: ' + cve)
        require([scores[d] for d in expected_dates] == state['scores'][idx] and
                [flags[d] for d in expected_dates] == state['statuses'][idx], 'State scores/status mismatch: ' + cve)
        n = 0
        for day_index, d in enumerate(expected_dates):
            value, flag = scores[d], flags[d]
            require(value is None or score_ok(value), 'Invalid EPSS number: ' + cve + ' ' + d)
            require((value is not None) == (flag in ('OBSERVED', 'EXISTING_UNVERIFIED')),
                    'Status/value contradiction: ' + cve + ' ' + d)
            n += value is not None
            status_counts[flag] += 1
            daily_coverage[str(day_index)][flag] += 1
            if flag in ('DOWNLOAD_ERROR', 'INVALID_SNAPSHOT', 'PENDING'):
                failed_dates[d] += 1
        score_coverage[n] += 1
        coverage = 'complete_91' if n == 91 else 'none' if n == 0 else 'partial'
        missing_day0_only += n == 90 and scores[expected_dates[0]] is None
        epss_counts[coverage] += 1
        year[coverage] += 1
        same_id = cve in old_ids
        counts['old_cve_overlap'] += same_id
        d_hash = digest(r['description'])
        description_index[d_hash].add(cve)
        if d_hash in old_desc:
            matched_old_description_cves.add(cve)
        counts['duplicate_reference_cves'] += len(r['all_sources_urls']) != len(set(r['all_sources_urls']))
        counts['publication_timestamp_date_mismatch'] += str(r.get('published_timestamp', ''))[:10] != pub.isoformat()
        counts['nvd_modified_after_day30'] += str(r.get('last_modified_timestamp', ''))[:10] > (pub + timedelta(days=30)).isoformat()
        local, usable, on_by_30 = Counter(), Counter(), 0
        for j, p in enumerate(r['social_media_posts']):
            post_fields.update(p.keys())
            counts['post_associations'] += 1
            key, text_hash = fingerprint(p), digest(p.get('post'))
            local[key] += 1
            text_index[text_hash].add(key)
            stored = posts.setdefault(key, {'cves': set(), 'text_hash': text_hash,
                         'date': p.get('date'), 'platform': p.get('platform'), 'text': p.get('post')})
            stored['cves'].add(cve)
            text = p.get('post') or ''
            referenced = set(x.upper() for x in CVE.findall(text))
            counts['associations_without_target_id_in_text'] += cve not in referenced
            counts['associations_with_multiple_cve_mentions'] += len(referenced) > 1
            if text_hash in old_posts:
                matched_old_text_cves.add(cve)
            day = iso_day(p.get('date'))
            event = dict(cve=cve, post_index=j, text_sha256=text_hash, fingerprint=key,
                         stored_post_date=p.get('date'), published_date=pub.isoformat(),
                         exact_post_id_available=bool(p.get('id') or p.get('post_id')),
                         historical_text_verified=False)
            if day is None:
                relation['invalid_or_missing_date'] += 1
                event.update(days_after_publication='', date_relation='invalid_or_missing_date')
            else:
                post_dates.append(day.isoformat())
                delta = (day - pub).days
                category = 'before_publication' if delta < 0 else 'same_day' if delta == 0 else 'after_publication'
                relation[category] += 1
                on_by_30 += day <= pub + timedelta(days=30)
                values = align(scores, day)
                event.update(days_after_publication=delta, date_relation=category,
                             **{'epss_offset_' + k: v for k, v in values.items()})
                for h in (1, 7, 14, 30):
                    endpoint = all(values[str(k)] is not None for k in (-h, 0, h))
                    complete = all(scores.get((day + timedelta(days=k)).isoformat()) is not None
                                   for k in range(-h, h + 1))
                    event[f'endpoints_{h}d_present'] = endpoint
                    event[f'complete_{h}d_each_side'] = complete
                    usable[f'endpoints_{h}d'] += endpoint
                    usable[f'complete_{h}d'] += complete
                    windows[str(h)]['associations_with_endpoints'] += endpoint
                    windows[str(h)]['associations_with_complete_daily_window'] += complete
                event['epss_on_post_date_present'] = values['0'] is not None
                counts['associations_with_epss_on_post_date'] += values['0'] is not None
                counts['associations_outside_publication_91day_window'] += delta < 0 or delta > 90
                if delta < 0 and len(date_issue_examples) < 8:
                    date_issue_examples.append({'cve': cve, 'publication': pub.isoformat(),
                        'post_date': day.isoformat(), 'text_excerpt': text[:350], 'delta_days': delta})
            event_rows.append(event)
        counts['duplicate_post_associations_within_cve'] += sum(v - 1 for v in local.values())
        counts['cves_with_multiple_post_entries'] += len(r['social_media_posts']) > 1
        counts['cves_with_date_screened_post_by_day30'] += bool(on_by_30)
        for h in (1, 7, 14, 30):
            windows[str(h)]['cves_with_endpoints'] += bool(usable[f'endpoints_{h}d'])
            windows[str(h)]['cves_with_complete_daily_window'] += bool(usable[f'complete_{h}d'])
        cve_rows.append(dict(cve=cve, published_date=pub.isoformat(), in_vckev=r['in_vckev'],
            post_entries=len(r['social_media_posts']), available_epss_days=n, missing_epss_days=91-n,
            coverage=coverage, old_dataset_overlap=same_id, day30_date_screened_posts=on_by_30,
            **{f'endpoints_{h}d_events': usable[f'endpoints_{h}d'] for h in (1, 7, 14, 30)}))
    del state
    require(report['cves'] == len(records) and report['score_slots'] == len(records) * 91,
            'Collector report totals disagree')
    require(dict(status_counts) == report['status_counts'], 'Collector status totals disagree')
    require(report['complete_windows'] == epss_counts['complete_91'] and
            report['filled_scores'] == sum(k*v for k, v in score_coverage.items()), 'Collector score totals disagree')
    require(set(failed_dates) == set(report['failed_dates']), 'Failed-date report mismatch')
    new_ids = {r['cve'] for r in records}
    role_overlap = {}
    roles_by_cve = {c: role for role, ids in roles.items() for c in ids}
    require(len(roles_by_cve) == sum(map(len,roles.values())) and set(roles_by_cve) == old_ids,
            'Old role map must cover the prior dataset without overlaps')
    for role, ids in roles.items():
        require(isinstance(ids, list) and all(isinstance(c, str) for c in ids), 'Unknown prior role data')
        role_overlap[role] = {'old_ids': len(ids), 'overlap': len(set(ids) & new_ids),
                              'overlapping_ids': sorted(set(ids) & new_ids)}
    cf_roles = read(CF_IDS)['ids']
    cf_overlap = {s:{'old_ids':len(ids),'overlap':len(set(ids)&new_ids)} for s,ids in cf_roles.items()}
    shared = sorted((dict(fingerprint=k, text_sha256=v['text_hash'], date=v['date'],
                cve_count=len(v['cves']), cves=sorted(v['cves']), excerpt=(v['text'] or '')[:500])
                for k,v in posts.items() if len(v['cves']) > 1), key=lambda v: (-v['cve_count'], v['fingerprint']))
    conflicts = [{'text_sha256': k, 'dates': sorted({posts[p]['date'] for p in ps}),
                  'fingerprints': sorted(ps)} for k, ps in sorted(text_index.items())
                 if len({posts[p]['date'] for p in ps}) > 1]
    counts['unique_text_date_platform_fingerprints'] = len(posts)
    counts['unique_exact_texts'] = len(text_index)
    counts['shared_fingerprints'] = len(shared)
    counts['texts_with_multiple_dates'] = len(conflicts)
    counts['new_cve_ids_not_in_old_dataset'] = len(new_ids - old_ids)
    counts['cves_with_exact_old_reddit_text'] = len(matched_old_text_cves)
    counts['cves_with_exact_old_description'] = len(matched_old_description_cves)
    counts['repeated_description_groups'] = sum(len(v) > 1 for v in description_index.values())
    counts['partial_windows_missing_only_day0'] = missing_day0_only
    counts['fingerprints_crossing_prior_train_validation'] = sum(
        {'train','validation'} <= {roles_by_cve.get(c,'new') for c in v['cves']} for v in posts.values())
    counts['fingerprints_linking_prior_test_to_new_cves'] = sum(
        {'test','new'} <= {roles_by_cve.get(c,'new') for c in v['cves']} for v in posts.values())
    checks = source_checks(repo)
    collector_hash = sha(repo / COLLECTOR)
    raw_collector = (repo / COLLECTOR).read_bytes()
    crlf_hash = hashlib.sha256(raw_collector.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')).hexdigest()
    snapshots = report['snapshots']
    archive_validation = Counter(v.get('date_validation', v.get('status', 'unknown')) for v in snapshots.values())
    findings = dict(schema='reddit_readiness_v1', commit=COMMIT, records=len(records), counts=counts,
        ranges={'cve_publication': [min(dates), max(dates)], 'post_date': [min(post_dates), max(post_dates)]},
        field_schema={k:{'types':v, 'non_null':field_presence[k]} for k,v in sorted(field_types.items())},
        post_field_occurrences=post_fields, post_publication_relation=relation,
        epss_coverage=epss_counts, available_days_histogram={str(k):v for k,v in sorted(score_coverage.items())},
        epss_status_counts=status_counts, failed_date_cells=failed_dates, per_year=years,
        epss_status_by_day=daily_coverage,
        post_centered_window_coverage=windows, source_lineage=sources,
        old_role_overlap=role_overlap, cf01_development_overlap=cf_overlap, offline_collector_checks=checks,
        collector_identity={'reported':report['collector_sha256'], 'branch_file':collector_hash,
                            'exact_match':collector_hash == report['collector_sha256'],
                            'crlf_equivalent_match':crlf_hash == report['collector_sha256'],
                            'input_hash_matches':sha(repo / REL / 'reddit_posts_vc_kev.json') == report['input_sha256']},
        epss_collection={'run_status':report['run_status'], 'as_of':report['as_of'],
                         'archive_date_evidence':archive_validation, 'failed_dates':report['failed_dates']},
        historical_text={'verified_cves':0, 'status':'NOT_ESTABLISHED_FROM_SAVED_FILES',
             'date_basis':'Scraper converts created_utc to UTC YYYY-MM-DD; saved JSON retains no time of day.',
             'gaps':['No Reddit IDs/permalinks', 'No edited/retrieved timestamps or archived content-version lineage',
                     'Current NVD description/CVSS have no historical value versions'],
             'interpretation':'Zero verified does not mean all saved text was unavailable or all post dates are wrong.'},
        previous_test_use='IDs and overlap only; no old test predictions, fitting, scoring or split changes',
        examples={'before_publication':date_issue_examples, 'largest_shared_posts':shared[:10]},
        training=False, source_scores_changed=False, pins=pins)
    findings['report_claim_checks'] = {k:{'reported':e,'observed':o,'matches':e==o} for k,e,o in [
        ('unique_cves',44985,len(records)), ('post_associations',74186,counts['post_associations']),
        ('on_or_after_launch',35113,counts['publication_on_or_after_epss_launch']), ('listed',2187,counts['listed'])]}
    findings['warnings'] = [
        '74,186 counts associations, not independently identified Reddit posts.',
        'Post IDs, edit history and content-version retrieval evidence are not retained.',
        'VulnCheck collector conflates failed requests with successful not-listed results.',
        'Nine EPSS dates have operational collection failures; missing values remain explicit.',
        'The recorded EPSS collector SHA-256 differs from the branch script, even after CRLF conversion.'
            if not findings['collector_identity']['exact_match'] and not findings['collector_identity']['crlf_equivalent_match']
            else 'EPSS collector hash comparison is recorded in collector_identity.',
        'The root branch README still describes an older dataset.',
        'Old evaluation CVEs and shared text must not be treated as new independent evidence.']
    stage = Path(tempfile.mkdtemp(prefix='reddit-report-', dir='/tmp'))
    (stage/'findings.json').write_text(dumps(findings))
    (stage/'shared_post_fingerprints.json').write_text(dumps(shared))
    (stage/'same_text_multiple_dates.json').write_text(dumps(conflicts))
    (stage/'findings.txt').write_text(report_text(findings))
    dump_rows(stage/'cve_coverage.csv.gz', cve_rows)
    # Missing/invalid dates use the same explicit column set, not silently dropped rows.
    keys = set().union(*(r.keys() for r in event_rows))
    dump_rows(stage/'post_epss_alignment.csv.gz', [{k:r.get(k) for k in keys} for r in event_rows])
    figures(stage, findings)
    for p, expected in tqdm(pins.items(), desc='Verify unchanged prior evidence and code', unit='file'):
        require(sha(p) == expected, 'Source changed during audit: ' + p)
    for rel, expected in manifest['files'].items():
        require(sha(base / rel) == expected, 'New source changed during audit: ' + rel)
    lock = {'files': {str(p.relative_to(stage)):sha(p) for p in sorted(stage.rglob('*')) if p.is_file()}}
    (stage/'artifact_lock.json').write_text(dumps(lock))
    if verify:
        saved = read(output/'artifact_lock.json')
        for name, expected in saved['files'].items():
            require(sha(output/name) == expected, 'Saved output modified: ' + name)
        require(lock == saved, 'Freshly reproduced audit differs; retained in ' + str(stage))
        shutil.rmtree(stage)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        require(shutil.disk_usage(output.parent).free > sum(p.stat().st_size for p in stage.rglob('*') if p.is_file()) + 20*1024**2,
                'Insufficient publication disk space; staged report is ' + str(stage))
        shutil.copytree(stage, output)
        shutil.rmtree(stage)
    print('Reddit audit ' + ('reproduced' if verify else 'saved') + ': ' + str(output))
    print('No training, split changes, EPSS downloading or test scoring.')


def report_text(f):
    c = f['counts']
    lines = ['Reddit dataset readiness audit', '', 'Pinned branch: ' + BRANCH + ' / ' + COMMIT,
        'Scope: saved data, source code and offline failure checks. No modeling.', '',
        '1. Size and identity', f"CVEs: {f['records']:,}; listed in the saved VulnCheck snapshot: {c['listed']:,}.",
        f"Stored CVE-post associations: {c['post_associations']:,}; distinct exact text/date/platform fingerprints: {c['unique_text_date_platform_fingerprints']:,}.",
        'Fingerprints are NOT verified unique Reddit posts because IDs and permalinks were discarded.',
        f"Texts attached to multiple stored dates: {c['texts_with_multiple_dates']:,}.",
        f"Fingerprints shared by multiple CVEs: {c['shared_fingerprints']:,}. Shared posts do not make distinct CVEs duplicates.", '',
        f"Exact duplicate CVE-post entries retained within CVEs: {c['duplicate_post_associations_within_cve']:,}.",
        f"The largest shared fingerprint is linked to {f['examples']['largest_shared_posts'][0]['cve_count']} CVEs.",
        'Example excerpt: ' + f['examples']['largest_shared_posts'][0]['excerpt'], '',
        '2. Post dates and historical content',
        'The available scraper selects created_utc, converts it to a UTC date and discards time of day.',
        'This is stronger date lineage than an unexplained date field, but the saved content is not version-linked to that day.',
        'No saved IDs, permalinks, edit times or archive retrieval times establish historical wording.',
        'The scraper deduplicates (CVE, cleaned text), ignoring date and post ID; repeated posts can be collapsed.',
        'The query selects submissions, not Reddit comments; failed yearly queries can be skipped without a completion manifest.',
        'Cleaning removes markup, symbols and whitespace. It is not a byte-preserving archive of the original text.',
        'Post-to-NVD publication day comparison: ' + str(dict(f['post_publication_relation'])),
        'Earlier posts are retained for review, not automatically treated as date errors.', '',
        '3. EPSS availability', 'CVEs by score coverage: ' + str(dict(f['epss_coverage'])),
        'Missing-score reasons: ' + str(dict(f['epss_status_counts'])),
        'Collection state: ' + f['epss_collection']['run_status'] + '; cutoff: ' + f['epss_collection']['as_of'],
        f"Partial histories missing only the publication-day score: {c['partial_windows_missing_only_day0']:,}.",
        f"Stored post associations with EPSS on the same date: {c['associations_with_epss_on_post_date']:,}.",
        f"Post associations outside the stored publication-to-Day-90 window: {c['associations_outside_publication_91day_window']:,}.",
        'Operationally failed dates and affected CVE-day scores: ' + str(dict(f['failed_date_cells'])),
        'The EPSS report and resume state are checked against every saved daily value and status.',
        'Snapshot header/URL metadata are inspected, but daily archive files are not downloaded or independently replayed here.',
        'Being published after EPSS launch is not equivalent to having a usable complete history.', '',
        '4. Post-centered coverage (descriptive candidates, not authenticated historical-text examples)',
        'Window | CVE-post associations with scores at -h/0/+h | CVEs | Complete daily window associations | CVEs']
    for h in ('1','7','14','30'):
        w=f['post_centered_window_coverage'][h]
        lines.append(f"{h} days each side | {w['associations_with_endpoints']} | {w['cves_with_endpoints']} | {w['associations_with_complete_daily_window']} | {w['cves_with_complete_daily_window']}")
    lines += ['A daily score on the post date cannot be assumed to precede the post: intraday EPSS release time is not established.',
        'Missing days are not filled. Endpoints present does not mean all intervening days are present.', '',
        '5. VulnCheck label risk',
        'Offline tests reproduce in_vckev=false after simulated HTTP 401/403/429/500 failures.',
        'A failed request and a successful empty catalogue lookup must be distinguished before reliable negative labels can be built.',
        'This demonstrates a code path, not how many saved labels were actually affected. No labels were changed.',
        'The PDF reports collection on 4 November 2026; no per-CVE successful-check receipt is present.', '',
        '6. Previous-study overlap',
        f"Previously seen CVE IDs: {c['old_cve_overlap']:,}; new IDs: {c['new_cve_ids_not_in_old_dataset']:,}.",
        'Prior roles: ' + str({k:{kk:vv for kk,vv in v.items() if kk!='overlapping_ids'} for k,v in f['old_role_overlap'].items()}),
        'Actual CF01 development overlap: ' + str(f['cf01_development_overlap']),
        f"New records with an exact old Reddit text: {c['cves_with_exact_old_reddit_text']:,}.",
        f"Shared fingerprints crossing the previous train/validation roles: {c['fingerprints_crossing_prior_train_validation']}.",
        f"Shared fingerprints connecting previously tested CVEs with new CVEs: {c['fingerprints_linking_prior_test_to_new_cves']}.",
        'This is an exposure audit, not a new split. Previously evaluated CVEs must not be advertised as an untouched new test.', '',
        '7. Source preservation and report checks', dumps(f['source_lineage']).strip(),
        'Collector identity: ' + str(f['collector_identity']),
        'A differing collector hash means the exact script that produced the saved scores is not the pinned branch file; it does not by itself prove the scores are incorrect.',
        'PDF count checks: ' + str(f['report_claim_checks']),
        f"The report's 'after launch' count includes the launch date itself: {c['publication_on_or_after_epss_launch']-c['publication_strictly_after_epss_launch']} CVEs were published on 2021-04-14.",
        'The branch README describes the older 9,218-row dataset, not this Reddit collection.', '',
        '8. Decision and next actions',
        'Proceed with dataset repair/provenance work, not a new training run.',
        'Preserve post ID, permalink, created_utc, edited, archive retrieval time, raw title/body and archive partition/version.',
        'Record successful empty VulnCheck responses separately from failures; rerun unresolved lookups without overwriting raw labels.',
        'Retry only operational EPSS failures with a separate versioned output; retain genuine pre-launch/absent/future missingness.',
        'Check the largest shared-post examples for bulk advisories or irrelevant CVE lists before choosing relevance/split rules.',
        'For future temporal evaluation, exclude already-exposed evaluation CVEs and document cross-split post reuse.',
        'A retrospective study may be feasible after labels are verified; strict historical text fusion is not established yet.', '',
        '9. Output guide',
        'findings.json / findings.txt: machine-readable and plain-English findings.',
        'cve_coverage.csv.gz: one CVE per row; number of available scores, posts and endpoint-qualified events.',
        'post_epss_alignment.csv.gz: one stored CVE-post association per row; dates and score coverage around the post.',
        'shared_post_fingerprints.json: exact text/date/platform sharing, with example excerpts and CVE IDs.',
        'same_text_multiple_dates.json: identical texts associated with different dates, not automatically timestamp errors.',
        'figures/: labeled coverage plots. artifact_lock.json: SHA-256 identities for every derived artifact.']
    return '\n'.join(lines) + '\n'


def figures(stage, f):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=stage/'figures'; dest.mkdir()
    def bars(name, labels, values, title, xlabel, ylabel):
        fig, ax=plt.subplots(figsize=(10,5))
        ax.bar(labels, values, color='#267b8d')
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        if len(labels)>12:
            ax.tick_params(axis='x', labelrotation=90, labelsize=7)
        if len(labels)>50:
            ticks=list(range(0,len(labels),7)) + [len(labels)-1]
            ax.set_xticks(ticks, [labels[i] for i in ticks])
        fig.tight_layout(); fig.savefig(dest/(name+'.png'), dpi=140, metadata={'Software':'reddit_readiness_v1'}); plt.close(fig)
    bars('epss_coverage', ['All 91 scores','Some scores','No scores'],
         [f['epss_coverage'][k] for k in ('complete_91','partial','none')],
         'Daily EPSS coverage per CVE: publication day through Day 90', 'Available daily scores', 'Number of CVEs')
    hist=f['available_days_histogram']
    bars('available_day_counts', list(map(str,range(92))), [hist.get(str(i),0) for i in range(92)],
         'How many of the 91 daily EPSS scores are available?', 'Number of available daily scores (0 to 91)', 'Number of CVEs')
    rel=f['post_publication_relation']
    bars('post_publication', ['Before NVD publication','Same day','After NVD publication'],
         [rel.get(k,0) for k in ('before_publication','same_day','after_publication')],
         'Stored Reddit post dates compared with NVD publication', 'Post date relative to publication', 'CVE-post associations (not unique posts)')
    years=sorted(f['per_year'])
    bars('publication_year', years, [f['per_year'][y]['cves'] for y in years],
         'Publication-year composition of the Reddit-selected CVEs', 'NVD publication year', 'Number of CVEs')
    bars('post_window_coverage', ['1 day','7 days','14 days','30 days'],
         [f['post_centered_window_coverage'][h]['cves_with_complete_daily_window'] for h in ('1','7','14','30')],
         'CVEs with an unbroken EPSS window on both sides of at least one post', 'Days required on EACH side, plus the post date', 'Number of CVEs (date-screened only)')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('fetch','audit'), required=True)
    p.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--verify-only', action='store_true')
    a=p.parse_args()
    if a.stage=='fetch':
        require(not a.verify_only, 'Fetch verifies sources automatically; --verify-only is for the audit')
        fetch(a.data_root)
    else:
        audit(a.data_root,a.output_dir,a.verify_only)


if __name__=='__main__':
    main()
