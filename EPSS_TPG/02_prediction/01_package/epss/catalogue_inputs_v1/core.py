"""Pure disclosure, role-assignment and EPSS validation contracts."""
from collections import defaultdict
import csv
from datetime import datetime
import gzip
import hashlib
import math
import re
from urllib.parse import unquote, urlsplit

from ..catalogue_classification_v1.snapshot import require, CVE
from ..training_fold_audit_v1.core import components, fingerprint
from ..vulncheck_study import policy_v2

CVSS = ('cvss_vector', 'cvss_vector_version', 'cvss_vector_score')
URLS = ('all_sources_urls', 'github_only_urls', 'non_github_urls')
BARE = re.compile(r'\s*CVE-\d{4}-\d{4,}\s*', re.I)


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def disclosure(text, cve, source):
    result = policy_v2.inspect(text, cve, source)
    # URL escaping can hide a literal catalogue cue from the lexical screen.
    escaped = unquote(text)
    if escaped != text and policy_v2.inspect(escaped, cve, source)['decision'] == 'quarantine':
        result['decision'] = 'quarantine'
        result['reasons'] = sorted(set(result['reasons']) | {'url_encoded_disclosure'})
    return result


def reference_urls(value, cve):
    if value is None:
        return None, {'missing': True, 'duplicates_removed': 0, 'quarantined': 0, 'invalid': 0}
    require(isinstance(value, list), 'Reference list has wrong type: ' + cve)
    kept, seen = [], set()
    stats = {'missing': False, 'duplicates_removed': 0, 'quarantined': 0, 'invalid': 0}
    for url in value:
        if not isinstance(url, str):
            stats['invalid'] += 1
            continue
        url = url.strip()
        if url in seen:
            stats['duplicates_removed'] += 1
            continue
        seen.add(url)
        try:
            parts = urlsplit(url)
            valid = parts.scheme in ('http', 'https') and parts.hostname and not parts.username and not parts.password
        except ValueError:
            valid = False
        if not valid:
            stats['invalid'] += 1
        elif disclosure(url, cve, 'description')['decision'] == 'quarantine':
            stats['quarantined'] += 1
        else:
            kept.append(url)
    return sorted(kept), stats


def screen(record):
    cve = record['cve']
    documents, audit, seen = [], [], set()
    posts = record.get('social_media_posts')
    require(posts is None or isinstance(posts, list), 'Malformed posts: ' + cve)
    candidates = [('D', 0, record.get('description'))]
    for index, post in enumerate(posts or []):
        require(isinstance(post, dict), 'Malformed post record: ' + cve)
        candidates.append(('M', index, post.get('post')))
    for source, index, text in candidates:
        item = {'source': source, 'index': index, 'text_sha256': digest(text) if isinstance(text, str) else None}
        if not isinstance(text, str) or not text.strip():
            item.update(decision='missing', reasons=['empty_or_non_text'])
        elif source == 'M' and BARE.fullmatch(text):
            item.update(decision='bare_id', reasons=['no_prose'])
        else:
            item.update(disclosure(text, cve, 'description' if source == 'D' else 'smp'))
            key = (source, fingerprint(text))
            if item['decision'] == 'accept':
                if key in seen:
                    item.update(decision='duplicate', reasons=['same_source_normalized_text'])
                else:
                    seen.add(key)
                    documents.append({'source': source, 'text': text, 'sha256': digest(text)})
        audit.append(item)
    structured = {k: record.get(k) for k in CVSS}
    urls = {}
    for name in URLS:
        structured[name], urls[name] = reference_urls(record.get(name), cve)
    # Derive disjoint subsets from the cleaned union; preserve null vs empty.
    if structured['all_sources_urls'] is not None:
        all_urls = sorted(set().union(*(set(structured[k] or []) for k in URLS)))
        structured['all_sources_urls'] = all_urls
        structured['github_only_urls'] = [u for u in all_urls if urlsplit(u).hostname.lower() in ('github.com', 'www.github.com')]
        structured['non_github_urls'] = [u for u in all_urls if u not in set(structured['github_only_urls'])]
    row = {'documents': documents, 'mask_D': int(any(d['source'] == 'D' for d in documents)),
           'mask_M': int(any(d['source'] == 'M' for d in documents)), 'structured': structured}
    return row, {'documents': audit, 'reference_lists': urls,
                 'excluded_fields': sorted(set(record) - {'cve', 'description', 'social_media_posts', *CVSS, *URLS})}


def freeze_roles(datasets, prior_groups, seed=6029):
    rows = {}
    for dataset in sorted(datasets):
        for cve, row in sorted(datasets[dataset].items()):
            target = rows.setdefault(cve, {'document_fingerprints': set()})['document_fingerprints']
            target.update('text:' + fingerprint(d['text']) for d in row['documents'])
    for group, members in prior_groups.items():
        for cve in set(members) & set(rows):
            rows[cve]['document_fingerprints'].add('prior:' + group)
    mapping, groups, _ = components(rows)
    assignment, details = {}, {}
    for key, members in groups.items():
        identity = digest(str(seed) + '\n' + '\n'.join(members))
        bucket = int(identity, 16) % 10000
        role = 'fitting' if bucket < 7000 else 'selection' if bucket < 8500 else 'assessment'
        details[key] = {'members': members, 'sha256': identity, 'role': role}
        assignment.update({c: {'role': role, 'group': key} for c in members})
    return {'by_cve': assignment, 'components': details,
            'datasets': {name: {role: sorted(c for c in records if assignment[c]['role'] == role)
                                for role in ('fitting', 'selection', 'assessment')}
                         for name, records in sorted(datasets.items())}}


def parse_epss(path, expected_date, ids, minimum_rows=100000, max_uncompressed=268435456):
    """Validate the entire file/CRC, but retain scores only for training IDs."""
    found, seen, size = {}, set(), 0
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
        header = stream.readline(4096)
        require(header.startswith('#') and header.endswith('\n'), 'Missing bounded EPSS metadata header')
        fields = {}
        for part in next(csv.reader([header[1:].strip()])):
            key, sep, value = part.partition(':')
            require(sep and key.strip() not in fields, 'Malformed/duplicate metadata key')
            fields[key.strip()] = value.strip()
        require(fields.get('model_version'), 'Missing model version')
        try:
            day = datetime.fromisoformat(fields.get('score_date', '').replace('Z', '+00:00')).date().isoformat()
        except ValueError:
            raise ValueError('Invalid EPSS header score_date') from None
        require(day == expected_date, 'EPSS internal date differs from required date')
        def lines():
            nonlocal size
            for line in stream:
                size += len(line.encode('utf-8'))
                require(size <= max_uncompressed and len(line) < 4096, 'EPSS CSV exceeds size limit')
                require(bool(line.strip()), 'Unexpected blank EPSS row')
                yield line
        reader = csv.DictReader(lines())
        require(reader.fieldnames == ['cve', 'epss', 'percentile'], 'Unexpected EPSS CSV columns')
        for row in reader:
            require(set(row) == {'cve', 'epss', 'percentile'} and all(row.values()), 'Malformed EPSS row')
            cve = row['cve']
            require(CVE.fullmatch(cve) and cve not in seen, 'Invalid/duplicate EPSS CVE')
            seen.add(cve)
            score, percentile = float(row['epss']), float(row['percentile'])
            require(all(math.isfinite(x) and 0 <= x <= 1 for x in (score, percentile)), 'Out-of-range EPSS value')
            if cve in ids:
                found[cve] = score
    require(len(seen) >= minimum_rows, 'EPSS snapshot unexpectedly small; completeness review required')
    return found, {'score_date': day, 'model_version': fields['model_version'], 'rows': len(seen),
                   'matched_training_ids': len(found), 'missing_training_ids': len(set(ids) - set(found))}


def join_epss(ids, scores):
    return {c: {'score': scores.get(c), 'observed': int(c in scores),
                'status': 'OBSERVED' if c in scores else 'CVE_NOT_IN_VERIFIED_SNAPSHOT'} for c in ids}
