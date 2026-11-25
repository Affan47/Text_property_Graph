"""Outcome-blind date cuts and conservative text-overlap grouping."""
from collections import defaultdict
from datetime import date, timedelta
import hashlib
import math
import unicodedata


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fingerprint(text):
    require(isinstance(text, str), 'Document text must be a string')
    value = ' '.join(unicodedata.normalize('NFC', text).split())
    require(bool(value), 'Empty accepted document')
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def components(rows):
    parent = {cid: cid for cid in rows}

    def root(cid):
        while parent[cid] != cid:
            parent[cid] = parent[parent[cid]]
            cid = parent[cid]
        return cid

    documents = defaultdict(set)
    for cid, row in sorted(rows.items()):
        for key in row['document_fingerprints']:
            documents[key].add(cid)
    for ids in documents.values():
        members = sorted(ids)
        for cid in members[1:]:
            left, right = sorted((root(members[0]), root(cid)))
            parent[right] = left
    groups = defaultdict(list)
    for cid in sorted(rows):
        groups[root(cid)].append(cid)
    mapping = {cid: members[0] for members in groups.values() for cid in members}
    shared = [{'fingerprint': key, 'cves': sorted(ids), 'degree': len(ids)}
              for key, ids in sorted(documents.items()) if len(ids) > 1]
    return mapping, dict(sorted(groups.items())), shared


def make_folds(rows, config):
    require(bool(rows), 'Empty training cohort')
    for cid, r in rows.items():
        require(r['outer_role'] == 'train', 'Non-training CVE in fold plan: '+cid)
        require(date.fromisoformat(r['anchor_date']).isoformat() == r['anchor_date'], 'Noncanonical anchor date')
    ordered = sorted(r['anchor_date'] for r in rows.values())
    fractions = config['cut_fractions']
    require(fractions == sorted(set(fractions)) and all(0 < f < 1 for f in fractions), 'Invalid cut fractions')
    starts = [ordered[math.floor(len(ordered)*f)] for f in fractions]
    ends = starts[1:]+[(date.fromisoformat(ordered[-1])+timedelta(days=1)).isoformat()]
    mapping, groups, shared = components(rows)
    folds = []
    for number, (start, end) in enumerate(zip(starts, ends), 1):
        fit = sorted(cid for cid, r in rows.items() if r['anchor_date'] < start)
        assessment = sorted(cid for cid, r in rows.items() if start <= r['anchor_date'] < end)
        future = sorted(cid for cid, r in rows.items() if r['anchor_date'] >= end)
        assessment_groups = {mapping[c] for c in assessment}
        for policy in config['policies']:
            purge = [c for c in fit if policy == 'evidence_grouped' and mapping[c] in assessment_groups]
            retained = sorted(set(fit)-set(purge))
            require(len(retained)+len(purge)+len(assessment)+len(future) == len(rows), 'Cohort accounting differs')
            require(not set(retained) & set(assessment), 'CVE crosses inner roles')
            require(all(rows[c]['anchor_date'] < start for c in retained), 'Immature target in fitting data')
            overlap = sorted({mapping[c] for c in retained} & assessment_groups)
            if policy == 'evidence_grouped':
                require(not overlap, 'Evidence component crosses grouped inner roles')
            folds.append({'fold': number, 'policy': policy, 'assessment_start_inclusive': start,
                          'assessment_end_exclusive': end, 'fit': retained, 'inner_assessment': assessment,
                          'purged_shared_evidence': purge, 'future_unseen': future,
                          'cross_role_components': overlap})
    unique = [c for f in folds if f['policy'] == 'evidence_grouped' for c in f['inner_assessment']]
    require(len(unique) == len(set(unique)), 'Repeated assessment CVE across folds')
    return {'cut_dates': starts, 'folds': folds, 'component_by_cve': mapping,
            'components': groups, 'shared_documents': shared,
            'oof_covered_ids': sorted(unique), 'warmup_only_ids': sorted(set(rows)-set(unique))}
