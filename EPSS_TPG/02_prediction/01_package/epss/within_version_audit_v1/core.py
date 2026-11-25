"""Date-only inner roles; preserve CT05 transitive evidence groups across versions."""
from collections import Counter
from datetime import date, timedelta
import math


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical_day(value):
    require(isinstance(value, str) and date.fromisoformat(value).isoformat() == value,
            'Noncanonical target date')
    return value


def select_pool(rows, components, window):
    start, end = (canonical_day(window[k]) for k in ('start_inclusive', 'end_exclusive'))
    require(start < end and bool(rows) and set(rows) == set(components), 'Invalid window/component coverage')
    projected = {}
    for cid, row in sorted(rows.items()):
        require(row['outer_role'] == 'train', 'Non-training CVE: ' + cid)
        day = canonical_day(row['anchor_date'])
        require(isinstance(components[cid], str) and bool(components[cid]), 'Invalid evidence component')
        require(set(row['source_mask']) == set('DMGN') and all(type(x) is bool for x in row['source_mask'].values()),
                'Malformed source mask')
        require(row['source_mask']['D'], 'Standalone description required')
        if start <= day < end:
            projected[cid] = {k: row[k] for k in ('outer_role', 'anchor_date', 'source_mask', 'D_key', 'DM_key')}
    require(bool(projected), 'No eligible original-training records')
    return projected, sorted(set(rows) - set(projected))


def verify_fold(fold, rows, components):
    names = ('optimization', 'selection', 'assessment', 'purged_to_assessment', 'purged_to_selection', 'future')
    assigned = [cid for role in names for cid in fold[role]]
    require(len(assigned) == len(set(assigned)) and set(assigned) == set(rows), 'Fold accounting differs')
    roles = ('optimization', 'selection', 'assessment')
    for i, earlier in enumerate(roles):
        for later in roles[i+1:]:
            a, b = fold[earlier], fold[later]
            require(not {components[c] for c in a} & {components[c] for c in b}, 'Evidence crosses inner roles')
            if a and b:
                require(max(rows[c]['anchor_date'] for c in a) < min(rows[c]['anchor_date'] for c in b),
                        'Targets not strictly chronological')
    start, end = fold['assessment_start_inclusive'], fold['assessment_end_exclusive']
    require(all(start <= rows[c]['anchor_date'] < end for c in fold['assessment']), 'Wrong assessment interval')
    require(all(rows[c]['anchor_date'] < start for role in names[:2] + names[3:5] for c in fold[role]),
            'Fitting/purged target crosses assessment boundary')
    require(all(rows[c]['anchor_date'] >= end for c in fold['future']), 'Wrong future interval')


def make_plan(rows, components, config, dataset):
    selected, excluded = select_pool(rows, components, config['datasets'][dataset])
    fractions = config['cut_fractions']; fraction = config['selection_fraction']
    require(fractions == sorted(set(fractions)) and all(0 < f < 1 for f in fractions) and 0 < fraction < 1,
            'Invalid date fractions')
    dates = sorted(row['anchor_date'] for row in selected.values())
    starts = [dates[math.floor(len(dates) * f)] for f in fractions]
    ends = starts[1:] + [(date.fromisoformat(dates[-1]) + timedelta(days=1)).isoformat()]
    folds = []
    for n, (start, end) in enumerate(zip(starts, ends), 1):
        before = sorted(c for c, r in selected.items() if r['anchor_date'] < start)
        assessment = sorted(c for c, r in selected.items() if start <= r['anchor_date'] < end)
        seen = {components[c] for c in assessment}
        purge_a = [c for c in before if components[c] in seen]
        fit = sorted(set(before) - set(purge_a))
        cut = sorted(selected[c]['anchor_date'] for c in fit)[math.floor(len(fit) * (1-fraction))] if fit else None
        selection = [c for c in fit if selected[c]['anchor_date'] >= cut] if cut else []
        earlier = [c for c in fit if selected[c]['anchor_date'] < cut] if cut else []
        selection_groups = {components[c] for c in selection}
        purge_s = [c for c in earlier if components[c] in selection_groups]
        fold = {'fold': n, 'assessment_start_inclusive': start, 'assessment_end_exclusive': end,
                'selection_start_inclusive': cut, 'optimization': sorted(set(earlier) - set(purge_s)),
                'selection': selection, 'assessment': assessment, 'purged_to_assessment': purge_a,
                'purged_to_selection': purge_s,
                'future': sorted(c for c, r in selected.items() if r['anchor_date'] >= end)}
        verify_fold(fold, selected, components)
        folds.append(fold)
    oof = sorted(c for f in folds for c in f['assessment'])
    require(len(oof) == len(set(oof)), 'Repeated OOF assessment CVE')
    return {'rows': selected, 'component_by_cve': {c: components[c] for c in selected},
            'out_of_version_ids': excluded, 'cut_dates': starts, 'folds': folds,
            'oof_ids': oof, 'warmup_only_ids': sorted(set(selected) - set(oof)),
            'all_original_training_ids': sorted(rows)}


def validate_labels(rows, lookup):
    labels = {}
    for cid, row in rows.items():
        record = lookup[cid]; value = record.get('value', record.get('target'))
        require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                'Invalid/missing Day-60 target: ' + cid)
        require(record.get('target_day') == 60 and record.get('observed', True) is True and
                record.get('split', 'train') == 'train' and record['target_date'] == row['anchor_date'],
                'Label date/role mismatch: ' + cid)
        labels[cid] = float(value)
    return labels


def support(ids, rows, labels, components, cutoff):
    import statistics
    values = [labels[c] for c in ids]
    high = sum(v >= cutoff for v in values)
    counts = Counter(components[c] for c in ids)
    present = [c for c in ids if rows[c]['source_mask']['M']]
    return {'cves': len(ids), 'high_score': high, 'lower_score': len(ids)-high,
            'M_present': len(present), 'M_absent': len(ids)-len(present),
            'high_and_M_present': sum(labels[c] >= cutoff for c in present),
            'high_and_M_absent': sum(labels[c] >= cutoff for c in ids if not rows[c]['source_mask']['M']),
            'evidence_components': len(counts), 'largest_component_cves': max(counts.values(), default=0),
            'high_score_components': len({components[c] for c in ids if labels[c] >= cutoff}),
            'M_present_components': len({components[c] for c in present}),
            'high_and_M_present_components': len({components[c] for c in present if labels[c] >= cutoff}),
            'minimum_target_date': min((rows[c]['anchor_date'] for c in ids), default=None),
            'maximum_target_date': max((rows[c]['anchor_date'] for c in ids), default=None),
            'median_score': statistics.median(values) if values else None,
            'mean_score': statistics.mean(values) if values else None}


def gates(summaries, config):
    checks = {role + '_count': summaries[role]['cves'] >= minimum
              for role, minimum in config['minimum_cves'].items()}
    for role in config['support_review_roles']:
        stats = summaries[role]
        checks.update({role + '_tail_support': min(stats['high_score'], stats['lower_score']) >= config['minimum_high_and_lower'],
                       role + '_M_support': stats['M_present'] >= config['minimum_M_present'],
                       role + '_flag_variation': stats['M_absent'] >= config['minimum_M_absent_for_flag_contrast'],
                       role + '_high_M_support': stats['high_and_M_present'] >= config['minimum_high_and_M_present']})
    failed = sorted(k for k, ok in checks.items() if not ok)
    return {'checks': checks, 'failed_checks': failed,
            'count_gates_pass': all(checks[r + '_count'] for r in config['minimum_cves']),
            'status': 'SUPPORT_REVIEW_REQUIRED' if failed else 'COUNTS_PASS_NOT_TRAINING_APPROVAL',
            'training_enabled': False}
