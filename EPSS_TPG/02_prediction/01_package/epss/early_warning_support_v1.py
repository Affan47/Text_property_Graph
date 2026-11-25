"""Outcome-blind date plans and offline crossing-label support; never trains."""
from collections import Counter
from datetime import date, timedelta
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import early_warning_review_v1 as old
from .early_warning_gap_review_v1 import gap_label

a = old.artifacts
ROOT = old.BASE/'22_early_warning_support_v1'
CLI = old.CLI.with_name('27_audit_early_warning_support.py')
CONFIG = {
    'schema': 'early_warning_support_v1', 'cut_fractions': [.4, .6, .8], 'selection_fraction': .2,
    'training_target_days': [60, 90], 'prediction_anchor_day': 7,
    'date_rule': 'Whole publication-through-Day90 range inside one frozen EPSS schedule interval',
    'maturity_rule': 'Every optimization target date strictly before first selection Day7; '
                     'every optimization/selection target date strictly before first assessment Day7',
    'group_rule': 'Reuse full original-training CT05 components, including paths through other versions',
    'role_allocation': 'Date-only 40/60/80 percent cuts; latest 20 percent selection after maturity/group exclusions; '
                       'no outcome-based date moves or weak-fold deletion',
    'label_policies': ['exact_day7_strict', 'last_known_by_day7_strict', 'isolated_day_assumption'],
    'support_warning': 'Fewer than 20 risers or 20 non-risers in an active role; not a power calculation',
    'interpretation': 'Support scenarios, not approved training folds or independent confirmation',
    'training': False, 'network': False, 'test_scoring': False,
}
COHORTS = {
    'reddit_v1': ('reddit', '2021-04-14', '2022-02-04'),
    'reddit_v2': ('reddit', '2022-02-04', '2023-03-07'),
    'previous_v3': ('previous', '2023-03-07', '2025-03-17'),
}
DATASETS = {
    'reddit': {'directory': '01_reddit', 'expected_train': 24213,
               'source': old.SOURCE, 'source_sha256': old.PINS[str(old.SOURCE)],
               'protocol_sha256': '76ff5ad01d652a389a6726115b5a71ff9278ba3f1bf5a6a213b8d44015bde2e2'},
    'previous': {'directory': '02_previous_multisource', 'expected_train': 2546,
                 'source': old.ROOT.parent/'SummTPGVul/Sec4AI4Sec-EPSS/Data_Files/epss_recollected_2026-09-30_smp.json',
                 'source_sha256': '506ae7da7c9202ca132e4e7ff8511f1fbbea9ab534630374dde5370e0ecfd43c',
                 'protocol_sha256': '63b2a8cc39520884cf35e55a47b99a5b4a96c44fc183cb2838889ea16a231ff2'},
}
ROLES = ('optimization', 'selection', 'assessment', 'immature_before_assessment',
         'purged_to_assessment', 'immature_before_selection', 'purged_to_selection', 'future')


def maturity(row, target_day):
    return (date.fromisoformat(row['publication'])+timedelta(days=target_day)).isoformat()


def assign_roles(rows, groups, start, end, target_day):
    result = {r: [] for r in ROLES}
    result['assessment'] = sorted(c for c, r in rows.items() if start <= r['anchor'] < end)
    result['future'] = sorted(c for c, r in rows.items() if r['anchor'] >= end)
    before = [c for c, r in rows.items() if r['anchor'] < start]
    result['immature_before_assessment'] = sorted(c for c in before if maturity(rows[c], target_day) >= start)
    eligible = sorted(set(before)-set(result['immature_before_assessment']))
    blocked = {groups[c] for c in result['assessment']}
    result['purged_to_assessment'] = sorted(c for c in eligible if groups[c] in blocked)
    fit = sorted(set(eligible)-set(result['purged_to_assessment']))
    cut = sorted(rows[c]['anchor'] for c in fit)[math.floor(len(fit)*(1-CONFIG['selection_fraction']))] if fit else None
    if cut:
        result['selection'] = sorted(c for c in fit if rows[c]['anchor'] >= cut)
        earlier = [c for c in fit if rows[c]['anchor'] < cut]
        result['immature_before_selection'] = sorted(c for c in earlier if maturity(rows[c], target_day) >= cut)
        earlier = sorted(set(earlier)-set(result['immature_before_selection']))
        blocked = {groups[c] for c in result['selection']}
        result['purged_to_selection'] = sorted(c for c in earlier if groups[c] in blocked)
        result['optimization'] = sorted(set(earlier)-set(result['purged_to_selection']))
    fold = {'target_day': target_day, 'assessment_start': start, 'assessment_end': end,
            'selection_start': cut, **result}
    validate_roles(fold, rows, groups)
    return fold


def validate_roles(fold, rows, groups):
    all_ids = [c for role in ROLES for c in fold[role]]
    a.require(len(all_ids) == len(set(all_ids)) and set(all_ids) == set(rows), 'Incomplete/overlapping role accounting')
    for i, role in enumerate(ROLES[:3]):
        for later in ROLES[i+1:3]:
            a.require(not {groups[c] for c in fold[role]} & {groups[c] for c in fold[later]}, 'Evidence group crosses active roles')
    h = fold['target_day']
    a.require(all(maturity(rows[c], h) < fold['assessment_start'] for c in fold['optimization']+fold['selection']),
              'Target not mature before assessment anchor')
    if fold['selection_start']:
        a.require(all(maturity(rows[c], h) < fold['selection_start'] for c in fold['optimization']),
                  'Optimization label not mature before selection anchor')
    a.require(all(fold['assessment_start'] <= rows[c]['anchor'] < fold['assessment_end'] for c in fold['assessment']),
              'Assessment interval changed')


def definitions(protocol, start, end):
    rows = {}
    for c, r in protocol['rows'].items():
        a.require(r['outer_role'] == 'train', 'Non-training record')
        publication = date.fromisoformat(r['anchor_date'])-timedelta(days=60)
        if start <= publication.isoformat() and (publication+timedelta(days=90)).isoformat() < end:
            rows[c] = {'publication': publication.isoformat(), 'anchor': (publication+timedelta(days=7)).isoformat(),
                       'M_present': r['source_mask']['M']}
    a.require(rows, 'Empty whole-version cohort')
    dates = sorted(r['anchor'] for r in rows.values())
    starts = [dates[math.floor(len(dates)*q)] for q in CONFIG['cut_fractions']]
    ends = starts[1:]+[(date.fromisoformat(dates[-1])+timedelta(days=1)).isoformat()]
    groups = {c: protocol['component_by_cve'][c] for c in rows}
    folds = [dict(fold=i+1, **assign_roles(rows, groups, lo, hi, h))
             for h in CONFIG['training_target_days'] for i, (lo, hi) in enumerate(zip(starts, ends))]
    return {'rows': rows, 'groups': groups, 'cut_dates': starts, 'folds': folds,
            'outside_whole_version': sorted(set(protocol['rows'])-set(rows))}


def source_context(cohort):
    dataset, start, end = COHORTS[cohort]; spec = DATASETS[dataset]
    directory = old.BASE/'11_training_only_pooled_mse_v1'/spec['directory']/'protocol'
    a.check_lock(directory)
    a.require(a.sha(directory/'protocol.json') == spec['protocol_sha256'], 'Frozen training protocol changed')
    protocol = a.read(directory/'protocol.json')
    a.require(not protocol['test_access'] and not protocol['outer_validation_access'], 'Unexpected parent scope')
    a.require(len(protocol['rows']) == spec['expected_train'], 'Original training cohort changed')
    pins = {str(directory/'protocol.json'): spec['protocol_sha256'],
            str(directory/'artifact_lock.json'): a.sha(directory/'artifact_lock.json')}
    for path in (Path(__file__), Path(old.__file__), Path(gap_label.__globals__['__file__']),
                 Path(a.__file__), CLI, CLI.with_name('test_early_warning_support.py')):
        pins[str(path)] = a.sha(path)
    return spec, protocol, start, end, pins


def plan(cohort, verify=False):
    spec, protocol, start, end, pins = source_context(cohort)
    value = {'cohort': cohort, 'config': CONFIG, 'window': {'start': start, 'end_exclusive': end},
             'original_training_cves': len(protocol['rows']), 'source_not_read': str(spec['source']),
             'source_sha256': spec['source_sha256'], 'pins': pins, **definitions(protocol, start, end)}
    a.publish(ROOT/cohort/'plan', {'plan.json': value, 'REPORT.txt':
        'Whole-version crossing support plan. Dates and existing training-role evidence groups only.\n'
        'No score values used to allocate roles. All cutoffs and weak/empty folds remain unchanged.\n'
        'Optimization labels precede selection anchors; selection labels precede assessment anchors.\n'
        '53/83 days describe target-minus-anchor spacing, not a universally sufficient embargo.\n'
        'This is not a training protocol, a power calculation, or an approved cohort selection.\n'}, pins, verify)
    print(f'{cohort}: date-only support plan '+('reproduced.' if verify else 'saved.')+' No training or downloads.')


def support(ids, labels, rows, groups):
    valid = [c for c in ids if labels[c]['label'] is not None]
    positives = [c for c in valid if labels[c]['label'] == 1]
    negatives = [c for c in valid if labels[c]['label'] == 0]
    return {'cves': len(ids), 'labelled': len(valid), 'risers': len(positives), 'non_risers': len(negatives),
            'unresolved_or_excluded': len(ids)-len(valid),
            'assumed_negatives': sum(labels[c].get('assumed_negative', False) for c in valid),
            'post_cves': sum(rows[c]['M_present'] for c in valid),
            'riser_post_cves': sum(rows[c]['M_present'] for c in positives),
            'positive_groups': len({groups[c] for c in positives}), 'eligible_groups': len({groups[c] for c in valid}),
            'risers_previously_high_before_day7': sum(labels[c]['observed_high_score_before_day7'] for c in positives),
            'sparse_class_warning': len(positives) < 20 or len(negatives) < 20,
            'label_status_counts': dict(Counter(labels[c]['status'] for c in ids))}


def audit(cohort, verify=False):
    spec, protocol, start, end, pins = source_context(cohort)
    directory = ROOT/cohort/'plan'; a.check_lock(directory)
    p = a.read(directory/'plan.json')
    a.require(p['config'] == CONFIG and p['pins'] == pins, 'Support plan/code identity changed')
    expected = definitions(protocol, start, end)
    a.require(all(p[k] == v for k, v in expected.items()), 'Date/group plan does not reproduce')
    pins[str(directory/'plan.json')] = a.sha(directory/'plan.json')
    pins[str(spec['source'])] = spec['source_sha256']; a.check_pins(pins)
    # The source container is decoded for integrity; only original-training pool rows are inspected/exported.
    wanted = set(p['rows']); records = {}
    for r in a.read(spec['source']):
        if r['cve'] in wanted:
            a.require(r['cve'] not in records, 'Duplicate selected CVE')
            records[r['cve']] = r
    a.require(set(records) == wanted, 'Missing selected training records')
    labels = {name: {} for name in CONFIG['label_policies']}; audit_rows = []; gaps = Counter()
    for c in tqdm(sorted(wanted), desc=cohort+': validate dates and labels; never fill scores', unit='CVE'):
        record = records[c]; a.require(record['published_date'] == p['rows'][c]['publication'], 'Publication date mismatch')
        dates, values = old.series(record)
        choices = [old.classify(values, 'exact_day7'), old.classify(values, 'last_known_by_day7'), gap_label(values, isolated=True)]
        for name, lab in zip(CONFIG['label_policies'], choices):
            labels[name][c] = lab
            audit_rows.append({'cve': c, 'policy': name, 'group': p['groups'][c], **lab})
        for i in np.flatnonzero(~np.isfinite(values[8:]))+8:
            status = (record.get('epss_score_status') or {}).get(dates[i], 'STATUS_NOT_RECORDED')
            gaps[dates[i], str(status)] += 1
    cohort_rows, table, oof = [], [], []
    for name, lab in labels.items():
        cohort_rows.append({'policy': name, **support(sorted(wanted), lab, p['rows'], p['groups'])})
        for fold in tqdm(p['folds'], desc=cohort+': inspect maturity-qualified roles ('+name+')', unit='fold'):
            for role in ROLES:
                table.append({'policy': name, 'fold': fold['fold'], 'training_target_day': fold['target_day'],
                              'role': role, 'assessment_start': fold['assessment_start'],
                              'selection_start': fold['selection_start'],
                              **support(fold[role], lab, p['rows'], p['groups'])})
        for h in CONFIG['training_target_days']:
            folds = [f for f in p['folds'] if f['target_day'] == h]
            for scope, fs in [('all_fixed_assessment_blocks', folds),
                              ('blocks_with_two_classes_in_optimization_and_selection',
                               [f for f in folds if all({lab[c]['label'] for c in f[r] if lab[c]['label'] is not None} == {0, 1}
                                                       for r in ('optimization', 'selection'))])]:
                ids = [c for f in fs for c in f['assessment']]
                a.require(len(ids) == len(set(ids)), 'Assessment CVEs repeat within scenario')
                oof.append({'policy': name, 'training_target_day': h, 'scope': scope,
                            'folds_counted': len(fs), **support(ids, lab, p['rows'], p['groups'])})
    findings = {'schema': CONFIG['schema'], 'cohort': cohort, 'config': CONFIG, 'window': p['window'],
                'original_training_cves': p['original_training_cves'], 'whole_version_cves': len(wanted),
                'cut_dates': p['cut_dates'], 'cohort_support': cohort_rows, 'assessment_support': oof,
                'role_support': table, 'pins': pins,
                'decision': 'SUPPORT_ONLY; NO_COHORT_PROMOTION_OR_TRAINING',
                'historical_inputs_verified': False, 'training': False, 'network_requests': 0, 'test_scoring': False}
    text = ['Whole-version Day-7 crossing-label and maturity support', '',
            'Original training roles only. No models, encoders, new NVD requests or official holdout scoring.',
            'The preserved Day-60-eligible parent cohorts limit scope; this is not every CVE in either dataset.',
            'Version intervals use the existing frozen release schedule; no new archive-header verification.',
            'A positive needs an observed crossing; strict negatives require all future days. An isolated-gap negative remains an assumption.',
            'The threshold is EPSS >=0.1, not confirmed exploitation. Re-crossings after an earlier high score are counted and flagged.',
            'Date cuts use no outcomes. Both optimization-to-selection and selection-to-assessment maturity are enforced.',
            'Active-role evidence groups are disjoint; links through excluded versions are preserved.',
            'Weak and empty folds remain in the report. Support counts do not prove statistical power.',
            'All fixed assessment rows are counted separately from rows in blocks with basic two-class fitting support.',
            'Do not choose a version/cohort because its model performance looks better. No fitted outcomes exist here.',
            'A future model also needs supported feature/encoder/preprocessing availability; chronology alone does not supply it.', '',
            'Label rule | Whole-version CVEs | Eligible | Risers | Non-risers | Assumed negatives']
    text += [f'{r["policy"]} | {r["cves"]} | {r["labelled"]} | {r["risers"]} | {r["non_risers"]} | {r["assumed_negatives"]}' for r in cohort_rows]
    text += ['', 'Role support is in role_support.csv, including all maturity exclusions and evidence purges.',
             'See assessment_support.csv before interpreting the whole-cohort riser count as evaluation support.']
    a.publish(ROOT/cohort/'audit', {'findings.json': findings, 'label_audit.csv': old.csv_text(audit_rows),
              'cohort_support.csv': old.csv_text(cohort_rows), 'role_support.csv': old.csv_text(table),
              'assessment_support.csv': old.csv_text(oof),
              'missing_dates.csv': old.csv_text([{'date': d, 'status': s, 'missing_cve_days': n} for (d, s), n in gaps.items()]),
              'REPORT.txt': '\n'.join(text)+'\n'}, pins, verify)
    print(f'{cohort}: corrected-label/maturity support '+('reproduced.' if verify else 'saved.')+' No training or downloads.')
