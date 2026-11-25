from collections import Counter
from pathlib import Path
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import collector, plan, reconstruction
from .. import early_warning_gap_review_v1 as gap

a = plan.a


def paired_gain(labels, reference, candidate, groups, draws=20000):
    if len(np.unique(labels)) < 2:
        return {'status': 'UNDEFINED_ONE_CLASS_OR_EMPTY', 'intervals': []}
    _, group = np.unique(groups, return_inverse=True)
    n = len(np.unique(group)); rng = np.random.default_rng(6029); differences = []
    for start in tqdm(range(0, draws, 64), desc='Frozen-score paired group intervals', unit='batch'):
        weights = rng.multinomial(n, np.full(n, 1/n), size=min(64, draws-start))[:, group].astype(float)
        baseline = np.stack(gap.weighted_metrics(labels, reference, weights), axis=1)
        result = np.stack(gap.weighted_metrics(labels, candidate, weights), axis=1)
        differences.append(result-baseline)
    differences = np.concatenate(differences)
    valid = np.isfinite(differences).all(axis=1)
    point_b = plan.old.metrics(labels, reference); point_c = plan.old.metrics(labels, candidate)
    rows = []
    for i, metric in enumerate(('auroc', 'average_precision')):
        low, high = np.quantile(differences[valid, i], [.025, .975]) if valid.any() else (None, None)
        rows.append({'metric': metric, 'difference': point_c[metric]-point_b[metric],
                     'low': float(low) if low is not None else None, 'high': float(high) if high is not None else None})
    return {'status': 'CONDITIONAL_DIAGNOSTIC' if valid.mean() >= .95 else 'UNSTABLE_CLASS_SUPPORT',
            'groups': n, 'draws': draws, 'valid_draws': int(valid.sum()), 'intervals': rows,
            'training_label_and_input_timing_uncertainty_included': False}


def subset_metrics(ids, metadata, predictions):
    metrics, intervals = [], {}
    for name, selected in ids.items():
        selected = [c for c in selected if metadata[c]['label_not_predictor']['label'] is not None]
        if not selected:
            metrics.append({'subset': name, 'cves': 0, 'positives': 0, 'metric_status': 'EMPTY_SUBSET'})
            continue
        y = np.array([metadata[c]['label_not_predictor']['label'] for c in selected])
        epss = np.array([metadata[c]['label_not_predictor']['anchor_score'] for c in selected])
        values = predictions.loc[selected, plan.old.SEEDS].to_numpy().mean(1)
        for method, score in [('past_epss', epss), ('frozen_D', values), ('frozen_D_plus_past_epss', gap.logit(values)+gap.logit(epss))]:
            metrics.append({'subset': name, 'method': method, 'cves': len(selected), 'positives': int(y.sum()),
                            **plan.old.metrics(y, score)})
        if name == 'description_cvss_supported_ONLY':
            intervals[name] = paired_gain(y, epss, gap.logit(values)+gap.logit(epss),
                                          np.array([metadata[c]['component'] for c in selected]))
    return metrics, intervals


def report(scope, name, allow_partial=False, verify=False):
    a.require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,79}', name), 'Unsafe report name')
    p = plan.load(); ids = p['assessment_ids'] if scope == 'assessment' else p['cohort_ids']
    pins = dict(p['input_hashes']); pins[str(plan.ROOT/'plan/plan.json')] = a.sha(plan.ROOT/'plan/plan.json')
    selected_ids = set(ids)
    records = {r['cve']: r for r in a.read(plan.old.SOURCE) if r['cve'] in selected_ids}
    output = plan.ROOT/'reports'/name
    frozen_received = None
    if verify:
        a.check_lock(output)
        saved = a.read(output/'findings.json')
        a.require(saved['scope'] == scope, 'Report scope changed')
        frozen_received = set(a.read(output/'field_reconstructions.json'))
    rows, reconstructions, missing = [], {}, []
    with collector.exclusive():
        for cve in tqdm(ids, desc='Reconstruct Day-7 fields from complete receipts', unit='CVE'):
            if frozen_received is not None and cve not in frozen_received:
                missing.append(cve); continue
            events, evidence = collector.read_complete(cve); pins.update(evidence)
            if events is None:
                missing.append(cve); continue
            result = reconstruction.reconstruct(events, records[cve], p['records'][cve])
            reconstructions[cve] = result
            r = {'cve': cve, 'assessment': p['records'][cve]['assessment'],
                 'description_status': result['description']['status'], 'cvss_status': result['cvss']['status'],
                 'description_supported': result['description']['supported'], 'cvss_supported': result['cvss']['supported'],
                 'description_matches_saved': result['description']['same_as_saved'], 'cvss_matches_saved': result['cvss']['same_as_saved'],
                 **{k: result[k] for k in ('description_cvss_match_supported', 'first_recorded_cvss_addition_day',
                                         'crossing_day_minus_cvss_addition_day', 'all_saved_D_inputs_verified')},
                 'label': result['label_not_predictor'].get('label'),
                 'label_status': result['label_not_predictor'].get('status', 'OUTSIDE_EARLY_WARNING_ASSESSMENT')}
            rows.append(r)
        a.require(allow_partial or not missing, 'Collection incomplete; use --allow-partial and a separate report name')
        metrics, intervals = [], {}
        if rows:
            o = pd.read_csv(plan.old.REPORT/'oof_predictions.csv')
            predictions = o[(o.policy == 'main') & (o.model == 'D')].pivot(index='cve', columns='seed', values='prediction')
            assessed = [c for c in reconstructions if p['records'][c]['assessment']]
            supported = [c for c in assessed if reconstructions[c]['description_cvss_match_supported']]
            metrics, intervals = subset_metrics({'downloaded_assessment': assessed, 'description_cvss_supported_ONLY': supported}, p['records'], predictions)
        supports = []
        for group in ('all_requested', 'assessment', 'observed_risers'):
            members = [c for c in ids if group == 'all_requested' or (p['records'][c]['assessment'] and
                       (group == 'assessment' or p['records'][c]['label_not_predictor']['label'] == 1))]
            received = [reconstructions[c] for c in members if c in reconstructions]
            support = sum(r['description_cvss_match_supported'] for r in received)
            supports.append({'group': group, 'requested_cves': len(members), 'complete_histories': len(received),
                             'supported_saved_description_and_cvss': support,
                             'share_of_requested': support/len(members) if members else None,
                             'share_of_received': support/len(received) if received else None})
        findings = {'schema': 'day7_availability_report_v1', 'scope': scope, 'requested': len(ids), 'downloaded': len(rows),
                    'pending': len(missing), 'coverage_partial': bool(missing), 'support': supports,
                    'description_status': dict(Counter(r['description_status'] for r in rows)),
                    'cvss_status': dict(Counter(r['cvss_status'] for r in rows)), 'ranking_metrics': metrics,
                    'paired_ranking_gain': intervals,
                    'strict_all_predictor_gate': 'BLOCKED_REFERENCE_HISTORY_AND_MODEL_TIMING',
                    'decision': 'NO_TRAINING_OR_AUTOMATIC_TARGET_SWITCH; FIELD_SUPPORT_IS_NOT_A_CAUSAL_RESULT',
                    'training': False, 'test_scoring': False, 'historical_reference_counts_verified': False}
        content = ['Day-7 public-history input-availability audit', '',
                   'No training, changed model predictions, raw-score imputation, or official validation/test scoring.',
                   'A complete API response means all advertised pages were received, not that initial field history exists.',
                   'The primary cutoff is end of Day 7 UTC; publication +168 hours is reported separately.',
                   'Fields are reversed from the saved NVD snapshot using exact event-value chains. Ambiguity stays unresolved.',
                   'No-assignment-recorded describes one source/version channel, not proof that no other provider had CVSS.',
                   'All new subset rankings are exploratory and on matched CVEs; changing eligibility changes prevalence.',
                   'Description/CVSS-only support does NOT verify reference counts/flags or that the fitted model existed by Day 7.',
                   'CVSS/crossing ordering uses dates; unknown daily EPSS release times prevent exact same-day ordering.',
                   'An association between NVD enrichment and EPSS change does not demonstrate causation.',
                   'Neither gain nor loss in a selected small subset automatically authorizes training or a target switch.',
                   '', a.dump(findings)]
        a.publish(output, {'findings.json': findings, 'field_reconstructions.json': reconstructions,
                  'availability.csv': plan.old.csv_text(rows), 'pending_cves.json': missing,
                  'ranking_metrics.csv': plan.old.csv_text(metrics), 'REPORT.txt': '\n'.join(content)+'\n'}, pins, verify)
    print('Day-7 availability report '+('verified' if verify else 'saved')+'. No training or test scoring.')
