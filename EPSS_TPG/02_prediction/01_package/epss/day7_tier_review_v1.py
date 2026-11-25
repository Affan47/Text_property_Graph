"""Offline, immutable receipt snapshots; assumptions never become verified inputs."""
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from .day7_availability_v1 import collector, plan, reconstruction
from .day7_availability_v1.report import paired_gain
from . import early_warning_gap_review_v1 as gap

a = plan.a
old = plan.old
ROOT = old.BASE/'21_day7_tier_review_v1/01_reddit'
CLI = old.CLI.with_name('26_review_day7_tiers.py')
TIERS = ('all_received', 'strict_dated_match', 'assumption_only_no_description_events',
         'dated_or_assumed_description_cvss_match', 'cvss_no_assignment_recorded')
POLICY = {
    'schema': 'day7_tier_review_v1',
    'scope': 'Frozen assessment IDs only, all from original training roles',
    'cutoffs': {'end_day6': 6, 'end_day7': 7},
    'label_policy': 'isolated_one_day_assumption; strict labels retained as sensitivity',
    'description_assumption': 'Nonempty saved wording with no Description event through the saved snapshot; '
                              'nonempty history required; assumes complete initial wording/change history',
    'ranking': 'Unchanged CT07 main D scores, mean over seeds 17/29/43; fixed logit sum with past EPSS',
    'intervals': {'draws': 20000, 'seed': 6029, 'unit': 'eligible evidence group',
                  'contrast': 'D plus past EPSS minus past EPSS; mean-seed fixed scores only'},
    'low_support_warning': 'Fewer than 10 positives/negatives or fewer than 5 positive/negative groups; '
                           'descriptive warning only, not a model selection gate',
    'tier_relationship': 'Strict and assumption-only are disjoint; their union is the relaxed tier',
    'training': False, 'network': False, 'official_validation_or_test_scoring': False,
    'reference_history_verified': False, 'model_time_verified': False,
}


def cutoff_at(record, day):
    publication = plan.timestamp(record['published_timestamp'])
    return datetime.combine(publication.date()+timedelta(days=day+1),
                            datetime.min.time(), timezone.utc)-timedelta(microseconds=1)


def review_fields(events, record, day):
    cutoff = cutoff_at(record, day)
    saved_at = plan.timestamp(record['last_modified_timestamp'])
    d = reconstruction.reconstruct_field(events, 'description', record, cutoff)
    c = reconstruction.reconstruct_field(events, 'cvss', record, cutoff)
    counts = Counter('through_snapshot' if plan.timestamp(e['created']) <= saved_at else 'after_snapshot'
                     for e in events for item in e['details'] if item.get('type') == 'Description')
    wording = record.get('description')
    assumed = bool(events and isinstance(wording, str) and wording.strip()
                   and counts['through_snapshot'] == 0 and saved_at >= cutoff
                   and d['status'] == 'HISTORY_CONSISTENT_BUT_ORIGIN_UNDATED' and d['same_as_saved'])
    dated_description = bool(d['supported'] and d['same_as_saved'])
    dated_cvss = bool(c['supported'] and c['same_as_saved'])
    flags = {'all_received': True, 'strict_dated_match': dated_description and dated_cvss,
             'assumption_only_no_description_events': assumed and dated_cvss,
             'dated_or_assumed_description_cvss_match': (dated_description or assumed) and dated_cvss,
             'cvss_no_assignment_recorded': c['status'] == 'NO_ASSIGNMENT_RECORDED_BY_CUTOFF'}
    return {'cutoff': cutoff.isoformat(), 'description': d, 'cvss': c,
            'description_event_counts': dict(counts), 'description_assumption_only': assumed,
            'tiers': flags, 'all_predictors_historically_verified': False}


def receipt_ids(p, saved=None):
    """Freeze IDs before reading immutable receipts; never acquire the collector lock."""
    if saved is not None:
        a.require(saved == sorted(set(saved)) and set(saved) <= set(p['assessment_ids']), 'Unsafe snapshot IDs')
        return saved
    return sorted(c for c in p['assessment_ids'] if (collector.STORE/c/'complete.json').is_file())


def summarize_members(members):
    labelled = [r for r in members if r['label'] is not None]
    positives = [r for r in labelled if r['label'] == 1]
    negatives = [r for r in labelled if r['label'] == 0]
    pg = {r['component'] for r in positives}; ng = {r['component'] for r in negatives}
    return {'received_cves': len(members), 'eligible_cves': len(labelled), 'risers': len(positives),
            'non_risers': len(negatives), 'excluded_label_cves': len(members)-len(labelled),
            'assumed_negative_cves': sum(r['assumed_negative'] for r in labelled),
            'positive_groups': len(pg), 'negative_groups': len(ng),
            'eligible_groups': len({r['component'] for r in labelled}),
            'prevalence': len(positives)/len(labelled) if labelled else None,
            'very_low_event_support': len(positives) < 10 or len(negatives) < 10 or len(pg) < 5 or len(ng) < 5}


def score_tables(rows, predictions, with_intervals=True):
    support, metrics, intervals, scores = [], [], [], []
    interval_cache = {}
    for cutoff in POLICY['cutoffs']:
        for tier in TIERS:
            members = [r for r in rows if r['cutoff_name'] == cutoff and r[tier]]
            stats = summarize_members(members)
            support.append({'cutoff': cutoff, 'tier': tier, **stats})
            selected = [r for r in members if r['label'] is not None]
            ids = [r['cve'] for r in selected]
            y = np.array([r['label'] for r in selected], dtype=int)
            ep = np.array([r['anchor_score'] for r in selected], dtype=float)
            fits = predictions.loc[ids, old.SEEDS].to_numpy(dtype=float)
            mean = fits.mean(axis=1)
            candidates = {'past_epss': ep, 'frozen_D_mean': mean,
                          'frozen_D_plus_past_epss': gap.logit(mean)+gap.logit(ep)}
            for j, seed in enumerate(old.SEEDS):
                candidates[f'frozen_D_plus_past_epss_seed_{seed}'] = gap.logit(fits[:, j])+gap.logit(ep)
            fold_ids = np.array([r['fold'] for r in selected], dtype=int)
            for fold in (0, 1, 2, 3):
                mask = np.ones(len(y), dtype=bool) if fold == 0 else fold_ids == fold
                for name, value in candidates.items():
                    metrics.append({'cutoff': cutoff, 'tier': tier, 'fold': fold, 'method': name,
                                    'cves': int(mask.sum()), 'risers': int(y[mask].sum()),
                                    **old.metrics(y[mask], value[mask])})
            # Strict-label sensitivity keeps the same availability rule, not an imputed target.
            strict = np.array([r['strict_label'] is not None for r in selected], dtype=bool)
            for name in ('past_epss', 'frozen_D_mean', 'frozen_D_plus_past_epss'):
                metrics.append({'cutoff': cutoff, 'tier': tier, 'fold': 0, 'method': name,
                                'label_sensitivity': 'strict_observed_future_only',
                                'cves': int(strict.sum()), 'risers': int(y[strict].sum()),
                                **old.metrics(y[strict], candidates[name][strict])})
            if with_intervals:
                key = tuple(ids)
                if key not in interval_cache:
                    interval_cache[key] = paired_gain(y, ep, candidates['frozen_D_plus_past_epss'],
                                                     np.array([r['component'] for r in selected]),
                                                     draws=POLICY['intervals']['draws'])
                result = interval_cache[key]
                for interval in result['intervals']:
                    intervals.append({'cutoff': cutoff, 'tier': tier, **interval,
                                      **{k: v for k, v in result.items() if k != 'intervals'},
                                      'very_low_event_support': stats['very_low_event_support'],
                                      'multiplicity_adjusted': False})
            if cutoff == 'end_day7' and tier == 'all_received':
                scores.extend({'cve': r['cve'], 'fold': r['fold'], 'label': r['label'],
                               'component': r['component'], **{k: float(v[i]) for k, v in candidates.items()}}
                              for i, r in enumerate(selected))
    return support, metrics, intervals, scores


def timing_summary(rows):
    risers = [r for r in rows if r['cutoff_name'] == 'end_day7' and r['label'] == 1]
    known = [r for r in risers if r['crossing_minus_first_cvss_addition_days'] is not None]
    counts = Counter()
    for r in known:
        lag = r['crossing_minus_first_cvss_addition_days']
        key = ('crossing_before_first_recorded_cvss' if lag < 0 else
               'same_date_order_unknown' if lag == 0 else
               'crossing_next_date' if lag == 1 else 'crossing_later')
        counts[key] += 1
    return {'observed_risers': len(risers), 'dated_first_cvss_addition': len(known),
            'missing_addition_date': len(risers)-len(known), 'counts': dict(counts),
            'cvss_first_added_on_day7': sum(r['first_cvss_addition_day'] == 7 for r in risers),
            'causal_claim': False, 'timing_resolution': 'Date-level EPSS; first recorded matching provider/version addition only'}


def run(name, allow_partial=False, verify=False):
    a.require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,79}', name), 'Unsafe report name')
    p = plan.load(); output = ROOT/name
    pins = dict(p['input_hashes'])
    files = (Path(__file__), CLI, CLI.with_name('test_day7_tier_review.py'), plan.ROOT/'plan/plan.json')
    pins.update({str(path): a.sha(path) for path in files})
    a.check_lock(old.REPORT); a.check_lock(old.PROTOCOL.parent)
    if verify:
        a.check_lock(output)
        snapshot = a.read(output/'snapshot.json')
        a.require(snapshot['policy'] == POLICY, 'Report policy changed')
        received = receipt_ids(p, snapshot['received_ids'])
    else:
        a.require(not output.exists(), 'Report exists; use --verify-only or a new name')
        received = receipt_ids(p)
        snapshot = {'policy': POLICY, 'captured_at_utc': datetime.now(timezone.utc).isoformat(),
                    'received_ids': received, 'requested_ids': p['assessment_ids'],
                    'pending_ids': sorted(set(p['assessment_ids'])-set(received))}
    a.require(snapshot['requested_ids'] == p['assessment_ids'], 'Requested cohort changed')
    a.require(snapshot['pending_ids'] == sorted(set(p['assessment_ids'])-set(received)), 'Pending IDs changed')
    a.require(allow_partial or not snapshot['pending_ids'], 'Collection incomplete; use --allow-partial with a new report name')
    selected_ids = set(received)
    raw = {r['cve']: r for r in a.read(old.SOURCE) if r['cve'] in selected_ids}
    a.require(set(raw) == selected_ids, 'Missing saved CVEs')
    protocol = a.read(old.PROTOCOL)
    folds = {c: f['fold'] for f in protocol['folds'] for c in f['main']['full']['assessment']}
    fields, rows, transport = {}, [], []
    for cve in tqdm(received, desc='Read immutable receipts; collector remains unlocked', unit='CVE'):
        events, receipt_pins = collector.read_complete(cve)
        a.require(events is not None, 'Frozen complete receipt disappeared: '+cve)
        pins.update(receipt_pins)
        meta, record = p['records'][cve], raw[cve]
        lab = meta['label_not_predictor']
        current = reconstruction.reconstruct(events, record, meta)
        transport.append({'cve': cve, 'events': len(events),
                          'pages': len(a.read(collector.STORE/cve/'complete.json')['pages'])})
        fields[cve] = {}
        for name_cutoff, day in POLICY['cutoffs'].items():
            value = review_fields(events, record, day); fields[cve][name_cutoff] = value
            rows.append({'cve': cve, 'fold': folds[cve], 'component': meta['component'], 'cutoff_name': name_cutoff,
                         'label': lab['label'], 'strict_label': lab['strict_label'],
                         'assumed_negative': lab['assumed_negative'], 'label_status': lab['status'],
                         'anchor_score': lab['anchor_score'], 'last_observation_day': lab['last_observation_day'],
                         'description_status': value['description']['status'], 'cvss_status': value['cvss']['status'],
                         'description_event_count': value['description_event_counts'].get('through_snapshot', 0),
                         'assumed_description': value['description_assumption_only'],
                         'first_cvss_addition_day': current['first_recorded_cvss_addition_day'],
                         'crossing_minus_first_cvss_addition_days': current['crossing_day_minus_cvss_addition_day'],
                         **value['tiers']})
    o = pd.read_csv(old.REPORT/'oof_predictions.csv')
    o = o[(o.policy == 'main') & (o.model == 'D') & o.cve.isin(selected_ids)]
    a.require(len(o) == 3*len(received) and not o.duplicated(['cve', 'seed']).any(), 'Incomplete D predictions')
    a.require(all(int(r.fold) == folds[r.cve] for r in o.itertuples()), 'Prediction fold mismatch')
    predictions = o.pivot(index='cve', columns='seed', values='prediction').reindex(columns=old.SEEDS)
    a.require(np.isfinite(predictions.to_numpy()).all(), 'Nonfinite frozen score')
    print('[NO FIT] Frozen CT07 scores; overlapping availability tiers; paired 20,000-draw group intervals.', flush=True)
    support, metrics, intervals, scores = score_tables(rows, predictions)
    end7 = [r for r in rows if r['cutoff_name'] == 'end_day7']
    findings = {'schema': POLICY['schema'], 'requested_cves': len(p['assessment_ids']),
                'received_cves': len(received), 'pending_cves': len(snapshot['pending_ids']),
                'partial_ordered_collection': bool(snapshot['pending_ids']),
                'description_no_recorded_event_through_snapshot': sum(r['description_event_count'] == 0 for r in end7),
                'empty_histories': sum(r['events'] == 0 for r in transport),
                'history_pages': dict(Counter(r['pages'] for r in transport)),
                'cvss_status_end_day7': dict(Counter(r['cvss_status'] for r in end7)),
                'support': support, 'timing': timing_summary(rows),
                'decision': 'TIMING_AUDIT_ONLY; NO_RETRAINING_AUTHORIZED',
                'all_predictor_gate': 'BLOCKED_REFERENCE_HISTORY_MODEL_TIMING_AND_DAY90_SPLITS',
                'source_pins': pins, 'training': False, 'network_requests': 0, 'test_scoring': False}
    content = ['Day-6 / Day-7 historical-input sensitivity report', '',
               'Only immutable completed receipts; no collector lock, download, fitting, or official holdout use.',
               'The snapshot fixes its received IDs. Verification ignores later arrivals.',
               'No recorded description event does not establish unchanged wording since publication.',
               'The relaxed tier assumes complete initial wording/change history; it is NOT verified history.',
               'Strict and assumption-only tiers are disjoint. Their union is the relaxed tier, not an independent cohort.',
               'End of Day 6 is a timing sensitivity, not a newly optimized cutoff. The EPSS comparator stays last known by Day 7.',
               'Date-only EPSS observations have unknown release times; same-date CVSS/EPSS order stays unknown.',
               'The no-assignment group is a retrospective leakage diagnostic, not an available-input cohort.',
               'One-provider/version absence is not proof that every provider lacked a severity assessment.',
               'The fixed logit sum is a ranking score, not a calibrated future-crossing probability.',
               'Intervals condition on fixed three-seed predictions and assumed gap labels, excluding training/lineage uncertainty.',
               'Very small event counts, many overlapping comparisons and ordered partial collection limit interpretation.',
               'Pooled folds and within-fold rankings are both reported; fold-specific score levels can affect pooled ranking.',
               'No description/CVSS subset verifies historical reference counts, fitted preprocessors, model availability or Day-90 maturity.',
               'Do not infer the number of available risers at completion by scaling this CVE-ID-ordered partial sample.',
               'A future retrain needs a new chronological Day-90 protocol and verified/missing/unknown feature states.',
               'A description-only arm must truly exclude current CVSS/reference features; existing D is not such an arm.',
               'Changed historical wording/vectors cannot reuse mismatched current embeddings/features.', '',
               'Cutoff | Tier | Eligible CVEs | Risers | Positive evidence groups']
    content += [f'{r["cutoff"]} | {r["tier"]} | {r["eligible_cves"]} | {r["risers"]} | {r["positive_groups"]}' for r in support]
    content += ['', 'No causal claim: CVSS arrival and EPSS changes may share upstream signals.', '',
                a.dump({k: v for k, v in findings.items() if k != 'source_pins'})]
    a.publish(output, {'snapshot.json': snapshot, 'findings.json': findings, 'field_reconstructions.json': fields,
                      'availability.csv': old.csv_text(rows), 'support.csv': old.csv_text(support),
                      'ranking_metrics.csv': old.csv_text(metrics), 'paired_intervals.csv': old.csv_text(intervals),
                      'ranking_scores.csv': old.csv_text(scores), 'transport_counts.csv': old.csv_text(transport),
                      'REPORT.txt': '\n'.join(content)+'\n'}, pins, verify)
    print('Day-7 tier review '+('reproduced: ' if verify else 'saved: ')+str(output)+'. No collector changes, requests or training.')
