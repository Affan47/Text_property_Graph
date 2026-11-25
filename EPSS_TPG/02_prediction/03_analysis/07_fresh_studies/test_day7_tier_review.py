from datetime import timedelta
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'02_prediction/01_package'))
import numpy as np
import pandas as pd
from epss import day7_tier_review_v1 as review

V = 'AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H'
RECORD = {'cve': 'CVE-2022-1234', 'description': 'saved description',
          'published_timestamp': '2022-05-01T12:00:00Z', 'last_modified_timestamp': '2026-01-01T00:00:00Z',
          'cvss_vector_version': '3.1', 'cvss_vector_source': 'nvd@nist.gov', 'cvss_vector': 'CVSS:3.1/'+V}


def event(day, field='CVSS V3.1', value='NIST '+V, action='Added', old=None):
    detail = {'type': field, 'action': action, 'newValue': value}
    if old is not None:
        detail['oldValue'] = old
    return {'created': day, 'cveChangeId': day+field, 'sourceIdentifier': 'nvd@nist.gov', 'details': [detail]}


class TierTests(unittest.TestCase):
    def test_assumption_does_not_become_strict(self):
        value = review.review_fields([event('2022-05-04T00:00:00Z')], RECORD, 7)
        self.assertTrue(value['tiers']['assumption_only_no_description_events'])
        self.assertFalse(value['tiers']['strict_dated_match'])
        self.assertFalse(value['all_predictors_historically_verified'])

    def test_empty_history_never_proves_unchanged(self):
        self.assertFalse(review.review_fields([], RECORD, 7)['description_assumption_only'])

    def test_dated_description_disjoint_from_assumed(self):
        events = [event('2022-05-04T00:00:00Z'), event('2022-05-01T12:00:00Z', 'Description', 'saved description')]
        value = review.review_fields(events, RECORD, 7)
        self.assertTrue(value['tiers']['strict_dated_match'])
        self.assertFalse(value['tiers']['assumption_only_no_description_events'])
        self.assertTrue(value['tiers']['dated_or_assumed_description_cvss_match'])

    def test_late_added_description_cannot_use_assumption(self):
        events = [event('2022-05-04T00:00:00Z'), event('2022-05-12T00:00:00Z', 'Description', 'saved description')]
        self.assertFalse(review.review_fields(events, RECORD, 7)['tiers']['dated_or_assumed_description_cvss_match'])

    def test_undated_old_value_not_absence_of_events(self):
        events = [event('2022-05-04T00:00:00Z'),
                  event('2022-05-12T00:00:00Z', 'Description', 'saved description', 'Changed', 'saved description')]
        self.assertFalse(review.review_fields(events, RECORD, 7)['description_assumption_only'])

    def test_snapshot_chain_mismatch_stays_out(self):
        events = [event('2022-05-12T00:00:00Z', value='NIST '+V.replace('AV:N', 'AV:L'))]
        self.assertFalse(review.review_fields(events, RECORD, 7)['tiers']['dated_or_assumed_description_cvss_match'])

    def test_day7_assignment_fails_day6_cutoff(self):
        events = [event('2022-05-08T12:00:00Z')]
        self.assertTrue(review.review_fields(events, RECORD, 7)['tiers']['dated_or_assumed_description_cvss_match'])
        self.assertTrue(review.review_fields(events, RECORD, 6)['tiers']['cvss_no_assignment_recorded'])

    def test_cutoffs_exactly_one_day_apart(self):
        self.assertEqual(review.cutoff_at(RECORD, 7)-review.cutoff_at(RECORD, 6), timedelta(days=1))
        self.assertEqual(review.cutoff_at(RECORD, 6).isoformat(), '2022-05-07T23:59:59.999999+00:00')

    def test_later_snapshot_events_do_not_rewrite_saved_text(self):
        events = [event('2022-05-04T00:00:00Z'),
                  event('2026-02-01T00:00:00Z', 'Description', 'new text', 'Changed', 'saved description')]
        value = review.review_fields(events, RECORD, 7)
        self.assertTrue(value['description_assumption_only'])
        self.assertEqual(value['description_event_counts']['after_snapshot'], 1)

    def test_missing_saved_wording_not_assumed(self):
        value = review.review_fields([event('2022-05-04T00:00:00Z')], {**RECORD, 'description': ''}, 7)
        self.assertFalse(value['description_assumption_only'])

    def test_empty_scoring_has_undefined_metrics(self):
        frame = pd.DataFrame(columns=review.old.SEEDS)
        support, metrics, intervals, _ = review.score_tables([], frame, with_intervals=False)
        self.assertTrue(all(r['eligible_cves'] == 0 for r in support))
        self.assertTrue(all(r['auroc'] is None for r in metrics))
        self.assertEqual(intervals, [])

    def test_same_frozen_scores_across_cutoffs_and_strict_labels(self):
        frame = pd.DataFrame({seed: [0.02, 0.8] for seed in review.old.SEEDS}, index=['A', 'B'])
        rows = []
        for cutoff in review.POLICY['cutoffs']:
            for c, label, anchor in [('A', 0, 0.01), ('B', 1, 0.02)]:
                rows.append({'cve': c, 'cutoff_name': cutoff, 'fold': 1, 'component': c,
                             'label': label, 'strict_label': None if c == 'A' else label,
                             'anchor_score': anchor, 'assumed_negative': c == 'A',
                             **{name: name == 'all_received' for name in review.TIERS}})
        _, metrics, _, scores = review.score_tables(rows, frame, with_intervals=False)
        selected = [r for r in metrics if r['tier'] == 'all_received' and r['fold'] == 0
                    and r['method'] == 'frozen_D_plus_past_epss' and 'label_sensitivity' not in r]
        self.assertEqual([r['auroc'] for r in selected], [1.0, 1.0])
        strict = [r for r in metrics if 'label_sensitivity' in r and r['tier'] == 'all_received']
        self.assertTrue(all(r['cves'] == 1 and r['auroc'] is None for r in strict))
        self.assertEqual(len(scores), 2)
        self.assertEqual(scores[0]['past_epss'], 0.01)

    def test_frozen_snapshot_ignores_later_arrivals(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(review.collector, 'STORE', Path(directory)):
            for c in ('A', 'B'):
                (Path(directory)/c).mkdir(); (Path(directory)/c/'complete.json').write_text('{}')
            plan = {'assessment_ids': ['A', 'B']}
            self.assertEqual(review.receipt_ids(plan), ['A', 'B'])
            self.assertEqual(review.receipt_ids(plan, ['A']), ['A'])
            with self.assertRaises(ValueError): review.receipt_ids(plan, ['C'])

    def test_one_class_and_excluded_labels_count_separately(self):
        rows = [{'label': 1, 'component': 'x', 'assumed_negative': False},
                {'label': None, 'component': 'y', 'assumed_negative': False}]
        stats = review.summarize_members(rows)
        self.assertEqual((stats['eligible_cves'], stats['risers'], stats['excluded_label_cves']), (1, 1, 1))
        self.assertTrue(stats['very_low_event_support'])

    def test_same_date_not_claimed_after(self):
        rows = [{'cutoff_name': 'end_day7', 'label': 1, 'first_cvss_addition_day': 7,
                 'crossing_minus_first_cvss_addition_days': 0}]
        result = review.timing_summary(rows)
        self.assertEqual(result['counts'], {'same_date_order_unknown': 1})
        self.assertFalse(result['causal_claim'])


if __name__ == '__main__':
    unittest.main()
