"""Synthetic checks: metadata-only projection, chronological maturity and grouping."""
import copy
from datetime import date, timedelta
import unittest

import review_forecast_cohort_design as review


def record(cve='CVE-2023-10000', published='2023-01-01'):
    start = date.fromisoformat(published)
    return {'cve': cve, 'published_date': published, 'published_timestamp': published + 'T12:00:00Z',
            'description': 'A sample vulnerability description.',
            'epss_scores': {(start + timedelta(days=i)).isoformat(): .2 for i in range(91)}}


def evidence(kind, ids, key, used=True):
    return {'kind': kind, 'cves': ids, 'key_sha256': key, 'used_for_grouping': used}


class CohortDesignTests(unittest.TestCase):
    def test_score_magnitude_cannot_change_projection(self):
        a = record()
        b = copy.deepcopy(a)
        b['epss_scores'] = {d: .9 for d in b['epss_scores']}
        self.assertEqual(review.availability(a), review.availability(b))
        self.assertNotIn('epss_scores', review.availability(a))

    def test_partial_history_and_zero_are_eligible(self):
        a = record()
        days = sorted(a['epss_scores'])
        a['epss_scores'] = {d: None for d in days}
        a['epss_scores'][days[60]] = 0
        a['epss_scores'][days[90]] = 0
        r = review.availability(a)
        self.assertTrue(r['matched_eligible'])
        self.assertEqual(r['recent_observed_days'], 1)

    def test_no_target_not_filled(self):
        a = record()
        a['epss_scores'][max(a['epss_scores'])] = None
        self.assertFalse(review.availability(a)['matched_eligible'])

    def test_older_input_is_separate_eligibility(self):
        a = record()
        for d in sorted(a['epss_scores'])[30:61]:
            a['epss_scores'][d] = None
        r = review.availability(a)
        self.assertFalse(r['matched_eligible'])
        self.assertTrue(r['long_history_eligible'])
        self.assertTrue(r['no_recent_input_with_description_and_target'])

    def test_absent_description_does_not_become_text_only(self):
        a = record()
        a['description'] = '  '
        self.assertFalse(review.availability(a)['description_only_potential'])

    def test_invalid_values_and_dates_rejected(self):
        for value in (True, float('nan'), 2, '0.2'):
            a = record()
            a['epss_scores'][min(a['epss_scores'])] = value
            with self.assertRaises(ValueError):
                review.availability(a)
        a = record()
        a['published_date'] = '2023-01-02'
        with self.assertRaises(ValueError):
            review.availability(a)

    def test_grouping_transitive_stable_and_ignored_links(self):
        es = [evidence('description', ['a', 'b'], 'd'), evidence('url', ['b', 'c'], 'u'),
              evidence('url', ['c', 'z'], 'ignored', False)]
        ids = ['a', 'b', 'c', 'z']
        g = review.grouping(ids, es)
        self.assertEqual(g['a'], g['c'])
        self.assertNotEqual(g['a'], g['z'])
        self.assertEqual(g, review.grouping(reversed(ids), list(reversed(es))))
        text = review.grouping(ids, es, keep_urls=False)
        self.assertEqual(text['a'], text['b'])
        self.assertNotEqual(text['a'], text['c'])
        self.assertEqual(text, review.grouping(ids, es, omitted_url_hash='u'))

    def test_maturity_boundary_and_observation_cutoff(self):
        policy = {'validation_start': '2024-01-01', 'test_start': '2025-01-01',
                  'test_end': '2026-01-01', 'observation_end': '2026-09-29'}
        rows = {}
        for i, published in enumerate(('2023-01-01', '2023-10-03', '2024-10-03', '2025-01-01', '2026-07-02')):
            c = f'CVE-2023-{10000+i}'
            rows[c] = review.availability(record(c, published))
        old = {c: {'split': 'test'} for c in rows}
        manifest, result = review.compare(rows, {c: c for c in rows}, policy, old)
        self.assertEqual(manifest['CVE-2023-10001']['exclusion_reason'], 'label_crosses_train_boundary')
        self.assertEqual(manifest['CVE-2023-10002']['exclusion_reason'], 'label_crosses_validation_boundary')
        self.assertEqual(manifest['CVE-2023-10004']['exclusion_reason'], 'target_after_observation_end')
        self.assertEqual(result['former_test_to_development'], 1)
        self.assertEqual(sum(v['assigned'] for v in result['counts'].values()), 5)

    def test_shared_group_is_quarantined_and_weaker_policy_flags_risk(self):
        a = review.availability(record('CVE-2023-10000', '2023-01-01'))
        b = review.availability(record('CVE-2024-10000', '2024-01-01'))
        rows = {r['cve']: r for r in (a, b)}
        old = {c: {'split': 'test'} for c in rows}
        es = [evidence('url', list(rows), 'u')]
        policy = {'validation_start': '2024-01-01', 'test_start': '2025-01-01',
                  'test_end': '2026-01-01', 'observation_end': '2026-09-29'}
        manifest, result = review.compare(rows, review.grouping(rows, es), policy, old)
        self.assertEqual(result['counts']['excluded']['assigned'], 2)
        weak, _ = review.compare(rows, review.grouping(rows, es, keep_urls=False), policy, old)
        self.assertEqual(review.cross_role_evidence(es, weak), {'url': 1})
        self.assertEqual(review.cross_role_evidence(es, manifest), {})


if __name__ == '__main__':
    unittest.main()
