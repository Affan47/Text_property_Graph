"""Offline verification and full-scope handoff regression tests."""
from contextlib import redirect_stdout
import io
import unittest

import review_forecast_snapshots as review
import verify_forecast_snapshots as verify
from epss.forecast_data import read_json, sha256, write_json
from test_forecast_evidence import fixture
import test_forecast_snapshots as snapshot_fixtures


class ReviewTests(unittest.TestCase):
    def test_independent_comparison(self):
        for before, after, valid, expected in [
            (.1, .1, True, ('MATCH', 0.0)),
            (.1, .11, True, ('MISMATCH', .01)),
            (None, None, True, ('BOTH_ABSENT', None)),
            (None, 0, True, ('MISSING_STORED_AVAILABLE', None)),
            (0, None, True, ('STORED_PRESENT_CVE_ABSENT', None)),
            (.1, None, False, ('UNRESOLVED_SNAPSHOT', None)),
        ]:
            with self.subTest(expected=expected):
                self.assertEqual(review.independent_comparison(before, after, valid), expected)

    def test_uniform_plan_not_limited_to_three_error_cases(self):
        _, x, y, _ = fixture()
        first, second = 'CVE-2024-1000', 'CVE-2024-2000'
        data = {'inputs.json': {first: x, second: x}, 'labels.json': {first: y, second: y},
                'task_cohorts.json': {'endpoint': {'train': [first], 'validation': [second], 'test': ['CVE-2025-3000']}},
                'preparation_summary.json': {'source_sha256': 'fixture'}}
        plan = review.expansion_plan(data, True)
        self.assertEqual(plan['cves_by_split'], {'train': 1, 'validation': 1})
        self.assertEqual(plan['cve_days'], 122)
        self.assertFalse(plan['test_comparison_allowed'])
        self.assertFalse(plan['training_allowed'])

    def test_no_clean_conclusion_if_values_disagree(self):
        result = {'scope': 'train-validation', 'archives_reparsed': 1, 'cve_summary': [],
                  'comparison_counts': {'MISMATCH': 1}, 'all_dates_resolved': True}
        text = review.explanation(result)
        self.assertIn('require review', text)
        self.assertNotIn('Every present score', text)

    def test_no_clean_conclusion_for_unresolved_dates(self):
        result = {'scope': 'diagnostic', 'archives_reparsed': 0, 'cve_summary': [],
                  'comparison_counts': {'UNRESOLVED_SNAPSHOT': 1}, 'all_dates_resolved': False}
        self.assertNotIn('No numerical', review.explanation(result))


class LockedRunTests(unittest.TestCase):
    def setUp(self):
        self.fixture = snapshot_fixtures.SnapshotTests()
        self.fixture.setUpClass()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.archive()
        self.output = self.fixture.root/'output'
        with redirect_stdout(io.StringIO()):
            verify.run({'2024-01-01': [self.fixture.row()]}, self.fixture.identity(),
                       self.fixture.module, self.fixture.cache, self.output, False, False)

    def test_lock_and_all_checkpoint_hashes(self):
        lock, state = review.load_checked_run(self.output)
        self.assertEqual(set(lock['files']), review.FILES)
        self.assertEqual(state['run_status'], 'complete')

    def test_changed_comparison_is_rejected(self):
        (self.output/'comparisons.json').write_text('[]')
        with self.assertRaisesRegex(ValueError, 'Changed result'):
            review.load_checked_run(self.output)

    def test_running_checkpoint_even_with_updated_hash_is_rejected(self):
        state = read_json(self.output/'state.json')
        state['run_status'] = 'running'
        write_json(self.output/'state.json', state)
        lock = read_json(self.output/'verification_lock.json')
        lock['files']['state.json'] = sha256(self.output/'state.json')
        write_json(self.output/'verification_lock.json', lock)
        with self.assertRaisesRegex(ValueError, 'not finished'):
            review.load_checked_run(self.output)


if __name__ == '__main__':
    unittest.main()
