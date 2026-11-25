"""Synthetic reproduction, arithmetic, tamper detection and preservation checks."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import review_forecast_extrema_shift as reviewer
from test_forecast_extrema_shift import predictions, records, artifacts, set_day
from test_forecast_minmax import evidence_bundle, task
from epss import forecast_minmax_sequence as direct
from epss import forecast_minmax_residual as residual
from epss.forecast_data import sha256, read_json, write_json

audit = reviewer.audit


def example():
    raw = records()
    data = artifacts(raw)
    return audit.analyze(data, {r['cve']: r for r in raw}, predictions(data))


class ReviewTests(unittest.TestCase):
    def test_independent_checks_and_summary(self):
        a = example()
        reviewer.independent_checks(a)
        result = reviewer.describe(a)
        self.assertEqual(result['cohort_counts'], a['analysis']['cohort_counts'])
        for s,v in result['split_summary'].items():
            self.assertEqual(sum(v['change_groups'].values()), result['cohort_counts'][s])
            self.assertEqual(sum(v['future_patterns'].values()), result['cohort_counts'][s])
        text = reviewer.explanation({**result, 'verification_scope': 'Synthetic check'})
        self.assertIn('not a new evaluation subset', text)
        self.assertIn('Leave the reserved test set closed', text)

    def test_independent_corruption_detection(self):
        a = example()
        for table,key,value in (
            ('cve_cases', 'latest_input_age_days', -1),
            ('cve_cases', 'change_category', 'incorrect'),
            ('cve_cases', 'largest_jump_date', '1900-01-01'),
            ('cve_cases', 'largest_jump_on_reported_transition_date', True),
            ('validation_errors', 'joint_absolute_error', 10),
            ('concentration', 'cve_ids', ['wrong']),
            ('model_group_metrics', 'sum_joint_absolute_error', 10)):
            changed = copy.deepcopy(a)
            index = 1 if table == 'model_group_metrics' else 0
            changed[table][index][key] = value
            with self.subTest(table=table, key=key), self.assertRaises(ValueError):
                reviewer.independent_checks(changed)

    def test_zero_denominator_and_missing_day30(self):
        raw = records()
        for r in raw:
            for day in range(61):
                set_day(r, day, None if day == 30 else .1)
        data = artifacts(raw)
        a = audit.analyze(data, {r['cve']: r for r in raw}, predictions(data))
        reviewer.independent_checks(a)
        self.assertTrue(all(r['percent_total_joint_error'] is None for r in a['concentration']))
        bad = copy.deepcopy(a)
        bad['concentration'][0]['percent_total_joint_error'] = 0
        with self.assertRaisesRegex(ValueError, 'Zero denominator'):
            reviewer.independent_checks(bad)

    def test_full_synthetic_review_preserves_files_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), \
             patch.dict(direct.CONFIG, {'epochs': 2, 'patience': 2}), \
             patch.dict(residual.CONFIG, {'epochs': 2, 'patience': 2}):
            root = Path(temp)
            args = evidence_bundle(root)
            task.run(*args, root/'linear')
            direct.run(*args, root/'linear', root/'direct')
            residual.run(root/'direct', root/'residual')
            audit.run(root/'residual', root/'audit')
            before = {str(p): sha256(p) for p in root.rglob('*') if p.is_file()}
            with patch.object(residual, 'fit_seed', side_effect=AssertionError('Must not fit')), \
                 patch.object(direct, 'fit_seed', side_effect=AssertionError('Must not fit')):
                result = reviewer.review(root/'audit')
            self.assertTrue(result['checks_passed'])
            self.assertFalse(result['test_analysis_performed'])
            self.assertEqual(before, {str(p): sha256(p) for p in root.rglob('*') if p.is_file()})
            original = read_json(root/'audit'/'concentration.json')
            changed = copy.deepcopy(original)
            changed[0]['sum_joint_absolute_error'] += 1
            write_json(root/'audit'/'concentration.json', changed)
            with self.assertRaisesRegex(ValueError, 'Changed artifact'):
                reviewer.review(root/'audit')
            # Even an updated checksum cannot hide an altered derived table.
            lock = read_json(root/'audit'/'analysis_lock.json')
            lock['files']['concentration.json'] = sha256(root/'audit'/'concentration.json')
            write_json(root/'audit'/'analysis_lock.json', lock)
            with self.assertRaisesRegex(ValueError, 'Audit does not reproduce: concentration'):
                reviewer.review(root/'audit')


if __name__ == '__main__':
    unittest.main()
