"""Synthetic-only F04 tests: full future labels, task isolation and frozen evidence."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import Ridge

from test_forecast_huber import records, bundle, artifacts, set_day
from epss import forecast_minmax as task
from epss.forecast_data import load_prepared, make_record, read_json, sha256, transform_statistics, write_json


def evidence_bundle(root):
    prepared, freeze, ridge = bundle(root)
    data = load_prepared(prepared)
    verification = root/'verification'
    rows = []
    for split in ('train', 'validation'):
        for c in task.cohort(data, split):
            lab = data['labels.json'][c]
            for day, (stamp, value) in enumerate(zip(lab['future_dates'], lab['y']['trajectory']), 31):
                rows.append({'cve': c, 'split': split, 'day_since_publication': day, 'score_date': stamp,
                    'stored_score': value, 'snapshot_score': value, 'comparison': 'MATCH'})
    write_json(verification/'comparisons.json', rows)
    write_json(verification/'verification_lock.json', {'schema': 'forecast_snapshot_verification_v2',
        'files': {'comparisons.json': sha256(verification/'comparisons.json')}})
    frozen = read_json(freeze/'freeze_manifest.json')
    frozen['pinned_files'][str(verification/'verification_lock.json')] = sha256(verification/'verification_lock.json')
    write_json(freeze/'freeze_manifest.json', frozen)
    lock = read_json(freeze/'analysis_lock.json')
    lock['files']['freeze_manifest.json'] = sha256(freeze/'freeze_manifest.json')
    write_json(freeze/'analysis_lock.json', lock)
    return prepared, freeze, ridge, verification


class MinmaxTests(unittest.TestCase):
    def test_full_future_required_but_missing_past_allowed(self):
        raw = records()
        set_day(raw[0], 45, None)
        set_day(raw[1], 0, None)
        set_day(raw[1], 1, 0.0)
        data = artifacts(raw)
        ids = task.cohort(data, 'train')
        self.assertNotIn(raw[0]['cve'], ids)
        self.assertIn(raw[1]['cve'], ids)
        model = task.fit_model(data)
        self.assertEqual(model['preprocessor']['task'], 'minmax')
        self.assertEqual(model['preprocessor']['fit_cve_ids'], ids)
        self.assertNotEqual(ids, data['training_preprocessor.json']['fit_cve_ids'])
        excluded = next(r for r in task.cohort_manifest(data) if not r['included'])
        self.assertEqual(excluded['missing_future_days'], 1)

    def test_exact_zero_is_valid_and_endpoint_not_extrema(self):
        raw = records()
        set_day(raw[0], 31, 0.0)
        set_day(raw[0], 42, .99)
        set_day(raw[0], 60, .3)
        data = artifacts(raw)
        self.assertEqual(data['labels.json'][raw[0]['cve']]['y']['minmax'], [0.0, .99])
        self.assertIn(raw[0]['cve'], task.cohort(data, 'train'))

    def test_portable_predictions_match_two_output_ridge(self):
        data = artifacts(records())
        model = task.fit_model(data)
        x = [transform_statistics(data['inputs.json'][c], model['preprocessor']) for c in model['fit_cve_ids']]
        y = [data['labels.json'][c]['y']['minmax'] for c in model['fit_cve_ids']]
        estimator = Ridge(**task.CONFIG).fit(x, y)
        for c in data['inputs.json']:
            record = data['inputs.json'][c]
            expected = estimator.predict([transform_statistics(record, model['preprocessor'])])[0]
            actual = task.predict(model, record)
            np.testing.assert_allclose([actual['ridge_raw_minimum'], actual['ridge_raw_maximum']], expected, atol=1e-14)
            self.assertLessEqual(actual['ridge_minimum'], actual['ridge_maximum'])

    def test_nontraining_targets_and_text_do_not_change_fit(self):
        raw = records()
        before = task.fit_model(artifacts(raw))
        for row in raw[-2:]:
            for day in range(31, 91):
                set_day(row, day, .99)
            row.update(in_vckev=True, llm_summary='Future text')
        self.assertEqual(before, task.fit_model(artifacts(raw)))

    def test_future_values_do_not_enter_predictors(self):
        raw = records()
        data = artifacts(raw)
        model = task.fit_model(data)
        before = task.predict(model, data['inputs.json'][raw[0]['cve']])
        for day in range(31, 91):
            set_day(raw[0], day, .95)
        x, _ = make_record(raw[0])
        self.assertEqual(before, task.predict(model, x))

    def test_fixed_ordering_and_clipping(self):
        data = artifacts(records())
        model = task.fit_model(data)
        model['coefficients'] = [[0.0]*7, [0.0]*7]
        model['intercepts'] = [1.2, -.2]
        pred = task.predict(model, next(iter(data['inputs.json'].values())))
        self.assertEqual([pred['ridge_minimum'], pred['ridge_maximum']], [0, 1])
        self.assertTrue(pred['ridge_raw_crossed'])
        self.assertEqual(pred['ridge_clipped_outputs'], 2)

    def test_validation_metrics_and_no_test_evaluation(self):
        data = artifacts(records())
        model = task.fit_model(data)
        rows, summary = task.evaluate(data, model)
        r = rows[0]
        expected = (abs(r['ridge_minimum']-r['actual_minimum'])+abs(r['ridge_maximum']-r['actual_maximum']))/2
        self.assertAlmostEqual(summary['metrics']['ridge']['joint_mae'], expected)
        self.assertEqual(summary['metrics']['ridge']['joint_mae'], summary['metrics']['ridge']['equal_group_joint_mae'])
        self.assertIsNone(summary['metrics']['ridge']['minimum']['spearman_rho'])
        self.assertIn('not a confidence interval', task.report(summary))
        with self.assertRaisesRegex(ValueError, 'Test scoring'):
            task.evaluate(data, model, split='test')
        with self.assertRaisesRegex(ValueError, 'Test cohort'):
            task.cohort(data, 'test')

    def test_cli_preflight_outputs_and_no_input_changes(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            args = evidence_bundle(root)
            output = root/'minmax'
            before = {str(p): sha256(p) for p in root.rglob('*') if p.is_file()}
            with patch.object(task, 'fit_model', side_effect=AssertionError('Must not fit')):
                task.run(*args, output, check_only=True)
            self.assertFalse(output.exists())
            task.run(*args, output)
            lock = read_json(output/'evaluation_lock.json')
            self.assertEqual(set(lock['files']), task.OUTPUT_FILES-{'evaluation_lock.json'})
            for name, digest in lock['files'].items():
                self.assertEqual(sha256(output/name), digest)
            for path, digest in before.items():
                self.assertEqual(sha256(path), digest)
            with self.assertRaisesRegex(ValueError, 'Output exists'):
                task.run(*args, output)
            with self.assertRaisesRegex(ValueError, 'overlaps protected'):
                task.run(*args, args[0]/'nested')

    def test_missing_future_evidence_is_rejected_even_if_rehashed(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            prepared, freeze, ridge, verification = evidence_bundle(root)
            rows = read_json(verification/'comparisons.json')
            rows.pop()
            write_json(verification/'comparisons.json', rows)
            lock = read_json(verification/'verification_lock.json')
            lock['files']['comparisons.json'] = sha256(verification/'comparisons.json')
            write_json(verification/'verification_lock.json', lock)
            frozen = read_json(freeze/'freeze_manifest.json')
            frozen['pinned_files'][str(verification/'verification_lock.json')] = sha256(verification/'verification_lock.json')
            write_json(freeze/'freeze_manifest.json', frozen)
            lock = read_json(freeze/'analysis_lock.json')
            lock['files']['freeze_manifest.json'] = sha256(freeze/'freeze_manifest.json')
            write_json(freeze/'analysis_lock.json', lock)
            with self.assertRaisesRegex(ValueError, 'Unverified minmax target'):
                task.preflight(prepared, freeze, ridge, verification)

    def test_failure_preserved_without_success_lock(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            args = evidence_bundle(root)
            output = root/'minmax'
            with patch.object(task, 'fit_model', side_effect=ValueError('synthetic failure')):
                with self.assertRaisesRegex(ValueError, 'synthetic failure'):
                    task.run(*args, output)
            self.assertEqual(read_json(output/'run_status.json')['status'], 'failed')
            self.assertFalse((output/'evaluation_lock.json').exists())


if __name__ == '__main__':
    unittest.main()
