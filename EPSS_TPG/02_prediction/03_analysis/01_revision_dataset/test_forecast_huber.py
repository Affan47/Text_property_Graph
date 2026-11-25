"""Synthetic-only Huber training and isolation tests; no real dataset fitting."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor

from test_forecast_baseline import artifacts
from test_forecast_data import fixture, set_day
from epss import forecast_huber as huber
from epss.forecast_data import make_record, read_json, save_prepared, sha256, transform_statistics, write_json


def records():
    rng = np.random.default_rng(12)
    rows = [fixture(f'CVE-2023-{2000+i}', '2023-06-01') for i in range(25)]
    rows += [fixture('CVE-2024-1234', '2024-06-01'), fixture('CVE-2025-1234', '2025-06-01')]
    for i, row in enumerate(rows):
        base = .05+.02*i
        for day in range(31):
            set_day(row, day, float(max(0, min(1, base+.005*np.sin(day)+float(rng.normal(0, .002))))))
        set_day(row, 60, max(0, min(1, .04+.7*base+float(rng.normal(0, .03)))))
    return rows


def bundle(root):
    source = root/'raw/data.json'
    raw = records()
    write_json(source, raw)
    data = artifacts(raw)
    data['preparation_summary.json'].update(source_path=str(source), source_sha256=sha256(source))
    prepared = root/'prepared'
    save_prepared(prepared, data, sha256(source))
    ridge_dir = root/'ridge'
    huber.ridge.run(prepared, ridge_dir)
    freeze = root/'freeze'
    policy = {'imputation': 'none', 'test_evaluation': False, 'future_values_as_inputs': False,
        'source_sha256': sha256(source), 'preparation_lock_sha256': sha256(prepared/'preparation_lock.json')}
    write_json(freeze/'freeze_manifest.json', {'policy': policy,
        'pinned_files': {str(source): sha256(source)},
        'cohort_ids': {s: data['task_cohorts.json']['endpoint'][s] for s in ('train', 'validation')}})
    for name in ('analysis.json', 'coverage.json', 'metrics.json'):
        write_json(freeze/name, {})
    write_json(freeze/'analysis_lock.json', {'schema': 'epss_missingness_sensitivity_v1',
        'files': {p.name: sha256(p) for p in freeze.iterdir()}})
    return prepared, freeze, ridge_dir


class HuberTests(unittest.TestCase):
    def test_train_fit_and_json_predict_match_estimator(self):
        data = artifacts(records())
        model, diagnostics = huber.fit_model(data)
        ids = data['task_cohorts.json']['endpoint']['train']
        self.assertEqual(model['fit_cve_ids'], ids)
        self.assertEqual(model['preprocessor'], data['training_preprocessor.json'])
        self.assertEqual(len(diagnostics), len(ids))
        self.assertTrue(all(r['retained_in_training'] for r in diagnostics))
        x = [transform_statistics(data['inputs.json'][c], model['preprocessor']) for c in ids]
        y = [data['labels.json'][c]['y']['endpoint'] for c in ids]
        estimator = HuberRegressor(**huber.CONFIG).fit(x, y)
        record = data['inputs.json']['CVE-2024-1234']
        restored = json.loads(json.dumps(model, allow_nan=False))
        expected = estimator.predict([transform_statistics(record, model['preprocessor'])])[0]
        self.assertAlmostEqual(huber.predict(restored, record)['huber_raw_prediction'], expected, places=13)

    def test_validation_test_and_unrelated_text_do_not_change_fit(self):
        raw = records()
        before, _ = huber.fit_model(artifacts(raw))
        for row in raw[-2:]:
            for day in range(91):
                set_day(row, day, .97)
            row.update(in_vckev=True, llm_summary='Future text')
        after, _ = huber.fit_model(artifacts(raw))
        self.assertEqual(before, after)

    def test_training_targets_change_model(self):
        raw = records()
        before, _ = huber.fit_model(artifacts(raw))
        for row in raw[:-2]:
            set_day(row, 60, .9-row['epss_scores'][row['published_date']])
        after, _ = huber.fit_model(artifacts(raw))
        self.assertNotEqual(before['coefficients'], after['coefficients'])

    def test_missing_target_excluded_and_zero_input_kept(self):
        raw = records()
        set_day(raw[0], 60, None)
        model, _ = huber.fit_model(artifacts(raw))
        self.assertNotIn(raw[0]['cve'], model['fit_cve_ids'])
        record = fixture()
        for day in range(30):
            set_day(record, day, None)
        set_day(record, 30, 0)
        x, _ = make_record(record)
        self.assertEqual(x['X']['statistics']['last_score'], 0)
        self.assertTrue(np.isfinite(huber.predict(model, x)['huber_prediction']))
        before = huber.predict(model, x)
        for day in range(31, 91):
            set_day(record, day, .99)
        self.assertEqual(before, huber.predict(model, make_record(record)[0]))

    def test_clipping_and_invalid_coefficients(self):
        model, _ = huber.fit_model(artifacts(records()))
        x, _ = make_record(fixture())
        for intercept, expected in [(-2, 0), (2, 1)]:
            modified = {**model, 'coefficients': [0]*7, 'intercept': intercept}
            result = huber.predict(modified, x)
            self.assertEqual(result['huber_prediction'], expected)
            self.assertTrue(result['huber_prediction_clipped'])
        with self.assertRaises(ValueError):
            huber.predict({**model, 'intercept': float('nan')}, x)

    def test_convergence_warning_rejected(self):
        def warn(*args, **kwargs):
            warnings.warn('Synthetic convergence failure', ConvergenceWarning)
        with patch.object(HuberRegressor, 'fit', side_effect=warn), self.assertRaisesRegex(ValueError, 'did not converge'):
            huber.fit_model(artifacts(records()))

    def test_validation_alignment_and_test_guard(self):
        data = artifacts(records())
        model, _ = huber.fit_model(data)
        ridge_model = huber.ridge.fit_model(data)
        rows, _ = huber.ridge.evaluate(data, ridge_model)
        predicted, summary = huber.evaluate(data, model, rows)
        self.assertEqual(len(predicted), len(data['task_cohorts.json']['endpoint']['validation']))
        self.assertFalse(summary['test_performance_evaluated'])
        with self.assertRaises(ValueError):
            huber.evaluate(data, model, rows, split='test')
        with self.assertRaises(ValueError):
            huber.evaluate(data, model, [{**rows[0], 'target_date': '2026-01-01'}])

    def test_check_only_output_and_completed_lock(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            prepared, freeze, ridge_dir = bundle(root)
            output = root/'huber'
            before = {str(p): sha256(p) for p in root.rglob('*') if p.is_file()}
            with patch.object(huber, 'fit_model', side_effect=AssertionError('No fit during preflight')):
                huber.run(prepared, freeze, ridge_dir, output, check_only=True)
            self.assertFalse(output.exists())
            huber.run(prepared, freeze, ridge_dir, output)
            self.assertEqual(set(p.name for p in output.iterdir()), huber.OUTPUT_FILES)
            self.assertEqual(read_json(output/'run_status.json')['status'], 'complete')
            for name, digest in read_json(output/'evaluation_lock.json')['files'].items():
                self.assertEqual(sha256(output/name), digest)
            for path, digest in before.items():
                self.assertEqual(sha256(path), digest)
            with self.assertRaises(ValueError):
                huber.run(prepared, freeze, ridge_dir, output)

    def test_failed_run_is_recorded_without_certified_results(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            prepared, freeze, ridge_dir = bundle(root)
            output = root/'failed'
            with patch.object(huber, 'fit_model', side_effect=ValueError('did not converge')), self.assertRaises(ValueError):
                huber.run(prepared, freeze, ridge_dir, output)
            self.assertEqual(read_json(output/'run_status.json')['status'], 'failed')
            self.assertTrue((output/'failure.txt').exists())
            self.assertFalse((output/'evaluation_lock.json').exists())

    def test_frozen_source_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            prepared, freeze, ridge_dir = bundle(root)
            (root/'raw/data.json').write_text('[]')
            with self.assertRaises(ValueError):
                huber.preflight(prepared, freeze, ridge_dir)


if __name__ == '__main__':
    unittest.main()
