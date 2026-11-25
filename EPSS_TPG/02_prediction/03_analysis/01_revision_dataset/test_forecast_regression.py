"""Offline safeguards for the first learned EPSS-only baseline."""
import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.forecast_regression import ALPHA, OUTPUT_FILES, evaluate, fit_model, predict, run
from epss.forecast_data import make_record, read_json, sha256, transform_statistics
from test_forecast_baseline import artifacts, bundle
from test_forecast_data import fixture, set_day, trio


def varied_records():
    records = trio() + [fixture('CVE-2023-7777', '2023-07-01'),
                        fixture('CVE-2023-8888', '2023-08-01')]
    for i, row in enumerate(records):
        for day in range(91):
            set_day(row, day, min(.99, .03*i + .001*day))
    return records


class RegressionTests(unittest.TestCase):
    def test_training_ids_and_scaling(self):
        data = artifacts(varied_records())
        model = fit_model(data)
        self.assertEqual(model['fit_cve_ids'], data['task_cohorts.json']['endpoint']['train'])
        self.assertEqual(len(model['fit_cve_ids']), 3)
        self.assertEqual(model['preprocessor'], data['training_preprocessor.json'])
        self.assertEqual(model['alpha'], 1)

    def test_heldout_values_do_not_change_fit(self):
        records = varied_records()
        before = fit_model(artifacts(records))
        for index in (1, 2):
            for day in range(91):
                set_day(records[index], day, .97)
        self.assertEqual(before, fit_model(artifacts(records)))

    def test_training_targets_do_change_fit(self):
        records = varied_records()
        before = fit_model(artifacts(records))
        set_day(records[0], 60, .99)
        after = fit_model(artifacts(records))
        self.assertNotEqual(before['intercept'], after['intercept'])

    def test_missing_training_endpoint_is_excluded(self):
        records = varied_records()
        set_day(records[0], 60, None)
        model = fit_model(artifacts(records))
        self.assertNotIn(records[0]['cve'], model['fit_cve_ids'])

    def test_json_inference_matches_sklearn(self):
        data = artifacts(varied_records())
        model = json.loads(json.dumps(fit_model(data), allow_nan=False))
        ids = model['fit_cve_ids']
        x = [transform_statistics(data['inputs.json'][c], model['preprocessor']) for c in ids]
        y = [data['labels.json'][c]['y']['endpoint'] for c in ids]
        reference = Ridge(alpha=ALPHA, solver='svd').fit(x, y)
        record = data['inputs.json']['CVE-2024-1234']
        expected = reference.predict([transform_statistics(record, model['preprocessor'])])[0]
        self.assertAlmostEqual(predict(model, record)['ridge_raw_prediction'], expected, places=14)

    def test_predictor_future_and_text_isolation(self):
        model = fit_model(artifacts(varied_records()))
        raw = fixture()
        before = predict(model, make_record(raw)[0])
        for day in range(31, 91):
            set_day(raw, day, .99)
        raw.update(in_vckev=True, description='New future text')
        self.assertEqual(before, predict(model, make_record(raw)[0]))

    def test_missing_scores_and_observed_zero(self):
        data = artifacts(varied_records())
        model = fit_model(data)
        raw = fixture()
        for day in range(30):
            set_day(raw, day, None)
        set_day(raw, 30, 0)
        x, _ = make_record(raw)
        self.assertEqual(x['X']['statistics']['last_score'], 0)
        self.assertEqual(x['X']['statistics']['slope_defined'], 0)
        self.assertTrue(np.isfinite(predict(model, x)['ridge_prediction']))
        set_day(raw, 30, None)
        with self.assertRaises(ValueError):
            predict(model, make_record(raw)[0])

    def test_clipping_keeps_raw_output(self):
        data = artifacts(trio())
        model = fit_model(data)
        model['coefficients'] = [0.0]*7
        for raw, expected in [(-.5, 0), (1.5, 1), (.4, .4)]:
            model['intercept'] = raw
            result = predict(model, data['inputs.json']['CVE-2024-1234'])
            self.assertEqual(result['ridge_raw_prediction'], raw)
            self.assertEqual(result['ridge_prediction'], expected)
            self.assertEqual(result['ridge_prediction_clipped'], raw != expected)

    def test_malformed_model_rejected(self):
        data = artifacts(trio())
        original = fit_model(data)
        for key, value in [('coefficients', [0]), ('intercept', float('nan')),
                           ('fit_split', 'test'), ('feature_names', ['in_vckev']),
                           ('alpha', 99), ('fit_cve_ids', ['CVE-2025-1234'])]:
            model = copy.deepcopy(original)
            model[key] = value
            with self.assertRaises(ValueError):
                predict(model, data['inputs.json']['CVE-2024-1234'])

    def test_tampered_scaler_rejected(self):
        data = artifacts(trio())
        data['training_preprocessor.json']['means'][0] = .99
        with self.assertRaisesRegex(ValueError, 'training-only'):
            fit_model(data)

    def test_test_evaluation_blocked(self):
        data = artifacts(trio())
        model = fit_model(data)
        for split in ('train', 'test', 'all'):
            with self.assertRaisesRegex(ValueError, 'validation evaluation only'):
                evaluate(data, model, split)

    def test_matched_comparison_and_no_mutation(self):
        data = artifacts(varied_records())
        before = copy.deepcopy(data)
        rows, summary = evaluate(data, fit_model(data))
        self.assertEqual(data, before)
        self.assertEqual([r['cve'] for r in rows], data['task_cohorts.json']['endpoint']['validation'])
        self.assertEqual(summary['ridge_metrics']['n'], summary['persistence_metrics']['n'])
        self.assertEqual(sum(summary['paired_absolute_error_counts'].values()), len(rows))
        self.assertTrue(summary['model_trained'])
        self.assertFalse(summary['test_performance_evaluated'])

    def test_cross_fold_groups_not_fitted(self):
        records = varied_records()
        records[0]['description'] = records[2]['description'] = 'shared'
        model = fit_model(artifacts(records))
        self.assertNotIn(records[0]['cve'], model['fit_cve_ids'])
        self.assertNotIn(records[2]['cve'], model['fit_cve_ids'])

    def test_full_run_portable_model_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared = bundle(temporary)
            before = {p.name: sha256(p) for p in prepared.iterdir()}
            output = Path(temporary)/'ridge'
            summary = run(prepared, output)
            self.assertEqual({p.name for p in output.iterdir()}, OUTPUT_FILES)
            lock = read_json(output/'evaluation_lock.json')
            self.assertTrue(all(sha256(output/k) == v for k,v in lock['files'].items()))
            model = read_json(output/'model.json')
            rows = read_json(output/'predictions.json')
            inputs = read_json(prepared/'inputs.json')
            for row in rows:
                self.assertEqual(predict(model, inputs[row['cve']])['ridge_prediction'], row['ridge_prediction'])
            with (output/'predictions.csv').open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), len(rows))
            with self.assertRaisesRegex(ValueError, 'already exists'):
                run(prepared, output)
            self.assertEqual(summary, run(prepared, output, overwrite=True))
            self.assertEqual(before, {p.name: sha256(p) for p in prepared.iterdir()})

    def test_tampered_source_or_prepared_data_blocks_training(self):
        for target in ('raw', 'prepared'):
            with tempfile.TemporaryDirectory() as temporary:
                prepared = bundle(temporary)
                file = Path(temporary)/'raw'/'data.json' if target == 'raw' else prepared/'labels.json'
                file.write_text('[]')
                output = Path(temporary)/'ridge'
                with self.assertRaises(ValueError):
                    run(prepared, output)
                self.assertFalse(output.exists())

    def test_output_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared = bundle(temporary)
            for output in (prepared, prepared/'nested', Path(temporary)/'raw'/'results'):
                with self.assertRaises(ValueError):
                    run(prepared, output, overwrite=True)


if __name__ == '__main__':
    unittest.main()
