"""Synthetic-only checks for the fixed tree and frozen-run handoff."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from test_forecast_huber import bundle, huber
from test_forecast_baseline import artifacts
from test_forecast_data import fixture, set_day
from epss import forecast_tree as tree
from epss.forecast_data import make_record, read_json, sha256, transform_statistics, write_json


def records():
    raw = [fixture(f'CVE-2023-{2000+i}', '2023-06-01') for i in range(80)]
    raw += [fixture('CVE-2024-1234', '2024-06-01'), fixture('CVE-2025-1234', '2025-06-01')]
    for i, row in enumerate(raw):
        for day in range(31):
            set_day(row, day, .01+.005*i+.0001*day)
        set_day(row, 60, .05 if i < 40 else .8)
    return raw


class TreeTests(unittest.TestCase):
    def test_portable_tree_matches_sklearn_on_branching_tree(self):
        data = artifacts(records())
        model = tree.fit_model(data)
        self.assertGreater(len(model['nodes']), 1)
        ids = model['fit_cve_ids']
        x = np.asarray([transform_statistics(data['inputs.json'][c], model['preprocessor']) for c in ids], dtype=np.float32)
        y = [data['labels.json'][c]['y']['endpoint'] for c in ids]
        estimator = DecisionTreeRegressor(**tree.CONFIG).fit(x, y)
        restored = json.loads(json.dumps(model, allow_nan=False))
        for c in data['inputs.json']:
            features = [transform_statistics(data['inputs.json'][c], model['preprocessor'])]
            self.assertEqual(tree.predict(restored, data['inputs.json'][c])['tree_prediction'], estimator.predict(features)[0])
        self.assertLessEqual(estimator.get_depth(), 3)

    def test_float32_input_threshold_boundary_matches_sklearn(self):
        data = artifacts(records())
        model = tree.fit_model(data)
        ids = model['fit_cve_ids']
        x = [transform_statistics(data['inputs.json'][c], model['preprocessor']) for c in ids]
        y = [data['labels.json'][c]['y']['endpoint'] for c in ids]
        estimator = DecisionTreeRegressor(**tree.CONFIG).fit(x, y)
        for node in model['nodes']:
            if node['left'] == -1:
                continue
            nearest = np.float32(node['threshold'])
            for value in (nearest, np.nextafter(nearest, np.float32(-np.inf)),
                          np.nextafter(nearest, np.float32(np.inf))):
                probe = np.zeros(len(model['feature_names']), dtype=np.float32)
                probe[node['feature']] = value
                with patch.object(tree, 'transform_statistics', return_value=probe):
                    actual = tree.predict(model, {})['tree_prediction']
                self.assertEqual(actual, estimator.predict([probe])[0])

    def test_validation_test_and_text_cannot_change_training(self):
        raw = records()
        before = tree.fit_model(artifacts(raw))
        for row in raw[-2:]:
            for day in range(91):
                set_day(row, day, .97)
        for row in raw:
            row.update(llm_summary='Future summary', in_vckev=True)
        self.assertEqual(before, tree.fit_model(artifacts(raw)))

    def test_training_targets_affect_model(self):
        raw = records()
        before = tree.fit_model(artifacts(raw))
        for row in raw[:-2]:
            set_day(row, 60, .4)
        self.assertNotEqual(before['nodes'], tree.fit_model(artifacts(raw))['nodes'])

    def test_missing_scores_zero_and_future_input_isolation(self):
        raw = records()
        set_day(raw[0], 0, None)
        set_day(raw[0], 1, 0.0)
        data = artifacts(raw)
        model = tree.fit_model(data)
        before = tree.predict(model, data['inputs.json'][raw[0]['cve']])
        for day in range(31, 91):
            set_day(raw[0], day, .99)
        x, _ = make_record(raw[0])
        self.assertIsNone(x['X']['values'][0])
        self.assertEqual(x['X']['values'][1], 0.0)
        self.assertEqual(before, tree.predict(model, x))

    def test_rejects_bad_model_structure(self):
        model = tree.fit_model(artifacts(records()))
        for field, value in [('left', 999), ('value', float('nan')), ('feature', 99)]:
            changed = copy.deepcopy(model)
            changed['nodes'][0][field] = value
            with self.assertRaises(ValueError):
                tree.validate_model(changed)

    def test_validation_alignment_and_test_guard(self):
        data = artifacts(records())
        ridge = huber.ridge.fit_model(data)
        refs, _ = huber.ridge.evaluate(data, ridge)
        hm, _ = huber.fit_model(data)
        refs, _ = huber.evaluate(data, hm, refs)
        model = tree.fit_model(data)
        rows, summary = tree.evaluate(data, model, refs)
        self.assertEqual(len(rows), 1)
        self.assertFalse(summary['test_performance_evaluated'])
        with self.assertRaisesRegex(ValueError, 'Test scoring'):
            tree.evaluate(data, model, refs, split='test')
        refs[0]['target_date'] = '2099-01-01'
        with self.assertRaisesRegex(ValueError, 'alignment'):
            tree.evaluate(data, model, refs)

    def test_cli_artifacts_preflight_and_preserved_inputs(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            prepared, freeze, ridge = bundle(root)
            huber_dir, output = root/'huber', root/'tree'
            huber.run(prepared, freeze, ridge, huber_dir)
            before = {str(p): sha256(p) for p in root.rglob('*') if p.is_file()}
            with patch.object(tree, 'fit_model', side_effect=AssertionError('Preflight fitted')):
                tree.run(prepared, freeze, ridge, huber_dir, output, check_only=True)
            self.assertFalse(output.exists())
            tree.run(prepared, freeze, ridge, huber_dir, output)
            lock = read_json(output/'evaluation_lock.json')
            self.assertEqual(set(lock['files']), tree.OUTPUT_FILES-{'evaluation_lock.json'})
            for name, digest in lock['files'].items():
                self.assertEqual(sha256(output/name), digest)
            self.assertEqual(read_json(output/'run_status.json')['status'], 'complete')
            for p, digest in before.items():
                self.assertEqual(sha256(p), digest)
            with self.assertRaisesRegex(ValueError, 'Output exists'):
                tree.run(prepared, freeze, ridge, huber_dir, output)
            with self.assertRaisesRegex(ValueError, 'overlaps protected'):
                tree.run(prepared, freeze, ridge, huber_dir, prepared/'nested')

    def test_failed_run_has_no_success_lock(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            prepared, freeze, ridge = bundle(root)
            huber_dir, output = root/'huber', root/'tree'
            huber.run(prepared, freeze, ridge, huber_dir)
            with patch.object(tree, 'fit_model', side_effect=ValueError('synthetic failure')):
                with self.assertRaisesRegex(ValueError, 'synthetic failure'):
                    tree.run(prepared, freeze, ridge, huber_dir, output)
            self.assertEqual(read_json(output/'run_status.json')['status'], 'failed')
            self.assertFalse((output/'evaluation_lock.json').exists())
            self.assertTrue((output/'failure.txt').exists())

    def test_changed_comparator_rejected(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            prepared, freeze, ridge = bundle(root)
            huber_dir = root/'huber'
            huber.run(prepared, freeze, ridge, huber_dir)
            write_json(huber_dir/'predictions.json', [])
            with self.assertRaisesRegex(ValueError, 'Changed artifact'):
                tree.preflight(prepared, freeze, ridge, huber_dir)


if __name__ == '__main__':
    unittest.main()
