"""Offline tests for GRU input boundaries, fit isolation and saved checkpoints."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.forecast_sequence import (CONFIG, CHANNELS, OUTPUT_FILES, evaluate, fit_seed,
                                    input_tensor, predict, restore_model, run)
from epss.forecast_regression import fit_model as fit_ridge, run as run_ridge
from epss.forecast_data import make_record, read_json, sha256
from test_forecast_baseline import artifacts, bundle
from test_forecast_data import fixture, set_day, trio
from test_forecast_regression import varied_records


class SequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def short_fit(self, data, seed=17):
        return fit_seed(data, seed, {**CONFIG, 'epochs': 2, 'patience': 2})

    def test_missing_zero_and_age_channels(self):
        raw = fixture()
        set_day(raw, 0, None)
        set_day(raw, 1, 0)
        set_day(raw, 2, None)
        x = input_tensor([make_record(raw)[0]])
        self.assertEqual(tuple(x.shape), (1, 31, 5))
        self.assertEqual(x[0, 0].tolist(), [0, 0, 0, 0, 0])
        self.assertEqual(x[0, 1, :4].tolist(), [0, 1, 0, 1])
        self.assertAlmostEqual(float(x[0, 2, 2]), 1/30)
        self.assertEqual(float(x[0, 30, 4]), 1)

    def test_future_mutation_does_not_change_inputs_or_prediction(self):
        data = artifacts(trio())
        model, _ = self.short_fit(data)
        raw = fixture()
        x = make_record(raw)[0]
        before = predict(model, [x])
        for day in range(31, 91):
            set_day(raw, day, .99)
        raw['in_vckev'] = True
        after = make_record(raw)[0]
        self.assertTrue(torch.equal(input_tensor([x]), input_tensor([after])))
        self.assertEqual(before, predict(model, [after]))

    def test_no_observed_input_rejected(self):
        raw = fixture()
        for day in range(31):
            set_day(raw, day, None)
        with self.assertRaises(ValueError):
            input_tensor([make_record(raw)[0]])

    def test_unexpected_channel_rejected(self):
        x = make_record(fixture())[0]
        x['X']['future_score'] = .5
        with self.assertRaises(ValueError):
            input_tensor([x])

    def test_training_updates_do_not_use_validation_or_test(self):
        records = varied_records()
        config = {**CONFIG, 'epochs': 1}
        a, _ = fit_seed(artifacts(records), 17, config)
        for i in (1, 2):
            for day in range(91):
                set_day(records[i], day, .95)
        b, _ = fit_seed(artifacts(records), 17, config)
        self.assertEqual(a, b)

    def test_test_values_do_not_change_training_or_selection(self):
        records = varied_records()
        a = self.short_fit(artifacts(records))
        for day in range(91):
            set_day(records[2], day, .9)
        self.assertEqual(a, self.short_fit(artifacts(records)))

    def test_training_target_changes_model(self):
        records = varied_records()
        a, _ = self.short_fit(artifacts(records))
        set_day(records[0], 60, .99)
        b, _ = self.short_fit(artifacts(records))
        self.assertNotEqual(a['state_dict'], b['state_dict'])

    def test_deterministic_and_rng_restored(self):
        data = artifacts(varied_records())
        state = torch.get_rng_state().clone()
        deterministic = torch.are_deterministic_algorithms_enabled()
        a = self.short_fit(data)
        b = self.short_fit(data)
        self.assertEqual(a, b)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), deterministic)

    def test_selection_matches_history_and_json_reload(self):
        data = artifacts(varied_records())
        model, history = self.short_fit(data)
        self.assertEqual(model['selected_epoch'], min(history, key=lambda r:r['validation_mae'])['epoch'])
        ids = data['task_cohorts.json']['endpoint']['validation']
        records = [data['inputs.json'][c] for c in ids]
        values = predict(json.loads(json.dumps(model)), records)
        self.assertTrue(all(0 <= v <= 1 for v in values))
        self.assertEqual(values, predict(model, records))
        y = [data['labels.json'][c]['y']['endpoint'] for c in ids]
        mae = sum(abs(v-t) for v,t in zip(values, y))/len(y)
        self.assertAlmostEqual(mae, min(r['validation_mae'] for r in history), places=12)

    def test_model_corruption_rejected(self):
        data = artifacts(trio())
        model, _ = self.short_fit(data)
        for key, value in [('channels', ['future_score']), ('hidden_size', 32), ('fit_split', 'test')]:
            bad = copy.deepcopy(model)
            bad[key] = value
            with self.assertRaises(ValueError):
                restore_model(bad)
        model['state_dict']['head.bias'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            restore_model(model)

    def test_all_seeds_and_validation_only(self):
        data = artifacts(varied_records())
        models = [self.short_fit(data, seed)[0] for seed in CONFIG['seeds']]
        ridge = fit_ridge(data)
        rows, summary = evaluate(data, models, ridge)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(summary['per_seed']), 3)
        self.assertFalse(summary['test_performance_evaluated'])
        with self.assertRaises(ValueError):
            evaluate(data, models[:1], ridge)
        for split in ('train', 'test'):
            with self.assertRaisesRegex(ValueError, 'validation evaluation only'):
                evaluate(data, models, ridge, split)

    def test_ineligible_and_related_training_rows_excluded(self):
        records = varied_records()
        set_day(records[0], 60, None)
        records[3]['description'] = records[2]['description'] = 'Shared description'
        model, _ = self.short_fit(artifacts(records))
        self.assertEqual(model['fit_cve_ids'], [records[4]['cve']])

    def test_run_checksums_readback_and_source_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared = bundle(temporary)
            before = {p.name:sha256(p) for p in prepared.iterdir()}
            ridge = Path(temporary)/'ridge'
            run_ridge(prepared, ridge)
            output = Path(temporary)/'gru'
            with patch.dict(CONFIG, {'epochs': 2, 'patience': 2}):
                summary = run(prepared, ridge, output)
                self.assertEqual(summary, run(prepared, ridge, output, overwrite=True))
            self.assertEqual({p.name for p in output.iterdir()}, OUTPUT_FILES)
            lock = read_json(output/'evaluation_lock.json')
            self.assertTrue(all(sha256(output/k)==v for k,v in lock['files'].items()))
            self.assertEqual(before, {p.name:sha256(p) for p in prepared.iterdir()})
            for protected in (prepared, ridge, prepared/'nested'):
                with self.assertRaises(ValueError):
                    run(prepared, ridge, protected, overwrite=True)
            with self.assertRaisesRegex(ValueError, 'Output exists'):
                run(prepared, ridge, output)
            (ridge/'model.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                run(prepared, ridge, Path(temporary)/'new_gru')


if __name__ == '__main__':
    unittest.main()
