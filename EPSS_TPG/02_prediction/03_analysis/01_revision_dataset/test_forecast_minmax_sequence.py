"""Ordered-head invariants, training isolation and synthetic F04-GRU runs."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_forecast_minmax import evidence_bundle, task, records, artifacts, set_day
from epss import forecast_minmax_sequence as gru
from epss.forecast_data import make_record, read_json, sha256, write_json


class MinmaxSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fit(self, raw=None, seed=17, epochs=2):
        return gru.fit_seed(artifacts(records() if raw is None else raw), seed,
            {**gru.CONFIG, 'epochs': epochs, 'patience': 2})

    def test_output_order_and_bounds_for_extreme_logits(self):
        x = gru.input_tensor([make_record(records()[0])[0]])
        net = gru.ExtremaGRU()
        for bias in ((-1000., -1000.), (-1000., 1000.), (1000., -1000.), (1000., 1000.), (0., 0.)):
            with torch.no_grad():
                net.head.weight.zero_()
                net.head.bias.copy_(torch.tensor(bias))
                y = net(x)
            gru.check_outputs(y)
        self.assertEqual(y.tolist(), [[.5, .75]])

    def test_training_mean_initialization_and_degenerate_targets(self):
        x = gru.input_tensor([make_record(records()[0])[0]])
        for pair in ((0., 0.), (1., 1.), (0., 1.), (.1, .4)):
            net = gru.ExtremaGRU()
            initial = net.initialize(pair)
            y = net(x)
            gru.check_outputs(y)
            self.assertAlmostEqual(float(y[0, 0].detach()), initial['bounded_initial_minimum'], places=6)
            self.assertAlmostEqual(float(y[0, 1].detach()), initial['bounded_initial_maximum'], places=6)
            self.assertEqual(torch.count_nonzero(net.head.weight).item(), 0)

    def test_both_head_outputs_receive_gradient(self):
        net = gru.ExtremaGRU()
        net.initialize((.2, .4))
        y = net(gru.input_tensor([make_record(records()[0])[0]]))
        nn_loss = torch.nn.functional.smooth_l1_loss(y, torch.tensor([[.1, .8]]), beta=.01)
        nn_loss.backward()
        self.assertTrue(torch.isfinite(net.head.bias.grad).all().item())
        self.assertTrue((net.head.bias.grad != 0).all().item())

    def test_missingness_channels_keep_zero_separate_from_missing(self):
        raw = records()[0]
        set_day(raw, 29, 0.0)
        set_day(raw, 30, None)
        x = gru.input_tensor([make_record(raw)[0]])
        self.assertEqual(x[0, 29, :2].tolist(), [0., 1.])
        self.assertEqual(x[0, 30, :2].tolist(), [0., 0.])
        with self.assertRaisesRegex(ValueError, 'all-missing'):
            gru.ExtremaGRU()(torch.zeros(1, 31, 5))

    def test_heldout_values_do_not_change_one_epoch_updates(self):
        raw = records()
        a = self.fit(raw, epochs=1)[0]
        for row in raw[-2:]:
            for day in range(91):
                set_day(row, day, .9)
        b = self.fit(raw, epochs=1)[0]
        self.assertEqual(a, b)

    def test_test_data_does_not_change_validation_selection(self):
        raw = records()
        a = self.fit(raw)
        for day in range(91):
            set_day(raw[-1], day, .8)
        self.assertEqual(a, self.fit(raw))

    def test_training_labels_change_weights(self):
        raw = records()
        before = self.fit(raw, epochs=1)[0]
        for row in raw[:-2]:
            for day in range(31, 61):
                set_day(row, day, .9)
        after = self.fit(raw, epochs=1)[0]
        self.assertNotEqual(before['state_dict'], after['state_dict'])

    def test_incomplete_future_excluded_only_from_this_task(self):
        raw = records()
        set_day(raw[0], 45, None)
        data = artifacts(raw)
        model, _, _ = self.fit(raw, epochs=1)
        self.assertNotIn(raw[0]['cve'], model['fit_cve_ids'])
        self.assertIn(raw[0]['cve'], data['task_cohorts.json']['endpoint']['train'])
        self.assertEqual(model['fit_cve_ids'], data['task_cohorts.json']['minmax']['train'])

    def test_future_and_unrelated_text_do_not_change_prediction(self):
        model = self.fit()[0]
        raw = records()[0]
        before = gru.predict(model, [make_record(raw)[0]])
        for day in range(31, 91):
            set_day(raw, day, .99)
        raw.update(in_vckev=True, llm_summary='Future summary')
        self.assertEqual(before, gru.predict(model, [make_record(raw)[0]]))

    def test_determinism_rng_and_selected_epoch(self):
        state = torch.get_rng_state().clone()
        deterministic = torch.are_deterministic_algorithms_enabled()
        a = self.fit()
        self.assertEqual(a, self.fit())
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(deterministic, torch.are_deterministic_algorithms_enabled())
        model, history, initial = a
        self.assertEqual(model['selected_epoch'], min(history, key=lambda h: h['validation_joint_mae'])['epoch'])
        self.assertFalse(initial['eligible_for_selection'])
        raw = [make_record(records()[0])[0]]
        self.assertEqual(gru.predict(model, raw), gru.predict(json.loads(json.dumps(model)), raw))

    def test_invalid_model_and_config_are_rejected(self):
        model = self.fit()[0]
        for key, value in [('head_rule', 'sort'), ('selected_epoch', 0), ('task', 'endpoint')]:
            changed = copy.deepcopy(model)
            changed[key] = value
            with self.assertRaises(ValueError):
                gru.restore_model(changed)
        model['state_dict']['head.bias'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            gru.restore_model(model)
        with self.assertRaisesRegex(ValueError, 'Changed fixed settings'):
            gru.fit_seed(artifacts(records()), 17, {**gru.CONFIG, 'learning_rate': .5})

    def test_comparison_requires_all_seeds_and_same_target(self):
        data = artifacts(records())
        linear = task.fit_model(data)
        refs, base = task.evaluate(data, linear)
        models = [self.fit(seed=seed)[0] for seed in gru.CONFIG['seeds']]
        rows, summary = gru.evaluate(data, models, refs, base)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(summary['per_seed']), 3)
        self.assertFalse(summary['test_performance_evaluated'])
        with self.assertRaisesRegex(ValueError, 'All fixed seeds'):
            gru.evaluate(data, models[:1], refs, base)
        with self.assertRaisesRegex(ValueError, 'Test scoring'):
            gru.evaluate(data, models, refs, base, split='test')
        refs[0]['actual_maximum'] = .99
        with self.assertRaisesRegex(ValueError, 'alignment'):
            gru.evaluate(data, models, refs, base)

    def test_cli_preflight_outputs_and_preserved_inputs(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            args = evidence_bundle(root)
            linear, output = root/'linear', root/'gru'
            task.run(*args, linear)
            before = {str(p): sha256(p) for p in root.rglob('*') if p.is_file()}
            with patch.object(gru, 'fit_seed', side_effect=AssertionError('Must not fit')):
                gru.run(*args, linear, output, check_only=True)
            self.assertFalse(output.exists())
            # Only synthetic integration uses a shorter budget; the CLI exposes no tuning flags.
            with patch.dict(gru.CONFIG, {'epochs': 2, 'patience': 2}):
                gru.run(*args, linear, output)
            lock = read_json(output/'evaluation_lock.json')
            self.assertEqual(set(lock['files']), gru.OUTPUT_FILES-{'evaluation_lock.json'})
            for name, digest in lock['files'].items():
                self.assertEqual(sha256(output/name), digest)
            for p, digest in before.items():
                self.assertEqual(sha256(p), digest)
            self.assertEqual([m['seed'] for m in read_json(output/'models.json')['models']], [17,29,43])
            with self.assertRaisesRegex(ValueError, 'Output exists'):
                gru.run(*args, linear, output)
            with self.assertRaisesRegex(ValueError, 'overlaps protected'):
                gru.run(*args, linear, linear/'nested')

    def test_failed_run_has_no_success_lock(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            args = evidence_bundle(root)
            linear, output = root/'linear', root/'gru'
            task.run(*args, linear)
            with patch.object(gru, 'fit_seed', side_effect=ValueError('synthetic failure')):
                with self.assertRaisesRegex(ValueError, 'synthetic failure'):
                    gru.run(*args, linear, output)
            self.assertEqual(read_json(output/'run_status.json')['status'], 'failed')
            self.assertFalse((output/'evaluation_lock.json').exists())

    def test_changed_linear_reference_rejected(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            args = evidence_bundle(root)
            linear = root/'linear'
            task.run(*args, linear)
            write_json(linear/'predictions.json', [])
            with self.assertRaisesRegex(ValueError, 'Changed artifact'):
                gru.preflight(*args, linear)


if __name__ == '__main__':
    unittest.main()
