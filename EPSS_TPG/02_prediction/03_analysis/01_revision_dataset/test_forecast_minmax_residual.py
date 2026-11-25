"""Synthetic F04-Residual checks; actual experiment training is researcher-run."""
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
from epss import forecast_minmax_sequence as direct
from epss import forecast_minmax_residual as residual
from epss.forecast_data import make_record, read_json, sha256
from review_forecast_minmax import measured
from review_forecast_minmax_sequence import close, history_check
from review_forecast_evidence import check_csv


class ResidualExtremaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fit(self, raw=None, seed=17, epochs=2):
        return residual.fit_seed(artifacts(records() if raw is None else raw), seed,
            {**residual.CONFIG, 'epochs': epochs, 'patience': 2})

    def test_zero_initialization_is_exact_persistence_including_zero_one_and_missing_day30(self):
        net = residual.ResidualExtremaGRU()
        for score in (0., .12345678, 1.):
            raw = records()[0]
            set_day(raw, 29, score)
            set_day(raw, 30, None)
            x = residual.input_tensor([make_record(raw)[0]])
            base, correction, proposed, result = net.details(x)
            self.assertTrue(torch.equal(base[:, 0], x[:, 29, 0]))
            self.assertTrue(torch.equal(result, base.expand(-1, 2)))
            self.assertEqual(torch.count_nonzero(correction), 0)
            self.assertTrue(torch.equal(result, proposed))
        with self.assertRaisesRegex(ValueError, 'All-missing'):
            net(torch.zeros_like(x))

    def test_order_bounds_clipping_and_both_extrema_can_move_same_direction(self):
        x = residual.input_tensor([make_record(records()[0])[0]])
        x[:, :, 0] = .5
        net = residual.ResidualExtremaGRU()
        for biases in ((1000., -1000.), (-1000., 1000.), (1000., 1000.), (-1000., -1000.), (0., 0.), (.1, .2), (-.1, -.2)):
            with torch.no_grad():
                net.head.bias.copy_(torch.tensor(biases))
                details = net.details(x)
            residual.check_outputs(details[-1])
            self.assertTrue(torch.equal(details[-1], torch.sort(details[2], dim=1, stable=True).values.clamp(0, 1)))
            if biases == (.1, .2):
                self.assertTrue((details[-1] > .5).all())
            if biases == (-.1, -.2):
                self.assertTrue((details[-1] < .5).all())

    def test_stable_tie_routes_distinct_gradients_and_updates_both_heads(self):
        x = residual.input_tensor([make_record(records()[0])[0]])
        x[:, :, 0] = .4
        net = residual.ResidualExtremaGRU()
        loss = torch.nn.functional.smooth_l1_loss(net(x), torch.tensor([[.1, .8]]), beta=.01)
        loss.backward()
        self.assertGreater(float(net.head.bias.grad[0]), 0)
        self.assertLess(float(net.head.bias.grad[1]), 0)
        self.assertTrue(torch.equal(net.gru.weight_ih_l0.grad, torch.zeros_like(net.gru.weight_ih_l0.grad)))
        torch.optim.Adam(net.parameters(), lr=.001).step()
        self.assertLess(float(net(x)[0, 0].detach()), .4)
        self.assertGreater(float(net(x)[0, 1].detach()), .4)

    def test_initial_input_gradient_preserves_latest_score_connection(self):
        raw = records()[0]
        set_day(raw, 29, .4)
        set_day(raw, 30, None)
        x = residual.input_tensor([make_record(raw)[0]]).requires_grad_(True)
        net = residual.ResidualExtremaGRU()
        gradient = torch.autograd.grad(net(x).sum(), x)[0][:, :, 0]
        expected = torch.zeros_like(gradient)
        expected[0, 29] = 2.
        self.assertTrue(torch.equal(gradient, expected))

    def test_encoder_initialization_matches_direct_seed(self):
        for seed in residual.CONFIG['seeds']:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                a = direct.ExtremaGRU()
                torch.manual_seed(seed)
                b = residual.ResidualExtremaGRU()
            for key,value in a.gru.state_dict().items():
                self.assertTrue(torch.equal(value, b.gru.state_dict()[key]))

    def test_heldout_labels_do_not_change_one_epoch_updates(self):
        raw = records()
        first = self.fit(raw, epochs=1)[0]
        for row in raw[-2:]:
            for day in range(91):
                set_day(row, day, .9)
        self.assertEqual(first, self.fit(raw, epochs=1)[0])

    def test_reserved_test_cannot_change_checkpoint_selection(self):
        raw = records()
        first = self.fit(raw)
        for day in range(91):
            set_day(raw[-1], day, .8)
        self.assertEqual(first, self.fit(raw))

    def test_training_targets_change_weights_but_not_initialization(self):
        raw = records()
        first = self.fit(raw, epochs=1)
        for row in raw[:-2]:
            for day in range(31, 61):
                set_day(row, day, .9)
        second = self.fit(raw, epochs=1)
        self.assertNotEqual(first[0]['state_dict'], second[0]['state_dict'])
        self.assertEqual(first[0]['initialization'], second[0]['initialization'])
        self.assertEqual(first[2], second[2])

    def test_future_scores_and_text_cannot_change_prediction(self):
        model = self.fit()[0]
        raw = records()[0]
        first = residual.predict(model, [make_record(raw)[0]])
        for day in range(31, 91):
            set_day(raw, day, .99)
        raw.update(in_vckev=True, llm_summary='Future text')
        self.assertEqual(first, residual.predict(model, [make_record(raw)[0]]))

    def test_determinism_reload_rng_and_best_epoch(self):
        state = torch.get_rng_state().clone()
        deterministic = torch.are_deterministic_algorithms_enabled()
        first = self.fit()
        self.assertEqual(first, self.fit())
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(deterministic, torch.are_deterministic_algorithms_enabled())
        model, history, initial = first
        self.assertEqual(model['selected_epoch'], min(history, key=lambda h: h['validation_joint_mae'])['epoch'])
        self.assertFalse(initial['eligible_for_selection'])
        row = [make_record(records()[0])[0]]
        self.assertEqual(residual.predict(model, row), residual.predict(json.loads(json.dumps(model)), row))

    def test_reject_invalid_contract_and_settings(self):
        model = self.fit()[0]
        for key,value in [('head_rule', 'other'), ('selected_epoch', 0), ('task', 'endpoint'), ('schema', direct.VERSION)]:
            with self.assertRaises(ValueError):
                residual.restore_model({**model, key: value})
        changed = copy.deepcopy(model)
        changed['state_dict']['head.bias'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            residual.restore_model(changed)
        with self.assertRaisesRegex(ValueError, 'Changed fixed settings'):
            residual.fit_seed(artifacts(records()), 17, {**residual.CONFIG, 'learning_rate': .5})

    def test_complete_future_cohort_is_unchanged(self):
        raw = records()
        set_day(raw[0], 45, None)
        data = artifacts(raw)
        model = self.fit(raw, epochs=1)[0]
        self.assertNotIn(raw[0]['cve'], model['fit_cve_ids'])
        self.assertEqual(model['fit_cve_ids'], task.cohort(data, 'train'))
        self.assertIn(raw[0]['cve'], data['task_cohorts.json']['endpoint']['train'])

    def test_synthetic_run_preflight_integrity_metrics_and_guards(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), \
             patch.dict(direct.CONFIG, {'epochs': 2, 'patience': 2}), \
             patch.dict(residual.CONFIG, {'epochs': 2, 'patience': 2}):
            root = Path(temp)
            args = evidence_bundle(root)
            linear, reference, output = root/'linear', root/'direct', root/'residual'
            task.run(*args, linear)
            direct.run(*args, linear, reference)
            before = {str(p): sha256(p) for p in root.rglob('*') if p.is_file()}
            with patch.object(residual, 'fit_seed', side_effect=AssertionError('Must not fit')):
                residual.run(reference, output, check_only=True)
            self.assertFalse(output.exists())
            summary = residual.run(reference, output)
            self.assertEqual({p.name for p in output.iterdir()}, residual.OUTPUT_FILES)
            lock = read_json(output/'evaluation_lock.json')
            for name,digest in lock['files'].items():
                self.assertEqual(sha256(output/name), digest)
            for path,digest in before.items():
                self.assertEqual(sha256(path), digest)
            data, refs, baseline, provenance = residual.preflight(reference)
            saved = read_json(output/'models.json')
            self.assertEqual(saved['provenance'], provenance)
            rows, repeated = residual.evaluate(data, saved['models'], refs, baseline)
            self.assertEqual(rows, read_json(output/'predictions.json'))
            check_csv(output/'predictions.csv', rows)
            self.assertEqual({**repeated, 'provenance': provenance}, summary)
            history = read_json(output/'training_history.json')
            for model, per_seed in zip(saved['models'], summary['per_seed']):
                history_check(history[str(model['seed'])], model)
                measured_metrics = measured(rows, f"residual_seed_{model['seed']}")
                close(measured_metrics['joint_mae'], per_seed['metrics']['joint_mae'], 'Independent MAE differs')
                for part in ('minimum', 'maximum', 'range_width'):
                    for metric in ('mae', 'rmse'):
                        close(measured_metrics[part][metric], per_seed['metrics'][part][metric], 'Independent metric differs')
            with self.assertRaisesRegex(ValueError, 'All fixed seeds'):
                residual.evaluate(data, saved['models'][:1], refs, baseline)
            with self.assertRaisesRegex(ValueError, 'Test scoring'):
                residual.evaluate(data, saved['models'], refs, baseline, split='test')
            changed = copy.deepcopy(refs)
            changed[0]['actual_minimum'] = .99
            with self.assertRaisesRegex(ValueError, 'Comparator alignment'):
                residual.evaluate(data, saved['models'], changed, baseline)
            with self.assertRaisesRegex(ValueError, 'Output exists'):
                residual.run(reference, output)
            with self.assertRaisesRegex(ValueError, 'overlaps protected'):
                residual.run(reference, reference/'bad')
            failed = root/'failed'
            with patch.object(residual, 'fit_seed', side_effect=RuntimeError('Synthetic failure')):
                with self.assertRaisesRegex(RuntimeError, 'Synthetic failure'):
                    residual.run(reference, failed)
            self.assertEqual(read_json(failed/'run_status.json')['status'], 'failed')
            self.assertFalse((failed/'evaluation_lock.json').exists())
            path = reference/'predictions.json'
            path.write_text('[]')
            with self.assertRaisesRegex(ValueError, 'Changed artifact'):
                residual.preflight(reference)


if __name__ == '__main__':
    unittest.main()
