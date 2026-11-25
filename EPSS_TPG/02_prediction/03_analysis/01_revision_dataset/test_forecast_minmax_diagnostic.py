"""Synthetic DQ05 tests: fixed weights, isolated cohorts and sensitivity checks."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_forecast_minmax import evidence_bundle, task, records, artifacts, set_day
from epss import forecast_minmax_sequence as gru
from epss.forecast_data import read_json, sha256
import diagnose_forecast_minmax_sequence as diagnostic


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def inputs(self):
        raw = records()
        set_day(raw[0], 29, 0.)
        set_day(raw[0], 30, None)
        data = artifacts(raw)
        ids = gru.task.cohort(data, 'train')
        return data, ids, gru.input_tensor([data['inputs.json'][c] for c in ids])

    def test_trace_and_analytic_derivatives(self):
        _, _, x = self.inputs()
        net = gru.ExtremaGRU().double()
        h, logits, outputs, derivatives = diagnostic.trace(net, x.double())
        self.assertTrue(torch.equal(outputs, net(x.double())))
        jac_low = torch.autograd.grad(outputs[:, 0].sum(), logits, retain_graph=True)[0]
        jac_high = torch.autograd.grad(outputs[:, 1].sum(), logits)[0]
        torch.testing.assert_close(jac_low[:, 0], derivatives[:, 0])
        self.assertEqual(torch.count_nonzero(jac_low[:, 1]), 0)
        torch.testing.assert_close(jac_high[:, 0], derivatives[:, 1])
        torch.testing.assert_close(jac_high[:, 1], derivatives[:, 2])
        self.assertEqual(h.shape, (len(x), 16))

    def test_input_derivative_matches_finite_difference_and_individual_sample(self):
        _, _, x = self.inputs()
        x = x.double().requires_grad_(True)
        net = gru.ExtremaGRU().double()
        gradient = torch.autograd.grad(net(x)[:, 1].sum(), x)[0]
        single = x[:1].detach().clone().requires_grad_(True)
        independent = torch.autograd.grad(net(single)[0, 1], single)[0]
        torch.testing.assert_close(gradient[:1], independent)
        plus, minus = x.detach().clone(), x.detach().clone()
        plus[0, 10, 0] += 1e-5
        minus[0, 10, 0] -= 1e-5
        finite = (net(plus)[0, 1]-net(minus)[0, 1])/2e-5
        torch.testing.assert_close(gradient[0, 10, 0], finite, atol=1e-8, rtol=1e-4)

    def test_perturbations_preserve_missingness_and_observed_zero(self):
        _, _, x = self.inputs()
        before = x.clone()
        mask = x[:, :, 1] == 1
        for name in diagnostic.PROTOCOL['perturbations']:
            changed = diagnostic.perturb(x, name)
            self.assertTrue(torch.equal(changed[:, :, 1:], x[:, :, 1:]))
            self.assertTrue(torch.equal(changed[:, :, 0][~mask], x[:, :, 0][~mask]))
            self.assertTrue(((changed[:, :, 0] >= 0) & (changed[:, :, 0] <= 1)).all())
        self.assertAlmostEqual(float(diagnostic.perturb(x, 'observed_plus')[0, 29, 0]), .01)
        self.assertEqual(float(diagnostic.perturb(x, 'observed_plus')[0, 30, 0]), 0.)
        self.assertTrue(torch.equal(x, before))
        with self.assertRaises(ValueError):
            diagnostic.perturb(torch.zeros_like(x), 'observed_plus')

    def test_constant_head_has_zero_input_sensitivity_despite_hidden_variation(self):
        data, ids, _ = self.inputs()
        net = gru.ExtremaGRU()
        net.initialize((.1, .3))
        summary, rows, spread, responses, gradients = diagnostic.analyze_split(net, data, ids, 17, 'train')
        self.assertGreater(summary['hidden_component_sd']['max'], 0.)
        self.assertTrue(all(r['mean_abs_d_minimum_d_observed_score'] == 0 for r in rows))
        self.assertTrue(all(r['mean_abs_d_maximum_d_observed_score'] == 0 for r in rows))
        self.assertTrue(all(r['mean_abs_maximum_change'] == 0 for r in responses))
        self.assertEqual(set(gradients), dict(net.named_parameters()).keys())
        self.assertTrue(all(p.grad is None for p in net.parameters()))

    def test_scope_guards(self):
        data, ids, x = self.inputs()
        net = gru.ExtremaGRU()
        for split in ('validation', 'test'):
            with self.assertRaisesRegex(ValueError, 'training-only'):
                diagnostic.loss_gradients(net, x, [[.1, .3]]*len(x), split)
        with self.assertRaisesRegex(ValueError, 'Unexpected split'):
            diagnostic.analyze_split(net, data, ids, 17, 'test')
        with self.assertRaisesRegex(ValueError, 'Unexpected split'):
            diagnostic.analyze_split(net, data, ids[::-1], 17, 'train')
        with self.assertRaisesRegex(ValueError, 'All fixed seeds'):
            diagnostic.diagnose(data, [])

    def test_test_records_and_future_labels_cannot_change_input_sensitivity(self):
        data, ids, _ = self.inputs()
        models = [gru.fit_seed(data, s, {**gru.CONFIG, 'epochs': 1, 'patience': 1})[0]
                  for s in gru.CONFIG['seeds']]
        weights = copy.deepcopy(models)
        rng = torch.get_rng_state().clone()
        with patch.object(gru, 'fit_seed', side_effect=AssertionError('Must not fit')), \
             patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Must not update')):
            first = diagnostic.diagnose(data, models)
            changed = copy.deepcopy(data)
            for c in data['task_cohorts.json']['minmax']['test']:
                changed['inputs.json'][c] = None
                changed['labels.json'][c] = None
            self.assertEqual(first, diagnostic.diagnose(changed, models))
            for c in ids:
                changed['labels.json'][c]['y']['minmax'] = [.8, .9]
                changed['labels.json'][c]['y']['trajectory'] = [.8]*29+[.9]
                changed['labels.json'][c]['y']['endpoint'] = .9
            other = diagnostic.diagnose(changed, models)
            for key in ('per_cve', 'perturbations', 'synthetic_probes'):
                self.assertEqual(first[key], other[key])
            self.assertNotEqual(first['parameter_diagnostics'], other['parameter_diagnostics'])
        self.assertEqual(weights, models)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_synthetic_run_locks_preservation_and_failure(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), \
             patch.dict(gru.CONFIG, {'epochs': 2, 'patience': 2}):
            root = Path(temp)
            args = evidence_bundle(root)
            linear, run_dir, output = root/'linear', root/'gru', root/'diagnostic'
            task.run(*args, linear)
            gru.run(*args, linear, run_dir)
            before = {str(p): sha256(p) for p in root.rglob('*') if p.is_file()}
            with patch.object(diagnostic, 'diagnose', side_effect=AssertionError('Preflight must not diagnose')):
                diagnostic.run(run_dir, output, check_only=True)
            self.assertFalse(output.exists())
            with patch.object(gru, 'fit_seed', side_effect=AssertionError('Must not fit')):
                diagnostic.run(run_dir, output)
            self.assertEqual({p.name for p in output.iterdir()}, diagnostic.OUTPUT_FILES)
            lock = read_json(output/'analysis_lock.json')
            for name, digest in lock['files'].items():
                self.assertEqual(sha256(output/name), digest)
            self.assertEqual(read_json(output/'run_status.json')['status'], 'complete')
            self.assertEqual(len(read_json(output/'synthetic_probes.json')), 18)
            self.assertEqual(len(read_json(output/'per_cve.json')), 26*3)
            for p, digest in before.items():
                self.assertEqual(sha256(p), digest)
            with self.assertRaisesRegex(ValueError, 'Output exists'):
                diagnostic.run(run_dir, output)
            with self.assertRaisesRegex(ValueError, 'overlaps protected'):
                diagnostic.run(run_dir, run_dir/'bad')
            failed = root/'failed'
            with patch.object(diagnostic, 'diagnose', side_effect=RuntimeError('Synthetic failure')):
                with self.assertRaisesRegex(RuntimeError, 'Synthetic failure'):
                    diagnostic.run(run_dir, failed)
            self.assertEqual(read_json(failed/'run_status.json')['status'], 'failed')
            self.assertFalse((failed/'analysis_lock.json').exists())


if __name__ == '__main__':
    unittest.main()
