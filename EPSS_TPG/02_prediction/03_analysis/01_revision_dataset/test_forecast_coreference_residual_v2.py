"""R04 v2 regression tests: synthetic fixtures only, no held-out test scoring."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_coreference_residual_v2 as run
from test_forecast_coreference_smoke import setup

torch = run.torch


class ScaledResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self, mode='tpg_epss'):
        manifest, graphs, data = setup()
        gain = run.training_gain(manifest['ids']['train'], data['inputs.json'])
        config = dict(run.smoke.config(), correction_gain=gain['gain'])
        initial = run.initialize(mode, manifest, graphs['all_coref'], config)
        net = run.ResidualTPGRegressor(**initial['spec'])
        net.load_state_dict(initial['state_dict'])
        return manifest, graphs, data, gain, config, initial, net

    def test_fixed_protocol_and_full_plan(self):
        p = run.settings()
        self.assertEqual(p['config'], run.v1.settings()['config'])
        self.assertEqual(run.read(run.FULL_PROTOCOL), run.planned_full_protocol())
        full = run.planned_full_protocol()
        self.assertEqual(full['config']['seeds'], [17, 29, 43])
        self.assertEqual(full['config']['epochs'], 100)
        self.assertEqual(full['config']['patience'], 15)
        self.assertFalse(full['test_performance_evaluated'])
        self.assertIn('not_runnable', full['status'])
        bad = deepcopy(p); bad['gain_rule'] = 'validation_optimum'
        with patch.object(run, 'read', return_value=bad), self.assertRaisesRegex(ValueError, 'gain rule'):
            run.settings()

    def test_gain_uses_only_training_inputs_and_is_order_invariant(self):
        m, graphs, data = setup()
        ids = m['ids']['train']
        inputs = {c: data['inputs.json'][c] for c in ids}
        a = run.training_gain(ids, inputs)
        b = run.training_gain(list(reversed(ids)), inputs)
        self.assertEqual(a['gain'], b['gain'])
        for c in m['ids']['validation']:
            data['inputs.json'][c] = {'invalid': 'must not be read'}
        data['labels.json'] = {'invalid': 'must not be read'}
        self.assertEqual(a, run.training_gain(ids, data['inputs.json']))
        self.assertFalse(a['uses_target_statistics'])
        self.assertFalse(a['uses_validation_inputs'])

    def test_gain_median_floor_ceiling_and_true_zero(self):
        m, graphs, data = setup()
        ids = m['ids']['train']
        for values, expected in (([0., 0.], .001), ([.01, .03], .02), ([.9, .8], .05)):
            with patch.object(run.task, 'predict_persistence', side_effect=[{'prediction': v} for v in values]):
                result = run.training_gain(ids, data['inputs.json'])
            self.assertAlmostEqual(result['gain'], expected)
            self.assertAlmostEqual(result['median_latest_observed_epss'], sum(values)/2)
        for ids_bad in ([], [ids[0], ids[0]]):
            with self.assertRaisesRegex(ValueError, 'Empty or duplicate'):
                run.training_gain(ids_bad, data['inputs.json'])

    def test_future_input_rejected_in_gain_and_model_batch(self):
        m, graphs, data, gain, config, initial, net = self.fixture()
        c = m['ids']['train'][0]
        data['inputs.json'][c]['X']['future_epss'] = .9
        with self.assertRaisesRegex(ValueError, 'Unexpected predictor'):
            run.training_gain(m['ids']['train'], data['inputs.json'])
        with self.assertRaisesRegex(ValueError, 'Unexpected predictor'):
            run.task.make_batch([c], graphs['all_coref'], data['inputs.json'])

    def test_exact_initial_persistence_and_unchanged_encoder_weights(self):
        m, graphs, data, gain, config, initial, net = self.fixture()
        old = run.v1.initialize('tpg_epss', m, graphs['all_coref'], run.smoke.config())
        self.assertEqual(initial['initial_state_sha256'], old['initial_state_sha256'])
        initial_states = {mode: run.initialize(mode, m, graphs['all_coref'], config)
                          for mode in ('tpg_epss', 'epss_only')}
        checks = run.initial_checks(initial_states, m, graphs, data, False)
        self.assertTrue(all(v['exact_float32_persistence'] for row in checks.values() for v in row.values()))
        batch = run.task.make_batch(m['ids']['train'], graphs['all_coref'], data['inputs.json'])
        # Fixture's Day 30 is absent: use the observed Day 29, including true zero.
        self.assertTrue(torch.equal(run.last_observed(batch.epss_sequence), batch.epss_sequence[:,29,0]))
        batch.epss_sequence[:,29,0] = 0
        self.assertTrue(torch.equal(net(batch), torch.zeros(2)))
        batch.epss_sequence[:,:,1] = 0
        with self.assertRaisesRegex(ValueError, 'No observed'):
            net(batch)

    def test_inside_tanh_scaling_not_small_hard_cap(self):
        m, graphs, data, gain, config, initial, net = self.fixture('epss_only')
        batch = run.task.make_batch(m['ids']['train'], graphs['all_coref'], data['inputs.json'])
        with torch.no_grad():
            net.head[-1].bias.fill_(.1)
        correction = net.details(batch)[1]
        self.assertTrue(torch.allclose(correction, torch.full((2,), .1*gain['gain']).tanh()))
        with torch.no_grad():
            net.head[-1].bias.fill_(2/gain['gain'])
        self.assertTrue(bool((net.details(batch)[1] > gain['gain']).all()))

    def test_gain_validation_and_checkpoint_version(self):
        m, graphs, data, gain, config, initial, net = self.fixture()
        for bad_gain in (None, True, 0., .5, float('nan')):
            spec = deepcopy(initial['spec']); spec['config']['correction_gain'] = bad_gain
            with self.assertRaisesRegex(ValueError, 'correction gain'):
                run.ResidualTPGRegressor(**spec)
        old = {'schema': run.v1.VERSION}
        with self.assertRaisesRegex(ValueError, 'contract'):
            run.restore(old)

    def test_gradient_routing_and_bounded_outputs(self):
        m, graphs, data, gain, config, initial, net = self.fixture()
        batch = run.task.make_batch(m['ids']['train'], graphs['all_coref'], data['inputs.json'])
        opt = torch.optim.SGD(net.parameters(), lr=.01)
        torch.nn.functional.smooth_l1_loss(net(batch), torch.full((2,), .8), beta=.01).backward()
        g = run.gradient_snapshot(net)
        self.assertGreater(g['head']['absolute_sum'], 0)
        for k in ('temporal', 'node_project', 'layers'):
            self.assertEqual(g[k]['absolute_sum'], 0)
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.nn.functional.smooth_l1_loss(net(batch), torch.full((2,), .8), beta=.01).backward()
        for k in ('temporal', 'node_project', 'layers'):
            self.assertGreater(run.gradient_snapshot(net)[k]['absolute_sum'], 0)
        for bias, expected in ((-100/gain['gain'], 0.), (100/gain['gain'], 1.)):
            with torch.no_grad():
                net.head[-1].weight.zero_(); net.head[-1].bias.fill_(bias)
            net.zero_grad(set_to_none=True)
            self.assertTrue(torch.equal(net(batch), torch.full((2,), expected)))
            net(batch).sum().backward()
            self.assertEqual(float(net.head[-1].bias.grad), 0.)

    def test_shared_initialization_and_epss_only_graph_control(self):
        m, graphs, data, gain, config, initial, net = self.fixture('epss_only')
        states = [run.initialize('tpg_epss', m, graphs[v], config) for v in run.cache.shared.MODES]
        self.assertEqual(len({s['initial_state_sha256'] for s in states}), 1)
        with torch.no_grad():
            net.head[-1].weight.fill_(.1)
        a = run.infer_details(net, m['ids']['validation'], graphs['all_coref'], data['inputs.json'], False)
        b = run.infer_details(net, m['ids']['validation'], graphs['no_coref'], data['inputs.json'], False)
        self.assertEqual(a, b)

    def test_deterministic_fit_reload_and_selection(self):
        m, graphs, data, gain, config, initial, net = self.fixture()
        orders = run.smoke.epoch_orders(m['ids']['train'], config)
        a, h, g = run.fit('all_coref', initial, orders, m, graphs['all_coref'], data, config, False)
        b, h2, g2 = run.fit('all_coref', initial, orders, m, graphs['all_coref'], data, config, False)
        self.assertEqual(h, h2); self.assertEqual(g, g2)
        self.assertEqual(run.prior.digest_state(a['state_dict']), run.prior.digest_state(b['state_dict']))
        self.assertEqual(a['selected_epoch'], min(h, key=lambda r: r['validation_mae'])['epoch'])
        self.assertEqual(a['spec']['config']['correction_gain'], gain['gain'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.pt'; torch.save(a, path)
            restored = run.restore(torch.load(path, weights_only=True))
            self.assertEqual(run.infer_details(restored, m['ids']['validation'], graphs['all_coref'], data['inputs.json'], False),
                             run.infer_details(run.restore(a), m['ids']['validation'], graphs['all_coref'], data['inputs.json'], False))
        bad = deepcopy(a); bad['selected_epoch'] = 0
        with self.assertRaisesRegex(ValueError, 'Epoch zero'):
            run.restore(bad)
        with patch.object(run.task, 'infer', side_effect=lambda net, ids, *a, **kw: [.5]*len(ids)):
            tied, _, _ = run.fit('all_coref', initial, orders, m, graphs['all_coref'], data, config, False)
        self.assertEqual(tied['selected_epoch'], 1)

    def test_save_verify_tamper_preflight_and_no_overwrite(self):
        m, graphs, data, gain, config, initial, net = self.fixture()
        identity = {'schema': run.VERSION, 'config': config, 'ids': m['ids'], 'gain_fit': gain,
                    'cache_lock_sha256': 'synthetic', 'pinned_files': {}, 'limitations': ['Synthetic only.']}
        with tempfile.TemporaryDirectory() as folder, patch.object(run, 'context', return_value=(m, graphs, data, identity)), \
             patch.object(run.task, 'RESULT_ROOT', Path(folder)):
            output = Path(folder)/'smoke'
            with patch.object(run, 'fit', side_effect=AssertionError('No fitting in preflight')):
                run.execute(output, check_only=True)
            self.assertFalse(output.exists())
            run.execute(output)
            with patch.object(run, 'fit', side_effect=AssertionError('No fitting in verification')):
                run.execute(output, verify_only=True)
            with self.assertRaisesRegex(ValueError, 'Output exists'):
                run.execute(output)
            (output/'summary.txt').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'Changed artifact'):
                run.execute(output, verify_only=True)


if __name__ == '__main__':
    unittest.main()
