"""GV01 synthetic contracts; no real-data full training or test scoring."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_gain_validation as run
import test_forecast_day60_pipeline as fixture


def setup():
    identity = fixture.identity()
    identity.update(schema=run.SCHEMA, original_gain=identity['gain'], gain=1., comparison_protocol=run.read(run.CONFIG))
    x = {s: run.old.tensor_inputs([fixture.inputs(c) for c in ids]) for s, ids in identity['ids'].items()}
    labels = {c: {'cve': c, 'target_day': 60, 'horizon_days': 30, 'target_observed': True,
                  'y': {'endpoint': .25 if c in ('a', 'c') else .08}}
              for ids in identity['ids'].values() for c in ids}
    store = fixture.legacy.MemoryStore()
    controls = {}
    for seed in identity['config']['seeds']:
        for variant in run.VARIANTS:
            net = run.initialize(identity, variant, seed)
            rows = run.prior.infer(net, identity['ids']['validation'], x['validation'], store, 2, False)
            run.old.scored(rows, labels)
            controls[f'{variant}__{seed}'] = {'rows': rows, 'initial_state_sha256': run.controls.state_digest(net.state_dict())}
    return identity, x, labels, store, controls


class GainValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        for redirect in (redirect_stdout, redirect_stderr):
            context = redirect(StringIO())
            context.__enter__()
            self.addCleanup(context.__exit__, None, None, None)
        self.args = setup()
        self.i, self.x, self.y, self.store, self.controls = self.args

    def test_same_weights_and_persistence_initialization(self):
        for variant in run.VARIANTS:
            new = run.initialize(self.i, variant, 17)
            old = run.prior.initialize(dict(self.i, gain=self.i['original_gain']), variant, 17)
            self.assertEqual(run.controls.state_digest(new.state_dict()), run.controls.state_digest(old.state_dict()))
            parts = new.details(self.x['train'], self.store, self.i['ids']['train'])
            self.assertTrue(torch.equal(parts[0], parts[3]))
            self.assertEqual(new.gain, 1.)
        run.preflight(self.i, self.x, self.store, self.controls)

    def test_diagnostics_distinguish_clipping_from_boundary_predictions(self):
        rows = [{'raw_prediction': raw, 'prediction': min(1., max(0., raw)), 'correction': raw,
                 'actual_day60_epss': .1} for raw in (-.5, 0., .5, 1., 1.5)]
        d = run.diagnostics(rows)
        self.assertEqual(d['clipped_below_zero'], 1)
        self.assertEqual(d['clipped_above_one'], 1)
        self.assertEqual(d['predicted_exactly_zero'], 2)
        self.assertEqual(d['predicted_exactly_one'], 2)
        self.assertEqual(d['clipped_fraction'], .4)
        self.assertGreater(d['raw_mae'], d['clipped_mae'])

    def test_instrumentation_preserves_optimizer_and_dropout(self):
        identity = deepcopy(self.i)
        identity['config'].update(epochs=1)
        for variant in run.VARIANTS:
            net = run.initialize(identity, variant, 17)
            cfg = identity['config']
            opt = torch.optim.Adam(net.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
            ids = identity['ids']['train']
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(17)
                net.train()
                order = run.controls.orders(ids, cfg, 17)[0]
                for start in range(0, len(order), cfg['batch_size']):
                    chosen = order[start:start+cfg['batch_size']]
                    ix = [ids.index(c) for c in chosen]
                    target = torch.tensor([self.y[c]['y']['endpoint'] for c in chosen], dtype=torch.float32)
                    opt.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.l1_loss(net(self.x['train'][ix], self.store, chosen), target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['gradient_clip_norm'])
                    opt.step()
            saved, history = run.fit(variant, 17, identity, self.x, self.y, self.store, False)
            self.assertEqual(saved['selected_epoch'], 1)
            for key, value in net.state_dict().items():
                torch.testing.assert_close(value, saved['state_dict'][key], rtol=0, atol=0)
            self.assertEqual(history[0]['training_examples'], len(ids))

    def test_labels_are_not_inputs_and_test_graphs_not_accessed(self):
        record = fixture.inputs('a')
        record['labels'] = {'target': .7}
        with self.assertRaises(ValueError):
            run.old.tensor_inputs([record])
        bad = deepcopy(self.y)
        bad['c']['target_day'] = 90
        with self.assertRaises(ValueError):
            run.old.scored([{'cve': 'c', 'prediction': .2}], bad)
        self.y['test'] = {'do_not_read': True}
        run.fit('all_sources_epss', 17, self.i, self.x, self.y, self.store, False)
        with self.assertRaises(ValueError):
            run.old.GraphStore({'index': {}}, {}, ['test'], 0)

    def test_matched_comparison_rejects_target_and_id_changes(self):
        control = self.controls['epss_only__17']
        result = run.compare(control['rows'], control)
        self.assertEqual(result['mae_minus_original'], 0.)
        for key, value in [('cve', 'wrong'), ('actual_day60_epss', .9), ('persistence_float32', .9)]:
            rows = deepcopy(control['rows'])
            rows[0][key] = value
            with self.assertRaises(ValueError):
                run.compare(rows, control)

    def test_lifecycle_resume_corruption_and_unchanged_old_results(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run, 'OUTPUT', Path(tmp)/'out'), \
                patch.object(run, 'context', return_value=self.args):
            run.execute('smoke', 1, 0, check_only=True)
            self.assertFalse(run.OUTPUT.exists())
            run.execute('smoke', 1, 0)
            run.execute('smoke', 1, 0, verify_only=True)
            with self.assertRaisesRegex(ValueError, 'Run exists'):
                run.execute('smoke', 1, 0)
            with self.assertRaisesRegex(ValueError, 'complete resume'):
                run.execute('smoke', 1, 0, resume=True)
            (run.OUTPUT/'smoke/evaluation_lock.json').unlink()
            with patch.object(run, 'fit', side_effect=AssertionError('Completed fits must not retrain')):
                run.execute('smoke', 1, 0, resume=True)
            path = run.OUTPUT/'smoke/fits/epss_only__17/metrics.json'
            run.write(path, {'corrupt': True})
            with self.assertRaisesRegex(ValueError, 'Changed/unsafe artifact'):
                run.execute('smoke', 1, 0, verify_only=True)

    def test_full_requires_smoke(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run, 'OUTPUT', Path(tmp)/'out'), \
                patch.object(run, 'context', return_value=self.args):
            with self.assertRaises(FileNotFoundError):
                run.execute('full', 1, 0)


if __name__ == '__main__':
    unittest.main()
