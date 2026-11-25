"""GB01 synthetic tests; no real full training or test scoring."""
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
from epss import forecast_group_balanced as run
import test_forecast_day60_pipeline as fixture


def setup(scale='smoke'):
    identity = fixture.identity()
    identity.update(schema=run.SCHEMA, scale=scale, original_gain=identity['gain'], gain=1.,
                    balanced_protocol=run.read(run.CONFIG))
    identity['ids'] = {'train': ['a', 'b', 'e', 'f', 'g', 'h'], 'validation': ['c', 'd', 'i']}
    if scale == 'full':
        identity['config']['seeds'] = [17, 29, 43]
    inputs = {c: fixture.inputs(c) for ids in identity['ids'].values() for c in ids}
    labels = {c: {'cve': c, 'target_day': 60, 'horizon_days': 30, 'target_observed': True,
                  'y': {'endpoint': .25 if c in ('a', 'c') else .05 if c in ('b', 'd') else .1}}
              for c in inputs}
    identity['movement_groups'] = {s: run.movement_groups(ids, inputs, labels) for s, ids in identity['ids'].items()}
    identity['training_weighting'] = run.training_weights(identity['movement_groups']['train'])
    x = {s: run.old.tensor_inputs([inputs[c] for c in ids]) for s, ids in identity['ids'].items()}
    store, controls = fixture.legacy.MemoryStore(), {}
    for seed in identity['config']['seeds']:
        for variant in run.VARIANTS:
            net = run.initialize(identity, variant, seed)
            rows = run.prior.infer(net, identity['ids']['validation'], x['validation'], store, 2, False)
            run.old.scored(rows, labels)
            controls[f'{variant}__{seed}'] = {'rows': rows, 'initial_state_sha256': run.controls.state_digest(net.state_dict())}
    return identity, x, labels, store, controls


class GroupBalancedTests(unittest.TestCase):
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

    def test_equal_group_mass_and_unit_mean(self):
        w = self.i['training_weighting']
        self.assertEqual(w['counts'], {'unchanged': 4, 'increased': 1, 'decreased': 1})
        self.assertEqual(w['weights_by_group'], {'unchanged': .5, 'increased': 2., 'decreased': 2.})
        self.assertEqual(set(w['total_weight_by_group'].values()), {2.})
        self.assertEqual(sum(w['weights_by_cve'].values()), 6.)

    def test_loss_and_gradient_not_renormalized_within_batch(self):
        pred = torch.tensor([.2, .4], requires_grad=True)
        target, weight = torch.tensor([.1, .1]), torch.tensor([.5, 2.])
        loss = run.weighted_mae(pred, target, weight)
        self.assertAlmostEqual(float(loss), .325, places=6)
        loss.backward()
        torch.testing.assert_close(pred.grad, torch.tensor([.25, 1.]))
        with self.assertRaises(ValueError):
            run.weighted_mae(pred, target, torch.zeros(2))

    def test_groups_use_original_scores_and_missing_anchor_fallback(self):
        inputs = {c: fixture.inputs(c) for c in self.i['ids']['train']}
        groups = run.movement_groups(self.i['ids']['train'], inputs, self.y)
        self.assertEqual(groups['e'], 'unchanged')
        self.assertIsNone(inputs['e']['histories']['days_0_30']['values'][-1])
        self.assertNotEqual(float(self.x['train'][2, -2, 0]), self.y['e']['y']['endpoint'])
        bad = deepcopy(self.y)
        bad['e']['target_day'] = 90
        with self.assertRaises(ValueError):
            run.movement_groups(['e'], inputs, bad)

    def test_training_weights_ignore_validation_test_and_inputs_are_unchanged(self):
        inputs = {c: fixture.inputs(c) for c in self.i['ids']['train']}
        original = deepcopy(inputs)
        labels = deepcopy(self.y)
        labels.update(c={'invalid': 'must not be read'}, test={'invalid': 'must not be read'})
        inputs.update(test={'invalid': True})
        groups = run.movement_groups(self.i['ids']['train'], inputs, labels)
        self.assertEqual(run.training_weights(groups), self.i['training_weighting'])
        self.assertEqual({c: inputs[c] for c in original}, original)
        self.assertNotIn('test', groups)

    def test_empty_groups_explicit_and_full_rejects_them(self):
        with self.assertRaises(ValueError):
            run.training_weights({'a': 'unchanged'})
        weights = run.training_weights({'a': 'unchanged'}, require_all=False)
        self.assertIsNone(weights['weights_by_group']['increased'])
        self.assertEqual(weights['weights_by_cve'], {'a': 1.})
        rows = self.controls['epss_only__17']['rows'][:1]
        metrics = run.grouped_metrics(rows, {rows[0]['cve']: 'unchanged'})
        self.assertEqual(metrics['increased'], {'n': 0, 'metrics': None, 'persistence_metrics': None})

    def test_unit_weights_reproduce_gv01_updates_and_selection(self):
        identity = deepcopy(self.i)
        identity['training_weighting']['weights_by_cve'] = dict.fromkeys(identity['ids']['train'], 1.)
        for variant in run.VARIANTS:
            new, nh = run.fit(variant, 17, identity, self.x, self.y, self.store, False)
            old, oh = run.gv.fit(variant, 17, identity, self.x, self.y, self.store, False)
            self.assertEqual(new['selected_epoch'], old['selected_epoch'])
            self.assertEqual([r['validation_mae'] for r in nh], [r['validation_mae'] for r in oh])
            for key, value in new['state_dict'].items():
                torch.testing.assert_close(value, old['state_dict'][key], rtol=0, atol=0)

    def test_weighted_updates_match_manual_optimizer(self):
        identity = deepcopy(self.i)
        identity['config'].update(epochs=1, batch_size=4, dropout=.1)
        net = run.initialize(identity, 'all_sources_epss', 17)
        cfg, ids = identity['config'], identity['ids']['train']
        opt = torch.optim.Adam(net.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            net.train()
            order = run.controls.orders(ids, cfg, 17)[0]
            for start in range(0, len(order), 4):
                chosen = order[start:start+4]
                ix = [ids.index(c) for c in chosen]
                targets = torch.tensor([self.y[c]['y']['endpoint'] for c in chosen])
                weights = torch.tensor([identity['training_weighting']['weights_by_cve'][c] for c in chosen])
                opt.zero_grad(set_to_none=True)
                loss = (weights*(net(self.x['train'][ix], self.store, chosen)-targets).abs()).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['gradient_clip_norm'])
                opt.step()
        saved, history = run.fit('all_sources_epss', 17, identity, self.x, self.y, self.store, False)
        for key, value in net.state_dict().items():
            torch.testing.assert_close(value, saved['state_dict'][key], rtol=0, atol=0)
        self.assertEqual(sum(r['n'] for r in history[0]['training_by_movement'].values()), len(ids))

    def test_selection_uses_ordinary_mae_and_comparison_is_matched(self):
        saved, h = run.fit('epss_only', 17, self.i, self.x, self.y, self.store, False)
        self.assertEqual(saved['selected_epoch'], min(h, key=lambda r: r['validation_mae'])['epoch'])
        rows, metrics = run.evaluate(saved, self.i, self.x, self.y, self.store, False)
        self.assertEqual(metrics['metrics']['n'], 3)
        self.assertEqual(sum(v['n'] for v in metrics['by_movement'].values()), 3)
        c = run.compare(rows, self.controls['epss_only__17'], self.i['movement_groups']['validation'])
        self.assertIn('mae_minus_gv01', c)
        bad = deepcopy(self.controls['epss_only__17'])
        bad['rows'][0]['actual_day60_epss'] = .99
        with self.assertRaises(ValueError):
            run.compare(rows, bad, self.i['movement_groups']['validation'])

    def test_smoke_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run, 'OUTPUT', Path(tmp)/'out'), \
                patch.object(run, 'context', return_value=self.args):
            run.execute('smoke', 1, 0, check_only=True)
            self.assertFalse(run.OUTPUT.exists())
            original = run.fit
            calls = []
            def interrupt(*args, **kwargs):
                calls.append(args[0])
                if len(calls) == 2:
                    raise RuntimeError('synthetic interruption')
                return original(*args, **kwargs)
            with patch.object(run, 'fit', side_effect=interrupt), self.assertRaises(RuntimeError):
                run.execute('smoke', 1, 0)
            path = run.OUTPUT/'smoke/fits/epss_only__17/checkpoint.pt'
            before = run.sha256(path)
            with patch.object(run, 'fit', wraps=original) as fitted:
                run.execute('smoke', 1, 0, resume=True)
                self.assertEqual(fitted.call_count, 1)
            self.assertEqual(run.sha256(path), before)
            run.execute('smoke', 1, 0, verify_only=True)
            with self.assertRaisesRegex(ValueError, 'Run exists'):
                run.execute('smoke', 1, 0)
            with self.assertRaisesRegex(ValueError, 'complete resume'):
                run.execute('smoke', 1, 0, resume=True)
            run.write(run.OUTPUT/'smoke/weighting.json', {'corrupt': True})
            with self.assertRaisesRegex(ValueError, 'Changed/unsafe artifact'):
                run.execute('smoke', 1, 0, verify_only=True)

    def test_six_fit_full_lifecycle_requires_smoke(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run, 'OUTPUT', Path(tmp)/'out'), \
                patch.object(run, 'context', side_effect=lambda scale, *args: setup(scale)):
            with self.assertRaises(FileNotFoundError):
                run.execute('full', 1, 0)
            run.execute('smoke', 1, 0)
            run.execute('full', 1, 0)
            run.execute('full', 1, 0, verify_only=True)
            self.assertEqual(len(run.read(run.OUTPUT/'full/summary.json')['fits']), 6)


if __name__ == '__main__':
    unittest.main()
