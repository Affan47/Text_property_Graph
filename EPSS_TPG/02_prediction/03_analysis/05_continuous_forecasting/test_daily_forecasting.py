"""CF01 boundary, masking, weighting, controls and deterministic integration tests."""
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss.continuous_forecasting import data, controls, evaluation as ev, runner

POLICY = {'validation_start': '2025-03-02', 'test_start': '2025-06-01',
          'test_end': '2025-12-02', 'observation_end': '2026-09-29'}


def record(pub='2024-01-01', cve='CVE-2024-10001'):
    start = date.fromisoformat(pub)
    return {'cve': cve, 'published_date': pub, 'published_timestamp': pub+'T12:00:00Z',
            'epss_scores': {(start+timedelta(days=i)).isoformat(): .123456789 for i in range(91)}}


def row(split='train', pub='2024-01-01', cve='CVE-2024-10001'):
    return data.sanitize(record(pub, cve), split, POLICY)[0]


class DailyTests(unittest.TestCase):
    def test_complete_matrix_and_unique_requests(self):
        r = row()
        self.assertEqual(int(data.eligible(r).sum()), 4095)
        cells = [(t, int(t+h)) for _, t, hs, _ in data.requests(r) for h in hs]
        self.assertEqual(len(set(cells)), 4095)
        self.assertIn((0, 90), cells)
        self.assertIn((89, 90), cells)
        self.assertTrue(all(t < u <= 90 for t, u in cells))

    def test_date_boundaries_ignore_poisoned_future(self):
        for split, pub, upper in [('train', '2025-01-01', POLICY['validation_start']),
                                  ('validation', '2025-04-01', POLICY['test_start'])]:
            rec = record(pub)
            for stamp in rec['epss_scores']:
                if stamp >= upper:
                    rec['epss_scores'][stamp] = 'DO NOT PARSE THIS FUTURE SCORE'
            r, _ = data.sanitize(rec, split, POLICY)
            for _, t, hs, _ in data.requests(r):
                for h in hs:
                    self.assertLess((date.fromisoformat(pub)+timedelta(days=int(t+h))).isoformat(), upper)
            self.assertTrue(any(v is None for v in r['values']))

    def test_observation_end_cannot_precede_reserved_period(self):
        policy = {**POLICY, 'observation_end': '2024-01-11'}
        with self.assertRaises(ValueError):
            data.sanitize(record(), 'train', policy)

    def test_validation_prefix_may_precede_window_but_anchor_may_not(self):
        r = row('validation', '2025-02-15')
        mask = data.eligible(r)
        self.assertFalse(mask[:15].any())
        self.assertTrue(mask[15].any())
        self.assertIsNotNone(r['values'][0])

    def test_test_role_is_refused(self):
        with self.assertRaises(ValueError):
            data.sanitize(record(), 'test', POLICY)
        r = row(); r['split'] = 'test'
        with self.assertRaises(ValueError):
            data.prefix(r, 30)

    def test_missing_and_true_zero_are_distinct(self):
        r = row()
        r['values'][:4] = [None]*4
        r['values'][4] = 0.
        r['values'][6] = None
        mask = data.eligible(r)
        self.assertFalse(mask[:4].any())
        self.assertFalse(mask[4, 1])
        self.assertTrue(mask[4, 0])
        past, latest, age = data.prefix(r, 6)
        self.assertEqual(past[4][:2], [0., 1])
        self.assertEqual(past[6][:2], [0., 0])
        self.assertEqual(age, 1)
        self.assertEqual(past[6][2], 1/90)
        self.assertEqual(latest, r['values'][5])

    def test_future_scores_do_not_reach_model(self):
        r = row()
        item = (r, 10, np.array([1, 2]), 1.)
        a = data.batch([item])[0]
        other = deepcopy(r)
        other['values'][11:] = [.99]*80
        b = data.batch([(other, 10, np.array([1, 2]), 1.)])[0]
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]))
        model = runner.model_for(17, data.settings())
        with torch.no_grad():
            model.head[-1].weight.fill_(.01)
        self.assertTrue(torch.equal(data.forward(model, a)[0], data.forward(model, b)[0]))
        with self.assertRaises(ValueError):
            data.forward(model, {**a, 'target': torch.tensor([.99])})

    def test_padding_does_not_impute_targets(self):
        r = row()
        inputs, targets, valid, weights = data.batch([(r, 0, np.array([1, 2]), 1.), (r, 89, np.array([1]), 1.)])
        self.assertTrue(torch.isnan(targets[~valid]).all())
        self.assertEqual(float(weights[~valid].sum()), 0.)
        self.assertEqual(int(inputs['lengths'][1]), 90)
        model = runner.model_for(17, data.settings())
        raw, _ = data.forward(model, inputs)
        self.assertTrue(torch.equal(raw, inputs['latest'].expand_as(raw)))

    def test_missing_target_request_is_rejected(self):
        r = row(); r['values'][31] = None
        with self.assertRaises(ValueError):
            data.batch([(r, 30, np.array([1]), 1.)])

    def test_cell_weights_sum_to_one_per_cve(self):
        r = row(); r['values'][50:55] = [None]*5
        total = 0.
        for items in data.packets(data.requests(r), 13):
            total += float(data.batch(items)[3].sum())
        self.assertAlmostEqual(total, 1., places=12)

    def test_equal_cves_not_flat_cells(self):
        mask1 = np.zeros((90, 90), bool); mask1[0, 0] = True
        mask2 = np.zeros_like(mask1); mask2[0, :2] = True; mask2[1, 0] = True
        y = np.zeros((90, 90))
        a = ev.cve_metrics(y, np.ones_like(y), mask1)
        b = ev.cve_metrics(y, np.zeros_like(y), mask2)
        out = ev.summarize([{'cve': 'A', 'metrics': a}, {'cve': 'B', 'metrics': b}])
        self.assertEqual(out['all']['mae'], .5)
        self.assertEqual(out['all']['cells'], 4)
        self.assertIsNone(out['day30_to_day60']['mae'])

    def test_anchor_then_lead_weights(self):
        mask = np.zeros((90, 90), bool); mask[0, :2] = True; mask[1, 0] = True
        y = np.zeros((90, 90)); raw = np.zeros_like(y); raw[1, 0] = 1.
        self.assertEqual(ev.cve_metrics(y, raw, mask)['all']['mae'], .5)

    def test_correction_training_only_and_weighted_median(self):
        self.assertEqual(controls.weighted_median([0., 1., 2.], [8., 1., 1.]), 0.)
        with self.assertRaises(ValueError):
            controls.fit([row('validation', '2025-03-02')])
        r = row(); r['values'] = [None]*89+[.2, .4]
        state = controls.fit([r])
        self.assertAlmostEqual(state['corrections'][0], .2)
        self.assertEqual(state['corrections'][1], 0.)
        self.assertEqual(state['leads'][1]['fallback'], 'no_training_cells_use_zero')
        self.assertEqual(state['training_ids'], [r['cve']])

    def test_clipping_and_raw_metrics(self):
        mask = np.zeros((90, 90), bool); mask[0, 0] = True
        y = np.zeros((90, 90)); raw = y.copy(); raw[0, 0] = -2.
        result = ev.cve_metrics(y, raw, mask)['all']
        self.assertEqual(result['mae'], 0.)
        self.assertEqual(result['raw_mae'], 2.)
        self.assertEqual(result['clipping_rate'], 1.)

    def test_persistence_uses_last_observed_and_full_precision(self):
        r = row(); r['values'][30] = None
        value = controls.raw_prediction(r, 30, [1, 30])
        self.assertTrue(np.array_equal(value, np.array([r['values'][29]]*2)))
        inputs, _, _, _ = data.batch([(r, 30, np.array([1, 30]), 1.)])
        model = runner.model_for(17, data.settings())
        self.assertTrue(np.array_equal(data.forward(model, inputs)[0].detach().numpy()[0], value))

    def test_seed_replay_and_checkpoint_reload(self):
        r = row(); r['values'] = [None]*88+[.1, .2, .3]
        p = data.settings()
        states = []
        for _ in range(2):
            model = runner.model_for(17, p)
            opt = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
            loss, norm = runner.train_epoch(model, [r], opt, 17, 1, p, progress=False)
            self.assertGreater(norm, 0)
            states.append(deepcopy(model.state_dict()))
        self.assertTrue(all(torch.equal(states[0][k], v) for k, v in states[1].items()))
        inputs, _, _, _ = data.batch(list(data.requests(r)))
        expected = data.forward(model, inputs)[0]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'model.pt'; torch.save(states[0], path)
            restored = runner.model_for(43, p)
            restored.load_state_dict(torch.load(path, weights_only=True))
            self.assertTrue(torch.equal(expected, data.forward(restored, inputs)[0]))

    def test_packets_preserve_objective_gradient(self):
        r = row(); r['values'] = [None]*87+[.1, .2, .3, .4]
        grads = []
        for size in (1, 32):
            model = runner.model_for(17, data.settings())
            for items in data.packets(data.requests(r), size):
                x, y, valid, weights = data.batch(items)
                raw, _ = data.forward(model, x)
                ((raw[valid]-y[valid]).abs()*weights[valid]).sum().backward()
            grads.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]))
        torch.testing.assert_close(grads[0], grads[1], atol=1e-7, rtol=1e-6)

    def test_matched_bootstrap_reproduces_and_rejects_unmatched(self):
        r = row(); r['values'] = [None]*88+[.1, .2, .3]
        result = ev.evaluate([r], 32, progress=False)
        results = {k: deepcopy(result) for k in ('persistence', 'lead_correction', 'seed_17', 'seed_29', 'seed_43')}
        a = ev.paired_intervals(results, 10)
        self.assertEqual(a, ev.paired_intervals(results, 10))
        self.assertEqual(a[0]['metrics']['mae']['difference'], 0.)
        results['seed_17']['per_cve'][0]['cve'] = 'wrong'
        with self.assertRaises(ValueError):
            ev.paired_intervals(results, 10)

    def test_full_fit_disabled(self):
        self.assertFalse(data.settings()['full_training_enabled'])
        self.assertFalse(data.settings()['test_access_enabled'])


if __name__ == '__main__':
    unittest.main()
