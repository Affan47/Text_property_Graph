"""Focused tests for the isolated DQ19 diagnostic."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import diagnose_forecast_residual_gradients as audit

torch = audit.torch


class GradientAuditTests(unittest.TestCase):
    def test_training_only_deterministic_selection(self):
        ids = ['CVE-' + str(i) for i in range(40)]
        splits = {c: {'split': 'train'} for c in ids}
        chosen = audit.select_ids(ids, splits)
        self.assertEqual(chosen, audit.select_ids(list(reversed(ids)), splits))
        self.assertEqual(len(set(chosen)), 32)
        splits[ids[0]]['split'] = 'validation'
        with self.assertRaisesRegex(ValueError, 'Non-training'):
            audit.select_ids(ids, splits)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            audit.select_ids(['a', 'a'], {'a': {'split': 'train'}}, 2)

    def test_known_gradient_and_decay_after_clipping(self):
        net = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            net.weight.fill_(2)
        before = deepcopy(net.state_dict())
        loss = net(torch.ones(1, 1)).sum() * 3
        with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No optimizer steps')):
            result = audit.measure_gradients(net, loss, .1, 1)
        r = result['parameters']['weight']
        self.assertAlmostEqual(r['raw_data_gradient_l2'], 3)
        self.assertAlmostEqual(r['clipped_data_gradient_l2'], 1, places=5)
        self.assertAlmostEqual(r['decay_l2'], .2)
        self.assertAlmostEqual(r['combined_gradient_l2'], 1.2, places=5)
        self.assertAlmostEqual(r['data_decay_cosine'], 1)
        self.assertTrue(result['clip_applied'])
        self.assertTrue(torch.equal(before['weight'], net.weight))
        self.assertIsNone(net.weight.grad)

    def test_zero_gradient_is_not_unused_parameter(self):
        net = torch.nn.Linear(1, 1)
        with torch.no_grad():
            net.weight.fill_(2)
            net.bias.fill_(3)
        result = audit.measure_gradients(net, (net.weight * 0).sum(), .1, 1)
        weight, bias = result['parameters']['weight'], result['parameters']['bias']
        self.assertEqual(weight['comparison'], 'decay_only')
        self.assertIsNone(weight['decay_to_data_ratio'])
        self.assertAlmostEqual(weight['decay_l2'], .2)
        self.assertEqual(bias['comparison'], 'no_active_gradient')
        self.assertEqual(bias['decay_l2'], 0)

    def test_opposing_directions(self):
        net = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            net.weight.fill_(2)
        row = audit.measure_gradients(net, -net.weight.sum(), .1, 10)['parameters']['weight']
        self.assertAlmostEqual(row['data_decay_cosine'], -1)
        self.assertAlmostEqual(row['combined_gradient_l2'], .8)

    def test_zero_final_head_blocks_upstream_data_not_decay(self):
        net = torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.Linear(2, 1))
        with torch.no_grad():
            net[1].weight.zero_()
            net[1].bias.zero_()
        loss = torch.nn.functional.smooth_l1_loss(net(torch.ones(2, 1)), torch.ones(2, 1), beta=.01)
        result = audit.measure_gradients(net, loss, .0001, 1)
        self.assertEqual(result['parameters']['0.weight']['comparison'], 'decay_only')
        self.assertGreater(result['parameters']['1.bias']['clipped_data_gradient_l2'], 0)

    def test_stale_gradients_refused(self):
        net = torch.nn.Linear(1, 1)
        net.weight.grad = torch.zeros_like(net.weight)
        with self.assertRaisesRegex(ValueError, 'Stale'):
            audit.measure_gradients(net, net.weight.sum(), .1, 1)

    def test_reproducible_no_updates(self):
        net = torch.nn.Linear(1, 1)
        def measure():
            return audit.measure_gradients(net, (net(torch.ones(2, 1)) ** 2).mean(), .0001, 1)
        self.assertEqual(measure(), measure())


if __name__ == '__main__':
    unittest.main()
