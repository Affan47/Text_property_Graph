"""Exact structural checks for inspecting immutable, differently numbered graphs."""
import copy
import unittest

import torch

from tpg_cached_structure import align_structure


class StructureTests(unittest.TestCase):
    def fixture(self):
        vocab = {'node_types': {'TOKEN': 0, 'ENTITY': 1}}
        x = torch.tensor([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 1., 0.]])
        cache = {'x': x, 'edge_index': torch.tensor([[2, 3, 2, 2], [0, 1, 3, 3]]),
                 'edge_type': torch.tensor([0, 0, 1, 1]),
                 'edge_features': torch.zeros(4, 64)}
        cache['edge_features'][3, 9] = -1
        return vocab, copy.deepcopy(cache), cache

    def test_exact_and_no_mutation(self):
        vocab, new, old = self.fixture()
        before = copy.deepcopy(old)
        result = align_structure(new, old, vocab)
        self.assertEqual(result['mode'], 'exact')
        self.assertEqual(result['node_index_to_cached_index'], [0, 1, 2, 3])
        self.assertTrue(all(torch.equal(old[k], before[k]) for k in old))

    def test_entity_permutation_and_edge_order(self):
        vocab, new, old = self.fixture()
        new['edge_index'] = torch.tensor([0, 1, 3, 2])[new['edge_index']]
        order = torch.tensor([3, 1, 0, 2])
        new['edge_index'] = new['edge_index'][:, order]
        for key in ('edge_type', 'edge_features'):
            new[key] = new[key][order]
        result = align_structure(new, old, vocab)
        self.assertEqual(result['mode'], 'entity_permutation')
        self.assertEqual(result['node_index_to_cached_index'], [0, 1, 3, 2])
        self.assertEqual(result['edge_index_to_cached_index'], order.tolist())

    def test_edge_order_only(self):
        vocab, new, old = self.fixture()
        new['edge_index'] = new['edge_index'].flip(1)
        for key in ('edge_type', 'edge_features'):
            new[key] = new[key].flip(0)
        self.assertEqual(align_structure(new, old, vocab)['mode'], 'edge_order_only')

    def test_wrong_direction_rejected(self):
        vocab, new, old = self.fixture()
        new['edge_index'][:, 2] = new['edge_index'][:, 2].flip(0)
        with self.assertRaisesRegex(ValueError, 'differs beyond'):
            align_structure(new, old, vocab)

    def test_wrong_type_rejected(self):
        vocab, new, old = self.fixture()
        new['edge_type'][2] = 99
        with self.assertRaisesRegex(ValueError, 'differs beyond'):
            align_structure(new, old, vocab)

    def test_changed_qualifier_feature_rejected(self):
        vocab, new, old = self.fixture()
        new['edge_features'][2, 63] = 1
        with self.assertRaisesRegex(ValueError, 'differs beyond'):
            align_structure(new, old, vocab)

    def test_parallel_edge_multiplicity_rejected(self):
        vocab, new, old = self.fixture()
        # Keep counts/label sets alike, but replace a distinct parallel edge.
        new['edge_features'][3] = new['edge_features'][2]
        with self.assertRaisesRegex(ValueError, 'differs beyond'):
            align_structure(new, old, vocab)

    def test_changed_token_anchor_rejected(self):
        vocab, new, old = self.fixture()
        new['edge_index'][1, 0] = 1
        with self.assertRaisesRegex(ValueError, 'differs beyond'):
            align_structure(new, old, vocab)

    def test_non_entity_relabeling_rejected(self):
        vocab, new, old = self.fixture()
        new['edge_index'] = torch.tensor([1, 0, 2, 3])[new['edge_index']]
        with self.assertRaisesRegex(ValueError, 'differs beyond'):
            align_structure(new, old, vocab)

    def test_unanchored_entities_cannot_move(self):
        vocab, new, old = self.fixture()
        old['edge_index'] = old['edge_index'][:, 2:]
        old['edge_type'] = old['edge_type'][2:]
        old['edge_features'] = old['edge_features'][2:]
        new = copy.deepcopy(old)
        new['edge_index'] = torch.tensor([0, 1, 3, 2])[new['edge_index']]
        with self.assertRaisesRegex(ValueError, 'differs beyond'):
            align_structure(new, old, vocab)

    def test_empty_edges_and_node_types(self):
        vocab, new, old = self.fixture()
        for key in ('edge_index', 'edge_type', 'edge_features'):
            old[key] = old[key][:, :0] if key == 'edge_index' else old[key][:0]
        new = {k: v.tolist() for k, v in old.items()}
        self.assertEqual(align_structure(new, old, vocab)['mode'], 'exact')
        new['x'][0] = [0., 1., 0.]
        with self.assertRaisesRegex(ValueError, 'node positions/types'):
            align_structure(new, old, vocab)


if __name__ == '__main__':
    unittest.main()
