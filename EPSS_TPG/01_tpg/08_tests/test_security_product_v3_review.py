"""Saved graph comparison preserves evidence while ignoring identifier shifts."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'02_examples/01_scripts'))
import review_security_product_v3 as review


class V3SavedReviewTests(unittest.TestCase):
    def graph(self):
        return {'vertices': [
            {'id': 1, 'label': 'ENTITY', 'properties': {'TEXT': '16.1.x', 'ENTITY_TYPE': 'CARDINAL',
                'CHAR_START': 13, 'CHAR_END': 19}},
            {'id': 2, 'label': 'ENTITY', 'properties': {'TEXT': '16.1.x', 'ENTITY_TYPE': 'VERSION',
                'CHAR_START': 13, 'CHAR_END': 19, 'EXTRA': {'extraction_policy': 'v2'}}}],
            'edges': [{'id': 'e0', 'outV': 1, 'inV': 2, 'label': 'COREF',
                       'properties': {'COREF_CLUSTER': 1, 'EXTRA': {'status': 'candidate'}}}]}

    def test_id_policy_changes_ignored_but_properties_retained(self):
        before = self.graph()
        after = deepcopy(before)
        after['vertices'][1]['id'] = 50
        after['vertices'][1]['properties']['EXTRA']['extraction_policy'] = 'v3'
        after['edges'][0].update(id='e99', inV=50)
        self.assertFalse(any(review.compare(before, after).values()))
        after['edges'][0]['properties']['EXTRA']['status'] = 'asserted'
        difference = review.compare(before, after)
        self.assertEqual(len(difference['added_edges']), 1)
        self.assertEqual(len(difference['removed_edges']), 1)

    def test_multiplicity_and_broken_endpoints(self):
        before = self.graph()
        after = deepcopy(before)
        after['edges'].append(deepcopy(after['edges'][0]))
        self.assertEqual(review.compare(before, after)['added_edges'][0]['count'], 1)
        after['edges'][0]['inV'] = 99
        with self.assertRaisesRegex(ValueError, 'Missing edge endpoint'):
            review.compare(before, after)

    def test_same_span_is_not_separate_occurrence(self):
        graph = self.graph()
        self.assertEqual(len(review.same_span_coreferences(graph)), 1)
        graph['vertices'][1]['properties'].update(CHAR_START=30, CHAR_END=36)
        self.assertFalse(review.same_span_coreferences(graph))


if __name__ == '__main__':
    unittest.main()
