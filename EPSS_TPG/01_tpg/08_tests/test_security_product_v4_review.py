"""Saved v4 review must reproduce evidence, not assign semantic judgments."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'02_examples/01_scripts'))
import review_security_product_v4 as review


class V4ReviewTests(unittest.TestCase):
    def test_different_type_flag_is_not_a_semantic_verdict(self):
        graph = {'vertices': [
            {'id': 0, 'label': 'ENTITY', 'properties': {'TEXT': 'Alpha', 'ENTITY_TYPE': 'SOFTWARE', 'CHAR_START': 0, 'CHAR_END': 5}},
            {'id': 1, 'label': 'ENTITY', 'properties': {'TEXT': 'Alpha', 'ENTITY_TYPE': 'ORG', 'CHAR_START': 6, 'CHAR_END': 11}}],
            'edges': [{'id': 'e0', 'label': 'COREF', 'outV': 0, 'inV': 1,
                       'properties': {'EXTRA': {'method': 'repeated_surface'}}}]}
        row = review.coreference_rows(graph)[0]
        self.assertTrue(row['different_entity_types'])
        self.assertEqual(row['semantic_judgment'], 'not_reviewed')
        graph['edges'][0]['properties']['EXTRA']['method'] = 'unambiguous_type_heuristic'
        self.assertFalse(review.coreference_rows(graph)[0]['different_entity_types'])

    def test_saved_graph_and_removal_evidence_reproduced(self):
        a = review.audit
        p = a.extension.v3.ProductVersionSecurityPipeline()
        text = 'On F5 BIG-IP 16.1.x versions prior to 16.1.2.2, the server stops.'
        graph = p.run(text, doc_id='synthetic')
        filtered = a.extension.filter_same_occurrence_coreference(graph)
        before, after = json.loads(p.export_graphson_string(graph)), json.loads(p.export_graphson_string(filtered))
        removed = a.check_delta(before, after, graph)
        row = {'cve': 'synthetic', 'split': 'train', 'description': text,
               'description_sha256': a.old.cache.text_digest(text), 'removed_edges': removed,
               'removed_count': len(removed), 'before': a.preview.previous.extraction(graph),
               'after': a.preview.previous.extraction(filtered)}
        self.assertEqual(review.inspect(before, after, row)['removed_count'], len(removed))
        bad = deepcopy(after)
        bad['metadata']['coreference_filter']['removed_count'] += 1
        with self.assertRaisesRegex(ValueError, 'does not reproduce'):
            review.inspect(before, bad, row)
        row['removed_edges'] = []
        with self.assertRaisesRegex(ValueError, 'Removal report differs'):
            review.inspect(before, after, row)
        row['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'split differs'):
            review.inspect(before, after, row)


if __name__ == '__main__':
    unittest.main()
