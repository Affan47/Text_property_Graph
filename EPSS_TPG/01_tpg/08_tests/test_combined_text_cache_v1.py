import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ('01_tpg/10_extensions', '01_tpg/01_core', '02_prediction/01_package'):
    sys.path.insert(0, str(ROOT / p))
from tpg_combined_v1 import common, sources, graphs
from tpg_combined_v1.embeddings import pool_spans, discarded_fragment, model_inputs


def doc(source, text, prose=True):
    h = hashlib.sha256(text.encode()).hexdigest()
    return {'source': source, 'text': text, 'text_sha256': h, 'document_id': source + h, 'is_prose': prose}


class CombinedTests(unittest.TestCase):
    def test_combines_original_text_and_spans(self):
        docs = [doc('M', 'Discussion.'), doc('D', 'Description.\nDetails.'), doc('G', 'Patch summary.')]
        text, spans = sources.combine(docs, 'DMGN', 'prose')
        self.assertEqual(text, 'Description.\nDetails.\n\nDiscussion.\n\nPatch summary.')
        for p in spans:
            self.assertEqual(hashlib.sha256(text[p['start']:p['end']].encode()).hexdigest(), p['text_sha256'])

    def test_missing_source_has_no_fallback(self):
        self.assertEqual(sources.combine([doc('D', 'Description.')], 'GN', 'prose'), ('', []))

    def test_prose_policy_explicit(self):
        docs = [doc('M', 'CVE-2022-46908', False)]
        self.assertEqual(sources.combine(docs, 'M', 'prose'), ('', []))
        self.assertEqual(sources.combine(docs, 'M', 'all')[0], 'CVE-2022-46908')

    def test_text_hash_failure(self):
        d = doc('D', 'Original')
        d['text'] = 'Changed'
        with self.assertRaises(ValueError):
            sources.combine([d], 'D', 'prose')

    def test_variant_alias_when_summary_missing(self):
        docs = [doc('D', 'Description'), doc('M', 'Post')]
        self.assertEqual(sources.combine(docs, 'DM', 'prose'), sources.combine(docs, 'DMGN', 'prose'))

    def test_offsets_byte_duplicates_and_special_tokens(self):
        offsets = [(0, 0), (0, 2), (2, 3), (2, 3), (5, 8), (0, 0)]
        vectors = np.arange(12).reshape(6, 2)
        sums, counts = pool_spans(offsets, vectors, [(0, 3), (5, 8), (-1, -1), (3, 5)])
        np.testing.assert_array_equal(counts, [3, 1, 0, 0])
        np.testing.assert_array_equal(sums[0], vectors[1:4].sum(0))

    def test_windows_sum_without_truncation(self):
        a, n = pool_spans([(0, 2)], [[1, 3]], [(0, 5)])
        b, m = pool_spans([(3, 5)], [[5, 7]], [(0, 5)])
        np.testing.assert_array_equal((a + b) / (n + m)[:, None], [[3, 5]])

    def test_no_blanket_unicode_exception(self):
        empty = lambda *args, **kwargs: {'input_ids': []}
        nonempty = lambda *args, **kwargs: {'input_ids': [1]}
        self.assertTrue(discarded_fragment('\xad\u200d\ufe0f', empty))
        self.assertFalse(discarded_fragment('ordinary', empty))
        self.assertFalse(discarded_fragment('\xad', nonempty))

    def test_gzip_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a.gz', Path(tmp) / 'b.gz'
            common.write(a, {'text': '\u200d', 'v': 2})
            common.write(b, {'v': 2, 'text': '\u200d'})
            self.assertEqual(common.sha(a), common.sha(b))

    def test_one_combined_graph_subtypes_and_sources(self):
        import torch
        text, spans = sources.combine([doc('D', 'Apache 2.4 allows remote attackers to execute code.'),
                                       doc('M', 'A critical vulnerability affects Apache 2.4.')], 'DM', 'prose')
        entry = {'text': text, 'spans': spans, 'processing_identity': 'unit'}
        graph = graphs.Builder().build('unit', entry)
        self.assertEqual(sum(n['label'] == 'DOCUMENT' for n in graph['vertices']), 1)
        self.assertEqual({p['source'] for p in graph['metadata']['combined_text']['spans']}, {'D', 'M'})
        tensor = graphs.tensors(graph)
        self.assertEqual(tensor['subtype_vocabulary']['SEVERITY'], 31)
        self.assertEqual(len(tensor['subtype_vocabulary']), 32)
        self.assertEqual(tensor['edge_features'].shape, (len(graph['edges']), 64))
        tensor.update(node_embeddings=torch.zeros(len(graph['vertices']), 768),
                      embedding_mask=torch.zeros(len(graph['vertices']), dtype=torch.bool))
        self.assertNotIn('y', model_inputs(tensor))
        self.assertEqual(model_inputs(tensor)['x'].shape[1], 782)


if __name__ == '__main__':
    unittest.main()
