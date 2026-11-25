from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
for rel in ('01_tpg/10_extensions', '01_tpg/01_core', '02_prediction/01_package', '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT / rel))
from tpg_combined_v2.graphs import Builder, tensors, relation_features
from tpg_combined_v2.validation import validate
from tpg_combined_v2.security_v7 import correct, normalize_cve


def entry(parts, cve='CVE-2022-1234'):
    text, spans = '', []
    for i, (source, content) in enumerate(parts):
        if spans:
            text += '\n\n'
        start = len(text)
        text += content
        spans.append({'source': source, 'document_id': f'doc-{i}', 'start': start, 'end': len(text),
                      'text_sha256': hashlib.sha256(content.encode()).hexdigest()})
    return dict(cve=cve, split='train', text=text, spans=spans, processing_identity='unit')


class SemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = Builder()

    def build(self, parts, cve='CVE-2022-1234'):
        e = entry(parts, cve)
        return e, self.builder.build('unit', e)

    def test_document_boundaries_same_and_different_sources(self):
        e, g = self.build([('D', 'Access can be disabled'), ('M', 'CVE-2022-1234 is critical.'),
                           ('M', 'However, a remote attacker can execute code.')])
        nodes = {n['id']: n for n in g['vertices']}
        cross = [x for x in g['edges'] if all(nodes[i]['properties']['DOC_IDX'] >= 0 for i in (x['outV'], x['inV']))
                 and nodes[x['outV']]['properties']['DOC_IDX'] != nodes[x['inV']]['properties']['DOC_IDX']]
        self.assertEqual(len(cross), 6)
        self.assertTrue(all(x['label'] in ('NEXT_TOKEN', 'NEXT_SENT', 'NEXT_PARA') and
                            x['properties']['EXTRA']['concatenation_boundary'] for x in cross))
        self.assertFalse(any(n['properties'].get('TEXT') == 'disabled\n\nCVE' for n in g['vertices']))
        validate(e, g)

    def test_unicode_offsets(self):
        for prefix in ('\u0130', '\u00df', '\ufb01', '\u0131'):
            e, g = self.build([('D', prefix + ' buffer overflow allows a remote attacker to cause denial of service using a template engine.')])
            entities = {n['properties']['TEXT'] for n in g['vertices'] if n['label'] == 'ENTITY'}
            self.assertIn('buffer overflow', entities)
            self.assertIn('denial of service', entities)
            self.assertIn('template engine', entities)

    def test_target_and_other_cves_and_unicode_hyphens(self):
        _, g = self.build([('D', 'CVE\u20112022\u20111234 is critical. CVE-2023-12345 is critical.')])
        mentions = [n for n in g['vertices'] if n['properties'].get('ENTITY_TYPE') == 'CVE_ID']
        self.assertEqual([n['properties']['EXTRA']['target_cve'] for n in mentions], [True, False])
        self.assertEqual(normalize_cve('cve\u20132022\u20131234'), 'CVE-2022-1234')
        t = tensors(g)
        self.assertEqual(t['target_cve'].sum().item(), 1)
        self.assertEqual(t['subtype_vocabulary']['SEVERITY'], 31)
        anchors = {x['properties']['EXTRA']['anchor'] for x in g['edges'] if x['label'] == 'SEC_HAS_SEVERITY'}
        self.assertEqual(anchors, {'target', 'other'})

    def test_absent_target_not_invented(self):
        _, g = self.build([('D', 'Apache 2.4.1 has a buffer overflow.')])
        self.assertFalse(g['metadata']['target_cve_mentioned'])
        self.assertFalse(tensors(g)['target_cve'].any())

    def test_security_twins_removed_but_verb_relations_retained(self):
        _, g = self.build([('D', 'CVE-2022-1234 affects Apache 2.4.1.')])
        sec = [e for e in g['edges'] if e['label'].startswith('SEC_')]
        self.assertTrue(sec)
        self.assertTrue(any(e['label'] == 'ENTITY_REL' and e['properties']['ENTITY_REL_TYPE'] == 'affect' for e in g['edges']))
        self.assertFalse(any(e['label'] == 'ENTITY_REL' and e['properties'].get('ENTITY_REL_TYPE') == 'HAS_VERSION' for e in g['edges']))

    def test_preposition_object_and_mark(self):
        _, g = self.build([('D', 'An attacker writes to files if the server allows access.')])
        args = [n for n in g['vertices'] if n['label'] == 'ARGUMENT']
        self.assertFalse(any(n['properties'].get('DEP_REL') in ('mark', 'prep') for n in args))
        prep = [n for n in args if n['properties'].get('SRL_ROLE') == 'ARGM-PREP']
        self.assertTrue(any(n['properties']['TEXT'] == 'files' and n['properties']['EXTRA']['preposition_lemma'] == 'to' for n in prep))

    def test_phrase_membership_and_text(self):
        e, g = self.build([('D', 'The attacker is-losing access because the server has not fully fixed the issue.')])
        phrases = [n for n in g['vertices'] if n['label'] in ('CLAUSE', 'VERB_PHRASE')]
        self.assertTrue(phrases)
        self.assertTrue(all(n['properties']['EXTRA']['member_token_ids'] for n in phrases))
        validate(e, g)

    def test_cve_2022_28859_clause_trailing_whitespace(self):
        text = ('The error response displays user supplied data without proper HTML or CSS escaping, '
                'allowing an attacker to inject malicious style rules or script URLs.  \nNext sentence.')
        e, g = self.build([('D', text)], cve='CVE-2022-28859')
        for n in g['vertices']:
            if n['label'] in ('CLAUSE', 'VERB_PHRASE'):
                self.assertEqual(n['properties']['TEXT'], n['properties']['TEXT'].strip())
        validate(e, g)

    def test_union_pooling_excludes_gap_and_does_not_double_count(self):
        import numpy as np
        from tpg_combined_v2.embeddings import pool_spans
        sums, counts = pool_spans([(0, 2), (2, 5), (5, 7)], np.array([[1.], [1000.], [3.]]),
                                  [[[0, 2], [5, 7]], [[0, 2], [1, 2]], [[-1, -1]]])
        self.assertEqual(sums[:, 0].tolist(), [4., 1., 0.])
        self.assertEqual(counts.tolist(), [2, 1, 0])

    def test_relation_flags_distinguish_semantics(self):
        variants = ({}, {'topic_relevance': True}, {'relation': 'continuation'}, {'cross_sentence': True},
                    {'concatenation_boundary': True}, {'anchor': 'target'}, {'anchor': 'other'}, {'preposition_lemma': 'to'})
        features = [tuple(relation_features({'properties': {'EXTRA': x}}, [0.] * 64)) for x in variants]
        self.assertEqual(len(set(features)), len(variants))

    def test_markdown_whitespace_topics(self):
        _, g = self.build([('M', '### [https://example.org/page] \n \n A parser vulnerability affects software. **')])
        from tpg_combined_v2.security_v7 import topic_word
        self.assertTrue(all(topic_word(n['properties']['TEXT']) for n in g['vertices'] if n['label'] == 'TOPIC'))

    def test_conflicts_and_compatible_views(self):
        _, g = self.build([('D', 'CVE-2022-1234 affects Apache 2.4.1 with buffer overflow.')])
        original = [n for n in g['vertices'] if n['properties'].get('ENTITY_TYPE') in ('CVE_ID', 'SOFTWARE', 'VERSION', 'VULN_TYPE')]
        expected_removed, compatible = [], []
        for n in original:
            kinds = {'CVE_ID': ['ORG', 'DATE', 'LAW', 'PRODUCT', 'CARDINAL', 'PERSON'],
                     'SOFTWARE': ['ORG'], 'VERSION': ['CARDINAL'], 'VULN_TYPE': ['PERSON']}[n['properties']['ENTITY_TYPE']]
            for kind in kinds:
                node = deepcopy(n)
                node['id'] = max(x['id'] for x in g['vertices']) + 1
                node['properties'].update(ENTITY_TYPE=kind, SOURCE='spacy_frontend')
                g['vertices'].append(node)
                (compatible if n['properties']['ENTITY_TYPE'] in ('SOFTWARE', 'VERSION') else expected_removed).append(node['id'])
        fixed = correct(g)
        ids = {n['id'] for n in fixed['vertices']}
        self.assertFalse(ids & set(expected_removed))
        self.assertTrue(set(compatible) <= ids)

    def test_validation_rejects_corruption(self):
        entry_, graph = self.build([('D', 'CVE-2022-1234 affects Apache 2.4.1.')])
        bad = deepcopy(graph)
        next(n for n in bad['vertices'] if n['label'] == 'ENTITY')['properties']['TEXT'] = 'wrong'
        with self.assertRaises(ValueError):
            validate(entry_, bad)
        bad = deepcopy(graph)
        next(e for e in bad['edges'] if e['label'].startswith('SEC_'))['properties']['EXTRA']['source_span'] = [0, 1]
        with self.assertRaises(ValueError):
            validate(entry_, bad)

    def test_deterministic_rebuild(self):
        e, g = self.build([('D', 'CVE-2022-1234 affects Apache 2.4.1.'), ('M', 'An attacker writes to files.')])
        self.assertEqual(g, self.builder.build('unit', e))

    def test_model_inputs_keep_types_target_and_document_features(self):
        import torch
        from tpg_combined_v2.embeddings import model_inputs
        _, g = self.build([('D', 'CVE-2022-1234 is critical.'), ('M', 'Apache 2.4.1 is affected.')])
        tensor = tensors(g)
        count = len(g['vertices'])
        tensor.update(node_embeddings=torch.zeros(count, 768), embedding_mask=torch.zeros(count, dtype=torch.bool))
        inputs = model_inputs(tensor)
        self.assertEqual(inputs['x'].shape, (count, tensor['vocabulary']['num_node_types'] + 770))
        self.assertTrue(torch.equal(inputs['x'][:, -1], tensor['target_cve'].float()))
        self.assertTrue(torch.equal(inputs['document_index'], tensor['document_index']))
        self.assertEqual(inputs['edge_features'].shape[1], 88)
        self.assertTrue(torch.equal(inputs['subtype'], tensor['subtype']))

    def test_retirement_rejects_unapproved_paths_and_symlinks(self):
        from tpg_combined_v2.retire import safe_path
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / 'full'
            obj = root / 'objects' / 'approved'
            obj.mkdir(parents=True)
            allowed = {root: {'approved'}}
            path = obj / 'graph.json.gz'
            self.assertEqual(safe_path(path, allowed), path)
            for invalid in (obj / 'plan.json', root / 'objects/unknown/graph.json.gz',
                            root / 'raw_dataset.json', root / 'other/approved/graph.json.gz'):
                with self.assertRaises(ValueError):
                    safe_path(invalid, allowed)
            path.symlink_to(root / 'raw_dataset.json')
            with self.assertRaises(ValueError):
                safe_path(path, allowed)


if __name__ == '__main__':
    unittest.main()
