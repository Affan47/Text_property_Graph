"""Focused v2 positive and negative controls; no dataset or trained models."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'01_core'), str(ROOT/'10_extensions')]
import tpg_security_product_v2 as v2
from tpg.schema.types import NodeType, EdgeType, SecurityEdgeType


class ProductV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = v2.ProductVersionSecurityPipeline()

    def graph(self, text):
        return self.pipeline.run(text, doc_id='regression-example')

    def pairs(self, graph):
        return [(graph.get_node(e.source).properties.text, graph.get_node(e.target).properties.text,
                 e.properties.extra['version_operator']) for e in graph.edges(SecurityEdgeType.HAS_VERSION)]

    def versions(self, graph):
        return [n for n in graph.nodes(NodeType.ENTITY) if n.properties.entity_type == 'VERSION']

    def test_parenthesized_desigo_and_unseen_name(self):
        text = ('A vulnerability has been identified in Desigo DXR2 (All versions < V01.21.142.5-22), '
                'Desigo PXC3 (All versions < V01.21.142.4-18), '
                'Desigo PXC4 (All versions < V02.20.142.10-10884), '
                'Desigo PXC5 (All versions < V02.20.142.10-10884).')
        g = self.graph(text)
        self.assertEqual([x[0] for x in self.pairs(g)], ['Desigo DXR2', 'Desigo PXC3', 'Desigo PXC4', 'Desigo PXC5'])
        self.assertTrue(all(x[2] == '<' for x in self.pairs(g)))
        self.assertEqual(self.pairs(self.graph('Example Router (All versions < v2.3.4) is affected.')),
                         [('Example Router', '2.3.4', '<')])

    def test_generic_words_not_products(self):
        for word in ('All', 'Any', 'Some', 'These', 'Affected', 'Previous', 'Latest'):
            with self.subTest(word=word):
                g = self.graph(f'{word} versions before 1.2.3 are affected.')
                self.assertEqual(self.pairs(g), [])
                self.assertTrue(self.versions(g)[0].properties.extra['ownership_abstention'])

    def test_parser_split_name_abstains_without_cross_sentence_guess(self):
        text = 'Oracle GraalVM Enterprise Edition: 19.3.5'
        match = next(v2.version_mentions(text))
        owner, reason = v2.owner_candidate(text, text.index('Enterprise'), match)
        self.assertIsNone(owner)
        self.assertEqual(reason, 'parser_split_product_requires_review')
        self.assertEqual(self.pairs(self.graph('Apache is affected. Version 2.0 is mentioned.')), [])

    def test_generic_package_prefix_removed(self):
        self.assertEqual(self.pairs(self.graph('The package Snyk CLI before 1.1294.0 is vulnerable.')),
                         [('Snyk CLI', '1.1294.0', '<')])

    def test_trailing_bounds_and_patch_preserved(self):
        for text, expected in [
            ('IpMatcher 1.0.4.1 and below is affected.', ('IpMatcher', '1.0.4.1', '<=')),
            ('Acrobat Reader DC versions 22.001.20085 (and earlier) are affected.',
             ('Acrobat Reader DC', '22.001.20085', '<=')),
            ('KiTTY versions 0.76.1.13 and before are affected.', ('KiTTY', '0.76.1.13', '<=')),
            ('Ivanti Cloud Services Appliance versions 4.6 Patch 518 and before are affected.',
             ('Ivanti Cloud Services Appliance', '4.6 Patch 518', '<=')),
            ('Example Server 2.0 (and later) is supported.', ('Example Server', '2.0', '>='))]:
            with self.subTest(text=text):
                g = self.graph(text)
                self.assertEqual(self.pairs(g), [expected])
                edge = g.edges(SecurityEdgeType.HAS_VERSION)[0]
                self.assertIn('and ', edge.properties.extra['qualifier_text'])

    def test_unsupported_release_mentions_explicit_not_shortened(self):
        cases = [
            ('Openscad openscad-2020.12-RC2 is affected.', '2020.12-RC2', 'product_prefixed_release'),
            ('Acronis Snap Deploy (Windows) before build 3640', 'build 3640', 'integer_build'),
            ('Illumina Universal Copy Service v1.x and v2.x are affected.', '1.x', 'wildcard_release'),
            ('Example 20.005.3031x is affected.', '20.005.3031x', 'wildcard_release'),
            ('Example 1.2.* is affected.', '1.2.*', 'wildcard_release')]
        for text, version, form in cases:
            with self.subTest(text=text):
                g = self.graph(text)
                self.assertEqual(self.pairs(g), [])
                n = self.versions(g)[0]
                self.assertEqual(n.properties.text, version)
                self.assertEqual(n.properties.extra['ownership_abstention'], 'unsupported_' + form)
                self.assertEqual(text[slice(*n.properties.extra['version_literal_span'])],
                                 n.properties.extra['version_literal'])

    def test_ambiguous_coordination_ranges_and_conflicts(self):
        for text in ('Apache and nginx before 2.0 are affected.', 'Apache from 1.0 to 2.0 is affected.',
                     'Apache 1.0 to 2.0 is affected.', 'Apache before 2.0 and later is affected.',
                     'Example Server before 2.0 (and later) is affected.'):
            with self.subTest(text=text):
                self.assertEqual(self.pairs(self.graph(text)), [])

    def test_previous_successes_retained(self):
        for text, expected in [
            ('The Contact Form Entries WordPress plugin before 1.3.0 is affected.',
             ('Contact Form Entries WordPress plugin', '1.3.0', '<')),
            ('An issue in modules_install.php of Pluck-CMS v4.7.18 permits code execution.',
             ('Pluck-CMS', '4.7.18', 'mentioned_version')),
            ('python-jose through 3.3.0 is affected.', ('python-jose', '3.3.0', '<=')),
            ('ESET File Security for Microsoft Windows Server 8.0.12013.0 is affected.',
             ('ESET File Security for Microsoft Windows Server', '8.0.12013.0', 'mentioned_version'))]:
            with self.subTest(text=text):
                self.assertEqual(self.pairs(self.graph(text)), [expected])

    def test_parallel_edges_provenance_and_no_labels(self):
        text = 'Example Server versions 4.6 Patch 518 and before is affected.'
        g = self.graph(text)
        typed = g.edges(SecurityEdgeType.HAS_VERSION)
        generic = [e for e in g.edges(EdgeType.ENTITY_REL) if e.properties.entity_rel_type == 'HAS_VERSION']
        self.assertEqual(len(typed), 1)
        self.assertEqual(typed[0].properties.extra, generic[0].properties.extra)
        e = typed[0].properties.extra
        self.assertEqual(e['extraction_policy'], v2.VERSION)
        self.assertEqual(e['release_form'], 'patch_release')
        self.assertEqual(text[slice(*e['qualifier_span'])], e['qualifier_text'])
        for n in g.nodes(NodeType.ENTITY):
            p = n.properties
            if p.source == 'security_frontend':
                self.assertEqual(text[p.char_start:p.char_end], p.text)
        data = self.pipeline.export_pyg(g, embedding_dim=0, use_security_edge_types=True)
        self.assertIsNone(data['y'])
        self.assertEqual(data['n_dropped_security_edges'], 0)
        self.assertFalse(g.validate())

    def test_no_score_ip_or_filename_fragments(self):
        self.assertEqual(list(v2.version_mentions('CVSS v3.1 score 9.8/10; IP address 192.168.1.1; x4.7.18; /app/1.2.3')), [])
        g = self.graph('python-jose through 3.3.0 uses install.php.')
        self.assertNotIn('php', [n.properties.text for n in g.nodes(NodeType.ENTITY)
                                if n.properties.entity_type == 'SOFTWARE'])

    def test_negation_generic_only_and_no_invented_cve(self):
        g = self.graph('CVE-2024-1000 may not affect Apache 2.4.51.')
        self.assertTrue(g.edges(SecurityEdgeType.AFFECTS))
        self.assertTrue(all(e.properties.extra['polarity'] == 'negated' for e in g.edges(SecurityEdgeType.AFFECTS)))
        g = v2.ProductVersionSecurityPipeline(False).run('Apache 2.4.51 is mentioned.', doc_id='CVE-2024-1000')
        self.assertFalse(g.edges(SecurityEdgeType.HAS_VERSION))
        self.assertFalse(any(n.properties.entity_type == 'CVE_ID' for n in g.nodes(NodeType.ENTITY)))

    def test_hash_seed_determinism_and_old_files_unchanged(self):
        paths = list((ROOT/'01_core/tpg').rglob('*.py')) + [ROOT/'10_extensions/tpg_security_product_v1.py']
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        code = f'''import sys,json,hashlib
sys.path[:0]={repr([str(ROOT/'01_core'),str(ROOT/'10_extensions')])}
from tpg_security_product_v2 import ProductVersionSecurityPipeline
p=ProductVersionSecurityPipeline()
g=p.run("Example Router (All versions < v2.3.4) uses git and php.",doc_id="test")
print(hashlib.sha256(json.dumps(p.export_pyg(g,embedding_dim=0,use_security_edge_types=True),sort_keys=True).encode()).hexdigest())
'''
        hashes = [subprocess.run([sys.executable, '-c', code], env={**os.environ, 'PYTHONHASHSEED': seed},
                   check=True, capture_output=True, text=True, timeout=90).stdout.strip() for seed in ('0', '4', '19')]
        self.assertEqual(len(set(hashes)), 1)
        self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


if __name__ == '__main__':
    unittest.main()
