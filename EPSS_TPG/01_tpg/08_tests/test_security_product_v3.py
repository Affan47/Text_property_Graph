"""Narrow v3 filter regression and genuine-IP negative controls."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'01_core'), str(ROOT/'10_extensions')]
import tpg_security_product_v2 as v2
import tpg_security_product_v3 as v3
from tpg.schema.types import NodeType, EdgeType, SecurityEdgeType


class ProductV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = v3.ProductVersionSecurityPipeline()

    @staticmethod
    def numbers(text, module=v3):
        return [m.group('number') for m in module.version_mentions(text)]

    def test_real_big_ip_wording_recovers_first_wildcard(self):
        text = ('On F5 BIG-IP 16.1.x versions prior to 16.1.2.2, 15.1.x versions prior to 15.1.5.1, '
                '14.1.x versions prior to 14.1.4.6, and 13.1.x versions prior to 13.1.5, '
                'when an Active mode-enabled FTP profile is configured on a virtual server.')
        self.assertNotIn('16.1.x', self.numbers(text, v2))
        self.assertEqual(self.numbers(text), ['16.1.x', '16.1.2.2', '15.1.x', '15.1.5.1',
                                             '14.1.x', '14.1.4.6', '13.1.x', '13.1.5'])
        g = self.pipeline.run(text, doc_id='CVE-2022-26130')
        versions = [n for n in g.nodes(NodeType.ENTITY) if n.properties.entity_type == 'VERSION']
        first = next(n for n in versions if n.properties.text == '16.1.x')
        self.assertEqual(first.properties.extra['ownership_abstention'], 'unsupported_wildcard_release')
        self.assertIsNone(first.properties.extra['ownership_candidate'])
        self.assertFalse(any(e.target == first.id for e in g.edges(SecurityEdgeType.HAS_VERSION)))
        self.assertEqual(first.properties.extra['extraction_policy'], v3.VERSION)
        self.assertEqual(text[first.properties.char_start:first.properties.char_end], '16.1.x')

    def test_compound_name_controls_not_hardcoded_big_ip(self):
        for prefix in ('BIG-IP ', 'BIG-IP: ', 'BIG-IP= ', 'big-ip ', 'Example-IP ',
                       'Vendor.IP ', 'Vendor/IP ', 'Example-Address ', 'Vendor_IP '):
            with self.subTest(prefix=prefix):
                self.assertEqual(self.numbers(prefix+'16.1.x'), ['16.1.x'])

    def test_genuine_ipv4_contexts_excluded(self):
        for text in ('IP address 192.168.1.1', 'IP: 10.0.0.1', 'source IP = 172.16.0.1',
                     'Address: 127.0.0.1', 'address=0.0.0.0', '(IP: 255.255.255.255)',
                     'destination ip address: 8.8.8.8', 'IPv4 address 192.0.2.1',
                     'F5 BIG-IP uses IP address 192.0.2.10', 'F5 BIG-IP address 192.0.2.10'):
            with self.subTest(text=text):
                self.assertEqual(self.numbers(text), [])
                g = self.pipeline.run(text)
                self.assertFalse(any(n.properties.entity_type == 'VERSION' for n in g.nodes(NodeType.ENTITY)))

    def test_four_part_releases_not_classified_by_number_alone(self):
        self.assertEqual(self.numbers('F5 BIG-IP 16.1.2.2 is affected.'), ['16.1.2.2'])
        self.assertEqual(self.numbers('Example Server version 192.168.1.1 is mentioned.'), ['192.168.1.1'])
        self.assertEqual(self.numbers('F5 BIG-IP 16.1.2.2 uses IP address 192.0.2.10.'), ['16.1.2.2'])

    def test_address_boundary_not_created_by_truncated_prefix(self):
        text = 'Example-IP' + ' '*63 + '16.1.x'
        self.assertEqual(self.numbers(text), ['16.1.x'])
        self.assertEqual(self.numbers('IP'+' '*70+'192.0.2.10'), [])

    def test_cvss_exclusions_and_unrelated_candidates_unchanged(self):
        for text in ('CVSS v3.1 score 9.8/10', 'CVSS:3.1/AV:N/AC:H', 'base score: 5.3',
                     'python-jose through 3.3.0', 'Example 4.6 Patch 518 and before',
                     'Example 1.2.x', 'Openscad openscad-2020.12-RC2', 'before build 3640'):
            with self.subTest(text=text):
                self.assertEqual(self.numbers(text), self.numbers(text, v2))

    def test_v3_policy_provenance_export_and_generic_only(self):
        text = 'F5 BIG-IP 16.1.2.2 is affected.'
        g = self.pipeline.run(text, doc_id='source-example')
        self.assertEqual(g.metadata['security_extraction_version'], v3.VERSION)
        typed = g.edges(SecurityEdgeType.HAS_VERSION)
        generic = [e for e in g.edges(EdgeType.ENTITY_REL) if e.properties.entity_rel_type == 'HAS_VERSION']
        self.assertEqual(len(typed), 1)
        self.assertEqual(typed[0].properties.extra, generic[0].properties.extra)
        self.assertEqual(typed[0].properties.extra['extraction_policy'], v3.VERSION)
        self.assertEqual(g.get_node(typed[0].source).properties.text, 'F5 BIG-IP')
        for n in g.nodes(NodeType.ENTITY):
            if n.properties.entity_type in ('SOFTWARE', 'VERSION'):
                self.assertEqual(n.properties.extra['extraction_policy'], v3.VERSION)
                self.assertEqual(text[n.properties.char_start:n.properties.char_end], n.properties.text)
        payload = self.pipeline.export_pyg(g, embedding_dim=0, use_security_edge_types=True)
        self.assertIsNone(payload['y'])
        self.assertEqual(payload['n_dropped_security_edges'], 0)
        self.assertFalse(g.validate())
        generic_graph = v3.ProductVersionSecurityPipeline(False).run(text)
        self.assertFalse(generic_graph.edges(SecurityEdgeType.HAS_VERSION))
        self.assertEqual(len([e for e in generic_graph.edges(EdgeType.ENTITY_REL)
                              if e.properties.entity_rel_type == 'HAS_VERSION']), 1)

    def test_determinism_and_frozen_modules_unchanged(self):
        paths = list((ROOT/'01_core/tpg').rglob('*.py')) + [ROOT/'10_extensions'/f'tpg_security_product_v{i}.py' for i in (1, 2)]
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        code = f'''import sys,json,hashlib
sys.path[:0]={repr([str(ROOT/'01_core'), str(ROOT/'10_extensions')])}
from tpg_security_product_v3 import ProductVersionSecurityPipeline
p=ProductVersionSecurityPipeline()
g=p.run("F5 BIG-IP 16.1.x and Example Server 2.0 use IP address 192.0.2.1.")
print(hashlib.sha256(json.dumps(p.export_pyg(g,embedding_dim=0,use_security_edge_types=True),sort_keys=True).encode()).hexdigest())
'''
        hashes = [subprocess.run([sys.executable, '-c', code], env={**os.environ, 'PYTHONHASHSEED': seed},
                                check=True, capture_output=True, text=True, timeout=90).stdout.strip()
                  for seed in ('0', '4', '19')]
        self.assertEqual(len(set(hashes)), 1)
        self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        self.assertNotIn('16.1.x', self.numbers('F5 BIG-IP 16.1.x', v2))


if __name__ == '__main__':
    unittest.main()
