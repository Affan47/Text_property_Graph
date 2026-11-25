"""Product/version extension regression tests; no dataset training or downloads."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'01_core'),str(ROOT/'10_extensions')]
from tpg_security_product_v1 import ProductVersionSecurityPipeline, version_mentions, VERSION
from tpg.schema.types import NodeType, SecurityEdgeType, EdgeType


class ProductVersionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = ProductVersionSecurityPipeline()

    def graph(self,text):
        return self.pipeline.run(text,doc_id='regression-example')

    def pairs(self,graph):
        return [(graph.get_node(e.source).properties.text,graph.get_node(e.target).properties.text,
                 e.properties.extra['version_operator']) for e in graph.edges(SecurityEdgeType.HAS_VERSION)]

    def test_real_wording_and_generalized_product_names(self):
        examples = [
            ('The Contact Form Entries WordPress plugin before 1.3.0 does not validate data.',
             ('Contact Form Entries WordPress plugin','1.3.0','<')),
            ('An arbitrary file upload vulnerability in /inc/modules_install.php of Pluck-CMS v4.7.18 allows code execution.',
             ('Pluck-CMS','4.7.18','mentioned_version')),
            ('SQL Injection in delete-tracker.php in Daily Habit Tracker v.1.0 allows remote code execution.',
             ('Daily Habit Tracker','1.0','mentioned_version')),
            ('python-jose through 3.3.0 allows attackers to cause a denial of service.',
             ('python-jose','3.3.0','<=')),
            ('This issue affects Brooklyn | Creative Multi-Purpose Responsive WordPress Theme: from n/a through 4.9.7.6.',
             ('Brooklyn | Creative Multi-Purpose Responsive WordPress Theme','4.9.7.6','<=')),
            ('ESET File Security for Microsoft Windows Server 8.0.12013.0 is affected.',
             ('ESET File Security for Microsoft Windows Server','8.0.12013.0','mentioned_version')),
            ('Mitsubishi Electric ICONICS Suite versions 10.97.3 and prior are affected.',
             ('Mitsubishi Electric ICONICS Suite','10.97.3','<=')),
            ('The Example Dashboard WordPress plugin before 9.12.3 has a vulnerability.',
             ('Example Dashboard WordPress plugin','9.12.3','<')),
            ('widget-engine through 2.3.4 is affected.',('widget-engine','2.3.4','<=')),
            ('Apache 2.4.51 is affected.',('Apache','2.4.51','mentioned_version')),
            ('Example Server up to 1.2 is mentioned.',('Example Server','1.2','upper_bound_unspecified_inclusivity')),
        ]
        for text,expected in examples:
            with self.subTest(text=text):
                graph=self.graph(text)
                self.assertEqual(self.pairs(graph),[expected])
                self.assertEqual(graph.metadata['security_extraction_version'],VERSION)
                self.assertFalse(graph.validate())

    def test_complete_versions_no_suffix_or_score_fragments(self):
        text='Pluck-CMS v4.7.18; Example v.2.3.4-rc.1+build.7; CVSS v3.1 score 9.8/10; IP address 192.168.1.1; x4.7.18; 1.2.x; 1.2.*.'
        self.assertEqual([m.group('number') for m in version_mentions(text)],['4.7.18','2.3.4-rc.1+build.7'])

    def test_no_filename_or_compound_substring_software(self):
        g=self.graph('python-jose through 3.3.0 uses install.php and tool.py.')
        software=[n.properties.text for n in g.nodes(NodeType.ENTITY) if n.properties.entity_type=='SOFTWARE']
        self.assertEqual(software,['python-jose'])

    def test_abstains_on_coordination_distance_and_cross_sentence(self):
        for text in ['Apache and nginx before 2.0 are affected.',
                     'Apache has a vulnerability. Version 2.0 is mentioned.',
                     'Apache uses a configuration identified by 2.0.',
                     'Apache from 1.0 to 2.0 is affected.',
                     'Apache before 2.0 and later is affected.']:
            with self.subTest(text=text):
                g=self.graph(text)
                self.assertEqual(self.pairs(g),[])
                versions=[n for n in g.nodes(NodeType.ENTITY) if n.properties.entity_type=='VERSION']
                self.assertTrue(versions)
                self.assertTrue(all(n.properties.extra['ownership_abstention'] for n in versions))

    def test_two_explicit_pairs_do_not_cross_attach(self):
        g=self.graph('Apache before 2.4.51 and OpenSSL before 3.0.1 are affected.')
        self.assertEqual(self.pairs(g),[('Apache','2.4.51','<'),('OpenSSL','3.0.1','<')])

    def test_qualifiers_source_spans_and_parallel_edges(self):
        text='Example Suite through v4.7.18 is affected.'
        g=self.graph(text)
        for e in g.edges(SecurityEdgeType.HAS_VERSION):
            p=e.properties.extra
            self.assertEqual(text[slice(*p['qualifier_span'])],p['qualifier_text'])
            self.assertIn('through v4.7.18',p['qualifier_text'])
            self.assertEqual(text[slice(*p['source_span'])],g.get_node(e.source).properties.text)
            self.assertEqual(text[slice(*p['target_span'])],g.get_node(e.target).properties.text)
            generic=[x for x in g.edges(EdgeType.ENTITY_REL) if x.properties.entity_rel_type=='HAS_VERSION']
            self.assertEqual(len(generic),1)
            self.assertEqual(generic[0].properties.extra,p)
        exported=self.pipeline.export_pyg(g,embedding_dim=0,use_security_edge_types=True)
        self.assertEqual(exported['n_dropped_security_edges'],0)
        self.assertIsNone(exported['y'])
        self.assertEqual(len(exported['edge_features']),len(g.edges()))

    def test_no_record_id_invented_as_text_entity(self):
        g=self.pipeline.run('Apache 2.4.51 is affected.',doc_id='CVE-2022-25866')
        self.assertFalse(any(n.properties.entity_type=='CVE_ID' for n in g.nodes(NodeType.ENTITY)))

    def test_other_security_relations_keep_negation(self):
        g=self.graph('CVE-2024-1000 may not affect Apache 2.4.51.')
        affects=g.edges(SecurityEdgeType.AFFECTS)
        self.assertTrue(affects)
        self.assertTrue(all(e.properties.extra['polarity']=='negated' for e in affects))

    def test_generic_only_option(self):
        g=ProductVersionSecurityPipeline(include_security_relations=False).run('Apache before 2.4.51 is affected.')
        self.assertEqual(g.edges(SecurityEdgeType.HAS_VERSION),[])
        self.assertEqual(len([e for e in g.edges(EdgeType.ENTITY_REL) if e.properties.entity_rel_type=='HAS_VERSION']),1)

    def test_hashseed_determinism_and_frozen_core_unchanged(self):
        files=list((ROOT/'01_core/tpg').rglob('*.py'))
        before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        code = f'''import sys,json,hashlib
sys.path[:0]={repr([str(ROOT/'01_core'),str(ROOT/'10_extensions')])}
from tpg_security_product_v1 import ProductVersionSecurityPipeline
p=ProductVersionSecurityPipeline()
g=p.run("The package czproject/git-php before 4.0.3 uses git and php.",doc_id="test")
payload=p.export_pyg(g,embedding_dim=0,use_security_edge_types=True)
print(hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest())
'''
        hashes=[]
        for seed in ('0','4','19'):
            result=subprocess.run([sys.executable,'-c',code],env={**os.environ,'PYTHONHASHSEED':seed},
                                  capture_output=True,text=True,check=True,timeout=90)
            hashes.append(result.stdout.strip())
        self.assertEqual(len(set(hashes)),1)
        self.assertEqual(before,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


if __name__=='__main__':
    unittest.main()
