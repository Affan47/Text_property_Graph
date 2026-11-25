from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
for rel in ('01_tpg/10_extensions','01_tpg/01_core','02_prediction/01_package','01_tpg/08_tests'):
    sys.path.insert(0,str(ROOT/rel))
import torch
torch.set_num_threads(1)
from tpg_combined_v2 import graphs, common as c
from tpg_combined_recovery_v1.boundary import Builder
from tpg_combined_quality_v1 import reference_verifier as verify
from tpg_combined_quality_v1 import reviewed_correction as repair
from tpg_combined_quality_v1 import correction
from test_combined_semantics_v2 import entry


class QualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = Builder()

    def build(self,text):
        e=entry([('D',text)])
        return e,self.builder.build(c.digest(e),e)[0]

    def repaired(self,text):
        e,g=self.build(text); old=c.digest(g)
        new=repair.correct(g)
        repair.validate_preview(e,g,new)
        self.assertEqual(old,c.digest(g))
        self.assertEqual(new,repair.correct(new))
        return g,new

    def test_agent_is_object_not_by(self):
        old,new=self.repaired('The file was deleted by attackers.')
        self.assertTrue(any(n['label']=='ARGUMENT' and n['properties']['TEXT']=='by' for n in old['vertices']))
        args=[n for n in new['vertices'] if n['label']=='ARGUMENT']
        self.assertFalse(any(n['properties']['TEXT']=='by' for n in args))
        self.assertTrue(any(n['properties']['TEXT']=='attackers' and n['properties']['SRL_ROLE']=='ARG0' for n in args))

    def test_by_time_is_not_passive_agent(self):
        old,new=self.repaired('Update Apache by Friday.')
        self.assertFalse(any(r['rule']=='AGENT_OBJECT' for r in new['metadata']['annotation_repair_preview']['findings']))

    def test_function_call_retained(self):
        for text in ('The System() function is vulnerable.','Call free() to release memory.','The system function accepts input.'):
            with self.subTest(text=text):
                old,new=self.repaired(text)
                before=[n for n in old['vertices'] if n['label']=='ENTITY' and n['properties'].get('ENTITY_TYPE')=='CODE_ELEMENT']
                self.assertTrue(before)
                self.assertTrue(all(n in new['vertices'] for n in before))

    def test_capitalized_system_without_function_removed(self):
        _,new=self.repaired('Attackers can compromise the SYSTEM and the System.')
        self.assertFalse(any(n['label']=='ENTITY' and n['properties'].get('ENTITY_TYPE')=='CODE_ELEMENT' and
            n['properties']['TEXT'].casefold()=='system' for n in new['vertices']))

    def test_vulnerability_words_not_code(self):
        _,new=self.repaired('A use after free and buffer overflow allow code execution.')
        self.assertFalse(any(n['label']=='ENTITY' and n['properties'].get('ENTITY_TYPE')=='CODE_ELEMENT' and
            n['properties']['TEXT'].casefold() in ('free','buffer') for n in new['vertices']))
        self.assertTrue(any(n['label']=='ENTITY' and n['properties'].get('ENTITY_TYPE')=='VULN_TYPE' for n in new['vertices']))

    def test_safeguard_control_not_software(self):
        old,new=self.repaired('CIS Controls recommend Safeguard 7.1: Establish a vulnerability management process.')
        self.assertTrue(any(n['properties'].get('ENTITY_TYPE')=='SOFTWARE' and n['properties']['TEXT']=='Safeguard' for n in old['vertices']))
        self.assertFalse(any(n['properties'].get('ENTITY_TYPE')=='SOFTWARE' and n['properties']['TEXT']=='Safeguard' for n in new['vertices']))

    def test_real_cis_citation_context(self):
        old,new=self.repaired('See https://learn.cisecurity.org/example. Safeguard 7.1: Establish a vulnerability management process.')
        self.assertTrue(any(n['properties'].get('ENTITY_TYPE')=='SOFTWARE' and n['properties']['TEXT']=='Safeguard' for n in old['vertices']))
        self.assertTrue(new['metadata']['annotation_repair_preview']['rule_counts'].get('CIS_SAFEGUARD_NOT_PRODUCT'))

    def test_genuine_safeguard_release_retained(self):
        old,new=self.repaired('Safeguard 7.1 contains a vulnerability.')
        for e in old['edges']:
            if e['label']=='SEC_HAS_VERSION': self.assertIn(e,new['edges'])
        self.assertTrue(any(e['label']=='SEC_HAS_VERSION' for e in new['edges']))

    def test_generic_id_and_topic_suppression_preserves_id(self):
        e,g=self.build('CVE-2022-1234 affects Apache 2.4.1. CVE-2022-1234 affects servers.')
        entity=next(n for n in g['vertices'] if n['label']=='ENTITY' and n['properties'].get('ENTITY_TYPE')=='CVE_ID')
        duplicate=deepcopy(entity);duplicate['id']=max(n['id'] for n in g['vertices'])+1
        duplicate['properties'].update(ENTITY_TYPE='EVENT',SOURCE='spacy_frontend')
        g['vertices'].append(duplicate)
        fixed=repair.correct(g)
        self.assertIn(entity,fixed['vertices']);self.assertNotIn(duplicate,fixed['vertices'])
        self.assertFalse(any(n['label']=='TOPIC' and n['properties']['TEXT'].startswith('cve-') for n in fixed['vertices']))

    def test_no_cross_document_cis_evidence(self):
        e=entry([('D','CIS Controls are useful.'),('M','Safeguard 7.1: a commercial product release.')])
        g=self.builder.build(c.digest(e),e)[0]
        new=repair.correct(g)
        self.assertTrue(any(n['properties'].get('ENTITY_TYPE')=='SOFTWARE' and n['properties']['TEXT']=='Safeguard' for n in new['vertices']))

    def test_independent_checker_is_read_only(self):
        e,g=self.build('CVE-2022-1234 affects Apache 2.4.1. Attackers delete files.')
        before=c.digest(g)
        _,bad,_=verify.check(g,c.digest(e),e['split'])
        self.assertFalse(bad);self.assertEqual(before,c.digest(g))

    def test_independent_checker_rejects_corruptions(self):
        e,g=self.build('CVE-2022-1234 affects Apache 2.4.1. Attackers delete files.')
        def token(x):return next(n for n in x['vertices'] if n['label']=='TOKEN')
        def edge(x,label):return next(e for e in x['edges'] if e['label']==label)
        def nlabel(x,label):return next(n for n in x['vertices'] if n['label']==label)
        mutations={
            'token_text':lambda x:token(x)['properties'].update(TEXT='wrong'),
            'token_span':lambda x:token(x)['properties'].update(CHAR_START=99),
            'node_type':lambda x:token(x).update(label='UNKNOWN'),
            'duplicate_node':lambda x:x['vertices'].append(deepcopy(token(x))),
            'edge_endpoint':lambda x:x['edges'][0].update(inV=999999),
            'edge_type':lambda x:x['edges'][0].update(label='UNKNOWN'),
            'duplicate_edge':lambda x:x['edges'].append(deepcopy(x['edges'][0])),
            'reading_chain':lambda x:x['edges'].remove(edge(x,'NEXT_TOKEN')),
            'dependency_label':lambda x:edge(x,'DEP')['properties'].update(DEP_LABEL='wrong'),
            'dependency_cycle':lambda x:edge(x,'DEP').update(outV=edge(x,'DEP')['inV']),
            'sentence_text':lambda x:nlabel(x,'SENTENCE')['properties'].update(TEXT='wrong'),
            'target_flag':lambda x:token(x)['properties']['EXTRA'].update(target_cve=True),
            'document_index':lambda x:token(x)['properties'].update(DOC_IDX=99),
            'source_label':lambda x:token(x)['properties']['EXTRA'].update(source_text_type='M'),
            'false_boundary':lambda x:edge(x,'NEXT_TOKEN')['properties']['EXTRA'].update(concatenation_boundary=True),
            'statistics':lambda x:x['stats'].update(total_nodes=1),
            'document_hash':lambda x:x['metadata']['combined_text']['spans'][0].update(text_sha256='bad'),
            'root_text':lambda x:nlabel(x,'DOCUMENT')['properties'].update(TEXT='wrong'),
            'missing_membership':lambda x:x['edges'].remove(edge(x,'BELONGS_TO')),
        }
        for name,mutate in mutations.items():
            with self.subTest(name=name):
                bad=deepcopy(g);mutate(bad)
                try: _,errors,_=verify.check(bad)
                except (KeyError,IndexError,ValueError):errors={'fail_closed':1}
                self.assertTrue(errors,name)


if __name__=='__main__':unittest.main()
