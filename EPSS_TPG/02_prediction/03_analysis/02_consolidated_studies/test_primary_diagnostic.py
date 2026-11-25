"""PS03 frozen-model parity, intervention isolation, denominators and evidence cues."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.primary_diagnostics import interventions as ops, evidence
from epss.primary_diagnostics.runner import select_ids
from epss.primary_training import batching
from test_primary_training import setup_batches, graph, RESOURCES


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_baseline_parity_and_frozen_state(self):
        with tempfile.TemporaryDirectory() as d:
            batches,v,s=setup_batches(d)
            for representation in ('security_tpg','pooled_text'):
                variant={**v,'representation':representation}
                net=batching.make_model(RESOURCES,len(batches.state['feature_names']),variant,s).eval()
                net.requires_grad_(False);before=ops.state_digest(net)
                expected,weights,masks=batching.predict(net,batches,['A','B'],variant,2)
                source,mask,numeric,_=ops.encode_sources(net,batches,['A','B'],variant)
                actual,w,m=ops.predict_fusion(net,source,mask,numeric,2)
                self.assertTrue(np.allclose(actual,expected,rtol=0,atol=1e-7))
                self.assertTrue(np.allclose(w,weights,rtol=0,atol=1e-7));self.assertEqual(m,masks)
                self.assertEqual(before,ops.state_digest(net))
                self.assertTrue(all(p.grad is None for p in net.parameters()))

    def test_edge_edits_use_vocab_and_do_not_modify_original(self):
        g=graph(1);g.edge_type=torch.tensor([7,3,7]);original=g.clone()
        vocab={'DEP':3,'SEC_AFFECTS':7}
        for mode,count in [('remove_security_edges',1),('remove_all_edges',0),('zero_edge_features',3)]:
            result=ops.changed_graph(g,mode,vocab)
            self.assertEqual(result.num_edges,count)
            self.assertTrue(torch.equal(result.x,original.x))
            if mode=='zero_edge_features':self.assertEqual(float(result.edge_features.abs().sum()),0.)
            for name in ('x','edge_type','edge_features','edge_index'):
                self.assertTrue(torch.equal(getattr(g,name),getattr(original,name)))

    def test_edge_affected_flags_and_unchanged_structured_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            batches,v,s=setup_batches(d);net=batching.make_model(RESOURCES,len(batches.state['feature_names']),v,s).eval()
            original,mask,numeric,affected=ops.encode_sources(net,batches,['A','B'],v)
            self.assertTrue(all(all(values) for values in affected.values()))
            for mode in ops.EDGE_MODES:
                altered,m,n,_=ops.encode_sources(net,batches,['A','B'],v,mode)
                self.assertTrue(torch.equal(m,mask));self.assertTrue(torch.equal(n,numeric))
                self.assertEqual(altered.shape,original.shape)

    def test_missing_source_removal_is_noop_and_all_text_is_masked(self):
        with tempfile.TemporaryDirectory() as d:
            batches,v,s=setup_batches(d);net=batching.make_model(RESOURCES,len(batches.state['feature_names']),v,s).eval()
            sources,masks,numeric,_=ops.encode_sources(net,batches,['A','B'],v)
            p,_,_=ops.predict_fusion(net,sources,masks,numeric,2)
            changed,present,affected=ops.remove_sources(sources,masks,[3])
            q,_,_=ops.predict_fusion(net,changed,present,numeric,2)
            self.assertEqual(affected,[False,False]);self.assertTrue(np.array_equal(p,q))
            changed,present,affected=ops.remove_sources(sources,masks,[0,1,2,3])
            _,weights,_=ops.predict_fusion(net,changed,present,numeric,2)
            self.assertEqual(affected,[True,True]);self.assertEqual(float(np.sum(weights)),0.)
            self.assertTrue(masks[:,0].all())

    def test_structured_removal_drops_observation_flags(self):
        with tempfile.TemporaryDirectory() as d:
            batches,v,_=setup_batches(d)
            original=batches.plan(['A','B'],v)['structured']
            cvss_off=ops.structured_without(batches,['A','B'],v,['cvss'])
            refs_off=ops.structured_without(batches,['A','B'],v,['reference_count'])
            self.assertEqual(float(cvss_off[:,:-2].abs().sum()),0.)
            self.assertTrue(torch.equal(cvss_off[:,-2:],original[:,-2:]))
            self.assertEqual(float(refs_off[:,-2:].abs().sum()),0.)
            self.assertTrue(torch.equal(refs_off[:,:-2],original[:,:-2]))

    def test_permutation_masks_determinism_singletons_and_identical_text(self):
        ids=['A','B','C','D','E','F'];masks=[[1,0,0,0]]*3+[[1,1,0,0]]*2+[[0,0,0,0]]
        fingerprints=['x','y','z','same','same','empty']
        a,changed=ops.mask_matched_permutation(ids,masks,fingerprints,17)
        b,again=ops.mask_matched_permutation(ids,masks,fingerprints,17)
        self.assertTrue(np.array_equal(a,b));self.assertTrue(np.array_equal(changed,again))
        self.assertEqual(changed.tolist(),[True,True,True,False,False,False])
        self.assertEqual(a[5],5);self.assertEqual(sorted(a.tolist()),list(range(6)))
        for i,j in enumerate(a):self.assertEqual(masks[i],masks[j])

    def test_response_has_separate_affected_denominator(self):
        result=ops.response([0,0],[.1,.1],[.1,.3],[False,True],1e-7)
        self.assertEqual(result['all_cves']['n'],2);self.assertEqual(result['affected_cves']['n'],1)
        self.assertAlmostEqual(result['all_cves']['mae_difference'],.1)
        self.assertAlmostEqual(result['affected_cves']['mae_difference'],.2)
        self.assertEqual(result['unaffected_cves']['changed_predictions_over_tolerance'],0)
        empty=ops.response([0],[.1],[.1],[False],1e-7)
        self.assertIsNone(empty['affected_cves']['mae_difference'])

    def test_evidence_unicode_cve_spans_and_real_excerpt(self):
        text='CVE-2025-0108 references CVE\u20112024\u201138475. Unable to access a page. \ufffd'
        found=evidence.mentions(text)
        self.assertEqual([m['cve'] for m in found],['CVE-2025-0108','CVE-2024-38475'])
        result=evidence.text_inventory('CVE-2025-0108','smp',text)
        self.assertEqual(result['assigned_cve_mentions'],1);self.assertEqual(result['other_cve_mentions'],1)
        self.assertEqual(result['replacement_characters'],1)
        sample=result['other_cve_examples'][0]
        self.assertEqual(sample['text'],text[sample['start']:sample['end']])
        self.assertEqual(result['review_cues']['possible_access_failure_language']['matches'],1)
        self.assertNotIn('irrelevant',result)

    def test_smoke_selection_is_input_only_and_rejects_other_roles(self):
        with tempfile.TemporaryDirectory() as d:
            batches,_,_=setup_batches(d)
            config={'smoke_max_chunks_per_cve':8,'smoke_cves':1}
            self.assertEqual(select_ids(batches,['C'],config,'smoke'),['C'])
            with self.assertRaises(ValueError):select_ids(batches,['A'],config,'smoke')
            batches.rows['C']['split']='test'
            with self.assertRaises(ValueError):select_ids(batches,['C'],config,'full')


if __name__=='__main__':unittest.main()
