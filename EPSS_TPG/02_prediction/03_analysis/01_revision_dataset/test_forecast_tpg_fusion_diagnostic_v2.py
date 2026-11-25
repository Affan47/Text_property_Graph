"""DQ09 synthetic regression tests; no real dataset training or test scoring."""
from contextlib import redirect_stdout, redirect_stderr
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import diagnose_forecast_tpg_fusion_v2 as dq
from test_forecast_tpg_review import with_graph_stats

task,cache,torch = dq.task,dq.cache,dq.torch


class FusionDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture_model(self):
        m,g,d=with_graph_stats()
        net=task.TPGRegressor(**task.model_spec('tpg_epss',m,g,task.config_for('full'))).eval()
        return m,g,d,net

    def test_group_permutations_deterministic_bijective(self):
        ids=[str(i) for i in range(100)]
        groups={c:str(int(c)//50) for c in ids}
        for seed in dq.PROTOCOL['permutation_seeds']:
            donors=dq.different_group_permutation(ids,groups,seed)
            self.assertEqual(donors,dq.different_group_permutation(list(reversed(ids)),groups,seed))
            self.assertEqual(set(donors),set(ids));self.assertEqual(set(donors.values()),set(ids))
            self.assertTrue(all(groups[c]!=groups[d] for c,d in donors.items()))
        with self.assertRaisesRegex(ValueError,'No different-group'):
            dq.different_group_permutation(['a','b','c'],{'a':'x','b':'x','c':'y'},101)

    def test_score_probe_preserves_masks_ages_padding_and_clips(self):
        m,g,d,net=self.fixture_model();ids=m['ids']['validation']
        batch=task.make_batch(ids,g,d['inputs.json']);x=batch.epss_sequence.clone()
        batch.epss_sequence[:,0,0]=1
        changed=dq.modify_batch(batch,'observed_plus',[])
        self.assertTrue(torch.equal(changed.epss_sequence[:,:,1:],x[:,:,1:]))
        self.assertTrue(torch.equal(changed.epss_sequence[:,:,0][x[:,:,1]==0],x[:,:,0][x[:,:,1]==0]))
        self.assertTrue((changed.epss_sequence[:,0,0]==1).all())

    def test_edge_removal_uses_vocab_not_hardcoded_indices(self):
        m,g,d,net=self.fixture_model();ids=m['ids']['validation']
        batch=task.make_batch(ids,g,d['inputs.json'])
        index=m['type_vocabulary']['edge_types']['SEC_AFFECTS']
        out=dq.modify_batch(batch,'remove_security_edges',[index])
        self.assertEqual(out.edge_type.numel(),2)
        self.assertFalse((out.edge_type==index).any())
        out=dq.modify_batch(out,'remove_all_edges',[index])
        self.assertEqual(tuple(out.edge_index.shape),(2,0));self.assertEqual(out.edge_features.shape[0],0)

    def test_probes_reproduce_without_mutating_inputs_or_weights(self):
        m,g,d,net=self.fixture_model();ids=m['ids']['validation']
        before_g=copy.deepcopy(g);before_d=copy.deepcopy(d)
        state={k:v.clone() for k,v in net.state_dict().items()}
        original,reps=dq.predict_variant(net,ids,g,d['inputs.json'],'original',[])
        self.assertEqual(original,task.infer(net,ids,g,d['inputs.json'],8,progress=False))
        self.assertEqual(tuple(reps.shape),(2,48))
        groups={c:d['split_manifest.json'][c]['group_id'] for c in ids}
        donors=dq.different_group_permutation(ids,groups,101)
        for variant in dq.PROTOCOL['variants']:
            a,_=dq.predict_variant(net,ids,g,d['inputs.json'],variant,[],donors if variant.startswith('swap_') else None)
            b,_=dq.predict_variant(net,ids,g,d['inputs.json'],variant,[],donors if variant.startswith('swap_') else None)
            self.assertEqual(a,b)
            if variant=='remove_security_edges':self.assertEqual(original,a)
        self.assertEqual(d,before_d)
        for c in g:
            for k,v in g[c].items():
                self.assertTrue(torch.equal(v,before_g[c][k]) if isinstance(v,torch.Tensor) else v==before_g[c][k])
        self.assertTrue(all(torch.equal(v,state[k]) for k,v in net.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in net.parameters()))
        self.assertEqual(len(net.head._forward_pre_hooks),0)

    def test_zero_representation_hooks_remove_only_the_intended_branch(self):
        m,g,d,net=self.fixture_model();ids=m['ids']['validation']
        _,rep=dq.predict_variant(net,ids,g,d['inputs.json'],'original',[])
        for variant,part in [('zero_epss_branch',slice(None,16)),('zero_graph_branch',slice(16,None))]:
            masked=rep.clone();masked[:,part]=0
            expected=torch.sigmoid(net.head(masked)).squeeze(-1).tolist()
            actual,_=dq.predict_variant(net,ids,g,d['inputs.json'],variant,[])
            self.assertEqual(actual,expected)

    def test_empty_security_subset_is_unknown_not_zero(self):
        row=dq.response_summary([],'with_security_edges')
        self.assertEqual(row['cves'],0)
        self.assertIsNone(row['mean_absolute_prediction_change'])
        self.assertIsNone(row['recipient_target_probe_mae'])

    def test_no_test_ids_or_targets_in_forward(self):
        m,g,d,net=self.fixture_model();ids=m['ids']['validation']
        batch=task.make_batch(ids,g,d['inputs.json'])
        self.assertFalse(any(k in batch for k in ('y','cve','target_date','group_id')))
        d['split_manifest.json'][ids[0]]['split']='test'
        identity={'selected_ids':m['ids']}
        with self.assertRaisesRegex(ValueError,'Only validation'):
            dq.diagnose(m,g,d,[{'seed':seed} for seed in dq.PROTOCOL['model_seeds']],identity)

    def test_reparsed_security_evidence_roundtrip_and_tamper(self):
        from tpg.pipeline import SecurityPipeline
        pipeline=SecurityPipeline(include_security_relations=True)
        text='CVE-2024-1000 may not affect Apache 2.4.51.'
        graph=pipeline.run(text,doc_id='synthetic')
        for n in graph.nodes():n.properties.extra['source_text_type']='description'
        for e in graph.edges():e.properties.extra['source_text_type']='description'
        p=pipeline.export_pyg(graph,embedding_dim=4,use_security_edge_types=True)
        cached={k:torch.tensor(p[k],dtype=torch.long if k in ('edge_index','edge_type') else torch.float32)
                for k in ('x','edge_index','edge_type','edge_features')}
        cached['description_sha256']=cache.text_digest(text)
        manifest={'embedding_dim':4,'type_vocabulary':p['type_vocabulary']}
        row,edges=dq.inspect_one(pipeline,'synthetic',text,cached,manifest,'validation')
        self.assertGreater(row['typed_security_edges'],0)
        self.assertTrue(any(e['relation']=='SEC_AFFECTS' and e['evidence']['polarity']=='negated' for e in edges))
        self.assertTrue(all(e['review_status']=='not_reviewed' for e in edges))
        cached['edge_features'][0,0]+=1
        with self.assertRaisesRegex(ValueError,'differs beyond'):
            dq.inspect_one(pipeline,'synthetic',text,cached,manifest,'validation')

    def test_end_to_end_synthetic_outputs_and_artifact_preservation(self):
        m,g,d=with_graph_stats();m['scale']='full'
        config=task.config_for('full');config.update(epochs=1,patience=1)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'raw';source.mkdir();source=source/'records.json';cache.write_json(source,[])
            prepared=root/'prepared';prepared.mkdir();graph_dir=root/'graphs';graph_dir.mkdir()
            m.update(source_pins={str(source):cache.sha256(source)},prepared_directory=str(prepared))
            d['preparation_summary.json']={'source_path':str(source)}
            cache.write_json(graph_dir/'manifest.json',m)
            cache.write_json(graph_dir/'cache_lock.json',{'files':{'manifest.json':cache.sha256(graph_dir/'manifest.json')}})
            coverage=[{'split':s,'cve':c,'typed_security_edges':1,'has_textual_cve_entity':True,'structure_alignment':{'mode':'exact'}} for s,ids in m['ids'].items() for c in ids]
            edges=[{'split':r['split'],'cve':r['cve'],'edge_id':1,'relation':'SEC_AFFECTS',
                    'also_present_as_generic_entity_relation':True,'source_text':'synthetic CVE',
                    'target_text':'synthetic product','evidence':{'text':'Synthetic evidence only.'}} for r in coverage]
            with patch.object(cache,'load_cache',return_value=(m,g,d)),patch.object(task,'config_for',return_value=config), \
                 patch.object(dq,'inspect_security',return_value=(coverage,edges,[])),redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                original=root/'original';task.run(graph_dir,original)
                hashes={str(p):cache.sha256(p) for p in original.rglob('*') if p.is_file()}
                with patch.object(task,'fit',side_effect=AssertionError('Diagnostic attempted training')):
                    dq.run(original,root/'check','smoke',check_only=True)
                    self.assertFalse((root/'check').exists())
                    dq.run(original,root/'diagnostic','smoke')
                result=cache.read_json(root/'diagnostic/analysis.json')
                self.assertFalse(result['training_performed']);self.assertFalse(result['test_scoring_performed'])
                lock=cache.read_json(root/'diagnostic/analysis_lock.json')
                self.assertTrue(all(cache.sha256(root/'diagnostic'/name)==digest for name,digest in lock['files'].items()))
                self.assertEqual(hashes,{str(p):cache.sha256(p) for p in original.rglob('*') if p.is_file()})
                with self.assertRaisesRegex(ValueError,'exists'):
                    dq.run(original,root/'diagnostic','smoke')


if __name__=='__main__':
    unittest.main()

