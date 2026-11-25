"""Synthetic invariants; no full fits, external encoder or test cohort."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
for p in ('01_tpg/01_core','01_tpg/10_extensions','02_prediction/01_package','01_tpg/02_examples/01_scripts'):
    sys.path.insert(0,str(ROOT/p))
import torch
from torch.nn import functional as F
from epss.combined_training_v1 import data,model,runner,reporting
from epss.edge_aware_layers import EdgeTypeGNNLayer


def fixture():
    metadata={'vocabulary':{'node_types':{'TOKEN':2,'ENTITY':0,'DOCUMENT':1},
        'edge_types':{'SEC_HAS_VERSION':0,'DEP':1},'num_node_types':3,'num_edge_types':2},
        'subtype_vocabulary':{'SEVERITY':0,'VERSION':1,'NOT_ENTITY':2},
        'edge_shape':[3,88],'edge_feature_schema':{'dimension':88,'named_flags':{'concatenation_boundary':67}}}
    resources=data.contract(metadata,{'trainable':False,'dtype':'float32'},['cvss_score_standardized'])
    t={'vocabulary':metadata['vocabulary'],'subtype_vocabulary':metadata['subtype_vocabulary'],
       'edge_feature_schema':metadata['edge_feature_schema'],'node_ids':torch.tensor([91,4,88]),
       'node_type':torch.tensor([2,0,1]),'subtype':torch.tensor([2,1,2]),
       'edge_index':torch.tensor([[0,1,2],[1,2,0]]),'edge_type':torch.tensor([0,1,0]),
       'edge_features':torch.randn(3,88),'node_embeddings':torch.randn(3,768),
       'token_mask':torch.tensor([True,False,False]),'embedding_mask':torch.ones(3,dtype=torch.bool),
       'target_cve':torch.tensor([False,True,False])}
    g={k:t[k] for k in ('edge_index','edge_type','edge_features','node_type','subtype','token_mask','embedding_mask')}
    g['x']=torch.cat((F.one_hot(t['node_type'],3).float(),t['node_embeddings'],t['embedding_mask'][:,None].float(),t['target_cve'][:,None].float()),1)
    return resources,t,g


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(17)
        self.p=runner.settings({'hidden':8,'layers':2,'threads':1,'edge_chunk_size':2})

    def test_nonstandard_vocabulary_ids_are_consumed(self):
        r,t,g=fixture();data.validate_tensors(t,r)
        net=model.CombinedRegressor(r,self.p,'tpg')
        self.assertEqual(net.node_type_vocab['TOKEN'],2)
        self.assertEqual(net.layers[0].edge_embed.num_embeddings,2)
        self.assertEqual(net.layers[0].edge_semantics.in_features,88)
        self.assertEqual(net.subtype_vocab['NOT_ENTITY'],2)

    def test_unknown_or_missing_edges_rejected(self):
        r,t,g=fixture();t['edge_type'][0]=2
        with self.assertRaisesRegex(ValueError,'edge_type'):data.validate_tensors(t,r)
        layer=model.TypedEdgeLayer(8,r['edge_type_vocab'],88,2)
        with self.assertRaisesRegex(ValueError,'mandatory'):layer(torch.randn(3,8),g['edge_index'],None,g['edge_features'])

    def test_swapped_edge_vocab_rejected(self):
        r,t,g=fixture();t['vocabulary']['edge_types']={'SEC_HAS_VERSION':1,'DEP':0}
        with self.assertRaisesRegex(ValueError,'vocabulary'):data.validate_tensors(t,r)

    def test_swapped_node_vocab_rejected(self):
        r,t,g=fixture();t['vocabulary']['node_types']={'TOKEN':0,'ENTITY':2,'DOCUMENT':1}
        with self.assertRaisesRegex(ValueError,'vocabulary'):data.validate_tensors(t,r)

    def test_old_width_and_bad_endpoints_rejected(self):
        r,t,g=fixture();t['edge_features']=t['edge_features'][:,:64]
        with self.assertRaisesRegex(ValueError,'dimension'):data.validate_tensors(t,r)
        r,t,g=fixture();t['edge_index'][0,0]=3
        with self.assertRaisesRegex(ValueError,'endpoints'):data.validate_tensors(t,r)

    def test_invalid_subtype_rejected(self):
        r,t,g=fixture();t['subtype'][0]=1
        with self.assertRaisesRegex(ValueError,'Non-entity'):data.validate_tensors(t,r)

    def test_edge_types_change_messages(self):
        r,t,g=fixture();layer=model.TypedEdgeLayer(8,r['edge_type_vocab'],88,2).eval();h=torch.randn(3,8)
        a=layer(h,g['edge_index'],g['edge_type'],g['edge_features'])
        b=layer(h,g['edge_index'],1-g['edge_type'],g['edge_features'])
        self.assertGreater(float((a-b).abs().max()),1e-6)

    def test_all_edge_property_columns_have_gradient_path(self):
        r,t,g=fixture();layer=model.TypedEdgeLayer(8,r['edge_type_vocab'],88,2)
        f=g['edge_features'].clone().requires_grad_(True)
        y=layer(torch.randn(3,8),g['edge_index'],g['edge_type'],f)
        (y*torch.randn_like(y)).sum().backward()
        self.assertTrue(bool((f.grad.abs().sum(0)>0).all()))
        self.assertTrue(bool((layer.edge_embed.weight.grad.abs().sum(1)>0).all()))

    def test_chunked_forward_and_gradients_match_existing_layer(self):
        r,t,g=fixture();a=model.TypedEdgeLayer(8,r['edge_type_vocab'],88,1);b=deepcopy(a)
        x=torch.randn(3,8,requires_grad=True);z=x.detach().clone().requires_grad_(True)
        left=a(x,g['edge_index'],g['edge_type'],g['edge_features'])
        right=EdgeTypeGNNLayer.forward(b,z,g['edge_index'],g['edge_type'],g['edge_features'])
        torch.testing.assert_close(left,right,atol=1e-6,rtol=1e-6)
        weight=torch.randn_like(left);(left*weight).sum().backward();(right*weight).sum().backward()
        torch.testing.assert_close(x.grad,z.grad,atol=2e-6,rtol=2e-5)
        for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa.grad,pb.grad,atol=2e-6,rtol=2e-5)

    def test_edgeless_graph_and_isolated_nodes(self):
        r,t,g=fixture();layer=model.TypedEdgeLayer(8,r['edge_type_vocab'],88,2)
        out=layer(torch.randn(3,8),torch.empty(2,0,dtype=torch.long),torch.empty(0,dtype=torch.long),torch.empty(0,88))
        self.assertEqual(out.shape,(3,8));self.assertTrue(torch.isfinite(out).all())

    def test_node_and_subtype_projection_receive_gradients(self):
        r,t,g=fixture();net=model.CombinedRegressor(r,self.p,'tpg')
        inp={'graph':g,'source_mask':torch.tensor([1.,0,0,0]),'structured':torch.tensor([.2])}
        net(inp).backward()
        self.assertGreater(float(net.node_project[0].weight.grad[:,:3].abs().sum()),0)
        self.assertGreater(float(net.subtype_project.weight.grad[:,1].abs().sum()),0)
        self.assertEqual(float(net.subtype_project.weight.grad[:,2].abs().sum()),0)

    def test_checkpoint_carries_vocab_and_refuses_remap(self):
        r,t,g=fixture();net=model.CombinedRegressor(r,self.p,'tpg');state=deepcopy(net.state_dict())
        other=deepcopy(r);other['edge_type_vocab']={'SEC_HAS_VERSION':1,'DEP':0}
        with self.assertRaisesRegex(ValueError,'Checkpoint vocabulary'):model.CombinedRegressor(other,self.p,'tpg').load_state_dict(state)

    def test_missing_view_and_extra_target_guard(self):
        r,t,g=fixture();net=model.CombinedRegressor(r,self.p,'tpg')
        inp={'graph':None,'source_mask':torch.zeros(4),'structured':torch.tensor([.2])}
        self.assertTrue(torch.isfinite(net(inp)))
        with self.assertRaisesRegex(ValueError,'targets'):net({**inp,'target':.9})
        with self.assertRaisesRegex(ValueError,'missingness'):net({**inp,'source_mask':torch.ones(4)})

    def test_masked_view_parity(self):
        r,t,g=fixture();net=model.CombinedRegressor(r,self.p,'tpg').eval()
        a={'graph':g,'source_mask':torch.tensor([1.,0,0,0]),'structured':torch.tensor([.2])}
        self.assertTrue(torch.equal(net(a),net(deepcopy(a))))

    def test_structured_only_and_pooled_controls(self):
        r,t,g=fixture()
        a=runner.initialize(r,self.p,'pooled',17);b=runner.initialize(r,self.p,'tpg',17)
        for k,v in a.structured.state_dict().items():self.assertTrue(torch.equal(v,b.structured.state_dict()[k]))
        for k,v in a.head.state_dict().items():self.assertTrue(torch.equal(v,b.head.state_dict()[k]))
        inp={'graph':g,'source_mask':torch.tensor([1.,0,0,0]),'structured':torch.tensor([.2])}
        p=a(inp);g['edge_type']=1-g['edge_type'];self.assertTrue(torch.equal(p,a(inp)))

    def test_hyperparameters_and_layer_count(self):
        r,t,g=fixture();p=runner.settings({'layers':3,'hidden':16,'learning_rate':.002,'seeds':[29]})
        self.assertEqual(len(model.CombinedRegressor(r,p,'tpg').layers),3)
        for bad in ({'layers':0},{'learning_rate':float('nan')},{'seeds':[17,17]}):
            with self.assertRaises(ValueError):runner.settings(bad)

    def test_target_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'labels.json';c=data.c
            for row in ({'target':None,'target_day':60},{'target':.2,'target_day':90},{'target':2.,'target_day':60}):
                data.write(path,{'CVE-X':row})
                with self.assertRaisesRegex(ValueError,'target'):data.load_labels(path,{'CVE-X':{'split':'train'}})

    def test_metrics_and_canonical_csv(self):
        self.assertIsNone(reporting.metrics([],[])['mae'])
        self.assertIsNone(reporting.metrics([1,1],[.5,.5])['spearman'])
        rows=[{'cve':'B','x':1},{'cve':'A','x':2}]
        self.assertEqual(reporting.csv_text(rows),reporting.csv_text(rows[::-1]))
        self.assertNotIn('\r',reporting.csv_text(rows))

    def smoke_identities(self):
        resources,_,_=fixture()
        resources['encoder_identity'].update(threads=4,revision='fixed-revision',device='cpu',
            files={'model.safetensors':'fixed-weight-hash'},window_tokens=1024,pooling='fixed-pooling')
        smoke={'schema':'combined_training_v1','task':'CT01','dataset':'previous','scale':'smoke',
               'settings':deepcopy(self.p),'source_pins':{'runner.py':'fixed-code-hash'},
               'runtime':{'torch':str(torch.__version__)},'test_access':False,'contract':resources}
        full=deepcopy(smoke);full['scale']='full';full['contract']['encoder_identity']['threads']=8
        return smoke,full

    def test_smoke_gate_allows_only_encoder_generation_thread_difference(self):
        smoke,full=self.smoke_identities();original=deepcopy((smoke,full))
        result=runner.smoke_compatibility(smoke,full)
        self.assertEqual(result['allowed_execution_difference']['encoder_identity.threads'],{'smoke':4,'full':8})
        self.assertEqual((smoke,full),original)
        full['contract']['encoder_identity']['threads']=4
        runner.smoke_compatibility(smoke,full)

    def test_smoke_gate_rejects_semantic_or_training_changes_with_field_names(self):
        changes=[('contract','node_type_vocab','TOKEN'),('contract','edge_type_vocab','DEP'),
                 ('contract','subtype_vocab','SEVERITY'),('contract','edge_feature_dim'),
                 ('contract','structured_names'),('contract','encoder_identity','revision'),
                 ('contract','encoder_identity','files','model.safetensors'),
                 ('contract','encoder_identity','window_tokens'),('contract','encoder_identity','pooling'),
                 ('contract','encoder_identity','dtype'),('contract','encoder_identity','device'),
                 ('settings','threads'),('settings','layers'),('source_pins','runner.py'),('dataset',)]
        for path in changes:
            smoke,full=self.smoke_identities();obj=full
            for key in path[:-1]:obj=obj[key]
            obj[path[-1]]='changed'
            with self.subTest(path=path),self.assertRaises(ValueError) as caught:
                runner.smoke_compatibility(smoke,full)
            self.assertIn('.'.join(path),str(caught.exception))

    def test_smoke_gate_rejects_missing_or_invalid_encoder_threads(self):
        for value in (None,0,True,'8'):
            smoke,full=self.smoke_identities();full['contract']['encoder_identity']['threads']=value
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'thread count'):
                runner.smoke_compatibility(smoke,full)
        smoke,full=self.smoke_identities();del full['contract']['encoder_identity']['threads']
        with self.assertRaisesRegex(ValueError,'thread count'):runner.smoke_compatibility(smoke,full)

    def test_checkpoint_identity_remains_exact_despite_cross_scale_gate(self):
        smoke,full=self.smoke_identities()
        runner.smoke_compatibility(smoke,full)
        net=model.CombinedRegressor(smoke['contract'],self.p,'tpg')
        other=model.CombinedRegressor(full['contract'],self.p,'tpg')
        with self.assertRaisesRegex(ValueError,'Checkpoint vocabulary'):
            other.load_state_dict(net.state_dict())

    def test_ready_stage_qualifies_before_any_model_fitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx={'contract':{},'provenance':{}}
            gate={'verified_neural_fits':39,'test_access':False}
            with patch.object(data,'context',return_value=ctx),patch.object(data,'GraphStore'), \
                 patch.object(runner,'destination',side_effect=lambda d,n,s:Path(tmp)/s), \
                 patch.object(runner,'preflight'),patch.object(runner,'smoke_gate',return_value=gate) as check, \
                 patch.object(runner,'parity') as parity,patch.object(runner,'controls') as controls, \
                 patch.object(runner,'fit') as fit:
                runner.run('previous','full','gate_test',self.p,'ready')
                self.assertEqual(check.call_count,1)
                parity.assert_not_called();controls.assert_not_called();fit.assert_not_called()
                runner.publication.check_lock(Path(tmp)/'full/smoke_gate')
                self.assertEqual(data.c.read(Path(tmp)/'full/smoke_gate/findings.json'),gate)
                check.side_effect=ValueError('mismatched smoke')
                with self.assertRaisesRegex(ValueError,'mismatched smoke'):
                    runner.run('previous','full','gate_test',self.p,'ready')
                fit.assert_not_called()

    def test_only_best_checkpoint_and_resume(self):
        r,_,_=fixture();p={**self.p,'epochs':3,'seeds':[17],'patience':5}
        ctx={'ids':{'train':['a','b'],'validation':['c','d']},'audit':{},
             'labels':{k:{'value':.7,'date':'2024-01-01'} for k in 'abcd'}}
        class Store:
            def inputs(self,cid,view,device):
                return {'graph':None,'source_mask':torch.zeros(4),'structured':torch.tensor([.2])}
        ident={'settings':p,'scale':'full','contract':r,'source_pins':{},'fit':'structured/S/seed_17'}
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'clean';b=Path(tmp)/'resumed'
            expected=runner.fit(a,ctx,Store(),ident,'structured','S',17,False,False)
            original=runner.train_epoch
            def stop(*args,**kw):
                if args[7]==3:raise InterruptedError('simulated interruption')
                return original(*args,**kw)
            with patch.object(runner,'train_epoch',stop):
                with self.assertRaises(InterruptedError):runner.fit(b,ctx,Store(),ident,'structured','S',17,False,False)
            actual=runner.fit(b,ctx,Store(),ident,'structured','S',17,False,True)
            self.assertEqual(expected,actual)
            runner.fit(b,ctx,Store(),ident,'structured','S',17,True,False)
            self.assertEqual([f.name for f in b.glob('*.pt')],['best_checkpoint.pt'])

    def test_tied_or_worse_epochs_do_not_replace_best_checkpoint(self):
        r,_,_=fixture();p={**self.p,'epochs':3,'seeds':[17],'patience':5}
        ident={'settings':p,'scale':'full','contract':r,'source_pins':{},'fit':'structured/S/seed_17'}
        evaluations=[{'metrics':{'all':{'mae':score}},'predictions':[]}
                     for score in (.1,.1,.2,.1)]
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'fit'
            with patch.object(runner,'train_epoch',return_value=(.2,0.)), \
                 patch.object(runner,'evaluate',side_effect=evaluations), \
                 patch.object(runner,'save_torch',wraps=runner.save_torch) as saves:
                runner.fit(directory,{},None,ident,'structured','S',17,False,False)
            self.assertEqual(saves.call_count,1)
            cp=torch.load(directory/'best_checkpoint.pt',weights_only=True)
            self.assertEqual(cp['best_epoch'],1)
            self.assertEqual(cp['best_mae'],.1)
            self.assertEqual(len(data.c.read(directory/'history.json')),3)
            self.assertEqual([f.name for f in directory.glob('*.pt')],['best_checkpoint.pt'])


if __name__=='__main__':unittest.main()
