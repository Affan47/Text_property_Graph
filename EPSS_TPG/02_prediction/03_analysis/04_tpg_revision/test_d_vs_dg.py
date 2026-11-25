"""Focused source isolation, parity, deterministic replay and diagnostic contracts."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Batch, Data

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.tpg_revision import dg_model_v1 as model, dg_training_v1 as train, dg_diagnostic_v1 as diagnostic


class DGTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.r = {'embedding_dim': 2, 'type_vocabulary': {'num_node_types': 2,
                  'node_types': {'TOKEN': 0, 'ENTITY': 1}, 'edge_types': {'DEP': 0}}}
        self.p = {'hidden': 4, 'layers': 1, 'learning_rate': .001, 'weight_decay': .0001,
                  'epochs': 2, 'patience': 2, 'batch_size': 2}
        self.sample = Data(x=torch.tensor([[1.,0.,.3,.4,1.],[0.,1.,.2,.7,1.]]),
                           edge_index=torch.tensor([[0,1],[1,0]]), edge_type=torch.zeros(2,dtype=torch.long),
                           edge_features=torch.zeros(2,64), token_mask=torch.tensor([True,False]), subtype=torch.tensor([0,4]))

    def plan(self, g=True):
        return {'paths': ['a','b','c'] if g else ['a','c'],
                'chunk_document': torch.arange(3 if g else 2),
                'document_source': torch.tensor([0,1,2] if g else [0,2]),
                'source_mask': torch.tensor([[1,int(g)],[1,0]],dtype=torch.bool),
                'structured': torch.tensor([[.1,.2],[.7,.8]])}

    def packets(self):
        sample = self.sample
        class Packets:
            def packets(self, plan, variant):
                for i, _ in enumerate(plan['paths']):
                    yield i, Batch.from_data_list([sample])
        return Packets()

    def test_exact_initialization_and_masked_optimizer_steps(self):
        for name in ('pooled_text','v5_node_mean'):
            a = model.make(self.r,2,self.p,17,name,False)
            b = model.make(self.r,2,self.p,17,name)
            train.shared_equal(a,b)
            pa = self.plan(False); pb = train.plain_plan(pa)
            optimizers = [torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001) for m in (a,b)]
            for _ in range(3):
                for m, o, plan in zip((a,b),optimizers,(pb,pa)):
                    o.zero_grad(set_to_none=True)
                    model.base.replay_backward(m,self.packets(),plan,model.representation(name),torch.tensor([0.,1.]),{0:1.,1:1.})
                train.shared_equal(a,b,True)
                for o in optimizers:o.step()
                train.shared_equal(a,b)

    def test_mixed_row_ignores_masked_g(self):
        m = model.make(self.r,2,self.p,17,'pooled_text')
        values = torch.randn(2,2,4); mask = torch.tensor([[1,1],[1,0]],dtype=torch.bool)
        a,w = m.from_sources(values,mask,self.plan()['structured'])
        changed = values.clone(); changed[1,1] = 999
        b,_ = m.from_sources(changed,mask,self.plan()['structured'])
        self.assertTrue(torch.equal(a,b)); self.assertEqual(float(w[1,1]),0.)

    def test_replay_matches_direct_gradients(self):
        for name in ('pooled_text','v5_node_mean'):
            a = model.make(self.r,2,self.p,17,name); b = deepcopy(a)
            plan = self.plan(); y = torch.tensor([0.,1.]); rep=model.representation(name)
            z = model.base.representations(a,self.packets(),plan,rep)
            model.base.loss_for(a.fuse(z,plan),y,{0:1.,1:1.}).backward()
            model.base.replay_backward(b,self.packets(),plan,rep,y,{0:1.,1:1.})
            for x,z in zip(a.parameters(),b.parameters()):
                if x.grad is None:self.assertIsNone(z.grad)
                else:torch.testing.assert_close(x.grad,z.grad,atol=1e-7,rtol=1e-5)

    def test_subtype_feature_disabled(self):
        m=model.make(self.r,2,self.p,17,'v5_node_mean')
        graph=Batch.from_data_list([self.sample]); altered=graph.clone(); altered.subtype[:]=2
        self.assertTrue(torch.equal(m.encode(graph,'security_tpg'),m.encode(altered,'security_tpg')))
        self.assertFalse(m.use_subtypes)

    def test_labels_rejected_by_fuse(self):
        m=model.make(self.r,2,self.p,17,'pooled_text')
        with self.assertRaises(ValueError):m.fuse(torch.randn(3,4),{**self.plan(),'target':torch.ones(2)})

    def test_bounded_fit_reload(self):
        packets=self.packets(); parent=self
        class Batches:
            def plan(self,ids):return parent.plan()
            def packets(self,plan,variant):return packets.packets(plan,variant)
        protocol={'settings':self.p,'ids':{'train':['a','b'],'validation':['c','d']},'class_weights':{0:1.,1:1.}}
        targets={'a':0,'b':1,'c':0,'d':1}
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('pooled_text','v5_node_mean'):
                path=Path(tmp)/name;path.mkdir()
                rows,metrics=train.fit(path,name,17,Batches(),targets,self.r,2,protocol)
                self.assertEqual((rows,metrics),train.verify_fit(path,name,17,Batches(),targets,self.r,2,protocol))
                with self.assertRaises(ValueError):train.verify_fit(path,name,29,Batches(),targets,self.r,2,protocol)

    def test_permutations_fixed_label_blind_no_self_donors(self):
        ids=['a','b','c','d','e']
        result=diagnostic.permutations(ids,[6029,6030])
        self.assertEqual(result,diagnostic.permutations(ids,[6029,6030]))
        for mapping in result.values():
            self.assertEqual(set(mapping),set(ids));self.assertEqual(set(mapping.values()),set(ids))
            self.assertTrue(all(k!=v for k,v in mapping.items()))

    def test_paired_bootstrap_identity_and_undefined_not_zero(self):
        p=np.asarray([[.1,.2,.6,.8],[.2,.1,.7,.9]])
        values=diagnostic.paired_interval([0,0,1,1],p,p,10,6029)
        self.assertTrue(all(v['lower_2_5']==v['upper_97_5']==0 for v in values.values()))
        metrics=train.dev.metric([0,1],[.1,.2])
        self.assertIsNone(diagnostic.mean_metrics([metrics,metrics])['precision'])

    def test_source_planner_excludes_g_when_masked_and_rejects_n(self):
        b=object.__new__(model.Batches)
        b.rows={'a':{'source_mask':[1,1,0,0],'structured':[.1,.2]}}
        b.index={'a':{'documents':[{'source_text_type':s,'chunks':[{'has_graph':True,'tensor':s}]} for s in model.SOURCES]}}
        self.assertEqual(b.plan(['a'])['paths'],list(model.SOURCES))
        self.assertEqual(b.plan(['a'],True)['paths'],['description'])
        b.index['a']['documents'].append({'source_text_type':'smp','chunks':[]})
        with self.assertRaises(ValueError):b.plan(['a'])

    def test_parity_publication_and_reproduction(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol={'pins':{},'settings':{}}
            with patch.object(train.cache,'OUTPUT',Path(tmp)), patch.object(train.cache,'load',return_value=(None,None,None,None,protocol)), \
                    patch.object(train,'parity_checks',return_value={'passed':True}):
                train.parity('smoke')
                train.parity('smoke',True)


if __name__ == '__main__':unittest.main()
