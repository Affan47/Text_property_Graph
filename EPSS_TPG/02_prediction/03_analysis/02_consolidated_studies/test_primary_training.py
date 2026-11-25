"""PS02 bounded batching, exact gradient replay, masks and resume contracts."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Data, Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.primary_training import batching as b, runner, reporting
from epss.primary_study import common
from epss.consolidated.contracts import StructuredEncoder


RESOURCES = {'embedding_dim':5, 'type_vocabulary':{'num_node_types':3,'node_types':{'TOKEN':2},
             'edge_types':{'DEP':0,'SEC_AFFECTS':1}}}


def fields(count=2):
    return {'cvss_vector':'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H',
            'cvss_vector_version':'3.1','cvss_vector_score':9.8,'n_references_nvd':count}


def graph(seed):
    generator = torch.Generator().manual_seed(seed)
    x = torch.zeros(4,9); x[:,2]=1; x[:,3:8]=torch.randn(4,5,generator=generator); x[:,-1]=1
    return Data(x=x, edge_index=torch.tensor([[0,1,2],[1,2,3]]), edge_type=torch.tensor([0,1,0]),
                edge_features=torch.randn(3,64,generator=generator), token_mask=torch.ones(4,dtype=torch.bool))


class MemoryStore:
    loads = hits = 0
    def load(self,path,view):
        return graph(int(path))


def setup_batches(directory, budget=8):
    rows = []
    index = {}
    for cve,split,docids in [('A','train',['1','2']),('B','train',['3']),('C','validation',['4'])]:
        rows.append({'cve':cve,'split':split,'documents':[{'document_id':'d','source_text_type':'description','text':'word'}],
                     'source_mask':[1,0,0,0],'structured':fields(len(docids))})
        index[cve] = {'documents':[{'source_text_type':'description','chunks':[{'has_graph':True,'tensor':s} for s in docids]}]}
    common.write(Path(directory)/'cve_index.json',index)
    state = StructuredEncoder.fit([{'cve':r['cve'],'split':'train',**r['structured']} for r in rows if r['split']=='train'])
    settings = runner.settings_for('smoke')
    settings.update(hidden=8,layers=1,graph_node_budget=budget,graph_edge_budget=20,graph_chunk_budget=2)
    batches = b.Batches(directory,rows,state,RESOURCES,settings,cache_mib=0)
    batches.store = MemoryStore()
    variant = next(v for v in runner.variants_for('core') if v['id']=='P01_D_security_tpg_S1')
    return batches,variant,settings


class PrimaryTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_registered_suites_and_no_normal_frontend_substitution(self):
        self.assertEqual(len(runner.variants_for('core')),9)
        self.assertEqual(len(runner.variants_for('sources')),63)
        self.assertEqual(len(runner.variants_for('all-supported')),68)
        self.assertEqual(len(runner.fits_for('core',runner.settings_for('full'))),21)
        self.assertEqual(len(runner.fits_for('sources',runner.settings_for('full'))),183)
        self.assertFalse(any('normal_text' in v['id'] for v in runner.variants_for('all-supported')))

    def test_streamed_predictions_gradients_and_update_match_dense(self):
        for representation in ('security_tpg','pooled_text'):
            with self.subTest(representation=representation), tempfile.TemporaryDirectory() as d:
                batches,v,s = setup_batches(d); v={**v,'representation':representation}
                runner.fixed_seed(17)
                dense = b.make_model(RESOURCES,len(batches.state['feature_names']),v,s); streamed=deepcopy(dense)
                plan = batches.plan(['A','B'],v); target=torch.tensor([.2,.8])
                graphs = Batch.from_data_list([graph(int(p)) for p in plan['paths']])
                numeric = {k:plan[k] for k in ('chunk_document','document_source','source_mask','structured')}
                prediction = dense({'graphs':graphs,**numeric})
                (prediction-target).abs().mean().backward()
                actual,_,_ = b.replay_backward(streamed,batches,plan,v,target)
                self.assertTrue(torch.allclose(actual,prediction,atol=1e-7,rtol=0))
                for (name,a),(other,c) in zip(dense.named_parameters(),streamed.named_parameters()):
                    self.assertEqual(name,other)
                    if a.grad is None: self.assertIsNone(c.grad)
                    else: self.assertTrue(torch.allclose(a.grad,c.grad,atol=2e-6,rtol=5e-5),name)
                for model in (dense,streamed):
                    torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001).step()
                for a,c in zip(dense.parameters(),streamed.parameters()):
                    self.assertTrue(torch.allclose(a,c,atol=2e-6,rtol=5e-5))

    def test_chunk_budgets_preserve_all_chunks_and_order(self):
        with tempfile.TemporaryDirectory() as d:
            batches,v,_=setup_batches(d)
            plan=batches.plan(['A','B'],v); packets=list(batches.packets(plan,v))
            self.assertEqual([p[0] for p in packets],[0,2])
            self.assertEqual(sum(p.num_graphs for _,p in packets),3)
            self.assertTrue(all(p.num_nodes<=8 and p.num_edges<=20 for _,p in packets))
            batches.settings['graph_node_budget']=3
            with self.assertRaisesRegex(ValueError,'refuse truncation'):list(batches.packets(plan,v))

    def test_disabled_sources_and_structured_blocks_cannot_contribute(self):
        with tempfile.TemporaryDirectory() as d:
            batches,v,s=setup_batches(d)
            v={**v,'sources':['smp'],'structured_blocks':[]}
            plan=batches.plan(['A','B'],v)
            self.assertFalse(plan['paths']); self.assertFalse(plan['source_mask'].any())
            self.assertEqual(float(plan['structured'].abs().sum()),0.)
            net=b.make_model(RESOURCES,len(batches.state['feature_names']),v,s)
            p,w,m=b.fuse(net,torch.empty(0,8),plan)
            self.assertTrue(torch.isfinite(p).all()); self.assertEqual(float(w.sum()),0.)
            with self.assertRaises(ValueError):b.fuse(net,torch.empty(0,8),{**plan,'target':torch.ones(2)})

    def test_replay_rejects_stochastic_layers(self):
        with tempfile.TemporaryDirectory() as d:
            batches,v,s=setup_batches(d); net=b.make_model(RESOURCES,len(batches.state['feature_names']),v,s)
            next(m for m in net.modules() if isinstance(m,torch.nn.Dropout)).p=.1
            with self.assertRaisesRegex(ValueError,'dropout'):
                b.replay_backward(net,batches,batches.plan(['A'],v),v,torch.ones(1))

    def test_edge_filter_uses_vocabulary_and_preserves_nodes(self):
        with tempfile.TemporaryDirectory() as d:
            g=graph(1)
            bundle={'shared':{'x':g.x},'edges':{'edge_index':g.edge_index,'edge_type':g.edge_type,'edge_features':g.edge_features},
                    'edge_masks':{'no_coref':torch.ones(3,dtype=torch.bool)}}
            torch.save(bundle,Path(d)/'chunk.pt')
            store=b.GraphStore(d,RESOURCES,cache_mib=0)
            item=store.load('chunk.pt','security_edges_removed')
            self.assertTrue(torch.equal(item.x,g.x)); self.assertEqual(item.edge_type.tolist(),[0,0])
            self.assertEqual(store.bytes,0)
            with self.assertRaises(ValueError):store.load('../escape.pt','no_coref')

    def test_baseline_verification_does_not_refit_ridge(self):
        with tempfile.TemporaryDirectory() as d:
            batches,_,s=setup_batches(d)
            v=next(v for v in runner.variants_for('core') if v['representation']=='structured_ridge')
            targets={c:{'value':p,'target_date':'2024-01-01'} for c,p in [('A',.2),('B',.7),('C',.4)]}
            identity={'ids':{'train':['A','B'],'validation':['C']},'settings':s}
            state,rows,diagnostics=runner.baseline(v,batches,targets,identity)
            with patch.object(runner.Ridge,'fit',side_effect=AssertionError('Must not refit')):
                self.assertEqual(runner.baseline(v,batches,targets,identity,{'baseline':state}),(state,rows,diagnostics))

    def test_interrupted_epoch_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); batches,v,s=setup_batches(d)
            targets={c:{'value':p,'target_date':'2024-01-01'} for c,p in [('A',.2),('B',.7),('C',.4)]}
            identity={'ids':{'train':['A','B'],'validation':['C']},'settings':s}
            for name in ('uninterrupted','resumed'): (root/name).mkdir()
            expected,_=runner.train_fit(root/'uninterrupted',root/'all.pt',v,17,batches,targets,identity)
            real_save=runner.atomic_torch
            def stop_after_save(path,obj):
                real_save(path,obj)
                if path==root/'resume.pt' and obj['epoch']==1:raise RuntimeError('simulated interruption')
            with patch.object(runner,'atomic_torch',side_effect=stop_after_save):
                with self.assertRaisesRegex(RuntimeError,'simulated'):
                    runner.train_fit(root/'resumed',root/'resume.pt',v,17,batches,targets,identity)
            actual,_=runner.train_fit(root/'resumed',root/'resume.pt',v,17,batches,targets,identity)
            self.assertEqual(expected,actual)
            bad={**identity,'settings':{**s,'learning_rate':.5}}
            with self.assertRaisesRegex(ValueError,'Resume checkpoint identity'):
                runner.train_fit(root/'resumed',root/'resume.pt',v,17,batches,targets,bad)

    def test_paired_intervals_are_deterministic_and_signed(self):
        settings={'bootstrap_seed':17,'bootstrap_repetitions':30}
        a=reporting.paired_interval([-.1,-.2,-.3],settings)
        self.assertEqual(a,reporting.paired_interval([-.1,-.2,-.3],settings))
        self.assertLess(a['ci95_upper'],0)


if __name__=='__main__':unittest.main()
