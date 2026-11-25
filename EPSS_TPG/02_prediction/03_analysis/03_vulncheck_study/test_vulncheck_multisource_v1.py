"""VK07 accepted identities, source isolation, pooling, replay, and D-reference parity."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Batch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.vulncheck_study import multisource_data_v1 as prep, multisource_model_v1 as network
from epss.vulncheck_study import multisource_training_v1 as training
from epss.vulncheck_study import multisource_cache_v1 as cache
from epss.vulncheck_study import graph_model_v1 as old, graph_training_v1 as original
from test_vulncheck_graph_v1 import resources, graph


def source_plan():
    return {'paths':['a','b','c','d','e'], 'chunk_document':torch.tensor([0,1,2,3,4]),
            'document_source':torch.tensor([0,1,2,3,4]),
            'source_mask':torch.tensor([[1,1,1,1],[1,0,0,0]],dtype=torch.bool),
            'structured':torch.tensor([[.1,.9],[.8,.2]])}


class Packets:
    def packets(self,plan,variant):
        for i in range(len(plan['paths'])):
            g=graph(); g.x[:,2:5] *= 1+i*.2
            yield i,Batch.from_data_list([g])


class MultisourceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.settings={**original.settings('smoke'),'hidden':4,'layers':1}

    def test_frozen_settings_and_comparison_counts(self):
        cfg=prep.config()
        self.assertEqual(len(cfg['variants'])*len(cfg['representations'])*3,24)
        self.assertEqual((len(cfg['variants'])-1)*len(cfg['representations'])*3,18)
        self.assertEqual(original.settings('full')['seeds'],[17,29,43])
        self.assertFalse(cfg['test_scoring'])
        self.assertFalse(cfg['threshold_tuning'])

    def test_exact_identity_and_platforms(self):
        fields=dict(zip(prep.SOURCES,prep.SOURCES))
        raw={'description':'Exact description','smp':[{'post':'Same text','platform':'A'},{'post':'Same text','platform':'B'}]}
        docs=[]
        for source,text,platform in [('description','Exact description',None),('smp','Same text','A'),('smp','Same text','B')]:
            docs.append({'document_id':source+':'+prep.data.digest({'source':source,'text':text,'platform':platform}),
                         'source_text_type':source,'text':text})
        row={'cve':'CVE-2025-12345','split':'train','documents':docs,'source_mask':[1,0,0,1],'structured':[.1]}
        converted,_=prep.canonical_documents(row,raw,fields)
        self.assertEqual(len({d['document_id'] for d in converted['documents']}),3)
        raw['in_vckev']=True
        self.assertEqual(converted,prep.canonical_documents(row,raw,fields)[0])
        changed=deepcopy(row);changed['documents'][0]['text']+=' '
        with self.assertRaises(ValueError):prep.canonical_documents(changed,raw,fields)
        with self.assertRaises(ValueError):prep.canonical_documents({**row,'target':1},raw,fields)
        with self.assertRaises(ValueError):prep.canonical_documents({**row,'split':'test'},raw,fields)
        row['documents']=docs[:1]
        self.assertEqual(len(prep.canonical_documents(row,raw,fields)[0]['documents']),1)

    def test_nested_pooling_not_flat_chunk_mean(self):
        p={'chunk_document':torch.tensor([0,0,1,2]),'document_source':torch.tensor([0,0,3]),'structured':torch.zeros(1,2)}
        values=network.pool_sources(torch.tensor([[1.],[3.],[10.],[7.]]),p)
        self.assertEqual(values[0,0,0].item(),6.)
        self.assertEqual(values[0,3,0].item(),7.)

    def test_missing_sources_masked_and_no_empty_fusion(self):
        attention=torch.nn.Linear(3,1)
        z=torch.randn(2,4,3);mask=torch.tensor([[1,0,0,0],[1,1,0,1]],dtype=torch.bool)
        first,weights=network.masked_fusion(z,mask,attention,torch.zeros(4))
        z[~mask]=1000
        second,_=network.masked_fusion(z,mask,attention,torch.zeros(4))
        torch.testing.assert_close(first,second,atol=0,rtol=0)
        torch.testing.assert_close(weights.sum(1),torch.ones(2))
        self.assertTrue(bool((weights[~mask]==0).all()))
        with self.assertRaises(ValueError):network.masked_fusion(z,torch.zeros_like(mask),attention,torch.zeros(4))

    def test_source_variants_gate_documents_and_paths(self):
        row={'cve':'CVE-2025-12345','split':'train','documents':[],'source_mask':[1,1,1,1],'structured':[0.,1.]}
        docs=[{'source_text_type':s,'chunks':[{'has_graph':True,'tensor':s}]} for s in prep.SOURCES]
        for variant,sources in prep.config()['variants'].items():
            with patch.object(old,'GraphStore'):
                b=network.Batches(Path('.'),[row],{row['cve']:{'documents':docs}},resources(),self.settings,sources)
            p=b.plan([row['cve']])
            self.assertEqual(p['paths'],sources)
            self.assertEqual(p['source_mask'][0].tolist(),[s in sources for s in prep.SOURCES])
            self.assertEqual(set(p),{'paths','chunk_document','document_source','source_mask','structured'})

    def test_D_initialization_and_checkpoint_numerical_parity(self):
        for rep in ('pooled_text','security_tpg'):
            before=original.model_for(resources(),2,self.settings,17)
            after=training.model_for(resources(),2,self.settings,17)
            for name,value in before.state_dict().items():
                torch.testing.assert_close(value,after.state_dict()[name],atol=0,rtol=0)
            missing=after.load_state_dict(before.state_dict(),strict=False)
            self.assertFalse(missing.unexpected_keys)
            p={'paths':['a','b','c'],'chunk_document':torch.tensor([0,0,1]),'document_source':torch.tensor([0,4]),
               'source_mask':torch.tensor([[1,0,0,0],[1,0,0,0]],dtype=torch.bool),'structured':torch.randn(2,2)}
            z=network.representations(after,Packets(),p,rep)
            old_plan={'paths':p['paths'],'chunk_cve':torch.tensor([0,0,1]),'structured':p['structured']}
            torch.testing.assert_close(before.fuse(z,old_plan),after.fuse(z,p),atol=0,rtol=0)

    def test_multisource_replay_matches_direct_gradients(self):
        for rep in ('pooled_text','security_tpg'):
            direct=training.model_for(resources(),2,self.settings,17);replay=deepcopy(direct)
            p=source_plan();target=torch.tensor([0.,1.]);weights={0:.75,1:1.5}
            loss=network.loss_for(direct.fuse(network.representations(direct,Packets(),p,rep),p),target,weights)
            loss.backward()
            actual,_=network.replay_backward(replay,Packets(),p,rep,target,weights)
            self.assertAlmostEqual(float(loss.detach()),actual,places=7)
            for a,b in zip(direct.parameters(),replay.parameters()):
                if a.grad is None:self.assertIsNone(b.grad)
                else:torch.testing.assert_close(a.grad,b.grad,atol=1e-7,rtol=1e-5)
            self.assertGreater(float(replay.source_bias.grad.abs().sum()),0.)
            with self.assertRaises(ValueError):direct.fuse(torch.zeros(5,4),{**p,'target':target})

    def test_new_checkpoint_reload(self):
        first=training.model_for(resources(),2,self.settings,17)
        second=training.model_for(resources(),2,self.settings,29)
        second.load_state_dict(first.state_dict(),strict=True)
        p=source_plan();z=torch.randn(5,4)
        torch.testing.assert_close(first.fuse(z,p),second.fuse(z,p),atol=0,rtol=0)

    def test_reference_routes_are_read_only_and_budget_bounded(self):
        with patch.object(old,'GraphStore') as factory:
            store=network.ReferencedStore('/tmp/vk07-local',resources(),512,{'tensors/a.pt':'/tmp/vk07-source'})
            self.assertEqual(sum(call.args[2] for call in factory.call_args_list),512)
            self.assertEqual(len(factory.call_args_list),2)
            store.load('tensors/a.pt','no_coref')
            factory.return_value.load.assert_called_once_with('tensors/a.pt','no_coref')

    def test_unpinned_reference_root_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            (directory/'chunk_references.json').write_text(prep.data.dump({'chunk':'/unapproved/cache'}))
            with self.assertRaises(ValueError):cache.reference_sources(directory,{'entries':{'chunk':{}}},[])

    def test_undefined_precision_not_silently_zero(self):
        self.assertEqual(training.summarize_metric([None,None]),
                         {'mean':None,'std_population':None,'defined_seeds':0,'undefined_seeds':2})
        self.assertEqual(training.summarize_metric([0.,None,1.]),
                         {'mean':.5,'std_population':.5,'defined_seeds':2,'undefined_seeds':1})
        self.assertIsNone(training.metric_difference(None,.4))
        self.assertIsNone(training.metric_difference(.4,None))
        self.assertEqual(training.metric_difference(0.,.4),-.4)
        with self.assertRaises(ValueError):training.summarize_metric([float('nan')])

    def test_reports_with_zero_predicted_positives(self):
        cfg=prep.config();ids={'train':['a','b'],'validation':['c','d']}
        identity={'config':cfg,'settings':{'seeds':[17]},'schema':'test','scale':'smoke','ids':ids}
        metrics=prep.dev.metric([0,1],[.1,.2])
        results={training.fit_key(v,r,17):deepcopy(metrics) for v in cfg['variants'] for r in cfg['representations']}
        report=training.reports(results,[{'cve':'c','actual':0,'predicted':.1}],identity,{})
        self.assertIsNone(report['summary.json']['aggregate']['DGN__pooled_text']['precision']['mean'])
        self.assertEqual(report['summary.json']['aggregate']['DGN__pooled_text']['precision']['undefined_seeds'],1)
        self.assertTrue(all(r['precision'] is None for r in report['paired_comparisons.json']))
        self.assertIn('no_predicted_positives',prep.data.dump(report['metrics.json']))


if __name__=='__main__':unittest.main()
