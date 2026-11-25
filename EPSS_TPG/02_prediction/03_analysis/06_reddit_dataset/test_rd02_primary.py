from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
from torch_geometric.data import Data,Batch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.reddit_primary import structured,model,runner,data
from epss.tpg_revision import subtypes_v6


def fields():
    return {'cvss_vector':'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H',
            'cvss_vector_version':3.1,'cvss_vector_score':9.8,'all_sources_urls':['a','a','b'],
            'github_only_urls':['a'],'non_github_urls':['b']}


class RD02Tests(unittest.TestCase):
    def test_structured_state_json_replay(self):
        rows=[{'cve':'A','split':'train','structured':fields()}];state=structured.fit(rows)
        a=structured.transform(fields(),state)[0];b=structured.transform(fields(),json.loads(data.dump(state)))[0]
        self.assertEqual(a,b);self.assertEqual(len(a),len(state['feature_names']))

    def test_allowlist_and_train_only_fit(self):
        for key in ('in_vckev','epss_scores','vckev_cwes','n_references_nvd'):
            with self.assertRaises(ValueError):structured.parse(fields()|{key:1})
        with self.assertRaises(ValueError):structured.fit([{'cve':'V','split':'validation','structured':fields()}])

    def test_deduplicated_counts_and_missing(self):
        f=fields();_,nums,_=structured.parse(f);self.assertAlmostEqual(nums['log1p_all_sources_urls'],np.log1p(2),places=14)
        f['all_sources_urls']=None;f['cvss_vector']=None
        cats,nums,_=structured.parse(f);self.assertEqual(cats['version'],'MISSING');self.assertIsNone(nums['cvss_score'])
        self.assertIsNone(nums['log1p_all_sources_urls'])

    def test_validation_cannot_change_fit(self):
        state=structured.fit([{'cve':'T','split':'train','structured':fields()}]);original=deepcopy(state)
        f=fields();f['all_sources_urls']=['x']*100;structured.transform(f,state)
        self.assertEqual(state,original)

    def test_subtype_append(self):
        self.assertEqual(len(subtypes_v6.VOCAB),32);self.assertEqual(subtypes_v6.VOCAB['SEVERITY'],31)

    def test_undefined_metrics_and_clipping(self):
        self.assertIsNone(runner.metrics([],[])['mae']);self.assertIsNone(runner.metrics([.1],[.2])['r2'])
        self.assertEqual(runner.metrics([.2],[-1])['clipped_fraction'],1.)

    def test_same_weight_masked_wrapper_and_replay_gradient(self):
        torch.set_num_threads(1);torch.manual_seed(17)
        res={'type_vocabulary':{'num_node_types':2,'edge_types':{'DEP':0}}}
        a=model.Model(res,3,'D');b=model.Model(res,3,'DM_all');state=deepcopy(a.state_dict());state['fusion.enabled']=b.fusion.enabled;b.load_state_dict(state)
        x=torch.randn(3,771);x[:,:2]=torch.tensor([[1,0],[0,1],[1,0]]);x[:,-1]=1
        g=Data(x=x,edge_index=torch.tensor([[0,1],[1,2]]),edge_type=torch.zeros(2,dtype=torch.long),
               edge_features=torch.zeros(2,64),token_mask=torch.tensor([1,0,1],dtype=torch.bool),subtype=torch.tensor([0,31,0]))
        graph=Batch.from_data_list([g,g]);variant=model.VARIANTS['D_v6_tpg']
        plan={'paths':['a','b'],'chunk_document':torch.tensor([0,1]),'document_source':torch.tensor([0,4]),
              'source_mask':torch.tensor([[1,0,0,0],[1,0,0,0]],dtype=torch.bool),'structured':torch.randn(2,3)}
        self.assertTrue(torch.equal(a.fuse(a.encode(graph,variant),plan),b.fuse(b.encode(graph,variant),plan)))
        full=deepcopy(a);target=torch.tensor([.2,.8],dtype=torch.float64)
        class Packets:
            def packets(self,*args):yield 0,graph
        model.backward(a,Packets(),plan,variant,target)
        (full.fuse(full.encode(graph,variant),plan).double()-target).abs().mean().backward()
        for (name,p),(_,q) in zip(a.named_parameters(),full.named_parameters()):
            if p.grad is None:self.assertIsNone(q.grad)
            else:torch.testing.assert_close(p.grad,q.grad,rtol=1e-6,atol=1e-7,msg=name)


if __name__=='__main__':unittest.main()
