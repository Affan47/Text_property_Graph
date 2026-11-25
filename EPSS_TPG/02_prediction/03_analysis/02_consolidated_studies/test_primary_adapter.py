"""PS01 predictor isolation, exact reuse, pooling, extraction and GNN checks."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.primary_study import data,cache,model,common
from epss.consolidated.contracts import StructuredEncoder
from epss.edge_aware_layers import RGATBlock,EDGE_FEATURE_DIM


def fields():
    return {'cvss_vector':'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H',
            'cvss_vector_version':'3.1','cvss_vector_score':9.8,'n_references_nvd':3}


def row():
    return {'cve':'CVE-2024-10001','split':'train','documents':[],
            'source_mask':[0,0,0,0],'structured':fields()}


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_labels_cannot_enter_predictor_rows(self):
        r=row();data.validate_input(r)
        for name in ('value','epss_scores','target_date','in_vckev'):
            with self.assertRaises(ValueError):data.validate_input({**r,name:.5})
        r['structured']['vckev_cwes']=[]
        with self.assertRaises(ValueError):data.validate_input(r)

    def test_test_rows_refused(self):
        r=row();r['split']='test'
        with self.assertRaises(ValueError):data.validate_input(r)

    def test_smoke_selection_covers_sources_without_target_values(self):
        masks={'A':[1,0,0,0],'B':[1,1,0,0],'C':[1,0,1,1]}
        self.assertEqual(data.smoke_ids(['A','B','C'],masks,2),['B','C'])
        self.assertEqual(data.smoke_ids(['C','B','A'],masks,2),['B','C'])

    def test_document_hash_retains_exact_text_and_platform(self):
        a={'source':'smp','platform':'Reddit','text':'Example'}
        self.assertNotEqual(data.document_id(a),data.document_id({**a,'text':'Example '}))
        self.assertNotEqual(data.document_id(a),data.document_id({**a,'platform':'Mastodon'}))

    def test_cache_requires_entry_and_file_identity(self):
        e={'text':'exact','key':'a','stem':'cve/source/doc/chunk','split':'train','text_sha256':'abc'}
        receipt={'entry':{k:v for k,v in e.items() if k!='text'},'files':dict.fromkeys(cache.base.files(e),'hash')}
        self.assertTrue(cache.reusable(receipt,e))
        for k,v in [('split','validation'),('text_sha256','def'),('stem','other')]:
            self.assertFalse(cache.reusable(receipt,{**e,k:v}))

    def test_hierarchical_pool_does_not_overweight_more_chunks(self):
        chunks=torch.tensor([[1.],[3.],[10.]],requires_grad=True)
        docs=model.mean_groups(chunks,torch.tensor([0,0,1]),2)
        sources=model.mean_groups(docs,torch.tensor([0,0]),4)
        self.assertEqual(sources.tolist(),[[6.],[0.],[0.],[0.]])
        sources.sum().backward();self.assertEqual(chunks.grad.tolist(),[[.25],[.25],[.5]])

    def test_structured_scaling_training_only_and_unknowns(self):
        state=StructuredEncoder.fit([{'cve':'A','split':'train',**fields()}])
        values,_=StructuredEncoder.transform(fields(),state,['cvss','reference_count'])
        self.assertEqual(values[-2:], [0.,1.])
        missing={**fields(),'n_references_nvd':None}
        self.assertEqual(StructuredEncoder.transform(missing,state,['reference_count'])[0][-2:],[0.,0.])

    def test_numeric_batch_and_structured_only_missing_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            common.write(Path(directory)/'cve_index.json',{row()['cve']:{'documents':[]}})
            state=StructuredEncoder.fit([{'cve':'A','split':'train',**fields()}])
            resources={'embedding_dim':3,'type_vocabulary':{'num_node_types':2,'node_types':{'TOKEN':1},'edge_types':{'DEP':0}}}
            collate=model.Collator(directory,[row()],state,resources)
            batch=collate([row()['cve']],[],True)
            variant={'sources':[],'representation':'pooled_text','structured':True}
            net=model.PrimaryModel(resources,len(state['feature_names']),variant,hidden=4,layers=1)
            value=net(batch);self.assertTrue(torch.isfinite(value).all());value.sum().backward()
            with self.assertRaises(ValueError):net({**batch,'target':torch.tensor([.1])})

    def test_exact_locks_detect_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);common.write(p/'inputs.json',{'x':1});common.lock(p);common.verify_lock(p)
            common.write(p/'inputs.json',{'x':2})
            with self.assertRaises(ValueError):common.verify_lock(p)

    def test_relation_features_affect_existing_gnn(self):
        from epss.edge_aware_layers import EdgeTypeGNNLayer
        torch.manual_seed(17);layer=EdgeTypeGNNLayer(8,3,0.)
        x=torch.randn(4,8);edges=torch.tensor([[0,1,2],[1,2,3]]);types=torch.tensor([0,1,2])
        zero=torch.zeros(3,EDGE_FEATURE_DIM)
        a=layer(x,edges,types,zero);b=layer(x,edges,types,zero+1)
        self.assertFalse(torch.allclose(a,b));a.square().sum().backward()
        self.assertIsNotNone(layer.edge_embed.weight.grad)

    def test_alternative_gnns_forward_backward_only(self):
        from torch_geometric.nn import RGCNConv,GATv2Conv
        x=torch.randn(4,8,requires_grad=True);edges=torch.tensor([[0,1,2],[1,2,3]]);types=torch.tensor([0,1,2])
        features=torch.randn(3,EDGE_FEATURE_DIM)
        candidates=[RGATBlock(8,3,heads=2,dropout=0.),RGCNConv(8,8,3,num_bases=2),
                    GATv2Conv(8,4,heads=2,edge_dim=EDGE_FEATURE_DIM,add_self_loops=False)]
        for layer in candidates:
            if isinstance(layer,RGATBlock):out=layer(x,edges,types,features)
            elif isinstance(layer,RGCNConv):out=layer(x,edges,types)
            else:out=layer(x,edges,features)
            self.assertEqual(out.shape,(4,8));self.assertTrue(torch.isfinite(out).all());out.sum().backward(retain_graph=True)

    def test_text_dependent_security_edges_and_qualifiers(self):
        audit=cache.dynamic_checks()
        self.assertFalse(audit['synthetic_checks']['neutral']['security_edges'])


if __name__=='__main__':unittest.main()
