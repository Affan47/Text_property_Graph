from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
for rel in ('01_tpg/10_extensions','01_tpg/01_core','02_prediction/01_package','01_tpg/08_tests'):
    sys.path.insert(0,str(ROOT/rel))
torch.set_num_threads(1)
from tpg_combined_v3 import compact,cache
from tpg_combined_v3.embeddings import Encoder
from tpg_combined_v2 import graphs,common as c
from tpg_combined_recovery_v1.boundary import Builder
from test_combined_semantics_v2 import entry


class FakeTokenizer:
    def __call__(self,text,**kwargs):
        return {'input_ids':[[1,2,3],[4,5,6]],'attention_mask':[[1,1,1],[1,1,1]],
                'offset_mapping':[[(0,0),(0,2),(2,4)],[(4,6),(6,8),(0,0)]]}


class FakeModel:
    def __call__(self,input_ids,attention_mask):
        return SimpleNamespace(last_hidden_state=input_ids.float()[:,:,None].repeat(1,1,768))


class CompactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e=entry([('D','CVE-2022-1234 affects Apache 2.4.1. The file was deleted by attackers.'),
                     ('M','A use after free compromises the system.')])
        cls.g=Builder().build(c.digest(cls.e),cls.e)[0]

    def packed(self):
        return compact.pack(graphs.tensors(self.g),self.g)

    def test_exact_structural_round_trip(self):
        arrays,meta=self.packed();actual=compact.unpack(arrays,meta)
        for k,v in graphs.tensors(self.g).items():
            if isinstance(v,torch.Tensor):
                self.assertEqual(v.dtype,actual[k].dtype);self.assertTrue(torch.equal(v,actual[k]),k)
            else:self.assertEqual(v,actual[k])
        self.assertEqual(actual['subtype_vocabulary']['SEVERITY'],31)

    def test_sparse_storage_smaller(self):
        a,_=self.packed()
        sparse=sum(a[k].nbytes for k in ('edge_ptr','edge_col','edge_value'))
        self.assertLess(sparse,graphs.tensors(self.g)['edge_features'].numel()*4)

    def test_pool_reuses_exact_spans(self):
        a,_=self.packed()
        self.assertLess(len(a['pool_ptr'])-1,len(self.g['vertices']))
        rows=compact.pooling_rows(a)
        from tpg_combined_v2.embeddings import node_spans
        for i,spans in enumerate(node_spans(self.g)):
            self.assertEqual(rows[a['pool_index'][i]],[[x,y] for x,y in spans if y>x>=0])

    def test_phrase_gap_is_not_included(self):
        g={'vertices':[{'label':'CLAUSE','properties':{'EXTRA':{'member_spans':[[0,2],[6,8]]}}},
                       {'label':'SENTENCE','properties':{'CHAR_START':0,'CHAR_END':8}}]}
        a=compact.pooling_layout(g)
        self.assertNotEqual(a['pool_index'][0],a['pool_index'][1])

    def test_serialization_is_deterministic_and_pickle_free(self):
        a,m=self.packed()
        with tempfile.TemporaryDirectory() as tmp:
            p,q=Path(tmp)/'a.npz',Path(tmp)/'b.npz'
            compact.save(p,a,m);compact.save(q,a,m)
            self.assertEqual(p.read_bytes(),q.read_bytes())
            b,n=compact.load(p);self.assertEqual(m,n)
            self.assertTrue(all(np.array_equal(a[k],b[k]) for k in a))
            with self.assertRaises(FileExistsError):compact.save(p,a,m)

    def test_sparse_corruption_rejected(self):
        a,m=self.packed();a['edge_ptr'][-1]+=1
        with self.assertRaises(ValueError):compact.unpack(a,m)

    def test_duplicate_sparse_column_rejected(self):
        a,m=self.packed();row=next(i for i in range(len(a['edge_ptr'])-1) if np.diff(a['edge_ptr'])[i]>=2)
        i=a['edge_ptr'][row];a['edge_col'][i+1]=a['edge_col'][i]
        with self.assertRaises(ValueError):compact.unpack(a,m)

    def test_integer_overflow_rejected(self):
        t=graphs.tensors(self.g);t['node_ids'][0]=2**40
        with self.assertRaises(ValueError):compact.pack(t,self.g)

    def test_nan_edge_rejected(self):
        t=graphs.tensors(self.g);t['edge_features'][0,0]=float('nan')
        with self.assertRaises(ValueError):compact.pack(t,self.g)

    def test_pool_index_out_of_range_rejected(self):
        a,m=self.packed();a['pool_index'][0]=len(a['pool_ptr'])
        with self.assertRaises(ValueError):compact.unpack(a,m)

    def test_float32_rows_expand_exactly(self):
        a,m=self.packed();n=len(a['pool_ptr'])-1
        rows=np.arange(n*768,dtype=np.float32).reshape(n,768);rows[0]=0
        mask=np.ones(n,dtype=bool);mask[0]=False
        t=compact.add_embeddings(compact.unpack(a,m),a,rows,mask)
        np.testing.assert_array_equal(t['node_embeddings'].numpy(),rows[a['pool_index']])
        with self.assertRaises(ValueError):compact.add_embeddings(t,a,rows.astype(np.float16),mask)

    def test_nonzero_missing_embedding_rejected(self):
        a,m=self.packed();n=len(a['pool_ptr'])-1
        with self.assertRaises(ValueError):
            compact.add_embeddings({},a,np.ones((n,768),np.float32),np.zeros(n,bool))

    def test_exact_windowed_dense_embedding_parity(self):
        g={'metadata':{'source_text':'abcdefgh'},'vertices':[
            {'id':i,'label':kind,'properties':props} for i,(kind,props) in enumerate([
                ('DOCUMENT',{'CHAR_START':0,'CHAR_END':8}),
                ('TOKEN',{'CHAR_START':2,'CHAR_END':6}),
                ('ENTITY',{'CHAR_START':2,'CHAR_END':6}),
                ('CLAUSE',{'EXTRA':{'member_spans':[[0,2],[6,8]]}}),
                ('TOPIC',{'CHAR_START':-1,'CHAR_END':-1})]) ]}
        encoder=Encoder.__new__(Encoder);encoder.device='cpu';encoder.tokenizer=FakeTokenizer();encoder.model=FakeModel()
        a=compact.pooling_layout(g);rows,mask,_=encoder.encode_compact('abcdefgh',g,a)
        expected,observed,_=encoder.encode('abcdefgh',g)
        self.assertTrue(torch.equal(expected,torch.from_numpy(rows[a['pool_index']])))
        self.assertTrue(torch.equal(observed,torch.from_numpy(mask[a['pool_index']])))
        with self.assertRaises(ValueError):encoder.encode_compact('changed',g,a)

    def test_correction_is_versioned_and_original_is_unchanged(self):
        before=c.digest(self.g);e={**self.e,'processing_identity':'new'}
        result=cache.transform(self.g,e,self.e,c.digest(self.e),'parenthash')
        self.assertEqual(before,c.digest(self.g));self.assertNotEqual(result['doc_id'],self.g['doc_id'])
        self.assertEqual(result['metadata']['source_text'],self.e['text'])
        self.assertTrue(result['metadata']['annotation_repair']['rule_counts'])
        graphs.validate(e,result)
        self.assertEqual(result,cache.transform(self.g,e,self.e,c.digest(self.e),'parenthash'))

    def test_no_new_object_overwrite(self):
        a,m=self.packed()
        with tempfile.TemporaryDirectory() as tmp:
            cache.publish(tmp,'abc',self.g,a,m,{})
            original=(Path(tmp)/'objects/abc/graph.json.gz').read_bytes()
            with self.assertRaises(ValueError):cache.publish(tmp,'abc',self.g,a,m,{})
            self.assertEqual(original,(Path(tmp)/'objects/abc/graph.json.gz').read_bytes())

    def test_changed_parent_blocks_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'parent/objects/old';source.mkdir(parents=True)
            c.create(source/'graph.json.gz',self.g);c.create(source/'graph_receipt.json',{})
            plan={'parent_directory':str(root/'parent'),'parent_keys':{'key':'old'},'processing':{'schema':'test'}}
            a,m=self.packed()
            cache.publish(root,'key',self.g,a,m,{'entry_sha256':'key','parent_key':'old',
                'processing_sha256':c.digest(plan['processing']),'parent_graph_sha256':c.sha(source/'graph.json.gz'),
                'parent_receipt_sha256':c.sha(source/'graph_receipt.json')})
            self.assertIsNotNone(cache.graph_receipt(root,'key',plan))
            (source/'graph_receipt.json').write_text('changed')
            with self.assertRaises(ValueError):cache.graph_receipt(root,'key',plan)

    def test_embedding_preflight_reserve(self):
        record={'compact_embedding_bytes':30*2**30}
        with patch.object(cache,'load_plan',return_value={'entries':{'key':{}}}),\
             patch.object(cache,'receipts',return_value={'key':record}),\
             patch.object(cache,'summarize',return_value={}),\
             patch.object(cache.shutil,'disk_usage',return_value=SimpleNamespace(free=40*2**30)):
            with self.assertRaisesRegex(ValueError,'Insufficient storage'):cache.preflight('reddit','full')

    def test_first_plan_creates_parent_and_preserves_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'new/smoke';source=Path(tmp)/'original/smoke'
            for name in ('plan.json.gz','plan_lock.json','parser_identity.json'):c.create(source/name,{})
            parent={'processing':{},'entries':{'old':self.e},'index':{'CVE-2022-1234':{
                'split':'train','views':{'D':{'object_key':'old','mask':True},'M':{'object_key':None,'mask':False}}}},
                'coverage':{},'variants':['D','M']}
            with patch.object(cache,'directory',return_value=root),patch.object(c,'directory',return_value=source),\
                 patch.object(c,'load_plan',return_value=parent):
                cache.prepare('reddit','smoke')
            self.assertTrue((root/'plan_lock.json').exists())
            plan=c.read(root/'plan.json.gz')
            self.assertFalse(plan['test_adapted'])
            self.assertEqual(plan['index']['CVE-2022-1234']['views']['M'],{'object_key':None,'mask':False})
            self.assertEqual(len(plan['entries']),1)
            self.assertEqual(plan['parent_keys'],{next(iter(plan['entries'])):'old'})


if __name__=='__main__':unittest.main()
