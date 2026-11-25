"""Narrow tokenizer-removal tests; no dataset fitting or full cache expansion."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import torch
from transformers import AutoTokenizer

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import reddit_soft_hyphen_repair_v1 as repair


class Encoder:
    dim=3
    def __init__(self,tokenizer,drop=None):
        self.tokenizer=tokenizer;self.drop=drop
    def embed(self,graph):
        for n in graph.nodes(repair.legacy.NodeType.TOKEN):
            if n.properties.text!=self.drop and self.tokenizer(n.properties.text,add_special_tokens=False)['input_ids']:
                n.properties.extra['embedding']=[.1,.2,.3]


class SoftHyphenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tokenizer=AutoTokenizer.from_pretrained('jackaduma/SecBERT',use_fast=True,local_files_only=True)
        repair.old.check_tokenizer(cls.tokenizer,repair.cache.resources()['encoder'])
        cls.pipeline=repair.cache.Pipeline()

    def documents(self,text):
        graph=self.pipeline.run(text,doc_id='RD02-SYNTHETIC-SOFT-HYPHEN')
        saved=json.loads(repair.cache.GraphSONExporter().export_string(graph))
        return repair.cache.layout.old.shared.preview.variants(saved)[0]

    def test_narrow_allowlist(self):
        for text in ('\u00ad','\u00ad'*20):self.assertTrue(repair.soft_only(text))
        for text in ('',' ','-','\u200b','\u00ade','\u00ad\u0301','\u00ad-'):
            self.assertFalse(repair.soft_only(text))

    def test_pinned_tokenizer_removes_soft_hyphens(self):
        text='\u00ad'*20
        self.assertEqual(self.tokenizer.backend_tokenizer.normalizer.normalize_str(text),'')
        self.assertEqual(self.tokenizer(text,add_special_tokens=False)['input_ids'],[])

    def test_original_coverage_remains_strict(self):
        docs=self.documents('Security '+ '\u00ad'*20 +' report.')
        with self.assertRaisesRegex(ValueError,'Missing ordinary text-token embeddings'):
            repair.old.build_bundle(docs,Encoder(self.tokenizer))
        before=deepcopy(docs)
        bundle,vocab,marks=repair.build_bundle(docs,Encoder(self.tokenizer))
        self.assertEqual(docs,before)
        self.assertEqual(len(marks),1);self.assertEqual(marks[0]['reason'],repair.REASON)
        self.assertEqual(marks[0]['codepoints'],['U+00AD']*20)
        graph=repair.legacy.import_graphson(docs['all_coref'])
        i=next(i for i,n in enumerate(graph.nodes()) if n.id==marks[0]['node_id'])
        self.assertEqual(float(bundle['shared']['x'][i,-1]),0.)
        self.assertTrue((bundle['shared']['x'][i,vocab['num_node_types']:-1]==0).all())

    def test_unchanged_inputs_have_identical_tensors(self):
        for text in ('Security report.','Security \u06e7 report.','Security soft\u00adhyphen report.'):
            docs=self.documents(text)
            old,vocab,marks=repair.old.build_bundle(docs,Encoder(self.tokenizer))
            new,new_vocab,new_marks=repair.build_bundle(docs,Encoder(self.tokenizer))
            self.assertEqual(vocab,new_vocab);self.assertEqual(marks,new_marks)
            for section in old:
                for key,value in old[section].items():
                    if isinstance(value,torch.Tensor):self.assertTrue(torch.equal(value,new[section][key]))
                    else:self.assertEqual(value,new[section][key])

    def test_ordinary_and_mixed_missing_tokens_fail(self):
        for word in ('Security','caf\u00e9','soft\u00adhyphen'):
            docs=self.documents(word+' '+ '\u00ad'*20 +' report.')
            with self.assertRaisesRegex(ValueError,'Missing ordinary text-token embeddings'):
                repair.build_bundle(docs,Encoder(self.tokenizer,drop=word))

    def test_contextual_overlap_requires_embedding(self):
        docs=self.documents('Security \u00ad report.')
        bundle,vocab,_=repair.build_bundle(docs,Encoder(self.tokenizer))
        class Covered:
            backend_tokenizer=self.tokenizer.backend_tokenizer
            def __call__(self,text,**kw):
                if kw.get('return_offsets_mapping'):return {'input_ids':[1],'offset_mapping':[(0,len(text))]}
                return {'input_ids':[]}
        with self.assertRaisesRegex(ValueError,'contextual tokenizer coverage'):
            repair.validate_bundle(bundle,docs,vocab,3,Covered())

    def test_source_and_zero_vector_guards(self):
        docs=self.documents('Security \u00ad report.')
        bundle,vocab,marks=repair.build_bundle(docs,Encoder(self.tokenizer))
        graph=repair.legacy.import_graphson(docs['all_coref'])
        i=next(i for i,n in enumerate(graph.nodes()) if n.id==marks[0]['node_id'])
        broken=deepcopy(bundle);broken['shared']['x'][i,vocab['num_node_types']]=1
        with self.assertRaisesRegex(ValueError,'zero vectors'):
            repair.validate_bundle(broken,docs,vocab,3,self.tokenizer)
        for document in docs.values():
            for node in document['vertices']:
                if node['label']=='TOKEN' and node['properties']['TEXT']=='\u00ad':
                    node['properties']['CHAR_START']-=1
        with self.assertRaisesRegex(ValueError,'span differs'):
            repair.validate_bundle(bundle,docs,vocab,3,self.tokenizer)


if __name__=='__main__':
    unittest.main()
