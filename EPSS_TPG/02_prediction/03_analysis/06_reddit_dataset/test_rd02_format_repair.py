"""Verified format-token exceptions, never a general missing-embedding bypass."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import torch
from transformers import AutoTokenizer

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import reddit_format_repair_v2 as repair
from epss import reddit_soft_hyphen_repair_v1 as previous


class Encoder:
    dim=3
    def __init__(self,tokenizer,drop=None):
        self.tokenizer=tokenizer;self.drop=drop
    def embed(self,graph):
        for n in graph.nodes(repair.legacy.NodeType.TOKEN):
            if n.properties.text!=self.drop and self.tokenizer(n.properties.text,add_special_tokens=False)['input_ids']:
                n.properties.extra['embedding']=[.1,.2,.3]


class FormatRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tokenizer=AutoTokenizer.from_pretrained('jackaduma/SecBERT',use_fast=True,local_files_only=True)
        repair.old.check_tokenizer(cls.tokenizer,repair.cache.resources()['encoder'])
        cls.pipeline=repair.cache.Pipeline()

    def documents(self,text):
        graph=self.pipeline.run(text,doc_id='RD02-SYNTHETIC-FORMATTING')
        saved=json.loads(repair.cache.GraphSONExporter().export_string(graph))
        return repair.cache.layout.old.shared.preview.variants(saved)[0]

    def test_character_category_gate(self):
        for text in ('\u200d\ufe0f','\u200d','\u00ad'*20,'\u200b\u0301','\x01'):
            self.assertTrue(repair.format_only(text))
        for text in ('',' ','\n','\t','word','caf\u00e9','x\u200d','\U0001f600\ufe0f','-\u200d'):
            self.assertFalse(repair.format_only(text))

    def test_exact_failure_sequences_preserve_nodes_and_masks(self):
        for token in ('\u200d\ufe0f','\u200d'):
            docs=self.documents('Security '+token+' report.');before=deepcopy(docs)
            for policy in (repair.old,previous):
                with self.assertRaisesRegex(ValueError,'Missing ordinary text-token embeddings'):
                    policy.build_bundle(docs,Encoder(self.tokenizer))
            bundle,vocab,marks=repair.build_bundle(docs,Encoder(self.tokenizer))
            self.assertEqual(docs,before);self.assertEqual(len(marks),1)
            self.assertEqual(marks[0]['reason'],repair.REASON)
            self.assertEqual(marks[0]['contextual_overlap_count'],0)
            graph=repair.legacy.import_graphson(docs['all_coref'])
            i=next(i for i,n in enumerate(graph.nodes()) if n.id==marks[0]['node_id'])
            self.assertEqual(float(bundle['shared']['x'][i,-1]),0.)
            self.assertTrue((bundle['shared']['x'][i,vocab['num_node_types']:-1]==0).all())

    def test_existing_successes_have_identical_tensors(self):
        for text in ('Security report.','Security \u06e7 report.','Security \u00ad report.'):
            docs=self.documents(text)
            before,vocab,old_marks=previous.build_bundle(docs,Encoder(self.tokenizer))
            after,new_vocab,new_marks=repair.build_bundle(docs,Encoder(self.tokenizer))
            self.assertEqual(vocab,new_vocab)
            self.assertEqual(len(old_marks),len(new_marks))
            for section in before:
                for key,value in before[section].items():
                    if isinstance(value,torch.Tensor):self.assertTrue(torch.equal(value,after[section][key]))
                    else:self.assertEqual(value,after[section][key])

    def test_missing_ordinary_and_mixed_words_fail(self):
        for token in ('Security','caf\u00e9','word\u200dword','\U0001f600'):
            docs=self.documents(token+' \u200d\ufe0f report.')
            with self.assertRaisesRegex(ValueError,'Missing ordinary text-token embeddings'):
                repair.build_bundle(docs,Encoder(self.tokenizer,drop=token))

    def test_contextual_overlap_requires_embedding(self):
        docs=self.documents('Security \u200d\ufe0f report.')
        bundle,vocab,_=repair.build_bundle(docs,Encoder(self.tokenizer))
        class Covered:
            backend_tokenizer=self.tokenizer.backend_tokenizer
            def __call__(self,text,**kw):
                if kw.get('return_offsets_mapping'):return {'input_ids':[1],'offset_mapping':[(0,len(text))]}
                return {'input_ids':[]}
        with self.assertRaisesRegex(ValueError,'contextual tokenizer coverage'):
            repair.validate_bundle(bundle,docs,vocab,3,Covered())

    def test_empty_normalization_is_required(self):
        from types import SimpleNamespace
        docs=self.documents('Security \u200d report.')
        bundle,vocab,_=repair.build_bundle(docs,Encoder(self.tokenizer))
        class Retained:
            backend_tokenizer=SimpleNamespace(normalizer=SimpleNamespace(normalize_str=lambda text:text))
            def __call__(self,text,**kw):return {'input_ids':[]}
        with self.assertRaisesRegex(ValueError,'not removed'):
            repair.validate_bundle(bundle,docs,vocab,3,Retained())

    def test_standalone_tokens_are_required_to_be_empty(self):
        docs=self.documents('Security \u200d report.')
        bundle,vocab,_=repair.build_bundle(docs,Encoder(self.tokenizer))
        class Retained:
            backend_tokenizer=self.tokenizer.backend_tokenizer
            def __call__(self,text,**kw):return {'input_ids':[1]}
        with self.assertRaisesRegex(ValueError,'not removed'):
            repair.validate_bundle(bundle,docs,vocab,3,Retained())

    def test_zero_vector_and_source_span_guards(self):
        docs=self.documents('Security \u200d\ufe0f report.')
        bundle,vocab,marks=repair.build_bundle(docs,Encoder(self.tokenizer))
        graph=repair.legacy.import_graphson(docs['all_coref'])
        i=next(i for i,n in enumerate(graph.nodes()) if n.id==marks[0]['node_id'])
        broken=deepcopy(bundle);broken['shared']['x'][i,vocab['num_node_types']]=1
        with self.assertRaisesRegex(ValueError,'zero vectors'):
            repair.validate_bundle(broken,docs,vocab,3,self.tokenizer)
        for document in docs.values():
            for node in document['vertices']:
                if node['label']=='TOKEN' and node['properties']['TEXT']=='\u200d\ufe0f':
                    node['properties']['CHAR_START']-=1
        with self.assertRaisesRegex(ValueError,'span differs'):
            repair.validate_bundle(bundle,docs,vocab,3,self.tokenizer)


if __name__=='__main__':
    unittest.main()
