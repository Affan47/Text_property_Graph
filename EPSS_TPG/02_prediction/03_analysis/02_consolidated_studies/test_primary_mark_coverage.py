"""Regression checks for tokenizer-removed marks, not a blanket Unicode exemption."""
from copy import deepcopy
from concurrent.futures import Future
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_primary_mark_coverage as policy
from epss import forecast_primary_cache_v2 as runner
from epss.primary_study import common


class Encoder:
    dim=3
    def __init__(self,tokenizer,drop=None):self.tokenizer=tokenizer;self.drop=drop
    def embed(self,graph):
        for n in graph.nodes(policy.legacy.NodeType.TOKEN):
            if n.properties.text!=self.drop and self.tokenizer(n.properties.text,add_special_tokens=False)['input_ids']:
                n.properties.extra['embedding']=[.1,.2,.3]


class MarkCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.resources=common.read(common.DATA/'smoke/resources.json')
        cls.tokenizer=runner.tokenizer_for(cls.resources)
        cls.pipeline=runner.old.base.old.extension.ProductVersionSecurityPipeline()

    def documents(self,text='Security \u06e7 report.'):
        graph=self.pipeline.run(text,doc_id='PS01-UNICODE-TEST')
        saved=json.loads(policy.legacy.GraphSONExporter().export_string(graph))
        return runner.old.base.old.shared.preview.variants(saved)[0]

    def build(self,text='Security \u06e7 report.',drop=None):
        docs=self.documents(text)
        return docs,policy.build_bundle(docs,Encoder(self.tokenizer,drop))

    def test_narrow_unicode_classification(self):
        for text in ('\u06e7','\u0301','\u06e7\u0301'):self.assertTrue(policy.mark_only(text))
        for text in ('','word','e\u0301','\u00e9','\u06e5','\u06e7x','\x00',' '):self.assertFalse(policy.mark_only(text))

    def test_pinned_tokenizer_removes_reported_mark_not_words(self):
        self.assertEqual(self.tokenizer.backend_tokenizer.normalizer.normalize_str('\u06e7'),'')
        self.assertEqual(self.tokenizer('\u06e7',add_special_tokens=False)['input_ids'],[])
        for text in ('security','caf\u00e9','\u0639\u0631\u0628\u064a'):
            self.assertTrue(self.tokenizer(text,add_special_tokens=False)['input_ids'])
        with self.assertRaises(ValueError):policy.check_tokenizer(self.tokenizer,{'tokenizer_sha256':'wrong'})

    def test_text_nodes_edges_masks_and_pooling_are_retained(self):
        docs=self.documents();before=deepcopy(docs)
        bundle,vocab,marks=policy.build_bundle(docs,Encoder(self.tokenizer))
        self.assertEqual(docs,before);self.assertEqual(len(marks),1)
        self.assertEqual(marks[0]['codepoints'],['U+06E7'])
        self.assertEqual(marks[0]['contextual_overlap_count'],0)
        graph=policy.legacy.import_graphson(docs['all_coref'])
        i=next(i for i,n in enumerate(graph.nodes()) if n.id==marks[0]['node_id'])
        self.assertEqual(float(bundle['shared']['x'][i,-1]),0.)
        self.assertTrue((bundle['shared']['x'][i,vocab['num_node_types']:-1]==0).all())
        self.assertEqual(len(bundle['shared']['x']),len(docs['all_coref']['vertices']))
        self.assertTrue(torch.isfinite(bundle['shared']['text_embedding']).all())

    def test_ordinary_missing_word_still_fails(self):
        with self.assertRaisesRegex(ValueError,'Missing ordinary text-token embeddings'):
            self.build(drop='Security')

    def test_non_ascii_word_is_not_a_mark_exception(self):
        with self.assertRaisesRegex(ValueError,'Missing ordinary text-token embeddings'):
            self.build('Security caf\u00e9 report.',drop='caf\u00e9')

    def test_mark_exception_cannot_hide_nonzero_missing_vector(self):
        docs,(bundle,vocab,marks)=self.build()
        graph=policy.legacy.import_graphson(docs['all_coref'])
        i=next(i for i,n in enumerate(graph.nodes()) if n.id==marks[0]['node_id'])
        bundle['shared']['x'][i,vocab['num_node_types']]=1.
        with self.assertRaisesRegex(ValueError,'zero vectors'):
            policy.validate_bundle(bundle,docs,vocab,3,self.tokenizer)

    def test_changed_source_span_fails(self):
        docs,(bundle,vocab,_)=self.build()
        for graph in docs.values():
            for node in graph['vertices']:
                if node['label']=='TOKEN' and node['properties']['TEXT']=='\u06e7':
                    node['properties']['TEXT']='\u0301'
        with self.assertRaisesRegex(ValueError,'span differs'):
            policy.validate_bundle(bundle,docs,vocab,3,self.tokenizer)

    def test_contextually_covered_mark_must_have_embedding(self):
        docs,(bundle,vocab,_)=self.build()
        class Covered:
            backend_tokenizer=self.tokenizer.backend_tokenizer
            def __call__(self,text,**kw):
                if kw.get('return_offsets_mapping'):return {'input_ids':[1],'offset_mapping':[(0,len(text))]}
                return {'input_ids':[]}
        with self.assertRaisesRegex(ValueError,'contextual tokenizer coverage'):
            policy.validate_bundle(bundle,docs,vocab,3,Covered())

    def test_legacy_validator_remains_strict(self):
        docs,(bundle,vocab,_)=self.build()
        with self.assertRaisesRegex(ValueError,'Missing ordinary text-token embeddings'):
            policy.controls.validate_bundle(bundle,docs,vocab,3)
        docs,(bundle,vocab,marks)=self.build('Security report.')
        self.assertEqual(marks,[])
        policy.controls.validate_bundle(bundle,docs,vocab,3)

    def test_bounded_queue_failure_does_not_submit_full_corpus(self):
        class Pool:
            def __init__(self,*args,**kwargs):self.submitted=0;self.closed=False
            def submit(self,fn,payload):
                self.submitted+=1;f=Future();f.set_exception(ValueError('deliberate'));return f
            def shutdown(self,**kwargs):self.closed=True
        pool=Pool()
        with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'ProcessPoolExecutor',return_value=pool):
            with self.assertRaisesRegex(ValueError,'deliberate'):
                runner.bounded_build(Path(tmp),[{'key':str(i)} for i in range(100)],{},2,{})
            self.assertEqual(pool.submitted,4);self.assertTrue(pool.closed)
            self.assertEqual(common.read(Path(tmp)/'chunk_provenance.json'),{})


if __name__=='__main__':unittest.main()
