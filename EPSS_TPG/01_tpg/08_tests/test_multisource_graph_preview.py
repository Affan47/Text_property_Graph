"""MS02 synthetic document isolation, shared tensors and checked resume tests."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

TPG = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TPG.parent/'02_prediction/01_package'))
from epss import forecast_multisource_graphs as cache
from test_security_coreference_cache import TinyEncoder
from tpg.schema.graph import TextPropertyGraph
from tpg.schema.types import NodeType, EdgeType, NodeProperties, EdgeProperties, SECURITY_SCHEMA


def synthetic_build(entry, encoder):
    g = TextPropertyGraph(schema=SECURITY_SCHEMA,doc_id=entry['graph_id'])
    text = entry['text']
    g.metadata.update(source_text=text,security_extraction_version=cache.extension.v3.VERSION)
    g.mark_pass(cache.extension.VERSION)
    g.add_node(NodeType.DOCUMENT,NodeProperties(text=text,char_start=0,char_end=len(text)))
    g.add_node(NodeType.SENTENCE,NodeProperties(text=text,char_start=0,char_end=len(text)))
    for start in (0,6):
        node = g.add_node(NodeType.TOKEN,NodeProperties(text='Alpha',char_start=start,char_end=start+5))
        g.add_edge(1,node,EdgeType.CONTAINS)
    g.add_edge(2,3,EdgeType.NEXT_TOKEN)
    g.add_edge(2,3,EdgeType.COREF,EdgeProperties(extra={'method':'repeated_surface','status':'candidate'}))
    cache.stamp_graph(g,entry)
    docs,_ = cache.shared.preview.variants(json.loads(cache.shared.GraphSONExporter().export_string(g)))
    cache.validate_documents(docs,entry)
    bundle,vocab = cache.coverage.build_bundle(docs,encoder)
    return docs,bundle,vocab


class FakeBuilder:
    instances = []
    stop_after = None

    def __init__(self,*args):
        self.encoder = TinyEncoder()
        self.instances.append(self)

    def build(self,entry):
        if self.stop_after is not None and self.encoder.calls == self.stop_after:
            raise RuntimeError('synthetic interruption')
        return synthetic_build(entry,self.encoder)


def fixture():
    ids = {'train':['CVE-2000-0001'],'validation':['CVE-2000-0002']}
    rows,splits = [],{}
    for split,chosen in ids.items():
        c = chosen[0]
        sources = cache.SOURCES if split == 'train' else ('description',)
        docs = [{'document_id':s+':fixed','source_text_type':s,'text':'Alpha Alpha'} for s in sources]
        rows.append({'cve':c,'split':split,'documents':docs,'source_mask':[int(s in sources) for s in cache.SOURCES]})
        splits[c] = {'split':split,'group_id':c}
    associations,entries = cache.collect_documents(ids,rows,splits)
    _,_,vocab = synthetic_build(next(iter(entries.values())),TinyEncoder())
    identity = {'schema':cache.SCHEMA,'ids':ids,'associations':associations,
                'documents':[{k:v for k,v in e.items() if k != 'text'} for e in entries.values()],
                'type_vocabulary':vocab,'embedding_dim':4,'source_pins':{},
                'protocol':{'limitations':['Synthetic fixtures only']},
                'comparisons':{'epss_only':[], 'description_epss':['description'],
                               'all_text_epss':list(cache.SOURCES), 'without_description':list(cache.SOURCES[1:])}}
    return {'identity':identity,'entries':entries},rows,splits


class MultiSourceTests(unittest.TestCase):
    def setUp(self):
        FakeBuilder.instances = []
        FakeBuilder.stop_after = None

    def test_unique_document_namespace_for_identical_text(self):
        ctx,_,_ = fixture()
        entries = list(ctx['entries'].values())
        self.assertEqual(len(entries),5)
        self.assertEqual(len({e['text_sha256'] for e in entries}),1)
        self.assertEqual(len({e['graph_id'] for e in entries}),5)

    def test_source_tags_and_no_cross_document_links(self):
        ctx,_,_ = fixture()
        for e in ctx['entries'].values():
            docs,bundle,vocab = synthetic_build(e,TinyEncoder())
            cache.coverage.validate_bundle(bundle,docs,vocab,4)
            graph = cache.shared.import_graphson(docs['no_coref'])
            self.assertTrue(all(o.properties.extra['source_text_type'] == e['source_text_type']
                                for o in [*graph.nodes(),*graph.edges()]))
            self.assertEqual(len(graph.edges(EdgeType.COREF)),0)
            self.assertLess(int(bundle['edges']['edge_index'].max()),len(bundle['shared']['x']))

    def test_one_embedding_call_and_shared_ablation_features(self):
        ctx,_,_ = fixture()
        encoder,bundles = TinyEncoder(),{}
        for key,e in ctx['entries'].items():
            _,bundles[key],_ = synthetic_build(e,encoder)
        self.assertEqual(encoder.calls,5)
        full = cache.materialize_view(ctx['identity'],bundles,'all_text_epss')
        desc = cache.materialize_view(ctx['identity'],bundles,'description_epss')
        empty = cache.materialize_view(ctx['identity'],bundles,'epss_only')
        for c in full:
            self.assertIs(full[c]['by_source']['description'][0]['x'],desc[c]['by_source']['description'][0]['x'])
            self.assertFalse(empty[c]['source_mask'].any())
            self.assertTrue(all(not v for v in empty[c]['by_source'].values()))
        self.assertEqual(encoder.calls,5)

    def test_missing_sources_keep_cve_and_do_not_fallback(self):
        ctx,_,_ = fixture()
        bundles = {k:synthetic_build(e,TinyEncoder())[1] for k,e in ctx['entries'].items()}
        view = cache.materialize_view(ctx['identity'],bundles,'without_description')
        self.assertEqual(len(view),2)
        val = view['CVE-2000-0002']
        self.assertEqual(val['source_mask'].tolist(),[False]*4)
        self.assertTrue(all(not v for v in val['by_source'].values()))
        rows = cache.audit_views(ctx['identity'],bundles)
        self.assertTrue(all(r['cves'] == 1 for r in rows))

    def test_forbidden_fields_and_mask_mismatch_rejected(self):
        ctx,rows,splits = fixture()
        for field in ('y','epss_history','vckev_date_added','platform'):
            changed = deepcopy(rows)
            changed[0]['documents'][0][field] = 'not input'
            with self.assertRaises(ValueError):
                cache.collect_documents(ctx['identity']['ids'],changed,splits)
        changed = deepcopy(rows)
        changed[0]['source_mask'] = [0]*4
        with self.assertRaises(ValueError):
            cache.collect_documents(ctx['identity']['ids'],changed,splits)

    def test_day60_exclusion_is_preserved_at_graph_boundary(self):
        ctx,rows,splits = fixture()
        adapter = cache.development.h01.sources
        fields = {s:s for s in cache.SOURCES}
        raw = {'cve':rows[0]['cve'],'description':'Alpha Alpha','github_summary':None,
               'non_github_summary':None,'smp':[
                   {'post':'before','date':'2022-03-01','platform':'telegram'},
                   {'post':'same day','date':'2022-03-02','platform':'telegram'},
                   {'post':'later','date':'2022-03-03','platform':'telegram'},
                   {'post':'conflict','date':'2022-02-01','platform':'telegram'},
                   {'post':'conflict','date':'2022-02-02','platform':'telegram'},
                   {'post':'missing','date':None,'platform':'telegram'}]}
        audit = adapter.adapt_record(raw,'2022-03-02',fields)
        rows[0] = {'cve':raw['cve'],'split':'train',**adapter.model_input(audit,list(cache.SOURCES))}
        _,entries = cache.collect_documents(ctx['identity']['ids'],rows,splits)
        self.assertEqual([e['text'] for e in entries.values() if e['source_text_type'] == 'smp'],['before'])

    def test_cached_future_fields_or_missing_token_embeddings_rejected(self):
        ctx,_,_ = fixture()
        e = next(iter(ctx['entries'].values()))
        docs,bundle,vocab = synthetic_build(e,TinyEncoder())
        changed = deepcopy(bundle)
        changed['shared']['y'] = .9
        with self.assertRaises(ValueError):
            cache.coverage.validate_bundle(changed,docs,vocab,4)
        changed = deepcopy(bundle)
        changed['shared']['x'][2,vocab['num_node_types']:] = 0
        with self.assertRaises(ValueError):
            cache.coverage.validate_bundle(changed,docs,vocab,4)

    def test_test_roles_duplicates_and_related_groups_rejected(self):
        ctx,rows,splits = fixture()
        ids = ctx['identity']['ids']
        with self.assertRaises(ValueError):
            cache.collect_documents({**ids,'test':['CVE-2000-0003']},rows,splits)
        with self.assertRaises(ValueError):
            cache.collect_documents(ids,rows+rows[:1],splits)
        changed = deepcopy(splits)
        changed['CVE-2000-0002']['group_id'] = 'CVE-2000-0001'
        with self.assertRaises(ValueError):
            cache.collect_documents(ids,rows,changed)

    def test_missing_source_label_and_changed_text_fail(self):
        ctx,_,_ = fixture()
        e = next(iter(ctx['entries'].values()))
        docs,_,_ = synthetic_build(e,TinyEncoder())
        graph = cache.shared.import_graphson(docs['no_coref'])
        next(iter(graph.nodes())).properties.extra.pop('source_text_type')
        docs['no_coref'] = json.loads(cache.shared.GraphSONExporter().export_string(graph))
        with self.assertRaises(ValueError):
            cache.validate_documents(docs,e)
        with self.assertRaises(ValueError):
            cache.validate_documents(synthetic_build(e,TinyEncoder())[0],{**e,'text':'Changed'})

    def test_preview_roundtrip_and_verify_without_models(self):
        ctx,_,_ = fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(cache,'context',return_value=ctx), \
                patch.object(cache,'OUTPUT',Path(tmp)/'preview'), patch.object(cache,'Builder',FakeBuilder):
            cache.prepare(check_only=True)
            self.assertFalse(cache.OUTPUT.exists())
            cache.prepare()
            identity,bundles = cache.verify(cache.OUTPUT,ctx)
            self.assertEqual(len(bundles),5)
            self.assertEqual(identity,ctx['identity'])
            with patch.object(cache,'Builder',side_effect=AssertionError('No encoder on verification')):
                cache.prepare(verify_only=True)
            (cache.OUTPUT/'analysis.txt').write_text('tampered')
            with self.assertRaises(ValueError):
                cache.verify(cache.OUTPUT,ctx)

    def test_resume_preserves_completed_document(self):
        ctx,_,_ = fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(cache,'context',return_value=ctx), \
                patch.object(cache,'OUTPUT',Path(tmp)/'preview'), patch.object(cache,'Builder',FakeBuilder):
            FakeBuilder.stop_after = 1
            with self.assertRaisesRegex(RuntimeError,'synthetic interruption'):
                cache.prepare()
            key = next(iter(ctx['entries']))
            path = cache.OUTPUT/'graphs'/(key+'.pt')
            digest = cache.sha256(path)
            FakeBuilder.stop_after = None
            cache.prepare(resume=True)
            self.assertEqual(FakeBuilder.instances[-1].encoder.calls,4)
            self.assertEqual(cache.sha256(path),digest)
            cache.prepare(verify_only=True)


if __name__ == '__main__':
    unittest.main()
