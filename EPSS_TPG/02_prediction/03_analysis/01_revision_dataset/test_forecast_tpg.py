"""Synthetic TPG forecasting contracts and small end-to-end training tests."""
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

PACKAGE = Path(__file__).resolve().parents[2]/'01_package'
sys.path.insert(0, str(PACKAGE))
from epss import forecast_tpg as task
from epss import forecast_tpg_data as cache
from tpg.contracts import EDGE_FEATURE_DIM
from tpg.exporters.exporters import PyGExporter
from tpg.schema.types import SECURITY_SCHEMA
import torch


def fixture():
    vocab = PyGExporter().export_vocab(SECURITY_SCHEMA, True)
    dim = 4
    ids = {'train':['CVE-2021-1000','CVE-2021-1001'], 'validation':['CVE-2024-1000','CVE-2024-1001']}
    inputs, labels, manifest, graphs = {}, {}, {}, {}
    for i, c in enumerate(sum(ids.values(), [])):
        year = c.split('-')[1]
        pub = year+'-01-01'
        from datetime import date, timedelta
        dates = [(date.fromisoformat(pub)+timedelta(days=j)).isoformat() for j in range(91)]
        values = {d: .01+(i+j)*.0001 for j,d in enumerate(dates)}
        values[dates[30]] = None
        raw = {'cve':c, 'published_date':pub, 'published_timestamp':pub+'T00:00:00Z', 'epss_scores':values}
        inputs[c], labels[c] = cache.contract.make_record(raw)
        split = 'train' if c in ids['train'] else 'validation'
        manifest[c] = {'split':split, 'group_id':c}
        n = len(vocab['node_types'])
        x = torch.zeros(3, n+dim+1)
        x[:, vocab['node_types']['TOKEN']] = 1
        x[:, n:-1] = torch.arange(12).reshape(3,4)/12+i*.01
        x[:, -1] = 1
        graphs[c] = {'x':x, 'edge_index':torch.tensor([[0,1],[1,2]]),
            'edge_type':torch.tensor([vocab['edge_types']['DEP'],vocab['edge_types']['SEC_AFFECTS']]),
            'edge_features':torch.ones(2, EDGE_FEATURE_DIM)*.1,
            'text_embedding':x[:,n:-1].mean(0), 'cve':c, 'description_sha256':'synthetic'}
    data = {'inputs.json':inputs, 'labels.json':labels, 'split_manifest.json':manifest,
            'task_cohorts.json':{'endpoint':ids}}
    return {'ids':ids, 'embedding_dim':dim, 'type_vocabulary':vocab, 'scale':'smoke'}, graphs, data


class TPGForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_description_allowlist_no_fallback(self):
        r = {'description':'Apache has a buffer overflow.', 'epss_scores':object(),
             'in_vckev':object(), 'llm_summary':'Do not use', 'social_media_posts':object()}
        self.assertEqual(cache.description_only(r), r['description'])
        with self.assertRaisesRegex(ValueError, 'Missing description'):
            cache.description_only({'orig_description':'Do not silently substitute'})

    def test_shared_security_pipeline_retains_typed_qualified_edges(self):
        from tpg.pipeline import SecurityPipeline
        pipeline=SecurityPipeline(include_security_relations=True)
        if not pipeline.frontend.nlp.has_pipe('parser'):
            self.skipTest('Local en_core_web_sm parser is not installed')
        graph=pipeline.run('CVE-2024-1000 may not affect Apache 2.4.51.',doc_id='synthetic')
        self.assertEqual(graph.validate(),[])
        exported=pipeline.export_pyg(graph,use_security_edge_types=True)
        index=exported['type_vocabulary']['edge_types']['SEC_AFFECTS']
        self.assertIn(index,exported['edge_type'])
        qualified=[p for t,p in zip(exported['edge_type'],exported['edge_properties']) if t==index]
        self.assertTrue(any(p['extra']['polarity']=='negated' and p['extra']['modality']=='possible' for p in qualified))
        self.assertEqual(exported['n_dropped_security_edges'],0)
        self.assertIsNone(exported['y'])

    def test_selection_deterministic_and_outcome_independent(self):
        m, _, data = fixture()
        selected = cache.select_ids(data, 'smoke')
        changed = copy.deepcopy(data)
        for r in changed['labels.json'].values():
            r['y']['endpoint'] = .99
        self.assertEqual(selected, cache.select_ids(changed, 'smoke'))
        self.assertEqual(selected, m['ids'])
        c = m['ids']['validation'][0]
        changed['split_manifest.json'][c]['group_id'] = m['ids']['train'][0]
        with self.assertRaisesRegex(ValueError, 'groups cross'):
            cache.select_ids(changed, 'smoke')

    def test_smoke_limits_are_enforced(self):
        m, _, data = fixture()
        for split, limit in [('train',32),('validation',16)]:
            ids = [f'CVE-2020-{10000+i}' for i in range(100)] if split=='train' else [f'CVE-2024-{10000+i}' for i in range(100)]
            data['task_cohorts.json']['endpoint'][split] = ids
            for c in ids:
                data['split_manifest.json'][c] = {'split':split,'group_id':c}
                data['labels.json'][c] = {'eligible':{'endpoint':True}}
        selected = cache.select_ids(data,'smoke')
        self.assertEqual({s:len(v) for s,v in selected.items()}, {'train':32,'validation':16})
        self.assertEqual({s:len(v) for s,v in cache.select_ids(data,'full').items()}, {'train':100,'validation':100})

    def test_graph_validation_rejects_targets_edges_and_nan(self):
        m, graphs, _ = fixture()
        item = next(iter(graphs.values()))
        cache.validate_item(item,m['type_vocabulary'],m['embedding_dim'])
        for fault in ('target','nan','edge','relation','mask'):
            changed = copy.deepcopy(item)
            if fault=='target': changed['y'] = torch.tensor([.9])
            if fault=='nan': changed['x'][0,0] = float('nan')
            if fault=='edge': changed['edge_index'][0,0] = 99
            if fault=='relation': changed['edge_type'][0] = 99
            if fault=='mask': changed['x'][0,-1] = .5
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                cache.validate_item(changed,m['type_vocabulary'],m['embedding_dim'])

    def test_batch_has_only_past_inputs_and_no_target(self):
        m, graphs, data = fixture()
        batch = task.make_batch(m['ids']['train'],graphs,data['inputs.json'])
        self.assertNotIn('y',batch.keys())
        self.assertNotIn('cve',batch.keys())
        self.assertEqual(batch.epss_sequence.shape,(2,31,5))
        self.assertEqual(batch.epss_sequence[0,30,1].item(),0)
        self.assertAlmostEqual(batch.epss_sequence[0,30,2].item(),1/30)
        changed = copy.deepcopy(data)
        for lab in changed['labels.json'].values(): lab['y']['endpoint'] = .99
        other = task.make_batch(m['ids']['train'],graphs,changed['inputs.json'])
        self.assertTrue(torch.equal(batch.epss_sequence,other.epss_sequence))

    def test_all_modes_bounded_and_gradients_finite(self):
        m, graphs, data = fixture()
        config = task.config_for('smoke')
        batch = task.make_batch(m['ids']['train'],graphs,data['inputs.json'])
        for mode in task.MODES:
            with self.subTest(mode=mode):
                net = task.TPGRegressor(**task.model_spec(mode,m,graphs,config))
                pred = net(batch)
                self.assertEqual(pred.shape,(2,))
                self.assertTrue(((pred>0)&(pred<1)).all())
                pred.sum().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))
                if mode=='tpg_epss':
                    edge_id = m['type_vocabulary']['edge_types']['SEC_AFFECTS']
                    self.assertGreater(float(net.layers[0].edge_embed.weight.grad[edge_id].abs().sum()),0)

    def test_ablation_and_modality_isolation(self):
        m, graphs, data = fixture()
        batch = task.make_batch(m['ids']['train'],graphs,data['inputs.json'])
        config = task.config_for('smoke')
        for mode in ('tpg_only','epss_only','text_epss','tpg_no_edges_epss'):
            net = task.TPGRegressor(**task.model_spec(mode,m,graphs,config)).eval()
            pred = net(batch)
            changed = batch.clone()
            if mode=='tpg_only': changed.epss_sequence = torch.randn_like(batch.epss_sequence)
            else:
                changed.edge_index = changed.edge_index.flip(0)
                changed.edge_features = changed.edge_features*7
                changed.edge_type = torch.zeros_like(changed.edge_type)
                if mode in ('epss_only','text_epss'): changed.x = changed.x*7
            self.assertTrue(torch.equal(pred,net(changed)), mode)

    def test_checkpoint_reloads_and_fit_is_reproducible(self):
        m, graphs, data = fixture()
        config = task.config_for('smoke');config.update(epochs=2,patience=2)
        first, history = task.fit('tpg_epss',17,m,graphs,data,config,progress=False)
        second, other_history = task.fit('tpg_epss',17,m,graphs,data,config,progress=False)
        self.assertEqual(history,other_history)
        self.assertTrue(all(torch.equal(v,second['state_dict'][k]) for k,v in first['state_dict'].items()))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.pt';torch.save(first,p)
            loaded=torch.load(p,weights_only=True)
            state=torch.get_rng_state().clone()
            net=task.restore(loaded)
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
            batch=task.make_batch(m['ids']['validation'],graphs,data['inputs.json'])
            self.assertTrue(torch.equal(task.restore(first)(batch),net(batch)))

    def test_cache_roundtrip_and_tamper_rejection(self):
        m, graphs, data = fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);prepared=root/'prepared';prepared.mkdir()
            raw={c:{'description':'Synthetic description for '+c} for c in graphs}
            pins=cache.code_pins()
            class Builder:
                identity={'synthetic':True}
                model=type('Model',(),{'config':type('Config',(),{'hidden_size':m['embedding_dim']})()})()
                def __init__(self,**kwargs): self.pipeline=self
                def build(self,text,cve):
                    item=copy.deepcopy(graphs[cve]);item['description_sha256']=cache.text_digest(text)
                    info={'cve':cve,'description_sha256':item['description_sha256'],
                          'stats':{'total_nodes':len(item['x']),'total_edges':len(item['edge_type']),
                                   'edge_types':{'DEP':1,'SEC_AFFECTS':1}}}
                    return item,info,m['type_vocabulary'],{}
                def export_graphson(self,graph,path): cache.write_json(path,{'synthetic':True})
            # Keep outputs separate from source.parent, matching the production safety rule.
            source_dir=root/'raw';source_dir.mkdir()
            source=source_dir/'source.json';cache.write_json(source,[])
            pins[str(source)]=cache.sha256(source)
            data['preparation_summary.json']={'source_path':str(source)}
            output=root/'cache'
            with patch.object(cache,'load_source',return_value=(data,raw,source,pins)), \
                 patch.object(cache,'GraphBuilder',Builder), \
                 patch.object(cache.contract,'load_prepared',return_value=data), redirect_stdout(io.StringIO()):
                cache.prepare(prepared,output)
                manifest,items,loaded=cache.load_cache(output)
                self.assertEqual(manifest['ids'],m['ids'])
                self.assertEqual(set(items),set(graphs))
                self.assertTrue(all('y' not in v for v in items.values()))
                graph_path=output/(m['ids']['train'][0]+'.pt')
                graph_path.write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError,'Changed/invalid cache artifact'):
                    cache.load_cache(output)

    def test_complete_synthetic_run_and_no_overwrite(self):
        m, graphs, data = fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cache_dir=root/'cache';cache_dir.mkdir()
            source_dir=root/'source';source_dir.mkdir()
            prepared=root/'prepared';prepared.mkdir()
            source=source_dir/'source.json';cache.write_json(source,[])
            data['preparation_summary.json']={'source_path':str(source)}
            m.update(source_pins={},prepared_directory=str(prepared))
            cache.write_json(cache_dir/'manifest.json',m)
            cache.write_json(cache_dir/'cache_lock.json',{'files':{'manifest.json':cache.sha256(cache_dir/'manifest.json')}})
            config=task.config_for('smoke');config.update(epochs=1,patience=1)
            output=root/'results'
            with patch.object(cache,'load_cache',return_value=(m,graphs,data)), \
                 patch.object(task,'config_for',return_value=config), redirect_stdout(io.StringIO()):
                task.run(cache_dir,output,check_only=True)
                self.assertFalse(output.exists())
                task.run(cache_dir,output)
                with self.assertRaisesRegex(ValueError,'already exists'):
                    task.run(cache_dir,output)
            summary=cache.read_json(output/'summary.json')
            self.assertEqual(len(summary['results']),6)
            self.assertFalse(summary['test_performance_evaluated'])
            self.assertEqual(summary['cohort_ids'],m['ids'])
            self.assertEqual(cache.read_json(output/'run_status.json')['status'],'complete')
            for name,digest in cache.read_json(output/'evaluation_lock.json')['files'].items():
                self.assertEqual(cache.sha256(output/name),digest)


if __name__=='__main__':
    unittest.main()
