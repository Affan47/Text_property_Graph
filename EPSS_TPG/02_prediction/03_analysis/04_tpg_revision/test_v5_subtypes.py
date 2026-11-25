"""Narrow extraction regressions, exact cache reuse, subtype contracts and replay."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch_geometric.data import Data, Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss.tpg_revision.v5_audit import policy
from epss.tpg_revision import v5_cache, subtypes_v5 as subtypes
from epss.vulncheck_study import graph_model_v1 as network
from tpg.exporters.exporters import PyGExporter, import_graphson


def node(i, text, word, kind, source='security_frontend', label='ENTITY', start=None):
    a = text.index(word) if start is None else start
    return {'id': i, 'label': label, 'properties': {'TEXT': word, 'CHAR_START': a,
            'CHAR_END': a+len(word), 'ENTITY_TYPE': kind, 'SOURCE': source}}


def graph(text, nodes, edges=()):
    return {'doc_id':'synthetic', 'metadata':{'source_text':text}, 'vertices':nodes,
            'edges':list(edges), 'schema':{'security_edge_types':['SEC_HAS_VERSION']}}


def edge(i, a, b, label='BELONGS_TO'):
    return {'id':f'e{i}', 'outV':a, 'inV':b, 'label':label, 'properties':{}}


class Corrections(unittest.TestCase):
    def test_attacker_is_not_vector_but_remote_and_local_remain(self):
        text='allows an attacker; remote attacker; local'
        g=graph(text,[node(0,text,'allows an attacker','ATTACK_VECTOR'),
                      node(1,text,'remote attacker','ATTACK_VECTOR'),node(2,text,'local','ATTACK_VECTOR')])
        self.assertEqual([n['id'] for n in policy.correct(g)['vertices']], [1,2])

    def test_product_remote_and_actual_remote_attack(self):
        text='TeamViewer Remote Clients allows remote attackers'
        g=graph(text,[node(0,text,'TeamViewer Remote Clients','ORG','spacy_frontend'),
                      node(1,text,'Remote','ATTACK_VECTOR'), node(2,text,'remote attackers','ATTACK_VECTOR')])
        self.assertEqual([n['id'] for n in policy.correct(g)['vertices']],[0,2])

    def test_function_evidence_preserved(self):
        for text in ('call system()', 'the system function fails', 'function system fails', 'uses `system`'):
            self.assertEqual(len(policy.correct(graph(text,[node(0,text,'system','CODE_ELEMENT')]))['vertices']),1)
        text='access to the system is required'
        self.assertEqual(policy.correct(graph(text,[node(0,text,'system','CODE_ELEMENT')]))['vertices'],[])

    def test_exact_conflict_not_cardinal_or_compatible_product(self):
        text='App 22.3'
        g=graph(text,[node(0,text,'22.3','DATE','spacy_frontend'),node(1,text,'22.3','VERSION'),
                      node(2,text,'22.3','CARDINAL','spacy_frontend'),node(3,text,'App','PRODUCT','spacy_frontend'),node(4,text,'App','SOFTWARE')])
        self.assertEqual([n['id'] for n in policy.correct(g)['vertices']],[1,2,3,4])

    def test_versions_false_predicate_not_genuine_verb(self):
        text='App versions 10.2; it versions the files'
        g=graph(text,[node(0,text,'versions','','spacy_frontend','PREDICATE'),node(1,text,'10.2','VERSION'),
                      node(2,text,'versions','','spacy_frontend','PREDICATE',text.rindex('versions'))])
        self.assertEqual([n['id'] for n in policy.correct(g)['vertices']],[1,2])

    def test_ambiguous_numeric_date_does_not_automatically_prefer_version(self):
        text='The advisory was released on 10.2.'
        g=graph(text,[node(0,text,'10.2','DATE','spacy_frontend'),node(1,text,'10.2','VERSION')])
        self.assertEqual([n['id'] for n in policy.correct(g)['vertices']],[0,1])

    def test_nominal_write_not_actual_write(self):
        text='out-of-bounds write allows users to write'
        g=graph(text,[node(0,text,'out-of-bounds write','VULN_TYPE'),node(1,text,'write','','spacy_frontend','PREDICATE'),
                      node(2,text,'write','','spacy_frontend','PREDICATE',text.rindex('write'))])
        self.assertEqual([n['id'] for n in policy.correct(g)['vertices']],[0,2])

    def test_hardware_retype_and_relation_suppression(self):
        text='App software 1.2 hardware 1.0'
        g=graph(text,[node(0,text,'App','SOFTWARE'),node(1,text,'1.2','VERSION'),node(2,text,'1.0','VERSION')],
                [edge(0,0,1,'SEC_HAS_VERSION'),edge(1,0,2,'SEC_HAS_VERSION')])
        c=policy.correct(g)
        self.assertEqual(c['vertices'][2]['properties']['ENTITY_TYPE'],'HARDWARE_VERSION')
        self.assertEqual([e['id'] for e in c['edges']],['e0'])

    def test_uncertain_section_left_explicitly_unrepaired(self):
        text='see 3.2.1'
        self.assertEqual(len(policy.correct(graph(text,[node(0,text,'3.2.1','VERSION')]))['vertices']),1)

    def test_semantic_relation_from_false_predicate_removed_even_with_surviving_endpoints(self):
        text='App versions 10.2'
        nodes=[node(0,text,'App','SOFTWARE'),node(1,text,'versions','','spacy_frontend','PREDICATE'),node(2,text,'10.2','VERSION')]
        derived=edge(0,0,2,'ENTITY_REL')
        derived['properties']={'EXTRA':{'method':'dependency_rule','predicate_span':[4,12]}}
        g=graph(text,nodes,[derived,edge(1,0,2,'SEC_HAS_VERSION')])
        result=policy.correct(g)
        self.assertEqual([e['id'] for e in result['edges']],['e1'])
        self.assertEqual(result['metadata']['v5_correction']['edge_findings'][0]['input_edge_id'],'e0')

    def test_orphan_argument_only_and_idempotence(self):
        text='path normalization works'
        nodes=[node(0,text,'path','','spacy_frontend','PREDICATE'),node(1,text,'works','','spacy_frontend','PREDICATE'),
               node(2,text,'normalization','','spacy_frontend','ARGUMENT'),node(3,text,'normalization','','spacy_frontend','ARGUMENT'),
               node(4,text,'path','','spacy_frontend','TOKEN')]
        g=graph(text,nodes,[edge(0,2,0,'SRL_ARG'),edge(1,3,0,'SRL_ARG'),edge(2,3,1,'SRL_ARG'),edge(3,4,0,'SRL_ARG')])
        snapshot=deepcopy(g);c=policy.correct(g)
        self.assertEqual([n['id'] for n in c['vertices']],[1,3,4])
        self.assertEqual(g,snapshot)
        self.assertEqual(policy.correct(c),c)
        self.assertEqual(c['stats']['total_nodes'],3)

    def test_target_fields_and_invalid_endpoints_rejected(self):
        for g in ({**graph('a',[]),'target':1}, graph('a',[],[edge(0,0,1)])):
            with self.assertRaises(ValueError):policy.correct(g)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.r={'embedding_dim':2,'type_vocabulary':{'num_node_types':2,'node_types':{'TOKEN':0,'ENTITY':1},'edge_types':{'DEP':0}}}
        self.p={'hidden':4,'layers':1}

    def model(self, readout, enabled):
        torch.manual_seed(17)
        return subtypes.Classifier(self.r,2,self.p,readout,enabled)

    def sample(self):
        return Data(x=torch.tensor([[1.,0.,.3,.4,1.],[0.,1.,.2,.7,1.]]),
                    edge_index=torch.tensor([[0,1],[1,0]]),edge_type=torch.zeros(2,dtype=torch.long),
                    edge_features=torch.zeros(2,64),token_mask=torch.tensor([True,False]),
                    subtype=torch.tensor([0,subtypes.VOCAB['SOFTWARE']]))

    def test_unknown_missing_nonentity(self):
        values=subtypes.encode_nodes([{'label':'TOKEN','properties':{}},{'label':'ENTITY','properties':{}},
                                     {'label':'ENTITY','properties':{'ENTITY_TYPE':'unseen'}}])
        self.assertEqual(values.tolist(),[0,1,2])
        self.assertEqual(float(subtypes.one_hot(values)[0].sum()),0.)

    def test_zero_initialization_and_explicit_subtype_effect(self):
        a,b=self.model('node_mean',False),self.model('node_mean',True)
        g=Batch.from_data_list([self.sample()])
        torch.testing.assert_close(a.encode(g,'security_tpg'),b.encode(g,'security_tpg'),atol=0,rtol=0)
        with torch.no_grad():b.encoder.subtype_project.weight[:,subtypes.VOCAB['SOFTWARE']]=torch.tensor([1.,2.,3.,4.])
        changed=g.clone();changed.subtype[1]=subtypes.VOCAB['VERSION']
        self.assertFalse(torch.allclose(b.encode(g,'security_tpg'),b.encode(changed,'security_tpg')))

    def test_replay_gradients_both_readouts_and_subtype_switch(self):
        sample=self.sample()
        class Packets:
            def packets(self,plan,variant):
                for i in range(2):yield i,Batch.from_data_list([sample])
        plan={'paths':['a','b'],'chunk_cve':torch.tensor([0,1]),'structured':torch.tensor([[.1,.2],[.7,.8]])}
        for readout in ('node_mean','type_balanced'):
            for enabled in (False,True):
                a,b=self.model(readout,enabled),self.model(readout,enabled)
                target=torch.tensor([0.,1.]);weights={0:.8,1:1.2}
                loss=network.loss_for(a.fuse(network.representations(a,Packets(),plan,'security_tpg'),plan),target,weights)
                loss.backward();network.replay_backward(b,Packets(),plan,'security_tpg',target,weights)
                for x,y in zip(a.parameters(),b.parameters()):
                    if x.grad is None:self.assertIsNone(y.grad)
                    else:torch.testing.assert_close(x.grad,y.grad,atol=1e-7,rtol=1e-5)
                if enabled:self.assertGreater(float(b.encoder.subtype_project.weight.grad.abs().sum()),0.)
                else:self.assertIsNone(b.encoder.subtype_project.weight.grad)

    def test_conversion_keeps_exact_embeddings_and_token_nodes(self):
        text='system allows'
        g=graph(text,[node(0,text,'system','','spacy_frontend','TOKEN'),node(1,text,'allows','','spacy_frontend','TOKEN'),
                      node(2,text,'system','CODE_ELEMENT')],[edge(0,0,1,'NEXT_TOKEN'),edge(1,2,0)])
        p=PyGExporter().export(import_graphson(g),embedding_dim=0,use_security_edge_types=True)
        x=torch.cat((torch.tensor(p['x']).float(),torch.arange(6).reshape(3,2).float(),torch.ones(3,1)),1)
        cached=Data(x=x,edge_index=torch.tensor(p['edge_index']),edge_type=torch.tensor(p['edge_type']),
                    edge_features=torch.tensor(p['edge_features']).float(),token_mask=torch.tensor([True,True,False]))
        c,item=v5_cache.convert(g,cached,{'type_vocabulary':p['type_vocabulary']})
        torch.testing.assert_close(item['x'],x[:2],atol=0,rtol=0)
        self.assertEqual(item['edge_index'].shape,(2,1))
        self.assertEqual([n['id'] for n in c['vertices']],[0,1])

    def test_synthetic_four_variant_fit_checkpoint_and_selection(self):
        from epss.tpg_revision import v5_training as training
        sample=self.sample()
        class Batches:
            def plan(self,ids):
                return {'paths':ids,'chunk_cve':torch.arange(len(ids)),
                        'structured':torch.tensor([[.1,.2] if c.endswith('0') else [.7,.8] for c in ids])}
            def packets(self,plan,variant):
                yield 0,Batch.from_data_list([sample for _ in plan['paths']])
        ids={'train':['a0','a1'],'validation':['b0','b1']}
        target={'a0':0,'a1':1,'b0':0,'b1':1}
        settings={**self.p,'learning_rate':.001,'weight_decay':.0001,'batch_size':2,'epochs':2,'patience':2}
        protocol={'settings':settings,'ids':ids,'class_weights':{0:1.,1:1.}}
        with tempfile.TemporaryDirectory() as tmp:
            for variant in training.VARIANTS:
                path=Path(tmp)/variant;path.mkdir()
                rows,metrics=training.fit(path,variant,17,Batches(),target,self.r,2,protocol)
                again=training.verify_fit(path,variant,17,Batches(),target,self.r,2,protocol)
                self.assertEqual((rows,metrics),again)
                with self.assertRaises(ValueError):
                    training.verify_fit(path,variant,29,Batches(),target,self.r,2,protocol)


class ReportingTests(unittest.TestCase):
    def test_no_predicted_positives_preserves_undefined_precision(self):
        from epss.tpg_revision import v5_training as training
        values = training.base.development.metric([0, 1], [.1, .2])
        self.assertIsNone(values['precision'])
        self.assertEqual(values['average_precision'], 1.)
        self.assertEqual(values['f1'], 0.)
        means, counts = training.aggregate_metrics({'v__seed_17': values}, ['v'], [17])
        self.assertIsNone(means['v']['precision'])
        self.assertEqual(counts['v']['precision'],
                         {'defined_seeds': 0, 'total_seeds': 1, 'undefined_seeds': [17]})
        self.assertNotIn('NaN', training.data.dump({'means': means, 'coverage': counts}))

    def test_mixed_undefined_seeds_are_not_silently_dropped(self):
        from epss.tpg_revision import v5_training as training
        undefined = training.base.development.metric([0, 1], [.1, .2])
        defined = training.base.development.metric([0, 1], [.1, .8])
        metrics = {'v__seed_17': undefined, 'v__seed_29': defined}
        means, counts = training.aggregate_metrics(metrics, ['v'], [17, 29])
        self.assertIsNone(means['v']['precision'])
        self.assertEqual(means['v']['f1'], .5)
        self.assertEqual(counts['v']['precision']['defined_seeds'], 1)
        metrics['v__seed_17'] = defined
        self.assertEqual(training.aggregate_metrics(metrics, ['v'], [17, 29])[0]['v']['precision'], 1.)
        metrics['v__seed_17'] = {**defined, 'brier': float('nan')}
        with self.assertRaisesRegex(ValueError, 'Nonfinite metric'):
            training.aggregate_metrics(metrics, ['v'], [17, 29])

    def test_report_recovery_only_accepts_exact_known_predecessor_with_all_fits(self):
        from epss.tpg_revision import v5_training as training
        source = str(Path(training.__file__))
        current = {'settings': {'seeds': [17], 'weight_decay': .0001},
                   'source_pins': {source: 'current', 'cache': 'unchanged'}, 'ids': ['fixed']}
        saved = deepcopy(current)
        saved['source_pins'][source] = training.REPORT_ONLY_PREDECESSOR
        original = deepcopy(saved)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self.assertEqual(training.saved_protocol_for_reports(current, current, path), current)
            with self.assertRaisesRegex(ValueError, 'every fit already completed'):
                training.saved_protocol_for_reports(saved, current, path)
            for variant in training.VARIANTS:
                fit = path/f'{variant}__seed_17'
                fit.mkdir()
                (fit/'artifact_lock.json').write_text('{}')
            self.assertEqual(training.saved_protocol_for_reports(saved, current, path), saved)
            self.assertEqual(saved, original)
            for changed in ('decay', 'cache', 'unknown_code'):
                bad = deepcopy(saved)
                if changed == 'decay': bad['settings']['weight_decay'] = .001
                elif changed == 'cache': bad['source_pins']['cache'] = 'changed'
                else: bad['source_pins'][source] = 'unknown'
                with self.assertRaises(ValueError):
                    training.saved_protocol_for_reports(bad, current, path)


if __name__=='__main__':unittest.main()
