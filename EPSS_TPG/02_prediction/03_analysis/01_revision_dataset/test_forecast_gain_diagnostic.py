"""Synthetic GD01 tests; no real-data full training or held-out scoring."""
from contextlib import redirect_stdout,redirect_stderr
from copy import deepcopy
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_gain_diagnostic as run
import test_forecast_day60_pipeline as old_tests


def fixture():
    ids = ['a','b','c','d','e','f']
    inputs = {c:old_tests.inputs(c) for c in ids}
    labels = {c:{'cve':c,'target_day':60,'horizon_days':30,'target_observed':True,
                 'y':{'endpoint':(.1,.2,.05)[i%3]}} for i,c in enumerate(ids)}
    selected,audit = run.select_training(ids,inputs,labels,1,'synthetic')
    identity = old_tests.identity()
    identity.pop('ids')
    identity.update(schema=run.SCHEMA,mode='smoke',protocol=run.read(run.CONFIG),original_gain=identity['gain'],
                    diagnostic_ids=selected,original_training_ids=ids,validation_scoring=False)
    x = run.old.tensor_inputs([inputs[c] for c in selected])
    labels = {c:{**labels[c],'diagnostic_movement':audit['selected'][c]['movement']} for c in selected}
    store = run.SelectedStore(old_tests.legacy.MemoryStore(),selected)
    return identity,x,labels,store,audit


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        out,err = redirect_stdout(StringIO()),redirect_stderr(StringIO())
        out.__enter__(); err.__enter__()
        self.addCleanup(out.__exit__,None,None,None)
        self.addCleanup(err.__exit__,None,None,None)
        self.i,self.x,self.labels,self.store,self.audit = fixture()

    def test_selection_training_only_reproducible_and_outcome_balanced(self):
        self.assertEqual(self.i['diagnostic_ids'],fixture()[0]['diagnostic_ids'])
        self.assertEqual(self.audit['selected_movement_counts'],{'unchanged':1,'increased':1,'decreased':1})
        inputs={c:old_tests.inputs(c) for c in self.i['diagnostic_ids']}
        with self.assertRaisesRegex(ValueError,'Insufficient'):
            run.select_training(self.i['diagnostic_ids'],inputs,self.labels,10,'salt')
        inputs['test']={'do_not_parse':True}
        labels={**self.labels,'test':{'do_not_parse':True}}
        selected,_=run.select_training(self.i['diagnostic_ids'],inputs,labels,1,'salt')
        self.assertNotIn('test',selected)

    def test_forbidden_graph_access_and_nonmutating_edge_removal(self):
        with self.assertRaises(ValueError): self.store.documents('test',run.old.model.SOURCES)
        with self.assertRaises(ValueError): self.store.load('not_selected')
        name=next(iter(self.store.paths))
        original=self.store.load(name)
        changed=run.InterventionStore(self.store,no_edges=True).load(name)
        self.assertEqual(changed['edge_index'].shape,(2,0))
        self.assertGreater(original['edge_index'].shape[1],0)

    def test_gain_does_not_change_weights_and_initialization_is_persistence(self):
        for variant in self.i['protocol']['variants']:
            models=[run.initialize(self.i,variant,17,c) for c in ('original','unit')]
            self.assertEqual(run.old.controls.state_digest(models[0].state_dict()),run.old.controls.state_digest(models[1].state_dict()))
            for net in models:
                parts=net.details(self.x,self.store,self.i['diagnostic_ids'])
                self.assertTrue(torch.equal(parts[0],parts[3]))

    def test_gain_derivative_is_not_a_hard_correction_cap(self):
        net=run.initialize(self.i,'epss_only',17,'original')
        with torch.no_grad(): net.head[-1].bias.fill_(100)
        parts=net.details(self.x,self.store,self.i['diagnostic_ids'])
        self.assertGreater(float(parts[1][0]),net.gain)
        parts[1].sum().backward()
        expected=len(self.x)*net.gain*(1-float(parts[1][0])**2)
        self.assertAlmostEqual(float(net.head[-1].bias.grad),expected,places=6)

    def test_decay_gradient_ratios_and_unused_parameters(self):
        net=run.initialize(self.i,'epss_only',17,'unit')
        parts=net.details(self.x,self.store,self.i['diagnostic_ids'])
        parts[3].sum().backward()
        result=run.gradient_stats(net,.0001)
        self.assertGreater(result['head']['data_gradient_l2'],0)
        self.assertTrue(result['temporal']['decay_nonzero_data_zero'])
        self.assertIsNone(result['temporal']['decay_over_data_ratio'])
        self.assertEqual(result['node_project']['adam_coupled_decay_term_l2'],0)
        self.assertGreater(result['node_project']['parameters_without_gradient'],0)

    def test_instrumented_training_matches_uninstrumented_optimizer_steps(self):
        identity=deepcopy(self.i)
        identity['config'].update(epochs=1,batch_size=3)
        variant,seed,condition='all_sources_epss',17,'original'
        net=run.initialize(identity,variant,seed,condition)
        optimizer=torch.optim.Adam(net.parameters(),lr=identity['config']['learning_rate'],weight_decay=identity['config']['weight_decay'])
        ids=identity['diagnostic_ids']
        order=run.old.controls.orders(ids,identity['config'],seed)[0]
        ix=[ids.index(c) for c in order]
        target=torch.tensor([self.labels[c]['y']['endpoint'] for c in order])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            net.train()
            loss=torch.nn.functional.l1_loss(net(self.x[ix],self.store,order),target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),identity['config']['gradient_clip_norm'])
            optimizer.step()
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            run.fit(d,identity,self.x,self.labels,self.store,variant,seed,condition,run.Log(),10)
            saved=torch.load(d/'checkpoint.pt',weights_only=True)
            for k,v in net.state_dict().items():
                torch.testing.assert_close(v,saved['final_state'][k],rtol=0,atol=0)

    def test_graph_interventions_leave_epss_only_unchanged(self):
        net=run.initialize(self.i,'epss_only',17,'unit')
        result=run.interventions(net,self.i,self.x,self.labels,self.store)
        for name,r in result['probes'].items():
            self.assertEqual(r['absolute_prediction_movement']['max'],0.)
        self.assertEqual({k:v['n'] for k,v in result['unmodified_training_metrics']['by_training_movement'].items()},
                         {'unchanged':1,'increased':1,'decreased':1})

    def test_complete_lifecycle_resume_and_artifact_corruption(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run,'OUTPUT',Path(tmp)/'results'), \
             patch.object(run,'context',side_effect=lambda *args:fixture()):
            run.execute('smoke',1,0,check_only=True)
            self.assertFalse(run.OUTPUT.exists())
            run.execute('smoke',1,0)
            run.execute('smoke',1,0,verify_only=True)
            with self.assertRaisesRegex(ValueError,'Run exists'): run.execute('smoke',1,0)
            with self.assertRaisesRegex(ValueError,'complete resume'): run.execute('smoke',1,0,resume=True)
            (run.OUTPUT/'smoke/diagnostic_lock.json').unlink()
            with patch.object(run,'fit',side_effect=AssertionError('do not retrain completed fits')):
                run.execute('smoke',1,0,resume=True)
            path=next((run.OUTPUT/'smoke/fits').glob('*/predictions.json'))
            value=run.read(path); value['best'][0]['prediction']=.999
            run.write(path,value)
            with self.assertRaisesRegex(ValueError,'Changed/unsafe artifact'): run.execute('smoke',1,0,verify_only=True)


if __name__=='__main__': unittest.main()
