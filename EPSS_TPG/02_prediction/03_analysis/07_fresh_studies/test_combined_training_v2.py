"""Execution parity, immutable inputs, loss selection and best-only checkpoint tests."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from test_combined_training import fixture
from epss.combined_training_v1 import runner as old, data
from epss.combined_training_v2 import pooled, runner, training


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.p = runner.settings({'hidden':8,'threads':1,'edge_chunk_size':2,'epochs':3,'patience':5,'seeds':[17]})
        self.contract, _, self.graph = fixture()
        self.inputs = {'graph':self.graph, 'structured':torch.tensor([.2]), 'source_mask':torch.tensor([1.,0,0,0])}

    def optimized(self):
        vector, observed = pooled.reduce_graph(self.graph, self.contract)
        return {**self.inputs, 'graph':{'pooled_vector':vector,'pooled_observed':observed}}

    def test_forward_gradient_and_optimizer_bitwise_parity(self):
        for rep in ('pooled','tpg','structured'):
            a = old.initialize(self.contract,self.p,rep,17)
            b = training.initialize(self.contract,self.p,rep,17)
            inp = self.inputs if rep != 'structured' else {**self.inputs,'graph':None,'source_mask':torch.zeros(4)}
            opt_a = torch.optim.AdamW(a.parameters(),lr=.001,weight_decay=.0001)
            opt_b = torch.optim.AdamW(b.parameters(),lr=.001,weight_decay=.0001)
            for _ in range(4):
                opt_a.zero_grad();opt_b.zero_grad()
                x = a(inp)
                y = b(self.optimized() if rep=='pooled' else inp)
                self.assertTrue(torch.equal(x,y), rep)
                (x-.2).abs().backward();(y-.2).abs().backward()
                for pa,pb in zip(a.parameters(),b.parameters()):
                    self.assertTrue(torch.equal(pa.grad,pb.grad), rep)
                opt_a.step();opt_b.step()
                for pa,pb in zip(a.parameters(),b.parameters()):
                    self.assertTrue(torch.equal(pa,pb), rep)

    def test_pooled_empty_embeddings_and_tpg_guard(self):
        self.graph['embedding_mask'].zero_()
        a = old.initialize(self.contract,self.p,'pooled',17)
        b = training.initialize(self.contract,self.p,'pooled',17)
        self.assertTrue(torch.equal(a(self.inputs),b(self.optimized())))
        tpg = training.initialize(self.contract,self.p,'tpg',17)
        with self.assertRaisesRegex(ValueError,'not a TPG'):
            tpg(self.optimized())

    def test_no_parameter_or_vocab_changes(self):
        for rep in ('pooled','tpg'):
            a = old.initialize(self.contract,self.p,rep,17)
            b = training.initialize(self.contract,self.p,rep,17)
            self.assertEqual(list(a.state_dict()),list(b.state_dict()))
            b.load_state_dict(a.state_dict())
            wrong = deepcopy(self.contract)
            wrong['edge_type_vocab'] = {'SEC_HAS_VERSION':1,'DEP':0}
            with self.assertRaisesRegex(ValueError,'vocabulary'):
                training.initialize(wrong,self.p,rep,17).load_state_dict(a.state_dict())

    def test_input_target_and_missingness_guards_preserved(self):
        net = training.initialize(self.contract,self.p,'pooled',17)
        with self.assertRaisesRegex(ValueError,'targets'):
            net({**self.optimized(),'target':.5})
        with self.assertRaisesRegex(ValueError,'missingness'):
            net({**self.optimized(),'source_mask':torch.zeros(4)})

    def test_cpu_and_objective_gate(self):
        self.assertEqual(training.selection_metric(self.p),'mae')
        self.assertEqual(training.selection_metric(runner.settings({'objective':'mse'})),'rmse')
        for bad in ({'objective':'logit'},{'device':'cuda'}):
            with self.assertRaises(ValueError):
                runner.settings(bad)

    def test_per_fit_lock_excludes_another_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'fit.lock'
            with runner.exclusive(path):
                with self.assertRaisesRegex(ValueError,'writer'):
                    with runner.exclusive(path):
                        pass
            with runner.exclusive(path):
                pass

    def context_and_store(self):
        ctx = {'ids':{'train':['a','b'],'validation':['c','d']}, 'audit':{},
               'labels':{k:{'value':.7,'date':'2024-01-01'} for k in 'abcd'}}
        class Store:
            def inputs(self,cid,view,device):
                return {'graph':None,'source_mask':torch.zeros(4),'structured':torch.tensor([.2])}
            def profile(self):
                return {}
        return ctx, Store()

    def test_mae_resume_and_only_best_checkpoint(self):
        ctx, store = self.context_and_store()
        ident = {'settings':self.p,'scale':'full','contract':self.contract,'source_pins':{}}
        with tempfile.TemporaryDirectory() as tmp:
            left, right = Path(tmp)/'left', Path(tmp)/'right'
            expected = training.fit(left,ctx,store,ident,'structured','S',17)
            original = training.train_epoch
            def stop(*args):
                if args[7]==3:
                    raise InterruptedError('test')
                return original(*args)
            with patch.object(training,'train_epoch',stop):
                with self.assertRaises(InterruptedError):
                    training.fit(right,ctx,store,ident,'structured','S',17)
            actual = training.fit(right,ctx,store,ident,'structured','S',17,resume=True)
            self.assertEqual(expected,actual)
            training.fit(right,ctx,store,ident,'structured','S',17,verify=True)
            self.assertEqual([p.name for p in right.glob('*.pt')],['best_checkpoint.pt'])

    def test_mse_uses_rmse_selection_and_earliest_tie(self):
        p = {**self.p,'objective':'mse'}
        ident = {'settings':p,'scale':'full','contract':self.contract,'source_pins':{}}
        rows = [{'metrics':{'all':{'mae':m,'rmse':r}},'predictions':[]}
                for m,r in ((.1,.2),(.3,.1),(.2,.1),(.3,.1))]
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)/'fit'
            with patch.object(training,'train_epoch',return_value=(.1,0.)), patch.object(old,'evaluate',side_effect=rows):
                training.fit(dest,{},type('Store',(),{'profile':lambda _: {}})(),ident,'structured','S',17)
            cp = torch.load(dest/'best_checkpoint.pt',weights_only=True)
            self.assertEqual(cp['best_epoch'],2)
            self.assertEqual(cp['selection_metric'],'rmse')

    def test_reuse_forbids_objective_and_arithmetic_changes(self):
        saved = {'fit':'pooled/D/seed_17','dataset':'previous','scale':'full','contract':{},'provenance':{},
                 'runtime':{},'test_access':False,'source_pins':{},'settings':old.settings({})}
        current = {**saved,'settings':runner.settings({})}
        training.reference_compatible(saved,current,'pooled','D',17)
        for key,value in (('objective','mse'),('threads',2),('layers',3)):
            changed = deepcopy(current);changed['settings'][key] = value
            with self.assertRaises(ValueError):
                training.reference_compatible(saved,changed,'pooled','D',17)

    def test_partial_cache_resume_corruption_and_no_test_ids(self):
        ctx = {'contract':self.contract,'provenance':{'inputs':{}},'receipts':{'one':['a','b']}}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/'vectors'
            with patch.object(data.GraphStore,'load',return_value=self.graph) as loader:
                actual = pooled.prepare(ctx,directory,1)
                self.assertEqual(loader.call_count,1)
                pooled.prepare(ctx,directory,1,verify=True)
                self.assertEqual(loader.call_count,1)
            self.assertEqual(set(actual),{'one'})
            (directory/'rows/one.json').write_text('{}')
            with self.assertRaises(ValueError):
                pooled.load(ctx,directory,1)


if __name__=='__main__':
    unittest.main()
