"""Residual-head invariants and tiny synthetic training fixtures only."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_residual as residual
from epss import forecast_sequence as direct
from epss.forecast_data import make_record, read_json, sha256
from epss.forecast_regression import run as run_ridge
from test_forecast_baseline import artifacts, bundle
from test_forecast_data import fixture, set_day, trio
from test_forecast_regression import varied_records


class ResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fit(self, data, seed=17, epochs=2):
        return residual.fit_seed(data, seed, {**residual.CONFIG, 'epochs':epochs, 'patience':2})

    def test_initialization_is_persistence_including_endpoints(self):
        records=[]
        for value in (0.0,.0001,.3,1.0):
            raw=fixture()
            set_day(raw,30,value)
            records.append(make_record(raw)[0])
        x=direct.input_tensor(records)
        base,correction,raw,pred=residual.ResidualGRU().details(x)
        self.assertTrue(torch.equal(base,pred))
        self.assertTrue(torch.equal(raw,pred))
        self.assertEqual(torch.count_nonzero(correction).item(),0)
        self.assertTrue(torch.equal(base,x[:,30,0]))

    def test_missing_day30_uses_latest_observed_not_padding(self):
        raw=fixture()
        set_day(raw,28,.6)
        set_day(raw,29,None)
        set_day(raw,30,None)
        x=direct.input_tensor([make_record(raw)[0]])
        self.assertAlmostEqual(float(residual.ResidualGRU()(x)[0]),.6,places=6)

    def test_observed_zero_not_skipped(self):
        raw=fixture()
        set_day(raw,29,0.0)
        set_day(raw,30,None)
        x=direct.input_tensor([make_record(raw)[0]])
        self.assertEqual(float(residual.ResidualGRU()(x)[0]),0)

    def test_all_missing_rejected(self):
        with self.assertRaisesRegex(ValueError,'No observed'):
            residual.ResidualGRU()(torch.zeros((1,31,5)))

    def test_correction_direction_clipping_and_diagnostics(self):
        raw=fixture()
        x=direct.input_tensor([make_record(raw)[0]])
        net=residual.ResidualGRU()
        for bias,expected in [(10.0,1.0),(-10.0,0.0),(0.0,.1)]:
            with torch.no_grad(): net.head.bias.fill_(bias)
            base,corr,raw,pred=net.details(x)
            self.assertAlmostEqual(float(pred[0]),expected,places=6)
            self.assertTrue(torch.equal(raw,base+corr))
            self.assertTrue(-1 <= float(corr[0]) <= 1)

    def test_endpoint_gradients_at_zero_initialization(self):
        for base,target in ((0.0,.2),(1.0,.8)):
            raw=fixture()
            set_day(raw,30,base)
            net=residual.ResidualGRU()
            pred=net(direct.input_tensor([make_record(raw)[0]]))
            torch.nn.functional.smooth_l1_loss(pred,torch.tensor([target]),beta=.01).backward()
            self.assertNotEqual(float(net.head.bias.grad[0]),0)

    def test_encoder_initialization_matches_direct_head(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            a=direct.ScoreGRU()
            torch.manual_seed(17)
            b=residual.ResidualGRU()
        for key,value in a.gru.state_dict().items():
            self.assertTrue(torch.equal(value,b.gru.state_dict()[key]))

    def test_future_mutation_does_not_affect_predictions(self):
        data=artifacts(varied_records())
        model,_,_=self.fit(data)
        raw=fixture()
        a=residual.predict(model,[make_record(raw)[0]])
        for day in range(31,91): set_day(raw,day,.9)
        raw.update(in_vckev=True,description='Future text')
        self.assertEqual(a,residual.predict(model,[make_record(raw)[0]]))

    def test_heldout_values_do_not_affect_training_updates(self):
        records=varied_records()
        a,_,_=self.fit(artifacts(records),epochs=1)
        for i in (1,2):
            for day in range(91): set_day(records[i],day,.9)
        b,_,_=self.fit(artifacts(records),epochs=1)
        self.assertEqual(a,b)

    def test_test_values_do_not_affect_selection(self):
        records=varied_records()
        a=self.fit(artifacts(records))
        for day in range(91): set_day(records[2],day,.9)
        self.assertEqual(a,self.fit(artifacts(records)))

    def test_determinism_and_caller_rng_state(self):
        data=artifacts(varied_records())
        state=torch.get_rng_state().clone()
        a=self.fit(data)
        self.assertEqual(a,self.fit(data))
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_epoch_zero_not_selected_and_reload_exact(self):
        data=artifacts(varied_records())
        model,history,initial=self.fit(data)
        self.assertFalse(initial['eligible_for_selection'])
        self.assertTrue(initial['zero_correction_verified'])
        self.assertEqual(model['selected_epoch'],min(history,key=lambda r:r['validation_mae'])['epoch'])
        self.assertGreaterEqual(model['selected_epoch'],1)
        x=[data['inputs.json']['CVE-2024-1234']]
        self.assertEqual(residual.predict(model,x),residual.predict(json.loads(json.dumps(model)),x))

    def test_invalid_saved_model_rejected(self):
        model,_,_=self.fit(artifacts(trio()))
        for key,value in [('schema',direct.VERSION),('selected_epoch',0),('epoch_zero_eligible_for_selection',True),
                           ('channels',['future_score']),('head_rule','unbounded')]:
            bad=copy.deepcopy(model)
            bad[key]=value
            with self.assertRaises(ValueError): residual.restore_model(bad)
        model['state_dict']['head.bias'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Non-finite'): residual.restore_model(model)

    def test_evaluation_test_access_and_missing_seeds_rejected(self):
        data=artifacts(trio())
        for split in ('test','train'):
            with self.assertRaisesRegex(ValueError,'Only validation'): residual.evaluate(data,[],[],split)
        with self.assertRaisesRegex(ValueError,'All fixed seeds'): residual.evaluate(data,[],[])

    def test_complete_toy_cli_flow_and_check_only_never_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            prepared=bundle(tmp)
            ridge=Path(tmp)/'ridge'
            reference=Path(tmp)/'direct'
            output=Path(tmp)/'residual'
            run_ridge(prepared,ridge)
            # Limit only these tiny synthetic fixtures, never the real dataset run.
            with patch.dict(direct.CONFIG,{'epochs':2,'patience':2}), patch.dict(residual.CONFIG,{'epochs':2,'patience':2}):
                direct.run(prepared,ridge,reference)
                before={p.name:sha256(p) for p in reference.iterdir()}
                with patch.object(residual,'fit_seed',side_effect=AssertionError('Check-only attempted training')):
                    self.assertIsNone(residual.run(prepared,reference,output,check_only=True))
                self.assertFalse(output.exists())
                result=residual.run(prepared,reference,output)
                self.assertFalse(result['test_performance_evaluated'])
                self.assertEqual(len(result['per_seed']),3)
                self.assertEqual({p.name for p in output.iterdir()},residual.OUTPUT_FILES)
                lock=read_json(output/'evaluation_lock.json')
                self.assertTrue(all(sha256(output/k)==v for k,v in lock['files'].items()))
                self.assertEqual(before,{p.name:sha256(p) for p in reference.iterdir()})
                self.assertEqual(result,residual.run(prepared,reference,output,overwrite=True))
                with self.assertRaisesRegex(ValueError,'already exists'): residual.run(prepared,reference,output)
                with self.assertRaisesRegex(ValueError,'Unrelated files'): residual.run(prepared,reference,ridge,overwrite=True)
                (reference/'predictions.json').write_text('[]')
                with self.assertRaisesRegex(ValueError,'artifact changed'):
                    residual.run(prepared,reference,Path(tmp)/'new_residual',check_only=True)


if __name__ == '__main__':
    unittest.main()
