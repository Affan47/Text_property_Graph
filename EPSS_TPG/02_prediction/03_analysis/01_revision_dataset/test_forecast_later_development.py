"""Focused synthetic tests: H02 performs no real-dataset training here."""
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_later_development as data
from epss import forecast_later_controls as run

torch = run.torch


def example(c='CVE-SYNTHETIC', values=None):
    values = list(values) if values is not None else [.01+i/10000 for i in range(91)]
    pub = date(2022,1,1)
    dates = [(pub+timedelta(days=i)).isoformat() for i in range(91)]
    return {'cve':c,'publication_date':dates[0],'input_end_date':dates[60],'anchor_day':60,
            'histories':{v:data.h01.history(dates,values,*bounds)
                         for v,bounds in zip(data.VARIANTS,((30,60),(0,60)))}}


class HistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_shapes_and_shared_time_scale(self):
        x = example()
        recent = run.input_tensor([x],'recent_30_60')
        longer = run.input_tensor([x],'long_0_60')
        self.assertEqual(tuple(recent.shape),(1,31,5))
        self.assertEqual(tuple(longer.shape),(1,61,5))
        self.assertTrue(torch.equal(recent,longer[:,30:]))
        self.assertEqual(recent[0,0,4],.5)

    def test_missing_is_not_zero_and_no_early_age_leak(self):
        v = [None]*91
        v[2],v[35] = .9,0.
        x = example(values=v)
        recent = run.input_tensor([x],data.VARIANTS[0])
        longer = run.input_tensor([x],data.VARIANTS[1])
        self.assertEqual(recent[0,0,3],0)
        self.assertEqual(longer[0,30,3],1)
        self.assertEqual(recent[0,5,1],1)
        self.assertEqual(recent[0,5,0],0)
        self.assertEqual(recent[0,-1,2],25/60)
        self.assertEqual(run.persistence(x),0)

    def test_future_and_other_text_fields_cannot_enter_tensor(self):
        v = [.02]*91
        x = example(values=v)
        v[61:] = [.99]*30
        other = example(values=v)
        other.update(y=.99, in_vckev=True, description='future information')
        for name in data.VARIANTS:
            self.assertTrue(torch.equal(run.input_tensor([x],name),run.input_tensor([other],name)))

    def test_reject_bad_masks_dates_lengths_and_age(self):
        for field, value in [('observed_mask',[0]*31),('age_known_mask',[0]*31),
                             ('age_since_observation_days',[99]*31),('values',[.1]*30),
                             ('day_offsets',list(range(31))),('dates',['2022-01-01']*31)]:
            x = example()
            x['histories']['recent_30_60'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                run.input_tensor([x],'recent_30_60')

    def test_reject_nonfinite_empty_and_unknown_history(self):
        for bad in (float('nan'),float('inf'),1.2,-.1):
            x = example()
            x['histories']['recent_30_60']['values'][0] = bad
            with self.assertRaises(ValueError):
                run.input_tensor([x],'recent_30_60')
        with self.assertRaises(ValueError):
            run.input_tensor([example(values=[None]*91)],'recent_30_60')
        with self.assertRaises(ValueError):
            run.input_tensor([example()],'future')

    def test_matched_initialization_and_missing_anchor_persistence(self):
        p,cfg = run.configuration('smoke')
        values = [.02]*91
        values[60] = None
        x = example(values=values)
        nets = [run.initialize(cfg,.01,17) for _ in data.VARIANTS]
        self.assertEqual(run.state_digest(nets[0].state_dict()),run.state_digest(nets[1].state_dict()))
        for net,v in zip(nets,data.VARIANTS):
            base,correction,_,pred = net.details(run.input_tensor([x],v))
            self.assertTrue(torch.equal(base,pred))
            self.assertEqual(correction.item(),0)
            self.assertAlmostEqual(pred.item(),.02)

    def test_gain_uses_training_inputs_only(self):
        inputs = {'train':example(values=[.02]*91),'validation':example(values=[.99]*91)}
        self.assertEqual(run.training_gain(['train'],inputs),.02)
        inputs['validation'] = example(values=[0]*91)
        self.assertEqual(run.training_gain(['train'],inputs),.02)

    def test_seed_schedule_reproducible_and_complete(self):
        _,cfg = run.configuration('smoke')
        ids = ['a','b','c','d','e']
        a = run.orders(ids,cfg,17)
        self.assertEqual(a,run.orders(ids,cfg,17))
        self.assertNotEqual(a,run.orders(ids,cfg,29))
        self.assertTrue(all(sorted(o)==ids for o in a))

    def test_state_initialization_preserves_rng(self):
        _,cfg = run.configuration('smoke')
        before = torch.random.get_rng_state().clone()
        run.initialize(cfg,.01,17)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))

    def test_selection_and_stopping_validation(self):
        _,cfg = run.configuration('smoke')
        h = [{'epoch':i+1,'training_objective_mae':.1,'training_mae':.1,
              'validation_mae':v,'temporal_data_gradient_abs_sum':0.,'head_data_gradient_abs_sum':1.}
             for i,v in enumerate((.1,.1,.2))]
        run.validate_history(h,cfg,1)
        with self.assertRaises(ValueError):
            run.validate_history(h,cfg,2)
        with self.assertRaises(ValueError):
            run.validate_history(h[:1],cfg,1)

    def synthetic_context(self):
        p,cfg = run.configuration('smoke')
        ids = {'train':['a','b'],'validation':['v']}
        x = {c:example(c, [.01+i*.01]*91) for i,c in enumerate(['a','b','v'])}
        labels = {c:{'y':{'endpoint':.04}} for c in x}
        ident = {'config':cfg,'gain':.015,'ids':ids,'scale':'smoke','protocol':p}
        tensors = {v:{s:run.input_tensor([x[c] for c in chosen],v) for s,chosen in ids.items()}
                   for v in data.VARIANTS}
        return ident,tensors,labels

    def test_synthetic_fit_reload_and_tamper_detection(self):
        ident,tensors,labels = self.synthetic_context()
        v = data.VARIANTS[0]
        saved,history = run.fit(v,17,ident,tensors[v],labels,False)
        second,h2 = run.fit(v,17,ident,tensors[v],labels,False)
        self.assertEqual(history,h2)
        self.assertEqual(run.state_digest(saved['state_dict']),run.state_digest(second['state_dict']))
        self.assertGreater(sum(r['head_data_gradient_abs_sum'] for r in history),0)
        rows,metrics = run.evaluate(saved,ident,tensors[v]['validation'],labels)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            run.save_checkpoint(path/'checkpoint.pt',saved)
            for n,val in [('history.json',history),('predictions.json',rows),('metrics.json',metrics)]:
                run.write(path/n,val)
            run.write(path/'fit_lock.json',{'schema':run.SCHEMA,
                      'files':{n:run.sha256(path/n) for n in run.FIT_FILES}})
            checked = run.check_fit(path,v,17,ident,tensors[v]['validation'],labels)
            self.assertEqual(checked['metrics'],metrics['metrics'])
            run.write(path/'predictions.json',[])
            with self.assertRaises(ValueError):
                run.check_fit(path,v,17,ident,tensors[v]['validation'],labels)

    def test_long_history_fit_and_output_bounds(self):
        ident,tensors,labels = self.synthetic_context()
        v = data.VARIANTS[1]
        saved,history = run.fit(v,17,ident,tensors[v],labels,False)
        run.validate_history(history,ident['config'],saved['selected_epoch'])
        rows,_ = run.evaluate(saved,ident,tensors[v]['validation'],labels)
        self.assertTrue(all(0 <= r['prediction'] <= 1 for r in rows))

    def test_development_rejects_test_role(self):
        with self.assertRaisesRegex(ValueError,'Development roles only'):
            data.validate({'run_identity.json':{'ids':{'train':['a'],'validation':['v'],'test':['t']}}})

    def test_resume_preserves_locked_fit_and_verifies_run(self):
        ident,tensors,labels = self.synthetic_context()
        ident.update(pinned_files={},schema=run.SCHEMA)
        x = {c:example(c) for c in ['a','b','v']}
        a = {'inputs.json':x,'labels.json':labels}
        original = run.fit
        calls = []

        def interrupted(v,*args,**kwargs):
            calls.append(v)
            if len(calls) == 2:
                raise RuntimeError('synthetic interruption')
            kwargs['progress'] = False
            return original(v,*args,**kwargs)

        with tempfile.TemporaryDirectory() as tmp, patch.object(run,'OUTPUT',Path(tmp)/'results'), \
                patch.object(run.data,'load',return_value=a), \
                patch.object(run,'make_context',return_value=(ident,tensors)), \
                patch.object(run,'tqdm',side_effect=lambda it,**kw:it):
            # tqdm.write is used during fitting only when progress=True.
            with patch.object(run,'fit',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'synthetic interruption'):
                    run.execute('smoke')
            path = run.OUTPUT/'smoke'
            checkpoint = path/'fits/recent_30_60__17/checkpoint.pt'
            digest = run.sha256(checkpoint)
            with patch.object(run,'fit',side_effect=lambda *args,**kw:original(*args,**kw,progress=False)):
                run.execute('smoke',resume=True)
            self.assertEqual(digest,run.sha256(checkpoint))
            with patch.object(run,'fit',side_effect=AssertionError('Verification must not fit')):
                run.execute('smoke',verify_only=True)
            run.write(path/'orders_seed_17.json',[])
            with self.assertRaises(ValueError):
                run.execute('smoke',verify_only=True)


if __name__ == '__main__':
    unittest.main()
