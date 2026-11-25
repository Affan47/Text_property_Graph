"""Synthetic R07 checks. No dataset fitting."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_mae_development as run
from test_forecast_coreference_smoke import setup


def tiny():
    p = run.protocol()
    p['config'].update(epochs=2,patience=2,seeds=[17])
    return p


class FullMAETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = run.torch.get_num_threads()
        run.torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        run.torch.set_num_threads(cls.threads)

    def test_fixed_handoff_settings(self):
        p = run.protocol()
        self.assertEqual(p['loss'],'mae')
        self.assertEqual((p['config']['epochs'],p['config']['patience']),(100,15))
        self.assertEqual(p['config']['seeds'],[17,29,43])
        self.assertEqual(p['config']['weight_decay'],.0001)
        self.assertEqual(p['cohort']['train'],1023)
        self.assertFalse(p['test_scoring'])

    def test_r06_update_parity_when_not_early_stopped(self):
        m,views,d = setup(); p = tiny()
        for model in run.MODELS:
            a,h,g = run.r06.fit_one(model,'mae',17,m,views[model],d,p,False)
            b,h2,g2 = run.fit_one(model,17,m,views[model],d,p,False)
            self.assertEqual(h,h2); self.assertEqual(g,g2)
            self.assertEqual(run.prior.digest_state(a['state_dict']),run.prior.digest_state(b['state_dict']))
            self.assertEqual(a['selected_epoch'],b['selected_epoch'])

    def test_strict_tie_early_stopping(self):
        m,views,d = setup(); p = tiny()
        p['config'].update(epochs=8,patience=2)
        with patch.object(run.task,'infer',side_effect=lambda net,ids,*a,**kw:[.5]*len(ids)):
            saved,h,g = run.fit_one('no_coref',17,m,views['no_coref'],d,p,False)
        self.assertEqual(len(h),3)
        self.assertEqual(saved['selected_epoch'],1)
        run.history_check(h,run.settings(p,17),1)
        with self.assertRaisesRegex(ValueError,'stopped too early'):
            run.history_check(h[:2],run.settings(p,17),1)
        with self.assertRaisesRegex(ValueError,'past stopping rule'):
            run.history_check(h+[{**h[-1],'epoch':4}],run.settings(p,17),1)
        self.assertTrue(all(v['steps_with_gradient']==3 for v in g['all_steps'].values()))

    def test_seed_and_state_repeatability(self):
        m,views,d = setup(); p = tiny()
        a,h,g = run.fit_one('epss_only',17,m,views['epss_only'],d,p,False)
        b,h2,g2 = run.fit_one('epss_only',17,m,views['epss_only'],d,p,False)
        self.assertEqual(h,h2); self.assertEqual(g,g2)
        self.assertEqual(run.prior.digest_state(a['state_dict']),run.prior.digest_state(b['state_dict']))
        initial = run.initialize('epss_only',m,views['epss_only'],run.settings(p,17))
        self.assertEqual(run.assess(a,initial,m,views['epss_only'],d),
                         run.assess(b,initial,m,views['epss_only'],d))

    def test_input_and_split_isolation(self):
        m,views,d = setup()
        batch = run.task.make_batch(m['ids']['train'],views['no_coref'],d['inputs.json'])
        self.assertFalse({'y','target_date','cve'}.intersection(batch.keys()))
        bad = deepcopy(m); bad['ids']['test'] = bad['ids'].pop('validation')
        with self.assertRaisesRegex(ValueError,'Only train/validation'):
            run.prior.smoke.select_data(bad,d)
        changed = deepcopy(d['inputs.json'])
        changed[m['ids']['train'][0]]['X']['future_epss'] = .9
        with self.assertRaisesRegex(ValueError,'Unexpected predictor'):
            run.task.make_batch(m['ids']['train'],views['no_coref'],changed)

    def test_partial_input_history_allowed(self):
        m,views,d = setup()
        altered = deepcopy(d['inputs.json'])
        c = m['ids']['train'][0]
        x = altered[c]['X']
        # The shared fixture already has partial histories; do not add a complete-window requirement.
        self.assertTrue(any(sum(altered[k]['X']['observed_mask'])<31 for k in m['ids']['train']))
        batch = run.task.make_batch(m['ids']['train'],views['no_coref'],altered)
        self.assertEqual(batch.epss_sequence.shape[1],31)

    def test_lifecycle_recovery_verification_and_tamper(self):
        m,views,d = setup(); p = tiny()
        refs = {run.fit_name(v,17):run.prior.score_predictions([.1]*len(m['ids']['validation']),m,d)
                for v in run.MODELS}
        identity = {'schema':run.VERSION,'protocol':p,'ids':m['ids'],'pinned_files':{},
                    'r06_decision':{'selected_loss':'mae'},'r04_references':refs}
        with tempfile.TemporaryDirectory() as folder:
            source,output = Path(folder)/'r04',Path(folder)/'r07'
            source.mkdir()
            for v in run.MODELS:
                init = run.initialize(v,m,views[v],run.settings(p,17))
                mode = 'epss_only' if v=='epss_only' else 'tpg_epss'
                run.prior.atomic_torch_save(source/f'initialization/seed_17/{mode}.pt',init)
            run.write(source/'orders_seed_17.json',run.prior.epoch_orders(m['ids']['train'],run.settings(p,17)))
            with patch.object(run,'context',return_value=(m,views,d,identity)),patch.object(run.prior,'DEFAULT_OUTPUT',source):
                with patch.object(run,'fit_one',side_effect=AssertionError('Preflight must not fit')):
                    run.execute(check_only=True,output=output)
                self.assertFalse(output.exists())
                original = run.fit_one
                calls = []
                def interrupted(*a,**kw):
                    calls.append(a[0])
                    if len(calls)==2:
                        raise RuntimeError('interruption')
                    return original(*a,**kw,progress=False)
                with patch.object(run,'fit_one',side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError,'interruption'):
                        run.execute(output=output)
                with patch.object(run,'fit_one',side_effect=lambda *a,**kw:original(*a,**kw,progress=False)):
                    run.execute(resume=True,output=output)
                with patch.object(run,'fit_one',side_effect=AssertionError('Verify must not fit')):
                    run.execute(verify_only=True,output=output)
                with self.assertRaisesRegex(ValueError,'Completed output'):
                    run.execute(resume=True,output=output)
                with self.assertRaisesRegex(ValueError,'Output exists'):
                    run.execute(output=output)
                (output/'summary.txt').write_text('changed')
                with self.assertRaisesRegex(ValueError,'Changed source/artifact'):
                    run.execute(verify_only=True,output=output)


if __name__ == '__main__':
    unittest.main()
