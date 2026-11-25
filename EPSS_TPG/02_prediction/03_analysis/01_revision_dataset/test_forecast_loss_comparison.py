"""Synthetic checks for R06. No dataset model training."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_loss_comparison as run
from test_forecast_coreference_smoke import setup


def tiny():
    p = run.protocol()
    p['config'].update(epochs=2,patience=2,seeds=[17])
    return p


class LossComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = run.torch.get_num_threads()
        run.torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        run.torch.set_num_threads(cls.threads)

    def test_fixed_settings_and_handoff(self):
        p = run.protocol()
        self.assertEqual(p['config'],run.r05.protocol()['config'])
        self.assertEqual(p['config']['weight_decay'],.0001)
        self.assertFalse(p['full_training_decision']['additional_bounded_tuning'])
        self.assertFalse(p['full_training_decision']['test_scoring'])

    def test_losses_and_zero_error_gradient(self):
        c = run.settings(tiny(),17)
        x = run.torch.tensor([0.,.02],requires_grad=True)
        y = run.torch.zeros(2)
        mae = run.criterion('mae',c)(x,y)
        huber = run.criterion('smooth_l1',c)(x,y)
        self.assertAlmostEqual(float(mae),.01)
        self.assertAlmostEqual(float(huber),.0075)
        mae.backward()
        self.assertEqual(x.grad.tolist(),[0.,.5])
        with self.assertRaises(ValueError):
            run.criterion('unknown',c)

    def test_exact_smooth_l1_algorithm_parity(self):
        m,views,d = setup(); p = tiny()
        for model in run.MODELS:
            old,h,g = run.r05.fit_one(model,'original_decay',17,m,views[model],d,p,False)
            new,h2,g2 = run.fit_one(model,'smooth_l1',17,m,views[model],d,p,False)
            self.assertEqual(run.prior.digest_state(old['state_dict']),run.prior.digest_state(new['state_dict']))
            self.assertEqual(h,run.normalized_control_history(h2))
            self.assertEqual(g,g2)
            self.assertEqual(old['selected_epoch'],new['selected_epoch'])
            self.assertEqual(old['initial_state_sha256'],new['initial_state_sha256'])

    def test_mae_deterministic_fixed_epochs_reload(self):
        m,views,d = setup(); p = tiny()
        a,h,g = run.fit_one('no_coref','mae',17,m,views['no_coref'],d,p,False)
        b,h2,g2 = run.fit_one('no_coref','mae',17,m,views['no_coref'],d,p,False)
        self.assertEqual(len(h),2); self.assertEqual(h,h2); self.assertEqual(g,g2)
        self.assertEqual(run.prior.digest_state(a['state_dict']),run.prior.digest_state(b['state_dict']))
        self.assertEqual(a['selected_epoch'],min(h,key=lambda r:r['validation_mae'])['epoch'])
        self.assertEqual(a['spec']['config']['weight_decay'],.0001)
        initial = run.initialize('no_coref',m,views['no_coref'],run.settings(p,17))
        values,_ = run.assess(a,initial,m,views['no_coref'],d)
        self.assertEqual(values,run.residual.infer_details(run.residual.restore(b),
                         m['ids']['validation'],views['no_coref'],d['inputs.json'],False))

    def test_no_future_values_in_model_inputs(self):
        m,views,d = setup()
        batch = run.task.make_batch(m['ids']['train'],views['no_coref'],d['inputs.json'])
        self.assertFalse({'y','target_date','cve'}.intersection(batch.keys()))
        changed = deepcopy(d['inputs.json'])
        changed[m['ids']['train'][0]]['X']['future_epss'] = .9
        with self.assertRaisesRegex(ValueError,'Unexpected predictor'):
            run.task.make_batch(m['ids']['train'],views['no_coref'],changed)

    def test_decision_complete_tie_and_one_common_loss(self):
        p = tiny()
        rows = {run.fit_name(v,l,17): {'metrics':{'mae':.1}}
                for v in run.MODELS for l in run.LOSSES}
        self.assertEqual(run.decision(rows,p)['selected_loss'],'smooth_l1')
        rows['no_coref__mae__17']['metrics']['mae'] = .08
        self.assertEqual(run.decision(rows,p)['selected_loss'],'mae')
        rows['epss_only__mae__17']['metrics']['mae'] = .13
        self.assertEqual(run.decision(rows,p)['selected_loss'],'smooth_l1')
        rows.pop('epss_only__mae__17')
        with self.assertRaisesRegex(ValueError,'all fits'):
            run.decision(rows,p)

    def test_full_synthetic_lifecycle_resume_and_control_parity(self):
        m,views,d = setup(); p = tiny()
        identity = {'schema':run.VERSION,'ids':m['ids'],'protocol':p,'pinned_files':{}}
        with tempfile.TemporaryDirectory() as folder:
            source,output = Path(folder)/'r05',Path(folder)/'r06'
            source.mkdir()
            run.write(source/'orders_seed_17.json',run.prior.epoch_orders(m['ids']['train'],run.settings(p,17)))
            run.write(source/'baselines.json',run.r05.baselines(m,d))
            for model in run.MODELS:
                saved,h,g = run.r05.fit_one(model,'original_decay',17,m,views[model],d,p,False)
                initial = run.initialize(model,m,views[model],run.settings(p,17))
                values,metrics = run.assess(saved,initial,m,views[model],d)
                dest = source/'fits'/run.fit_name(model,'original_decay',17)
                dest.mkdir(parents=True)
                run.prior.atomic_torch_save(dest/'checkpoint.pt',saved)
                for name,obj in [('history.json',h),('gradients.json',g),('predictions.json',values),('metrics.json',metrics)]:
                    run.write(dest/name,obj)
            with patch.object(run,'context',return_value=(m,views,d,identity)),patch.object(run,'R05_SOURCE',source):
                with patch.object(run,'fit_one',side_effect=AssertionError('preflight must not fit')):
                    run.execute(check_only=True,output=output)
                self.assertFalse(output.exists())
                original = run.fit_one
                calls = []
                def interrupted(*a,**kw):
                    calls.append(a[1])
                    if len(calls)==2:
                        raise RuntimeError('interruption')
                    return original(*a,**kw,progress=False)
                with patch.object(run,'fit_one',side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError,'interruption'):
                        run.execute(output=output)
                with patch.object(run,'fit_one',side_effect=lambda *a,**kw:original(*a,**kw,progress=False)):
                    run.execute(resume=True,output=output)
                with patch.object(run,'fit_one',side_effect=AssertionError('verify must not fit')):
                    run.execute(verify_only=True,output=output)
                self.assertTrue(run.read(output/'summary.json')['smooth_l1_controls_reproduce_r05'])
                self.assertFalse(run.read(output/'full_training_decision.json')['full_training_started'])
                with self.assertRaisesRegex(ValueError,'Completed output'):
                    run.execute(resume=True,output=output)
                with self.assertRaisesRegex(ValueError,'Output exists'):
                    run.execute(output=output)
                (output/'full_training_decision.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'Changed source/artifact'):
                    run.execute(verify_only=True,output=output)


if __name__ == '__main__':
    unittest.main()
