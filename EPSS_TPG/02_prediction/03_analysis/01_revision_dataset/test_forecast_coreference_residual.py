"""R04 smoke regression tests; synthetic data only, no reserved test scoring."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_coreference_residual as run
from test_forecast_coreference_smoke import setup

torch = run.torch


class ResidualSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def initial(self, mode='tpg_epss'):
        m, views, data = setup()
        initial = run.initialize(mode, m, views['all_coref'], run.smoke.config())
        net = run.ResidualTPGRegressor(**initial['spec'])
        net.load_state_dict(initial['state_dict'])
        return m, views, data, initial, net

    def test_fixed_scope_and_settings(self):
        p = run.settings()
        self.assertEqual(p['config']['seeds'], [17])
        self.assertEqual(p['config']['epochs'], 3)
        self.assertFalse(p['epoch_zero_eligible_for_selection'])
        bad = deepcopy(p); bad['config']['epochs'] = 100
        with patch.object(run, 'read', return_value=bad), self.assertRaisesRegex(ValueError, 'settings'):
            run.settings()

    def test_last_observed_missing_day30_and_actual_zero(self):
        m, views, data, initial, net = self.initial()
        batch = run.task.make_batch(m['ids']['train'], views['all_coref'], data['inputs.json'])
        self.assertTrue(bool((batch.epss_sequence[:,30,1] == 0).all()))
        base = run.last_observed(batch.epss_sequence)
        self.assertTrue(torch.equal(base, batch.epss_sequence[:,29,0]))
        batch.epss_sequence[:,29,0] = 0
        self.assertTrue(torch.equal(run.last_observed(batch.epss_sequence), torch.zeros(2)))
        batch.epss_sequence[:,:,1] = 0
        with self.assertRaisesRegex(ValueError, 'No observed'):
            run.last_observed(batch.epss_sequence)

    def test_exact_persistence_and_target_independent_initialization(self):
        m, views, data, initial, net = self.initial()
        init = {mode:run.initialize(mode,m,views['all_coref'],run.smoke.config()) for mode in ('tpg_epss','epss_only')}
        checks = run.initial_checks(init, m, views, data, False)
        self.assertTrue(all(v['exact_float32_persistence'] for x in checks.values() for v in x.values()))
        for row in data['labels.json'].values():
            row['y']['endpoint'] = .99
        other = run.initialize('tpg_epss',m,views['all_coref'],run.smoke.config())
        self.assertEqual(initial['initial_state_sha256'],other['initial_state_sha256'])
        self.assertFalse(initial['uses_target_statistics'])

    def test_same_frozen_encoder_representation_as_direct_model(self):
        m, views, data, initial, net = self.initial()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            direct = run.task.TPGRegressor(**initial['spec'])
        for name, value in net.state_dict().items():
            if not name.startswith('head.2.'):
                self.assertTrue(torch.equal(value,direct.state_dict()[name]), name)
        captured = []
        hook = direct.head.register_forward_pre_hook(lambda module,args: captured.append(args[0].detach().clone()))
        batch = run.task.make_batch(m['ids']['train'],views['all_coref'],data['inputs.json'])
        direct.eval(); net.eval()
        try:
            with torch.no_grad():
                direct(batch)
                rep = net.representation(batch)
        finally:
            hook.remove()
        self.assertTrue(torch.equal(captured[0],rep))

    def test_shared_graph_initialization_and_schedules(self):
        m, views, data = setup(); c = run.smoke.config()
        states = [run.initialize('tpg_epss',m,views[v],c) for v in run.cache.shared.MODES]
        self.assertEqual(len({s['initial_state_sha256'] for s in states}),1)
        self.assertEqual(run.smoke.epoch_orders(m['ids']['train'],c),run.smoke.epoch_orders(m['ids']['train'],c))

    def test_first_step_gradient_routing_and_later_encoder_gradients(self):
        m, views, data, initial, net = self.initial()
        batch = run.task.make_batch(m['ids']['train'],views['all_coref'],data['inputs.json'])
        optimizer = torch.optim.SGD(net.parameters(),lr=.01)
        loss = torch.nn.functional.smooth_l1_loss(net(batch),torch.full((2,),.8),beta=.01)
        loss.backward()
        first = run.gradient_snapshot(net)
        self.assertGreater(first['head']['absolute_sum'],0)
        for prefix in ('temporal','node_project','layers'):
            self.assertEqual(first[prefix]['absolute_sum'],0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.smooth_l1_loss(net(batch),torch.full((2,),.8),beta=.01).backward()
        later = run.gradient_snapshot(net)
        for prefix in ('temporal','node_project','layers'):
            self.assertGreater(later[prefix]['absolute_sum'],0)

    def test_clipping_boundaries_and_saturation_gradient(self):
        m, views, data, initial, net = self.initial('epss_only')
        batch = run.task.make_batch(m['ids']['train'],views['all_coref'],data['inputs.json'])
        for base,bias,expected in ((.9,2.,1.),(.1,-2.,0.)):
            batch.epss_sequence[:,29,0] = base
            with torch.no_grad():
                net.head[-1].bias.fill_(bias)
            net.zero_grad(set_to_none=True)
            values = net.details(batch)
            self.assertTrue(torch.equal(values[-1],torch.full((2,),expected)))
            values[-1].sum().backward()
            self.assertTrue(torch.equal(net.head[-1].bias.grad,torch.zeros(1)))
        with torch.no_grad():
            net.head[-1].bias.zero_()
        batch.epss_sequence[:,29,0] = 0
        net.zero_grad(set_to_none=True)
        net(batch).sum().backward()
        self.assertGreater(float(net.head[-1].bias.grad),0)

    def test_epss_only_graph_invariance_and_future_input_rejection(self):
        m, views, data, initial, net = self.initial('epss_only')
        with torch.no_grad():
            net.head[-1].weight.fill_(.1)
        a = run.task.infer(net,m['ids']['validation'],views['all_coref'],data['inputs.json'],8,progress=False)
        b = run.task.infer(net,m['ids']['validation'],views['no_coref'],data['inputs.json'],8,progress=False)
        self.assertEqual(a,b)
        changed = deepcopy(data['inputs.json'])
        changed[m['ids']['validation'][0]]['X']['future_epss'] = .5
        with self.assertRaisesRegex(ValueError,'Unexpected predictor'):
            run.task.make_batch(m['ids']['validation'],views['all_coref'],changed)

    def test_deterministic_fit_restore_and_epoch_zero_exclusion(self):
        m, views, data, initial, net = self.initial()
        config = run.smoke.config(); orders = run.smoke.epoch_orders(m['ids']['train'],config)
        a,h,g = run.fit('all_coref',initial,orders,m,views['all_coref'],data,config,False)
        b,h2,g2 = run.fit('all_coref',initial,orders,m,views['all_coref'],data,config,False)
        self.assertEqual(h,h2); self.assertEqual(g,g2)
        self.assertEqual(run.prior.digest_state(a['state_dict']),run.prior.digest_state(b['state_dict']))
        self.assertGreaterEqual(a['selected_epoch'],1)
        self.assertEqual(a['selected_epoch'],min(h,key=lambda x:x['validation_mae'])['epoch'])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint.pt'; torch.save(a,path)
            saved=torch.load(path,weights_only=True)
            x=run.infer_details(run.restore(a),m['ids']['validation'],views['all_coref'],data['inputs.json'],False)
            y=run.infer_details(run.restore(saved),m['ids']['validation'],views['all_coref'],data['inputs.json'],False)
            self.assertEqual(x,y)
        bad=deepcopy(a); bad['selected_epoch']=0
        with self.assertRaisesRegex(ValueError,'Epoch zero'):
            run.restore(bad)

    def test_tied_validation_selects_earliest_trained_epoch(self):
        m, views, data, initial, net = self.initial()
        c=run.smoke.config()
        with patch.object(run.task,'infer',side_effect=lambda net,ids,*a,**k:[.5]*len(ids)):
            saved,h,g=run.fit('all_coref',initial,run.smoke.epoch_orders(m['ids']['train'],c),m,views['all_coref'],data,c,False)
        self.assertEqual(saved['selected_epoch'],1)
        self.assertEqual(len(h),3)

    def test_end_to_end_preflight_save_verify_no_overwrite(self):
        m, views, data = setup()
        config=run.smoke.config()
        identity={'schema':run.VERSION,'config':config,'ids':m['ids'],'cache_lock_sha256':'synthetic',
                  'pinned_files':{},'limitations':['Synthetic fixture only.']}
        with tempfile.TemporaryDirectory() as folder, patch.object(run,'context',return_value=(m,views,data,identity)), \
             patch.object(run.task,'RESULT_ROOT',Path(folder)):
            output=Path(folder)/'smoke'
            with patch.object(run,'fit',side_effect=AssertionError('No preflight fitting')):
                run.execute(output,check_only=True)
            self.assertFalse(output.exists())
            run.execute(output)
            with patch.object(run,'fit',side_effect=AssertionError('No verify fitting')):
                run.execute(output,verify_only=True)
            result=run.read(output/'summary.json')
            self.assertEqual(len(result['results']),5)
            self.assertEqual(len(run.read(output/'predictions.json')),2)
            with self.assertRaisesRegex(ValueError,'Output exists'):
                run.execute(output)
            (output/'summary.txt').write_text('tampered')
            with self.assertRaisesRegex(ValueError,'Changed artifact'):
                run.execute(output,verify_only=True)


if __name__ == '__main__':
    unittest.main()
