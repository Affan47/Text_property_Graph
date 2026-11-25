"""Synthetic R05 tests; no dataset model training."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_decay_comparison as run
from test_forecast_coreference_smoke import setup


def tiny_protocol():
    p = run.protocol()
    p['config'].update(epochs=2, patience=2, seeds=[17])
    return p


class DecayComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = run.torch.get_num_threads()
        run.torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        run.torch.set_num_threads(cls.threads)

    def test_fixed_protocol(self):
        p = run.protocol()
        self.assertEqual(p['config']['epochs'], 30)
        self.assertEqual(p['config']['seeds'], [17,29,43])
        self.assertEqual(p['models'], ['epss_only', 'no_coref'])
        self.assertEqual(p['cohort']['train'], 128)
        a, b = [run.settings(p, 17, c) for c in run.CONDITIONS]
        self.assertEqual({k for k in a if a[k] != b[k]}, {'weight_decay'})

    def test_hash_selection_split_and_group_guards(self):
        ids = {'train': ['a','b','c'], 'validation': ['d','e']}
        splits = {c: {'split': s, 'group_id': c} for s, cs in ids.items() for c in cs}
        sizes = {'train': 2, 'validation': 1}
        selected = run.select_cohort(ids, splits, sizes)
        self.assertEqual(selected, run.select_cohort({s: list(reversed(v)) for s,v in ids.items()}, splits, sizes))
        splits[selected['validation'][0]]['group_id'] = selected['train'][0]
        with self.assertRaisesRegex(ValueError, 'Related group'):
            run.select_cohort(ids, splits, sizes)
        splits['a']['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'Changed split'):
            run.select_cohort(ids, splits, sizes)

    def test_weights_and_orders_matched(self):
        m, views, _ = setup()
        p = tiny_protocol()
        for v in run.MODELS:
            a, b = [run.initialize(v, m, views['no_coref'], run.settings(p,17,c)) for c in run.CONDITIONS]
            self.assertEqual(a['initial_state_sha256'], b['initial_state_sha256'])
        a, b = [run.prior.epoch_orders(m['ids']['train'], run.settings(p,17,c)) for c in run.CONDITIONS]
        self.assertEqual(a,b)

    def test_scalar_uses_train_targets_only(self):
        m, _, d = setup()
        a = run.scalar_control(m['ids']['train'], d['inputs.json'], d['labels.json'])
        altered = deepcopy(d)
        for c in m['ids']['validation']:
            altered['labels.json'][c]['y']['endpoint'] = .999
        b = run.scalar_control(m['ids']['train'], altered['inputs.json'], altered['labels.json'])
        self.assertEqual(a,b)

    def test_fixed_epochs_zero_decay_and_reload(self):
        m, views, d = setup(); p = tiny_protocol()
        for condition in run.CONDITIONS:
            a, h, g = run.fit_one('no_coref',condition,17,m,views['no_coref'],d,p,False)
            b, h2, g2 = run.fit_one('no_coref',condition,17,m,views['no_coref'],d,p,False)
            self.assertEqual(len(h),2)
            self.assertEqual(h,h2); self.assertEqual(g,g2)
            self.assertEqual(run.prior.digest_state(a['state_dict']),run.prior.digest_state(b['state_dict']))
            self.assertEqual(a['spec']['config']['weight_decay'],run.CONDITIONS[condition])
            init = run.initialize('no_coref',m,views['no_coref'],run.settings(p,17,condition))
            values,metrics = run.assess(a,init,m,views['no_coref'],d)
            reloaded = run.residual.restore(a)
            self.assertEqual(values,run.residual.infer_details(reloaded,m['ids']['validation'],views['no_coref'],d['inputs.json'],False))

    def test_preflight_does_not_train(self):
        m, views, d = setup(); p = tiny_protocol()
        with patch.object(run.residual,'fit',side_effect=AssertionError('No fit in preflight')):
            run.preflight(m,views,d,p)

    def test_execute_verify_resume_and_tamper(self):
        m, views, d = setup(); p = tiny_protocol()
        ident = {'schema':run.VERSION,'protocol':p,'pinned_files':{},'ids':m['ids']}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'run'
            with patch.object(run,'context',return_value=(m,views,d,ident)):
                run.execute(check_only=True,output=out)
                self.assertFalse(out.exists())
                original = run.fit_one
                calls = []
                def interrupt(*a,**kw):
                    calls.append(a[1])
                    if len(calls) == 2:
                        raise RuntimeError('interrupted')
                    return original(*a,**kw,progress=False)
                with patch.object(run,'fit_one',side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError,'interrupted'):
                        run.execute(output=out)
                with patch.object(run,'fit_one',side_effect=lambda *a,**kw: original(*a,**kw,progress=False)):
                    run.execute(resume=True,output=out)
                with patch.object(run,'fit_one',side_effect=AssertionError('Verification must not train')):
                    run.execute(verify_only=True,output=out)
                with self.assertRaisesRegex(ValueError,'Completed output'):
                    run.execute(resume=True,output=out)
                with self.assertRaisesRegex(ValueError,'Output exists'):
                    run.execute(output=out)
                (out/'summary.txt').write_text('changed',encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'Changed source/artifact'):
                    run.execute(verify_only=True,output=out)


if __name__ == '__main__':
    unittest.main()
