"""SF01/SF02 synthetic tests, without real-data training or test inspection."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_structured_features as a
from epss import forecast_structured_training as run
import test_forecast_day60_pipeline as old_tests

VECTOR = 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H'


def raw():
    return {'cve':'CVE-2022-12345','published_date':'2022-01-01',
            'cvss_vector':VECTOR,'cvss_vector_version':'3.1','cvss_vector_score':9.8,
            'last_modified_timestamp':'2026-09-01T12:00:00', 'in_vckev':True,
            'url_change_dates':[{'url':'https://example.org','earliest_added_date':'2022-01-02'}]}


class AuditTests(unittest.TestCase):
    def test_fixed_encoding(self):
        r,issues = a.input_row(raw())
        self.assertEqual(issues,[])
        self.assertEqual(len(a.check_input(r)),27)
        self.assertAlmostEqual(r['values'][-3],.98)
        self.assertEqual(r['values'][-2:],[1.,1.])
        self.assertEqual(sum(r['values'][:-3]),9)

    def test_three_zero_version_kept_separate(self):
        r = raw()
        r.update(cvss_vector=VECTOR.replace('3.1','3.0'),cvss_vector_version='3.0')
        x,_ = a.input_row(r)
        self.assertEqual(x['values'][22:24],[1,0])

    def test_real_dataset_numeric_versions_match_string_versions(self):
        for version in (3,3.0,3.1):
            r = raw()
            r.update(cvss_vector=VECTOR.replace('3.1',f'{version:.1f}'),cvss_vector_version=version)
            numeric,issues = a.input_row(r)
            r['cvss_vector_version'] = f'{version:.1f}'
            self.assertEqual(numeric,a.input_row(r)[0])
            self.assertEqual(issues,[])
            self.assertEqual(numeric['values'][-1],1.)

    def test_bad_vectors_masked(self):
        for vector in (None,'',2,VECTOR+'/AV:N',VECTOR.replace('/AC:L',''),VECTOR.replace('AV:N','AV:X'),
                       VECTOR.replace('3.1','4.0'),VECTOR+'/BAD:X',VECTOR+'/RC:ZZ'):
            r = raw()
            r['cvss_vector'] = vector
            x,issues = a.input_row(r)
            self.assertTrue(issues)
            self.assertEqual(a.check_input(x),[0.]*27)

    def test_version_mismatch_is_not_silently_repaired(self):
        r = raw()
        r['cvss_vector_version'] = '3.0'
        x,issues = a.input_row(r)
        self.assertEqual(issues,['version_field_disagrees_with_vector'])
        self.assertEqual(x['values'][-1],0)

    def test_invalid_scores_have_explicit_masks(self):
        for v in (None,float('nan'),float('inf'),True,-1,11,'9.8'):
            r = raw()
            r['cvss_vector_score'] = v
            x,issues = a.input_row(r)
            self.assertTrue(issues)
            self.assertEqual(a.check_input(x)[-3:],[0,0,1])
        r['cvss_vector_score'] = 0
        self.assertEqual(a.input_row(r)[0]['values'][-3:],[0,1,1])

    def test_optional_metrics_not_sneaked_into_base_inputs(self):
        r = raw()
        r['cvss_vector'] += '/E:H/RL:O/MAV:P'
        x,issues = a.input_row(r)
        self.assertEqual(x,a.input_row(raw())[0])
        self.assertEqual(issues,['optional_metrics_not_used'])

    def test_no_future_or_catalogue_fields_in_encoder(self):
        r = {k:raw()[k] for k in a.CVSS_FIELDS}
        with self.assertRaisesRegex(ValueError,'allowlist'):
            a.encode_cvss({**r,'in_vckev':True})
        first = a.input_row(raw())[0]
        r = raw()
        r.update(in_vckev=False,vckev_cwes=['CWE-79'],epss_version_changed=True,target_day60=1.)
        self.assertEqual(a.input_row(r)[0],first)
        self.assertFalse(any('vckev' in name or 'reference' in name for name in a.FEATURES))

    def test_update_date_never_authenticates_historical_cvss(self):
        for value, relation in [('2026-09-01T12:00:00','after_day30'),('2022-01-02','before_day30'),
                                ('2022-01-31T23:59:59','same_day_as_day30'),('bad','unparseable_or_missing')]:
            r = raw()
            r['last_modified_timestamp'] = value
            audit,_ = a.inspect(r,'2022-01-31')
            self.assertEqual(audit['nvd_update_relation'],relation)
            self.assertEqual(audit['cvss_availability'],'not_verified_at_day30')
            self.assertFalse(audit['historical_reference_count_verified'])

    def test_historical_and_non_opt_in_modes_block_before_data_access(self):
        with patch.object(run,'context') as ctx:
            with self.assertRaisesRegex(ValueError,'Historical CVSS'):
                run.execute('smoke',1,0)
            with self.assertRaisesRegex(ValueError,'allow-retrospective'):
                run.execute('smoke',1,0,policy='retrospective')
            ctx.assert_not_called()

    def test_model_input_rejects_unknown_fields_and_inconsistent_onehot(self):
        r,_ = a.input_row(raw())
        with self.assertRaises(ValueError):
            a.check_input({**r,'target':1})
        r['values'][0] = 0
        with self.assertRaises(ValueError):
            a.check_input(r)

    def test_audit_does_not_adapt_holdout_values(self):
        heldout = {'cve':'heldout', 'published_date':'must not parse', 'cvss_vector':object()}
        c = raw()['cve']
        ctx = ({'train':[c],'validation':[]},{c:{'input_end_date':'2022-01-31'}},Path('/synthetic'),{})
        with patch.object(a,'context',return_value=ctx), patch.object(a,'read',return_value=[raw(),heldout]):
            output = a.build('smoke')
        self.assertEqual(set(output['cve_audit.json']),{c})
        self.assertEqual(set(output['retrospective_inputs.json']),{c})
        self.assertEqual(output['summary.json']['historically_verified_cvss_cves'],0)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        self.out = redirect_stdout(StringIO())
        self.err = redirect_stderr(StringIO())
        self.out.__enter__()
        self.err.__enter__()
        self.addCleanup(self.out.__exit__,None,None,None)
        self.addCleanup(self.err.__exit__,None,None,None)
        self.i = old_tests.identity()
        self.i.update(schema=run.SCHEMA,availability_policy='retrospective',structured_feature_names=a.FEATURES)
        self.i['protocol'].update(run.read(run.CONFIG))
        self.x = {s:run.old.tensor_inputs([old_tests.inputs(c) for c in ids]) for s,ids in self.i['ids'].items()}
        self.labels = {c:{'cve':c,'target_day':60,'horizon_days':30,'target_observed':True,'y':{'endpoint':.24}}
                       for ids in self.i['ids'].values() for c in ids}
        self.rows = {c:a.input_row(raw())[0] for c in self.labels}
        self.store = run.model.StructuredStore(old_tests.legacy.MemoryStore(),self.rows,self.labels)

    def test_matched_initialization_and_persistence(self):
        states = []
        for v in run.model.VARIANTS:
            net = run.initialize(self.i,v,17)
            states.append(run.old.controls.state_digest(net.state_dict()))
            parts = net.details(self.x['train'],self.store,self.i['ids']['train'])
            self.assertTrue(torch.equal(parts[0],parts[3]))
        self.assertEqual(len(set(states)),1)

    def test_control_does_not_read_cvss(self):
        net = run.initialize(self.i,run.model.VARIANTS[0],17)
        with patch.object(self.store,'structured',side_effect=AssertionError('must not read')):
            net(self.x['train'],self.store,self.i['ids']['train']).sum().backward()
        self.assertTrue(all(p.grad is None for p in net.structured_encoder.parameters()))

    def test_cvss_branch_receives_gradient_after_head_opens(self):
        net = run.initialize(self.i,run.model.VARIANTS[1],17)
        with torch.no_grad():
            net.head[-1].weight.fill_(.2)
        net(self.x['train'],self.store,self.i['ids']['train']).sum().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in net.structured_encoder.parameters()),0)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in net.node_project.parameters()),0)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in net.temporal.parameters()),0)

    def test_missing_cvss_and_missing_text_stay_finite(self):
        net = run.initialize(self.i,run.model.VARIANTS[1],17)
        store = run.model.StructuredStore(old_tests.legacy.MemoryStore(),{'missing':{'values':[0.]*27}},['missing'])
        parts = net.details(self.x['train'][:1],store,['missing'])
        self.assertTrue(torch.isfinite(parts[3]).all())
        self.assertEqual(float(parts[4].sum()),0)

    def test_no_unselected_structured_rows(self):
        with self.assertRaises(ValueError):
            run.model.StructuredStore(self.store,self.rows,['a'])
        with self.assertRaises(ValueError):
            self.store.structured(['heldout'],self.x['train'])

    def test_two_variant_checkpoint_reproduction_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run,'OUTPUT',Path(tmp)/'results'), \
                patch.object(run,'context',return_value=(self.i,self.x,self.labels,self.store)):
            opts = dict(policy='retrospective',allow_retrospective=True)
            run.execute('smoke',1,0,check_only=True,**opts)
            self.assertFalse(run.OUTPUT.exists())
            run.execute('smoke',1,0,**opts)
            run.execute('smoke',1,0,verify_only=True,**opts)
            with self.assertRaisesRegex(ValueError,'Run exists'):
                run.execute('smoke',1,0,**opts)
            with self.assertRaisesRegex(ValueError,'complete resume'):
                run.execute('smoke',1,0,resume=True,**opts)
            (run.OUTPUT/'smoke/evaluation_lock.json').unlink()
            with patch.object(run,'fit',side_effect=AssertionError('completed fits must not retrain')):
                run.execute('smoke',1,0,resume=True,**opts)
            path = run.OUTPUT/f'smoke/fits/{run.model.VARIANTS[0]}__17/predictions.json'
            rows = run.read(path)
            rows[0]['prediction'] = 1.
            run.write(path,rows)
            with self.assertRaisesRegex(ValueError,'Changed/unsafe artifact'):
                run.execute('smoke',1,0,verify_only=True,**opts)

    def test_checkpoint_cannot_move_to_different_cohort(self):
        saved,history = run.fit(run.model.VARIANTS[1],17,self.i,self.x,self.labels,self.store,False)
        self.assertEqual(saved['identity_sha256'],run.identity_digest(self.i))
        changed = deepcopy(self.i)
        changed['ids']['train'] = list(reversed(changed['ids']['train']))
        self.assertNotEqual(saved['identity_sha256'],run.identity_digest(changed))
        rows,metrics = run.evaluate(saved,self.i,self.x,self.labels,self.store,False)
        self.assertTrue(all('actual_day60_epss' in r for r in rows))
        self.assertEqual(metrics['metrics']['mae'],history[saved['selected_epoch']-1]['validation_mae'])

    def test_full_six_fit_lifecycle_requires_completed_smoke(self):
        full = deepcopy(self.i)
        full['scale'] = 'full'
        full['config']['seeds'] = [17,29,43]
        def ctx(scale,threads,cache_gib):
            return deepcopy(full if scale == 'full' else self.i),self.x,self.labels,self.store
        with tempfile.TemporaryDirectory() as tmp, patch.object(run,'OUTPUT',Path(tmp)/'results'), patch.object(run,'context',side_effect=ctx):
            opts = dict(policy='retrospective',allow_retrospective=True)
            with self.assertRaises(FileNotFoundError):
                run.execute('full',1,0,**opts)
            self.assertFalse((run.OUTPUT/'full').exists())
            run.execute('smoke',1,0,**opts)
            run.execute('full',1,0,**opts)
            run.execute('full',1,0,verify_only=True,**opts)
            self.assertEqual(len(run.read(run.OUTPUT/'full/summary.json')['fits']),6)


if __name__ == '__main__':
    unittest.main()
