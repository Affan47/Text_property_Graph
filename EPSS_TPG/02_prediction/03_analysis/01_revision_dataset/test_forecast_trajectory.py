"""TR01 leakage boundaries, masks, matched windows and training lifecycle."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_trajectory as run
data = run.data


def raw(cve, published):
    pub = date.fromisoformat(published)
    dates = [(pub+timedelta(days=d)).isoformat() for d in range(91)]
    values = [None if d in (0, 21, 22, 30, 37) else .1 + d/1000 for d in range(91)]
    return {'cve': cve, 'published_date': published, 'published_timestamp': published+'T12:00:00Z',
            'epss_scores': dict(zip(dates, values))}


def artifacts(scale='smoke'):
    records = [raw('CVE-2024-10001', '2024-01-01'), raw('CVE-2024-10002', '2024-02-01'),
               raw('CVE-2025-10003', '2025-03-01'), raw('CVE-2025-10004', '2025-03-02')]
    for r in records[1::2]:
        r['epss_scores'][(date.fromisoformat(r['published_date'])+timedelta(days=37)).isoformat()] = .2
    ids = {'train': [r['cve'] for r in records[:2]], 'validation': [r['cve'] for r in records[2:]]}
    x, y, manifest = {}, {}, {}
    for r in records:
        c = r['cve']
        x[c], old_y, _ = data.parent.derive(r)
        y[c] = data.derive_labels(r, x[c], old_y)
        manifest[c] = {'split': 'train' if c in ids['train'] else 'validation',
                       'input_end_date': x[c]['input_end_date'], 'target_date': old_y['target_date']}
    a = {'protocol.json': data.protocol(), 'run_identity.json': {'schema': data.SCHEMA, 'scale': scale, 'ids': ids,
        'source_pins': {}, 'test_scoring': False, 'split_policy': {'validation_start': '2025-03-02',
            'test_start': '2025-06-01', 'test_end': '2025-12-02', 'observation_end': '2026-09-29'}},
         'inputs.json': x, 'labels.json': y, 'split_manifest.json': manifest}
    data.validate(a)
    a['coverage.json'] = data.coverage(a)
    a['summary.txt'] = data.report(a)
    return a


def fixture(scale='smoke'):
    a = artifacts(scale)
    p = a['protocol.json']
    cfg = deepcopy(p['config'])
    cfg.update(epochs=2, patience=2, batch_size=2, temporal_hidden_size=3, hidden_size=4,
               seeds=[17] if scale == 'smoke' else [17,29,43])
    i = {'schema': run.SCHEMA, 'scale': scale, 'ids': a['run_identity.json']['ids'], 'config': cfg,
         'protocol': p, 'pinned_files': {}, 'test_scoring': False}
    t = {length: {s: data.input_tensor([a['inputs.json'][c] for c in ids], length)
                  for s, ids in i['ids'].items()} for length in data.LENGTHS}
    return i, t, a['labels.json']


class DailyTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        for redirect in (redirect_stdout, redirect_stderr):
            cm = redirect(StringIO())
            cm.__enter__()
            self.addCleanup(cm.__exit__, None, None, None)
        self.i, self.t, self.y = fixture()

    def test_lengths_offsets_and_recomputed_observation_age(self):
        a = artifacts(); row = next(iter(a['inputs.json'].values()))
        for n, start in ((10,21), (20,11), (31,0)):
            h = data.input_view(row, n)
            self.assertEqual(h['day_offsets'], list(range(start,31)))
            self.assertEqual(data.input_tensor([row],n).shape, (1,n,5))
        h = data.input_view(row,10)
        self.assertEqual(h['age_known_mask'][:2], [0,0])
        self.assertEqual(h['age_since_observation_days'][:2], [None,None])
        self.assertTrue(all(v is None for v in h['values'][:2]))
        self.assertEqual(h['age_since_observation_days'][-1],1)

    def test_target_and_outside_window_changes_cannot_change_short_inputs(self):
        r = raw('CVE-2024-10001','2024-01-01')
        x, label, _ = data.parent.derive(r)
        before = data.input_tensor([x],10)
        changed = deepcopy(x)
        # An earlier observed score changes the long view, never the short view.
        changed['histories']['days_0_30']['values'][1] = .9
        self.assertTrue(torch.equal(before,data.input_tensor([changed],10)))
        r['epss_scores']['2024-02-14'] = .99
        self.assertTrue(torch.equal(before,data.input_tensor([x],10)))
        with self.assertRaises(ValueError):
            data.input_tensor([{**x,'labels':label}],10)

    def test_empty_short_history_has_no_hidden_long_history_fallback(self):
        r = raw('CVE-2024-10001','2024-01-01')
        for day in range(21,31):
            r['epss_scores'][(date(2024,1,1)+timedelta(days=day)).isoformat()] = None
        x, _, _ = data.parent.derive(r)
        self.assertTrue(any(data.input_view(x,31)['observed_mask']))
        with self.assertRaisesRegex(ValueError,'No observed score'):
            data.input_view(x,10)

    def test_exact_future_dates_masks_and_parent_target_integrity(self):
        r = raw('CVE-2024-10001','2024-01-01')
        x, y, _ = data.parent.derive(r)
        label = data.derive_labels(r,x,y)
        self.assertEqual(label['target_dates'], [(date(2024,1,1)+timedelta(days=d)).isoformat() for d in range(31,61)])
        self.assertEqual(label['observed_mask'], [int(d != 37) for d in range(31,61)])
        self.assertIsNone(label['values'][6])
        r['epss_scores']['2024-03-01'] = .9
        with self.assertRaisesRegex(ValueError,'Raw source differs'):
            data.derive_labels(r,x,y)

    def test_temporal_maturity_roles_and_missing_targets(self):
        for mutate in ('test','maturity','mask'):
            a = artifacts(); c = a['run_identity.json']['ids']['train'][0]
            if mutate=='test':
                a['split_manifest.json'][c]['split']='test'
            elif mutate=='maturity':
                a['run_identity.json']['split_policy']['validation_start']='2024-02-10'
            else:
                a['labels.json'][c]['observed_mask'][6]=1
            with self.assertRaises(ValueError): data.validate(a)

    def test_loss_masks_absent_targets_and_equalizes_cve_weight(self):
        pred=torch.full((2,30), .4, requires_grad=True)
        target=torch.full((2,30), float('nan'))
        target[0,0]=.3; target[0,2]=.1; target[1,29]=-.1
        mask=torch.isfinite(target)
        loss=run.masked_mae(pred,target,mask)
        self.assertAlmostEqual(float(loss),.35,places=6)
        loss.backward()
        expected=torch.zeros((2,30)); expected[0,0]=.25; expected[0,2]=.25; expected[1,29]=.5
        torch.testing.assert_close(pred.grad,expected)
        with self.assertRaises(ValueError):run.masked_mae(pred,target,torch.zeros_like(mask))

    def test_initial_weights_persistence_shared_across_windows(self):
        run.preflight(self.i,self.t,self.y)
        hashes=[]
        for length in data.LENGTHS:
            net=run.initialize(self.i,17)
            hashes.append(run.controls.state_digest(net.state_dict()))
            parts=net.details(self.t[length]['train'])
            self.assertEqual(parts[3].shape,(2,30))
            self.assertTrue(torch.equal(parts[0],parts[3]))
        self.assertEqual(len(set(hashes)),1)

    def test_daily_correction_uses_cve_loss_weights_and_lower_median_tie(self):
        ids = ['A', 'B', 'C']
        tensor = torch.zeros((3,31,5)); tensor[:,:,1] = 1
        labels = {c: {'cve': c, 'horizons': list(data.HORIZONS), 'values': [v]*30,
                      'observed_mask': [1]*30} for c,v in zip(ids,[.1,.2,.9])}
        labels['C']['values'][1:] = [None]*29
        labels['C']['observed_mask'][1:] = [0]*29
        fitted = run.fit_daily_corrections(ids,tensor,labels)
        self.assertEqual(fitted['offsets'][0], .9)
        self.assertEqual(fitted['offsets'][1:], [.1]*29)
        self.assertEqual(fitted['observed_training_targets'], [3]+[2]*29)
        self.assertAlmostEqual(fitted['weight_sums'][0], 1+2/30)
        tied = {c: deepcopy(labels[c]) for c in ids[:2]}
        self.assertEqual(run.fit_daily_corrections(ids[:2],tensor[:2],tied)['offsets'], [.1]*30)

    def test_daily_correction_cannot_use_validation_labels(self):
        before = run.references(self.i,self.t,self.y)
        changed = deepcopy(self.y)
        for c in self.i['ids']['validation']:
            changed[c]['values'] = [.9 if v is not None else None for v in changed[c]['values']]
        after = run.references(self.i,self.t,changed)
        self.assertEqual(before['daily_correction']['fit'],after['daily_correction']['fit'])
        self.assertEqual([r['prediction'] for r in before['daily_correction']['rows']],
                         [r['prediction'] for r in after['daily_correction']['rows']])
        self.assertNotEqual(before['daily_correction']['metrics'],after['daily_correction']['metrics'])
        with self.assertRaisesRegex(ValueError,'training records only'):
            run.fit_daily_corrections(self.i['ids']['train'],self.t[31]['train'],self.y)

    def test_daily_correction_masks_and_empty_days_are_checked(self):
        ids = self.i['ids']['train']; labels = {c: deepcopy(self.y[c]) for c in ids}
        for c in ids:
            labels[c]['values'][2] = None; labels[c]['observed_mask'][2] = 0
        with self.assertRaisesRegex(ValueError,'No training observations'):
            run.fit_daily_corrections(ids,self.t[31]['train'],labels)
        labels[ids[0]]['observed_mask'][2] = 1
        with self.assertRaisesRegex(ValueError,'Invalid correction targets'):
            run.fit_daily_corrections(ids,self.t[31]['train'],labels)

    def test_constant_control_clipping_and_daily_outputs_are_explicit(self):
        offsets = [-1., 1.]+[.1]*28
        base, correction, raw, prediction = run.DailyCorrection(offsets).details(self.t[10]['validation'])
        self.assertEqual(prediction.shape, (2,30))
        self.assertTrue(bool((prediction[:,0] == 0).all()))
        self.assertTrue(bool((prediction[:,1] == 1).all()))
        self.assertTrue(bool((raw[:,0] < 0).all()))
        self.assertTrue(bool((raw[:,1] > 1).all()))
        torch.testing.assert_close(correction[0],correction[1])

    def test_correction_baseline_and_daily_comparisons_reproduce(self):
        refs = run.references(self.i,self.t,self.y)
        results = {}
        for length in data.LENGTHS:
            net = run.initialize(self.i,17)
            rows = run.infer(net,self.i['ids']['validation'],self.t[length]['validation'],self.y,2,False)
            results[f'history{length}__17'] = {'metrics': run.metrics(rows)}
        summary = run.summarize(results,refs,self.i)
        for length in data.LENGTHS:
            for h in data.HORIZONS:
                entry = summary['aggregates'][str(length)][str(h)]
                self.assertEqual(entry['mae_minus_daily_correction'],
                    refs['metrics']['per_horizon'][str(h)]['mae']-refs['daily_correction']['metrics']['per_horizon'][str(h)]['mae'])
                groups = refs['metrics']['per_horizon'][str(h)]['change_groups']
                self.assertEqual(sum(g['n'] for g in groups.values()), entry['n'])

    def test_metrics_skip_missing_targets_and_match_complete_cases(self):
        net=run.initialize(self.i,17)
        rows=run.infer(net,self.i['ids']['validation'],self.t[10]['validation'],self.y,2,False)
        m=run.metrics(rows)
        self.assertEqual(m['per_horizon']['7']['n'],1)
        self.assertEqual(m['per_horizon']['14']['n'],2)
        self.assertEqual(m['complete_case_n'],1)
        self.assertIsNone(rows[0]['actual'][6])
        changed=deepcopy(rows);changed[0]['prediction'][6]=.99
        self.assertEqual(run.metrics(changed),m)

    def test_training_matches_manual_masked_optimizer(self):
        i=deepcopy(self.i);i['config']['epochs']=1
        net=run.initialize(i,17);cfg=i['config'];ids=i['ids']['train']
        opt=torch.optim.Adam(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        order=run.controls.orders(ids,cfg,17)[0];ix=[ids.index(c) for c in order]
        mask=torch.tensor([self.y[c]['observed_mask'] for c in order],dtype=torch.bool)
        y=torch.tensor([[v if v is not None else 0. for v in self.y[c]['values']] for c in order])
        net.train();opt.zero_grad(set_to_none=True)
        pred=net(self.t[10]['train'][ix])
        errors=(pred-y).abs().masked_fill(~mask,0)
        loss=(errors.sum(1)/mask.sum(1)).mean();loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),cfg['gradient_clip_norm']);opt.step()
        saved,h=run.fit(10,17,i,self.t,self.y,False)
        for k,v in net.state_dict().items():torch.testing.assert_close(v,saved['state_dict'][k],rtol=0,atol=0)
        self.assertEqual(h[0]['observed_training_targets'],59)

    def test_preparation_reproduction_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(data,'OUTPUT',Path(tmp)/'data'),patch.object(data,'context',side_effect=artifacts):
            data.prepare('smoke',check_only=True)
            self.assertFalse(data.OUTPUT.exists())
            data.prepare('smoke');data.prepare('smoke',verify_only=True)
            with self.assertRaisesRegex(ValueError,'Preparation exists'):data.prepare('smoke')
            data.write(data.OUTPUT/'smoke/labels.json',{})
            with self.assertRaisesRegex(ValueError,'Changed/unsafe artifact'):data.prepare('smoke',verify_only=True)

    def test_training_resume_preserves_completed_fits_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(run,'OUTPUT',Path(tmp)/'out'),patch.object(run,'context',side_effect=lambda scale,*args:fixture(scale)):
            run.execute('smoke',1,check_only=True)
            self.assertFalse(run.OUTPUT.exists())
            real=run.fit;calls=[]
            def fail(*args,**kw):
                calls.append(args[0])
                if len(calls)==2:raise RuntimeError('interrupted')
                return real(*args,**kw)
            with patch.object(run,'fit',side_effect=fail),self.assertRaises(RuntimeError):run.execute('smoke',1)
            path=run.OUTPUT/'smoke/fits/history10__17/checkpoint.pt';before=run.sha256(path)
            run.execute('smoke',1,resume=True)
            self.assertEqual(before,run.sha256(path));run.execute('smoke',1,verify_only=True)
            with self.assertRaises(ValueError):run.execute('smoke',1,resume=True)
            run.write(run.OUTPUT/'smoke/fits/history10__17/metrics.json',{})
            with self.assertRaisesRegex(ValueError,'Changed/unsafe artifact'):run.execute('smoke',1,verify_only=True)

    def test_nine_fit_full_lifecycle_requires_smoke(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(run,'OUTPUT',Path(tmp)/'out'),patch.object(run,'context',side_effect=lambda scale,*args:fixture(scale)):
            with self.assertRaises(FileNotFoundError):run.execute('full',1)
            run.execute('smoke',1);run.execute('full',1);run.execute('full',1,verify_only=True)
            summary=run.read(run.OUTPUT/'full/summary.json')
            self.assertEqual(len(summary['fits']),9)
            self.assertEqual(set(summary['aggregates']),{'10','20','31'})


if __name__ == '__main__':
    unittest.main()
