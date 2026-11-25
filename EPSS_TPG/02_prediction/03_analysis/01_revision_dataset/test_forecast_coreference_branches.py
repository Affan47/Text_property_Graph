"""Synthetic-only tests for DQ18; no dataset fitting or reserved test access."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import diagnose_forecast_coreference_branches as diag
from test_forecast_coreference_smoke import setup

torch, run = diag.torch, diag.run


def fixture():
    manifest, views, data = setup()
    manifest['graphs'] = [{'cve': c, 'edges_by_mode': {m: views[m][c]['edge_index'].shape[1]
                           for m in run.cache.shared.MODES}} for c in sum(manifest['ids'].values(), [])]
    ids = manifest['ids']['validation']
    identity = {'schema': diag.VERSION, 'scale': 'smoke', 'protocol': deepcopy(diag.PROTOCOL),
        'ids': ids, 'pinned_files': {}, 'donor_maps': {str(s): diag.donor_map(ids,
            {c: data['split_manifest.json'][c]['group_id'] for c in ids}, s) for s in diag.PROTOCOL['donor_seeds']}}
    models, baseline = {}, {c: {'actual_day60_epss': data['labels.json'][c]['y']['endpoint'], 'predictions': {}} for c in ids}
    for mode in diag.PROTOCOL['variants']:
        for seed in diag.PROTOCOL['model_seeds']:
            spec = run.task.model_spec('epss_only' if mode == 'epss_only' else 'tpg_epss', manifest, views[mode], run.smoke.config())
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                net = run.task.TPGRegressor(**spec).eval()
            key = f'{mode}__{seed}'
            models[key] = {'schema': run.task.VERSION, 'spec': spec, 'state_dict': net.state_dict(),
                           'test_performance_evaluated': False, 'seed': seed, 'edge_configuration': mode}
            pred = run.task.infer(net, ids, views[mode], data['inputs.json'], 8, progress=False)
            for c,p in zip(ids, pred):
                baseline[c]['predictions'][key] = p
    return manifest, views, data, identity, models, baseline


class BranchDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_selection_and_donors(self):
        ids = [f'c{i}' for i in range(40)]
        selected = diag.select_ids(ids, 'smoke')
        self.assertEqual(len(selected), 16)
        self.assertEqual(selected, diag.select_ids(ids, 'smoke'))
        self.assertEqual(selected, [c for c in ids if c in selected])
        groups = {c: str(i//2) for i,c in enumerate(ids)}
        for seed in diag.PROTOCOL['donor_seeds']:
            donors = diag.donor_map(ids, groups, seed)
            self.assertEqual(set(donors.values()), set(ids))
            self.assertTrue(all(groups[c] != groups[d] for c,d in donors.items()))
        with self.assertRaisesRegex(ValueError, 'No different-group'):
            diag.donor_map(['a','b','c','d'], {'a':1, 'b':1, 'c':1, 'd':2}, 101)

    def test_shift_preserves_masks_and_missing_padding(self):
        x = torch.tensor([[[.999, 1, .1, 1, 0], [0, 0, .2, 1, .1], [.001, 1, 0, 1, .2]]])
        original = x.clone()
        for amount in (.01, -.01):
            changed = diag.shifted_sequence(x, amount)
            self.assertTrue(torch.equal(x, original))
            self.assertTrue(torch.equal(changed[:,:,1:], x[:,:,1:]))
            self.assertEqual(float(changed[0,1,0]), 0)
            self.assertTrue(bool(((changed[:,:,0] >= 0) & (changed[:,:,0] <= 1)).all()))
        self.assertEqual(float(diag.shifted_sequence(x, .01)[0,0,0]), 1)
        self.assertEqual(float(diag.shifted_sequence(x, -.01)[0,2,0]), 0)

    def test_latent_swaps_match_input_swaps(self):
        m, views, data, identity, models, baseline = fixture()
        ids = identity['ids']; donors = identity['donor_maps']['101']
        net = run.task.restore(models['all_coref__17'])
        pred, rep = diag.encode(net, ids, views['all_coref'], data['inputs.json'], progress=False)
        before = rep.clone()
        for probe in ('swap_epss', 'swap_graph'):
            x = {c: data['inputs.json'][donors[c] if probe == 'swap_epss' else c] for c in ids}
            graphs = {c: views['all_coref'][donors[c] if probe == 'swap_graph' else c] for c in ids}
            direct = run.task.infer(net, ids, graphs, x, 8, progress=False)
            latent = diag.head_predictions(net, diag.intervene(rep, ids, 16, probe, donors))
            self.assertTrue(diag.close(direct, latent))
        self.assertTrue(torch.equal(rep, before))
        self.assertTrue(diag.close(pred, diag.head_predictions(net, rep)))
        self.assertEqual(len(net.head._forward_pre_hooks), 0)

    def test_zeroing_branch_boundaries_and_control(self):
        x = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        y = diag.intervene(x, ['a','b'], 2, 'zero_epss')
        self.assertTrue(torch.equal(y[:,:2], torch.zeros(2,2)))
        self.assertTrue(torch.equal(y[:,2:], x[:,2:]))
        y = diag.intervene(x, ['a','b'], 2, 'zero_graph')
        self.assertTrue(torch.equal(y[:,:2], x[:,:2]))
        self.assertTrue(torch.equal(y[:,2:], torch.zeros(2,4)))
        self.assertTrue(torch.equal(diag.intervene(x, ['a','b'], 6, 'zero_graph'), x))

    def test_response_empty_direction_and_tolerance(self):
        self.assertIsNone(diag.response([], [], [])['mae_difference'])
        with self.assertRaisesRegex(ValueError, 'alignment'):
            diag.response([.1], [], [])
        r = diag.response([.2,.4], [.3,.2], [.2,.4])
        self.assertAlmostEqual(r['mean_absolute_prediction_change'], .15)
        self.assertAlmostEqual(r['mae_difference'], .15)
        self.assertEqual(r['changed_cves'], 2)
        self.assertEqual(diag.response([0], [1e-8], [0])['changed_cves'], 0)

    def test_hook_cleanup_and_future_feature_rejection(self):
        m, views, data, identity, models, baseline = fixture()
        net = run.task.restore(models['all_coref__17'])
        changed = deepcopy(data['inputs.json'])
        changed[identity['ids'][0]]['X']['future_epss'] = 1
        with self.assertRaisesRegex(ValueError, 'Unexpected predictor'):
            diag.encode(net, identity['ids'], views['all_coref'], changed, progress=False)
        self.assertEqual(len(net.head._forward_pre_hooks), 0)

    def test_all_models_repeatable_no_mutation_or_test_access(self):
        m, views, data, identity, models, baseline = fixture()
        original = deepcopy(views)
        hashes = {k: run.digest_state(s['state_dict']) for k,s in models.items()}
        a, rows = diag.diagnose(m, views, data, identity, models, baseline, False)
        b, rows2 = diag.diagnose(m, views, data, identity, models, baseline, False)
        self.assertEqual(a,b); self.assertEqual(rows,rows2)
        self.assertEqual(len(rows), 12*11*2)
        self.assertEqual(len(a['checks']), 12)
        self.assertEqual(len(a['seed_aggregates']), 4*7)
        for mode in views:
            for c in views[mode]:
                for key, value in views[mode][c].items():
                    if torch.is_tensor(value):
                        self.assertTrue(torch.equal(value, original[mode][c][key]))
        self.assertEqual(hashes, {k: run.digest_state(s['state_dict']) for k,s in models.items()})
        for r in rows:
            if r['variant'] == 'epss_only' and r['probe'] in ('swap_graph','zero_graph'):
                self.assertEqual(r['original_prediction'], r['probe_prediction'])
        bad = deepcopy(identity); bad['ids'] = m['ids']['train']
        with self.assertRaisesRegex(ValueError, 'Only selected validation'):
            diag.diagnose(m, views, data, bad, models, baseline, False)
        bad_models = dict(models); bad_models.pop('epss_only__43')
        with self.assertRaisesRegex(ValueError, 'All fixed models'):
            diag.diagnose(m, views, data, identity, bad_models, baseline, False)

    def test_save_verify_smoke_gate_and_no_overwrite(self):
        values = fixture()
        def context(scale):
            result = list(values)
            result[3] = deepcopy(values[3]); result[3]['scale'] = scale
            return result
        with tempfile.TemporaryDirectory() as folder, patch.object(diag, 'ROOT', Path(folder)/'diagnostic'), \
             patch.object(diag, 'context', side_effect=context), \
             patch.object(run, 'fit', side_effect=AssertionError('No training allowed')):
            diag.execute('smoke', check_only=True)
            self.assertFalse(diag.ROOT.exists())
            with self.assertRaises(FileNotFoundError):
                diag.execute('full')
            diag.execute('smoke')
            diag.execute('smoke', verify_only=True)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                diag.execute('smoke')
            diag.execute('full')
            diag.execute('full', verify_only=True)
            (diag.ROOT/'smoke/analysis.txt').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'Changed diagnostic'):
                diag.execute('full', verify_only=True)


if __name__ == '__main__':
    unittest.main()
