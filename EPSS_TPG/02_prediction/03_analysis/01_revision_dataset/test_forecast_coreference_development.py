"""R03 synthetic training, matching, recovery and leakage-contract tests."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_coreference_training as run
from test_forecast_coreference_smoke import setup
import torch


def settings(seed=17):
    result = run.smoke.config()
    result.update(epochs=2, patience=2, seeds=[seed])
    return result


class DevelopmentTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_fixed_protocol_no_tunable_seed_or_budget(self):
        p = run.read_json(run.cache.PROTOCOL)
        self.assertEqual(run.settings_for(p)['seeds'], [17, 29, 43])
        self.assertEqual(run.settings_for(p, 29)['seeds'], [29])
        self.assertEqual(run.settings_for(p)['epochs'], 100)
        with self.assertRaisesRegex(ValueError, 'Unregistered seed'):
            run.settings_for(p, 1)
        p['config']['dropout'] = .1
        with self.assertRaisesRegex(ValueError, 'protocol differs'):
            run.settings_for(p)

    def test_initialization_batch_matching_and_seed_variation(self):
        m, views, data = setup()
        hashes = []
        for seed in (17, 29, 43):
            states = [run.initialize('tpg_epss', m, views[v], data, settings(seed)) for v in run.cache.shared.MODES]
            self.assertEqual(len({x['initial_state_sha256'] for x in states}), 1)
            hashes.append(states[0]['initial_state_sha256'])
            orders = run.epoch_orders(m['ids']['train'], settings(seed))
            self.assertEqual(orders, run.epoch_orders(m['ids']['train'], settings(seed)))
            self.assertTrue(all(sorted(o) == sorted(m['ids']['train']) for o in orders))
        self.assertEqual(len(set(hashes)), 3)

    def test_early_stop_strict_ties_and_earliest_checkpoint(self):
        m, views, data = setup()
        c = settings(); c.update(epochs=8, patience=2)
        init = run.initialize('tpg_epss', m, views['all_coref'], data, c)
        orders = run.epoch_orders(m['ids']['train'], c)
        with patch.object(run.task, 'infer', side_effect=lambda net, ids, *a, **k: [.5]*len(ids)):
            saved, history, grad = run.fit('all_coref', init, orders, m, views['all_coref'], data, c, False)
        self.assertEqual(len(history), 3)
        self.assertEqual(saved['selected_epoch'], 1)
        run.history_check(history, c, 1)
        self.assertTrue(all(g['steps_with_gradient'] == 3 for g in grad.values()))
        with self.assertRaisesRegex(ValueError, 'stopped too early'):
            run.history_check(history[:2], c, 1)
        continued = history + [{**history[-1], 'epoch': 4}]
        with self.assertRaisesRegex(ValueError, 'past stopping rule'):
            run.history_check(continued, c, 1)

    def test_fit_deterministic_and_restorable(self):
        m, views, data = setup(); c = settings()
        init = run.initialize('tpg_epss', m, views['all_coref'], data, c)
        orders = run.epoch_orders(m['ids']['train'], c)
        a, h, g = run.fit('all_coref', init, orders, m, views['all_coref'], data, c, False)
        b, h2, g2 = run.fit('all_coref', init, orders, m, views['all_coref'], data, c, False)
        self.assertEqual(h, h2); self.assertEqual(g, g2)
        self.assertEqual(run.digest_state(a['state_dict']), run.digest_state(b['state_dict']))
        self.assertNotEqual(run.digest_state(a['state_dict']), init['initial_state_sha256'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.pt'; run.atomic_torch_save(path, a)
            loaded = torch.load(path, weights_only=True)
            pred = run.task.infer(run.task.restore(a), m['ids']['validation'], views['all_coref'], data['inputs.json'], 8, progress=False)
            pred2 = run.task.infer(run.task.restore(loaded), m['ids']['validation'], views['all_coref'], data['inputs.json'], 8, progress=False)
            self.assertEqual(pred, pred2)

    def test_input_and_target_separation_and_initialization(self):
        m, views, data = setup(); c = settings()
        changed = deepcopy(data)
        for cv in m['ids']['validation']:
            changed['labels.json'][cv]['y']['endpoint'] = .99
        a = run.initialize('tpg_epss', m, views['all_coref'], data, c)
        b = run.initialize('tpg_epss', m, views['all_coref'], changed, c)
        self.assertEqual(a['initial_state_sha256'], b['initial_state_sha256'])
        batch = run.task.make_batch(m['ids']['train'], views['all_coref'], data['inputs.json'])
        self.assertFalse({'y', 'target_date', 'cve'} & set(batch.keys()))
        self.assertEqual(tuple(batch.epss_sequence.shape), (2, 31, 5))
        changed['inputs.json'][m['ids']['train'][0]]['X']['future_epss'] = .99
        with self.assertRaisesRegex(ValueError, 'Unexpected predictor'):
            run.task.make_batch(m['ids']['train'], views['all_coref'], changed['inputs.json'])
        bad = deepcopy(m); bad['ids']['test'] = bad['ids'].pop('validation')
        with self.assertRaisesRegex(ValueError, 'Only train/validation'):
            run.smoke.select_data(bad, data)

    def test_missingness_empty_groups_and_all_seed_aggregation(self):
        m, _, data = setup()
        scores = [.1]*len(m['ids']['validation'])
        result = run.score_predictions(scores, m, data)
        self.assertEqual(sum(g['n'] for g in result['missingness'].values()), len(scores))
        for group in result['missingness'].values():
            if group['n'] == 0:
                self.assertIsNone(group['metrics'])
        rows = [{'variant': 'persistence', 'seed': None, **result}]
        for v in run.VARIANTS:
            for seed, error in ((17, .1), (29, .2), (43, .3)):
                r = deepcopy(result); r['metrics']['mae'] = error
                rows.append({'variant': v, 'seed': seed, **r})
        aggregated, pairs = run.aggregate(rows, [17, 29, 43])
        self.assertAlmostEqual(aggregated['all_coref']['mae']['mean'], .2)
        self.assertAlmostEqual(aggregated['all_coref']['mae']['sample_std'], .1)
        self.assertEqual(aggregated['all_coref']['mae']['defined_seeds'], 3)
        self.assertTrue(pairs)

    def test_lazy_cache_shares_nodes_and_rejects_unselected_ids(self):
        m, views, _ = setup()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'graphs').mkdir()
            ids = m['ids']['train']
            for cve in ids:
                g = views['all_coref'][cve]
                edges = {k: g[k] for k in run.cache.shared.EDGE_FIELDS}
                shared = {k: v for k,v in g.items() if k not in edges}
                mask = torch.ones(len(g['edge_type']), dtype=torch.bool)
                torch.save({'shared': shared, 'edges': edges,
                            'edge_masks': {mode: mask for mode in run.cache.shared.MODES}}, root/'graphs'/(cve+'.pt'))
            bank = run.GraphBank(root, ids, capacity=1)
            a = bank.view('all_coref')[ids[0]]; b = bank.view('no_coref')[ids[0]]
            self.assertIs(a['x'], b['x'])
            bank.bundle(ids[1]); self.assertEqual(len(bank.items), 1)
            with self.assertRaisesRegex(ValueError, 'outside development'):
                bank.bundle('unselected-test-record')

    def test_end_to_end_resume_all_seeds_and_tamper_detection(self):
        m, views, data = setup()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root/'cache'; source.mkdir()
            lock = source/'cache_lock.json'; run.write_json(lock, {'synthetic': True})
            cfg = settings(); cfg['seeds'] = [17, 29, 43]
            protocol = {'config': cfg, 'limitations': ['Synthetic test only']}
            identity = {'schema': run.VERSION, 'protocol': protocol, 'ids': m['ids'],
                        'cache_directory': str(source), 'pinned_files': {str(lock): run.sha256(lock)},
                        'cache_lock_sha256': run.sha256(lock)}
            output = root/'results/full'
            original_fit = run.fit
            calls = []
            def interrupted(*args, **kwargs):
                calls.append(args[0])
                if len(calls) == 2:
                    raise RuntimeError('synthetic interruption')
                return original_fit(*args, **kwargs, progress=False)
            with patch.object(run, 'context', return_value=(m, views, data, identity)), \
                    patch.object(run.task, 'RESULT_ROOT', root/'results'):
                with patch.object(run, 'fit', side_effect=AssertionError('preflight must not train')):
                    run.run(source, output, check_only=True)
                self.assertFalse(output.exists())
                with patch.object(run, 'fit', side_effect=interrupted), self.assertRaisesRegex(RuntimeError, 'interruption'):
                    run.run(source, output)
                completed = run.fit_directory(output, 17, 'no_coref')/'checkpoint.pt'
                digest = run.sha256(completed)
                with patch.object(run, 'fit', wraps=original_fit) as fit:
                    run.run(source, output, resume=True)
                    self.assertEqual(fit.call_count, 11)
                self.assertEqual(run.sha256(completed), digest)
                run.run(source, output, verify_only=True)
                summary = run.read_json(output/'summary.json')
                self.assertEqual(len(summary['results']), 13)
                self.assertFalse(summary['test_performance_evaluated'])
                self.assertEqual(len(run.read_json(output/'predictions.json')), 2)
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    run.run(source, output)
                with self.assertRaisesRegex(ValueError, 'unfinished output'):
                    run.run(source, output, resume=True)
                (output/'predictions.json').write_text('[]')
                with self.assertRaisesRegex(ValueError, 'Changed result'):
                    run.run(source, output, verify_only=True)


if __name__ == '__main__':
    unittest.main()
