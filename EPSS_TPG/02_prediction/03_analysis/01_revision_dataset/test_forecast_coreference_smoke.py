"""R02 synthetic-only matching, leakage, checkpoint and training-loop checks."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_coreference_smoke as run
from test_forecast_tpg import fixture
import torch


def setup():
    manifest, graphs, data = fixture()
    views = {v: deepcopy(graphs) for v in run.VARIANTS}
    # Synthetic graphs have one ordinary and one COREF edge. No fixture data
    # comes from the saved dataset or reserved test split.
    idx = manifest['type_vocabulary']['edge_types']['COREF']
    for v in run.VARIANTS:
        for graph in views[v].values():
            graph['edge_type'][1] = idx
            if v == 'no_coref':
                graph['edge_index'] = graph['edge_index'][:, :1]
                graph['edge_type'] = graph['edge_type'][:1]
                graph['edge_features'] = graph['edge_features'][:1]
    return manifest, views, data


class CoreferenceSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_fixed_smoke_budget_and_dropout(self):
        c = run.config()
        self.assertEqual((c['epochs'], c['batch_size'], c['seeds'], c['dropout']), (3, 8, [17], 0))
        self.assertEqual(c['device'], 'cpu')

    def test_initial_weights_and_batches_match(self):
        manifest, views, data = setup()
        c = run.config()
        states = [run.initialize('tpg_epss', manifest, views[v], data, c) for v in run.cache.MODES]
        self.assertEqual(len({s['initial_state_sha256'] for s in states}), 1)
        for key in states[0]['state_dict']:
            self.assertTrue(all(torch.equal(states[0]['state_dict'][key], s['state_dict'][key]) for s in states))
        self.assertEqual(run.epoch_orders(manifest['ids']['train'], c), run.epoch_orders(manifest['ids']['train'], c))

    def test_validation_targets_do_not_initialize_model(self):
        m, views, data = setup()
        original = run.initialize('tpg_epss', m, views['all_coref'], data, run.config())
        other = deepcopy(data)
        for c in m['ids']['validation']:
            other['labels.json'][c]['y']['endpoint'] = .99
        self.assertEqual(original['initial_state_sha256'],
                         run.initialize('tpg_epss', m, views['all_coref'], other, run.config())['initial_state_sha256'])

    def test_selection_excludes_unselected_records_and_rejects_test(self):
        m, _, data = setup()
        changed = deepcopy(data)
        for name in ('inputs.json', 'labels.json', 'split_manifest.json'):
            changed[name]['unused-test-record'] = object()
        scoped = run.select_data(m, changed)
        self.assertTrue(all('unused-test-record' not in values for values in scoped.values()))
        bad = deepcopy(m); bad['ids']['test'] = bad['ids'].pop('validation')
        with self.assertRaisesRegex(ValueError, 'Only train/validation'):
            run.select_data(bad, data)
        changed = deepcopy(data)
        changed['labels.json'][m['ids']['validation'][0]]['target_date'] = '2024-03-05'
        with self.assertRaisesRegex(ValueError, 'Day-30 anchor'):
            run.select_data(m, changed)

    def test_forward_inputs_exclude_targets_and_preserve_missingness(self):
        m, views, data = setup()
        batch = run.task.make_batch(m['ids']['train'], views['all_coref'], data['inputs.json'])
        self.assertFalse({'y', 'cve', 'target', 'target_date'} & set(batch.keys()))
        self.assertEqual(tuple(batch.epss_sequence.shape), (2, 31, 5))
        self.assertEqual(float(batch.epss_sequence[0, 30, 1]), 0)
        changed = deepcopy(data['inputs.json'])
        changed[m['ids']['train'][0]]['X']['future_score'] = .9
        with self.assertRaisesRegex(ValueError, 'Unexpected predictor'):
            run.task.make_batch(m['ids']['train'], views['all_coref'], changed)

    def test_fit_reproducible_gradients_and_checkpoint_reload(self):
        m, views, data = setup()
        settings = run.config()
        initial = run.initialize('tpg_epss', m, views['all_coref'], data, settings)
        orders = run.epoch_orders(m['ids']['train'], settings)
        first, history, grad = run.fit('all_coref', initial, orders, m, views['all_coref'], data, settings, False)
        second, history2, grad2 = run.fit('all_coref', initial, orders, m, views['all_coref'], data, settings, False)
        self.assertEqual(history, history2); self.assertEqual(grad, grad2)
        self.assertEqual(run.digest_state(first['state_dict']), run.digest_state(second['state_dict']))
        self.assertNotEqual(run.digest_state(first['state_dict']), initial['initial_state_sha256'])
        self.assertEqual(first['selected_epoch'], min(history, key=lambda r:r['validation_mae'])['epoch'])
        self.assertTrue(all(g['steps_with_nonzero_gradient'] > 0 for g in grad.values()))
        self.assertEqual(run.digest_state(initial['state_dict']), initial['initial_state_sha256'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.pt'; torch.save(first, path)
            loaded = torch.load(path, weights_only=True)
            batch = run.task.make_batch(m['ids']['validation'], views['all_coref'], data['inputs.json'])
            with torch.no_grad():
                self.assertTrue(torch.equal(run.task.restore(first)(batch), run.task.restore(loaded)(batch)))

    def test_graph_changes_do_not_affect_epss_only_reference(self):
        m, views, data = setup()
        initial = run.initialize('epss_only', m, views['all_coref'], data, run.config())
        with torch.random.fork_rng():
            model = run.task.TPGRegressor(**initial['spec']).eval()
            model.load_state_dict(initial['state_dict'])
        ids = m['ids']['validation']
        with torch.no_grad():
            a = model(run.task.make_batch(ids, views['all_coref'], data['inputs.json']))
            b = model(run.task.make_batch(ids, views['no_coref'], data['inputs.json']))
        self.assertTrue(torch.equal(a, b))

    def test_end_to_end_smoke_and_tamper_detection(self):
        m, views, data = setup()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root/'cache'; source.mkdir()
            lock = source/'cache_lock.json'; run.write_json(lock, {'synthetic': True})
            identity = {'schema': run.VERSION, 'config': run.config(), 'ids': m['ids'],
                        'cache_directory': str(source), 'pinned_files': {str(lock): run.sha256(lock)}}
            output = root/'results/smoke'
            with patch.object(run, 'context', return_value=(m, views, data, identity)), \
                    patch.object(run.task, 'RESULT_ROOT', root/'results'):
                with patch.object(run, 'fit', side_effect=AssertionError('preflight must not train')):
                    run.run(source, output, check_only=True)
                self.assertFalse(output.exists())
                run.run(source, output)
                run.run(source, output, verify_only=True)
                summary = run.read_json(output/'summary.json')
                self.assertEqual(len(summary['results']), 5)
                self.assertFalse(summary['test_performance_evaluated'])
                self.assertEqual(len(run.read_json(output/'predictions.json')), 2)
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    run.run(source, output)
                (output/'predictions.json').write_text('[]', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Changed result'):
                    run.run(source, output, verify_only=True)


if __name__ == '__main__':
    unittest.main()
