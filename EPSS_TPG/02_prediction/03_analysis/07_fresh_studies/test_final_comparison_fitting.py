"""Synthetic-only final fit/store tests; no dataset, encoder, network or scientific fits."""
from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_combined_training import fixture
from epss.combined_training_v1 import data as old_data
from epss.combined_training_v2 import pooled, training as old_training
from epss.final_comparison_v1 import fitting, model, store
from epss.final_comparison_v1.common import c, exclusive, read, write


class LabelBomb(dict):
    def __getitem__(self, key):
        raise AssertionError('Predictor-only path accessed a label')


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        torch.manual_seed(222)
        self.contract, _, self.graph = fixture()
        self.graphs = {}
        self.ctx = {
            'contract': self.contract, 'directory': self.root / 'cache',
            'index': {}, 'receipts': {}, 'vectors': {}, 'pooled': {'D': {}, 'DM': {}},
            'ids': {'train': ['a', 'b'], 'validation': ['c', 'd'], 'assessment': ['e']},
            'labels': {cid: {'value': value, 'date': '2024-01-01'}
                       for cid, value in zip('abcde', [.15, .8, .3, .9, .2])},
        }
        for i, cid in enumerate('abcde'):
            self.ctx['vectors'][cid] = [i * .1]
            self.ctx['index'][cid] = {'views': {}}
            posts = cid in 'ace'
            for view in ('D', 'DM'):
                key = f'{cid}_{view}' if view == 'D' or posts else f'{cid}_D'
                flags = {'D': True, 'M': view == 'DM' and posts, 'G': False, 'N': False}
                self.ctx['index'][cid]['views'][view] = {'object_key': key, 'source_mask': flags}
                if key not in self.graphs:
                    graph = deepcopy(self.graph)
                    graph['x'][:, len(self.contract['node_type_vocab']):][:, :768] += i * .01
                    self.graphs[key] = graph
                    obj = self.ctx['directory'] / 'objects' / key
                    write(obj / 'graph_receipt.json', {'statistics': {'nodes': 3, 'edges': 3}})
                    write(obj / 'embedding_receipt.json', {'synthetic': key})
                    self.ctx['receipts'][key] = [c.sha(obj / name) for name in
                                                ('graph_receipt.json', 'embedding_receipt.json')]
                self.ctx['pooled'][view][cid] = pooled.reduce_graph(self.graphs[key], self.contract)[0].tolist()
        self.loader = patch.object(old_data.GraphStore, 'load', side_effect=lambda key: deepcopy(self.graphs[key]))
        self.disk = self.loader.start()

    def tearDown(self):
        self.loader.stop()
        self.tmp.cleanup()

    def identity(self, name='structured', seed=17):
        return {'settings': {**model.SETTINGS, 'seeds': list(model.SEEDS)}, 'contract': self.contract,
                'model': name, 'seed': seed, 'source_pins': {}, 'run': 'synthetic', 'fold': 1}

    def fit(self, destination='fit', name='structured', seed=17, **kwargs):
        return fitting.fit(self.root / destination, self.ctx, self.identity(name, seed), name, seed, **kwargs)

    def test_all_five_models_one_epoch_reload_and_parameter_counts(self):
        for name in model.MODELS:
            with self.subTest(model=name):
                result = self.fit(name, name, epochs_override=1)
                replay = self.fit(name, name, verify=True, epochs_override=1, budget_gib=33)
                self.assertEqual(result, replay)
                self.assertEqual(result['best_epoch'], 1)
                self.assertEqual([r['cve'] for r in result['predictions']['assessment']], ['e'])
                self.assertEqual([r['cve'] for r in result['predictions']['validation']], ['c', 'd'])
                frozen = read(self.root / name / 'fit_contract.json')
                count = sum(p.numel() for p in model.initialize(self.contract, name, 17).parameters())
                self.assertEqual(frozen['architecture']['parameter_count'], count)
                self.assertNotIn('budget_gib', frozen)
                self.assertEqual([p.name for p in (self.root / name).glob('*.pt')], ['best_checkpoint.pt'])
                locked = read(self.root / name / 'artifact_lock.json')['files']
                self.assertNotIn('execution_profile.json', locked)
                self.assertNotIn(b'\r', (self.root / name / 'predictions.csv').read_bytes())

    def test_fixed_settings_seeds_and_integration_override(self):
        self.assertEqual(model.settings(self.identity(), 1)['epochs'], 1)
        for key, value in [('layers', 3), ('hidden', 64), ('edge_chunk_size', 4096), ('threads', 8),
                           ('objective', 'mae'), ('weight_decay', .01), ('patience', 2)]:
            identity = self.identity()
            identity['settings'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'fixed'):
                model.settings(identity)
        for value in (0, 2, True):
            with self.assertRaisesRegex(ValueError, 'one-epoch'):
                model.settings(self.identity(), value)
        for seed in model.SEEDS:
            left = model.initialize(self.contract, 'pooled_D', seed)
            right = model.initialize(self.contract, 'pooled_D', seed)
            for a, b in zip(left.parameters(), right.parameters()):
                self.assertTrue(torch.equal(a, b))
        with self.assertRaisesRegex(ValueError, 'seed'):
            model.initialize(self.contract, 'tpg_D', 19)

    def test_model_forward_gradient_parity_with_existing_architecture(self):
        for name, (representation, _) in model.MODELS.items():
            net = model.initialize(self.contract, name, 17)
            reference = old_training.initialize(self.contract, model.SETTINGS, representation, 17)
            resident = store.ResidentStore(self.ctx, name).preload(['a'], 'synthetic parity')
            inputs = resident.inputs('a')
            a, b = net(inputs), reference(inputs)
            self.assertTrue(torch.equal(a, b), name)
            (a - .2).square().backward()
            (b - .2).square().backward()
            for left, right in zip(net.parameters(), reference.parameters()):
                self.assertTrue(torch.equal(left, right))
                if left.grad is not None:
                    self.assertTrue(torch.equal(left.grad, right.grad), name)

    def test_typed_edges_properties_and_node_subtypes_have_gradient_paths(self):
        net = model.initialize(self.contract, 'tpg_D', 17)
        resident = store.ResidentStore(self.ctx, 'tpg_D').preload(['a'], 'synthetic graph')
        inputs = resident.inputs('a')
        prediction = net(inputs)
        prediction.backward()
        self.assertEqual(len(net.layers), 2)
        self.assertEqual(net.layers[0].edge_type_vocab, self.contract['edge_type_vocab'])
        self.assertGreater(float(net.layers[0].edge_embed.weight.grad.abs().sum()), 0)
        self.assertGreater(float(net.layers[0].edge_semantics.weight.grad.abs().sum()), 0)
        self.assertGreater(float(net.node_project[0].weight.grad[:, :3].abs().sum()), 0)
        self.assertGreater(float(net.subtype_project.weight.grad.abs().sum()), 0)
        self.assertFalse(inputs['graph']['x'].requires_grad)
        changed = deepcopy(inputs)
        changed['graph']['edge_type'] = 1 - changed['graph']['edge_type']
        self.assertGreater(abs(float(net(changed)) - float(prediction)), 0)
        with self.assertRaisesRegex(ValueError, 'targets'):
            net({**inputs, 'target': .2})

    def test_pooled_uses_provided_mean_and_never_loads_graphs(self):
        self.disk.reset_mock()
        resident = store.ResidentStore(self.ctx, 'pooled_DM').preload(['a', 'b'], 'pooled')
        self.disk.assert_not_called()
        self.assertTrue(torch.equal(resident.inputs('a')['graph']['pooled_vector'],
                                    torch.tensor(self.ctx['pooled']['DM']['a'])))
        self.assertFalse(resident.inputs('a')['graph']['pooled_vector'].requires_grad)
        for name in ('pooled_D', 'pooled_DM'):
            a = store.ResidentStore(self.ctx, name).preload(['b'], 'missing posts').inputs('b')
            net = model.initialize(self.contract, name, 17)
            if name == 'pooled_D':
                expected = net(a).detach()
            else:
                self.assertTrue(torch.equal(expected, net(a)))

    def test_complete_ram_residency_without_eviction(self):
        resident = store.ResidentStore(self.ctx, 'tpg_DM').preload(['a', 'b', 'c', 'd'], 'both roles')
        self.assertEqual(self.disk.call_count, 4)
        for _ in range(3):
            for cid in 'abcd':
                resident.inputs(cid)
        self.assertEqual(self.disk.call_count, 4)
        self.assertEqual(len(resident.items), 4)
        self.assertEqual(resident.hits, 12)
        with self.assertRaisesRegex(ValueError, 'preloaded'):
            resident.inputs('e')
        resident.preload(['e'], 'assessment')
        self.assertEqual(len(resident.items), 1)
        self.assertEqual(set(resident.rows), {'e'})

    def test_ram_rejection_precedes_loading_and_does_not_drop_cves(self):
        resident = store.ResidentStore(self.ctx, 'tpg_DM', budget_gib=.00002)
        with self.assertRaisesRegex(ValueError, 'No CVEs were dropped'):
            resident.preload(['a', 'b', 'c', 'd'], 'optimization and selection')
        self.disk.assert_not_called()
        self.assertEqual(resident.items, {})
        self.assertEqual(resident.rows, {})
        with self.assertRaisesRegex(ValueError, 'preloaded'):
            resident.inputs('a')
        with self.assertRaisesRegex(ValueError, 'Positive'):
            store.ResidentStore(self.ctx, 'tpg_D', float('nan'))

    def test_corrupt_receipt_and_expanded_size_fail_closed(self):
        path = self.ctx['directory'] / 'objects/a_D/graph_receipt.json'
        write(path, {'statistics': {'nodes': 4, 'edges': 3}})
        with self.assertRaisesRegex(ValueError, 'receipt changed'):
            store.ResidentStore(self.ctx, 'tpg_D').preload(['a'], 'receipt guard')
        self.ctx['receipts']['a_D'][0] = c.sha(path)
        resident = store.ResidentStore(self.ctx, 'tpg_D')
        with self.assertRaisesRegex(ValueError, 'size differs'):
            resident.preload(['a'], 'shape guard')
        self.assertEqual(resident.items, {})
        self.assertEqual(resident.used, 0)

    def test_assessment_predictors_labels_and_graphs_wait_until_selected(self):
        allowed = {'assessment': False}
        labels = self.ctx['labels']
        pooled_vectors = self.ctx['pooled']['D']

        class Guarded(dict):
            def __getitem__(self, cid):
                if cid == 'e' and not allowed['assessment']:
                    raise AssertionError('Assessment accessed before selected checkpoint reload')
                return super().__getitem__(cid)

        self.ctx['labels'] = Guarded(labels)
        self.ctx['pooled']['D'] = Guarded(pooled_vectors)
        original = fitting._finish_predictions

        def finish(*args, **kwargs):
            self.assertTrue((self.root / 'guarded/best_checkpoint.pt').exists())
            allowed['assessment'] = True
            return original(*args, **kwargs)

        with patch.object(fitting, '_finish_predictions', side_effect=finish):
            self.fit('guarded', 'pooled_D', epochs_override=1)
        self.assertTrue(allowed['assessment'])

    def test_integration_empty_assessment_never_reads_assessment_material(self):
        self.ctx['ids']['assessment'] = []

        class SelectionOnly(dict):
            def __getitem__(self, cid):
                if cid == 'e':
                    raise AssertionError('Integration read assessment material')
                return super().__getitem__(cid)

        self.ctx['labels'] = SelectionOnly(self.ctx['labels'])
        self.ctx['pooled']['D'] = SelectionOnly(self.ctx['pooled']['D'])
        result = self.fit('integration_no_assessment', 'pooled_D', epochs_override=1)
        self.assertEqual(result['predictions']['assessment'], [])
        self.assertIsNone(result['assessment_identity'])
        self.assertEqual(result, self.fit('integration_no_assessment', 'pooled_D',
                                         epochs_override=1, verify=True))

    def test_raw_mse_weights_cves_equally_and_normalizes_last_batch(self):
        class Scalar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(.3))

            def forward(self, inputs):
                return self.value

        class Inputs:
            model = 'synthetic'

            def inputs(self, cid):
                return {}

        ids = [str(i) for i in range(9)]
        ctx = {'labels': {cid: {'value': int(cid) / 10, 'date': '2024-01-01'} for cid in ids}}
        net = Scalar()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.)
        order = list(ids)
        random.Random(17001).shuffle(order)
        gradients = []
        step = optimizer.step

        def measured_step():
            gradients.append(float(net.value.grad))
            return step()

        with patch.object(optimizer, 'step', side_effect=measured_step):
            result = fitting.train_epoch(net, optimizer, Inputs(), ctx, ids, model.SETTINGS, 17, 1)
        self.assertEqual(result['optimizer_steps'], 2)
        self.assertAlmostEqual(result['train_mse'], np.mean([(.3 - i / 10)**2 for i in range(9)]), places=7)
        self.assertAlmostEqual(gradients[0], 2 * (.3 - np.mean([int(cid) / 10 for cid in order[:8]])), places=6)
        self.assertAlmostEqual(gradients[1], 2 * (.3 - int(order[-1]) / 10), places=6)

    def _constant_selection(self):
        original = fitting.evaluate

        def evaluate(*args, **kwargs):
            result = original(*args, **kwargs)
            # Force equal RMSE to exercise best-only/tie/resume paths while
            # predictions still come from actual trained synthetic models.
            if 'selection' in args[-1]:
                result['metrics'] = {'n': 2, 'mse': .04, 'rmse': .2, 'mae': .15}
            return result

        return patch.object(fitting, 'evaluate', side_effect=evaluate)

    def test_earliest_tie_patience_and_only_one_best_checkpoint_write(self):
        with self._constant_selection(), patch.object(fitting, 'save_torch', wraps=fitting.save_torch) as saves:
            result = self.fit('tie')
        self.assertEqual(result['best_epoch'], 1)
        self.assertEqual(len(result['history']), 16)
        self.assertEqual(result['history'][-1]['patience_used'], 15)
        self.assertEqual(saves.call_count, 1)
        checkpoint = torch.load(self.root / 'tie/best_checkpoint.pt', weights_only=True)
        self.assertEqual(len(checkpoint['history']), 1)
        self.assertTrue(checkpoint['optimizer']['state'])
        self.assertEqual(set(checkpoint['rng']), {'python', 'numpy', 'torch'})

    def test_resume_replays_post_best_history_optimizer_and_rng_exactly(self):
        original = fitting.train_epoch

        def probed(*args, **kwargs):
            result = original(*args, **kwargs)
            result['rng_probe'] = [random.random(), float(np.random.random()), float(torch.rand(()))]
            return result

        def interrupt(*args, **kwargs):
            if args[-1] == 4:
                raise InterruptedError('synthetic interruption')
            return probed(*args, **kwargs)

        with self._constant_selection():
            with patch.object(fitting, 'train_epoch', side_effect=probed):
                expected = self.fit('clean')
            with patch.object(fitting, 'train_epoch', side_effect=interrupt):
                with self.assertRaises(InterruptedError):
                    self.fit('resume')
            self.assertEqual(len(read(self.root / 'resume/history.json')), 3)
            epochs = []

            def replay(*args, **kwargs):
                epochs.append(args[-1])
                return probed(*args, **kwargs)

            with patch.object(fitting, 'train_epoch', side_effect=replay):
                actual = self.fit('resume', resume=True, budget_gib=48)
            self.assertEqual(epochs[:3], [2, 3, 4])
            self.assertEqual(actual, expected)
            self.assertEqual(self.fit('resume', verify=True, budget_gib=32), expected)
        self.assertEqual((self.root / 'clean/predictions.csv').read_bytes(),
                         (self.root / 'resume/predictions.csv').read_bytes())

    def test_resume_rejects_altered_post_best_epoch(self):
        original = fitting.train_epoch

        def interrupt(*args, **kwargs):
            if args[-1] == 4:
                raise InterruptedError()
            return original(*args, **kwargs)

        with self._constant_selection():
            with patch.object(fitting, 'train_epoch', side_effect=interrupt):
                with self.assertRaises(InterruptedError):
                    self.fit('bad_replay')
            path = self.root / 'bad_replay/history.json'
            history = read(path)
            history[1]['train_mse'] += .001
            write(path, history)
            with self.assertRaisesRegex(ValueError, 'post-best epoch replay differs'):
                self.fit('bad_replay', resume=True)

    def test_resume_recovers_checkpoint_before_history_write(self):
        original = fitting.write

        def interrupt(path, value):
            if Path(path).name == 'history.json':
                raise InterruptedError('checkpoint already published')
            return original(path, value)

        with patch.object(fitting, 'write', side_effect=interrupt):
            with self.assertRaises(InterruptedError):
                self.fit('cp_first', epochs_override=1)
        self.assertTrue((self.root / 'cp_first/best_checkpoint.pt').exists())
        self.assertFalse((self.root / 'cp_first/history.json').exists())
        expected = self.fit('cp_reference', epochs_override=1)
        actual = self.fit('cp_first', epochs_override=1, resume=True)
        self.assertEqual(actual, expected)

    def test_completed_resume_no_parameter_updates_and_changed_inputs_rejected(self):
        expected = self.fit('done', epochs_override=1)
        with patch.object(fitting, 'train_epoch', side_effect=AssertionError('No retraining')):
            self.assertEqual(self.fit('done', epochs_override=1, resume=True, budget_gib=64), expected)
        self.ctx['labels']['e']['value'] += .01
        with self.assertRaisesRegex(ValueError, 'assessment inputs changed'):
            self.fit('done', epochs_override=1, resume=True)
        self.ctx['vectors']['a'][0] += .01
        with self.assertRaisesRegex(ValueError, 'inputs/settings/runtime differ'):
            self.fit('done', epochs_override=1, verify=True)

    def test_completed_artifact_tampering_is_rejected(self):
        self.fit('tamper', epochs_override=1)
        write(self.root / 'tamper/evaluation.json', {})
        with self.assertRaisesRegex(ValueError, 'artifact'):
            self.fit('tamper', epochs_override=1, verify=True)

    def test_frozen_predictor_never_reads_labels_or_constructs_optimizer(self):
        result = self.fit('predict', 'tpg_DM', epochs_override=1)
        self.ctx['labels'] = LabelBomb()
        with patch.object(torch.optim, 'AdamW', side_effect=AssertionError('No optimizer')):
            rows = fitting.predict_checkpoint(self.root / 'predict', self.ctx,
                self.identity('tpg_DM'), 'tpg_DM', 17, ['e'], 32)
        self.assertEqual(rows, [{k: row[k] for k in ('cve', 'prediction')}
                                for row in result['predictions']['assessment']])
        self.ctx['contract'] = deepcopy(self.contract)
        self.ctx['contract']['edge_type_vocab'] = {'DEP': 0, 'SEC_HAS_VERSION': 1}
        with self.assertRaisesRegex(ValueError, 'contract mismatch'):
            fitting.predict_checkpoint(self.root / 'predict', self.ctx,
                self.identity('tpg_DM'), 'tpg_DM', 17, ['e'], 32)

    def test_test_flag_routes_only_to_test_cache_loader_and_fit_forbids_it(self):
        self.fit('test_helper', 'tpg_D', epochs_override=1)
        self.ctx['is_test'] = True
        self.ctx['labels'] = LabelBomb()
        from epss.final_comparison_v1 import test_cache
        with patch.object(test_cache, 'load_graph', side_effect=lambda ctx, key: deepcopy(self.graphs[key])) as test_load, \
             patch.object(old_data.GraphStore, 'load', side_effect=AssertionError('Wrong cache loader')):
            rows = fitting.predict_checkpoint(self.root / 'test_helper', self.ctx,
                self.identity('tpg_D'), 'tpg_D', 17, ['e'], 32)
            self.assertEqual(len(rows), 1)
            self.assertEqual(test_load.call_count, 1)
        with self.assertRaisesRegex(ValueError, 'official test'):
            self.fit('forbidden', epochs_override=1)

    def test_overlapping_roles_missing_targets_and_same_fit_lock_rejected(self):
        self.ctx['ids']['assessment'] = ['a']
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            self.fit('overlap', epochs_override=1)
        self.ctx['ids']['assessment'] = []
        self.ctx['labels']['a']['value'] = None
        with self.assertRaisesRegex(ValueError, 'target'):
            self.fit('missing', epochs_override=1)
        with exclusive(self.root / '.locked.writer.lock'):
            with self.assertRaisesRegex(ValueError, 'Another process'):
                self.fit('locked', epochs_override=1)

    def test_runtime_pins_native_threads_without_shell_exports(self):
        with patch.dict('os.environ', {'OPENBLAS_NUM_THREADS': '88'}):
            with fitting.cpu_runtime():
                runtime = fitting._runtime()
                self.assertEqual(runtime['threads'], 4)
                self.assertTrue(all(p['num_threads'] == 4 for p in runtime['native_pools']))


if __name__ == '__main__':
    unittest.main()
