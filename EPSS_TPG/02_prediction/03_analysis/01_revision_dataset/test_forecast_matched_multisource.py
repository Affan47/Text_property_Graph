"""Synthetic MS03 contracts; no real dataset training or test-set access."""
from copy import deepcopy
from contextlib import redirect_stdout, redirect_stderr
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss import forecast_matched_multisource as run

m = run.model
VOCAB = {'num_node_types': 2, 'num_edge_types': 2, 'edge_types': {'DEP': 0, 'SEC_HAS_VERSION': 1}}
CFG = dict(run.configuration('smoke')[1], hidden_size=4, temporal_hidden_size=3,
           graph_layers=1, dropout=0., graph_batch_nodes=4, graph_batch_edges=4,
           epochs=2, patience=2, batch_size=2)


def graph(value=1.):
    return {'x': torch.tensor([[1., 0., value, .2, 1.], [0., 1., .1, value, 1.]]),
            'edge_index': torch.tensor([[0, 1], [1, 0]]), 'edge_type': torch.tensor([0, 1]),
            'edge_features': torch.zeros(2, m.EDGE_FEATURE_DIM)}


class MemoryStore:
    def __init__(self):
        self.calls = []
        self.items = {'a': graph(), 'b': graph(.4), 'c': graph(.9)}

    def documents(self, cve, enabled):
        self.calls.append((cve, enabled))
        if cve == 'missing':
            return []
        return [(i, ['a'] if i == 0 else ['b', 'c'])
                for i, source in enumerate(m.SOURCES) if source in enabled]

    def load(self, name):
        return self.items[name]


def record(cve='a'):
    pub = date(2024, 1, 1)
    values = [None, .1] + [.2] * 28 + [None]
    last, ages, known = None, [], []
    for i, v in enumerate(values):
        if v is not None:
            last = i
        ages.append(None if last is None else i-last)
        known.append(int(last is not None))
    return {'cve': cve, 'publication_date': pub.isoformat(), 'input_end_date': (pub+timedelta(days=60)).isoformat(),
            'anchor_day': 60, 'time_policy': 'synthetic', 'histories': {'recent_30_60': {
                'dates': [(pub+timedelta(days=i)).isoformat() for i in range(30, 61)],
                'day_offsets': list(range(30, 61)), 'values': values,
                'observed_mask': [int(v is not None) for v in values],
                'age_since_observation_days': ages, 'age_known_mask': known}}}


def identity():
    return {'schema': run.SCHEMA, 'config': deepcopy(CFG), 'gain': .01,
            'ids': {'train': ['a', 'b'], 'validation': ['c', 'd']},
            'vocab': VOCAB, 'embedding_dim': 2, 'graph_cache_lock_sha256': 'synthetic'}


class MatchedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        self.store = MemoryStore()
        self.x = run.tensor_inputs([record(), record('b')])

    def net(self, variant='all_sources_epss', config=None):
        i = identity()
        if config:
            i['config'] = config
        return run.initialize(i, variant, 17)

    def test_persistence_initialization_and_shared_state(self):
        hashes = []
        for variant in m.VARIANTS:
            net = self.net(variant).eval()
            hashes.append(run.controls.state_digest(net.state_dict()))
            base, correction, _, pred, _, _ = net.details(self.x, self.store, ['a', 'b'])
            self.assertTrue(torch.equal(base, pred))
            self.assertTrue(torch.equal(base, torch.full((2,), .2)))
            self.assertEqual(float(correction.abs().sum()), 0.)
        self.assertEqual(len(set(hashes)), 1)

    def test_epss_only_never_loads_graphs(self):
        net = self.net('epss_only')
        net(self.x, None, ['a', 'b']).sum().backward()
        self.assertTrue(all(p.grad is None for p in net.layers.parameters()))
        self.assertTrue(all(p.grad is None for p in net.node_project.parameters()))

    def test_description_only_does_not_visit_other_sources(self):
        parts = self.net('description_epss').details(self.x, self.store, ['a', 'b'])
        self.assertTrue(all(enabled == ('description',) for _, enabled in self.store.calls))
        self.assertEqual(parts[5].tolist(), [[True, False, False, False]] * 2)
        self.assertEqual(parts[4].tolist(), [[1., 0., 0., 0.]] * 2)

    def test_missing_source_attention_is_finite_and_zero(self):
        parts = self.net().details(self.x, self.store, ['missing', 'a'])
        self.assertTrue(torch.isfinite(parts[3]).all())
        self.assertEqual(parts[4][0].tolist(), [0.] * 4)
        self.assertFalse(parts[5][0].any())
        self.assertAlmostEqual(float(parts[4][1].sum()), 1., places=6)

    def test_hierarchical_pool_equal_documents_not_chunk_counts(self):
        vectors = [torch.tensor([2.]), torch.tensor([2.]), torch.tensor([8.])]
        pooled, mask = m.hierarchical_pool(vectors, [(0, 0, 0), (0, 0, 0), (0, 0, 1)], 1, 1, vectors[0])
        self.assertEqual(float(pooled[0, 0]), 5.)
        self.assertEqual(mask.tolist(), [[True, False, False, False]])

    def test_microbatches_keep_every_chunk_and_owner(self):
        batches = list(m.graph_batches(self.store, ['a', 'b'], m.SOURCES, 4, 4))
        owners = [o for _, group in batches for o in group]
        self.assertEqual(len(owners), 14)
        self.assertTrue(all(len(b.x) <= 4 and b.num_edges <= 4 for b, _ in batches))
        with self.assertRaisesRegex(ValueError, 'no truncation'):
            list(m.graph_batches(self.store, ['a'], m.SOURCES, 1, 4))

    def test_checkpointed_gradients_match_ordinary_gradients(self):
        first = self.net(config=dict(CFG, dropout=.2))
        second = self.net(config=dict(CFG, dropout=.2, checkpoint_graph_activations=False))
        with torch.no_grad():
            first.head[-1].weight.fill_(.2)
        second.load_state_dict(first.state_dict())
        torch.manual_seed(123)
        a = first(self.x, self.store, ['a', 'b'])
        torch.manual_seed(123)
        b = second(self.x, self.store, ['a', 'b'])
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.sum().backward()
        b.sum().backward()
        for (name, p), (_, q) in zip(first.named_parameters(), second.named_parameters()):
            self.assertIsNotNone(p.grad, name)
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        self.assertGreater(float(first.layers[0].edge_embed.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(first.temporal.weight_ih_l0.grad.abs().sum()), 0.)

    def test_security_relation_type_affects_encoded_graph(self):
        net = self.net().eval()
        g = graph()
        args = (g['x'], g['edge_index'], g['edge_type'], g['edge_features'], torch.zeros(2, dtype=torch.long))
        a = net.encode_graphs(*args)
        b = net.encode_graphs(args[0], args[1], 1-args[2], args[3], args[4])
        self.assertFalse(torch.equal(a, b))

    def test_missing_history_masks_and_target_rejection(self):
        self.assertEqual(self.x.shape, (2, 31, 5))
        self.assertEqual(self.x[0, -1, :2].tolist(), [0., 0.])
        self.assertEqual(float(self.x[0, -1, 3]), 1.)
        bad = record()
        bad['y'] = {'endpoint': .9}
        with self.assertRaisesRegex(ValueError, 'target field'):
            run.tensor_inputs([bad])
        bad = record()
        bad['histories']['recent_30_60']['day_offsets'][-1] = 90
        with self.assertRaises(ValueError):
            run.tensor_inputs([bad])

    def test_targets_do_not_enter_forward_or_gain(self):
        records = {c: record(c) for c in ['a', 'b']}
        gain = run.controls.training_gain(['a'], records)
        records['b']['histories']['recent_30_60']['values'][29] = .9
        self.assertEqual(run.controls.training_gain(['a'], records), gain)
        net = self.net().eval()
        rows = run.infer(net, ['a', 'b'], self.x, self.store, 2, False)
        original = [r['prediction'] for r in rows]
        run.scored(rows, {c: {'y': {'endpoint': .9}} for c in ['a', 'b']})
        self.assertEqual([r['prediction'] for r in rows], original)

    def test_graph_store_allowlist_masks_hashes_and_lru(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            g = graph()
            payload = {'shared': {'x': g['x'], 'forbidden_label': .9},
                       'edges': {k: v for k, v in g.items() if k != 'x'},
                       'edge_masks': {'no_coref': torch.tensor([True, False])}}
            path = root / 'a.pt'
            torch.save(payload, path)
            index = {'a': {'split': 'train', 'documents': [{'source_text_type': 'description',
                      'chunks': [{'has_graph': True, 'tensor': 'a.pt'}]}]}}
            store = m.GraphStore(root, index, VOCAB, 2, ['a'], {'a.pt': run.sha256(path)}, cache_gib=0)
            item = store.load('a.pt')
            self.assertEqual(set(item), {'x', 'edge_index', 'edge_type', 'edge_features'})
            self.assertEqual(item['edge_type'].tolist(), [0])
            self.assertFalse(store.items)
            with self.assertRaises(ValueError):
                store.documents('held_out', m.SOURCES)
            index['a']['split'] = 'test'
            with self.assertRaises(ValueError):
                store.documents('a', m.SOURCES)
            with self.assertRaises(ValueError):
                store.load('../outside.pt')
            store.verified.clear()
            torch.save({'changed': True}, path)
            with self.assertRaisesRegex(ValueError, 'Changed graph tensor'):
                store.load('a.pt')

    def test_synthetic_fit_reload_and_fixed_seed_reproduction(self):
        i = identity()
        tensors = {'train': self.x, 'validation': self.x.clone()}
        labels = {c: {'y': {'endpoint': .25 + .01*j}} for j, c in enumerate(['a', 'b', 'c', 'd'])}
        for variant in m.VARIANTS:
            saved, history = run.fit(variant, 17, i, tensors, labels, self.store, False)
            repeated, again = run.fit(variant, 17, i, tensors, labels, self.store, False)
            self.assertEqual(history, again)
            self.assertEqual(run.controls.state_digest(saved['state_dict']), run.controls.state_digest(repeated['state_dict']))
            rows, metrics = run.evaluate(saved, i, tensors, labels, self.store, False)
            self.assertEqual(metrics['metrics']['mae'], history[saved['selected_epoch']-1]['validation_mae'])
            with tempfile.TemporaryDirectory() as d:
                directory = Path(d)
                run.controls.save_checkpoint(directory/'checkpoint.pt', saved)
                for name, value in [('history.json', history), ('predictions.json', rows), ('metrics.json', metrics)]:
                    run.write(directory/name, value)
                run.write(directory/'fit_lock.json', {'schema': run.SCHEMA,
                          'files': {n: run.sha256(directory/n) for n in run.FIT_FILES}})
                replay = run.check_fit(directory, variant, 17, i, tensors, labels, self.store)
                self.assertEqual(replay['metrics'], metrics['metrics'])

    def test_result_writer_excludes_concurrent_processes(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / '.writer.lock'
            with run.writer_lock(path):
                with self.assertRaisesRegex(ValueError, 'holds the result lock'):
                    with run.writer_lock(path):
                        self.fail('Concurrent writer entered')
            with run.writer_lock(path):
                pass

    def test_complete_synthetic_smoke_full_verify_and_resume(self):
        base = identity()
        inputs = {c: record(c) for c in ['a', 'b', 'c', 'd']}
        base.update(gain=run.controls.training_gain(['a', 'b'], inputs), gain_fit_ids=['a', 'b'], pinned_files={},
                    source_coverage={s: {src: 2 for src in m.SOURCES} for s in ['train', 'validation']},
                    protocol={'limitations': ['Synthetic integration fixture only.']})
        self.store.index = {c: {'source_mask': [True]*4} for c in inputs}
        labels = {c: {'y': {'endpoint': .25 + j*.01}} for j, c in enumerate(inputs)}
        tensors = {'train': self.x, 'validation': self.x.clone()}

        def context(scale, threads, cache_gib):
            return dict(deepcopy(base), scale=scale), tensors, labels, self.store

        with tempfile.TemporaryDirectory() as d, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            root = Path(d)
            prepared = root / 'prepared'
            (prepared / 'smoke').mkdir(parents=True)
            (prepared / 'full').mkdir()
            run.write(prepared/'smoke/run_identity.json', {'ids': base['ids']})
            run.write(prepared/'full/inputs.json', inputs)
            with patch.object(run, 'OUTPUT', root/'results'), patch.object(run, 'context', context), \
                    patch.object(run.cache.base.preparation, 'OUTPUT', prepared), \
                    patch.object(run, 'configuration', return_value=(base['protocol'], base['config'])):
                run.execute('smoke', 1, 0, check_only=True)
                self.assertFalse((root/'results').exists())
                run.execute('smoke', 1, 0)
                run.execute('smoke', 1, 0, verify_only=True)
                with self.assertRaisesRegex(ValueError, 'Run exists'):
                    run.execute('smoke', 1, 0)
                # Simulate interruption after completed fits but before the final lock.
                (root/'results/smoke/evaluation_lock.json').unlink()
                with patch.object(run, 'fit', side_effect=AssertionError('Completed fits must not retrain')):
                    run.execute('smoke', 1, 0, resume=True)
                run.execute('full', 1, 0)
                run.execute('full', 1, 0, verify_only=True)
                summary = run.read(root/'results/full/summary.json')
                self.assertEqual(set(summary['aggregates']), set(m.VARIANTS))
                self.assertEqual(len(summary['fits']), 3)
                self.assertFalse(summary['test_scoring'])


if __name__ == '__main__':
    unittest.main()
