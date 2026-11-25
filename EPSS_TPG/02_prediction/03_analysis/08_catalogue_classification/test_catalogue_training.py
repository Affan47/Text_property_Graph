"""Synthetic classification and provenance regressions; no scientific fits."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('cc03_cli', Path(__file__).with_name('05_run_catalogue_training.py'))
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
sys.path.insert(0, str(cli.ROOT/'02_prediction/03_analysis/07_fresh_studies'))

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from test_combined_training import fixture
from epss.catalogue_training_v1 import common as c, cache, data, fitting, metrics, runner
from epss.combined_training_v1.data import GraphStore
from epss.combined_training_v2.pooled import Regressor, reduce_graph


class MetricTests(unittest.TestCase):
    def test_cutoff_ties_and_selection_not_assessment(self):
        self.assertEqual(metrics.cutoff([0, 1, 0, 1], [.2, .8, .2, .8]), .8)
        self.assertEqual(metrics.cutoff([0, 1], [.3, .3]), .3)
        with self.assertRaisesRegex(ValueError, 'both classes'):
            metrics.cutoff([0, 0], [.1, .2])

    def test_weighted_bootstrap_ap_matches_sklearn_with_ties(self):
        y, p = [0, 1, 1, 0, 1], [.1, .5, .5, .9, .5]
        weights = np.asarray([[1, 1, 1, 1, 1], [4, 2, 0, 3, 1], [0, 1, 1, 0, 2]])
        actual = metrics.weighted_ap(y, p, weights)
        expected = [average_precision_score(y, p, sample_weight=w) for w in weights]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)

    def test_group_bootstrap_reproduces_and_identical_models_zero(self):
        y, p = [0, 1, 1, 0], [.1, .6, .4, .7]
        a = metrics.paired_ap(y, [p]*3, [p]*3, ['a', 'a', 'b', 'b'], 200, 6029)
        b = metrics.paired_ap(y, [p]*3, [p]*3, ['a', 'a', 'b', 'b'], 200, 6029)
        self.assertEqual(a, b)
        self.assertEqual(a['interval95'], [0., 0.])
        self.assertEqual(a['groups'], 2)

    def test_budgets_ties_are_id_stable_and_calibration_complete(self):
        r = metrics.score([1, 0], [.4, .4], .5, ['b', 'a'], [.5])
        self.assertEqual(r['review_budgets'][0]['hits'], 0)
        self.assertEqual(sum(x['n'] for x in r['calibration']), 2)
        self.assertEqual(r['fn'], 1)
        self.assertIsNone(metrics.score([0, 0], [.1, .9], .5, ['a', 'b'])['average_precision'])

    def test_canonical_csv_sorting(self):
        a = [{'cve': 'z', 'p': .2}, {'cve': 'a', 'p': .1}]
        text = c.csv_text(a, ['cve', 'p'], ['cve'])
        self.assertEqual(text, c.csv_text(a[::-1], ['cve', 'p'], ['cve']))
        self.assertNotIn('\r', text)

    def test_reuse_requires_exact_text_not_similar_words(self):
        entry = {'text': 'A patch.', 'spans': [{'start': 0, 'end': 8, 'source': 'D',
                 'document_id': 'doc', 'text_sha256': 'sha'}]}
        self.assertTrue(cache.matching_documents([{'source': 'D', 'text': 'A patch.'}], entry))
        self.assertFalse(cache.matching_documents([{'source': 'D', 'text': 'A  patch.'}], entry))
        self.assertFalse(cache.matching_documents([{'source': 'M', 'text': 'A patch.'}], entry))

    def test_keyed_CC02_labels_and_unknown_rejection(self):
        row = {'cve': 'CVE-2022-12345', 'label': 1, 'status': 'LISTED'}
        with patch.object(c, 'read', return_value={row['cve']: row}):
            self.assertEqual(c.labels_at(Path('/unused'), 'reddit'), {row['cve']: 1})
        with patch.object(c, 'read', return_value={row['cve']: {**row, 'status': 'UNKNOWN', 'label': None}}):
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                c.labels_at(Path('/unused'), 'reddit')

    def test_cross_filesystem_reuse_preserves_bytes(self):
        import errno
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp)/'source', Path(temp)/'target'
            source.write_bytes(b'fixture bytes')
            with patch.object(cache.os, 'link', side_effect=OSError(errno.EXDEV, 'different filesystem')):
                cache.link_verified(source, target)
            self.assertEqual(source.read_bytes(), target.read_bytes())


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        torch.manual_seed(222)
        contract, _, graph = fixture()
        contract['structured_names'] += data.EPSS_NAMES
        self.cfg = c.configuration()
        self.cfg['settings']['epochs'] = 3
        self.cfg['settings']['patience'] = 2
        self.cfg['smoke']['epochs'] = 3
        self.cfg['settings']['tree']['max_iter'] = 2
        self.graphs = {}
        self.ctx = {'directory': self.root/'cache', 'contract': contract, 'index': {}, 'receipts': {},
                    'vectors': {}, 'pooled': {'D': {}, 'DM': {}}, 'observed': {}, 'epss': {}, 'groups': {}, 'statistics': {},
                    'roles': {'fitting': ['a', 'b'], 'selection': ['c', 'd'], 'assessment': ['e', 'f']},
                    'labels': {'a': 0, 'b': 1, 'c': 0, 'd': 1}, 'identity': {'source_pins': {}}}
        for i, cid in enumerate('abcdef'):
            self.ctx['vectors'][cid] = [i*.1, .01+i*.1, 1.]
            self.ctx['groups'][cid] = cid
            self.ctx['index'][cid] = {'views': {}}
            for view in ('D', 'DM'):
                posts = cid in 'ace' and view == 'DM'
                key = cid+('_DM' if posts else '_D')
                self.ctx['index'][cid]['views'][view] = {'object_key': key, 'source_mask': {
                    'D': True, 'M': posts, 'G': False, 'N': False}}
                obj = self.ctx['directory']/'objects'/key
                if key not in self.graphs:
                    self.graphs[key] = deepcopy(graph)
                    self.graphs[key]['x'][:, len(contract['node_type_vocab']):][:, :768] += i*.01
                    c.atomic_json(obj/'graph_receipt.json', {'statistics': {'nodes': 3, 'edges': 3}})
                    c.atomic_json(obj/'embedding_receipt.json', {'synthetic': key})
                    self.ctx['receipts'][key] = [c.sha(obj/name) for name in ('graph_receipt.json', 'embedding_receipt.json')]
                    self.ctx['observed'][key] = True
                    self.ctx['statistics'][key] = {'nodes': 3, 'edges': 3}
                self.ctx['pooled'][view][cid] = reduce_graph(self.graphs[key], contract)[0].tolist()
        self.loader = patch.object(GraphStore, 'load', side_effect=lambda key: deepcopy(self.graphs[key]))
        self.disk = self.loader.start()

    def tearDown(self):
        self.loader.stop()
        self.tmp.cleanup()

    def fit(self, arm='tpg_D', name=None, **kwargs):
        return fitting.fit(self.root/(name or arm), self.ctx, self.cfg, arm, 17, 'smoke', .1, **kwargs)

    def test_same_weight_original_graph_and_pooled_prediction_parity(self):
        for arm, representation in fitting.NEURAL.items():
            net = fitting.initialize(self.ctx['contract'], arm, 17, self.cfg['settings'])
            reference = Regressor(self.ctx['contract'], self.cfg['settings'], representation)
            reference.load_state_dict(net.state_dict())
            store = data.Store(self.ctx, arm, .1).preload(['a'], 'parity')
            self.assertTrue(torch.equal(net(store.inputs('a')), reference(store.inputs('a'))))
            self.assertEqual(list(net.state_dict()), list(reference.state_dict()))

    def test_node_type_subtype_edge_type_property_gradients(self):
        net = fitting.initialize(self.ctx['contract'], 'tpg_D', 17, self.cfg['settings'])
        row = data.Store(self.ctx, 'tpg_D', .1).preload(['a'], 'gradient').inputs('a')
        torch.nn.functional.binary_cross_entropy_with_logits(net.logits(row), torch.tensor(1.)).backward()
        for value in (net.node_project[0].weight.grad[:, :3], net.subtype_project.weight.grad,
                      net.layers[0].edge_embed.weight.grad, net.layers[0].edge_semantics.weight.grad):
            self.assertGreater(float(value.abs().sum()), 0)
        with self.assertRaisesRegex(ValueError, 'Targets'):
            net({**row, 'label': 1})

    def test_masked_DM_equals_D_and_no_pooled_graph_io(self):
        self.disk.reset_mock()
        outputs = []
        for arm in ('pooled_D', 'pooled_DM'):
            store = data.Store(self.ctx, arm, .1).preload(['b'], 'masked')
            net = fitting.initialize(self.ctx['contract'], arm, 17, self.cfg['settings'])
            outputs.append(net(store.inputs('b')).detach())
        self.assertTrue(torch.equal(*outputs))
        self.disk.assert_not_called()

    def test_all_six_arms_checkpoint_prediction_replay(self):
        for arm in self.cfg['arms']:
            with self.subTest(arm=arm):
                result = self.fit(arm)
                self.assertEqual(result, self.fit(arm, verify=True))
                if arm in fitting.NEURAL:
                    self.assertEqual([x.name for x in (self.root/arm).glob('*.pt')], ['best_checkpoint.pt'])
                self.assertEqual(set(result['probabilities']), {'c', 'd'})
                self.assertNotIn('\r', (self.root/arm/'completed/selection_predictions.csv').read_text())

    def test_assessment_labels_refused_by_training(self):
        self.ctx['labels']['e'] = 1
        with self.assertRaisesRegex(ValueError, 'assessment labels'):
            self.fit()

    def test_changed_checkpoint_rejected_before_deserialization(self):
        self.fit('pooled_D')
        path = self.root/'pooled_D/best_checkpoint.pt'
        with path.open('ab') as stream:
            stream.write(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            self.fit('pooled_D', verify=True)

    def test_interrupted_best_only_resume_exact(self):
        expected = self.fit('pooled_D', 'clean')
        original = fitting.train_epoch
        counter = []
        def interrupted(*args, **kwargs):
            counter.append(1)
            if len(counter) == 2:
                raise RuntimeError('synthetic interruption')
            return original(*args, **kwargs)
        with patch.object(fitting, 'train_epoch', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'interruption'):
                self.fit('pooled_D', 'partial')
        resumed = self.fit('pooled_D', 'partial', resume=True)
        self.assertEqual(resumed['probabilities'], expected['probabilities'])
        self.assertEqual(c.read(self.root/'partial/history.json'), c.read(self.root/'clean/history.json'))
        self.fit('pooled_D', 'partial', verify=True)

    def test_completed_assessment_replay_keeps_frozen_threshold(self):
        self.fit('pooled_D')
        expected = c.read(self.root/'pooled_D/completed/result.json')['threshold']
        ctx = deepcopy(self.ctx)
        ctx['labels'] = {'e': 1, 'f': 0}
        a = fitting.assessment_predictions(self.root/'pooled_D', ctx, self.cfg, 'pooled_D', 17, 'smoke', .1)
        b = fitting.assessment_predictions(self.root/'pooled_D', ctx, self.cfg, 'pooled_D', 17, 'smoke', .1)
        self.assertEqual(a, b)
        self.assertEqual(a[1], expected)

    def test_missing_EPSS_blocks_real_adapter_without_fallback(self):
        with patch.object(c, 'input_scope', return_value=(self.root, {})):
            with self.assertRaisesRegex(ValueError, 'No stale EPSS substitution'):
                data.prepare(self.cfg, 'reddit', 'full')

    def test_end_to_end_release_assessment_and_reproduction(self):
        cfg = deepcopy(self.cfg)
        cfg['output_directory'] = str(self.root/'pipeline')
        cfg['seeds'] = cfg['smoke']['seeds'] = [17]
        cfg['bootstrap']['repetitions'] = 50
        def context(*args, **kwargs):
            row = deepcopy(self.ctx)
            if kwargs.get('assessment'):
                row['labels'] = {'e': 1, 'f': 0}
            return row
        with patch.object(data, 'context', side_effect=context):
            with self.assertRaisesRegex(ValueError, 'smoke'):
                runner.train(cfg, 'reddit', 'full', 1, .1, False, False)
            runner.train(cfg, 'reddit', 'smoke', 1, .1, False, False)
            runner.train(cfg, 'reddit', 'smoke', 1, .1, False, True)
            runner.train(cfg, 'reddit', 'full', 1, .1, False, False)
            with self.assertRaisesRegex(ValueError, 'Fresh-process'):
                runner.assess(cfg, 'reddit', 'full', .1, False)
            runner.train(cfg, 'reddit', 'full', 1, .1, False, True)
            runner.assess(cfg, 'reddit', 'full', .1, False)
            runner.assess(cfg, 'reddit', 'full', .1, True)
        output = c.location(cfg, 'reddit', 'full')/'assessment'
        self.assertFalse(c.read(output/'identity.json')['official_test_scored'])
        self.assertEqual(len(c.read(output/'summary.json')), 8)


if __name__ == '__main__':
    unittest.main()
