import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('ct03_cli', Path(__file__).with_name('05_run_source_readout.py'))
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
import numpy as np
import torch
from epss.combined_readout_v1 import data, model, runner


def fixture():
    torch.manual_seed(17)
    ids = {'train': ['a', 'b', 'c'], 'validation': ['d', 'e']}
    inputs = {cid: {'merged': torch.randn(768), 'source': torch.randn(2, 768), 'observed': torch.ones(2),
                    'mask': torch.tensor([1., 1., 0., 0.]), 'structured': torch.randn(53)} for cid in 'abcde'}
    ctx = {'ids': ids, 'labels': {cid: {'value': value, 'date': '2025-01-01'} for cid, value in zip('abcde', [.01, .3, .7, .02, .4])},
           'index': {cid: {'views': {'DM': {'object_key': cid, 'source_mask': dict(zip('DMGN', [True, True, False, False]))}}} for cid in 'abcde'},
           'audit': {}}
    ident = {'settings': data.settings(), 'scale': 'smoke', 'source_pins': {}, 'contract': {'structured_names': [str(i) for i in range(53)]},
             'arm': 'source', 'seed': 17, 'objective': 'mse'}
    return ctx, inputs, ident


class SourceReadoutTests(unittest.TestCase):
    def test_parameter_count_matches(self):
        self.assertEqual(model.parameter_count(model.Regressor(53, 'merged')), 28705)
        self.assertEqual(model.parameter_count(model.Regressor(53, 'source')), 28705)

    def test_no_label_inputs(self):
        _, inputs, _ = fixture()
        with self.assertRaises(ValueError): model.Regressor(53, 'source')({**inputs['a'], 'target': .3})

    def test_missing_source_is_inactive_after_projection(self):
        _, inputs, _ = fixture(); x = inputs['a']; x['observed'][1] = 0; x['mask'][1] = 0
        net = model.Regressor(53, 'source'); before = net(x).detach()
        x['source'][1].fill_(1000.)
        self.assertTrue(torch.equal(before, net(x)))

    def test_source_extraction_excludes_global_and_unobserved(self):
        rows = np.stack([np.ones(768)*i for i in range(4)]).astype(np.float32)
        result = data.reduce_tokens(rows, np.array([True, True, False, True]), np.array([3, 0, 1, 2]),
            np.array([False, True, True, True]), np.array([[1,1,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,0]], dtype=bool),
            np.array([-1, 0, 1, 1]), ['D', 'M'])
        self.assertEqual(result['counts'], [1, 1]); self.assertEqual(result['merged'][0], .5)

    def test_ambiguous_token_fails(self):
        with self.assertRaises(ValueError):
            data.reduce_tokens(np.zeros((1, 768), np.float32), np.array([True]), np.array([0]), np.array([True]),
                               np.array([[1,1,0,0]], bool), np.array([0]), ['D'])

    def test_source_document_disagreement_fails(self):
        with self.assertRaises(ValueError):
            data.reduce_tokens(np.zeros((1, 768), np.float32), np.array([True]), np.array([0]), np.array([True]),
                               np.array([[0,1,0,0]], bool), np.array([0]), ['D'])

    def test_checkpoint_replay_and_best_only(self):
        ctx, inputs, ident = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'fit'
            result = runner.fit(folder, ctx, inputs, ident)
            self.assertEqual(result, runner.fit(folder, ctx, inputs, ident, verify=True))
            self.assertEqual(len(list(folder.glob('*.pt'))), 1)

    def test_interrupted_fit_replays(self):
        ctx, inputs, ident = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp)/'one', Path(tmp)/'two'
            expected = runner.fit(first, ctx, inputs, ident)
            original = runner.evaluate; calls = 0
            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2: raise RuntimeError('test interruption')
                return original(*args, **kwargs)
            with patch.object(runner, 'evaluate', side_effect=interrupted):
                with self.assertRaises(RuntimeError): runner.fit(second, ctx, inputs, ident)
            self.assertEqual(expected, runner.fit(second, ctx, inputs, ident, resume=True))
            self.assertEqual(data.c.read(first/'history.json'), data.c.read(second/'history.json'))

    def test_logistic_is_not_epss_output(self):
        ctx, _, _ = fixture()
        result = runner.linear_evaluation(ctx, np.array([.2, .8]), 'logistic')
        self.assertFalse(result['output_is_epss_score'])
        self.assertNotIn('prediction', result['predictions'][0])

    def test_training_only_scaler(self):
        ctx, inputs, ident = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            runner.controls(out, ctx, inputs, ident)
            states = data.c.read(out/'controls/states.json')
            x = runner.linear_matrix(inputs, ctx['ids']['train'], 'DM')
            self.assertTrue(np.allclose(states['ridge/DM']['mean'], x.mean(0)))
            runner.controls(out, ctx, inputs, ident, verify=True)

    def test_smoke_single_class_is_explicit(self):
        ctx, inputs, ident = fixture()
        for cid in ctx['ids']['train']: ctx['labels'][cid]['value'] = .01
        with tempfile.TemporaryDirectory() as tmp:
            runner.controls(Path(tmp), ctx, inputs, ident)
            states = data.c.read(Path(tmp)/'controls/states.json')
            self.assertEqual(states['logistic/DM']['status'], 'SMOKE_SINGLE_CLASS_NO_CLASSIFIER_FIT')

    def test_full_bootstrap_report_identical_predictions(self):
        ctx, _, ident = fixture()
        p = {**ident['settings'], 'bootstrap_repetitions': 32}
        result = runner.evaluate(ctx, [.1, .2])
        results = {f'{o}/{a}/{s}': result for o in p['objectives'] for a in p['arms'] for s in p['seeds']}
        summary = runner.summarize(ctx, results, p, 'full')
        self.assertTrue(summary['paired_intervals'])
        for row in summary['paired_intervals']:
            self.assertEqual(row['difference'], 0.)
            self.assertEqual(row['lower_2_5'], 0.)
            self.assertEqual(row['upper_97_5'], 0.)

    def test_validation_labels_do_not_change_linear_fit(self):
        ctx, inputs, ident = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            one, two = Path(tmp)/'one', Path(tmp)/'two'
            runner.controls(one, ctx, inputs, ident)
            for cid in ctx['ids']['validation']: ctx['labels'][cid]['value'] = .99
            runner.controls(two, ctx, inputs, ident)
            self.assertEqual(data.c.read(one/'controls/states.json'), data.c.read(two/'controls/states.json'))


if __name__ == '__main__': unittest.main()
