"""MS05/MS06 synthetic regression tests; no real-data training or holdout access."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_day60_training as run
import test_forecast_day60_multisource as data_tests
import test_forecast_matched_multisource as legacy

cache = run.cache


def inputs(cve):
    raw = data_tests.record('CVE-2024-12345', '2024-01-01')
    data_tests.change_score(raw, 0, None)
    data_tests.change_score(raw, 30, None)
    row = run.prep.derive(raw)[0]
    row['cve'] = cve
    return row


def identity(scale='smoke'):
    i = legacy.identity()
    p, _ = run.configuration(scale)
    i.update(schema=run.SCHEMA, scale=scale, protocol=p, pinned_files={}, source_coverage={})
    i['config']['seeds'] = [17]
    return i


def cache_fixture(root):
    c = 'CVE-2024-12345'
    rows = [{'cve': c, 'split': 'train', 'source_mask': [1,0,0,0],
             'documents': [{'document_id': 'description', 'source_text_type': 'description',
                            'text': 'A product before version 2 allows an attack.'}]}]
    ctx = cache.base.plan(rows, {'train': [c], 'validation': []}, 4096)
    key, e = next(iter(ctx['entries'].items()))
    reuse = root/'prior'
    names = cache.base.files(e)
    for n in names:
        cache.write(reuse/n, {'synthetic': n})
    stat = {**{k: e[k] for k in ('key','cve','split','document_id','source_text_type','text_sha256')},
            'nodes': 2, 'embedded_nodes': 2, 'unembedded_nodes': 0, 'edges_no_coref': 1, 'security_edges': 1}
    receipt_name = 'receipts/'+key+'.json'
    receipt = {'entry': {k: v for k,v in e.items() if k != 'text'}, 'statistics': stat,
               'files': {n: cache.sha256(reuse/n) for n in names}}
    cache.write(reuse/receipt_name, receipt)
    loc = {'root': str(reuse), 'statistics': stat, 'files': {receipt_name: cache.sha256(reuse/receipt_name), **receipt['files']}}
    ctx['reuse'] = {key: loc}
    ctx['identity'] = {'schema': cache.SCHEMA, 'scale': 'smoke', 'ids': {'train': [c], 'validation': []},
                       'protocol': {'limitations': []}, 'source_pins': {str(reuse/n): h for n,h in loc['files'].items()},
                       'type_vocabulary': legacy.VOCAB, 'embedding_dim': 2}
    return ctx, reuse


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        self.stdout = redirect_stdout(StringIO())
        self.stderr = redirect_stderr(StringIO())
        self.stdout.__enter__()
        self.stderr.__enter__()
        self.addCleanup(self.stdout.__exit__, None,None,None)
        self.addCleanup(self.stderr.__exit__, None,None,None)
        self.i = identity()
        self.x = {s: run.tensor_inputs([inputs(c) for c in ids]) for s,ids in self.i['ids'].items()}
        self.labels = {c: {'cve': c, 'target_day': 60, 'horizon_days': 30, 'target_observed': True,
                           'y': {'endpoint': .25 if c in ('a','c') else .08}}
                       for ids in self.i['ids'].values() for c in ids}
        self.store = legacy.MemoryStore()

    def test_day30_channels_and_missing_day30_persistence(self):
        self.assertEqual(self.x['train'].shape, (2,31,5))
        self.assertEqual(float(self.x['train'][0,-1,-1]), .5)
        self.assertEqual(float(self.x['train'][0,-1,0]), 0.)
        r = run.references(self.i, self.x, self.labels)
        self.assertAlmostEqual(r['rows'][0]['prediction'], .1)
        self.assertIn('actual_day60_epss', r['rows'][0])
        self.assertNotIn('actual_day90_epss', r['rows'][0])

    def test_day60_input_cannot_be_used_as_day30(self):
        with self.assertRaises(ValueError):
            run.tensor_inputs([legacy.record()])
        x = inputs('a')
        x['labels'] = {'target': .9}
        with self.assertRaises(ValueError):
            run.tensor_inputs([x])

    def test_gain_uses_training_past_only(self):
        x = {'a': inputs('a'), 'b': inputs('b'), 'validation': {'malformed': 'must not read'}}
        self.assertEqual(run.training_gain(['a','b'], x), .05)
        self.labels['a']['y']['endpoint'] = .99
        self.assertEqual(run.training_gain(['a','b'], x), .05)
        with self.assertRaises(ValueError):
            run.training_gain(['a','a'], x)

    def test_wrong_target_rejected(self):
        self.labels['c']['target_day'] = 90
        with self.assertRaisesRegex(ValueError, 'Day-60 target'):
            run.scored([{'cve':'c', 'prediction': .1}], self.labels)

    def test_hyperparameters_remain_matched(self):
        for scale in ('smoke','full'):
            self.assertEqual(run.configuration(scale)[1], run.prior.configuration(scale)[1])
        self.assertEqual(run.configuration('full')[1]['seeds'], [17,29,43])

    def test_all_three_training_loops_match_frozen_numerics(self):
        for variant in run.model.VARIANTS:
            new, history = run.fit(variant,17,self.i,self.x,self.labels,self.store,False)
            old, old_history = run.prior.fit(variant,17,self.i,self.x,self.labels,self.store,False)
            self.assertEqual(history, old_history)
            self.assertEqual(new['selected_epoch'], old['selected_epoch'])
            self.assertEqual(new['schema'], run.SCHEMA)
            for k, value in new['state_dict'].items():
                torch.testing.assert_close(value, old['state_dict'][k], rtol=0,atol=0)
            rows, metrics = run.evaluate(new,self.i,self.x,self.labels,self.store,False)
            self.assertTrue(all('actual_day60_epss' in r and 'actual_day90_epss' not in r for r in rows))
            self.assertGreaterEqual(metrics['metrics']['mae'], 0.)

    def test_smoke_run_checkpoint_reproduction_and_guards(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run, 'context', return_value=(self.i,self.x,self.labels,self.store)), \
                patch.object(run, 'OUTPUT', Path(tmp)/'results'):
            run.execute('smoke',1,0,check_only=True)
            self.assertFalse(run.OUTPUT.exists())
            run.execute('smoke',1,0)
            run.execute('smoke',1,0,verify_only=True)
            with self.assertRaisesRegex(ValueError, 'Run exists'):
                run.execute('smoke',1,0)
            path = run.OUTPUT/'smoke/fits/epss_only__17/predictions.json'
            rows = run.read(path)
            rows[0]['prediction'] = .999
            run.write(path,rows)
            with self.assertRaisesRegex(ValueError, 'Changed/unsafe artifact'):
                run.execute('smoke',1,0,verify_only=True)

    def test_unselected_and_test_graphs_forbidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx, reuse = cache_fixture(Path(tmp))
            with patch.object(cache,'REUSE',reuse):
                store = run.GraphStore(ctx,ctx['reuse'],[],0)
                with self.assertRaisesRegex(ValueError,'Forbidden'):
                    store.documents('CVE-2024-12345',('description',))
                with self.assertRaisesRegex(ValueError,'Unselected'):
                    store.load(next(iter(ctx['reuse'].values()))['root'])
                ctx['index']['CVE-2024-12345']['split'] = 'test'
                store = run.GraphStore(ctx,ctx['reuse'],['CVE-2024-12345'],0)
                with self.assertRaisesRegex(ValueError,'Test graphs'):
                    store.documents('CVE-2024-12345',('description',))

    def test_full_requires_smoke_and_reproduces_all_fixed_fits(self):
        full = deepcopy(self.i)
        full.update(scale='full')
        full['config']['seeds'] = [17,29,43]
        def context(scale, *args):
            return (self.i if scale == 'smoke' else full), self.x, self.labels, self.store
        with tempfile.TemporaryDirectory() as tmp, patch.object(run,'context',side_effect=context), \
                patch.object(run,'OUTPUT',Path(tmp)/'out'):
            with self.assertRaises(FileNotFoundError):
                run.execute('full',1,0)
            run.execute('smoke',1,0)
            run.execute('full',1,0)
            run.execute('full',1,0,verify_only=True)
            summary = run.read(run.OUTPUT/'full/summary.json')
            self.assertEqual(len(summary['fits']),9)
            self.assertEqual(summary['target_day'],60)

    def test_resume_keeps_completed_fit_and_restarts_incomplete_fit(self):
        original = run.fit
        calls = []
        def interrupted(variant, *args, **kwargs):
            calls.append(variant)
            if len(calls) == 2:
                raise RuntimeError('synthetic interruption')
            return original(variant,*args,**kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch.object(run,'context',return_value=(self.i,self.x,self.labels,self.store)), \
                patch.object(run,'OUTPUT',Path(tmp)/'out'):
            with patch.object(run,'fit',side_effect=interrupted), self.assertRaisesRegex(RuntimeError,'synthetic'):
                run.execute('smoke',1,0)
            path = run.OUTPUT/'smoke/fits/epss_only__17/checkpoint.pt'
            digest = run.sha256(path)
            with patch.object(run,'fit',wraps=original) as fitted:
                run.execute('smoke',1,0,resume=True)
                self.assertEqual(fitted.call_count,2)
            self.assertEqual(run.sha256(path),digest)
            run.execute('smoke',1,0,verify_only=True)

    def test_reuse_requires_exact_text_role_source_and_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx,reuse = cache_fixture(Path(tmp))
            key,e = next(iter(ctx['entries'].items()))
            receipt = cache.read(reuse/'receipts'/(key+'.json'))
            cache.receipt_matches(receipt,e)
            for field,value in [('text_sha256','changed'),('split','validation'),('source_text_type','smp'),('parent_text_sha256','changed')]:
                bad = dict(e, **{field:value})
                with self.assertRaisesRegex(ValueError,'identity differs'):
                    cache.receipt_matches(receipt,bad)

    def test_cache_reuses_without_encoder_or_copies_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx,reuse = cache_fixture(root)
            with patch.object(cache.frozen, 'Builder', side_effect=AssertionError('must not encode')):
                cache.prepare(root/'new',ctx,1)
                cache.verify(root/'new',ctx)
            self.assertFalse((root/'new/tensors').exists())
            self.assertEqual(cache.read(root/'new/analysis.json')['reused_chunks'],1)
            path = next(reuse.rglob('*.pt'))
            cache.write(path,{'modified':True})
            with self.assertRaisesRegex(ValueError,'Changed reused artifact'):
                cache.verify(root/'new',ctx)

    def test_unsafe_reuse_paths_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'Unsafe'):
                cache.safe_path(Path(tmp),'../outside.pt')


if __name__ == '__main__':
    unittest.main()
