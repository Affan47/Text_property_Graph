"""Synthetic regression checks for DQ17; no dataset training or downloads."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

TPG = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TPG.parent/'02_prediction/01_package'), str(Path(__file__).resolve().parent)]
from epss import forecast_coreference_development as dev
from test_security_coreference_cache import documents, TinyEncoder


class FakeBuilder:
    calls = []
    fail_at = None

    def __init__(self, *args):
        self.encoder = TinyEncoder()

    def build(self, cve, text):
        self.calls.append(cve)
        if cve == self.fail_at:
            raise RuntimeError('simulated interruption')
        docs = documents(cve)
        bundle, vocab = dev.shared.build_bundle(docs, self.encoder)
        return docs, bundle, vocab


class DevelopmentTests(unittest.TestCase):
    def fixture(self, root):
        ids = {'train': ['CVE-2000-0001', 'CVE-2000-0002', 'CVE-2000-0003'],
               'validation': ['CVE-2001-0001', 'CVE-2001-0002', 'CVE-2001-0003']}
        prior_ids = {s: v[:1] for s,v in ids.items()}
        source = root/'prior'
        saved_docs = {}
        for split in dev.ROLES:
            cve = prior_ids[split][0]
            docs = documents(cve)
            bundle, vocab = dev.shared.build_bundle(docs, TinyEncoder())
            (source/'graphs').mkdir(parents=True, exist_ok=True)
            torch.save(bundle, source/'graphs'/(cve+'.pt'))
            saved_docs[cve] = docs
        prior = {'ids': prior_ids, 'type_vocabulary': vocab, 'embedding_dim': 4,
                 'encoder': {'test': 'synthetic'}}
        ctx = {'full_ids': ids, 'smoke_ids': ids, 'prior': prior, 'pins': {},
               'protocol': {'test': True},
               'descriptions': {c: 'Alpha Alpha 1.0' for v in ids.values() for c in v}}
        return ctx, source, saved_docs

    def test_selection_full_smoke_and_role_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx, _, _ = self.fixture(Path(temp))
        ids = ctx['full_ids']
        data = {'task_cohorts.json': {'endpoint': ids},
                'split_manifest.json': {c: {'split': s, 'group_id': c} for s in ids for c in ids[s]},
                'labels.json': {c: {'eligible': {'endpoint': True}} for s in ids for c in ids[s]}}
        expected = {'train': 3, 'validation': 3}
        self.assertEqual(dev.select_ids(data, ctx['prior']['ids'], 'full', expected), ids)
        self.assertEqual(dev.select_ids(data, ctx['prior']['ids'], 'smoke', expected), ids)
        with self.assertRaisesRegex(ValueError, 'counts changed'):
            dev.select_ids(data, ctx['prior']['ids'], 'full', {'train': 4, 'validation': 3})
        changed = deepcopy(data)
        changed['split_manifest.json'][ids['validation'][0]]['group_id'] = ids['train'][0]
        with self.assertRaisesRegex(ValueError, 'Related groups'):
            dev.select_ids(changed, ctx['prior']['ids'], 'full', expected)
        changed = deepcopy(data)
        changed['split_manifest.json'][ids['validation'][0]]['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'role/eligibility'):
            dev.select_ids(changed, ctx['prior']['ids'], 'full', expected)

    def test_encoder_identity_rejects_different_weights_or_tokenizer(self):
        fields = ('model_name', 'revision', 'weights_sha256', 'tokenizer_sha256', 'config_sha256',
                  'pooler', 'frozen', 'device', 'embedding_dim', 'embedding_method', 'packages')
        ident = {k: k for k in fields}
        dev.check_encoder(ident, ident)
        for field in fields:
            changed = dict(ident, **{field: 'changed'})
            with self.assertRaisesRegex(ValueError, 'Encoder identity changed'):
                dev.check_encoder(changed, ident)

    def test_preflight_resume_full_gate_and_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ctx, source, docs = self.fixture(root)
            output = root/'cache/smoke'
            FakeBuilder.calls = []
            FakeBuilder.fail_at = None
            with patch.object(dev, 'context', return_value=ctx), \
                    patch.object(dev, 'ROOT', output.parent), \
                    patch.object(dev, 'Builder', FakeBuilder), \
                    patch.object(dev.shared, 'DEFAULT_OUTPUT', source), \
                    patch.object(dev.shared, 'documents_for', side_effect=lambda c: deepcopy(docs[c])):
                dev.prepare(output, check_only=True)
                self.assertFalse(output.exists())
                self.assertEqual(len(FakeBuilder.calls), 2)
                with self.assertRaisesRegex(ValueError, 'small DQ17'):
                    dev.prepare(root/'cache/full', scale='full')
                FakeBuilder.calls = []
                FakeBuilder.fail_at = ctx['full_ids']['train'][2]
                with self.assertRaisesRegex(RuntimeError, 'interruption'):
                    dev.prepare(output)
                old = dev.sha256(output/'graphs'/(ctx['full_ids']['train'][1]+'.pt'))
                FakeBuilder.calls = []
                FakeBuilder.fail_at = None
                dev.prepare(output, resume=True)
                self.assertEqual(len(FakeBuilder.calls), 3)
                self.assertEqual(old, dev.sha256(output/'graphs'/(ctx['full_ids']['train'][1]+'.pt')))
                manifest = dev.verify(output, ctx, 'smoke')
                self.assertEqual(manifest['reused_dq16'], 2)
                self.assertEqual(len(manifest['graphs']), 6)
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    dev.prepare(output)
                with self.assertRaisesRegex(ValueError, 'unfinished output'):
                    dev.prepare(output, resume=True)
                full = root/'cache/full'
                dev.prepare(full, scale='full', smoke_dir=output)
                self.assertEqual(dev.verify(full, ctx, 'full')['smoke_cache_lock_sha256'],
                                 dev.sha256(output/'cache_lock.json'))
                (full/'analysis.txt').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'Changed artifact'):
                    dev.verify(full, ctx)

    def test_resume_rejects_changed_identity_and_completed_tensor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ctx, source, docs = self.fixture(root)
            output = root/'cache/smoke'
            with patch.object(dev, 'context', return_value=ctx), \
                    patch.object(dev, 'ROOT', output.parent), \
                    patch.object(dev, 'Builder', FakeBuilder), \
                    patch.object(dev.shared, 'DEFAULT_OUTPUT', source), \
                    patch.object(dev.shared, 'documents_for', side_effect=lambda c: deepcopy(docs[c])):
                FakeBuilder.fail_at = ctx['full_ids']['train'][1]
                with self.assertRaises(RuntimeError):
                    dev.prepare(output)
                FakeBuilder.fail_at = None
                ctx['protocol']['test'] = False
                with self.assertRaisesRegex(ValueError, 'Resume source/settings'):
                    dev.prepare(output, resume=True)
                ctx['protocol']['test'] = True
                (output/'graphs'/(ctx['full_ids']['train'][0]+'.pt')).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'Changed completed CVE artifact'):
                    dev.prepare(output, resume=True)

    def test_protocol_has_fixed_seeds_budget_and_no_test(self):
        protocol = dev.read_json(dev.PROTOCOL)
        self.assertEqual(protocol['config']['seeds'], [17, 29, 43])
        self.assertEqual(protocol['config']['epochs'], 100)
        self.assertEqual(protocol['config']['patience'], 15)
        self.assertEqual(protocol['config']['dropout'], 0.0)
        self.assertFalse(protocol['cohort']['test_scoring'])
        self.assertEqual(protocol['target']['input_days_inclusive'], [0, 30])
        self.assertEqual(protocol['target']['target_day'], 60)
        self.assertEqual(protocol['variants'], [*dev.shared.MODES, 'epss_only'])


if __name__ == '__main__':
    unittest.main()
