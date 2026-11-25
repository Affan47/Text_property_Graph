"""Synthetic-only tests. No dataset, encoder download, fits or held-out reads."""
from copy import deepcopy
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
for relative in ('02_prediction/01_package', '01_tpg/10_extensions', '01_tpg/01_core'):
    sys.path.insert(0, str(ROOT/relative))

from epss.final_comparison_v1 import test_cache as tc
from tpg_combined_v3 import compact
from tpg_combined_v2 import graphs


def write_csv_gz(path, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(gzip.compress(stream.getvalue().encode(), mtime=0))


def write_jsonl(path, rows):
    path.write_bytes(gzip.compress(b''.join(tc.c.c.encoded(r) for r in rows), mtime=0))


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root/'output'
        self.addCleanup(patch.stopall)
        patch.object(tc.c, 'OUT', self.out).start()
        self.ids = {'train': ['CVE-2098-1000'], 'validation': ['CVE-2098-2000'],
                    'test': ['CVE-2099-1001', 'CVE-2099-1002']}
        self.raw = []
        for i, cid in enumerate(self.ids['test']):
            posts = [{'post': 'A remote attacker can execute code.', 'platform': 'reddit', 'date': '2025-01-04'},
                     {'post': cid, 'platform': 'reddit', 'date': '2025-01-04'},
                     {'post': 'EPSS discussion quarantined.', 'platform': 'reddit', 'date': '2025-01-04'}]
            self.raw.append({'cve': cid, 'published_date': '2025-01-01',
                'description': 'Apache 2.4.1 contains a buffer overflow.',
                'social_media_posts': posts if i == 0 else posts[1:],
                **{k: None for k in tc.structured.FIELDS},
                'in_vckev': True, 'epss_score': 0.999})
        self.decisions = []
        for row in self.raw:
            choices = [('D', row['description'], '')]+[
                ('M', p['post'], tc._fingerprint(p)) for p in row['social_media_posts']]
            for source, text, fp in choices:
                self.decisions.append({'task': 'RD02', 'role': 'test', 'cve': row['cve'], 'source': source,
                    'text_sha256': hashlib.sha256(text.encode()).hexdigest(), 'post_fingerprint': fp,
                    'quarantine': 'true' if text.startswith('EPSS') else 'false',
                    'bare_id': 'true' if text.startswith('CVE-') else 'false'})
        self.paths = {name: self.root/(name+suffix) for name, suffix in (
            ('manifest', '.json'), ('policy', '.json'), ('approval', '.json'),
            ('candidates', '.jsonl.gz'), ('decisions', '.csv.gz'), ('outcomes', '.jsonl.gz'))}
        manifest = {'role_state': 'FROZEN', 'prior_test_connected_excluded': 4987,
                    'tasks': {'RD02': {'ids': self.ids, 'role_ids_sha256': 'frozen-roles'}},
                    'release_schedule': {'v1': '2021-04-14', 'v2': '2022-02-04', 'v3': '2023-03-07'},
                    'publication_intervals': {'test': ['2025-01-01', '2026-11-03']}, 'cutoff': '2026-11-03'}
        tc.c.write(self.paths['manifest'], manifest)
        tc.c.write(self.paths['policy'], {'policy': 'synthetic frozen policy'})
        tc.c.write(self.paths['approval'], {'source_gate': 'ACCEPTED_WITH_DECLARED_DESCRIPTION_OVERLAP',
                   'original_manifest_sha256': tc.c.sha(self.paths['manifest']), 'role_ids_sha256': 'frozen-roles'})
        write_csv_gz(self.paths['decisions'], self.decisions)
        write_jsonl(self.paths['candidates'], self.raw)
        # Deliberately invalid outcomes prove predictor preparation cannot parse them.
        self.paths['outcomes'].write_bytes(b'OUTCOMES MUST NOT BE OPENED HERE')
        state = tc.structured.fit([{'cve': self.ids['train'][0], 'split': 'train',
                                    'structured': {k: None for k in tc.structured.FIELDS}}])
        tc.c.write(tc._state_path(), state)
        self.constants = tc._state_path().with_name('constants.json')
        tc.c.write(self.constants, {'training_mean': 0.02, 'training_median': 0.001})
        self.protocol = {'settings': {'seeds': list(tc.SEEDS)}, 'official_ids': deepcopy(self.ids),
            'contract_template': {'structured_names': state['feature_names'], 'embedding_dim': 768},
            'cache_processing': {'schema': 'combined_text_securebert2_v3'},
            'cache_encoder': {'trainable': False, 'dtype': 'float32', 'device': 'cpu', 'threads': 4},
            'test_source_paths': {k: str(v) for k, v in self.paths.items()},
            'input_pins': {str(v): tc.c.sha(v) for v in self.paths.values()},
            'code_pins': {str(Path(tc.__file__)): tc.c.sha(tc.__file__)}}
        self.fits = []
        for model in tc.c.MODELS:
            for seed in tc.SEEDS:
                directory = self.out/'fits'/f'{model}__seed_{seed}'
                checkpoint = directory/'best_checkpoint.pt'
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(f'synthetic checkpoint {model}/{seed}'.encode())
                tc.c.publication.lock(directory)
                self.fits.append({'model': model, 'seed': seed, 'path': checkpoint,
                                  'completion_lock': directory/'artifact_lock.json'})
        self.parent_processing = {'max_characters': 500000, 'smp_policy': 'prose'}

    def freeze(self):
        return tc.freeze_release(self.protocol, self.fits, self.constants)

    def inputs(self):
        release = self.freeze()
        with patch.object(tc, '_development_metadata', return_value=(self.parent_processing, {'parser': 'fixture'})):
            ctx = tc.prepare(self.protocol, release, stage='inputs')
        return release, ctx

    def test_no_label_reads_without_explicit_evaluate_release(self):
        with patch.object(tc, '_jsonl', side_effect=AssertionError('Must not open outcomes')):
            with self.assertRaisesRegex(ValueError, 'closed'):
                tc.load_test_labels(self.protocol)
            with self.assertRaisesRegex(ValueError, 'evaluate --release-test'):
                with tc.evaluation_release(self.protocol, None):
                    pass
            with self.assertRaisesRegex(ValueError, 'evaluate --release-test'):
                with tc.evaluation_release(self.protocol, None, release_test=True, stage='train'):
                    pass

    def test_incomplete_duplicate_checkpoint_sets_block_before_sources(self):
        for records in (self.fits[:-1], self.fits[:-1]+[self.fits[0]]):
            with self.assertRaisesRegex(ValueError, '25 official'):
                tc.freeze_release(self.protocol, records, self.constants)
        self.assertFalse((self.out/'release_lock.json').exists())

    def test_complete_set_and_frozen_transform_required(self):
        release = self.freeze()
        self.assertEqual(release, tc.verify_release(self.protocol, release))
        self.assertEqual(release, self.freeze())
        state = tc.c.read(tc._state_path())
        state['fit_ids'].append(self.ids['test'][0])
        tc.c.write(tc._state_path(), state)
        with self.assertRaises(ValueError):
            tc.verify_release(self.protocol, release)
        with self.assertRaisesRegex(ValueError, 'training-only'):
            self.freeze()

    def test_changed_checkpoint_and_incomplete_lock_are_rejected(self):
        release = self.freeze()
        self.fits[0]['path'].write_bytes(b'changed')
        with self.assertRaises(ValueError):
            tc.verify_release(self.protocol, release)
        tc.c.write(self.fits[0]['completion_lock'], {'files': {}})
        with self.assertRaisesRegex(ValueError, 'covered'):
            self.freeze()

    def test_changed_protocol_rejected_before_test_adapter(self):
        release = self.freeze()
        self.protocol['settings']['loss'] = 'changed'
        with patch.object(tc, '_jsonl', side_effect=AssertionError('No source reads')):
            with self.assertRaisesRegex(ValueError, 'release'):
                tc.prepare(self.protocol, release)

    def test_outcome_path_must_be_pinned_without_reading_its_values(self):
        del self.protocol['input_pins'][str(self.paths['outcomes'])]
        with self.assertRaisesRegex(ValueError, 'Freeze every test-source'):
            self.freeze()

    def test_source_screening_matches_rd02_and_never_copies_outcomes(self):
        decisions = {(r['cve'], r['source'], r['text_sha256'], r['post_fingerprint']): r for r in self.decisions}
        row = tc.screen_predictor(self.raw[0], decisions)
        self.assertEqual(set(row), {'cve', 'split', 'documents', 'structured'})
        self.assertEqual(set(row['structured']), set(tc.structured.FIELDS))
        self.assertEqual([d['source'] for d in row['documents']], ['D', 'M'])
        self.assertTrue(row['documents'][1]['is_prose'])
        self.assertEqual(row['documents'][0]['text'], self.raw[0]['description'])
        with self.assertRaisesRegex(ValueError, 'Missing frozen'):
            tc.screen_predictor(self.raw[0], {})
        bad = deepcopy(decisions)
        next(iter(bad.values()))['quarantine'] = ''
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            tc.screen_predictor(self.raw[0], bad)

    def test_input_preparation_never_reads_labels_or_fits_preprocessing(self):
        real_jsonl = tc._jsonl
        def guarded(path):
            self.assertNotEqual(path, self.paths['outcomes'])
            return real_jsonl(path)
        with patch.object(tc, '_jsonl', side_effect=guarded), patch.object(tc.structured, 'fit', side_effect=AssertionError('No fitting')):
            release, ctx = self.inputs()
        self.assertNotIn('labels', ctx)
        self.assertTrue(ctx['is_test'])
        self.assertEqual(ctx['ids'], {'train': [], 'validation': self.ids['test'], 'assessment': []})
        self.assertEqual({r['split'] for r in ctx['index'].values()}, {'test'})
        no_posts = ctx['index'][self.ids['test'][1]]['views']
        self.assertEqual(no_posts['D']['object_key'], no_posts['DM']['object_key'])
        self.assertFalse(no_posts['DM']['source_mask']['M'])
        self.assertEqual(ctx['metadata'][self.ids['test'][0]]['target_date'], '2025-03-02')
        self.assertEqual(ctx['metadata'][self.ids['test'][0]]['target_version'], 'v3')
        self.assertFalse((self.out/'test_evaluation_release.json').exists())
        self.assertFalse((tc._root()/'cache_verification.json').exists())
        self.assertEqual(len(tc._load_plan(self.protocol, release)['entries']), 3)

    def test_no_fake_development_split_in_extraction_arguments(self):
        release, _ = self.inputs()
        plan = tc._load_plan(self.protocol, release)
        for entry in plan['entries'].values():
            self.assertEqual(entry['split'], 'test')
            self.assertNotIn('split', tc._content(entry))
        self.assertTrue(plan['test_adapted'])

    def test_verify_only_cannot_publish_missing_inputs_or_cache_markers(self):
        release = self.freeze()
        with self.assertRaisesRegex(ValueError, 'No test cache'):
            tc.prepare(self.protocol, release, stage='inputs', verify_only=True)
        self.assertFalse(tc._root().exists())
        self.inputs()
        old = {str(p): tc.c.sha(p) for p in tc._root().rglob('*') if p.is_file()}
        with patch.object(tc, '_development_metadata', return_value=(self.parent_processing, {'parser': 'fixture'})):
            tc.prepare(self.protocol, release, stage='inputs', verify_only=True)
        self.assertEqual(old, {str(p): tc.c.sha(p) for p in tc._root().rglob('*') if p.is_file()})
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            tc.prepare(self.protocol, release, stage='verify', verify_only=True)
        self.assertFalse((tc._root()/'cache_verification.json').exists())

    def test_changed_source_pin_blocks_before_any_adaptation(self):
        release = self.freeze()
        self.paths['candidates'].write_bytes(b'changed')
        with patch.object(tc, '_prepare_inputs', side_effect=AssertionError('Must fail first')):
            with self.assertRaisesRegex(ValueError, 'Changed frozen'):
                tc.prepare(self.protocol, release)

    def test_missing_selected_cve_rejected_without_publish(self):
        write_jsonl(self.paths['candidates'], self.raw[:1])
        self.protocol['input_pins'][str(self.paths['candidates'])] = tc.c.sha(self.paths['candidates'])
        release = self.freeze()
        with self.assertRaisesRegex(ValueError, 'Missing test predictors'):
            tc.prepare(self.protocol, release)
        self.assertFalse((tc._root()/'inputs').exists())

    def test_grouping_matches_existing_label_blind_rule(self):
        from epss.training_fold_audit_v1 import core
        rows = [{'cve': 'C', 'documents': [{'text': 'caf\u00e9 shared'}, {'text': 'second'}]},
                {'cve': 'B', 'documents': [{'text': 'cafe\u0301   shared'}]},
                {'cve': 'A', 'documents': [{'text': 'second'}]},
                {'cve': 'D', 'documents': [{'text': 'Second'}]}]
        expected = core.components({r['cve']: {'document_fingerprints': [core.fingerprint(d['text'])
                   for d in r['documents']]} for r in rows})[0]
        self.assertEqual(tc.document_groups(rows), expected)
        self.assertEqual(expected, {'A': 'A', 'B': 'A', 'C': 'A', 'D': 'D'})


class SyntheticGraphTests(ReleaseTests):
    """A small synthetic graph uses real local extraction and correction functions."""
    @classmethod
    def setUpClass(cls):
        from tpg_combined_recovery_v1.boundary import Builder
        torch.set_num_threads(1)
        cls.builder = Builder()

    def built(self):
        # Derive the real vocabulary from synthetic text only; no saved graph is opened.
        documents = [{'source': 'D', 'document_id': 'D-unit', 'text': 'Apache contains a buffer overflow.',
                      'text_sha256': hashlib.sha256(b'Apache contains a buffer overflow.').hexdigest(), 'is_prose': False}]
        text, spans = tc.combine(documents, 'D', 'prose')
        entry = {'cve': 'CVE-2099-1001', 'text': text, 'spans': spans, 'processing_identity': 'fixture'}
        graph, _ = self.builder.build('fixture', entry)
        _, metadata = compact.pack(graphs.tensors(graph), graph)
        self.protocol['contract_template'] = tc.c.base.contract(metadata, self.protocol['cache_encoder'],
                                                                tc.c.read(tc._state_path())['feature_names'])
        release = self.freeze()
        with patch.object(tc, '_development_metadata', return_value=(self.parent_processing, self.builder.identity)):
            tc.prepare(self.protocol, release, stage='inputs')
        plan = tc._load_plan(self.protocol, release)
        identity = self.protocol['cache_encoder']
        tc.graph_common.create(tc._root()/'encoder_identity.json', identity)
        for key, entry in plan['entries'].items():
            record = tc._build_one(tc._root(), key, entry, plan, self.builder, self.builder.identity)
            obj = tc._root()/'objects'/key
            layout, _ = compact.load(obj/'structure.npz')
            count = len(layout['pool_ptr'])-1
            rows = np.arange(count*768, dtype=np.float32).reshape(count, 768)/100000
            observed = np.ones(count, dtype=bool)
            rows[0] = 0
            observed[0] = False
            compact.save(obj/'embeddings.npz', {'rows': rows, 'observed': observed},
                         {'format': compact.FORMAT, 'encoder_sha256': tc.c.digest(identity)})
            tc.graph_common.create(obj/'embedding_receipt.json', {
                'graph_sha256': record['files']['graph.json.gz'], 'structure_sha256': record['files']['structure.npz'],
                'encoder_sha256': tc.c.digest(identity), 'coverage': {'exceptions': [], 'expanded_nodes': len(layout['pool_index']),
                'stored_rows': count}, 'files': {'embeddings.npz': tc.c.sha(obj/'embeddings.npz')}})
        return release, plan

    def test_corrected_compact_graph_model_input_and_pooled_parity(self):
        release, plan = self.built()
        ctx = tc.prepare(self.protocol, release, stage='verify')
        self.assertNotIn('labels', ctx)
        self.assertTrue(ctx['cache_complete'])
        for key in plan['entries']:
            actual = tc.load_graph(ctx, key)
            reference = tc.c.base.GraphStore(ctx, cache_mib=0).load(key)
            for name, tensor in actual.items():
                self.assertTrue(torch.equal(tensor, reference[name]), name)
            self.assertEqual(actual['edge_features'].shape[1], 88)
        from epss.training_only_pooled_v1.data import vector_job
        for cid, views in ctx['index'].items():
            for view in tc.VIEWS:
                key = views['views'][view]['object_key']
                expected = vector_job((str(tc._root()), key, [s['source'] for s in plan['entries'][key]['spans']],
                                       tc.c.digest(self.protocol['cache_encoder']), None))
                np.testing.assert_array_equal(ctx['pooled'][view][cid], np.asarray(expected['vector'], np.float32))
        with self.assertRaisesRegex(ValueError, 'Unverified'):
            tc.load_graph({**ctx, 'is_test': False}, next(iter(plan['entries'])))

    def test_independent_development_and_test_extraction_have_identical_tensors(self):
        from tpg_combined_v3.embeddings import Encoder

        class CharacterTokenizer:
            def __call__(self, text, **kwargs):
                return {'input_ids': [[ord(ch) for ch in text]], 'attention_mask': [[1]*len(text)],
                        'offset_mapping': [[(i, i+1) for i in range(len(text))]]}

        class CharacterModel:
            def __call__(self, input_ids, attention_mask):
                from types import SimpleNamespace
                return SimpleNamespace(last_hidden_state=input_ids.float()[:, :, None].repeat(1, 1, 768)/1000)

        encoder = Encoder.__new__(Encoder)
        encoder.device, encoder.tokenizer, encoder.model = 'cpu', CharacterTokenizer(), CharacterModel()
        _, plan = self.built()
        # Independent old-path build with a real development-role entry, from
        # invented text. No saved development or held-out graph is read here.
        for key, entry in plan['entries'].items():
            parent_entry = {**entry, 'split': 'train', 'processing_identity': tc.c.digest(plan['parent_processing'])}
            parent_key = tc.c.digest(parent_entry)
            original, _ = self.builder.build(parent_key, parent_entry)
            dev_entry = {**entry, 'split': 'train'}
            reference = tc.compact_cache.transform(original, dev_entry, parent_entry, parent_key, tc.c.digest(original))
            actual = tc.c.read(tc._root()/'objects'/key/'graph.json.gz')
            self.assertEqual(actual['metadata']['evaluation_role'], 'test')
            self.assertNotIn('evaluation_role', reference['metadata'])
            for name, value in graphs.tensors(reference).items():
                other = graphs.tensors(actual)[name]
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(value, other), name)
                else:
                    self.assertEqual(value, other, name)
            reference_layout = compact.pooling_layout(reference)
            actual_layout = compact.pooling_layout(actual)
            for field in reference_layout:
                np.testing.assert_array_equal(reference_layout[field], actual_layout[field])
            ref_vectors, ref_mask, _ = encoder.encode_compact(entry['text'], reference, reference_layout)
            vectors, mask, _ = encoder.encode_compact(entry['text'], actual, actual_layout)
            np.testing.assert_array_equal(ref_vectors, vectors)
            np.testing.assert_array_equal(ref_mask, mask)

    def test_parallel_build_and_resume_keep_parent_publications_independent(self):
        release = self.freeze()
        with patch.object(tc, '_development_metadata', return_value=(self.parent_processing, self.builder.identity)):
            tc.prepare(self.protocol, release, stage='inputs')
        # Real two-process parsing of only three invented graphs, no encoder or fit.
        tc.prepare(self.protocol, release, stage='graphs', workers=2)
        plan = tc._load_plan(self.protocol, release)
        before = {key: tc.c.sha(tc._root()/'objects'/key/'graph_receipt.json') for key in plan['entries']}
        tc.prepare(self.protocol, release, stage='graphs', workers=2, resume=True)
        self.assertEqual(before, {key: tc.c.sha(tc._root()/'objects'/key/'graph_receipt.json') for key in plan['entries']})
        self.assertFalse((tc._root()/'cache_verification.json').exists())

    def test_verification_does_not_create_markers_and_payload_corruption_fails(self):
        release, plan = self.built()
        with self.assertRaisesRegex(ValueError, 'No published'):
            tc.prepare(self.protocol, release, stage='verify', verify_only=True)
        self.assertFalse((tc._root()/'cache_verification.json').exists())
        tc.prepare(self.protocol, release, stage='verify')
        marker = tc._root()/'cache_verification.json'
        before = marker.read_bytes()
        tc.prepare(self.protocol, release, stage='verify', verify_only=True)
        self.assertEqual(before, marker.read_bytes())
        key = next(iter(plan['entries']))
        (tc._root()/'objects'/key/'embeddings.npz').write_bytes(b'bad')
        with self.assertRaises(ValueError):
            tc.prepare(self.protocol, release, stage='verify', verify_only=True)

    def test_evaluation_release_replay_and_context_reset(self):
        # Synthetic labels are inserted before freezing, not into an existing release.
        targets = [{'cve': cid, 'epss_outcomes': {'epss_scores': {'2025-03-02': 0.04},
                    'epss_score_status': {'2025-03-02': 'OBSERVED'}}} for cid in self.ids['test']]
        write_jsonl(self.paths['outcomes'], targets)
        self.protocol['input_pins'][str(self.paths['outcomes'])] = tc.c.sha(self.paths['outcomes'])
        release, _ = self.built()
        tc.prepare(self.protocol, release, stage='verify')
        with self.assertRaisesRegex(ValueError, 'No existing scientific'):
            with tc.evaluation_release(self.protocol, release, release_test=True, verify_only=True):
                pass
        self.assertFalse((self.out/'test_evaluation_release.json').exists())
        with tc.evaluation_release(self.protocol, release, release_test=True):
            labels = tc.load_test_labels(self.protocol)
        self.assertEqual(set(labels), set(self.ids['test']))
        self.assertEqual(labels[self.ids['test'][0]], {'value': 0.04, 'date': '2025-03-02'})
        marker = (self.out/'test_evaluation_release.json').read_bytes()
        with tc.evaluation_release(self.protocol, release, release_test=True, verify_only=True):
            self.assertEqual(labels, tc.load_test_labels(self.protocol))
        self.assertEqual(marker, (self.out/'test_evaluation_release.json').read_bytes())
        with self.assertRaisesRegex(ValueError, 'closed'):
            tc.load_test_labels(self.protocol)


if __name__ == '__main__':
    unittest.main()
