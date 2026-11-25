"""DQ16 synthetic-only embedding, cache and downstream tensor contract tests."""
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

TPG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TPG.parent/'02_prediction/01_package'))
from epss import forecast_coreference_cache as cache
from tpg.schema.graph import TextPropertyGraph
from tpg.schema.types import NodeType, EdgeType, SecurityEdgeType, NodeProperties, EdgeProperties, SECURITY_SCHEMA
from tpg.exporters.exporters import GraphSONExporter


class TinyTokenizer:
    """Deterministic source offsets; no external weights or downloads."""
    def __call__(self, text, **kwargs):
        spans = [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]
        return {'input_ids': torch.arange(1, len(spans)+1)[None],
                'offset_mapping': torch.tensor([spans]), 'attention_mask': torch.ones(1, len(spans), dtype=torch.long)}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, max_position_embeddings=512)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=input_ids.float()[..., None].repeat(1, 1, 4))


class TinyEncoder:
    dim = 4
    def __init__(self, *args):
        self.calls = 0
        self.identity = {'model_name': 'synthetic-only', 'embedding_dim': 4}

    def embed(self, graph):
        self.calls += 1
        cache.embed_graph(graph, TinyTokenizer(), TinyModel().eval(), 'cpu')


def documents(cve='CVE-2000-0001', with_whitespace=False):
    g = TextPropertyGraph(schema=SECURITY_SCHEMA, doc_id=cve)
    text = 'Alpha Alpha 1.0' + ('\n' if with_whitespace else '')
    g.metadata['source_text'] = text
    g.add_node(NodeType.DOCUMENT, NodeProperties(text=text, char_start=0, char_end=len(text)))
    g.add_node(NodeType.SENTENCE, NodeProperties(text=text, char_start=0, char_end=len(text)))
    for label, start, end in [('Alpha', 0, 5), ('Alpha', 6, 11), ('1.0', 12, 15)]:
        g.add_node(NodeType.TOKEN, NodeProperties(text=label, char_start=start, char_end=end))
    g.add_node(NodeType.ENTITY, NodeProperties(text='Alpha', char_start=0, char_end=5, entity_type='SOFTWARE'))
    g.add_node(NodeType.ENTITY, NodeProperties(text='Alpha', char_start=6, char_end=11, entity_type='ORG'))
    g.add_node(NodeType.ENTITY, NodeProperties(text='1.0', char_start=12, char_end=15, entity_type='VERSION'))
    g.add_edge(5, 6, EdgeType.COREF, EdgeProperties(extra={'method': 'repeated_surface', 'status': 'candidate'}))
    g.add_edge(3, 5, EdgeType.COREF, EdgeProperties(extra={'method': 'unambiguous_type_heuristic', 'status': 'candidate'}))
    g.add_edge(5, 7, SecurityEdgeType.HAS_VERSION, EdgeProperties(entity_rel_type='HAS_VERSION'))
    for node in (2, 3, 4):
        g.add_edge(1, node, EdgeType.CONTAINS)
    g.add_edge(2, 3, EdgeType.NEXT_TOKEN)
    g.add_edge(3, 4, EdgeType.NEXT_TOKEN)
    if with_whitespace:
        node = g.add_node(NodeType.TOKEN, NodeProperties(text='\n', char_start=15, char_end=16))
        g.add_edge(1, node, EdgeType.CONTAINS)
        g.add_edge(4, node, EdgeType.NEXT_TOKEN)
    g.mark_pass(cache.preview.semantic.frozen.audit.extension.VERSION)
    return {m: json.loads(GraphSONExporter().export_string(cache.preview.policy.ablate(g, m))) for m in cache.MODES}


class SharedCacheTests(unittest.TestCase):
    def test_encoder_offline_freeze_and_loading_failure(self):
        tokenizer = TinyTokenizer()
        tokenizer.is_fast = True
        tokenizer.backend_tokenizer = SimpleNamespace(to_str=lambda: 'synthetic-tokenizer')
        model = TinyModel()
        model.config.to_json_string = lambda: '{"hidden_size":4}'
        info = {'missing_keys': [], 'mismatched_keys': [], 'error_msgs': [], 'unexpected_keys': []}
        with patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer) as tok, \
                patch('transformers.AutoModel.from_pretrained', return_value=(model, info)) as weights:
            encoder = cache.FrozenEncoder()
            self.assertTrue(tok.call_args.kwargs['local_files_only'])
            self.assertTrue(weights.call_args.kwargs['local_files_only'])
            self.assertFalse(weights.call_args.kwargs['add_pooling_layer'])
            self.assertFalse(encoder.model.training)
            cache.FrozenEncoder(allow_download=True)
            self.assertFalse(tok.call_args.kwargs['local_files_only'])
            self.assertFalse(weights.call_args.kwargs['local_files_only'])
            info['missing_keys'] = ['encoder.layer.0.weight']
            with self.assertRaisesRegex(ValueError, 'Incomplete encoder weights'):
                cache.FrozenEncoder()

    def test_embed_once_shared_features_and_three_edges(self):
        docs, encoder = documents(), TinyEncoder()
        before = deepcopy(docs)
        bundle, vocab = cache.build_bundle(docs, encoder)
        self.assertEqual(encoder.calls, 1)
        self.assertEqual(docs, before)
        self.assertEqual([int(bundle['edge_masks'][m].sum()) for m in cache.MODES], [6, 7, 8])
        for mode in cache.MODES:
            item = cache.materialize(bundle, mode)
            cache.base.validate_item(item, vocab, 4)
            self.assertIs(item['x'], bundle['shared']['x'])
            self.assertIs(item['text_embedding'], bundle['shared']['text_embedding'])
        self.assertEqual(bundle['shared']['x'][0, -1].item(), 0)
        self.assertTrue(torch.equal(bundle['shared']['x'][0, vocab['num_node_types']:-1], torch.zeros(4)))

    def test_mutated_mask_and_future_fields_rejected(self):
        docs = documents()
        bundle, vocab = cache.build_bundle(docs, TinyEncoder())
        changed = deepcopy(bundle)
        changed['edge_masks']['no_coref'][0] = True
        with self.assertRaisesRegex(ValueError, 'mask differs'):
            cache.validate_bundle(changed, docs, vocab, 4)
        for field in ('target', 'epss_sequence', 'vckev_date_added'):
            changed = deepcopy(bundle)
            changed['shared'][field] = 1
            with self.assertRaisesRegex(ValueError, 'forbidden'):
                cache.validate_bundle(changed, docs, vocab, 4)

    def test_missing_token_embedding_fails(self):
        with patch.object(TinyEncoder, 'embed', return_value=None):
            with self.assertRaisesRegex(ValueError, 'Missing text-token embeddings'):
                cache.build_bundle(documents(), TinyEncoder())

    def test_whitespace_token_is_preserved_without_fake_embedding(self):
        docs = documents(with_whitespace=True)
        bundle, vocab = cache.build_bundle(docs, TinyEncoder())
        self.assertEqual(bundle['shared']['x'][-1, -1].item(), 0)
        self.assertEqual(cache.row_statistics(bundle, vocab)['unembedded_whitespace_tokens'], 1)
        ordinary, _ = cache.build_bundle(documents(), TinyEncoder())
        self.assertTrue(torch.equal(bundle['shared']['text_embedding'], ordinary['shared']['text_embedding']))

    def test_finite_values_and_text_pool_checked(self):
        docs = documents()
        bundle, vocab = cache.build_bundle(docs, TinyEncoder())
        changed = deepcopy(bundle)
        changed['shared']['x'][2, -2] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            cache.validate_bundle(changed, docs, vocab, 4)
        changed = deepcopy(bundle)
        changed['shared']['text_embedding'] += 1
        with self.assertRaisesRegex(ValueError, 'pooling differs'):
            cache.validate_bundle(changed, docs, vocab, 4)

    def test_frozen_split_and_eligibility_guards(self):
        rows = [{'cve': 'CVE-2000-0001', 'split': 'train'}, {'cve': 'CVE-2000-0002', 'split': 'validation'}]
        splits = {r['cve']: {'split': r['split'], 'group_id': r['cve']} for r in rows}
        cohorts = {'endpoint': {r['split']: [r['cve']] for r in rows}}
        self.assertEqual(cache.selected_ids(rows, splits, cohorts), cohorts['endpoint'])
        changed = deepcopy(rows); changed[1]['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'Forbidden split'):
            cache.selected_ids(changed, splits, cohorts)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            cache.selected_ids(rows + rows[:1], splits, cohorts)
        changed = deepcopy(splits); changed[rows[1]['cve']]['group_id'] = rows[0]['cve']
        with self.assertRaisesRegex(ValueError, 'related CVEs'):
            cache.selected_ids(rows, changed, cohorts)
        changed = deepcopy(cohorts); changed['endpoint']['train'] = []
        with self.assertRaisesRegex(ValueError, 'eligibility'):
            cache.selected_ids(rows, splits, changed)

    def test_export_alignment_does_not_depend_on_numeric_relation_ids(self):
        docs = documents()
        bundle, vocab = cache.build_bundle(docs, TinyEncoder())
        idx = vocab['edge_types']['COREF']
        for mode, count in zip(cache.MODES, (0, 1, 2)):
            item = cache.materialize(bundle, mode)
            self.assertEqual(int((item['edge_type'] == idx).sum()), count)
        docs['no_coref']['vertices'][0]['properties']['TEXT'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'nodes differ'):
            cache.validate_bundle(bundle, docs, vocab, 4)

    def test_synthetic_gnn_forward_contract(self):
        from torch_geometric.data import Data, Batch
        from epss.forecast_tpg import TPGRegressor, config_for
        bundle, vocab = cache.build_bundle(documents(), TinyEncoder())
        with torch.random.fork_rng():
            torch.manual_seed(17)
            model = TPGRegressor('tpg_only', bundle['shared']['x'].shape[1], 4, vocab['edge_types'], config_for('smoke')).eval()
            for mode in cache.MODES:
                item = cache.materialize(bundle, mode)
                batch = Batch.from_data_list([Data(**{k: item[k] for k in ('x', *cache.EDGE_FIELDS)})])
                with torch.inference_mode():
                    value = model(batch)
                self.assertEqual(tuple(value.shape), (1,))
                self.assertTrue(bool(torch.isfinite(value).all()))

    def test_save_reload_check_only_and_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root/'source/smoke', root/'cache/smoke'
            rows, ids = [], {'train': [], 'validation': []}
            for cve, split in [('CVE-2000-0001', 'train'), ('CVE-2000-0002', 'validation')]:
                docs = documents(cve)
                for mode in cache.MODES:
                    path = source/'graphs'/mode/(cve+'.json')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_json(path, docs[mode])
                rows.append({'cve': cve, 'split': split, 'description_sha256': cache.base.text_digest('Alpha Alpha 1.0')})
                ids[split].append(cve)
            ctx = (rows, ids, {})
            with patch.object(cache, 'context', return_value=ctx), patch.object(cache, 'SOURCE', source), \
                    patch.object(cache, 'DEFAULT_OUTPUT', output), patch.object(cache, 'FrozenEncoder', TinyEncoder):
                cache.prepare(output, check_only=True)
                self.assertFalse(output.exists())
                cache.prepare(output)
                cache.prepare(output, verify_only=True)
                manifest, items = cache.load_cache(output, 'repeated_only')
                self.assertEqual(len(items), 2)
                self.assertEqual(manifest['ids'], ids)
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    cache.prepare(output)
                (output/'analysis.txt').write_text('tampered', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Changed artifact'):
                    cache.verify(output)


if __name__ == '__main__':
    unittest.main()
