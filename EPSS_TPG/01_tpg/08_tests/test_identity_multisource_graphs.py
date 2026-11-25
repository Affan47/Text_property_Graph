"""CD03: lossless chunking, visible graph files, source routing and safe resume."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

TPG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TPG.parent / '02_prediction/01_package'))
from epss import forecast_identity_graphs as cache
from test_multisource_graph_preview import synthetic_build
from test_security_coreference_cache import TinyEncoder


class FakeBuilder:
    calls = 0
    stop = None

    def __init__(self, *args):
        self.encoder = TinyEncoder()

    def build(self, entry):
        if self.stop is not None and type(self).calls == self.stop:
            raise RuntimeError('synthetic interruption')
        type(self).calls += 1
        docs, bundle, vocab = synthetic_build(entry, self.encoder)
        for g in docs.values():
            graph = cache.old.shared.import_graphson(g)
            cache.annotate(graph, entry)
            g.update(json.loads(cache.old.shared.GraphSONExporter().export_string(graph)))
        return docs, bundle, vocab


def fixture():
    ids = {'train': ['CVE-2000-0001'], 'validation': ['CVE-2000-0002']}
    rows = []
    for split, chosen in ids.items():
        kinds = ['description', 'smp'] if split == 'train' else ['description']
        rows.append({'cve': chosen[0], 'split': split, 'source_mask': [int(s in kinds) for s in cache.old.SOURCES],
            'documents': [{'document_id': s + ':example', 'source_text_type': s, 'text': 'Alpha Alpha'} for s in kinds]})
    ctx = cache.plan(rows, ids, 4096)
    e = next(iter(ctx['entries'].values()))
    _, _, vocab = synthetic_build(e, TinyEncoder())
    ctx['identity'] = {'schema': cache.SCHEMA, 'scale': 'smoke', 'ids': ids, 'source_pins': {},
        'embedding_dim': 4, 'type_vocabulary': vocab, 'protocol': {'limitations': ['Synthetic test']}}
    return ctx, rows


class GraphTests(unittest.TestCase):
    def setUp(self):
        FakeBuilder.calls, FakeBuilder.stop = 0, None

    def test_chunk_coverage_and_bound(self):
        for text in ('alpha beta\n' * 50, 'x' * 87, 'a' + ' ' * 40 + 'b', 'é text\n' * 8):
            s = cache.segments(text, 16)
            self.assertEqual(''.join(text[r['start']:r['end']] for r in s), text)
            self.assertTrue(all(0 < r['end'] - r['start'] <= 16 for r in s))
            self.assertEqual([r['end'] for r in s[:-1]], [r['start'] for r in s[1:]])

    def test_forced_and_non_graph_segments_are_explicit(self):
        self.assertEqual(cache.segments('x' * 30, 16)[0]['boundary'], 'forced_character_boundary')
        s = cache.segments('Alpha' + ' ' * 40 + 'Beta', 16)
        self.assertTrue(any(not r['has_graph'] and r['non_graph_reason'] for r in s))
        with self.assertRaises(ValueError):
            cache.segments('text', 0)

    def test_test_cves_and_extra_target_fields_rejected(self):
        ctx, rows = fixture()
        rows[0]['target'] = .4
        with self.assertRaises(ValueError):
            cache.plan(rows, ctx['identity']['ids'], 4096)
        del rows[0]['target']
        rows[0]['split'] = 'test'
        with self.assertRaises(ValueError):
            cache.plan(rows, ctx['identity']['ids'], 4096)

    def test_namespace_and_indexes_retain_cves(self):
        ctx, _ = fixture()
        self.assertEqual(len(ctx['entries']), 3)
        self.assertEqual(len({e['graph_id'] for e in ctx['entries'].values()}), 3)
        for e in ctx['entries'].values():
            self.assertIn(e['cve'] + '/' + e['source_text_type'], e['stem'])

    def test_save_verify_actual_graphs_and_numeric_isolation(self):
        ctx, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'cache'
            cache.prepare(out, ctx, builder_factory=FakeBuilder)
            a = cache.verify(out, ctx)
            self.assertEqual(FakeBuilder.calls, 3)
            self.assertEqual(sum(r['chunks'] for r in a['coverage']), 3)
            item = cache.numeric_cve(out, ctx, 'CVE-2000-0001', ['description'])
            self.assertEqual(item['source_mask'], [True, False, False, False])
            graph = item['by_source']['description'][0][0]
            self.assertEqual(set(graph), set(cache.old.NUMERIC_FIELDS))
            no_text = cache.numeric_cve(out, ctx, 'CVE-2000-0001', [])
            self.assertFalse(any(no_text['source_mask']))
            self.assertTrue((out / 'by_cve/CVE-2000-0001.json').exists())

    def test_interruption_resume_and_receipt_integrity(self):
        ctx, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'cache'
            FakeBuilder.stop = 1
            with self.assertRaises(RuntimeError):
                cache.prepare(out, ctx, builder_factory=FakeBuilder)
            first = next(iter(ctx['entries'].values()))
            before = cache.sha256(out / 'tensors' / (first['stem'] + '.pt'))
            FakeBuilder.stop = None
            cache.prepare(out, ctx, resume=True, builder_factory=FakeBuilder)
            self.assertEqual(FakeBuilder.calls, 3)
            self.assertEqual(before, cache.sha256(out / 'tensors' / (first['stem'] + '.pt')))
            cache.verify(out, ctx)
            with self.assertRaises(ValueError):
                cache.prepare(out, ctx, resume=True, builder_factory=FakeBuilder)

    def test_reuse_does_not_encode_again(self):
        ctx, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            small, full = Path(tmp) / 'small', Path(tmp) / 'full'
            cache.prepare(small, ctx, builder_factory=FakeBuilder)
            cache.prepare(full, ctx, builder_factory=FakeBuilder, reuse=small)
            self.assertEqual(FakeBuilder.calls, 3)
            cache.verify(full, ctx)

    def test_tampered_graph_is_rejected(self):
        ctx, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'cache'
            cache.prepare(out, ctx, builder_factory=FakeBuilder)
            e = next(iter(ctx['entries'].values()))
            path = out / 'tpg_json/no_coref' / (e['stem'] + '.json')
            path.write_text('{}')
            with self.assertRaises(ValueError):
                cache.verify(out, ctx)

    def test_node_parent_offsets(self):
        ctx, _ = fixture()
        e = deepcopy(next(iter(ctx['entries'].values())))
        e['chunk']['start'] = 100
        e['chunk']['end'] = 111
        docs, _, _ = FakeBuilder().build(e)
        for n in docs['no_coref']['vertices']:
            p = n['properties']
            self.assertEqual(p['EXTRA']['source_document_char_start'], 100 + p['CHAR_START'])


if __name__ == '__main__':
    unittest.main()
