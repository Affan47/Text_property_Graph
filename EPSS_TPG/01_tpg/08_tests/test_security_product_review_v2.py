"""DQ11 source-check and review validation tests. No real-data training."""
from copy import deepcopy
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'02_examples/01_scripts'))
import review_security_product_v2 as review


class SourceReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = review.preview.extension.ProductVersionSecurityPipeline()

    def fixture(self, text='Example Router (All versions < v2.3.4) is affected.'):
        g = self.pipeline.run(text, doc_id='CVE-EXAMPLE')
        data = review.preview.extraction(g)
        row = {'cve': 'CVE-EXAMPLE', 'split': 'train', 'description': text,
               'description_sha256': review.old.cache.text_digest(text),
               'before': deepcopy(data), 'after': data, 'previously_reviewed': False, 'fixed_sample': True}
        return row, g

    def packet(self, text=None):
        r, g = self.fixture() if text is None else self.fixture(text)
        return {'schema': review.SCHEMA, 'records': [review.inspect_record(r, g)]}

    def complete(self, packet):
        result = deepcopy(packet)
        r = result['records'][0]
        r.update(full_source_reviewed=True, reviewer='test reviewer', reviewed_at='2026-10-19T12:00:00+02:00',
                 notes='Checked complete source, including missed release mentions.')
        for item in r['items']:
            item.update(judgment='supported_link' if item['state'] == 'linked' else 'appropriately_unresolved')
        return result

    def test_source_roundtrip_and_blank_review(self):
        row, g = self.fixture()
        payload = review.old.json.loads(self.pipeline.export_graphson_string(g))
        r = review.inspect_record(row, review.import_graphson(payload))
        self.assertFalse(r['full_source_reviewed'])
        self.assertEqual(r['items'][0]['judgment'], 'pending')
        self.assertEqual(r['items'][0]['change'], 'unchanged')

    def test_generic_copy_disagreement_rejected(self):
        row, g = self.fixture()
        e = next(e for e in g.edges(review.EdgeType.ENTITY_REL) if e.properties.entity_rel_type == 'HAS_VERSION')
        e.properties.extra['version_operator'] = '>'
        with self.assertRaisesRegex(ValueError, 'Generic/typed'):
            review.inspect_record(row, g)

    def test_invalid_quote_even_if_comparison_agrees(self):
        row, g = self.fixture()
        for e in g.edges():
            if e.properties.entity_rel_type == 'HAS_VERSION':
                e.properties.extra['qualifier_text'] = 'invented'
        row['after'] = review.preview.extraction(g)
        with self.assertRaisesRegex(ValueError, 'Evidence quote differs'):
            review.inspect_record(row, g)

    def test_test_split_rejected(self):
        row, g = self.fixture()
        row['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'Test inspection'):
            review.inspect_record(row, g)

    def test_unresolved_partition_and_no_auto_negative(self):
        packet = self.packet('Example 1.2.x is affected.')
        self.assertEqual(packet['records'][0]['items'][0]['state'], 'unresolved')
        stats = review.summary(packet['records'])
        self.assertEqual(stats['states'], {'unresolved': 1})
        self.assertIsNone(stats['extraction_accuracy'])
        result = review.validate_reviews(packet, packet)
        self.assertEqual(result['pending_cves'], 1)
        self.assertEqual(result['judgment_counts'], {})

    def test_source_evidence_edits_and_omissions_rejected(self):
        packet = self.packet()
        for key in ('description', 'description_sha256', 'split'):
            copy = self.complete(packet)
            copy['records'][0][key] = 'modified'
            with self.assertRaises(ValueError):
                review.validate_reviews(packet, copy)
        copy = self.complete(packet)
        copy['records'][0]['items'] = []
        with self.assertRaisesRegex(ValueError, 'Missing review items'):
            review.validate_reviews(packet, copy)

    def test_completed_review_requires_identity_time_and_all_judgments(self):
        packet = self.packet()
        for field, value in [('reviewer', ''), ('reviewed_at', '2026-10-19T12:00:00')]:
            copy = self.complete(packet)
            copy['records'][0][field] = value
            with self.assertRaises(ValueError):
                review.validate_reviews(packet, copy)
        copy = self.complete(packet)
        copy['records'][0]['items'][0]['judgment'] = 'pending'
        with self.assertRaisesRegex(ValueError, 'pending items'):
            review.validate_reviews(packet, copy)

    def test_wrong_and_uncertain_need_explanations(self):
        packet = self.packet()
        copy = self.complete(packet)
        copy['records'][0]['items'][0]['judgment'] = 'incorrect_link'
        with self.assertRaisesRegex(ValueError, 'Explain the review concern'):
            review.validate_reviews(packet, copy)
        copy['records'][0]['items'][0]['notes'] = 'The product is not the one named in the source.'
        result = review.validate_reviews(packet, copy)
        self.assertEqual(result['judgment_counts'], {'incorrect_link': 1})
        self.assertFalse(result['full_expansion_authorized'])

    def test_missed_mentions_require_exact_unique_spans(self):
        packet = self.packet('Example release 2024 is affected.')
        copy = self.complete(packet)
        text = copy['records'][0]['description']
        start = text.index('2024')
        mention = {'span': [start, start+4], 'text': '2024', 'notes': 'Integer-only release was missed.'}
        copy['records'][0]['missed_mentions'] = [mention]
        self.assertEqual(review.validate_reviews(packet, copy)['reported_missed_mentions'], 1)
        copy['records'][0]['missed_mentions'].append(mention)
        with self.assertRaises(ValueError):
            review.validate_reviews(packet, copy)
        copy['records'][0]['missed_mentions'] = [{**mention, 'text': '2025'}]
        with self.assertRaisesRegex(ValueError, 'Missed quote differs'):
            review.validate_reviews(packet, copy)

    def test_draft_not_promoted_and_empty_sources_need_note(self):
        packet = self.packet()
        draft = deepcopy(packet)
        draft['records'][0]['items'][0]['judgment'] = 'supported_link'
        with self.assertRaisesRegex(ValueError, 'Incomplete reviews'):
            review.validate_reviews(packet, draft)
        empty = self.packet('A vulnerability was reported.')
        copy = self.complete(empty)
        copy['records'][0]['notes'] = ''
        with self.assertRaisesRegex(ValueError, 'Explain reviewed empty'):
            review.validate_reviews(empty, copy)

    def test_packet_lifecycle_hashes_and_original_preservation(self):
        row, graph = self.fixture()
        old = review.old
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'inputs'/'v2_smoke'
            (source/'graphs').mkdir(parents=True)
            self.pipeline.export_graphson(graph, str(source/'graphs'/f"{row['cve']}.json"))
            manifest = {'scale': 'smoke', 'inspected_ids': [row['cve']], 'pinned_files': {},
                        'statistics': review.preview.metrics([row]), 'training': False, 'test_scoring': False,
                        'test_graphs': False, 'embeddings_generated': False, 'model_ready_cache': False}
            old.write_json(source/'analysis.json', manifest)
            old.write_json(source/'comparisons.json', [row])
            old.write_json(source/'preview_lock.json', {})
            before = {str(p): old.sha256(p) for p in source.rglob('*') if p.is_file()}
            # Separate destination from the protected source parent.
            output = root/'outputs'/'packet'
            with patch.object(review, 'SOURCE', source), patch.object(review.preview, 'run'), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                review.prepare(output, check_only=True)
                self.assertFalse(output.exists())
                review.prepare(output)
                packet = review.check_packet(output)
                self.assertEqual(review.validate_reviews(packet, packet)['pending_cves'], 1)
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    review.prepare(output)
                (output/'analysis.txt').write_text('tampered', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Packet artifact changed'):
                    review.check_packet(output)
            self.assertEqual(before, {str(p): old.sha256(p) for p in source.rglob('*') if p.is_file()})


if __name__ == '__main__':
    unittest.main()
