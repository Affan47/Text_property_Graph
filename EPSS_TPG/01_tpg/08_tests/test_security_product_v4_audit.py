"""V4 saved-graph preview lifecycle and strict off-scope change checks."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'02_examples/01_scripts'))
import audit_security_product_versions_v4 as audit


class V4PreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = audit.extension.v3.ProductVersionSecurityPipeline()

    def source_graph(self, text, cve):
        graph = self.pipeline.run(text, doc_id=cve)
        for node in graph.nodes(): node.properties.extra['source_text_type'] = 'description'
        for edge in graph.edges(): edge.properties.extra['source_text_type'] = 'description'
        return graph

    def test_off_scope_changes_rejected(self):
        graph = self.source_graph('Example Router 2.3 is affected.', 'synthetic')
        exporter = audit.GraphSONExporter()
        before = json.loads(exporter.export_string(graph))
        after = json.loads(exporter.export_string(audit.extension.filter_same_occurrence_coreference(graph)))
        audit.check_delta(before, after, graph)
        altered = deepcopy(after)
        altered['vertices'][0]['properties']['TEXT'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'Node content'):
            audit.check_delta(before, altered, graph)
        altered = deepcopy(after)
        altered['edges'].pop()
        with self.assertRaisesRegex(ValueError, 'Retained edges'):
            audit.check_delta(before, altered, graph)
        altered = deepcopy(after)
        altered['edges'][0]['properties']['EXTRA']['new_claim'] = True
        with self.assertRaisesRegex(ValueError, 'Retained edges'):
            audit.check_delta(before, altered, graph)

    def test_invalid_selection_rejected(self):
        for ids, rows in [(['a'], [{'cve': 'a', 'split': 'test'}]),
                          (['a', 'a'], [{'cve': 'a', 'split': 'train'}]*2),
                          (['b'], [{'cve': 'a', 'split': 'train'}])]:
            with self.assertRaises(ValueError):
                audit.selected_rows({'inspected_ids': ids}, rows)

    def test_saved_preview_no_parser_or_training_and_provenance(self):
        old = audit.old
        texts = ['On F5 BIG-IP 16.1.x versions prior to 16.1.2.2, a server can stop.',
                 'Example Router 2.3 is affected. Example Router 2.3 is fixed.']
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, review, output = root/'v3/smoke', root/'v3/smoke_review', root/'v4/smoke'
            (source/'graphs').mkdir(parents=True); review.mkdir()
            rows = []
            for i, (text, split) in enumerate(zip(texts, ('train', 'validation'))):
                cve = f'CVE-EXAMPLE-{i}'
                graph = self.source_graph(text, cve)
                self.pipeline.export_graphson(graph, str(source/'graphs'/f'{cve}.json'))
                after = audit.preview.previous.extraction(graph)
                rows.append({'cve': cve, 'split': split, 'description': text,
                    'description_sha256': old.cache.text_digest(text), 'fixed_sample': True,
                    'previously_reviewed': False, 'after': after,
                    'v3_links': sum(e['relation'] == 'SEC_HAS_VERSION' for e in after['typed_security_relations'])})
            ids = [r['cve'] for r in rows]
            summary = {'pinned_files': {}, 'inspected_ids': ids,
                       'fixed_sample_ids': {'train': ids[:1], 'validation': ids[1:]}, 'previously_reviewed_ids': []}
            old.write_json(source/'analysis.json', summary)
            old.write_json(source/'comparisons.json', rows)
            old.write_json(source/'preview_lock.json', {})
            old.write_json(review/'review_lock.json', {})
            originals = {str(p): old.sha256(p) for p in (root/'v3').rglob('*') if p.is_file()}
            with patch.object(audit, 'SOURCE', source), patch.object(audit, 'REVIEW', review), \
                 patch.object(audit, 'verify_sources'), \
                 patch.object(audit.extension, 'ProductVersionSecurityPipeline', side_effect=AssertionError('NLP forbidden')), \
                 patch.object(old.frozen.task, 'fit', side_effect=AssertionError('Training forbidden')), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                audit.run(output, check_only=True)
                self.assertFalse(output.exists())
                audit.run(output)
                audit.run(output, verify_only=True)
                result = old.read_json(output/'analysis.json')
                self.assertFalse(result['nlp_rerun'])
                self.assertFalse(result['training'])
                self.assertFalse(result['test_inspection'])
                stats = result['statistics']
                self.assertGreater(stats['removed_same_occurrence_edges'], 0)
                self.assertEqual(stats['same_span_coref_after'], 0)
                self.assertEqual(stats['version_mentions_before'], stats['version_mentions_after'])
                self.assertEqual(stats['product_version_links_before'], stats['product_version_links_after'])
                self.assertGreater(stats['coref_after'], 0)
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    audit.run(output)
                with self.assertRaises(ValueError):
                    audit.run(source/'child', check_only=True)
                self.assertEqual(originals, {str(p): old.sha256(p) for p in (root/'v3').rglob('*') if p.is_file()})
                (output/'analysis.txt').write_text('tampered', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Changed v4 artifact'):
                    audit.run(output, verify_only=True)
                with patch.object(audit, 'check_delta', side_effect=ValueError('Injected off-scope change')):
                    with self.assertRaisesRegex(ValueError, 'Injected'):
                        audit.run(root/'failed')
                self.assertEqual(old.read_json(root/'failed/run_status.json')['status'], 'failed')
                self.assertFalse((root/'failed/preview_lock.json').exists())


if __name__ == '__main__':
    unittest.main()
