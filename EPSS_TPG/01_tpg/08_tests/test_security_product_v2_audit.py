"""V2 preview lifecycle and train/validation-only guard tests."""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'02_examples/01_scripts'))
import audit_security_product_versions_v2 as audit


class V2PreviewTests(unittest.TestCase):
    def test_wrong_split_and_duplicates_rejected(self):
        m = {'ids': {'train': ['a'], 'validation': ['b']}}
        for ids, rows in [(['a', 'a'], [{'cve': 'a', 'split': 'train'}]*2),
                          (['b'], [{'cve': 'b', 'split': 'train'}]),
                          (['x'], [{'cve': 'x', 'split': 'test'}])]:
            with self.assertRaises(ValueError):
                audit.selected_rows({'inspected_ids': ids}, rows, m)

    def test_preview_and_verify_without_source_mutation(self):
        old = audit.old
        pipeline = old.extension.ProductVersionSecurityPipeline()
        resource = hashlib.sha256(pipeline.frontend.nlp.to_bytes()).hexdigest()
        ids = {'train': ['CVE-EXAMPLE-A'], 'validation': ['CVE-EXAMPLE-B']}
        raw = [{'cve': 'CVE-EXAMPLE-A', 'description': 'Example Router (All versions < v2.3.4) is affected.'},
               {'cve': 'CVE-EXAMPLE-B', 'description': 'Example Server 4.6 Patch 518 and before is affected.'}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, review = root/'v1', root/'review'
            (source/'graphs').mkdir(parents=True); review.mkdir()
            raw_path = root/'raw.json'; old.write_json(raw_path, raw)
            rows = []
            for record, split in zip(raw, ('train', 'validation')):
                cve, text = record['cve'], record['description']
                graph = pipeline.run(text, doc_id=cve)
                for node in graph.nodes(): node.properties.extra['source_text_type'] = 'description'
                for edge in graph.edges(): edge.properties.extra['source_text_type'] = 'description'
                pipeline.export_graphson(graph, str(source/'graphs'/f'{cve}.json'))
                rows.append({'cve': cve, 'split': split, 'description': text,
                             'description_sha256': old.cache.text_digest(text), 'fixed_sample': True,
                             'previously_reviewed': False, 'after': old.extraction_rows(graph)})
            summary = {'pinned_files': {str(raw_path): old.sha256(raw_path)},
                       'inspected_ids': [r['cve'] for r in rows], 'fixed_sample_ids': ids,
                       'previously_reviewed_ids': []}
            old.write_json(source/'analysis.json', summary)
            old.write_json(source/'comparisons.json', rows)
            old.write_json(source/'preview_lock.json', {})
            old.write_json(review/'analysis.json', {})
            old.write_json(review/'review_lock.json', {})
            original = {str(p): old.sha256(p) for folder in (source, review) for p in folder.rglob('*') if p.is_file()}
            manifest = {'ids': ids, 'resources': {'spacy_pipeline_sha256': resource}}
            data = {'preparation_summary.json': {'source_path': str(raw_path)}}
            output = root/'v2'
            with patch.object(audit, 'SOURCE', source), patch.object(audit, 'REVIEW', review), \
                 patch.object(audit.old_review, 'run'), \
                 patch.object(old.cache, 'load_cache', return_value=(manifest, {}, data)), \
                 patch.object(old.frozen.task, 'fit', side_effect=AssertionError('Training forbidden')), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                audit.run(output, check_only=True)
                self.assertFalse(output.exists())
                audit.run(output)
                audit.run(output, verify_only=True)
                result = old.read_json(output/'analysis.json')
                self.assertFalse(result['training'])
                self.assertFalse(result['test_graphs'])
                self.assertEqual(result['statistics']['after']['typed_version_links'], 2)
                comparisons = old.read_json(output/'comparisons.json')
                self.assertEqual(comparisons[0]['after']['typed_security_relations'][0]['source'], 'Example Router')
                self.assertEqual(comparisons[1]['after']['versions'][0]['text'], '4.6 Patch 518')
                self.assertEqual(len(list((output/'graphs').glob('*.json'))), 2)
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    audit.run(output)
                (output/'analysis.txt').write_text('tampered', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Changed v2 artifact'):
                    audit.run(output, verify_only=True)
            self.assertEqual(original, {str(p): old.sha256(p) for folder in (source, review)
                                       for p in folder.rglob('*') if p.is_file()})
            self.assertEqual(old.sha256(raw_path), summary['pinned_files'][str(raw_path)])


if __name__ == '__main__':
    unittest.main()
