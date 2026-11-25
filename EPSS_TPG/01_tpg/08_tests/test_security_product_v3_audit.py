"""V3 preview lifecycle, provenance and narrowly scoped change reporting."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'02_examples/01_scripts'))
import audit_security_product_versions_v3 as audit


class V3PreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = audit.previous.extension.ProductVersionSecurityPipeline()
        cls.candidate = audit.extension.ProductVersionSecurityPipeline()

    def test_change_detection_keeps_bounds_and_abstentions(self):
        text = 'F5 BIG-IP 16.1.x is affected. Example Router before 2.3 is affected.'
        left = audit.previous.extraction(self.baseline.run(text, doc_id='example'))
        right = audit.previous.extraction(self.candidate.run(text, doc_id='example'))
        changes = audit.differences(left, right)
        self.assertEqual([v['text'] for v in changes['added_versions']], ['16.1.x'])
        self.assertFalse(changes['security_relations_changed'])
        self.assertFalse(changes['changed_versions'])
        modified = deepcopy(right)
        modified['typed_security_relations'][0]['evidence']['version_operator'] = '<='
        self.assertTrue(audit.differences(left, modified)['security_relations_changed'])
        modified = deepcopy(right)
        modified['versions'][-1]['abstention'] = 'changed'
        self.assertTrue(audit.differences(left, modified)['changed_versions'])

    def test_genuine_address_remains_excluded_in_preview(self):
        text = 'F5 BIG-IP uses IP address 192.0.2.10. Example Server 16.1.2.2 is affected.'
        before = audit.previous.extraction(self.baseline.run(text, doc_id='example'))
        graph = self.candidate.run(text, doc_id='example')
        audit.validate_graph(graph, text)
        after = audit.previous.extraction(graph)
        self.assertFalse(any(audit.differences(before, after).values()))
        self.assertEqual([v['text'] for v in after['versions']], ['16.1.2.2'])

    def test_invalid_evidence_rejected(self):
        text = 'Example Router before 2.3 is affected.'
        graph = self.candidate.run(text, doc_id='example')
        graph.edges(audit.SecurityEdgeType.HAS_VERSION)[0].properties.extra['qualifier_text'] = 'invented'
        with self.assertRaises(ValueError):
            audit.validate_graph(graph, text)

    def test_unexpected_changes_and_missing_recovery_need_review(self):
        text = 'In F5 BIG-IP 16.1.x a problem exists.'
        before = audit.previous.extraction(self.baseline.run(text, doc_id='CVE-2022-26130'))
        after = audit.previous.extraction(self.candidate.run(text, doc_id='CVE-2022-26130'))
        row = {'cve': 'CVE-2022-26130', 'split': 'train', 'before': before, 'after': after,
               'differences': audit.differences(before, after)}
        self.assertEqual(audit.statistics([row])['review_status'], 'expected_change_only')
        altered = deepcopy(row)
        altered['differences']['edge_count_change'] = 100
        self.assertEqual(audit.statistics([altered])['unexpected_change_cves'], ['CVE-2022-26130'])
        row['cve'] = 'other'
        self.assertEqual(audit.statistics([row])['unexpected_change_cves'], ['other'])
        self.assertEqual(audit.statistics([])['review_status'], 'requires_investigation')

    def test_test_split_and_duplicate_ids_rejected(self):
        manifest = {'ids': {'train': ['a'], 'validation': ['b']}}
        for ids, rows in [(['a', 'a'], [{'cve': 'a', 'split': 'train'}]*2),
                          (['b'], [{'cve': 'b', 'split': 'train'}]),
                          (['x'], [{'cve': 'x', 'split': 'test'}])]:
            with self.assertRaises(ValueError):
                audit.previous.selected_rows({'inspected_ids': ids}, rows, manifest)

    def test_preview_lifecycle_and_reconstruction_guard(self):
        old = audit.old
        baseline = audit.previous.extension.ProductVersionSecurityPipeline()
        resource = hashlib.sha256(baseline.frontend.nlp.to_bytes()).hexdigest()
        ids = {'train': ['CVE-EXAMPLE-A'], 'validation': ['CVE-EXAMPLE-B']}
        raw = [{'cve': 'CVE-EXAMPLE-A', 'description': 'F5 BIG-IP 16.1.x is affected.'},
               {'cve': 'CVE-EXAMPLE-B', 'description': 'Example Server 4.6 is affected. IP address 192.0.2.1.'}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, packet = root/'v2/smoke', root/'packet'
            (source/'graphs').mkdir(parents=True); packet.mkdir()
            raw_path = root/'raw.json'; old.write_json(raw_path, raw)
            rows = []
            for record, split in zip(raw, ('train', 'validation')):
                cve, text = record['cve'], record['description']
                graph = baseline.run(text, doc_id=cve)
                for node in graph.nodes(): node.properties.extra['source_text_type'] = 'description'
                for edge in graph.edges(): edge.properties.extra['source_text_type'] = 'description'
                baseline.export_graphson(graph, str(source/'graphs'/f'{cve}.json'))
                rows.append({'cve': cve, 'split': split, 'description': text,
                             'description_sha256': old.cache.text_digest(text), 'fixed_sample': True,
                             'previously_reviewed': False, 'after': audit.previous.extraction(graph)})
            source_summary = {'pinned_files': {str(raw_path): old.sha256(raw_path)},
                              'inspected_ids': [r['cve'] for r in rows], 'fixed_sample_ids': ids,
                              'previously_reviewed_ids': []}
            old.write_json(source/'analysis.json', source_summary)
            old.write_json(source/'comparisons.json', rows)
            old.write_json(source/'preview_lock.json', {})
            old.write_json(packet/'review_lock.json', {})
            original = {str(p): old.sha256(p) for folder in (source, packet)
                        for p in folder.rglob('*') if p.is_file()}
            manifest = {'ids': ids, 'resources': {'spacy_pipeline_sha256': resource}}
            data = {'preparation_summary.json': {'source_path': str(raw_path)}}
            output = root/'output'
            with patch.object(audit, 'SOURCE', source), patch.object(audit, 'PACKET', packet), \
                 patch.object(audit, 'verify_sources') as verify, \
                 patch.object(old.cache, 'load_cache', return_value=(manifest, {}, data)), \
                 patch.object(old.frozen.task, 'fit', side_effect=AssertionError('Training forbidden')), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                audit.run(output, check_only=True)
                self.assertFalse(output.exists())
                audit.run(output)
                self.assertGreaterEqual(verify.call_count, 3)
                audit.run(output, verify_only=True)
                result = old.read_json(output/'analysis.json')
                self.assertEqual(result['statistics']['after']['version_mentions'], 2)
                self.assertEqual(result['statistics']['after']['typed_version_links'], 1)
                self.assertFalse(result['training'])
                self.assertFalse(result['test_graphs'])
                self.assertFalse(result['full_expansion_authorized'])
                self.assertEqual(len(list((output/'graphs').glob('*.json'))), 2)
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    audit.run(output)
                with self.assertRaises(ValueError):
                    audit.run(source/'child', check_only=True)
                self.assertEqual(original, {str(p): old.sha256(p) for folder in (source, packet)
                                           for p in folder.rglob('*') if p.is_file()})
                (output/'analysis.txt').write_text('tampered', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Changed v3 artifact'):
                    audit.run(output, verify_only=True)
                old.write_json(source/'graphs'/'CVE-EXAMPLE-A.json', {'tampered': True})
                failed = root/'failed'
                with self.assertRaisesRegex(ValueError, 'V2 reconstruction differs'):
                    audit.run(failed)
                self.assertEqual(old.read_json(failed/'run_status.json')['status'], 'failed')
                self.assertFalse((failed/'preview_lock.json').exists())
                self.assertEqual(old.sha256(raw_path), source_summary['pinned_files'][str(raw_path)])


if __name__ == '__main__':
    unittest.main()
