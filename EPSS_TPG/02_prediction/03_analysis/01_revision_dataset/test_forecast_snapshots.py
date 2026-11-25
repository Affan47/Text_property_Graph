"""Synthetic archives only: no requests to EPSS and no model fitting."""
from contextlib import redirect_stdout
from datetime import date
import gzip
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import verify_forecast_snapshots as verify
from epss.forecast_data import read_json, sha256, write_json
from test_forecast_evidence import fixture


class SnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = verify.collector_module()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root/'old_cache'
        self.cache.mkdir()
        self.day = date(2024, 1, 1)
        self.cve = 'CVE-2024-1000'

    def archive(self, body=None, header=None):
        header = '#model_version:v2023.03.01,score_date:2024-01-01T00:00:00+0000\n' if header is None else header
        body = 'cve,epss,percentile\nCVE-2024-1000,0.1,0.8\n' if body is None else body
        payload = gzip.compress((header+body).encode())
        (self.cache/'2024-01-01.csv.gz').write_bytes(payload)
        write_json(self.cache/'2024-01-01.json', {'requested_date': '2024-01-01',
            'url': self.module.CONFIG['EPSS_SNAPSHOT_URL'].format(date='2024-01-01'),
            'http_status': 200, 'sha256': hashlib.sha256(payload).hexdigest()})

    def row(self, value=.1):
        return {'cve': self.cve, 'split': 'validation', 'day_since_publication': 60,
                'score_date': '2024-01-01', 'role': 'endpoint', 'stored_score': value}

    def test_valid_archive_and_read_only_cache(self):
        self.archive()
        before = {p.name: sha256(p) for p in self.cache.iterdir()}
        result = verify.cached_snapshot(self.module, self.cache, self.day)
        self.assertEqual(result.status, 'VALID')
        self.assertEqual(result.scores[self.cve], .1)
        self.assertEqual(before, {p.name: sha256(p) for p in self.cache.iterdir()})

    def test_offline_cannot_download(self):
        with patch.object(self.module, 'SnapshotClient', side_effect=AssertionError('Network forbidden')):
            result = verify.obtain_snapshot(self.module, self.cache, self.root/'output', self.day, False)
        self.assertEqual(result.status, 'CACHE_MISSING')
        self.assertFalse((self.root/'output').exists())

    def test_hash_tampering(self):
        self.archive()
        (self.cache/'2024-01-01.csv.gz').write_bytes(b'changed')
        self.assertEqual(verify.cached_snapshot(self.module, self.cache, self.day).status, 'INVALID_CACHE')

    def test_optin_downloader_does_not_write_old_cache(self):
        client = Mock()
        client.fetch.return_value = self.module.Snapshot('DOWNLOAD_ERROR', {}, {'reason': 'synthetic failure'})
        output = self.root/'out'
        with patch.object(self.module, 'SnapshotClient', return_value=client) as factory:
            result = verify.obtain_snapshot(self.module, self.cache, output, self.day, True)
        factory.assert_called_once_with(output/'downloaded_snapshots')
        client.close.assert_called_once_with()
        self.assertEqual(result.status, 'DOWNLOAD_ERROR')
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_wrong_date_header(self):
        self.archive(header='#model_version:v2023.03.01,score_date:2024-01-02\n')
        self.assertEqual(verify.cached_snapshot(self.module, self.cache, self.day).status, 'INVALID_CACHE')

    def test_wrong_model_header(self):
        self.archive(header='#model_version:unexpected,score_date:2024-01-01\n')
        self.assertEqual(verify.cached_snapshot(self.module, self.cache, self.day).status, 'INVALID_CACHE')

    def test_missing_header(self):
        self.archive(header='')
        self.assertEqual(verify.cached_snapshot(self.module, self.cache, self.day).status, 'INVALID_CACHE')

    def test_invalid_numeric_and_duplicate_cves(self):
        for body in ('cve,epss\nCVE-2024-1000,nan\n', 'cve,epss\nCVE-2024-1000,1.1\n',
                     'cve,epss\nCVE-2024-1000,.1\nCVE-2024-1000,.2\n'):
            with self.subTest(body=body):
                self.archive(body=body)
                self.assertEqual(verify.cached_snapshot(self.module, self.cache, self.day).status, 'INVALID_CACHE')

    def test_bad_manifest(self):
        for key, value in [('requested_date', '2024-01-02'), ('http_status', 404), ('url', 'https://example.invalid')]:
            with self.subTest(key=key):
                self.archive()
                path = self.cache/'2024-01-01.json'
                meta = read_json(path)
                meta[key] = value
                write_json(path, meta)
                self.assertEqual(verify.cached_snapshot(self.module, self.cache, self.day).status, 'INVALID_CACHE')

    def test_comparison_states_and_zero(self):
        cases = [(.1, {self.cve: .1}, 'MATCH'), (.1, {self.cve: .2}, 'MISMATCH'),
                 (.1, {}, 'STORED_PRESENT_CVE_ABSENT'), (None, {}, 'BOTH_ABSENT'),
                 (None, {self.cve: 0}, 'MISSING_STORED_AVAILABLE'), (0, {self.cve: 0}, 'MATCH')]
        for original, scores, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(verify.compare_row(self.row(original), 'VALID', scores, {})['comparison'], expected)
        unresolved = verify.compare_row(self.row(.1), 'DOWNLOAD_ERROR', {}, {})
        self.assertEqual(unresolved['comparison'], 'UNRESOLVED_SNAPSHOT')
        self.assertIsNone(unresolved['snapshot_score'])

    def test_decimal_difference(self):
        row = verify.compare_row(self.row(.1), 'VALID', {self.cve: .11}, {})
        self.assertEqual(row['snapshot_minus_stored'], .01)

    def test_plan_does_not_access_test_records(self):
        _, x, y, _ = fixture()
        data = {'inputs.json': {self.cve: x}, 'labels.json': {self.cve: y},
                'task_cohorts.json': {'endpoint': {'train': [self.cve], 'validation': [], 'test': ['CVE-2025-9000']}}}
        plan = verify.build_plan(data, 'train-validation')
        self.assertEqual(len(plan), 61)
        self.assertEqual(sum(len(v) for v in plan.values()), 61)
        self.assertEqual([r['role'] for v in plan.values() for r in v].count('input'), 31)
        data['task_cohorts.json']['endpoint']['test'].append(self.cve)
        with self.assertRaises(ValueError):
            verify.build_plan(data, 'train-validation')

    def identity(self):
        source, prepared = self.root/'source.json', self.root/'prep.json'
        write_json(source, {'unchanged': True})
        write_json(prepared, {'unchanged': True})
        return {'scope': 'diagnostic', 'source_path': str(source), 'source_sha256': sha256(source),
                'preparation_path': str(prepared), 'preparation_sha256': sha256(prepared),
                'verification_script_sha256': sha256(verify.__file__), 'collector_sha256': sha256(verify.COLLECTOR)}

    def test_end_to_end_resume_and_filtered_headers(self):
        self.archive()
        identity = self.identity()
        output = self.root/'out'
        plan = {'2024-01-01': [self.row()]}
        with redirect_stdout(io.StringIO()):
            verify.run(plan, identity, self.module, self.cache, output, False, False)
            with patch.object(verify, 'obtain_snapshot', side_effect=AssertionError('Valid checkpoint should be reused')):
                verify.run(plan, identity, self.module, self.cache, output, False, True)
        summary = read_json(output/'analysis.json')
        self.assertEqual(summary['comparison_counts'], {'MATCH': 1})
        self.assertEqual(read_json(output/'state.json')['run_status'], 'complete')
        self.assertIn('stored_score', (output/'disagreements.csv').read_text())
        for name, digest in read_json(output/'verification_lock.json')['files'].items():
            self.assertEqual(sha256(output/name), digest)

    def test_resume_retries_unresolved_and_rejects_changed_identity(self):
        identity = self.identity()
        output = self.root/'out'
        plan = {'2024-01-01': [self.row()]}
        with redirect_stdout(io.StringIO()):
            verify.run(plan, identity, self.module, self.cache, output, False, False)
            self.assertFalse(read_json(output/'analysis.json')['all_dates_resolved'])
            self.archive()
            verify.run(plan, identity, self.module, self.cache, output, False, True)
        self.assertTrue(read_json(output/'analysis.json')['all_dates_resolved'])
        with self.assertRaises(ValueError):
            verify.run(plan, {**identity, 'scope': 'changed'}, self.module, self.cache, output, False, True)

    def test_checkpoint_tampering_rejected(self):
        self.archive()
        identity = self.identity()
        output = self.root/'out'
        plan = {'2024-01-01': [self.row()]}
        with redirect_stdout(io.StringIO()):
            verify.run(plan, identity, self.module, self.cache, output, False, False)
        (output/'dates/2024-01-01.json').write_text('{}')
        with self.assertRaises(ValueError):
            verify.run(plan, identity, self.module, self.cache, output, False, True)


if __name__ == '__main__':
    unittest.main()
