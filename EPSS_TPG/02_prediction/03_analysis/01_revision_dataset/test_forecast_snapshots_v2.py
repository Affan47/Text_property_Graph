"""Synthetic legacy/modern archives; no real downloads or training."""
from contextlib import redirect_stdout
from datetime import date
import gzip
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import verify_forecast_snapshots_v2 as verify
from epss.forecast_data import read_json, sha256, write_json


class VersionedSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = verify.v1.collector_module()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root/'old_cache'
        self.cache.mkdir()
        self.output = self.root/'new_run'
        self.caches = [self.output/'downloaded_snapshots', self.cache]
        self.cve = 'CVE-2021-1000'

    def archive(self, stamp='2021-04-14', header='', body=None):
        body = body or f'cve,epss\n{self.cve},0.1\n'
        archive = self.cache/f'{stamp}.csv.gz'
        archive.write_bytes(gzip.compress((header+body).encode()))
        manifest = self.cache/f'{stamp}.json'
        write_json(manifest, {'requested_date': stamp,
            'url': self.module.CONFIG['EPSS_SNAPSHOT_URL'].format(date=stamp),
            'http_status': 200, 'sha256': sha256(archive), 'archive_rows': 1})
        return archive, manifest

    def row(self, stamp, score=.1):
        return {'cve': self.cve, 'split': 'train', 'role': 'input',
            'day_since_publication': 0, 'score_date': stamp, 'stored_score': score}

    def identity(self, retry_dates=()):
        source = self.root/'source.json'
        if not source.exists():
            write_json(source, {'unchanged': True})
        return {'schema': verify.SCHEMA, 'retry_dates': list(retry_dates),
            'pinned_files': {str(source): sha256(source)}}

    def test_legacy_accepted_with_weaker_date_evidence(self):
        self.archive()
        result = verify.cached_snapshot(self.module, self.cache, date(2021, 4, 14))
        self.assertEqual(result.status, 'VALID')
        self.assertEqual(result.metadata['date_evidence'], 'legacy_requested_url_and_manifest_only')
        self.assertIsNone(result.metadata['score_date'])
        self.assertEqual(result.scores[self.cve], .1)

    def test_headerless_modern_archive_rejected(self):
        for stamp in ('2022-02-04', '2024-01-01'):
            self.archive(stamp)
            self.assertEqual(verify.cached_snapshot(self.module, self.cache, date.fromisoformat(stamp)).status, 'INVALID_CACHE')

    def test_modern_date_model_and_numeric_validation(self):
        stamp = '2024-01-01'
        header = '#model_version:v2023.03.01,score_date:2024-01-01\n'
        self.archive(stamp, header)
        result = verify.cached_snapshot(self.module, self.cache, date.fromisoformat(stamp))
        self.assertEqual(result.status, 'VALID')
        self.assertEqual(result.metadata['date_evidence'], 'internal_date_header_and_url_manifest')
        for bad_header, body in [(header.replace('2024-01-01', '2024-01-02'), None),
                                 (header.replace('v2023.03.01', 'invalid'), None),
                                 (header, f'cve,epss\n{self.cve},NaN\n'),
                                 (header, f'cve,epss\n{self.cve},1.1\n'),
                                 (header, f'cve,epss\n{self.cve},.1\n{self.cve},.2\n')]:
            with self.subTest(header=bad_header, body=body):
                self.archive(stamp, bad_header, body)
                self.assertEqual(verify.cached_snapshot(self.module, self.cache, date.fromisoformat(stamp)).status, 'INVALID_CACHE')

    def test_legacy_manifest_tampering_rejected(self):
        for key, value in [('http_status', 403), ('requested_date', '2021-04-15'),
                           ('url', 'https://example.invalid'), ('sha256', 'changed'), ('archive_rows', 2)]:
            with self.subTest(key=key):
                _, manifest = self.archive()
                meta = read_json(manifest)
                meta[key] = value
                write_json(manifest, meta)
                self.assertEqual(verify.cached_snapshot(self.module, self.cache, date(2021, 4, 14)).status, 'INVALID_CACHE')

    def test_pre_start_never_reads_cache_or_downloads(self):
        with patch.object(verify, 'cached_snapshot', side_effect=AssertionError('Cache should be skipped')):
            with patch.object(self.module, 'SnapshotClient', side_effect=AssertionError('No network')):
                result = verify.obtain_snapshot(self.module, self.caches, self.output, '2021-04-13', ['2021-04-13'], True)
        self.assertEqual(result.status, 'BEFORE_EPSS')
        entry = {'status': result.status, 'scores': {}, 'metadata': result.metadata}
        self.assertEqual(verify.comparison(self.row('2021-04-13', None), entry)['comparison'], 'BEFORE_EPSS')
        self.assertEqual(verify.comparison(self.row('2021-04-13', 0), entry)['comparison'], 'STORED_SCORE_BEFORE_EPSS')

    def test_download_requires_optin_and_allowlisted_date(self):
        for enabled, dates in [(False, ['2021-04-22']), (True, [])]:
            with patch.object(self.module, 'SnapshotClient', side_effect=AssertionError('No network')):
                result = verify.obtain_snapshot(self.module, self.caches, self.output, '2021-04-22', dates, enabled)
            self.assertEqual(result.status, 'UNRESOLVED_SNAPSHOT')
        self.assertFalse(self.output.exists())

    def test_download_writes_only_new_cache_and_failures_stay_unresolved(self):
        client = Mock()
        client.fetch.return_value = self.module.Snapshot('DOWNLOAD_ERROR', {}, {'reason': 'HTTP 403'})
        with patch.object(self.module, 'SnapshotClient', return_value=client) as factory:
            result = verify.obtain_snapshot(self.module, self.caches, self.output, '2021-04-22', ['2021-04-22'], True)
        factory.assert_called_once_with(self.output/'downloaded_snapshots')
        client.fetch.assert_called_once_with(date(2021, 4, 22), refresh=True)
        client.close.assert_called_once_with()
        self.assertEqual(result.status, 'DOWNLOAD_ERROR')
        row = verify.comparison(self.row('2021-04-22', None), {'status': result.status, 'scores': {}, 'metadata': result.metadata})
        self.assertEqual(row['comparison'], 'UNRESOLVED_SNAPSHOT')

    def test_reuse_legacy_cache_even_with_download_enabled(self):
        self.archive()
        with patch.object(self.module, 'SnapshotClient', side_effect=AssertionError('No redundant download')):
            result = verify.obtain_snapshot(self.module, self.caches, self.output, '2021-04-14', ['2021-04-14'], True)
        self.assertEqual(result.status, 'VALID')

    def test_recovery_candidate_zero_is_preserved(self):
        row = verify.comparison(self.row('2021-04-14', None), {'status': 'VALID',
            'scores': {self.cve: 0}, 'metadata': {'date_evidence': 'legacy_requested_url_and_manifest_only'}})
        self.assertEqual(row['comparison'], 'MISSING_STORED_AVAILABLE')
        self.assertEqual(row['snapshot_score'], 0)

    def test_end_to_end_outputs_resume_and_source_unchanged(self):
        self.archive()
        identity = self.identity(['2021-04-22'])
        plan = {d: [self.row(d, value)] for d, value in [('2021-04-13', None), ('2021-04-14', .1), ('2021-04-22', None)]}
        before = {str(p): sha256(p) for p in self.cache.iterdir()}
        with redirect_stdout(io.StringIO()):
            verify.run(plan, identity, self.module, self.caches, self.output)
        summary = read_json(self.output/'analysis.json')
        self.assertEqual(summary['comparison_counts'], {'BEFORE_EPSS': 1, 'MATCH': 1, 'UNRESOLVED_SNAPSHOT': 1})
        self.assertFalse(summary['all_in_range_dates_resolved'])
        self.assertEqual(read_json(self.output/'recovery_candidates.json'), [])
        self.assertIn('date_evidence', (self.output/'recovery_candidates.csv').read_text())
        self.assertEqual(before, {str(p): sha256(p) for p in self.cache.iterdir()})
        self.archive('2021-04-22')
        with redirect_stdout(io.StringIO()):
            verify.run(plan, identity, self.module, self.caches, self.output, resume=True)
        summary = read_json(self.output/'analysis.json')
        self.assertTrue(summary['all_in_range_dates_resolved'])
        self.assertEqual(summary['possible_recovery_cve_days'], 1)
        for name, digest in read_json(self.output/'verification_lock.json')['files'].items():
            self.assertEqual(sha256(self.output/name), digest)
        verify.check_identity(identity)

    def test_resume_rejects_modified_archive_checkpoint_or_source(self):
        for kind in ('archive', 'checkpoint', 'source'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                output = Path(temp)/'out'
                archive, _ = self.archive()
                identity = self.identity()
                with redirect_stdout(io.StringIO()):
                    verify.run({'2021-04-14': [self.row('2021-04-14')]}, identity, self.module, [self.cache], output)
                target = {'archive': archive, 'checkpoint': output/'dates/2021-04-14.json',
                          'source': self.root/'source.json'}[kind]
                target.write_text('tampered')
                with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):
                    verify.run({'2021-04-14': [self.row('2021-04-14')]}, identity, self.module, [self.cache], output, resume=True)


if __name__ == '__main__':
    unittest.main()
