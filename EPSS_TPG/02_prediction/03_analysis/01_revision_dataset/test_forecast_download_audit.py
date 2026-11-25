"""Distinguish archive-format policy from network transfer failures."""
from contextlib import redirect_stdout
import gzip
import hashlib
import io
import shutil
import unittest

from audit_forecast_downloads import reason_group, inspect
import test_forecast_snapshots as fixtures
import verify_forecast_snapshots as verify
from epss.forecast_data import write_json


class ReasonTests(unittest.TestCase):
    def test_pre_start(self):
        self.assertEqual(reason_group('2021-04-13', {'status': 'DOWNLOAD_ERROR'}), 'before_daily_archives_started')

    def test_legacy_header_not_network_failure(self):
        entry = {'status': 'INVALID_CACHE', 'metadata': {'reason': 'No header date; cannot certify daily identity'}}
        self.assertEqual(reason_group('2021-04-14', entry), 'legacy_file_missing_optional_header')
        self.assertEqual(reason_group('2022-02-04', entry), 'other_unresolved')

    def test_only_explicit_403_is_grouped(self):
        entry = {'status': 'DOWNLOAD_ERROR', 'metadata': {'attempt_errors': [{'message': 'HTTP 403'}]}}
        self.assertEqual(reason_group('2024-12-01', entry), 'http_403_after_archive_start')
        entry['metadata']['attempt_errors'][0]['message'] = 'HTTP 503'
        self.assertEqual(reason_group('2024-12-01', entry), 'other_unresolved')


class LegacyTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.SnapshotTests()
        self.f.setUpClass()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)

    def test_payload_can_match_without_changing_original_status(self):
        stamp = '2021-04-14'
        payload = gzip.compress(b'cve,epss\nCVE-2024-1000,0.1\n')
        (self.f.cache/f'{stamp}.csv.gz').write_bytes(payload)
        write_json(self.f.cache/f'{stamp}.json', {'requested_date': stamp,
            'url': self.f.module.CONFIG['EPSS_SNAPSHOT_URL'].format(date=stamp), 'http_status': 200,
            'sha256': hashlib.sha256(payload).hexdigest(), 'archive_rows': 1})
        output = self.f.root/'out'
        identity = {**self.f.identity(), 'collector_configuration': self.f.module.CONFIG}
        row = {**self.f.row(), 'score_date': stamp}
        with redirect_stdout(io.StringIO()):
            verify.run({stamp: [row]}, identity, self.f.module, self.f.cache, output, False, False)
        (output/'downloaded_snapshots').mkdir()
        for suffix in ('.csv.gz', '.json'):
            shutil.copyfile(self.f.cache/f'{stamp}{suffix}', output/'downloaded_snapshots'/f'{stamp}{suffix}')
        with redirect_stdout(io.StringIO()):
            result, rows = inspect(output)
        self.assertEqual(result['original_comparison_counts'], {'UNRESOLVED_SNAPSHOT': 1})
        self.assertEqual(result['legacy_payload_comparison_counts'], {'MATCH': 1})
        self.assertTrue(rows[0]['original_status_retained'])
        (output/'downloaded_snapshots'/f'{stamp}.csv.gz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            inspect(output)


if __name__ == '__main__':
    unittest.main()
