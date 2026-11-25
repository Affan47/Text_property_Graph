"""Synthetic network responses only. No live requests or production labels."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'02_prediction/01_package'))
sys.path.insert(0, str(ROOT/'02_prediction/03_analysis/06_reddit_dataset'))
from epss.catalogue_checks_v1 import runner as r
from epss.catalogue_checks_v1 import saved_export as s


class CatalogueChecksTests(unittest.TestCase):
    def protocol(self):
        return {'schema': 'ct09_catalogue_checks_v1', 'dataset': 'synthetic',
                'ids': ['CVE-2022-1000', 'CVE-2022-1001'], 'candidate_count': 100,
                'legacy_flags': {'CVE-2022-1000': True, 'CVE-2022-1001': False},
                'input_hashes': {}, 'endpoint': r.evidence.ENDPOINT, 'query_parameters': {'limit': 100, 'page': 1},
                'timeout_seconds': 30, 'minimum_interval_seconds': 0, 'max_requests_per_invocation': 20,
                'stop_http_codes': [401, 403, 429], 'stop_consecutive_transport_failures': 3}

    def response(self, code=200, positive=False, incomplete=False):
        body = {'_meta': {'index': 'vulncheck-kev', 'total_documents': int(positive)},
                'data': [{'cve': ['CVE-2022-1000']}] if positive else []}
        if incomplete:
            body['_meta'].pop('total_documents')
        value = Mock(status_code=code, content=json.dumps(body).encode())
        return value

    def fake_session(self, responses):
        session = Mock()
        session.__enter__ = Mock(return_value=session); session.__exit__ = Mock(return_value=False)
        session.get.side_effect = responses
        return session

    def test_plan_is_network_free_and_no_writes(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(r, 'load_plan', return_value=(self.protocol(), 'synthetic')):
            output = Path(tmp)/'audit'
            with patch.object(r.requests, 'Session', side_effect=AssertionError('network forbidden')):
                r.run('synthetic', 'plan', output_dir=output)
            self.assertFalse(output.exists())

    def test_collection_requires_opt_in_and_local_key(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(r, 'load_plan', return_value=(self.protocol(), 'synthetic')):
            output = Path(tmp)/'audit'
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, 'allow-network'):
                    r.run('synthetic', 'collect', output_dir=output)
                with self.assertRaisesRegex(ValueError, 'VC_API_KEY'):
                    r.run('synthetic', 'collect', allow_network=True, output_dir=output)
            self.assertFalse(output.exists())

    def test_end_to_end_response_replay_resume_and_secret_absence(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(r, 'load_plan', return_value=(self.protocol(), 'synthetic')):
            output = Path(tmp)/'audit'; session = self.fake_session([self.response(positive=True), self.response()])
            with patch.dict(os.environ, {'VC_API_KEY': 'FAKE_SECRET_NOT_REAL'}), patch.object(r.requests, 'Session', return_value=session):
                findings = r.run('synthetic', 'collect', True, output_dir=output)
                self.assertEqual(findings['pilot_outcomes'], {'LISTED': 1, 'NOT_LISTED': 1, 'UNKNOWN': 0})
                self.assertEqual(session.get.call_count, 2)
                r.run('synthetic', 'collect', True, resume=True, output_dir=output)
                self.assertEqual(session.get.call_count, 2)
            with patch.object(r.requests, 'Session', side_effect=AssertionError('offline verification')):
                r.run('synthetic', 'verify', output_dir=output)
            for path in output.rglob('*'):
                if path.is_file():
                    self.assertNotIn('FAKE_SECRET_NOT_REAL', path.read_text())
            kwargs = session.get.call_args_list[0].kwargs
            self.assertFalse(kwargs['allow_redirects'])
            self.assertEqual(kwargs['params']['cve'], 'CVE-2022-1000')
            self.assertEqual(kwargs['timeout'], 30)

    def test_authentication_failure_stops_and_never_labels_negative(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(r, 'load_plan', return_value=(self.protocol(), 'synthetic')):
            session = self.fake_session([self.response(403)])
            with patch.dict(os.environ, {'VC_API_KEY': 'FAKE'}), patch.object(r.requests, 'Session', return_value=session):
                findings = r.run('synthetic', 'collect', True, output_dir=Path(tmp)/'audit')
            self.assertEqual(session.get.call_count, 1)
            self.assertEqual(findings['pilot_outcomes']['UNKNOWN'], 2)
            self.assertEqual(findings['stop_reason'], 'HTTP_ERROR')
            self.assertEqual(findings['candidate_cves_not_verified_by_this_pilot'], 100)

    def test_incomplete_empty_response_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self.fake_session([self.response(incomplete=True)])
            receipt = r.collect_one(session, 'FAKE', 'CVE-2022-1000', self.protocol(), Path(tmp))
            self.assertIsNone(receipt['verified_in_vckev'])

    def test_timeout_does_not_write_exception_or_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self.fake_session([requests.Timeout('FAKE_SECRET_NOT_REAL')])
            receipt = r.collect_one(session, 'FAKE', 'CVE-2022-1000', self.protocol(), Path(tmp))
            self.assertEqual(receipt['status'], 'NETWORK_ERROR')
            self.assertIsNone(receipt['verified_in_vckev'])
            self.assertNotIn('FAKE_SECRET_NOT_REAL', (Path(tmp)/'000001.receipt.json').read_text())

    def test_explicit_retry_preserves_attempts(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(r, 'load_plan', return_value=(self.protocol(), 'synthetic')):
            output = Path(tmp)/'audit'
            with patch.dict(os.environ, {'VC_API_KEY': 'FAKE'}), patch.object(r.requests, 'Session',
                    return_value=self.fake_session([self.response(500), self.response()])):
                r.run('synthetic', 'collect', True, output_dir=output)
            old = output/'receipts/CVE-2022-1000/000001.receipt.json'; old_hash = r.SHA(old)
            session = self.fake_session([self.response(positive=True)])
            with patch.dict(os.environ, {'VC_API_KEY': 'FAKE'}), patch.object(r.requests, 'Session', return_value=session):
                r.run('synthetic', 'collect', True, resume=True, retry_unknown=True, output_dir=output)
            self.assertEqual(r.SHA(old), old_hash)
            self.assertTrue((old.parent/'000002.receipt.json').exists())
            self.assertEqual(session.get.call_count, 1)
            r.run('synthetic', 'verify', output_dir=output)

    def test_out_of_scope_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/'receipts/CVE-2025-9999'; directory.mkdir(parents=True)
            (directory/'000001.receipt.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'out-of-scope'):
                r.read_attempts(Path(tmp), self.protocol())

    def test_orphan_response_survives_and_next_attempt_is_new(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); (folder/'000001.response.json').write_text('partial response')
            session = self.fake_session([self.response(positive=True)])
            receipt = r.collect_one(session, 'FAKE', 'CVE-2022-1000', self.protocol(), folder)
            self.assertEqual(receipt['attempt'], 2)
            self.assertEqual((folder/'000001.response.json').read_text(), 'partial response')

    def test_tampering_cannot_be_resumed(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(r, 'load_plan', return_value=(self.protocol(), 'synthetic')):
            output = Path(tmp)/'audit'
            with patch.dict(os.environ, {'VC_API_KEY': 'FAKE'}), patch.object(r.requests, 'Session',
                    return_value=self.fake_session([self.response(positive=True), self.response()])):
                r.run('synthetic', 'collect', True, output_dir=output)
                path = output/'receipts/CVE-2022-1000/000001.response.json'; path.write_text('{}')
                with self.assertRaisesRegex(ValueError, 'Changed prior artifact'):
                    r.run('synthetic', 'collect', True, resume=True, output_dir=output)

    def test_unknown_cases_do_not_increase_disagreements(self):
        reports = r.reports(self.protocol(), {})
        self.assertEqual(reports['findings.json']['saved_flag_disagreements'], 0)
        self.assertEqual(reports['findings.json']['pilot_outcomes']['UNKNOWN'], 2)
        self.assertTrue(all(row['saved_flag_disagreement'] is None for row in reports['verified_labels.json']))

    def test_augmented_requests_remain_bounded_training_only(self):
        ids = ['CVE-2022-1000', 'CVE-2022-1001']
        self.assertEqual(r.approved_ids(ids[:1], ids, ids), ids)
        with self.assertRaisesRegex(ValueError, 'allowlist'):
            r.approved_ids(ids, ['CVE-2025-9999'], ids)
        lots = [f'CVE-2022-{1000+i}' for i in range(41)]
        with self.assertRaisesRegex(ValueError, 'allowlist'):
            r.approved_ids(lots, [], lots)

    def test_snapshot_presence_not_current_label_and_no_heldout_export(self):
        records = [{'cve': ['CVE-2022-1000'], 'date_added': '2022-01-01'},
                   {'cve': ['CVE-2025-9999'], 'shortDescription': 'DO_NOT_EXPORT'}]
        rows = [{'cve': 'CVE-2022-1000', 'saved_in_vckev': 'False', 'vckev_date_added': ''},
                {'cve': 'CVE-2022-1001', 'saved_in_vckev': 'True', 'vckev_date_added': '2026-11-01'}]
        result = s.compare(records, rows, r.audit.parse_time('2026-10-02')[0])
        self.assertEqual(result[0]['comparison'], 'SAVED_FALSE_BUT_PRESENT')
        self.assertEqual(result[1]['comparison'], 'SAVED_LISTED_BUT_ABSENT')
        self.assertTrue(result[1]['saved_addition_after_repository_commit'])
        self.assertTrue(all(row['verified_current_label'] is None and row['released_training_label'] is None for row in result))
        self.assertNotIn('CVE-2025-9999', json.dumps(result))
        self.assertNotIn('DO_NOT_EXPORT', json.dumps(result))

    def test_absence_is_not_verified_negative(self):
        rows = [{'cve': 'CVE-2022-1000', 'saved_in_vckev': 'False', 'vckev_date_added': ''}]
        result = s.compare([], rows, r.audit.parse_time('2026-10-02')[0])
        self.assertEqual(result[0]['comparison'], 'SAVED_FALSE_AND_ABSENT_UNVERIFIED')
        self.assertIsNone(result[0]['verified_current_label'])

    def test_duplicate_snapshot_matches_are_flagged(self):
        record = {'cve': ['CVE-2022-1000'], 'date_added': '2022-01-01'}
        row = {'cve': 'CVE-2022-1000', 'saved_in_vckev': 'True', 'vckev_date_added': '2022-01-01'}
        result = s.compare([record, record], [row], r.audit.parse_time('2026-10-02')[0])
        self.assertEqual(result[0]['matching_export_records'], 2)
        self.assertFalse(result[0]['saved_listed_addition_matches'])


if __name__ == '__main__':
    unittest.main()
