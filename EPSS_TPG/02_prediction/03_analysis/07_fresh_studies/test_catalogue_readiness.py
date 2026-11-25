"""Offline regression tests; synthetic records only, no models or requests."""
import copy
from datetime import timedelta
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'02_prediction/01_package'))
sys.path.insert(0, str(ROOT/'02_prediction/03_analysis/06_reddit_dataset'))
from epss.catalogue_readiness_v1 import audit as a


class CatalogueReadinessTests(unittest.TestCase):
    def stamp(self, value):
        return a.parse_time(value)[0]

    def record(self):
        return {'cve': 'CVE-2022-1234', 'in_vckev': True,
                'published_timestamp': '2022-02-01T14:00:00Z', 'published_date': '2022-02-01',
                'vckev_date_added': '2022-02-01T00:00:00Z', 'description': 'A buffer overflow.'}

    def test_deterministic_dates_and_timezone(self):
        self.assertEqual(self.stamp('2022-02-01T15:00:00+01:00'), self.stamp('2022-02-01T14:00:00Z'))
        self.assertEqual(a.parse_time('2022-02-01T14:00:00')[1], 'naive_assumed_UTC')
        self.assertEqual(a.parse_time('2022-02-01')[1], 'date_only')
        for value in ('01/02/2022', '2022-02-30', 'yesterday'):
            self.assertEqual(a.parse_time(value), (None, 'invalid'))

    def test_negative_time_not_converted_to_zero(self):
        info, _, _, _ = a.timing(self.record())
        self.assertAlmostEqual(info['delta_days_exact_stored'], -14/24)
        self.assertEqual(info['timing_category'], 'EARLIER_TIME_SAME_DAY')
        self.assertFalse(info['stored_nonnegative_within_60'])
        self.assertTrue(info['date_only_within_60'])

    def test_earlier_day_and_inconsistent_publication(self):
        r = self.record(); r['vckev_date_added'] = '2022-01-30'
        r['published_date'] = '2022-01-31'
        info, _, _, _ = a.timing(r)
        self.assertEqual(info['timing_category'], 'EARLIER_DAY')
        self.assertTrue(info['publication_date_timestamp_disagree'])

    def test_missing_date_is_explicit(self):
        r = self.record(); r['vckev_date_added'] = None
        info, pub, lo, hi = a.timing(r)
        self.assertIsNone(info['delta_days_exact_stored'])
        self.assertEqual(a.interval_relation(lo, hi, pub, 30), 'MISSING_OR_INVALID_TIME')

    def test_midnight_interval_is_ambiguous_at_publication(self):
        _, pub, lo, hi = a.timing(self.record())
        self.assertEqual(hi-lo, timedelta(days=1))
        self.assertEqual(a.interval_relation(lo, hi, pub, 30), 'BOUNDARY_OR_EVENT_ORDER_UNCERTAIN')

    def test_exact_horizon_boundaries(self):
        anchor = self.stamp('2022-02-01T14:00:00Z'); end = anchor+timedelta(days=30)
        self.assertEqual(a.interval_relation(anchor, anchor, anchor, 30), 'ALREADY_ADDED_AT_OR_BEFORE_ANCHOR')
        self.assertEqual(a.interval_relation(end, end, anchor, 30), 'WITHIN_HORIZON_STORED_TIME_SCENARIO')
        self.assertEqual(a.interval_relation(end+timedelta(seconds=1), end+timedelta(seconds=1), anchor, 30), 'AFTER_HORIZON')
        self.assertEqual(a.interval_relation(end-timedelta(hours=1), end+timedelta(hours=1), anchor, 30), 'BOUNDARY_OR_EVENT_ORDER_UNCERTAIN')

    def test_string_true_not_positive(self):
        r = self.record(); r['in_vckev'] = 'true'
        self.assertFalse(a.timing(r)[0]['stored_nonnegative_within_60'])

    def test_failed_lookup_is_never_negative(self):
        for status in (401, 403, 429, 500):
            self.assertIsNone(a.receipts.classify_response(status, {})[0])
        incomplete = {'data': [], '_meta': {'index': 'vulncheck-kev'}}
        self.assertIsNone(a.receipts.classify_response(200, incomplete)[0])

    def test_complete_empty_and_wrong_cve(self):
        empty = {'data': [], '_meta': {'index': 'vulncheck-kev', 'total_documents': 0}}
        self.assertIs(a.receipts.classify_response(200, empty)[0], False)
        empty['_meta']['truncated'] = True
        self.assertIsNone(a.receipts.classify_response(200, empty)[0])
        positive = {'data': [{'cve': ['CVE-2022-1234']}], '_meta': {'index': 'vulncheck-kev'}}
        self.assertIs(a.receipts.classify_cve('CVE-2022-1234', 200, positive)[0], True)
        self.assertIsNone(a.receipts.classify_cve('CVE-2022-9999', 200, positive)[0])

    def receipt(self):
        return {'cve': 'CVE-2022-1234', 'attempt': 1, 'legacy_in_vckev': True,
                'verified_in_vckev': None, 'status': 'NETWORK_ERROR', 'http_status': None,
                'request_started_at_utc': '2026-11-01T00:00:00+00:00',
                'checked_at_utc': '2026-11-01T00:00:01+00:00',
                'label_date_basis': 'fresh_lookup_not_original_snapshot'}

    def test_receipt_failure_timestamp_and_relabel_guards(self):
        r = self.receipt()
        self.assertEqual(a.receipt_state(r, r['cve'], Path('/tmp'))[0], 'UNKNOWN')
        r['verified_in_vckev'] = False
        with self.assertRaises(ValueError):
            a.receipt_state(r, r['cve'], Path('/tmp'))
        r = self.receipt(); r['checked_at_utc'] = '2026-10-01T00:00:00Z'
        with self.assertRaises(ValueError):
            a.receipt_state(r, r['cve'], Path('/tmp'))

    def test_receipt_hash_and_scope_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); response = directory/'response.json'
            response.write_text(json.dumps({'data': [], '_meta': {'index': 'vulncheck-kev', 'total_documents': 0}}))
            r = self.receipt(); r.update(verified_in_vckev=False, status='SUCCESS_NOT_LISTED', http_status=200,
                                       response_file=response.name, response_sha256=a.SHA(response))
            self.assertEqual(a.receipt_state(r, r['cve'], directory)[0], 'NOT_LISTED')
            with self.assertRaises(ValueError):
                a.receipt_state(r, 'CVE-2022-9999', directory)
            response.write_text('{}')
            with self.assertRaises(ValueError):
                a.receipt_state(r, r['cve'], directory)

    def test_latest_failed_receipt_does_not_reuse_old_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); r = self.receipt(); cve = r['cve']
            directory = folder/'receipts'/cve; directory.mkdir(parents=True)
            response = directory/'response.json'
            response.write_text(json.dumps({'data': [], '_meta': {'index': 'vulncheck-kev', 'total_documents': 0}}))
            old = dict(r, verified_in_vckev=False, status='SUCCESS_NOT_LISTED', http_status=200,
                       response_file=response.name, response_sha256=a.SHA(response))
            (directory/'000001.receipt.json').write_text(json.dumps(old))
            r.update(attempt=2, checked_at_utc='2026-11-01T00:00:02Z')
            (directory/'000002.receipt.json').write_text(json.dumps(r))
            (folder/'protocol.json').write_text(json.dumps({'schema': 'reddit_kev_recheck_v1',
                'endpoint': a.receipts.ENDPOINT, 'source_sha256': 'source', 'ids': [cve, 'CVE-2025-9999'],
                'script_sha256': a.SHA(Path(a.receipts.__file__))}))
            latest, _, inventory = a.audit_receipts(folder, {cve: self.record()}, 'source')
            self.assertEqual(set(latest), {cve})
            self.assertEqual(latest[cve]['verified_state_at_check'], 'UNKNOWN')
            self.assertFalse(inventory['other_roles_receipts_opened'])
            with self.assertRaises(ValueError):
                a.audit_receipts(folder, {cve: self.record()}, 'different-source')

    def test_component_roles_without_epss_requirement(self):
        intervals = {'train': ['2021-01-01', '2023-12-31'], 'validation': ['2024-01-01', '2024-12-31']}
        pubs = {'a': '2022-01-01', 'b': '2022-01-01', 'c': '2024-01-01', 'd': '2022-01-01', 'e': '2022-01-01'}
        groups = [dict(current_cves=['a'], prior_roles=[], connected_to_prior_test=False),
                  dict(current_cves=['b', 'c'], prior_roles=[], connected_to_prior_test=False),
                  dict(current_cves=['d'], prior_roles=['train'], connected_to_prior_test=True),
                  dict(current_cves=['e'], prior_roles=['validation'], connected_to_prior_test=False)]
        chosen, _ = a.candidate_train_ids(groups, pubs, intervals)
        self.assertEqual(chosen, {'a'})
        with self.assertRaises(ValueError):
            a.candidate_train_ids(groups+groups[:1], pubs, intervals)

    def test_post_date_is_not_historical_wording_proof(self):
        r = self.record(); post = {'post': 'The software crashes.', 'date': '2022-02-02', 'platform': 'Reddit'}
        r['social_media_posts'] = [post, dict(post)]
        docs = a.screen_documents(r, self.stamp(r['published_timestamp']), True)
        self.assertEqual(len(docs), 2)
        self.assertTrue(docs[-1]['date_screen_by_day30'])
        self.assertFalse(docs[-1]['historical_wording_verified'])
        r['cve'] = 'CVE-2022-9999'
        self.assertEqual(len(a.screen_documents(r, self.stamp(r['published_timestamp']), True)), 2)

    def test_disclosure_and_audit_fields_not_predictors(self):
        r = self.record(); r['description'] = 'Added to VulnCheck KEV. EPSS score: 0.93.'
        docs = a.screen_documents(r, self.stamp(r['published_timestamp']), True)
        self.assertEqual(docs[0]['lexical_screen'], 'quarantine')
        self.assertEqual(len(a.feature_policy()['catalogue_fields_excluded']), 11)
        self.assertNotIn('text', docs[0])

    def test_csv_canonical_order_and_line_endings(self):
        rows = [{'cve': 'b', 'value': None}, {'cve': 'a', 'value': 1}]
        one = a.csv_text(rows, ['cve', 'value'])
        self.assertEqual(one, a.csv_text(list(reversed(rows)), ['cve', 'value']))
        self.assertNotIn('\r', one)

    def test_end_to_end_replay_and_holdout_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); spec = copy.deepcopy(a.READ(a.CONFIG)); cfg = spec['datasets']['previous']
            first = self.record(); second = dict(first, cve='CVE-2022-9999', in_vckev=False, epss_scores={})
            heldout = {'cve': 'CVE-2025-9999', 'description': 'DO_NOT_EXPORT_HELDOUT', 'in_vckev': True}
            objects = {'raw': [first, second, heldout],
                       'roles': {first['cve']: {'split': 'train'}, second['cve']: {'split': 'train'}, heldout['cve']: {'split': 'test'}},
                       'kev_identity': {'ids': {'train': [first['cve'], second['cve']]}},
                       'epss_protocol': {'rows': {first['cve']: {'outer_role': 'train'}}}}
            for key, obj in objects.items():
                path = root/(key+'.json'); path.write_text(json.dumps(obj))
                cfg[key], cfg[key+'_sha256'] = str(path), a.SHA(path)
            cfg.update(expected_candidates=2, expected_epss_train=1)
            config = root/'config.json'; config.write_text(json.dumps(spec)); dest = root/'output'
            with patch.object(a, 'CONFIG', config):
                a.run('previous', output_dir=dest)
                a.run('previous', output_dir=dest, verify=True)
                findings = json.loads((dest/'findings.json').read_text())
                self.assertEqual(findings['additional_candidates'], 1)
                self.assertEqual(findings['support'][0]['unknown'], 2)
                self.assertFalse(findings['snapshot_training_enabled'])
                for path in dest.iterdir():
                    self.assertNotIn('DO_NOT_EXPORT_HELDOUT', path.read_text())
                    self.assertNotIn('CVE-2025-9999', path.read_text())
                with self.assertRaises(ValueError):
                    a.run('previous', output_dir=dest)
                (dest/'REPORT.txt').write_text('tampered')
                with self.assertRaises(ValueError):
                    a.run('previous', output_dir=dest, verify=True)


if __name__ == '__main__':
    unittest.main()
