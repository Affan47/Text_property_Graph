import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'02_prediction/01_package'))
from epss.day7_availability_v1 import collector as c, reconstruction as r, plan
from epss.day7_availability_v1.report import paired_gain
import numpy as np


def event(day, action='Added', typ='CVSS V3.1', old=None, new=None, provider='nvd@nist.gov'):
    detail = {'action': action, 'type': typ}
    if old is not None: detail['oldValue'] = old
    if new is not None: detail['newValue'] = new
    return {'cveId': 'CVE-2022-1234', 'created': day, 'sourceIdentifier': provider,
            'cveChangeId': day+typ, 'details': [detail]}


V = 'AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H'
RECORD = {'cve': 'CVE-2022-1234', 'description': 'saved wording',
          'published_timestamp': '2022-05-01T12:00:00', 'last_modified_timestamp': '2026-06-01T00:00:00',
          'cvss_vector_version': '3.1', 'cvss_vector_source': 'nvd@nist.gov', 'cvss_vector': 'CVSS:3.1/'+V}
CUTOFF = plan.timestamp('2022-05-08T23:59:59.999999Z')


class ReconstructionTests(unittest.TestCase):
    def test_absent_history_does_not_prove_availability(self):
        value = r.reconstruct_field([], 'description', RECORD, CUTOFF)
        self.assertEqual(value['status'], 'HISTORY_CONSISTENT_BUT_ORIGIN_UNDATED')
        self.assertFalse(value['supported'])

    def test_late_cvss_addition(self):
        value = r.reconstruct_field([event('2022-05-12T00:00:00', new='NIST '+V)], 'cvss', RECORD, CUTOFF)
        self.assertEqual(value['status'], 'NO_ASSIGNMENT_RECORDED_BY_CUTOFF')
        self.assertIsNone(value['value_at_cutoff'])

    def test_early_cvss_addition(self):
        value = r.reconstruct_field([event('2022-05-05T00:00:00', new='NIST '+V)], 'cvss', RECORD, CUTOFF)
        self.assertTrue(value['supported'] and value['same_as_saved'])

    def test_exact_description_chain(self):
        events = [event('2022-05-01T00:00:00', typ='Description', new='original'),
                  event('2022-05-10T00:00:00', 'Changed', 'Description', 'original', 'saved wording')]
        value = r.reconstruct_field(events, 'description', RECORD, CUTOFF)
        self.assertTrue(value['supported'])
        self.assertEqual(value['value_at_cutoff'], 'original')
        self.assertFalse(value['same_as_saved'])

    def test_undated_old_description_not_promoted(self):
        events = [event('2022-05-10T00:00:00', 'Changed', 'Description', 'original', 'saved wording')]
        value = r.reconstruct_field(events, 'description', RECORD, CUTOFF)
        self.assertFalse(value['supported'])
        self.assertEqual(value['value_at_cutoff'], 'original')

    def test_chain_conflict(self):
        value = r.reconstruct_field([event('2022-05-10T00:00:00', 'Changed', 'Description', 'old', 'wrong')], 'description', RECORD, CUTOFF)
        self.assertIn('MISMATCH', value['status'])

    def test_events_after_saved_snapshot_not_applied(self):
        value = r.reconstruct_field([event('2026-07-01T00:00:00', 'Changed', 'Description', 'saved wording', 'new live value')], 'description', RECORD, CUTOFF)
        self.assertEqual(value['value_at_cutoff'], 'saved wording')
        self.assertFalse(value['supported'])

    def test_provider_not_silently_changed(self):
        value = r.reconstruct_field([event('2022-05-05T00:00:00', new='Vendor '+V, provider='vendor@example.org')], 'cvss', RECORD, CUTOFF)
        self.assertFalse(value['supported'])

    def test_ambiguous_nist_edit_of_other_provider(self):
        value = r.reconstruct_field([event('2022-05-05T00:00:00', new='Vendor '+V)], 'cvss', RECORD, CUTOFF)
        self.assertIn('UNPARSED', value['status'])

    def test_timezone_and_boundary(self):
        self.assertEqual(plan.timestamp('2022-05-08T12:00:00'), plan.timestamp('2022-05-08T14:00:00+02:00'))
        value = r.reconstruct_field([event('2022-05-08T23:59:59', new='NIST '+V)], 'cvss', RECORD, CUTOFF)
        self.assertTrue(value['supported'])

    def test_simultaneous_updates_not_arbitrarily_ordered(self):
        e = event('2022-05-05T00:00:00', new='NIST '+V)
        value = r.reconstruct_field([e, e], 'cvss', RECORD, CUTOFF)
        self.assertIn('SIMULTANEOUS', value['status'])

    def test_reference_gate_does_not_open(self):
        meta = {'cutoff': CUTOFF.isoformat(), 'cutoff_168h': CUTOFF.isoformat()}
        value = r.reconstruct([], RECORD, meta)
        self.assertFalse(value['all_saved_D_inputs_verified'])

    def test_already_high_is_not_a_new_future_crossing(self):
        meta = {'cutoff': CUTOFF.isoformat(), 'cutoff_168h': CUTOFF.isoformat(),
                'label_not_predictor': {'label': None, 'first_observed_crossing_day': 8}}
        result = r.reconstruct([event('2022-05-05T00:00:00', new='NIST '+V)], RECORD, meta)
        self.assertIsNone(result['crossing_day_minus_cvss_addition_day'])

    def test_paired_intervals_are_matched_and_one_class_is_undefined(self):
        y = np.array([0, 1]*10); score = np.arange(20)/20
        result = paired_gain(y, score, score, np.arange(20), draws=128)
        self.assertTrue(all(r['difference'] == 0 and r['low'] == 0 and r['high'] == 0 for r in result['intervals']))
        self.assertEqual(paired_gain(np.zeros(2), np.zeros(2), np.zeros(2), np.arange(2))['status'], 'UNDEFINED_ONE_CLASS_OR_EMPTY')


class TransportTests(unittest.TestCase):
    def page(self, events=None, total=None, start=0):
        events = [event('2022-05-05T00:00:00', new='NIST '+V)] if events is None else events
        return {'format': 'NVD_CVEHistory', 'version': '2.0', 'startIndex': start,
                'resultsPerPage': len(events), 'totalResults': len(events) if total is None else total,
                'cveChanges': [{'change': e} for e in events]}

    def test_page_counts_and_wrong_cve(self):
        with self.assertRaises(ValueError): c.validate_page(self.page(), 'CVE-2022-9999', 0)
        with self.assertRaises(ValueError): c.validate_page(self.page(), 'CVE-2022-1234', 1)
        with self.assertRaises(ValueError): c.validate_page(self.page(), 'CVE-2022-1234', 0, 5)
        value = self.page(); value['resultsPerPage'] = 2
        with self.assertRaises(ValueError): c.validate_page(value, 'CVE-2022-1234', 0)

    def test_empty_page_and_duplicate_detection(self):
        self.assertEqual(c.validate_page(self.page([], 0), 'CVE-2022-1234', 0), (0, []))
        with self.assertRaises(ValueError): c.validate_page(self.page([], 2), 'CVE-2022-1234', 0)
        e = event('2022-05-05T00:00:00', new='NIST '+V)
        with self.assertRaises(ValueError): c.validate_page(self.page([e, e]), 'CVE-2022-1234', 0)

    def test_retry_after_is_honored(self):
        self.assertEqual(c.retry_delay({'Retry-After': '120'}, 0), 120)
        self.assertEqual(c.retry_delay({}, 2), 120)

    def test_long_cooldown_survives_restart_without_request(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(c, 'STORE', Path(directory)):
            c.cooldown(3600)
            with patch.object(c, 'urlopen') as opener:
                with self.assertRaises(StopIteration): c.PublicClient().get('https://example.invalid', Path(directory)/'new')
                opener.assert_not_called()

    def test_short_cooldown_is_persisted(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(c, 'STORE', Path(directory)):
            c.cooldown(120)
            state = c.a.read(Path(directory)/'.last_request.json')
            self.assertGreaterEqual(state['not_before_epoch']-state['epoch'], 119)

    def test_raw_receipts_and_resume_integrity(self):
        payload = self.page()
        class FakeClient:
            def get(self, url, directory):
                body = json.dumps(payload).encode()
                info = {'url': url, 'http_status': 200, 'body_sha256': c.sha_bytes(body)}
                c.write_new(directory/'body.json', body); c.write_new(directory/'receipt.json', info)
                return info, body
        with tempfile.TemporaryDirectory() as directory, patch.object(c, 'STORE', Path(directory)):
            self.assertTrue(c.fetch('CVE-2022-1234', FakeClient()))
            events, pins = c.read_complete('CVE-2022-1234')
            self.assertEqual(len(events), 1); self.assertEqual(len(pins), 3)
            body = next(Path(directory).rglob('body.json')); body.write_bytes(b'changed')
            with self.assertRaises(ValueError): c.read_complete('CVE-2022-1234')

    def test_http_failure_never_becomes_empty_success(self):
        class FailedClient:
            def get(self, url, directory):
                info = {'http_status': 401, 'error': 'HTTPError'}
                c.write_new(directory/'body.json', b'Unauthorized'); c.write_new(directory/'receipt.json', info)
                return info, b'Unauthorized'
        with tempfile.TemporaryDirectory() as directory, patch.object(c, 'STORE', Path(directory)):
            self.assertFalse(c.fetch('CVE-2022-1234', FailedClient()))
            self.assertIsNone(c.read_complete('CVE-2022-1234')[0])

    def test_budget_refuses_before_network(self):
        client = c.PublicClient(max_requests=0)
        with patch.object(c, 'urlopen') as opener:
            with self.assertRaises(StopIteration): client.get('https://example.invalid', Path('/tmp/not-written'))
            opener.assert_not_called()

    def test_receipts_are_create_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'receipt.json'; c.write_new(path, {'a': 1})
            with self.assertRaises(FileExistsError): c.write_new(path, {'a': 2})
            self.assertEqual(c.a.read(path), {'a': 1})

    def test_multiple_pages_are_assembled_without_duplicate_events(self):
        e1 = event('2022-05-05T00:00:00', new='NIST '+V)
        e2 = event('2022-05-06T00:00:00', typ='Description', new='saved wording')
        pages = [self.page([e1], 2, 0), self.page([e2], 2, 1)]
        class FakeClient:
            def get(inner, url, directory):
                body = json.dumps(pages.pop(0)).encode()
                info = {'url': url, 'http_status': 200, 'body_sha256': c.sha_bytes(body)}
                c.write_new(directory/'body.json', body); c.write_new(directory/'receipt.json', info)
                return info, body
        with tempfile.TemporaryDirectory() as directory, patch.object(c, 'STORE', Path(directory)):
            self.assertTrue(c.fetch('CVE-2022-1234', FakeClient()))
            self.assertEqual(len(c.read_complete('CVE-2022-1234')[0]), 2)

if __name__ == '__main__':
    unittest.main()
