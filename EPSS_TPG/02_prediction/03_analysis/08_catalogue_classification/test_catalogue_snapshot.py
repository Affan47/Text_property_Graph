"""Offline tests: no credentials, external requests or real catalogue labels."""
from copy import deepcopy
from pathlib import Path
import hashlib
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / '02_prediction/01_package'))
from epss.catalogue_classification_v1 import snapshot as s


def archive(rows=None, name='catalogue.json'):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr(name, json.dumps(rows if rows is not None else [
            {'cve': ['CVE-2022-1234'], 'date_added': '2026-11-16T00:00:00Z'}]))
    return stream.getvalue()


def metadata(body):
    return {'_meta': {'index': 'vulncheck-kev'}, 'data': [{
        'filename': 'catalogue.zip', 'sha256': hashlib.sha256(body).hexdigest(),
        'date_added': '2026-11-17T00:00:00Z',
        'url': 'https://serve.vulncheck.com/latest/catalogue.zip?Signature=PRIVATE'}]}


class Response:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, chunk_size):
        yield self.body


class Session:
    def __init__(self, responses):
        self.responses, self.calls = iter(responses), []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.responses)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.cfg = s.read(s.CONFIG)
        self.now = '2026-11-17T12:00:00+00:00'

    def decode(self, body):
        return s.decode_records(io.BytesIO(body), self.cfg, '2026-11-17T00:00:00Z')

    def test_valid_archive(self):
        ids, inventory = self.decode(archive())
        self.assertEqual(set(ids), {'CVE-2022-1234'})
        self.assertEqual(inventory['records'], 1)

    def test_no_verified_archive_means_unknown_not_zero(self):
        rows = s.label_rows(['CVE-2022-1234'])
        self.assertIsNone(rows[0]['label'])
        self.assertEqual(rows[0]['status'], 'UNKNOWN')

    def test_verified_absence_is_not_listed(self):
        rows = s.label_rows(['CVE-2022-1235', 'CVE-2022-1234'], {'CVE-2022-1234': 'date'})
        self.assertEqual([r['label'] for r in rows], [1, 0])
        self.assertEqual(rows[1]['status'], 'NOT_LISTED')

    def test_unrequested_holdout_never_joined(self):
        rows = s.label_rows(['CVE-2022-1234'], {'CVE-2022-1234': 'date', 'CVE-2022-9999': 'date'})
        self.assertEqual([r['cve'] for r in rows], ['CVE-2022-1234'])

    def test_labels_are_canonical(self):
        self.assertEqual(s.label_rows(['CVE-2022-1235', 'CVE-2022-1234']),
                         s.label_rows(['CVE-2022-1234', 'CVE-2022-1235']))

    def test_duplicate_candidates_rejected(self):
        with self.assertRaises(ValueError):
            s.label_rows(['CVE-2022-1234'] * 2)

    def test_bad_record_packaging_rejected(self):
        for rows in ([], {'unrelated': []}, [{'cve': []}], [{'cve': ['not-a-CVE']}],
                     [{'cve': ['CVE-2022-1234'], 'date_added': '2027-01-01T00:00:00Z'}],
                     [{'cve': ['CVE-2022-1234'], 'date_added': '2026-11-16'}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.decode(archive(rows))

    def test_duplicate_records_rejected(self):
        row = {'cve': ['CVE-2022-1234'], 'date_added': '2026-11-16T00:00:00Z'}
        with self.assertRaises(ValueError):
            self.decode(archive([row, row]))

    def test_json_data_envelope_supported(self):
        row = {'cve': ['CVE-2022-1234'], 'date_added': '2026-11-16T00:00:00Z'}
        self.assertEqual(self.decode(archive({'data': [row]}))[1]['records'], 1)

    def test_unsafe_zip_and_unsupported_member_rejected(self):
        for name in ('../catalogue.json', '/catalogue.json', 'a\\catalogue.json', 'partial.txt'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.decode(archive(name=name))

    def test_decompression_budget(self):
        self.cfg['max_uncompressed_bytes'] = 2
        with self.assertRaises(ValueError):
            self.decode(archive())

    def test_metadata_host_credentials_and_redirect_targets(self):
        for url in ('http://serve.vulncheck.com/a', 'https://evil.example/a',
                    'https://serve.vulncheck.com.evil.example/a',
                    'https://token@serve.vulncheck.com/a', 'https://serve.vulncheck.com:444/a'):
            payload = metadata(archive())
            payload['data'][0]['url'] = url
            with self.subTest(url=url), self.assertRaises(ValueError):
                s.validate_metadata(payload, self.cfg, self.now)

    def test_wrong_index_hash_ambiguity_and_date(self):
        original = metadata(archive())
        for mode in ('index', 'hash', 'multiple', 'stale', 'future'):
            payload = deepcopy(original)
            if mode == 'index': payload['_meta']['index'] = 'cisa-kev'
            if mode == 'hash': payload['data'][0]['sha256'] = ''
            if mode == 'multiple': payload['data'] *= 2
            if mode == 'stale': payload['data'][0]['date_added'] = '2020-01-01T00:00:00Z'
            if mode == 'future': payload['data'][0]['date_added'] = '2027-01-01T00:00:00Z'
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                s.validate_metadata(payload, self.cfg, self.now)

    def create_plan(self, root):
        plan = {'schema': self.cfg['schema'], 'specification': self.cfg,
                'training_ids': {'synthetic': ['CVE-2022-1234', 'CVE-2022-1235']}}
        (root / 'plan').mkdir()
        s.write(root / 'plan/plan.json', plan)
        return plan

    def test_collect_verify_and_credentials_not_forwarded(self):
        body = archive()
        fake = Session([Response(json.dumps(metadata(body)).encode()), Response(body)])
        with tempfile.TemporaryDirectory() as tmp, patch.object(s, 'utc', return_value=self.now):
            root = Path(tmp)
            plan = self.create_plan(root)
            s.collect(root, plan, 'SECRET_TOKEN', fake)
            f = s.verify_snapshot(root, plan)
            self.assertEqual(f['counts']['synthetic'], {'LISTED': 1, 'NOT_LISTED': 1, 'UNKNOWN': 0})
            self.assertEqual(fake.calls[0][1]['headers']['Authorization'], 'Bearer SECRET_TOKEN')
            self.assertNotIn('Authorization', fake.calls[1][1]['headers'])
            self.assertTrue(all(not c[1]['allow_redirects'] for c in fake.calls))
            self.assertEqual((root / 'snapshot/metadata.json').stat().st_mode & 0o777, 0o600)
            for name in ('receipt.json', 'REPORT.txt', 'findings.json'):
                text = (root / 'snapshot' / name).read_text()
                self.assertNotIn('SECRET_TOKEN', text)
                self.assertNotIn('PRIVATE', text)
            (root / 'snapshot/synthetic_labels.json').write_text('[]')
            with self.assertRaises(ValueError):
                s.verify_snapshot(root, plan)

    def test_failed_http_or_hash_cannot_publish_negatives(self):
        body = archive()
        for mode in ('unauthorized', 'redirect', 'wrong_hash', 'truncated'):
            payload = metadata(body)
            if mode == 'wrong_hash': payload['data'][0]['sha256'] = '0' * 64
            if mode == 'truncated': body_to_send = body[:-10]
            else: body_to_send = body
            responses = [Response(json.dumps(payload).encode(), 401 if mode == 'unauthorized' else 302 if mode == 'redirect' else 200),
                         Response(body_to_send)]
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp, patch.object(s, 'utc', return_value=self.now):
                root = Path(tmp)
                plan = self.create_plan(root)
                with self.assertRaises(ValueError):
                    s.collect(root, plan, 'SECRET', Session(responses))
                self.assertFalse((root / 'snapshot').exists())
                self.assertFalse(list((root / 'attempts').rglob('*_labels.json')))
                failure = s.read(next((root / 'attempts').rglob('failure.json')))
                self.assertEqual(failure['outcomes'], 'UNKNOWN')

    def test_download_size_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                s.request_bytes(Session([Response(b'abcdef')]), 'https://example.com', Path(tmp)/'data', 3, {}, [], 'test')

    def test_empty_key_never_requests(self):
        fake = Session([])
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            s.collect(Path(tmp), {'specification': self.cfg}, '', fake)
        self.assertEqual(fake.calls, [])

    def test_frozen_protocol_has_no_training_or_test(self):
        self.assertFalse(self.cfg['training_enabled'])
        self.assertFalse(self.cfg['test_access_enabled'])
        self.assertFalse(self.cfg['fc01_modified'])
        self.assertEqual(self.cfg['comparison_design']['primary_metric'], 'average_precision')
        self.assertGreaterEqual(self.cfg['comparison_design']['paired_bootstrap_repetitions'], 10000)


if __name__ == '__main__':
    unittest.main()
