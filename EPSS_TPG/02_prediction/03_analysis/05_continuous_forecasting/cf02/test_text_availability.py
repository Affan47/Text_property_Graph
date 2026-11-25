"""Focused checks for the read-only historical-text evidence gate."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('cf02_audit', Path(__file__).with_name('07_audit_text_availability.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class EvidenceGateTests(unittest.TestCase):
    def pair(self):
        post = {'post':'sample', 'platform':'reddit', 'date':'2025-01-01T12:00:00'}
        evidence = dict(post_sha256=hashlib.sha256(b'sample').hexdigest(), platform='reddit',
            candidate_timestamp=post['date'], historical_text_verified=False,
            post_publication_verified=False, verified_publication_timestamp=None,
            publication_evidence_missing=['HISTORICAL_TEXT_VERSION_UNVERIFIED'])
        return post, evidence

    def test_exact_match_does_not_authenticate(self):
        post, evidence = self.pair()
        audit.match_post(post, evidence)
        self.assertFalse(evidence['historical_text_verified'])

    def test_changed_text_date_platform_fail(self):
        for key, value in [('post','changed'), ('date','2025-01-02T12:00:00'), ('platform','telegram')]:
            post, evidence = self.pair()
            post[key] = value
            with self.assertRaises(ValueError):
                audit.match_post(post, evidence)

    def test_new_provenance_requires_review(self):
        post, evidence = self.pair()
        post['permalink'] = 'https://example.invalid/post'
        with self.assertRaises(ValueError):
            audit.match_post(post, evidence)
        post, evidence = self.pair()
        evidence['historical_text_verified'] = True
        with self.assertRaises(ValueError):
            audit.match_post(post, evidence)

    def test_naive_time_not_promoted_to_utc_or_verified(self):
        parsed = audit.ms.stored_date('2025-01-01T12:00:00', '2025-01-02')
        self.assertEqual(parsed['relation'], 'before_anchor')
        self.assertEqual(parsed['timezone'], 'not_supplied')
        self.assertFalse(parsed['historical_verified'])

    def test_canonical_csv_bytes(self):
        rows = [dict(split='validation', cve='CVE-2025-2', source='smp'),
                dict(split='train', cve='CVE-2025-1', source='description')]
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a.csv', Path(tmp)/'b.csv'
            audit.write_csv(a, rows)
            audit.write_csv(b, list(reversed(rows)))
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertNotIn(b'\r\n', a.read_bytes())


if __name__ == '__main__':
    unittest.main()
