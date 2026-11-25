import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

spec = importlib.util.spec_from_file_location('publication_cli', Path(__file__).with_name('04_check_epss_publication.py'))
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
from epss import catalogue_publication_v1 as publication
from epss.catalogue_classification_v1 import snapshot as frozen


def events(github=404, dated=403, advertised='2026-11-16'):
    return [{'source': 'configured_archive', 'http_status': github},
            {'source': 'publisher_dated', 'http_status': dated},
            {'source': 'publisher_current', 'http_status': 302,
             'location': 'epss_scores-' + advertised + '.csv.gz', 'ended_at': '2026-11-17T11:35:00+00:00'}]


class PublicationTests(unittest.TestCase):
    def test_observed_delay_is_not_missing_scores(self):
        f = publication.summarize(events(), '2026-11-17')
        self.assertEqual(f['state'], 'WAITING_FOR_REQUIRED_DATE')
        self.assertEqual(f['latest_advertised_date'], '2026-11-16')
        self.assertFalse(f['scores_verified'])
        self.assertFalse(f['can_attempt_configured_download'])

    def test_http_200_is_only_permission_to_attempt_original_verifier(self):
        f = publication.summarize(events(200, 200, '2026-11-17'), '2026-11-17')
        self.assertEqual(f['state'], 'CONFIGURED_ARCHIVE_AVAILABLE')
        self.assertTrue(f['can_attempt_configured_download'])
        self.assertFalse(f['scores_verified'])

    def test_mirror_lag_does_not_change_source(self):
        f = publication.summarize(events(404, 200, '2026-11-17'), '2026-11-17')
        self.assertEqual(f['state'], 'PUBLISHER_AVAILABLE_MIRROR_PENDING')
        self.assertFalse(f['can_attempt_configured_download'])

    def test_403_without_old_current_link_is_inconclusive(self):
        f = publication.summarize(events(403, 403, '2026-11-17'), '2026-11-17')
        self.assertEqual(f['state'], 'PUBLICATION_STATUS_UNRESOLVED')

    def test_network_failure_never_means_negative(self):
        f = publication.summarize(events(None, None), '2026-11-17')
        self.assertEqual(f['state'], 'PUBLICATION_STATUS_UNRESOLVED')

    def test_wrong_redirect_is_not_followed_or_used(self):
        for url in ['https://evil.example/epss_scores-2026-11-16.csv.gz',
                    'http://epss.empiricalsecurity.com/epss_scores-2026-11-16.csv.gz',
                    'https://' + 'x:y@epss.empiricalsecurity.com/epss_scores-2026-11-16.csv.gz',
                    'https://epss.empiricalsecurity.com:bad/epss_scores-2026-11-16.csv.gz',
                    'epss_scores-2026-99-99.csv.gz', 'epss_scores-2026-11-16.csv.gz?secret=x']:
            with self.subTest(url=url):
                self.assertIsNone(publication.advertised_date({'http_status': 302, 'location': url}))

    def test_valid_relative_and_absolute_redirects(self):
        for url in ['epss_scores-2026-11-16.csv.gz', 'https://epss.empiricalsecurity.com/epss_scores-2026-11-16.csv.gz']:
            self.assertEqual(publication.advertised_date({'http_status': 302, 'location': url}), '2026-11-16')

    def test_head_only_no_redirects_no_cookies_recorded(self):
        session = MagicMock()
        response = session.head.return_value.__enter__.return_value
        response.status_code = 302
        response.headers = {'Location': 'epss_scores-2026-11-16.csv.gz', 'Set-Cookie': 'private', 'Date': 'today'}
        row = publication.probe(session, 'publisher_current', publication.CURRENT)
        session.head.assert_called_once_with(publication.CURRENT, timeout=(10, 30), allow_redirects=False)
        session.get.assert_not_called()
        self.assertNotIn('Set-Cookie', row['headers'])

    def test_timeout_is_recorded_without_exception_url(self):
        session = MagicMock()
        session.head.side_effect = requests.Timeout('secret proxy credentials')
        row = publication.probe(session, 'publisher_current', publication.CURRENT)
        self.assertEqual(row['error_type'], 'Timeout')
        self.assertNotIn('secret', str(row))

    def test_incomplete_probe_rejected(self):
        with self.assertRaises(ValueError):
            publication.summarize(events()[:2], '2026-11-17')

    def test_delay_does_not_invoke_collector_or_create_epss(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {'output': tmp}
            f = publication.summarize(events(), '2026-11-17')
            with patch.object(publication, 'check', return_value=(Path(tmp), f)), patch.object(publication.inputs, 'run') as collect:
                self.assertEqual(publication.collect_checked(cfg, True, False, True), 2)
                collect.assert_not_called()
                self.assertFalse((Path(tmp) / 'epss').exists())

    def test_ready_delegates_without_patching_original_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {'output': tmp}
            f = publication.summarize(events(200, 200, '2026-11-17'), '2026-11-17')
            with patch.object(publication, 'check', return_value=(Path(tmp), f)), patch.object(publication.inputs, 'run') as collect:
                self.assertEqual(publication.collect_checked(cfg, True, False, True), 0)
                collect.assert_called_once_with('collect-epss', allow_network=True, resume=True)

    def test_missing_verify_is_offline_actionable(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(publication, 'check') as check:
            self.assertEqual(publication.collect_checked({'output': tmp}, False, True, False), 2)
            check.assert_not_called()

    def test_completed_resume_makes_no_status_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'epss').mkdir()
            with patch.object(publication, 'check') as check, patch.object(publication.inputs, 'run') as collect:
                self.assertEqual(publication.collect_checked({'output': tmp}, False, True, False), 0)
                collect.assert_called_once_with('collect-epss', verify=True, resume=False)
                check.assert_not_called()

    def test_download_validation_failure_has_receipt_but_no_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            f = publication.summarize(events(200, 200), '2026-11-17')
            with patch.object(publication, 'OUTPUT', base / 'status'), patch.object(publication, 'check', return_value=(base, f)), \
                 patch.object(publication.inputs, 'run', side_effect=ValueError('EPSS internal date differs')):
                self.assertEqual(publication.collect_checked({'output': str(base / 'data')}, True, False, True), 2)
            receipts = list((base / 'status/collector_failures').glob('*/receipt.json'))
            self.assertEqual(len(receipts), 1)
            self.assertFalse(frozen.read(receipts[0])['collector_succeeded'])
            self.assertIn('internal date differs', frozen.read(receipts[0])['error_message'])
            self.assertFalse((base / 'data/epss').exists())

    def test_receipt_replay_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / 'freeze').mkdir()
            frozen.write(base / 'freeze/roles.json', {'fixed': True})
            cfg = {'output': tmp, 'epss_date': '2026-11-17', 'epss_url': 'https://example.org/frozen-test-url'}
            pins = {str(base / 'freeze/roles.json'): frozen.sha(base / 'freeze/roles.json')}
            with patch.object(publication, 'OUTPUT', base / 'status'), patch.object(publication, 'identities', return_value=pins), \
                 patch.object(publication, 'probe', side_effect=events()), patch.object(publication.requests, 'Session') as session:
                output, findings = publication.check(cfg, 'test')
                self.assertFalse(session.return_value.__enter__.return_value.trust_env)
                original = frozen.read
                with patch.object(publication.frozen, 'read', wraps=original) as read:
                    read.side_effect = lambda p: cfg if Path(p) == publication.inputs.CONFIG else original(p)
                    _, replay = publication.verify_check('test')
                self.assertEqual(replay, findings)
                (output / 'receipt.json').write_text('{}')
                with self.assertRaises(ValueError):
                    publication.verify_check('test')


if __name__ == '__main__':
    unittest.main()
