"""Offline contract tests; synthetic records, no model fitting or test cohort."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import prepare_rd02_summary_extension as s


CVE = 'CVE-2021-12345'
FIELDS = {'G': {'text': 'summary_g', 'references': 'github_only_urls'},
          'N': {'text': 'summary_n', 'references': 'non_github_urls'}}


def sample():
    d = dict(document_id='D-example', source='D', text='A buffer overflow affects a parser.',
             text_sha256=s.text_hash('A buffer overflow affects a parser.'), is_prose=False)
    row = dict(cve=CVE, split='train', structured={'github_only_urls': ['https://github.com/example/project'],
               'non_github_urls': ['https://example.org/advisory']}, documents=[d])
    old = dict(cve=CVE, summary_g='The parser can read beyond its input buffer.',
               summary_n='The advisory describes an input parsing flaw.',
               github_only_urls=['https://github.com/example/project'],
               non_github_urls=['https://example.org/advisory'], epss_scores={'future': .99}, in_vckev=True)
    return row, old


class SummaryAdapterTests(unittest.TestCase):
    def test_preserves_exact_summary_and_excludes_outcomes(self):
        row, old = sample()
        old['summary_g'] = '  A parser defect.\n'
        doc, audit = s.adapt_summary(row, old, 'G', FIELDS['G'], 'abc')
        self.assertEqual(doc['text'], old['summary_g'])
        self.assertEqual(set(doc), {'document_id', 'source', 'text', 'text_sha256', 'is_prose'})
        self.assertTrue(audit['accepted'])
        self.assertFalse(audit['historical_availability_verified'])

    def test_reference_set_dedup_and_subset(self):
        row, old = sample()
        old['github_only_urls'] *= 2
        row['structured']['github_only_urls'].append('https://github.com/example/another')
        doc, a = s.adapt_summary(row, old, 'G', FIELDS['G'], 'abc')
        self.assertIsNotNone(doc)
        self.assertEqual(a['reference_relation'], 'original_subset_of_current')
        self.assertEqual(a['original_reference_count'], 1)

    def test_mismatched_references_do_not_enter_encoder(self):
        row, old = sample()
        old['github_only_urls'] = ['https://github.com/unrelated/project']
        doc, a = s.adapt_summary(row, old, 'G', FIELDS['G'], 'abc')
        self.assertIsNone(doc)
        self.assertIn('unmatched_reference_set', a['reasons'])

    def test_missing_summary_retains_cve_and_false_mask(self):
        row, _ = sample()
        overlay, audits = s.overlay_rows([row], {}, FIELDS, 'abc')
        self.assertEqual(len(overlay), 1)
        self.assertFalse(overlay[0]['mask_G'])
        self.assertFalse(overlay[0]['mask_N'])
        self.assertEqual(len(audits), 2)

    def test_shared_url_does_not_allow_cross_cve_summary(self):
        row, old = sample()
        old['cve'] = 'CVE-2021-67890'
        with self.assertRaises(ValueError):
            s.adapt_summary(row, old, 'G', FIELDS['G'], 'abc')

    def test_duplicate_source_records_fail(self):
        _, old = sample()
        with self.assertRaises(ValueError):
            s.index_records([old, deepcopy(old)])

    def test_epss_topic_quarantined_even_without_number(self):
        row, old = sample()
        old['summary_g'] = 'EPSS places this issue among the likely targets.'
        doc, a = s.adapt_summary(row, old, 'G', FIELDS['G'], 'abc')
        self.assertIsNone(doc)
        self.assertIn('epss_topic_disclosure', a['reasons'])

    def test_other_cve_reference_is_not_automatically_rejected(self):
        row, old = sample()
        old['summary_g'] = 'This flaw differs from CVE-2020-12345.'
        doc, a = s.adapt_summary(row, old, 'G', FIELDS['G'], 'abc')
        self.assertIsNotNone(doc)
        self.assertIn('mentions_other_cves_requires_context', a['flags'])
        old['summary_g'] = 'CVE identifier is CVE-2020-12345.'
        self.assertIsNone(s.adapt_summary(row, old, 'G', FIELDS['G'], 'abc')[0])

    def test_placeholders_and_insufficient_language_are_different(self):
        row, old = sample()
        old['summary_g'] = 'N/A'
        self.assertIsNone(s.adapt_summary(row, old, 'G', FIELDS['G'], 'abc')[0])
        old['summary_g'] = 'The page has no useful details about the parser.'
        doc, a = s.adapt_summary(row, old, 'G', FIELDS['G'], 'abc')
        self.assertIsNotNone(doc)
        self.assertIn('insufficient_content_language_requires_review', a['flags'])

    def test_test_role_rejected(self):
        row, old = sample()
        row['split'] = 'test'
        with self.assertRaises(ValueError):
            s.overlay_rows([row], {CVE: old}, FIELDS, 'abc')

    def test_matched_d_view_and_source_masks(self):
        row, old = sample()
        row['documents'].append(dict(document_id='M-id', source='M', text=CVE,
                                    text_sha256=s.text_hash(CVE), is_prose=False))
        overlays, _ = s.overlay_rows([row], {CVE: old}, FIELDS, 'abc')
        d = s.compose(row, overlays[0], ['D'])
        self.assertEqual(d['documents'], row['documents'][:1])
        full = s.compose(row, overlays[0], ['D', 'G', 'N', 'M_prose'])
        self.assertEqual(full['source_mask'], [True, True, True, False])
        self.assertIs(full['structured'], row['structured'])

    def test_request_queue_contains_no_targets(self):
        row, _ = sample()
        _, audit = s.overlay_rows([row], {}, FIELDS, 'abc')
        requests, bad = s.requests_for([row], audit)
        self.assertEqual(len(requests), 2)
        self.assertEqual(bad, [])
        self.assertTrue(all(r['status'] == 'NOT_REQUESTED' for r in requests))
        for r in requests:
            self.assertNotIn('epss_scores', r)
            self.assertNotIn('target', r)
            self.assertNotIn('in_vckev', r)

    def test_non_http_references_not_sent_to_scraper(self):
        row, _ = sample()
        row['structured']['github_only_urls'] = ['file:///etc/passwd']
        _, audits = s.overlay_rows([row], {}, FIELDS, 'abc')
        requests, bad = s.requests_for([row], audits)
        self.assertEqual(len(requests), 1)
        self.assertEqual(len(bad), 1)

    def test_source_overlap_audit_only(self):
        row, old = sample()
        other = deepcopy(row)
        other.update(cve='CVE-2024-12345', split='validation')
        other['documents'][0]['text'] = old['summary_g']
        base = sorted([row, other], key=lambda r: r['cve'])
        overlay, _ = s.overlay_rows(base, {CVE: old}, FIELDS, 'abc')
        groups = s.overlaps(base, overlay)
        self.assertEqual(len(groups), 1)
        self.assertEqual({a['split'] for a in groups[0]['associations']}, {'train', 'validation'})
        self.assertFalse(overlay[1]['mask_G'])

    def test_serialization_byte_identical(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder)/'a.gz', Path(folder)/'b.gz'
            s.write_jsonl(a, [{'b': 2, 'a': 1}]); s.write_jsonl(b, [{'a': 1, 'b': 2}])
            self.assertEqual(a.read_bytes(), b.read_bytes())
            with gzip.open(a, 'rt') as f:
                self.assertEqual(json.loads(f.readline()), {'a': 1, 'b': 2})
            s.csv_table(Path(folder)/'rows.csv', [{'a': 1}], ['a'])
            self.assertEqual((Path(folder)/'rows.csv').read_bytes(), b'a\n1\n')


if __name__ == '__main__':
    unittest.main()
