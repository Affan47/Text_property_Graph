"""Synthetic review checks; no network or held-out data."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('cc01_review', Path(__file__).with_name('02_review_catalogue_snapshot.py'))
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class ReviewTests(unittest.TestCase):
    def fixtures(self):
        ids = ['CVE-2022-1234', 'CVE-2022-1235']
        labels = [{'cve': c, 'label': i, 'status': 'LISTED' if i else 'NOT_LISTED',
                   'catalogue_date_added': '2026-11-17' if i else None} for i, c in enumerate(ids)]
        legacy = [{'cve': c, 'saved_catalogue_state': 'NOT_LISTED_UNVERIFIED'} for c in ids]
        return ids, labels, legacy

    def test_counts_disagreement_and_baseline(self):
        s, t, c = review.summarize(*self.fixtures())
        self.assertEqual((s['cves'], s['listed'], s['not_listed'], s['unknown']), (2, 1, 1, 0))
        self.assertEqual(s['listed_fraction'], .5)
        self.assertEqual(s['all_not_listed_positive_f1'], 0)
        self.assertEqual(c[0]['cve'], 'CVE-2022-1235')
        self.assertEqual(sum(r['cves'] for r in t), 2)

    def test_rows_are_order_independent(self):
        ids, labels, old = self.fixtures()
        self.assertEqual(review.summarize(ids, labels, old), review.summarize(ids[::-1], labels[::-1], old[::-1]))

    def test_extra_heldout_identity_is_rejected(self):
        ids, labels, old = self.fixtures()
        with self.assertRaises(ValueError):
            review.summarize(ids[:1], labels, old)

    def test_duplicates_rejected(self):
        ids, labels, old = self.fixtures()
        with self.assertRaises(ValueError):
            review.summarize(ids, labels + labels[:1], old)

    def test_unknown_never_becomes_negative(self):
        ids, labels, old = self.fixtures()
        labels[0].update(status='UNKNOWN', label=None)
        with self.assertRaises(ValueError):
            review.summarize(ids, labels, old)

    def test_csv_stable_order_and_unix_newline(self):
        rows = [{'a': 'b', 'n': 1}, {'a': 'a', 'n': 2}]
        self.assertEqual(review.csv_text(rows, ['a', 'n']), review.csv_text(rows[::-1], ['a', 'n']))
        self.assertNotIn('\r', review.csv_text(rows, ['a', 'n']))


if __name__ == '__main__':
    unittest.main()
