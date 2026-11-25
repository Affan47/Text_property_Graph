"""Regression checks for the read-only external CT05 review."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).with_name('12_review_external_ct05.py')
spec = importlib.util.spec_from_file_location('external_ct05_review', path)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class TargetVersionTest(unittest.TestCase):
    def test_publishing_boundaries(self):
        for before, release, old, new in (
                ('2021-04-13', '2021-04-14', 'before_EPSS', 'v1'),
                ('2022-02-03', '2022-02-04', 'v1', 'v2'),
                ('2023-03-06', '2023-03-07', 'v2', 'v3'),
                ('2025-03-16', '2025-03-17', 'v3', 'v4')):
            self.assertEqual(review.target_version(before), old)
            self.assertEqual(review.target_version(release), new)

    def test_target_not_window_end(self):
        self.assertEqual(review.target_version('2025-03-01'), 'v3')
        self.assertEqual(review.target_version('2025-03-31'), 'v4')

    def test_bad_date_fails(self):
        with self.assertRaises(ValueError):
            review.target_version('2023-02-30')

    def test_support_does_not_merge_versions(self):
        rows = [{'date': '2023-03-06', 'y': .00885},
                {'date': '2023-03-07', 'y': .00048},
                {'date': '2023-03-08', 'y': .2}]
        result = review.support(rows, 'assessment')
        self.assertEqual([r['cves'] for r in result], [1, 2])
        self.assertEqual([r['high_score_cves'] for r in result], [0, 1])
        self.assertEqual([r['at_0_00885'] for r in result], [1, 0])


if __name__ == '__main__':
    unittest.main()
