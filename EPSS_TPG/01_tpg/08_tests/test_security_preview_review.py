"""Checks for the separate DQ10 reviewer; no dataset, model or NLP loading."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'02_examples/01_scripts'))
import review_security_product_preview as review


def example():
    side = {'software': [{'text': 'B', 'span': [3, 4]}, {'text': 'A', 'span': [0, 1]}],
            'versions': [], 'typed_security_relations': []}
    return {'cve': 'CVE-example', 'split': 'train', 'before': deepcopy(side),
            'after': deepcopy(side), 'old_version_links': 0, 'new_version_links': 0,
            'unresolved_versions': 0, 'typed_relation_endpoints_changed': False}


class ReviewTests(unittest.TestCase):
    def test_order_only_normalization(self):
        first = example()
        second = deepcopy(first)
        second['before']['software'].reverse()
        self.assertEqual(review.canonical(first), review.canonical(second))
        self.assertEqual(first['before']['software'][0]['text'], 'B')

    def test_content_and_span_changes_not_hidden(self):
        first = example()
        for key, value in [('text', 'Wrong'), ('span', [4, 3])]:
            second = deepcopy(first)
            second['before']['software'][0][key] = value
            self.assertNotEqual(review.canonical(first), review.canonical(second))

    def test_duplicates_and_test_cohort_rejected(self):
        with self.assertRaises(ValueError):
            review.statistics([example(), example()])
        row = example()
        row['split'] = 'test'
        with self.assertRaises(ValueError):
            review.statistics([row])

    def test_empty_extraction_is_not_unresolved_owner(self):
        result = review.statistics([example()])
        self.assertEqual(result['after']['recognized_versions'], 0)
        self.assertEqual(result['unresolved_versions'], 0)
        self.assertEqual(result['cves'], 1)


if __name__ == '__main__':
    unittest.main()
