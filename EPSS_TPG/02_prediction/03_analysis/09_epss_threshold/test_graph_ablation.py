"""ET02 contract tests; no study labels, training or disk caches required."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
for path in ('01_tpg/01_core', '01_tpg/10_extensions', '02_prediction/01_package', '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT/path))

from epss import epss_threshold_graph_v1 as et


class ThresholdGraphTest(unittest.TestCase):
    def test_observed_boundary(self):
        self.assertEqual([et.label(x) for x in [None, 0, .09999, .1, 1]], [None, 0, 0, 1, 1])
        with self.assertRaises(ValueError):
            et.label(float('nan'))

    def test_absent_summary_keeps_both_classes_in_every_view(self):
        def views():
            return {v: {'object_key': 'd' if 'D' in v else None,
                        'source_mask': {s: s == 'D' and 'D' in v for s in 'DMGN'}} for v in et.VIEWS}
        ctx = {'index': {k: {'views': views()} for k in ('a', 'b')},
               'records': {'d': {'statistics': {'nodes': 2}}}}
        ys = {'a': 0, 'b': 1}
        chosen = et.smoke_ids(ctx, ys, ['a', 'b'])
        self.assertEqual(chosen, ['a', 'b'])
        r = et.coverage(ctx, ys, {'validation': chosen})['validation']
        self.assertEqual((r['cves'], r['positive']), (2, 1))
        self.assertEqual(r['views']['U']['text_present'], 0)
        self.assertEqual(r['views']['DMU']['positive_with_text'], 1)
        self.assertEqual(r['views']['DMU']['source_cves'], {'D': 2, 'M': 0, 'G': 0, 'N': 0})

    def test_label_cutoff_is_not_the_probability_cutoff(self):
        m = et.score(['a','b'], {'a': 1, 'b': 0}, [.2,.1])
        self.assertEqual((m['tp'], m['fn'], m['high_score']), (0, 1, 1))
        self.assertEqual(m['review_budgets'][1]['hits'], 1)
        self.assertEqual(et.summary_row('tpg', 'M', [m])['recall_at_5pct'], 1.)


if __name__ == '__main__':
    unittest.main()
