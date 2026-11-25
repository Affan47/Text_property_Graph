"""Review-only numerical and interpretation checks; never train models."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('fc01_review_cli',Path(__file__).with_name('35_review_final_run1.py'))
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
from epss import fc01_completed_review_v1 as review


class ReviewTests(unittest.TestCase):
    def test_metrics_preserve_error_scale_and_high_score_definition(self):
        result=review.metrics(np.array([0.,.1,1.]),np.array([.1,.1,.8]))
        self.assertAlmostEqual(result['rmse'],np.sqrt(.05/3))
        self.assertAlmostEqual(result['mae'],.1)
        self.assertEqual(result['top_50_hits'],2)

    def test_constant_rank_and_single_class_are_explicit(self):
        result=review.metrics(np.array([.01,.02]),np.array([.1,.1]))
        self.assertIsNone(result['spearman'])
        self.assertIsNone(result['auroc_high_score'])
        self.assertIsNone(result['ap_high_score'])

    def test_nominal_gain_does_not_override_adjusted_interval(self):
        rows=[{'partition':'all','unit':'evidence_group_and_seed','contrast':a+'_minus_'+b,
               'difference':-.1,'nominal_low':-.2,'nominal_high':-.01,'family_low':-.3,
               'family_high':.01,'seed_differences':[{'difference':-.1}]*5}
              for a,b in review.reporting.CONTRASTS]
        results=review.interval_decisions(rows)
        self.assertFalse(any(r['improvement_supported'] for r in results))
        self.assertTrue(all(r['favorable_seeds']==5 for r in results))

    def test_incomplete_primary_family_rejected(self):
        with self.assertRaises(ValueError):review.interval_decisions([])

    def test_csv_uses_unix_newlines(self):
        value=review.csv_text([{'x':1,'y':2}])
        self.assertEqual(value,'x,y\n1,2\n')


if __name__=='__main__':unittest.main()
