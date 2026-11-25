import importlib.util
from pathlib import Path
import unittest
import numpy as np

path = Path(__file__).with_name('11_review_training_only_pooled.py')
spec = importlib.util.spec_from_file_location('ct05_review',path)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class ReviewTests(unittest.TestCase):
    def test_saving_is_additive_squared_error(self):
        d=np.tile([.1,.4],(3,1));dm=np.tile([.2,.8],(3,1))
        rows=review.contributions(['a','b'],[0,.9],d,dm)
        np.testing.assert_allclose([r['squared_error_saving'] for r in rows],[-.03,.24])
        self.assertGreater(rows[1]['share_of_net_saving_percent'],100)

    def test_seed_metrics_not_ensemble_errors(self):
        d=np.array([[0.],[1.],[0.]]);dm=np.full((3,1),1/3)
        rows=review.contributions(['a'],[0.],d,dm)
        self.assertAlmostEqual(rows[0]['D_mean_prediction'],rows[0]['DM_mean_prediction'])
        self.assertAlmostEqual(rows[0]['squared_error_saving'],2/9)

    def test_zero_net_share_explicit(self):
        rows=review.contributions(['a'],[.5],np.zeros((3,1)),np.zeros((3,1)))
        self.assertIsNone(rows[0]['share_of_net_saving_percent'])

    def test_unmatched_seed_shape_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unmatched'):
            review.contributions(['a'],[.2],np.zeros((2,1)),np.zeros((3,1)))

    def test_partitions_do_not_mix_absent_and_low_scores(self):
        row={'fold':2,'M_present':False,'actual':.2}
        self.assertTrue(review.in_partition(row,'M_absent'))
        self.assertTrue(review.in_partition(row,'score_ge_0.1'))
        self.assertFalse(review.in_partition(row,'M_present'))
        self.assertFalse(review.in_partition(row,'score_lt_0.1'))
        self.assertFalse(review.in_partition(row,'fold_1'))

    def test_metric_guard(self):
        review.assert_metrics({'rmse':.1,'spearman':None},{'rmse':.1,'spearman':None})
        with self.assertRaisesRegex(ValueError,'differs'):
            review.assert_metrics({'rmse':.2},{'rmse':.1})


if __name__=='__main__':unittest.main()
