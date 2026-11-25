import runpy
from pathlib import Path
import unittest
import numpy as np

runpy.run_path(str(Path(__file__).with_name('31_review_description_surrogate.py')),run_name='review_import')
from epss import description_surrogate_review_v1 as review


class ReviewTests(unittest.TestCase):
    def test_error_metrics(self):
        m=review.regression_metrics([0.,1.],[.1,.8])
        self.assertAlmostEqual(m['mae'],.15);self.assertAlmostEqual(m['rmse'],np.sqrt(.025))
        self.assertAlmostEqual(m['signed_bias'],-.05)

    def test_one_class_is_not_perfect_ranking(self):
        self.assertIsNone(review.ranking_metrics([1,1],[.2,.4])['auroc'])

    def test_tied_ranking(self):
        r=review.ranking_metrics([0,0,1,1],[.2,.2,.2,.2])
        self.assertEqual(r,{'auroc':.5,'average_precision':.5})

    def test_benchmark_includes_preload_separately(self):
        b={'all_parameters_unchanged':True,'training_fits':0,'labels_loaded':False,'test_access':False,
           'config':{'limit':24,'repetitions':3,'threads':4},'timings':[],
           'sample':{'x':{'nodes':10}},'candidate_graphs':100,'resident_tensor_bytes':1048576,
           'peak_process_rss_kib':1048576,'preload_seconds':3.}
        for store,load in [('uncached_verified_loader',3.),('resident_RAM',0.)]:
            b['timings'] += [{'store':store,'repeat':i,'graphs':24,'load_seconds':load,
                             'forward_backward_seconds':1.,'total_seconds':load+1} for i in (1,2,3)]
        r=review.summarize_benchmark(b)
        self.assertEqual(r['warm_measured_segment_speedup'],4.)
        self.assertEqual(r['three_pass_speedup_including_preload'],2.)
        self.assertFalse(r['full_epoch_speedup_established'])

    def test_invalid_scope_rejected(self):
        with self.assertRaises(ValueError):review.summarize_benchmark({'all_parameters_unchanged':False})

    def test_changed_number_rejected(self):
        with self.assertRaises(ValueError):review.equal(.1,.11)


if __name__=='__main__':unittest.main()
