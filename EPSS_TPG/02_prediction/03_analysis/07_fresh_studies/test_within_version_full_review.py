"""Focused tests for saved-result review semantics; no dataset training."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('ct07_full_review', Path(__file__).with_name('18_review_within_version_training.py'))
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class FullReviewTests(unittest.TestCase):
    def test_history_earliest_best_and_stop(self):
        rows = [{'epoch': i+1, 'selection_rmse': .1 if i < 2 else .2,
                 'optimization_mse': .02, 'patience': i} for i in range(16)]
        ev = {'best_epoch': 1, 'selection_rmse': .1, 'history_epochs': 16}
        self.assertEqual(review.inspect_history(rows, ev)['best_epoch'], 1)
        with self.assertRaises(ValueError):
            review.inspect_history(rows, dict(ev, best_epoch=2))
        with self.assertRaises(ValueError):
            review.inspect_history(rows[:-1], dict(ev, history_epochs=15))

    def test_mean_of_seed_metrics_not_ensemble(self):
        keys = ('rmse', 'mae', 'r2', 'pearson', 'spearman', 'signed_bias', 'auroc_high_score',
                'ap_high_score', 'top_10_hits', 'top_25_hits', 'top_50_hits', 'top_100_hits')
        rows = [{'policy': 'main', 'partition': 'all', 'model': 'D', 'seed': seed,
                 'n': 4, 'high_score_cves': 1, **{k: v for k in keys}, 'pearson': None}
                for seed, v in [(17, .1), (29, .2), (43, .3)]]
        out = review.mean_metrics(rows)
        self.assertAlmostEqual(out[0]['rmse'], .2)
        self.assertIsNone(out[0]['pearson'])
        self.assertEqual(out, review.mean_metrics(rows[::-1]))

    def test_csv_flag_and_duplicate_guard(self):
        header = 'M_present,actual,cve,fold,model,policy,prediction,seed,target_date\n'
        row = 'False,0.1,CVE-2022-1,1,D,main,0.2,17,2022-03-01\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'predictions.csv'
            path.write_text(header+row)
            self.assertFalse(review.predictions(path)[0]['M_present'])
            path.write_text(header+row+row)
            with self.assertRaises(ValueError):
                review.predictions(path)
            path.write_text(header+row.replace('False', 'unknown'))
            with self.assertRaises(ValueError):
                review.predictions(path)


if __name__ == '__main__':
    unittest.main()
