import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('flag_review', Path(__file__).with_name('13_verify_flag_checkpoints.py'))
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class HistoryTests(unittest.TestCase):
    def rows(self):
        return [{'epoch': i + 1, 'optimization_mse': .01, 'selection_rmse': score, 'patience': stale}
                for i, (score, stale) in enumerate(((.2, 0), (.1, 0), (.1, 1), (.12, 2)))]

    def test_earliest_strict_best_with_ties(self):
        self.assertEqual(review.check_history(self.rows(), 2, patience=2), .1)

    def test_reject_later_tie_checkpoint(self):
        with self.assertRaises(ValueError):
            review.check_history(self.rows(), 3, patience=2)

    def test_reject_incomplete_history(self):
        with self.assertRaises(ValueError):
            review.check_history(self.rows()[:3], 2, patience=2)

    def test_accept_epoch_budget_stop(self):
        self.assertEqual(review.check_history(self.rows()[:3], 2, max_epochs=3, patience=2), .1)

    def test_reject_resetting_patience_on_tie(self):
        rows = self.rows()
        rows[2]['patience'] = 0
        with self.assertRaises(ValueError):
            review.check_history(rows, 2, patience=2)


if __name__ == '__main__':
    unittest.main()
