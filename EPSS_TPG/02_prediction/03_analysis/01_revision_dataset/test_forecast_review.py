"""Post-run review checks only; these tests do not train models."""
import unittest

from review_forecast_sequence import metrics, verify_history


def history(values):
    return [{'epoch': i+1, 'mean_training_huber_loss': .01,
             'training_mae': .02, 'validation_mae': value} for i,value in enumerate(values)]


class ReviewTests(unittest.TestCase):
    def test_patience_stop(self):
        result = verify_history(history([.1,.2,.3]), 1, {'epochs': 10, 'patience': 2})
        self.assertEqual(result['stop_reason'], 'patience_exhausted')

    def test_epoch_limit(self):
        result = verify_history(history([.3,.2,.1]), 3, {'epochs': 3, 'patience': 2})
        self.assertEqual(result['stop_reason'], 'epoch_limit')

    def test_ties_keep_first_epoch(self):
        verify_history(history([.1,.1,.1]), 1, {'epochs': 10, 'patience': 2})
        with self.assertRaises(ValueError):
            verify_history(history([.1,.1,.1]), 3, {'epochs': 10, 'patience': 2})

    def test_unexplained_or_late_termination_rejected(self):
        for values in ([.1,.2], [.1,.2,.3,.4]):
            with self.assertRaises(ValueError):
                verify_history(history(values), 1, {'epochs': 10, 'patience': 2})

    def test_nonfinite_history_rejected(self):
        with self.assertRaises(ValueError):
            verify_history(history([float('nan')]), 1, {'epochs': 1, 'patience': 2})

    def test_metric_scale_and_undefined_correlation(self):
        self.assertAlmostEqual(metrics([.1,.1],[.2,.3])['mae'], .15)
        self.assertIsNone(metrics([.1,.1],[.2,.3])['spearman_rho'])


if __name__ == '__main__':
    unittest.main()
