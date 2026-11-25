"""Small regression checks for the saved-artifact MS03 review."""
import unittest

import review_forecast_matched_multisource as review


class ReviewTests(unittest.TestCase):
    def test_endpoint_equality_is_not_float32_rounding(self):
        rows = [{'target': .1, 'last_observed_score': .1, 'observed_days': 31},
                {'target': .3, 'last_observed_score': .1, 'observed_days': 30},
                {'target': .1, 'last_observed_score': .2, 'observed_days': 31}]
        result = review.movement(rows)
        self.assertEqual((result['unchanged'], result['increased'], result['decreased']), (1, 1, 1))
        self.assertEqual(result['partial_recent_histories'], 1)
        self.assertAlmostEqual(result['raw_persistence_mae'], .1)

    def test_all_unchanged_has_defined_zero_error_shares(self):
        result = review.movement([{'target': .2, 'last_observed_score': .2, 'observed_days': 31}])
        self.assertEqual(result['top_error_shares'], {'3': 0., '10': 0., '20': 0.})

    def test_mask_only_correction_does_not_imply_text_variation(self):
        rows = [{'correction': 1e-6, 'source_mask': [True, False, False, False], 'source_attention': [1., 0., 0., 0.]},
                {'correction': 2e-6, 'source_mask': [True, True, False, False], 'source_attention': [.5, .5, 0., 0.]},
                {'correction': 2e-6, 'source_mask': [True, True, False, False], 'source_attention': [.5, .5, 0., 0.]}]
        result = review.corrections(rows)
        self.assertEqual(result['distinct_values'], 2)
        self.assertEqual(result['max_within_mask_group_range'], 0.)
        self.assertEqual(result['max_attention_difference_from_uniform_present_sources'], 0.)
        rows[2]['correction'] = 3e-6
        self.assertGreater(review.corrections(rows)['max_within_mask_group_range'], 0.)

    def test_parameter_norms_include_only_requested_branch(self):
        t = review.run.torch.tensor
        state = {'temporal.weight': t([3., 4.]), 'head.weight': t([12.]), 'temporal_other.weight': t([9.])}
        self.assertEqual(review.norm(state, 'temporal'), 5.)


if __name__ == '__main__':
    unittest.main()
