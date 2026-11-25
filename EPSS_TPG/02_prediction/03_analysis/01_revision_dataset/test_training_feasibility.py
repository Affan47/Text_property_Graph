"""No model imports or training; test eligibility boundaries and missingness."""
import unittest

from assess_training_feasibility import valid, window_counts, stamp


class FeasibilityTests(unittest.TestCase):
    def test_no_future_in_input(self):
        values = [None] * 31 + [0.5] * 60
        counts = window_counts(values, 30, 30)
        self.assertFalse(counts["endpoint_and_any_input"])
        self.assertTrue(counts["all_future_days"])

    def test_partial_past_kept(self):
        values = [None] + [0.0] * 90
        self.assertTrue(window_counts(values, 30, 30)["endpoint_and_any_input"])
        self.assertTrue(window_counts(values, 30, 30)["partial_input_with_usable_endpoint"])

    def test_missing_middle_future_blocks_extrema_not_endpoint(self):
        values = [0.2] * 91
        values[45] = None
        c = window_counts(values, 30, 30)
        self.assertTrue(c["endpoint_and_any_input"])
        self.assertFalse(c["minmax_and_any_input"])

    def test_missing_anchor_blocks_exact_change_not_endpoint(self):
        values = [0.2] * 91
        values[30] = None
        c = window_counts(values, 30, 30)
        self.assertTrue(c["endpoint_and_any_input"])
        self.assertFalse(c["direction_or_delta_and_anchor"])

    def test_no_scores_is_not_zero_series(self):
        c = window_counts([None] * 91, 30, 30)
        self.assertFalse(c["endpoint_and_any_input"])
        self.assertFalse(c["minmax_and_any_input"])

    def test_saved_boundary(self):
        values = [0.0] * 91
        self.assertTrue(window_counts(values, 60, 30)["within_saved_window"])
        self.assertFalse(window_counts(values, 30, 90)["within_saved_window"])

    def test_finite_scores(self):
        self.assertTrue(valid(0.0))
        for v in (True, None, "0.2", float("nan"), float("inf")):
            self.assertFalse(valid(v))

    def test_date_parsing_explicit_assumption(self):
        self.assertEqual(stamp("2025-01-01T00:00:00"), stamp("2025-01-01T01:00:00+01:00"))


if __name__ == "__main__":
    unittest.main()
