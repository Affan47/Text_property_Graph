"""Unit checks for H01 readiness review, without dataset access."""
from copy import deepcopy
from datetime import date, timedelta
import unittest
import review_forecast_later_anchor as review


class ReviewTests(unittest.TestCase):
    def test_exclusions_are_disjoint(self):
        rows = [{'target_observed':target,'observed_input_days':{'recent_30_60':n},
                 'matched_eligible':target and n>0} for target,n in ((True,31),(False,0),(False,31),(True,0))]
        self.assertEqual(review.eligibility_partition(rows),
                         {'eligible':1,'both_missing':1,'target_only_missing':1,'recent_input_only_missing':1})

    def test_forged_eligibility_rejected(self):
        with self.assertRaises(ValueError):
            review.eligibility_partition([{'target_observed':False,'observed_input_days':{'recent_30_60':31},
                                           'matched_eligible':True}])

    def test_masks_and_ages_independently_checked(self):
        days = [(date(2023,1,1)+timedelta(days=i)).isoformat() for i in range(61)]
        values = [None]*60+[0.0]
        x = {'publication_date':days[0],'input_end_date':days[60],
             'histories':{'recent_30_60':review.prep.history(days,values,30,60)}}
        self.assertEqual(review.check_history(x,'recent_30_60')['observed_days'],1)
        damaged = deepcopy(x)
        damaged['histories']['recent_30_60']['age_since_observation_days'][0] = 30
        with self.assertRaises(ValueError):
            review.check_history(damaged,'recent_30_60')


if __name__ == '__main__':
    unittest.main()
