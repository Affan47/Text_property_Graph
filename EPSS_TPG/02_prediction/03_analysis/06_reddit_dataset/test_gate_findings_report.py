"""Checks for report arithmetic; no models or network access."""
from copy import deepcopy
import unittest

from audit_reddit import read
from build_gate_findings_report import AUDIT, RESULTS, REVIEW, reconcile, document


class GateFindingsTests(unittest.TestCase):
    def setUp(self):
        self.review = read(REVIEW/'review.json')
        self.initial = read(AUDIT/'findings.json')
        self.recovery = read(RESULTS/'stage1_epss/report.json')

    def test_totals_and_denominators(self):
        d = reconcile(self.review,self.initial,self.recovery)
        self.assertEqual(d['post_launch_day60_observed'],34921)
        self.assertEqual(d['post_launch_day60_not_observed'],192)
        self.assertEqual(d['post_launch_without_any_score'],118)
        self.assertEqual(d['other_partial_histories'],11139)
        self.assertEqual(d['component_non_singleton_cves'],10047)

    def test_refuse_wrong_component_totals(self):
        bad=deepcopy(self.review); bad['component_size_distribution']['1']+=1
        with self.assertRaises(ValueError):
            reconcile(bad,self.initial,self.recovery)

    def test_refuse_stale_no_recovery_interpretation(self):
        bad=deepcopy(self.recovery); bad['counts']['recovered_scores']=1
        with self.assertRaises(ValueError):
            reconcile(self.review,self.initial,bad)

    def test_refuse_inconsistent_daily_total(self):
        bad=deepcopy(self.initial); bad['epss_status_counts']['DOWNLOAD_ERROR']-=1
        with self.assertRaises(ValueError):
            reconcile(self.review,bad,self.recovery)


if __name__=='__main__':
    unittest.main()
