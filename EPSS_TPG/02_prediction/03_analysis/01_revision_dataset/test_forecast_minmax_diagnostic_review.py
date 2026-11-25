"""Synthetic review checks; no real-dataset fitting or test predictions."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_forecast_minmax import evidence_bundle, task
from epss import forecast_minmax_sequence as gru
from epss.forecast_data import read_json, sha256, write_json
import diagnose_forecast_minmax_sequence as diagnostic
import review_forecast_minmax_diagnostic as review


class DiagnosticReviewTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        budget = patch.dict(gru.CONFIG, {'epochs': 2, 'patience': 2})
        budget.start()
        self.addCleanup(budget.stop)
        self.output = self.root/'diagnostic'
        with redirect_stdout(io.StringIO()):
            args = evidence_bundle(self.root)
            task.run(*args, self.root/'linear')
            gru.run(*args, self.root/'linear', self.root/'gru')
            diagnostic.run(self.root/'gru', self.output)

    def test_review_reproduces_without_updates_or_writes(self):
        before = {str(p): sha256(p) for p in self.root.rglob('*') if p.is_file()}
        rng = torch.get_rng_state().clone()
        with patch.object(gru, 'fit_seed', side_effect=AssertionError('Must not fit')), \
             patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Must not update')):
            result = review.review(self.output)
        self.assertTrue(result['checks_passed'])
        self.assertEqual(result['cohort_counts'], {'train': 25, 'validation': 1})
        self.assertEqual(result['artifact_row_counts']['per_cve'], 78)
        self.assertFalse(result['test_scoring_performed'])
        self.assertEqual(before, {str(p): sha256(p) for p in self.root.rglob('*') if p.is_file()})
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertIn('not actual CVEs', review.explanation(result))

    def test_detects_modified_output(self):
        write_json(self.output/'per_cve.json', [])
        with self.assertRaisesRegex(ValueError, 'Changed artifact'):
            review.review(self.output)

    def test_rehashing_wrong_derivative_does_not_pass(self):
        rows = read_json(self.output/'per_cve.json')
        rows[0]['d_minimum_d_a'] += .1
        write_json(self.output/'per_cve.json', rows)
        lock = read_json(self.output/'analysis_lock.json')
        lock['files']['per_cve.json'] = sha256(self.output/'per_cve.json')
        write_json(self.output/'analysis_lock.json', lock)
        with self.assertRaisesRegex(ValueError, 'does not reproduce'):
            review.review(self.output)

    def test_independent_derivatives_and_aggregates_detect_changes(self):
        artifacts = {k: read_json(self.output/f'{k}.json')
                     for k in ('per_cve', 'perturbations', 'feature_spread')}
        review.independent_checks(artifacts)
        bad = copy.deepcopy(artifacts)
        bad['per_cve'][0]['d_maximum_d_b'] += .1
        with self.assertRaisesRegex(ValueError, 'derivative differs'):
            review.independent_checks(bad)
        bad = copy.deepcopy(artifacts)
        bad['perturbations'][0]['mean_abs_maximum_change'] += .1
        with self.assertRaisesRegex(ValueError, 'Response mean differs'):
            review.independent_checks(bad)


if __name__ == '__main__':
    unittest.main()
