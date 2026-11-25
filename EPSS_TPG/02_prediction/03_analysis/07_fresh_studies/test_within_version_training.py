import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
ROOT = Path(__file__).resolve().parents[3]
for rel in ('01_tpg/01_core', '01_tpg/10_extensions', '02_prediction/01_package', '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT/rel))

import numpy as np
from epss.within_version_training_v1 import protocol, reporting, runner
from epss.within_version_pooled_v1 import data


class ControlAndExecutionTests(unittest.TestCase):
    def test_constants_are_optimization_mean_and_median(self):
        values = reporting.constants([0, .2, 1])
        self.assertAlmostEqual(values['optimization_mean'], .4)
        self.assertEqual(values['optimization_median'], .2)
        for bad in ([], [None], [float('nan')], [2], [-.1]):
            with self.assertRaises((TypeError, ValueError)):
                reporting.constants(bad)

    def test_settings_preserve_frozen_parameters(self):
        p = protocol.settings(); old = data.settings()
        self.assertEqual(p['threads_per_fit'], 4)
        self.assertFalse(p['test_access']); self.assertFalse(p['official_validation_access'])
        for key in ('hidden', 'batch_size', 'learning_rate', 'weight_decay', 'epochs', 'patience', 'seeds'):
            self.assertEqual(p[key], old[key])
        self.assertEqual(runner.parameters('smoke')['epochs'], 2)
        self.assertEqual(runner.parameters('full')['epochs'], 100)

    def test_all_arm_control_contrasts_declared(self):
        pairs = reporting.comparisons(protocol.settings())
        self.assertEqual(len(pairs), 9)
        self.assertEqual(set(pairs), {('DM', 'D'), ('DM', 'DF'), ('DF', 'D')} |
                         {(a, c) for a in ('D', 'DF', 'DM') for c in ('optimization_mean', 'optimization_median')})

    def test_shared_bootstrap_exact_old_procedure(self):
        rng = np.random.default_rng(123)
        errors = {arm: rng.random((3, 12)) for arm in ('D', 'DF', 'DM')}
        errors['optimization_mean'] = np.tile(rng.random(12), (3, 1))
        errors['optimization_median'] = np.tile(rng.random(12), (3, 1))
        pairs = reporting.comparisons(protocol.settings())
        for groups in ([str(i) for i in range(12)], ['a']*7+['b']*3+['c']*2):
            rows = reporting.paired_many(errors, groups, pairs, 75, 6029)
            for (a, b), row in zip(pairs, rows):
                expected = reporting.old_report.paired_ci(errors[a], errors[b], groups, 75, 6029)
                self.assertEqual({k: row[k] for k in expected}, expected)

    def test_bootstrap_matched_row_permutation(self):
        a = np.array([[1., 2., 3., 4.]]*3); b = a+.1
        x = reporting.paired_many({'A': a, 'B': b}, ['a', 'b', 'c', 'd'], [('A', 'B')], 30, 6029)
        order = [3, 1, 0, 2]
        y = reporting.paired_many({'A': a[:, order], 'B': b[:, order]}, [str('abcd'[i]) for i in order], [('A', 'B')], 30, 6029)
        for key in ('low', 'high', 'difference'):
            self.assertAlmostEqual(x[0][key], y[0][key], places=14)

    def test_invalid_bootstrap_inputs_rejected(self):
        with self.assertRaises(ValueError):
            reporting.paired_many({'A': np.zeros((3, 2)), 'B': np.ones((3, 2))}, ['x'], [('A', 'B')], 20, 1)

    def test_relative_gain_does_not_pass_constant_gate(self):
        settings = {**protocol.settings(), 'policies': ['main'], 'interval_partitions': ['all']}
        intervals = [{'policy': 'main', 'partition': 'all', 'unit': unit, 'contrast': a+'_minus_'+b,
                      'high': -.1 if b in ('D', 'DF') else .1}
                     for a, b in reporting.comparisons(settings) for unit in settings['bootstrap_units']]
        result = reporting.outcome_gates(intervals, settings)[0]
        self.assertTrue(result['DM_vs_D_both_units'])
        self.assertFalse(result['useful_estimation_reporting_gate'])
        self.assertFalse(result['content_specific_reporting_gate'])

    def test_both_interval_units_and_both_controls_required(self):
        settings = {**protocol.settings(), 'policies': ['main'], 'interval_partitions': ['all']}
        rows = [{'policy': 'main', 'partition': 'all', 'unit': unit, 'contrast': a+'_minus_'+b, 'high': -.1}
                for a, b in reporting.comparisons(settings) for unit in settings['bootstrap_units']]
        self.assertTrue(reporting.outcome_gates(rows, settings)[0]['content_specific_reporting_gate'])
        rows = [r for r in rows if not (r['contrast'] == 'DM_minus_optimization_mean' and r['unit'] == 'evidence_component')]
        self.assertFalse(reporting.outcome_gates(rows, settings)[0]['useful_estimation_reporting_gate'])

    def test_smoke_and_full_interval_counts(self):
        p = protocol.settings()
        self.assertEqual(p['bootstrap_repetitions'], 20000)
        self.assertEqual(p['smoke']['bootstrap_repetitions'], 20)

    def test_older_adapter_stays_disabled(self):
        self.assertFalse(data.settings()['full_training_enabled'])
        self.assertFalse(data.settings()['older_v3_training_enabled'])

    def test_between_fit_source_changes_rejected(self):
        frozen = {'source_pins': {'code': 'original'}, 'runtime': {'threads': 4}, 'input_hashes': {}}
        with patch.object(protocol, 'pins', return_value={'code': 'edited'}), \
             patch.object(data.old, 'runtime', return_value={'threads': 4}):
            with self.assertRaisesRegex(ValueError, 'between fits'):
                runner.assert_frozen_identity(frozen)


if __name__ == '__main__':
    unittest.main()
