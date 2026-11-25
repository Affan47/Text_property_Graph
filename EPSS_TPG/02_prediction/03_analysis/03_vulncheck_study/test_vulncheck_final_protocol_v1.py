"""Final freeze selection, exact cohort, inventory and no-scoring contracts."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.vulncheck_study import final_protocol_v1 as final


class FinalProtocolTests(unittest.TestCase):
    def test_selection_uses_all_seed_mean_ap_not_best_f1_or_seed(self):
        metrics = {}
        for family, values in [('A', [.9, .1, .1]), ('B', [.5, .5, .5])]:
            for seed, ap in zip([17, 29, 43], values):
                metrics[f'{family}__seed_{seed}'] = {'average_precision': ap, 'f1': 1 if family == 'A' else 0}
        self.assertEqual(final.choose_family(metrics, [17, 29, 43])[0], 'B')

    def test_exact_ties_are_deterministic(self):
        metrics = {f'{f}__seed_{s}': {'average_precision': .5} for f in ('Z', 'A') for s in (17, 29, 43)}
        self.assertEqual(final.choose_family(metrics, [17, 29, 43])[0], 'A')

    def test_missing_extra_or_undefined_seed_rejected(self):
        metrics = {f'A__seed_{s}': {'average_precision': .5} for s in (17, 29, 43)}
        missing = deepcopy(metrics); missing.pop('A__seed_17')
        extra = {**metrics, 'A__seed_99': {'average_precision': .9}}
        undefined = deepcopy(metrics); undefined['A__seed_17']['average_precision'] = None
        for invalid in (missing, extra, undefined):
            with self.assertRaises(ValueError):
                final.choose_family(invalid, [17, 29, 43])

    def test_exact_cohort_not_just_count(self):
        manifest = {c: {'cve': c, 'split': s} for c, s in [('a','train'), ('b','validation'), ('c','test'), ('d','excluded')]}
        counts = {'train':1, 'validation':1, 'test':1, 'excluded':1}
        result = final.validate_cohort(manifest, {'train':['a'], 'validation':['b']}, counts)
        self.assertEqual(result['test'], ['c'])
        swapped = deepcopy(manifest)
        swapped['a']['split'], swapped['c']['split'] = 'test', 'train'
        with self.assertRaises(ValueError):
            final.validate_cohort(swapped, {'train':['a'], 'validation':['b']}, counts)

    def test_inventory_and_control_cutoff_protection(self):
        items = []
        for kind, n in [('source_neural',24), ('edge_free',3), ('logistic',8), ('constant',1)]:
            for i in range(n):
                items.append({'model': kind+str(i), 'kind': kind, 'fixed_threshold': .5,
                    'selected_threshold': .7 if kind == 'source_neural' else None,
                    'checkpoint': None if kind == 'constant' else 'saved.pt',
                    'checkpoint_sha256': None if kind == 'constant' else 'a'*64})
        final.validate_inventory(items)
        bad = deepcopy(items); bad[-2]['selected_threshold'] = .2
        with self.assertRaises(ValueError):
            final.validate_inventory(bad)
        with self.assertRaises(ValueError):
            final.validate_inventory(items[:-1])

    def test_frozen_config_disables_execution_and_independence_claims(self):
        cfg = final.data.read(final.CONFIG)
        self.assertTrue(cfg['execution']['freeze_only'])
        self.assertFalse(cfg['execution']['training_enabled'])
        self.assertFalse(cfg['execution']['test_scoring_enabled'])
        self.assertFalse(cfg['predictors']['historical_availability_claim'])
        self.assertEqual(cfg['selection']['seeds'], [17,29,43])
        self.assertEqual(cfg['cohort']['counts']['test'], 651)
        self.assertTrue(cfg['decision_rules']['no_evaluation_threshold_search'])

    def test_output_lock_reproduction_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'freeze'
            artifacts = {'protocol.json': {'test_scoring': False}}
            final.threshold.save_report(output, artifacts, False)
            final.threshold.save_report(output, artifacts, True)
            with self.assertRaises(ValueError):
                final.threshold.save_report(output, artifacts, False)
            (output/'protocol.json').write_text('{}')
            with self.assertRaises(ValueError):
                final.threshold.save_report(output, artifacts, True)


if __name__ == '__main__':
    unittest.main()
