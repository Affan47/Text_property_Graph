"""PS04 exact fit selection, provenance isolation, pairing and source masks."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss.primary_source_comparison import runner, reporting
from epss.primary_study import common
from epss.primary_training import runner as original
from epss.consolidated.evaluation import score_predictions
from test_primary_training import setup_batches


class PrimarySourceComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_full_fit_plan_and_reuse(self):
        plan = runner.plan('full')
        self.assertEqual(len(plan), 27)
        self.assertEqual(len({r['key'] for r in plan}), 27)
        new = [r for r in plan if r['origin'] == 'ps04_new']
        self.assertEqual(len(new), 12)
        self.assertEqual({r['seed'] for r in new}, {17, 29, 43})
        self.assertEqual({r['variant']['id'].split('_')[1] for r in new}, {'DM', 'DGN'})
        self.assertTrue(all(r['variant']['structured_blocks'] == ['cvss', 'reference_count'] for r in new))
        self.assertTrue(all(r['variant']['edge_view'] == 'no_coref' for r in new if r['variant']['representation'] == 'security_tpg'))

    def test_smoke_and_full_use_same_variants_not_same_budget(self):
        smoke, full = runner.plan('smoke'), runner.plan('full')
        self.assertEqual(len(smoke), 11)
        self.assertEqual({r['variant']['id'] for r in smoke}, {r['variant']['id'] for r in full})
        self.assertEqual({r['seed'] for r in smoke}, {None, 17})
        self.assertEqual(original.settings_for('smoke')['epochs'], 2)
        self.assertEqual(original.settings_for('full')['epochs'], 100)

    def test_output_resolution_never_writes_into_old_fit(self):
        for item in runner.plan('full'):
            path = runner.fit_directory(Path('/new'), Path('/original'), item)
            self.assertEqual(path.parts[1], 'original' if item['origin'] == 'ps02_reused' else 'new')

    def test_reused_fit_cannot_be_trained(self):
        item = next(r for r in runner.plan('smoke') if r['origin'] == 'ps02_reused')
        with patch.object(original, 'train_fit', side_effect=AssertionError('No reuse training')):
            with self.assertRaisesRegex(ValueError, 'Refuse training'):
                runner.finish_new_fit(None, None, item, None, None, None)

    def test_identity_changes_and_test_roles_rejected(self):
        good = {'ids': {'train': ['A'], 'validation': ['B']}, 'settings': {'epochs': 100}, 'test_scoring': False}
        runner.validate_source_identity(good, deepcopy(good))
        bad = deepcopy(good); bad['settings']['epochs'] = 10
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            runner.validate_source_identity(good, bad)
        bad = deepcopy(good); bad['ids']['test'] = ['C']
        with self.assertRaisesRegex(ValueError, 'development roles'):
            runner.validate_source_identity(bad, bad)
        bad = deepcopy(good); bad['ids']['validation'] = ['A']
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            runner.validate_source_identity(bad, bad)

    def test_source_masks_and_structured_inputs_match(self):
        with tempfile.TemporaryDirectory() as d:
            batches, _, _ = setup_batches(d)
            for c in batches.rows:
                batches.rows[c]['source_mask'] = [1, 1, 1, 1]
                for i, source in enumerate(('github_summary', 'non_github_summary', 'smp')):
                    batches.index[c]['documents'].append({'source_text_type': source,
                        'chunks': [{'has_graph': True, 'tensor': str(10+i)}]})
            structured = None
            for item in runner.plan('smoke'):
                v = item['variant']
                if item['seed'] is None:
                    continue
                plan = batches.plan(['A', 'B', 'C'], v)
                mask = [s in v['sources'] for s in common.cs.SOURCES]
                self.assertEqual(plan['source_mask'].tolist(), [mask]*3)
                if structured is None:
                    structured = plan['structured']
                self.assertTrue(torch.equal(plan['structured'], structured))
                self.assertNotIn('target', plan)

    def test_planned_pair_directions(self):
        specs = reporting.comparison_specs()
        self.assertEqual(len(specs), 12)
        pairs = {(s['left'], s['right']) for s in specs}
        self.assertIn(('P01_DM_security_tpg_S1', 'P01_D_security_tpg_S1'), pairs)
        self.assertIn(('P01_DGNM_security_tpg_S1', 'P01_DGN_security_tpg_S1'), pairs)
        self.assertIn(('P01_DGN_pooled_text_S1', 'P01_D_pooled_text_S1'), pairs)

    def fixtures(self, root):
        plan = runner.plan('smoke')
        identity = {'scale': 'smoke', 'ids': {'train': ['A'], 'validation': ['B', 'C']},
                    'settings': original.settings_for('smoke'), 'protocol': runner.protocol(), 'test_scoring': False}
        locations = {}
        for item in plan:
            out = root / item['key']; out.mkdir()
            p = .3 if '_DM_' in item['key'] else .4
            rows = [{'cve': c, 'split': 'validation', 'target_date': '2024-01-01', 'actual': y, 'predicted': p}
                    for c, y in [('B', .1), ('C', .2)]]
            common.write(out / 'predictions.json', rows)
            common.write(out / 'metrics.json', score_predictions([.1, .2], [p, p]))
            common.lock(out)
            locations[item['key']] = out
        return locations, plan, identity

    def test_report_pairs_same_targets_and_cve_balancing(self):
        with tempfile.TemporaryDirectory() as d:
            locations, plan, identity = self.fixtures(Path(d))
            result = reporting.aggregate(locations, plan, identity)
            self.assertEqual(result, reporting.aggregate(locations, plan, identity))
            self.assertEqual(len(result['paired_comparisons']), 20)
            p = next(p for p in result['paired_comparisons'] if p['left'] == 'P01_DM_security_tpg_S1'
                     and p['right'] == 'P01_D_security_tpg_S1')
            self.assertAlmostEqual(p['mean_mae_difference'], -.1)
            self.assertEqual(p['n_cves'], 2)
            key = plan[1]['key']; rows = common.read(locations[key] / 'predictions.json')
            rows[0]['target_date'] = '2024-02-01'; common.write(locations[key] / 'predictions.json', rows)
            with self.assertRaisesRegex(ValueError, 'target values or dates'):
                reporting.aggregate(locations, plan, identity)

    def test_report_rejects_missing_seed_and_test_rows(self):
        with tempfile.TemporaryDirectory() as d:
            locations, plan, identity = self.fixtures(Path(d))
            identity['settings']['seeds'] = [17, 29]
            with self.assertRaisesRegex(ValueError, 'seed group'):
                reporting.aggregate(locations, plan, identity)
            identity['settings']['seeds'] = [17]
            out = locations[plan[0]['key']]; rows = common.read(out / 'predictions.json')
            rows[0]['split'] = 'test'; common.write(out / 'predictions.json', rows)
            with self.assertRaisesRegex(ValueError, 'report cohort'):
                reporting.aggregate(locations, plan, identity)


if __name__ == '__main__':
    unittest.main()
