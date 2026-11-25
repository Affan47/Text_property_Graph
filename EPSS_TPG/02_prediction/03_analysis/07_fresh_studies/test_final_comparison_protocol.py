"""Final-run role/fit/resource/release tests; synthetic data, no dataset fits."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[name]='4'
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
for rel in ('01_tpg/01_core','01_tpg/10_extensions','02_prediction/01_package','01_tpg/02_examples/01_scripts'):
    sys.path.insert(0,str(ROOT/rel))
from epss.final_comparison_v1 import data, protocol
from epss.final_comparison_v1.common import MODELS, read, write


class FinalProtocolTests(unittest.TestCase):
    def test_fixed_settings(self):
        p=protocol.settings()
        self.assertEqual(p['seeds'],[17,29,43,59,71])
        self.assertEqual(p['scientific_fits'],{'run1':75,'run2':25})
        self.assertEqual(len(p['primary_contrasts']),6)
        self.assertFalse(p['test_access_default'])
        self.assertTrue(p['no_further_experiments_after_release'])

    def test_run1_cannot_adapt_official_validation_or_test(self):
        p={'official_ids':{'train':['a'],'validation':['v'],'test':['t']}}
        data.validate_roles({'train':['a'],'validation':[],'assessment':[]},p,'run1')
        for cid in ('v','t'):
            with self.assertRaisesRegex(ValueError,'Test or unassigned'):
                data.validate_roles({'train':['a'],'validation':[cid],'assessment':[]},p,'run1')

    def test_run2_still_forbids_test_and_duplicate_ids(self):
        p={'official_ids':{'train':['a'],'validation':['v'],'test':['t']}}
        data.validate_roles({'train':['a'],'validation':['v'],'assessment':[]},p,'run2')
        with self.assertRaises(ValueError):data.validate_roles({'train':['a'],'validation':['t'],'assessment':[]},p,'run2')
        with self.assertRaises(ValueError):data.validate_roles({'train':['a'],'validation':['a'],'assessment':[]},p,'run2')

    def test_version_uses_target_date(self):
        schedule={'v1':'2021-04-14','v2':'2022-02-04','v3':'2023-03-07'}
        self.assertEqual(data.version_on('2022-02-03',schedule),'v1')
        self.assertEqual(data.version_on('2022-02-04',schedule),'v2')
        with self.assertRaises(ValueError):data.version_on('2020-01-01',schedule)

    def test_ram_estimate_deduplicates_same_graph(self):
        index={c:{'views':{'DM':{'object_key':'k'}}} for c in ('a','b')}
        records={'k':{'statistics':{'nodes':3,'edges':2}}}
        result=data.estimate_view_bytes(index,records,{'node_feature_dim':10,'edge_feature_dim':4},['a','b'],'DM')
        self.assertEqual(result['graphs'],1)
        self.assertEqual(result['resident_tensor_bytes'],3*(40+18)+2*(16+24))

    def test_roles_frozen_without_changing_assessment(self):
        roles={'train':['a'],'validation':['b'],'assessment':['c']}
        p={'folds':{'1':roles},'official_ids':{'train':['a','b','c'],'validation':['v'],'test':['t']}}
        self.assertEqual(data.selected_folds(p,'run1'),{'1':roles})
        self.assertEqual(data.selected_folds(p,'run2')['0']['assessment'],[])

    def test_all_models_and_seeds_are_required(self):
        from epss.final_comparison_v1 import runner
        p={'settings':protocol.settings(),'folds':{str(i):{} for i in (1,2,3)},
           'official_ids':{'train':[],'validation':[],'test':[]}}
        self.assertEqual(len(runner.specs(p,'run1')),75)
        self.assertEqual(len(runner.specs(p,'run2')),25)
        self.assertEqual(len(runner.specs(p,'run1',True)),5)
        self.assertEqual({m for _,m,_ in runner.specs(p,'run1')},set(MODELS))

    def test_release_permanently_blocks_training_before_other_reads(self):
        from epss.final_comparison_v1 import runner
        with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'OUT',Path(tmp)):
            write(Path(tmp)/'test_evaluation_release.json',{})
            with self.assertRaisesRegex(ValueError,'permanently closed'):runner.guard({},'run1')

    def test_six_contrasts_must_not_promote_test_direction_alone(self):
        from epss.final_comparison_v1 import runner, reporting
        rows=[{'partition':'all','unit':'evidence_group_and_seed','contrast':a+'_minus_'+b,
               'family_low':-.004,'family_high':-.001,'difference':-.002} for a,b in reporting.CONTRASTS]
        with patch.object(runner,'read',return_value={'intervals':rows}),patch.object(runner.publication,'check_lock'):
            result=runner.cross_run_decisions({'intervals':[{**r,'family_high':.001} for r in rows]})
            opposite=runner.cross_run_decisions({'intervals':[{**r,'difference':.001} for r in rows]})
        self.assertEqual(len(result['decisions']),6)
        self.assertEqual(result['decisions'][0]['claim'],'development_support_directionally_replicated_only')
        self.assertFalse(result['further_training_allowed'])
        self.assertTrue(all(r['claim']=='improvement_not_established_by_frozen_rule' for r in opposite['decisions']))

    def test_primary_interval_filter_matches_actual_report_api(self):
        from epss.final_comparison_v1 import runner, reporting
        rows=[]
        for model in reporting.MODELS:
            for cid,y in [('a',.01),('b',.3)]:
                for seed in ([0] if model.startswith('training_') else [17]):
                    rows.append({'cve':cid,'fold':1,'seed':seed,'model':model,'target':y,'prediction':.05,
                                 'group':cid,'has_posts':cid=='b','target_version':'v2'})
        values=reporting.generate(rows,{'seeds':[17],'bootstrap_repetitions':20,'progress':False})
        self.assertEqual(len(runner.primary_intervals(values)),6)
        with tempfile.TemporaryDirectory() as tmp:
            runner.figures(Path(tmp),values)
            self.assertTrue((Path(tmp)/'paired_rmse.png').exists())
        for intervals in ([],values['intervals'][1:]):
            with self.assertRaisesRegex(ValueError,'All six'):runner.primary_intervals({'intervals':intervals})

    def test_canonical_csv_complex_columns(self):
        from epss.final_comparison_v1 import runner
        text=runner.csv_text([{'b':{'x':2,'a':1},'a':'value'}])
        self.assertNotIn('\r',text)
        self.assertTrue(text.startswith('a,b\n'))

    def test_evaluation_checks_ram_before_authorizing_labels(self):
        from epss.final_comparison_v1 import runner
        with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'OUT',Path(tmp)),\
             patch.object(runner.protocol,'load',return_value={}),\
             patch.object(runner.test_cache,'prepare',return_value={}),\
             patch.object(runner,'test_resources',side_effect=ValueError('RAM insufficient')),\
             patch.object(runner.test_cache,'evaluation_release') as release:
            with self.assertRaisesRegex(ValueError,'RAM insufficient'):runner.evaluate(release_test=True)
            release.assert_not_called()

    def test_evaluation_verify_cannot_create_missing_release_marker(self):
        from epss.final_comparison_v1 import runner
        with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'OUT',Path(tmp)),\
             patch.object(runner.protocol,'load',return_value={}),\
             patch.object(runner,'completed',return_value=True),\
             patch.object(runner.test_cache,'prepare',return_value={}) as prepare,\
             patch.object(runner,'test_resources'),\
             patch.object(runner.test_cache,'evaluation_release',side_effect=ValueError('marker missing')) as release:
            with self.assertRaisesRegex(ValueError,'marker missing'):runner.evaluate(release_test=True,verify=True)
            self.assertTrue(prepare.call_args.kwargs['verify_only'])
            self.assertTrue(release.call_args.kwargs['verify_only'])

    def test_masked_test_graph_has_no_tensor_size_but_keeps_cve(self):
        result=data.estimate_view_bytes({'a':{'views':{'D':{'object_key':None}}}}, {}, {}, ['a'],'D')
        self.assertEqual(result['resident_tensor_bytes'],0)


if __name__=='__main__':unittest.main()
