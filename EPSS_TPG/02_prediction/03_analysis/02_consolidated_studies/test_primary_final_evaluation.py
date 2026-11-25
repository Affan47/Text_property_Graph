"""PS05 release guards, unchanged source/label contracts and inference-only baselines."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.primary_final_evaluation import protocol as p, data, evaluation, cache
from epss.primary_source_comparison import runner as sources
from epss.primary_training import runner as original
from epss.primary_study import common
from epss.consolidated.contracts import StructuredEncoder
from test_primary_training import setup_batches


def record():
    return {'cve':'CVE-2025-12345','published_date':'2025-06-01','published_timestamp':'2025-06-01T12:00:00.000',
            'epss_scores':{'2025-07-31':.12},'description':'Apache Tomcat version 9.0 has a vulnerability.',
            'social_media_posts':[],'n_references_nvd':2}


def frozen():
    return {'reserved_test_ids':['CVE-2025-12345'],
            'split_policy':{'validation_start':'2025-03-02','test_start':'2025-06-01','test_end':'2025-12-02','observation_end':'2026-09-29'},
            'source_fields':common.cs.load()['source_fields']}


class FinalEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_guard_precedes_all_full_data_access(self):
        for operation in (data.prepare,data.load,cache.prepare,evaluation.execute):
            with self.subTest(operation=operation.__name__), patch.object(p,'load',side_effect=AssertionError('Must reject before reading')):
                with self.assertRaisesRegex(ValueError,'release-test'):operation('full')
        p.access('full',True);p.access('smoke',False)

    def test_eligibility_observed_zero_and_null(self):
        r=record();r['epss_scores']['2025-07-31']=0.
        row,label,audit=data.heldout_record(r,frozen())
        self.assertEqual(label['value'],0.);self.assertEqual(row['split'],'test')
        self.assertEqual(audit['reason'],'included')
        r['epss_scores']['2025-07-31']=None
        row,label,audit=data.heldout_record(r,frozen())
        self.assertIsNone(row);self.assertIsNone(label);self.assertEqual(audit['reason'],'missing_day60_target')

    def test_unknown_ids_dates_and_observation_end(self):
        r=record();r['cve']='CVE-2025-99999'
        with self.assertRaisesRegex(ValueError,'reserved'):data.heldout_record(r,frozen())
        r=record();r['published_date']='2025-06-02'
        with self.assertRaisesRegex(ValueError,'Publication'):data.heldout_record(r,frozen())
        f=frozen();f['split_policy']['observation_end']='2025-07-01'
        row,label,a=data.heldout_record(record(),f)
        self.assertIsNone(row);self.assertEqual(a['reason'],'target_after_observation_end')

    def test_target_value_cannot_change_predictors(self):
        first=record();second=deepcopy(first);second['epss_scores']['2025-07-31']=.9
        a,la,_=data.heldout_record(first,frozen());b,lb,_=data.heldout_record(second,frozen())
        self.assertEqual(a,b);self.assertNotEqual(la['value'],lb['value'])
        a['target']=.12
        with self.assertRaisesRegex(ValueError,'predictor fields'):data.validate_input(a,'test')

    def test_disclosure_quarantine_and_missing_sources_keep_cve(self):
        r=record();r['social_media_posts']=[{'post':'EPSS score 0.99','platform':'Reddit'}]
        row,label,a=data.heldout_record(r,frozen())
        self.assertEqual(row['source_mask'],[1,0,0,0]);self.assertEqual(a['quarantined_documents'],1)
        r['description']='';row,label,_=data.heldout_record(r,frozen())
        self.assertEqual(row['source_mask'],[0,0,0,0]);self.assertTrue(label['observed'])

    def test_test_graph_plan_preserves_roles_and_matches_development_algorithm(self):
        row,_,_=data.heldout_record(record(),frozen());a=data.graph_plan([row],'test',20)
        self.assertTrue(all(e['split']=='test' for e in a['entries'].values()))
        row['split']='validation'
        b=data.graph_plan([row],'validation',20)
        old=data.chunks.plan([{k:v for k,v in row.items() if k!='structured'}],{'train':[],'validation':[row['cve']]},20)
        self.assertEqual(b,old)
        self.assertEqual(set(a['entries']),set(b['entries']))
        for k in a['entries']:
            self.assertEqual({**a['entries'][k],'split':'validation'},b['entries'][k])
        with self.assertRaisesRegex(ValueError,'role'):data.graph_plan([row],'test',20)

    def test_baselines_use_checkpoint_without_any_fit(self):
        with tempfile.TemporaryDirectory() as d:
            batches,_,settings=setup_batches(d)
            variants=original.variants_for('core')
            with patch.object(StructuredEncoder,'fit',side_effect=AssertionError('No preprocessing fit')), \
                 patch.object(original.Ridge,'fit',side_effect=AssertionError('No ridge fit')):
                for kind in ('training_mean','training_median','structured_ridge'):
                    v=next(v for v in variants if v['representation']==kind)
                    state={'kind':kind,'value':.25} if kind!='structured_ridge' else {'kind':kind,'coef':[0.]*len(batches.state['feature_names']),'intercept':1.2}
                    values,diag=evaluation.baseline_values({'baseline':state},batches,['C'],v)
                    self.assertEqual(values,[.25] if kind!='structured_ridge' else [1.])
                    self.assertEqual(diag['raw_above_one'],int(kind=='structured_ridge'))

    def test_neural_inference_unchanged_and_no_optimizer(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);batches,v,s=setup_batches(d)
            net=evaluation.batching.make_model(batches.resources,len(batches.state['feature_names']),v,s)
            checkpoint={'state_dict':net.state_dict(),'variant':v,'seed':17,'settings':s,'identity_digest':'frozen'}
            torch.save(checkpoint,root/'checkpoint.pt');common.lock(root)
            item={'directory':str(root),'variant':v,'seed':17,'run_digest':'frozen','checkpoint_sha256':common.sha256(root/'checkpoint.pt')}
            with patch.object(torch.optim,'AdamW',side_effect=AssertionError('No optimizer')):
                a=evaluation.inference(item,batches,['C'],{'settings':s},'test')
                b=evaluation.inference(item,batches,['C'],{'settings':s},'test')
            self.assertEqual(a,b);self.assertTrue(a[-1]['parameters_unchanged'])
            item['checkpoint_sha256']='wrong'
            with self.assertRaisesRegex(ValueError,'Checkpoint changed'):evaluation.inference(item,batches,['C'],{'settings':s},'test')

    def test_reload_rejects_changed_role_target_or_score(self):
        row={'cve':'CVE-2025-12345','split':'test','target_date':'2025-07-31','actual':.2,
             'predicted':.3,'source_mask':[True,False,False,False],'source_weights':[1.,0.,0.,0.]}
        evaluation.compare_predictions([row],[deepcopy(row)])
        for key,value in [('split','validation'),('actual',.3),('predicted',.9)]:
            with self.assertRaises(ValueError):evaluation.compare_predictions([row],[{**row,key:value}])

    def test_model_list_complete_and_no_seed_cherry_picking(self):
        plan=sources.plan('full')
        self.assertEqual(len(plan),27)
        for name in {r['variant']['id'] for r in plan if r['seed'] is not None}:
            self.assertEqual([r['seed'] for r in plan if r['variant']['id']==name],[17,29,43])


if __name__=='__main__':unittest.main()
