"""Synthetic tests for final release, predictor isolation and frozen reporting."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
import numpy as np
from epss.vulncheck_study import final_evaluation_data_v1 as prep
from epss.vulncheck_study import final_evaluation_report_v1 as report
from epss.vulncheck_study import final_evaluation_v1 as runner
from epss.vulncheck_study import classifier_smoke_v2 as legacy
from epss.primary_final_evaluation import data as prior
from epss.consolidated import contracts


def example(role='test'):
    text = 'A buffer overflow affects Example Server before version 2.0.'
    return {'cve': 'CVE-2025-12345', 'split': role, 'structured': [0.]*29, 'source_mask': [1,0,0,0],
            'documents': [{'document_id': 'description:verified', 'source_text_type': 'description', 'text': text}]}


class FinalEvaluationTests(unittest.TestCase):
    def test_explicit_release(self):
        with self.assertRaises(ValueError): prep.access('full', False)
        prep.access('full', True); prep.access('smoke', False)

    def test_original_guard_remains_closed(self):
        with self.assertRaises(ValueError): legacy.check_rows([example()])
        prep.validate(example(), 'test')

    def test_no_labels_or_ep_ss_in_predictors(self):
        for key in ('target', 'in_vckev', 'epss', 'decisions'):
            row = example(); row[key] = 1
            with self.assertRaises(ValueError): prep.validate(row, 'test')

    def test_roles_masks_dimensions_and_duplicates(self):
        invalid = []
        row=example(); row['split']='train'; invalid.append(row)
        row=example(); row['source_mask']=[1,1,0,0]; invalid.append(row)
        row=example(); row['structured'][0]=float('nan'); invalid.append(row)
        row=example(); row['structured'].pop(); invalid.append(row)
        row=example(); row['documents']*=2; invalid.append(row)
        for row in invalid:
            with self.assertRaises(ValueError): prep.validate(row, 'test')

    def test_chunk_planner_matches_original_evaluation(self):
        row=example(); original=deepcopy(row)
        original['structured']={k:None for k in (*contracts.cvss.CVSS_FIELDS, 'n_references_nvd')}
        self.assertEqual(prep.graph_plan([row], 'test', 32), prior.graph_plan([original], 'test', 32))
        self.assertEqual(row, example())

    def test_missing_label_fails_before_transform(self):
        with self.assertRaisesRegex(ValueError, 'Missing/nonboolean'):
            prep.convert({'cve':'CVE-2025-12345', 'in_vckev':None}, 'test', {}, {})

    def test_ten_calibration_bins_include_endpoints(self):
        scores=[0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.]
        m=report.metrics([0]*5+[1]*6, scores, .5)
        self.assertEqual(len(m['calibration']),10)
        self.assertEqual(sum(b['n'] for b in m['calibration']),11)
        self.assertEqual(m['calibration'][-1]['n'],2)
        self.assertEqual(m['confusion_matrix_true_rows_predicted_columns'],[[5,0],[0,6]])

    def test_undefined_metrics_are_explicit(self):
        m=report.metrics([0,0],[0.,0.],.5)
        for key in ('precision','recall','f1','average_precision','roc_auc','balanced_accuracy'):
            self.assertIsNone(m[key]); self.assertIn(key,m['undefined_reasons'])

    def test_equal_cutoff_predicts_positive(self):
        self.assertEqual(report.metrics([0,1],[.49,.5],.5)['f1'],1.)

    def test_overlap_does_not_merge_cves(self):
        a=example('validation'); b=example('train'); b['cve']='CVE-2025-54321'
        audit=prep.overlap([a],[b])
        self.assertEqual(audit['shared_with_training'],[a['cve']])
        self.assertEqual(len(audit['matches']),1)

    def test_bootstrap_identical_predictions_zero_and_repeatable(self):
        inventory=[]; predictions={}
        for seed in (17,29,43):
            for rep in ('security_tpg','pooled_text'):
                name=f'DGNM__{rep}__seed_{seed}'
                inventory.append({'model':name,'selected_threshold':.4})
                predictions[name]=[.1,.8,.3,.9]
        a=report.paired_bootstrap([0,1,0,1],predictions,inventory,10,8809)
        b=report.paired_bootstrap([0,1,0,1],predictions,inventory,10,8809)
        self.assertEqual(a,b)
        self.assertTrue(all(draw==[0.,0.,0.] for draw in a['draws']))

    def test_single_class_bootstrap_not_estimable(self):
        self.assertEqual(report.paired_bootstrap([0,0],{},[])['status'],'not_estimable_single_class')

    def test_constant_inference_uses_frozen_prevalence(self):
        item={'kind':'constant','constant_score':.13}
        self.assertEqual(runner.infer(item,[example()],None,None,None),[.13])

    def test_logits_never_use_labels(self):
        import inspect
        self.assertEqual(list(inspect.signature(runner.infer).parameters),['item','rows','resources','batches','protocol'])
        source=inspect.getsource(runner)
        for forbidden in ('.fit(', '.fit_transform(', '.backward(', 'optimizer.step(', 'threshold_search('):
            self.assertNotIn(forbidden,source)


if __name__=='__main__':
    unittest.main()
