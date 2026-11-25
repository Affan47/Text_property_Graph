"""Checks for decay-only settings, diagnostics, deterministic fits and null reports."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch_geometric.data import Batch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.tpg_revision import regularization_v1 as reg
import test_v5_subtypes as fixtures


class RegularizationTests(unittest.TestCase):
    def test_sigmoid_preserves_original_small_batch_execution(self):
        torch.manual_seed(3)
        logits=torch.randn(1104)*10
        batches=list(logits.split(8))
        joined,scores=reg.joined_outputs(batches)
        expected=np.array([v for part in batches for v in part.sigmoid().tolist()])
        self.assertTrue(torch.equal(joined,logits))
        self.assertTrue(np.array_equal(scores,expected))

    def test_recovery_rejects_any_non_diagnostic_change(self):
        source=str(Path(reg.__file__))
        old={'source_pins':{source:reg.DIAGNOSTIC_PREDECESSOR,'cache':'unchanged'},'settings':{'decay':.001}}
        current=deepcopy(old);current['source_pins'][source]='new diagnostic hash'
        reg.check_diagnostic_recovery(old,current)
        for key in ('cache','settings'):
            changed=deepcopy(current)
            if key=='cache':changed['source_pins']['cache']='changed'
            else:changed['settings']['decay']=.01
            with self.assertRaises(ValueError):reg.check_diagnostic_recovery(old,changed)

    def test_only_weight_decay_changes_and_unknown_arms_rejected(self):
        control={'settings':{'weight_decay':.0001,'epochs':100,'patience':15,'seeds':[17,29,43]},
                 'ids':{'train':['a'],'validation':['b']},'class_weights':{0:1.,1:2.},'source_pins':{},'threshold':.5}
        cfg=reg.data.read(reg.CONFIG);identity={'config':cfg,'source_pins':{}}
        before=deepcopy(control)
        for family in cfg['families']:
            for decay in cfg['new_decays']:
                result=reg.fit_protocol(control,identity,family,decay)
                self.assertEqual({k:v for k,v in result['settings'].items() if k!='weight_decay'},
                                 {k:v for k,v in control['settings'].items() if k!='weight_decay'})
                self.assertEqual(result['ids'],control['ids'])
                self.assertEqual(result['class_weights'],control['class_weights'])
                self.assertEqual(result['settings']['weight_decay'],decay)
        self.assertEqual(control,before)
        for family,decay in [('v5_type_balanced',.001),('v5_node_mean',.0001),('v5_node_mean',.1)]:
            with self.assertRaises(ValueError):reg.fit_protocol(control,identity,family,decay)

    def test_initial_weights_match_for_each_decay(self):
        fixture=fixtures.ModelTests();fixture.setUp()
        for family in reg.data.read(reg.CONFIG)['families']:
            for seed in (17,29,43):
                reference=reg.original.make(fixture.r,2,{**fixture.p,'weight_decay':.0001},seed,family)
                for decay in (.001,.01):
                    other=reg.original.make(fixture.r,2,{**fixture.p,'weight_decay':decay},seed,family)
                    self.assertTrue(all(torch.equal(v,other.state_dict()[k]) for k,v in reference.state_dict().items()))

    def test_real_fitter_synthetic_reload_and_diagnostics(self):
        fixture=fixtures.ModelTests();fixture.setUp();sample=fixture.sample()
        class Batches:
            def plan(self,ids):
                return {'paths':ids,'chunk_cve':torch.arange(len(ids)),
                        'structured':torch.tensor([[.1,.2] if c.endswith('0') else [.7,.8] for c in ids])}
            def packets(self,plan,variant):
                yield 0,Batch.from_data_list([sample for _ in plan['paths']])
        ids={'train':['a0','a1'],'validation':['b0','b1']};targets={'a0':0,'a1':1,'b0':0,'b1':1}
        control={'settings':{**fixture.p,'learning_rate':.001,'weight_decay':.0001,'batch_size':2,'epochs':2,'patience':2},
                 'ids':ids,'class_weights':{0:1.,1:1.},'subtype_schema':list(reg.original.subtypes.NAMES),'source_pins':{}}
        identity={'config':reg.data.read(reg.CONFIG),'source_pins':{}}
        ctx=(Batches(),targets,fixture.r,2,control,identity)
        with tempfile.TemporaryDirectory() as tmp:
            for family in identity['config']['families']:
                protocol=reg.fit_protocol(json.loads(reg.data.dump(control)),identity,family,.01)
                self.assertEqual(protocol['class_weights'], control['class_weights'])
                directory=Path(tmp)/family;directory.mkdir()
                rows,metrics=reg.original.fit(directory,family,17,*ctx[:4],protocol)
                before={str(p):reg.data.sha(p) for p in directory.iterdir() if p.is_file()}
                diag=reg.diagnostics(directory,family,17,ctx,protocol,rows)
                self.assertEqual(diag,reg.diagnostics(directory,family,17,ctx,protocol,rows))
                self.assertEqual(before,{str(p):reg.data.sha(p) for p in directory.iterdir() if p.is_file()})
                self.assertTrue(np.isfinite(diag['logit_min']))
                self.assertEqual(diag['selected_epoch'],1)
                if 'subtypes' in family:self.assertGreater(diag['subtype_frobenius_norm'],0)
                else:self.assertEqual(diag['subtype_frobenius_norm'],0)
                self.assertEqual(metrics,reg.original.verify_fit(directory,family,17,*ctx[:4],protocol)[1])

    def test_grid_summary_keeps_undefined_precision(self):
        cfg=reg.data.read(reg.CONFIG);metrics={};diagnostic={}
        for family in cfg['families']:
            for decay in [cfg['control_decay'],*cfg['new_decays']]:
                for seed in cfg['seeds']:
                    key=reg.condition(family,decay)+f'__seed_{seed}'
                    metrics[key]=reg.original.base.development.metric([0,1],[.1,.2])
                    diagnostic[key]={k:0 for k in ('subtype_frobenius_norm','subtype_rms_weight','absolute_logit_above_10',
                                       'confident_false_negatives_at_or_below_0p0001','confident_false_positives_at_or_above_0p9999')}
        result=reg.summarize({'config':cfg},metrics,diagnostic)
        self.assertEqual(len(metrics),27)
        self.assertTrue(all(v['precision'] is None for v in result['means'].values()))
        self.assertEqual(result['reused_controls'],9);self.assertEqual(result['new_fits'],18)
        self.assertTrue(all(v['precision']==0 for v in ({'precision':x['precision']['defined_seeds']} for x in result['defined_metric_seeds'].values())))


if __name__=='__main__':unittest.main()
