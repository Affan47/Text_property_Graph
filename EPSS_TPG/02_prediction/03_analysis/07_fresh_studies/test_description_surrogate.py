import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='4'
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
for rel in ('01_tpg/01_core','01_tpg/10_extensions','02_prediction/01_package','01_tpg/02_examples/01_scripts'):
    sys.path.insert(0,str(ROOT/rel))
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from epss.description_surrogate_v1 import data, model, fitting, reporting, runner


class SurrogateTests(unittest.TestCase):
    def test_only_text_is_accepted(self):
        net=model.DescriptionRegressor();x=torch.ones(2,768)
        self.assertEqual(net({'text':x}).shape,(2,))
        for key in ('structured','source_mask','epss','label','target'):
            with self.assertRaises(ValueError):net({'text':x,key:torch.ones(2)})
        with self.assertRaises(ValueError):net({'text':torch.ones(2,769)})

    def test_exact_parameter_count(self):
        self.assertEqual(sum(p.numel() for p in model.DescriptionRegressor().parameters()),25761)

    def test_export_contract(self):
        ids={r:['a','b'] for r in data.ROLES};arrays={r:np.ones((2,768),np.float32) for r in ids}
        data.validate_predictors(arrays,ids)
        with self.assertRaises(ValueError):data.validate_predictors(dict(arrays,structured=np.ones((2,3))),ids)
        arrays['selection'][0,0]=np.nan
        with self.assertRaises(ValueError):data.validate_predictors(arrays,ids)

    def test_best_checkpoint_and_replay(self):
        rng=np.random.default_rng(5);x=torch.from_numpy(rng.normal(size=(8,768)).astype(np.float32));y=rng.random(8)
        p={'hidden':32,'epochs':3,'patience':3,'batch_size':4,'learning_rate':.001,'weight_decay':.0001}
        ident={'seed':17,'fold':1};sx=x[:4];sy=y[:4]
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'first';net,cp=fitting.fit(root,ident,p,x,y,sx,sy)
            pred=model.predict(net,sx);hist=data.a.read(root/'history.json')
            self.assertEqual(cp['epoch'],min(hist,key=lambda r:r['selection_rmse'])['epoch'])
            self.assertEqual([f.name for f in root.glob('*.pt')],['best_checkpoint.pt'])
            second,_=fitting.fit(Path(t)/'second',ident,p,x,y,sx,sy)
            self.assertTrue(np.array_equal(pred,model.predict(second,sx)))
            again,_=fitting.fit(root,ident,p,x,y,sx,sy,resume=True)
            self.assertTrue(np.array_equal(pred,model.predict(again,sx)))
            self.assertEqual(hist,data.a.read(root/'history.json'))
            with self.assertRaises(ValueError):fitting.fit(root,{'seed':29,'fold':1},p,x,y,sx,sy,resume=True)

    def test_tie_aware_weighted_metrics(self):
        y=np.array([0,1,1,0,0]);v=np.array([.2,.2,.7,.1,.7]);w=np.array([[2,1,3,2,1]],float)
        auc,ap=reporting.gap.weighted_metrics(y,v,w)
        self.assertAlmostEqual(auc[0],roc_auc_score(y,v,sample_weight=w[0]))
        self.assertAlmostEqual(ap[0],average_precision_score(y,v,sample_weight=w[0]))

    def test_seed_identity_shared_across_rows(self):
        p=np.array([[.1,.8,.3],[.2,.9,.4],[.3,.7,.5]])
        np.testing.assert_array_equal(reporting.seed_mean(p,[0,3,0]),p[:,1])

    def test_bootstrap_replay_and_one_class(self):
        rng=np.random.default_rng(11);y=np.array([0,1,0,1,0,1,0,0]);anchor=rng.random(8)*.09
        p=rng.random((8,3));groups=np.array(['a','a','b','c','c','d','e','f'])
        first=reporting.intervals(y,anchor,p,groups,30,6029)
        self.assertEqual(first,reporting.intervals(y,anchor,p,groups,30,6029))
        self.assertEqual(len(first),4)
        self.assertEqual(reporting.intervals(np.zeros(8),anchor,p,groups,30,6029),[])

    def test_csv_canonical_and_lf(self):
        rows=[{'seed':29,'cve':'b','value':.2},{'seed':17,'cve':'a','value':.1}]
        self.assertEqual(reporting.csv_text(rows),reporting.csv_text(rows[::-1]))
        self.assertNotIn('\r',reporting.csv_text(rows))

    def test_progression_is_not_forecast_claim(self):
        row={'policy':'isolated_day_assumption','unit':'group_and_seed','metric':'average_precision','difference':.01,'low':-.01}
        self.assertTrue(reporting.decision([row],False).startswith('STOP'))
        row['low']=.001
        self.assertTrue(reporting.decision([row],False).startswith('EXPLORATORY'))
        self.assertEqual(reporting.decision([],True),'SMOKE_ONLY_NO_SCIENTIFIC_DECISION')

    def test_synthetic_end_to_end_report_replay(self):
        torch.set_num_threads(4);rng=np.random.default_rng(71)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);roles={};records={};groups={};s=data.settings()
            s['smoke']['bootstrap_repetitions']=20
            for f in range(1,4):
                ids={r:[f'synthetic_{f}_{r}_{i}' for i in range(4 if r=='optimization' else 2)] for r in data.ROLES}
                roles[str(f)]=ids;dest=root/'smoke/prepared'/f'fold_{f}';dest.mkdir(parents=True)
                fitting.write(dest/'ids.json',ids)
                fitting.write(dest/'labels.json',{r:[.02,.2]*(len(ids[r])//2) for r in data.ROLES})
                data.parent.old.npz_write(dest/'predictors.npz',{r:rng.normal(size=(len(ids[r]),768)).astype(np.float32) for r in ids})
                for i,c in enumerate(ids['assessment']):
                    groups[c]=c
                    records[c]={'label_not_predictor':{'label':i,'strict_label':i,'anchor_score':.01+.01*i}}
            plan=root/'plan/plan.json';fitting.write(plan,{'assessment_ids':sorted(records),'records':records})
            p={'settings':s,'roles':{'smoke':roles},'component_by_cve':groups,'source_pins':{},
               'runtime':data.parent.old.runtime(),'input_hashes':{str(plan):data.a.sha(plan)}}
            fitting.write(root/'protocol/protocol.json',p);data.parent.old.publication.lock(root/'smoke/prepared')
            with patch.object(data,'OUTPUT',root),patch.object(data,'pins',return_value={}),patch.object(runner,'check',return_value=p),patch.object(runner,'controls',side_effect=lambda *args:[]):
                runner.run('smoke',jobs=1)
                runner.run('smoke',jobs=1,verify=True)
            self.assertEqual(data.a.read(root/'smoke/report/findings.json')['decision'],'SMOKE_ONLY_NO_SCIENTIFIC_DECISION')
            self.assertEqual(len(list(root.glob('smoke/fits/*/*/best_checkpoint.pt'))),3)


if __name__=='__main__':unittest.main()
