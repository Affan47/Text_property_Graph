import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[name]='4'
ROOT=Path(__file__).resolve().parents[3]
for rel in ('01_tpg/01_core','01_tpg/10_extensions','02_prediction/01_package','01_tpg/02_examples/01_scripts'):
    sys.path.insert(0,str(ROOT/rel))

import numpy as np
import torch
from epss.training_only_pooled_v1 import data, fitting, model, report, runner


def fixture():
    rng=np.random.default_rng(54); arrays={}; ys={}; ids={}
    for role,n in [('optimization',16),('selection',8),('assessment',8)]:
        arrays[role+'_structured']=rng.normal(size=(n,5)).astype('float32')
        arrays[role+'_D']=rng.normal(size=(n,768)).astype('float32')
        arrays[role+'_DM']=arrays[role+'_D'].copy()
        arrays[role+'_D_mask']=np.tile(np.array([1,0,0,0],dtype='float32'),(n,1))
        arrays[role+'_DM_mask']=arrays[role+'_D_mask'].copy()
        ys[role]=rng.uniform(0,.2,n).tolist();ids[role]=[role+str(i) for i in range(n)]
    return {'arrays':arrays,'labels':ys,'ids':ids}


class TrainingOnlyTests(unittest.TestCase):
    def test_runtime_order_not_identity_but_threads_are(self):
        a={'threadpools':[{'filepath':'z','user_api':'blas','num_threads':4},{'filepath':'a','user_api':'openmp','num_threads':4}]}
        b={'threadpools':a['threadpools'][::-1]}
        self.assertEqual(data.canonical_runtime(a),data.canonical_runtime(b))
        b={'threadpools':[dict(r,num_threads=1) for r in b['threadpools']]}
        self.assertNotEqual(data.canonical_runtime(a),data.canonical_runtime(b))

    def test_nested_chronology_and_components(self):
        rows={str(i):{'anchor_date':f'2022-01-{i+1:02d}'} for i in range(12)}
        components={c:c for c in rows};components['0']='8';components['8']='8'
        f={'fit':[str(i) for i in range(10)],'inner_assessment':['10','11']}
        r=data.nested_roles(f,rows,components,.2)
        self.assertEqual(r['selection'],['8','9']);self.assertEqual(r['optimization_purged'],['0'])
        self.assertEqual(r['assessment'],['10','11']);self.assertEqual(len(r['optimization']),7)

    def test_selection_date_ties_not_split(self):
        rows={str(i):{'anchor_date':f'2022-01-{i+1:02d}'} for i in range(12)}
        rows['7']['anchor_date']=rows['8']['anchor_date']
        r=data.nested_roles({'fit':[str(i) for i in range(10)],'inner_assessment':['10','11']},rows,{c:c for c in rows},.2)
        self.assertEqual(r['selection'],['7','8','9'])

    def test_model_rejects_target(self):
        f=fixture();x=model.inputs(f['arrays'],'optimization','D');x['target']=torch.zeros(16)
        with self.assertRaisesRegex(ValueError,'Unexpected'):model.Regressor(5)(x)

    def test_wrapper_parity_and_capacity(self):
        result=runner.parity(fixture(),data.settings())
        self.assertEqual(result['parameters_D'],result['parameters_DM'])
        self.assertEqual(result['absent_M_cases'],8)

    def test_input_masks_do_not_expose_other_source_to_D(self):
        f=fixture();f['arrays']['optimization_DM_mask'][:,1]=1
        self.assertFalse(model.inputs(f['arrays'],'optimization','D')['source_mask'][:,1].any())

    def test_npz_and_csv_are_canonical(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b=Path(folder)/'a.npz',Path(folder)/'b.npz'
            data.npz_write(a,{'z':np.ones(3),'a':np.zeros(2)})
            data.npz_write(b,{'a':np.zeros(2),'z':np.ones(3)})
            self.assertEqual(a.read_bytes(),b.read_bytes())
        rows=[{'cve':'b','score':1},{'cve':'a','score':0}]
        self.assertEqual(data.reporting.csv_text(rows),data.reporting.csv_text(rows[::-1]))
        self.assertNotIn('\r',data.reporting.csv_text(rows))

    def test_paired_bootstrap_reproducible_and_identical_predictions_zero(self):
        a=np.arange(12).reshape(3,4)/20
        v=report.paired_ci(a,a,['a','a','b','c'],100,6029)
        self.assertEqual((v['difference'],v['low'],v['high']),(0,0,0))
        self.assertEqual(v,report.paired_ci(a,a,['a','a','b','c'],100,6029))
        self.assertEqual(v['units'],3)

    def test_ridge_fit_does_not_use_assessment(self):
        f=fixture();p=data.settings();p['ridge_selection_grid']=[1.]
        original=fitting.controls(f['arrays'],f['labels'],p)
        f['labels']['assessment']=[1.]*8
        self.assertEqual(original,fitting.controls(f['arrays'],f['labels'],p))

    def test_linear_replay(self):
        x=np.array([[1.,2.],[3.,4.]])
        s={'kind':'ridge','mean':[1,2],'scale':[2,2],'coef':[2,3],'intercept':.1}
        np.testing.assert_allclose(fitting.linear_predict(s,x),[.1,5.1])

    def test_small_training_resume_and_replay(self):
        f=fixture();p=data.settings();p.update(epochs=3,patience=3)
        identity={'dataset':'unit','arm':'D','seed':17}
        x=model.inputs(f['arrays'],'optimization','D');sx=model.inputs(f['arrays'],'selection','D')
        with tempfile.TemporaryDirectory() as folder:
            a,b=Path(folder)/'fresh',Path(folder)/'interrupted'
            fresh,cp=fitting.fit(a,identity,x,f['labels']['optimization'],sx,f['labels']['selection'],p,17,1)
            original=data.base.write
            def interrupt(path,value):
                original(path,value)
                if Path(path).name=='history.json' and len(value)==2:raise RuntimeError('interrupt')
            with patch.object(data.base,'write',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'interrupt'):
                    fitting.fit(b,identity,x,f['labels']['optimization'],sx,f['labels']['selection'],p,17,1)
            resumed,rcp=fitting.fit(b,identity,x,f['labels']['optimization'],sx,f['labels']['selection'],p,17,1,resume=True)
            self.assertEqual(data.c.read(a/'history.json'),data.c.read(b/'history.json'))
            np.testing.assert_array_equal(model.predict(fresh,sx),model.predict(resumed,sx))
            self.assertEqual(len(list(b.glob('*.pt'))),1)
            self.assertEqual(rcp['epoch'],min(data.c.read(b/'history.json'),key=lambda r:r['selection_rmse'])['epoch'])

    def test_assessment_not_fit_argument(self):
        import inspect
        names=set(inspect.signature(fitting.fit).parameters)
        self.assertFalse(any('assessment' in n for n in names))

    def test_locked_settings(self):
        p=data.settings()
        self.assertEqual(p['bootstrap_repetitions'],20000)
        self.assertFalse(p['test_access']);self.assertFalse(p['outer_validation_access'])
        self.assertFalse(p['refit_after_selection'])


if __name__=='__main__':unittest.main()
