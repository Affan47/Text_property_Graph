import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.epss_threshold_v1 import runner as et


class ThresholdTest(unittest.TestCase):
    def test_label(self):
        self.assertEqual([et.label(x) for x in (None,0,.099999,.1,1)], [None,0,0,1,1])
        for x in (True,-.1,1.1,float('nan'),float('inf'),'0.1'):
            with self.assertRaises(ValueError):
                et.label(x)

    def test_weighted_ranking_matches_explicit_resampling(self):
        ids=np.array(['d','c','a','b','e'])
        y=np.array([0,1,0,1,1]);s=np.array([1.,1.,1.,.2,.2])
        weights=np.array([[0,6,0,20,1],[1,2,1,10,12],[6,0,1,6,0],[1,1,1,1,1]])
        got=et.Ranking(ids,y,s).evaluate(weights)
        for row,w in zip(got,weights):
            idx=np.arange(5).repeat(w)
            top=idx[np.lexsort((ids[idx],-s[idx]))[:(len(idx)+19)//20]]
            np.testing.assert_allclose(row,[average_precision_score(y[idx],s[idx]),y[top].sum()/y[idx].sum()])
        with self.assertRaises(ValueError):
            et.Ranking(['a','a'],[0,1],[1,2])

    def test_replay_and_assessment_synthetic_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);cfg=et.read(et.CFG)
            cfg['bootstrap']['draws']=64
            config=tmp/'synthetic.json';et.write(config,cfg)
            parts={};gold={};refs={(v,s):{} for v in et.VIEWS for s in cfg['reference_seeds']}
            rng=np.random.default_rng(730)
            for fold in (1,2,3):
                ids={};eligible={};arrays={}
                for role in et.ROLES:
                    cs=[f'synthetic-{fold}-{role}-{i}' for i in range(24)]
                    ids[role]=cs
                    eligible[role]=[(i,c,i%2) for i,c in enumerate(cs)]
                    for suffix,dim in [('D',768),('DM',768),('structured',3)]:
                        arrays[role+'_'+suffix]=rng.normal(size=(24,dim))
                    if role=='assessment':
                        for i,c,y in eligible[role]:
                            gold[c]=(fold,'group-'+str(i//2),y,.2*y)
                            for scores in refs.values():
                                scores[c]=float(rng.random())
                path=tmp/f'{fold}.npz';np.savez(path,**arrays)
                parts[fold]=path,ids,eligible,3
            with patch.object(et,'OUT',tmp/'out'), patch.object(et,'CFG',config), patch.object(et,'context',
                    return_value=(parts,gold,refs,{'ready':True,'support':[{'test':True}],'assessed':72})):
                et.run('preflight')
                et.run('train')
                with patch.object(et,'fit',side_effect=AssertionError('unexpected fitting')):
                    et.run('train',True)
                    et.run('assess')
                    et.run('assess',True)
                self.assertEqual(et.read(et.OUT/'assessment/primary.json')['paired']['valid_draws'],64)
                with self.assertRaises(ValueError):
                    et.run('train')

    def test_unknown_rows_excluded_and_extra_array_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'x.npz'
            arrays={r+'_'+s:np.ones((3,d)) for r in et.ROLES for s,d in [('structured',1),('D',768),('DM',768)]}
            arrays['train_structured'][:,0]=[1,9,2]
            part=path,{'train':['a','unknown','b']},{'train':[(0,'a',0),(2,'b',1)]},1
            np.savez(path,**arrays)
            np.testing.assert_array_equal(et.design(part,'train','structured')[:,0],[1,2])
            np.savez(path,**arrays,train_day60=[0,.5,.1])
            with self.assertRaises(ValueError):
                et.design(part,'train','pooled_D')

if __name__=='__main__':
    unittest.main()
