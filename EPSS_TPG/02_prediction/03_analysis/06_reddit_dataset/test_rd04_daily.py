from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.reddit_forecasting import data, evaluation
from epss.continuous_forecasting.runner import model_for


def row(pub='2023-03-01', role='train'):
    manifest={'publication_intervals':{'train':['2021-04-14','2023-12-31'],
              'validation':['2024-01-01','2024-12-31']},'cutoff':'2026-11-03',
              'release_schedule':{'v1':'2021-04-14','v2':'2022-02-04','v3':'2023-03-07',
                                  'v4':'2025-03-17','v5':'2026-06-15'}}
    dates=[(date.fromisoformat(pub)+timedelta(days=i)).isoformat() for i in range(91)]
    return data.adapt('CVE-SYNTHETIC',pub,{d:.01+i*.0001 for i,d in enumerate(dates)},
                      dict.fromkeys(dates,'OBSERVED'),role,manifest)


class RD04Tests(unittest.TestCase):
    def test_boundaries(self):
        r=row('2023-12-30'); m=data.eligible(r)
        self.assertEqual(int(m.sum()),1)
        self.assertTrue(all(v is None for v in r['values'][2:]))
        self.assertTrue(data.eligible(row('2024-12-30','validation'))[0,0])
        with self.assertRaises(ValueError): row(role='test')

    def test_no_history_or_target_imputation(self):
        r=row(); r['values'][0]=None; r['values'][8]=None
        m=data.eligible(r); self.assertFalse(m[0].any()); self.assertFalse(m[7,0])
        self.assertEqual(data.prefix(r,8)[2],1)
        self.assertEqual(sum(int(v.sum()) for v in data.coverage(r).values()),4095)

    def test_future_values_not_inputs(self):
        r=row(); b=deepcopy(r); b['values'][31:]=[.9]*60
        a=data.batch([(r,30,np.array([30]),1.)]); z=data.batch([(b,30,np.array([30]),1.)])
        self.assertTrue(all(torch.equal(a[0][k],z[0][k]) for k in a[0]))
        self.assertFalse(torch.equal(a[1],z[1]))

    def test_weights_and_masks(self):
        r=row(); total=0.
        for packet in data.packets(data.requests(r),7):
            _,_,valid,w=data.batch(packet); total+=float(w[valid].sum())
        self.assertAlmostEqual(total,1.,places=12)

    def test_release_boundaries_and_different_start(self):
        r=row(); p=data.partitions(r)
        self.assertTrue(p['anchor_target_crossing'][5,0])
        self.assertFalse(p['anchor_target_crossing'][6,0])
        self.assertTrue(p['history_target_crossing'][6,0])
        for stem in ('anchor_target','history_target'):
            a,b=p[stem+'_crossing'],p[stem+'_single_version']
            self.assertFalse((a&b).any()); np.testing.assert_array_equal(a|b,data.eligible(r))
        self.assertFalse(data.partitions(row('2024-01-01','validation'))['anchor_target_crossing'].any())

    def test_initialization_and_seed_replay(self):
        p=data.settings(); r=row(); inputs,_,_,_=data.batch([next(data.requests(r))])
        for seed in p['seeds']:
            a=model_for(seed,p); b=model_for(seed,p)
            self.assertTrue(all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict()))
            raw,_=data.forward(a,inputs)
            self.assertTrue(torch.equal(raw,inputs['latest'].expand_as(raw)))

    def test_empty_version_partition_not_zero_error(self):
        report=evaluation.evaluate([row('2024-01-01','validation')],32)
        self.assertIsNone(report['summary']['all/anchor_target_crossing']['mae'])
        self.assertEqual(report['summary']['all']['cells'],4095)


if __name__=='__main__': unittest.main()
