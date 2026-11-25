from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'01_package'))
from epss.continuous_forecasting.stage3 import diagnostic as d


def example():
    return dict(cve='CVE-2024-10001', split='validation', publication_date='2024-01-01',
                values=[.1, None, 0., .3, None, None]+[.4]*85, date_allowed=[True]*91)


def inputs():
    return d.data.batch([(example(), 5, np.array([1, 30]), 1.)])[0]


class InterventionTests(unittest.TestCase):
    def test_flat_preserves_masks_age_and_latest_without_mutation(self):
        a = inputs(); before = {k:v.clone() for k,v in a.items()}
        b = d.transform(a, 'flat_history')
        torch.testing.assert_close(b['past'][0, [0, 2, 3], 0], torch.tensor([.3]*3))
        self.assertTrue(torch.equal(a['past'][:,:,1:], b['past'][:,:,1:]))
        self.assertEqual(b['past'][0,1,0], 0)
        for k in a:
            self.assertTrue(torch.equal(a[k], before[k]))
            if k != 'past':self.assertTrue(torch.equal(a[k], b[k]))

    def test_last_only_keeps_time_length_and_correct_age(self):
        a=inputs(); b=d.transform(a, 'last_obs_only')
        self.assertTrue(torch.equal(b['past'][0,:3,:4], torch.zeros(3,4)))
        self.assertTrue(torch.equal(b['past'][0,3:], a['past'][0,3:]))
        self.assertTrue(torch.equal(b['past'][:,:,4], a['past'][:,:,4]))
        self.assertEqual(int(b['past'][:,:,1].sum()),1)
        self.assertEqual(float(b['past'][0,5,2]), float(a['past'][0,5,2]))
        self.assertTrue(torch.equal(b['lengths'], a['lengths']))
        self.assertTrue(torch.equal(b['latest'], a['latest']))

    def test_true_zero_is_an_observation(self):
        row=example();row['values'][3]=0.
        a=d.data.batch([(row,5,np.array([1]),1.)])[0]
        b=d.transform(a,'last_obs_only')
        self.assertEqual(float(b['past'][0,3,0]),0.)
        self.assertEqual(float(b['past'][0,3,1]),1.)

    def test_anchor_zero_and_single_observation_are_noops(self):
        a=d.data.batch([(example(),0,np.array([2]),1.)])[0]
        for mode in ('intact','flat_history','last_obs_only'):
            b=d.transform(a,mode)
            self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))

    def test_future_values_and_targets_are_not_inputs(self):
        a=inputs();row=example();row['values'][6:]=[.99]*85
        b=d.data.batch([(row,5,np.array([1,30]),1.)])[0]
        for mode in ('intact','flat_history','last_obs_only'):
            x,y=d.transform(a,mode),d.transform(b,mode)
            self.assertTrue(all(torch.equal(x[k],y[k]) for k in x))
        with self.assertRaises(ValueError):d.transform({**a,'labels':torch.zeros(1)},'flat_history')

    def test_interventions_preserve_frozen_eligibility_and_missing_targets(self):
        row=example();row['values'][60]=None
        model=d.runner.model_for(17,d.data.settings()).eval()
        with torch.inference_mode():
            truth,preds,mask=d.predict(row,model,32,.01)
        self.assertTrue(np.array_equal(mask,d.data.eligible(row)))
        self.assertTrue(np.isnan(truth[~mask]).all())
        for mode in ('intact','flat_history','last_obs_only'):
            self.assertTrue(np.array_equal(preds[mode][mask],preds['persistence'][mask]))
        np.testing.assert_array_equal(preds['constant_shift_control'][mask],preds['persistence'][mask]+.01)

    def test_diff_exact_zero_and_hierarchical_weights(self):
        mask=np.zeros((90,90),bool);mask[0,:2]=True;mask[1,0]=True
        a=np.zeros((90,90));b=a.copy();b[1,0]=1
        out=d.differences(b,a,mask)['all']
        self.assertEqual(out['mean_abs_raw_diff'],.5)
        self.assertEqual(out['raw_changed_cells'],1)
        self.assertEqual(d.differences(a,a,mask)['all']['max_abs_raw_diff'],0.)

    def test_canonical_csv_survives_json_reload_and_input_order(self):
        values={'lead7':.1,'lead14':.2,'all':.3}
        loaded=json.loads(json.dumps(values,sort_keys=True))
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a.csv',Path(tmp)/'b.csv'
            for path,items in ((a,values),(b,loaded)):
                d.small_csv(path,['slice','value'],[((k,),[k,v]) for k,v in items.items()])
            self.assertEqual(a.read_bytes(),b.read_bytes())

    def test_stream_rejects_duplicates_and_wrong_order(self):
        w=d.OrderedWriter(io.StringIO(),['key']);w.write(('B',1),['B'])
        for key in (('B',1),('A',1)):
            with self.assertRaises(ValueError):w.write(key,['invalid'])

    def test_gzip_replay_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=[Path(tmp)/x for x in ('a.gz','b.gz')]
            for path in paths:
                with d.compressed_csv(path,['key','value']) as w:w.write(('A',),['A',.1])
            self.assertEqual(paths[0].read_bytes(),paths[1].read_bytes())


if __name__ == '__main__':unittest.main()
