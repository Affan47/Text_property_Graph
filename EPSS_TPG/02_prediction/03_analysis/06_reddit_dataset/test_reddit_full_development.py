"""Synthetic regression tests. No real-cohort fitting or test-data access."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.reddit_development_v1 import common as c, primary, forecast, counterfactuals
from epss.reddit_forecasting import data, evaluation as original_evaluation
from epss.continuous_forecasting import runner as base
from epss.continuous_forecasting.stage3.diagnostic import transform


class FullDevelopmentTests(unittest.TestCase):
    def row(self):
        return dict(cve='SYNTHETIC',split='validation',publication_date='2022-01-01',
                    values=[.01+i*.0001 if i%4 else None for i in range(91)],
                    date_allowed=[True]*91,release_dates=['2022-02-01'])

    def test_eleven_gate_hashes(self):
        self.assertEqual(len(c.EXPECTED),11)
        self.assertTrue(all(len(v)==64 for v in c.EXPECTED.values()))

    def test_empty_and_constant_metrics(self):
        m=primary.measure([],[])
        self.assertIsNone(m['mae']);self.assertIsNone(m['high_score_recall'])
        m=primary.measure([.2,.01],[.2,.3])
        self.assertEqual(m['high_score_precision'],.5);self.assertEqual(m['high_score_recall'],1.)
        self.assertIsNone(primary.measure([.01,.01],[.01,.01])['pearson'])

    def test_seed_metric_mean_not_ensemble(self):
        r=c.compare_ci([[1,1],[0,0]],[[0,0],[0,0]],'rmse',10)
        self.assertEqual(r['difference'],.5)  # mean(1, 0), not sqrt(mean MSE)=sqrt(.5)
        self.assertIsNone(c.mean_metrics([{'a':None},{'a':1}])['a'])

    def test_canonical_csv_and_compressed_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=[{'model':'z','value':None},{'model':'a','value':.1}]
            c.csv_table(root/'a.csv',rows);c.csv_table(root/'b.csv',list(reversed(rows)))
            self.assertEqual((root/'a.csv').read_bytes(),(root/'b.csv').read_bytes())
            self.assertNotIn(b'\r\n',(root/'a.csv').read_bytes())
            for name in ('a.gz','b.gz'):
                with forecast.prediction_writer(root/name) as writer:writer.writerow(['same',.1])
            self.assertEqual(c.sha(root/'a.gz'),c.sha(root/'b.gz'))
            self.assertTrue(c.report_replay(root,False,True))
            self.assertFalse(c.report_replay(root,False,False))

    def test_forecast_evaluation_parity_and_empty_partitions(self):
        r=self.row();r['release_dates']=[];p={'anchor_packet':32}
        expected=original_evaluation.evaluate([r],32)
        self.assertEqual(forecast.evaluate([r],p),expected)
        self.assertIsNone(expected['summary']['all/anchor_target_crossing']['mae'])

    def test_observed_only_and_future_excluded(self):
        r=self.row();items=list(data.requests(r));inputs,targets,valid,weights=data.batch(items[:1])
        changed=dict(r,values=list(r['values']));changed['values'][90]=.99
        other=data.batch(list(data.requests(changed))[:1])[0]
        self.assertTrue(all(torch.equal(inputs[k],other[k]) for k in inputs))
        self.assertTrue(torch.isfinite(targets[valid]).all());self.assertTrue(torch.isnan(targets[~valid]).all())
        self.assertAlmostEqual(sum(float(data.batch([item])[3].sum()) for item in items),1.)

    def test_interventions_no_mutation(self):
        r=self.row();item=next(x for x in data.requests(r) if x[1]==30)
        inputs=data.batch([item])[0];original={k:v.clone() for k,v in inputs.items()}
        flat=transform(inputs,'flat_history');last=transform(inputs,'last_obs_only')
        self.assertTrue(torch.equal(flat['past'][:,:,1:],inputs['past'][:,:,1:]))
        self.assertLess(float(last['past'][:,:,1].sum()),float(inputs['past'][:,:,1].sum()))
        self.assertTrue(all(torch.equal(inputs[k],original[k]) for k in inputs))

    def test_gain_share_and_ineligible_primary(self):
        self.assertIsNone(counterfactuals.gain_share(.1,.1,.09)['percent'])
        self.assertAlmostEqual(counterfactuals.gain_share(.1,.09,.095)['percent'],50.)
        p={'per_cve':[{'cve':'a','metrics':{'day30_to_day60':{'raw_bias':.2}}},
                      {'cve':'b','metrics':{'day30_to_day60':None}}]}
        b={'per_cve':[{'cve':'a','metrics':{'day30_to_day60':{'raw_bias':.1}}},
                      {'cve':'b','metrics':{'day30_to_day60':None}}]}
        self.assertEqual(counterfactuals.constants(p,b),(.1,1))

    def test_atomic_resume_and_checkpoint_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ident=dict(task='synthetic',settings={'learning_rate':.001,'weight_decay':.0001},source_pins={})
            def make():
                torch.manual_seed(17);return torch.nn.Linear(1,1)
            def train(net,opt,epoch):
                opt.zero_grad();loss=(net(torch.ones(2,1))-.25).abs().mean();loss.backward();opt.step()
                return float(loss.detach()),1.
            def evaluate(net,stage):
                with torch.no_grad():v=float(net(torch.ones(1,1))[0,0])
                return {'metrics':{'mae':abs(v-.25)},'prediction':v}
            kwargs=dict(ident=ident,key='fit',make_model=make,evaluate=evaluate,
                summary=lambda r:r['metrics'],score=lambda m:m['mae'],epochs=2,patience=2)
            a=c.fit_unit(root/'a',root/'recovery/a.pt',train_epoch=train,**kwargs)
            replay=c.fit_unit(root/'a',root/'recovery/a.pt',train_epoch=train,verify=True,**kwargs)
            self.assertEqual(a,replay)
            with self.assertRaisesRegex(ValueError,'already exists'):
                c.fit_unit(root/'a',root/'recovery/a.pt',train_epoch=train,**kwargs)
            def interrupted(net,opt,epoch):
                if epoch==2:raise RuntimeError('simulated interruption')
                return train(net,opt,epoch)
            with self.assertRaisesRegex(RuntimeError,'simulated'):
                c.fit_unit(root/'b',root/'recovery/b.pt',train_epoch=interrupted,**kwargs)
            b=c.fit_unit(root/'b',root/'recovery/b.pt',train_epoch=train,resume=True,**kwargs)
            self.assertEqual(a,b)

    def test_synthetic_forecast_report_and_counterfactual_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            p=dict(data.settings(),bootstrap_repetitions=10)
            p['smoke']=dict(p['smoke'],epochs=1,patience=1)
            rows={s:[] for s in ('train','validation')}
            for s in rows:
                for i in range(2):
                    r=self.row();r.update(cve=f'SYNTHETIC-{s}-{i}',split=s)
                    r['values'][60]=.04+i*.001
                    r['date_allowed']=[d in (30,60) for d in range(91)]
                    rows[s].append(r)
            ident=dict(task='RD04',settings=p,source_pins={},upstream={},counts={})
            pre=root/'preflight/full';pre.mkdir(parents=True)
            c.put(pre,'summary.json',{'full_preflight':{'train':{'synthetic':True}}})
            with patch.object(forecast,'context',return_value=(rows,ident)),patch.object(data,'OUTPUT',root):
                forecast.run('smoke')
                forecast.run('smoke',verify=True)
                # Simulate interruption after report publication, before the outer lock.
                (forecast.destination('smoke')/'artifact_lock.json').unlink()
                with patch.object(base,'train_epoch',side_effect=AssertionError('unexpected retraining')):
                    forecast.run('smoke',resume=True)
                    forecast.run('smoke',verify=True)
                counterfactuals.run('smoke')
                counterfactuals.run('smoke',verify=True)

    def test_synthetic_primary_all_families_replay(self):
        from types import SimpleNamespace
        from torch_geometric.data import Data, Batch
        from tpg.contracts import EDGE_FEATURE_DIM
        ids={'train':['train-a','train-b'],'validation':['val-a','val-b']}
        names=ids['train']+ids['validation']
        resources={'type_vocabulary':{'num_node_types':13,'edge_types':{'DEP':0}}}
        vectors={x:[float(i%2),float(i)/4] for i,x in enumerate(names)}
        labels={x:{'target':.02+i*.01,'target_date':'2024-01-01'} for i,x in enumerate(names)}
        rows=[{'cve':x,'mask_M_prose':i%2==0} for i,x in enumerate(names)]
        audit={x:{'overlap_partition':'no_detected_overlap_with_train' if i%2 else 'overlap_with_train',
                  'local_release_crossing':False,'whole91day_version_changed':False} for i,x in enumerate(names)}
        class SyntheticBatches:
            def plan(self,cs,variant):
                sources=[];paths=[];masks=[]
                for i,x in enumerate(cs):
                    m=variant['view']!='D' and (variant['view']=='DM_all' or names.index(x)%2==0)
                    masks.append([True,False,False,m]);sources.append(4*i);paths.append(x+'-D')
                    if m:sources.append(4*i+3);paths.append(x+'-M')
                return dict(paths=paths,chunk_document=torch.arange(len(paths)),
                    document_source=torch.tensor(sources),source_mask=torch.tensor(masks),
                    structured=torch.tensor([vectors[x] for x in cs]))
            def packets(self,plan,variant):
                graphs=[]
                for path in plan['paths']:
                    x=torch.zeros(2,782);x[:,3]=1;x[:,-1]=1;x[:,13:781]=.01 if path.endswith('D') else .02
                    graphs.append(Data(x=x,edge_index=torch.tensor([[0,1],[1,0]]),edge_type=torch.zeros(2,dtype=torch.long),
                        edge_features=torch.zeros(2,EDGE_FEATURE_DIM),token_mask=torch.ones(2,dtype=torch.bool),
                        subtype=torch.tensor([0,31])))
                yield 0,Batch.from_data_list(graphs)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=root/'cache';cache.mkdir();c.put(cache,'artifact_lock.json',{'synthetic':True})
            batches=SyntheticBatches();batches.resources=resources;batches.vectors=vectors;batches.directory=cache
            p={'learning_rate':.001,'weight_decay':.0001,'batch_size':8,'epochs':100,'patience':15}
            ident={'task':'RD02','settings':p,'source_pins':{},'upstream':{}}
            original_comparisons=primary.comparisons
            with patch.object(primary,'context',return_value=(rows,batches,ids,labels,audit,ident)),\
                 patch.object(primary.data,'OUT',root),\
                 patch.object(primary,'comparisons',side_effect=lambda r,p:original_comparisons(r,p,10)):
                primary.run('smoke')
                primary.run('smoke',verify=True)
                (root/'development_smoke_v1/artifact_lock.json').unlink()
                with patch.object(primary.model,'backward',side_effect=AssertionError('unexpected retraining')):
                    primary.run('smoke',resume=True)
                    primary.run('smoke',verify=True)


if __name__=='__main__':unittest.main()
