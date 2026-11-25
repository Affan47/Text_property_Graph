"""CS01 source-factorial, retrospective input and all-day forecast contracts."""
from copy import deepcopy
from datetime import date,timedelta
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.consolidated import protocol as p,contracts as c,evaluation as e

POLICY={'validation_start':'2025-03-02','test_start':'2025-06-01','test_end':'2025-12-02','observation_end':'2026-09-29'}


def record(publication='2024-01-01'):
    pub=date.fromisoformat(publication)
    r={'cve':'CVE-2024-10001','published_date':publication,'published_timestamp':publication+'T12:00:00Z',
       'epss_scores':{(pub+timedelta(days=d)).isoformat():.1 for d in range(91)},
       'description':'A vulnerability in the parser.','social_media_posts':[{'post':'A new advisory.','platform':'Mastodon','date':'2024-01-04'}]}
    fields=p.load()['source_fields']
    r[fields['github_summary']]='A patch checks buffer bounds.'
    r[fields['non_github_summary']]='The vendor describes a flaw.'
    return r


def structured():
    return {'cvss_vector':'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H',
            'cvss_vector_version':'3.1','cvss_vector_score':9.8,'n_references_nvd':10}


class ContractsTests(unittest.TestCase):
    def test_full_source_factorial_and_names(self):
        rows=p.experiment_matrix();self.assertEqual(len(rows),73)
        self.assertEqual(len({r['id'] for r in rows}),73)
        primary=[r for r in rows if r['id'].startswith('P01_') and r['sources']]
        self.assertEqual(len(primary),60)
        combos={tuple(r['sources']) for r in primary};self.assertEqual(len(combos),15)
        for sources in combos:
            matches=[r for r in primary if tuple(r['sources'])==sources]
            self.assertEqual(len(matches),4)

    def test_sources_individually_removed_and_no_missingness_carryover(self):
        docs=c.text_documents(record(),p.load()['source_fields'])
        for removed in p.SOURCES:
            mask=c.source_mask(docs,[s for s in p.SOURCES if s!=removed])
            self.assertEqual(mask[p.SOURCES.index(removed)],0);self.assertEqual(sum(mask),3)
        self.assertEqual(c.source_mask(docs,[]),[0,0,0,0])

    def test_missing_text_does_not_create_fake_documents(self):
        r=record();r['description']=None;r['social_media_posts']=[]
        docs=c.text_documents(r,p.load()['source_fields'])
        self.assertEqual(c.source_mask(docs,p.SOURCES),[0,1,1,0])

    def test_explicit_epss_disclosure_is_quarantined(self):
        r=record();r['description']='This CVE has an EPSS score of 0.83.'
        docs=c.text_documents(r,p.load()['source_fields'])
        self.assertEqual(c.source_mask(docs,p.SOURCES),[0,1,1,1])
        self.assertTrue(next(d for d in docs if d['source']=='description')['quarantined_epss_disclosure'])

    def test_primary_missing_is_not_zero(self):
        r=record();r['epss_scores']['2024-03-01']=None
        self.assertFalse(c.primary_label(r)['observed'])
        r['epss_scores']['2024-03-01']=0
        self.assertTrue(c.primary_label(r)['observed']);self.assertEqual(c.primary_label(r)['value'],0)
        del r['epss_scores']['2024-03-01']
        with self.assertRaises(ValueError):c.primary_label(r)

    def test_cvss_encoding_and_structured_removal(self):
        fields=structured();state=c.StructuredEncoder.fit([{'cve':'A','split':'train',**fields}])
        v,issues=c.StructuredEncoder.transform(fields,state,['cvss','reference_count'])
        self.assertEqual(len(v),len(c.cvss.FEATURES)+2)
        self.assertAlmostEqual(v[-5],.98);self.assertEqual(v[-1],1)
        removed,_=c.StructuredEncoder.transform(fields,state,[])
        self.assertEqual(removed,[0.]*len(v))

    def test_training_only_scaling_and_strict_allowlist(self):
        fields=structured()
        with self.assertRaises(ValueError):c.StructuredEncoder.fit([{'cve':'A','split':'validation',**fields}])
        state=c.StructuredEncoder.fit([{'cve':'A','split':'train',**fields}])
        changed={**fields,'n_references_nvd':999999}
        before=deepcopy(state);c.StructuredEncoder.transform(changed,state,['reference_count']);self.assertEqual(state,before)
        with self.assertRaises(ValueError):c.StructuredEncoder.transform({**fields,'in_vckev':True},state,['cvss'])
        with self.assertRaises(ValueError):c.StructuredEncoder.fit([{'cve':'A','split':'train',**fields,'target':.9}])

    def test_multihot_unseen_missing_and_train_vocab(self):
        vocab=c.fit_multihot([['Reddit','Mastodon'],['Reddit'],None]);self.assertEqual(vocab,['Mastodon','Reddit'])
        self.assertEqual(c.encode_multihot(['Reddit','Telegram'],vocab),[0,1,0,1])
        self.assertEqual(c.encode_multihot(None,vocab),[0,0,1,0])
        with self.assertRaises(ValueError):c.fit_multihot([['X']],split='validation')

    def test_all_daily_anchor_horizon_pairs(self):
        r=c.temporal_coverage(record(),'train',POLICY)
        self.assertEqual(r['anchor_count'],90);self.assertEqual(r['pair_count'],4095)
        self.assertTrue(r['eligible'][0,89]);self.assertTrue(r['eligible'][89,0]);self.assertFalse(r['eligible'][89,1])

    def test_future_targets_cannot_change_prefix_inputs(self):
        values=[.1]*91;before=c.past_channels(values,20);values[21:]=[.99]*70
        self.assertEqual(c.past_channels(values,20),before)
        values[0]=None;channels=c.past_channels(values,20)
        self.assertEqual(channels[0],[0.,0,0.,0,0.])

    def test_missing_histories_and_targets_masked(self):
        r=record();pub=date(2024,1,1)
        for day in range(21):r['epss_scores'][(pub+timedelta(days=day)).isoformat()]=None
        out=c.temporal_coverage(r,'train',POLICY)
        self.assertFalse(out['eligible'][:21].any());self.assertEqual(out['pair_count'],2415)
        r['epss_scores'][(pub+timedelta(days=90)).isoformat()]=None
        out=c.temporal_coverage(r,'train',POLICY);self.assertEqual(out['pair_count'],2346)

    def test_date_boundary_prevents_late_value_inspection(self):
        r=record('2025-01-01');upper=POLICY['validation_start']
        for stamp in r['epss_scores']:
            if stamp>=upper:r['epss_scores'][stamp]=float('nan')
        out=c.temporal_coverage(r,'train',POLICY)
        self.assertFalse(out['eligible'][60:].any());self.assertTrue(out['pair_count']>0)

    def test_test_series_are_refused(self):
        with self.assertRaises(ValueError):c.temporal_coverage(record(),'test',POLICY)

    def test_validation_early_anchors_do_not_move_to_train(self):
        r=record('2025-02-15');out=c.temporal_coverage(r,'validation',POLICY)
        self.assertFalse(out['date_admissible'][:15].any());self.assertTrue(out['date_admissible'][15].any())

    def test_metrics_defined_and_undefined_cases(self):
        out=e.score_predictions([.0,.2],[.0,.2]);self.assertEqual(out['mae'],0)
        self.assertEqual(out['threshold_categories']['0.1']['f1'],1)
        out=e.score_predictions([0,0],[0,0]);self.assertIsNone(out['r2']);self.assertIsNone(out['spearman'])
        self.assertIsNone(out['threshold_categories']['0.1']['roc_auc'])
        self.assertIsNone(out['threshold_categories']['0.1']['f1'])
        with self.assertRaises(ValueError):e.score_predictions([float('nan')],[.1])

    def test_cve_anchor_horizon_balanced_evaluation(self):
        rows=[{'cve':'A','anchor':0,'lead':1,'actual':0.,'prediction':1.},
              {'cve':'B','anchor':0,'lead':1,'actual':0.,'prediction':0.},
              {'cve':'B','anchor':0,'lead':2,'actual':0.,'prediction':0.},
              {'cve':'B','anchor':1,'lead':1,'actual':0.,'prediction':0.}]
        self.assertEqual(e.balanced_forecast_mae(rows),.5)
        with self.assertRaises(ValueError):e.balanced_forecast_mae(rows+[rows[0]])

    def test_snapshot_masks_remove_sources_and_structured_values(self):
        import torch
        from epss.consolidated.models import SnapshotRegressor
        torch.manual_seed(17)
        model=SnapshotRegressor(3,[1,0,0,0],False,hidden=4)
        sources=torch.randn(2,4,4);present=torch.ones((2,4),dtype=torch.bool);structured=torch.randn(2,3)
        before=model(sources,present,structured)[0]
        changed=sources.clone();changed[:,1:]=999
        torch.testing.assert_close(before,model(changed,present,structured+200)[0],rtol=0,atol=0)
        result,weights,mask=model(sources,torch.zeros_like(present),structured)
        self.assertTrue(torch.isfinite(result).all());self.assertEqual(float(weights.sum()),0)

    def test_daily_model_padding_and_all_feasible_leads(self):
        import torch
        from epss.consolidated.models import DailyForecaster
        model=DailyForecaster();past=torch.tensor([c.past_channels([.1]*91,9),c.past_channels([.2]*91,9)])
        lengths=torch.tensor([5,10]);leads=torch.tensor([[1.,30.,86.],[1.,30.,81.]])
        a=model(past,lengths,leads);self.assertTrue(torch.equal(a['prediction'],a['persistence']))
        with torch.no_grad():model.head[-1].weight.fill_(.01)
        before=model(past,lengths,leads)['prediction'];changed=past.clone();changed[0,5:]=500
        torch.testing.assert_close(before,model(changed,lengths,leads)['prediction'],rtol=0,atol=0)
        before.sum().backward();self.assertTrue(any(v.grad is not None for v in model.parameters()))
        with self.assertRaises(ValueError):model(past,lengths,leads+10)

    def test_reuses_edge_layer_and_non_graph_control_ignores_edges(self):
        import torch
        from epss.consolidated.models import SourceEncoder
        from epss.edge_aware_layers import EDGE_FEATURE_DIM
        model=SourceEncoder(6,3,2,{'DEP':0,'SEC_AFFECTS':1},hidden=4,layers=1)
        x=torch.randn(3,6);x[:,-1]=1
        edges=torch.tensor([[0,1],[1,2]]);types=torch.tensor([0,1]);features=torch.zeros(2,EDGE_FEATURE_DIM)
        batch=torch.zeros(3,dtype=torch.long);mask=torch.ones(3,dtype=torch.bool)
        a=model(x,edges,types,features,batch,mask,'pooled_text')
        b=model(x,edges.flip(0),types.flip(0),features+1,batch,mask,'pooled_text')
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        g=model(x,edges,types,features,batch,mask,'security_tpg');self.assertEqual(g.shape,(1,4));g.sum().backward()


if __name__=='__main__':unittest.main()
