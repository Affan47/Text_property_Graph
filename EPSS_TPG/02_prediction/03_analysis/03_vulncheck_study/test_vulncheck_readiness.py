"""Focused VK01 contract tests using synthetic fixtures, not experimental predictions."""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.vulncheck_study import readiness as vk


def record(cve='CVE-2025-10001', listed=True, description='A buffer overflow affects the product.'):
    return {'cve':cve,'in_vckev':listed,'published_timestamp':'2025-01-01T12:00:00',
            'published_date':'2025-01-01','vckev_date_added':'2025-01-01T00:00:00Z',
            'description':description,'social_media_posts':[]}


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.policy=vk.read(vk.CONFIG)

    def test_only_actual_booleans_are_labels(self):
        self.assertEqual(vk.label(True),1)
        self.assertEqual(vk.label(False),0)
        for value in (None,1,0,'true','false',float('nan'),[],{}):
            self.assertIsNone(vk.label(value))

    def test_single_score_is_not_required(self):
        r=record();r['epss_scores']={};r['epss_score']=None
        out=vk.audit([r],{r['cve']:{'split':'train'}},self.policy,'full',
                     [{'cve':r['cve'],'split':'train','primary_eligible':False}])
        self.assertEqual(out['labels_audit.json'][0]['target'],1)
        self.assertEqual(out['summary.json']['previous_training_without_day60_now_label_eligible'],1)
        self.assertFalse(out['summary.json']['training_ready'])

    def test_negative_dates_do_not_remove_snapshot_labels(self):
        r=record();info=vk.temporal(r)
        self.assertEqual(info['delta_days_stored'],-.5)
        self.assertTrue(info['negative_delta'])
        self.assertTrue(info['same_day_negative'])
        self.assertEqual(vk.label(r['in_vckev']),1)

    def test_timezone_deterministic(self):
        for zone in ('Europe/Rome','Pacific/Honolulu'):
            with patch.dict(os.environ,{'TZ':zone}):
                a,flag=vk.timestamp('2025-01-01T12:00:00')
                b,_=vk.timestamp('2025-01-01T14:00:00+02:00')
                self.assertEqual(a,b);self.assertEqual(flag,'assumed_utc')
        self.assertIsNone(vk.timestamp('not-a-date')[0])

    def test_catalogue_cues_quarantine_without_label_use(self):
        for text in ('Listed in VulnCheck.', 'Added to CISA KEV.',
                     'Known exploited vulnerabilities catalogue',
                     'https://www.cisa.gov/known-exploited-vulnerabilities-catalog'):
            self.assertTrue(vk.inspect_text(text,'CVE-2025-10001')['candidate_quarantine'])
        self.assertFalse(vk.inspect_text('Kevin found a bug.','CVE-2025-10001')['candidate_quarantine'])

    def test_epss_disclosure_and_exploitation_differ(self):
        self.assertTrue(vk.inspect_text('EPSS score: 0.98','CVE-2025-10001')['candidate_quarantine'])
        x=vk.inspect_text('This is actively exploited.','CVE-2025-10001')
        self.assertTrue(x['exploitation_statement']);self.assertFalse(x['candidate_quarantine'])

    def test_other_cve_is_review_not_duplicate_or_automatic_drop(self):
        x=vk.inspect_text('CVE-2025-10002 is discussed with CVE-2025-10001.','CVE-2025-10001')
        self.assertEqual(x['other_cve_ids'],['CVE-2025-10002'])
        self.assertTrue(x['target_cve_mentioned']);self.assertFalse(x['candidate_quarantine'])

    def test_predictor_allowlist_blocks_derived_leakage(self):
        vk.check_predictor_keys(['description','cvss_vector'],self.policy)
        for field in ('in_vckev','vckev_date_added','vendor_project','vckev_count','in_vckev_mask','epss_scores'):
            with self.assertRaises(ValueError):vk.check_predictor_keys(['description',field],self.policy)

    def test_source_masks_dont_drop_cves(self):
        r=record(description='Added to VulnCheck KEV.')
        r['social_media_posts']=[{'post':'Product advisory.','platform':'Reddit'}]*2
        out=vk.audit([r],{r['cve']:{'split':'validation'}},self.policy,'full',[])
        m=out['source_masks_audit.json'][0]
        self.assertEqual(m['raw_source_mask']['description'],1)
        self.assertEqual(m['candidate_source_mask']['description'],0)
        self.assertEqual(m['candidate_source_mask']['smp'],1)
        self.assertEqual(len(out['labels_audit.json']),1)
        self.assertEqual(out['summary.json']['source_format_issues']['within_cve_source_duplicate'],1)

    def test_no_review_examples_from_previous_test(self):
        r=record(description='VulnCheck KEV catalogue')
        out=vk.audit([r],{r['cve']:{'split':'test'}},self.policy,'full',[])
        self.assertEqual(out['review_examples.json'],[])
        self.assertEqual(out['source_audit.json'][0]['previous_role'],'test')
        self.assertFalse(out['summary.json']['independent_final_evaluation_ready'])

    def test_duplicates_and_manifest_mismatch_fail(self):
        r=record()
        with self.assertRaises(ValueError):vk.audit([r,r],{r['cve']:{'split':'train'}},self.policy,'full',[])
        with self.assertRaises(ValueError):vk.audit([r],{},self.policy,'full',[])
        with self.assertRaises(ValueError):vk.audit([],{},self.policy,'full',[])

    def test_normalized_overlap_preserves_distinct_cves(self):
        self.assertEqual(vk.normalize(' Product\n A '),vk.normalize('product a'))
        rows=[{'normalized_text_sha256':'abc','cve':c,'previous_role':role,'source':'description'}
              for c,role in [('CVE-2025-10001','train'),('CVE-2025-10002','validation')]]
        g=vk.overlap_groups(rows)
        self.assertEqual(len(g),1);self.assertEqual(len(g[0]['cves']),2)

    def test_matrix_bounded_and_not_trained(self):
        rows=vk.matrix(self.policy)
        self.assertEqual(len(rows),18)
        self.assertEqual(len({r['id'] for r in rows}),18)
        self.assertEqual(sum(len(r.get('seeds',[])) for r in rows),48)
        self.assertTrue(all(r['target']=='in_vckev' and not r['epss_input'] for r in rows))
        self.assertTrue(all(r['status']=='proposed_not_trained' for r in rows))

    def test_smoke_covers_sources_without_using_labels(self):
        a=record();b=record(cve='CVE-2025-10002',listed=False)
        b[self.policy['source_fields']['github_summary']]='GitHub reference summary.'
        p=copy.deepcopy(self.policy);p['smoke_text_counts']={'train':1}
        m={r['cve']:{'split':'train'} for r in (a,b)}
        first=vk.select_text_ids([a,b],m,p,'smoke')
        a['in_vckev']=False;b['in_vckev']=True
        self.assertEqual(first,{b['cve']})
        self.assertEqual(first,vk.select_text_ids([a,b],m,p,'smoke'))

    def test_deterministic_and_nonmutating(self):
        r=record(); original=copy.deepcopy(r); m={r['cve']:{'split':'train'}}
        a=vk.audit([r],m,self.policy,'full',[])
        b=vk.audit([r],m,self.policy,'full',[])
        self.assertEqual(a,b);self.assertEqual(r,original)

    def test_missing_label_is_counted_not_negative(self):
        r=record(listed=None)
        out=vk.audit([r],{r['cve']:{'split':'train'}},self.policy,'full',[])
        self.assertEqual(out['summary.json']['labels']['invalid_or_missing_label'],1)
        self.assertEqual(out['summary.json']['labels']['not_listed'],0)


if __name__=='__main__':unittest.main()
