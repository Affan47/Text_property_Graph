"""Offline safety checks for targeted recovery and source isolation."""
import copy
import json
import unittest

from reddit_gates import CATALOGUE, ALLOWLIST, UnionFind, dedup_record, receipt_audit
from recover_reddit_epss_v2 import apply_recovery
from specify_reddit_studies import specifications, size_distribution
from reddit_official_recovery import git_blob_hash


class GateTests(unittest.TestCase):
    def record(self):
        r={k:None for k in ALLOWLIST}
        r.update(cve='CVE-2021-1000',description='Example',published_date='2021-04-20',
                 in_vckev=False,n_references_nvd=3,all_sources_urls=['a','a','b'],
                 github_only_urls=['a','a'],non_github_urls=['b'],social_media_posts=[
                     {'post':'CVE-2021-1000','date':'2021-04-20','platform':'reddit'},
                     {'post':'CVE-2021-1000','date':'2021-04-20','platform':'reddit'}])
        for k in CATALOGUE: r[k]='restricted'
        return r

    def test_dedup_keeps_raw_count_and_quarantines_all_labels(self):
        r=self.record(); original=copy.deepcopy(r)
        p,urls,removed,posts=dedup_record(r)
        self.assertEqual(r,original)
        self.assertEqual(p['n_references_nvd'],3)
        self.assertEqual(p['all_sources_urls'],['a','b'])
        self.assertEqual(p['distinct_reference_counts']['all_sources_urls'],2)
        self.assertEqual(removed,1); self.assertTrue(posts[0]['bare_cve_id'])
        self.assertFalse(set(CATALOGUE+('in_vckev',))&set(p))

    def test_different_dates_and_urls_are_not_fuzzy_deduplicated(self):
        r=self.record(); r['social_media_posts'][1]['date']='2021-04-21'
        r['all_sources_urls']=['https://a/p','https://a/p/']
        p,_,removed,_=dedup_record(r)
        self.assertEqual(removed,0); self.assertEqual(len(p['all_sources_urls']),2)

    def test_components_transitive_and_order_invariant(self):
        edges=[['a','b'],['b','external'],['external','c']]
        first=UnionFind(['a','b','c','d']); second=UnionFind(['d','c','b','a'])
        for e in edges: first.join(e)
        for e in reversed(edges): second.join(reversed(e))
        self.assertEqual(first.groups(['a','b','c','d']),second.groups(['a','b','c','d']))
        self.assertEqual(first.find('a'),first.find('c'))
        self.assertNotEqual(first.find('a'),first.find('d'))

    def test_failed_lookup_is_not_inferred_from_legacy_flag(self):
        report,labels=receipt_audit(None,[self.record()])
        self.assertEqual(report['verified_not_listed'],0)
        self.assertEqual(report['unknown'],1); self.assertEqual(labels,{})

    def test_recovery_changes_only_download_error(self):
        r={'cve':'CVE-2021-1000','epss_scores':{'a':None,'b':None,'c':.2},
           'epss_score_status':{'a':'DOWNLOAD_ERROR','b':'BEFORE_EPSS','c':'OBSERVED'},'other':'kept'}
        changes,counts=apply_recovery([r],{'a':{'status':'VALID','scores':{'CVE-2021-1000':0.0},
                                                    'metadata':{'date_validation':'header'}}})
        self.assertEqual(r['epss_scores'],{'a':0.0,'b':None,'c':.2})
        self.assertEqual(r['epss_score_status']['b'],'BEFORE_EPSS')
        self.assertEqual(counts['recovered_scores'],1); self.assertEqual(r['other'],'kept')

    def test_valid_absence_and_failed_archive_remain_missing_not_zero(self):
        for status,expected in [('VALID','CVE_NOT_IN_SNAPSHOT'),('DOWNLOAD_ERROR','DOWNLOAD_ERROR')]:
            r={'cve':'CVE-2021-1000','epss_scores':{'a':None},'epss_score_status':{'a':'DOWNLOAD_ERROR'}}
            apply_recovery([r],{'a':{'status':status,'scores':{},'metadata':{'date_validation':'header'}}})
            self.assertIsNone(r['epss_scores']['a']); self.assertEqual(r['epss_score_status']['a'],expected)

    def test_refuse_to_overwrite_score_on_failed_entry(self):
        r={'cve':'CVE-2021-1000','epss_scores':{'a':.5},'epss_score_status':{'a':'DOWNLOAD_ERROR'}}
        with self.assertRaises(ValueError): apply_recovery([r],{'a':{'status':'DOWNLOAD_ERROR'}})

    def test_specifications_are_bounded_and_disabled(self):
        specs=specifications()
        self.assertEqual(set(specs),{'RD01','RD02','RD03','RD04'})
        for spec in specs.values():
            self.assertFalse(spec['training_enabled']); self.assertFalse(spec['test_access_enabled'])
            self.assertEqual(spec['seeds'],[17,29,43])
        for name in ['RD02','RD03']:
            self.assertEqual(sum(m['fits'] for m in specs[name]['models']),specs[name]['total_fits'])
            self.assertEqual(specs[name]['graph']['subtype_vocabulary']['SEVERITY'],31)
            self.assertEqual(len(specs[name]['graph']['subtype_vocabulary']),32)
        self.assertTrue(specs['RD01']['not_yet_assigned'])
        self.assertFalse(specs['RD04']['event_study']['causal_claim'])

    def test_git_blob_identity_includes_object_header(self):
        self.assertEqual(git_blob_hash(b''),'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391')

    def test_component_counts_survive_json_roundtrip(self):
        result=size_distribution([{'current_count':1},{'current_count':1},{'current_count':12}])
        self.assertEqual(result,{'1':2,'12':1})
        self.assertEqual(result,json.loads(json.dumps(result)))


if __name__=='__main__': unittest.main()
