"""Synthetic checks of RD01 boundaries, exposure memory and event coverage."""
from datetime import date, timedelta
import unittest

from audit_rd01_splits import (CONFIG, read, role_for, post_group, keep_post,
                               groups_for, daily_cells, component_reason, event_window)


class RD01Tests(unittest.TestCase):
    def setUp(self):
        self.p=read(CONFIG)

    def test_publication_boundaries(self):
        self.assertEqual(role_for('2021-04-13',self.p),'outside')
        self.assertEqual(role_for('2021-04-14',self.p),'train')
        self.assertEqual(role_for('2023-12-31',self.p),'train')
        self.assertEqual(role_for('2024-01-01',self.p),'validation')
        self.assertEqual(role_for('2025-01-01',self.p),'test')
        self.assertEqual(role_for('2026-11-04',self.p),'outside')

    def test_strict_threshold(self):
        self.assertTrue(keep_post(10,10)); self.assertFalse(keep_post(11,10))
        self.assertTrue(keep_post(25,25)); self.assertFalse(keep_post(26,25))
        self.assertTrue(keep_post(10000,None))

    def test_disjoint_text_groups(self):
        self.assertEqual(post_group('CVE-2021-1234'),'bare_id_only')
        self.assertEqual(post_group('Patch CVE-2021-1234 today'),'single_cve_descriptive')
        self.assertEqual(post_group('CVE-2021-1234 and CVE-2021-1235'),'multi_cve_text')

    def test_one_score_is_not_forecast_pair(self):
        self.assertEqual(daily_cells([]),0)
        self.assertEqual(daily_cells([30]),0)
        self.assertEqual(daily_cells([30,31]),1)
        self.assertEqual(daily_cells([1,60],59),0)

    def test_lazy_cell_count_matches_enumeration(self):
        for obs in ([0,1,3,90],[1,4,60],[88,90]):
            for bound in (0,1,30,60,90):
                expected=sum(any(d<=t for d in obs) and u in obs and u<=bound
                             for t in range(90) for u in range(t+1,91))
                self.assertEqual(daily_cells(obs,bound),expected)

    def test_transitive_components_and_whole_post_removal(self):
        records={x:{} for x in ('A','B','C','D')}
        posts={'x':{'cves':{'A','B','C'},'endpoints':{'A','B','C','old'}},
               'y':{'cves':{'C','D'},'endpoints':{'C','D'}}}
        self.assertEqual(len(groups_for(records,posts)),1)
        self.assertEqual(sorted(len(g['current']) for g in groups_for(records,posts,2)),[1,1,2])

    def test_exposure_not_erased_by_isolation(self):
        records={'A':{'published_date':'2025-01-01'}}
        for mode in self.p['exposure_policies']:
            reason,role=component_reason(['A'],records,{'A':{'test'}},mode,self.p)
            self.assertEqual(reason,'prior_test_connected'); self.assertIsNone(role)

    def test_prior_role_match_and_conflict(self):
        records={'A':{'published_date':'2024-01-01'}}
        self.assertEqual(component_reason(['A'],records,{'A':{'validation'}},'retain_matching_prior_roles',self.p),(None,'validation'))
        self.assertEqual(component_reason(['A'],records,{'A':{'train'}},'retain_matching_prior_roles',self.p)[0],'prior_role_conflict')
        self.assertEqual(component_reason(['A'],records,{'A':{'excluded'}},'retain_matching_prior_roles',self.p)[0],'prior_role_conflict')

    def test_cross_period_and_prior_test_protection(self):
        records={'A':{'published_date':'2023-01-01'},'B':{'published_date':'2024-01-01'}}
        self.assertEqual(component_reason(['A','B'],records,{'A':set(),'B':set()},'never_exposed',self.p)[0],'cross_publication_intervals')
        records={'A':{'published_date':'2025-01-01'}}
        self.assertEqual(component_reason(['A'],records,{'A':{'train'}},'retain_matching_prior_roles',self.p)[0],'prior_exposure_barred_from_test')

    def test_window_missing_inside_not_imputed(self):
        pub=date(2022,1,1); post=pub+timedelta(days=7)
        dates={(pub+timedelta(days=i)).isoformat() for i in range(91)}
        missing=post.isoformat(); observed=dates-{missing}
        statuses={d:'OBSERVED' for d in dates}; statuses[missing]='DOWNLOAD_ERROR'
        r=event_window(pub,post,observed,statuses,7,[])
        self.assertFalse(r['complete']); self.assertEqual(r['reason'],'missing_daily_score')
        self.assertEqual(r['missing_reasons'],{'DOWNLOAD_ERROR':1})
        self.assertEqual(event_window(pub,pub,observed,statuses,1,[])['reason'],'outside_stored_window')

    def test_release_boundary_local_not_global_flag(self):
        pub=date(2022,1,1); dates={(pub+timedelta(days=i)).isoformat() for i in range(91)}
        statuses={d:'OBSERVED' for d in dates}; event=pub+timedelta(days=10)
        self.assertFalse(event_window(pub,event,dates,statuses,1,[pub+timedelta(days=30)])['crosses_release'])
        self.assertFalse(event_window(pub,event,dates,statuses,1,[event-timedelta(days=1)])['crosses_release'])
        self.assertTrue(event_window(pub,event,dates,statuses,1,[event+timedelta(days=1)])['crosses_release'])


if __name__=='__main__':
    unittest.main()
