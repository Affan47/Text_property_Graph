"""Offline tests for EPSS targets, temporal splits and predictor isolation."""
import copy
import json
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss.forecast_data import (make_record, prepare, predictor_view, sequence_arrays,
    fit_training_preprocessor, transform_statistics, read_json, candidate_split,
    validate_prepared, duplicate_groups, normalized_url)
from epss.forecast_data import (validated_split_policy, DEFAULT_SPLIT_POLICY,
    task_cohorts, save_prepared, load_prepared, write_json, sha256, VERSION)


def fixture(cve='CVE-2023-1234', published='2023-06-01'):
    p = date.fromisoformat(published)
    return {'cve': cve, 'published_date': published, 'published_timestamp': published+'T12:15:00.000',
            'description': 'Unique description '+cve, 'all_sources_urls': [], 'social_media_posts': [],
            'epss_scores': {(p+timedelta(days=d)).isoformat(): 0.1 for d in range(91)}}


def set_day(r, day, value):
    r['epss_scores'][(date.fromisoformat(r['published_date'])+timedelta(days=day)).isoformat()] = value


def trio():
    return [fixture(), fixture('CVE-2024-1234', '2024-06-01'), fixture('CVE-2025-1234', '2025-06-01')]


class ForecastDataTests(unittest.TestCase):
    def test_future_mutation_does_not_change_inputs(self):
        r = fixture()
        before, _ = make_record(r)
        for day in range(31,91):
            set_day(r, day, day/100)
        after, labels = make_record(r)
        self.assertEqual(before, after)
        self.assertEqual(labels['y']['endpoint'], .6)
        self.assertEqual(max(after['X']['day_offsets']), 30)

    def test_missing_input_vs_observed_zero(self):
        r = fixture()
        set_day(r, 0, None)
        set_day(r, 1, 0.0)
        x, y = make_record(r)
        a = sequence_arrays(x)
        self.assertEqual(a['values'][:2], [0.0, 0.0])
        self.assertEqual(a['observed_mask'][:2], [0,1])
        self.assertEqual(a['age_known_mask'][:2], [0,1])
        self.assertIsNone(x['X']['values'][0])
        self.assertTrue(y['eligible']['endpoint'])

    def test_age_uses_only_past(self):
        r = fixture()
        for d in (0,1,27,29,30):
            set_day(r, d, None)
        x, _ = make_record(r)
        self.assertEqual(x['X']['age_since_observation_days'][:3], [None,None,0])
        self.assertEqual(x['X']['age_since_observation_days'][27:], [1,0,1,2])

    def test_missing_endpoint_not_zero(self):
        r = fixture()
        set_day(r,60,None)
        _, y = make_record(r)
        self.assertIsNone(y['y']['endpoint'])
        self.assertFalse(y['eligible']['endpoint'])
        self.assertTrue(y['eligible']['trajectory'])

    def test_missing_day30_not_missing_endpoint(self):
        r = fixture()
        set_day(r,30,None)
        x, y = make_record(r)
        self.assertTrue(y['eligible']['endpoint'])
        self.assertFalse(y['eligible']['delta'])
        self.assertIsNone(y['y']['direction'])
        self.assertEqual(x['X']['statistics']['last_observation_age_days'],1)

    def test_future_gap_not_exact_extrema(self):
        r = fixture()
        set_day(r,40,None)
        _, y = make_record(r)
        self.assertTrue(y['eligible']['endpoint'])
        self.assertFalse(y['eligible']['minmax'])
        self.assertIsNone(y['y']['minmax'])

    def test_extrema_exclude_day30_and_day61(self):
        r = fixture()
        set_day(r,30,1.0)
        set_day(r,31,.03)
        set_day(r,60,.4)
        set_day(r,61,0.0)
        _, y = make_record(r)
        self.assertEqual(y['y']['minmax'], [.03,.4])
        self.assertEqual(y['y']['delta'], -.6)

    def test_epsilon_boundary_is_stable(self):
        for anchor,endpoint,expected in [(.01,.02,'stable'),(.02,.01,'stable'),(.01,.02001,'increase'),(.02,.00999,'decrease')]:
            r = fixture()
            set_day(r,30,anchor)
            set_day(r,60,endpoint)
            self.assertEqual(make_record(r)[1]['y']['direction'],expected)

    def test_all_missing_input_retains_labels_but_ineligible(self):
        r = fixture()
        for day in range(31):
            set_day(r,day,None)
        x, y = make_record(r)
        self.assertIsNone(x['X']['statistics'])
        self.assertTrue(y['label_mask']['endpoint'])
        self.assertFalse(y['eligible']['endpoint'])

    def test_single_input_slope_explicit(self):
        r = fixture()
        for day in range(30):
            set_day(r,day,None)
        stats = make_record(r)[0]['X']['statistics']
        self.assertEqual(stats['slope_defined'],0)
        self.assertEqual(stats['slope_per_day'],0.0)

    def test_text_catalogue_and_metadata_cannot_enter_inputs(self):
        r = fixture()
        before, _ = make_record(r)
        r.update(description='Future exploitation evidence', in_vckev=True,
                 epss_version_changed=True, llm_summary='future', social_media_posts=[{'post':'future'}])
        after, _ = make_record(r)
        self.assertEqual(before,after)
        for mode in ('tpg','description','smp','cvss','vckev'):
            with self.assertRaises(ValueError):
                predictor_view(after, ('epss', mode))

    def test_unexpected_predictor_rejected(self):
        x, _ = make_record(fixture())
        x['X']['vckev_date_added'] = '2025-01-01'
        with self.assertRaises(ValueError):
            predictor_view(x)

    def test_forged_age_or_future_date_rejected(self):
        for mode in ('age','date','statistics'):
            x, _ = make_record(fixture())
            if mode == 'age': x['X']['age_since_observation_days'][5] = -10
            elif mode == 'date': x['X']['dates'][-1] = '2025-01-01'
            else: x['X']['statistics']['mean'] = .9
            with self.assertRaises(ValueError): predictor_view(x)

    def test_canonical_ids_not_collapsed(self):
        rows = [fixture('CVE-2023-1234'),fixture('CVE-2023-5678')]
        x, *_ = prepare(rows)
        self.assertEqual(len(x),2)

    def test_duplicate_id_rejected(self):
        with self.assertRaises(ValueError): prepare([fixture(),fixture()])

    def test_publication_date_not_id_year(self):
        x, y = make_record(fixture('CVE-2013-10001','2022-05-17'))
        self.assertEqual(x['X']['dates'][0], '2022-05-17')
        self.assertEqual(y['target_date'],'2022-07-16')

    def test_inclusive_final_date(self):
        self.assertEqual(candidate_split('2023-12-01','2023-12-31')[0], 'train')
        self.assertEqual(candidate_split('2023-12-02','2024-01-01')[0], 'excluded')
        self.assertEqual(candidate_split('2024-12-01','2024-12-31')[0], 'validation')

    def test_cross_fold_duplicates_quarantined(self):
        rows = trio()
        rows[0]['description'] = rows[2]['description'] = ' SAME   description '
        _, _, m, _, _ = prepare(rows)
        self.assertEqual(m[rows[0]['cve']]['split'], 'excluded')
        self.assertEqual(m[rows[2]['cve']]['split'], 'excluded')
        self.assertEqual(m[rows[1]['cve']]['split'], 'validation')

    def test_transitive_url_and_post_grouping(self):
        rows = trio()
        rows[0]['all_sources_urls'] = ['https://EXAMPLE.org/advisory#section1']
        rows[1]['all_sources_urls'] = ['https://example.org/advisory#section2']
        rows[1]['social_media_posts'] = rows[2]['social_media_posts'] = [{'post':'Shared discussion'}]
        _, _, m, g, _ = prepare(rows)
        self.assertEqual(len(g),1)
        self.assertTrue(all(r['split']=='excluded' for r in m.values()))

    def test_cve_url_queries_not_stripped(self):
        self.assertNotEqual(normalized_url('https://example.org/?cve=1'), normalized_url('https://example.org/?cve=2'))

    def test_training_preprocessing_ignores_test_values(self):
        rows = trio()
        x,y,m,_,_ = prepare(rows)
        first = fit_training_preprocessor(x,y,m)
        for day in range(91): set_day(rows[2],day,.99)
        x2,y2,m2,_,_ = prepare(rows)
        second = fit_training_preprocessor(x2,y2,m2)
        self.assertEqual(first,second)
        self.assertEqual(first['fit_cve_ids'],[rows[0]['cve']])
        self.assertTrue(all(abs(v)<1e-12 for v in transform_statistics(x[rows[0]['cve']],first)))

    def test_invalid_split_injection_rejected(self):
        x,y,m,_,_ = prepare(trio())
        m['CVE-2025-1234']['split'] = 'train'
        with self.assertRaises(ValueError): fit_training_preprocessor(x,y,m)

    def test_group_leak_injection_rejected(self):
        x,y,m,_,_ = prepare(trio())
        m['CVE-2025-1234']['group_id'] = m['CVE-2023-1234']['group_id']
        with self.assertRaises(ValueError): validate_prepared(x,y,m)

    def test_invalid_values_and_dates_rejected(self):
        for value in (float('nan'),float('inf'),-.1,1.1,True,'0.1'):
            r=fixture(); set_day(r,3,value)
            with self.assertRaises(ValueError): make_record(r)
        r=fixture(); r['published_date']='2020-01-01'
        with self.assertRaises(ValueError): make_record(r)

    def test_json_duplicate_keys_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'invalid.json'
            p.write_text('{"cve":"a","cve":"b"}')
            with self.assertRaises(ValueError): read_json(p)

    def test_mutated_targets_are_rejected(self):
        for field in ('delta','direction','minmax'):
            x,y,m,_,_ = prepare(trio())
            y['CVE-2023-1234']['y'][field] = None
            with self.assertRaises(ValueError): validate_prepared(x,y,m)

    def test_missing_date_key_rejected(self):
        r=fixture()
        del r['epss_scores'][r['published_date']]
        with self.assertRaises(ValueError): make_record(r)

    def test_deterministic_build_order(self):
        self.assertEqual(prepare(trio()),prepare(list(reversed(trio()))))

    def test_configurable_boundaries(self):
        policy = dict(DEFAULT_SPLIT_POLICY, validation_start='2023-07-01', test_start='2024-07-01')
        self.assertEqual(candidate_split('2023-07-01','2023-07-31',policy)[0], 'validation')
        self.assertEqual(candidate_split('2024-07-01','2024-07-31',policy)[0], 'test')
        self.assertEqual(candidate_split('2023-06-15','2023-07-15',policy)[0], 'excluded')
        x,y,m,_,_ = prepare(trio(),policy)
        validate_prepared(x,y,m,policy)
        with self.assertRaises(ValueError):
            validate_prepared(x,y,m)

    def test_policy_dates_must_increase(self):
        for policy in (dict(DEFAULT_SPLIT_POLICY, test_start='2023-01-01'),
                       dict(DEFAULT_SPLIT_POLICY, test_end='2025-01-01'),
                       dict(DEFAULT_SPLIT_POLICY, validation_start='20240101'),
                       dict(DEFAULT_SPLIT_POLICY, observation_end='2024-12-31'),
                       dict(DEFAULT_SPLIT_POLICY, hidden_option=True)):
            with self.assertRaises(ValueError):
                validated_split_policy(policy)

    def test_observation_cutoff_inclusive(self):
        policy = dict(DEFAULT_SPLIT_POLICY,observation_end='2025-06-30')
        self.assertEqual(candidate_split('2025-05-31','2025-06-30',policy)[0],'test')
        self.assertEqual(candidate_split('2025-06-01','2025-07-01',policy),('excluded','target_after_observation_end'))

    def test_test_period_end_is_exclusive(self):
        self.assertEqual(candidate_split('2025-12-31','2026-01-30')[0],'test')
        self.assertEqual(candidate_split('2026-01-01','2026-01-31')[0],'excluded')

    def test_target_cannot_precede_anchor(self):
        for end in ('2023-01-01','2023-01-02'):
            with self.assertRaises(ValueError):
                candidate_split('2023-01-02',end)

    def test_homepage_only_flagged_not_silently_removed(self):
        rows = trio()
        rows[0]['all_sources_urls'] = rows[2]['all_sources_urls'] = ['https://example.org/']
        evidence = []
        _,_,manifest,_,_ = prepare(rows,evidence=evidence)
        self.assertEqual(manifest[rows[0]['cve']]['split'],'excluded')
        self.assertTrue(evidence[0]['homepage_review_hint'])
        self.assertTrue(evidence[0]['used_for_grouping'])

    def test_reviewed_exact_url_exception(self):
        rows = trio()
        rows[0]['all_sources_urls'] = rows[2]['all_sources_urls'] = ['https://example.org/']
        evidence = []
        _,_,manifest,_,summary = prepare(rows,url_exclusions={'https://example.org/':'Reviewed generic homepage'},evidence=evidence)
        self.assertEqual(manifest[rows[0]['cve']]['split'],'train')
        self.assertEqual(manifest[rows[2]['cve']]['split'],'test')
        self.assertFalse(evidence[0]['used_for_grouping'])
        self.assertEqual(summary['reviewed_url_exclusions']['https://example.org/'],'Reviewed generic homepage')

    def test_url_exception_never_removes_text_link(self):
        rows = trio()
        rows[0]['all_sources_urls'] = rows[2]['all_sources_urls'] = ['https://example.org/']
        rows[0]['description'] = rows[2]['description'] = 'Identical description'
        _,_,manifest,_,_ = prepare(rows,url_exclusions={'https://example.org/':'Generic homepage'})
        self.assertEqual(manifest[rows[0]['cve']]['split'],'excluded')

    def test_unknown_or_unjustified_exception_rejected(self):
        for exceptions in ({'https://unknown.example/':'Reason'}, {'example.org':'Reason'}, {'https://example.org/':''}):
            with self.assertRaises(ValueError):
                prepare(trio(),url_exclusions=exceptions)

    def test_manifest_false_exclusion_rejected(self):
        x,y,m,_,_ = prepare(trio())
        row = m['CVE-2023-1234']
        row.update(split='excluded',exclusion_reason='related_group_crosses_time_folds')
        with self.assertRaises(ValueError):
            validate_prepared(x,y,m)

    def test_forged_manifest_input_id_or_dates_rejected(self):
        for mode in ('id','date','candidate'):
            x,y,m,_,_ = prepare(trio())
            if mode == 'id': x['CVE-2023-1234']['cve']='CVE-2023-9999'
            if mode == 'date': m['CVE-2023-1234']['input_end_date']='2023-01-01'
            if mode == 'candidate': m['CVE-2023-1234']['candidate_split']='test'
            with self.assertRaises(ValueError): validate_prepared(x,y,m)

    def test_preprocessor_bad_dimensions_rejected(self):
        x,y,m,_,_ = prepare(trio())
        p = fit_training_preprocessor(x,y,m)
        p['scales'] = p['scales'][:-1]
        with self.assertRaises(ValueError): transform_statistics(x['CVE-2023-1234'],p)

    def test_lfs_pointer_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'pointer.json'
            path.write_text('version https://git-lfs.github.com/spec/v1\n')
            with self.assertRaisesRegex(ValueError,'Git LFS pointer'):
                read_json(path)

    def bundle(self, tmp):
        raw_path=Path(tmp)/'raw.json'
        write_json(raw_path,trio())
        evidence=[]
        x,y,m,g,s = prepare(trio(),evidence=evidence)
        s.update(source_path=str(raw_path),source_sha256=sha256(raw_path))
        artifacts={'inputs.json':x,'labels.json':y,'split_manifest.json':m,
                   'duplicate_groups.json':g,'preparation_summary.json':s,
                   'training_preprocessor.json':fit_training_preprocessor(x,y,m),
                   'split_policy.json':dict(DEFAULT_SPLIT_POLICY),'reviewed_url_exclusions.json':{},
                   'grouping_evidence.json':evidence,'task_cohorts.json':task_cohorts(y,m)}
        target=Path(tmp)/'prepared'
        save_prepared(target,artifacts,sha256(raw_path))
        return target

    def test_locked_bundle_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=load_prepared(self.bundle(tmp))
            self.assertEqual(data['task_cohorts.json']['endpoint']['train'],['CVE-2023-1234'])

    def test_locked_bundle_detects_changed_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=self.bundle(tmp)
            (target/'inputs.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Prepared file changed'):
                load_prepared(target)

    def test_locked_bundle_detects_changed_raw_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=self.bundle(tmp)
            (Path(tmp)/'raw.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'Raw source changed'):
                load_prepared(target)

    def test_locked_bundle_detects_changed_groups_even_with_new_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=self.bundle(tmp)
            groups=read_json(target/'duplicate_groups.json')
            first=next(iter(groups))
            groups[first]=['CVE-2023-0000']
            write_json(target/'duplicate_groups.json',groups)
            lock=read_json(target/'preparation_lock.json')
            lock['files']['duplicate_groups.json']=sha256(target/'duplicate_groups.json')
            write_json(target/'preparation_lock.json',lock)
            with self.assertRaises(ValueError):load_prepared(target)


if __name__ == '__main__':
    unittest.main()
