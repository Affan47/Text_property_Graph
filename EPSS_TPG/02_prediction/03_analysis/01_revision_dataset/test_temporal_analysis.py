"""Synthetic temporal tests; no live APIs, NLP models or raw dataset mutation."""
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest

from temporal_analysis import (
    HORIZONS, CATEGORIES, DAY, category, date_info, derive_row, distribution,
    elapsed_microseconds, feature_policy, horizon_label, iso, label_counts,
    research_answers, schema, summarize, validate_derived, vector_components, write_table,
)


def record(event='2025-01-21T12:00:00Z', member=True, anchor='2025-01-01T12:00:00Z', cve='CVE-2025-1234'):
    row = {'cve': cve, 'in_vckev': member, 'published_timestamp': anchor,
           'published_date': anchor[:10] if anchor else None, 'url_change_dates': [],
           'cvss_vector_version': 3.1, 'cvss_vector_score': 9.8, 'cvss_vector_severity': 'CRITICAL',
           'cvss_vector': 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H'}
    if event is not None:
        row['vckev_date_added'] = event
    return row


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.anchor = date_info('2025-01-01T12:00:00Z')

    def label(self, days, h, **kwargs):
        event = date_info(iso(self.anchor['dt']+timedelta(days=days)))
        return horizon_label(True, self.anchor, event, h, **kwargs)

    def test_user_example_horizons_are_cumulative(self):
        for delay, expected in ((20,[1,1,1]),(45,[0,1,1]),(75,[0,0,1]),(150,[0,0,0])):
            with self.subTest(delay=delay):
                self.assertEqual([self.label(delay,h)[0] for h in HORIZONS], expected)

    def test_schema_distinguishes_event_metadata_from_binary_targets(self):
        fields = {x['column']: x['meaning'] for x in schema([derive_row(record())])['fields']}
        self.assertIn('not a binary target', fields['target_event_timestamp'])
        self.assertIn('Source field', fields['target_event_field'])
        self.assertIn('0/1 or null', fields['target_30d'])

    def test_asserted_cutoff_duration_partition_includes_short_followup(self):
        raw = [record(event=None, member=False)]
        cutoff = self.anchor['dt'] + timedelta(days=45)
        rows = [derive_row(raw[0], cutoff, cutoff)]
        report = summarize(raw, rows, {}, {})
        coverage = report['observation_coverage']
        self.assertEqual(coverage[0]['primary_full_duration'], 1)
        self.assertEqual(coverage[1]['primary_short_duration'], 1)
        for item in coverage:
            self.assertEqual(sum(item[k] for k in ('primary_full_duration', 'primary_short_duration',
                                                  'primary_unknown_duration')), 1)

    def test_exact_boundaries_and_microseconds(self):
        for h in HORIZONS:
            self.assertEqual(self.label(h,h)[0],1)
            event = self.anchor['dt']+timedelta(days=h, microseconds=1)
            self.assertEqual(horizon_label(True,self.anchor,date_info(iso(event)),h)[0],0)
            self.assertEqual(elapsed_microseconds(self.anchor['dt'],event), h*86400000000+1)

    def test_exclusive_categories_boundary_priority(self):
        a=self.anchor['dt']
        cases=[(-1,'NEGATIVE'),(0,'SAME_DAY'),(.1,'SAME_DAY'),(.75,'0_30_DAYS'),
               (30,'0_30_DAYS'),(30.0001,'31_60_DAYS'),(60,'31_60_DAYS'),
               (60.0001,'61_90_DAYS'),(90,'61_90_DAYS'),(90.0001,'OVER_90_DAYS')]
        for delta,expected in cases:
            self.assertEqual(category(True,a,a+timedelta(days=delta)),expected)
        self.assertEqual(category(True,a,None),'MISSING_DATE')
        self.assertEqual(category(False,a,None),'NOT_LISTED')

    def test_negative_same_day_stays_negative(self):
        r=derive_row(record(event='2025-01-01T00:00:00Z'))
        self.assertEqual(r['delta_category'],'NEGATIVE')
        self.assertEqual(r['negative_subtype'],'same_calendar_day_earlier_timestamp')
        self.assertEqual(r['temporal_status'],'anomalous_or_requires_review')
        self.assertGreater(r['day_sensitivity_delta_upper_days'],0)
        for h in HORIZONS:
            self.assertIsNone(r[f'target_{h}d'])
            self.assertIsNone(r[f'stored_timestamp_target_{h}d'])

    def test_earlier_calendar_day_remains_negative_under_day_sensitivity(self):
        r=derive_row(record(event='2024-12-31T00:00:00Z'))
        self.assertEqual(r['negative_subtype'],'earlier_calendar_day')
        self.assertLess(r['day_sensitivity_delta_upper_days'],0)

    def test_zero_at_anchor_is_not_future_incidence(self):
        r=derive_row(record(event='2025-01-01T12:00:00Z'))
        self.assertEqual(r['delta_category'],'SAME_DAY')
        self.assertIsNone(r['target_30d'])
        self.assertEqual(r['target_30d_status'],'ALREADY_AT_ANCHOR_REQUIRES_REVIEW')

    def test_positive_same_day_event_can_be_target(self):
        r=derive_row(record(event='2025-01-01T13:00:00Z'))
        self.assertEqual(r['delta_category'],'SAME_DAY')
        self.assertEqual(r['target_30d'],1)

    def test_midnight_boundary_sensitivity_keeps_nominal_separate(self):
        r=derive_row(record(event='2025-01-31T00:00:00Z'))
        self.assertEqual(r['delta_days_exact'],29.5)
        self.assertEqual(r['stored_timestamp_target_30d'],1)
        self.assertIsNone(r['target_30d'])
        self.assertEqual(r['target_30d_status'],'PRECISION_AMBIGUOUS')
        self.assertEqual(r['target_60d'],1)

    def test_unknown_snapshot_does_not_create_unlisted_zeros(self):
        r=derive_row(record(event=None,member=False))
        self.assertIsNone(r['target_30d'])
        self.assertEqual(r['target_30d_status'],'UNKNOWN_OBSERVATION_CUTOFF')
        self.assertIsNone(r['target_over_90d'])

    def test_known_short_observation_right_censors_unlisted(self):
        r=derive_row(record(event=None,member=False),self.anchor['dt']+timedelta(days=10))
        for h in HORIZONS:
            self.assertIsNone(r[f'target_{h}d'])
            self.assertEqual(r[f'target_{h}d_status'],'RIGHT_CENSORED')

    def test_full_horizon_observation_and_partial_longer_window(self):
        r=derive_row(record(event=None,member=False),self.anchor['dt']+timedelta(days=60))
        self.assertEqual([r[f'target_{h}d'] for h in HORIZONS],[0,0,None])
        self.assertEqual(r['target_90d_status'],'RIGHT_CENSORED')

    def test_observed_early_event_resolves_target_without_full_horizon_duration(self):
        r=derive_row(record(),self.anchor['dt']+timedelta(days=21))
        self.assertFalse(r['full_followup_90d'])
        self.assertEqual(r['target_90d'],1)
        self.assertEqual(r['censoring_status_90d'],'EVENT_WITHIN_HORIZON')

    def test_observed_late_event_can_resolve_zero_without_global_cutoff(self):
        r=derive_row(record(event='2025-06-01T12:00:00Z'))
        self.assertEqual([r[f'target_{h}d'] for h in HORIZONS],[0,0,0])
        self.assertEqual(r['target_over_90d'],1)

    def test_missing_event_does_not_turn_member_into_negative(self):
        r=derive_row(record(event=None),self.anchor['dt']+timedelta(days=365))
        self.assertEqual(r['delta_category'],'MISSING_DATE')
        self.assertIsNone(r['target_90d'])

    def test_missing_timestamp_does_not_silently_use_publication_date(self):
        raw=record(anchor=None)
        raw['published_date']='2025-01-01'
        r=derive_row(raw)
        self.assertIsNone(r['prediction_anchor_timestamp'])
        self.assertIsNone(r['delta_days_exact'])

    def test_invalid_membership_fails(self):
        for value in ('false',0,1,None):
            raw=record(member=value)
            with self.assertRaises(ValueError):
                derive_row(raw)

    def test_membership_date_conflict_is_not_a_zero(self):
        r=derive_row(record(member=False),self.anchor['dt']+timedelta(days=365))
        self.assertEqual(r['temporal_status'],'membership_event_conflict')
        self.assertIsNone(r['target_30d'])

    def test_event_after_asserted_cutoff_is_not_accepted(self):
        r=derive_row(record(),self.anchor['dt']+timedelta(days=10))
        self.assertIsNone(r['target_30d'])
        self.assertEqual(r['target_30d_status'],'EVENT_AFTER_CUTOFF_CONFLICT')

    def test_documented_scenario_never_overwrites_primary(self):
        r=derive_row(record(event=None,member=False),documented_cutoff=self.anchor['dt']+timedelta(days=365))
        self.assertIsNone(r['target_90d'])
        self.assertEqual(r['documented_snapshot_target_90d'],0)

    def test_date_parsing_deterministic_timezone_and_precision(self):
        self.assertEqual(date_info('2025-01-01T14:00:00+02:00')['dt'],self.anchor['dt'])
        self.assertEqual(date_info('2025-01-01T12:00:00')['dt'],self.anchor['dt'])
        self.assertEqual(date_info('2025-01-01T12:00:00')['timezone'],'assumed_UTC')
        self.assertEqual(date_info('2025-01-01')['precision'],'date_only')
        for bad in ('01/02/2025','20250101','2025-W01-1','2025-02-30','2025-01-01T12:00:00.123456789Z'):
            self.assertIsNone(date_info(bad)['dt'])

    def test_date_only_anchor_conservative_followup(self):
        r=derive_row(record(event=None,member=False,anchor='2025-01-01'),date_info('2025-01-31T00:00:00Z')['dt'])
        self.assertIsNone(r['target_30d'])
        self.assertEqual(r['stored_timestamp_target_30d'],0)

    def test_cisa_and_reference_dates_do_not_change_primary_event(self):
        raw=record()
        raw.update(cisa_date_added='2025-02-20T12:00:00Z',url_change_dates=[
            {'earliest_added_date':'2025-01-02T12:00:00Z'}, {'earliest_added_date':'bad'}])
        r=derive_row(raw)
        self.assertEqual(r['delta_days_exact'],20)
        self.assertEqual(r['publication_to_cisa_days'],50)
        self.assertEqual(r['vckev_to_cisa_days'],30)
        self.assertEqual(r['publication_to_earliest_reference_days'],1)
        self.assertEqual(r['reference_history_unusable_dates'],1)

    def test_cvss_v2_does_not_fabricate_v3_privileges_or_scope(self):
        parsed,status=vector_components('AV:N/AC:M/Au:S/C:P/I:N/A:C',2.0)
        self.assertEqual(status,'valid_base_tokens_only')
        self.assertEqual(parsed['current_authentication_v2'],'S')
        self.assertEqual(parsed['current_attack_complexity'],'M')
        for key in ('current_privileges_required','current_scope','current_user_interaction'):
            self.assertIsNone(parsed[key])

    def test_cvss_parser_rejects_version_mismatch_or_duplicate_component(self):
        vector=record()['cvss_vector']
        self.assertEqual(vector_components(vector,3.0)[1],'version_prefix_mismatch')
        self.assertEqual(vector_components(vector+'/AV:L',3.1)[1],'malformed_or_duplicate_token')
        self.assertEqual(vector_components(vector.replace('/AC:L','/AC:Z'),3.1)[1],'invalid_or_missing_base_component')

    def test_feature_policy_has_no_outcome_linked_predictors(self):
        policy=feature_policy()
        self.assertEqual(policy['approved_predictor_features'],[])
        for field in policy['candidate_predictor_fields_requiring_asof_evidence']:
            self.assertNotIn(field,policy['forbidden_raw_fields'])
            self.assertFalse(field.startswith(('vckev_','target_','delta_','documented_')))
        self.assertIn('counts',policy['forbidden_derivations'])

    def test_recomputation_and_all_category_partition_invariants(self):
        raw=[record(),record(event=None,member=False,cve='CVE-2025-1235'),
             record(event='2024-12-31T00:00:00Z',cve='CVE-2025-1236')]
        rows=[derive_row(r) for r in raw]
        self.assertTrue(validate_derived(raw,rows)['passed'])
        self.assertEqual(sum(sum(r['delta_category']==cat for r in rows) for cat in CATEGORIES),len(rows))
        rows[0]['target_60d']=0
        with self.assertRaises(AssertionError):
            validate_derived(raw,rows)

    def test_duplicate_cve_rows_are_rejected(self):
        raw=[record(),record()]
        with self.assertRaises(ValueError):
            validate_derived(raw,[derive_row(r) for r in raw])

    def test_stats_and_export_preserve_unknowns(self):
        self.assertEqual(distribution([1,2,3])['median'],2)
        self.assertIsNone(distribution([])['median'])
        rows=[derive_row(record(event=None,member=False))]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'table'
            write_table(path,rows)
            data=json.loads(path.with_suffix('.json').read_text(),parse_constant=lambda x:self.fail(x))
            self.assertIsNone(data[0]['target_30d'])
            self.assertIn('NA',path.with_suffix('.csv').read_text())
            self.assertTrue(path.with_suffix('.txt').is_file())
        self.assertEqual(schema(rows)['csv_null'],'NA')

    def test_summary_counts_are_coherent(self):
        raw=[record(),record(event=None,member=False,cve='CVE-2025-1235')]
        rows=[derive_row(r) for r in raw]
        report=summarize(raw,rows,{},validate_derived(raw,rows))
        self.assertEqual(report['cves'],2)
        self.assertEqual(len(research_answers(report)),12)
        for row in label_counts(rows):
            self.assertEqual(row['positive']+row['negative']+row['unknown'],2)
        json.dumps(report,allow_nan=False)

    def test_plots_support_no_listed_events(self):
        from temporal_plots import make_temporal_figures
        raw=[record(event=None,member=False)]
        rows=[derive_row(r) for r in raw]
        report=summarize(raw,rows,{},validate_derived(raw,rows))
        with tempfile.TemporaryDirectory() as temp:
            figures=make_temporal_figures(Path(temp),rows,report)
            self.assertEqual(len(figures),9)
            json.dumps(figures,allow_nan=False)
            self.assertEqual(len(list(Path(temp).glob('*.png'))),9)


if __name__=='__main__':
    unittest.main()
