"""CD02 regression checks on synthetic CVEs, not model performance."""
import copy
from datetime import date, timedelta
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss import forecast_identity_development as prep


def doc(source, text):
    return {'document_id': source + ':' + prep.sources.digest(text), 'source_text_type': source, 'text': text}


def source_row(cve, split, documents):
    return {'cve': cve, 'split': split, 'documents': documents,
            'source_mask': [int(any(d['source_text_type'] == s for d in documents)) for s in prep.sources.SOURCES]}


def record(cve, published):
    start = date.fromisoformat(published)
    return {'cve': cve, 'published_timestamp': published + 'T00:00:00Z', 'published_date': published,
            'description': 'Shared generic title', 'orig_description': 'Shared generic title',
            'summ_github_only_urls_fast.gpt-oss:120b': 'Shared optional summary',
            'summ_non_github_urls_fast.gpt-oss:120b': 'Distinct summary for ' + cve,
            'social_media_posts': [{'platform': 'Example', 'post': 'Post for ' + cve,
                                    'date': (start + timedelta(days=1)).isoformat()}],
            'epss_scores': {(start + timedelta(days=i)).isoformat(): .1 for i in range(91)}}


class IdentityTests(unittest.TestCase):
    def test_description_census_is_not_identity_grouping(self):
        records = [{'cve': 'A', 'description': 'Same text'}, {'cve': 'B', 'description': 'Same text'},
                   {'cve': 'C', 'description': ' SAME  text '}, {'cve': 'D', 'description': None}]
        c = prep.description_census(records)['description']
        self.assertEqual(c['exact']['records_in_repeated_string_groups'], 2)
        self.assertEqual(c['lower_whitespace']['records_in_repeated_string_groups'], 3)
        self.assertEqual(c['exact']['nonempty_records'], 3)

    def test_keep_description_mask_optional_not_cve(self):
        training = [source_row('A', 'train', [doc('description', 'Same TEXT')])]
        validation = [source_row('B', 'validation', [doc('description', 'Same TEXT'), doc('github_summary', 'same  text')])]
        kept, audit = prep.route_validation(validation, prep.registry(training))
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]['source_mask'], [1, 0, 0, 0])
        self.assertEqual([d['reason'] for d in audit], ['description_overlap_retained', 'optional_text_seen_in_training'])
        self.assertEqual(len(validation[0]['documents']), 2)

    def test_missing_all_sources_preserves_cve(self):
        rows, audit = prep.route_validation([source_row('B', 'validation', [])], {})
        self.assertEqual(rows[0]['source_mask'], [0, 0, 0, 0])
        self.assertEqual(audit, [])

    def test_no_quality_filter_on_imperfect_unique_summary(self):
        text = 'I need the excerpt to analyze.'
        rows, _ = prep.route_validation([source_row('B', 'validation', [doc('non_github_summary', text)])], {})
        self.assertEqual(rows[0]['documents'][0]['text'], text)

    def test_validation_does_not_modify_training_registry(self):
        reg = {'abc': ['description']}
        before = copy.deepcopy(reg)
        prep.route_validation([source_row('B', 'validation', [doc('smp', 'New text')])], reg)
        self.assertEqual(reg, before)
        with self.assertRaises(ValueError):
            prep.route_validation([source_row('T', 'test', [])], reg)

    def test_cross_source_overlap_and_original_wording(self):
        reg = prep.registry([source_row('A', 'train', [doc('smp', 'Repeated TEXT')])])
        rows, _ = prep.route_validation([source_row('B', 'validation', [doc('github_summary', ' repeated\ntext '),
                  doc('non_github_summary', 'Keep   exact\nwording')])], reg)
        self.assertEqual(rows[0]['documents'][0]['text'], 'Keep   exact\nwording')

    def fixture(self, temp):
        records = [record('CVE-2024-10000', '2024-01-01'), record('CVE-2024-10001', '2024-02-01'),
                   record('CVE-2025-10000', '2025-02-15'), record('CVE-2025-10001', '2025-02-16'),
                   record('CVE-2025-20000', '2025-06-01')]
        # Test score contents are deliberately unusable: development export must not derive them.
        records[-1]['epss_scores'] = {'SECRET_TEST_DATE': 'SECRET_TEST_SCORE'}
        records[-1]['description'] = 'SECRET_TEST_TEXT'
        path = Path(temp) / 'raw.json'
        prep.write(path, records)
        availability = {}
        for r in records:
            day = date.fromisoformat(r['published_date'])
            availability[r['cve']] = {'input_end_date': (day + timedelta(days=60)).isoformat(),
                'target_date': (day + timedelta(days=90)).isoformat(), 'matched_eligible': True,
                'day90_observed': True, 'recent_observed_days': 31, 'long_observed_days': 61}
        p = {'split_policy': {'validation_start': '2025-04-01', 'test_start': '2025-07-01',
                             'test_end': '2026-01-01', 'observation_end': '2026-09-29'},
             'expected_source_cves': 5, 'smoke_counts': {'train': 1, 'validation': 1}, 'selection_salt': 'test'}
        manifest, full = prep.assign(availability, p['split_policy'])
        return (p, {}, path, availability, manifest, full)

    def build_fixture(self, ctx, scale):
        real_read = prep.read
        def read(path):
            if str(path).endswith('development/split_manifest.json'):
                return ctx[4]
            return real_read(path)
        with patch.object(prep, 'read', side_effect=read):
            return prep.build(ctx, scale)

    def test_full_export_contract_and_test_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self.fixture(tmp)
            a = self.build_fixture(ctx, 'full')
            self.assertEqual(a['analysis.json']['exported_counts'], {'train': 2, 'validation': 2})
            self.assertNotIn('CVE-2025-20000', a['inputs.json'])
            dumped = json.dumps(a)
            self.assertNotIn('SECRET_TEST_SCORE', dumped)
            self.assertNotIn('SECRET_TEST_TEXT', dumped)
            self.assertNotIn('SECRET_TEST_DATE', dumped)
            self.assertNotIn('y', a['inputs.json']['CVE-2024-10000'])
            self.assertEqual(a['analysis.json']['full_validation_overlap']['masked_optional_documents'], 2)

    def test_smoke_and_full_share_registry_and_source_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self.fixture(tmp)
            small, full = self.build_fixture(ctx, 'smoke'), self.build_fixture(ctx, 'full')
            self.assertEqual(small['training_text_registry.json'], full['training_text_registry.json'])
            by_cve = {r['cve']: r for r in full['source_inputs.json']['rows']}
            self.assertTrue(all(r == by_cve[r['cve']] for r in small['source_inputs.json']['rows']))

    def test_new_identity_groups_do_not_merge_shared_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self.fixture(tmp)
            self.assertEqual(len(set(r['group_id'] for r in ctx[4].values())), 5)
            self.assertEqual({k: len(v) for k, v in ctx[5].items()}, {'train': 2, 'validation': 2, 'test': 1})

    def test_tampered_future_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.build_fixture(self.fixture(tmp), 'full')
            c = next(iter(a['inputs.json']))
            a['inputs.json'][c]['histories']['recent_30_60']['day_offsets'][-1] = 90
            with self.assertRaises(ValueError):
                prep.validate(a)

    def test_temporal_maturity_unchanged(self):
        p = {'validation_start': '2025-04-01', 'test_start': '2025-07-01',
             'test_end': '2026-01-01', 'observation_end': '2026-09-29'}
        rows = {'A': {'input_end_date': '2025-03-02', 'target_date': '2025-04-01', 'matched_eligible': True}}
        m, ids = prep.assign(rows, p)
        self.assertEqual(m['A']['exclusion_reason'], 'label_crosses_train_boundary')
        self.assertEqual(ids['train'], [])


if __name__ == '__main__':
    unittest.main()
