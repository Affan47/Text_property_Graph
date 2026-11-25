"""Synthetic MS04 preparation checks; no real model training or test inspection."""
from copy import deepcopy
from contextlib import redirect_stdout, redirect_stderr
from datetime import date, timedelta
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_day60_multisource as prep


def record(cve, published):
    pub = date.fromisoformat(published)
    return {'cve': cve, 'published_date': published, 'published_timestamp': published+'T00:00:00Z',
            'description': 'Shared vulnerability title',
            'summ_github_only_urls_fast.gpt-oss:120b': 'Common optional summary',
            'summ_non_github_urls_fast.gpt-oss:120b': 'Unique summary for '+cve,
            'social_media_posts': [{'platform': 'Example', 'post': 'Unique post for '+cve,
                                   'date': (pub+timedelta(days=2)).isoformat()}],
            'epss_scores': {(pub+timedelta(days=j)).isoformat(): .1 for j in range(91)}}


def change_score(raw, day, value):
    raw['epss_scores'][(date.fromisoformat(raw['published_date'])+timedelta(days=day)).isoformat()] = value


def fixture(directory):
    records = [record('CVE-2024-10000', '2024-01-01'), record('CVE-2024-10001', '2024-02-01'),
               record('CVE-2025-10000', '2025-02-15'), record('CVE-2025-10001', '2025-02-16'),
               record('CVE-2025-20000', '2025-06-01'), record('CVE-2025-20001', '2025-04-15')]
    old_policy = {'validation_start': '2025-04-01', 'test_start': '2025-07-01',
                  'test_end': '2026-01-01', 'observation_end': '2026-09-29'}
    x = {r['cve']: {'input_end_date': (date.fromisoformat(r['published_date'])+timedelta(days=60)).isoformat()} for r in records}
    y = {r['cve']: {'target_date': (date.fromisoformat(r['published_date'])+timedelta(days=90)).isoformat()} for r in records}
    old, _ = prep.contract.split_manifest(x, y, {c: 'cve_identity:'+c for c in x}, old_policy)
    p = deepcopy(prep.read(prep.CONFIG))
    p.update(expected_source_cves=len(records), smoke_counts={'train': 1, 'validation': 1})
    m = prep.assignment(old, p['split_policy'])
    for r in records:
        if m[r['cve']]['split'] not in ('train', 'validation'):
            r['epss_scores'] = {'SECRET_TEST_OR_EXCLUDED_SCORE': 'not a score'}
            r['description'] = {'SECRET_TEST_OR_EXCLUDED_TEXT': 'not text'}
            r['social_media_posts'] = 'must not adapt'
    source = Path(directory)/'raw.json'
    prep.write(source, records)
    return {'protocol': p, 'pins': {}, 'source': source, 'manifest': m, 'old_manifest': old,
            'old_ids': {s: sorted(c for c, r in old.items() if r['split'] == s) for s in ('train', 'validation')},
            'same_calendar_assignment_counts': {}}, records


class Day60Tests(unittest.TestCase):
    def setUp(self):
        self.out = redirect_stdout(StringIO())
        self.err = redirect_stderr(StringIO())
        self.out.__enter__()
        self.err.__enter__()
        self.addCleanup(self.out.__exit__, None, None, None)
        self.addCleanup(self.err.__exit__, None, None, None)

    def test_boundary_shift_preserves_roles_and_observation_end(self):
        with tempfile.TemporaryDirectory() as d:
            ctx, _ = fixture(d)
            self.assertEqual({c: r['split'] for c, r in ctx['old_manifest'].items()},
                             {c: r['split'] for c, r in ctx['manifest'].items()})
            self.assertEqual(ctx['protocol']['split_policy']['observation_end'], '2026-09-29')
            self.assertEqual(ctx['protocol']['split_policy']['test_start'], '2025-06-01')
            self.assertEqual(ctx['manifest']['CVE-2025-20000']['split'], 'test')
            self.assertEqual(ctx['manifest']['CVE-2025-20001']['split'], 'excluded')

    def test_wrong_shift_cannot_move_prior_roles(self):
        with tempfile.TemporaryDirectory() as d:
            ctx, _ = fixture(d)
            policy = dict(ctx['protocol']['split_policy'], validation_start='2025-05-01')
            with self.assertRaisesRegex(ValueError, 'changed a reserved or development role'):
                prep.assignment(ctx['old_manifest'], policy)

    def test_target_boundary_is_excluded_not_training(self):
        p = prep.read(prep.CONFIG)['split_policy']
        self.assertEqual(prep.contract.candidate_split('2025-01-31', '2025-03-02', p)[0], 'excluded')
        self.assertEqual(prep.contract.candidate_split('2025-05-02', '2025-06-01', p)[0], 'excluded')

    def test_future_scores_do_not_enter_inputs(self):
        raw = record('CVE-2024-12345', '2024-01-01')
        x, y, _ = prep.derive(raw)
        for j in range(31, 91):
            change_score(raw, j, .9)
        new_x, new_y, _ = prep.derive(raw)
        self.assertEqual(x, new_x)
        self.assertNotEqual(y, new_y)
        channels = prep.sequence_channels(x)
        self.assertEqual((len(channels), len(channels[0])), (31, 5))
        self.assertEqual(channels[-1][-1], .5)

    def test_day90_is_not_required_and_does_not_affect_day60_labels(self):
        raw = record('CVE-2024-12345', '2024-01-01')
        original = prep.derive(raw)
        for j in range(61, 91):
            change_score(raw, j, None)
        self.assertEqual(prep.derive(raw), original)

    def test_missing_target_and_empty_history_are_explicit(self):
        raw = record('CVE-2024-12345', '2024-01-01')
        change_score(raw, 60, None)
        _, y, audit = prep.derive(raw)
        self.assertIsNone(y['y']['endpoint'])
        self.assertFalse(audit['eligible'])
        for j in range(31):
            change_score(raw, j, None)
        x, _, audit = prep.derive(raw)
        self.assertEqual(len(audit['exclusion_reasons']), 2)
        with self.assertRaises(ValueError):
            prep.sequence_channels(x)

    def test_observed_zero_and_missing_zero_have_different_masks(self):
        raw = record('CVE-2024-12345', '2024-01-01')
        for j in range(31):
            change_score(raw, j, None)
        change_score(raw, 2, 0.)
        x, _, audit = prep.derive(raw)
        a = prep.sequence_channels(x)
        self.assertTrue(audit['eligible'])
        self.assertEqual(a[0], [0., 0, 0., 0, 0.])
        self.assertEqual(a[2][:4], [0., 1, 0., 1])
        self.assertEqual(a[30][2:4], [28/60, 1])

    def test_forged_future_or_target_input_fails(self):
        x, _, _ = prep.derive(record('CVE-2024-12345', '2024-01-01'))
        bad = deepcopy(x)
        bad['y'] = {'endpoint': .1}
        with self.assertRaises(ValueError):
            prep.sequence_channels(bad)
        x['histories'][prep.VIEW]['day_offsets'][-1] = 60
        with self.assertRaises(ValueError):
            prep.sequence_channels(x)

    def test_posts_screened_at_day30_not_day60(self):
        with tempfile.TemporaryDirectory() as d:
            ctx, records = fixture(d)
            raw = records[0]
            pub = date.fromisoformat(raw['published_date'])
            raw['social_media_posts'] = [{'platform': 'Example', 'post': 'day'+str(j),
                 'date': (pub+timedelta(days=j)).isoformat()} for j in (29, 30, 31, 45)]
            prep.write(ctx['source'], records)
            a = prep.build(ctx, 'full')
            row = next(r for r in a['source_inputs.json']['rows'] if r['cve'] == raw['cve'])
            self.assertEqual([x['text'] for x in row['documents'] if x['source_text_type'] == 'smp'], ['day29'])
            audit = next(r for r in a['source_date_checks.json'] if r['cve'] == raw['cve'])
            bad = next(x for x in audit['documents'] if x['source_text_type'] == 'smp' and x['stored_date_evidence'][0]['relation'] == 'before_anchor')
            bad['stored_date_evidence'][0]['value'] = (pub+timedelta(days=31)).isoformat()
            with self.assertRaisesRegex(ValueError, 'Late, undated or conflicting SMP'):
                prep.validate(a)

    def test_conflicting_or_undated_posts_stay_out(self):
        raw = record('CVE-2024-12345', '2024-01-01')
        raw['social_media_posts'] = [{'post': 'same', 'platform': 'Example', 'date': '2024-01-02'},
                                    {'post': 'same', 'platform': 'Example', 'date': '2024-01-03'},
                                    {'post': 'undated', 'platform': 'Example', 'date': None}]
        fields = prep.read(prep.sources.CONFIG)['source_fields']
        row = prep.sources.adapt_record(raw, '2024-01-31', fields)
        self.assertEqual(prep.sources.model_input(row, list(prep.sources.SOURCES))['source_mask'][-1], 0)

    def test_full_export_never_adapts_test_or_excluded_contents(self):
        with tempfile.TemporaryDirectory() as d:
            ctx, _ = fixture(d)
            a = prep.build(ctx, 'full')
            self.assertNotIn('SECRET_TEST_OR_EXCLUDED', json.dumps(a))
            self.assertEqual(len(a['inputs.json']), 4)
            self.assertEqual(a['analysis.json']['test_day60_eligibility'], 'not_inspected')
            self.assertNotIn('CVE-2025-20000', a['development_eligibility.json'])
            self.assertEqual(set(a['labels.json']), set(a['inputs.json']))

    def test_overlap_keeps_cve_and_description_masks_optional_only(self):
        with tempfile.TemporaryDirectory() as d:
            ctx, _ = fixture(d)
            a = prep.build(ctx, 'full')
            rows = [r for r in a['source_inputs.json']['rows'] if r['split'] == 'validation']
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r['source_mask'] == [1, 0, 1, 1] for r in rows))
            self.assertEqual(a['analysis.json']['masked_optional_validation_documents'], 2)

    def test_missing_text_preserves_eligible_cve(self):
        with tempfile.TemporaryDirectory() as d:
            ctx, records = fixture(d)
            raw = records[0]
            for field in prep.read(prep.sources.CONFIG)['source_fields'].values():
                raw[field] = [] if field == 'social_media_posts' else None
            prep.write(ctx['source'], records)
            a = prep.build(ctx, 'full')
            row = next(r for r in a['source_inputs.json']['rows'] if r['cve'] == raw['cve'])
            self.assertEqual(row['source_mask'], [0, 0, 0, 0])
            self.assertIn(raw['cve'], a['labels.json'])

    def test_smoke_full_use_same_registry_and_records(self):
        with tempfile.TemporaryDirectory() as d:
            ctx, _ = fixture(d)
            small, full = prep.build(ctx, 'smoke'), prep.build(ctx, 'full')
            self.assertEqual(small['training_text_registry.json'], full['training_text_registry.json'])
            for name in ('inputs.json', 'labels.json'):
                self.assertTrue(all(full[name][c] == row for c, row in small[name].items()))

    def test_cli_preflight_export_reproduction_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as d:
            ctx, _ = fixture(d)
            output = Path(d)/'out'
            with patch.object(prep, 'context', return_value=ctx), patch.object(prep, 'OUTPUT', output):
                def cli(*args):
                    with patch.object(sys, 'argv', ['prepare', *args]):
                        prep.main()
                cli('--check-only')
                self.assertFalse(output.exists())
                cli('--scale', 'smoke')
                cli('--scale', 'smoke', '--verify-only')
                cli('--scale', 'full')
                cli('--scale', 'full', '--verify-only')
                with self.assertRaisesRegex(ValueError, 'Preparation exists'):
                    cli('--scale', 'full')
                labels = prep.read(output/'full/labels.json')
                labels[next(iter(labels))]['y']['endpoint'] = .9
                prep.write(output/'full/labels.json', labels)
                with self.assertRaisesRegex(ValueError, 'Changed/unsafe prerequisite'):
                    cli('--scale', 'full', '--verify-only')


if __name__ == '__main__':
    unittest.main()
