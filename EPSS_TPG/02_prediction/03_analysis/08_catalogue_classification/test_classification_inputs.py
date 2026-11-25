import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('cc02_cli', Path(__file__).with_name('03_prepare_classification_inputs.py'))
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
from epss.catalogue_inputs_v1 import core, runner
from epss.catalogue_classification_v1 import snapshot as frozen

C = 'CVE-2022-10000'


def record(cve=C):
    return {'cve': cve, 'description': 'A parser accepts malformed input.',
            'social_media_posts': [{'post': 'A patch fixes the parser.'}],
            'cvss_vector': 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H',
            'cvss_vector_version': '3.1', 'cvss_vector_score': 9.8,
            'all_sources_urls': ['https://example.org/advisory'],
            'github_only_urls': [], 'non_github_urls': ['https://example.org/advisory'],
            'in_vckev': True, 'vckev_date_added': '2026-11-17', 'epss_scores': [0.99]}


class SourceTests(unittest.TestCase):
    def test_catalogue_columns_never_predictors(self):
        row, audit = core.screen(record())
        self.assertEqual(set(row), {'documents', 'structured', 'mask_D', 'mask_M'})
        self.assertEqual(set(row['structured']), set(core.CVSS + core.URLS))
        self.assertIn('in_vckev', audit['excluded_fields'])
        self.assertNotIn('vckev', json.dumps(row))

    def test_disclosure_masks_not_cve_deletion(self):
        r = record()
        r['description'] = 'Listed in VulnCheck KEV.'
        r['social_media_posts'] = [{'post': 'Not in CISA KEV.'}]
        row, audit = core.screen(r)
        self.assertEqual(row['documents'], [])
        self.assertEqual((row['mask_D'], row['mask_M']), (0, 0))
        self.assertTrue(all(x['decision'] == 'quarantine' for x in audit['documents']))

    def test_encoded_url_and_epss_disclosures(self):
        self.assertEqual(core.disclosure('https://example.org/%4b%45%56', C, 'smp')['decision'], 'quarantine')
        self.assertEqual(core.disclosure('EPSS score: 0.99', C, 'smp')['decision'], 'quarantine')

    def test_valid_function_and_exploitation_wording(self):
        self.assertEqual(core.disclosure('def vulncheck(ip): return True', C, 'description')['decision'], 'accept')
        self.assertEqual(core.disclosure('An exploit exists.', C, 'description')['decision'], 'accept')

    def test_bare_and_duplicate_posts(self):
        r = record()
        r['social_media_posts'] = [{'post': C}, {'post': 'A patch exists.'}, {'post': 'A  patch exists.'}]
        row, audit = core.screen(r)
        self.assertEqual([x['decision'] for x in audit['documents']], ['accept', 'bare_id', 'accept', 'duplicate'])
        self.assertEqual(sum(d['source'] == 'M' for d in row['documents']), 1)

    def test_references_within_record_dedup_shared_preserved(self):
        urls = ['https://example.org/advisory'] * 2 + ['https://vulncheck.com/kev', 'ftp://bad/x']
        result, stats = core.reference_urls(urls, C)
        self.assertEqual(result, ['https://example.org/advisory'])
        self.assertEqual(stats, {'missing': False, 'duplicates_removed': 1, 'quarantined': 1, 'invalid': 1})
        self.assertEqual(core.reference_urls(urls, 'CVE-2022-10001')[0], result)
        self.assertIsNone(core.reference_urls(None, C)[0])
        self.assertEqual(core.reference_urls([], C)[0], [])

    def test_url_subsets_consistent(self):
        r = record()
        r['github_only_urls'] = ['https://github.com/vendor/patch']
        row, _ = core.screen(r)
        s = row['structured']
        self.assertEqual(set(s['all_sources_urls']), set(s['github_only_urls']) | set(s['non_github_urls']))
        self.assertFalse(set(s['github_only_urls']) & set(s['non_github_urls']))


class GroupTests(unittest.TestCase):
    def test_transitive_cross_dataset_role_parity(self):
        rows = {c: core.screen(record(c))[0] for c in ['CVE-2022-10000', 'CVE-2022-10001', 'CVE-2022-10002']}
        datasets = {'reddit': rows, 'previous': {C: rows[C]}}
        roles = core.freeze_roles(datasets, {})
        self.assertEqual(len(roles['components']), 1)
        self.assertEqual(len({v['role'] for v in roles['by_cve'].values()}), 1)
        self.assertEqual(sum(len(x) for x in roles['datasets']['reddit'].values()), 3)

    def test_order_and_outcome_invariant(self):
        rows = {}
        for i in range(80):
            c = f'CVE-2022-{10000+i}'
            r = record(c)
            r['description'] = 'Distinct wording ' + str(i)
            r['social_media_posts'] = []
            rows[c] = core.screen(r)[0]
        first = core.freeze_roles({'reddit': rows}, {})
        reversed_rows = dict(reversed(list(rows.items())))
        for row in reversed_rows.values():
            row['outcome_for_test_only'] = 1
        self.assertEqual(first, core.freeze_roles({'reddit': reversed_rows}, {}))
        self.assertEqual({r['role'] for r in first['by_cve'].values()}, {'fitting', 'selection', 'assessment'})

    def test_preserve_prior_groups_even_without_text(self):
        rows = {c: {'documents': []} for c in [C, 'CVE-2022-10001']}
        roles = core.freeze_roles({'reddit': rows}, {'prior1': list(rows)})
        self.assertEqual(len(roles['components']), 1)


class EpssTests(unittest.TestCase):
    def parse(self, body, ids=None, truncate=False):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'scores.gz'
            payload = gzip.compress(body.encode())
            path.write_bytes(payload[:-5] if truncate else payload)
            return core.parse_epss(path, '2026-11-17', {C} if ids is None else ids, 1)

    def body(self, row=None, day='2026-11-17'):
        return f'#model_version:v2026.06.15,score_date:{day}\ncve,epss,percentile\n' + (row or C + ',0.0,0.1\n')

    def test_zero_is_observed_missing_is_null(self):
        scores, inventory = self.parse(self.body())
        joined = core.join_epss([C, 'CVE-2022-10001'], scores)
        self.assertEqual(joined[C]['observed'], 1)
        self.assertEqual(joined[C]['score'], 0.)
        self.assertIsNone(joined['CVE-2022-10001']['score'])
        self.assertEqual(inventory['rows'], 1)

    def test_wrong_date_rejected(self):
        with self.assertRaisesRegex(ValueError, 'date differs'):
            self.parse(self.body(day='2026-11-16'))

    def test_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.parse(self.body((C + ',0.1,0.2\n') * 2))

    def test_invalid_scores(self):
        for value in ('NaN', '-0.1', '1.1', 'inf'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.parse(self.body(C + ',' + value + ',0.2\n'))

    def test_truncation_rejected(self):
        with self.assertRaises(EOFError):
            self.parse(self.body(), truncate=True)

    def test_no_unselected_scores_retained(self):
        scores, inventory = self.parse(self.body(C + ',0.1,0.2\nCVE-2022-10001,0.2,0.3\n'))
        self.assertEqual(set(scores), {C})
        self.assertEqual(inventory['rows'], 2)


class AdapterTests(unittest.TestCase):
    def test_synthetic_end_to_end_replay_and_fitting_only_preprocessing(self):
        cfg = frozen.read(runner.CONFIG)
        cfg['epss_minimum_rows'] = 1
        ids = [f'CVE-2022-{10000+i}' for i in range(5)]
        plan = {'training_ids': {name: ids for name in ('reddit', 'previous')}}
        roles = {'datasets': {name: {'fitting': ids[:3], 'selection': ids[3:4], 'assessment': ids[4:]} for name in plan['training_ids']}}
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            freeze, epss = base / 'freeze', base / 'epss'
            freeze.mkdir()
            epss.mkdir()
            for name in plan['training_ids']:
                rows = {c: core.screen(record(c))[0] for c in ids}
                rows[ids[-1]]['structured']['cvss_vector_score'] = 0.0
                frozen.write(freeze / (name + '_screened.json'), rows)
                frozen.write(freeze / (name + '_labels.json'), {c: {'label': int(i % 2), 'status': 'LISTED' if i % 2 else 'NOT_LISTED'} for i, c in enumerate(ids)})
            frozen.write(freeze / 'identity.json', {'source_pins': {}})
            frozen.write(freeze / 'protocol.json', cfg)
            frozen.write(freeze / 'roles.json', roles)
            frozen.write(freeze / 'findings.json', {'role_support_sufficient': True})
            frozen.lock(freeze)
            path = epss / 'epss_scores-2026-11-17.csv.gz'
            body = '#model_version:v2026.06.15,score_date:2026-11-17\ncve,epss,percentile\n'
            body += '\n'.join(c + ',' + str(.1 * i) + ',0.1' for i, c in enumerate(ids[:3])) + '\n'
            path.write_bytes(gzip.compress(body.encode()))
            _, inventory = core.parse_epss(path, cfg['epss_date'], set(ids), 1)
            frozen.write(epss / 'receipt.json', {'requested_url': cfg['epss_url'], 'http_status': 200,
                         'sha256': frozen.sha(path), 'bytes': path.stat().st_size, 'inventory': inventory})
            frozen.lock(epss)
            runner.prepare(cfg, plan, base, {}, False, False)
            runner.prepare(cfg, plan, base, {}, True, False)
            p = frozen.read(base / 'prepared/reddit_predictors.json')
            state = frozen.read(base / 'prepared/reddit_preprocessing.json')
            self.assertEqual(state['fit_ids'], ids[:3])
            self.assertEqual(state['numeric']['cvss_score']['median'], 9.8)
            self.assertEqual(state['epss']['median'], .1)
            self.assertEqual(p[ids[-1]]['epss_features'], [.1, 0])
            self.assertIsNone(p[ids[-1]]['epss']['score'])
            self.assertNotIn('label', json.dumps(p))
            with patch.object(runner.requests, 'get', side_effect=AssertionError('No network')):
                runner.collect(cfg, plan, base, False, True, False)


if __name__ == '__main__':
    unittest.main()
