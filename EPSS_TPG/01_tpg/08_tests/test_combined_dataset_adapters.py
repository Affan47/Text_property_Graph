from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
for p in ('01_tpg/10_extensions', '01_tpg/01_core', '02_prediction/01_package'):
    sys.path.insert(0, str(ROOT / p))
from tpg_combined_datasets_v1 import adapters as a
from tpg_combined_datasets_v1.reporting import exclusion_accounting
from tpg_combined_v1 import common as c, sources


class DatasetTests(unittest.TestCase):
    def test_same_join_and_source_contract(self):
        docs = [a.canonical_document({'source_text_type': name, 'document_id': name, 'text': text})
                for name, text in [('description', 'Flaw.'), ('github_summary', 'Patch.'), ('smp', 'Post.')]]
        text, spans = sources.combine(docs, c.VARIANTS['DMU'], 'prose')
        self.assertEqual(text, 'Flaw.\n\nPost.\n\nPatch.')
        self.assertEqual([p['source'] for p in spans], ['D', 'M', 'G'])
        self.assertNotIn('label', docs[0])

    def test_bare_id_does_not_turn_into_prose(self):
        doc = a.canonical_document({'source_text_type': 'smp', 'document_id': 'm', 'text': 'CVE-2022-46908'})
        self.assertEqual(sources.combine([doc], 'M', 'prose'), ('', []))
        self.assertEqual(sources.combine([doc], 'M', 'all')[0], doc['text'])

    def test_roles_do_not_include_test(self):
        rows = [{'cve': 'a', 'split': 'train'}, {'cve': 'b', 'split': 'validation'}]
        ids = {'train': ['a'], 'validation': ['b']}
        a.validate_roles(rows, ids, ['c'])
        with self.assertRaises(ValueError):
            a.validate_roles(rows, ids, ['b'])
        with self.assertRaises(ValueError):
            a.validate_roles(rows + rows[:1], ids, [])

    def test_label_missing_or_wrong_role_rejected(self):
        rows = [{'cve': 'a', 'split': 'train'}]
        good = {'cve': 'a', 'split': 'train', 'value': .2, 'target_day': 60, 'observed': True}
        a.validate_labels([good], rows)
        for patch in ({'value': None}, {'observed': False}, {'split': 'test'}, {'target_day': 90}):
            with self.assertRaises(ValueError):
                a.validate_labels([{**good, **patch}], rows)

    def test_structured_train_only_and_outcome_guard(self):
        fields = {key: None for key in a.structured.FIELDS}
        row = {'cve': 'a', 'split': 'train', 'structured': fields}
        state = a.structured.fit([row])
        self.assertEqual(state['fit_ids'], ['a'])
        with self.assertRaises(ValueError):
            a.structured.fit([{**row, 'split': 'validation'}])
        with self.assertRaises(ValueError):
            a.structured.parse({**fields, 'in_vckev': True})

    def test_dataset_roots_are_separate(self):
        self.assertNotEqual(a.CACHES['reddit'], a.CACHES['previous'])
        self.assertEqual(a.CACHES['reddit'], c.DEFAULT)

    def test_deterministic_publish_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            values = {'inputs.json.gz': {'x': 1}, 'REPORT.txt': 'Only a report.\n'}
            a.publish(path, values, False)
            before = c.sha(path / 'inputs.json.gz')
            a.publish(path, values, True)
            self.assertEqual(before, c.sha(path / 'inputs.json.gz'))
            with self.assertRaises(ValueError):
                a.publish(path, {**values, 'REPORT.txt': 'Changed'}, True)

    def test_reddit_exclusions_are_disjoint_and_exhaustive(self):
        flow = {'post_launch_exclusion_first_reason': {'prior_test_connected': 4245,
                'cross_publication_intervals': 626, 'prior_exposure_barred_from_test': 277, 'prior_role_conflict': 274},
                'task_exclusions': {'RD02': {'train:missing_day60': 69, 'validation:missing_day60': 10,
                  'test:missing_day60': 1, 'train:target_after_role_end': 317, 'validation:target_after_role_end': 132}}}
        counts = {'train': 24213, 'validation': 972, 'test': 3977}
        result = exclusion_accounting({'all_cves': 44985, 'post_launch': 35113}, flow, counts)
        self.assertEqual(sum(r['count'] for r in result), 15823)
        with self.assertRaises(ValueError):
            exclusion_accounting({'all_cves': 44985, 'post_launch': 35113}, flow, {**counts, 'train': 24214})

    def test_additional_disclosure_screen(self):
        self.assertTrue(a.disclosure('EPSS score is 0.8')['quarantine'])
        self.assertFalse(a.disclosure('Apache allows remote code execution.')['quarantine'])


if __name__ == '__main__':
    unittest.main()
