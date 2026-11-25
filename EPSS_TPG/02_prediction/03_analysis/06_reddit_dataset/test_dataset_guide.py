"""Focused report-format and schema guards; no source files are changed."""
import tempfile
from pathlib import Path
import unittest

import build_dataset_guide as guide


class GuideTests(unittest.TestCase):
    def test_field_dictionary_has_34_distinct_definitions(self):
        keys=[k for group in guide.GROUPS.values() for k in group]
        self.assertEqual(len(keys),34)
        self.assertEqual(len(set(keys)),34)
        self.assertEqual(set(guide.NESTED['social_media_posts']),{'post','platform','date'})

    def test_examples_do_not_turn_missing_values_into_zero(self):
        self.assertEqual(guide.render(None),'null')
        self.assertEqual(guide.render(False),'false')
        self.assertEqual(guide.render([]),'[]')
        self.assertEqual(guide.render({'2021-01-01':None}),'{'+'"2021-01-01": null}')

    def test_table_is_breakable_and_retains_plain_text(self):
        doc=guide.Document()
        doc.table(['Key','Value'],[[('code','a_b'),'x & y']],'Explanation',[.4,.6])
        latex='\n'.join(doc.body)
        self.assertIn(r'\allowbreak{}',latex)
        self.assertIn(r'x \& y',latex)
        self.assertIn(r'\toprule Key',latex)
        self.assertIn('a_b | x & y',doc.plain)

    def test_lock_detects_modified_report(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            (p/'report.txt').write_text('original')
            (p/'artifact_lock.json').write_text(guide.dumps({'files':{'report.txt':guide.sha(p/'report.txt')}}))
            guide.verify_lock(p)
            (p/'report.txt').write_text('changed')
            with self.assertRaises(ValueError):
                guide.verify_lock(p)


if __name__=='__main__':
    unittest.main()
