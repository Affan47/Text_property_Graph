import json
from pathlib import Path
import tempfile
import unittest

from compare_summary_samples import strict_load, indexed, excerpt, parse_log, transitions, OUT, digest, SOURCE
from check_revised_summary_generator import run_checks


class Rules(unittest.TestCase):
    def test_duplicate_cves_rejected(self):
        with self.assertRaises(ValueError):
            indexed([{"cve": "x"}, {"cve": "x"}])

    def test_strict_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "test.json"
            for text in ('{"score":NaN}', '{"score":1,"score":2}'):
                p.write_text(text)
                with self.assertRaises(ValueError):
                    strict_load(p)

    def test_prompt_id_is_not_source_evidence(self):
        self.assertEqual(excerpt("CVE-2025-0001 '''\nNotion\n''' analyze this CVE"), "Notion")
        self.assertEqual(excerpt(""), "")
        with self.assertRaises(ValueError):
            excerpt("unknown template")

    def test_log_url_prefixes(self):
        e = parse_log("C -> https://example.org/123: extracted 6 chars (html).", "C", ["https://example.org/1", "https://example.org/123"])
        self.assertEqual(e[0]["url"], "https://example.org/123")
        self.assertEqual(e[0]["extracted_chars"], 6)

    def test_http_status_not_success(self):
        e = parse_log("C -> Exception 404 Client Error occurred for https://example.org/.", "C", ["https://example.org/"])
        self.assertEqual(e[0]["status"], "http_404")

    def test_paired_missingness(self):
        old = {"a": {"nonempty": False}, "b": {"nonempty": True, "summary": "a", "flags": {}}}
        new = {"a": {"nonempty": True}, "b": {"nonempty": False}}
        r = transitions(old, new)
        self.assertEqual(r["availability"], {"empty_to_text": 1, "text_to_empty": 1})
        self.assertEqual(r["both_have_text"], 0)

    def test_offline_generator_checks(self):
        report = run_checks()
        self.assertEqual(len(report["findings"]), 12)
        self.assertEqual(report["findings"][0]["severity"], "critical")


class SavedArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = strict_load(OUT / "sample_comparison.json")
        cls.cells = strict_load(OUT / "summary_cells.json")

    def test_coverage_and_cohort(self):
        self.assertEqual(len(self.cells), 700)
        self.assertEqual(len({(c['variant'], c['cve'], c['group']) for c in self.cells}), 700)
        self.assertEqual(len({c['cve'] for c in self.cells}), 50)
        self.assertFalse(self.report['integrity_errors'])

    def test_sources_unchanged(self):
        for rel, sha in self.report['source_hashes'].items():
            self.assertEqual(digest(SOURCE / rel), sha, rel)

    def test_counts_reconcile(self):
        for variant, groups in self.report['statistics'].items():
            for group, s in groups.items():
                cc = [c for c in self.cells if c['variant'] == variant and c['group'] == group]
                self.assertEqual(s['nonempty'], sum(c['nonempty'] for c in cc))
                self.assertEqual(s['nonempty'] + s['empty'], 50)

    def test_all_logged_urls_matched(self):
        for value in self.report['log_consistency'].values():
            self.assertFalse(value)

    def test_missing_old_prompts_are_unknown_not_zero(self):
        for s in self.report['statistics']['old_gpt'].values():
            self.assertIsNone(s['foreign_cves_absent_from_saved_source'])

    def test_only_excerpt_used_for_grounding(self):
        for c in self.cells:
            if c['cve'] == 'CVE-2025-24513' and c['variant'] != 'old_gpt' and c['group'] == 'non_github':
                self.assertEqual(c['source_excerpt'], 'NetApp Product Security')
                self.assertFalse(c['target_cve_in_source_excerpt'])


if __name__ == '__main__':
    unittest.main()
