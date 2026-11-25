"""Rules and offline characterization checks, with no web or LLM calls."""
import unittest

from analyze_summary_quality import SOURCE, SCRIPT, cve_ids, duplicate_groups, flag_text, normalized, similar_pairs, source_window
from summary_scraper_checks import run_checks


class QualityRuleTests(unittest.TestCase):
    def test_unicode_cve_id(self):
        self.assertEqual(cve_ids("CVE\u20112024\u201112345"), {"cve-2024-12345"})

    def test_exact_and_normalized_are_different(self):
        cells = [{"cve": "CVE-2024-1234", "group": "github", "summary": s, "nonempty": True}
                 for s in ("No excerpt was provided.", "NO  excerpt was provided.")]
        self.assertEqual(duplicate_groups(cells, "exact"), [])
        self.assertEqual(duplicate_groups(cells, "normalized")[0]["summary_cells"], 2)

    def test_empty_values_do_not_form_duplicate_groups(self):
        cells = [{"cve": "CVE-2024-1234", "group": "github", "summary": s, "nonempty": False} for s in (None, "")]
        self.assertEqual(duplicate_groups(cells, "normalized"), [])

    def test_missing_input_response(self):
        self.assertIn("requests_missing_input", flag_text("I need the excerpt to analyze.", "CVE-2024-1234", ""))

    def test_specific_excerpt_and_relevant_content(self):
        for text in ("I need the specific excerpt.", "Please provide the relevant content."):
            self.assertIn("requests_missing_input", flag_text(text, "CVE-2024-1234", ""))

    def test_anti_bot_is_review_signal(self):
        self.assertIn("anti_bot_or_scraping_topic", flag_text("The page implements a proof-of-work browser check.", "CVE-2024-1234", ""))

    def test_legitimate_no_auth_does_not_imply_no_content(self):
        self.assertNotIn("says_insufficient_vulnerability_content", flag_text("No authentication is required to exploit this vulnerability.", "CVE-2024-1234", ""))

    def test_related_cve_in_context_not_flagged(self):
        self.assertNotIn("foreign_cve_not_in_description_or_urls", flag_text("A regression of CVE-2023-1234.", "CVE-2024-1234", "Related to CVE-2023-1234"))

    def test_foreign_cve_review(self):
        self.assertIn("foreign_cve_not_in_description_or_urls", flag_text("CVE-2023-1234 is discussed here.", "CVE-2024-1234", ""))

    def test_incomplete_ending(self):
        self.assertIn("possible_cutoff_ending", flag_text("The attacker can modify data and", "CVE-2024-1234", ""))

    def test_window_upper_bound_inclusive(self):
        row = {"published_timestamp": "2024-01-01T12:00:00Z", "urls": ["a", "b", "c", "a"],
               "url_change_dates": [{"url":"a","earliest_added_date":"2024-03-31T12:00:00Z"},
                                    {"url":"b","earliest_added_date":"2024-03-31T12:00:01Z"}]}
        got = source_window(row,"urls")
        self.assertEqual(got["eligible_under_documented_upper_bound"],["a"])
        self.assertEqual(got["missing_history"],["c"])

    def test_similarity_pairs_not_double_counted(self):
        cells = [{"cve":c,"group":"github","summary":s,"nonempty":True,"urls":["u"],"description":"same"}
                 for c,s in [("CVE-2024-1234","The same vulnerability text appears here."),
                             ("CVE-2024-1235","The same vulnerability text appears here.")]]
        pairs,method = similar_pairs(cells)
        self.assertEqual(len(pairs),1)
        self.assertTrue(pairs[0]["normalized_identical"])
        self.assertEqual(method["all_pairs_compared"],1)


class OfflineScraperCharacterizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_checks(SOURCE/SCRIPT)

    def test_legitimate_content_and_navigation(self):
        text = self.result["valid_article"]["text_returned"]
        self.assertIn("Actual advisory body",text)
        self.assertNotIn("navigation",text)
        self.assertNotIn("hidden()",text)

    def test_known_and_unknown_challenges(self):
        self.assertEqual(self.result["recognized_challenge"]["text_returned"],"")
        self.assertIn("Anubis",self.result["unrecognized_challenge"]["text_returned"])

    def test_plain_patch_is_skipped(self):
        self.assertEqual(self.result["plain_text_patch"]["text_returned"],"")

    def test_duplicate_fetch_and_unused_minimum(self):
        self.assertEqual(self.result["duplicate_url"]["request_count"],2)
        self.assertEqual(self.result["below_declared_minimum"]["text_returned"].strip(),"OK")

    def test_prompt_identity_and_truncation(self):
        checks = self.result["prompt_and_completion"]
        self.assertFalse(checks["cve_id_supplied_to_function_reaches_prompt"])
        self.assertFalse(checks["source_url_reaches_prompt"])
        self.assertTrue(checks["finish_reason_length_returned_as_ordinary_summary"])

    def test_missing_key_call(self):
        self.assertIn("api_key",self.result["summary_call_signature_failures"][0]["error"])


if __name__ == "__main__":
    unittest.main()
