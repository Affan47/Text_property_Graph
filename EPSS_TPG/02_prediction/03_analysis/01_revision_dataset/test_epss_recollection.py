"""Offline regression cases for the independent missing-only comparison."""
from collections import Counter
from copy import deepcopy
from datetime import date, timedelta
import tempfile
import gzip
import hashlib
from pathlib import Path
import unittest

from compare_epss_recollection import compare, strict_json, verify_cached_examples, availability_distribution


class RecollectionTests(unittest.TestCase):
    def test_availability_group_boundaries_and_zero_scores(self):
        records = [{"epss_scores": {str(i): 0.0 if i < n else None for i in range(91)}}
                   for n in range(92)]
        result = availability_distribution(records)
        self.assertEqual([r["cves"] for r in result["grouped_distribution"]],
                         [1, 30, 30, 15, 12, 1, 1, 1, 1])
        self.assertEqual(result["cves_with_at_least_one_missing_score"], 91)
        self.assertTrue(all(r["days_with_score"] + r["days_without_score"] == 91
                            and r["cves"] == 1 for r in result["exact_distribution"]))

    def test_availability_rejects_wrong_window(self):
        with self.assertRaisesRegex(ValueError, "91-day"):
            availability_distribution([{"epss_scores": {"day": None}}])

    def setUp(self):
        self.days = [(date(2024, 1, 1) + timedelta(days=i)).isoformat() for i in range(91)]
        self.before = [{"cve": "CVE-2024-12345", "published_date": self.days[0],
                        "description": "Keep source text", "epss_scores": dict.fromkeys(self.days, .1)}]
        self.before[0]["epss_scores"][self.days[1]] = None
        self.before[0]["epss_scores"][self.days[2]] = None
        self.after = deepcopy(self.before)
        self.after[0]["epss_scores"][self.days[1]] = .2
        self.after[0]["epss_score_status"] = dict.fromkeys(self.days, "EXISTING_UNVERIFIED")
        self.after[0]["epss_score_status"].update({self.days[1]: "OBSERVED", self.days[2]: "CVE_NOT_IN_SNAPSHOT"})
        self.report = {"as_of": "2024-04-01", "cves": 1, "days_per_cve": 91,
                       "score_slots": 91, "filled_scores": 90, "complete_windows": 0,
                       "run_status": "complete", "failed_dates": [], "conflicting_dates": [],
                       "status_counts": dict(Counter(self.after[0]["epss_score_status"].values())),
                       "snapshots": {d: {"status": "VALID", "requested_date": d,
                                         "sha256": "fixture", "url": "https://example.invalid/archive"}
                                     for d in self.days[1:3]}}

    def test_valid_recovery_and_confirmed_absence(self):
        result = compare(self.before, self.after, self.report)
        self.assertEqual(result["newly_recovered_scores"], 1)
        self.assertEqual(result["remaining_null_reasons"], {"CVE_NOT_IN_SNAPSHOT": 1})

    def test_changed_existing_score_rejected(self):
        self.after[0]["epss_scores"][self.days[0]] = .9
        with self.assertRaisesRegex(ValueError, "existing score changed"):
            compare(self.before, self.after, self.report)

    def test_changed_description_rejected(self):
        self.after[0]["description"] = "Changed"
        with self.assertRaisesRegex(ValueError, "non-EPSS"):
            compare(self.before, self.after, self.report)

    def test_missing_day_rejected(self):
        del self.after[0]["epss_scores"][self.days[0]]
        with self.assertRaisesRegex(ValueError, "91-day"):
            compare(self.before, self.after, self.report)

    def test_fabricated_recovery_without_provenance_rejected(self):
        del self.report["snapshots"][self.days[1]]
        with self.assertRaisesRegex(ValueError, "provenance"):
            compare(self.before, self.after, self.report)

    def test_invalid_scores_rejected(self):
        for value in [True, -1, 1.01, float("nan"), float("inf")]:
            with self.subTest(value=value):
                self.after[0]["epss_scores"][self.days[1]] = value
                with self.assertRaisesRegex(ValueError, "invalid score"):
                    compare(self.before, self.after, self.report)

    def test_wrong_count_rejected(self):
        self.report["filled_scores"] = 91
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            compare(self.before, self.after, self.report)

    def test_null_mislabeled_observed_rejected(self):
        self.after[0]["epss_score_status"][self.days[2]] = "OBSERVED"
        with self.assertRaisesRegex(ValueError, "invalid null status"):
            compare(self.before, self.after, self.report)

    def test_failure_must_be_reported(self):
        self.after[0]["epss_score_status"][self.days[2]] = "DOWNLOAD_ERROR"
        self.report["status_counts"] = dict(Counter(self.after[0]["epss_score_status"].values()))
        self.report["snapshots"][self.days[2]]["status"] = "DOWNLOAD_ERROR"
        self.report["failed_dates"] = [self.days[2]]
        with self.assertRaisesRegex(ValueError, "Run status mismatch"):
            compare(self.before, self.after, self.report)
        self.report["run_status"] = "incomplete"
        self.assertEqual(compare(self.before, self.after, self.report)["verification"], "passed")

    def test_nonstandard_json_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            path.write_text('[{"score": NaN}]')
            with self.assertRaisesRegex(ValueError, "Non-standard"):
                strict_json(path)

    def test_direct_archive_content_check(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)
            content = gzip.compress(b"#model_version:test\ncve,epss,percentile\nCVE-2024-12345,0.2,0.9\n")
            (cache / f"{self.days[1]}.csv.gz").write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            result = {"recovered_scores": [{"cve": "CVE-2024-12345", "date": self.days[1],
                                             "new_score": .2, "archive_sha256": digest}],
                      "confirmed_archive_absence_examples": []}
            verify_cached_examples(result, cache)
            self.assertEqual(result["direct_cache_checks"]["recovered_scores_checked"], 1)
            result["recovered_scores"][0]["new_score"] = .3
            with self.assertRaisesRegex(ValueError, "archive content"):
                verify_cached_examples(result, cache)

    def test_direct_archive_hash_check(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)
            (cache / f"{self.days[1]}.csv.gz").write_bytes(b"changed")
            result = {"recovered_scores": [], "confirmed_archive_absence_examples": [
                {"cve": "CVE-2024-12345", "date": self.days[1], "archive_sha256": "wrong"}]}
            with self.assertRaisesRegex(ValueError, "cache hash mismatch"):
                verify_cached_examples(result, cache)


if __name__ == "__main__":
    unittest.main()
