"""Small offline regression tests for the new snapshot's parsing and EPSS checks."""
from datetime import date, timedelta
import json
from pathlib import Path
import tempfile
import unittest

from analyze_enrichment_update import epss_profile, field_profile, load_json, no_duplicate_keys


class EnrichmentUpdateTests(unittest.TestCase):
    def row(self):
        anchor = date(2025, 1, 1)
        return {"cve": "CVE-2025-12345", "published_date": anchor.isoformat(),
                "epss_scores": {(anchor + timedelta(days=i)).isoformat(): 0.1 for i in range(91)},
                "epss_version_changed": False, "epss_from_to_version": "v3"}

    def test_exact_window(self):
        p = epss_profile([self.row()])
        self.assertEqual(p["complete_91_scores"], 1)
        self.assertEqual(p["observed_scores"], 91)
        self.assertEqual(p["invalid_series_cves"], [])

    def test_null_is_not_zero(self):
        row = self.row()
        row["epss_scores"] = {day: None for day in row["epss_scores"]}
        p = epss_profile([row])
        self.assertEqual(p["all_missing_scores"], 1)
        self.assertIsNone(p["score_min"])
        self.assertEqual(p["rows_with_day_0_score"], 0)

    def test_wrong_day_key(self):
        row = self.row()
        row["epss_scores"]["2030-01-01"] = row["epss_scores"].pop("2025-01-01")
        self.assertEqual(epss_profile([row])["invalid_series_cves"], [row["cve"]])

    def test_invalid_scores(self):
        for value in (True, "0.1", float("nan"), -0.1, 1.1):
            with self.subTest(value=value):
                row = self.row()
                row["epss_scores"]["2025-01-01"] = value
                self.assertEqual(epss_profile([row])["invalid_series_cves"], [row["cve"]])

    def test_duplicate_keys(self):
        with self.assertRaises(ValueError):
            no_duplicate_keys([("a", 1), ("a", 2)])

    def test_nonstandard_json_counted_without_source_change(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "fixture.json"
            raw = '[{"cve":"CVE-2025-12345","score":NaN}]'
            p.write_text(raw)
            rows, constants = load_json(p)
            self.assertEqual(constants, {"NaN": 1})
            self.assertIsNone(rows[0]["score"])
            self.assertEqual(p.read_text(), raw)

    def test_duplicate_cve_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "fixture.json"
            p.write_text(json.dumps([self.row(), self.row()]))
            with self.assertRaises(ValueError):
                load_json(p)

    def test_missingness_distinctions(self):
        p = field_profile([None, "", [], {}, "text"], 6)
        self.assertEqual((p["absent"], p["null"], p["blank_text"], p["empty_container"]), (1, 1, 1, 2))


if __name__ == "__main__":
    unittest.main()
