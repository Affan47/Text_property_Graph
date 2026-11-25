import unittest
from datetime import date, timedelta
from unittest.mock import patch

from verify_smp_dates import align_coverage, classify_source, embedded_utc_hint, legacy_cve_id, match_dates, parse_source_time, score_at, sequence_matches
from verify_smp_dates import candidate_text_key, publication_evidence_gaps, pinned_csv_inventory
from smp_embedded_dates import extract_body_dates, compare_embedded_date, score_lookup


class SMPDateTests(unittest.TestCase):
    def test_body_version_and_patch_dates_are_not_publication_templates(self):
        self.assertEqual(extract_body_dates('reddit', 'OTP-26.2.5.10 patch released May 5, 2025'), [])
        self.assertEqual(extract_body_dates('mastodon', 'UPDATE 07.05.2025: fixed versions'), [])

    def test_reddit_digest_date_is_explicit_day_first(self):
        r = extract_body_dates('reddit', 'Top 10 Trending CVEs (07/05/2025) Here is a list')[0]
        self.assertEqual(r['timestamp'], '2025-05-07T00:00:00')
        self.assertEqual(r['precision'], 'day')
        self.assertFalse(r['explicit_timezone'])

    def test_invalid_digest_date_is_not_coerced(self):
        self.assertEqual(extract_body_dates('reddit', 'Top 10 Trending CVEs (31/02/2025)')[0]['kind'], 'INVALID_DATE')

    def test_mastodon_12_hour_clock(self):
        for stamp, hour in [('12:15AM', 0), ('12:15PM', 12), ('02:15PM', 14)]:
            r = extract_body_dates('mastodon', 'CVE notice March 11, 2025 at ' + stamp)[0]
            self.assertIn(f'T{hour:02d}:15:00', r['timestamp'])

    def test_telegram_offset_can_change_score_day(self):
        r = extract_body_dates('telegram', '2025-05-14 01:30:00 (UTC +09:00) alert')[0]
        self.assertEqual(r['utc_timestamp'], '2025-05-13T16:30:00+00:00')
        self.assertIsNone(extract_body_dates('telegram', '2025-05-14 01:30:00 alert')[0]['utc_timestamp'])

    def test_invalid_explicit_timezone_not_silently_dropped(self):
        self.assertEqual(extract_body_dates('telegram', '2025-05-14 01:30:00 (UTC nonsense)')[0]['kind'], 'INVALID_DATE')

    def test_day60_is_in_plan_but_day61_only_in_raw_window(self):
        pub = date(2025, 1, 1)
        day = pub + timedelta(days=61)
        scores = {day.isoformat(): 0.0}
        self.assertEqual(score_lookup(scores, day, pub, 60)['status'], 'OUTSIDE_SELECTED_WINDOW')
        self.assertEqual(score_lookup(scores, day, pub, 90)['score'], 0.0)

    def test_embedded_date_alignment_rejects_invalid_scores(self):
        day = date(2025, 1, 1)
        for invalid in (True, float('nan'), float('inf'), -0.1, 1.1):
            with self.assertRaises(ValueError):
                score_lookup({day.isoformat():invalid}, day, day, 60)

    def test_body_date_matching_nvd_does_not_verify_post(self):
        record = {'published_timestamp': '2025-03-11T14:15:25.427', 'epss_scores': {},
                  'social_media_posts': [{'platform': 'mastodon', 'post': 'March 11, 2025 at 02:15PM'}]}
        row = {'post_index': 0, 'candidate_timestamp': '2025-05-01T00:00:00'}
        result = compare_embedded_date(record, row)
        self.assertTrue(result['same_nvd_stored_minute'])
        self.assertFalse(result['verified_post_publication'])
        self.assertEqual(result['day_after_nvd_publication'], 0)

    def test_multiple_body_timestamps_are_not_silently_selected(self):
        record = {'social_media_posts': [{'platform': 'mastodon', 'post': 'March 11, 2025 at 02:15PM March 12, 2025 at 02:15PM'}]}
        self.assertEqual(compare_embedded_date(record, {'post_index': 0})['selection_status'], 'MULTIPLE_DATES_REQUIRE_REVIEW')

    def test_raw_candidates_require_same_cve_and_platform(self):
        csv_text = ('cve,source,text,created_at\n'
                    'CVE-2025-3600,Reddit,See https://example.com/a,2025-05-01\n'
                    'CVE-2025-3601,Telegram,See https://example.com/a,2025-05-01\n')
        with patch('verify_smp_dates.git', return_value=csv_text):
            result = pinned_csv_inventory('sample.csv', set(),
                {('CVE-2025-3600', 'telegram', 'See <URL>')})
        self.assertEqual(result['candidate_matches'], [])

    def test_raw_candidate_date_conflicts_are_not_resolved_by_earliest(self):
        csv_text = ('cve,source,text,created_at\n'
                    'CVE-2025-3600,Telegram,See https://example.com/a,2025-05-02\n'
                    'CVE-2025-3600,Telegram,See https://example.com/b,2025-05-01\n')
        with patch('verify_smp_dates.git', return_value=csv_text):
            result = pinned_csv_inventory('sample.csv', set(),
                {('CVE-2025-3600', 'telegram', 'See <URL>')})
        candidates = result['candidate_matches'][0]['raw_rows']
        self.assertEqual([r['time_fields']['created_at'] for r in candidates], ['2025-05-02', '2025-05-01'])
        self.assertTrue(all(not r['publication_verified'] for r in candidates))
        self.assertEqual(result['timestamp_precision_counts'], {'DATE_ONLY': 2})

    def test_raw_timestamp_precision_and_bad_dates_reported(self):
        csv_text = ('cve,source,text,created_at\n'
                    'CVE-2025-3600,Telegram,Text,2025-05-02T10:00:00Z\n'
                    'CVE-2025-3600,Telegram,Text,2025-05-02T10:00:00\n'
                    'CVE-2025-3600,Telegram,Text,2025-02-30\n')
        with patch('verify_smp_dates.git', return_value=csv_text):
            result = pinned_csv_inventory('sample.csv', set(),
                {('CVE-2025-3600', 'telegram', 'Text')})
        self.assertEqual(result['timestamp_precision_counts'],
                         {'TIMESTAMP_WITH_OFFSET': 1, 'TIMESTAMP_WITHOUT_OFFSET': 1, 'MISSING_OR_INVALID': 1})

    def test_cleaned_text_candidates_handle_urls_and_entities(self):
        self.assertEqual(candidate_text_key("CVE-2025-3600 &amp; test https://example.com/a\n next"),
                         candidate_text_key("CVE-2025-3600 & test <URL> next"))

    def test_cleaned_match_does_not_change_words_or_dates(self):
        self.assertNotEqual(candidate_text_key("Not exploited 2025-05-01"),
                            candidate_text_key("Exploited 2025-05-02"))

    def test_url_masking_collision_cannot_verify_publication(self):
        self.assertEqual(candidate_text_key("See https://example.com/1"),
                         candidate_text_key("See https://example.com/2"))
        match = {"identity_fields": {"post_id": "123"}, "explicit_timezone": True,
                 "match_kind": "CLEANED_TEXT_CANDIDATE"}
        gaps = publication_evidence_gaps([match])
        self.assertIn("LOSSY_TEXT_MATCH_ONLY", gaps)
        self.assertIn("PUBLICATION_EVENT_SEMANTICS_UNVERIFIED", gaps)

    def test_exact_raw_match_still_needs_lineage_and_historical_text(self):
        match = {"identity_fields": {"post_id": "123"}, "explicit_timezone": True,
                 "match_kind": "EXACT_TEXT"}
        gaps = publication_evidence_gaps([match])
        self.assertNotIn("LOSSY_TEXT_MATCH_ONLY", gaps)
        self.assertIn("COLLECTOR_TO_ROW_LINEAGE_UNVERIFIED", gaps)
        self.assertIn("HISTORICAL_TEXT_VERSION_UNVERIFIED", gaps)

    def test_absent_original_export_is_not_verified(self):
        gaps = publication_evidence_gaps([])
        self.assertIn("NO_RAW_PLATFORM_TEXT_CANDIDATE", gaps)
        self.assertIn("ORIGINAL_POST_ID_OR_PERMALINK_UNAVAILABLE", gaps)
        self.assertIn("ORIGINAL_PUBLICATION_TIMEZONE_UNAVAILABLE", gaps)

    def test_ambiguous_raw_candidates_preserved_for_review(self):
        match = {"identity_fields": {}, "explicit_timezone": False, "match_kind": "EXACT_TEXT"}
        self.assertIn("MULTIPLE_RAW_ROWS_REQUIRE_IDENTITY_REVIEW", publication_evidence_gaps([match, match]))

    def test_text_timezone_hint_is_not_a_source_timestamp(self):
        value = embedded_utc_hint("2025-05-14 23:30:18.190614 (UTC +09:00) 1. CVE-2025-3600")
        self.assertEqual(value.isoformat(), "2025-05-14T14:30:18.190614+00:00")
        self.assertIsNone(embedded_utc_hint("CVE-2025-3600"))

    def test_sequence_replay_checks_order_and_platform(self):
        rows = [{"text": "a", "source": "telegram_batch1"}, {"text": "b", "source": "reddit"}]
        posts = [{"post": "a", "platform": "telegram"}, {"post": "b", "platform": "reddit"}]
        self.assertTrue(sequence_matches(posts, rows))
        self.assertFalse(sequence_matches(posts[::-1], rows))
        self.assertFalse(sequence_matches(posts[:1], rows))

    def test_legacy_suffix_never_strips_canonical_cve_number(self):
        self.assertEqual(legacy_cve_id("CVE-2025-3600-1"), "CVE-2025-3600")
        self.assertEqual(legacy_cve_id("CVE-2025-3600"), "CVE-2025-3600")

    def test_naive_timezone_not_invented(self):
        self.assertIsNone(parse_source_time("2025-05-01", "00:00").tzinfo)

    def test_explicit_offset_preserved(self):
        self.assertEqual(parse_source_time("2025-05-01", "01:15+02:00").utcoffset(), timedelta(hours=2))

    def test_invalid_dates_not_coerced(self):
        for day, clock in (("2025-02-30", "00:00"), ("01/05/2025", "12:00"), ("2025-05-01", "")):
            with self.assertRaises(ValueError):
                parse_source_time(day, clock)

    def test_repeated_rows_do_not_make_multiple_dates(self):
        r = {"date": "2025-05-01", "time": "00:00"}
        self.assertEqual(match_dates([r, r]), ("ONE_SOURCE_TIMESTAMP", ["2025-05-01T00:00:00"]))

    def test_same_day_multiple_times_remain_ambiguous(self):
        rows = [{"date": "2025-05-01", "time": t} for t in ("00:00", "12:00")]
        self.assertEqual(match_dates(rows)[0], "MULTIPLE_SOURCE_TIMESTAMPS")

    def test_bad_match_never_selects_good_date(self):
        rows = [{"date": "2025-05-01", "time": t} for t in ("00:00", "bad")]
        self.assertEqual(match_dates(rows)[0], "UNPARSEABLE_SOURCE_DATE")

    def test_platform_unknown_not_guessed(self):
        self.assertIsNone(classify_source("unknown"))
        self.assertIsNone(classify_source("reddit_telegram"))
        self.assertEqual(classify_source("telegram_batch1_cleaned"), "telegram")

    def test_null_zero_outside_are_distinct(self):
        scores = {"2025-05-01": None, "2025-05-02": 0.0}
        self.assertEqual(score_at(scores, date(2025, 5, 1))[1], "MISSING_IN_SAVED_WINDOW")
        self.assertEqual(score_at(scores, date(2025, 5, 2)), (0.0, "OBSERVED"))
        self.assertEqual(score_at(scores, date(2025, 5, 3))[1], "OUTSIDE_SAVED_WINDOW")

    def test_invalid_score_rejected(self):
        for value in (True, float("nan"), float("inf"), -0.1, 1.1):
            with self.assertRaises(ValueError):
                score_at({"2025-05-01": value}, date(2025, 5, 1))

    def test_coverage_excludes_event_day_for_before_after(self):
        anchor = date(2025, 5, 1)
        scores = {(anchor + timedelta(days=k)).isoformat(): 0.5 for k in range(-30, 31) if k != 0}
        offsets, windows = align_coverage(scores, anchor)
        self.assertTrue(windows["30"]["all_before_and_after_days"])
        self.assertEqual(offsets["0"]["status"], "OUTSIDE_SAVED_WINDOW")
        self.assertTrue(windows["30"]["minus1_plusH_pair"])
        self.assertFalse(windows["30"]["same_day_plusH_pair"])

    def test_gap_blocks_complete_window_not_endpoint_pair(self):
        anchor = date(2025, 5, 1)
        scores = {(anchor + timedelta(days=k)).isoformat(): 0.5 for k in range(-30, 31)}
        scores[(anchor + timedelta(days=3)).isoformat()] = None
        _, windows = align_coverage(scores, anchor)
        self.assertFalse(windows["7"]["all_before_and_after_days"])
        self.assertTrue(windows["7"]["minus1_plusH_pair"])

    def test_join_absence_explicit(self):
        self.assertEqual(match_dates([]), ("NO_SOURCE_MATCH", []))


if __name__ == "__main__":
    unittest.main()
