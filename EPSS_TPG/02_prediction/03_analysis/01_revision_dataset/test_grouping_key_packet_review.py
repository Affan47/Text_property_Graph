"""Tests for read-only interpretation of a completed DQ08 packet."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import review_grouping_key_packet as review
from test_grouping_key_review import packet_fixture, reviewed


class PacketReviewTests(unittest.TestCase):
    def test_counts_are_unique_cves_not_occurrences(self):
        packet, _, _ = packet_fixture()
        before = copy.deepcopy(packet)
        result = review.summarize(packet)
        expected = {x for c in packet['contexts'] for x in c['cve_ids']}
        self.assertEqual(result['distinct_cves'], len(expected))
        self.assertEqual(sum(r['shared_values'] for r in result['by_kind']), len(packet['contexts']))
        self.assertEqual(sum(result['distinct_cves_by_assigned_role'].values()), len(expected))
        self.assertEqual(packet, before)

    def test_active_split_text_rejected(self):
        packet, _, _ = packet_fixture()
        c = next(c for c in packet['contexts'] if c['kind'] == 'post')
        for context in packet['contexts']:
            for o in context['source_occurrences']:
                if o['cve'] == c['cve_ids'][0]:
                    o['assigned_split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'Active-split text'):
            review.summarize(packet)
        c['content_status'] = 'withheld_active_split_text'
        c['full_normalized_value'] = None
        result = review.summarize(packet)
        self.assertNotIn(c['key_sha256'], [e['key_sha256'] for e in result['preliminary_examples_not_decisions']])

    def test_cve_mentions_not_inferred_from_excerpt(self):
        packet, _, _ = packet_fixture()
        c = next(c for c in packet['contexts'] if c['kind'] == 'post')
        c['full_normalized_value'] = 'x '*400 + c['cve_ids'][0].lower()
        result = review.summarize(packet)
        example = next(e for e in result['preliminary_examples_not_decisions'] if e['key_sha256'] == c['key_sha256'])
        self.assertEqual(example['associated_ids_present_in_text'], 1)
        self.assertNotIn(c['cve_ids'][0].lower(), example['excerpt'])

    def test_saved_validation_reproduces_and_stale_results_fail(self):
        packet, _, _ = packet_fixture()
        w = review.workflow
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp).resolve()
            decisions = w.template(packet, 'synthetic_sha')
            review.write_json(directory/'decisions.json', decisions)
            before = review.sha256(directory/'decisions.json')
            with patch.object(w, 'load_packet', return_value=(packet, 'synthetic_sha')), \
                 patch.object(w, 'check_pins'), redirect_stdout(io.StringIO()):
                w.validate(directory)
            result, initial = review.verify_saved(directory, packet, 'synthetic_sha')
            self.assertTrue(initial)
            self.assertFalse(result['policy_application_authorized'])
            self.assertEqual(before, review.sha256(directory/'decisions.json'))
            text = directory/'validation/analysis.txt'
            text.write_text('Changed report', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'validation text'):
                review.verify_saved(directory, packet, 'synthetic_sha')
            reviewed(decisions['decisions'][0])
            review.write_json(directory/'decisions.json', decisions)
            with self.assertRaisesRegex(ValueError, 'stale'):
                review.verify_saved(directory, packet, 'synthetic_sha')


if __name__ == '__main__':
    unittest.main()
