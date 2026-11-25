"""Synthetic DQ08 packet, decision validation and preservation tests."""
import copy
from contextlib import redirect_stdout
from datetime import datetime
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import review_grouping_keys as workflow
from test_forecast_cohort_coverage import grouped
from test_forecast_baseline import artifacts
from epss.forecast_data import read_json, write_json, sha256, save_prepared


def packet_fixture():
    raw = grouped()
    for r in raw[:2]:
        r['social_media_posts'] = [{'post': 'SAMPLE full page '+('context '*70)}]
    data = artifacts(raw)
    a = workflow.audit.analyze(data)
    packet = workflow.build_packet(a['review_queue'], data['grouping_evidence.json'],
                                   data['split_manifest.json'], raw, {'fixture': True})
    return packet, data, raw


def reviewed(row, decision='retain'):
    row.update(decision=decision, reviewer='Synthetic reviewer',
               reviewed_at='2026-10-18T12:00:00+00:00', reason='Synthetic source-context reason',
               evidence_note='Synthetic CVE field inspected', predictor_safety_plan='')


class KeyReviewTests(unittest.TestCase):
    def test_full_context_and_unresolved_defaults(self):
        packet, data, raw = packet_fixture()
        p = next(c for c in packet['contexts'] if c['kind'] == 'post')
        self.assertGreater(len(p['full_normalized_value']), 240)
        self.assertEqual(len(p['normalized_excerpt']), 240)
        self.assertTrue(all(r['field_path'] == 'social_media_posts[0].post' for r in p['source_occurrences']))
        self.assertTrue(all(r['assigned_split'] == 'excluded' for r in p['source_occurrences']))
        decisions = workflow.template(packet, 'fixture_digest')
        result = workflow.validate_decisions(packet, 'fixture_digest', decisions)
        self.assertEqual(result['decision_counts']['unresolved'], len(packet['contexts']))
        self.assertEqual(result['keys_with_review_metadata'], 0)
        self.assertFalse(result['policy_application_authorized'])
        self.assertNotIn('epss_scores', str(packet))

    def test_no_numeric_outcomes_or_unrelated_content_in_packet(self):
        packet, data, raw = packet_fixture()
        modified = copy.deepcopy(raw)
        for r in modified:
            r['epss_scores'] = object()
            r['in_vckev'] = object()
            r['summ_other'] = 'Not grouping evidence'
        a = workflow.audit.analyze(data)
        self.assertEqual(packet, workflow.build_packet(a['review_queue'], data['grouping_evidence.json'],
                                                       data['split_manifest.json'], modified, {'fixture': True}))

    def test_active_text_withheld_and_cannot_be_excluded(self):
        packet, data, raw = packet_fixture()
        manifest = copy.deepcopy(data['split_manifest.json'])
        manifest[raw[0]['cve']]['split'] = 'test'
        a = workflow.audit.analyze(data)
        p = workflow.build_packet(a['review_queue'], data['grouping_evidence.json'], manifest, raw, {})
        c = next(c for c in p['contexts'] if c['kind'] == 'post')
        self.assertIsNone(c['full_normalized_value'])
        self.assertIsNone(c['normalized_excerpt'])
        d = workflow.template(p, 'fixture')
        row = next(r for r in d['decisions'] if r['kind'] == 'post')
        reviewed(row, 'exclude')
        row['predictor_safety_plan'] = 'Synthetic removal plan'
        with self.assertRaisesRegex(ValueError, 'withheld'):
            workflow.validate_decisions(p, 'fixture', d)

    def test_complete_coverage_identity_and_immutable_binding(self):
        p,_,_ = packet_fixture()
        d = workflow.template(p, 'fixture')
        for modification in ('missing', 'duplicate', 'unknown', 'context', 'kind', 'packet'):
            changed = copy.deepcopy(d)
            if modification == 'missing':
                changed['decisions'].pop()
            elif modification == 'duplicate':
                changed['decisions'].append(changed['decisions'][0])
            elif modification == 'unknown':
                changed['decisions'][0]['key_sha256'] = 'unknown'
            elif modification == 'context':
                changed['decisions'][0]['context_sha256'] = 'wrong'
            elif modification == 'kind':
                changed['decisions'][0]['kind'] = 'wrong'
            else:
                changed['packet_sha256'] = 'wrong'
            with self.subTest(modification=modification), self.assertRaises(ValueError):
                workflow.validate_decisions(p, 'fixture', changed)

    def test_review_metadata_and_exclusion_plan_required(self):
        p,_,_ = packet_fixture()
        d = workflow.template(p, 'fixture')
        reviewed(d['decisions'][0], 'exclude')
        with self.assertRaisesRegex(ValueError, 'safety plan'):
            workflow.validate_decisions(p, 'fixture', d)
        d['decisions'][0]['predictor_safety_plan'] = 'Remove exact shared value from future predictors; audit remaining duplicates separately.'
        result = workflow.validate_decisions(p, 'fixture', d)
        self.assertEqual(result['decision_counts']['exclude'], 1)
        self.assertFalse(result['scientific_decisions_independently_verified'])
        self.assertFalse(result['policy_application_authorized'])
        for field in ('reason', 'reviewer', 'evidence_note'):
            changed = copy.deepcopy(d)
            changed['decisions'][0][field] = ' '
            with self.subTest(field=field), self.assertRaises(ValueError):
                workflow.validate_decisions(p, 'fixture', changed)
        d['decisions'][0]['reviewed_at'] = '2026-10-18T12:00:00'
        with self.assertRaisesRegex(ValueError, 'timezone'):
            workflow.validate_decisions(p, 'fixture', d)

    def test_recorded_unresolved_not_counted_as_resolved(self):
        p,_,_ = packet_fixture()
        d = workflow.template(p, 'fixture')
        reviewed(d['decisions'][0], 'unresolved')
        result = workflow.validate_decisions(p, 'fixture', d)
        self.assertEqual(result['keys_with_review_metadata'], 1)
        self.assertEqual(result['decision_counts']['unresolved'], len(p['contexts']))
        self.assertFalse(result['all_keys_resolved'])

    def test_prepare_record_validate_and_preserve_inputs(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            packet, data, raw = packet_fixture()
            source = root/'raw'/'data.json'
            write_json(source, raw)
            data['preparation_summary.json'].update(source_path=str(source), source_sha256=sha256(source))
            prepared = root/'prepared'
            save_prepared(prepared, data, sha256(source))
            audit_dir = root/'audit'
            workflow.audit.run(prepared, audit_dir)
            before = {str(p):sha256(p) for p in root.rglob('*') if p.is_file()}
            output = root/'review'
            workflow.prepare(audit_dir, output, check_only=True)
            self.assertFalse(output.exists())
            workflow.prepare(audit_dir, output)
            p, h = workflow.load_packet(output)
            self.assertEqual(len(p['contexts']), len(packet['contexts']))
            workflow.validate(output)
            self.assertTrue(read_json(output/'validation'/'analysis.json')['structure_valid'])
            lock = read_json(output/'packet_lock.json')
            old_decisions_sha = sha256(output/'decisions.json')
            workflow.record(output, 1, None, 'retain', 'Synthetic reviewer', 'Specific source-context explanation', 'Examined the stored exact URL and associated CVE source locations')
            self.assertEqual(sha256(output/'decision_history'/(old_decisions_sha+'.json')), old_decisions_sha)
            d = read_json(output/'decisions.json')
            self.assertEqual(d['decisions'][0]['decision'], 'retain')
            self.assertIsNotNone(datetime.fromisoformat(d['decisions'][0]['reviewed_at']).tzinfo)
            workflow.validate(output)
            result = read_json(output/'validation'/'analysis.json')
            self.assertEqual(result['decisions_sha256'], sha256(output/'decisions.json'))
            for name,digest in lock['files'].items():
                self.assertEqual(sha256(output/name), digest)
            for path,digest in before.items():
                self.assertEqual(sha256(path), digest)
            with self.assertRaisesRegex(ValueError, 'Output exists'):
                workflow.prepare(audit_dir, output)
            with self.assertRaisesRegex(ValueError, 'overlaps protected'):
                workflow.prepare(audit_dir, prepared/'bad')
            # Rehashing an edited immutable context must not bypass source reconstruction.
            p['contexts'][0]['review_hints'] = ['fabricated']
            write_json(output/'review_packet.json', p)
            lock['files']['review_packet.json'] = sha256(output/'review_packet.json')
            write_json(output/'packet_lock.json', lock)
            with self.assertRaisesRegex(ValueError, 'does not reproduce'):
                workflow.load_packet(output)

    def test_invalid_rank_and_decision(self):
        p,_,_ = packet_fixture()
        for rank in (0, -1, len(p['contexts'])+1):
            with self.assertRaises(ValueError):
                workflow.choose(p, rank=rank)
        with self.assertRaises(ValueError):
            workflow.choose(p, key='missing')
        d = workflow.template(p, 'fixture')
        d['decisions'][0]['decision'] = 'automatically_approved'
        with self.assertRaisesRegex(ValueError, 'Unsupported decision'):
            workflow.validate_decisions(p, 'fixture', d)


if __name__ == '__main__':
    unittest.main()
