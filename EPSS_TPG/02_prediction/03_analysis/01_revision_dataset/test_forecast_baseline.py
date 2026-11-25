"""Offline persistence and validation-only evaluation regression tests."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss.forecast_baseline import (evaluate_validation, predict_persistence,
                                    regression_metrics, run, OUTPUT_FILES)
from epss.forecast_data import (DEFAULT_SPLIT_POLICY, fit_training_preprocessor,
                               make_record, prepare, read_json, save_prepared,
                               sha256, task_cohorts, write_json)
from test_forecast_data import fixture, set_day, trio


def artifacts(records):
    evidence = []
    x, y, m, groups, summary = prepare(records, evidence=evidence)
    return {'inputs.json': x, 'labels.json': y, 'split_manifest.json': m,
            'duplicate_groups.json': groups, 'preparation_summary.json': summary,
            'training_preprocessor.json': fit_training_preprocessor(x, y, m),
            'split_policy.json': dict(DEFAULT_SPLIT_POLICY), 'reviewed_url_exclusions.json': {},
            'grouping_evidence.json': evidence, 'task_cohorts.json': task_cohorts(y, m)}


def bundle(directory):
    root = Path(directory)
    source = root/'raw'/'data.json'
    write_json(source, trio())
    data = artifacts(trio())
    data['preparation_summary.json'].update(source_path=str(source), source_sha256=sha256(source))
    prepared = root/'prepared'
    save_prepared(prepared, data, sha256(source))
    return prepared


class PersistenceTests(unittest.TestCase):
    def test_uses_day30_not_mean(self):
        raw = fixture()
        set_day(raw, 30, .8)
        pred = predict_persistence(make_record(raw)[0])
        self.assertEqual(pred['prediction'], .8)
        self.assertEqual(pred['last_score_age_at_anchor_days'], 0)

    def test_missing_latest_score_records_age(self):
        raw = fixture()
        for day in (29, 30):
            set_day(raw, day, None)
        pred = predict_persistence(make_record(raw)[0])
        self.assertEqual(pred['prediction'], .1)
        self.assertEqual(pred['last_score_day_after_publication'], 28)
        self.assertEqual(pred['last_score_age_at_anchor_days'], 2)
        self.assertEqual(pred['observed_input_days'], 29)

    def test_zero_is_an_observation(self):
        raw = fixture()
        set_day(raw, 30, 0)
        self.assertEqual(predict_persistence(make_record(raw)[0])['prediction'], 0)

    def test_all_missing_rejected(self):
        raw = fixture()
        for day in range(31):
            set_day(raw, day, None)
        with self.assertRaisesRegex(ValueError, 'at least one'):
            predict_persistence(make_record(raw)[0])

    def test_future_scores_and_text_do_not_change_prediction(self):
        raw = fixture()
        before = predict_persistence(make_record(raw)[0])
        for day in range(31, 91):
            set_day(raw, day, .99)
        raw.update(description='Future description', in_vckev=True,
                   social_media_posts=[{'post': 'Future post', 'date': '2026-01-01'}])
        self.assertEqual(before, predict_persistence(make_record(raw)[0]))

    def test_injected_predictor_rejected(self):
        x, _ = make_record(fixture())
        x['X']['future_score'] = .9
        with self.assertRaises(ValueError):
            predict_persistence(x)

    def test_metrics_known_values(self):
        m = regression_metrics([0, .5, 1], [.1, .5, .9])
        self.assertAlmostEqual(m['mae'], .2/3)
        self.assertAlmostEqual(m['rmse'], (.02/3)**.5)
        self.assertAlmostEqual(m['mean_signed_error'], 0)
        self.assertEqual(m['spearman_rho'], 1)

    def test_rank_ties(self):
        self.assertAlmostEqual(regression_metrics([0, 0, 1], [0, 1, 1])['spearman_rho'], .5)

    def test_undefined_rank_not_nan(self):
        for y, p, reason in [([], [], 'empty_cohort'), ([0], [1], 'fewer_than_two_cves'),
                             ([0, 0], [0, 1], 'constant_actual_scores'),
                             ([0, 1], [0, 0], 'constant_predictions')]:
            m = regression_metrics(y, p)
            self.assertIsNone(m['spearman_rho'])
            self.assertEqual(m['spearman_status'], 'undefined_'+reason)

    def test_metrics_reject_invalid_scores_and_misalignment(self):
        for actual, predicted in [([0], []), ([None], [0]), ([0], [float('nan')]),
                                  ([0], [float('inf')]), ([0], [1.1]), ([False], [0])]:
            with self.assertRaises(ValueError):
                regression_metrics(actual, predicted)

    def test_validation_only(self):
        data = artifacts(trio())
        rows, s = evaluate_validation(data)
        self.assertEqual([r['cve'] for r in rows], ['CVE-2024-1234'])
        self.assertEqual(rows[0]['forecast_horizon_days'], 30)
        self.assertFalse(s['test_performance_evaluated'])
        self.assertFalse(s['model_trained'])
        self.assertEqual(s['exactly_matching_recorded_scores'], 1)
        self.assertEqual(s['median_absolute_error'], 0)
        for split in ('test', 'train', 'all'):
            with self.assertRaisesRegex(ValueError, 'validation evaluation only'):
                evaluate_validation(data, split)

    def test_test_targets_do_not_affect_validation_metrics(self):
        data = artifacts(trio())
        before = evaluate_validation(data)
        data['labels.json']['CVE-2025-1234']['y']['endpoint'] = .99
        self.assertEqual(before, evaluate_validation(data))

    def test_missing_endpoint_excluded_not_zero(self):
        records = trio() + [fixture('CVE-2024-7777', '2024-07-01')]
        set_day(records[-1], 60, None)
        rows, s = evaluate_validation(artifacts(records))
        self.assertEqual(len(rows), 1)
        self.assertEqual(s['validation_cves_without_usable_endpoint_or_input'], 1)

    def test_empty_cohort_fails_explicitly(self):
        records = trio()
        set_day(records[1], 60, None)
        with self.assertRaisesRegex(ValueError, 'Empty or inconsistent'):
            evaluate_validation(artifacts(records))

    def test_forged_cohort_rejected(self):
        data = artifacts(trio())
        data['task_cohorts.json']['endpoint']['validation'].append('CVE-2025-1234')
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            evaluate_validation(data)

    def test_cross_split_related_cves_are_not_scored(self):
        records = trio() + [fixture('CVE-2024-7777', '2024-07-01')]
        records[0]['description'] = records[1]['description'] = 'Shared description'
        rows, _ = evaluate_validation(artifacts(records + [fixture('CVE-2023-9999')]))
        self.assertEqual([r['cve'] for r in rows], ['CVE-2024-7777'])

    def test_equal_group_weight_metric(self):
        records = trio() + [fixture('CVE-2024-7777', '2024-07-01'),
                            fixture('CVE-2024-8888', '2024-07-02')]
        records[1]['description'] = records[3]['description'] = 'Related validation CVEs'
        set_day(records[1], 60, .5)
        set_day(records[3], 60, .5)
        _, s = evaluate_validation(artifacts(records))
        self.assertAlmostEqual(s['equal_group_weight_mae'], .2)
        self.assertAlmostEqual(s['metrics']['mae'], .8/3)

    def test_run_artifacts_reproducible_and_inputs_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            prepared = bundle(tmp)
            initial = {p.name: sha256(p) for p in prepared.iterdir()}
            output = Path(tmp)/'results'
            summary = run(prepared, output)
            self.assertEqual(set(p.name for p in output.iterdir()), OUTPUT_FILES)
            self.assertEqual(summary['metrics']['n'], 1)
            lock = read_json(output/'evaluation_lock.json')
            self.assertTrue(all(sha256(output/name) == digest for name, digest in lock['files'].items()))
            with (output/'predictions.csv').open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                run(prepared, output)
            self.assertEqual(summary, run(prepared, output, overwrite=True))
            self.assertEqual(initial, {p.name: sha256(p) for p in prepared.iterdir()})

    def test_raw_or_prepared_tampering_blocks_outputs(self):
        for target in ('raw', 'prepared'):
            with tempfile.TemporaryDirectory() as tmp:
                prepared = bundle(tmp)
                path = Path(tmp)/'raw'/'data.json' if target == 'raw' else prepared/'inputs.json'
                path.write_text('[]')
                output = Path(tmp)/'results'
                with self.assertRaises(ValueError):
                    run(prepared, output)
                self.assertFalse(output.exists())

    def test_output_cannot_overwrite_inputs_or_unrelated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            prepared = bundle(tmp)
            for out in (prepared, prepared/'nested', Path(tmp), Path(tmp)/'raw'/'results'):
                with self.assertRaises(ValueError):
                    run(prepared, out, overwrite=True)
            output = Path(tmp)/'results'
            output.mkdir()
            (output/'user_notes.txt').write_text('keep')
            with self.assertRaisesRegex(ValueError, 'unrelated'):
                run(prepared, output, overwrite=True)
            self.assertEqual((output/'user_notes.txt').read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
