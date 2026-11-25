"""VK03 synthetic checks; no dataset training or test-set access."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss.vulncheck_study import development_v1 as dev
from epss.vulncheck_study import classifier_smoke_v2 as common
from epss.vulncheck_study import adapter_v2 as data


def fixture():
    rows, labels = [], []
    for i in range(12):
        split = 'train' if i < 8 else 'validation'
        y = int(i % 3 == 0)
        rows.append({'cve': f'CVE-2024-{10000+i}', 'split': split,
                     'documents': [{'document_id': f'description:{i}', 'source_text_type': 'description',
                                    'text': 'memory corruption remote attack' if y else 'configuration update product'}],
                     'source_mask': [1, int(i % 2 == 0), 0, 0], 'structured': [float(i % 3), float(i % 2)]})
        labels.append({'cve': rows[-1]['cve'], 'split': split, 'target': y})
    return rows, labels


class DevelopmentTests(unittest.TestCase):
    def test_unique_config_keys(self):
        def pairs(values):
            self.assertEqual(len(values), len({k for k, _ in values}))
            return dict(values)
        json.loads(dev.CONFIG.read_text(), object_pairs_hook=pairs)

    def test_training_weights(self):
        self.assertEqual(dev.weights([0, 0, 0, 1], 'balanced'), {0: 2/3, 1: 2.})
        self.assertIsNone(dev.weights([0, 1], 'unweighted'))
        for y in ([0, 0], [1, 1], [0, 2], []):
            with self.assertRaises(ValueError):
                dev.weights(y, 'balanced')

    def test_csv_export_round_trip(self):
        text = dev.csv_text([{'split': 'train', 'count': 10}])
        self.assertNotIn('\r', text)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'plan'
            data.publish(out, {'table.csv': text}, {})
            data.publish(out, {'table.csv': text}, {}, verify=True)

    def test_class_filter_audit(self):
        rows, labels = fixture()
        # One listed training CVE loses two SMP documents, but counts once as affected.
        decisions = []
        for r in rows:
            decisions.append({'cve': r['cve'], 'split': r['split'], **r['documents'][0], 'decision': 'accept', 'reasons': []})
        for i in range(2):
            decisions.append({'cve': rows[0]['cve'], 'split': 'train', 'document_id': f'smp:{i}',
                              'source_text_type': 'smp', 'decision': 'quarantine', 'reasons': ['cue', 'other']})
        audit = dev.audit_sources(rows, labels, decisions, ['description', 'github_summary', 'non_github_summary', 'smp'])
        listed_train = next(r for r in audit['by_class'] if r['split'] == 'train' and r['listed'] == 1)
        self.assertEqual(listed_train['cves_with_any_exclusion'], 1)
        self.assertEqual(listed_train['documents_excluded'], 2)
        smp = next(r for r in audit['by_source_and_class'] if r['split'] == 'train' and r['listed'] == 1 and r['source'] == 'smp')
        self.assertEqual(smp['cves_losing_entire_source'], 1)
        with self.assertRaises(ValueError):
            dev.audit_sources(rows, labels, decisions + decisions[:1], ['description', 'smp'])

    def test_shared_wording_not_identity(self):
        rows, _ = fixture()
        before = copy.deepcopy(rows)
        audit = dev.overlap_audit(rows)
        self.assertEqual(audit['groups_by_source'], {'description': 2})
        self.assertEqual(audit['distinct_validation_cves'], 4)
        self.assertEqual(rows, before)
        rows[-1]['documents'][0]['source_text_type'] = 'smp'
        self.assertEqual(dev.overlap_audit(rows)['distinct_validation_cves'], 3)

    def test_join_and_roles(self):
        rows, labels = fixture()
        ids = {s: sorted(r['cve'] for r in rows if r['split'] == s) for s in ('train', 'validation')}
        identity = {'scale': 'full', 'ids': ids}
        state = {'fit_ids': ids['train']}
        p = {'expected_counts': {'train': 8, 'validation': 4}, 'expected_listed': {'train': 3, 'validation': 1}}
        dev.validate_join(rows, labels, identity, state, p)
        for badlabels in (labels[::-1], labels[:-1]):
            with self.assertRaises(ValueError):
                dev.validate_join(rows, badlabels, identity, state, p)
        bad = copy.deepcopy(rows)
        bad[-1]['split'] = 'test'
        with self.assertRaises(ValueError):
            dev.validate_join(bad, labels, identity, state, p)
        with self.assertRaises(ValueError):
            dev.validate_join(rows, labels, identity, {'fit_ids': ids['validation']}, p)

    def test_all_eight_fits_reload_and_report_verification(self):
        rows, labels = fixture()
        train, val = rows[:8], rows[8:]
        y, v = np.array([r['target'] for r in labels[:8]]), np.array([r['target'] for r in labels[8:]])
        p = dev.config()
        vec = common.fit_vectorizer(train, p)
        prior = float(y.mean())
        predictions = dev.prediction_rows('training_prevalence', val, v, np.full(len(v), prior))
        metrics = {'training_prevalence': dev.metric(v, np.full(len(v), prior))}
        diagnostics = {}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for mode in p['weight_modes']:
                for variant in p['variants']:
                    bundle, values, diag = dev.fit_one(train, val, y, v, variant, mode, vec, p, ['one', 'two'])
                    name = variant + '__' + mode
                    joblib.dump(bundle, out / (name + '.joblib'))
                    np.testing.assert_array_equal(values, common.predict(joblib.load(out / (name + '.joblib')), val))
                    metrics[name], diagnostics[name] = diag['validation_metrics'], diag
                    predictions.extend(dev.prediction_rows(name, val, v, values))
            identity = {'fixture': True, 'balanced_training_weights': dev.weights(y, 'balanced')}
            for name, obj in {'run_identity.json': identity, 'predictions.json': predictions,
                              'metrics.json': metrics, 'diagnostics.json': diagnostics}.items():
                (out / name).write_text(data.dump(obj))
            (out / 'artifact_lock.json').write_text(data.dump({'files': {f.name: data.sha(f) for f in out.iterdir()}}))
            dev.evaluate_saved(out, identity, val, v, prior, p)
            (out / 'predictions.json').write_text('[]')
            with self.assertRaises(ValueError):
                dev.evaluate_saved(out, identity, val, v, prior, p)

    def test_validation_labels_cannot_change_fit(self):
        rows, labels = fixture()
        y = np.array([r['target'] for r in labels[:8]])
        v = np.array([r['target'] for r in labels[8:]])
        p = dev.config()
        a, first, _ = dev.fit_one(rows[:8], rows[8:], y, v, 'structured_only', 'balanced', None, p, ['a', 'b'])
        b, second, _ = dev.fit_one(rows[:8], rows[8:], y, 1-v, 'structured_only', 'balanced', None, p, ['a', 'b'])
        np.testing.assert_array_equal(first, second)
        self.assertEqual(a['model'].class_weight, b['model'].class_weight)

    def test_no_validation_vocab_and_no_labels_as_features(self):
        rows, _ = fixture()
        vec = common.fit_vectorizer(rows[:8], dev.config())
        rows[-1]['documents'][0]['text'] = 'validationonlytoken'
        self.assertNotIn('validationonlytoken', vec.vocabulary_)
        self.assertEqual(common.features(rows[-1:], 'description_tfidf', vec).nnz, 0)
        rows[-1]['target'] = 1
        with self.assertRaises(ValueError):
            common.features(rows[-1:], 'structured_only')


if __name__ == '__main__':
    unittest.main()
