"""Offline extraction contract and benchmark tests; no model weights required."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from tpg.extraction import (SourceDocument, extract_validated, fingerprint, load_security_schema,
                            make_result, parse_timestamp, require_available, strict_json_load,
                            validate_extraction, validate_schema)
from tpg.extraction_benchmark import main, prepare_annotations
from tpg.extraction_evaluation import evaluate_extractions, validate_annotations


class ExtractionContracts(unittest.TestCase):
    def setUp(self):
        self.schema = load_security_schema()
        self.source = SourceDocument("post:1", "Acme 2.4 is not affected. Upgrade to Acme 2.5.",
                                     "social_media_post", record_id="CVE-fixture")
        self.result = make_result(self.source, self.schema, extractor_name="reviewer-A",
                                  extractor_revision="annotation-v1", method="human")
        self.result["entities"] = [
            {"id": "product", "type": "SOFTWARE", "span": self.span(0, 4), "score": None},
            {"id": "version", "type": "VERSION", "span": self.span(5, 8), "score": None}]
        self.result["claims"] = [{"id": "claim-1", "relation": "affected_version",
            "arguments": {"product": ["product"], "version": ["version"]},
            "evidence": [self.span(0, 24)], "polarity": "negated", "modality": "asserted", "score": None}]

    def span(self, start, end):
        return {"start": start, "end": end, "text": self.source.text[start:end]}

    def annotated(self, result=None):
        return {"source": self.source.to_dict(), "annotation_status": "complete",
                "result": self.result if result is None else result}

    def test_valid_result_is_copied_and_roundtrips(self):
        validated = validate_extraction(self.result, self.source, self.schema)
        validated["entities"].clear()
        self.assertEqual(len(self.result["entities"]), 2)
        self.assertEqual(validate_extraction(json.loads(json.dumps(self.result)), self.source, self.schema), self.result)

    def test_invalid_offsets_text_and_boolean_offsets(self):
        for span in ({"start": False, "end": 4, "text": "Acme"},
                     {"start": -1, "end": 4, "text": "Acme"},
                     {"start": 0, "end": 999, "text": "Acme"},
                     {"start": 0, "end": 4, "text": "Fake"}):
            with self.subTest(span=span), self.assertRaises(ValueError):
                result = deepcopy(self.result)
                result["entities"][0]["span"] = span
                validate_extraction(result, self.source, self.schema)

    def test_unicode_offsets_use_original_codepoints(self):
        source = SourceDocument("unicode", "\u0130 Acme", "description")
        result = make_result(source, self.schema, extractor_name="test", extractor_revision="1", method="fixture")
        result["entities"] = [{"id": "e", "type": "SOFTWARE", "span": {"start": 2, "end": 6, "text": "Acme"}, "score": 1.0}]
        validate_extraction(result, source, self.schema)
        result["entities"][0]["span"]["start"] = 3
        with self.assertRaises(ValueError):
            validate_extraction(result, source, self.schema)

    def test_source_identity_metadata_and_text_are_bound(self):
        for source in (replace(self.source, text=self.source.text + " "),
                       replace(self.source, source_id="post:2"),
                       replace(self.source, record_id="CVE-other"),
                       replace(self.source, available_at="2026-01-01T00:00:00Z")):
            with self.subTest(source=source), self.assertRaises(ValueError):
                validate_extraction(self.result, source, self.schema)

    def test_schema_fingerprint_prevents_same_version_drift(self):
        schema = deepcopy(self.schema)
        schema["relations"]["affected_version"]["description"] += " Changed."
        with self.assertRaisesRegex(ValueError, "schema definition"):
            validate_extraction(self.result, self.source, schema)

    def test_extra_target_fields_rejected(self):
        for key in ("in_vckev", "epss_score", "binary_label"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                result = dict(self.result, **{key: 1})
                validate_extraction(result, self.source, self.schema)
        with self.assertRaises(ValueError):
            SourceDocument.from_dict(dict(self.source.to_dict(), in_vckev=True))

    def test_invalid_scores_rejected(self):
        for score in (float("nan"), float("inf"), -0.1, 1.1, True, "0.9"):
            with self.subTest(score=score), self.assertRaises(ValueError):
                result = deepcopy(self.result)
                result["claims"][0]["score"] = score
                validate_extraction(result, self.source, self.schema)

    def test_duplicates_dangling_and_wrong_roles_rejected(self):
        mutations = [
            lambda r: r["entities"].append(deepcopy(r["entities"][0])),
            lambda r: r["entities"].append(dict(r["entities"][0], id="another")),
            lambda r: r["claims"].append(dict(r["claims"][0], id="another")),
            lambda r: r["claims"][0]["arguments"].update(product=["missing"]),
            lambda r: r["claims"][0]["arguments"].update(product=["version"]),
            lambda r: r["claims"][0]["arguments"].pop("version"),
            lambda r: r["claims"][0]["arguments"].update(version=[]),
            lambda r: r["claims"][0].update(relation="not_defined"),
            lambda r: r["claims"][0].update(polarity="true"),
            lambda r: r["claims"][0].update(modality="probably"),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                result = deepcopy(self.result)
                mutation(result)
                validate_extraction(result, self.source, self.schema)

    def test_participants_must_be_inside_evidence(self):
        self.result["claims"][0]["evidence"] = [self.span(25, len(self.source.text))]
        with self.assertRaisesRegex(ValueError, "outside supplied evidence"):
            validate_extraction(self.result, self.source, self.schema)

    def test_extensible_schema_without_python_relation_enum(self):
        relation = self.schema["relations"].pop("affected_version")
        self.schema["relations"]["custom_reviewed_relation"] = relation
        self.result["claims"][0]["relation"] = "custom_reviewed_relation"
        self.result["schema_sha256"] = fingerprint(self.schema)
        validate_extraction(self.result, self.source, self.schema)

    def test_timestamp_cutoff_requires_known_availability(self):
        self.assertEqual(parse_timestamp("2026-01-01T01:00:00+01:00"), parse_timestamp("2026-01-01T00:00:00Z"))
        for bad in ("2026-01-01", "2026-01-01T00:00:00", "01/02/2026", "bad"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_timestamp(bad)
        with self.assertRaisesRegex(ValueError, "unknown"):
            require_available(self.source, "2026-01-01T00:00:00Z")
        source = replace(self.source, available_at="2026-01-01T01:00:00+01:00")
        require_available(source, "2026-01-01T00:00:00Z")
        with self.assertRaisesRegex(ValueError, "after"):
            require_available(source, "2025-12-31T23:59:59Z")

    def test_backend_validation_and_cutoff_before_inference(self):
        backend = Mock()
        backend.extract.return_value = self.result
        self.assertEqual(extract_validated(backend, self.source, self.schema), self.result)
        backend.reset_mock()
        with self.assertRaises(ValueError):
            extract_validated(backend, self.source, self.schema, cutoff="2026-01-01T00:00:00Z")
        backend.extract.assert_not_called()
        backend.extract.return_value = {"bad": "output"}
        with self.assertRaises(ValueError):
            extract_validated(backend, self.source, self.schema)

    def test_strict_json_rejects_nan_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            for text in ('{"score": NaN}', '{"a": 1, "a": 2}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    strict_json_load(path)

    def test_invalid_schema_rejected(self):
        schema = deepcopy(self.schema)
        schema["relations"]["affects"]["roles"]["product"]["types"] = ["UNDEFINED"]
        with self.assertRaises(ValueError):
            validate_schema(schema)

    def test_gold_unreviewed_is_not_negative(self):
        unreviewed = {"source": self.source.to_dict(), "annotation_status": "unreviewed", "result": None}
        validate_annotations([unreviewed], self.schema)
        with self.assertRaisesRegex(ValueError, "No complete"):
            evaluate_extractions([unreviewed], [], self.schema)
        with self.assertRaises(ValueError):
            validate_annotations([dict(unreviewed, result=self.result)], self.schema)

    def test_gold_must_identify_human_review(self):
        result = deepcopy(self.result)
        result["extractor"]["method"] = "model"
        with self.assertRaisesRegex(ValueError, "human reviewer"):
            validate_annotations([self.annotated(result)], self.schema)

    def test_evaluation_ignores_ids_but_respects_qualifiers(self):
        prediction = deepcopy(self.result)
        prediction["entities"][0]["id"] = "renamed"
        prediction["claims"][0]["arguments"]["product"] = ["renamed"]
        prediction["claims"][0]["id"] = "different-claim-id"
        report = evaluate_extractions([self.annotated()], [prediction], self.schema)
        self.assertEqual(report["metrics"]["evidence_claims"]["f1"], 1.0)
        prediction["claims"][0]["polarity"] = "affirmed"
        report = evaluate_extractions([self.annotated()], [prediction], self.schema)
        self.assertEqual(report["metrics"]["relations"]["f1"], 1.0)
        self.assertEqual(report["metrics"]["qualified_claims"]["f1"], 0.0)

    def test_evidence_offsets_are_scored_separately(self):
        prediction = deepcopy(self.result)
        prediction["claims"][0]["evidence"] = [self.span(0, len(self.source.text))]
        report = evaluate_extractions([self.annotated()], [prediction], self.schema)
        self.assertEqual(report["metrics"]["qualified_claims"]["f1"], 1.0)
        self.assertEqual(report["metrics"]["evidence_claims"]["f1"], 0.0)

    def test_reviewed_empty_and_explicit_empty_prediction(self):
        empty = dict(self.result, entities=[], claims=[])
        report = evaluate_extractions([self.annotated(empty)], [empty], self.schema)
        self.assertIsNone(report["metrics"]["qualified_claims"]["f1"])
        report = evaluate_extractions([self.annotated()], [empty], self.schema)
        self.assertEqual(report["metrics"]["qualified_claims"]["false_negative"], 1)
        self.assertEqual(report["metrics"]["qualified_claims"]["recall"], 0)

    def test_missing_and_extra_predictions_fail(self):
        for predictions in ([], [self.result, self.result], [dict(self.result, source_id="other")]):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                evaluate_extractions([self.annotated()], predictions, self.schema)

    def test_unreviewed_sources_excluded_explicitly(self):
        source = replace(self.source, source_id="unreviewed")
        unreviewed = {"source": source.to_dict(), "annotation_status": "unreviewed", "result": None}
        prediction = make_result(source, self.schema, extractor_name="test", extractor_revision="1", method="fixture")
        report = evaluate_extractions([self.annotated(), unreviewed], [self.result, prediction], self.schema)
        self.assertEqual(report["reviewed_sources"], 1)
        self.assertEqual(report["predictions_on_unreviewed_sources_excluded"], 1)

    def test_repeated_relation_occurrences_not_hidden_by_set_scoring(self):
        prediction = deepcopy(self.result)
        prediction["claims"].append(dict(prediction["claims"][0], id="extra",
                                         evidence=[self.span(0, len(self.source.text))]))
        report = evaluate_extractions([self.annotated()], [prediction], self.schema)
        self.assertEqual(report["metrics"]["qualified_claims"]["false_positive"], 1)

    def test_evaluate_cli_writes_reports_without_editing_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = {"schema.json": self.schema, "annotations.json": [self.annotated()],
                        "predictions.json": [self.result]}
            for name, payload in payloads.items():
                (root / name).write_text(json.dumps(payload))
            original = (root / "annotations.json").read_bytes()
            with unittest.mock.patch("builtins.print"):
                self.assertEqual(main(["evaluate", "--annotations", str(root / "annotations.json"),
                    "--schema", str(root / "schema.json"), "--predictions", str(root / "predictions.json"),
                    "--output", str(root / "report")]), 0)
            report = strict_json_load(root / "report/evaluate.json")
            self.assertEqual(report["metrics"]["qualified_claims"]["f1"], 1.0)
            self.assertTrue((root / "report/evaluate.txt").exists())
            self.assertEqual((root / "annotations.json").read_bytes(), original)

    def test_optional_roles_and_multiple_evidence_spans(self):
        schema = deepcopy(self.schema)
        schema["relations"]["affected_version"]["roles"]["other_product"] = {
            "types": ["SOFTWARE"], "required": False}
        result = deepcopy(self.result)
        result["schema_sha256"] = fingerprint(schema)
        result["entities"].append({"id": "other", "type": "SOFTWARE",
                                   "span": self.span(36, 40), "score": None})
        result["claims"][0]["arguments"]["other_product"] = ["other"]
        result["claims"][0]["evidence"].append(self.span(25, len(self.source.text)))
        validate_extraction(result, self.source, schema)

    def test_duplicate_annotation_sources_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate source_id"):
            validate_annotations([self.annotated(), self.annotated()], self.schema)


class BenchmarkPreparation(unittest.TestCase):
    def test_preparation_is_deterministic_label_free_and_non_destructive(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            source.write_text('[{"cve":"CVE-2","description":"Same text","epss_score":NaN,"in_vckev":true},'
                              '{"cve":"CVE-1","description":"same  text","binary_label":1},'
                              '{"cve":"CVE-3","description":"Different text"}]')
            original = source.read_bytes()
            records, manifest = prepare_annotations(source, limit=10)
            self.assertEqual(prepare_annotations(source, limit=10)[0], records)
            self.assertEqual(manifest["selected_records"], 2)
            self.assertEqual(manifest["nonstandard_constants_replaced_in_memory"], {"NaN": 1})
            self.assertTrue(all(r["result"] is None and r["annotation_status"] == "unreviewed" for r in records))
            self.assertTrue(all(r["source"]["available_at"] is None for r in records))
            self.assertNotIn("epss_score", json.dumps(records))
            self.assertNotIn("in_vckev", json.dumps(records))
            output = Path(directory) / "annotations"
            with unittest.mock.patch("builtins.print"):
                self.assertEqual(main(["prepare", "--source", str(source), "--output", str(output)]), 0)
                validation = Path(directory) / "validation"
                self.assertEqual(main(["validate", "--annotations", str(output / "annotations.json"),
                                       "--schema", str(output / "schema.json"), "--output", str(validation)]), 0)
            self.assertEqual(strict_json_load(validation / "validate.json")["complete"], 0)
            self.assertEqual(source.read_bytes(), original)
            with unittest.mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                main(["prepare", "--source", str(source), "--output", str(output)])

    def test_reject_duplicate_source_ids_and_invalid_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            source.write_text('[{"cve":"CVE-1","description":"Text"},{"cve":"CVE-1","description":"Other"}]')
            with self.assertRaises(ValueError):
                prepare_annotations(source)
            for limit in (0, -1, True):
                with self.assertRaises(ValueError):
                    prepare_annotations(source, limit=limit)


if __name__ == "__main__":
    unittest.main()
