"""Offline adapter tests; scripted outputs test plumbing, not model accuracy."""
from copy import deepcopy
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tpg.extraction import SourceDocument, extract_validated, load_security_schema, strict_json_load
from tpg.extraction_predict import main, predict_records
from tpg.learned_extraction import (ExtractionSettings, GLiNERClaimExtractor, _choice,
                                    _no_custom_code, _span, load_gliner)


class Plan:
    def __init__(self):
        self.schema = self
        self.kind = None
        self.fields = {}

    def entities(self, types):
        self.kind = "entities"
        self.types = types
        return self

    def structure(self, name):
        self.kind = name
        return self

    def field(self, name, **kwargs):
        self.fields[name] = kwargs
        return self

    def build(self):
        return self


class ScriptedEngine:
    """Deterministic fixture, deliberately not presented as a learned model."""
    def __init__(self):
        self.encoder = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512))
        self.processor = SimpleNamespace(change_mode=lambda **kw: None,
            transform_and_format=lambda text, schema: SimpleNamespace(input_ids=list(text)))
        self.calls = []
        self.mutate = lambda task, raw: raw
        self.fail = False

    def create_schema(self):
        return Plan()

    def extract(self, text, plan, **kwargs):
        self.calls.append((text, plan, kwargs))
        if self.fail:
            raise RuntimeError("Fixture inference failure")

        def spans(term):
            return [{"text": m.group(), "start": m.start(), "end": m.end(), "confidence": 0.9}
                    for m in re.finditer(re.escape(term), text)]

        if plan.kind == "entities":
            raw = {"entities": {"SOFTWARE": spans("Acme"), "VERSION": spans("2.4")}}
        elif plan.kind == "affected_version" and spans("Acme") and spans("2.4"):
            raw = {plan.kind: [{"product": spans("Acme")[:1], "version": spans("2.4")[:1],
                               "polarity": {"text": "negated", "confidence": 0.9},
                               "modality": {"text": "asserted", "confidence": 0.9}}]}
        else:
            raw = {plan.kind: []}
        return self.mutate(plan.kind, raw)


class LearnedAdapterTests(unittest.TestCase):
    def setUp(self):
        self.schema = load_security_schema()
        self.source = SourceDocument("post:1", "Acme 2.4 is not affected.", "social_media_post", "CVE-fixture")
        self.engine = ScriptedEngine()
        self.adapter = GLiNERClaimExtractor(self.engine, {"fixture": True})

    def test_source_bound_claim_retains_negation_roles_and_context(self):
        result = extract_validated(self.adapter, self.source, self.schema)
        self.assertEqual(len(result["entities"]), 2)
        claim = result["claims"][0]
        self.assertEqual(claim["polarity"], "negated")
        self.assertEqual(claim["arguments"], {"product": ["e1"], "version": ["e2"]})
        self.assertEqual(claim["evidence"][0]["text"], self.source.text)
        self.assertIsNone(claim["score"])
        self.assertFalse(any(e["type"] == "CVE_ID" for e in result["entities"]))
        self.assertEqual(result["extractor"]["method"], "model")
        for _, _, kwargs in self.engine.calls:
            self.assertIsNone(kwargs["max_len"])
            self.assertTrue(kwargs["include_spans"])

    def test_definitions_and_roles_reach_model_schema(self):
        plans = dict(self.adapter._schemas(self.schema))
        self.assertEqual(plans["entities"].types, self.schema["entity_types"])
        self.assertIn(self.schema["relations"]["affects"]["description"], plans["affects"].fields["product"]["description"])
        self.assertEqual(plans["affects"].fields["polarity"]["choices"], ["affirmed", "negated", "unresolved"])

    def test_missing_qualifiers_abstain(self):
        def mutate(task, raw):
            if task == "affected_version":
                raw[task][0].pop("polarity")
                raw[task][0].pop("modality")
            return raw
        self.engine.mutate = mutate
        claim = self.adapter.extract(self.source, self.schema)["claims"][0]
        self.assertEqual((claim["polarity"], claim["modality"]), ("unresolved", "unresolved"))

    def test_bad_entity_span_rejected_not_repaired(self):
        def mutate(task, raw):
            if task == "entities":
                raw["entities"]["SOFTWARE"][0]["text"] = "invented"
            return raw
        self.engine.mutate = mutate
        result = self.adapter.extract(self.source, self.schema)
        self.assertEqual(result["claims"], [])
        self.assertEqual(len(self.adapter.last_audit["rejections"]), 2)

    def test_required_role_missing_is_rejected(self):
        def mutate(task, raw):
            if task == "affected_version":
                raw[task][0]["version"] = []
            return raw
        self.engine.mutate = mutate
        self.assertEqual(self.adapter.extract(self.source, self.schema)["claims"], [])
        self.assertIn("Required role", self.adapter.last_audit["rejections"][0]["reason"])

    def test_ambiguous_type_not_chosen_arbitrarily(self):
        def mutate(task, raw):
            if task == "entities":
                raw["entities"]["VERSION_RANGE"] = deepcopy(raw["entities"]["VERSION"])
            return raw
        self.engine.mutate = mutate
        self.assertEqual(self.adapter.extract(self.source, self.schema)["claims"], [])

    def test_unknown_backend_shape_fails_document(self):
        self.engine.mutate = lambda task, raw: {"wrong": []}
        with self.assertRaisesRegex(ValueError, "incompatible format"):
            self.adapter.extract(self.source, self.schema)

    def test_upstream_empty_record_dictionary_is_abstention(self):
        self.engine.mutate = lambda task, raw: raw if task == "entities" else {task: {}}
        self.assertEqual(self.adapter.extract(self.source, self.schema)["claims"], [])
        self.assertEqual(self.adapter.last_audit["rejections"], [])

    def test_duplicate_records_deduplicated(self):
        def mutate(task, raw):
            if task == "affected_version":
                raw[task] *= 2
            return raw
        self.engine.mutate = mutate
        self.assertEqual(len(self.adapter.extract(self.source, self.schema)["claims"]), 1)
        self.assertEqual(self.adapter.last_audit["duplicate_claims"], 1)

    def test_nary_roles_kept_in_one_record(self):
        schema = deepcopy(self.schema)
        schema["relations"]["affected_version"]["roles"]["context_product"] = {"types": ["SOFTWARE"], "required": True}
        def mutate(task, raw):
            if task == "affected_version":
                raw[task][0]["context_product"] = deepcopy(raw[task][0]["product"])
            return raw
        self.engine.mutate = mutate
        claim = self.adapter.extract(self.source, schema)["claims"][0]
        self.assertEqual(len(claim["arguments"]), 3)

    def test_separate_records_not_cartesian_product(self):
        def mutate(task, raw):
            if task == "affected_version":
                record = deepcopy(raw[task][0])
                record["polarity"]["text"] = "affirmed"
                raw[task].append(record)
            return raw
        self.engine.mutate = mutate
        result = self.adapter.extract(self.source, self.schema)
        self.assertEqual({c["polarity"] for c in result["claims"]}, {"negated", "affirmed"})

    def test_unicode_and_window_offset_mapping(self):
        source = SourceDocument("unicode", "\u0130 Intro text. Acme 2.4 is not affected. More text.", "description")
        adapter = GLiNERClaimExtractor(self.engine, {}, ExtractionSettings(window_words=6, overlap_words=3))
        result = extract_validated(adapter, source, self.schema)
        self.assertGreater(len(adapter.last_audit["windows"]), 1)
        for entity in result["entities"]:
            span = entity["span"]
            self.assertEqual(source.text[span["start"]:span["end"]], span["text"])
        self.assertEqual(len(result["entities"]), 2)
        self.assertEqual(len(result["claims"]), 1)

    def test_window_budget_shrinks_without_losing_text(self):
        adapter = GLiNERClaimExtractor(self.engine, {}, ExtractionSettings(max_input_tokens=16, window_words=8, overlap_words=2))
        windows = list(adapter._windows(self.source.text, adapter._schemas(self.schema)))
        self.assertEqual(windows[0][0], 0)
        self.assertEqual(windows[-1][1], len(self.source.text))
        self.assertTrue(all(b[0] <= a[1] for a, b in zip(windows, windows[1:])))
        self.assertTrue(all(right > left for left, right in windows))

    def test_schema_that_does_not_fit_fails_before_inference(self):
        adapter = GLiNERClaimExtractor(self.engine, {}, ExtractionSettings(max_input_tokens=1))
        with self.assertRaisesRegex(ValueError, "silently truncated"):
            adapter.extract(self.source, self.schema)
        self.assertEqual(self.engine.calls, [])

    def test_cutoff_checked_before_inference(self):
        with self.assertRaisesRegex(ValueError, "availability is unknown"):
            extract_validated(self.adapter, self.source, self.schema, cutoff="2026-01-01T00:00:00Z")
        self.assertEqual(self.engine.calls, [])

    def test_failures_never_become_empty_predictions(self):
        self.engine.fail = True
        predictions, audits, statuses = predict_records([(self.source, "unreviewed", None)], self.schema, self.adapter)
        self.assertEqual(predictions, [])
        self.assertEqual(statuses[0]["status"], "failed")
        self.assertEqual(audits[0]["source_id"], self.source.source_id)

    def test_low_qualifier_confidence_unresolved(self):
        self.assertEqual(_choice({"text": "affirmed", "confidence": 0.1}, {"affirmed"}, 0.5), "unresolved")

    def test_invalid_span_scores_and_offsets(self):
        for changes in ({"confidence": float("nan")}, {"confidence": True}, {"start": True}, {"end": 999}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                _span(dict(text="Acme", start=0, end=4, confidence=0.9) | changes, self.source.text)

    def test_invalid_settings_and_encoder_limit(self):
        for settings in ({"threshold": 0}, {"threshold": float("nan")}, {"window_words": True},
                         {"overlap_words": 128}, {"max_input_tokens": -1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ExtractionSettings(**settings)
        with self.assertRaisesRegex(ValueError, "encoder limit"):
            GLiNERClaimExtractor(self.engine, {}, ExtractionSettings(max_input_tokens=1024))

    def test_revision_changes_with_settings(self):
        other = GLiNERClaimExtractor(self.engine, {"fixture": True}, ExtractionSettings(threshold=0.7))
        self.assertNotEqual(self.adapter.revision, other.revision)

    def test_hub_requires_immutable_revision_before_download(self):
        with patch("tpg.learned_extraction.version", return_value="2.0.0"):
            with self.assertRaisesRegex(ValueError, "immutable"):
                load_gliner("example/nonexistent", revision="main")

    def test_local_checkpoint_requires_safetensors(self):
        with tempfile.TemporaryDirectory() as directory, patch("tpg.learned_extraction.version", return_value="2.0.0"):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_gliner(directory)

    def test_custom_model_code_rejected(self):
        with self.assertRaisesRegex(ValueError, "Custom"):
            _no_custom_code({"nested": [{"auto_map": {"AutoModel": "custom.py"}}]})

    def test_cli_preserves_annotations_and_writes_separate_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations, schema = root / "annotations.json", root / "schema.json"
            annotations.write_text(json.dumps([{"source": self.source.to_dict(), "annotation_status": "unreviewed", "result": None}]))
            schema.write_text(json.dumps(self.schema))
            before = annotations.read_bytes()
            args = ["--annotations", str(annotations), "--schema", str(schema), "--output", str(root / "predictions"), "--model", "fixture"]
            with patch("tpg.extraction_predict.load_gliner", return_value=(self.engine, {"fixture": True})) as loader:
                self.assertEqual(main(args), 0)
                self.assertEqual(annotations.read_bytes(), before)
                report = strict_json_load(root / "predictions/prediction_report.json")
                self.assertEqual(report["status_counts"], {"completed": 1})
                self.assertFalse(loader.call_args.kwargs["allow_download"])
                with self.assertRaises(SystemExit):
                    main(args)
                self.assertEqual(loader.call_count, 1)

    def test_cli_nonzero_on_inference_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.json").write_text(json.dumps([{"source": self.source.to_dict(), "annotation_status": "unreviewed", "result": None}]))
            (root / "s.json").write_text(json.dumps(self.schema))
            self.engine.fail = True
            with patch("tpg.extraction_predict.load_gliner", return_value=(self.engine, {})):
                result = main(["--annotations", str(root / "a.json"), "--schema", str(root / "s.json"),
                               "--output", str(root / "out"), "--model", "fixture"])
            self.assertEqual(result, 1)
            self.assertEqual(strict_json_load(root / "out/predictions.json"), [])


if __name__ == "__main__":
    unittest.main()
