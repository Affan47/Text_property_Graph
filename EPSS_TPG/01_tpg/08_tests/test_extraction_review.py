"""Review safety and semantic baseline projection, using synthetic fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from tpg.extraction import SourceDocument, load_security_schema, make_result
from tpg.extraction_review import (compare_systems, finalize_review, main,
                                   prediction_map, prepare_review, review_text)
from tpg.rule_extraction import project_graph
from tpg.schema.graph import TextPropertyGraph
from tpg.schema.types import EdgeProperties, NodeProperties, NodeType, SecurityEdgeType


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.schema = load_security_schema()
        self.source = SourceDocument("fixture:1", "Acme 2.4 is affected.", "description")
        self.annotations = [{"source": self.source.to_dict(), "annotation_status": "unreviewed", "result": None}]
        self.prediction = make_result(self.source, self.schema, extractor_name="fixture-model",
                                      extractor_revision="1", method="model")
        self.prediction["entities"] = [{"id": "p", "type": "SOFTWARE", "span": {"start": 0, "end": 4, "text": "Acme"}, "score": 0.9}]
        self.packet = prepare_review(self.annotations, [self.prediction], self.schema)

    def complete_empty(self):
        packet = deepcopy(self.packet)
        packet["records"][0].update(status="complete", reviewer="fixture-reviewer", reviewed_at="2026-09-23T12:00:00Z",
                                     full_source_reviewed=True, notes="Synthetic test of a deliberate empty review, not an actual annotation.")
        return packet

    def test_prepare_is_blank_and_does_not_promote_suggestions(self):
        record = self.packet["records"][0]
        self.assertEqual(record["entities"], [])
        self.assertFalse(record["full_source_reviewed"])
        self.assertEqual(record["status"], "pending")
        self.assertIsNone(self.annotations[0]["result"])
        self.assertNotIn("fixture-model", review_text(self.packet, self.schema))

    def test_pending_cannot_finalize(self):
        with self.assertRaisesRegex(ValueError, "No completed"):
            finalize_review(self.packet, self.annotations, self.schema)

    def test_full_source_review_confirmation_required(self):
        packet = self.complete_empty()
        packet["records"][0]["full_source_reviewed"] = False
        with self.assertRaisesRegex(ValueError, "full-source"):
            finalize_review(packet, self.annotations, self.schema)

    def test_reviewer_and_timezone_required(self):
        for field, value in (("reviewer", " "), ("reviewed_at", "2026-09-23"), ("full_source_reviewed", 1)):
            packet = self.complete_empty()
            packet["records"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                finalize_review(packet, self.annotations, self.schema)

    def test_empty_review_requires_reason(self):
        packet = self.complete_empty()
        packet["records"][0]["notes"] = ""
        with self.assertRaisesRegex(ValueError, "explanatory note"):
            finalize_review(packet, self.annotations, self.schema)

    def test_finalization_copies_and_keeps_unselected_sources(self):
        source = SourceDocument("fixture:2", "Another source", "description")
        annotations = self.annotations + [{"source": source.to_dict(), "annotation_status": "unreviewed", "result": None}]
        packet = prepare_review(annotations, [self.prediction], self.schema)
        packet["records"] = self.complete_empty()["records"]
        result, report = finalize_review(packet, annotations, self.schema)
        self.assertEqual(result[0]["annotation_status"], "complete")
        self.assertEqual(result[0]["result"]["extractor"]["method"], "human")
        self.assertEqual(result[1], annotations[1])
        self.assertEqual(annotations[0]["annotation_status"], "unreviewed")
        self.assertEqual(report["newly_reviewed_sources"], ["fixture:1"])

    def test_source_and_schema_tampering_fail(self):
        for field in ("source", "schema", "base", "duplicate"):
            packet = self.complete_empty()
            if field == "source":
                packet["records"][0]["source"]["text"] = "Other text"
            elif field == "schema":
                packet["schema_sha256"] = "changed"
            elif field == "base":
                packet["annotations_sha256"] = "changed"
            else:
                packet["records"] *= 2
            with self.subTest(field=field), self.assertRaises(ValueError):
                finalize_review(packet, self.annotations, self.schema)

    def test_bad_reviewed_span_rejected(self):
        packet = self.complete_empty()
        packet["records"][0]["entities"] = deepcopy(self.prediction["entities"])
        packet["records"][0]["entities"][0]["span"]["text"] = "invented"
        with self.assertRaises(ValueError):
            finalize_review(packet, self.annotations, self.schema)

    def test_pending_cannot_hide_partial_annotations(self):
        packet = deepcopy(self.packet)
        packet["records"][0]["entities"] = self.prediction["entities"]
        with self.assertRaisesRegex(ValueError, "Pending"):
            finalize_review(packet, self.annotations, self.schema)

    def test_compare_requires_real_reviewed_cohort(self):
        with self.assertRaisesRegex(ValueError, "No complete human"):
            compare_systems(self.annotations, {"a": [self.prediction], "b": [self.prediction]}, self.schema)

    def test_compare_fails_missing_predictions_not_intersection(self):
        annotations, _ = finalize_review(self.complete_empty(), self.annotations, self.schema)
        with self.assertRaisesRegex(ValueError, "Missing predictions"):
            compare_systems(annotations, {"a": [self.prediction], "b": []}, self.schema)

    def test_compare_same_cohort_and_per_source_metrics(self):
        annotations, _ = finalize_review(self.complete_empty(), self.annotations, self.schema)
        empty = dict(self.prediction, entities=[])
        report = compare_systems(annotations, {"a": [self.prediction], "b": [empty]}, self.schema)
        self.assertEqual(report["reviewed_source_ids"], ["fixture:1"])
        self.assertEqual(report["systems"]["a"]["metrics"]["entities"]["false_positive"], 1)
        self.assertIsNone(report["systems"]["b"]["metrics"]["entities"]["precision"])
        self.assertIn("fixture:1", report["per_source"])

    def test_duplicate_predictions_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
            prediction_map([self.prediction] * 2, self.annotations, self.schema)

    def test_reviewed_sources_cannot_be_overwritten(self):
        annotations, _ = finalize_review(self.complete_empty(), self.annotations, self.schema)
        with self.assertRaisesRegex(ValueError, "Already reviewed"):
            prepare_review(annotations, [self.prediction], self.schema)

    def test_prepare_cli_preserves_input_and_blinds_suggestions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in (("a", self.annotations), ("s", self.schema), ("p", [self.prediction])):
                (root / (name + ".json")).write_text(json.dumps(value))
            before = (root / "a.json").read_bytes()
            args = ["prepare", "--annotations", str(root / "a.json"), "--schema", str(root / "s.json"),
                    "--predictions", str(root / "p.json"), "--output", str(root / "out")]
            self.assertEqual(main(args), 0)
            self.assertEqual((root / "a.json").read_bytes(), before)
            self.assertNotIn("fixture-model", (root / "out/review_guide.txt").read_text())
            with self.assertRaises(SystemExit):
                main(args)


class RuleProjectionTests(unittest.TestCase):
    def setUp(self):
        self.schema = load_security_schema()
        self.source = SourceDocument("fixture", "CVE-2024-1234 affects Acme 2.4.", "description")
        self.graph = TextPropertyGraph()
        self.graph.metadata["source_text"] = self.source.text
        self.nodes = []
        for kind, text in (("CVE_ID", "CVE-2024-1234"), ("SOFTWARE", "Acme"), ("VERSION", "2.4")):
            start = self.source.text.index(text)
            end = start + len(text)
            self.nodes.append(self.graph.add_node(NodeType.ENTITY, NodeProperties(
                text=text, entity_type=kind, char_start=start, char_end=end)))
        self.evidence = {"char_start": 0, "char_end": len(self.source.text), "text": self.source.text,
                         "polarity": "negated", "modality": "possible"}

    def test_affects_preserves_qualifiers(self):
        self.graph.add_edge(self.nodes[0], self.nodes[1], SecurityEdgeType.AFFECTS, EdgeProperties(extra=self.evidence))
        result, audit = project_graph(self.graph, self.source, self.schema, "fixture")
        self.assertEqual(result["claims"][0]["relation"], "affects")
        self.assertEqual(result["claims"][0]["polarity"], "negated")
        self.assertEqual(result["claims"][0]["modality"], "possible")
        self.assertEqual(audit["rejections"], [])

    def test_has_version_never_becomes_affected_or_fixed(self):
        self.graph.add_edge(self.nodes[1], self.nodes[2], SecurityEdgeType.HAS_VERSION, EdgeProperties(extra=self.evidence))
        result, audit = project_graph(self.graph, self.source, self.schema, "fixture")
        self.assertEqual(result["claims"], [])
        self.assertEqual(audit["omitted_edges"][0]["edge_type"], "HAS_VERSION")

    def test_unspecified_qualifiers_remain_unresolved(self):
        self.evidence.update(polarity="unspecified", modality="unspecified")
        self.graph.add_edge(self.nodes[0], self.nodes[1], SecurityEdgeType.AFFECTS, EdgeProperties(extra=self.evidence))
        result, _ = project_graph(self.graph, self.source, self.schema, "fixture")
        self.assertEqual(result["claims"][0]["polarity"], "unresolved")

    def test_wrong_source_fails(self):
        self.graph.metadata["source_text"] = "changed"
        with self.assertRaisesRegex(ValueError, "source text"):
            project_graph(self.graph, self.source, self.schema, "fixture")

    def test_bad_evidence_rejected_and_audited(self):
        self.evidence["text"] = "invented"
        self.graph.add_edge(self.nodes[0], self.nodes[1], SecurityEdgeType.AFFECTS, EdgeProperties(extra=self.evidence))
        result, audit = project_graph(self.graph, self.source, self.schema, "fixture")
        self.assertEqual(result["claims"], [])
        self.assertEqual(len(audit["rejections"]), 1)


if __name__ == "__main__":
    unittest.main()
