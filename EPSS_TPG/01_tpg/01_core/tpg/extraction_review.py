"""Offline human-review packets, current-rule baseline and paired evaluation.

No automatic conversion of model suggestions into human annotations. No training.
"""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path

from tpg.extraction import (SourceDocument, _object, fingerprint, make_result,
                            parse_timestamp, strict_json_load, validate_extraction, validate_schema)
from tpg.extraction_benchmark import _json, _write
from tpg.extraction_evaluation import evaluate_extractions, validate_annotations
from tpg.extraction_predict import predict_records


def prediction_map(predictions, annotations, schema):
    sources = {s.source_id: s for s, _, _ in validate_annotations(annotations, schema)}
    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a list")
    result = {}
    for candidate in predictions:
        if (not isinstance(candidate, dict) or not isinstance(candidate.get("source_id"), str)
                or candidate["source_id"] not in sources):
            raise ValueError("Prediction has an unknown source")
        key = candidate["source_id"]
        if key in result:
            raise ValueError(f"Duplicate prediction: {key}")
        result[key] = validate_extraction(candidate, sources[key], schema)
    return result


def prepare_review(annotations, predictions, schema):
    candidates = prediction_map(predictions, annotations, schema)
    if not candidates:
        raise ValueError("No successful predictions to review")
    records = []
    for source, status, _ in validate_annotations(annotations, schema):
        if source.source_id not in candidates:
            continue
        if status == "complete":
            raise ValueError(f"Already reviewed: {source.source_id}; use a separate adjudication workflow")
        records.append({"source": source.to_dict(), "source_sha256": fingerprint(source.to_dict()),
                        "status": "pending", "reviewer": "", "reviewed_at": None,
                        "full_source_reviewed": False, "entities": [], "claims": [], "notes": ""})
    return {"packet_version": "1.0", "schema_sha256": fingerprint(schema),
            "annotations_sha256": fingerprint(annotations), "records": records}


def finalize_review(packet, annotations, schema):
    _object(packet, {"packet_version", "schema_sha256", "annotations_sha256", "records"}, set(), "packet")
    if packet["packet_version"] != "1.0" or packet["schema_sha256"] != fingerprint(schema):
        raise ValueError("Review packet version or schema mismatch")
    if packet["annotations_sha256"] != fingerprint(annotations):
        raise ValueError("Original annotations changed; reconcile the review packet explicitly")
    validate_annotations(annotations, schema)
    if not isinstance(packet["records"], list):
        raise ValueError("Review records must be a list")
    output = deepcopy(annotations)
    indexed = {r["source"]["source_id"]: r for r in output}
    seen, completed = set(), []
    for record in packet["records"]:
        _object(record, {"source", "source_sha256", "status", "reviewer", "reviewed_at",
                         "full_source_reviewed", "entities", "claims", "notes"}, set(), "review")
        source = SourceDocument.from_dict(record["source"])
        key = source.source_id
        if key in seen or key not in indexed:
            raise ValueError("Duplicate or unknown review source")
        seen.add(key)
        original_source = SourceDocument.from_dict(indexed[key]["source"])
        if record["source_sha256"] != fingerprint(source.to_dict()) or source != original_source:
            raise ValueError("Review source text or metadata changed")
        if indexed[key]["annotation_status"] != "unreviewed":
            raise ValueError("Cannot overwrite a completed annotation")
        if type(record["full_source_reviewed"]) is not bool or not isinstance(record["notes"], str):
            raise ValueError("Review confirmation must be boolean and notes must be text")
        if record["status"] == "pending":
            if (record["entities"] != [] or record["claims"] != [] or record["reviewer"] != ""
                    or record["reviewed_at"] is not None or record["full_source_reviewed"]):
                raise ValueError("Pending records must remain blank; keep unfinished drafts separate")
            continue
        if record["status"] != "complete" or not record["full_source_reviewed"]:
            raise ValueError("Completion requires explicit full-source review, including missed claims")
        if not isinstance(record["reviewer"], str) or not record["reviewer"].strip():
            raise ValueError("A named reviewer is required")
        parse_timestamp(record["reviewed_at"], "reviewed_at")
        if not record["entities"] and not record["claims"] and not record["notes"].strip():
            raise ValueError("A reviewed empty result requires an explanatory note")
        result = make_result(source, schema, extractor_name=record["reviewer"],
                             extractor_revision="review:" + fingerprint(record), method="human")
        result["entities"], result["claims"] = record["entities"], record["claims"]
        result = validate_extraction(result, source, schema)
        indexed[key].update(annotation_status="complete", result=result)
        completed.append(key)
    if not completed:
        raise ValueError("No completed human reviews; nothing can be promoted to gold")
    validate_annotations(output, schema)
    return output, {"newly_reviewed_sources": completed, "review_packet_sha256": fingerprint(packet),
                    "review_is_declared_not_authenticated": True}


def compare_systems(annotations, named_predictions, schema):
    if len(named_predictions) < 2:
        raise ValueError("At least two named systems are required")
    scores = {name: evaluate_extractions(annotations, predictions, schema)
              for name, predictions in named_predictions.items()}
    reviewed = [s.source_id for s, status, _ in validate_annotations(annotations, schema) if status == "complete"]
    per_source = {}
    for record in annotations:
        key = record["source"]["source_id"]
        if key not in reviewed:
            continue
        per_source[key] = {name: evaluate_extractions([record], [r for r in predictions if r["source_id"] == key], schema)["metrics"]
                           for name, predictions in named_predictions.items()}
    return {"scope": "Paired exact-span comparison on the same completed human annotations",
            "reviewed_source_ids": reviewed, "systems": scores, "per_source": per_source,
            "notes": ["No intersection-based source dropping: missing reviewed predictions fail the comparison.",
                      "All schema relations are scored, including relations a baseline cannot extract.",
                      "Compare per-relation results before interpreting aggregate scores.",
                      "Whole-window and sentence-level evidence differ; exact evidence scores are not entailment scores.",
                      "Reviewing only supplied candidates would miss false negatives; full-source review is required.",
                      "This is not a held-out benchmark or evidence of statistical significance by itself."]}


def review_text(packet, schema, suggestions=None):
    lines = ["SECURITY EXTRACTION REVIEW", "", "No source in this packet is marked reviewed automatically.",
             "Annotate every supported mention/claim, not just the supplied suggestions.",
             "Offsets count Python Unicode characters: start included, end excluded.",
             "The index at the left is the start offset of that exact 80-character slice.",
             "Adjacent slices join without extra spaces; displayed JSON escapes are not source characters.",
             "Use the full source string in review.json to check exact quotations.", "",
             "Entity definitions:", _json(schema["entity_types"]), "Relation definitions:", _json(schema["relations"])]
    for record in packet["records"]:
        source = record["source"]
        lines.extend(["", source["source_id"], "Source text:", source["text"], "Indexed source slices:"])
        for start in range(0, len(source["text"]), 80):
            lines.append(f"{start:6d}: " + _json(source["text"][start:start + 80]).strip())
        if suggestions is not None:
            lines.extend(["UNREVIEWED MODEL SUGGESTIONS (may bias annotation):", _json(suggestions[source["source_id"]])])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "baseline", "finalize", "compare"):
        command = commands.add_parser(name)
        command.add_argument("--annotations", type=Path, required=True)
        command.add_argument("--schema", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name in {"prepare", "baseline"}:
            command.add_argument("--predictions", type=Path, required=True,
                                 help="Validated predictions select the explicit source cohort")
        if name == "prepare":
            command.add_argument("--show-suggestions", action="store_true")
        if name == "finalize":
            command.add_argument("--review", type=Path, required=True)
        if name == "compare":
            command.add_argument("--system", action="append", required=True, metavar="NAME=PATH")
    args = parser.parse_args(argv)
    input_fingerprints = {}

    def read(path):
        value = strict_json_load(path)
        input_fingerprints[str(Path(path).resolve())] = fingerprint(value)
        return value

    try:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("Output directory exists; choose a new directory")
        annotations, schema = read(args.annotations), validate_schema(read(args.schema))
        records = validate_annotations(annotations, schema)
        inputs = [args.annotations, args.schema]
        files, exit_code = {}, 0
        if args.command in {"prepare", "baseline"}:
            inputs.append(args.predictions)
            predictions = read(args.predictions)
            mapping = prediction_map(predictions, annotations, schema)
            if not mapping:
                raise ValueError("Prediction file selects no sources")
        if args.command == "prepare":
            packet = prepare_review(annotations, predictions, schema)
            files = {"review.json": _json(packet), "review_guide.txt": review_text(packet, schema, mapping if args.show_suggestions else None)}
            report = {"selected_sources": len(packet["records"]), "completed_reviews": 0,
                      "suggestions_shown": args.show_suggestions,
                      "selection": "Explicit successful-prediction cohort; not a representative benchmark",
                      "instructions": "Edit review.json after full-source human review, then run finalize into a separate directory."}
        elif args.command == "baseline":
            from tpg.rule_extraction import RuleClaimExtractor
            backend = RuleClaimExtractor()
            selected = [(s, status, r) for s, status, r in records if s.source_id in mapping]
            predictions, audits, statuses = predict_records(selected, schema, backend)
            files = {"predictions.json": _json(predictions), "audit.json": _json(audits)}
            report = {"provenance": backend.provenance, "records": statuses,
                      "status_counts": dict(Counter(r["status"] for r in statuses)),
                      "selected_source_ids": [s.source_id for s, _, _ in selected],
                      "limitations": "Current rules with audited semantic mapping, not the historical pre-fix code. Unsupported claims remain absent, not reinterpreted."}
            exit_code = int(any(s["status"] == "failed" for s in statuses))
        elif args.command == "finalize":
            inputs.append(args.review)
            packet = read(args.review)
            reviewed, report = finalize_review(packet, annotations, schema)
            files = {"annotations.json": _json(reviewed), "review_record.json": _json(packet)}
        else:
            systems = {}
            for value in args.system:
                name, sep, path = value.partition("=")
                if not sep or not name.strip() or not path or name in systems:
                    raise ValueError("Each --system must have a unique nonempty NAME=PATH")
                inputs.append(Path(path))
                systems[name] = read(path)
            report = compare_systems(annotations, systems, schema)
        report["input_fingerprints"] = input_fingerprints
        report["schema_sha256"] = fingerprint(schema)
        files.update({"report.json": _json(report), "report.txt": _json(report)})
        _write(args.output, files, inputs)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.error(str(error))
    print(_json({"command": args.command, "output": str(args.output)}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
