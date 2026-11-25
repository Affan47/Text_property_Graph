"""Offline preparation, validation and evaluation of extraction annotations.

Run with python -m tpg.extraction_benchmark --help. No model is loaded.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from tpg.extraction import (SourceDocument, _unique_json_object, fingerprint,
                            load_security_schema, strict_json_load, validate_schema)
from tpg.extraction_evaluation import evaluate_extractions, validate_annotations


def prepare_annotations(source_path, *, limit=300, seed=42):
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    raw = Path(source_path).read_bytes()
    constants = Counter()

    def missing(value):
        constants[value] += 1
        return None

    # The documented revision source has NaN values outside the selected text
    # fields. Count and replace them only in memory; derived files are strict JSON.
    rows = json.loads(raw, object_pairs_hook=_unique_json_object, parse_constant=missing)
    if not isinstance(rows, list):
        raise ValueError("Expected the revision dataset JSON array")
    groups, seen = {}, set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every source record must be an object")
        cve = row.get("cve")
        if not isinstance(cve, str) or not cve.strip() or cve in seen:
            raise ValueError(f"Missing or duplicate CVE ID: {cve!r}")
        seen.add(cve)
        source = SourceDocument(source_id=f"description:{cve}", record_id=cve,
                                source_text_type="description", text=row.get("description"))
        group_id = hashlib.sha256(" ".join(source.text.casefold().split()).encode("utf-8")).hexdigest()
        groups.setdefault(group_id, []).append(source)
    candidates = [(key, min(values, key=lambda s: s.source_id)) for key, values in sorted(groups.items())]
    selected = random.Random(seed).sample(candidates, min(limit, len(candidates)))
    annotations = [{"source": source.to_dict(), "text_group_id": group,
                    "annotation_status": "unreviewed", "result": None}
                   for group, source in sorted(selected, key=lambda pair: pair[1].source_id)]
    manifest = {
        "purpose": "Unreviewed description sample for extraction annotation; not a training/evaluation result",
        "source_path": str(Path(source_path).resolve()), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "input_records": len(rows), "unique_normalized_description_groups": len(groups),
        "nonstandard_constants_replaced_in_memory": dict(constants),
        "requested_records": limit, "selected_records": len(annotations), "seed": seed,
        "sampling": "One lexicographically first CVE per casefolded/whitespace-normalized text group, then seeded sampling",
        "source_fields_copied": ["cve", "description"],
        "outcome_labels_copied": False, "availability_known": False,
        "notes": ["Original text is preserved exactly; normalization is used only for grouping.",
                  "available_at is null: current snapshot text is not proven available at original publication.",
                  "No split or gold labels are generated. Human review is required.",
                  "This deduplicated sample is not prevalence-weighted or a complete near-duplicate audit.",
                  "Group related CVEs, advisories and future SMP records before assigning splits."]}
    return annotations, manifest


def _write(output_dir, files, inputs):
    output_dir = Path(output_dir)
    targets = {output_dir / name: content for name, content in files.items()}
    protected = {Path(path).resolve() for path in inputs if path is not None}
    for target in targets:
        if target.resolve() in protected or target.exists():
            raise ValueError(f"Refusing to overwrite input or existing artifact: {target}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for target, content in targets.items():
        with target.open("x", encoding="utf-8") as handle:
            handle.write(content)


def _json(value):
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Sample descriptions without outcome labels")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--limit", type=int, default=300)
    prepare.add_argument("--seed", type=int, default=42)
    for name in ("validate", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--annotations", type=Path, required=True)
        command.add_argument("--schema", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            records, report = prepare_annotations(args.source, limit=args.limit, seed=args.seed)
            schema = load_security_schema()
            validate_annotations(records, schema)
            report["schema_sha256"] = fingerprint(schema)
            _write(args.output, {"annotations.json": _json(records), "schema.json": _json(schema),
                                "preparation.json": _json(report), "preparation.txt": _json(report)}, [args.source])
        else:
            schema = validate_schema(strict_json_load(args.schema))
            annotations = strict_json_load(args.annotations)
            inputs = [args.schema, args.annotations]
            if args.command == "validate":
                records = validate_annotations(annotations, schema)
                report = {"scope": "Structural/source-alignment validation only, not semantic correctness",
                          "records": len(records), "complete": sum(s == "complete" for _, s, _ in records),
                          "unreviewed": sum(s == "unreviewed" for _, s, _ in records)}
            else:
                inputs.append(args.predictions)
                report = evaluate_extractions(annotations, strict_json_load(args.predictions), schema)
            report["schema_sha256"] = fingerprint(schema)
            report["input_sha256"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
            _write(args.output, {f"{args.command}.json": _json(report), f"{args.command}.txt": _json(report)}, inputs)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
