"""Generate separate, unreviewed learned-extraction suggestions. Never trains a model."""
import argparse
from collections import Counter
from copy import deepcopy
from importlib.metadata import PackageNotFoundError
from pathlib import Path
import sys

from tpg.extraction import (extract_validated, fingerprint, require_available,
                            strict_json_load, validate_schema)
from tpg.extraction_benchmark import _json, _write
from tpg.extraction_evaluation import validate_annotations
from tpg.learned_extraction import ExtractionSettings, GLiNERClaimExtractor, _file_hash, load_gliner


def predict_records(records, schema, extractor, *, cutoff=None):
    """Failures are explicit and never masquerade as empty successful predictions."""
    predictions, audits, statuses = [], [], []
    for source, _, _ in records:
        try:
            result = extract_validated(extractor, source, schema, cutoff=cutoff)
            predictions.append(result)
            rejected = len(extractor.last_audit["rejections"])
            status = {"source_id": source.source_id,
                      "status": "completed_with_rejections" if rejected else "completed",
                      "entities": len(result["entities"]), "claims": len(result["claims"]),
                      "rejected_candidates": rejected}
        except Exception as error:
            status = {"source_id": source.source_id, "status": "failed",
                      "error_type": type(error).__name__, "error": str(error)}
        audit = deepcopy(extractor.last_audit)
        if audit is not None and audit.get("source_id") == source.source_id:
            audits.append(audit)
        statuses.append(status)
        print(f"[{len(statuses)}/{len(records)}] {source.source_id}: {status['status']}", file=sys.stderr)
    return predictions, audits, statuses


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory, separate from human annotations")
    parser.add_argument("--model", required=True, help="Local checkpoint directory or Hugging Face model ID")
    parser.add_argument("--revision", help="Required immutable Hub commit SHA; omit for local checkpoints")
    parser.add_argument("--allow-download", action="store_true", help="Explicitly permit downloading model files only")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--window-words", type=int, default=128)
    parser.add_argument("--overlap-words", type=int, default=32)
    parser.add_argument("--limit", type=int, help="First N input records for an explicitly limited smoke test")
    parser.add_argument("--cutoff", help="Require documented source availability at/before this ISO timestamp")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("Output directory already exists; choose a new directory to preserve previous artifacts")
        if args.threads < 1 or (args.limit is not None and args.limit < 1):
            raise ValueError("threads and limit must be positive")
        settings = ExtractionSettings(args.threshold, args.max_input_tokens, args.window_words, args.overlap_words)
        schema = validate_schema(strict_json_load(args.schema))
        records = validate_annotations(strict_json_load(args.annotations), schema)
        total = len(records)
        if args.limit is not None:
            records = records[:args.limit]
        if not records:
            raise ValueError("No source records selected")
        if args.cutoff is not None:
            for source, _, _ in records:
                require_available(source, args.cutoff)
        import torch
        torch.set_num_threads(args.threads)
        engine, identity = load_gliner(args.model, revision=args.revision,
                                       allow_download=args.allow_download, device=args.device, seed=args.seed)
        extractor = GLiNERClaimExtractor(engine, identity, settings)
        predictions, audits, statuses = predict_records(records, schema, extractor, cutoff=args.cutoff)
        counts = dict(Counter(s["status"] for s in statuses))
        report = {"purpose": "Unreviewed extraction suggestions, not gold labels or prediction-model results",
                  "input_records": total, "selected_records": len(records), "status_counts": counts,
                  "cutoff": args.cutoff, "schema_sha256": fingerprint(schema),
                  "extractor_sha256": extractor.revision, "provenance": extractor.provenance,
                  "threads": args.threads,
                  "inputs_sha256": {str(p.resolve()): _file_hash(p) for p in (args.annotations, args.schema)},
                  "source_sha256": {s.source_id: fingerprint(s.to_dict()) for s, _, _ in records},
                  "records": statuses,
                  "limitations": ["All suggestions require review; confidence is not calibrated probability.",
                                  "Evidence is the entire model input window, not a selected supporting phrase.",
                                  "Claims spanning different windows can be missed; overlap does not guarantee recall.",
                                  "Independent entity matching can reject correct role spans with different boundaries.",
                                  "No annotations, graph edges, outcome labels or predictive models were changed.",
                                  "A seed and hashes aid reproducibility but do not ensure cross-device bitwise identity."]}
        _write(args.output, {"predictions.json": _json(predictions), "audit.json": _json(audits),
                            "prediction_report.json": _json(report), "prediction_report.txt": _json(report)},
               [args.annotations, args.schema])
    except (ValueError, OSError, ImportError, PackageNotFoundError, RuntimeError) as error:
        parser.error(str(error))
    print(_json({"output": str(args.output), "status_counts": counts}))
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
