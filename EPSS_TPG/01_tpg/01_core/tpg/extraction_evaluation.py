"""Strict, span-based evaluation independent of local entity/claim IDs."""
from collections import Counter

from tpg.extraction import SourceDocument, _object, validate_extraction, validate_schema


def validate_annotations(records, schema):
    validate_schema(schema)
    if not isinstance(records, list):
        raise ValueError("Annotations must be a list")
    seen, validated = set(), []
    for record in records:
        _object(record, {"source", "annotation_status", "result"},
                {"text_group_id"}, "annotation")
        source = SourceDocument.from_dict(record["source"])
        if source.source_id in seen:
            raise ValueError(f"Duplicate source_id: {source.source_id}")
        seen.add(source.source_id)
        status = record["annotation_status"]
        if status not in ("unreviewed", "complete"):
            raise ValueError("annotation_status must be unreviewed or complete")
        result = record["result"]
        if status == "unreviewed" and result is not None:
            raise ValueError("Unreviewed records must have result=null; store model suggestions separately")
        if status == "complete":
            result = validate_extraction(result, source, schema)
            if result["extractor"]["method"] != "human":
                raise ValueError("Complete gold annotations must identify a human reviewer")
        validated.append((source, status, result))
    return validated


def _metrics(gold, predicted):
    tp = sum((gold & predicted).values())
    fp = sum((predicted - gold).values())
    fn = sum((gold - predicted).values())
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def _keys(result, source_id):
    entities = {e["id"]: (e["type"], e["span"]["start"], e["span"]["end"])
                for e in result["entities"]}
    keys = {name: Counter() for name in ("entities", "relations", "qualified_claims", "evidence_claims")}
    keys["entities"].update((source_id, *e) for e in entities.values())
    for claim in result["claims"]:
        arguments = tuple(sorted((role, tuple(sorted(entities[e] for e in references)))
                                 for role, references in claim["arguments"].items()))
        relation = (source_id, claim["relation"], arguments)
        qualified = (*relation, claim["polarity"], claim["modality"])
        evidence = tuple(sorted((s["start"], s["end"]) for s in claim["evidence"]))
        keys["relations"][relation] += 1
        keys["qualified_claims"][qualified] += 1
        keys["evidence_claims"][(*qualified, evidence)] += 1
    return keys


def evaluate_extractions(annotations, predictions, schema):
    """Missing predictions are errors, not silent abstentions or skipped cases.

    A complete empty result represents no extracted claims. Unreviewed gold
    records never contribute negative examples. Counters penalize repeated
    claims even when their differing evidence spans are valid.
    """
    records = validate_annotations(annotations, schema)
    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a list of extraction results")
    source_map = {source.source_id: source for source, _, _ in records}
    pred_map = {}
    for result in predictions:
        if not isinstance(result, dict) or not isinstance(result.get("source_id"), str):
            raise ValueError("Prediction must identify its source_id")
        source_id = result["source_id"]
        if source_id not in source_map:
            raise ValueError(f"Prediction for unknown source: {source_id}")
        if source_id in pred_map:
            raise ValueError(f"Duplicate prediction: {source_id}")
        pred_map[source_id] = validate_extraction(result, source_map[source_id], schema)
    complete = [(s, r) for s, status, r in records if status == "complete"]
    if not complete:
        raise ValueError("No complete human annotations; unreviewed is not a negative label")
    missing = [s.source_id for s, _ in complete if s.source_id not in pred_map]
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} reviewed sources: {missing[:5]}")
    gold_totals, pred_totals = {}, {}
    for source, gold in complete:
        for totals, result in ((gold_totals, gold), (pred_totals, pred_map[source.source_id])):
            for name, keys in _keys(result, source.source_id).items():
                totals.setdefault(name, Counter()).update(keys)
    metrics = {name: _metrics(gold_totals[name], pred_totals[name]) for name in gold_totals}
    per_relation = {}
    for relation in schema["relations"]:
        subsets = [Counter({k: v for k, v in totals["qualified_claims"].items() if k[1] == relation})
                   for totals in (gold_totals, pred_totals)]
        per_relation[relation] = _metrics(*subsets)
    return {
        "scope": "Exact source-span extraction evaluation, not factual verification or outcome prediction",
        "reviewed_sources": len(complete), "unreviewed_sources_excluded": len(records) - len(complete),
        "predictions_on_unreviewed_sources_excluded": sum(
            s.source_id in pred_map for s, status, _ in records if status == "unreviewed"),
        "metrics": metrics, "per_relation_qualified_claims": per_relation,
        "notes": ["Local annotation IDs are ignored; role, entity type and offsets are compared.",
                  "Relations metric ignores polarity/modality; qualified_claims requires both.",
                  "evidence_claims additionally requires exact evidence offsets, not semantic entailment.",
                  "Undefined metrics are null, not zero or a perfect score.",
                  "All entities participate in entity scoring, including those unused by a claim."]}
