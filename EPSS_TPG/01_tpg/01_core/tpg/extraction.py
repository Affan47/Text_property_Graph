"""Model-independent, source-bound extraction contracts.

Validation checks structure and source alignment, not whether a claim is true
or entailed. Entity and claim labels come from a declarative schema, not rules.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Optional, Protocol


CONTRACT_VERSION = "1.0"
POLARITIES = {"affirmed", "negated", "unresolved"}
MODALITIES = {"asserted", "possible", "conditional", "recommended", "unresolved"}


def _fail(message):
    raise ValueError(message)


def _string(value, path):
    if not isinstance(value, str) or not value.strip():
        _fail(f"{path}: expected a non-empty string")


def _object(value, required, optional, path):
    if not isinstance(value, dict):
        _fail(f"{path}: expected an object")
    if any(not isinstance(key, str) for key in value):
        _fail(f"{path}: object keys must be strings")
    missing = set(required) - value.keys()
    unknown = value.keys() - set(required) - set(optional)
    if missing or unknown:
        _fail(f"{path}: missing fields {sorted(missing)}, unknown fields {sorted(unknown)}")


def _list(value, path, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        _fail(f"{path}: expected {'non-empty ' if nonempty else ''}list")


def _unique_json_object(items):
    result = {}
    for key, value in items:
        if key in result:
            _fail(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object,
                      parse_constant=lambda value: _fail(f"Non-standard JSON constant: {value}"))


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_timestamp(value, path="timestamp"):
    _string(value, path)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{path}: invalid ISO timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(f"{path}: an explicit timezone offset is required")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class SourceDocument:
    source_id: str
    text: str
    source_text_type: str
    record_id: Optional[str] = None
    available_at: Optional[str] = None

    def __post_init__(self):
        for key in ("source_id", "text", "source_text_type"):
            _string(getattr(self, key), f"source.{key}")
        if self.record_id is not None:
            _string(self.record_id, "source.record_id")
        if self.available_at is not None:
            parse_timestamp(self.available_at, "source.available_at")

    @property
    def text_sha256(self):
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        _object(value, {"source_id", "text", "source_text_type"},
                {"record_id", "available_at"}, "source")
        return cls(**value)


def require_available(source, cutoff):
    cutoff_time = parse_timestamp(cutoff, "cutoff")
    if source.available_at is None:
        _fail(f"{source.source_id}: availability is unknown; cannot apply a historical cutoff")
    if parse_timestamp(source.available_at) > cutoff_time:
        _fail(f"{source.source_id}: source was available after the cutoff")


def validate_schema(schema):
    _object(schema, {"schema_id", "version", "entity_types", "relations"}, set(), "schema")
    for key in ("schema_id", "version"):
        _string(schema[key], f"schema.{key}")
    types = schema["entity_types"]
    if not isinstance(types, dict) or not types:
        _fail("schema.entity_types: expected non-empty type/description mapping")
    for name, description in types.items():
        _string(name, "entity type")
        _string(description, f"entity_types.{name}")
    if not isinstance(schema["relations"], dict) or not schema["relations"]:
        _fail("schema.relations: expected a non-empty mapping")
    for name, relation in schema["relations"].items():
        _string(name, "relation name")
        _object(relation, {"description", "roles"}, set(), f"relations.{name}")
        _string(relation["description"], f"relations.{name}.description")
        if not isinstance(relation["roles"], dict) or not relation["roles"]:
            _fail(f"relations.{name}.roles: expected a non-empty mapping")
        for role, spec in relation["roles"].items():
            _string(role, "role name")
            _object(spec, {"types", "required"}, set(), f"{name}.{role}")
            _list(spec["types"], f"{name}.{role}.types", nonempty=True)
            if any(not isinstance(t, str) or t not in types for t in spec["types"]):
                _fail(f"{name}.{role}: undeclared entity type")
            if len(set(spec["types"])) != len(spec["types"]):
                _fail(f"{name}.{role}: duplicate entity types")
            if type(spec["required"]) is not bool:
                _fail(f"{name}.{role}.required: expected boolean")
        if not any(r["required"] for r in relation["roles"].values()):
            _fail(f"{name}: at least one participant role must be required")
    return deepcopy(schema)


def load_security_schema():
    return validate_schema(strict_json_load(Path(__file__).with_name("security_claim_schema.json")))


def make_result(source, schema, *, extractor_name, extractor_revision, method):
    """Create an explicit empty result; this does not mark an annotation reviewed."""
    schema = validate_schema(schema)
    return {
        "contract_version": CONTRACT_VERSION,
        "source_id": source.source_id,
        "source_sha256": fingerprint(source.to_dict()),
        "schema_sha256": fingerprint(schema),
        "extractor": {"name": extractor_name, "revision": extractor_revision, "method": method},
        "entities": [], "claims": [],
    }


def _span(span, source, path):
    _object(span, {"start", "end", "text"}, set(), path)
    start, end = span["start"], span["end"]
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source.text):
        _fail(f"{path}: invalid half-open character offsets")
    if not isinstance(span["text"], str) or source.text[start:end] != span["text"]:
        _fail(f"{path}: span text does not match the original source")


def _score(value, path):
    if value is not None and (type(value) not in (int, float)
                              or not math.isfinite(value) or not 0 <= value <= 1):
        _fail(f"{path}: expected null or a finite score in [0, 1]")


def validate_extraction(result, source, schema):
    schema = validate_schema(schema)
    _object(result, {"contract_version", "source_id", "source_sha256", "schema_sha256",
                     "extractor", "entities", "claims"}, set(), "result")
    if result["contract_version"] != CONTRACT_VERSION:
        _fail("Unsupported extraction contract version")
    if result["source_id"] != source.source_id or result["source_sha256"] != fingerprint(source.to_dict()):
        _fail("Result does not match this source document and its metadata")
    if result["schema_sha256"] != fingerprint(schema):
        _fail("Result does not match this schema definition")
    _object(result["extractor"], {"name", "revision", "method"}, set(), "extractor")
    for key in ("name", "revision", "method"):
        _string(result["extractor"][key], f"extractor.{key}")
    if result["extractor"]["method"] not in {"model", "human", "rule", "fixture"}:
        _fail("Unsupported extraction method")
    _list(result["entities"], "entities")
    _list(result["claims"], "claims")
    entities, entity_keys = {}, set()
    for index, entity in enumerate(result["entities"]):
        path = f"entities[{index}]"
        _object(entity, {"id", "type", "span", "score"}, set(), path)
        _string(entity["id"], path + ".id")
        _string(entity["type"], path + ".type")
        if entity["id"] in entities:
            _fail(f"{path}: duplicate entity ID")
        if entity["type"] not in schema["entity_types"]:
            _fail(f"{path}: undeclared entity type")
        _span(entity["span"], source, path + ".span")
        _score(entity["score"], path + ".score")
        key = (entity["type"], entity["span"]["start"], entity["span"]["end"])
        if key in entity_keys:
            _fail(f"{path}: duplicate typed mention")
        entity_keys.add(key)
        entities[entity["id"]] = entity
    claim_ids, claim_keys = set(), set()
    for index, claim in enumerate(result["claims"]):
        path = f"claims[{index}]"
        _object(claim, {"id", "relation", "arguments", "evidence", "polarity", "modality", "score"},
                set(), path)
        for key in ("id", "relation", "polarity", "modality"):
            _string(claim[key], path + "." + key)
        if claim["id"] in claim_ids:
            _fail(f"{path}: duplicate claim ID")
        claim_ids.add(claim["id"])
        if claim["relation"] not in schema["relations"]:
            _fail(f"{path}: undeclared relation")
        if claim["polarity"] not in POLARITIES or claim["modality"] not in MODALITIES:
            _fail(f"{path}: invalid polarity or modality")
        _score(claim["score"], path + ".score")
        roles = schema["relations"][claim["relation"]]["roles"]
        required = {name for name, spec in roles.items() if spec["required"]}
        _object(claim["arguments"], required, set(roles) - required, path + ".arguments")
        _list(claim["evidence"], path + ".evidence", nonempty=True)
        for span in claim["evidence"]:
            _span(span, source, path + ".evidence")
        evidence_keys = [(s["start"], s["end"]) for s in claim["evidence"]]
        if len(set(evidence_keys)) != len(evidence_keys):
            _fail(f"{path}: duplicate evidence span")
        for role, references in claim["arguments"].items():
            _list(references, path + ".arguments." + role, nonempty=True)
            for reference in references:
                if not isinstance(reference, str) or reference not in entities:
                    _fail(f"{path}: unknown entity reference")
                entity = entities[reference]
                if entity["type"] not in roles[role]["types"]:
                    _fail(f"{path}: incompatible participant type for {role}")
                if not any(s["start"] <= entity["span"]["start"] and entity["span"]["end"] <= s["end"]
                           for s in claim["evidence"]):
                    _fail(f"{path}: participant outside supplied evidence")
            if len(set(references)) != len(references):
                _fail(f"{path}: repeated participant in one role")
        key = fingerprint({"relation": claim["relation"], "polarity": claim["polarity"],
                           "modality": claim["modality"], "evidence": sorted(evidence_keys),
                           "arguments": {k: sorted(v) for k, v in claim["arguments"].items()}})
        if key in claim_keys:
            _fail(f"{path}: duplicate claim")
        claim_keys.add(key)
    return deepcopy(result)


class Extractor(Protocol):
    def extract(self, source: SourceDocument, schema: dict) -> dict:
        """Return contract data; no downstream outcome labels are passed here."""
        ...


def extract_validated(extractor: Extractor, source: SourceDocument, schema: dict, *, cutoff=None):
    schema = validate_schema(schema)
    if cutoff is not None:
        require_available(source, cutoff)
    result = extractor.extract(source, deepcopy(schema))
    return validate_extraction(result, source, schema)
