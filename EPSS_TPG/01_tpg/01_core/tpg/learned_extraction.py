"""Opt-in GLiNER2 adapter. No model imports or downloads at module import time.

Schema definitions specify what to extract; learned weights propose the spans
and role bindings. Returned claims are unreviewed suggestions, not graph facts.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import math
from pathlib import Path
import re

from tpg.extraction import (MODALITIES, POLARITIES, fingerprint, make_result,
                            strict_json_load, validate_extraction, validate_schema)


GLINER_VERSION = "2.0.0"
ADAPTER_VERSION = "1.0"


@dataclass(frozen=True)
class ExtractionSettings:
    threshold: float = 0.5
    max_input_tokens: int = 512
    window_words: int = 128
    overlap_words: int = 32

    def __post_init__(self):
        if type(self.threshold) not in (int, float) or not math.isfinite(self.threshold) or not 0 < self.threshold <= 1:
            raise ValueError("threshold must be finite and in (0, 1]")
        for key in ("max_input_tokens", "window_words", "overlap_words"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} must be an integer")
        if self.max_input_tokens < 1 or self.window_words < 1:
            raise ValueError("Token and window limits must be positive")
        if not 0 <= self.overlap_words < self.window_words:
            raise ValueError("overlap_words must be nonnegative and smaller than window_words")


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _no_custom_code(value):
    if isinstance(value, dict):
        if value.get("auto_map"):
            raise ValueError("Custom model/tokenizer code is not supported")
        for child in value.values():
            _no_custom_code(child)
    elif isinstance(value, list):
        for child in value:
            _no_custom_code(child)


def load_gliner(model, *, revision=None, allow_download=False, device="cpu", seed=42):
    """Load a local safe checkpoint or an explicitly pinned Hub snapshot.

Resolve the whole snapshot first: upstream tokenizer loading does not forward
the model revision, so passing a Hub ID directly would mix file revisions.
"""
    if version("gliner2") != GLINER_VERSION:
        raise ValueError(f"This adapter requires gliner2=={GLINER_VERSION}")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    path = Path(model).expanduser()
    identity = {"requested_model": str(model), "revision": revision}
    if path.is_dir():
        if revision is not None:
            raise ValueError("Local checkpoints use file hashes; omit --revision")
    else:
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Hub models require an immutable 40-character commit SHA as --revision")
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(
            str(model), revision=revision, local_files_only=not allow_download,
            allow_patterns=["*.json", "encoder_config/*.json", "model.safetensors", "spm.model"],
        ))
    required = ["config.json", "encoder_config/config.json", "tokenizer_config.json",
                "tokenizer.json", "model.safetensors"]
    for name in required:
        if not (path / name).is_file():
            raise ValueError(f"Checkpoint is incomplete: missing {name}")
    config = strict_json_load(path / "config.json")
    if config.get("architecture", "span") != "span":
        raise ValueError("This adapter supports GLiNER2 span checkpoints, not boundary checkpoints")
    artifacts = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in {".json", ".model", ".safetensors"})
    for artifact in artifacts:
        if artifact.suffix == ".json":
            _no_custom_code(strict_json_load(artifact))
    hashes = {str(p.relative_to(path)): _file_hash(p) for p in artifacts}
    import torch
    from gliner2 import GLiNER2
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    engine = GLiNER2.from_pretrained(str(path.resolve()), map_location=device, use_flashdeberta=False)
    engine.eval()
    identity.update({"adapter_version": ADAPTER_VERSION, "checkpoint_files_sha256": hashes,
                     "checkpoint_sha256": fingerprint(hashes), "device": device, "seed": seed,
                     "packages": {name: version(name) for name in
                                  ("gliner2", "torch", "transformers", "huggingface_hub", "safetensors", "peft")}})
    return engine, identity


def _span(value, text):
    if not isinstance(value, dict) or not {"start", "end", "text", "confidence"} <= value.keys():
        raise ValueError("Expected a model span with offsets, text and confidence")
    start, end, score = value["start"], value["end"], value["confidence"]
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
        raise ValueError("Invalid source offsets")
    if text[start:end] != value["text"]:
        raise ValueError("Model span does not exactly quote the source")
    if type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid model confidence")
    return {"start": start, "end": end, "text": value["text"]}, score


def _choice(value, allowed, threshold):
    if value is None:
        return "unresolved"
    if not isinstance(value, dict) or value.get("text") not in allowed:
        raise ValueError("Invalid qualifier output")
    score = value.get("confidence")
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid qualifier confidence")
    return value["text"] if score >= threshold else "unresolved"


class GLiNERClaimExtractor:
    """Independent typed mentions plus model-bound, possibly n-ary records.

No Cartesian products, fuzzy offset repair, CVE metadata injection, or inferred
affirmation. A whole input window supplies context, not an entailment proof.
"""

    def __init__(self, engine, identity, settings=None):
        self.engine = engine
        self.identity = deepcopy(identity)
        self.settings = settings or ExtractionSettings()
        encoder_limit = engine.encoder.config.max_position_embeddings
        if self.settings.max_input_tokens > encoder_limit:
            raise ValueError(f"max_input_tokens exceeds the encoder limit of {encoder_limit}")
        self.provenance = {"backend": "gliner2", "adapter_version": ADAPTER_VERSION,
                           "model": self.identity, "settings": asdict(self.settings),
                           "evidence_policy": "whole input window; not a model-selected rationale",
                           "claim_score_policy": "null; no joint claim score is supplied by this backend"}
        self.revision = fingerprint(self.provenance)
        self.last_audit = None

    def _schemas(self, schema):
        if "entities" in schema["relations"]:
            raise ValueError("Relation name entities is reserved by this adapter")
        if any(name in {"polarity", "modality"} for r in schema["relations"].values() for name in r["roles"]):
            raise ValueError("Role names polarity and modality are reserved by this adapter")
        plans = [("entities", self.engine.create_schema().entities(schema["entity_types"]))]
        for name, relation in schema["relations"].items():
            builder = self.engine.create_schema().structure(name)
            for role, spec in relation["roles"].items():
                definitions = "; ".join(schema["entity_types"][t] for t in spec["types"])
                builder.field(role, dtype="list", description=(
                    f"{relation['description']} Participant role: {role}. {definitions} "
                    "Use only explicit spans participating in this particular claim; otherwise leave empty."))
            builder.field("polarity", dtype="str", choices=sorted(POLARITIES), description=(
                "For this claim: affirmed if expressed, negated if explicitly denied, "
                "unresolved if unclear. Do not equate mention with affirmation."))
            builder.field("modality", dtype="str", choices=sorted(MODALITIES), description=(
                "For this claim: asserted, possible, conditional, recommended, or unresolved. "
                "Preserve uncertainty and conditions in the text."))
            # build finalizes the pending structure; retain the Schema for its field metadata.
            builder.build()
            plans.append((name, builder.schema))
        return plans

    def _fits(self, text, plans):
        prepared_text = text if text.endswith((".", "!", "?")) else text + "."
        self.engine.processor.change_mode(is_training=False)
        for _, plan in plans:
            record = self.engine.processor.transform_and_format(prepared_text, plan.build())
            if len(record.input_ids) > self.settings.max_input_tokens:
                return False
        return True

    def _windows(self, text, plans):
        words = list(re.finditer(r"\S+", text))
        start = 0
        while start < len(words):
            stop = min(len(words), start + self.settings.window_words)
            left = 0 if start == 0 else words[start].start()
            while True:
                right = len(text) if stop == len(words) else words[stop].start()
                if self._fits(text[left:right], plans):
                    break
                if stop == start + 1:
                    raise ValueError("Schema plus one source word exceeds the input budget; no text was silently truncated")
                stop = start + max(1, (stop - start) // 2)
            yield left, right
            if stop == len(words):
                break
            start = stop - min(self.settings.overlap_words, stop - start - 1)

    def extract(self, source, schema):
        schema = validate_schema(schema)
        self.last_audit = {"source_id": source.source_id, "windows": [], "rejections": [],
                           "duplicate_entities": 0, "duplicate_claims": 0}
        plans = self._schemas(schema)
        result = make_result(source, schema, extractor_name="gliner2-security-claims",
                             extractor_revision=self.revision, method="model")
        entities, claims = {}, set()
        for left, right in self._windows(source.text, plans):
            text = source.text[left:right]
            audit = {"start": left, "end": right, "outputs": {}}
            self.last_audit["windows"].append(audit)
            local_entities = []
            for task, plan in plans:
                raw = self.engine.extract(text, plan, threshold=self.settings.threshold,
                                          include_spans=True, include_confidence=True, max_len=None)
                audit["outputs"][task] = raw
                if not isinstance(raw, dict):
                    raise ValueError(f"Unexpected backend output for {task}")
                if task == "entities":
                    if set(raw) != {"entities"} or not isinstance(raw["entities"], dict):
                        raise ValueError("Backend entity output has an incompatible format")
                    for kind, values in raw["entities"].items():
                        if kind not in schema["entity_types"] or not isinstance(values, list):
                            raise ValueError("Backend returned an unknown entity type or invalid entity list")
                        for value in values:
                            try:
                                span, score = _span(value, text)
                                if score < self.settings.threshold:
                                    raise ValueError("Entity score below threshold")
                            except ValueError as error:
                                self._reject(task, left, value, error)
                                continue
                            span.update(start=span["start"] + left, end=span["end"] + left)
                            key = (kind, span["start"], span["end"])
                            if key not in entities:
                                entity = {"id": f"e{len(entities) + 1}", "type": kind, "span": span, "score": score}
                                entities[key] = entity
                                result["entities"].append(entity)
                            else:
                                entities[key]["score"] = max(score, entities[key]["score"])
                                self.last_audit["duplicate_entities"] += 1
                            local_entities.append(entities[key])
                    continue
                # GLiNER2 2.0.0 uses {} when its record-count head predicts zero.
                if set(raw) == {task} and raw[task] == {}:
                    continue
                if set(raw) != {task} or not isinstance(raw[task], list):
                    raise ValueError(f"Backend record output has an incompatible format: {task}")
                for record in raw[task]:
                    try:
                        claim = self._claim(task, record, schema, text, left, right, local_entities)
                    except ValueError as error:
                        self._reject(task, left, record, error)
                        continue
                    # Overlap can repeat a claim with a different context window.
                    key = fingerprint({k: claim[k] for k in ("relation", "arguments", "polarity", "modality")})
                    if key in claims:
                        self.last_audit["duplicate_claims"] += 1
                        continue
                    claims.add(key)
                    claim["id"] = f"c{len(result['claims']) + 1}"
                    result["claims"].append(claim)
        return validate_extraction(result, source, schema)

    def _reject(self, task, offset, value, error):
        self.last_audit["rejections"].append({"task": task, "window_start": offset,
                                                "candidate": value, "reason": str(error)})

    def _claim(self, task, record, schema, text, left, right, entities):
        roles = schema["relations"][task]["roles"]
        if not isinstance(record, dict) or set(record) - set(roles) - {"polarity", "modality"}:
            raise ValueError("Record has unknown fields or is not an object")
        arguments = {}
        for role, spec in roles.items():
            values = record.get(role, [])
            if not isinstance(values, list):
                raise ValueError(f"Role {role} is not a list")
            if spec["required"] and not values:
                raise ValueError(f"Required role {role} is empty")
            references = []
            for value in values:
                span, score = _span(value, text)
                if score < self.settings.threshold:
                    raise ValueError(f"Role {role} score below threshold")
                matches = {e["id"] for e in entities if e["type"] in spec["types"]
                           and e["span"]["start"] == left + span["start"]
                           and e["span"]["end"] == left + span["end"]}
                if len(matches) != 1:
                    raise ValueError(f"Role {role} has no unique independently typed entity match")
                references.extend(matches)
            if references:
                arguments[role] = sorted(set(references))
        return {"relation": task, "arguments": arguments,
                "evidence": [{"start": left, "end": right, "text": text}],
                "polarity": _choice(record.get("polarity"), POLARITIES, self.settings.threshold),
                "modality": _choice(record.get("modality"), MODALITIES, self.settings.threshold), "score": None}
