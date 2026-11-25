"""Expose current security rules in the shared extraction contract, without retraining.

This is an audited semantic projection, not a claim that the legacy graph's
edge ontology is identical to the extraction schema.
"""
from importlib.metadata import version
from pathlib import Path

from tpg.extraction import (MODALITIES, POLARITIES, fingerprint, make_result,
                            validate_extraction, validate_schema)
from tpg.learned_extraction import _file_hash


ENTITY_MAP = {name: name for name in ("CVE_ID", "SOFTWARE", "VERSION", "CWE_ID",
                                     "IMPACT", "REMEDIATION", "SEVERITY")}
ENTITY_MAP["VULN_TYPE"] = "VULNERABILITY"
RELATION_MAP = {
    "AFFECTS": ("affects", "vulnerability", "product"),
    "CLASSIFIED_AS": ("classified_as", "vulnerability", "weakness"),
    "MITIGATED_BY": ("mitigated_by", "vulnerability", "remediation"),
    "HAS_SEVERITY": ("has_severity", "vulnerability", "severity"),
}


def project_graph(graph, source, schema, revision):
    from tpg.schema.types import NodeType, SecurityEdgeType
    schema = validate_schema(schema)
    if graph.metadata.get("source_text") != source.text:
        raise ValueError("Rule graph does not match the source text")
    result = make_result(source, schema, extractor_name="current-security-rules",
                         extractor_revision=revision, method="rule")
    audit = {"source_id": source.source_id, "rejections": [], "omitted_entities": [],
             "omitted_edges": [], "mapped_graph_nodes": {}, "mapped_graph_edges": []}
    typed, node_ids, seen = {}, {}, set()

    def span(start, end):
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source.text):
            raise ValueError("Rule output has invalid source offsets")
        return {"start": start, "end": end, "text": source.text[start:end]}

    for node in graph.nodes(NodeType.ENTITY) + graph.nodes(NodeType.CONCEPT):
        kind = node.properties.entity_type.upper()
        mapped = ENTITY_MAP.get(kind)
        if mapped not in schema["entity_types"]:
            audit["omitted_entities"].append({"node_id": node.id, "type": kind,
                                               "reason": "No declared semantic mapping"})
            continue
        mention = span(node.properties.char_start, node.properties.char_end)
        key = (mapped, mention["start"], mention["end"])
        if key not in typed:
            entity = {"id": f"e{len(typed) + 1}", "type": mapped, "span": mention, "score": None}
            typed[key] = entity["id"]
            result["entities"].append(entity)
        node_ids[node.id] = typed[key]
    audit["mapped_graph_nodes"] = node_ids
    for edge in graph.edges():
        if not isinstance(edge.edge_type, SecurityEdgeType):
            continue
        mapping = RELATION_MAP.get(edge.edge_type.name)
        if mapping is None or mapping[0] not in schema["relations"]:
            audit["omitted_edges"].append({"edge_type": edge.edge_type.name, "source": edge.source,
                "target": edge.target, "reason": "No equivalent claim; HAS_VERSION does not mean affected or fixed"})
            continue
        name, subject, obj = mapping
        extra = edge.properties.extra
        try:
            if edge.source not in node_ids or edge.target not in node_ids:
                raise ValueError("Participant has no mapped entity")
            evidence = span(extra.get("char_start"), extra.get("char_end"))
            if extra.get("text") != evidence["text"]:
                raise ValueError("Rule evidence does not quote the source")
            claim = {"relation": name, "arguments": {subject: [node_ids[edge.source]], obj: [node_ids[edge.target]]},
                     "evidence": [evidence], "polarity": extra.get("polarity", "unresolved"),
                     "modality": extra.get("modality", "unresolved"), "score": None}
            for key, allowed in (("polarity", POLARITIES), ("modality", MODALITIES)):
                if claim[key] == "unspecified":
                    claim[key] = "unresolved"
                if claim[key] not in allowed:
                    raise ValueError(f"Unrecognized rule {key}")
            identity = fingerprint(claim)
            if identity in seen:
                continue
            claim["id"] = f"c{len(result['claims']) + 1}"
            validate_extraction(dict(result, claims=[claim]), source, schema)
            result["claims"].append(claim)
            seen.add(identity)
            audit["mapped_graph_edges"].append({"claim_id": claim["id"], "edge_type": edge.edge_type.name,
                                               "properties": extra})
        except ValueError as error:
            audit["rejections"].append({"edge_type": edge.edge_type.name, "source": edge.source,
                                         "target": edge.target, "reason": str(error)})
    return validate_extraction(result, source, schema), audit


class RuleClaimExtractor:
    def __init__(self):
        from tpg.pipeline import SecurityPipeline
        self.pipeline = SecurityPipeline(include_security_relations=True)
        root = Path(__file__).parent
        self.provenance = {"backend": "current security rules plus spaCy; no transformer embeddings",
                           "entity_mapping": ENTITY_MAP, "relation_mapping": RELATION_MAP,
                           "source_sha256": {str(p.relative_to(root)): _file_hash(p) for p in sorted(root.rglob("*.py"))},
                           "packages": {n: version(n) for n in ("spacy", "en-core-web-sm")},
                           "unsupported_relations": ["affected_version", "fixed_version", "has_impact", "reports_exploitation"]}
        self.revision = fingerprint(self.provenance)
        self.last_audit = None

    def extract(self, source, schema):
        self.last_audit = None
        graph = self.pipeline.run(source.text, doc_id=source.source_id)
        result, self.last_audit = project_graph(graph, source, schema, self.revision)
        return result
