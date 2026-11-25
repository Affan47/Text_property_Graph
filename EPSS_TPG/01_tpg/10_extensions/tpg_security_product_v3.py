"""Opt-in v3: distinguish standalone address cues from compound product names.

Only version-candidate filtering changes. V2 ownership, bounds, abstentions,
graph rebuilding and validation are reused without modifying frozen modules.
"""
import re

import tpg_security_product_v2 as v2
from tpg.pipeline import TPGPipeline
from tpg.passes.security_relations import SecurityRelationsPass
from tpg.schema.types import NodeType, EdgeType, SecurityEdgeType, SECURITY_SCHEMA

VERSION = 'security_product_version_v3'
# A word boundary also matches after '-' in BIG-IP. An address cue must not
# be the suffix of a compound name, path or dotted identifier.
ADDRESS_CONTEXT_RE = re.compile(r'(?<![\w./-])(?:ip|ip address|address)\s*[:=]?\s*$', re.I)


def version_mentions(text):
    for match in v2.VERSION_RE.finditer(text):
        # Match address cues against the original text with an end position;
        # clipping a prefix could invent a boundary inside a compound name.
        prefix = text[max(0, match.start()-65):match.start()]
        suffix = text[match.end():match.end()+32]
        if v2.v1.SCORE_RE.search(prefix) or re.match(r'\s*/\s*10\b', suffix):
            continue
        if ADDRESS_CONTEXT_RE.search(text, 0, match.start()):
            continue
        yield match


class ProductVersionFrontend(v2.ProductVersionFrontend):
    def _extract_versions(self, text, graph):
        graph.metadata['security_extraction_version'] = VERSION
        for match in version_mentions(text):
            start, end = match.span('number')
            nid = self._add_security_entity(graph, text[start:end], 'VERSION', 'software_version',
                                            start, end, confidence=.8)
            node = graph.get_node(nid)
            sentence = self._find_sentence_node(start, graph)
            owner, reason = v2.owner_candidate(text, sentence.properties.char_start, match) if sentence else (None, 'no_sentence')
            node.properties.extra.update(version_literal=match.group(), version_literal_span=list(match.span()),
                release_form=v2.release_form(match), ownership_candidate=owner,
                ownership_abstention=reason, extraction_policy=VERSION)

    def _extract_software(self, text, graph):
        super()._extract_software(text, graph)
        for node in graph.nodes(NodeType.ENTITY):
            if node.properties.entity_type == 'SOFTWARE':
                node.properties.extra['extraction_policy'] = VERSION


class ProductVersionSecurityPipeline(v2.ProductVersionSecurityPipeline):
    def __init__(self, include_security_relations=True):
        TPGPipeline.__init__(self, frontend=ProductVersionFrontend(schema=SECURITY_SCHEMA), schema=SECURITY_SCHEMA)
        self.include_security_relations = include_security_relations
        if include_security_relations:
            self.passes.append(SecurityRelationsPass())

    def run(self, text, doc_id=''):
        graph = super().run(text, doc_id)
        for edge in graph.edges():
            if edge.edge_type == SecurityEdgeType.HAS_VERSION or (
                    edge.edge_type == EdgeType.ENTITY_REL and edge.properties.entity_rel_type == 'HAS_VERSION'):
                edge.properties.extra['extraction_policy'] = VERSION
        graph.mark_pass(VERSION)
        return graph
