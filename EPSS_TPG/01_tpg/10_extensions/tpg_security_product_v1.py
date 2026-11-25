"""Opt-in, source-bound product/version extraction; frozen core is unchanged.

This narrow rule correction is not a learned extractor or a new graph schema.
Unresolved version mentions remain explicit; proximity alone is not ownership.
"""
from copy import deepcopy
import re

from tpg.contracts import attach_provenance, sentence_evidence
from tpg.frontends.security_frontend import SecurityFrontend, _SOFTWARE_NAMES
from tpg.pipeline import TPGPipeline
from tpg.passes.security_relations import SecurityRelationsPass
from tpg.schema.graph import TextPropertyGraph
from tpg.schema.types import NodeType, EdgeType, SecurityEdgeType, EdgeProperties, SECURITY_SCHEMA

VERSION = 'security_product_version_v1'
VERSION_RE = re.compile(
    r'(?<![\w./-])(?:v\.?)?(?P<number>\d+\.\d+(?:\.\d+)*'
    r'(?:-[a-z0-9]+(?:[.-][a-z0-9]+)*)?(?:\+[a-z0-9]+(?:[.-][a-z0-9]+)*)?)'
    r'(?![\w*]|\.[\w*])', re.I)
BRIDGE_RE = re.compile(
    r'(?P<bridge>\s*(?::\s*)?(?:versions?\s+)?(?:from\s+n/a\s+)?'
    r'(?P<bound>before|prior to|through|up to and including|up to|at least|<=|>=|<|>)?\s*)$', re.I)
WORD = r'[A-Z][A-Za-z0-9_+.-]*'
COMPOUND = r'[A-Za-z][A-Za-z0-9_]*(?:[-./][A-Za-z0-9_+]+)+'
PRODUCT_RE = re.compile(
    rf'(?<![\w./-])(?P<product>(?:{WORD}|{COMPOUND})'
    rf'(?:\s+(?:{WORD}|{COMPOUND}|for|of|the|and|plugin|theme|server|package|\|)){{0,14}})$')
SCORE_RE = re.compile(r'(?:cvss(?:\s+v?\d+(?:\.\d+)?)?(?:\s+base)?\s*(?:score)?|base score|severity score)\s*[:=]?\s*$', re.I)
OPERATORS = {'before':'<', 'prior to':'<', 'through':'<=', 'up to and including':'<=',
             'up to':'upper_bound_unspecified_inclusivity', 'at least':'>=', '<':'<', '<=':'<=', '>':'>', '>=':'>='}


def version_mentions(text):
    """Retain full numeric spans and literal v/v. prefix separately."""
    for match in VERSION_RE.finditer(text):
        prefix, suffix = text[max(0,match.start()-65):match.start()], text[match.end():match.end()+32]
        if SCORE_RE.search(prefix) or re.match(r'\s*/\s*10\b', suffix):
            continue
        if re.search(r'\b(?:ip|ip address|address)\s*[:=]?\s*$', prefix, re.I):
            continue
        yield match


def owner_candidate(text, sentence_start, match):
    """Recognize an adjacent named product, not the nearest SOFTWARE anywhere."""
    prefix = text[sentence_start:match.start()]
    bridge = BRIDGE_RE.search(prefix)
    if bridge is None:
        return None, 'unsupported_version_context'
    end = sentence_start + bridge.start()
    left = text[sentence_start:end]
    found = PRODUCT_RE.search(left)
    candidates = []
    if found:
        start = sentence_start + found.start('product')
        surface = text[start:end]
        article = re.match(r'(?:The|A|An)\s+',surface)
        if article:
            start += article.end()
        candidates.append((start,end))
    for name in sorted(_SOFTWARE_NAMES,key=lambda s:(-len(s),s)):
        found = re.search(r'(?<![\w./-])'+re.escape(name)+r'$',left,re.I)
        if found:
            candidates.append((sentence_start+found.start(),end))
    if not candidates:
        return None, 'no_adjacent_named_product'
    start,end = min(candidates,key=lambda span:(span[0],-span[1]))
    surface = text[start:end]
    # Never narrow a coordinated owner to the final name, or treat a path as a product.
    if re.search(r'\b(?:and|or)\b',surface,re.I) or re.search(r'\b(?:and|or)\s*$',text[sentence_start:start],re.I):
        previous_versions = list(version_mentions(text[sentence_start:start]))
        if re.search(r'\b(?:and|or)\b',surface,re.I) or not previous_versions:
            return None, 'coordinated_owner_requires_review'
    if (not surface or surface.lower() in {'version','versions','score','cvss','ip','address','release','releases'}
            or re.search(r'\.(?:php|py|js|exe|dll|so|c|h)$',surface,re.I)
            or '://' in surface or surface.startswith('/') or surface.endswith('.')):
        return None, 'not_a_supported_product_name'
    bound = (bridge.group('bound') or '').lower()
    operator = OPERATORS.get(bound,'mentioned_version')
    after = re.match(r'\s+and\s+(prior|earlier|later|above)\b',text[match.end():],re.I)
    qualifier_end = match.end()
    if after:
        trailing = '<=' if after.group(1).lower() in {'prior','earlier'} else '>='
        if operator != 'mentioned_version' and operator != trailing:
            return None, 'conflicting_version_bounds'
        operator = trailing
        qualifier_end += after.end()
    return {'owner_span':[start,end], 'operator':operator,
            'qualifier_span':[end,qualifier_end],
            'qualifier_text':text[end:qualifier_end]}, None


class ProductVersionFrontend(SecurityFrontend):
    def _extract_versions(self,text,graph):
        graph.metadata['security_extraction_version'] = VERSION
        for match in version_mentions(text):
            start,end = match.span('number')
            nid = self._add_security_entity(graph,text[start:end],'VERSION','software_version',start,end,confidence=.8)
            node = graph.get_node(nid)
            sentence = self._find_sentence_node(start,graph)
            owner,reason = owner_candidate(text,sentence.properties.char_start,match) if sentence else (None,'no_sentence')
            node.properties.extra.update(version_literal=match.group(),version_literal_span=list(match.span()),
                                         ownership_candidate=owner,ownership_abstention=reason,
                                         extraction_policy=VERSION)

    def _extract_software(self,text,graph):
        spans = {}
        for node in graph.nodes(NodeType.ENTITY):
            if node.properties.entity_type != 'VERSION':
                continue
            owner = node.properties.extra.get('ownership_candidate')
            if owner:
                spans[tuple(owner['owner_span'])] = .8
        # Explicit product phrases take precedence over embedded platform names.
        for name in sorted(_SOFTWARE_NAMES,key=lambda s:(-len(s),s)):
            for m in re.finditer(r'(?<![\w./-])'+re.escape(name)+r'(?![\w./-])',text,re.I):
                span = m.span()
                if not any(span[0]<b and a<span[1] for a,b in spans):
                    spans[span] = .9
        for (start,end),confidence in sorted(spans.items()):
            nid = self._add_security_entity(graph,text[start:end],'SOFTWARE','software_product',start,end,confidence)
            graph.get_node(nid).properties.extra['extraction_policy'] = VERSION


def replace_version_edges(graph,typed=True):
    """Rebuild through public APIs, removing both old HAS_VERSION copies."""
    result = TextPropertyGraph(schema=graph.schema,doc_id=graph.doc_id)
    result.metadata = deepcopy(graph.metadata)
    for node in graph.nodes():
        if result.add_node(node.node_type,deepcopy(node.properties)) != node.id:
            raise ValueError('Non-contiguous graph IDs; refuse implicit remapping')
    for edge in graph.edges():
        if edge.edge_type == SecurityEdgeType.HAS_VERSION or (
                edge.edge_type == EdgeType.ENTITY_REL and edge.properties.entity_rel_type == 'HAS_VERSION'):
            continue
        result.add_edge(edge.source,edge.target,edge.edge_type,deepcopy(edge.properties),allow_duplicate=True)
    for name in graph.passes_applied:
        result.mark_pass(name)
    software = {tuple((n.properties.char_start,n.properties.char_end)):n
                for n in result.nodes(NodeType.ENTITY) if n.properties.entity_type=='SOFTWARE'}
    for target in result.nodes(NodeType.ENTITY):
        if target.properties.entity_type != 'VERSION':
            continue
        p = target.properties
        owner = p.extra.get('ownership_candidate')
        if owner is None:
            continue
        source = software.get(tuple(owner['owner_span']))
        evidence = sentence_evidence(result,source,target) if source else None
        if evidence is None:
            p.extra.update(ownership_candidate=None,ownership_abstention='owner_not_in_same_sentence')
            continue
        evidence.update(method='explicit_adjacent_product_version',relation_kind='attribute',
                        polarity='unspecified',modality='unspecified',extraction_policy=VERSION,
                        version_operator=owner['operator'],qualifier_span=owner['qualifier_span'],
                        qualifier_text=owner['qualifier_text'],
                        interpretation='Named product/version mention; not a claim of affected or fixed status.')
        for kind in ([EdgeType.ENTITY_REL,SecurityEdgeType.HAS_VERSION] if typed else [EdgeType.ENTITY_REL]):
            result.add_edge(source.id,target.id,kind,EdgeProperties(entity_rel_type='HAS_VERSION',extra=deepcopy(evidence)))
    result.mark_pass(VERSION)
    return attach_provenance(result)


class ProductVersionSecurityPipeline(TPGPipeline):
    """Explicit opt-in. Does not replace SecurityPipeline or alter cached graphs."""
    def __init__(self,include_security_relations=True):
        super().__init__(frontend=ProductVersionFrontend(schema=SECURITY_SCHEMA),schema=SECURITY_SCHEMA)
        self.include_security_relations = include_security_relations
        if include_security_relations:
            self.passes.append(SecurityRelationsPass())

    def run(self,text,doc_id=''):
        graph = replace_version_edges(super().run(text,doc_id),self.include_security_relations)
        for node in graph.nodes():
            if node.node_type == NodeType.ENTITY and node.properties.source == 'security_frontend':
                p = node.properties
                if text[p.char_start:p.char_end] != p.text:
                    raise ValueError('Security entity does not match its source span')
        issues = graph.validate()
        if issues:
            raise ValueError(f'Invalid product/version graph: {issues[:3]}')
        return graph
