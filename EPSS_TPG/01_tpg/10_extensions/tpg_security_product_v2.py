"""Conservative, opt-in DQ10 correction. No changes to the core or v1 policy."""
import re

import tpg_security_product_v1 as v1
from tpg.pipeline import TPGPipeline
from tpg.passes.security_relations import SecurityRelationsPass
from tpg.schema.types import NodeType, EdgeType, SecurityEdgeType, SECURITY_SCHEMA

VERSION = 'security_product_version_v2'
COMPONENT = r'(?:\d+[x*]?|[x*])'
VERSION_RE = re.compile(
    rf'(?<![\w./-])(?:(?P<product_prefix>[A-Za-z][A-Za-z0-9_]*-)|v\.?)?'
    rf'(?P<number>(?:\d+\.{COMPONENT}(?:\.{COMPONENT})*'
    r'(?:-[a-z0-9]+(?:[.-][a-z0-9]+)*)?(?:\+[a-z0-9]+(?:[.-][a-z0-9]+)*)?'
    r'(?:\s+Patch\s+\d+)?)|(?:build\s+\d+))(?![\w*]|\.[\w*])', re.I)
BRIDGE_RE = re.compile(
    r'\s*(?::\s*)?(?:\(\s*(?:all\s+)?versions?\s+|versions?\s+)?'
    r'(?:from\s+n/a\s+)?'
    r'(?P<bound>before|prior to|through|up to and including|up to|at least|<=|>=|<|>)?\s*$', re.I)
TRAILING_RE = re.compile(
    r'\s+(?:\(\s*and\s+(?P<paren>prior|earlier|below|before|later|above)\s*\)'
    r'|and\s+(?P<plain>prior|earlier|below|before|later|above)\b)', re.I)
GENERIC = {'all', 'any', 'some', 'these', 'those', 'both', 'other', 'affected',
           'vulnerable', 'supported', 'unsupported', 'previous', 'earlier',
           'later', 'current', 'latest', 'new', 'old', 'version', 'versions',
           'release', 'releases', 'software', 'product', 'products', 'package',
           'score', 'cvss', 'ip', 'address'}


def version_mentions(text):
    for match in VERSION_RE.finditer(text):
        prefix = text[max(0, match.start()-65):match.start()]
        suffix = text[match.end():match.end()+32]
        if v1.SCORE_RE.search(prefix) or re.match(r'\s*/\s*10\b', suffix):
            continue
        if re.search(r'\b(?:ip|ip address|address)\s*[:=]?\s*$', prefix, re.I):
            continue
        yield match


def release_form(match):
    if match.group('product_prefix'):
        return 'product_prefixed_release'
    text = match.group('number')
    if re.search(r'[x*]', text.split('-')[0], re.I):
        return 'wildcard_release'
    if text.lower().startswith('build'):
        return 'integer_build'
    if re.search(r'\s+Patch\s+', text, re.I):
        return 'patch_release'
    return 'dotted_release'


def owner_candidate(text, sentence_start, match):
    form = release_form(match)
    # Preserve difficult release mentions without pretending to support their semantics.
    if form in ('product_prefixed_release', 'wildcard_release', 'integer_build'):
        return None, 'unsupported_' + form
    prefix = text[sentence_start:match.start()]
    bridge = BRIDGE_RE.search(prefix)
    if bridge is None:
        return None, 'unsupported_version_context'
    end = sentence_start + bridge.start()
    left = text[sentence_start:end]
    candidates = []
    found = v1.PRODUCT_RE.search(left)
    if found:
        start = sentence_start + found.start('product')
        file_context = re.match(r'\S+\.(?:php|py|js|exe|dll|so|c|h)\s+of\s+', text[start:end], re.I)
        if file_context:
            start += file_context.end()
        modifier = re.match(r'(?:(?:The|A|An)\s+)?(?:(?:package|plugin|product|software)\s+)?', text[start:end])
        start += modifier.end()
        candidates.append((start, end))
    for name in sorted(v1._SOFTWARE_NAMES, key=lambda s: (-len(s), s)):
        found = re.search(r'(?<![\w./-])' + re.escape(name) + r'$', left, re.I)
        if found:
            candidates.append((sentence_start + found.start(), end))
    if not candidates:
        return None, 'no_adjacent_named_product'
    start, end = min(candidates, key=lambda span: (span[0], -span[1]))
    surface = text[start:end]
    if (not surface or surface.lower() in GENERIC or
            re.search(r'\.(?:php|py|js|exe|dll|so|c|h)(?:\s|$)', surface, re.I) or
            '://' in surface or surface.startswith('/') or surface.endswith('.')):
        return None, 'not_a_supported_product_name'
    if surface.lower().split()[0] in GENERIC:
        return None, 'generic_product_prefix_requires_review'
    if re.search(r'\b(?:and|or)\b', surface, re.I) or re.search(r'\b(?:and|or)\s*$', text[sentence_start:start], re.I):
        if re.search(r'\b(?:and|or)\b', surface, re.I) or not list(version_mentions(text[sentence_start:start])):
            return None, 'coordinated_owner_requires_review'
    # Do not repair a parser boundary by joining across sentences. A name beginning
    # at a boundary without punctuation may be only the suffix of the full name.
    preceding = text[:sentence_start].rstrip(' \t')
    if start == sentence_start and preceding and preceding[-1] not in '.!?;:\n\r':
        return None, 'parser_split_product_requires_review'
    after = text[match.end():]
    if re.match(r'\s*(?:to\b|through\b|-)\s*v?\d', after, re.I):
        return None, 'release_range_requires_review'
    operator = v1.OPERATORS.get((bridge.group('bound') or '').lower(), 'mentioned_version')
    qualifier_end = match.end()
    trailing = TRAILING_RE.match(after)
    if trailing:
        word = (trailing.group('paren') or trailing.group('plain')).lower()
        tail_operator = '<=' if word in ('prior', 'earlier', 'below', 'before') else '>='
        if operator not in ('mentioned_version', tail_operator):
            return None, 'conflicting_version_bounds'
        operator = tail_operator
        qualifier_end += trailing.end()
    return {'owner_span': [start, end], 'operator': operator,
            'qualifier_span': [end, qualifier_end],
            'qualifier_text': text[end:qualifier_end]}, None


class ProductVersionFrontend(v1.ProductVersionFrontend):
    def _extract_versions(self, text, graph):
        graph.metadata['security_extraction_version'] = VERSION
        for match in version_mentions(text):
            start, end = match.span('number')
            nid = self._add_security_entity(graph, text[start:end], 'VERSION', 'software_version', start, end, confidence=.8)
            node = graph.get_node(nid)
            sentence = self._find_sentence_node(start, graph)
            owner, reason = owner_candidate(text, sentence.properties.char_start, match) if sentence else (None, 'no_sentence')
            node.properties.extra.update(version_literal=match.group(), version_literal_span=list(match.span()),
                release_form=release_form(match), ownership_candidate=owner,
                ownership_abstention=reason, extraction_policy=VERSION)

    def _extract_software(self, text, graph):
        super()._extract_software(text, graph)
        for node in graph.nodes(NodeType.ENTITY):
            if node.properties.entity_type == 'SOFTWARE':
                node.properties.extra['extraction_policy'] = VERSION


class ProductVersionSecurityPipeline(TPGPipeline):
    def __init__(self, include_security_relations=True):
        super().__init__(frontend=ProductVersionFrontend(schema=SECURITY_SCHEMA), schema=SECURITY_SCHEMA)
        self.include_security_relations = include_security_relations
        if include_security_relations:
            self.passes.append(SecurityRelationsPass())

    def run(self, text, doc_id=''):
        graph = v1.replace_version_edges(super().run(text, doc_id), self.include_security_relations)
        # Reuse the public-API graph rebuild; expose v2 as the decision policy.
        for edge in graph.edges():
            if edge.edge_type == SecurityEdgeType.HAS_VERSION or (
                    edge.edge_type == EdgeType.ENTITY_REL and edge.properties.entity_rel_type == 'HAS_VERSION'):
                edge.properties.extra['extraction_policy'] = VERSION
                edge.properties.extra['release_form'] = graph.get_node(edge.target).properties.extra['release_form']
        graph.mark_pass(VERSION)
        for node in graph.nodes(NodeType.ENTITY):
            p = node.properties
            if p.source == 'security_frontend' and text[p.char_start:p.char_end] != p.text:
                raise ValueError('Security entity does not match source span')
        issues = graph.validate()
        if issues:
            raise ValueError(f'Invalid v2 product/version graph: {issues[:3]}')
        return graph
