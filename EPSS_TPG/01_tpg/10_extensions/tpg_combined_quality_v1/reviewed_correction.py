"""Add the independently inspected CIS advisory citation case to the preview."""
from collections import Counter
import re
from . import correction as base
from tpg_combined_v2 import common as c

POLICY, RULES = base.POLICY, base.RULES
validate_preview = base.validate_preview


def citation_findings(graph):
    text = graph['metadata']['source_text']
    rows = []
    for n in graph['vertices']:
        p = n['properties']
        if n['label'] != 'ENTITY' or p.get('ENTITY_TYPE') != 'SOFTWARE' or not re.fullmatch(r'Safeguards?', p['TEXT'], re.I):
            continue
        a, b = base.span(n)
        part = graph['metadata']['combined_text']['spans'][p['DOC_IDX']]
        # A nearby actual CIS link plus a numbered control heading, not the
        # word Safeguard alone, licenses this additional bounded correction.
        local = text[max(part['start'], a-1200):min(part['end'], b+1200)]
        if not re.search(r'https?://(?:[A-Za-z0-9-]+\.)*cisecurity\.org(?:[/\s)]|$)', local, re.I):
            continue
        if not re.match(r'\s+\d+\.\d+\s*:\s*', text[b:part['end']]):
            continue
        rows.append({'rule':'CIS_SAFEGUARD_NOT_PRODUCT','node_id':n['id'], 'label':n['label'],
            'entity_type':'SOFTWARE','text':p['TEXT'],'span':[a,b],'document_index':p['DOC_IDX'],
            'context':text[max(part['start'],a-100):min(part['end'],b+160)],
            'evidence':'Numbered Safeguard heading with a cisecurity.org citation within 1200 characters in the same document.'})
    return rows


def correct(graph):
    result = base.correct(graph)
    rows = citation_findings(result)
    meta = result['metadata']['annotation_repair_preview']
    if meta.get('citation_extension'):
        c.require(not rows, 'Previously repaired citation case remains')
        return result
    removed = {r['node_id'] for r in rows}
    edges = {e['id'] for e in result['edges'] if e['outV'] in removed or e['inV'] in removed}
    result['vertices'] = [n for n in result['vertices'] if n['id'] not in removed]
    result['edges'] = [e for e in result['edges'] if e['id'] not in edges]
    meta['findings'].extend(rows)
    meta['rule_counts'] = dict(Counter(r['rule'] for r in meta['findings']))
    meta['removed_node_ids'] = sorted(set(meta['removed_node_ids']) | removed)
    meta['removed_edge_ids'] = sorted(set(meta['removed_edge_ids']) | edges)
    meta['citation_extension'] = 'same_document_cis_link_numbered_heading_v1'
    result['stats'].update(total_nodes=len(result['vertices']),total_edges=len(result['edges']),
        node_types=dict(Counter(n['label'] for n in result['vertices'])),
        edge_types=dict(Counter(e['label'] for e in result['edges'])))
    c.require(not citation_findings(result), 'Citation correction incomplete')
    return result
