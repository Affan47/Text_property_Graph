"""Read-only DQ12 graph comparison; preserve the frozen preview and its warning."""
import argparse
from collections import Counter
import json
from pathlib import Path

import audit_security_product_versions_v3 as preview

old = preview.old
SOURCE = preview.DEFAULT_OUTPUT
BASELINE = preview.SOURCE
DEFAULT_OUTPUT = SOURCE.parent/'smoke_review'
SCHEMA = 'dq12_v3_saved_graph_review_v1'


def serialized(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def canonical(graph):
    """Use complete node properties as endpoints, retaining edge multiplicity."""
    nodes = {n['id']: {'label': n['label'], 'properties': preview.without_policy(n['properties'])}
             for n in graph['vertices']}
    old.require(len(nodes) == len(graph['vertices']), 'Duplicate GraphSON node IDs')
    edges = []
    for edge in graph['edges']:
        old.require(edge['outV'] in nodes and edge['inV'] in nodes, 'Missing edge endpoint')
        edges.append({'source': nodes[edge['outV']], 'target': nodes[edge['inV']],
                      'label': edge['label'], 'properties': preview.without_policy(edge['properties'])})
    return Counter(serialized(n) for n in nodes.values()), Counter(serialized(e) for e in edges)


def unpack(counter):
    return [{'count': count, 'value': json.loads(value)} for value, count in sorted(counter.items())]


def compare(before, after):
    bn, be = canonical(before)
    an, ae = canonical(after)
    return {'added_nodes': unpack(an-bn), 'removed_nodes': unpack(bn-an),
            'added_edges': unpack(ae-be), 'removed_edges': unpack(be-ae)}


def same_span_coreferences(graph):
    nodes = {n['id']: n for n in graph['vertices']}
    result = []
    for edge in graph['edges']:
        if edge['label'] != 'COREF':
            continue
        left, right = nodes[edge['outV']]['properties'], nodes[edge['inV']]['properties']
        span = [left.get('CHAR_START'), left.get('CHAR_END')]
        if (all(type(v) is int for v in span) and 0 <= span[0] < span[1] and
                span == [right.get('CHAR_START'), right.get('CHAR_END')]):
            result.append({'span': span, 'source_text': left.get('TEXT'), 'target_text': right.get('TEXT'),
                           'source_entity_type': left.get('ENTITY_TYPE'),
                           'target_entity_type': right.get('ENTITY_TYPE'), 'properties': edge['properties']})
    return result


def report(result):
    lines = ['# DQ12 v3 Saved-Preview Review', '',
        f"Reviewed {result['cves']} saved CVEs; split counts: {result['split_counts']}.",
        'The original preview and its requires_investigation flag are preserved.', '',
        '| Measure | v2 | v3 |', '|---|---:|---:|']
    stats = result['preview_statistics']
    for key, label in [('version_mentions', 'Recognized release mentions'),
                       ('typed_version_links', 'Candidate product/release links'),
                       ('unresolved_versions', 'Releases without an assigned product')]:
        lines.append(f"| {label} | {stats['before'][key]} | {stats['after'][key]} |")
    lines += ['', f"Graph changes beyond ID and extractor-policy tags: {result['changed_cves']}.",
        f"Same-span candidate coreference edges: {result['same_span_coref_totals']}.", '',
        '## Why the Preview Flagged the Result', '',
        'The preview expected one extra node and exactly two edges. That expectation came from '
        'a shorter test sentence and was too narrow for the complete description. The saved result '
        'recovers 16.1.x at [13, 19) in CVE-2022-26130, but adds three edges:', '',
        '- CONTAINS: the sentence contains the new VERSION annotation.',
        '- BELONGS_TO: the new VERSION annotation links to its existing text token.',
        '- COREF: the existing spaCy CARDINAL annotation links to the VERSION annotation '
        'over exactly the same text span.', '',
        'The recovery is real despite expected_big_ip_recovery_present=false in the original report. '
        'That flag combined recovery with the overly strict edge-count expectation. Do not interpret '
        'it as evidence that the version was still missing.', '',
        '## Remaining Semantic Concern', '',
        'The core repeated_surface rule groups entities by equal normalized text without excluding '
        'identical spans or checking compatible entity types. Here, CARDINAL and VERSION annotate '
        'one occurrence of 16.1.x. This is not evidence of two separate mentions referring to the same '
        'thing. The edge is marked candidate, but its COREF meaning is questionable. It was not '
        'created by a new v3 relation rule; the recovered entity triggered the existing shared pass.',
        'A later version should distinguish same-span annotations from genuine repeated mentions. '
        'Do not patch the frozen core or delete the edge from these saved graphs.', '',
        '## Verification and Limits', '',
        'All original checksums and preview statistics were verified. The comparison uses complete '
        'node properties and edge properties, retaining multiplicity and direction. IDs are replaced '
        'by node content, and only extraction_policy fields are removed. Graph-level metadata and '
        'pass-history tags are not included in this node/edge comparison.',
        'This review reads saved graphs only: no NLP rerun, embeddings, training, inference, test '
        'inspection or expansion. Human semantic judgments remain pending. The recovered wildcard '
        'still has no assigned product, and HAS_VERSION counts remain unchanged. No forecast benefit '
        'or extraction accuracy gain has been established.', '',
        'Detailed per-CVE differences and every same-span coreference case are in analysis.json.', '']
    return '\n'.join(lines)


def run(output, check_only=False):
    output = Path(output).resolve()
    preview.run(SOURCE, verify_only=True)
    summary = old.read_json(SOURCE/'analysis.json')
    pins = dict(summary['pinned_files'])
    for path in (SOURCE/'preview_lock.json', BASELINE/'preview_lock.json', Path(__file__),
                 old.TPG/'08_tests/test_security_product_v3_review.py'):
        pins[str(path.resolve())] = old.sha256(path)
    old.cache.check_pins(pins)
    if check_only:
        lock = old.read_json(output/'review_lock.json')
        old.require({p.name for p in output.iterdir()} == set(lock) | {'review_lock.json'}, 'Review file set differs')
        for name, digest in lock.items():
            old.require(Path(name).name == name and old.sha256(output/name) == digest, 'Changed review artifact')
        old.require(old.read_json(output/'analysis.json')['pinned_files'] == pins, 'Review identity differs')
        print('DQ12 saved v3 review verified; original preview unchanged.'); return
    old.cache.check_destination(output, [SOURCE, BASELINE.parent, preview.previous.SOURCE.parent,
        preview.review.ROOT, old.TPG/'01_core', old.TPG/'10_extensions', old.TPG/'08_tests',
        old.TPG/'02_examples', old.cache.CACHE_ROOT, old.cache.PREPARED, old.EPSS.parent/'SummTPGVul'])
    comparisons = old.read_json(SOURCE/'comparisons.json')
    old.require([r['cve'] for r in comparisons] == summary['inspected_ids'], 'Selection differs')
    rows = []
    for row in old.tqdm(comparisons, desc='Compare saved v2/v3 graph content', unit='CVE'):
        old.require(row['split'] in ('train', 'validation'), 'Test inspection forbidden')
        cve = row['cve']
        before = old.read_json(BASELINE/'graphs'/f'{cve}.json')
        after = old.read_json(SOURCE/'graphs'/f'{cve}.json')
        old.require(before['doc_id'] == after['doc_id'] == cve and
                    before['metadata']['source_text'] == after['metadata']['source_text'] == row['description'],
                    'Saved graph/source differs')
        rows.append({'cve': cve, 'split': row['split'], 'graph_difference': compare(before, after),
                     'before_same_span_coreferences': same_span_coreferences(before),
                     'after_same_span_coreferences': same_span_coreferences(after)})
    changed = [r for r in rows if any(r['graph_difference'].values())]
    # This report interprets this specific result; fail rather than reuse its prose for a different run.
    old.require([r['cve'] for r in changed] == ['CVE-2022-26130'], 'Different changed cases; review report wording')
    diff = changed[0]['graph_difference']
    edge_counts = Counter()
    for item in diff['added_edges']:
        edge_counts[item['value']['label']] += item['count']
    old.require(not diff['removed_nodes'] and not diff['removed_edges'] and
                sum(x['count'] for x in diff['added_nodes']) == 1 and
                edge_counts == Counter({'CONTAINS': 1, 'BELONGS_TO': 1, 'COREF': 1}),
                'Different graph delta; review wording')
    node = diff['added_nodes'][0]['value']['properties']
    old.require(node['TEXT'] == '16.1.x' and node['ENTITY_TYPE'] == 'VERSION' and
                [node['CHAR_START'], node['CHAR_END']] == [13, 19] and
                node['EXTRA']['ownership_abstention'] == 'unsupported_wildcard_release', 'Different recovered node')
    coref = next(x['value'] for x in diff['added_edges'] if x['value']['label'] == 'COREF')
    for endpoint, entity_type in (('source', 'CARDINAL'), ('target', 'VERSION')):
        properties = coref[endpoint]['properties']
        old.require(properties['ENTITY_TYPE'] == entity_type and properties['TEXT'] == '16.1.x' and
                    [properties['CHAR_START'], properties['CHAR_END']] == [13, 19],
                    'Different coreference endpoints; review wording')
    old.require(coref['properties']['EXTRA']['method'] == 'repeated_surface' and
                coref['properties']['EXTRA']['status'] == 'candidate', 'Different coreference method/status')
    result = {'schema': SCHEMA, 'pinned_files': pins, 'cves': len(rows),
        'split_counts': dict(Counter(r['split'] for r in rows)), 'preview_statistics': summary['statistics'],
        'changed_cves': [r['cve'] for r in changed], 'rows': rows,
        'same_span_coref_totals': {side: sum(len(r[f'{side}_same_span_coreferences']) for r in rows)
                                  for side in ('before', 'after')},
        'conclusion': 'version_recovered_with_existing_same_span_coreference_concern',
        'findings': [
            {'type': 'audit_expectation_flaw', 'expected_edges': 2, 'actual_edges': 3,
             'explanation': 'The original recovery flag also required two edges; the release was recovered.'},
            {'type': 'existing_coreference_semantic_concern', 'cve': 'CVE-2022-26130',
             'span': [13, 19], 'explanation': 'CARDINAL and VERSION annotate one occurrence, not two mentions.'}],
        'training': False, 'test_inspection': False, 'graphs_regenerated': False,
        'semantic_review_complete': False, 'full_expansion_authorized': False}
    old.cache.check_pins(pins)
    preview.run(SOURCE, verify_only=True)
    output.mkdir(parents=True, exist_ok=False)
    old.write_json(output/'analysis.json', result)
    for name in ('analysis.txt', 'README.md'):
        (output/name).write_text(report(result), encoding='utf-8')
    old.write_json(output/'review_lock.json', {p.name: old.sha256(p) for p in sorted(output.iterdir())})
    print(f'DQ12 v3 saved-graph review written to {output}. Original artifacts unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    old.torch.set_num_threads(1)
    run(args.output_dir, args.check_only)
