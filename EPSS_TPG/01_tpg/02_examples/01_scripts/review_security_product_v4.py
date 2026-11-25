"""Verify the completed DQ13 preview and describe retained candidate edges."""
import argparse
from collections import Counter
import json
from pathlib import Path

import audit_security_product_versions_v4 as audit

old = audit.old
SOURCE = audit.DEFAULT_OUTPUT
DEFAULT_OUTPUT = SOURCE.parent/'smoke_review'
SCHEMA = 'dq13_saved_preview_review_v1'


def coreference_rows(graph):
    nodes = {n['id']: n for n in graph['vertices']}
    rows = []
    for edge in graph['edges']:
        if edge['label'] != 'COREF':
            continue
        left, right = nodes[edge['outV']], nodes[edge['inV']]
        a, b = left['properties'], right['properties']
        method = edge['properties'].get('EXTRA', {}).get('method')
        rows.append({'edge_id': edge['id'], 'method': method,
            'source_node_type': left['label'], 'target_node_type': right['label'],
            'source_text': a.get('TEXT'), 'target_text': b.get('TEXT'),
            'source_span': [a.get('CHAR_START'), a.get('CHAR_END')],
            'target_span': [b.get('CHAR_START'), b.get('CHAR_END')],
            'source_entity_type': a.get('ENTITY_TYPE'), 'target_entity_type': b.get('ENTITY_TYPE'),
            'different_entity_types': method == 'repeated_surface' and a.get('ENTITY_TYPE') != b.get('ENTITY_TYPE'),
            'semantic_judgment': 'not_reviewed'})
    return rows


def inspect(before, after, row):
    old.require(before['doc_id'] == after['doc_id'] == row['cve'] and row['split'] in ('train', 'validation'),
                'Graph identity or split differs')
    old.require(before['metadata']['source_text'] == after['metadata']['source_text'] == row['description'] and
                old.cache.text_digest(row['description']) == row['description_sha256'], 'Description differs')
    graph = audit.import_graphson(before)
    exporter = audit.GraphSONExporter()
    rebuilt = audit.extension.filter_same_occurrence_coreference(graph)
    old.require(json.loads(exporter.export_string(rebuilt)) == after, 'Saved v4 graph does not reproduce')
    old.require(json.loads(exporter.export_string(graph)) == before, 'Input graph mutated')
    removed = audit.check_delta(before, after, graph)
    old.require(removed == row['removed_edges'] and len(removed) == row['removed_count'], 'Removal report differs')
    old.require(audit.preview.previous.extraction(graph) == row['before'] and
                audit.preview.previous.extraction(rebuilt) == row['after'], 'Extraction report differs')
    for candidate in (graph, rebuilt):
        audit.preview.validate_graph(candidate, row['description'])
    return {'cve': row['cve'], 'split': row['split'], 'removed_count': len(removed),
            'nodes': len(after['vertices']), 'edges_before': len(before['edges']), 'edges_after': len(after['edges']),
            'retained_coreference': coreference_rows(after)}


def report(result):
    s = result['preview_statistics']
    lines = ['# DQ13: Verified v4 Preview', '',
        f"All {result['reproduced_graphs']} saved graphs reproduce exactly from their frozen v3 inputs.",
        'No NLP rerun, embeddings, model inference, training or test inspection was performed.', '',
        '| Measure | v3 | v4 |', '|---|---:|---:|',
        f"| Release mentions | {s['version_mentions_before']} | {s['version_mentions_after']} |",
        f"| Product/release links | {s['product_version_links_before']} | {s['product_version_links_after']} |",
        f"| All coreference edges | {s['coref_before']} | {s['coref_after']} |",
        f"| Coreference between annotations of the same text occurrence | {s['same_span_coref_before']} | {s['same_span_coref_after']} |",
        '', f"Changed CVEs by split: {result['changed_cves_by_split']}.",
        f"Removed edges by split: {result['removed_edges_by_split']}.",
        f"Unchanged nodes: {result['node_count']}. Total edges: {result['edges_before']} -> {result['edges_after']}.", '',
        '## What Was Preserved', '',
        'Node properties and IDs, every retained edge and its evidence, release ownership, bounds, '
        'abstentions, sentence links and token links remain unchanged. Edge IDs can shift after '
        'rebuilding. Graph metadata records the new filter and removals.',
        'For CVE-2022-26130, the recovered 16.1.x VERSION annotation stays. Its false same-occurrence '
        'COREF link to the CARDINAL annotation is removed; neither annotation is deleted.', '',
        '## What the Remaining Edges Mean', '',
        f"Retained methods: {result['retained_methods']}.",
        f"Repeated-surface candidates with different entity types: {result['different_type_candidate_count']}.",
        'Matching text at different positions is a candidate relationship, not proof that both '
        'mentions refer to the same object. Different entity types can reflect inconsistent parser '
        'annotations or a genuinely wrong association; the counts alone cannot decide which. '
        'Pronoun links also remain heuristic, not human-verified. All semantic judgments in this '
        'report are explicitly not_reviewed. Do not automatically delete different-type links.', '',
        '## Recommended Next Step', '',
        'Keep v4 fixed for a bounded review of the existing product/release evidence and the retained '
        'coreference candidates. Resolve or explicitly document those findings before choosing a '
        'larger development preview or a matched training ablation. Do not continue adding rules '
        'merely to increase counts. This sample is not an independent accuracy benchmark, and no '
        'forecast improvement has been established.', '',
        'The original preview, extractor, core, caches, models and human-review packet are unchanged. '
        'analysis.json provides per-CVE retention details and source spans for follow-up review.', '']
    return '\n'.join(lines)


def run(output, check_only=False):
    output = Path(output).resolve()
    audit.run(SOURCE, verify_only=True)
    summary = old.read_json(SOURCE/'analysis.json')
    pins = dict(summary['pinned_files'])
    for path in (SOURCE/'preview_lock.json', Path(__file__), old.TPG/'08_tests/test_security_product_v4_review.py'):
        pins[str(path.resolve())] = old.sha256(path)
    old.cache.check_pins(pins)
    if check_only:
        lock = old.read_json(output/'review_lock.json')
        old.require({p.name for p in output.iterdir()} == set(lock) | {'review_lock.json'}, 'Review file set differs')
        for name, digest in lock.items():
            old.require(Path(name).name == name and old.sha256(output/name) == digest, 'Changed review artifact')
        old.require(old.read_json(output/'analysis.json')['pinned_files'] == pins, 'Review identity differs')
        print('DQ13 v4 saved review verified; original artifacts unchanged.'); return
    old.cache.check_destination(output, [SOURCE, audit.SOURCE.parent, audit.preview.SOURCE.parent,
        audit.preview.previous.SOURCE.parent, audit.preview.review.ROOT, old.TPG/'01_core',
        old.TPG/'10_extensions', old.TPG/'08_tests', old.TPG/'02_examples', old.cache.CACHE_ROOT,
        old.cache.PREPARED, old.REVIEW, old.DIAGNOSTIC, old.EPSS.parent/'SummTPGVul'])
    saved = old.read_json(SOURCE/'comparisons.json')
    audit.selected_rows(summary, saved)
    rows = []
    for row in old.tqdm(saved, desc='Reproduce and inspect saved v4 graphs', unit='CVE'):
        rows.append(inspect(old.read_json(audit.SOURCE/'graphs'/f"{row['cve']}.json"),
                            old.read_json(SOURCE/'graphs'/f"{row['cve']}.json"), row))
    retained = [e for r in rows for e in r['retained_coreference']]
    result = {'schema': SCHEMA, 'pinned_files': pins, 'reproduced_graphs': len(rows),
        'preview_statistics': summary['statistics'], 'rows': rows,
        'changed_cves_by_split': dict(Counter(r['split'] for r in rows if r['removed_count'])),
        'removed_edges_by_split': {s: sum(r['removed_count'] for r in rows if r['split'] == s)
                                  for s in ('train', 'validation')},
        'node_count': sum(r['nodes'] for r in rows), 'edges_before': sum(r['edges_before'] for r in rows),
        'edges_after': sum(r['edges_after'] for r in rows),
        'retained_methods': dict(Counter(e['method'] for e in retained)),
        'different_type_candidate_count': sum(e['different_entity_types'] for e in retained),
        'training': False, 'nlp_rerun': False, 'test_inspection': False,
        'semantic_review_complete': False, 'full_expansion_authorized': False}
    audit.run(SOURCE, verify_only=True)
    old.cache.check_pins(pins)
    output.mkdir(parents=True, exist_ok=False)
    old.write_json(output/'analysis.json', result)
    for name in ('analysis.txt', 'README.md'):
        (output/name).write_text(report(result), encoding='utf-8')
    old.write_json(output/'review_lock.json', {p.name: old.sha256(p) for p in sorted(output.iterdir())})
    print(f'DQ13 v4 review saved to {output}. No training or original-artifact changes.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    old.torch.set_num_threads(1)
    run(args.output_dir, args.check_only)
