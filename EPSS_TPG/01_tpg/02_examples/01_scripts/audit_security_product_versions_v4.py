"""DQ13: apply the narrow coreference correction to the same saved v3 sample.

No parser, model, embeddings, expanded cohort or training is used by this CLI.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import review_security_product_v3 as prior_review
import tpg_security_product_v4 as extension
from tpg.exporters.exporters import GraphSONExporter, PyGExporter, import_graphson

preview = prior_review.preview
old = preview.old
SOURCE = prior_review.SOURCE
REVIEW = prior_review.DEFAULT_OUTPUT
DEFAULT_OUTPUT = old.TPG/'05_workspace/security_product_version_v4/smoke'
SCHEMA = 'dq13_same_occurrence_coreference_preview_v1'


def verify_sources():
    prior_review.run(REVIEW, check_only=True)


def selected_rows(summary, rows):
    ids = [r['cve'] for r in rows]
    old.require(len(ids) == len(set(ids)) and ids == summary['inspected_ids'], 'Selection differs')
    old.require(all(r['split'] in ('train', 'validation') for r in rows), 'Test inspection forbidden')


def edge_content(edge):
    return {k: v for k, v in edge.items() if k != 'id'}


def check_delta(before, after, input_graph):
    """Compare actual removed content, never an assumed total edge count."""
    old.require(before['vertices'] == after['vertices'], 'Node content or IDs changed')
    old.require(before['schema'] == after['schema'] and before['doc_id'] == after['doc_id'], 'Graph identity changed')
    graph_edges = input_graph.edges()
    old.require(len(graph_edges) == len(before['edges']), 'Input edge count differs')
    removed, retained = [], []
    for edge, saved in zip(graph_edges, before['edges']):
        old.require(edge.source == saved['outV'] and edge.target == saved['inV'] and
                    edge.edge_type.name == saved['label'].removeprefix('SEC_'), 'Imported edge order differs')
        (removed if extension.same_occurrence_candidate(input_graph, edge) else retained).append(saved)
    old.require([edge_content(e) for e in retained] == [edge_content(e) for e in after['edges']],
                'Retained edges, ordering or evidence changed')
    old.require(after['metadata']['coreference_filter']['removed_count'] == len(removed), 'Removal audit differs')
    return removed


def statistics(rows):
    return {'cves': len(rows), 'counts_by_split': dict(Counter(r['split'] for r in rows)),
        'removed_same_occurrence_edges': sum(r['removed_count'] for r in rows),
        'changed_cves': [r['cve'] for r in rows if r['removed_count']],
        'same_span_coref_before': sum(r['same_span_coref_before'] for r in rows),
        'same_span_coref_after': sum(r['same_span_coref_after'] for r in rows),
        'coref_before': sum(r['coref_before'] for r in rows),
        'coref_after': sum(r['coref_after'] for r in rows),
        'version_mentions_before': sum(len(r['before']['versions']) for r in rows),
        'version_mentions_after': sum(len(r['after']['versions']) for r in rows),
        'product_version_links_before': sum(r['version_links_before'] for r in rows),
        'product_version_links_after': sum(r['version_links_after'] for r in rows),
        'remaining_same_span_requires_review': any(r['same_span_coref_after'] for r in rows)}


def verify_output(output, pins):
    lock = old.read_json(output/'preview_lock.json')
    old.require(lock['schema'] == SCHEMA, 'Wrong v4 preview schema')
    old.require({str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()} ==
                set(lock['files']) | {'preview_lock.json'}, 'Unexpected v4 file set')
    for name, digest in lock['files'].items():
        old.require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe output path')
        old.require(old.sha256(output/name) == digest, 'Changed v4 artifact')
    summary = old.read_json(output/'analysis.json')
    old.require(summary['pinned_files'] == pins and summary['schema'] == SCHEMA, 'V4 identity differs')
    old.require(old.read_json(output/'run_status.json')['status'] == 'complete', 'Incomplete v4 run')
    rows = old.read_json(output/'comparisons.json')
    selected_rows(summary, rows)
    old.require(summary['statistics'] == statistics(rows), 'V4 statistics differ')


def report(summary, rows):
    s = summary['statistics']
    lines = ['# DQ13: Same-Occurrence Coreference Preview', '',
        f"Same saved development sample: {s['cves']} CVEs; {s['counts_by_split']}.",
        'This is a structural correction, not a new extraction benchmark or model run.', '',
        '| Measure | Saved v3 | v4 |', '|---|---:|---:|',
        f"| Release mentions | {s['version_mentions_before']} | {s['version_mentions_after']} |",
        f"| Candidate product/release links | {s['product_version_links_before']} | {s['product_version_links_after']} |",
        f"| All coreference edges | {s['coref_before']} | {s['coref_after']} |",
        f"| Coreference edges over exactly the same text span | {s['same_span_coref_before']} | {s['same_span_coref_after']} |",
        '', f"Removed same-occurrence candidate edges: {s['removed_same_occurrence_edges']}.",
        f"Remaining same-span edges require review: {s['remaining_same_span_requires_review']}.", '',
        '## What Changes', '',
        'For example, spaCy CARDINAL and security VERSION annotations can both cover the same '
        '16.1.x occurrence. The repeated_surface candidate edge between them is removed. Both '
        'annotations, their token and sentence links, and the source wording remain intact.',
        'Only candidate repeated_surface COREF edges between ENTITY nodes with identical, '
        'nonempty, source-verified spans are removed. Unknown offsets, asserted links, other '
        'methods, pronoun links and distinct-span links are not changed. No new links are invented.',
        'Version extraction remains v3. A separate v4 coreference_filter metadata record records '
        'each removal. Edge IDs can shift after rebuilding; node IDs do not.', '',
        '| CVE | Split | Coreference edges before | Removed same-occurrence edges | Coreference edges after |',
        '|---|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['cve']} | {r['split']} | {r['coref_before']} | {r['removed_count']} | {r['coref_after']} |")
    lines += ['', '## Checks and Limitations', '',
        'Every node and retained edge property is compared with the saved v3 graph. The only '
        'permitted edge removals are those meeting the explicit rule, not a hardcoded count. '
        'Product/version evidence, abstentions and other security links must remain identical.',
        'Distinct-span matches remain heuristic candidates, not proven identity. Type conflicts '
        'across separate mentions, pronoun ambiguity, and broader coreference quality are not fixed here. '
        'No claim of improved forecast performance or extraction accuracy follows from fewer edges.',
        'analysis.json holds totals and pins. comparisons.json holds source descriptions, '
        'before/after extraction and every removed edge. comparison_counts.csv is a compact table. '
        'graphs/ contains separate, unembedded v4 GraphSON files and preview_lock.json hashes the output.',
        'No NLP rerun, embeddings, model inference, training, full expansion or test inspection. '
        'Original source data, core code, graphs, models and human-review decisions remain unchanged.', '']
    return '\n'.join(lines)


def run(output, check_only=False, verify_only=False):
    output = Path(output).resolve()
    with old.tqdm(total=2, desc='Verify frozen v3 preview and review', unit='stage') as bar:
        verify_sources(); bar.update()
        source_summary = old.read_json(SOURCE/'analysis.json')
        pins = dict(source_summary['pinned_files'])
        for path in (Path(__file__), Path(extension.__file__), Path(prior_review.__file__),
                     SOURCE/'preview_lock.json', REVIEW/'review_lock.json',
                     old.TPG/'08_tests/test_security_product_v4.py',
                     old.TPG/'08_tests/test_security_product_v4_audit.py'):
            pins[str(path.resolve())] = old.sha256(path)
        old.cache.check_pins(pins); bar.update()
    if verify_only:
        verify_output(output, pins)
        print('DQ13 v4 saved preview verified; no graphs regenerated.'); return
    old.cache.check_destination(output, [SOURCE.parent, preview.SOURCE.parent, preview.previous.SOURCE.parent,
        preview.review.ROOT, old.TPG/'01_core', old.TPG/'10_extensions', old.TPG/'08_tests',
        old.TPG/'02_examples', old.cache.CACHE_ROOT, old.cache.PREPARED, old.REVIEW, old.DIAGNOSTIC,
        old.EPSS.parent/'SummTPGVul'])
    saved = old.read_json(SOURCE/'comparisons.json')
    selected_rows(source_summary, saved)
    if check_only:
        print(f'DQ13 v4 preflight passed: {len(saved)} saved development CVEs; no output created.'); return
    output.mkdir(parents=True, exist_ok=False)
    old.write_json(output/'run_status.json', {'schema': SCHEMA, 'status': 'running'})
    try:
        (output/'graphs').mkdir()
        exporter, pyg = GraphSONExporter(), PyGExporter()
        rows = []
        for prior in old.tqdm(saved, desc='Filter same-occurrence edges in saved sample', unit='CVE'):
            cve = prior['cve']
            original = old.read_json(SOURCE/'graphs'/f'{cve}.json')
            graph = import_graphson(original)
            old.require(graph.doc_id == cve and graph.metadata['source_text'] == prior['description'] and
                        old.cache.text_digest(prior['description']) == prior['description_sha256'], 'Source identity differs')
            old.require(json.loads(exporter.export_string(graph)) == original, 'GraphSON round trip differs')
            preview.validate_graph(graph, prior['description'])
            old.require(preview.previous.extraction(graph) == prior['after'], 'V3 extraction differs')
            filtered = extension.filter_same_occurrence_coreference(graph)
            preview.validate_graph(filtered, prior['description'])
            after_json = json.loads(exporter.export_string(filtered))
            removed = check_delta(original, after_json, graph)
            old.require(json.loads(exporter.export_string(graph)) == original, 'Input graph was mutated')
            before, after = prior['after'], preview.previous.extraction(filtered)
            old.require({k: v for k, v in before.items() if k != 'edges'} ==
                        {k: v for k, v in after.items() if k != 'edges'}, 'Non-coreference extraction changed')
            payload = pyg.export(filtered, embedding_dim=0, use_security_edge_types=True)
            old.require(payload['y'] is None and payload['n_dropped_security_edges'] == 0, 'Label/export contract differs')
            row = {k: prior[k] for k in ('cve', 'split', 'description', 'description_sha256',
                                         'fixed_sample', 'previously_reviewed')}
            row.update(before=before, after=after, removed_edges=removed, removed_count=len(removed),
                same_span_coref_before=len(prior_review.same_span_coreferences(original)),
                same_span_coref_after=len(prior_review.same_span_coreferences(after_json)),
                coref_before=sum(e['label'] == 'COREF' for e in original['edges']),
                coref_after=sum(e['label'] == 'COREF' for e in after_json['edges']),
                version_links_before=prior['v3_links'],
                version_links_after=sum(e['relation'] == 'SEC_HAS_VERSION' for e in after['typed_security_relations']))
            rows.append(row)
            exporter.export(filtered, str(output/'graphs'/f'{cve}.json'))
        summary = {'schema': SCHEMA, 'scale': 'smoke', 'extension': extension.VERSION,
            'extraction_policy': extension.v3.VERSION, 'pinned_files': pins,
            'inspected_ids': source_summary['inspected_ids'], 'fixed_sample_ids': source_summary['fixed_sample_ids'],
            'previously_reviewed_ids': source_summary['previously_reviewed_ids'], 'statistics': statistics(rows),
            'training': False, 'test_inspection': False, 'embeddings_generated': False, 'nlp_rerun': False,
            'model_ready_cache': False, 'full_expansion_authorized': False, 'semantic_review_complete': False}
        with old.tqdm(total=3, desc='Save v4 preview and verify frozen inputs', unit='stage') as bar:
            old.write_json(output/'analysis.json', summary)
            old.write_json(output/'comparisons.json', rows)
            fields = ('cve', 'split', 'coref_before', 'removed_count', 'coref_after',
                      'same_span_coref_after', 'version_links_before', 'version_links_after')
            with (output/'comparison_counts.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader(); writer.writerows({k: r[k] for k in fields} for r in rows)
            for name in ('analysis.txt', 'README.md'):
                (output/name).write_text(report(summary, rows), encoding='utf-8')
            bar.update()
            verify_sources(); old.cache.check_pins(pins); bar.update()
            old.write_json(output/'run_status.json', {'schema': SCHEMA, 'status': 'complete'})
            old.write_json(output/'preview_lock.json', {'schema': SCHEMA, 'files': {
                str(p.relative_to(output)): old.sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}})
            verify_output(output, pins); bar.update()
        print(f'DQ13 v4 smoke preview saved to {output}. No training, embeddings, NLP rerun or test inspection.')
    except Exception as exc:
        old.write_json(output/'run_status.json', {'schema': SCHEMA, 'status': 'failed', 'error': str(exc)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    old.torch.set_num_threads(1)
    run(args.output_dir, args.check_only, args.verify_only)
