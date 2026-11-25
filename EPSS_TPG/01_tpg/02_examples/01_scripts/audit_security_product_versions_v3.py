"""DQ12: compare v3 with frozen v2 on the same small development sample only."""
import argparse
import csv
import hashlib
from pathlib import Path

import audit_security_product_versions_v2 as previous
import review_security_product_v2 as review
import tpg_security_product_v3 as extension
from tpg.contracts import sentence_evidence
from tpg.schema.types import EdgeType, NodeType, SecurityEdgeType

old = previous.old
SCHEMA = 'dq12_security_product_preview_v3'
SOURCE = previous.DEFAULT_OUTPUT
PACKET = review.ROOT/'packet_v1'
DEFAULT_OUTPUT = old.TPG/'05_workspace/security_product_version_v3/smoke'


def verify_sources():
    previous.run(SOURCE, verify_only=True)
    review.check_packet(PACKET)


def without_policy(value):
    """Ignore only the declared extractor version, not source evidence or bounds."""
    if isinstance(value, dict):
        return {k: without_policy(v) for k, v in value.items() if k != 'extraction_policy'}
    if isinstance(value, list):
        return [without_policy(v) for v in value]
    return value


def differences(before, after):
    left = {tuple(v['span']): v for v in before['versions']}
    right = {tuple(v['span']): v for v in after['versions']}
    old.require(len(left) == len(before['versions']) and len(right) == len(after['versions']),
                'Duplicate version spans')
    return {
        'added_versions': [right[s] for s in sorted(right.keys() - left.keys())],
        'removed_versions': [left[s] for s in sorted(left.keys() - right.keys())],
        'changed_versions': [{'before': left[s], 'after': right[s]}
                             for s in sorted(left.keys() & right.keys()) if left[s] != right[s]],
        'software_changed': before['software'] != after['software'],
        'security_relations_changed': without_policy(before['typed_security_relations']) !=
                                      without_policy(after['typed_security_relations']),
        'node_count_change': after['nodes'] - before['nodes'],
        'edge_count_change': after['edges'] - before['edges'],
    }


def validate_graph(graph, text):
    old.require(not graph.validate(), 'Invalid v3 graph')
    old.require(graph.metadata['source_text'] == text and
                graph.metadata['security_extraction_version'] == extension.VERSION, 'Wrong v3 source/policy')
    typed = graph.edges(SecurityEdgeType.HAS_VERSION)
    generic = [e for e in graph.edges(EdgeType.ENTITY_REL) if e.properties.entity_rel_type == 'HAS_VERSION']
    old.require(len(typed) == len(generic), 'Generic/typed version count differs')
    for edge in typed:
        matches = [e for e in generic if (e.source, e.target) == (edge.source, edge.target)]
        old.require(len(matches) == 1 and matches[0].properties.extra == edge.properties.extra,
                    'Generic/typed evidence differs')
        source, target = graph.get_node(edge.source), graph.get_node(edge.target)
        evidence = edge.properties.extra
        expected = sentence_evidence(graph, source, target)
        old.require(source.properties.entity_type == 'SOFTWARE' and target.properties.entity_type == 'VERSION',
                    'Wrong version edge endpoints')
        old.require(expected is not None and all(evidence[k] == expected[k] for k in
                    ('sentence_id', 'sent_idx', 'char_start', 'char_end', 'text', 'source_span', 'target_span')),
                    'Version evidence differs from source')
        old.require(evidence['extraction_policy'] == extension.VERSION and evidence['status'] == 'candidate',
                    'Wrong edge policy/status')
        old.require(review.quote(text, evidence['source_span']) == source.properties.text and
                    review.quote(text, evidence['target_span']) == target.properties.text and
                    review.quote(text, evidence['qualifier_span']) == evidence['qualifier_text'], 'Invalid quote')
        old.require(evidence['qualifier_span'][0] == source.properties.char_end and
                    target.properties.char_end <= evidence['qualifier_span'][1] <= expected['char_end'],
                    'Invalid qualifier bounds')
    for node in graph.nodes(NodeType.ENTITY):
        p = node.properties
        if p.entity_type not in ('VERSION', 'SOFTWARE'):
            continue
        old.require(p.extra['extraction_policy'] == extension.VERSION and
                    review.quote(text, [p.char_start, p.char_end]) == p.text, 'Invalid entity source/policy')
        if p.entity_type != 'VERSION':
            continue
        old.require(review.quote(text, p.extra['version_literal_span']) == p.extra['version_literal'],
                    'Invalid version literal')
        links = [e for e in typed if e.target == node.id]
        reason, owner = p.extra.get('ownership_abstention'), p.extra.get('ownership_candidate')
        old.require((len(links) == 1 and reason is None and owner is not None) or
                    (not links and isinstance(reason, str) and bool(reason) and owner is None),
                    'Version needs one owner or explicit abstention')
        if links:
            evidence = links[0].properties.extra
            old.require(owner['owner_span'] == evidence['source_span'] and
                        owner['operator'] == evidence['version_operator'] and
                        owner['qualifier_span'] == evidence['qualifier_span'] and
                        owner['qualifier_text'] == evidence['qualifier_text'], 'Node/edge ownership differs')


def statistics(rows):
    stats = previous.metrics(rows)
    changed, unexpected = [], []
    recovered = False
    for row in rows:
        d = row['differences']
        is_changed = any(d.values())
        if is_changed:
            changed.append(row['cve'])
        added = d['added_versions']
        expected = (row['cve'] == 'CVE-2022-26130' and len(added) == 1 and
                    added[0]['text'] == '16.1.x' and added[0]['span'] == [13, 19] and
                    added[0]['abstention'] == 'unsupported_wildcard_release' and
                    added[0]['owner_candidate'] is None and
                    d['node_count_change'] == 1 and d['edge_count_change'] == 2 and
                    not any(d[k] for k in ('removed_versions', 'changed_versions', 'software_changed',
                                          'security_relations_changed')))
        recovered |= expected
        if is_changed and not expected:
            unexpected.append(row['cve'])
    stats.update(changed_extraction_cves=changed, unexpected_change_cves=unexpected,
                 expected_big_ip_recovery_present=recovered,
                 review_status='expected_change_only' if recovered and not unexpected else 'requires_investigation')
    return stats


def verify_output(output, pins):
    lock = old.read_json(output/'preview_lock.json')
    old.require(lock['schema'] == SCHEMA, 'Wrong v3 schema')
    old.require({str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()} ==
                set(lock['files']) | {'preview_lock.json'}, 'Unexpected v3 file set')
    for name, digest in lock['files'].items():
        old.require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe v3 artifact path')
        old.require(old.sha256(output/name) == digest, 'Changed v3 artifact')
    old.require(old.read_json(output/'run_status.json')['status'] == 'complete', 'Incomplete v3 preview')
    summary = old.read_json(output/'analysis.json')
    old.require(summary['schema'] == SCHEMA and summary['pinned_files'] == pins, 'V3 identity differs')
    old.require(summary['statistics'] == statistics(old.read_json(output/'comparisons.json')),
                'V3 statistics differ')


def report(summary, rows):
    stats = summary['statistics']
    lines = ['# DQ12: v3 Small Product/Version Preview', '',
        f"Same frozen v2 sample: {len(rows)} CVEs; split counts: {stats['counts_by_split']}.",
        'Each row below is one development description, not an independent accuracy test.', '',
        '| Measure | Frozen v2 | v3 |', '|---|---:|---:|']
    for key, label in [('version_mentions', 'Recognized release mentions'),
                       ('typed_version_links', 'Candidate product/release links'),
                       ('unresolved_versions', 'Release mentions without an assigned product')]:
        lines.append(f"| {label} | {stats['before'][key]} | {stats['after'][key]} |")
    lines += ['', 'Links count the typed copy once, not its generic duplicate.',
        'V3 changes only address-context filtering. It should retain 16.1.x after F5 BIG-IP, '
        'without inventing an owner for that wildcard release. The narrow expectation is one added '
        'node and two added graph edges; other changes are flagged for investigation, not silently accepted.',
        f"Expected BIG-IP recovery present: {stats['expected_big_ip_recovery_present']}.",
        f"Other changed CVEs requiring investigation: {stats['unexpected_change_cves']}.",
        f"Review status: {stats['review_status']}.", '',
        '| CVE | Split | v2 releases | v3 releases | v2 links | v3 links | Added releases | Removed releases |',
        '|---|---|---:|---:|---:|---:|---|---|']
    for row in rows:
        d = row['differences']
        lines.append(f"| {row['cve']} | {row['split']} | {len(row['before']['versions'])} | "
                     f"{len(row['after']['versions'])} | {row['v2_links']} | {row['v3_links']} | "
                     f"{', '.join(v['text'] for v in d['added_versions']) or 'None'} | "
                     f"{', '.join(v['text'] for v in d['removed_versions']) or 'None'} |")
    lines += ['', '## Reading the Files', '',
        '- analysis.json: counts, flags, selected CVEs and checksums of dependencies.',
        '- comparisons.json: exact descriptions, source spans, before/after extraction and every detected difference.',
        '- comparison_counts.csv: one compact count row per CVE.',
        '- graphs/<CVE>.json: separate v3 GraphSON graphs, without embeddings or labels.',
        '- preview_lock.json: hashes for verifying the completed output.', '',
        '## Scope and Limits', '',
        'The complete v2 GraphSON is reconstructed and compared exactly before accepting each v3 graph. '
        'V3 graphs are checked for valid source spans, explicit owner abstentions and matching generic/typed evidence. '
        'Extraction comparisons ignore only extraction_policy tags; evidence and bounds remain compared. '
        'Node/edge count changes are reported, but this is not a claim that every graph property is unchanged.',
        'A recovered mention is not a verified affected-version fact or a measured accuracy improvement. '
        'Complex ranges, release lists, missed mentions and historical text availability remain unresolved.',
        'The human-review packet remains unchanged and pending. No full expansion, embeddings, '
        'model inference, training or test-description inspection. These graphs are not a training cache.', '']
    return '\n'.join(lines)


def run(output, check_only=False, verify_only=False):
    output = Path(output).resolve()
    with old.tqdm(total=3, desc='Verify frozen inputs and v3 preflight', unit='stage') as bar:
        verify_sources(); bar.update()
        source_summary = old.read_json(SOURCE/'analysis.json')
        pins = dict(source_summary['pinned_files'])
        for path in (Path(__file__), Path(extension.__file__), Path(review.__file__),
                     SOURCE/'preview_lock.json', PACKET/'review_lock.json',
                     old.TPG/'08_tests/test_security_product_v3.py',
                     old.TPG/'08_tests/test_security_product_v3_audit.py'):
            pins[str(path.resolve())] = old.sha256(path)
        old.cache.check_pins(pins); bar.update()
        if verify_only:
            verify_output(output, pins); bar.update()
            print('DQ12 v3 saved preview checks passed; no graphs regenerated.'); return
        old.cache.check_destination(output, [SOURCE.parent, previous.SOURCE.parent, previous.REVIEW,
            review.ROOT, old.TPG/'01_core', old.TPG/'10_extensions', old.TPG/'08_tests',
            old.TPG/'02_examples', old.cache.CACHE_ROOT, old.cache.PREPARED, old.REVIEW,
            old.DIAGNOSTIC, old.EPSS.parent/'SummTPGVul'])
        manifest, _, data = old.cache.load_cache(old.cache.CACHE_ROOT/'description_full_v1')
        saved = old.read_json(SOURCE/'comparisons.json')
        previous.selected_rows(source_summary, saved, manifest)
        raw = {r['cve']: r for r in old.read_json(data['preparation_summary.json']['source_path'])}
        baseline = previous.extension.ProductVersionSecurityPipeline()
        candidate = extension.ProductVersionSecurityPipeline()
        for pipeline in (baseline, candidate):
            old.require(pipeline.frontend.model_name == 'en_core_web_sm' and
                hashlib.sha256(pipeline.frontend.nlp.to_bytes()).hexdigest() ==
                manifest['resources']['spacy_pipeline_sha256'], 'NLP pipeline changed')
        for row in saved:
            text = old.cache.description_only(raw[row['cve']])
            old.require(text == row['description'] and old.cache.text_digest(text) ==
                        row['description_sha256'], 'Source description changed')
        bar.update()
    if check_only:
        print(f'DQ12 v3 preflight passed: {len(saved)} development CVEs; no output created.'); return
    output.mkdir(parents=True, exist_ok=False)
    old.write_json(output/'run_status.json', {'schema': SCHEMA, 'status': 'running'})
    try:
        (output/'graphs').mkdir()
        rows = []
        for prior in old.tqdm(saved, desc='Compare frozen v2 with v3', unit='CVE'):
            cve, text = prior['cve'], prior['description']
            before_graph, after_graph = baseline.run(text, doc_id=cve), candidate.run(text, doc_id=cve)
            for graph in (before_graph, after_graph):
                for node in graph.nodes(): node.properties.extra['source_text_type'] = 'description'
                for edge in graph.edges(): edge.properties.extra['source_text_type'] = 'description'
            old.require(old.json.loads(baseline.export_graphson_string(before_graph)) ==
                        old.read_json(SOURCE/'graphs'/f'{cve}.json'), f'V2 reconstruction differs: {cve}')
            old.require(previous.extraction(before_graph) == prior['after'], 'V2 extraction differs')
            validate_graph(after_graph, text)
            payload = candidate.export_pyg(after_graph, embedding_dim=0, use_security_edge_types=True)
            old.require(payload['y'] is None and payload['n_dropped_security_edges'] == 0,
                        'Unexpected labels or dropped security edges')
            before, after = prior['after'], previous.extraction(after_graph)
            row = {k: prior[k] for k in ('cve', 'split', 'fixed_sample', 'previously_reviewed',
                                         'description', 'description_sha256')}
            row.update(before=before, after=after, differences=differences(before, after),
                v2_links=sum(e['relation'] == 'SEC_HAS_VERSION' for e in before['typed_security_relations']),
                v3_links=sum(e['relation'] == 'SEC_HAS_VERSION' for e in after['typed_security_relations']),
                v3_unresolved=sum(v['abstention'] is not None for v in after['versions']))
            rows.append(row)
            candidate.export_graphson(after_graph, str(output/'graphs'/f'{cve}.json'))
        summary = {'schema': SCHEMA, 'scale': 'smoke', 'extension': extension.VERSION,
            'pinned_files': pins, 'inspected_ids': source_summary['inspected_ids'],
            'fixed_sample_ids': source_summary['fixed_sample_ids'],
            'previously_reviewed_ids': source_summary['previously_reviewed_ids'],
            'statistics': statistics(rows), 'training': False, 'test_scoring': False, 'test_graphs': False,
            'embeddings_generated': False, 'model_ready_cache': False, 'original_artifacts_modified': False,
            'full_expansion_authorized': False, 'semantic_review_complete': False}
        with old.tqdm(total=3, desc='Save v3 reports and recheck frozen artifacts', unit='stage') as bar:
            old.write_json(output/'analysis.json', summary)
            old.write_json(output/'comparisons.json', rows)
            fields = ('cve', 'split', 'fixed_sample', 'previously_reviewed', 'v2_links', 'v3_links', 'v3_unresolved')
            with (output/'comparison_counts.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader(); writer.writerows({k: row[k] for k in fields} for row in rows)
            for name in ('analysis.txt', 'README.md'):
                (output/name).write_text(report(summary, rows), encoding='utf-8')
            bar.update()
            verify_sources(); old.cache.check_pins(pins); bar.update()
            old.write_json(output/'run_status.json', {'schema': SCHEMA, 'status': 'complete'})
            old.write_json(output/'preview_lock.json', {'schema': SCHEMA, 'files': {
                str(p.relative_to(output)): old.sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}})
            verify_output(output, pins); bar.update()
        print(f'DQ12 v3 smoke preview saved to {output}. No training, embeddings or test inspection.')
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
