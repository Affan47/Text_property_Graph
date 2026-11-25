"""DQ10 v2: same small development sample only; no full mode or training."""
import argparse
from collections import Counter
import csv
import hashlib
from pathlib import Path

import audit_security_product_versions as old
import review_security_product_preview as old_review
import tpg_security_product_v2 as extension

SCHEMA = 'dq10_security_product_preview_v2'
SOURCE = old.DEFAULT_ROOT/'smoke'
REVIEW = old.DEFAULT_ROOT/'smoke_review'
DEFAULT_OUTPUT = old.TPG/'05_workspace/security_product_version_v2/smoke'


def selected_rows(summary, rows, manifest):
    ids = [r['cve'] for r in rows]
    old.require(len(set(ids)) == len(ids) and ids == summary['inspected_ids'], 'Preview selection differs')
    allowed = {c: split for split in ('train', 'validation') for c in manifest['ids'][split]}
    old.require(all(r['split'] in ('train', 'validation') and allowed.get(r['cve']) == r['split']
                    for r in rows), 'Preview contains forbidden or incorrect split')
    return ids


def extraction(graph):
    result = old.extraction_rows(graph)
    forms = {(n.properties.char_start, n.properties.char_end): n.properties.extra.get('release_form')
             for n in graph.nodes(old.NodeType.ENTITY) if n.properties.entity_type == 'VERSION'}
    for node in result['versions']:
        node['release_form'] = forms[tuple(node['span'])]
    return result


def metrics(rows):
    result = {'cves': len(rows), 'counts_by_split': dict(Counter(r['split'] for r in rows))}
    for side in ('before', 'after'):
        result[side] = {
            'version_mentions': sum(len(r[side]['versions']) for r in rows),
            'typed_version_links': sum(e['relation'] == 'SEC_HAS_VERSION' for r in rows
                                      for e in r[side]['typed_security_relations']),
            'unresolved_versions': sum(v['abstention'] is not None for r in rows for v in r[side]['versions']),
            'abstention_reasons': dict(Counter(v['abstention'] for r in rows for v in r[side]['versions'] if v['abstention'])),
            'operators': dict(Counter(e['evidence']['version_operator'] for r in rows
                for e in r[side]['typed_security_relations'] if e['relation'] == 'SEC_HAS_VERSION'))}
    result['changed_endpoint_cves'] = [r['cve'] for r in rows if old.relation_keys(r['before']) != old.relation_keys(r['after'])]
    result['non_version_relation_change_cves'] = [r['cve'] for r in rows if
        [e for e in old.relation_keys(r['before']) if e[0] != 'SEC_HAS_VERSION'] !=
        [e for e in old.relation_keys(r['after']) if e[0] != 'SEC_HAS_VERSION']]
    return result


def verify_output(directory, pins):
    lock = old.read_json(directory/'preview_lock.json')
    old.require(lock['schema'] == SCHEMA, 'Wrong v2 schema')
    old.require({str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()} ==
                set(lock['files']) | {'preview_lock.json'}, 'Unexpected v2 file set')
    for name, digest in lock['files'].items():
        old.require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe v2 path')
        old.require(old.sha256(directory/name) == digest, 'Changed v2 artifact')
    old.require(old.read_json(directory/'run_status.json')['status'] == 'complete', 'Incomplete v2 preview')
    old.require(old.read_json(directory/'analysis.json')['pinned_files'] == pins, 'V2 source/code identity differs')


def report(summary, rows):
    s = summary['statistics']
    lines = ['# DQ10 v2: Small Product/Version Preview', '',
        'This compares v1 with v2 on exactly the same 56 development descriptions. '
        'The sample contains 32 training and 24 validation CVEs; eight validation '
        'cases supplement the fixed 32/16 hash sample. Previously reviewed examples '
        'are engineering regression cases, not independent accuracy measurements.', '',
        '## What v2 Changes', '',
        '- Reject generic owners such as All; support explicit Product (All versions < release) wording.',
        '- Abstain when a product name appears to start at a parser boundary inside a phrase; '
        'do not join across sentences or silently accept an incomplete suffix.',
        '- Remove a leading generic package descriptor and an explicit filename-of-product prefix.',
        '- Retain trailing and below, and before, and parenthesized bounds; keep Patch numbers in the release.',
        '- Preserve wildcard, integer-build and product-prefixed releases as version nodes with '
        'explicit unsupported-form reasons, without assigning owners.',
        '- Reuse the core graph, NLP, provenance, passes and exports, plus v1 software placement '
        'and edge-rebuild helpers. The frozen core and v1 files are not edited.', '',
        '## Counts', '', '| Measure | v1 | v2 |', '|---|---:|---:|']
    for key, label in [('version_mentions', 'Recognized version mentions'),
                       ('typed_version_links', 'Candidate product/version links'),
                       ('unresolved_versions', 'Recognized versions without an owner')]:
        lines.append(f"| {label} | {s['before'][key]} | {s['after'][key]} |")
    lines += ['', 'Links count the typed copy once, not its generic duplicate. '
        'Unresolved counts can increase because v2 recognizes more release forms and explicitly '
        'rejects doubtful ownership. Neither more links nor fewer abstentions establishes accuracy.', '',
        '## Per-CVE Comparison', '',
        '| CVE | Split | v1 links | v2 links | v2 unresolved versions |', '|---|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['cve']} | {r['split']} | {r['v1_links']} | {r['v2_links']} | {r['v2_unresolved']} |")
    lines += ['', 'Full source descriptions and before/after evidence are in comparisons.json. '
        'graphs/<CVE>.json contains v2 GraphSON with source spans, qualifiers and explicit abstentions.', '',
        '## Remaining Limits', '',
        'Complex release lists, ranges, integer-only releases without a build marker, unknown '
        'lowercase products and NLP sentence errors remain limited. Unsupported-form flags cover '
        'recognized patterns only; zero abstentions is not proof of complete extraction.',
        'Version bounds and release forms remain metadata. The existing 64-value edge encoding '
        'does not encode them. These unembedded graphs are not a model-ready cache.',
        'This is still rule-based extraction of candidate mentions, not learned extraction and '
        'not an assertion that each named release is affected or fixed.', '',
        '## Integrity and Scope', '',
        'The v1 preview/review and current source/code pins are verified. The v1 graphs are '
        'reconstructed on the same 56 descriptions and compared exactly with their saved GraphSON. '
        'The original cache is verified, but no model forward pass or parameter update runs.',
        f"Non-version typed relation changes: {s['non_version_relation_change_cves']}.",
        'No full expansion, training, embeddings, test-description inspection or test scoring. '
        'Do not use this preview as a training cache. Review every changed and unresolved case '
        'before planning a separate expansion.', '']
    return '\n'.join(lines)


def run(output, check_only=False, verify_only=False):
    output = Path(output).resolve()
    old_review.run(SOURCE, REVIEW, check_only=True)
    source_summary = old.read_json(SOURCE/'analysis.json')
    pins = dict(source_summary['pinned_files'])
    for path in (Path(__file__), Path(extension.__file__), Path(old_review.__file__),
                 SOURCE/'preview_lock.json', REVIEW/'review_lock.json', REVIEW/'analysis.json',
                 old.TPG/'08_tests/test_security_product_v2.py',
                 old.TPG/'08_tests/test_security_product_v2_audit.py'):
        pins[str(path.resolve())] = old.sha256(path)
    old.cache.check_pins(pins)
    if verify_only:
        verify_output(output, pins)
        print('DQ10 v2 saved preview checks passed; no graphs regenerated.'); return
    old.cache.check_destination(output, [old.DEFAULT_ROOT, old.TPG/'01_core', old.TPG/'10_extensions',
        old.TPG/'08_tests', old.cache.CACHE_ROOT, old.cache.PREPARED, old.REVIEW, old.DIAGNOSTIC,
        old.EPSS.parent/'SummTPGVul'])
    manifest, _, data = old.cache.load_cache(old.cache.CACHE_ROOT/'description_full_v1')
    saved = old.read_json(SOURCE/'comparisons.json')
    selected_rows(source_summary, saved, manifest)
    raw = {r['cve']: r for r in old.read_json(data['preparation_summary.json']['source_path'])}
    baseline = old.extension.ProductVersionSecurityPipeline()
    candidate = extension.ProductVersionSecurityPipeline()
    for pipeline in (baseline, candidate):
        old.require(pipeline.frontend.model_name == 'en_core_web_sm', 'Wrong NLP model')
        old.require(hashlib.sha256(pipeline.frontend.nlp.to_bytes()).hexdigest() ==
                    manifest['resources']['spacy_pipeline_sha256'], 'NLP pipeline changed')
    if check_only:
        print(f'DQ10 v2 preflight passed: {len(saved)} development CVEs; no output created.'); return
    output.mkdir(parents=True, exist_ok=False)
    old.write_json(output/'run_status.json', {'status': 'running', 'schema': SCHEMA})
    try:
        (output/'graphs').mkdir()
        rows = []
        for previous in old.tqdm(saved, desc='Compare v1 and v2 on fixed smoke sample', unit='CVE'):
            cve = previous['cve']
            text = old.cache.description_only(raw[cve])
            old.require(text == previous['description'] and old.cache.text_digest(text) ==
                        previous['description_sha256'], f'Source description changed: {cve}')
            before_graph = baseline.run(text, doc_id=cve)
            after_graph = candidate.run(text, doc_id=cve)
            for graph in (before_graph, after_graph):
                for node in graph.nodes(): node.properties.extra['source_text_type'] = 'description'
                for edge in graph.edges(): edge.properties.extra['source_text_type'] = 'description'
            # Compare the full v1 GraphSON object, not only relation counts.
            old.require(old.json.loads(baseline.export_graphson_string(before_graph)) ==
                        old.read_json(SOURCE/'graphs'/f'{cve}.json'), f'V1 reconstruction differs: {cve}')
            payload = candidate.export_pyg(after_graph, embedding_dim=0, use_security_edge_types=True)
            old.require(payload['y'] is None and payload['n_dropped_security_edges'] == 0,
                        'Unexpected labels or dropped security edges')
            before, after = previous['after'], extraction(after_graph)
            row = {k: previous[k] for k in ('cve', 'split', 'fixed_sample', 'previously_reviewed',
                                           'description', 'description_sha256')}
            row.update(before=before, after=after,
                v1_links=sum(e['relation'] == 'SEC_HAS_VERSION' for e in before['typed_security_relations']),
                v2_links=sum(e['relation'] == 'SEC_HAS_VERSION' for e in after['typed_security_relations']),
                v2_unresolved=sum(v['abstention'] is not None for v in after['versions']))
            rows.append(row)
            candidate.export_graphson(after_graph, str(output/'graphs'/f'{cve}.json'))
        summary = {'schema': SCHEMA, 'scale': 'smoke', 'extension': extension.VERSION,
            'pinned_files': pins, 'inspected_ids': source_summary['inspected_ids'],
            'fixed_sample_ids': source_summary['fixed_sample_ids'],
            'previously_reviewed_ids': source_summary['previously_reviewed_ids'],
            'statistics': metrics(rows), 'training': False, 'test_scoring': False,
            'test_graphs': False, 'embeddings_generated': False, 'model_ready_cache': False,
            'original_artifacts_modified': False}
        with old.tqdm(total=4, desc='Save v2 preview and verify original artifacts', unit='stage') as bar:
            old.write_json(output/'analysis.json', summary)
            old.write_json(output/'comparisons.json', rows); bar.update()
            fields = ('cve', 'split', 'fixed_sample', 'previously_reviewed', 'v1_links', 'v2_links', 'v2_unresolved')
            with (output/'comparison_counts.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader(); writer.writerows({k: r[k] for k in fields} for r in rows)
            for name in ('analysis.txt', 'README.md'):
                (output/name).write_text(report(summary, rows), encoding='utf-8')
            bar.update()
            old_review.run(SOURCE, REVIEW, check_only=True)
            old.cache.check_pins(pins); bar.update()
            old.write_json(output/'run_status.json', {'status': 'complete', 'schema': SCHEMA})
            old.write_json(output/'preview_lock.json', {'schema': SCHEMA, 'files': {
                str(p.relative_to(output)): old.sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}})
            bar.update()
        print(f'DQ10 v2 smoke saved to {output}. No full expansion, training, embeddings or test inspection.')
    except Exception as exc:
        old.write_json(output/'run_status.json', {'status': 'failed', 'schema': SCHEMA, 'error': str(exc)})
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
