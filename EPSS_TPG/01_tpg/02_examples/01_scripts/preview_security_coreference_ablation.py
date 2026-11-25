"""DQ15: three matched edge-only variants of the fixed v4 development preview."""
import argparse
from collections import Counter
import json
from pathlib import Path

import review_security_product_v4_semantics as semantic
import tpg_coreference_ablation_v1 as policy
from tpg.exporters.exporters import GraphSONExporter, PyGExporter, import_graphson

old = semantic.old
SOURCE = semantic.frozen.SOURCE
DEFAULT_OUTPUT = old.TPG/'05_workspace/coreference_ablation_v1/smoke'
SCHEMA = 'dq15_coreference_ablation_preflight_v1'


def source_context():
    semantic.run(semantic.DEFAULT_OUTPUT, check_only=True)
    summary = old.read_json(SOURCE/'analysis.json')
    rows = old.read_json(SOURCE/'comparisons.json')
    semantic.frozen.audit.selected_rows(summary, rows)
    old.require(len(rows) == 56 and Counter(r['split'] for r in rows) ==
                {'train': 32, 'validation': 24}, 'Bounded development cohort changed')
    pins = dict(old.read_json(semantic.DEFAULT_OUTPUT/'analysis.json')['pinned_files'])
    for path in (semantic.DEFAULT_OUTPUT/'review_lock.json', Path(__file__), Path(policy.__file__),
                 old.TPG/'08_tests/test_security_coreference_ablation.py'):
        pins[str(path.resolve())] = old.sha256(path)
    old.cache.check_pins(pins)
    return rows, pins


def variants(saved):
    """Validate GraphSON and tensor contracts without generating embeddings."""
    exporter, pyg = GraphSONExporter(), PyGExporter()
    graph = import_graphson(saved)
    old.require(json.loads(exporter.export_string(graph)) == saved, 'Source round trip differs')
    old.require(semantic.frozen.audit.extension.VERSION in graph.passes_applied, 'Expected fixed v4 graph')
    old.require(not any('embedding' in n.properties.extra for n in graph.nodes()), 'Expected unembedded source')
    original = pyg.export(graph, embedding_dim=0, use_security_edge_types=True)
    old.require(original['y'] is None and original['n_dropped_security_edges'] == 0, 'Invalid source export')
    outputs, counts = {}, {}
    for mode in policy.MODES:
        filtered = policy.ablate(graph, mode)
        doc = json.loads(exporter.export_string(filtered))
        expected = [edge for edge, obj in zip(saved['edges'], graph.edges()) if policy.retain(obj, mode)]
        content = semantic.frozen.audit.edge_content
        old.require(doc['vertices'] == saved['vertices'], 'Nodes changed')
        old.require([content(e) for e in doc['edges']] == [content(e) for e in expected],
                    'Retained edge content/order differs')
        payload = pyg.export(filtered, embedding_dim=0, use_security_edge_types=True)
        old.require(payload['type_vocabulary'] == original['type_vocabulary'], 'Vocabulary changed')
        old.require(payload['x'] == original['x'] and payload['y'] is None and
                    payload['n_dropped_security_edges'] == 0, 'Node/export contract changed')
        keep = [i for i, e in enumerate(graph.edges()) if policy.retain(e, mode)]
        for field in ('edge_type', 'edge_features'):
            old.require(payload[field] == [original[field][i] for i in keep], f'{field} changed')
        old.require(payload['edge_index'] == [[axis[i] for i in keep] for axis in original['edge_index']],
                    'Tensor endpoints changed')
        outputs[mode] = doc
        counts[mode] = {'nodes': len(doc['vertices']), 'edges': len(doc['edges']),
            'coref': sum(e['label'] == 'COREF' for e in doc['edges']),
            'product_version_links': sum(e['label'] == 'SEC_HAS_VERSION' for e in doc['edges']),
            'removed': len(saved['edges']) - len(doc['edges'])}
    old.require(json.loads(exporter.export_string(graph)) == saved, 'Source graph mutated')
    return outputs, counts


def build_rows(source_rows, destination=None, verify=False):
    rows = []
    for row in old.tqdm(source_rows, desc='Check three matched edge variants', unit='CVE'):
        cve = row['cve']
        old.require(Path(cve).name == cve and '/' not in cve, 'Unsafe CVE identifier')
        source = old.read_json(SOURCE/'graphs'/f'{cve}.json')
        old.require(source['doc_id'] == cve and source['metadata']['source_text'] == row['description']
                    and old.cache.text_digest(row['description']) == row['description_sha256'], 'Source differs')
        graphs, counts = variants(source)
        for mode, graph in graphs.items():
            if destination is not None:
                path = destination/'graphs'/mode/f'{cve}.json'
                if verify:
                    old.require(old.read_json(path) == graph, f'Variant does not reproduce: {cve}/{mode}')
                else:
                    old.write_json(path, graph)
        rows.append({'cve': cve, 'split': row['split'], 'description_sha256': row['description_sha256'],
                     'variants': counts})
    return rows


def statistics(rows):
    return {mode: {key: sum(r['variants'][mode][key] for r in rows)
                   for key in ('nodes', 'edges', 'coref', 'product_version_links', 'removed')}
            for mode in policy.MODES}


def report(summary):
    lines = ['DQ15: Three Ways to Use Candidate Coreference Edges', '',
        'Same fixed 56 development CVEs: 32 training-role and 24 validation-role records.',
        'These are graph checks, not training or performance results. No test records are inspected.', '',
        'no_coref: remove every COREF edge.',
        'repeated_only: retain repeated_surface candidate edges; remove pronoun candidate edges.',
        'all_coref: retain all fixed v4 edges. This is not the older v3 graph.', '',
        'Variant | Nodes kept | All edges kept | COREF edges kept | Product/version links kept | Edges removed',
        '--------|------------|----------------|------------------|----------------------------|--------------']
    for mode, s in summary['statistics'].items():
        lines.append(f"{mode} | {s['nodes']} | {s['edges']} | {s['coref']} | {s['product_version_links']} | {s['removed']}")
    lines += ['', 'What stays identical',
        'Every node, source description, other relation and retained edge property is unchanged.',
        'Edge IDs can change. Each graph records its input-to-output edge ID mapping and removed IDs.',
        'The security edge vocabulary is kept fixed, including absent relation types. No numeric type IDs are assumed.',
        'Unembedded PyG exports are checked in memory for matching features, endpoints and vocabulary.', '',
        'Limits of this comparison',
        'This isolates COREF edges only. Mention nodes, coreference-cluster properties and other derived structure remain.',
        'Repeated text means a candidate association, not verified entity identity. Equal release numbers can belong to different products.',
        'The earlier 17/20/7 semantic judgments are provisional automated assessments, not accuracy or training labels.',
        'In particular, FTP and SCM were assessed at different conceptual scopes; human review needs a consistent rubric.',
        'Five pronoun links had unsupported antecedents; another 15 reviewed links lacked product-release identity evidence.',
        'That does not establish that all 20 associations are useless or false at every interpretation level.',
        'Variants are selected by extraction method, never by review verdict, forecast errors or target values.',
        'The fixed sample includes earlier reviewed cases; it is a debugging sample, not an unbiased benchmark.',
        'Current descriptions are not proven to be the historical descriptions available on prediction Day 30.', '',
        'Next decision, after this preview is verified',
        'Prepare a separate model-ready small cache with shared node embeddings and these three edge masks.',
        'Do not reuse old trained weights as if they had been trained on the corrected graphs.',
        'A later matched training comparison should keep cohort, targets, EPSS history, architecture, seeds, optimizer,',
        'budget and validation rules identical. Include the EPSS-only and persistence references on identical CVEs.',
        'No performance benefit is assumed. Full expansion and training need a separate decision.', '',
        'Files',
        'analysis.json: provenance, per-CVE counts and aggregate checks. analysis.txt: this explanation.',
        'graphs/<variant>/<CVE>.json: source-preserving unembedded GraphSON variants.',
        'preview_lock.json: checksums. run_status.json: completion state.',
        'No source files, frozen v4 code, earlier graphs, embeddings, models or results are overwritten.', '']
    return '\n'.join(lines)


def verify_output(output, source_rows, pins):
    lock = old.read_json(output/'preview_lock.json')
    old.require(lock['schema'] == SCHEMA, 'Wrong preview schema')
    old.require({str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()} ==
                set(lock['files']) | {'preview_lock.json'}, 'Output file set differs')
    for name, digest in lock['files'].items():
        old.require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe artifact path')
        old.require(old.sha256(output/name) == digest, f'Changed artifact: {name}')
    summary = old.read_json(output/'analysis.json')
    old.require(summary['schema'] == SCHEMA and summary['pinned_files'] == pins, 'Preview identity differs')
    old.require(old.read_json(output/'run_status.json')['status'] == 'complete', 'Incomplete preview')
    rows = build_rows(source_rows, output, verify=True)
    old.require(summary['rows'] == rows and summary['statistics'] == statistics(rows), 'Counts do not reproduce')
    old.require((output/'analysis.txt').read_text(encoding='utf-8') == report(summary), 'Text report differs')
    old.cache.check_pins(pins)


def run(output, check_only=False, verify_only=False):
    output = Path(output).resolve()
    with old.tqdm(total=1, desc='Verify frozen v4 and bounded review', unit='stage') as bar:
        source_rows, pins = source_context(); bar.update()
    if verify_only:
        verify_output(output, source_rows, pins)
        print('DQ15 saved variants, counts and source checksums verified. No training.'); return
    # Only this new workspace may receive preview artifacts; frozen folders stay protected.
    root = DEFAULT_OUTPUT.parent.resolve()
    old.require(root in output.parents, 'Output must be a new folder inside coreference_ablation_v1')
    old.cache.check_destination(output, [SOURCE.parent, semantic.ROOT, old.TPG/'01_core',
        old.TPG/'10_extensions', old.cache.CACHE_ROOT, old.cache.PREPARED])
    if check_only:
        rows = build_rows(source_rows)
        old.cache.check_pins(pins)
        print(f'DQ15 preflight passed: {len(rows)} fixed development CVEs; no output created.'); return
    output.mkdir(parents=True, exist_ok=False)
    old.write_json(output/'run_status.json', {'status': 'running'})
    try:
        for mode in policy.MODES:
            (output/'graphs'/mode).mkdir(parents=True)
        rows = build_rows(source_rows, output)
        summary = {'schema': SCHEMA, 'pinned_files': pins, 'rows': rows, 'statistics': statistics(rows),
            'training': False, 'test_inspection': False, 'embeddings_generated': False,
            'nlp_rerun': False, 'model_ready_cache': False, 'full_expansion_authorized': False}
        with old.tqdm(total=3, desc='Save reports and verify provenance', unit='stage') as bar:
            old.write_json(output/'analysis.json', summary)
            (output/'analysis.txt').write_text(report(summary), encoding='utf-8'); bar.update()
            old.cache.check_pins(pins); bar.update()
            old.write_json(output/'run_status.json', {'status': 'complete'})
            old.write_json(output/'preview_lock.json', {'schema': SCHEMA, 'files': {
                str(p.relative_to(output)): old.sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}})
            bar.update()
        verify_output(output, source_rows, pins)
    except Exception as exc:
        old.write_json(output/'run_status.json', {'status': 'failed', 'error': str(exc)})
        raise
    print(f'DQ15 matched COREF preview saved to {output}. No training, embeddings or test inspection.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    old.torch.set_num_threads(1)
    run(args.output_dir, args.check_only, args.verify_only)
