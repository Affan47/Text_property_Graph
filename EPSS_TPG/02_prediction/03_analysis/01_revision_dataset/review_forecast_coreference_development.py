"""Review verified DQ17 v2 cache coverage only; no predictions or training."""
import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from tqdm.auto import tqdm

EPSS = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(EPSS/'02_prediction/01_package'))
from epss import forecast_coreference_development_v2 as cache

DEFAULT_OUTPUT = EPSS/'05_results/02_evaluation/epss_day30_to_day60/coreference_development_v2_review'


def distribution(values):
    return {'min': min(values), 'median': float(np.median(values)),
            'mean': float(np.mean(values)), 'p95': float(np.percentile(values, 95)), 'max': max(values)}


def summarize(manifest, source):
    by_id = {r['cve']: r for r in manifest['graphs']}
    node_types, edge_types, embedded = Counter(), Counter(), Counter()
    rows, removed = [], 0
    for split in cache.ROLES:
        for cve in tqdm(manifest['ids'][split], desc=f'Inspect {split} graph coverage', unit='CVE'):
            saved = cache.read_json(source/'graphson/all_coref'/(cve+'.json'))
            stats = by_id[cve]
            nodes = Counter(n['label'] for n in saved['vertices'])
            edges = Counter(e['label'] for e in saved['edges'])
            cache.require(sum(nodes.values()) == stats['nodes'] and
                          sum(edges.values()) == stats['edges_by_mode']['all_coref'], 'Counts differ')
            node_types.update(nodes); edge_types.update(edges)
            embedded.update(stats['embedded_by_node_type'])
            methods = Counter(e['properties']['EXTRA']['method'] for e in saved['edges'] if e['label'] == 'COREF')
            counts = stats['edges_by_mode']
            cache.require(edges['COREF'] == counts['all_coref']-counts['no_coref'] and
                          methods['repeated_surface'] == counts['repeated_only']-counts['no_coref'] and
                          methods['unambiguous_type_heuristic'] == counts['all_coref']-counts['repeated_only'],
                          'COREF counts differ from masks')
            removed_here = saved['metadata']['coreference_filter']['removed_count']
            removed += removed_here
            rows.append({'cve': cve, 'split': split, 'nodes': stats['nodes'],
                'embedded_nodes': stats['embedded_nodes'], 'unembedded_nodes': stats['unembedded_nodes'],
                'edges': dict(counts), 'coref': edges['COREF'], 'repeated_coref': methods['repeated_surface'],
                'pronoun_coref': methods['unambiguous_type_heuristic'],
                'typed_security_edges': sum(n for name,n in edges.items() if name.startswith('SEC_')),
                'product_version_edges': edges['SEC_HAS_VERSION'], 'v4_removed_edges': removed_here,
                'missing_whitespace_tokens': stats['unembedded_whitespace_tokens'],
                'missing_replacement_tokens': stats['unembedded_replacement_tokens_count']})
    groups = {}
    for split in (*cache.ROLES, 'all'):
        selected = [r for r in rows if split == 'all' or r['split'] == split]
        groups[split] = {'cves': len(selected),
            **{key: sum(r[key] for r in selected) for key in ('nodes', 'embedded_nodes', 'unembedded_nodes',
                'coref', 'repeated_coref', 'pronoun_coref', 'typed_security_edges', 'product_version_edges',
                'missing_whitespace_tokens', 'missing_replacement_tokens')},
            'edges_by_mode': {m: sum(r['edges'][m] for r in selected) for m in cache.shared.MODES},
            'cves_with': {k: sum(r[k] > 0 for r in selected) for k in
                         ('coref', 'repeated_coref', 'pronoun_coref', 'typed_security_edges', 'product_version_edges')},
            'nodes_per_cve': distribution([r['nodes'] for r in selected]),
            'all_coref_edges_per_cve': distribution([r['edges']['all_coref'] for r in selected])}
    node_rows = {kind: {'total': node_types[kind], 'embedded': embedded[kind],
                       'unembedded': node_types[kind]-embedded[kind]}
                 for kind in manifest['type_vocabulary']['node_types']}
    cache.require(sum(r['unembedded'] for r in node_rows.values()) == groups['all']['unembedded_nodes'],
                  'Unembedded-node accounting differs')
    return {'schema': 'dq17_v2_cache_coverage_review_v1', 'groups': groups, 'node_types': node_rows,
            'all_coref_edge_types': dict(sorted(edge_types.items())), 'v4_removed_same_occurrence_edges': removed,
            'replacement_cases': {c: v for c,v in manifest['unembedded_replacement_tokens'].items() if v},
            'reuse': {'dq16': manifest['reused_dq16'], 'v1': manifest['reused_dq17_v1'],
                     'overlap': manifest['reused_dq16']+manifest['reused_dq17_v1']-manifest['reused_total'],
                     'distinct_reused': manifest['reused_total'],
                     'newly_built': len(rows)-manifest['reused_total']},
            'largest_graphs': sorted(rows, key=lambda r: (-r['nodes'], r['cve']))[:10], 'rows': rows,
            'embedding_dim': manifest['embedding_dim'],
            'node_feature_width': manifest['type_vocabulary']['num_node_types']+manifest['embedding_dim']+1,
            'training': False, 'test_scoring': False,
            'limitations': [cache.base.LIMITATION,
                'Coverage is not semantic accuracy or evidence of prediction improvement.',
                'COREF changes only explicit edges; mention nodes and derived structure remain.',
                'Generic copies of security relations can coexist with typed SEC_ edges.',
                'Repeated validation inspection limits confirmatory interpretation.']}


def report(result):
    total = result['groups']['all']
    lines = ['DQ17 v2: Verified Full Development Cache', '',
        'This is a graph/data-integrity review, not a model evaluation.',
        'The frozen verifier checks source pins, tensor contents, masks, source text and artifact hashes.',
        f"CVEs: {total['cves']}; training: {result['groups']['train']['cves']}; validation: {result['groups']['validation']['cves']}.",
        f"Nodes: {total['nodes']}; embedded: {total['embedded_nodes']}; unembedded: {total['unembedded_nodes']}.",
        f"Feature width: {result['node_feature_width']} = node-type indicators + {result['embedding_dim']} embedding values + presence mask.", '',
        'Configuration | Same nodes | Total edges | COREF edges',
        *[f"{mode} | {total['nodes']} | {total['edges_by_mode'][mode]} | {total['edges_by_mode'][mode]-total['edges_by_mode']['no_coref']}"
          for mode in cache.shared.MODES], '',
        'Coverage counts CVEs with at least one edge of the named kind; these groups overlap.',
        'Split | Any COREF | Repeated-text COREF | Pronoun COREF | Typed security | Product/version']
    for split in (*cache.ROLES, 'all'):
        c = result['groups'][split]['cves_with']
        lines.append(f"{split} | {c['coref']} | {c['repeated_coref']} | {c['pronoun_coref']} | {c['typed_security_edges']} | {c['product_version_edges']}")
    lines += ['', 'Embedding coverage by node type: total / embedded / unembedded',
              *[f"{k}: {v['total']} / {v['embedded']} / {v['unembedded']}" for k,v in result['node_types'].items()],
              '', f"Unembedded whitespace tokens: {total['missing_whitespace_tokens']}.",
              f"Unembedded U+FFFD tokens: {total['missing_replacement_tokens']}; CVEs: {', '.join(result['replacement_cases'])}.",
              'U+FFFD characters remain in the source and graph, with zero embedding and mask zero.',
              'Ordinary text tokens must be embedded. Zero structural embeddings are expected, not discarded nodes.',
              '', 'Reuse: '+str(result['reuse']),
              'The overlap is counted once in distinct_reused. Verified prior tensor files are copied unchanged.',
              '', 'Graph sizes (minimum, median, mean, 95th percentile, maximum)',
              'Nodes per CVE: '+str(total['nodes_per_cve']),
              'All-COREF edges per CVE: '+str(total['all_coref_edges_per_cve']),
              '', 'Interpretation',
              'The three configurations are ready as matched inputs for the predefined development experiment.',
              'No COREF winner can be selected from cache coverage. Only later matched validation runs can',
              'measure whether the edge differences help. Compare persistence and EPSS-only on the same CVEs.',
              'Inspect large-graph memory use in the training preflight; do not silently truncate or drop CVEs.',
              'R03 is still planned, not trained: three seeds, up to 100 epochs, patience 15.',
              '', *result['limitations'], '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = cache.ROOT/'full'
    output = args.output_dir.resolve()
    cache.require(DEFAULT_OUTPUT.parent.resolve() in output.parents, 'Use evaluation-results directory')
    cache.base.check_destination(output, [source, cache.ROOT, cache.base.PREPARED])
    cache.torch.set_num_threads(1)
    manifest = cache.verify(source, expected_scale='full')
    lock_digest = cache.sha256(source/'cache_lock.json')
    result = summarize(manifest, source)
    result.update(cache_directory=str(source), cache_lock_sha256=lock_digest,
                  review_script_sha256=cache.sha256(__file__), checks_passed=True)
    cache.require(cache.sha256(source/'cache_lock.json') == lock_digest, 'Source lock changed during review')
    output.mkdir(parents=True, exist_ok=False)
    cache.write_json(output/'analysis.json', result)
    (output/'analysis.txt').write_text(report(result), encoding='utf-8')
    cache.write_json(output/'review_lock.json', {'files': {n: cache.sha256(output/n)
                      for n in ('analysis.json', 'analysis.txt')}})
    print(f'DQ17 v2 review saved to {output}. No encoder rerun, training or test scoring.')


if __name__ == '__main__':
    main()
