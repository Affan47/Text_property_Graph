"""Review CD03 v2 hashes and saved statistics, without expensive tensor replay."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss import forecast_identity_graphs_v2 as cache

OUTPUT = cache.OUTPUT / 'completed_review_v1'


def analyze():
    ctx = cache.context('full')
    root = cache.OUTPUT / 'full'
    lock = cache.read(root / 'cache_lock.json')
    require = cache.require
    require(lock['schema'] == cache.SCHEMA and set(lock['files']) == cache.expected(ctx), 'Wrong locked file set')
    require({str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()} == set(lock['files']) | {'cache_lock.json'},
            'Unexpected/missing cache files')
    for name, digest in cache.tqdm(lock['files'].items(), desc='Recheck cache file hashes (no tensor replay)', unit='file'):
        require(cache.sha256(root / name) == digest, 'Changed artifact: ' + name)
    require(cache.read(root / 'run_identity.json') == ctx['identity'] and
            cache.read(root / 'run_status.json') == {'status': 'complete'} and
            cache.read(root / 'cve_index.json') == ctx['index'], 'Cache identity/index/status differs')
    saved = cache.read(root / 'analysis.json')
    rows = saved['chunks']
    require(len(rows) == len(ctx['entries']) and {r['key'] for r in rows} == set(ctx['entries']), 'Wrong chunk statistics')
    require(cache.summary(ctx, rows) == saved, 'Aggregate statistics do not reproduce')
    docs, cves = {}, defaultdict(lambda: Counter(nodes=0, chunks=0, edges=0))
    types, reused = Counter(), Counter()
    for row in cache.tqdm(rows, desc='Review chunk statistics and receipts', unit='chunk'):
        entry = ctx['entries'][row['key']]
        require(all(row[k] == entry[k] for k in ('cve', 'split', 'document_id', 'source_text_type', 'text_sha256')) and
                row['characters'] == len(entry['text']), 'Wrong chunk source identity')
        require(row['nodes'] == row['embedded_nodes'] + row['unembedded_nodes'] and
                sum(row['embedded_by_node_type'].values()) == row['embedded_nodes'] and
                sum(row['edge_type_counts'].values()) == row['edges_no_coref'], 'Inconsistent graph counts')
        require(row['source_span_tokens'] - row['embedded_by_node_type']['TOKEN'] ==
                row['unembedded_whitespace_tokens'] + len(row['unembedded_replacement_tokens']) + len(row['unembedded_control_tokens']),
                'Unaccounted missing token embeddings in saved statistics')
        types.update({k: v for k, v in row['edge_type_counts'].items() if k.startswith('SEC_')})
        key = (row['cve'], row['document_id'])
        d = docs.setdefault(key, {k: row[k] for k in ('cve', 'split', 'document_id', 'source_text_type')})
        for field in ('characters', 'nodes'):
            d[field] = d.get(field, 0) + row[field]
        d['chunks'] = d.get('chunks', 0) + 1
        cves[row['cve']].update(nodes=row['nodes'], chunks=1, edges=row['edges_no_coref'])
        receipt_path = root / 'receipts' / (row['key'] + '.json')
        receipt = cache.read(receipt_path)
        if receipt.get('embedding_coverage_policy') is None:
            original = cache.base.OUTPUT / 'full' / 'receipts' / receipt_path.name
            require(cache.sha256(original) == cache.sha256(receipt_path), 'Reused receipt differs from v1')
            require(all(lock['files'][name] == digest for name, digest in receipt['files'].items()), 'Reused file hash differs')
            reused['v1_reused_chunks'] += 1
        else:
            require(receipt['embedding_coverage_policy'] == cache.coverage.POLICY, 'Unknown coverage policy')
            reused['v2_built_chunks'] += 1
    require(len(docs) == len(ctx['documents']), 'Document coverage differs')
    totals = {k: sum(r[k] for r in rows) for k in
              ('nodes', 'embedded_nodes', 'unembedded_nodes', 'edges_no_coref', 'security_edges')}
    coverage = []
    for r in saved['coverage']:
        subset = [d for d in rows if d['split'] == r['split'] and d['source_text_type'] == r['source']]
        per_cve = Counter()
        for d in subset:
            per_cve[d['cve']] += d['nodes']
        split_nodes = sum(d['nodes'] for d in rows if d['split'] == r['split'])
        coverage.append({**r, 'coverage_percent': 100 * r['cves_with_graphs'] / r['cves'],
                         'split_node_share_percent': 100 * r['nodes'] / split_nodes,
                         'cves_with_security_edges': len({d['cve'] for d in subset if d['security_edges']}),
                         'median_nodes_per_present_cve': median(per_cve.values()),
                         'max_nodes_per_present_cve': max(per_cve.values())})
    missing = {'whitespace_tokens': sum(r['unembedded_whitespace_tokens'] for r in rows),
               'replacement_tokens': sum(len(r['unembedded_replacement_tokens']) for r in rows),
               'control_tokens': saved['control_token_count']}
    missing['other_unembedded_nodes'] = totals['unembedded_nodes'] - sum(missing.values())
    return {'schema': 'cd03_v2_completed_review_v1', 'source_pins': {
        str(root / 'cache_lock.json'): cache.sha256(root / 'cache_lock.json'),
        str(Path(__file__).resolve()): cache.sha256(Path(__file__).resolve())},
        'verification_scope': 'All file hashes and saved statistics rechecked; user completed full tensor/graph verification. No tensor replay here.',
        'locked_files': len(lock['files']), 'cohort': {s: len(ids) for s, ids in ctx['identity']['ids'].items()},
        'documents': len(docs), 'chunks': len(rows), 'totals': totals, 'coverage': coverage,
        'reuse': dict(reused), 'missing_embeddings': missing, 'controls': saved['unembedded_control_tokens'],
        'security_edge_types': dict(types), 'cves_with_security_edges': len({r['cve'] for r in rows if r['security_edges']}),
        'largest_documents': sorted(docs.values(), key=lambda d: d['nodes'], reverse=True)[:10],
        'largest_cves': [{'cve': c, **d} for c, d in sorted(cves.items(), key=lambda x: x[1]['nodes'], reverse=True)[:10]],
        'forced_boundaries': saved['forced_boundaries'], 'non_graph_segments': saved['non_graph_segments'],
        'training': False, 'test_scoring': False,
        'next_step': 'Matched EPSS-only, description-plus-EPSS and all-source-plus-EPSS fusion development; document/source-aware pooling, one integration smoke, then fixed multi-seed full development fits.',
        'limitations': ['Graph validity is not semantic accuracy or evidence of predictive improvement.',
                       'Descriptions/summaries remain retrospective; stored post dates alone do not authenticate availability.',
                       'A long document is not many independent CVEs; keep chunks nested within documents and sources.',
                       'Earlier H02/R07 metrics are not matched controls for this changed cohort.',
                       'No per-node tensor validation was repeated during this report review.']}


def report(a):
    lines = ['CD03 v2 completed multi-source cache review', '', a['verification_scope'], '',
             f"Cohort: {a['cohort']}; documents: {a['documents']}; graph chunks: {a['chunks']}",
             f"Totals (no-COREF edge view): {a['totals']}", f"Reuse: {a['reuse']}", '',
             'Split | Source | CVEs with graphs | Coverage % | Nodes | Split node share % | CVEs with security edges']
    for r in a['coverage']:
        lines.append(' | '.join(str(round(r[k], 2)) if isinstance(r[k], float) else str(r[k]) for k in
            ('split', 'source', 'cves_with_graphs', 'coverage_percent', 'nodes', 'split_node_share_percent', 'cves_with_security_edges')))
    lines += ['', 'Embedding coverage', str(a['missing_embeddings']),
              'All missing TOKEN embeddings in the saved statistics are accounted for by whitespace, replacement or allowlisted controls.',
              'Other unembedded nodes are not automatically missing word embeddings; structural/span availability differs by node type.',
              '', 'Security relations', str(a['security_edge_types']),
              f"{a['cves_with_security_edges']} CVEs have at least one typed security edge. Counts are coverage, not correctness or model performance.",
              f"SEC_HAS_VERSION contributes {100 * a['security_edge_types'].get('SEC_HAS_VERSION', 0) / a['totals']['security_edges']:.2f}% of typed security edges.",
              'The security component is mostly product/version structure, not a broad set of verified exploitation claims.',
              '', 'Source-size imbalance']
    for r in a['coverage']:
        if r['source'] == 'smp':
            lines.append(f"{r['split']}: SMP accounts for {r['split_node_share_percent']:.2f}% of graph nodes.")
    lines += ['Do not concatenate or average every node across all sources indiscriminately. Pool chunks within each document, then documents within each source, with explicit source masks.',
              'Shared posts can discuss several distinct vulnerabilities. This does not make their CVE identities duplicates.',
              'No CVEs or sources were removed by this review.',
              '', 'Largest source documents (counts per CVE association)']
    lines += [str(r) for r in a['largest_documents']]
    lines += ['', 'Next step', a['next_step'], '', 'Limitations'] + a['limitations']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    a = analyze()
    if args.verify_only:
        cache.require(cache.read(OUTPUT / 'analysis.json') == a and (OUTPUT / 'analysis.txt').read_text() == report(a),
                      'Saved review differs')
    else:
        cache.write(OUTPUT / 'analysis.json', a)
        (OUTPUT / 'analysis.txt').write_text(report(a), encoding='utf-8')
    print(f'CD03 v2 completed review {"reproduced" if args.verify_only else "saved"}: {OUTPUT}. No tensor replay, training or test scoring.')


if __name__ == '__main__':
    main()
