"""Read-only full-cache inventory/security review and bounded tensor revalidation."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss import forecast_primary_cache_v2 as cache
from epss.primary_study import common

read, require, tqdm = common.read, common.require, common.tqdm


def distribution(values):
    import numpy as np
    a = np.asarray(values)
    return {'count': len(values), 'sum': int(a.sum()), 'min': int(a.min()),
            'median': float(np.median(a)), 'p95': float(np.percentile(a, 95)),
            'max': int(a.max())}


def review():
    cache.torch.set_num_threads(1)
    directory = cache.OUTPUT / 'full'
    lock = common.verify_lock(directory)
    plan, identity, resources = cache.context('full')
    require(read(directory / 'run_identity.json') == identity, 'Identity changed')
    require(read(directory / 'resources.json') == resources, 'Resources changed')
    require(read(directory / 'cve_index.json') == plan['index'], 'Index changed')
    common.check_pins(identity['source_pins'])
    common.check_pins(resources['source_pins'])
    provenance = read(directory / 'chunk_provenance.json')
    require(set(provenance) == set(plan['entries']), 'Chunk set differs')
    audit = read(directory / 'coverage_audit.json')
    rows = {r['key']: r for r in audit['chunks']}
    require(len(rows) == len(audit['chunks']) and set(rows) == set(plan['entries']), 'Audit coverage differs')
    sources = resources['protocol']['source_order']
    coverage = {s: {source: 0 for source in sources} for s in ('train', 'validation')}
    documents = Counter()
    cve_sizes = []
    for cve, record in plan['index'].items():
        require(record['split'] in coverage, 'Unexpected test role')
        for i, source in enumerate(sources):
            coverage[record['split']][source] += record['source_mask'][i]
        chunks = [ch for d in record['documents'] for ch in d['chunks'] if ch['has_graph']]
        documents.update(d['source_text_type'] for d in record['documents'])
        cve_sizes.append({'cve': cve, 'split': record['split'], 'chunks': len(chunks),
                          'nodes': sum(rows[ch['key']]['nodes'] for ch in chunks)})
    for name, doc in tqdm(plan['documents'].items(), desc='Match unchanged source documents', unit='document'):
        require(read(directory / name) == doc, 'Source document differs: ' + name)
    by_source = defaultdict(list)
    counts, methods, polarities, modalities = Counter(), Counter(), Counter(), Counter()
    security_cves = set()
    examples = defaultdict(list)
    missing = Counter()
    affected = defaultdict(set)
    for key, entry in tqdm(plan['entries'].items(), desc='Review receipts and security graphs', unit='chunk'):
        row = rows[key]
        receipt = read(directory / 'receipts' / (key + '.json'))
        require(cache.old.reusable(receipt, entry), 'Receipt identity differs')
        saved = dict(receipt['statistics'])
        # Strict pre-control receipts predate both explicit exception lists.
        saved.setdefault('unembedded_control_tokens', [])
        saved.setdefault('unembedded_nonspacing_tokens', [])
        require(saved == row, 'Receipt/audit mismatch')
        for path, value in receipt['files'].items():
            require(lock['files'].get(path) == value, 'Receipt/lock hash mismatch')
        graph = read(directory / 'tpg_json/no_coref' / (entry['stem'] + '.json'))
        edge_counts = Counter(e['label'] for e in graph['edges'])
        require(dict(edge_counts) == row['edge_type_counts'], 'Graph/audit edge mismatch')
        require(len(graph['vertices']) == row['nodes'] and len(graph['edges']) == row['edges_no_coref'],
                'Graph/audit size mismatch')
        require(row['embedded_nodes'] + row['unembedded_nodes'] == row['nodes'], 'Embedding count mismatch')
        nodes = {n['id']: n for n in graph['vertices']}
        sec = [e for e in graph['edges'] if e['label'].startswith('SEC_')]
        require(len(sec) == row['security_edges'], 'Security total differs')
        if sec:
            security_cves.add(entry['cve'])
        for edge in sec:
            extra = edge['properties'].get('EXTRA', {})
            require(extra.get('source_text_type') == entry['source_text_type'], 'Lost security source label')
            require(extra.get('status') == 'candidate', 'Security edge lacks candidate status')
            counts[edge['label']] += 1
            methods[extra.get('method', 'not_recorded')] += 1
            polarities[extra.get('polarity', 'not_recorded')] += 1
            modalities[extra.get('modality', 'not_recorded')] += 1
            if len(examples[edge['label']]) < 3:
                examples[edge['label']].append({'cve': entry['cve'], 'source': entry['source_text_type'],
                    'source_text': nodes[edge['outV']]['properties']['TEXT'],
                    'target_text': nodes[edge['inV']]['properties']['TEXT'], 'evidence': extra})
        by_source[entry['source_text_type']].append(row)
        for field in ('unembedded_control_tokens', 'unembedded_replacement_tokens', 'unembedded_nonspacing_tokens'):
            missing[field] += len(row[field])
            if row[field]:
                affected[field].add(entry['cve'])
    require(dict(counts) == audit['security_edge_counts'], 'Security audit does not reproduce')
    # Deterministic coverage of every source/role, largest graph, and each exceptional token class.
    selected = {cache.FAILURE, max(rows, key=lambda k: rows[k]['nodes'])}
    for source in sources:
        for split in coverage:
            selected.add(min(k for k, r in rows.items() if r['source_text_type'] == source and r['split'] == split))
    for field in missing:
        candidates = [k for k, r in rows.items() if r[field]]
        if candidates:
            selected.add(min(candidates))
    tokenizer = cache.tokenizer_for(resources)
    for key in tqdm(sorted(selected), desc='Revalidate bounded tensor sample', unit='chunk'):
        _, row = cache.check_entry(directory, plan['entries'][key], resources, tokenizer)
        require(row == rows[key], 'Tensor sample audit differs')
    source_stats = {}
    for source, items in by_source.items():
        source_stats[source] = {'documents': documents[source], 'chunks': len(items),
            'nodes': sum(r['nodes'] for r in items),
            'edges_no_coref': sum(r['edges_no_coref'] for r in items),
            'security_edges': sum(r['security_edges'] for r in items),
            'chunks_with_security_edges': sum(r['security_edges'] > 0 for r in items),
            'cves_with_security_edges': len({r['cve'] for r in items if r['security_edges']}),
            'nodes_per_chunk': distribution([r['nodes'] for r in items])}
    result = {'schema': 'ps01_full_cache_review_v1',
        'verification': {'all_artifact_hashes_checked': len(lock['files']),
            'index_and_source_documents_reproduced': True, 'all_receipts_and_no_coref_graph_counts_checked': len(rows),
            'all_security_edges_provenance_checked': sum(counts.values()),
            'tensor_semantic_recheck_keys': sorted(selected),
            'full_tensor_semantic_recheck_this_review': False,
            'scope_note': 'Build performed full tensor validation; this independent review repeats it only for the listed sample. All saved files are hash-checked.'},
        'cves': dict(Counter(r['split'] for r in plan['index'].values())),
        'documents': len(plan['documents']), 'chunks': len(rows),
        'provenance': dict(Counter(provenance.values())), 'source_coverage': coverage,
        'source_statistics': source_stats,
        'total_nodes': sum(r['nodes'] for r in rows.values()),
        'embedded_nodes': sum(r['embedded_nodes'] for r in rows.values()),
        'total_edges_no_coref': sum(r['edges_no_coref'] for r in rows.values()),
        'security_edge_counts': dict(counts), 'security_cves': len(security_cves),
        'security_methods': dict(methods), 'security_polarities': dict(polarities),
        'security_modalities': dict(modalities), 'security_examples': dict(examples),
        'missing_token_counts': dict(missing), 'affected_cve_counts': {k: len(v) for k, v in affected.items()},
        'nonspacing_cases': audit['removed_nonspacing_tokens'],
        'chunks_per_cve': distribution([r['chunks'] for r in cve_sizes]),
        'nodes_per_cve': distribution([r['nodes'] for r in cve_sizes]),
        'largest_cves_by_chunks': sorted(cve_sizes, key=lambda r: (-r['chunks'], r['cve']))[:10],
        'training': False, 'test_scoring': False,
        'limitations': ['Candidate edge counts do not measure semantic accuracy.',
            'SMP stored documents may aggregate posts or contain scraped/garbled material.',
            'Node/edge totals count chunk occurrences, not unique real-world facts.',
            'Current source text and CVSS are retrospective, not proven Day-60-available inputs.',
            'Cache completion establishes readiness, not prediction quality.']}
    output = common.RESULTS / 'cache_v2_full_review'
    lines = ['PS01 full cache review', '', 'Verification scope', str(result['verification']), '',
        'Dataset and graph totals', f"CVEs: {result['cves']}; documents: {result['documents']}; chunks: {result['chunks']}",
        f"Reused/new: {result['provenance']}", f"Source coverage: {coverage}",
        f"Nodes: {result['total_nodes']}; embedded nodes: {result['embedded_nodes']}; edges (no COREF): {result['total_edges_no_coref']}",
        '', 'Source-by-source graph statistics', *[f'{k}: {v}' for k, v in source_stats.items()], '',
        'Security candidates', str(dict(counts)), f'CVEs with any security edge: {len(security_cves)}',
        '', 'Missing-token exceptions', str(result['missing_token_counts']), str(result['affected_cve_counts']),
        '', 'Compute requirements', f"Chunks per CVE: {result['chunks_per_cve']}",
        f"Largest records: {result['largest_cves_by_chunks']}", '', 'Limitations', *result['limitations'], '',
        'Next: version-aware, memory-bounded matched primary-study training runner; no further cache expansion needed.',
        'Do not use the old v1 smoke reader unchanged for the new v2 full cache.',
        'No training, embeddings, raw-data edits or test scoring performed.']
    output.mkdir(parents=True, exist_ok=True)
    common.write_report(output, result, lines)
    print(f'PS01 full cache review saved to {output}. No training or test scoring.')


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    review()
