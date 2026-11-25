"""Review the completed MS02 cache without NLP, embedding or model execution."""
import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss import forecast_multisource_graphs as preview

OUTPUT = preview.OUTPUT.parent / 'preview_review_v1'


def analyze():
    identity, _ = preview.verify()
    read, require = preview.read, preview.require
    saved = read(preview.OUTPUT / 'analysis.json')
    docs = saved['documents']
    node_types, embedded, security = Counter(), Counter(), Counter()
    for d in docs:
        graph = read(preview.OUTPUT / 'graphson/no_coref' / (d['key'] + '.json'))
        node_types.update(n['label'] for n in graph['vertices'])
        embedded.update(d['embedded_by_node_type'])
        security.update({k: v for k, v in d['edge_type_counts'].items() if k.startswith('SEC_')})
    unembedded = {k: v - embedded[k] for k, v in node_types.items() if v != embedded[k]}
    require(all(n >= 0 for n in unembedded.values()), 'Invalid embedding counts')
    totals = {k: sum(d[k] for d in docs) for k in
              ('nodes', 'embedded_nodes', 'unembedded_nodes', 'edges_no_coref', 'security_edges')}
    require(sum(unembedded.values()) == totals['unembedded_nodes'], 'Node accounting differs')
    require(sum(security.values()) == totals['security_edges'], 'Security edge accounting differs')
    full = preview.development.OUTPUT
    manifest = read(full / 'split_manifest.json')
    cohort = read(full / 'cohort_audit.json')
    source = read(full / 'source_coverage.json')
    counts = Counter(r['split'] for r in manifest.values())
    eligible = Counter(r['split'] for r in cohort.values() if r['matched_eligible'])
    require(all(r['split'] in ('train', 'validation') for r in cohort.values()), 'Unexpected test audit')
    accounting = {'eligible_training': eligible['train'], 'eligible_validation': eligible['validation'],
                  'reserved_test_not_eligibility_scored': counts['test'],
                  'excluded_by_split_policy': counts['excluded'],
                  'development_missing_required_input_or_target': sum(not r['matched_eligible'] for r in cohort.values())}
    require(sum(accounting.values()) == len(manifest), 'Dataset accounting differs')
    full_docs = Counter()
    for r in source['coverage']:
        full_docs[r['split']] += r['date_screened_documents']
    descriptions = sum(d['nodes'] for d in docs if d['source_text_type'] == 'description')
    largest = max(docs, key=lambda d: d['nodes'])
    pins = {str(preview.OUTPUT / 'cache_lock.json'): preview.sha256(preview.OUTPUT / 'cache_lock.json'),
            str(Path(__file__).resolve()): preview.sha256(Path(__file__).resolve())}
    for name in ('split_manifest.json', 'cohort_audit.json', 'source_coverage.json', 'preparation_lock.json'):
        pins[str(full / name)] = preview.sha256(full / name)
    return {'schema': 'ms02_completed_preview_review_v1', 'source_pins': pins,
            'preview_cves': {s: len(ids) for s, ids in identity['ids'].items()},
            'documents': len(docs), 'totals': totals, 'coverage': saved['coverage'],
            'unembedded_by_node_type': unembedded,
            'unembedded_whitespace_tokens': sum(d['unembedded_whitespace_tokens'] for d in docs),
            'unembedded_replacement_tokens': sum(len(d['unembedded_replacement_tokens']) for d in docs),
            'typed_security_edges': dict(security),
            'documents_with_security_edges': sum(d['security_edges'] > 0 for d in docs),
            'cves_with_security_edges': len({d['cve'] for d in docs if d['security_edges'] > 0}),
            'all_sources_to_description_node_ratio': totals['nodes'] / descriptions,
            'largest_document': {k: largest[k] for k in ('cve', 'source_text_type', 'characters', 'nodes', 'edges_no_coref')},
            'dataset_total': len(manifest), 'dataset_accounting': accounting,
            'split_exclusions': dict(Counter(r['exclusion_reason'] for r in manifest.values() if r['split'] == 'excluded')),
            'full_source_coverage': source['coverage'], 'full_candidate_documents': dict(full_docs),
            'next_step': 'Full matched development cache and fixed source-aware fusion training protocol; one integration smoke before full fits.',
            'limitations': ['Cache verification does not establish predictive improvement or semantic correctness.',
                           'Stored-date masks do not authenticate historical text availability.',
                           'Full multi-source cache and fusion runner are not implemented by MS02.',
                           'Sparse typed security edges are coverage observations, not extractor accuracy.',
                           'Reserved test count is split assignment, not verified target eligibility.'],
            'training': False, 'test_scoring': False}


def report(a):
    t = a['totals']
    lines = ['MS02 completed multi-source preview: findings and next training stage', '',
             'The original cache passes hash, graph, embedding, vocabulary and source-mask verification.',
             'No parser, encoder or trained prediction model was executed. No test scores were inspected.', '',
             '1. What is ready?',
             f"{a['preview_cves']['train']} training and {a['preview_cves']['validation']} validation CVEs; {a['documents']} independent source documents.",
             f"{t['nodes']} nodes; {t['embedded_nodes']} with contextual embeddings; {t['unembedded_nodes']} without them.",
             f"{t['edges_no_coref']} edges in the no-COREF view; {t['security_edges']} typed security edges.",
             'Split | Source | CVEs with graphs | Documents | Nodes | Edges | Typed security edges']
    for r in a['coverage']:
        lines.append(' | '.join(str(r[k]) for k in ('split', 'source', 'cves_with_graphs', 'documents', 'nodes', 'edges', 'security_edges')))
    lines += ['', '2. Interpretation',
              'Missing sources preserve the CVE and use an explicit source mask, not a fake graph.',
              'All ten source-routing views retain the same 24 CVEs. They are not trained ablations.',
              'There are no GitHub-summary graphs in preview validation; full validation has 51 such CVEs.',
              f"Nodes without contextual embeddings by type: {a['unembedded_by_node_type']}.",
              f"Unembedded whitespace tokens: {a['unembedded_whitespace_tokens']}; replacement-character tokens: {a['unembedded_replacement_tokens']}.",
              'Embedding-present masks distinguish these nodes from embedded ones; node-type/graph structure remains available.',
              f"Typed security edges: {a['typed_security_edges']}; present in {a['documents_with_security_edges']} documents across {a['cves_with_security_edges']} CVEs.",
              'The graph is mostly linguistic and structural. Edge counts do not measure relation correctness or prediction quality.',
              f"All sources contain {a['all_sources_to_description_node_ratio']:.2f} times as many nodes as description-only in this preview.",
              'Pool within documents and sources so a long post does not dominate simply by node count.',
              f"Largest document: {a['largest_document']}.", '',
              '3. Where are all the CVEs?']
    lines += [f'{k}: {v}' for k, v in a['dataset_accounting'].items()]
    lines += [f"Total: {a['dataset_total']}", f"Split exclusions: {a['split_exclusions']}",
              'These are chronological/grouping and task-eligibility rules, not a demand for perfect EPSS histories.',
              'Partial histories are already retained. Missing target scores are not invented.',
              'The large reserved test and conservative shared-link grouping constrain training size.',
              'They are not the only possible scientific design, but cannot be silently changed to enlarge this run.', '',
              '4. What has already been trained?',
              'R07: full Day-30-to-Day-60 development, 1023 training / 197 validation CVEs, six fits.',
              'It compared description TPG+EPSS with EPSS-only. Neither beat persistence; text benefit remains unproven.',
              'H02: full Day-60-to-Day-90 development, 1029 training / 173 validation CVEs, six fits.',
              'It compared 31-day and 61-day EPSS histories, not textual sources.',
              'The verified H02 review reports persistence MAE 0.00830393063584; all learned MAEs are slightly worse.',
              '150 of 173 validation targets equal the last observed input score. Two CVEs account for 94.99% of absolute persistence error.',
              'This concerns endpoints, not a claim that all intervening scores were unchanged.',
              'Recent/long histories yield identical final predictions within each seed, with negligible learned corrections.',
              'This does not prove that older history or text has no useful information in other models.', '',
              '5. Finite next step',
              f"Prepare the existing full development sources: {a['full_candidate_documents']['train']} training and {a['full_candidate_documents']['validation']} validation documents.",
              'Keep the same 1029/173 CVEs, Day-60 masks, no-COREF extractor and frozen text encoder.',
              'Implement document GNN -> within-source pooling -> source masks -> fusion with past EPSS -> Day-90 score.',
              'Use recent EPSS history as the simpler provisional control, not as a proven superior history.',
              'Freeze EPSS-only, description+EPSS and all-text+EPSS comparisons with seeds 17, 29 and 43: nine fits.',
              'Use one small integration smoke for shapes, missing sources, gradients and checkpoint reloads; then run full training.',
              'Do not repeat optimizer/coreference tuning rounds or select settings from smoke scores.',
              'Source-removal ablations can follow the first full comparison. Persistence remains a reference.',
              'This is a recommendation, not an implemented or approved new training protocol.',
              'The full multi-source training CLI does not exist yet. MS02 has only a bounded cache CLI.',
              'Classify current text fusion as retrospective: supplied dates do not prove text was available at Day 60.', '',
              '6. Limitations']
    lines += ['- ' + x for x in a['limitations']]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    a = analyze()
    text = report(a)
    if args.verify_only:
        preview.require(preview.read(OUTPUT / 'analysis.json') == a, 'Review JSON changed')
        preview.require((OUTPUT / 'analysis.txt').read_text() == text, 'Review text changed')
        print('MS02 review reproduced. No training, parser/encoder rerun or test scoring.')
        return
    preview.require(not OUTPUT.exists(), 'Review exists; use --verify-only')
    OUTPUT.mkdir(parents=True)
    preview.write(OUTPUT / 'analysis.json', a)
    (OUTPUT / 'analysis.txt').write_text(text, encoding='utf-8')
    print(f'MS02 review saved to {OUTPUT}. Original cache unchanged.')


if __name__ == '__main__':
    main()
