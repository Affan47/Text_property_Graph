"""Source-grounded review of the fixed 16-CVE training packet; never model labels."""
from collections import Counter
from pathlib import Path
import sys

from tqdm import tqdm
from ..vulncheck_study import adapter_v2 as data
from .experiment_v1 import publish

ROOT = data.ROOT
sys.path.insert(0, str(ROOT / '01_tpg/10_extensions'))
import tpg_security_product_v5 as policy

SOURCE = ROOT / '01_tpg/09_annotation/08_type_readout_review_v1/full'
CONFIG = ROOT / '02_prediction/04_configs/03_tpg_revision/01_v5_semantic_review.json'
OUTPUT = ROOT / '01_tpg/09_annotation/09_v5_semantic_audit_v1'


def run(verify=False):
    data.check_lock(SOURCE)
    protocol = data.read(CONFIG)
    rows = data.read(SOURCE / 'review_packet.json')
    selected = data.read(SOURCE / 'summary.json')['selected_training_cves']
    ids = data.read(SOURCE / 'run_identity.json')['ids']
    data.require(len(rows) == 16 and set(selected) == set(protocol['decisions']) == {r['cve'] for r in rows}
                 and set(selected) <= set(ids['train']) and not set(selected) & set(ids['validation']), 'Audit cohort changed')
    counts, affected, edges, node_counts = Counter(), {}, Counter(), Counter()
    weighting = {name: [] for name in ('ENTITY', 'PREDICATE', 'DOCUMENT')}
    reviewed = []
    artifacts = {}
    for row in tqdm(rows, desc='Audit 16 training CVEs against source text', unit='CVE'):
        graph = row['graph']
        corrected = policy.correct(graph)
        events = corrected['metadata']['v5_correction']['findings'] + corrected['metadata']['v5_correction']['edge_findings']
        counts.update(e['rule'] for e in events)
        for e in events:
            affected.setdefault(e['rule'], set()).add(row['cve'])
        edges.update(e['label'] for e in graph['edges'])
        node_counts.update(n['label'] for n in graph['vertices'])
        local = Counter(n['label'] for n in graph['vertices'])
        for name in weighting:
            if local[name]:
                weighting[name].append({'cve': row['cve'], 'balanced_to_original_weight':
                                        len(graph['vertices']) / (len(local) * local[name])})
        artifacts['graphs/' + row['cve'] + '.json'] = corrected
        reviewed.append({'cve': row['cve'], 'source_text': graph['metadata']['source_text'],
                         'reviewer': protocol['reviewer'], 'assessment': protocol['decisions'][row['cve']],
                         'confirmed_correction_events': events,
                         'nodes_before': len(graph['vertices']), 'nodes_after': len(corrected['vertices']),
                         'edges_before': len(graph['edges']), 'edges_after': len(corrected['edges']),
                         'deferred_categories': [k for k,v in protocol['deferred_taxonomy'].items() if row['cve'] in v]})
    # Original published packet remains pending; this is a separate automated review.
    summary = {'schema': 'trv02_v5_semantic_audit_v1', 'cves': 16, 'reviewer': protocol['reviewer'],
               'taxonomy': {k: {'meaning': policy.TAXONOMY[k], 'events': counts[k], 'cves': len(affected[k])} for k in sorted(counts)},
               'source_node_counts': dict(node_counts), 'source_edge_counts': dict(edges),
               'type_weighting_in_16_source_graphs': {name: {
                   'increased': sum(r['balanced_to_original_weight'] > 1 for r in rows),
                   'decreased': sum(r['balanced_to_original_weight'] < 1 for r in rows),
                   'per_cve': rows} for name, rows in weighting.items()},
               'changed_cves': sum(bool(r['confirmed_correction_events']) for r in reviewed),
               'nodes_suppressed': sum(r['nodes_before']-r['nodes_after'] for r in reviewed),
               'edges_suppressed': sum(r['edges_before']-r['edges_after'] for r in reviewed),
               'deferred_taxonomy': protocol['deferred_taxonomy'],
               'limitations': ['Automated review, not independent annotation; no population precision estimate.',
                              'v5 is a bounded correction, not clean or complete parsing.',
                              'No causal claim about seed-specific model failures.',
                              'No labels, EPSS values, CVSS vectors, validation mistakes or test text used to select fixes.']}
    pins = {str(p): data.sha(p) for p in (SOURCE / 'artifact_lock.json', CONFIG, Path(__file__), Path(policy.__file__))}
    lines = ['TRV02: fixed 16-CVE training semantic audit', protocol['reviewer'], '',
             'Category | annotation events | distinct CVEs | meaning']
    lines += [f'{k} | {v["events"]} | {v["cves"]} | {v["meaning"]}' for k,v in summary['taxonomy'].items()]
    lines += ['', 'Counts may overlap across categories. Derived argument removals are consequences of predicate corrections.',
              'Missed relations, uncertain section references and complex product boundaries are not guessed.', '']
    for r in reviewed:
        lines += [r['cve'], 'Source: '+r['source_text'], 'Assessment: '+r['assessment'],
                  'Exact correction events: '+data.dump(r['confirmed_correction_events']),
                  f'Nodes {r["nodes_before"]} -> {r["nodes_after"]}; edges {r["edges_before"]} -> {r["edges_after"]}', '']
    lines += summary['limitations']
    artifacts.update({'review.json': reviewed, 'summary.json': summary, 'summary.txt': '\n'.join(lines)+'\n',
                      'source_pins.json': pins, 'protocol.json': protocol})
    # publish supports nested graph paths through a local, immutable transaction.
    from .v5_cache import publish_nested
    publish_nested(OUTPUT, artifacts, verify)
    data.check_pins(pins)
    print(f'V5 semantic audit '+('verified.' if verify else f'saved to {OUTPUT}.')+' No training or test access.')
