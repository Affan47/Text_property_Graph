"""TRV05 full development diff and frozen specification; no training/test access."""
from collections import Counter
from pathlib import Path
import sys

import torch
from tqdm import tqdm

from ..vulncheck_study import adapter_v2 as data
from . import v5_cache, subtypes_v5, subtypes_v6

ROOT = data.ROOT
sys.path.insert(0, str(ROOT / '01_tpg/10_extensions'))
import tpg_security_product_v6 as policy

SOURCE = v5_cache.OUTPUT / 'full'
CONTROLS = ROOT / '05_results/06_tpg_revision/03_v5_subtypes_comparison_v1/full'
OUTPUT = ROOT / '05_results/06_tpg_revision/10_v6_diff_audit_v1'
CLI = ROOT / '02_prediction/03_analysis/04_tpg_revision/09_audit_v6_schema.py'
VARIANTS = {
    'v6_node_mean': {'readout': 'node_mean', 'use_subtypes': False, 'control': 'v5_node_mean'},
    'v6_subtypes_node_mean': {'readout': 'node_mean', 'use_subtypes': True, 'control': 'v5_subtypes_node_mean'},
    'v6_subtypes_type_balanced': {'readout': 'type_balanced', 'use_subtypes': True, 'control': 'v5_subtypes_type_balanced'}}


def assert_export(graph, tensor, resources):
    from tpg.exporters.exporters import PyGExporter, import_graphson
    expected = PyGExporter().export(import_graphson(graph), embedding_dim=0, use_security_edge_types=True)
    data.require(expected['type_vocabulary'] == resources['type_vocabulary'], 'Changed type vocabulary')
    ntypes = resources['type_vocabulary']['num_node_types']
    data.require(torch.equal(tensor['x'][:, :ntypes], torch.tensor(expected['x'], dtype=torch.float32)),
                 'Node types/order differ from GraphSON')
    for key in ('edge_index', 'edge_type', 'edge_features'):
        value = torch.tensor(expected[key], dtype=tensor[key].dtype).reshape(tensor[key].shape)
        data.require(torch.equal(value, tensor[key]), 'GraphSON/tensor mismatch: ' + key)


def convert(graph, tensor, resources):
    """Pure v5-to-v6 conversion. No source text is re-encoded or rewritten."""
    data.require(set(tensor) == {'x', 'token_mask', 'edge_index', 'edge_type', 'edge_features', 'subtype'},
                 'Unexpected input tensor fields')
    data.require(graph['metadata'].get('v5_correction', {}).get('version') == policy.v5.VERSION,
                 'Expected a corrected v5 parent')
    assert_export(graph, tensor, resources)
    data.require(torch.equal(tensor['subtype'], subtypes_v5.encode_nodes(graph['vertices'])), 'Wrong v5 subtypes')
    corrected = policy.correct(graph)
    row_by_id = {n['id']: i for i, n in enumerate(graph['vertices'])}
    edge_by_id = {e['id']: i for i, e in enumerate(graph['edges'])}
    rows = [row_by_id[n['id']] for n in corrected['vertices']]
    edges = [edge_by_id[e['id']] for e in corrected['edges']]
    remap = {n['id']: i for i, n in enumerate(corrected['vertices'])}
    data.require(all(n == graph['vertices'][i] for n, i in zip(corrected['vertices'], rows)),
                 'Surviving node properties changed')
    data.require(all(e == graph['edges'][i] for e, i in zip(corrected['edges'], edges)),
                 'Surviving edge properties changed')
    item = {'x': tensor['x'][rows].clone(), 'token_mask': tensor['token_mask'][rows].clone(),
            'edge_type': tensor['edge_type'][edges].clone(), 'edge_features': tensor['edge_features'][edges].clone(),
            'edge_index': torch.tensor([[remap[e['outV']] for e in corrected['edges']],
                                        [remap[e['inV']] for e in corrected['edges']]], dtype=torch.long).reshape(2, -1),
            'subtype': subtypes_v6.encode_nodes(corrected['vertices'])}
    old_tokens = [i for i, n in enumerate(graph['vertices']) if n['label'] == 'TOKEN']
    new_tokens = [i for i, n in enumerate(corrected['vertices']) if n['label'] == 'TOKEN']
    data.require(len(old_tokens) == len(new_tokens) and
                 torch.equal(tensor['x'][old_tokens], item['x'][new_tokens]) and
                 torch.equal(tensor['token_mask'][old_tokens], item['token_mask'][new_tokens]), 'Token values changed')
    data.require(bool(torch.isfinite(item['x']).all()), 'Nonfinite node features')
    assert_export(corrected, item, resources)
    data.require(policy.correct(corrected) == corrected, 'Correction is not idempotent')
    return corrected, item


def specification(protocol, pins):
    p = protocol['settings']
    data.require(p['weight_decay'] == .0001 and p['seeds'] == [17, 29, 43] and p['threshold'] == .5,
                 'Locked control settings differ')
    keys = ('hidden', 'layers', 'learning_rate', 'weight_decay', 'batch_size', 'epochs', 'patience',
            'threads', 'cache_mib', 'graph_node_budget', 'graph_edge_budget', 'graph_chunk_budget',
            'loss', 'selection', 'seeds', 'threshold', 'edge_view')
    references = {}
    for variant, definition in VARIANTS.items():
        for seed in p['seeds']:
            name = f'{definition["control"]}__seed_{seed}'
            path = CONTROLS / name
            saved = torch.load(path / 'checkpoint.pt', map_location='cpu', weights_only=True)
            data.require(saved['variant'] == definition['control'] and saved['seed'] == seed and
                         saved['identity'] == data.digest(protocol), 'Control checkpoint identity differs')
            rows = data.read(path / 'predictions.json')
            data.require([r['cve'] for r in rows] == protocol['ids']['validation'], 'Control prediction CVEs differ')
            references[f'{variant}__seed_{seed}'] = {
                'control': name, 'checkpoint': str(path / 'checkpoint.pt'),
                'checkpoint_sha256': data.sha(path / 'checkpoint.pt'),
                'predictions_sha256': data.sha(path / 'predictions.json'), 'best_epoch': saved['best_epoch']}
    return {'schema': 'trv05_v6_runner_spec_v1', 'stage': 'audit_and_specification_only',
            'task': 'Retrospective binary in_vckev classification, not EPSS or future exploitation.',
            'ids': protocol['ids'], 'settings': {k: p[k] for k in keys}, 'optimizer': 'AdamW',
            'class_weights': protocol['class_weights'], 'packages': protocol['packages'],
            'variants': VARIANTS, 'new_full_fits': 9, 'frozen_control_fits': 9, 'controls': references,
            'source_policy': 'D only; same accepted descriptions and train-fitted structured vectors; no G/N/M.',
            'graph_policy': 'v6 correction of exact frozen v5 descriptions; no COREF; exact surviving tensor rows/edges.',
            'embedding_policy': 'No NLP rerun, no encoder rerun, no embedding fine-tuning.',
            'subtype_vocabulary': subtypes_v6.VOCAB,
            'initialization': 'Fresh fixed seed; common parameters bit-identical to v5; subtype projection zero-initialized.',
            'projection_parameters': {'v5': 31 * p['hidden'], 'v6': 32 * p['hidden'],
                                      'extra': p['hidden'], 'inactive_in_node_mean': True},
            'training': 'Deterministic two-pass replay; one optimizer step per CVE batch; no dropout; no retuning.',
            'metrics': ['average_precision', 'roc_auc', 'brier', 'log_loss', 'f1', 'precision',
                        'recall', 'accuracy', 'tp', 'fp', 'tn', 'fn'],
            'reporting': 'Every seed, all-seed means, same-seed v6-minus-v5 deltas; undefined metrics remain null.',
            'paired_intervals': {'unit': 'validation CVE', 'repetitions': 1000, 'seed': 6029,
                                 'method': 'paired class-stratified CVE bootstrap of mean same-seed delta',
                                 'level': .95, 'interpretation': 'Exploratory, no multiplicity adjustment; not seed uncertainty.'},
            'diagnostics': ['training BCE and validation AP/Brier/log-loss by epoch',
                            'selected epoch and early stopping', 'subtype norm and SEVERITY-channel norm',
                            'logit saturation and probability histograms', 'paired PR/ROC and calibration plots'],
            'gates_before_training': ['review this audit', 'immutable v6 cache and exact tensor reproduction',
                                     'matched input/label and common-weight checks',
                                     'small smoke checkpoint and prediction reload'],
            'test_access': False, 'training_started': False, 'source_pins': pins,
            'limitations': ['Previously exposed test remains excluded; validation has informed earlier development.',
                            'v6 is a narrow repair, not a clean-graph guarantee.',
                            'Subtype arms combine predicate repair and schema restoration, not a full factorial attribution.',
                            'Append-only SEVERITY adds 32 active parameters only in subtype-enabled arms.']}


def source_path(relative):
    path = (SOURCE / relative).resolve()
    data.require(SOURCE.resolve() in path.parents, 'Unsafe source path')
    return path


def run(verify=False):
    from ..forecast_matched_multisource import writer_lock
    from tpg.frontends import spacy_frontend
    torch.set_num_threads(2)
    roots = [SOURCE, CONTROLS, ROOT / '05_results/06_tpg_revision/06_v5_regularization_v1/full',
             ROOT / '05_results/06_tpg_revision/09_d_vs_dg_v1/training/full',
             ROOT / '05_results/06_tpg_revision/09_d_vs_dg_v1/diagnostics/full',
             ROOT / '05_results/06_tpg_revision/09_d_vs_dg_v1/review/full']
    for directory in tqdm(roots, desc='Verify preserved studies and source cache', unit='tree'):
        data.check_lock(directory)
    pins = {str(d / 'artifact_lock.json'): data.sha(d / 'artifact_lock.json') for d in roots}
    for path in (Path(__file__), CLI, Path(policy.__file__), Path(policy.v5.__file__),
                 Path(spacy_frontend.__file__), Path(subtypes_v5.__file__), Path(subtypes_v6.__file__)):
        pins[str(path)] = data.sha(path)
    identity = data.read(SOURCE / 'run_identity.json')
    data.check_pins(identity['source_pins'])
    pins.update(identity['source_pins'])
    protocol = data.read(CONTROLS / 'run_identity.json')
    data.require(identity['ids'] == protocol['ids'], 'Cache/control cohort differs')
    ids = identity['ids']
    data.require(set(ids) == {'train', 'validation'} and len(ids['train']) == 2870 and
                 len(ids['validation']) == 1104 and len(set(ids['train'] + ids['validation'])) == 3974,
                 'Expected fixed development cohort only')
    index, resources = data.read(SOURCE / 'cve_index.json'), data.read(SOURCE / 'resources.json')
    data.require(set(index) == set(ids['train'] + ids['validation']), 'Index/cohort differs')
    spec = specification(protocol, pins)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(OUTPUT.parent / '.v6-audit-writer.lock'):
        data.require(OUTPUT.exists() if verify else not OUTPUT.exists(), 'Use --verify-only only for an existing audit')
        diffs, details, artifacts, severity_examples = [], [], {}, []
        totals = {s: Counter() for s in ids}
        nodes_before, nodes_after, edges_before, edges_after = Counter(), Counter(), Counter(), Counter()
        sub_before, sub_after, rules, edge_rules = Counter(), Counter(), Counter(), Counter()
        for split, cves in ids.items():
            for cve in tqdm(cves, desc=f'Audit exact {split} v5-to-v6 diff', unit='CVE'):
                entry = index[cve]
                data.require(entry['split'] == split, 'Wrong split')
                row = {'cve': cve, 'split': split, 'chunks': 0, 'nodes_before': 0, 'nodes_after': 0,
                       'edges_before': 0, 'edges_after': 0, 'severity_channels_restored': 0,
                       'removed_predicates': 0, 'removed_arguments': 0}
                for doc in entry['documents']:
                    data.require(doc['source_text_type'] == 'description', 'Unexpected source')
                    for chunk in doc['chunks']:
                        data.require(chunk['has_graph'], 'Expected description graph')
                        graph = data.read(source_path(chunk['graph']))
                        tensor = torch.load(source_path(chunk['tensor']), map_location='cpu', weights_only=True)
                        corrected, item = convert(graph, tensor, resources)
                        meta = corrected['metadata']['v6_correction']
                        changes = []
                        previous = {n['id']: int(v) for n, v in zip(graph['vertices'], tensor['subtype'])}
                        for n, v in zip(corrected['vertices'], item['subtype']):
                            if previous[n['id']] != int(v):
                                data.require(previous[n['id']] == subtypes_v5.VOCAB['UNKNOWN'] and
                                             int(v) == subtypes_v6.VOCAB['SEVERITY'], 'Unexpected subtype change')
                                change = {'node': n, 'before': 'UNKNOWN', 'after': 'SEVERITY',
                                          'before_index': previous[n['id']], 'after_index': int(v)}
                                changes.append(change)
                                if split == 'train' and len(severity_examples) < 5:
                                    severity_examples.append({'cve': cve, 'source_text': graph['metadata']['source_text'], **change})
                        row['chunks'] += 1
                        for label, value in (('nodes_before', len(graph['vertices'])), ('nodes_after', len(corrected['vertices'])),
                                             ('edges_before', len(graph['edges'])), ('edges_after', len(corrected['edges'])),
                                             ('severity_channels_restored', len(changes))):
                            row[label] += value
                        row['removed_predicates'] += sum(r['rule'] == 'VERSION_SHAPED_PREDICATE' for r in meta['findings'])
                        row['removed_arguments'] += sum(r['rule'] == 'ORPHAN_VERSION_ARGUMENT' for r in meta['findings'])
                        nodes_before.update(n['label'] for n in graph['vertices'])
                        nodes_after.update(n['label'] for n in corrected['vertices'])
                        edges_before.update(e['label'] for e in graph['edges'])
                        edges_after.update(e['label'] for e in corrected['edges'])
                        sub_before.update(subtypes_v5.NAMES[i] for i in tensor['subtype'].tolist())
                        sub_after.update(subtypes_v6.NAMES[i] for i in item['subtype'].tolist())
                        rules.update(r['rule'] for r in meta['findings'])
                        edge_rules.update(reason for r in meta['edge_findings'] for reason in r['reasons'])
                        if meta['findings'] or meta['edge_findings'] or changes:
                            details.append({'cve': cve, 'split': split, 'chunk': chunk['key'],
                                            'source_text': graph['metadata']['source_text'],
                                            'removed_nodes': [n for n in graph['vertices'] if n['id'] in meta['removed_node_ids']],
                                            'removed_edges': [e for e in graph['edges'] if e['id'] in meta['removed_edge_ids']],
                                            'node_findings': meta['findings'], 'edge_findings': meta['edge_findings'],
                                            'subtype_changes': changes})
                            if meta['findings'] or meta['edge_findings']:
                                artifacts[f'changed_graphs/{cve}/{chunk["key"]}.json'] = corrected
                diffs.append(row)
                totals[split].update({k: v for k, v in row.items() if isinstance(v, int)})
                totals[split].update(cves=1, graph_changed_cves=int(row['nodes_before'] != row['nodes_after'] or row['edges_before'] != row['edges_after']),
                                     severity_changed_cves=int(row['severity_channels_restored'] > 0))
        summary = {'schema': 'trv05_v6_diff_audit_v1', 'by_split': {k: dict(v) for k, v in totals.items()},
                   'nodes_before': dict(nodes_before), 'nodes_after': dict(nodes_after),
                   'edges_before': dict(edges_before), 'edges_after': dict(edges_after),
                   'subtypes_before': dict(sub_before), 'subtypes_after': dict(sub_after),
                   'node_rules': dict(rules), 'edge_reasons_overlapping': dict(edge_rules),
                   'audit_checks': ['all original locks valid', 'exact node/edge/tensor alignment',
                                    'all surviving properties and token embeddings/masks unchanged',
                                    'only UNKNOWN-to-SEVERITY channel changes', 'no dangling endpoints',
                                    'idempotent correction', 'nine frozen reference identities verified'],
                   'new_embeddings': 0, 'training_fits': 0, 'test_cves_accessed': 0,
                   'severity_training_examples': severity_examples, 'limitations': spec['limitations']}
        report = render(summary, details, spec)
        artifacts.update({'summary.json': summary, 'summary.txt': report, 'REPORT.md': report,
                          'cve_diffs.json': diffs, 'exact_changes.json': details,
                          'trv05_runner_spec.json': spec, 'source_pins.json': pins,
                          'cve_diffs.csv': csv_rows(diffs)})
        data.check_pins(pins)
        v5_cache.publish_nested(OUTPUT, artifacts, verify)
        data.check_pins(pins)
    print(f'TRV05 full diff audit '+('reproduced.' if verify else f'saved to {OUTPUT}.')+
          ' No training, parser/encoder rerun or test scoring.')


def csv_rows(rows):
    import csv
    from io import StringIO
    out = StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def render(s, details, spec):
    lines = ['# TRV05: v6 description-only diff audit and fixed comparison', '',
             '## Status', '', 'Audit and specification only. No training, new embeddings or test access.',
             'Every saved v5 graph and tensor was checked; the v5, TRV03 and TRV04 artifacts remain unchanged.', '',
             '## What changes', '',
             'The opt-in SpacyFrontendV6 rejects whole version-shaped tokens before predicate/argument creation.',
             'The v6 correction applies the same guard to saved v5 graphs and removes relations whose recorded',
             'dependency-rule evidence refers to a version-shaped verb. It does not classify numbers as versions.',
             'Only arguments owned exclusively by removed predicates are removed. Shared arguments remain.',
             'Parser token POS labels, dependency edges, verb phrases and entity annotations are not rewritten.',
             'This is a narrow derived-predicate repair, not a fresh NLP parse or a guarantee of clean graphs.', '',
             'SEVERITY is appended at subtype index 31. The existing 31 indices keep their meanings.',
             'The source entities were already SEVERITY; only their numeric model channel changes from UNKNOWN.', '',
             '## Exact development diff', '',
             '| Split | CVEs | Graphs changed | Nodes before -> after | Edges before -> after | Severity nodes restored | CVEs with severity |',
             '|---|---:|---:|---|---|---:|---:|']
    for split, r in s['by_split'].items():
        lines.append(f'| {split} | {r["cves"]} | {r["graph_changed_cves"]} | {r["nodes_before"]} -> {r["nodes_after"]} | '
                     f'{r["edges_before"]} -> {r["edges_after"]} | {r["severity_channels_restored"]} | {r["severity_changed_cves"]} |')
    lines += ['', 'Graph changes and severity-channel changes can affect the same CVE; do not add these CVE counts.', '',
              '| Removed node type | Nodes |', '|---|---:|']
    lines += [f'| {k} | {v - s["nodes_after"].get(k, 0)} |' for k, v in sorted(s['nodes_before'].items()) if v != s['nodes_after'].get(k, 0)]
    lines += ['', '| Removed edge type | Edges |', '|---|---:|']
    lines += [f'| {k} | {v - s["edges_after"].get(k, 0)} |' for k, v in sorted(s['edges_before'].items()) if v != s['edges_after'].get(k, 0)]
    lines += ['', '## Real examples', '']
    for r in details:
        if r['node_findings'] and r['split'] == 'train':
            lines += [f'### {r["cve"]}', '', 'Saved description: '+r['source_text'], '',
                      'Removed predicates: '+', '.join('`'+n['text']+'`' for n in r['node_findings'] if n['rule'] == 'VERSION_SHAPED_PREDICATE'),
                      f'Removed {len(r["removed_nodes"])} nodes and {len(r["removed_edges"])} edges; exact IDs and evidence are in exact_changes.json.', '']
    for r in s['severity_training_examples']:
        lines += [f'- {r["cve"]}: `{r["node"]["properties"]["TEXT"]}`, source subtype SEVERITY; numeric channel UNKNOWN (2) -> SEVERITY (31).']
    lines += ['', '## Fixed TRV05 specification', '',
              '| New model | Frozen matching control | Readout | Subtype channels used |', '|---|---|---|---|']
    lines += [f'| {v} | {d["control"]} | {d["readout"]} | {d["use_subtypes"]} |' for v, d in VARIANTS.items()]
    lines += ['', '- Nine fresh fits: three variants times seeds 17, 29 and 43. Nine saved v5 fits are references, not retrained.',
              '- Same 2,870 training / 1,104 validation CVEs; saved VulnCheck listed/unlisted target. D plus unchanged structured branch.',
              '- Frozen embeddings; two typed-edge GNN layers, width 32; no COREF, G, N, M or EPSS input.',
              '- AdamW; learning rate 0.001; weight decay 0.0001; batch size 8; up to 100 epochs; patience 15.',
              '- Training-only balanced class weights and weighted BCE; earliest strict best full-validation AP; threshold fixed at 0.5.',
              '- Node-mean and type-balanced readouts are unchanged. Subtype projection grows from 992 to 1,024 parameters and starts at zero.',
              '- All common initial weights must match v5 exactly at each seed. No dropout, new tuning, or warm-started control weights.',
              '- Report AP, ROC-AUC, Brier, log-loss and fixed-threshold confusion counts, precision, recall, F1 and accuracy.',
              '- Save all seeds, same-seed deltas and 1,000 paired class-stratified CVE bootstrap intervals (seed 6029).',
              '- Inspect training/validation curves, subtype norms, saturation, PR/ROC and calibration. No change to checkpoint selection.',
              '- The training runner and final v6 tensor cache remain the next stage after this audit is reviewed.', '',
              '## Interpretation limits', '']
    lines += ['- '+x for x in spec['limitations']]
    lines += ['- TRV04 AP is not a weighted average of subgroup AP. Downward average G-present score shifts are consistent with cross-group ranking changes,',
              '  but do not establish the proposed cause, uniform score shifts, or preserved within-group ordering. No TRV04 thresholds or results changed.', '',
              '## Files', '', '- summary.json / summary.txt / REPORT.md: counts, interpretation and specification.',
              '- cve_diffs.csv / cve_diffs.json: all 3,974 development CVEs, including unchanged cases.',
              '- exact_changes.json: removed nodes/edges and changed subtype indices with real source text.',
              '- changed_graphs/: GraphSON examples for every graph with removed nodes/edges, retaining original node IDs.',
              '- trv05_runner_spec.json: exact settings, split IDs, nine control hashes and gates before training.',
              '- source_pins.json / artifact_lock.json: preservation checks and reproducible audit identity.', '']
    return '\n'.join(lines)
