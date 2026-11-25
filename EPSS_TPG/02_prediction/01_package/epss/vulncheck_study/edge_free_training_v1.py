"""VK06 edge-free retraining with unchanged VK04 model, optimizer and batch path."""
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
from tqdm import tqdm

from . import graph_training_v1 as original, graph_model_v1 as network

data = original.data
CONFIG = data.ROOT / '02_prediction/04_configs/02_vulncheck/05_edge_free_control_v1.json'
CLI = data.ROOT / '02_prediction/03_analysis/03_vulncheck_study/10_run_edge_free_control.py'


def without_edges(graph):
    result = graph.clone()
    result.edge_index = graph.edge_index[:, :0].clone()
    result.edge_type = graph.edge_type[:0].clone()
    result.edge_features = graph.edge_features[:0].clone()
    data.require(result.num_nodes == graph.num_nodes and result.num_edges == 0 and
                 result.edge_features.shape == (0, graph.edge_features.shape[1]), 'Invalid edge-free graph')
    return result


class EdgeFreeBatches:
    """Keep original graph budgets and packet boundaries before removing edges."""
    def __init__(self, source):
        self.source = source

    def plan(self, ids):
        return self.source.plan(ids)

    def packets(self, plan, variant):
        data.require(variant == {'edge_view': 'no_coref'}, 'Unexpected base edge view')
        for start, graph in self.source.packets(plan, variant):
            yield start, without_edges(graph)


def audit_contract(batches, ids, batch_size):
    counts = {'cves': len(ids), 'graphs': 0, 'nodes_preserved': 0, 'edges_removed': 0, 'edges_remaining': 0}
    for offset in tqdm(range(0, len(ids), batch_size), desc='Check matched nodes and empty edges', unit='batch'):
        plan = batches.plan(ids[offset:offset+batch_size])
        for _, graph in batches.packets(plan, {'edge_view': 'no_coref'}):
            changed = without_edges(graph)
            for field in ('x', 'batch', 'ptr', 'token_mask'):
                data.require(torch.equal(changed[field], graph[field]), f'Node/packet field changed: {field}')
            counts['graphs'] += graph.num_graphs
            counts['nodes_preserved'] += graph.num_nodes
            counts['edges_removed'] += graph.num_edges
            counts['edges_remaining'] += changed.num_edges
    return counts


def context(scale):
    batches, targets, resources, dim, base_identity = original.context(scale)
    protocol = data.read(CONFIG)
    p = base_identity['settings']
    data.require(base_identity['schema'] == protocol['reference_schema'] and p['threshold'] == .5,
                 'Unexpected reference protocol or threshold')
    reference = data.ROOT / p['result_root'] / scale
    data.check_lock(reference)
    data.require(data.dump(data.read(reference / 'run_identity.json')) == data.dump(base_identity),
                 'Saved VK04 settings/cohort differ from current source')
    identity = {**base_identity, 'schema': protocol['schema'], 'protocol': protocol,
                'reference_identity_digest': data.digest(base_identity),
                'pins': {**base_identity['pins'], str(CONFIG): data.sha(CONFIG), str(CLI): data.sha(CLI),
                         str(Path(__file__)): data.sha(Path(__file__)),
                         str(reference / 'artifact_lock.json'): data.sha(reference / 'artifact_lock.json')},
                'interpretation': 'Retrospective snapshot classification; trained edge-free control, not independent test.'}
    print(f'[VK06] {scale}: {len(p["seeds"])} new edge-free fits; '
          f'CVEs={ {s: len(v) for s,v in identity["ids"].items()} }; '
          f'fresh initialization, unchanged VK04 settings; no test scoring.', flush=True)
    return batches, targets, resources, dim, identity, reference


def reference_results(reference, ids, targets):
    results = {}
    all_predictions = data.read(reference / 'predictions.json')
    for name, expected in data.read(reference / 'metrics.json').items():
        rep, seed = name.split('__seed_')
        rows = [r for r in all_predictions if r['model'] == rep and r['seed'] == int(seed)]
        data.require([r['cve'] for r in rows] == ids and
                     all(r['split'] == 'validation' and r['actual'] == targets[r['cve']] for r in rows), 'Reference CVE/label mismatch')
        metrics = original.development.metric([r['actual'] for r in rows], [r['predicted'] for r in rows])
        data.require(metrics == expected, 'Reference metrics changed')
        results[name] = metrics
    return results


def run(scale='smoke', verify=False, check_only=False, resume=False):
    from ..forecast_matched_multisource import writer_lock
    base, targets, resources, dim, identity, reference = context(scale)
    p, protocol = identity['settings'], identity['protocol']
    data.check_pins(identity['pins'])
    # Audit every prepared development graph before any fit; no embeddings are computed.
    audit = {split: audit_contract(base, ids, p['batch_size']) for split, ids in identity['ids'].items()}
    data.require(set(audit) == {'train', 'validation'}, 'Unexpected development role')
    references = reference_results(reference, identity['ids']['validation'], targets)
    logistic = data.read(reference / 'reference_metrics.json')
    batches = EdgeFreeBatches(base)
    if check_only:
        print('VK06 edge-free preflight passed. No fitting, outputs or test scoring.')
        return
    output = data.ROOT / protocol['result_root'] / scale
    output.parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(output.parent / '.writer.lock'):
        if output.exists():
            data.require(verify or resume, 'Run exists: use --resume or --verify-only')
            data.require(data.dump(data.read(output / 'run_identity.json')) == data.dump(identity), 'Run identity differs')
            if verify or (output / 'artifact_lock.json').exists():
                data.check_lock(output)
        else:
            data.require(not verify, 'No VK06 run to verify; run training first')
            output.mkdir()
            (output / 'run_identity.json').write_text(data.dump(identity))
        results, predictions = {}, []
        for seed in tqdm(p['seeds'], desc='Matched edge-free classifier fits', unit='seed'):
            name = f'security_tpg_edge_free__seed_{seed}'
            directory = output / name
            print(f'[VK06] seed={seed}; original VK04 training loop follows; input packets have zero edges.', flush=True)
            if directory.exists():
                rows, metrics = original.verify_fit(directory, 'security_tpg', seed, identity, resources, dim, batches, targets)
            else:
                data.require(not verify, f'Missing completed seed {seed}')
                stage = Path(tempfile.mkdtemp(prefix='.vk06-fit-', dir=output.parent))
                try:
                    model = original.model_for(resources, dim, p, seed)
                    rows, metrics = original.save_fit(stage, model, seed, 'security_tpg', identity,
                                                      resources, dim, batches, targets)
                    data.require(not directory.exists(), 'Fit appeared; refuse overwrite')
                    stage.rename(directory)
                finally:
                    if stage.exists():
                        shutil.rmtree(stage)
            results[name] = metrics
            predictions += [{**r, 'model': protocol['reported_representation']} for r in rows]
        metrics_keys = ('average_precision', 'roc_auc', 'precision', 'recall', 'f1', 'accuracy', 'log_loss', 'brier')
        paired = [{'seed': seed, 'edge_free_minus_original_tpg': {
            key: results[f'security_tpg_edge_free__seed_{seed}'][key] - references[f'security_tpg__seed_{seed}'][key]
            if results[f'security_tpg_edge_free__seed_{seed}'][key] is not None and references[f'security_tpg__seed_{seed}'][key] is not None
            else None for key in metrics_keys}} for seed in p['seeds']]
        aggregate = {key: {'mean': float(np.mean([r[key] for r in results.values()])),
                           'std_population': float(np.std([r[key] for r in results.values()]))}
                     if all(r[key] is not None for r in results.values()) else None for key in metrics_keys}
        summary = {'schema': protocol['schema'], 'scale': scale, 'cves': {s:len(v) for s,v in identity['ids'].items()},
                   'aggregate_edge_free': aggregate, 'paired_seed_comparisons': paired, 'graph_contract': audit,
                   'threshold': .5, 'test_scoring': False, 'threshold_tuning': False, 'automatic_model_selection': False,
                   'reference_note': 'Same-scale VK04 GNN and pooled references. Smoke is integration only; smoke logistic references were full-trained.',
                   'interpretation': 'Matched edge-free retraining, not the VK05 inference-time edge removal. Validation has already informed this design.',
                   'limitations': [
                       'Nodes and their security-related text remain; this is not removal of all security information.',
                       'Unused message/edge parameters remain in the same architecture but receive no edge-driven signal.',
                       'Residual update and normalization layers are retained; this is not a replacement with pooled tokens.',
                       'No SMP, summaries or EPSS predictors are added. Current structured/text inputs remain retrospective.',
                       'Repeated validation, source filtering, shared text and prior test exposure preclude an independent-test claim.',
                       'Across-seed standard deviations are not confidence intervals. No significance claim or threshold search.'
                   ]}
        lines = ['VK06 matched edge-free retraining',
                 f'Scale: {scale}; training/validation: {summary["cves"]}; threshold=0.5.',
                 'Same VK04 models, seeds, input rows, frozen node embeddings, packet boundaries, loss and optimizer.',
                 'Only edges are disabled during both training and validation; no test scoring.', '',
                 'Model | AP | ROC-AUC | Precision | Recall | F1 | Accuracy | Log loss | Brier']
        for name, metric in {**references, **results}.items():
            lines.append(name+' | '+' | '.join(str(metric[k]) for k in metrics_keys))
        lines += ['', 'Same-seed differences (edge-free minus original TPG):', data.dump(paired),
                  'Positive AP/F1 differences favor edge-free; negative log-loss/Brier differences favor edge-free.',
                  '', *summary['limitations']]
        artifacts = {'metrics.json': results, 'reference_metrics.json': references, 'logistic_reference_metrics.json': logistic,
                     'predictions.json': predictions, 'predictions.csv': original.development.csv_text(predictions),
                     'summary.json': summary, 'summary.txt': '\n'.join(lines)+'\n', 'graph_contract.json': audit}
        if verify or (output / 'artifact_lock.json').exists():
            for name, value in artifacts.items():
                data.require((output / name).read_text() == (value if isinstance(value,str) else data.dump(value)), 'Report differs: '+name)
        else:
            for name, value in artifacts.items():
                (output / name).write_text(value if isinstance(value,str) else data.dump(value))
            original.lock_tree(output)
        data.check_lock(reference)
        data.check_pins(identity['pins'])
    print(f'VK06 {scale} edge-free checkpoints and predictions '+('verified.' if verify else f'saved to {output}.')+
          ' Original VK04 runs unchanged; no test scoring or threshold tuning.')
