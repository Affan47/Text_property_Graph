"""TRV01: training-only graph review and matched type-balanced readout pilot."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch
from tqdm import tqdm

from . import preservation, model_v1
from ..vulncheck_study import graph_training_v1 as base, graph_model_v1 as network

data = base.data
ROOT = data.ROOT
RESULT = ROOT / '05_results/06_tpg_revision/01_type_readout_v1'
PACKET = ROOT / '01_tpg/09_annotation/08_type_readout_review_v1'
CLI = ROOT / '02_prediction/03_analysis/04_tpg_revision/01_run_type_readout.py'


def context(scale):
    preservation.verify()
    batches, targets, resources, dim, old = base.context(scale)
    reference = ROOT / old['settings']['result_root'] / scale
    data.check_lock(reference)
    data.require(data.dump(data.read(reference / 'run_identity.json')) == data.dump(old), 'VK04 identity changed')
    pins = {**old['pins'], str(reference / 'artifact_lock.json'): data.sha(reference / 'artifact_lock.json')}
    for path in (Path(__file__), Path(model_v1.__file__), Path(preservation.__file__), CLI):
        pins[str(path)] = data.sha(path)
    identity = {**old, 'schema': 'trv01_type_balanced_readout_v1', 'pins': pins,
                'readout': 'mean within type then equal mean of present types',
                'preservation_manifest_sha256': data.sha(preservation.OUTPUT / 'manifest.json'),
                'change': 'readout only; no new parameters, nodes, edges, embeddings, sources or labels',
                'evaluation': 'Exploratory development; previous final cohort is exposed and is not accessed here.'}
    data.check_pins(pins)
    print(f'[TRV01] {scale}: type-balanced readout vs saved node-mean GNN; '
          f'{ {k: len(v) for k, v in old["ids"].items()} }; no test scoring.', flush=True)
    return batches, targets, resources, dim, identity, reference


def model(resources, dim, settings, seed, readout='type_balanced'):
    base.seed_everything(seed)
    return model_v1.Classifier(resources, dim, settings, readout)


def audit_graphs(batches, identity, resources):
    totals, per_cve = Counter(), []
    names = resources['type_vocabulary']['node_types']
    for cve in tqdm(identity['ids']['train'], desc='Inspect training TPG node types', unit='CVE'):
        counts = Counter()
        for _, graph in batches.packets(batches.plan([cve]), {'edge_view': 'no_coref'}):
            types = graph.x[:, :len(names)]
            # Exercise the exact new readout contract on every training graph.
            model_v1.type_balanced_mean(graph.x[:, :1], types, graph.batch)
            counts.update({name: int(types[:, i].sum()) for name, i in names.items()})
        totals.update(counts)
        per_cve.append({'cve': cve, 'node_counts': dict(counts),
                        'token_fraction': counts['TOKEN'] / sum(counts.values())})
    return {'training_node_counts': dict(totals), 'training_cves': per_cve,
            'meaning': 'Counts describe readout weighting, not semantic correctness or causality.'}


def review(scale, verify=False):
    batches, _, resources, _, identity, _ = context(scale)
    counts = audit_graphs(batches, identity, resources)
    # Stable sample, independent of scores, predictions and final-test mistakes.
    selected = sorted(identity['ids']['train'], key=lambda c: data.digest(['TRV01', c]))[:16]
    examples = []
    for cve in tqdm(selected, desc='Export training-only review examples', unit='CVE'):
        for doc in batches.index[cve]['documents']:
            for part in doc['chunks']:
                if not part['has_graph']:
                    continue
                graph = data.read(batches.store.directory / part['graph'])
                examples.append({'cve': cve, 'source': 'description', 'chunk_key': part['key'],
                                 'graph': graph, 'semantic_judgment': 'pending',
                                 'questions': ['Are entity boundaries and types correct?',
                                               'Is each security edge supported by the quoted text?',
                                               'Are negation, uncertainty and version meaning preserved?']})
    report = {'schema': identity['schema'], 'selected_training_cves': selected,
              'semantic_judgments': 'Pending human review; no extraction precision is claimed.',
              'extractor_changed': False, 'counts': counts['training_node_counts']}
    artifacts = {'run_identity.json': identity, 'graph_audit.json': counts,
                 'review_packet.json': examples, 'summary.json': report,
                 'summary.txt': 'TRV01 training-only TPG review\n' + data.dump(report) +
                 'Graphs are copied for inspection, not repaired or relabelled. No validation/test examples.\n'}
    publish(PACKET / scale, artifacts, verify)
    data.check_pins(identity['pins'])
    print('TRV01 TPG review ' + ('verified.' if verify else 'saved to ' + str(PACKET / scale)))


def publish(directory, artifacts, verify):
    if verify:
        data.check_lock(directory)
        for name, value in artifacts.items():
            expected = value if isinstance(value, str) else data.dump(value)
            data.require((directory / name).read_text() == expected, 'Changed report: ' + name)
        return
    data.require(not directory.exists(), 'Output exists; use --verify-only')
    directory.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.trv01-report-', dir=directory.parent))
    try:
        for name, value in artifacts.items():
            (stage / name).write_text(value if isinstance(value, str) else data.dump(value))
        base.lock_tree(stage)
        data.require(not directory.exists(), 'Destination appeared')
        stage.rename(directory)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def verify_fit(directory, resources, dim, identity, batches, targets, seed):
    data.check_lock(directory)
    checkpoint = torch.load(directory / 'checkpoint.pt', map_location='cpu', weights_only=True)
    data.require(checkpoint['identity'] == data.digest(identity) and checkpoint['seed'] == seed
                 and checkpoint['readout'] == 'type_balanced', 'Checkpoint identity/readout differs')
    current = model(resources, dim, identity['settings'], seed)
    current.load_state_dict(checkpoint['state_dict'], strict=True)
    ids = identity['ids']['validation']
    probabilities = network.predict(current, batches, ids, 'security_tpg', identity['settings']['batch_size'])
    saved = data.read(directory / 'predictions.json')
    data.require([r['cve'] for r in saved] == ids and all(r['actual'] == targets[r['cve']] for r in saved), 'Prediction label join differs')
    data.require(np.allclose(probabilities, [r['predicted'] for r in saved], atol=1e-7, rtol=0), 'Prediction reload differs')
    history = data.read(directory / 'history.json')
    best = max(history, key=lambda r: r['validation']['average_precision'])
    metrics = base.development.metric([targets[c] for c in ids], probabilities)
    data.require(best['epoch'] == checkpoint['best_epoch'] and metrics == data.read(directory / 'metrics.json')
                 and metrics == best['validation'], 'Selection or metrics differ')
    return saved, metrics


def train_fit(directory, resources, dim, identity, batches, targets, seed):
    p, ids = identity['settings'], identity['ids']
    current = model(resources, dim, p, seed)
    optimizer = torch.optim.AdamW(current.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    history, best_state, best_ap, best_epoch, stale = [], None, -1., 0, 0
    for epoch in tqdm(range(1, p['epochs'] + 1), desc=f'TRV01 seed {seed}', unit='epoch'):
        current.train()
        order = list(ids['train'])
        random.Random(seed + epoch).shuffle(order)
        total, gradients = 0., Counter()
        for offset in tqdm(range(0, len(order), p['batch_size']), desc='Train CVE batches', unit='batch', leave=False):
            group = order[offset:offset + p['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss, _ = network.replay_backward(current, batches, batches.plan(group), 'security_tpg',
                                              torch.tensor([targets[c] for c in group], dtype=torch.float32), identity['class_weights'])
            for name, value in current.named_parameters():
                if value.grad is not None:
                    data.require(bool(torch.isfinite(value.grad).all()), 'Nonfinite gradient')
                    gradients[name.split('.')[0]] += float(value.grad.square().sum())
            optimizer.step()
            total += loss * len(group)
        probs = network.predict(current, batches, ids['validation'], 'security_tpg', p['batch_size'])
        metrics = base.development.metric([targets[c] for c in ids['validation']], probs)
        ap = metrics['average_precision']
        data.require(ap is not None and np.isfinite(ap), 'Invalid validation AP')
        if ap > best_ap:
            best_ap, best_epoch, stale, best_state = ap, epoch, 0, deepcopy(current.state_dict())
        else:
            stale += 1
        history.append({'epoch': epoch, 'train_weighted_bce': total / len(order), 'validation': metrics,
                        'data_gradient_l2_across_batches': {k: v**.5 for k, v in gradients.items()},
                        'best_epoch': best_epoch, 'stale': stale})
        print(f'[TRV01] seed={seed} epoch={epoch} loss={total / len(order):.6f} AP={ap:.6f} '
              f'F1@0.5={metrics["f1"]:.6f} best={best_epoch} patience={stale}/{p["patience"]}', flush=True)
        if stale >= p['patience']:
            break
    current.load_state_dict(best_state)
    probs = network.predict(current, batches, ids['validation'], 'security_tpg', p['batch_size'])
    rows = [{'cve': c, 'split': 'validation', 'seed': seed, 'actual': targets[c], 'predicted': float(v)}
            for c, v in zip(ids['validation'], probs)]
    torch.save({'state_dict': best_state, 'identity': data.digest(identity), 'readout': 'type_balanced',
                'seed': seed, 'best_epoch': best_epoch}, directory / 'checkpoint.pt')
    for name, value in {'history.json': history, 'predictions.json': rows,
                        'metrics.json': base.development.metric([targets[c] for c in ids['validation']], probs)}.items():
        (directory / name).write_text(data.dump(value))
    base.lock_tree(directory)
    return verify_fit(directory, resources, dim, identity, batches, targets, seed)


def run(scale, verify=False, check_only=False, resume=False):
    from ..forecast_matched_multisource import writer_lock
    batches, targets, resources, dim, identity, reference = context(scale)
    p, ids = identity['settings'], identity['ids']
    packet = PACKET / scale
    data.check_lock(packet)
    data.require(data.dump(data.read(packet / 'run_identity.json')) == data.dump(identity), 'Create/review the matching graph packet first')
    references = {}
    reference_identity = data.read(reference / 'run_identity.json')
    # JSON object keys are strings; torch checkpoints retain integer class keys.
    reference_identity['class_weights'] = {int(k): v for k, v in reference_identity['class_weights'].items()}
    for seed in tqdm(p['seeds'], desc='Reproduce original GNN reference', unit='seed'):
        _, metrics = base.verify_fit(reference / f'security_tpg__seed_{seed}', 'security_tpg', seed,
                                     reference_identity, resources, dim, batches, targets)
        references[str(seed)] = metrics
        a = base.model_for(resources, dim, p, seed)
        b = model(resources, dim, p, seed)
        data.require(all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()), 'Initialization differs')
    if check_only:
        print('TRV01 preflight passed; no training or test scoring.')
        return
    output = RESULT / scale
    output.parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(output.parent / '.writer.lock'):
        if output.exists():
            data.require(verify or resume, 'Run exists; use --resume or --verify-only')
            data.require(data.dump(data.read(output / 'run_identity.json')) == data.dump(identity), 'Run identity changed')
            if verify or (output / 'artifact_lock.json').exists():
                data.check_lock(output)
        else:
            data.require(not verify, 'Nothing to verify; run first')
            output.mkdir()
            (output / 'run_identity.json').write_text(data.dump(identity))
        metrics, predictions = {}, []
        for seed in tqdm(p['seeds'], desc='Matched type-readout fits', unit='seed'):
            directory = output / f'seed_{seed}'
            if directory.exists():
                rows, values = verify_fit(directory, resources, dim, identity, batches, targets, seed)
            else:
                data.require(not verify, 'Missing completed fit')
                stage = Path(tempfile.mkdtemp(prefix='.trv01-fit-', dir=output.parent))
                try:
                    rows, values = train_fit(stage, resources, dim, identity, batches, targets, seed)
                    data.require(not directory.exists(), 'Fit appeared')
                    stage.rename(directory)
                finally:
                    if stage.exists():
                        shutil.rmtree(stage)
            metrics[str(seed)] = values
            predictions.extend(rows)
        keys = ('average_precision', 'roc_auc', 'f1', 'precision', 'recall', 'accuracy', 'log_loss', 'brier')
        paired = {s: {k: metrics[s][k] - references[s][k] for k in keys} for s in metrics}
        summary = {'schema': identity['schema'], 'cves': {s: len(v) for s, v in ids.items()},
                   'metrics': metrics, 'original_node_mean_metrics': references,
                   'type_balanced_minus_node_mean': paired, 'test_scoring': False,
                   'limitations': ['Development-only; the earlier final cohort is already exposed.',
                                   'Mean readout is not a confirmed bug; balanced readout may help or harm.',
                                   'Semantic extraction judgments remain pending. No extractor repair is claimed.',
                                   'F1 uses 0.5 for this matched control, not a newly optimized cutoff.',
                                   'Description and structured inputs remain retrospective; no SMP/summaries in this first comparison.']}
        artifacts = {'metrics.json': metrics, 'predictions.json': predictions,
                     'predictions.csv': base.development.csv_text(predictions), 'summary.json': summary,
                     'summary.txt': 'TRV01 matched readout comparison\n' + data.dump(summary)}
        locked = verify or (output / 'artifact_lock.json').exists()
        for name, value in artifacts.items():
            text = value if isinstance(value, str) else data.dump(value)
            if locked:
                data.require((output / name).read_text() == text, 'Report differs: ' + name)
            else:
                (output / name).write_text(text)
        if not locked:
            base.lock_tree(output)
        data.check_pins(identity['pins'])
        preservation.verify()
    print(f'TRV01 {scale} ' + ('verified.' if verify else 'saved to ' + str(output)) + ' No test scoring.')
