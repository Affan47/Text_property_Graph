"""VK05 frozen VK04 branch/edge sensitivity; development data only."""
import hashlib
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
from tqdm import tqdm

from . import adapter_v2 as data, development_v1 as dev, graph_training_v1 as training
from ..primary_study.model import mean_groups

BRANCHES = ('baseline', 'zero_description', 'zero_structured', 'zero_both',
            'swap_description', 'swap_structured')
GROUPS = {
    'dependency': ('DEP',),
    'sequence': ('NEXT_TOKEN', 'NEXT_SENT', 'NEXT_PARA'),
    'containment': ('CONTAINS', 'BELONGS_TO'),
    'semantic': ('SRL_ARG', 'AMR_EDGE', 'ENTITY_REL'),
    'discourse': ('DISCOURSE', 'RST_RELATION'),
    'referential': ('COREF', 'SIMILARITY'),
}
ROOT = data.ROOT / '05_results/05_vulncheck_study/09_graph_diagnostic_v1'


def edge_groups(vocab):
    data.require(sorted(vocab.values()) == list(range(len(vocab))), 'Invalid saved vocabulary')
    groups = {'security': tuple(k for k in vocab if k.startswith('SEC_')), **GROUPS}
    names = [name for group in groups.values() for name in group if name in vocab]
    data.require(len(names) == len(set(names)) and set(names) == set(vocab),
                 'Unmapped/overlapping edge vocabulary; explicitly review new types')
    return {**{group: [vocab[k] for k in keys if k in vocab] for group, keys in groups.items()},
            'all': list(vocab.values())}


def remove_edges(graph, edge_ids):
    """Return a separate graph with all edge-aligned arrays filtered together."""
    result = graph.clone()
    keep = ~torch.isin(graph.edge_type, graph.edge_type.new_tensor(edge_ids))
    result.edge_index = graph.edge_index[:, keep]
    result.edge_type = graph.edge_type[keep]
    result.edge_features = graph.edge_features[keep]
    data.require(result.edge_features.shape[0] == result.edge_type.numel() == result.edge_index.shape[1],
                 'Edge arrays are misaligned')
    return result, ~keep


def selected_ids(ids, scale, batch_size=8):
    data.require(scale in ('smoke', 'full') and len(ids) == len(set(ids)), 'Invalid cohort')
    if scale == 'full':
        return list(ids)
    data.require(batch_size > 0 and 64 % batch_size == 0, 'Smoke size must preserve original batch boundaries')
    blocks = [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)]
    complete = [block for block in blocks if len(block) == batch_size]
    if len(ids) <= 64:
        return list(ids)
    selected = {c for block in sorted(complete, key=lambda b: hashlib.sha256(('vk05-smoke:'+','.join(b)).encode()).hexdigest())[:64//batch_size]
                for c in block}
    return [c for c in ids if c in selected]


def donors(ids):
    """One label-independent derangement shared by all models and interventions."""
    data.require(len(ids) >= 2 and len(ids) == len(set(ids)), 'Need distinct CVEs')
    order = sorted(range(len(ids)), key=lambda i: hashlib.sha256(('vk05-donor:'+ids[i]).encode()).hexdigest())
    mapping = [None] * len(ids)
    for i, recipient in enumerate(order):
        mapping[recipient] = order[(i+1) % len(order)]
    data.require(sorted(mapping) == list(range(len(ids))) and all(i != j for i, j in enumerate(mapping)),
                 'Invalid donor permutation')
    return torch.tensor(mapping, dtype=torch.long)


def branch_values(text, structured, condition, mapping):
    data.require(condition in BRANCHES and text.shape == structured.shape, 'Invalid branch intervention')
    if condition in ('zero_description', 'zero_both'):
        text = torch.zeros_like(text)
    if condition in ('zero_structured', 'zero_both'):
        structured = torch.zeros_like(structured)
    if condition == 'swap_description':
        text = text[mapping]
    if condition == 'swap_structured':
        structured = structured[mapping]
    return text, structured


@torch.no_grad()
def encode_views(model, batches, ids, rep, groups, batch_size):
    parts = {'baseline': [], **{'remove_'+g: [] for g in groups}}
    structured, coverage = [], []
    for offset in tqdm(range(0, len(ids), batch_size), desc=f'Encode frozen {rep} edge views', unit='batch', leave=False):
        current = ids[offset:offset+batch_size]
        plan = batches.plan(current)
        encoded = {key: [] for key in parts}
        counts = {group: torch.zeros(len(current), dtype=torch.long) for group in groups}
        totals = torch.zeros(len(current), dtype=torch.long)
        for start, graph in batches.packets(plan, {'edge_view': 'no_coref'}):
            owner = plan['chunk_cve'][start + graph.batch[graph.edge_index[0]]]
            totals += torch.bincount(owner, minlength=len(current))
            encoded['baseline'].append(model.encode(graph, rep))
            for group, edge_ids in groups.items():
                changed, removed = remove_edges(graph, edge_ids)
                counts[group] += torch.bincount(owner[removed], minlength=len(current))
                encoded['remove_'+group].append(model.encode(changed, rep))
        for key, values in encoded.items():
            parts[key].append(mean_groups(torch.cat(values), plan['chunk_cve'], len(current)))
        structured.append(model.tabular(plan['structured']))
        coverage += [{'cve': c, 'total_edges': int(totals[i]),
                      'removed_edges': {g: int(v[i]) for g, v in counts.items()}} for i, c in enumerate(current)]
    return {k: torch.cat(v) for k, v in parts.items()}, torch.cat(structured), coverage


@torch.no_grad()
def scores(model, text, structured, batch_size):
    logits = torch.cat([model.head(torch.cat((text[i:i+batch_size], structured[i:i+batch_size]), dim=1)).squeeze(1)
                        for i in range(0, len(text), batch_size)])
    data.require(bool(torch.isfinite(logits).all()), 'Nonfinite diagnostic score')
    return logits.sigmoid().tolist(), logits.tolist()


def compare(y, prediction, baseline, logits, base_logits):
    current = dev.metric(y, prediction)
    previous = dev.metric(y, baseline)
    p, b = np.asarray(prediction), np.asarray(baseline)
    return {'metrics': current,
            'metric_changes': {k: None if current[k] is None or previous[k] is None else current[k]-previous[k]
                               for k in ('average_precision', 'roc_auc', 'precision', 'recall', 'f1', 'log_loss', 'brier')},
            'mean_absolute_score_change': float(np.abs(p-b).mean()),
            'max_absolute_score_change': float(np.abs(p-b).max()),
            'mean_absolute_logit_change': float(np.abs(np.asarray(logits)-np.asarray(base_logits)).mean()),
            'decision_flips_at_0_5': int(((p >= .5) != (b >= .5)).sum())}


def mistake_cases(ids, labels, score_maps, rows, feature_names):
    chosen, summaries = {}, {}
    for name, values in score_maps.items():
        fp = sorted([i for i, c in enumerate(ids) if labels[c] == 0 and values[i] >= .5],
                    key=lambda i: (-values[i], ids[i]))
        fn = sorted([i for i, c in enumerate(ids) if labels[c] == 1 and values[i] < .5],
                    key=lambda i: (values[i], ids[i]))
        summaries[name] = {'false_positives': len(fp), 'false_negatives': len(fn),
                           'false_positives_score_at_least_0_9': sum(values[i] >= .9 for i in fp),
                           'false_negatives_score_at_most_0_1': sum(values[i] <= .1 for i in fn)}
        for kind, indices in (('false_positive', fp), ('false_negative', fn)):
            for i in indices[:3]:
                chosen.setdefault(ids[i], []).append({'model': name, 'reason': kind})
    positions = {c: i for i, c in enumerate(ids)}
    examples = []
    for c in sorted(chosen):
        docs = [d for d in rows[c]['documents'] if d['source_text_type'] == 'description']
        data.require(len(docs) == 1, 'Expected one description for review')
        text = docs[0]['text']
        examples.append({'cve': c, 'actual_listed': labels[c], 'selection_reasons': chosen[c],
                         'description_excerpt': text[:800], 'excerpt_truncated': len(text) > 800,
                         'description_sha256': hashlib.sha256(text.encode()).hexdigest(),
                         'encoded_structured_values': dict(zip(feature_names, rows[c]['structured'])),
                         'scores': {name: values[positions[c]] for name, values in score_maps.items()},
                         'interpretation': 'Mistake relative to saved catalogue label; not proof of factual exploitation status.'})
    return summaries, examples


def run(scale='smoke', verify=False, check_only=False):
    batches, labels, resources, dim, original = training.context('full')
    cfg = original['settings']
    source = data.ROOT / cfg['result_root'] / 'full'
    reference = data.ROOT / cfg['reference_root']
    for directory in (source, reference):
        data.check_lock(directory)
    data.require(data.dump(data.read(source / 'run_identity.json')) == data.dump(original), 'Full training identity differs')
    ids = selected_ids(original['ids']['validation'], scale, cfg['batch_size'])
    data.require(set(ids) <= set(original['ids']['validation']) and not set(ids) & set(original['ids']['train']), 'Wrong diagnostic role')
    groups = edge_groups(resources['type_vocabulary']['edge_types'])
    mapping = donors(ids)
    pins = {**original['pins'], str(source / 'artifact_lock.json'): data.sha(source / 'artifact_lock.json'),
            str(reference / 'artifact_lock.json'): data.sha(reference / 'artifact_lock.json'),
            str(Path(__file__)): data.sha(Path(__file__))}
    identity = {'schema': 'vk05_frozen_diagnostic_v1', 'scale': scale, 'ids': ids, 'pins': pins,
                'edge_groups_from_saved_vocabulary': groups, 'branch_conditions': list(BRANCHES),
                'donor_cve_by_recipient': {c: ids[int(mapping[i])] for i, c in enumerate(ids)},
                'models': ['pooled_text', 'security_tpg'], 'seeds': cfg['seeds'], 'threshold': .5,
                'smoke_selection': '64 validation CVEs: eight full original batches selected by fixed hash of their CVE IDs, not labels or scores. Original batch boundaries/order preserved.',
                'intervention': 'Zero post-encoder latents or swap whole latents between fixed CVE donors; never zero raw categories.',
                'training': False, 'test_scoring': False, 'threshold_tuning': False,
                'interpretation': 'Frozen-model sensitivity, not matched retraining, causal importance or independent evaluation.'}
    print(f'[VK05] {len(ids)} validation CVEs, six FULL-trained checkpoints; {len(groups)} edge removals; no fitting.', flush=True)
    if check_only:
        print('VK05 preflight passed; no diagnostic files written or model scoring.')
        return
    output = ROOT / scale
    data.require(output.exists() if verify else not output.exists(), 'Use --verify-only for existing output; cannot verify missing output')
    if verify:
        data.check_lock(output)
        data.require(data.read(output / 'run_identity.json') == identity, 'Diagnostic identity differs')
    saved = data.read(source / 'predictions.json')
    results, predictions, baseline_maps, coverage = {}, [], {}, None
    y = [labels[c] for c in ids]
    fits = [(rep, seed) for rep in cfg['representations'] for seed in cfg['seeds']]
    for rep, seed in tqdm(fits, desc='Diagnose six frozen full-trained models', unit='model'):
        name = f'{rep}__seed_{seed}'
        checkpoint = torch.load(source / name / 'checkpoint.pt', map_location='cpu', weights_only=True)
        data.require(checkpoint['identity_digest'] == data.digest(original) and checkpoint['representation'] == rep
                     and checkpoint['seed'] == seed and checkpoint['settings'] == cfg, 'Wrong checkpoint')
        model = training.model_for(resources, dim, cfg, seed)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.eval().requires_grad_(False)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        views, structured, current_coverage = encode_views(model, batches, ids, rep, groups, cfg['batch_size'])
        if coverage is None:
            coverage = current_coverage
        data.require(coverage == current_coverage, 'Models saw different graph edges')
        baseline, base_logits = scores(model, views['baseline'], structured, cfg['batch_size'])
        saved_rows = [r for r in saved if r['model'] == rep and r['seed'] == seed]
        data.require([r['cve'] for r in saved_rows] == original['ids']['validation'] and
                     all(r['split'] == 'validation' and r['actual'] == labels[r['cve']] for r in saved_rows),
                     'Saved baseline labels or CVE ordering changed')
        expected = {r['cve']: r['predicted'] for r in saved_rows}
        data.require(np.allclose(baseline, [expected[c] for c in ids], atol=1e-7, rtol=0), 'Baseline does not reproduce saved predictions')
        baseline_maps[name] = baseline
        conditions = {condition: branch_values(views['baseline'], structured, condition, mapping) for condition in BRANCHES}
        conditions.update({key: (value, structured) for key, value in views.items() if key != 'baseline'})
        results[name] = {}
        for condition, (text, tabular) in conditions.items():
            values, logits = scores(model, text, tabular, cfg['batch_size'])
            report = compare(y, values, baseline, logits, base_logits)
            if condition.startswith('remove_'):
                group = condition[len('remove_'):]
                affected = [i for i, r in enumerate(coverage) if r['removed_edges'][group] > 0]
                unaffected = sorted(set(range(len(ids))) - set(affected))
                data.require(np.allclose(np.asarray(values)[unaffected], np.asarray(baseline)[unaffected], atol=1e-7, rtol=0), 'No-edge-removal CVE changed')
                if rep == 'pooled_text':
                    data.require(np.allclose(values, baseline, atol=1e-7, rtol=0), 'Non-graph negative control changed with edges')
                report['affected_cves'] = len(affected)
                report['removed_edges'] = sum(r['removed_edges'][group] for r in coverage)
                report['affected_subset'] = compare([y[i] for i in affected], [values[i] for i in affected],
                    [baseline[i] for i in affected], [logits[i] for i in affected], [base_logits[i] for i in affected]) if affected else None
            results[name][condition] = report
            predictions.extend({'model': name, 'condition': condition, 'cve': c, 'split': 'validation',
                                'actual': y[i], 'score': values[i], 'logit': logits[i]} for i, c in enumerate(ids))
        data.require(all(torch.equal(v, state[k]) for k, v in model.state_dict().items()) and
                     all(p.grad is None for p in model.parameters()), 'Frozen model mutated or accumulated gradients')
        print(f'[VK05] {name}: baseline reproduced, interventions complete, parameters unchanged.', flush=True)
    references = {}
    ref_predictions = data.read(reference / 'predictions.json')
    for name in ('description_tfidf_structured__balanced', 'description_tfidf_structured__unweighted'):
        rows = {r['cve']: r for r in ref_predictions if r['model'] == name}
        data.require(all(rows[c]['split'] == 'validation' and rows[c]['actual'] == labels[c] for c in ids), 'Reference label join differs')
        values = [rows[c]['predicted'] for c in ids]
        baseline_maps['logistic:'+name] = values
        references[name] = dev.metric(y, values)
    state = data.read(data.ROOT / dev.config()['data_root'] / 'structured_encoder.json')
    errors, examples = mistake_cases(ids, labels, baseline_maps, batches.rows, state['feature_names'])
    aggregate = {rep: {condition: {key: {'mean': float(np.mean([results[f'{rep}__seed_{s}'][condition][key] for s in cfg['seeds']])),
                                         'std_population': float(np.std([results[f'{rep}__seed_{s}'][condition][key] for s in cfg['seeds']]))}
                                     for key in ('mean_absolute_score_change', 'mean_absolute_logit_change', 'decision_flips_at_0_5')}
                       for condition in results[f'{rep}__seed_{cfg["seeds"][0]}']} for rep in cfg['representations']}
    limitations = [
        'Zero latents and mismatched donors can create out-of-distribution inputs. Changes measure sensitivity, not causal contribution.',
        'Swapping uses one fixed label-independent derangement, not a permutation significance test. Smoke donors differ from full donors.',
        'Edge removal retains node types, node text, embeddings and pooling. All-edge removal still retains node projections and residual layer behavior.',
        'Security removal retains security-related nodes and their embeddings. A null effect does not prove the security information is absent.',
        'Pooled edge-removal predictions must remain unchanged: this is an implementation negative control.',
        'Report affected-subset comparisons against the same original subset. Do not compare affected CVEs directly to a different unaffected cohort as a causal test.',
        'Top-three errors per model/direction are selected examples, not a representative sample. Encoded structured values are not raw CVSS values.',
        'False positives mean not listed in the saved snapshot, not never exploited. Scores are not calibrated probabilities.',
        'Repeated validation, shared text, retrospective inputs and prior test exposure remain limitations. No independent-test claim.'
    ]
    summary = {'schema': identity['schema'], 'scale': scale, 'validation_cves': len(ids), 'listed': sum(y),
               'model_interventions': results, 'aggregate_sensitivity': aggregate, 'logistic_references': references,
               'error_counts': errors, 'limitations': limitations, 'training': False, 'test_scoring': False}
    lines = ['VK05 frozen description/structured and graph sensitivity',
             f'Validation CVEs: {len(ids)}; listed: {sum(y)}; six FULL-trained models; fixed threshold 0.5.',
             'AP change = perturbed AP minus original AP on the same CVEs. Negative means ranking worsened.',
             'Score change = mean absolute difference; flips count changed decisions at 0.5.', '',
             'Model | Condition | AP | AP change | Mean score change | Flips | Affected CVEs (edge removal only)']
    for name, conditions in results.items():
        for condition, r in conditions.items():
            lines.append(f'{name} | {condition} | {r["metrics"]["average_precision"]} | {r["metric_changes"]["average_precision"]} | '
                         f'{r["mean_absolute_score_change"]:.8f} | {r["decision_flips_at_0_5"]} | {r.get("affected_cves", "not applicable")}')
    lines += ['', 'Confident-mistake counts:', data.dump(errors), '', 'Limitations:', *limitations,
              '', 'Read confident_mistakes.json for real excerpts, encoded features and all baseline model scores.',
              'Predictions retain every CVE/condition; graph_coverage.json records exact removed-edge counts.']
    artifacts = {'run_identity.json': identity, 'summary.json': summary, 'summary.txt': '\n'.join(lines)+'\n',
                 'predictions.json': predictions, 'predictions.csv': dev.csv_text(predictions),
                 'graph_coverage.json': coverage, 'confident_mistakes.json': examples}
    data.check_pins(pins)
    data.check_lock(source)
    data.check_lock(reference)
    if verify:
        for name, value in artifacts.items():
            data.require((output / name).read_text() == (value if isinstance(value, str) else data.dump(value)), 'Diagnostic differs: '+name)
        print(f'VK05 {scale} frozen diagnostic reproduced. No training, embeddings or test scoring.')
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.vk05-', dir=ROOT))
    try:
        for name, value in artifacts.items():
            (stage / name).write_text(value if isinstance(value, str) else data.dump(value))
        training.lock_tree(stage)
        data.require(not output.exists(), 'Output appeared; refuse overwrite')
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f'VK05 {scale} frozen diagnostic saved to {output}. No training, embeddings or test scoring.')
