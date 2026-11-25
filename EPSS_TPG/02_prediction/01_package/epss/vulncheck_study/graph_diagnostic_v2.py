"""VK05 v2: preserve the original per-batch sigmoid evaluation.

The v1 helper definitions and its archived artifacts remain unchanged. The
versioned runner retains every intervention and the original 1e-7 tolerance.
"""
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
from tqdm import tqdm

from . import graph_diagnostic_v1 as legacy
from .graph_diagnostic_v1 import (
    data, dev, training, BRANCHES, edge_groups, selected_ids, donors,
    branch_values, encode_views, compare, mistake_cases,
)

ROOT = data.ROOT / '05_results/05_vulncheck_study/09_graph_diagnostic_v2'


@torch.no_grad()
def scores(model, text, structured, batch_size):
    data.require(batch_size > 0 and len(text) > 0 and text.shape == structured.shape,
                 'Invalid diagnostic latent batch')
    logit_batches, probability_batches = [], []
    for i in range(0, len(text), batch_size):
        logits = model.head(torch.cat((text[i:i+batch_size], structured[i:i+batch_size]), dim=1)).squeeze(1)
        data.require(bool(torch.isfinite(logits).all()), 'Nonfinite diagnostic score')
        logit_batches.append(logits)
        # Match graph_model_v1.predict: sigmoid before joining the CVE batches.
        probability_batches.append(logits.sigmoid())
    return torch.cat(probability_batches).tolist(), torch.cat(logit_batches).tolist()


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
            str(Path(__file__)): data.sha(Path(__file__)),
            str(Path(legacy.__file__)): data.sha(Path(legacy.__file__))}
    identity = {'schema': 'vk05_frozen_diagnostic_v2', 'scale': scale, 'ids': ids, 'pins': pins,
                'edge_groups_from_saved_vocabulary': groups, 'branch_conditions': list(BRANCHES),
                'donor_cve_by_recipient': {c: ids[int(mapping[i])] for i, c in enumerate(ids)},
                'score_policy': 'Head and sigmoid applied per original CVE batch before concatenation; baseline atol=1e-7 unchanged.',
                'models': ['pooled_text', 'security_tpg'], 'seeds': cfg['seeds'], 'threshold': .5,
                'smoke_selection': '64 validation CVEs: eight full original batches selected by fixed hash of their CVE IDs, not labels or scores. Original batch boundaries/order preserved.',
                'intervention': 'Zero post-encoder latents or swap whole latents between fixed CVE donors; never zero raw categories.',
                'training': False, 'test_scoring': False, 'threshold_tuning': False,
                'interpretation': 'Frozen-model sensitivity, not matched retraining, causal importance or independent evaluation.'}
    print(f'[VK05 v2] {len(ids)} validation CVEs, six FULL-trained checkpoints; {len(groups)} edge removals; no fitting.', flush=True)
    if check_only:
        print('VK05 v2 preflight passed; no diagnostic files written or model scoring.')
        return
    output = ROOT / scale
    data.require(output.exists() if verify else not output.exists(),
                 f'No saved diagnostic at {output}; run without --verify-only first.' if verify else
                 f'Diagnostic already exists at {output}; use --verify-only.')
    if verify:
        data.check_lock(output)
        data.require(data.read(output / 'run_identity.json') == identity, 'Diagnostic identity differs')
    saved = data.read(source / 'predictions.json')
    results, predictions, baseline_maps, baseline_checks, coverage = {}, [], {}, {}, None
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
        difference = np.abs(np.asarray(baseline) - [expected[c] for c in ids])
        worst = int(difference.argmax())
        baseline_checks[name] = {'max_absolute_score_difference': float(difference[worst]),
                                 'worst_cve': ids[worst], 'tolerance': 1e-7}
        data.require(np.allclose(baseline, [expected[c] for c in ids], atol=1e-7, rtol=0),
                     f'Baseline mismatch: {name}, {ids[worst]}, saved={expected[ids[worst]]}, '
                     f'recomputed={baseline[worst]}, max difference={difference[worst]:.12g}; tolerance=1e-7')
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
        print(f'[VK05 v2] {name}: baseline reproduced, interventions complete, parameters unchanged.', flush=True)
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
               'baseline_reproduction': baseline_checks, 'model_interventions': results, 'aggregate_sensitivity': aggregate, 'logistic_references': references,
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
        print(f'VK05 v2 {scale} frozen diagnostic reproduced. No training, embeddings or test scoring.')
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
    print(f'VK05 v2 {scale} frozen diagnostic saved to {output}. No training, embeddings or test scoring.')
