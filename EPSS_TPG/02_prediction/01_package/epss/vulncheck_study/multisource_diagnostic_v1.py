"""VK08 frozen all-source reliance checks; thresholds fixed before interventions."""
from pathlib import Path
import hashlib

import numpy as np
import torch
from tqdm import tqdm

from . import threshold_selection_v1 as threshold
from . import multisource_training_v1 as training

data, dev, network = training.data, training.dev, training.network
ROOT = data.ROOT/'05_results/05_vulncheck_study/16_multisource_diagnostic_v1'
GROUPS = {'description': [0], 'github_summary': [1], 'non_github_summary': [2],
          'summaries': [1, 2], 'posts': [3], 'optional_sources': [1, 2, 3]}
CONDITIONS = {'baseline': ('none', [])}
for _name in ('description', 'summaries', 'posts', 'optional_sources'):
    CONDITIONS['zero_'+_name] = ('zero', GROUPS[_name])
    CONDITIONS['swap_'+_name] = ('swap', GROUPS[_name])
for _name in ('github_summary', 'non_github_summary', 'summaries', 'posts', 'optional_sources'):
    CONDITIONS['remove_'+_name] = ('remove', GROUPS[_name])
CONDITIONS['zero_structured'] = ('structured', [])


def source_donors(ids, mask):
    """A fixed label-blind cycle within each complete source-availability pattern."""
    data.require(len(ids) == len(set(ids)) == len(mask) and mask.shape[1] == 4, 'Invalid donor inputs')
    groups = {}
    for i, row in enumerate(mask.tolist()):
        groups.setdefault(tuple(row), []).append(i)
    mapping = list(range(len(ids)))
    for group in groups.values():
        order = sorted(group, key=lambda i: (hashlib.sha256(('vk08-donor:'+ids[i]).encode()).hexdigest(), ids[i]))
        for j, i in enumerate(order):
            mapping[i] = order[(j+1) % len(order)]
    mapping = torch.tensor(mapping, dtype=torch.long)
    data.require(sorted(mapping.tolist()) == list(range(len(ids))) and torch.equal(mask[mapping], mask), 'Invalid donor permutation')
    return mapping


def intervene(sources, mask, structured, condition, mapping):
    data.require(condition in CONDITIONS, 'Unknown intervention')
    z, m, t = sources.clone(), mask.clone(), structured.clone()
    action, columns = CONDITIONS[condition]
    if action == 'zero':
        z[:, columns] = 0
    elif action == 'swap':
        z[:, columns] = sources[mapping][:, columns]
    elif action == 'remove':
        m[:, columns] = False
        z[:, columns] = 0
    elif action == 'structured':
        t.zero_()
    data.require(bool(m.any(1).all()), 'Intervention removed every text source')
    return z, m, t


@torch.no_grad()
def source_scores(model, sources, mask, structured, batch_size):
    data.require(sources.shape[:2] == mask.shape and sources.shape[0] == len(structured)
                 and batch_size > 0, 'Invalid source scoring shapes')
    values, weights = [], []
    for i in range(0, len(sources), batch_size):
        z, m = sources[i:i+batch_size], mask[i:i+batch_size]
        if not bool(m[:, 1:].any()):
            text = z[:, 0]
            w = torch.zeros_like(m, dtype=z.dtype)
            w[:, 0] = 1
        else:
            text, w = network.masked_fusion(z, m, model.source_attention, model.source_bias)
        logits = model.head(torch.cat((text, structured[i:i+batch_size]), dim=1)).squeeze(1)
        data.require(bool(torch.isfinite(logits).all()), 'Nonfinite diagnostic logit')
        # Preserve the original batch-local sigmoid to avoid the former VK05 discrepancy.
        values.extend(logits.sigmoid().tolist())
        weights.extend(w.tolist())
    return np.array(values), weights


@torch.no_grad()
def encode(model, batches, ids, rep, batch_size):
    sources, masks, structured, direct = [], [], [], []
    for start in tqdm(range(0, len(ids), batch_size), desc='Read cached graphs and encode source representations', unit='batch', leave=False):
        plan = batches.plan(ids[start:start+batch_size])
        z = network.representations(model, batches, plan, rep)
        sources.append(network.pool_sources(z, plan))
        masks.append(plan['source_mask'])
        structured.append(model.tabular(plan['structured']))
        direct.extend(model.fuse(z, plan).sigmoid().tolist())
    return torch.cat(sources), torch.cat(masks), torch.cat(structured), np.array(direct)


def evidence_cases(ids, y, values, cutoff, rows, shared):
    chosen = {
        'unlisted_flagged': sorted((i for i in range(len(ids)) if y[i] == 0 and values[i] >= cutoff), key=lambda i: (-values[i], ids[i]))[:3],
        'listed_missed': sorted((i for i in range(len(ids)) if y[i] == 1 and values[i] < cutoff), key=lambda i: (values[i], ids[i]))[:3],
    }
    result = {}
    for kind, selected in chosen.items():
        result[kind] = []
        for i in selected:
            c = ids[i]
            documents = rows[c]['documents']
            result[kind].append({'cve': c, 'listed': int(y[i]), 'score': float(values[i]),
                'shared_wording': c in shared, 'source_mask': rows[c]['source_mask'],
                'evidence': {source: {'accepted_documents': sum(d['source_text_type'] == source for d in documents),
                    'examples': [{'document_id': d['document_id'], 'excerpt': d['text'][:800],
                                  'full_characters': len(d['text']), 'truncated': len(d['text']) > 800}
                                 for d in documents if d['source_text_type'] == source][:3]}
                            for source in training.prep.SOURCES}})
    return result


def run(scale='smoke', verify=False, check_only=False):
    source, export, saved_identity, ids, y, saved_scores, pins = threshold.load_predictions(scale)
    threshold_root = threshold.ROOT/scale
    data.check_lock(threshold_root)
    threshold_identity = data.read(threshold_root/'run_identity.json')
    data.require(threshold_identity['ids'] == ids and threshold_identity['scale'] == scale
                 and threshold_identity['source_identity_digest'] == data.digest(saved_identity)
                 and threshold_identity['policy'] == threshold.POLICY, 'Threshold scope/protocol mismatch')
    data.check_pins(threshold_identity['pins'])
    thresholds = data.read(threshold_root/'thresholds.json')
    cfg, batches, targets, resources, dim, original, _ = training.context('full')
    data.require(data.dump(original) == data.dump(saved_identity), 'Frozen model context changed')
    data.require([targets[c] for c in ids] == y.tolist(), 'Diagnostic label join changed')
    pins.update({str(threshold_root/'artifact_lock.json'): data.sha(threshold_root/'artifact_lock.json'),
                 str(Path(__file__)): data.sha(Path(__file__))})
    identity = {'schema': 'vk08_multisource_diagnostic_v1', 'scale': scale, 'ids': ids, 'pins': pins,
                'conditions': CONDITIONS, 'thresholds_frozen_before_interventions': True,
                'donor_policy': 'One label-blind cycle per full source-mask pattern; singletons unchanged and counted.',
                'training': False, 'test_scoring': False, 'new_embeddings': False}
    if check_only:
        print(f'[VK08] Diagnostic preflight: {len(ids)} validation CVEs, six frozen all-source models. No inference or test scoring.')
        return
    output = ROOT/scale
    data.require(output.exists() if verify else not output.exists(),
                 'Run without --verify-only to create missing output; use --verify-only for existing output')
    if verify:
        data.check_lock(output)
        data.require(data.dump(data.read(output/'run_identity.json')) == data.dump(identity), 'Diagnostic identity changed')
    shared = {c for g in data.read(export/'shared_text.json')['groups'] for c in g['validation']}
    subsets = {'all': np.ones(len(ids), dtype=bool),
               'no_recorded_shared_wording': np.array([c not in shared for c in ids]),
               'recorded_shared_wording': np.array([c in shared for c in ids])}
    results, predictions, cases, attention, comparisons, checks = {}, [], {}, [], [], {}
    batch_size = original['settings']['batch_size']
    fits = [(r, s) for r in cfg['representations'] for s in original['settings']['seeds']]
    donor_rows = None
    for rep, seed in tqdm(fits, desc='Diagnose six frozen all-source models', unit='model'):
        name = training.fit_key('DGNM', rep, seed)
        checkpoint_path = source/name/'checkpoint.pt'
        data.require(data.sha(checkpoint_path) == thresholds[name]['checkpoint_sha256'], 'Threshold/checkpoint mismatch')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        expected = training.fit_metadata(original, 'DGNM', rep, seed)
        data.require(all(checkpoint[k] == v for k, v in expected.items()) and checkpoint['settings'] == original['settings'], 'Wrong checkpoint')
        model = training.model_for(resources, dim, original['settings'], seed)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.eval().requires_grad_(False)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        z, mask, tabular, direct = encode(model, batches['DGNM'], ids, rep, batch_size)
        baseline, weights = source_scores(model, z, mask, tabular, batch_size)
        error = float(np.max(np.abs(baseline-saved_scores[name])))
        data.require(np.allclose(direct, saved_scores[name], atol=1e-7, rtol=0), 'Original baseline failed to reproduce: '+name)
        data.require(np.allclose(baseline, direct, atol=1e-7, rtol=0), 'Source pooling reconstruction differs: '+name)
        checks[name] = {'maximum_saved_probability_error': error, 'tolerance': 1e-7}
        mapping = source_donors(ids, mask)
        current_donors = [{'cve': c, 'donor_cve': ids[int(mapping[i])], 'source_mask': mask[i].tolist(),
                           'singleton_unchanged': int(mapping[i]) == i} for i, c in enumerate(ids)]
        data.require(donor_rows is None or donor_rows == current_donors, 'Models use different source donors')
        donor_rows = current_donors
        cutoff = thresholds[name]['threshold']
        data.require(np.array_equal(baseline >= cutoff, saved_scores[name] >= cutoff)
                     and np.array_equal(baseline >= .5, saved_scores[name] >= .5),
                     'Reconstructed baseline crosses a frozen decision boundary')
        results[name] = {}
        for condition in tqdm(CONDITIONS, desc='Fixed-threshold source interventions', unit='condition', leave=False):
            altered = intervene(z, mask, tabular, condition, mapping)
            values, _ = source_scores(model, *altered, batch_size)
            action, columns = CONDITIONS[condition]
            unaffected = ~mask[:, columns].any(1).numpy() if columns else np.zeros(len(ids), dtype=bool)
            if columns:
                data.require(np.allclose(values[unaffected], baseline[unaffected], atol=1e-7, rtol=0), 'Missing-source negative control changed')
            if action == 'swap':
                unchanged = (mapping == torch.arange(len(ids))).numpy()
                data.require(np.allclose(values[unchanged], baseline[unchanged], atol=1e-7, rtol=0), 'Singleton swap changed')
            reports = {}
            for subset, keep in subsets.items():
                if not keep.any():
                    reports[subset] = {'cves': 0, 'metrics': None}
                    continue
                a, p, b = y[keep], values[keep], baseline[keep]
                ranked = dev.metric(a, p)
                baseline_ranked = dev.metric(a, b)
                fixed = threshold.decision_metrics(a, p >= .5)
                selected = threshold.decision_metrics(a, p >= cutoff)
                base_selected = threshold.decision_metrics(a, b >= cutoff)
                report = {'cves': int(keep.sum()), 'listed': int(a.sum()), 'average_precision': ranked['average_precision'],
                          'roc_auc': ranked['roc_auc'], 'fixed_0_5': fixed, 'frozen_selected_threshold': selected,
                          'AP_change': training.metric_difference(ranked['average_precision'], baseline_ranked['average_precision']),
                          'F1_change_at_frozen_threshold': selected['f1']-base_selected['f1'],
                          'mean_absolute_score_change': float(np.abs(p-b).mean()),
                          'decision_flips_frozen_threshold': int(((p >= cutoff) != (b >= cutoff)).sum())}
                reports[subset] = report
                comparisons.append({'model': name, 'condition': condition, 'subset': subset,
                                    'threshold': cutoff, **{k: v for k, v in report.items() if not isinstance(v, dict)},
                                    'fixed_f1': fixed['f1'], 'selected_f1': selected['f1']})
            results[name][condition] = reports
            predictions.extend({'model': name, 'condition': condition, 'cve': c, 'split': 'validation',
                                'actual': int(y[i]), 'score': float(values[i]), 'frozen_threshold': cutoff,
                                'selected_prediction': int(values[i] >= cutoff)} for i, c in enumerate(ids))
        attention.extend({'model': name, 'cve': c, 'source_weights': weights[i], 'source_mask': mask[i].tolist()}
                         for i, c in enumerate(ids))
        cases[name] = {'threshold': cutoff, 'cases': evidence_cases(ids, y, baseline, cutoff, batches['DGNM'].rows, shared)}
        data.require(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
                     and all(p.grad is None for p in model.parameters()), 'Frozen model changed')
        print(f'[VK08] {name}: baseline reproduced; threshold={cutoff:.9g} held fixed; parameters unchanged.', flush=True)
    limitations = [
        'All results are development sensitivity checks. Validation has informed checkpoints and threshold selection.',
        'No threshold is retuned on interventions or wording subsets; no model/seed winner is automatically selected.',
        'Mask removal reweights available sources. Zeroing keeps original masks; neither is matched retraining.',
        'Donor swaps preserve availability, not semantic plausibility. One deterministic permutation is not a significance test.',
        'Representation interventions may be out of distribution; they do not establish causal source effects.',
        'Attention weights are descriptive, not proof of causal importance. Shared wording does not mean duplicate CVEs.',
        'Confident cases contain bounded excerpts, not independently verified source claims. Unlisted does not mean never exploited.',
        'Smoke uses 64 CVEs from complete original validation batches; smoke thresholds are not deployment thresholds.',
    ]
    summary = {'validation_cves': len(ids), 'listed': int(y.sum()), 'baseline_checks': checks,
               'results': results, 'limitations': limitations, 'training': False, 'test_scoring': False}
    lines = ['VK08 frozen multi-source diagnostic', '', *limitations, '',
             'Model | Intervention | Validation subset | AP change | F1 change at frozen threshold | Mean score change']
    lines += [f'{r["model"]} | {r["condition"]} | {r["subset"]} | {r["AP_change"]} | '
              f'{r["F1_change_at_frozen_threshold"]:.6f} | {r["mean_absolute_score_change"]:.6f}' for r in comparisons]
    artifacts = {'run_identity.json': identity, 'summary.json': summary, 'summary.txt': '\n'.join(lines)+'\n',
                 'comparisons.csv': dev.csv_text(comparisons), 'predictions.json': predictions,
                 'source_donors.json': donor_rows, 'baseline_attention.json': attention, 'confident_cases.json': cases}
    data.check_pins(pins)
    threshold.save_report(output, artifacts, verify)
    print(f'VK08 {scale} source diagnostic {"reproduced" if verify else "saved to "+str(output)}. No training or test scoring.')
