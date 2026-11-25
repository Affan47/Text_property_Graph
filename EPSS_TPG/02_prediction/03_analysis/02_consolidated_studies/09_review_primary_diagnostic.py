"""Interpret saved PS03 predictions; no inference, training or test access."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss.primary_diagnostics import interventions, runner

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / '05_results/04_consolidated_studies/01_primary_day60/saved_model_diagnostic_v1/full'
OUTPUT = SOURCE.parent.parent / 'diagnostic_review_v1'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run():
    lock = read(SOURCE / 'diagnostic_lock.json')
    for name, expected in tqdm(lock['files'].items(), desc='Verify original diagnostic hashes', unit='file'):
        require(sha(SOURCE / name) == expected, 'Changed diagnostic: ' + name)
    identity = read(SOURCE / 'run_identity.json')
    summary = read(SOURCE / 'analysis.json')
    require(identity['scale'] == 'full' and not identity['training'] and not identity['test_scoring'], 'Wrong scope')
    ids = identity['ids']
    require(len(ids) == len(set(ids)) == 1104 and len(identity['model_ids']) == 18, 'Unexpected cohort')
    results, rows, attention, predictions = [], [], [], []
    for key in tqdm(identity['model_ids'], desc='Recompute saved prediction differences', unit='model'):
        result = read(SOURCE / 'models' / key / 'diagnostic.json')
        require(result['baseline_reproduced'] and not result['parameter_updates'], 'Failed original checks')
        groups = defaultdict(list)
        for row in result['predictions']:
            groups[row['intervention']].append(row)
        baseline = groups['baseline']
        y = np.array([r['actual'] for r in baseline])
        b = np.array([r['predicted'] for r in baseline])
        weights = np.array([r['source_weights'] for r in baseline])
        attention.append({'model': key, 'seed': result['seed'], 'configuration': result['configuration'],
                          'mean_attention_all_cves': dict(zip(('description', 'github_summary', 'non_github_summary', 'smp'), weights.mean(axis=0).tolist()))})
        for mode, members in groups.items():
            require([r['cve'] for r in members] == ids, 'Changed CVE order')
            require(all(r['actual'] == baseline[i]['actual'] and r['baseline_prediction'] == b[i]
                        for i, r in enumerate(members)), 'Changed pairing')
            p = np.array([r['predicted'] for r in members])
            affected = np.array([r['affected'] for r in members])
            reproduced = interventions.response(y, b, p, affected, summary['response_tolerance'])
            require(reproduced == result['summaries'][mode]['response'], 'Saved arithmetic differs: ' + key + mode)
            delta = abs(y - p) - abs(y - b)
            shift = abs(p - b)
            without_influential = np.array([c != 'CVE-2025-0108' for c in ids])
            rows.append({'model': key, 'configuration': result['configuration'], 'seed': result['seed'],
                         'intervention': mode, 'affected_cves': int(affected.sum()),
                         'baseline_mae': float(abs(y-b).mean()), 'intervention_mae': float(abs(y-p).mean()),
                         'mae_change': float(delta.mean()), 'mean_absolute_prediction_change': float(shift.mean()),
                         'maximum_absolute_prediction_change': float(shift.max()),
                         'median_absolute_prediction_change': float(np.median(shift)),
                         'mae_change_excluding_CVE_2025_0108_descriptive_only': float(delta[without_influential].mean())})
        for cve in ('CVE-2025-0108', 'CVE-2024-48248', 'CVE-2025-1974'):
            r = baseline[ids.index(cve)]
            predictions.append({'model': key, 'cve': cve, 'actual': r['actual'], 'predicted': r['predicted']})
        results.append(result)
    require(runner.aggregate(results) == summary['intervention_summary'], 'Aggregate report differs')
    evidence = read(SOURCE / 'evidence_review.json')
    docs = [(case['cve'], d) for case in evidence['cases'] for d in case['documents']]
    def count(predicate):
        selected = [(c, d) for c, d in docs if predicate(d)]
        return {'document_occurrences': len(selected), 'cves': len({c for c, _ in selected})}
    inventory = {
        'selected_cves': len(evidence['cases']), 'document_occurrences': len(docs),
        'document_occurrences_by_source': dict(Counter(d['source'] for _, d in docs)),
        'documents_mentioning_other_cves': count(lambda d: d['other_cve_mentions'] > 0),
        'documents_shared_with_training_cves': count(lambda d: d['exact_text_shared_with_other_cves']['train'] > 0),
        'replacement_characters': count(lambda d: d['replacement_characters'] > 0),
        'non_whitespace_controls': count(lambda d: d['non_whitespace_controls'] > 0),
        'lexical_cues': {cue: count(lambda d: d['review_cues'][cue]['matches'] > 0)
                        for cue in docs[0][1]['review_cues']},
        'cves_with_candidate_edges_naming_other_cves': sum(c['security_edges_naming_other_cves'] > 0 for c in evidence['cases']),
        'candidate_edges_naming_other_cves': sum(c['security_edges_naming_other_cves'] for c in evidence['cases']),
        'interpretation': 'Outcome-selected cases, not dataset prevalence. Repeated document occurrences are not unique texts. Other-CVE mentions or shared articles do not prove duplicate vulnerabilities or irrelevant text.'}
    examples = []
    for c, d in docs:
        if (c == 'CVE-2024-48248' and d['characters'] == 20012) or (c == 'CVE-2025-0108' and d['source'] == 'non_github_summary'):
            examples.append({'cve': c, 'source': d['source'], 'source_file': d['source_file'],
                             'characters': d['characters'], 'other_cve_ids': d['other_cve_ids'],
                             'first_text_excerpt': d['first_text_excerpt'], 'other_cve_examples': d['other_cve_examples']})
    findings = [
        'Saved arithmetic and aggregate summaries reproduce for all 18 checkpoints and 1,104 validation CVEs. This review does not rerun inference; the original --verify-only command does.',
        'Security-edge removal has very small average effects. All-edge removal has larger individual effects but mixed seed-level MAE changes. These results do not demonstrate useful security-edge contributions.',
        'All-source TPG S1 attention is seed-dependent: SMP dominates seeds 17 and 29; description dominates seed 43. Attention is descriptive, not causal importance.',
        'Mask-matched text permutation worsens all-source TPG S1 MAE for every seed, including when CVE-2025-0108 is excluded descriptively. This supports sensitivity to CVE-text matching, not graph superiority.',
        'Removing SMP or description can create large artificial errors, but these are inference-time missing-source patterns, not results from models trained without those sources.',
        'Removing CVSS or reference count worsens all-source TPG S1 MAE in all three seeds. Current structured fields remain retrospective, not verified historical inputs.',
        'Summary removals have small average effects; GitHub summaries exist for 201 validation CVEs and non-GitHub summaries for 745. Low average effects do not establish poor summary quality.',
        'The selected evidence includes multi-CVE articles and shared texts. Target-focused content and contextual references must be distinguished manually; CVEs remain distinct.',
        'No selected document triggers the implemented access-failure/missing-information/markup cues. This does not establish successful scraping or factual summary accuracy.',
        'Recommended next: a bounded subset of registered source comparisons with structured features fixed: D, DM, DGN and DGNM, each pooled-text and TPG, at seeds 17/29/43. Reuse the 12 completed D/DGNM fits; DM and DGN require 12 new fits. This subset needs a versioned selector; the current sources CLI launches the much larger complete suite.',
        'A separate matched edge-removal training comparison is needed for any claim about graph/security-edge benefit. Freeze final model and reporting choices before opening the test set.'
    ]
    report = {'schema': 'ps03_interpretation_v1', 'source_lock_sha256': sha(SOURCE / 'diagnostic_lock.json'),
              'review_script_sha256': sha(Path(__file__)), 'validation_cves': len(ids), 'frozen_models': len(results),
              'per_seed_interventions': rows, 'source_attention': attention,
              'source_evidence_inventory': inventory, 'real_text_examples': examples,
              'real_prediction_examples': predictions, 'findings': findings,
              'training': False, 'new_predictions': False, 'test_scoring': False,
              'limitations': summary['limitations']}
    lines = ['PS03 full diagnostic interpretation', '', *findings, '', 'All-source TPG + structured: seed-level findings',
             'Seed | Intervention | Affected CVEs | Original MAE | Changed-input MAE | MAE change | Mean absolute prediction change']
    lines += [f"{r['seed']} | {r['intervention']} | {r['affected_cves']} | {r['baseline_mae']:.9f} | {r['intervention_mae']:.9f} | {r['mae_change']:+.9f} | {r['mean_absolute_prediction_change']:.9f}"
              for r in rows if r['configuration'] == 'P01_DGNM_security_tpg_S1']
    lines += ['', 'Selected evidence inventory (not population prevalence)', json.dumps(inventory, indent=2),
              '', 'Real text examples from the saved evidence', json.dumps(examples, indent=2, ensure_ascii=False),
              '', 'Limitations', *summary['limitations']]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'analysis.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    (OUTPUT / 'analysis.txt').write_text('\n'.join(lines) + '\n')
    files = {name: sha(OUTPUT / name) for name in ('analysis.json', 'analysis.txt')}
    (OUTPUT / 'review_lock.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
    require(sha(SOURCE / 'diagnostic_lock.json') == report['source_lock_sha256'], 'Source lock changed')
    print(f'PS03 interpretation saved to {OUTPUT}. No inference, training or test scoring.')


if __name__ == '__main__':
    run()
