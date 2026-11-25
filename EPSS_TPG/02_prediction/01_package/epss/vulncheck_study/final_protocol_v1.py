"""VK09 complete development reporting and immutable final-evaluation specification."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import threshold_selection_v1 as threshold

data, dev, training = threshold.data, threshold.dev, threshold.training
CONFIG = data.ROOT/'02_prediction/04_configs/02_vulncheck/07_final_evaluation_v1.json'


def choose_family(metrics, seeds):
    families = sorted({n.rsplit('__seed_', 1)[0] for n in metrics})
    averages = {}
    for family in families:
        names = [family+'__seed_'+str(s) for s in seeds]
        data.require(all(n in metrics for n in names), 'Incomplete seed family')
        data.require(sum(n.startswith(family+'__seed_') for n in metrics) == len(seeds), 'Unexpected seed')
        values = [metrics[n]['average_precision'] for n in names]
        data.require(all(v is not None and np.isfinite(v) for v in values), 'Undefined selection AP')
        averages[family] = float(np.mean(values))
    winner = min(families, key=lambda n: (-averages[n], n))
    return winner, averages


def validate_cohort(manifest, development, expected):
    data.require(dict(Counter(v['split'] for v in manifest.values())) == expected, 'Frozen role counts changed')
    roles = {role: sorted(c for c, r in manifest.items() if r['split'] == role) for role in expected}
    data.require(all(c == r['cve'] for c, r in manifest.items()), 'Manifest CVE identity mismatch')
    for role in ('train', 'validation'):
        data.require(roles[role] == sorted(development[role]), 'Development roles changed')
    data.require(not set(roles['test']) & (set(roles['train']) | set(roles['validation'])), 'Evaluation overlaps development CVE IDs')
    return roles


def validate_inventory(items):
    data.require(len(items) == 36 and len({r['model'] for r in items}) == 36, 'Expected 36 unique predictors')
    data.require(Counter(r['kind'] for r in items) == {'source_neural': 24, 'edge_free': 3, 'logistic': 8, 'constant': 1}, 'Wrong model inventory')
    for item in items:
        data.require(item['fixed_threshold'] == .5, 'Fixed reporting threshold changed')
        if item['kind'] != 'constant':
            data.require(item['checkpoint'] and len(item['checkpoint_sha256']) == 64, 'Missing checkpoint identity')
        if item['kind'] == 'source_neural':
            data.require(item['selected_threshold'] is not None and 0 <= item['selected_threshold'] <= 1, 'Missing selected cutoff')
        else:
            data.require(item['selected_threshold'] is None, 'Unexpected new tuning of reference controls')


def gather():
    cfg = data.read(CONFIG)
    data.require(cfg['schema'] == 'vk09_final_protocol_v1' and cfg['execution']['freeze_only']
                 and not cfg['execution']['test_scoring_enabled'] and not cfg['execution']['training_enabled'], 'Freeze must not train or score')
    source, export, identity, ids, y, scores, pins = threshold.load_predictions('full')
    roots = {
        'logistic': data.ROOT/dev.config()['result_root'],
        'edge_free': data.ROOT/'05_results/05_vulncheck_study/11_edge_free_control_v1/full',
        'thresholds': threshold.ROOT/'full',
        'source_review': data.ROOT/'05_results/05_vulncheck_study/17_threshold_source_review_v1',
    }
    for root in tqdm(roots.values(), desc='Verify frozen reports and reference checkpoints', unit='artifact tree'):
        data.check_lock(root)
        record = data.read(root/'run_identity.json')
        if 'pins' in record:
            data.check_pins(record['pins'])
        pins[str(root/'artifact_lock.json')] = data.sha(root/'artifact_lock.json')
    manifest_path = data.ROOT/cfg['cohort']['manifest']
    manifest = data.read(manifest_path)
    roles = validate_cohort(manifest, identity['ids'], cfg['cohort']['counts'])
    state_path = data.ROOT/dev.config()['data_root']/'structured_encoder.json'
    state = data.read(state_path)
    data.require(sorted(state['fit_ids']) == roles['train'] and len(state['feature_names']) == 29, 'Structured fit changed')
    metrics = data.read(source/'metrics.json')
    winner, family_ap = choose_family(metrics, cfg['selection']['seeds'])
    data.require(winner == cfg['selection']['expected_primary'] and len(family_ap) == 8, 'Selection differs from frozen decision')
    cutoffs = data.read(roots['thresholds']/'thresholds.json')
    references = data.read(source/'description_references.json')
    inventory, report_rows = [], []
    for name in tqdm(sorted(metrics), desc='Bind saved neural checkpoints and thresholds', unit='model'):
        checkpoint = (Path(references[name]['path']) if name.startswith('D__') else source/name)/'checkpoint.pt'
        digest = data.sha(checkpoint)
        data.require(digest == cutoffs[name]['checkpoint_sha256'], 'Threshold bound to different checkpoint')
        data.require(dev.metric(y, scores[name]) == metrics[name], 'Validation metrics changed')
        inventory.append({'model': name, 'kind': 'source_neural', 'family': name.rsplit('__seed_', 1)[0],
                          'seed': int(name.rsplit('__seed_', 1)[1]), 'checkpoint': str(checkpoint), 'checkpoint_sha256': digest,
                          'fixed_threshold': .5, 'selected_threshold': cutoffs[name]['threshold'],
                          'best_epoch': data.read(checkpoint.parent/'summary.json')['best_epoch'], 'constant_score': None})
        report_rows.append({'model': name, 'kind': 'source_neural', **{k: metrics[name][k] for k in ('average_precision', 'roc_auc', 'f1', 'precision', 'recall', 'brier', 'log_loss')},
                            'selected_validation_f1': cutoffs[name]['selected_on_same_validation']['f1'],
                            'threshold_crossfold_f1': cutoffs[name]['crossfold']['metrics']['f1']})
    for kind in ('edge_free', 'logistic'):
        root = roots[kind]
        values = data.read(root/'metrics.json')
        by_model = defaultdict(list)
        for row in data.read(root/'predictions.json'):
            key = row['model']+'__seed_'+str(row['seed']) if kind == 'edge_free' else row['model']
            by_model[key].append(row)
        data.require(set(by_model) == set(values), 'Reference model list differs')
        for name, metric in values.items():
            rows = by_model[name]
            data.require([r['cve'] for r in rows] == ids and all(r['split'] == 'validation' and r['actual'] == int(y[i]) for i, r in enumerate(rows)), 'Reference cohort differs')
            data.require(dev.metric(y, [r['predicted'] for r in rows]) == metric, 'Reference metrics differ')
            constant = name == 'training_prevalence'
            checkpoint = root/name/'checkpoint.pt' if kind == 'edge_free' else root/(name+'.joblib')
            inventory.append({'model': name, 'kind': 'constant' if constant else kind,
                              'family': name.rsplit('__seed_', 1)[0] if kind == 'edge_free' else name,
                              'seed': int(name.rsplit('__seed_', 1)[1]) if kind == 'edge_free' else (None if constant else 17),
                              'checkpoint': None if constant else str(checkpoint),
                              'checkpoint_sha256': None if constant else data.sha(checkpoint),
                              'fixed_threshold': .5, 'selected_threshold': None,
                              'best_epoch': data.read(checkpoint.parent/'summary.json')['best_epoch'] if kind == 'edge_free' else None,
                              'constant_score': 376/2870 if constant else None})
            if constant:
                data.require(all(r['predicted'] == 376/2870 for r in rows), 'Prevalence reference changed')
            report_rows.append({'model': name, 'kind': 'constant' if constant else kind,
                                **{k: metric[k] for k in ('average_precision', 'roc_auc', 'f1', 'precision', 'recall', 'brier', 'log_loss')},
                                'selected_validation_f1': None, 'threshold_crossfold_f1': None})
    validate_inventory(inventory)
    for item in inventory:
        if item['checkpoint']:
            pins[item['checkpoint']] = item['checkpoint_sha256']
    for path in (CONFIG, Path(__file__), manifest_path, state_path,
                 data.SCRIPTS/'16_freeze_final_protocol.py', data.SCRIPTS/'test_vulncheck_final_protocol_v1.py'):
        pins[str(path)] = data.sha(path)
    review = data.read(roots['source_review']/'analysis.json')
    protocol = {'schema': cfg['schema'], 'configuration': cfg, 'primary_family': winner,
                'family_selection_validation_AP': family_ap, 'checkpoint_inventory': inventory,
                'source_training_identity_digest': data.digest(identity),
                'source_dataset': data.config()['source'], 'source_dataset_sha256': data.config()['source_sha256'],
                'training_settings': identity['settings'], 'class_weights': identity['class_weights'],
                'source_policy': data.config()['source_policy'], 'source_fields': data.config()['source_fields'],
                'structured_feature_names': state['feature_names'],
                'evaluation_ids_sha256': data.digest(roles['test']), 'development_complete': True,
                'final_evaluation_performed': False, 'independent_final_test_available': False}
    return cfg, protocol, roles, report_rows, review, pins


def render_report(protocol, roles, rows, review):
    lines = ['# VulnCheck Snapshot Classification: Completed Development Report', '',
        '## Status and Research Question', '',
        'Development is complete and the final evaluation protocol is frozen. Final evaluation has NOT been performed.',
        'The question is whether CVE descriptions, URL summaries, related posts and structured features help identify CVEs listed in the saved VulnCheck catalogue, and whether typed TPG graphs improve on pooled text.',
        'This is retrospective binary classification, not EPSS regression, CISA KEV classification, future exploitation prediction or causal analysis of social discussion.', '',
        '## Data and Roles', '',
        'The frozen snapshot has 5,968 distinct CVE IDs. Existing role assignments are preserved; shared wording is not treated as proof of duplicate vulnerabilities.',
        '| Role | CVEs | Use |', '|---|---:|---|',
        '| Training | 2,870 | Fit model parameters and structured/TF-IDF transforms; 376 listed |',
        '| Validation | 1,104 | Select checkpoints, compare sources and select thresholds; 74 listed |',
        '| Previously reserved evaluation | 651 | Not scored in this study yet; previously exposed during EPSS research |',
        '| Excluded | 1,343 | Remain excluded; no expansion or reassignment |', '',
        'The exact role manifest is the authority. Its date boundaries originate in the prior EPSS task; they do not establish historical availability of the current text or structured features for this snapshot task.',
        'A zero label means not listed in this snapshot, not never exploited. Acquisition completeness, historical label availability and negative-label interpretation remain limitations.', '',
        '## Architecture and Predictors', '',
        'Description, GitHub summaries, non-GitHub summaries and posts are kept separate. Accepted documents become bounded text chunks; the frozen security-v4 extractor builds typed TPG nodes and candidate relations, using frozen SecBERT node embeddings. The GNN is not a newly trained relation extractor.',
        'The security encoder projects node features to 32 dimensions and applies two EdgeTypeGNNLayer layers. Messages use sender/receiver node vectors plus learned relation-type embeddings and edge features, with sum aggregation, residual updates and layer normalization. This is the saved custom typed-message GNN, not a claim that RGAT or a new architecture was trained here.',
        'The pooled control projects and averages embedded text tokens without graph message passing. Its active encoder capacity and node population differ from the GNN. Chunks are averaged within documents and documents within each source. Masked source attention combines available sources; the encoded structured vector is concatenated before the binary classifier head.',
        'Structured inputs are current CVSS vector categories, CVSS version, normalized base score, observation masks and a training-standardized log reference count: 29 columns. CVSS categories use one-hot indicators. Relation types enter the GNN, not the tabular vector. No EPSS score or catalogue-derived field is a predictor.',
        'Source disclosure filters are engineering-reviewed heuristics, not independent semantic validation. The current descriptions, summaries, posts, CVSS and reference counts are retrospective. SMP includes related scraped text as well as posts; it is not uniformly a verified dated social message.', '',
        '## Completed Comparisons', '',
        'VK03 fitted eight logistic controls plus a constant prevalence reference. VK04 compared pooled description and description TPG with structured features. VK05 diagnosed those frozen models. VK06 trained three matched edge-free description controls. VK07 added description+summaries, description+posts and all-source variants. VK08 selected thresholds and diagnosed frozen all-source models.',
        'There are 36 retained predictors: 24 source/representation fits, three edge-free fits, eight logistic fits and one constant. All neural source variants retain structured inputs. Therefore no claim is made that a complete structured-off or independently trained GitHub-only/non-GitHub-only ablation was performed.',
        'Neural training used seeds 17, 29 and 43, balanced BCE with training-only class weights, AdamW, learning rate 0.001, weight decay 0.0001, at most 100 epochs and patience 15. Checkpoints maximize validation AP; earliest epochs win ties. These settings and all checkpoint files are frozen, not refitted on training plus validation.', '',
        '## Full Validation Results', '',
        'Each row is an individual saved predictor. AP is average precision, not trapezoidal PR-AUC. Higher AP, ROC-AUC and F1 are better. Lower Brier/log loss are better. Selected F1 uses the same labels that selected its cutoff and is optimistic; cross-fold checks only threshold selection, not model training.',
        '| Predictor | AP | ROC-AUC | F1 at 0.5 | Selected-validation F1 | Cross-fold threshold F1 | Brier | Log loss |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(value):
        return 'Not selected' if value is None else f'{value:.4f}'
    for row in rows:
        lines.append('| '+row['model']+' | '+' | '.join(fmt(row[k]) for k in ('average_precision', 'roc_auc', 'f1', 'selected_validation_f1', 'threshold_crossfold_f1', 'brier', 'log_loss'))+' |')
    lines += ['', 'D=description; G=GitHub summary; N=non-GitHub summary; M=posts/related text. All D/DGN/DM/DGNM neural models include the same structured branch. Controls without VK08 threshold searches have no selected F1; this is not a zero.', '',
        '## Findings', '',
        '- All-source TPG mean AP is 0.4149 versus 0.3781 for all-source pooled text and 0.2992 for description TPG. The graph/pooled AP difference favors graphs in two of three seeds, not every seed. No significance claim follows from three seed means.',
        '- All-source TPG mean F1 is 45.85% at 0.5, 48.23% at thresholds selected on the same validation data and 45.76% in threshold-only cross-fold checks. The highest individual selected F1 is 49.67%, but the final protocol does not promote that seed as the winner.',
        '- URL summaries alone reduce mean AP relative to description. With posts already included, adding summaries improves TPG AP in all three seeds. Individual summary groups were not separately retrained.',
        '- The description edge-free control has mean AP 0.2873 versus 0.2992 with edges. AP favors edges in only one seed; ROC-AUC favors edges in all three. This is not uniform predictive superiority of graph relations.',
        '- On 1,018 validation CVEs without recorded wording shared with training, all-source TPG AP is 0.3182 versus 0.2713 for description TPG. The posts-only TPG gain disappears there. These are descriptive subsets with different class proportions, not repaired independent splits.',
        '- Frozen all-source TPG AP falls to 0.1213 after exchanging post vectors, to 0.2983 after exchanging descriptions, and to 0.3206 after zeroing structured vectors. Swapping summaries gives 0.3962. These out-of-distribution interventions show sensitivity, not causality or isolated security-edge value.',
        '- Some accepted summaries describe a different product/CVE or contain unresolved placeholders. CVE-2025-21253 has an Edge-spoofing description but a Print-Spooler summary naming CVE-2024-12345; CVE-2025-26630 has an Access description but a Print-Spooler summary containing CVE-2023-XXXXX. Removing summaries does not correct the highlighted false flags.',
        '- Attention favors posts, but attention is not causal importance. Reported 80.02% mean post attention concerns only text fusion and excludes the structured branch.', '',
        '## Final Evaluation Is Frozen, Not Executed', '',
        'The primary family is DGNM__security_tpg, selected by the highest mean validation AP across the eight source/representation families. The matched comparator is DGNM__pooled_text. Seeds 17, 29 and 43 are all retained. No seed ensemble is constructed.',
        'The planned cohort is the exact 651 previously reserved CVE IDs. Because this cohort was exposed in the earlier EPSS study, any later result must be labelled reused-cohort retrospective evaluation, not a fresh independent or preregistered test. This protocol is frozen after development, not before the study began.',
        'Every completed predictor remains reportable. Report fixed-0.5 metrics for all 36 and frozen VK08 thresholds for the 24 source-neural fits only. The primary comparison is AP; tuned neural F1 must not be used to claim superiority over fixed-only logistic or edge-free controls.',
        'No threshold search, probability calibration, checkpoint or seed replacement, source repair, cohort filtering, vocabulary/scaler fitting or training is allowed during evaluation. Existing development adapters remain test-blocked. A separate tested runner and explicit authorization are required before final scoring.',
        'The frozen JSON specifies exact checkpoint hashes, selected thresholds, CVE IDs, feature state, bootstrap rules, required metrics and figures, failure handling and provenance. Missing labels or unprocessable required graphs stop the complete evaluation rather than silently shrink its cohort.', '',
        '## Limitations and Conclusions', '',
        'The defensible result is a development improvement for the retrospective all-source system, with a smaller seed-dependent graph advantage. Neither threshold tuning nor source interventions establish independent generalization. Shared text, repeated validation use, source relevance, noisy catalogue negatives and previous test exposure remain material.',
        'This study does not demonstrate that TPG replaces tokenization, that it is universally superior to knowledge graphs, or that it is a state-of-the-art future exploitation predictor. Typed graphs, source separation, provenance and reproducible controls are engineering contributions; predictive claims must stay within the measured task.',
        'Do not run another broad training sweep for this study. Preserve and report negative controls and all seeds. Corrected-source experiments or new external/future snapshots require separate versioned protocols and cannot be silently substituted into this completed comparison.', '',
        '## Artifact Guide', '',
        '- final_protocol.json: complete frozen decisions and checkpoint inventory.',
        '- cohort_ids.json: exact existing role IDs; no new evaluation labels or text.',
        '- model_inventory.csv and validation_metrics.csv: reproducible reporting tables.',
        '- final_report.md, final_report.txt and final_report.json: completed development report and numerical evidence.',
        '- run_identity.json and artifact_lock.json: dependency hashes and output lock.',
        '- Final evaluation metrics are deliberately absent. No test scores are fabricated.']
    return '\n'.join(lines)+'\n'


def run(verify=False, check_only=False):
    cfg, protocol, roles, rows, review, pins = gather()
    output = data.ROOT/cfg['result_root']
    print(f'[VK09] Primary={protocol["primary_family"]}; 36 frozen predictors; 651 reused-cohort IDs; no evaluation scoring.', flush=True)
    if check_only:
        print('VK09 final protocol preflight passed. No output writes, training or test scoring.')
        return
    report = render_report(protocol, roles, rows, review)
    identity = {'schema': cfg['schema'], 'pins': pins, 'protocol_sha256': data.digest(protocol),
                'training': False, 'test_scoring': False, 'new_inference': False}
    artifacts = {'final_protocol.json': protocol, 'cohort_ids.json': roles,
                 'model_inventory.csv': dev.csv_text(protocol['checkpoint_inventory']),
                 'validation_metrics.csv': dev.csv_text(rows), 'final_report.md': report, 'final_report.txt': report,
                 'final_report.json': {'status': cfg['status'], 'protocol_sha256': data.digest(protocol),
                                       'all_predictor_metrics': rows, 'source_and_threshold_review': review,
                                       'evaluation_performed': False},
                 'run_identity.json': identity}
    data.check_pins(pins)
    threshold.save_report(output, artifacts, verify)
    print(f'VK09 final reporting and protocol {"reproduced" if verify else "frozen at "+str(output)}. No training, inference or test scoring.')
