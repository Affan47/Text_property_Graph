"""Freeze date/text-only inner roles, then describe their training-label support."""
from collections import Counter
from datetime import date
from pathlib import Path
import platform
import numpy as np
from threadpoolctl import threadpool_info
from tqdm import tqdm

from ..combined_training_v1 import data as base, reporting
from ..continuous_forecasting import data as publication
from . import core

c, require, ROOT = base.c, core.require, base.ROOT
CONFIG = ROOT/'02_prediction/04_configs/07_fresh_studies/04_training_fold_audit_v1.json'
CLI = ROOT/'02_prediction/03_analysis/07_fresh_studies/08_audit_training_folds.py'
OUTPUT = ROOT/'05_results/08_fresh_studies/10_training_fold_audit_v1'
NAMES = {'previous': '02_previous_multisource', 'reddit': '01_reddit'}


def code_pins():
    files = [CONFIG, CLI, Path(base.__file__), Path(reporting.__file__), Path(publication.__file__),
             *Path(__file__).parent.glob('*.py')]
    return {str(p): c.sha(p) for p in sorted(files)}


def runtime():
    import os
    pools = threadpool_info()
    require(all(p['num_threads'] == 4 for p in pools if p['user_api'] == 'blas'), 'BLAS threads not pinned')
    return {'python': platform.python_version(), 'numpy': np.__version__, 'threadpools': pools,
            'environment': {name: os.environ.get(name) for name in
                            ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS')}}


def label_map(path):
    raw = c.read(path)
    if isinstance(raw, list):
        require(len(raw) == len({r['cve'] for r in raw}), 'Duplicate label IDs')
        return {r['cve']: r for r in raw}
    require(isinstance(raw, dict), 'Unexpected label container')
    return raw


def metadata(dataset):
    directory = base.cache.directory(dataset, 'full')
    saved = base.cache.load_plan(directory)
    require(not saved['test_adapted'] and saved['dataset'] == dataset, 'Wrong development graph plan')
    export = Path(saved['structured_vectors_path']).parent
    publication.check_lock(export)
    identity = c.read(export/'run_identity.json')
    require(not identity['test_adapted'] and set(identity['ids']) == {'train', 'validation'}, 'Unexpected outer roles')
    ids = sorted(identity['ids']['train'])
    require(len(ids) == len(set(ids)) and not set(ids) & set(identity['ids']['validation']), 'Outer role overlap')
    require(ids == sorted(cid for cid, r in saved['index'].items() if r['split'] == 'train'), 'Training IDs changed')
    labels = label_map(saved['labels_path_not_loaded'])
    require(set(labels) == set(saved['index']), 'Development labels and index differ')
    # Shared files contain development rows. Only training payloads are projected;
    # no validation value, document, or structured feature is consumed below.
    rows = {}
    for cid in tqdm(ids, desc=f'{dataset}: read training-only dates and evidence', unit='CVE'):
        label = labels[cid]
        require(label.get('split', 'train') == 'train' and label['target_day'] == 60 and
                label.get('observed', True) is True, 'Wrong/unobserved target metadata')
        anchor = label['target_date']; require(date.fromisoformat(anchor).isoformat() == anchor, 'Bad target date')
        views = saved['index'][cid]['views']; item = views['DMU']; key = item['object_key']
        require(key is not None, 'Missing accepted description')
        entry = saved['entries'][key]
        require(entry['cve'] == cid and entry['split'] == 'train', 'Wrong document owner')
        document_hashes, counts = [], Counter()
        for span in entry['spans']:
            source = span['source']; require(source in 'DMGN', 'Unrecognized source')
            text = entry['text'][span['start']:span['end']]
            import hashlib
            require(hashlib.sha256(text.encode()).hexdigest() == span['text_sha256'], 'Document hash/span mismatch')
            document_hashes.append(core.fingerprint(text)); counts[source] += 1
        mask = {s: counts[s] > 0 for s in 'DMGN'}
        require(mask == item['source_mask'], 'Source mask differs from document provenance')
        d = saved['entries'][views['D']['object_key']]
        require(d['cve'] == cid and d['split'] == 'train' and {s['source'] for s in d['spans']} == {'D'},
                'D cache is not standalone-description context')
        dm = saved['entries'][views['DM']['object_key']]
        require(dm['cve'] == cid and dm['split'] == 'train' and {s['source'] for s in dm['spans']} <= {'D', 'M'},
                'DM cache contains other sources')
        rows[cid] = {'outer_role': 'train', 'anchor_date': anchor, 'source_mask': mask,
                     'document_counts': {s: counts[s] for s in 'DMGN'},
                     'document_fingerprints': sorted(set(document_hashes)),
                     'D_key': views['D']['object_key'], 'DM_key': views['DM']['object_key']}
    paths = [directory/'plan_lock.json', directory/'plan.json.gz', export/'artifact_lock.json',
             export/'run_identity.json', Path(saved['labels_path_not_loaded'])]
    pins = {str(p): c.sha(p) for p in paths}
    return rows, {'dataset': dataset, 'training_ids_sha256': c.digest(ids),
                  'outer_validation_ids_sha256': c.digest(sorted(identity['ids']['validation'])),
                  'input_hashes': pins, 'source_pins': code_pins(), 'runtime': runtime(),
                  'label_source': saved['labels_path_not_loaded'], 'training': False,
                  'validation_features_or_scores_used': False, 'test_access': False,
                  'shared_file_note': 'Development JSON containers decoded for integrity; only train payloads used. No held-out raw sources opened.'}


def plan(dataset, verify):
    config = c.read(CONFIG)
    require(config['schema'] == 'ct04_training_fold_feasibility_v1' and not config['training_enabled']
            and not config['test_enabled'] and not config['outer_validation_used'], 'Invalid scope')
    rows, identity = metadata(dataset)
    result = core.make_folds(rows, config)
    manifest = {'schema': config['schema'], 'config': config, 'identity': identity, 'rows': rows, **result}
    out = OUTPUT/NAMES[dataset]/'plan'
    with publication.transaction(out, verify=verify) as target:
        base.write(target/'fold_manifest.json', manifest)
        report = ['Training-only inner-fold plan. No outcome values used to choose cut points.',
                  f'Dataset: {dataset}; training CVEs: {len(rows)}',
                  f'Score-date cut points: {result["cut_dates"]}',
                  f'Assessment-covered CVEs: {len(result["oof_covered_ids"])}; warmup only: {len(result["warmup_only_ids"])}',
                  'Anchor = date of the observed Day-60 score being estimated, NOT CVE ID year.',
                  'All fit targets strictly predate each inner assessment block.',
                  'Grouped exclusions apply only inside this plan; official roles remain unchanged.',
                  'Identical text links evidence; it does not establish duplicate vulnerabilities.',
                  'Grouping is exact normalized text only, not a complete near-duplicate or vulnerability-family audit.',
                  'Current text/CVSS remain retrospective. Calendar ordering does not establish historical input availability.',
                  'Support thresholds are review gates, not power calculations. No cuts will be adjusted after reading labels.']
        for fold in result['folds']:
            report.append(f"Fold {fold['fold']} / {fold['policy']}: fit={len(fold['fit'])}, "
                          f"assessment={len(fold['inner_assessment'])}, purged={len(fold['purged_shared_evidence'])}")
        (target/'REPORT.txt').write_text('\n'.join(report)+'\n')
        base.write(target/'shared_document_components.json', {'groups': result['components'], 'shared': result['shared_documents']})
        publication.check_pins(identity['input_hashes']); publication.check_pins(identity['source_pins'])
    print(f'CT04 {dataset} outcome-blind fold plan {"reproduced" if verify else "saved"}: {out}; no training.')


def support(ids, rows, labels, config):
    y = np.array([labels[c] for c in ids], dtype=float)
    return {'cves': len(ids), 'high_epss': int((y >= config['high_score_cutoff']).sum()),
            'lower_epss': int((y < config['high_score_cutoff']).sum()),
            'very_high_epss': int((y >= config['very_high_score_cutoff']).sum()),
            'median_epss': float(np.median(y)) if len(y) else None,
            'mean_epss': float(y.mean()) if len(y) else None,
            'min_anchor': min((rows[c]['anchor_date'] for c in ids), default=None),
            'max_anchor': max((rows[c]['anchor_date'] for c in ids), default=None),
            **{f'{s}_present': sum(rows[c]['source_mask'][s] for c in ids) for s in 'DMGN'},
            'U_present': sum(rows[c]['source_mask']['G'] or rows[c]['source_mask']['N'] for c in ids),
            'high_and_M_present': sum(labels[c] >= config['high_score_cutoff'] and rows[c]['source_mask']['M'] for c in ids)}


def audit(dataset, verify):
    folder = OUTPUT/NAMES[dataset]; plan_folder = folder/'plan'
    publication.check_lock(plan_folder)
    manifest = c.read(plan_folder/'fold_manifest.json'); identity = manifest['identity']; config = manifest['config']
    require(config == c.read(CONFIG) and identity['source_pins'] == code_pins() and identity['runtime'] == runtime(),
            'Changed plan/code/runtime; version the audit rather than rewriting a lock')
    publication.check_pins(identity['input_hashes'])
    rows = manifest['rows']; raw = label_map(identity['label_source']); labels = {}
    for cid in rows:
        r = raw[cid]; value = r.get('value', r.get('target'))
        require(type(value) in (int, float) and np.isfinite(value) and 0 <= value <= 1
                and r['target_day'] == 60 and r.get('observed', True) is True
                and r['target_date'] == rows[cid]['anchor_date'], 'Missing/invalid training target: '+cid)
        labels[cid] = float(value)
    table, gates, membership = [], [], []
    for fold in tqdm(manifest['folds'], desc=f'{dataset}: quantify locked fold support', unit='fold'):
        summaries = {}
        for role in ('fit', 'inner_assessment', 'purged_shared_evidence', 'future_unseen'):
            stats = support(fold[role], rows, labels, config); summaries[role] = stats
            table.append({'fold': fold['fold'], 'policy': fold['policy'], 'role': role, **stats})
            membership.extend({'fold': fold['fold'], 'policy': fold['policy'], 'cve': cid, 'inner_role': role,
                               'anchor_date': rows[cid]['anchor_date'], 'evidence_component': manifest['component_by_cve'][cid]}
                              for cid in fold[role])
        fit, assessment = summaries['fit'], summaries['inner_assessment']
        regression = fit['cves'] >= config['minimum_fit_cves'] and assessment['cves'] >= config['minimum_assessment_cves']
        tail = all(r[k] >= config['minimum_high_and_lower_per_role'] for r in (fit, assessment) for k in ('high_epss', 'lower_epss'))
        posts = all(r['M_present'] >= config['minimum_M_per_role'] for r in (fit, assessment))
        summaries_ok = all(r['U_present'] >= config['minimum_U_per_role'] for r in (fit, assessment))
        gates.append({'fold': fold['fold'], 'policy': fold['policy'], 'regression_count_gate': regression,
                      'high_score_support_gate': tail, 'M_support_gate': posts, 'U_support_gate': summaries_ok,
                      'overlap_components': len(fold['cross_role_components']),
                      'status': 'COUNT_GATES_PASS_NOT_TRAINING_APPROVAL' if regression and tail and posts else 'SUPPORT_REVIEW_REQUIRED'})
    covered = manifest['oof_covered_ids']; warmup = manifest['warmup_only_ids']
    oof = {name: support(ids, rows, labels, config) for name, ids in
           (('all_outer_training', sorted(rows)), ('potential_oof_assessment', covered), ('warmup_only', warmup))}
    for key in ('cves', 'high_epss', 'very_high_epss', 'M_present'):
        require(oof['potential_oof_assessment'][key]+oof['warmup_only'][key] == oof['all_outer_training'][key], 'OOF coverage mismatch')
    source_support = []
    for fold in manifest['folds']:
        for role in ('fit', 'inner_assessment'):
            for stratum, ids in [('M_present', [c for c in fold[role] if rows[c]['source_mask']['M']]),
                                 ('M_absent', [c for c in fold[role] if not rows[c]['source_mask']['M']])]:
                source_support.append({'fold': fold['fold'], 'policy': fold['policy'], 'role': role,
                                       'stratum': stratum, **support(ids, rows, labels, config)})
    finding = {'schema': 'ct04_training_fold_support_v1', 'dataset': dataset,
               'plan_sha256': c.sha(plan_folder/'fold_manifest.json'), 'plan_lock_sha256': c.sha(plan_folder/'artifact_lock.json'),
               'identity': identity, 'gates': gates, 'oof_coverage': oof, 'training': False, 'test_access': False,
               'decision': 'No training unlocked. Retain all fold counts, including failed support gates; review before defining a fitting protocol.',
               'limitations': ['Forward chaining cannot assess the initial warmup without future-data training.',
                   'Grouped policy conservatively purges fitting evidence overlap, not duplicate CVE identities.',
                   'Exact normalized documents do not detect every paraphrase, shared vulnerability family or historical dependence.',
                   'Inner folds reuse previously trained-on records; this is not fresh independent test evidence.',
                   'Refit structured preprocessing inside fitting folds. Existing outer-training vectors must not be reused as fold-fitted inputs.',
                   'Standalone D embeddings must come from D graphs; D tokens from DM context are not a D-only control.',
                   'Choose epochs/hyperparameters without inner-assessment outcomes; use an additional fitting-only chronological selection step.',
                   'Pooled OOF metrics supplement per-fold metrics, not replace them. Report exact covered CVEs/high-score counts.',
                   'Future intervals need at least 10,000 resamples; this specification fixes 20,000. More resamples reduce numerical error, not selection bias or data limitations.',
                   'Historical wording/CVSS availability is not repaired by chronological splitting.']}
    report = ['CT04: training-only fold support', f'Dataset: {dataset}',
              'Original validation and test roles unchanged. No model fit or predictions produced.',
              'fit = earlier training CVEs; inner_assessment = later CVEs within the original training role.',
              'high_epss means observed Day-60 EPSS >= 0.1; M means accepted prose, not all raw posts.',
              'grouped excludes fitting members linked to assessment through accepted text; URLs alone never link records.',
              'fold | policy | role | CVEs | high EPSS | M present | high EPSS with M | U present']
    for r in table:
        report.append(f"{r['fold']} | {r['policy']} | {r['role']} | {r['cves']} | {r['high_epss']} | "
                      f"{r['M_present']} | {r['high_and_M_present']} | {r['U_present']}")
    report += ['', 'Potential out-of-fold coverage (NO predictions yet):', publication.dump(oof),
               'Operational support gates (not power guarantees):', publication.dump(gates),
               'Limitations and next-step constraints:', *['- '+s for s in finding['limitations']], finding['decision']]
    with publication.transaction(folder/'audit', verify=verify) as target:
        base.write(target/'findings.json', finding)
        (target/'REPORT.txt').write_text('\n'.join(report)+'\n')
        for name, records in [('fold_support.csv', table), ('source_strata.csv', source_support),
                              ('fold_membership.csv', membership), ('support_gates.csv', gates)]:
            (target/name).write_text(reporting.csv_text(records))
        make_figure(target, table, dataset)
        publication.check_pins(identity['source_pins']); publication.check_pins(identity['input_hashes'])
    print(f'CT04 {dataset} support audit {"reproduced" if verify else "saved"}: {folder/"audit"}; no training or test scoring.')


def make_figure(target, table, dataset):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    subset = [r for r in table if r['policy'] == 'evidence_grouped' and r['role'] in ('fit', 'inner_assessment')]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, key, title in zip(axes, ('cves', 'high_epss', 'M_present'),
                              ('All CVEs', 'CVEs with Day-60 EPSS >= 0.1', 'CVEs with accepted post prose')):
        for role, offset, color in [('fit', -.18, '#287f8e'), ('inner_assessment', .18, '#b34c60')]:
            rs = [r for r in subset if r['role'] == role]
            bars = ax.bar(np.array([r['fold'] for r in rs])+offset, [r[key] for r in rs], .36,
                          label=role.replace('_', ' '), color=color)
            ax.bar_label(bars, fontsize=8, padding=2)
        ax.set_title(title, fontsize=10); ax.set_xlabel('Inner chronological fold'); ax.set_ylabel('Number of CVEs')
        ax.set_xticks([1, 2, 3]); ax.margins(y=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'{dataset}: grouped training-only fold support; no model results')
    fig.tight_layout(); fig.savefig(target/'fold_support.png', dpi=140); plt.close(fig)


def run(dataset, stage, verify=False):
    require(dataset in NAMES and stage in ('plan', 'audit'), 'Unsupported audit stage')
    (plan if stage == 'plan' else audit)(dataset, verify)
