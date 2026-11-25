"""Publish immutable CT06 plans before inspecting their outcome support."""
from pathlib import Path

from tqdm import tqdm
from ..training_only_pooled_v1 import data
from ..training_fold_audit_v1 import core as prior_core
from . import core

ROOT = data.ROOT
CONFIG = ROOT / '02_prediction/04_configs/07_fresh_studies/07_within_version_audit_v1.json'
CLI = ROOT / '02_prediction/03_analysis/07_fresh_studies/14_audit_within_version.py'
OUTPUT = ROOT / '05_results/08_fresh_studies/12_within_version_audit_v1'


def settings():
    value = data.c.read(CONFIG)
    core.require(value['schema'] == 'ct06_within_version_preflight_v1'
                 and not any(value[k] for k in ('training_enabled', 'outer_validation_used', 'test_enabled'))
                 and value['future_comparison']['arms'] == ['D', 'DF', 'DM']
                 and value['future_comparison']['seeds'] == [17, 29, 43], 'Wrong CT06 scope')
    return value


def pins():
    paths = [CONFIG, CLI, Path(prior_core.__file__), *Path(__file__).parent.glob('*.py')]
    return {**data.pins(), **{str(p): data.c.sha(p) for p in sorted(paths)}}


def location(dataset, config):
    return OUTPUT / data.NAMES[dataset] / config['datasets'][dataset]['version']


def parent(dataset):
    saved = data.load_protocol(dataset)
    core.require(not saved['outer_validation_access'] and not saved['test_access'], 'Unexpected CT05 scope')
    mapping, _, _ = prior_core.components(saved['rows'])
    core.require(mapping == saved['component_by_cve'], 'CT05 full-training evidence mapping does not reproduce')
    folder = data.OUT / data.NAMES[dataset] / 'protocol'
    identity = {'input_hashes': {**saved['parents'],
                                **{str(folder / name): data.c.sha(folder / name)
                                   for name in ('protocol.json', 'artifact_lock.json')}},
                'source_pins': pins(), 'runtime': data.runtime(),
                'original_training_ids_sha256': data.c.digest(sorted(saved['rows'])),
                'label_source_not_used_for_plan': saved['label_source'],
                'training': False, 'outer_validation_payloads_used': False, 'test_access': False}
    return saved, identity


def membership(manifest):
    rows, comp = manifest['rows'], manifest['component_by_cve']
    return [{'fold': f['fold'], 'role': role, 'cve': c, 'target_date': rows[c]['anchor_date'],
             'evidence_component': comp[c], 'M_present': rows[c]['source_mask']['M']}
            for f in manifest['folds'] for role in
            ('optimization', 'selection', 'assessment', 'purged_to_assessment', 'purged_to_selection', 'future')
            for c in f[role]]


def plan(dataset, verify=False):
    config = settings()
    saved, identity = parent(dataset)
    with tqdm(total=3, desc=f'{dataset}: freeze date-only within-version roles', unit='stage') as bar:
        result = core.make_plan(saved['rows'], saved['component_by_cve'], config, dataset)
        core.require(len(result['rows']) == config['datasets'][dataset]['expected_pool'],
                     'Eligible pool differs from reviewed metadata; investigate without changing cuts')
        bar.update()
        value = {'schema': config['schema'], 'dataset': dataset, 'config': config,
                 'identity': identity, 'target_version': config['datasets'][dataset]['version'], **result}
        table = membership(value)
        bar.update()
        destination = location(dataset, config) / 'plan'
        with data.publication.transaction(destination, verify=verify) as temp:
            data.base.write(temp / 'fold_manifest.json', value)
            data.base.write(temp / 'comparison_spec.json', {**config['future_comparison'],
                            'training_enabled': False, 'dataset': dataset,
                            'target_version': value['target_version'], 'fold_manifest_sha256': data.c.sha(temp / 'fold_manifest.json')})
            (temp / 'fold_membership.csv').write_text(data.reporting.csv_text(table))
            lines = [f"CT06 {dataset}: frozen {value['target_version']} target-date plan",
                     'No outcome values used to allocate roles. No model inputs prepared or model fitted.',
                     f"Original training CVEs: {len(saved['rows'])}; selected version: {len(result['rows'])}; "
                     f"other versions retained outside this study: {len(result['out_of_version_ids'])}.",
                     f"Assessment starts: {result['cut_dates']}",
                     f"Potential OOF assessment: {len(result['oof_ids'])}; warmup-only: {len(result['warmup_only_ids'])}.",
                     'Optimization fits weights/transforms; selection chooses checkpoints; assessment is only scored after fitting.',
                     'Within each fold, targets in earlier roles strictly predate targets in later roles.',
                     'Evidence groups come from ALL original training CVEs. Version filtering does not sever transitive links.',
                     'Shared evidence is not a declaration that two CVEs describe the same vulnerability.',
                     'D/DF/DM retain the same CVEs and structured branch. No G/N summaries enter these proposed models.',
                     'Future structured transforms must be fitted again on optimization IDs; frozen text means can be reused.',
                     'No official train/validation/test identity changes. No training CLI exists for this stage.',
                     '', 'Fold | optimization | selection | assessment | purged for assessment | purged for selection']
            lines += [' | '.join(str(f[k]) if k == 'fold' else str(len(f[k])) for k in
                                  ('fold', 'optimization', 'selection', 'assessment', 'purged_to_assessment', 'purged_to_selection'))
                      for f in result['folds']]
            (temp / 'REPORT.txt').write_text('\n'.join(lines) + '\n')
            data.publication.check_pins(identity['input_hashes'])
            data.publication.check_pins(identity['source_pins'])
        bar.update()
    print(f'CT06 {dataset} within-version plan {"reproduced" if verify else "saved"}: {destination}. No training or test access.')


def audit(dataset, verify=False):
    config = settings(); root = location(dataset, config)
    data.publication.check_lock(root / 'plan')
    manifest = data.c.read(root / 'plan/fold_manifest.json')
    saved, identity = parent(dataset)
    core.require(manifest['config'] == config and manifest['identity'] == identity, 'Changed frozen plan identity')
    reconstructed = core.make_plan(saved['rows'], saved['component_by_cve'], config, dataset)
    core.require(all(manifest[k] == v for k, v in reconstructed.items()), 'Role plan does not reproduce')
    raw = data.c.read(saved['label_source'])
    if isinstance(raw, list):
        lookup = {r['cve']: r for r in raw}
        core.require(len(lookup) == len(raw), 'Duplicate source labels')
    else:
        core.require(isinstance(raw, dict), 'Unexpected source labels')
        lookup = raw
    rows, comp = manifest['rows'], manifest['component_by_cve']
    labels = core.validate_labels(rows, lookup)
    del raw, lookup
    # The shared development container is decoded, but only selected original-training labels are consumed.
    table, strata, checks = [], [], []
    for fold in tqdm(manifest['folds'], desc=f'{dataset}: audit locked fold support', unit='fold'):
        core.verify_fold(fold, rows, comp)
        summaries = {}
        for role in ('optimization', 'selection', 'assessment', 'purged_to_assessment', 'purged_to_selection', 'future'):
            stats = core.support(fold[role], rows, labels, comp, config['high_score_cutoff'])
            summaries[role] = stats
            table.append({'fold': fold['fold'], 'role': role, **stats})
            if role in ('optimization', 'selection', 'assessment'):
                for present in (True, False):
                    ids = [c for c in fold[role] if rows[c]['source_mask']['M'] == present]
                    strata.append({'fold': fold['fold'], 'role': role, 'stratum': 'M_present' if present else 'M_absent',
                                   **core.support(ids, rows, labels, comp, config['high_score_cutoff'])})
        checks.append({'fold': fold['fold'], **core.gates(summaries, config)})
    cohort = {name: core.support(ids, rows, labels, comp, config['high_score_cutoff']) for name, ids in
              [('eligible_pool', sorted(rows)), ('potential_oof', manifest['oof_ids']), ('warmup_only', manifest['warmup_only_ids'])]}
    for key in ('cves', 'high_score', 'lower_score', 'M_present', 'M_absent', 'high_and_M_present'):
        core.require(cohort['eligible_pool'][key] == cohort['potential_oof'][key] + cohort['warmup_only'][key],
                     'OOF/warmup accounting differs')
    failed = [r['fold'] for r in checks if r['failed_checks']]
    finding = {'schema': 'ct06_within_version_support_v1', 'dataset': dataset,
               'target_version': manifest['target_version'], 'priority': config['datasets'][dataset]['priority'],
               'plan_sha256': data.c.sha(root / 'plan/fold_manifest.json'),
               'plan_lock_sha256': data.c.sha(root / 'plan/artifact_lock.json'),
               'cut_dates': manifest['cut_dates'], 'identity': identity, 'cohort_support': cohort,
               'fold_support': table, 'source_strata': strata, 'support_gates': checks,
               'original_training_cves': len(manifest['all_original_training_ids']),
               'other_versions_outside_new_study': len(manifest['out_of_version_ids']),
               'all_count_gates_pass': all(r['count_gates_pass'] for r in checks),
               'folds_requiring_support_review': failed,
               'status': 'SUPPORT_REVIEW_REQUIRED' if failed else 'COUNTS_PASS_NOT_TRAINING_APPROVAL',
               'training_enabled': False, 'test_access': False,
               'outer_validation_payloads_used': False,
               'shared_container_note': 'Existing development label container decoded for integrity; only selected original-training targets used.',
               'limitations': [
                   'The cohort/version choice follows previously inspected results. This is new exploratory development, not independent confirmation.',
                   'This is retrospective Day-60 EPSS estimation, not a publication-time or historically verified temporal-text forecast.',
                   'One model-version interval removes a release transition, not changes over time within that version.',
                   'Version assignment uses published release dates; archived score headers are not revalidated here.',
                   'Exact-document components are conservative evidence groups, not a complete near-duplicate/family audit.',
                   'Warmup CVEs cannot get OOF assessment without future-data training. Report actual assessment support, not all eligible positives.',
                   'Support thresholds are review triggers, not statistical power guarantees. Weak folds stay recorded and dates never move automatically.',
                   'Repeated forward folds are dependent. Future paired intervals keep shared components together and condition on fitted models.',
                   'Rank, high-score and source subgroups are secondary; EPSS >=0.1 is not a confirmed exploitation label.',
                   'Old CT05 fits and official role IDs stay unchanged. New fold-specific preprocessing and a new smoke gate are required before fitting.',
                   'No training, test scoring, graph extraction, embedding generation or wider ridge search is implemented in this preflight.']}
    lines = [f"CT06 {dataset}: {manifest['target_version']} within-version support",
             'Status: ' + finding['status'] + '. Training remains disabled.',
             'Roles: optimization fits weights and preprocessing; selection chooses the checkpoint; assessment measures held-out inner-fold error.',
             'High score means observed Day-60 EPSS >=0.1. M means accepted post prose, not merely a raw attachment.',
             'All roles are subsets of the ORIGINAL TRAINING role. Official validation and test remain untouched.',
             'Fold | role | CVEs | high scores | accepted posts | high scores with posts | evidence groups']
    lines += [f"{r['fold']} | {r['role']} | {r['cves']} | {r['high_score']} | {r['M_present']} | "
              f"{r['high_and_M_present']} | {r['evidence_components']}" for r in table]
    lines += ['', 'Eligible pool, potential OOF assessment and initial warmup:', data.publication.dump(cohort),
              'Support gates, fixed before reading these fold outcomes:', data.publication.dump(checks),
              '', 'Interpretation and limitations:', *finding['limitations'], '',
              'Next gate: review these exact counts and support warnings before implementing separate labels,',
              'optimization-fitted structured inputs, cached D/DM means and a bounded D/DF/DM training smoke test.',
              'No training commands are enabled by this audit. Do not change cuts merely to improve support or metrics.']
    with data.publication.transaction(root / 'audit', verify=verify) as temp:
        data.base.write(temp / 'findings.json', finding)
        (temp / 'REPORT.txt').write_text('\n'.join(lines) + '\n')
        for name, records in [('fold_support', table), ('source_strata', strata), ('support_gates', checks)]:
            (temp / (name + '.csv')).write_text(data.reporting.csv_text(records))
        figure(temp, table, dataset, manifest['target_version'])
        data.publication.check_pins(identity['input_hashes'])
        data.publication.check_pins(identity['source_pins'])
    print(f'CT06 {dataset} within-version support {"reproduced" if verify else "saved"}: {root / "audit"}. No training or test scoring.')


def figure(directory, table, dataset, version):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, key, title in zip(axes, ('cves', 'high_score', 'M_present'),
                              ('All CVEs', 'Day-60 EPSS >= 0.1', 'CVEs with accepted posts')):
        for role, offset, color in [('optimization', -.25, '#277c91'), ('selection', 0, '#bb4765'), ('assessment', .25, '#568744')]:
            records = [r for r in table if r['role'] == role]
            bars = ax.bar(np.array([r['fold'] for r in records]) + offset, [r[key] for r in records],
                          .24, label=role, color=color)
            ax.bar_label(bars, fontsize=7, padding=2)
        ax.set(title=title, xlabel='Chronological inner fold', ylabel='Number of CVEs', xticks=[1, 2, 3])
        ax.margins(y=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'{dataset} / {version}: original-training records only; no models fitted')
    fig.tight_layout()
    fig.savefig(directory / 'fold_support.png', dpi=150, metadata={'Software': 'CT06 within-version preflight'})
    plt.close(fig)
