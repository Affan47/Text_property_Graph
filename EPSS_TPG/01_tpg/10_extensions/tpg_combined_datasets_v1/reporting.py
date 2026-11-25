"""Reproduce dataset accounting without adapting any test text or outcomes."""
import gzip
import json
from . import adapters as a
from tpg_combined_v1 import common as c, sources

ANALYSIS = c.ROOT / '05_results/03_dataset_analysis/02_reddit_dataset'
DIAG = ANALYSIS / '06_rd01_diagnostic_v1'
LOCK = ANALYSIS / '08_rd01_source_lock_v1'
OUTPUT = c.ROOT / '05_results/08_fresh_studies/00_dataset_coverage_v1'


def exclusion_accounting(cohort, flow, counts):
    reasons = flow['post_launch_exclusion_first_reason']
    task = flow['task_exclusions']['RD02']
    rows = [{'reason': 'Published before the EPSS start date, 2021-04-14',
             'count': cohort['all_cves'] - cohort['post_launch']}]
    names = {'prior_test_connected': 'Linked to earlier test evidence (post-launch CVEs)',
             'cross_publication_intervals': 'Connected group spans different publication-role intervals',
             'prior_exposure_barred_from_test': 'Earlier train/validation exposure prevents a new test assignment',
             'prior_role_conflict': 'Earlier role conflicts with the proposed role'}
    c.require(set(reasons) == set(names), 'Unreviewed role exclusion reason')
    rows += [{'reason': names[key], 'count': reasons[key]} for key in names]
    c.require(all(key.split(':')[1] in ('missing_day60', 'target_after_role_end') for key in task),
              'Unreviewed target exclusion reason')
    for suffix, name in (('missing_day60', 'Day-60 EPSS target missing after preceding exclusions'),
                         ('target_after_role_end', 'Day-60 target date beyond the assigned role end')):
        rows.append({'reason': name, 'count': sum(n for k, n in task.items() if k.endswith(':' + suffix))})
    c.require(sum(r['count'] for r in rows) + sum(counts.values()) == cohort['all_cves'],
              'Cohort accounting does not conserve all CVEs')
    return rows


def report(resume=False):
    for directory in (DIAG, LOCK):
        c.check_files(directory, c.read(directory / 'artifact_lock.json')['files'])
    old = a.previous_rows()
    provenance = c.read(a.EXPORT / 'run_identity.json')
    manifest = c.read(LOCK / 'rd01_frozen_manifest.json')
    roles = manifest['tasks']['RD02']
    c.require(not manifest['test_release_enabled'], 'Unexpected test release')
    reddit = c.read(sources.BASE / 'predictor_inputs.json.gz')
    a.validate_roles(reddit, {k: roles['ids'][k] for k in ('train', 'validation')}, roles['ids']['test'])
    for directory in (sources.BASE, sources.OVERLAY):
        c.check_files(directory, c.read(directory / 'artifact_lock.json')['files'])
    with gzip.open(sources.OVERLAY / 'summary_inputs.jsonl.gz', 'rt') as stream:
        overlay = {r['cve']: r for r in map(json.loads, stream)}
    for row in reddit:
        c.require(overlay[row['cve']]['split'] == row['split'], 'Summary role mismatch')
        row['documents'] += overlay[row['cve']]['documents']
    findings = c.read(DIAG / 'findings.json')
    flow = c.read(DIAG / 'candidate_exclusion_flow.json')['retain_all/retain_matching_prior_roles']
    exclusions = exclusion_accounting(findings['cohort'], flow, roles['counts'])
    old_counts = {s: len(v) for s, v in provenance['ids'].items()}
    old_counts['test'] = provenance['test_ids_count']
    overlap = {f'previous_{s}__reddit_{t}': len(set(provenance['ids'][s]) & set(roles['ids'][t]))
               for s in ('train', 'validation') for t in ('train', 'validation', 'test')}
    coverage = {'reddit': a.source_coverage(reddit), 'previous': a.source_coverage(old)}
    dimensions = {'reddit': len(c.read(sources.BASE / 'structured_state.json')['feature_names']),
                  'previous': len(c.read(a.EXPORT / 'structured_state.json')['feature_names'])}
    c.require(c.read(sources.BASE / 'structured_state.json')['feature_names'] ==
              c.read(a.EXPORT / 'structured_state.json')['feature_names'], 'Structured feature layouts differ')
    paths = [DIAG / 'findings.json', DIAG / 'candidate_exclusion_flow.json',
             LOCK / 'rd01_frozen_manifest.json', a.EXPORT / 'artifact_lock.json',
             sources.BASE / 'artifact_lock.json', sources.OVERLAY / 'artifact_lock.json']
    result = {'schema': 'shared_dataset_accounting_v1',
              'datasets': {'reddit': {'total': findings['cohort']['all_cves'], 'roles': roles['counts'],
                                      'development': len(reddit), 'disjoint_exclusions': exclusions},
                           'previous': {'total': provenance['raw_cves'], 'roles': old_counts,
                                        'development': len(old),
                                        'outside_preserved_primary_roles': provenance['raw_cves'] - sum(old_counts.values())}},
              'source_coverage_prose_policy': coverage, 'structured_dimensions': dimensions,
              'development_id_overlap': overlap, 'source_hashes': {str(p): c.sha(p) for p in paths},
              'adapter_hashes': a.adapter_pins(), 'training': False, 'test_adapted': False,
              'limitations': ['Older test records were evaluated in previous studies; cleanup does not make them unseen.',
                              'The two datasets overlap; separate runs are not independent replication.',
                              'Reddit has no accepted validation webpage summaries; U-dependent comparisons remain blocked.',
                              'Older accepted summaries are available, but coverage does not establish their quality.',
                              'Current text and structured values are retrospective, not historically verified predictors.',
                              'No new GNN training runner or fitted model is provided by this cache adapter.']}
    lines = ['Two datasets, one combined-text TPG architecture', '',
             'The six variants are D, M, U, DM, DU, DMU. U means webpage summaries, not URL strings.',
             'M means Reddit posts in the new collection and the saved multi-platform posts in the older collection.',
             'Both use the same v6 parser, relation schema, 32 subtype channels and frozen SecureBERT 2.0.',
             'The graph is built first. Contextual embeddings are then assigned to graph-node text spans.',
             'The same structured encoder is fitted separately on each dataset training split.', '',
             'Dataset | Total CVEs | Training | Validation | Test (not built) | Development graphs planned']
    for dataset, info in result['datasets'].items():
        counts = info['roles']
        lines.append(f'{dataset} | {info["total"]} | {counts["train"]} | {counts["validation"]} | {counts["test"]} | {info["development"]} CVEs x six views')
    lines += ['', 'Where the Reddit CVEs went (non-overlapping exclusions):']
    lines += [f'{r["count"]}: {r["reason"]}' for r in exclusions]
    lines += ['The remaining 29,162 CVEs are 24,213 train + 972 validation + 3,977 test.',
              'The source has 192 post-launch CVEs without Day-60 scores in total; only 80 remain',
              'after preceding exclusions. Subtracting both counts would double count exclusions.',
              'There are 4,987 prior-test-linked CVEs across all publication years, of which 4,245',
              'are in the post-launch group. Shared evidence is not proof that vulnerabilities are identical.',
              '', 'The older 1,667 CVEs outside the saved primary roles are not silently reassigned.',
              'This adapter preserves the existing task; it does not impose an arbitrary graph-capacity limit.',
              'Generating a graph for an unlabelled CVE would not create a supervised EPSS target.',
              '', 'Source availability (CVEs with nonempty accepted text, M uses prose policy):',
              'Dataset | Split | CVEs | D | M | G | N | U (G or N)']
    for dataset, splits in coverage.items():
        for split, counts in splits.items():
            lines.append(f'{dataset} | {split} | ' + ' | '.join(str(counts[k]) for k in ('cves', 'D', 'M', 'G', 'N', 'U')))
    audit = c.read(a.EXPORT / 'audit.json')
    lines += ['', f'Older text screen: {audit["additional_quarantined_documents"]} additional EPSS-discussing documents',
              f'across {audit["quarantined_cves"]} CVEs were quarantined in the new export. No CVE was dropped.',
              'The raw dataset and old prepared inputs remain unchanged.', '', 'Cross-dataset ID overlap:']
    lines += [f'{key}: {value}' for key, value in overlap.items()]
    lines += ['', 'Limitations:'] + result['limitations']
    a.publish(OUTPUT, {'findings.json': result, 'REPORT.txt': '\n'.join(lines) + '\n'}, resume)
    print(f'Both-dataset accounting verified: {OUTPUT}. No graphs, training or test outcomes read.')
