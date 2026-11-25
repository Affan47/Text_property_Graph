"""Compare a hash-pinned repository export with training flags, not holdout data."""
from collections import Counter, defaultdict
import csv
from pathlib import Path

from tqdm import tqdm

from ..catalogue_readiness_v1 import audit

ROOT = audit.ROOT
CONFIG = ROOT/'02_prediction/04_configs/07_fresh_studies/11_catalogue_snapshot_v1.json'
CLI = ROOT/'02_prediction/03_analysis/07_fresh_studies/21_review_saved_catalogue.py'
OUTPUT = ROOT/'05_results/08_fresh_studies/16_catalogue_checks_v1'


def compare(records, rows, commit_time):
    ids = {r['cve'] for r in rows}
    audit.REQUIRE(len(ids) == len(rows), 'Duplicate audit CVE')
    matches = defaultdict(list)
    for record in records:
        audit.REQUIRE(isinstance(record, dict) and isinstance(record.get('cve'), list), 'Unexpected export schema')
        audit.REQUIRE(all(isinstance(c, str) and audit.common.CVE.fullmatch(c) for c in record['cve']), 'Invalid CVE IDs in export')
        for c in set(record['cve']) & ids:
            matches[c].append(record)
    output = []
    for r in rows:
        c = r['cve']; entries = matches[c]
        audit.REQUIRE(r['saved_in_vckev'] in ('True', 'False'), 'Unresolved saved flag schema')
        saved = r['saved_in_vckev'] == 'True'
        present = bool(entries)
        category = ('SAVED_LISTED_AND_PRESENT' if saved else 'SAVED_FALSE_BUT_PRESENT') if present else (
            'SAVED_LISTED_BUT_ABSENT' if saved else 'SAVED_FALSE_AND_ABSENT_UNVERIFIED')
        dates = sorted({e.get('date_added') for e in entries if isinstance(e.get('date_added'), str)})
        added, _ = audit.parse_time(r.get('vckev_date_added'))
        output.append({'cve': c, 'saved_in_vckev': saved, 'present_in_saved_export': present,
                       'comparison': category, 'matching_export_records': len(entries),
                       'saved_addition': r.get('vckev_date_added'), 'export_addition_values': dates,
                       'saved_listed_addition_matches': saved and len(entries) == 1 and dates == [r.get('vckev_date_added')],
                       'saved_addition_after_repository_commit': added > commit_time if added else None,
                       'verified_current_label': None, 'released_training_label': None,
                       'interpretation': 'Positive corroboration only; absent entries are not authenticated negatives'})
    return sorted(output, key=lambda r: r['cve'])


def run(dataset, verify=False):
    spec = audit.READ(CONFIG); path = ROOT/spec['snapshot']
    audit.REQUIRE(path.stat().st_size == spec['bytes'] and audit.SHA(path) == spec['sha256'], 'Saved export differs from Git LFS pointer')
    records = audit.READ(path)
    audit.REQUIRE(isinstance(records, list) and len(records) == spec['records'], 'Export count/schema changed')
    original = audit.READ(audit.CONFIG); cfg = original['datasets'][dataset]
    base = ROOT/original['output']/cfg['output_name']/'audit'
    audit.artifact_io.check_lock(base)
    pins = dict(audit.READ(base/'identity.json')['input_hashes'])
    audit.artifact_io.check_pins(pins)
    with (base/'cve_readiness.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    audit.REQUIRE({r['cve'] for r in rows} == set(audit.READ(base/'training_scope_ids.json')), 'Training-only row mismatch')
    commit, _ = audit.parse_time(spec['commit_timestamp'])
    with tqdm(total=1, desc='Compare training-only flags with saved repository export', unit='stage') as bar:
        output = compare(records, rows, commit); bar.update()
    counts = dict(Counter(r['comparison'] for r in output))
    differences = [r for r in output if r['saved_in_vckev'] != r['present_in_saved_export']]
    summary = {'schema': spec['schema'], 'dataset': dataset, 'training_candidates': len(rows),
               'export_records': len(records), 'comparisons': counts,
               'saved_listed_addition_matches': sum(r['saved_listed_addition_matches'] for r in output),
               'difference_count': len(differences),
               'absent_saved_listed_added_after_commit': sum(r['comparison'] == 'SAVED_LISTED_BUT_ABSENT'
                                                           and r['saved_addition_after_repository_commit'] is True for r in output),
               'duplicate_target_record_cves': sum(r['matching_export_records'] > 1 for r in output),
               'verified_current_labels': 0, 'verified_original_snapshot_negatives': 0,
               'network_calls_in_review': 0, 'training': False, 'holdout_outcomes_analyzed': False,
               'decision': 'POSITIVE_CORROBORATION_FOUND; NEGATIVES_AND_SNAPSHOT_TIME_REQUIRE_VERIFICATION'}
    lines = [f'CT09 {dataset}: saved VulnCheck export comparison', '',
             f"Training candidates: {len(rows)}. Export records: {len(records)}.",
             'The file was found on the modify-summ-generation branch, not the two dataset branches.',
             'Its bytes match the Git LFS SHA-256. It is a list with no enclosing retrieval-time/completeness metadata.',
             'The Git commit time is not the original download time or a catalogue cutoff.', '',
             'Comparison | Training CVEs']
    lines += [f'{k} | {v}' for k, v in sorted(counts.items())]
    lines += ['', f"Matching saved addition dates among saved-listed CVEs present in export: {summary['saved_listed_addition_matches']}.",
              f"Saved-listed CVEs absent from export whose addition date is after the commit: {summary['absent_saved_listed_added_after_commit']}.",
              'Later additions are compatible with catalogue changes. Earlier addition dates do not prove that a record was in this export.',
              'No difference is automatically called a collector error; backfilling, coverage and collection time remain possible explanations.', '',
              'CVE | Saved flag | In export | Saved addition | Export addition']
    lines += [f"{r['cve']} | {r['saved_in_vckev']} | {r['present_in_saved_export']} | {r['saved_addition']} | {r['export_addition_values']}" for r in differences]
    lines += ['', 'Decision', 'The export corroborates positive listings; it cannot authenticate a fresh or original-snapshot negative.',
              'Original flags, CT08 reports, splits and all completed models remain unchanged.',
              'Next: fresh training-only lookup receipts for the bounded pilot and these discrepancies, with errors kept UNKNOWN.',
              'Then approve a single retrospective catalogue target policy and verify the whole approved cohort before fitting.',
              'Historical future-inclusion targets still require follow-up and exact-text history evidence.']
    for p in [CONFIG, CLI, CLI.with_name('test_catalogue_checks.py'), Path(__file__), path, *base.iterdir()]:
        if p.is_file():
            pins[str(p)] = audit.SHA(p)
    artifacts = {'identity.json': {'provenance': spec, 'pins': pins}, 'findings.json': summary,
                 'cve_comparisons.csv': audit.csv_text(output, list(output[0])),
                 'discrepancies.json': differences,
                 'additional_training_request_plan.json': {'ids': sorted(r['cve'] for r in differences),
                    'reason': 'Saved flag versus repository export discrepancy; not prevalence sampling', 'training_only': True},
                 'REPORT.txt': '\n'.join(lines)+'\n'}
    dest = OUTPUT/cfg['output_name']/'saved_export_review'
    audit.publish(dest, artifacts, pins, verify)
    print(f'CT09 {dataset} saved-export review {"reproduced" if verify else "saved"}: {dest}; no live checks or training.')
