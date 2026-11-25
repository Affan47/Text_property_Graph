"""Offline CC01 label review; no requests, training or held-out label joins."""
import argparse
from collections import Counter
import csv
import io
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / '02_prediction/01_package'))
from epss.catalogue_classification_v1 import snapshot as frozen


def csv_text(rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda r: tuple(str(r[k]) for k in fields)))
    return stream.getvalue()


def summarize(ids, labels, legacy):
    current = {r['cve']: r for r in labels}
    previous = {r['cve']: r for r in legacy}
    frozen.require(len(current) == len(labels) and len(previous) == len(legacy)
                   and set(current) == set(previous) == set(ids), 'Scope mismatch or duplicate training CVE')
    counts, transitions, changed = Counter(), Counter(), []
    for cve in sorted(ids):
        row = current[cve]
        frozen.require(row['status'] in ('LISTED', 'NOT_LISTED')
                       and row['label'] == int(row['status'] == 'LISTED'), 'Unverified or inconsistent label')
        prior = previous[cve]['saved_catalogue_state']
        frozen.require(prior in ('LISTED', 'NOT_LISTED_UNVERIFIED', 'UNKNOWN'), 'Unknown legacy state schema')
        counts[row['status']] += 1
        transitions[(prior, row['status'])] += 1
        if (prior == 'LISTED') != (row['status'] == 'LISTED') or prior == 'UNKNOWN':
            changed.append({'cve': cve, 'prior_state': prior, 'snapshot_state': row['status'],
                            'snapshot_date_added': row['catalogue_date_added']})
    n = len(ids)
    frozen.require(n > 0, 'Empty training cohort')
    return {'cves': n, 'listed': counts['LISTED'], 'not_listed': counts['NOT_LISTED'],
            'unknown': 0, 'listed_fraction': counts['LISTED'] / n,
            'all_not_listed_accuracy': counts['NOT_LISTED'] / n,
            'all_not_listed_positive_recall': 0., 'all_not_listed_positive_f1': 0.,
            'legacy_disagreements': len(changed)}, [
                {'prior_state': a, 'snapshot_state': b, 'cves': n}
                for (a, b), n in sorted(transitions.items())], changed


def generate(snapshot_id):
    frozen.require(__import__('re').fullmatch('[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}', snapshot_id), 'Unsafe snapshot ID')
    cfg = frozen.read(frozen.CONFIG)
    source = ROOT / cfg['output'] / snapshot_id
    frozen.check_lock(source / 'plan')
    plan = frozen.read(source / 'plan/plan.json')
    frozen.require(plan['source_pins'] == frozen.pins() and plan['specification'] == cfg, 'Frozen CC01 identity changed')
    frozen.check_pins(plan['prerequisite_pins'])
    findings = frozen.verify_snapshot(source, plan)
    receipt = frozen.read(source / 'snapshot/receipt.json')
    sources = {str(p): frozen.sha(p) for p in (source / 'plan/plan.json', source / 'plan/artifact_lock.json',
               source / 'snapshot/receipt.json', source / 'snapshot/artifact_lock.json',
               source / 'snapshot/findings.json', Path(__file__), Path(__file__).with_name('test_catalogue_review.py'))}
    datasets, transitions, changed, maps = {}, [], [], {}
    for name, spec in cfg['datasets'].items():
        audit = ROOT / spec['audit']
        frozen.check_lock(audit)
        old_file = audit / 'cve_readiness.csv'
        label_file = source / 'snapshot' / (name + '_labels.json')
        labels = frozen.read(label_file)
        with old_file.open(newline='') as stream:
            legacy = list(csv.DictReader(stream))
        summary, moves, differences = summarize(plan['training_ids'][name], labels, legacy)
        frozen.require(summary['listed'] == findings['counts'][name]['LISTED']
                       and summary['not_listed'] == findings['counts'][name]['NOT_LISTED'], 'Label summary mismatch')
        datasets[name] = summary
        maps[name] = {r['cve']: r['label'] for r in labels}
        transitions.extend({'dataset': name, **r} for r in moves)
        changed.extend({'dataset': name, **r} for r in differences)
        for path in (old_file, label_file, audit / 'artifact_lock.json'):
            sources[str(path)] = frozen.sha(path)
    shared = set(maps['reddit']) & set(maps['previous'])
    frozen.require(all(maps['reddit'][c] == maps['previous'][c] for c in shared), 'Same-snapshot overlap labels disagree')
    result = {'schema': 'cc01_snapshot_review_v1', 'snapshot_id': snapshot_id,
              'snapshot_at': receipt['snapshot_at'], 'collection_completed_at': receipt['request_events'][-1]['ended_at'],
              'records': receipt['inventory']['records'], 'unique_catalogue_cves': receipt['inventory']['unique_cves'],
              'archive_sha256': receipt['official_archive_sha256'], 'receipt_sha256': frozen.sha(source / 'snapshot/receipt.json'),
              'archive_bytes': receipt['request_events'][-1]['bytes'], 'datasets': datasets,
              'shared_training_cves': len(shared), 'shared_training_listed': sum(maps['reddit'][c] for c in shared),
              'label_gate': findings['label_gate'], 'remaining_gates': findings['remaining_gates'],
              'training': False, 'network_requests': 0, 'heldout_label_joins': False,
              'interpretation': findings['meaning'],
              'limitations': ['Not listed is not proof of no exploitation.',
                              'Same-snapshot classification is not future-listing prediction.',
                              'Legacy disagreements do not identify historical additions/removals or collector error causes.',
                              'The collections overlap and are not independent replication cohorts.',
                              'EPSS timing, predictor disclosures and inner roles remain unresolved; training is not released.']}
    return result, transitions, changed, sources


def run(snapshot_id, verify):
    result, transitions, changed, sources = generate(snapshot_id)
    output = ROOT / '05_results/09_catalogue_classification/02_snapshot_review_v1' / snapshot_id
    with frozen.exclusive(output.parent):
        frozen.require(output.exists() if verify else not output.exists(), 'Use --verify-only for an existing review')
        with tempfile.TemporaryDirectory(prefix='.cc01-review-', dir=output.parent) as tmp:
            stage = Path(tmp) / 'ready'
            stage.mkdir()
            frozen.write(stage / 'findings.json', result)
            frozen.write(stage / 'identity.json', {'source_pins': sources})
            (stage / 'legacy_transitions.csv').write_text(csv_text(transitions, ['dataset', 'prior_state', 'snapshot_state', 'cves']))
            (stage / 'legacy_disagreements.csv').write_text(csv_text(changed, ['dataset', 'cve', 'prior_state', 'snapshot_state', 'snapshot_date_added']))
            lines = ['CC01: completed snapshot review', '', 'Snapshot UTC: ' + result['snapshot_at'],
                     'The archive hash, complete records and all training-only labels reproduce.', '',
                     'Dataset | CVEs | Listed | Not listed | Listed % | All-negative accuracy %']
            for name, row in result['datasets'].items():
                lines.append(f"{name} | {row['cves']} | {row['listed']} | {row['not_listed']} | "
                             f"{100 * row['listed_fraction']:.2f} | {100 * row['all_not_listed_accuracy']:.2f}")
            lines.extend(['', 'All-negative positive recall and F1 are zero, despite high accuracy.',
                          'The two datasets share ' + str(result['shared_training_cves']) + ' training CVEs.',
                          'Prior-state differences are descriptive snapshot disagreements, not new listing dates.', '',
                          'Next: obtain date-aligned EPSS, screen predictor disclosures, freeze inner roles/settings.',
                          'FC01 stays unchanged. No requests, training or held-out label joins by this review.', '',
                          *result['limitations']])
            (stage / 'REPORT.txt').write_text('\n'.join(lines) + '\n')
            frozen.lock(stage)
            frozen.check_pins(sources)
            if verify:
                frozen.check_lock(output)
                frozen.require(frozen.read(stage / 'artifact_lock.json') == frozen.read(output / 'artifact_lock.json'),
                               'Snapshot review does not reproduce')
            else:
                stage.rename(output)
    print('CC01 completed-snapshot review ' + ('reproduced: ' if verify else 'saved: ') + str(output))
    print(frozen.dump(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-id', default='catalogue_v1')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    run(args.snapshot_id, args.verify_only)
