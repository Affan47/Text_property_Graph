"""Check the known CF01 resumed-report CSV ordering difference, without scoring."""
import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / '01_package'))
from epss.continuous_forecasting import data

ROOT = data.OUTPUT.parent / 'stage2_v1/full'
OUTPUT = data.OUTPUT.parent / 'stage2_report_order_audit_v1'


def csv_rows(path):
    with path.open(newline='') as stream:
        rows = list(csv.reader(stream))
    expected = {(p, s) for p in ('persistence', 'lead_correction', 'seed_17', 'seed_29', 'seed_43')
                for s in ('all', 'day30_to_day60', 'lead7', 'lead14', 'lead30')}
    data.require(len(rows) == 26 and {tuple(r[:2]) for r in rows[1:]} == expected,
                 'Missing, duplicate or unexpected metric rows')
    return rows[0], {tuple(r[:2]): r[2:] for r in rows[1:]}


def audit(path, verify=False):
    original = data.check_lock(ROOT / 'report')
    replay = data.check_lock(path)
    data.require(original.keys() == replay.keys(), 'Report artifact set differs')
    differences = [k for k in original if original[k] != replay[k]]
    data.require(differences == ['metrics.csv'], 'Not the isolated CSV ordering issue')
    data.require(csv_rows(ROOT / 'report/metrics.csv') == csv_rows(path / 'metrics.csv'),
                 'CSV header or keyed values differ; cannot accept as row ordering')
    data.check_pins(data.read(ROOT / 'report/run_identity.json')['source_pins'])
    result = dict(schema='cf01_report_order_audit_v1', differences=differences,
                  exact_keyed_rows=25, numeric_tolerance_used=False,
                  original_lock_sha256=data.sha(ROOT / 'report/artifact_lock.json'),
                  replay_lock_sha256=data.sha(path / 'artifact_lock.json'),
                  audit_script_sha256=data.sha(Path(__file__)),
                  interpretation='Report-only equivalence check, not a new checkpoint replay. All other report bytes match.',
                  original_outputs_modified=False)
    with data.transaction(OUTPUT, verify) as stage:
        (stage / 'audit.json').write_text(data.dump(result))
        (stage / 'audit.txt').write_text(
            'CF01 report ordering audit\n\n'
            'The original report and fresh replay differ only in metrics.csv row order.\n'
            'The header and all 25 uniquely keyed predictor/slice rows match exactly, including every value.\n'
            'No numerical tolerance is used. Every other report file has an identical byte hash.\n'
            'Cause: resume reloads sorted JSON keys; fresh evaluation uses the declared slice insertion order.\n'
            'The frozen runner still rejects this byte-level difference. Original code, outputs and locks are preserved.\n'
            'This audit checks the saved report replay only; it does not retrain or independently replay checkpoints.\n')
    print('CF01 report-only CSV ordering audit ' + ('reproduced.' if verify else 'saved.') + ' All 25 keyed rows match exactly.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reproduced-report', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    audit(args.reproduced_report, args.verify_only)
