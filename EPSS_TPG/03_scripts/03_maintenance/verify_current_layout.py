#!/usr/bin/env python3
"""Validate the selected revision data and current layout, without training."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess

from audit_revision_data import audit

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.tpg-project-root').is_file())
REVISION = 'a1e32ca3f16052b987767c013a49e2441b972e55'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expect-empty', action='store_true', help='Also verify the post-cleanup empty working directories')
    args = parser.parse_args()
    errors = []
    submodule = ROOT.parent / 'SummTPGVul'
    revision = subprocess.check_output(['git', '-C', str(submodule), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        errors.append('Submodule HEAD differs from the verified dataset snapshot')
    source = submodule / 'Sec4AI4Sec-EPSS/Data_Files'
    for name in ['cves_merged_refetched.json', 'cves_merged_with_url_dates_vc_kev.json']:
        records = json.loads((source / name).read_text())
        if len(records) != 5968 or len({r['cve'] for r in records}) != 5968:
            errors.append(f'{name}: unexpected count or duplicate CVEs')
        if not all(isinstance(r.get('description'), str) and r['description'].strip() for r in records):
            errors.append(f'{name}: missing description')
        if name.endswith('_vc_kev.json'):
            if not all(type(r.get('in_vckev')) is bool for r in records):
                errors.append('Membership flags must be native booleans')
            if sum(r.get('in_vckev') is True for r in records) != 548:
                errors.append('Unexpected VulnCheck membership count')
        print(f'{name}: {len(records)} records')
    with (source / 'cves_merged_refetched.csv').open(newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if len(rows) != 5968 or len(reader.fieldnames) != 26:
            errors.append('Refetched CSV count/schema differs from verified snapshot')
    for base, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:] = [d for d in dirs if d not in {'__pycache__', '01_texlive_installer'}]
        for name in dirs + files:
            p = Path(base) / name
            if p.is_symlink() and not p.exists() and p.name not in {'tpg_workspace.db', 'tpg_workspace.db-shm', 'tpg_workspace.db-wal'}:
                errors.append(f'Broken alias: {p.relative_to(ROOT)}')
    if args.expect_empty:
        for name in ['04_data/01_records_and_graphs', '05_results/01_training',
                     '05_results/02_evaluation', '05_results/03_dataset_analysis', '06_runtime/01_logs']:
            for p in (ROOT / name).rglob('*'):
                if p.is_file() and p.name not in {'.gitkeep', 'README.md'}:
                    errors.append(f'Unexpected retained artifact: {p.relative_to(ROOT)}')
    snapshot = audit()
    errors.extend(snapshot['integrity_errors'])
    for warning in snapshot['readiness_warnings']:
        print('WARNING:', warning)
    for error in errors:
        print('ERROR:', error)
    print(f'Validation errors: {len(errors)}')
    print('This validates the downloaded snapshot, not training readiness; review the warnings.')
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
