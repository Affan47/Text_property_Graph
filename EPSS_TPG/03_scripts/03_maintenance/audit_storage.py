"""Read-only storage inventory and bounded verification of lock-listed duplicates.

Only the requested report directory is written. No deletion or deduplication mode.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat

ROOT = Path(__file__).resolve().parents[3]
WORK = 'EPSS_TPG/01_tpg/05_workspace'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def group_savings(rows):
    # Count each inode once; linked aliases must never be counted as extra copies.
    unique = {(r['device'], r['inode']): r for r in rows}
    values = list(unique.values())
    return max(0, sum(r['allocated_bytes'] for r in values)
               - max((r['allocated_bytes'] for r in values), default=0))


def run(root, output, budget):
    root, output = root.resolve(), output.resolve()
    records, errors, symlinks = {}, [], []
    print('[1/3] Inventory regular files; no symlink traversal', flush=True)
    for directory, dirs, files in os.walk(root, followlinks=False,
                                          onerror=lambda e: errors.append(str(e))):
        dirs.sort()
        dirs[:] = [d for d in dirs if (Path(directory) / d).resolve() != output]
        for name in sorted(files):
            p = Path(directory) / name
            rel = p.relative_to(root).as_posix()
            try:
                s = p.lstat()
                if stat.S_ISLNK(s.st_mode):
                    symlinks.append(rel)
                    continue
                if not stat.S_ISREG(s.st_mode):
                    continue
                records[rel] = dict(path=rel, bytes=s.st_size,
                    allocated_bytes=s.st_blocks * 512, device=s.st_dev,
                    inode=s.st_ino, links=s.st_nlink, mtime_ns=s.st_mtime_ns)
            except OSError as exc:
                errors.append(str(exc))
    areas, suffixes = defaultdict(dict), defaultdict(dict)
    all_inodes, temporary = {}, []
    for rel, r in records.items():
        key = (r['device'], r['inode'])
        all_inodes[key] = r
        parts = Path(rel).parts
        area = '/'.join(parts[:2]) if parts[0] == 'EPSS_TPG' else parts[0]
        areas[area][key] = r
        if rel.startswith(WORK + '/'):
            areas['/'.join(parts[:4])][key] = r
        suffixes[Path(rel).suffix or '(no suffix)'][key] = r
        if '__pycache__' in parts or '.pytest_cache' in parts:
            temporary.append(r)
    totals = lambda table: sorted([
        dict(path=k, allocated_bytes=sum(r['allocated_bytes'] for r in v.values()),
             unique_files=len(v)) for k, v in table.items()],
        key=lambda r: (-r['allocated_bytes'], r['path']))
    print(f'[2/3] Read cache hash inventories ({len(records):,} files inventoried)', flush=True)
    claims, locks = defaultdict(set), []
    for rel in sorted(records):
        is_lock = rel.endswith('artifact_lock.json')
        is_receipt = '/receipts/' in rel and rel.endswith('.json')
        if not rel.startswith(WORK + '/') or not (is_lock or is_receipt):
            continue
        try:
            document = json.loads((root / rel).read_text())
            entries = document.get('files', {})
            if not isinstance(entries, dict):
                continue
            if is_lock:
                locks.append(dict(path=rel, sha256=sha(root / rel)))
            base = (root / rel).parent if is_lock else (root / rel).parent.parent
            for local, digest in entries.items():
                if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
                    continue
                target = base / local
                if target.is_symlink():
                    continue
                try:
                    key = target.resolve().relative_to(root).as_posix()
                except ValueError:
                    continue
                if key in records:
                    claims[key].add(digest)
        except (ValueError, OSError) as exc:
            errors.append(f'{rel}: {exc}')
    by_digest = defaultdict(list)
    conflicts = []
    for rel, digests in claims.items():
        if len(digests) != 1:
            conflicts.append(rel)
            continue
        r = records[rel]
        by_digest[(next(iter(digests)), r['bytes'], r['device'])].append(r)
    candidates = []
    for (digest, size, device), rows in by_digest.items():
        savings = group_savings(rows)
        if savings:
            candidates.append(dict(sha256=digest, bytes_per_file=size,
                potential_allocated_bytes=savings, paths=sorted(r['path'] for r in rows)))
    candidates.sort(key=lambda r: (-r['potential_allocated_bytes'], r['sha256']))
    print(f'[3/3] Verify duplicate samples (read budget {budget / 2**30:.1f} GiB)', flush=True)
    verified, used = [], 0
    for c in candidates:
        unique = {}
        for rel in c['paths']:
            r = records[rel]
            unique.setdefault((r['device'], r['inode']), r)
        cost = sum(r['bytes'] for r in unique.values())
        if used + cost > budget:
            continue
        if len(verified) >= 100:
            break
        good = True
        for r in unique.values():
            p = root / r['path']
            digest = sha(p)
            after = p.stat()
            good &= (digest == c['sha256'] and after.st_size == r['bytes']
                     and after.st_mtime_ns == r['mtime_ns']
                     and after.st_ino == r['inode'])
        used += cost
        verified.append(dict(**c, matches_current_bytes=bool(good)))
    large_data = [r for rel, r in records.items()
                  if (rel.startswith('EPSS_TPG/04_data/') or rel.startswith('SummTPGVul/'))
                  and r['bytes'] >= 10 * 2**20]
    report = dict(schema='storage_audit_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        root=str(root), policy='AUDIT ONLY: no deletion, no relinking, no source changes',
        filesystem=dict(total=shutil.disk_usage(root).total,
                        free=shutil.disk_usage(root).free),
        unique_regular_file_allocated_bytes=sum(r['allocated_bytes'] for r in all_inodes.values()),
        regular_file_paths=len(records), unique_regular_files=len(all_inodes),
        areas=totals(areas), extensions=totals(suffixes),
        regenerable_python_test_cache=dict(paths=[r['path'] for r in temporary],
            allocated_bytes=sum(r['allocated_bytes'] for r in
                {(r['device'], r['inode']): r for r in temporary}.values())),
        largest_files=sorted(records.values(), key=lambda r: -r['bytes'])[:40],
        dataset_files=sorted(large_data, key=lambda r: -r['bytes']),
        cache_locks=locks, conflicting_hash_claims=conflicts,
        lock_claimed_duplicate_groups=len(candidates),
        lock_claimed_potential_allocated_bytes=sum(c['potential_allocated_bytes'] for c in candidates),
        largest_lock_claimed_duplicate_groups=candidates[:100],
        content_verification_read_bytes=used, verified_duplicate_samples=verified,
        verified_sample_potential_allocated_bytes=sum(c['potential_allocated_bytes']
            for c in verified if c['matches_current_bytes']),
        errors=errors, file_symlinks_skipped=len(symlinks),
        limitations=[
            'Hash claims include artifact locks and per-chunk receipts (including incomplete caches); only explicitly verified samples were rehashed.',
            'Area totals overlap; existing hard links are counted once within each area.',
            'Potential savings require identical metadata/permissions, immutable files, all aliases handled and a fresh verification before any relinking.',
            'No dependency closure or remote-backup proof is implied. Old and untracked files are not automatically disposable.',
            'Directory blocks are excluded; figures differ slightly from du. No host-wide open-file check was possible in the isolated process namespace.',
            'No experimental arrays, labels, text or model tensors were deserialized.'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'storage_inventory.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['PROJECT STORAGE AUDIT', report['policy'], '',
        f"Free home-filesystem space: {report['filesystem']['free'] / 2**30:.2f} GiB",
        f"Unique regular-file storage: {report['unique_regular_file_allocated_bytes'] / 2**30:.2f} GiB", '',
        'AREAS (independent totals; do not add nested areas):']
    lines += [f"{r['allocated_bytes'] / 2**30:9.3f} GiB  {r['path']}" for r in report['areas']]
    lines += ['', 'DUPLICATE CACHE STORAGE',
        f"Lock-claimed potential: {report['lock_claimed_potential_allocated_bytes'] / 2**30:.3f} GiB (not all rehashed)",
        f"Content-verified sample potential: {report['verified_sample_potential_allocated_bytes'] / 2**30:.3f} GiB",
        f"Regenerable Python/test caches: {report['regenerable_python_test_cache']['allocated_bytes'] / 2**20:.2f} MiB",
        '', 'LIMITATIONS'] + report['limitations']
    (output / 'storage_inventory.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'Reports: {output}; nothing deleted.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT / 'EPSS_TPG/00_documentation/07_maintenance/storage_audit_v1')
    parser.add_argument('--verify-gib', type=float, default=2.0)
    args = parser.parse_args()
    if args.verify_gib < 0:
        parser.error('--verify-gib must be nonnegative')
    run(args.root, args.output, int(args.verify_gib * 2**30))
