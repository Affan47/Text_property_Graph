#!/usr/bin/env python3
"""Check the numbered layout without loading models or altering experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / '.tpg-project-root').is_file())
DOCS = ROOT / '00_documentation'
MAINTENANCE = DOCS / '07_maintenance'


def local_links(text):
    for value in re.findall(r'\]\(([^)]+)\)', text):
        value = value.strip().strip('<>')
        if value.startswith(('#', 'http:', 'https:', 'mailto:', 'app:')):
            continue
        yield unquote(value.split('#', 1)[0])


def normalized(base, target):
    return Path(os.path.normpath(os.path.join(str(base), target)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-report', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((MAINTENANCE / 'layout_manifest.json').read_text())
    originals_path = MAINTENANCE / 'documentation_before_reorganization.json'
    originals = json.loads(originals_path.read_text()) if originals_path.exists() else {}
    known = {str(ROOT / item['old']) for item in manifest['inventory']}
    known.update(str(parent) for path in list(known) for parent in Path(path).parents)

    def relocated(path):
        try:
            relative = str(path.relative_to(ROOT))
        except ValueError:
            return path
        if relative in manifest['files']:
            return ROOT / manifest['files'][relative]
        for old in sorted(manifest['directories'], key=len, reverse=True):
            if relative == old or relative.startswith(old + '/'):
                return ROOT / (manifest['directories'][old] + relative[len(old):])
        return path

    errors = []
    checked = 0
    for item in manifest['inventory']:
        p = ROOT / item['new']
        if not p.is_file():
            errors.append(f"Missing migrated file: {item['new']}")
            continue
        editable = p.suffix in {'.py', '.sh', '.md'} or p.name in {'.gitignore', 'Dockerfile'}
        if editable:
            continue
        checked += 1
        if p.stat().st_size != item['size']:
            errors.append(f"Changed artifact size: {item['new']}")
        elif 'sha256' in item and hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
            errors.append(f"Changed artifact content: {item['new']}")

    broken_aliases = []
    for base, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__'}]
        for name in dirs + files:
            p = Path(base) / name
            if p.is_symlink() and not p.exists():
                broken_aliases.append(str(p.relative_to(ROOT)))
    errors.extend('Broken alias: ' + p for p in broken_aliases)

    old_by_new = {item['new']: item['old'] for item in manifest['inventory']}
    historical = []
    checked_links = 0
    for p in DOCS.rglob('*.md'):
        if p.is_symlink() or p.name == '03_VERIFICATION_REPORT.md':
            continue
        relative = str(p.relative_to(ROOT))
        old = old_by_new.get(relative)
        old_missing_targets = set()
        if old in originals:
            for target in local_links(originals[old]):
                original_target = normalized((ROOT / old).parent, target)
                if str(original_target) not in known:
                    old_missing_targets.add(str(relocated(original_target)))
        missing = []
        for target in local_links(p.read_text()):
            checked_links += 1
            if not normalized(p.parent, target).exists():
                missing.append(target)
        if missing:
            new_missing = [target for target in missing
                           if str(normalized(p.parent, target)) not in old_missing_targets]
            if not new_missing:
                historical.append({'document': relative, 'links': missing,
                                   'preexisting_missing_count': len(old_missing_targets)})
            else:
                errors.extend(f'Unresolved link: {relative}: {link}' for link in new_missing)

    lines = [
        '# Layout Verification', '',
        f'- Original files checked for presence: {len(manifest["inventory"])}.',
        f'- Non-code/document artifacts checked for preservation: {checked}.',
        f'- Documentation links inspected: {checked_links}.',
        f'- Broken compatibility links: {len(broken_aliases)}.',
        f'- Migration errors: {len(errors)}.', '',
        'This checks files and paths. It does not retrain models or validate scientific claims.', '',
        '## Existing Historical Link Gaps', '',
        'The following documents already contained unavailable references before the move. '
        'Each unresolved target also appears in the pre-migration document after path mapping. '
        'Missing historical artifacts are not replaced with invented results.', '',
    ]
    for entry in historical:
        lines.append(f"- `{entry['document']}`: {len(entry['links'])} unresolved references.")
    if not historical:
        lines.append('None found.')
    if errors:
        lines.extend(['', '## Errors', ''] + ['- ' + error for error in errors])
    if args.write_report:
        (MAINTENANCE / '03_VERIFICATION_REPORT.md').write_text('\n'.join(lines) + '\n')
        (MAINTENANCE / 'historical_link_gaps.json').write_text(json.dumps(historical, indent=2) + '\n')
    print('\n'.join(lines[:10]))
    for error in errors:
        print(error)
    print(f'Historical documents with unresolved references: {len(historical)}')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
