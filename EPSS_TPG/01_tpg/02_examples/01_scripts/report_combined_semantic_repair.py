"""Publish checked Phase 2 evidence; never build graphs or load an encoder."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for rel in ('01_tpg/10_extensions', '01_tpg/01_core', '02_prediction/01_package', '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT / rel))
from tpg_combined_v2 import common as c

OUTPUT = c.REPORTS / 'completion_v1'


def location(filename, name):
    path = ROOT / '01_tpg/10_extensions/tpg_combined_v2' / filename
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    return f'{path.relative_to(ROOT)}:{node.lineno}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    sources = {}

    def read(path):
        path = Path(path)
        sources[str(path)] = c.sha(path)
        return c.read(path)

    audit_dir = c.REPORTS / 'regression_v2'
    audit = read(audit_dir / 'REPORT.json')
    c.check_files(audit_dir, read(audit_dir / 'artifact_lock.json')['files'])
    c.require(audit['code'] == c.pins() and not any(audit['after'].values()), 'Audit is not current/passing')
    tests = read(c.REPORTS / 'tests_v2/suite.json')
    test_runs = read(c.REPORTS / 'tests_v2/REPORT.json')
    c.require(tests['success'] and not tests['errors'] and not tests['failures'], 'Unit suite failed')
    for path in sorted((c.REPORTS / 'tests_v2').glob('*.log')):
        sources[str(path)] = c.sha(path)
    retired = read(c.REPORTS / 'retirement_v1/applied.json')
    retirement_plan = read(c.REPORTS / 'retirement_v1/plan.json')
    c.require(retired['plan_sha256'] == c.sha(c.REPORTS / 'retirement_v1/plan.json'), 'Retirement plan differs')
    c.require(all(not Path(r['path']).exists() for r in retirement_plan['files']), 'Retired payload returned')
    phase1 = read(c.REPORTS.parent / 'phase1/artifact_lock.json')
    c.check_files(ROOT, phase1['files'])
    datasets, examples = {}, {}
    for dataset in c.DATASETS:
        smoke_dir, full_dir = c.directory(dataset, 'smoke'), c.directory(dataset, 'full')
        smoke, full = c.load_plan(smoke_dir), c.load_plan(full_dir)
        for directory in (smoke_dir, full_dir):
            read(directory / 'plan_lock.json')
        checked = read(smoke_dir / 'graph_verification.json')
        c.check_files(smoke_dir, checked['files'])
        datasets[dataset] = {'roles': dict(Counter(r['split'] for r in full['index'].values())),
                             'variants': full['variants'], 'full_objects_planned': len(full['entries']),
                             'smoke': checked['statistics'], 'full_root': str(full_dir)}
        candidates = [(key, row) for key, row in smoke['entries'].items() if len(row['spans']) > 1]
        key, row = min(candidates, key=lambda pair: (len(pair[1]['text']), pair[0]))
        graph = read(smoke_dir / 'objects' / key / 'graph.json.gz')
        examples[f'{dataset}_{row["cve"]}.json'] = graph
        datasets[dataset]['inspection_example'] = {'cve': row['cve'], 'object_key': key,
            'documents': [{'source': s['source'], 'id': s['document_id']} for s in row['spans']],
            'file': f'{dataset}_{row["cve"]}.json'}
    timings = []
    items = read(c.REPORTS.parent / 'regression_set/items.json.gz')
    for row in items:
        new = read(audit_dir / 'records' / (row['id'] + '.json'))
        if row['kind'] not in ('fresh', 'smoke', 'synthetic'):
            continue
        old = read(c.REPORTS.parent / 'regression_set/reference' / (row['id'] + '.json'))
        timings.append({'case': row['id'], 'characters': len(row['entry']['text']),
                        'old_reference_seconds': old['seconds'], 'new_build_seconds': new['seconds']})
    changes = [(location('graphs.py', 'Builder'), 'Extract each source document separately, assemble one graph, mark concatenation and target CVEs.'),
               (location('security_v7.py', 'Frontend'), 'Match security keywords on original Unicode text without lowercase offset shifts.'),
               (location('security_v7.py', 'correct'), 'Resolve exact-span conflicts, preserve phrase members, replace prep-only arguments, align whitespace and cross-sentence entities, remove SEC twins.'),
               (location('security_v7.py', 'TopicPass'), 'Retain up to five valid word topics per source document.'),
               (location('graphs.py', 'relation_features'), 'Expose relation flags and preposition lemma hashing in 88 edge features.'),
               (location('validation.py', 'validate'), 'Reject incorrect spans, document boundaries, evidence, memberships and target markers.'),
               (location('embeddings.py', 'pool_spans'), 'Pool only member subwords for phrases, deduplicating overlapping subwords; tested with synthetic vectors.'),
               (location('embeddings.py', 'model_inputs'), 'Expose target-CVE flag and document index to the future model; retain subtype IDs.'),
               (location('cache.py', 'build'), 'Bound CPU workers, verify receipts and code identity, refuse silent partial-cache overwrite.'),
               (location('retire.py', 'safe_path'), 'Delete only hash-checked, approved v1 object payloads; preserve original goldens.')]
    notes = [
        'Both datasets retain six variants: D, M, U, DM, DU, DMU. Missing views remain masked; no text is invented.',
        'Four defaults: marked cross-document reading order; explicit edge flags; typed SEC only; object-based ARGM-PREP.',
        'The 523-case before/after counts are node/edge occurrences in a bounded sample, not distinct CVEs or whole-dataset totals.',
        'Some categories are missing annotations rather than false extracted facts. Several categories can describe the same node.',
        'The first semantic audit stopped at CVE-2022-28859: trailing whitespace extended a clause beyond its sentence. That attempt is preserved.',
        'The fix removes boundary whitespace from phrase membership, not from source text or TOKEN nodes. A regression test covers it.',
        'Early smoke plan probes are preserved separately; accepted caches use the final code identity.',
        'Old rules/core files were not changed in Phase 2; v7 lives inside the new v2 package.',
        'Tensor structure and phrase-pooling math were checked without running SecureBERT. Real embedding coverage still needs the separate stage.',
        'API integration timed out at 45 seconds with both original and current implementations. It is not counted as a passing test.',
        'Timing rows compare previously recorded old builds with new parallel builds. Old first-use initialization may be included; these are descriptive, not controlled speedup estimates.',
        f'The audit records {audit["build_seconds_sum"]:.3f} seconds of summed worker build time, not wall-clock time or a full-build ETA.',
        'Security edges remain heuristic candidates. No improvement in predictions is claimed.',
        'Open-vocabulary label/preposition hash collisions remain possible. Document boundaries do not isolate transformer attention within a 1024-subword window.',
        'CARDINAL+VERSION and ORG+SOFTWARE are retained as compatible views. Cross-sentence entities keep token evidence but no sentence-bound security assertions.',
        'Reddit validation still has no accepted webpage summaries. These repairs do not resolve that missing-source problem.',
        'No full-cache generation, real embeddings, training or test-cohort access was performed in this phase.'
    ]
    report = {'schema': 'combined_semantic_repair_completion_v1', 'audit': audit, 'tests': tests,
              'test_runs': test_runs, 'datasets': datasets, 'retirement': retired, 'changes': changes,
              'timings': timings, 'notes': notes}
    lines = ['PHASE 2 COMPLETE: DOCUMENT-BOUNDED COMBINED-TEXT TPG', '',
             '1. What was implemented', *[f'{path}: {message}' for path, message in changes], '',
             '2. Verified before/after occurrences (523 regression graphs)', 'Check | old count | new count']
    lines += [f'{k} | {audit["before"][k]} | {audit["after"][k]}' for k in sorted(audit['before'])]
    lines += ['', f'Old totals: {audit["old_totals"]}', f'New totals: {audit["new_totals"]}', '',
              '3. Development plans and completed smoke graphs']
    for dataset, record in datasets.items():
        lines += [f'{dataset}: roles={record["roles"]}; planned full objects={record["full_objects_planned"]}',
                  f'Smoke: {record["smoke"]}', f'New root: {record["full_root"]}',
                  f'Real readable JSON example: {record["inspection_example"]["file"]}']
    lines += ['', f'4. Tests: {tests["tests_run"]} passed; 16 are focused v2 tests.',
              'The separate API integration check timed out for both reference and current code.', '',
              '5. Authorized retirement', json.dumps(retired, sort_keys=True), '', '6. Limits and interpretation', *notes, '',
              '7. Build commands: see 01_tpg/10_extensions/tpg_combined_v2/USAGE.txt',
              'Use build_dataset_tpgs_v2.py, not the retired v1 cache commands.',
              'Graph stage first, verification next, frozen embeddings separately, then verify --require-embeddings.']
    fresh_ids = {row['id'] for row in items if row['kind'] == 'fresh'}
    lines += ['', '8. Descriptive build timings for the 12 fresh-reference cases',
              'Case | characters | old reference seconds | new v2 seconds',
              'Different runs/concurrency and possible old initialization: do not interpret as a controlled speedup test.']
    lines += [f'{r["case"]} | {r["characters"]} | {r["old_reference_seconds"]:.3f} | {r["new_build_seconds"]:.3f}'
              for r in timings if r['case'] in fresh_ids]
    text = '\n'.join(lines) + '\n'
    sources[str(Path(__file__).resolve())] = c.sha(__file__)
    for path in (ROOT / '01_tpg/08_tests/test_combined_semantics_v2.py',
                 ROOT / '01_tpg/02_examples/01_scripts/build_dataset_tpgs_v2.py',
                 ROOT / '01_tpg/02_examples/01_scripts/repair_combined_semantics.py',
                 ROOT / '01_tpg/10_extensions/tpg_combined_v2/USAGE.txt',
                 ROOT / '00_documentation/02_tpg/11_COMBINED_TEXT_FRESH_START.md'):
        sources[str(path)] = c.sha(path)
    if args.verify_only:
        c.check_files(ROOT, c.read(OUTPUT / 'source_lock.json')['files'])
        c.check_files(OUTPUT, c.read(OUTPUT / 'artifact_lock.json')['files'])
        c.require(c.read(OUTPUT / 'REPORT.json') == json.loads(json.dumps(report)), 'Report differs')
        c.require((OUTPUT / 'REPORT.txt').read_text() == text, 'Text report differs')
        for name, graph in examples.items():
            c.require(c.read(OUTPUT / name) == graph, 'Inspection example differs')
        print(f'Phase 2 completion report verified: {OUTPUT}')
        return
    c.require(not OUTPUT.exists(), 'Report exists; use --verify-only')
    c.create(OUTPUT / 'REPORT.json', report)
    with (OUTPUT / 'REPORT.txt').open('x') as stream:
        stream.write(text)
    for name, graph in examples.items():
        with (OUTPUT / name).open('x') as stream:
            json.dump(graph, stream, indent=2, ensure_ascii=True, allow_nan=False)
            stream.write('\n')
    c.create(OUTPUT / 'source_lock.json', {'files': sources})
    files = sorted(OUTPUT.iterdir())
    c.create(OUTPUT / 'artifact_lock.json', {'files': {p.name: c.sha(p) for p in files}})
    print(f'Phase 2 completion report saved: {OUTPUT}')


if __name__ == '__main__':
    main()
