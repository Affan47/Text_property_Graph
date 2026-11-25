"""Publish the bounded performance gate without changing any earlier artifacts."""
import difflib
import hashlib
import math
from pathlib import Path

from .harness import (ROOT, OUTPUT, REGRESSION, REFERENCE, canonical, read, sha,
                      create, check, verify_reference, execution_lock, checked_plan, DATASETS)

PHASE = OUTPUT / 'phase1'

CHANGES = (
    ('01_tpg/01_core/tpg/span_index.py', 'class SpanIndex',
     'Insertion-order-preserving interval lookup; source offsets remain immutable during extraction.'),
    ('01_tpg/01_core/tpg/contracts.py', 'def covering_sentence',
     'Index sentence coverage, token provenance and assertion mention spans; preserve edge insertion order.'),
    ('01_tpg/01_core/tpg/frontends/security_frontend.py', 'def _find_sentence_node',
     'Use the same first-inserted covering sentence without scanning all sentences in each extractor.'),
    ('01_tpg/01_core/tpg/frontends/spacy_frontend.py', 'def _index_noun_chunks',
     'Enumerate paragraph noun chunks once, retaining the exact per-sentence filtering and order.'),
    ('01_tpg/01_core/tpg/schema/graph.py', 'def add_edge',
     'Bucket duplicate checks by endpoints/type, compare mutable properties live; count sentence tokens once.'),
    ('01_tpg/01_core/tpg/passes/security_relations.py', 'def emit_security_relations',
     'Group entities by sentence/type once; keep original source/target iteration order and relation rules.'),
    ('01_tpg/10_extensions/tpg_security_product_v5.py', 'def inspect',
     'Index same-span entities, versions and argument owners; use sets for endpoint and removal membership.'),
    ('01_tpg/10_extensions/tpg_security_product_v6.py', 'def inspect',
     'Index argument-owner edges once instead of rescanning every edge for each argument.'),
)


def gather():
    from tqdm import tqdm
    frozen = verify_reference()
    manifest = read(REGRESSION / 'manifest.json')
    check(sha(REGRESSION / 'items.json.gz') == manifest['items_sha256'], 'Regression inputs changed')
    rows = read(REGRESSION / 'items.json.gz')
    reports = {mode: read(path / 'REPORT.json') for mode, path in (
        ('reference', REGRESSION / 'reference'), ('replay', PHASE / 'replay_v1'))}
    original_plan_paths = set()
    for dataset in DATASETS:
        for scale in ('smoke', 'full'):
            directory, _ = checked_plan(dataset, scale)
            original_plan_paths.add(str(directory / 'plan_lock.json'))
            original_plan_paths.update(str(directory / p) for p in read(directory / 'plan_lock.json')['files'])
    execution_lock(PHASE / 'replay_v1', 'replay')
    execution_lock(PHASE / 'benchmark_v2', 'benchmark')
    execution_lock(PHASE / 'tests_v2', 'tests')
    existing_paths, failures = original_plan_paths, []
    for row, old, new in tqdm(zip(rows, reports['reference']['records'], reports['replay']['records']),
                             total=len(rows), desc='Verify all graph/tensor comparison records', unit='item'):
        input_sha = hashlib.sha256(canonical(row)).hexdigest()
        for mode, record in (('reference', old), ('replay', new)):
            directory = REGRESSION / 'reference' if mode == 'reference' else PHASE / 'replay_v1'
            check(record == read(directory / (row['id'] + '.json')), 'Summary/record mismatch')
            check(record['id'] == row['id'] and record['input_sha256'] == input_sha and record['equal'],
                  'Incorrect comparison identity or failure')
        for name in ('error', 'statistics', 'tensor_identity', 'graph_canonical_sha256'):
            check(old[name] == new[name], f'Comparison differs for {row["id"]}: {name}')
        if old['error']:
            check(row['kind'] == 'synthetic', 'Non-synthetic failure is not acceptable')
            failures.append({'id': row['id'], **old['error']})
        else:
            check(sha(old['graph_path']) == old['graph_file_sha256'], 'Golden bytes changed')
            existing_paths.add(old['graph_path'])
        if 'receipt_path' in row:
            check(sha(row['receipt_path']) == row['receipt_sha256'], 'Original receipt changed')
            existing_paths.add(row['receipt_path'])
    check(all(r['items'] == len(rows) == len(r['records']) == r['successes'] for r in reports.values()),
          'Not every regression item was verified')
    benchmark = read(PHASE / 'benchmark_v2/REPORT.json')
    for item in benchmark['rows']:
        size = item['characters']
        a, b = [read(PHASE / f'benchmark_v2/{v}_{size}.json') for v in ('reference', 'optimized')]
        check(a['input_sha256'] == b['input_sha256'] == sha(PHASE / f'benchmark_v2/input_{size}.json.gz'),
              'Benchmark input hash differs')
        check(a['graph_canonical_sha256'] == b['graph_canonical_sha256'], 'Benchmark graphs differ')
        check(a['statistics'] == b['statistics'], 'Benchmark statistics differ')
        check(item['old_seconds'] == a['seconds_profiled'] and item['new_seconds'] == b['seconds_profiled'],
              'Benchmark table differs')
    large = read(PHASE / 'benchmark_v2/optimized_large.json')
    check(benchmark['large_new_only'] == large, 'Large-case report differs')
    check(large['input_sha256'] == sha(PHASE / 'benchmark_v2/input_large.json.gz'), 'Large-case input differs')
    check(large['characters'] >= 400000, 'Large case not measured')
    tests = read(PHASE / 'tests_v2/REPORT.json')
    suite = read(PHASE / 'tests_v2/suite.json')
    check(suite['success'] and not suite['errors'] and not suite['failures'], 'Unit suite failed')
    changes = []
    for rel, signature, explanation in CHANGES:
        text = (ROOT / rel).read_text()
        line = next(i for i, s in enumerate(text.splitlines(), 1) if signature in s)
        changes.append({'file': rel, 'line': line, 'change': explanation, 'sha256': sha(ROOT / rel)})
    expected_changes = {r['file'] for r in changes}
    unexpected_changes = [p for p, old_sha in frozen['pins'].items()
                          if sha(ROOT / p) != old_sha and p not in expected_changes]
    check(not unexpected_changes, f'Unreviewed extractor changes: {unexpected_changes}')
    sizes = benchmark['rows']
    exponents = {name: math.log(sizes[-1][name] / sizes[0][name]) / math.log(8)
                 for name in ('old_seconds', 'new_seconds')}
    report = {
        'stage': 'Phase 1: output-preserving performance repair; stop for review',
        'changes': changes, 'reference_files_matching_both_plans': len(frozen['pins']),
        'regression_counts': manifest['counts'], 'comparison_items': len(rows),
        'successful_graphs_equal': len(rows) - len(failures), 'preserved_synthetic_failures': failures,
        'graph_differences': 0, 'tensor_differences': 0,
        'identity_scope': 'Full parsed GraphSON dictionary including order, IDs, metadata and statistics; tensors use dtype/shape/byte hashes and exact vocabularies. No embeddings generated.',
        'benchmarks': benchmark, 'measured_10k_to_80k_scaling_exponents': exponents,
        'unit_tests': {'harness': tests, 'suite': suite},
        'stale_locks_preserved': frozen['stale_locks'],
        'not_changed': ['Original cache payloads/receipts/previews/plans/pins', 'Raw datasets, labels and splits',
                        'Security extraction meaning, cross-document behavior, phrase pooling and relation encoding'],
        'scope_limits': ['Finite regression evidence, not proof for every possible input.',
                         'Single cProfile run per size; not a full-cache ETA or guaranteed linear complexity.',
                         'Existing API integration test has separate original/current timeout evidence; see tests.',
                         'benchmark_v1 was superseded after its large-case interrupt; tests_v1 used an incorrect API selector. Neither is acceptance evidence.',
                         'No full-cache build, training, embedding generation, test-cohort access or pin migration.'],
        'phase2_decisions_pending': [
            'Keep document-crossing reading-order edges with encoded concatenation markers, or drop them?',
            'Encode discourse distinctions in edge features, or introduce separate labels?',
            'Keep only typed SEC_* edges while retaining non-SEC ENTITY_REL edges?',
            'Replace bare prepositions with their object using ARGM-PREP plus the preposition lemma?'],
        'next_action': 'Researcher reviews Phase 1; do not resume original pinned caches or start Phase 2 automatically.'}
    return report, existing_paths


def publish():
    check(not (PHASE / 'REPORT.json').exists() and not (PHASE / 'artifact_lock.json').exists(),
          'Final report already exists; use --stage verify')
    report, originals = gather()
    create(PHASE / 'REPORT.json', report)
    lines = ['TPG builder performance repair: Phase 1', '',
             'The old extractor was frozen before editing. All 45 original plan-pinned files matched both datasets.',
             'The same graph now takes less repeated scanning. This does not repair graph meaning or demonstrate prediction improvements.', '',
             '1. Exact-output gate',
             f'{report["comparison_items"]} cases: {report["successful_graphs_equal"]} complete graphs match; '
             f'{len(report["preserved_synthetic_failures"])} original synthetic failures remain identical.',
             'Full graph dictionaries, ordering, IDs, metadata, statistics and graph-tensor hashes match. No embeddings were generated.',
             str(report['regression_counts']), '', '2. Timing (one profiled CPU build per case)',
             'Characters | old seconds | new seconds | speedup | canonical graph identical']
    for row in report['benchmarks']['rows']:
        lines.append(f'{row["characters"]} | {row["old_seconds"]:.3f} | {row["new_seconds"]:.3f} | {row["speedup"]:.2f}x | yes')
    large = report['benchmarks']['large_new_only']
    lines += [f'Large case: {large["cve"]}, {large["characters"]} characters, {large["seconds_profiled"]:.3f} seconds.',
              f'Peak worker RSS during build: {large["peak_rss_kib"] / 1024**2:.3f} GiB; '
              f'including canonical hashing: {large["peak_rss_after_hash_kib"] / 1024**2:.3f} GiB.',
              'Measured 10k-to-80k scaling exponents: ' + str(report['measured_10k_to_80k_scaling_exponents']),
              'Do not extrapolate these profiled single-case times to an 88-core full-build ETA.', '', '3. Changes']
    lines += [f'{r["file"]}:{r["line"]}: {r["change"]}' for r in report['changes']]
    lines += ['', 'Additional checks: version-edge replacement and same-occurrence filtering rebuild once through graph APIs.',
              'Their duplicate-edge checks benefit from endpoint/type indexing; copy costs remain. No extraction rules changed.',
              'A full edge-identity dictionary was deliberately avoided because EdgeProperties can be mutated after insertion.',
              '', '4. Tests', str(report['unit_tests']), '', '5. Preserved original synthetic failures']
    lines += [str(row) for row in report['preserved_synthetic_failures']]
    lines += ['', '6. Limits and preserved artifacts'] + report['scope_limits'] + report['not_changed']
    lines += [str(row) for row in report['stale_locks_preserved']]
    lines += ['', '7. Stop for Phase 1 review'] + report['phase2_decisions_pending'] + [report['next_action']]
    with (PHASE / 'REPORT.txt').open('x') as f:
        f.write('\n'.join(lines) + '\n')
    diff = []
    for row in report['changes']:
        old, new = REFERENCE / row['file'], ROOT / row['file']
        diff.extend(difflib.unified_diff(old.read_text().splitlines(keepends=True) if old.exists() else [],
                                        new.read_text().splitlines(keepends=True),
                                        fromfile='reference/' + row['file'], tofile='optimized/' + row['file']))
    with (PHASE / 'code_diff.txt').open('x') as f:
        f.writelines(diff)
    paths = {Path(p) for p in originals}
    for directory in (REFERENCE, REFERENCE.parent / 'reference_import_support_v1',
                      REFERENCE.parent / 'reference_extension_support_v1', REGRESSION,
                      PHASE / 'replay_v1', PHASE / 'benchmark_v2', PHASE / 'tests_v2', Path(__file__).parent):
        paths.update(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.update(ROOT / r['file'] for r in report['changes'])
    paths.update([PHASE / 'REPORT.json', PHASE / 'REPORT.txt', PHASE / 'code_diff.txt',
                  ROOT / '01_tpg/08_tests/test_combined_performance_equivalence.py',
                  ROOT / '01_tpg/02_examples/01_scripts/repair_combined_tpg.py'])
    create(PHASE / 'artifact_lock.json', {'files': {str(p): sha(p) for p in sorted(paths)},
                                         'purpose': 'Phase 1 reports, reference and goldens; no old pins migrated'})
    verify()
    print(f'Phase 1 report published: {PHASE / "REPORT.txt"}')


def verify():
    from tqdm import tqdm
    lock = read(PHASE / 'artifact_lock.json')
    for name, expected in tqdm(lock['files'].items(), desc='Verify preserved inputs and repair evidence', unit='file'):
        check(sha(name) == expected, f'Artifact changed: {name}')
    report, _ = gather()
    check(read(PHASE / 'REPORT.json') == report, 'Final report does not reproduce')
    print('Phase 1 hashes, graph/tensor comparisons and reports verified. No graph rebuild, embeddings, training or test access.')
