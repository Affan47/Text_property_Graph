"""Publish hash-checked diagnosis, tests and the four-graph resume result."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for rel in ('01_tpg/10_extensions', '01_tpg/01_core', '02_prediction/01_package', '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT / rel))
from tpg_combined_v2 import common as c, graphs, cache
from tpg_combined_recovery_v1 import runner

BASE = ROOT / '05_results/08_fresh_studies/04_graph_execution_repair_v1'
OUTPUT = BASE / 'completion'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    if args.verify_only:
        c.check_files(ROOT, c.read(OUTPUT / 'source_lock.json')['files'])
        c.check_files(OUTPUT, c.read(OUTPUT / 'artifact_lock.json')['files'])
        print(f'Recovery completion report and preserved-file hashes verified: {OUTPUT}')
        return
    c.require(not OUTPUT.exists(), 'Completion report exists; use --verify-only')
    sources = {}

    def read(path):
        sources[str(path)] = c.sha(path)
        return c.read(path)

    diagnosis = read(BASE / 'diagnosis/REPORT.json')
    verification = read(BASE / 'verification/REPORT.json')
    for name in ('diagnosis', 'verification'):
        lock = read(BASE / name / 'artifact_lock.json')
        c.check_files(BASE / name, lock['files'])
        sources.update({str(BASE / name / path): sha for path, sha in lock['files'].items()})
    suite = read(BASE / 'tests_v1/suite.json')
    harness = read(BASE / 'tests_v1/REPORT.json')
    c.require(suite['success'], 'Test suite failed')
    runs = sorted((runner.RUNTIME / '01_reddit').glob('full_*/execution.json'))
    candidates = [p for p in runs if c.read(p)['status'] == 'bounded_check_complete' and c.read(p)['planned_jobs'] == 4]
    c.require(len(candidates) == 1, 'Expected exactly one four-graph integration run')
    run_dir = candidates[0].parent
    execution, before = read(candidates[0]), read(run_dir / 'before.json')
    directory = Path(before['directory'])
    c.check_files(directory, before['retained_files'])
    sources.update({str(directory / path): sha for path, sha in before['retained_files'].items()})
    sources[str(run_dir / 'events.jsonl')] = c.sha(run_dir / 'events.jsonl')
    events = [json.loads(line) for line in (run_dir / 'events.jsonl').read_text().splitlines()]
    keys = [row['value']['key'] for row in events if row['kind'] == 'result']
    c.require(len(keys) == len(set(keys)) == 4, 'Bounded result count differs')
    plan = c.load_plan(directory)
    contract = read(directory / 'recovery_identity.json')
    c.require(contract == runner.identity(plan), 'Recovery code changed')
    sources.update({str(ROOT / name): sha for name, sha in contract['pins'].items()})
    new = []
    for key in keys:
        receipt = read(directory / 'objects' / key / 'graph_receipt.json')
        graph = read(directory / 'objects' / key / 'graph.json.gz')
        c.check_files(directory, receipt['files'])
        c.require(receipt['recovery_identity_sha256'] == c.digest(contract), 'Receipt identity differs')
        c.require(receipt['boundary_repair'] == graph['metadata'].get('compatibility_extension'), 'Repair record differs')
        graphs.validate(plan['entries'][key], graph)
        tensors = graphs.tensors(graph)
        c.require(tensors['edge_features'].shape[1] == 88, 'Feature layout changed')
        new.append({'cve': plan['entries'][key]['cve'], 'key': key, 'repair': receipt['boundary_repair']})
    quarantine = read(run_dir / 'quarantine_complete.json')
    for row in quarantine['objects']:
        for name, record in row['files'].items():
            path = run_dir / 'quarantine' / row['key'] / name
            c.require(c.sha(path) == record['sha256'], 'Quarantined bytes changed')
            sources[str(path)] = c.sha(path)
    for dataset in c.DATASETS:
        smoke = c.directory(dataset, 'smoke')
        saved = read(smoke / 'graph_verification.json')
        c.check_files(smoke, saved['files'])
    progress = {}
    for dataset in c.DATASETS:
        path = c.directory(dataset, 'full')
        saved_plan = c.load_plan(path)
        read(path / 'plan_lock.json')
        receipts = list((path / 'objects').glob('*/graph_receipt.json'))
        progress[dataset] = {'planned': len(saved_plan['entries']), 'receipts_at_report': len(receipts),
                             'remaining_at_report': len(saved_plan['entries']) - len(receipts)}
    report = {'diagnosis': diagnosis, 'verification': verification, 'tests': suite, 'test_harness': harness,
              'bounded_resume': execution, 'new_graphs': new, 'quarantine': quarantine, 'progress': progress,
              'original_receipts_preserved': len(before['retained_files']) // 2,
              'original_plans_and_code_pins_unchanged': True, 'full_run_started': False,
              'embeddings': False, 'training': False, 'test_access': False}
    lines = ['GRAPH BUILD RECOVERY: CAUSE, REPAIR AND VERIFIED RESUME', '',
        '1. Triggering error',
        'The bounded replay tested 91 jobs near the stopped build. Only CVE-2024-42271 failed.',
        'An ENTITY contained "fput+0xba/0x268\\n" at offsets 1101:1117; its sentence ended at 1116.',
        'The extra newline caused the strict CONTAINS check to reject the graph. Validation was not weakened.', '',
        '2. Why it then appeared to run for over 13 hours',
        'The old stop_executor killed workers, then waited for ProcessPoolExecutor shutdown.',
        'A queue feeder was blocked writing tasks after its consumers had exited. The manager and parent waited for it.',
        f'The controlled reproduction reached its deadline: {verification["legacy_shutdown"]}.',
        'verification/legacy_shutdown.log contains the captured feeder, manager and shutdown thread stacks.', '',
        '3. Fixes',
        'Dedicated spawn workers use per-worker pipes, without background executor queue feeders.',
        'Original worker errors are printed and saved before bounded termination/join.',
        'Worker crashes, initialization errors, task timeouts, Ctrl+C and SIGTERM are handled explicitly.',
        'Graphs and receipts are prepared in staging, then published together by an atomic directory rename.',
        'Unfinished objects are preserved in quarantine. No completed graph is deleted or overwritten.',
        'The fallback trims only boundary whitespace when the resulting entity fits a sentence.',
        'Source text, TOKEN nodes, other CVEs, source masks and labels are unchanged.',
        'Successful original graph builds bypass the fallback and retain identical output.',
        'Original v2 pins and plans remain frozen. Recovery code has a separate recorded identity; repaired graphs have explicit metadata.', '',
        '4. Verification',
        f'{verification["cases"]} graph cases: {verification["unchanged_successful_graphs"]} previously successful outputs identical; '
        f'{verification["repaired_previously_failing_graphs"]} previously failing output repaired.',
        f'{suite["tests_run"]} graph/unit tests passed, including 12 focused recovery tests.',
        'The unrelated API test still timed out for both original and current code; it is not counted as passed.',
        'Both existing smoke caches verified: 14 Reddit graphs and 36 previous-dataset graphs.',
        f'Bounded actual resume: {execution["completed"]} new graphs; {execution["repaired"]} used the repair.',
        f'Original {report["original_receipts_preserved"]} completed graphs and receipts remain hash-identical.',
        'The zero-byte CVE-2022-2692 graph is preserved in quarantine. It was not the reproduced trigger.',
        f'Runtime evidence: {run_dir}', '', '5. Current counts, at report creation']
    lines += [f'{dataset}: {row["receipts_at_report"]}/{row["planned"]} graph receipts; {row["remaining_at_report"]} remaining.'
              for dataset, row in progress.items()]
    lines += ['', '6. What has not been done',
        'The full build has not been launched. No embeddings, training or test-cohort adaptation was performed.',
        'This is a narrow repair with bounded regression evidence, not a guarantee that every unseen text parses correctly.',
        'A different failure will stop visibly with its CVE/key and traceback; it will not be silently skipped.',
        'Use build_dataset_tpgs_recovery.py for graph building and graph verification. Commands are in the recovery USAGE.txt.']
    c.create(OUTPUT / 'REPORT.json', report)
    with (OUTPUT / 'REPORT.txt').open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    for path in (Path(__file__), ROOT / '01_tpg/02_examples/01_scripts/build_dataset_tpgs_recovery.py',
                 ROOT / '01_tpg/08_tests/test_combined_recovery.py',
                 ROOT / '01_tpg/10_extensions/tpg_combined_recovery_v1/USAGE.txt'):
        sources[str(path.resolve())] = c.sha(path)
    c.create(OUTPUT / 'source_lock.json', {'files': sources})
    c.create(OUTPUT / 'artifact_lock.json', {'files': {p.name: c.sha(p) for p in sorted(OUTPUT.iterdir()) if p.is_file()}})
    print(f'Recovery report saved: {OUTPUT}')


if __name__ == '__main__':
    main()
