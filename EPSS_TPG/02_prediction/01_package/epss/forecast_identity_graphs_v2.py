"""CD03 v2: explicit tokenizer-removed controls; reuse immutable v1 chunks."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

from . import forecast_identity_graphs as base
from . import forecast_control_coverage as coverage
from . import forecast_identity_graphs_parallel as parallel

SCHEMA = 'cd03_identity_multisource_graph_cache_v2'
OUTPUT = base.TPG / '05_workspace/cve_identity_multisource_v2'
SCRIPT = base.TPG / '02_examples/01_scripts/prepare_identity_multisource_graphs_v2.py'
TEST = base.TPG / '08_tests/test_identity_multisource_controls.py'
require, read, write, sha256, torch, tqdm = base.require, base.read, base.write, base.sha256, base.torch, base.tqdm
files = base.files


def context(scale):
    ctx = base.context(scale)
    identity = ctx['identity']
    identity['schema'] = SCHEMA
    identity['protocol'] = {**identity['protocol'], 'schema': SCHEMA, 'embedding_coverage_policy': coverage.POLICY,
        'allowed_control_codepoints': sorted(f'U+{ord(ch):04X}' for ch in coverage.REMOVED_CONTROLS)}
    identity['protocol']['limitations'] = identity['protocol']['limitations'] + [
        'Allowlisted tokenizer-removed control-only tokens retain zero embeddings and explicit masks; original text and offsets remain unchanged.']
    for path in (Path(__file__).resolve(), Path(coverage.__file__).resolve(), Path(parallel.__file__).resolve(), SCRIPT, TEST):
        identity['source_pins'][str(path)] = sha256(path)
    return ctx


class Builder(base.Builder):
    def build(self, entry):
        graph = base.annotate(self.pipeline.run(entry['text'], doc_id=entry['graph_id']), entry)
        saved = json.loads(base.old.shared.GraphSONExporter().export_string(graph))
        docs, _ = base.old.shared.preview.variants(saved)
        base.old.validate_documents(docs, entry)
        bundle, vocab = coverage.build_bundle(docs, self.encoder)
        return docs, bundle, vocab


def statistics(entry, docs, bundle, vocab):
    row = base.old.statistics_for(entry, docs, bundle, vocab)
    controls = coverage.missing_control_tokens(bundle, docs)
    row['unembedded_whitespace_tokens'] -= len(controls)
    require(row['unembedded_whitespace_tokens'] >= 0, 'Incorrect missing-token accounting')
    row['unembedded_control_tokens'] = controls
    return row


def check_entry(directory, entry, identity):
    receipt = read(directory / 'receipts' / (entry['key'] + '.json'))
    require(receipt['entry'] == {k: v for k, v in entry.items() if k != 'text'} and
            set(receipt['files']) == files(entry), 'Wrong chunk receipt')
    for name, digest in receipt['files'].items():
        require(sha256(directory / name) == digest, 'Changed chunk artifact: ' + name)
    docs = {mode: read(directory / 'tpg_json' / mode / (entry['stem'] + '.json')) for mode in base.old.shared.MODES}
    base.old.validate_documents(docs, entry)
    for graph in docs.values():
        require(graph['metadata']['source_chunk'] == {'parent_text_sha256': entry['parent_text_sha256'], **entry['chunk']},
                'Chunk offsets differ')
        for node in graph['vertices']:
            p = node['properties']
            available = bool(p['EXTRA'].get('span_available', True) and 0 <= p['CHAR_START'] < p['CHAR_END'] <= len(entry['text']))
            require(p['EXTRA']['source_span_available'] == available and
                    p['EXTRA']['source_document_char_start'] == (entry['chunk']['start'] + p['CHAR_START'] if available else None) and
                    p['EXTRA']['source_document_char_end'] == (entry['chunk']['start'] + p['CHAR_END'] if available else None),
                    'Global node offsets differ')
    bundle = torch.load(directory / 'tensors' / (entry['stem'] + '.pt'), weights_only=True, map_location='cpu')
    coverage.validate_bundle(bundle, docs, identity['type_vocabulary'], identity['embedding_dim'])
    row = statistics(entry, docs, bundle, identity['type_vocabulary'])
    policy = receipt.get('embedding_coverage_policy')
    if policy is None:
        # V1 receipts must also pass their original strict coverage policy.
        base.old.coverage.validate_bundle(bundle, docs, identity['type_vocabulary'], identity['embedding_dim'])
        expected = base.old.statistics_for(entry, docs, bundle, identity['type_vocabulary'])
    else:
        require(policy == coverage.POLICY, 'Unknown receipt embedding policy')
        expected = row
    require(receipt['statistics'] == expected, 'Chunk statistics differ')
    return bundle, row


def save_entry(directory, entry, identity, builder):
    docs, bundle, vocab = builder.build(entry)
    require(vocab == identity['type_vocabulary'], 'Vocabulary changed')
    for mode, graph in docs.items():
        write(directory / 'tpg_json' / mode / (entry['stem'] + '.json'), graph)
    path = directory / 'tensors' / (entry['stem'] + '.pt')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + '.tmp')
    torch.save(bundle, temporary)
    temporary.replace(path)
    receipt = directory / 'receipts' / (entry['key'] + '.json')
    temporary = Path(str(receipt) + '.tmp')
    write(temporary, {'entry': {k: v for k, v in entry.items() if k != 'text'},
        'embedding_coverage_policy': coverage.POLICY, 'statistics': statistics(entry, docs, bundle, vocab),
        'files': {name: sha256(directory / name) for name in sorted(files(entry))}})
    temporary.replace(receipt)


def summary(ctx, rows):
    result = base.summary(ctx, rows)
    result['schema'] = SCHEMA
    result['embedding_coverage_policy'] = coverage.POLICY
    cases = [{**{k: row[k] for k in ('key', 'cve', 'split', 'source_text_type')}, **case}
             for row in rows for case in row['unembedded_control_tokens']]
    result['unembedded_control_tokens'] = cases
    result['control_token_count'] = len(cases)
    result['cves_with_unembedded_controls'] = len({r['cve'] for r in cases})
    result['control_codepoint_counts'] = dict(sorted(Counter(c for r in cases for c in r['codepoints']).items()))
    return result


def guide(ctx, analysis):
    return base.guide(ctx, analysis) + (
        '\n## V2 Control-Character Coverage\n\n'
        f"Unembedded control-only nodes: {analysis['control_token_count']}; "
        f"affected CVEs: {analysis['cves_with_unembedded_controls']}.\n"
        'These source characters were removed by the frozen tokenizer, not silently dropped from the graph. '
        'Their embedding-present mask is zero; they are excluded from text-embedding pooling. '
        'Ordinary word tokens still require embeddings. See analysis.json / unembedded_control_tokens '
        'for exact CVEs, graph keys, node IDs, source spans and Unicode codepoints. '
        'V1 reused graph and tensor files retain their original bytes and receipts.\n')


expected = base.expected


def prepare(directory, ctx, resume=True, builder_factory=None):
    """Finalize already built/reused chunks; encoding is exclusively in workers."""
    require(resume and read(directory / 'run_identity.json') == ctx['identity'] and
            not (directory / 'cache_lock.json').exists(), 'Wrong/complete output')
    rows = []
    for entry in tqdm(ctx['entries'].values(), desc='Verify v2 graph coverage and build reports', unit='chunk'):
        _, row = check_entry(directory, entry, ctx['identity'])
        rows.append(row)
    for name, doc in ctx['documents'].items():
        require(read(directory / name) == doc, 'Source document changed')
    analysis = summary(ctx, rows)
    for cve, row in tqdm(ctx['index'].items(), desc='Write CVE indexes', unit='CVE'):
        write(directory / 'by_cve' / (cve + '.json'), row)
    write(directory / 'cve_index.json', ctx['index'])
    write(directory / 'protocol.json', ctx['identity']['protocol'])
    write(directory / 'analysis.json', analysis)
    for name in ('README.md', 'analysis.txt'):
        (directory / name).write_text(guide(ctx, analysis), encoding='utf-8')
    base.preparation.sources.check_pins(ctx['identity']['source_pins'])
    write(directory / 'run_status.json', {'status': 'complete'})
    require({str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()} == expected(ctx),
            'Unexpected cache files')
    write(directory / 'cache_lock.json', {'schema': SCHEMA, 'files': {
        name: sha256(directory / name) for name in tqdm(sorted(expected(ctx)), desc='Lock cache hashes', unit='file')}})


def verify(directory, ctx):
    lock = read(directory / 'cache_lock.json')
    require(lock['schema'] == SCHEMA and set(lock['files']) == expected(ctx), 'Wrong completed file set')
    require({str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()} == expected(ctx) | {'cache_lock.json'},
            'Unexpected/missing files')
    for name, digest in tqdm(lock['files'].items(), desc='Verify artifact hashes', unit='file'):
        require(sha256(directory / name) == digest, 'Changed artifact: ' + name)
    require(read(directory / 'run_identity.json') == ctx['identity'] and
            read(directory / 'protocol.json') == ctx['identity']['protocol'] and
            read(directory / 'cve_index.json') == ctx['index'] and
            read(directory / 'run_status.json') == {'status': 'complete'}, 'Changed cache contract')
    for name, doc in ctx['documents'].items():
        require(read(directory / name) == doc, 'Document/chunk coverage differs')
    for cve, row in ctx['index'].items():
        require(read(directory / 'by_cve' / (cve + '.json')) == row, 'CVE index differs')
    rows = [check_entry(directory, e, ctx['identity'])[1] for e in
            tqdm(ctx['entries'].values(), desc='Verify graphs and control masks', unit='chunk')]
    analysis = summary(ctx, rows)
    require(read(directory / 'analysis.json') == analysis and all(
        (directory / name).read_text() == guide(ctx, analysis) for name in ('README.md', 'analysis.txt')),
        'Reports do not reproduce')
    base.preparation.sources.check_pins(ctx['identity']['source_pins'])
    return analysis


def numeric_cve(directory, ctx, cve, enabled):
    require(cve in ctx['index'] and set(enabled) <= set(base.old.SOURCES), 'Unknown CVE/source')
    result = {source: [] for source in base.old.SOURCES}
    for doc in ctx['index'][cve]['documents']:
        source = doc['source_text_type']
        if source not in enabled:
            continue
        chunks = []
        for part in doc['chunks']:
            if part['has_graph']:
                bundle, _ = check_entry(directory, ctx['entries'][part['key']], ctx['identity'])
                item = base.old.shared.materialize(bundle, 'no_coref')
                chunks.append({k: item[k] for k in base.old.NUMERIC_FIELDS})
        result[source].append(chunks)
    return {'by_source': result, 'source_mask': [bool(result[s]) for s in base.old.SOURCES]}


def check_controls(ctx):
    entries = [e for e in ctx['entries'].values() if set(e['text']) & coverage.REMOVED_CONTROLS]
    require(0 < len(entries) <= 64, 'Control check is bounded to 64 chunks; review changed input scope')
    builder = Builder(ctx['identity'], False)
    coverage.check_tokenizer(builder.encoder.tokenizer)
    rows = []
    with tempfile.TemporaryDirectory(prefix='cd03_controls_') as tmp:
        for entry in tqdm(entries, desc='Check real control-character chunks', unit='chunk'):
            save_entry(Path(tmp), entry, ctx['identity'], builder)
            _, row = check_entry(Path(tmp), entry, ctx['identity'])
            rows.append(row)
    counts = Counter(ch for e in entries for ch in e['text'] if ch in coverage.REMOVED_CONTROLS)
    report = {'schema': SCHEMA, 'scope': 'bounded development control-character checks; no full cache or training',
              'source_pins': ctx['identity']['source_pins'], 'chunks_checked': len(rows),
              'source_control_counts': {f'U+{ord(k):04X}': v for k, v in sorted(counts.items())},
              'control_token_count': sum(len(r['unembedded_control_tokens']) for r in rows),
              'rows': rows, 'test_graphs': False}
    base.preparation.sources.check_pins(ctx['identity']['source_pins'])
    output = OUTPUT / 'control_character_check' / ctx['identity']['scale']
    write(output / 'analysis.json', report)
    (output / 'analysis.txt').write_text(
        f"CD03 v2 control-character check\nChunks checked: {len(rows)}\n"
        f"Explicitly unembedded control-only nodes: {report['control_token_count']}\n"
        'Every checked graph passed source, edge, embedding and offset validation.\n'
        'Original text is unchanged. Ordinary lexical tokens require embeddings.\n'
        'Temporary graphs removed. No full cache, training or test graphs.\n', encoding='utf-8')
    print(f'CD03 v2 bounded control check saved to {output}.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', choices=('smoke', 'full'), default='full')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--threads-per-worker', type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--plan-only', action='store_true')
    mode.add_argument('--check-controls', action='store_true')
    args = parser.parse_args()
    parallel.cpu_settings(args.workers, args.threads_per_worker)
    torch.set_num_threads(1)
    ctx = context(args.scale)
    directory = OUTPUT / args.scale
    if args.check_controls:
        check_controls(ctx)
        return
    if args.verify_only:
        verify(directory, ctx)
        print(f'CD03 v2 {args.scale} cache verified. No encoder rerun, training or test graphs.')
        return
    reuse = base.OUTPUT / args.scale
    original = base.context(args.scale)
    require(read(reuse / 'run_identity.json') == original['identity'], 'V1 reuse identity differs')
    if args.plan_only:
        print(f"CD03 v2: {len(ctx['entries'])} chunks; reuse verified receipts from {reuse}; output {directory}. No writes.")
        return
    base.verify(base.OUTPUT / 'smoke', base.context('smoke'))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report_dir = OUTPUT / 'parallel_execution' / (args.scale + '_' + stamp)
    report = {'status': 'running', 'source_pins': ctx['identity']['source_pins'],
              'workers': args.workers, 'threads_per_worker': args.threads_per_worker,
              'reuse_directory': str(reuse), 'original_identity_sha256': sha256(reuse / 'run_identity.json'),
              'embedding_coverage_policy': coverage.POLICY, 'training': False, 'test_graphs': False}
    write(report_dir / 'execution.json', report)
    try:
        parallel.prepare_parallel(directory, ctx, args.workers, args.threads_per_worker, args.resume,
                                  reuse, Builder, report, backend=sys.modules[__name__])
        report['status'] = 'complete'
        report['cache_lock_sha256'] = sha256(directory / 'cache_lock.json')
    except BaseException as exc:
        report['status'], report['error'] = 'interrupted_or_failed', repr(exc)
        raise
    finally:
        write(report_dir / 'execution.json', report)
        (report_dir / 'execution.txt').write_text(
            f"CD03 v2 execution: {report['status']}\nWorkers: {args.workers}\n"
            f"Original cache retained: {reuse}\nOutput: {directory}\n"
            'Completed v1 chunks reused without encoding. Controls masked explicitly; text unchanged.\n', encoding='utf-8')
    print(f'CD03 v2 {args.scale} cache saved to {directory}. No training or test graphs.')


if __name__ == '__main__':
    main()
