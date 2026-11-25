"""MS05: Day-30 source routing over immutable reused and newly built TPGs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import multiprocessing as mp
from pathlib import Path

from . import forecast_day60_multisource as preparation
from . import forecast_identity_graphs_v2 as frozen

base, parallel = frozen.base, frozen.parallel
read, write, require, sha256, tqdm = frozen.read, frozen.write, frozen.require, frozen.sha256, frozen.tqdm
ROOT = preparation.ROOT
SCHEMA = 'ms05_day30_multisource_cache_v1'
OUTPUT = ROOT/'01_tpg/05_workspace/day30_multisource_v1'
REUSE = frozen.OUTPUT/'full'
SCRIPT = ROOT/'01_tpg/02_examples/01_scripts/prepare_day30_multisource_graphs.py'
TEST = ROOT/'02_prediction/03_analysis/01_revision_dataset/test_forecast_day60_pipeline.py'


def safe_path(root, name):
    path = (root/name).resolve()
    require(root.resolve() in path.parents, 'Unsafe cache artifact path')
    return path


def receipt_matches(receipt, entry):
    require(receipt['entry'] == {k: v for k, v in entry.items() if k != 'text'}
            and set(receipt['files']) == base.files(entry), 'Reuse text, source, role or chunk identity differs')


def context(scale):
    require(scale in ('smoke', 'full'), 'Unknown cache scale')
    data = preparation.OUTPUT/scale
    pins = preparation.check_lock(data, 'preparation_lock.json')
    lock = read(data/'preparation_lock.json')
    require(lock['schema'] == preparation.SCHEMA, 'Wrong preparation schema')
    a = {name: read(data/name) for name in lock['files'] if name.endswith('.json')}
    preparation.validate(a)
    source = a['run_identity.json']
    require(source['schema'] == preparation.SCHEMA and source['scale'] == scale, 'Wrong MS04 preparation')
    preparation.sources.check_pins(source['source_pins'])
    pins.update(source['source_pins'])
    old_lock = read(REUSE/'cache_lock.json')
    require(old_lock['schema'] == frozen.SCHEMA, 'Wrong reuse cache version')
    pins[str(REUSE/'cache_lock.json')] = sha256(REUSE/'cache_lock.json')
    for name in ('run_identity.json', 'protocol.json', 'analysis.json', 'run_status.json'):
        require(sha256(REUSE/name) == old_lock['files'][name], 'Changed reuse resource metadata')
        pins[str(REUSE/name)] = old_lock['files'][name]
    old = read(REUSE/'run_identity.json')
    require(old['schema'] == frozen.SCHEMA and old['scale'] == 'full'
            and read(REUSE/'run_status.json') == {'status': 'complete'}
            and old['protocol'] == read(REUSE/'protocol.json')
            and old['protocol']['extractor'] == base.old.extension.VERSION
            and old['protocol']['embedding_coverage_policy'] == frozen.coverage.POLICY,
            'Incomplete or incompatible frozen cache')
    preparation.sources.check_pins(old['source_pins'])
    pins.update(old['source_pins'])
    for path in (Path(__file__).resolve(), SCRIPT, TEST, Path(frozen.__file__).resolve(),
                 Path(parallel.__file__).resolve(), Path(preparation.__file__).resolve()):
        pins[str(path)] = sha256(path)
    p = {**deepcopy(old['protocol']), 'schema': SCHEMA, 'anchor_day': 30,
         'prepared_schema': preparation.SCHEMA, 'reuse_policy': 'Exact receipt identity and locked artifact hashes; no shared-file writes',
         'limitations': a['protocol.json']['limitations'] + [
             'Reused graphs/tensors stay in the old cache; do not delete or modify it.',
             'Independent 4096-character chunks do not add cross-chunk relations.']}
    ctx = base.plan(a['source_inputs.json']['rows'], source['ids'], p['max_chunk_characters'])
    old_stats = {r['key']: r for r in read(REUSE/'analysis.json')['chunks']}
    reuse = {}
    for key, entry in tqdm(ctx['entries'].items(), desc='Match frozen chunk receipts', unit='chunk'):
        name = 'receipts/'+key+'.json'
        if name not in old_lock['files']:
            continue
        require(sha256(REUSE/name) == old_lock['files'][name], 'Changed reuse receipt')
        receipt = read(REUSE/name)
        receipt_matches(receipt, entry)
        require(key in old_stats, 'Missing frozen chunk statistics')
        row = old_stats[key]
        comparable = deepcopy(row)
        if receipt.get('embedding_coverage_policy') is None:
            require(not comparable.pop('unembedded_control_tokens'), 'Legacy receipt unexpectedly has control tokens')
        else:
            require(receipt['embedding_coverage_policy'] == frozen.coverage.POLICY, 'Unknown reuse policy')
        require(receipt['statistics'] == comparable, 'Reuse statistics disagree with locked receipt')
        hashes = {name: old_lock['files'][name], **receipt['files']}
        require(all(old_lock['files'].get(n) == h for n, h in hashes.items()), 'Receipt is outside original cache lock')
        reuse[key] = {'root': str(REUSE), 'files': hashes, 'statistics': row}
        pins.update({str(safe_path(REUSE, n)): h for n, h in hashes.items()})
    identity = {'schema': SCHEMA, 'scale': scale, 'protocol': p, 'ids': source['ids'], 'source_pins': pins,
                **{k: old[k] for k in ('encoder', 'embedding_dim', 'type_vocabulary', 'spacy_pipeline_sha256')},
                'reused_keys': sorted(reuse), 'chunk_count': len(ctx['entries']),
                'document_count': len(ctx['documents']), 'training': False, 'test_graphs': False}
    ctx.update(identity=identity, reuse=reuse)
    # Local indexes retain the new source masks; absolute locations make reuse inspectable.
    for row in ctx['index'].values():
        for doc in row['documents']:
            for part in doc['chunks']:
                if part['has_graph']:
                    root = REUSE if part['key'] in reuse else OUTPUT/scale
                    part.update(storage_root=str(root), graph_absolute=str(root/part['graph']),
                                tensor_absolute=str(root/part['tensor']))
    preparation.sources.check_pins(pins)
    return ctx


def locations(directory, ctx, deep=False):
    result = {}
    for key, entry in tqdm(ctx['entries'].items(), desc='Verify chunk locations and coverage', unit='chunk'):
        if key in ctx['reuse']:
            location = ctx['reuse'][key]
            root = Path(location['root'])
            for name, h in location['files'].items():
                require(sha256(safe_path(root, name)) == h, 'Changed reused artifact')
            if deep:
                require(frozen.check_entry(root, entry, ctx['identity'])[1] == location['statistics'], 'Reused graph no longer reproduces')
        else:
            _, row = frozen.check_entry(directory, entry, ctx['identity'])
            name = 'receipts/'+key+'.json'
            receipt = read(directory/name)
            location = {'root': str(directory), 'files': {name: sha256(directory/name), **receipt['files']}, 'statistics': row}
        result[key] = location
    return result


def analysis(ctx, loc):
    a = base.summary(ctx, [loc[k]['statistics'] for k in ctx['entries']])
    a.update(schema=SCHEMA, anchor_day=30, reused_chunks=len(ctx['reuse']),
             newly_built_chunks=len(ctx['entries'])-len(ctx['reuse']),
             cohort={s: len(ids) for s, ids in ctx['identity']['ids'].items()})
    return a


def report(ctx, a):
    return '\n'.join([
        'MS05: Day-30-aware TPG cache', '', f"CVEs: {a['cohort']}",
        f"Reused verified chunks: {a['reused_chunks']}; new chunks: {a['newly_built_chunks']}",
        'Day-30 source masks come from MS04, never from the Day-60 experiment.',
        'No EPSS values, target labels or test text enter graph construction.',
        'Reused files are referenced, not copied, edited or re-encoded. Keep the original cache.',
        'cve_index.json and by_cve/*.json contain absolute graph/tensor locations.',
        'documents/ contains original source text and chunk boundaries. chunk_locations.json records hashes and statistics.',
        'New graphs live in tpg_json/ and tensors/; no_coref is the training edge configuration.',
        'Fast verification rechecks every referenced byte against the original locked cache.',
        '--deep-verify additionally reloads every reused graph/tensor; it is slower and never reruns the encoder.', '',
        'Split | Source | CVEs with graphs | Documents | Chunks | Nodes | No-COREF edges',
        *[' | '.join(str(r[k]) for k in ('split','source','cves_with_graphs','documents','chunks','nodes','edges_no_coref')) for r in a['coverage']],
        '', 'No training or test scoring.', *ctx['identity']['protocol']['limitations'], ''])


def expected(ctx):
    names = {'run_identity.json', 'protocol.json', 'cve_index.json', 'chunk_locations.json',
             'analysis.json', 'analysis.txt', 'run_status.json'} | set(ctx['documents'])
    names.update('by_cve/'+c+'.json' for c in ctx['index'])
    for key, e in ctx['entries'].items():
        if key not in ctx['reuse']:
            names.update(base.files(e))
            names.add('receipts/'+key+'.json')
    return names


def verify(directory, ctx, deep=False):
    lock = read(directory/'cache_lock.json')
    require(lock['schema'] == SCHEMA and set(lock['files']) == expected(ctx), 'Wrong Day-30 cache lock')
    require({str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()} == expected(ctx) | {'cache_lock.json'}, 'Unexpected cache files')
    for name, h in tqdm(lock['files'].items(), desc='Verify Day-30 cache files', unit='file'):
        require(sha256(safe_path(directory, name)) == h, 'Changed Day-30 cache artifact')
    require(read(directory/'run_identity.json') == ctx['identity'] and
            read(directory/'cve_index.json') == ctx['index'] and
            read(directory/'protocol.json') == ctx['identity']['protocol'] and
            read(directory/'run_status.json') == {'status': 'complete'}, 'Changed Day-30 cache contract')
    for name, doc in ctx['documents'].items():
        require(read(directory/name) == doc, 'Changed source document')
    for c, row in ctx['index'].items():
        require(read(directory/'by_cve'/(c+'.json')) == row, 'Changed CVE index')
    loc = locations(directory, ctx, deep)
    a = analysis(ctx, loc)
    require(read(directory/'chunk_locations.json') == loc and read(directory/'analysis.json') == a
            and (directory/'analysis.txt').read_text() == report(ctx, a), 'Reports do not reproduce')
    preparation.sources.check_pins(ctx['identity']['source_pins'])
    return loc


def prepare(directory, ctx, workers, resume=False):
    with parallel.writer_lock(directory):
        if resume:
            require(directory.exists() and not (directory/'cache_lock.json').exists()
                    and read(directory/'run_identity.json') == ctx['identity'], 'Wrong or completed resume')
        else:
            require(not directory.exists(), 'Cache exists; use --resume or --verify-only')
            directory.mkdir(parents=True)
            write(directory/'run_identity.json', ctx['identity'])
        write(directory/'run_status.json', {'status': 'running'})
        for name, doc in tqdm(ctx['documents'].items(), desc='Save Day-30 source documents', unit='document'):
            if (directory/name).exists():
                require(read(directory/name) == doc, 'Changed partial source document')
            else:
                write(directory/name, doc)
        pending = []
        for key, e in ctx['entries'].items():
            if key in ctx['reuse']:
                continue
            if (directory/'receipts'/(key+'.json')).exists():
                frozen.check_entry(directory, e, ctx['identity'])
            else:
                pending.append(e)
        if pending:
            with parallel.thread_environment(1), ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                initializer=parallel.initialize_worker, initargs=(str(directory), ctx['identity'], None, 1,
                frozen.Builder, frozen.__name__)) as pool:
                for _ in tqdm(pool.map(parallel.build_chunk, pending, chunksize=1), total=len(pending),
                              desc='Build missing Day-30 TPG chunks', unit='chunk'):
                    pass
        loc = locations(directory, ctx)
        a = analysis(ctx, loc)
        for c, row in ctx['index'].items():
            write(directory/'by_cve'/(c+'.json'), row)
        for name, value in {'protocol.json': ctx['identity']['protocol'], 'cve_index.json': ctx['index'],
                            'chunk_locations.json': loc, 'analysis.json': a, 'run_status.json': {'status': 'complete'}}.items():
            write(directory/name, value)
        (directory/'analysis.txt').write_text(report(ctx, a), encoding='utf-8')
        preparation.sources.check_pins(ctx['identity']['source_pins'])
        require({str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()} == expected(ctx), 'Unexpected output files')
        write(directory/'cache_lock.json', {'schema': SCHEMA, 'files': {n: sha256(directory/n) for n in sorted(expected(ctx))}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--workers', type=int, default=4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan-only', action='store_true')
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--resume', action='store_true')
    parser.add_argument('--deep-verify', action='store_true')
    args = parser.parse_args()
    require(not args.deep_verify or args.verify_only, '--deep-verify requires --verify-only')
    parallel.cpu_settings(args.workers, 1)
    frozen.torch.set_num_threads(1)
    ctx = context(args.scale)
    directory = OUTPUT/args.scale
    if args.plan_only:
        print(f"MS05 {args.scale}: {len(ctx['entries'])} chunks; {len(ctx['reuse'])} reusable; {len(ctx['entries'])-len(ctx['reuse'])} new. No writes or encoding.")
        return
    if args.verify_only:
        verify(directory, ctx, args.deep_verify)
        print(f'MS05 {args.scale} cache verified. No encoder rerun, training or test scoring.')
        return
    if args.scale == 'full':
        small = context('smoke')
        verify(OUTPUT/'smoke', small)
        require(all(ctx['entries'].get(k) == e for k, e in small['entries'].items()), 'Smoke/full chunks differ')
    prepare(directory, ctx, args.workers, args.resume)
    print(f'MS05 {args.scale} cache saved to {directory}. No training or test scoring.')


if __name__ == '__main__':
    main()
