"""CD03: inspectable, chunked CD02 TPGs and streaming model-ready caches."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil

from . import forecast_identity_development as preparation
from . import forecast_multisource_graphs as old

ROOT = preparation.ROOT
TPG = ROOT / '01_tpg'
CONFIG = ROOT / '02_prediction/04_configs/cve_identity_graph_cache_v1.json'
OUTPUT = TPG / '05_workspace/cve_identity_multisource_v1'
SCRIPT = TPG / '02_examples/01_scripts/prepare_identity_multisource_graphs.py'
TEST = TPG / '08_tests/test_identity_multisource_graphs.py'
SCHEMA = 'cd03_identity_multisource_graph_cache_v1'
read, write, require, sha256, tqdm, torch = old.read, old.write, old.require, old.sha256, old.tqdm, old.torch


def segments(text, limit):
    require(isinstance(text, str) and text.strip() and type(limit) is int and limit >= 8, 'Invalid chunk input')
    result, start = [], 0
    while start < len(text):
        end = min(start + limit, len(text))
        reason = 'document_end'
        if end < len(text):
            boundary = next((i + 1 for i in range(end - 1, start + limit // 2 - 1, -1) if text[i].isspace()), None)
            if boundary is not None:
                end, reason = boundary, 'whitespace_boundary'
            else:
                reason = 'forced_character_boundary'
        chunk = text[start:end]
        material = any(not ch.isspace() and ch != '\ufffd' for ch in chunk)
        result.append({'index': len(result), 'start': start, 'end': end, 'boundary': reason,
                       'has_graph': material, 'non_graph_reason': None if material else 'whitespace_or_replacement_only'})
        start = end
    require(''.join(text[s['start']:s['end']] for s in result) == text, 'Chunk coverage lost text')
    return result


def files(entry):
    stem = entry['stem']
    return {f'tensors/{stem}.pt', *[f'tpg_json/{m}/{stem}.json' for m in old.shared.MODES]}


def plan(rows, ids, limit):
    require(set(ids) == {'train', 'validation'} and not set(ids['train']) & set(ids['validation']), 'Invalid development roles')
    selected = {c for v in ids.values() for c in v}
    require(len(rows) == len(selected) and {r['cve'] for r in rows} == selected, 'Source rows differ from CVE cohort')
    entries, documents, index = {}, {}, {}
    for r in rows:
        c, split = r['cve'], r['split']
        require(re.fullmatch(r'CVE-\d{4}-\d{4,}', c) and split in ids and c in ids[split], 'Unsafe CVE or test role')
        require(set(r) == {'cve', 'split', 'documents', 'source_mask'}, 'Unexpected source fields')
        require(r['source_mask'] == [int(any(d['source_text_type'] == s for d in r['documents'])) for s in old.SOURCES], 'Source masks differ')
        index[c] = {'cve': c, 'split': split, 'source_mask': r['source_mask'], 'documents': []}
        seen = set()
        for d in r['documents']:
            require(set(d) == {'document_id', 'source_text_type', 'text'} and d['source_text_type'] in old.SOURCES, 'Unexpected document fields')
            require(d['document_id'] not in seen, 'Duplicate source document')
            seen.add(d['document_id'])
            source = d['source_text_type']
            digest = old.shared.base.text_digest(d['document_id'])
            stem = f'{c}/{source}/{digest}'
            path = 'documents/' + stem + '.json'
            parent_hash = old.shared.base.text_digest(d['text'])
            parts = segments(d['text'], limit)
            require(any(s['has_graph'] for s in parts), 'Document has no embeddable text; explicit review needed, no silent drop')
            doc = {'cve': c, 'split': split, **d, 'text_sha256': parent_hash, 'chunks': []}
            for s in parts:
                part = dict(s)
                if s['has_graph']:
                    text = d['text'][s['start']:s['end']]
                    key = old.shared.base.text_digest(c + '\0' + d['document_id'] + '\0' + str(s['index']))
                    require(key not in entries, 'Chunk identity collision')
                    entry = {'key': key, 'graph_id': 'CD03-' + key, 'cve': c, 'split': split,
                        'document_id': d['document_id'], 'source_text_type': source, 'text': text,
                        'text_sha256': old.shared.base.text_digest(text), 'parent_text_sha256': parent_hash,
                        'chunk': dict(s), 'stem': stem + f"/chunk_{s['index']:05d}"}
                    entries[key] = entry
                    part.update(key=key, graph=f"tpg_json/no_coref/{entry['stem']}.json", tensor=f"tensors/{entry['stem']}.pt")
                doc['chunks'].append(part)
            documents[path] = doc
            index[c]['documents'].append({'document_id': d['document_id'], 'source_text_type': source,
                'source_file': path, 'characters': len(d['text']), 'chunks': doc['chunks']})
    return {'entries': entries, 'documents': documents, 'index': index}


def checked_files(directory, lock_name):
    lock = read(directory / lock_name)
    pins = {str(directory / lock_name): sha256(directory / lock_name)}
    for n, h in lock['files'].items():
        path = (directory / n).resolve()
        require(directory.resolve() in path.parents and sha256(path) == h, 'Changed/unsafe prerequisite: ' + n)
        pins[str(path)] = h
    return pins


def context(scale):
    p = read(CONFIG)
    require(p['schema'] == SCHEMA and p['source_order'] == list(old.SOURCES) and
            p['edge_configuration'] == 'no_coref' and p['extractor'] == old.extension.VERSION and
            not p['training'] and not p['test_graphs'], 'Wrong graph contract')
    data = preparation.OUTPUT / scale
    with tqdm(total=3, desc='Check prepared data and fixed graph resources', unit='stage') as bar:
        pins = checked_files(data, 'preparation_lock.json')
        identity = read(data / 'run_identity.json')
        require(identity['schema'] == preparation.SCHEMA and identity['scale'] == scale, 'Wrong CD02 export')
        preparation.sources.check_pins(identity['source_pins'])
        pins.update(identity['source_pins'])
        payload = read(data / 'source_inputs.json')
        require(payload['anchor_day'] == 60 and payload['view'] == 'retrospective_date_screened_train_overlap_masked', 'Wrong text view')
        ctx = plan(payload['rows'], identity['ids'], p['max_chunk_characters'])
        bar.update()
        ref = read(old.OUTPUT / 'manifest.json')
        ref_lock = read(old.OUTPUT / 'cache_lock.json')
        require(sha256(old.OUTPUT / 'manifest.json') == ref_lock['files']['manifest.json'], 'Changed MS02 resource reference')
        # Reuse frozen resources directly; calling the old fixed-cohort context would reject the new cohort.
        for name, h in ref['source_pins'].items():
            if name.endswith('.py'):
                require(sha256(name) == h, 'Frozen TPG/helper code changed: ' + name)
                pins[name] = h
        bar.update()
        for path in (CONFIG, SCRIPT, TEST, Path(__file__).resolve(), Path(old.__file__).resolve(),
                     Path(preparation.__file__).resolve(), old.OUTPUT / 'manifest.json', old.OUTPUT / 'cache_lock.json'):
            pins[str(path)] = sha256(path)
        meta = {k: ref[k] for k in ('encoder', 'embedding_dim', 'type_vocabulary', 'spacy_pipeline_sha256')}
        ctx['identity'] = {'schema': SCHEMA, 'scale': scale, 'protocol': p, 'ids': identity['ids'],
            'source_pins': pins, **meta, 'chunk_count': len(ctx['entries']), 'document_count': len(ctx['documents']),
            'training': False, 'test_graphs': False}
        bar.update()
    return ctx


def annotate(graph, e):
    old.stamp_graph(graph, e)
    graph.metadata['source_chunk'] = {'parent_text_sha256': e['parent_text_sha256'], **e['chunk']}
    for node in graph.nodes():
        p = node.properties
        available = bool(p.extra.get('span_available', True) and 0 <= p.char_start < p.char_end <= len(e['text']))
        p.extra['source_span_available'] = available
        p.extra['source_document_char_start'] = e['chunk']['start'] + p.char_start if available else None
        p.extra['source_document_char_end'] = e['chunk']['start'] + p.char_end if available else None
    return graph


class Builder(old.Builder):
    def build(self, entry):
        graph = annotate(self.pipeline.run(entry['text'], doc_id=entry['graph_id']), entry)
        saved = json.loads(old.shared.GraphSONExporter().export_string(graph))
        docs, _ = old.shared.preview.variants(saved)
        old.validate_documents(docs, entry)
        bundle, vocab = old.coverage.build_bundle(docs, self.encoder)
        return docs, bundle, vocab


def check_entry(directory, e, identity):
    receipt = read(directory / 'receipts' / (e['key'] + '.json'))
    require(receipt['entry'] == {k: v for k, v in e.items() if k != 'text'} and set(receipt['files']) == files(e), 'Wrong chunk receipt')
    for n, h in receipt['files'].items():
        require(sha256(directory / n) == h, 'Changed chunk artifact: ' + n)
    docs = {m: read(directory / 'tpg_json' / m / (e['stem'] + '.json')) for m in old.shared.MODES}
    old.validate_documents(docs, e)
    for g in docs.values():
        require(g['metadata']['source_chunk'] == {'parent_text_sha256': e['parent_text_sha256'], **e['chunk']}, 'Chunk offsets differ')
        for node in g['vertices']:
            props = node['properties']
            available = bool(props['EXTRA'].get('span_available', True) and 0 <= props['CHAR_START'] < props['CHAR_END'] <= len(e['text']))
            require(props['EXTRA']['source_span_available'] == available and
                    props['EXTRA']['source_document_char_start'] == (e['chunk']['start'] + props['CHAR_START'] if available else None) and
                    props['EXTRA']['source_document_char_end'] == (e['chunk']['start'] + props['CHAR_END'] if available else None), 'Global node offsets differ')
    bundle = torch.load(directory / 'tensors' / (e['stem'] + '.pt'), weights_only=True, map_location='cpu')
    old.coverage.validate_bundle(bundle, docs, identity['type_vocabulary'], identity['embedding_dim'])
    stats = old.statistics_for(e, docs, bundle, identity['type_vocabulary'])
    require(stats == receipt['statistics'], 'Chunk statistics differ')
    return bundle, stats


def save_entry(directory, e, identity, builder):
    docs, bundle, vocab = builder.build(e)
    require(vocab == identity['type_vocabulary'], 'Vocabulary changed')
    for m, g in docs.items():
        write(directory / 'tpg_json' / m / (e['stem'] + '.json'), g)
    path = directory / 'tensors' / (e['stem'] + '.pt')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + '.tmp')
    torch.save(bundle, temp)
    temp.replace(path)
    write(directory / 'receipts' / (e['key'] + '.json'), {
        'entry': {k: v for k, v in e.items() if k != 'text'},
        'statistics': old.statistics_for(e, docs, bundle, vocab),
        'files': {n: sha256(directory / n) for n in sorted(files(e))}})


def summary(ctx, stats):
    result = []
    for split, ids in ctx['identity']['ids'].items():
        for source in old.SOURCES:
            rows = [r for r in stats if r['split'] == split and r['source_text_type'] == source]
            result.append({'split': split, 'source': source, 'cves': len(ids),
                'cves_with_graphs': len({r['cve'] for r in rows}),
                'documents': len({(r['cve'], r['document_id']) for r in rows}), 'chunks': len(rows),
                **{k: sum(r[k] for r in rows) for k in ('nodes', 'embedded_nodes', 'unembedded_nodes', 'edges_no_coref', 'security_edges')}})
    return {'schema': SCHEMA, 'scale': ctx['identity']['scale'], 'coverage': result,
            'chunks': stats, 'training': False, 'test_graphs': False,
            'forced_boundaries': sum(s['boundary'] == 'forced_character_boundary' for d in ctx['documents'].values() for s in d['chunks']),
            'non_graph_segments': sum(not s['has_graph'] for d in ctx['documents'].values() for s in d['chunks'])}


def guide(ctx, a):
    lines = ['# CD03: Where the Real TPGs Are', '',
        'This cache contains actual source-derived graphs, not illustrative diagrams. No training or test graphs.', '',
        '## Open in This Order', '',
        '1. `cve_index.json`: all selected CVEs, source masks and document/chunk paths.',
        '2. `by_cve/CVE-....json`: one CVE, its original documents and every chunk.',
        '3. `documents/<CVE>/<source>/<document-hash>.json`: full unchanged text, chunk ranges and coverage.',
        '4. `tpg_json/no_coref/<CVE>/<source>/<document-hash>/chunk_00000.json`: the primary real TPG.',
        '5. `tensors/<CVE>/<source>/<document-hash>/chunk_00000.pt`: node embeddings/types/masks and edge tensors.',
        '6. `analysis.json` and `analysis.txt`: node/edge/embedding counts. `receipts` and `cache_lock.json` protect artifacts.', '',
        '## Graph Fields', '',
        '`vertices` are nodes (id, label, properties). `edges` are typed links between local node IDs.',
        '`metadata.source_text` is the exact chunk text. `metadata.source_chunk` locates it in the parent document.',
        '`CHAR_START/CHAR_END` are chunk-local, end-exclusive character offsets. Node EXTRA source_document_char_* are parent offsets, or null when source_span_available is false.',
        'Node and edge EXTRA source_text_type identifies description, github_summary, non_github_summary or smp.',
        'A global node identity is (graph doc_id, local node id). Separate chunks have no invented cross-chunk edges.',
        'GraphSON intentionally omits large embeddings; tensors store them separately. This does not mean the encoded graph lacks embeddings.',
        'Tensor shared.x contains node-type channels, contextual embedding channels and a final embedding-present mask.',
        'The saved type_vocabulary determines all indices, including SEC_* relations. Missing sources have no fake graphs.',
        'Three edge-mask variants are stored for compatibility; no_coref is the fixed primary model view.', '',
        '## Source Data', '',
        f"The prepared EPSS histories/targets/text remain in `{preparation.OUTPUT / ctx['identity']['scale']}`.",
        '`inputs.json` is past EPSS history; `labels.json` contains separate development targets. Neither enters graph construction.', '',
        '## Example Real Graph Links', '']
    for e in list(ctx['entries'].values())[:8]:
        path = f"tpg_json/no_coref/{e['stem']}.json"
        lines.append(f"- [{e['cve']} / {e['source_text_type']} / chunk {e['chunk']['index']}]({path})")
    lines += ['', '## Counts', '', '| Split | Source | CVEs with graphs | Documents | Chunks | Nodes | Edges |', '|---|---|---:|---:|---:|---:|---:|']
    for r in a['coverage']:
        lines.append('| ' + ' | '.join(str(r[k]) for k in ('split', 'source', 'cves_with_graphs', 'documents', 'chunks', 'nodes', 'edges_no_coref')) + ' |')
    lines += ['', '## Limits', ''] + ['- ' + s for s in ctx['identity']['protocol']['limitations']]
    return '\n'.join(lines) + '\n'


def expected(ctx):
    names = {'run_identity.json', 'protocol.json', 'cve_index.json', 'analysis.json', 'analysis.txt', 'README.md', 'run_status.json'}
    names.update(ctx['documents'])
    names.update(f'by_cve/{c}.json' for c in ctx['index'])
    for e in ctx['entries'].values():
        names.update(files(e))
        names.add('receipts/' + e['key'] + '.json')
    return names


def verify(directory, ctx):
    lock = read(directory / 'cache_lock.json')
    require(lock['schema'] == SCHEMA and set(lock['files']) == expected(ctx), 'Wrong completed file set')
    require({str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()} == expected(ctx) | {'cache_lock.json'}, 'Unexpected/missing cache files')
    for n, h in tqdm(lock['files'].items(), desc='Verify artifact hashes', unit='file'):
        require(sha256(directory / n) == h, 'Changed artifact: ' + n)
    require(read(directory / 'run_identity.json') == ctx['identity'] and read(directory / 'cve_index.json') == ctx['index'] and
            read(directory / 'protocol.json') == ctx['identity']['protocol'] and read(directory / 'run_status.json') == {'status': 'complete'}, 'Changed cache contract')
    for n, d in ctx['documents'].items():
        require(read(directory / n) == d, 'Document/chunk coverage differs')
    for c, row in ctx['index'].items():
        require(read(directory / 'by_cve' / (c + '.json')) == row, 'CVE index differs')
    stats = []
    for e in tqdm(ctx['entries'].values(), desc='Verify real graphs and embeddings', unit='chunk'):
        _, row = check_entry(directory, e, ctx['identity'])
        stats.append(row)
    a = summary(ctx, stats)
    require(read(directory / 'analysis.json') == a and (directory / 'README.md').read_text() == guide(ctx, a) and
            (directory / 'analysis.txt').read_text() == guide(ctx, a), 'Reports do not reproduce')
    preparation.sources.check_pins(ctx['identity']['source_pins'])
    return a


def numeric_cve(directory, ctx, cve, enabled):
    """Load one CVE at a time, keeping document/chunk nesting for later pooling."""
    require(cve in ctx['index'] and set(enabled) <= set(old.SOURCES), 'Unknown CVE/source')
    result = {s: [] for s in old.SOURCES}
    for d in ctx['index'][cve]['documents']:
        s = d['source_text_type']
        if s not in enabled:
            continue
        chunks = []
        for part in d['chunks']:
            if part['has_graph']:
                bundle, _ = check_entry(directory, ctx['entries'][part['key']], ctx['identity'])
                item = old.shared.materialize(bundle, 'no_coref')
                chunks.append({k: item[k] for k in old.NUMERIC_FIELDS})
        result[s].append(chunks)
    return {'by_source': result, 'source_mask': [bool(result[s]) for s in old.SOURCES]}


def prepare(directory, ctx, resume=False, builder_factory=Builder, allow_download=False, reuse=None):
    if resume:
        require(directory.is_dir() and not (directory / 'cache_lock.json').exists() and read(directory / 'run_identity.json') == ctx['identity'], 'Wrong/complete resume directory')
        allowed = expected(ctx)
        actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
        require(actual <= allowed | {n + '.tmp' for n in allowed}, 'Unexpected resume artifacts')
    else:
        require(not directory.exists(), 'Cache exists; use --verify-only or --resume')
        directory.mkdir(parents=True)
        write(directory / 'run_identity.json', ctx['identity'])
    write(directory / 'run_status.json', {'status': 'running'})
    builder, stats = None, []
    for n, d in tqdm(ctx['documents'].items(), desc='Save source documents and chunk coverage', unit='document'):
        if (directory / n).exists():
            require(read(directory / n) == d, 'Changed saved source document')
        else:
            write(directory / n, d)
    for e in tqdm(ctx['entries'].values(), desc='Build or reuse source chunk TPGs', unit='chunk'):
        receipt = directory / 'receipts' / (e['key'] + '.json')
        if not receipt.exists():
            previous = reuse / 'receipts' / (e['key'] + '.json') if reuse else None
            if previous and previous.exists():
                check_entry(reuse, e, ctx['identity'])
                for n in sorted(files(e)):
                    target = directory / n
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temp = Path(str(target) + '.tmp')
                    shutil.copy2(reuse / n, temp)
                    temp.replace(target)
                write(receipt, read(previous))
            else:
                if builder is None:
                    builder = builder_factory(ctx['identity'], allow_download)
                save_entry(directory, e, ctx['identity'], builder)
        _, row = check_entry(directory, e, ctx['identity'])
        stats.append(row)
    a = summary(ctx, stats)
    for c, row in ctx['index'].items():
        write(directory / 'by_cve' / (c + '.json'), row)
    write(directory / 'cve_index.json', ctx['index'])
    write(directory / 'protocol.json', ctx['identity']['protocol'])
    write(directory / 'analysis.json', a)
    for name in ('README.md', 'analysis.txt'):
        (directory / name).write_text(guide(ctx, a), encoding='utf-8')
    preparation.sources.check_pins(ctx['identity']['source_pins'])
    write(directory / 'run_status.json', {'status': 'complete'})
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    require(actual == expected(ctx), 'Unexpected files before locking cache')
    write(directory / 'cache_lock.json', {'schema': SCHEMA, 'files': {n: sha256(directory / n) for n in sorted(expected(ctx))}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--allow-model-download', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan-only', action='store_true')
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--inspect-cve')
    args = parser.parse_args()
    ctx = context(args.scale)
    directory = OUTPUT / args.scale
    if args.plan_only:
        print(f"CD03 {args.scale}: {len(ctx['index'])} CVEs, {len(ctx['documents'])} documents, {len(ctx['entries'])} graph chunks. Output: {directory}")
        print('No parser, encoder, files or training. Every source character is covered.')
        return
    if args.inspect_cve:
        require(args.inspect_cve in ctx['index'], 'CVE is not in this development cache')
        name = f'by_cve/{args.inspect_cve}.json'
        lock = read(directory / 'cache_lock.json')
        require(sha256(directory / name) == lock['files'][name], 'Changed CVE index')
        print(f'Index: {directory / name}')
        print(json.dumps(read(directory / name), indent=2))
        return
    if args.verify_only:
        require(not args.allow_model_download, 'Verification never downloads models')
        verify(directory, ctx)
        print(f'CD03 {args.scale} graph cache verified. No parser/encoder rerun, training or test graphs.')
        return
    reuse = None
    if args.scale == 'full':
        smoke_ctx = context('smoke')
        reuse = OUTPUT / 'smoke'
        verify(reuse, smoke_ctx)
        require(all(k in ctx['entries'] and ctx['entries'][k] == e for k, e in smoke_ctx['entries'].items()), 'Smoke/full chunks differ')
    prepare(directory, ctx, args.resume, allow_download=args.allow_model_download, reuse=reuse)
    print(f'CD03 {args.scale} real TPGs saved to {directory}. Read README.md and cve_index.json. No training or test graphs.')


if __name__ == '__main__':
    main()
