"""DQ17 v2: source-preserving coverage exception for U+FFFD-only tokens.

No model fitting or target scoring. Preserve the original smoke artifacts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch
from tqdm.auto import tqdm

from . import forecast_coreference_cache as shared
from . import forecast_coreference_development as v1
from . import forecast_coreference_coverage as coverage
import tpg_security_product_v4 as extension

base = shared.base
require, read_json, write_json, sha256 = base.require, base.read_json, base.write_json, base.sha256
SCHEMA = 'dq17_matched_coreference_development_cache_v2'
ROOT = base.CACHE_ROOT/'coreference_development_v2'
PROTOCOL = base.EPSS_ROOT/'02_prediction/04_configs/coreference_development_v1.json'
SCRIPT = shared.SCRIPTS/'prepare_security_coreference_development_v2.py'
TEST = shared.TPG/'08_tests/test_security_coreference_development_v2.py'
ROLES = ('train', 'validation')




def context():
    ctx = v1.context()
    smoke = v1.ROOT/'smoke'
    v1.verify(smoke, ctx, 'smoke')
    failed = v1.ROOT/'full'
    recovery = {}
    pins = dict(ctx['pins'])
    if failed.exists():
        require(read_json(failed/'run_status.json')['status'] == 'failed',
                'Expected the unchanged failed v1 run; do not run v1 concurrently')
        require(read_json(failed/'run_identity.json') ==
                v1.identity(ctx, 'full', sha256(smoke/'cache_lock.json'), smoke),
                'Failed v1 source/settings differ')
        allowed = {c for values in ctx['full_ids'].values() for c in values}
        for receipt in tqdm(sorted((failed/'receipts').glob('*.json')),
                            desc='Verify completed v1 entries for recovery', unit='CVE'):
            cve = receipt.stem
            require(cve in allowed, 'Recovery CVE outside development cohort')
            row = v1.check_entry(failed, cve, ctx)
            recovery[cve] = str(failed.resolve())
            for name, digest in row['files'].items():
                pins[str((failed/name).resolve())] = digest
            pins[str(receipt.resolve())] = sha256(receipt)
        for name in ('run_identity.json', 'run_status.json'):
            pins[str((failed/name).resolve())] = sha256(failed/name)
    # Include the observed failure as a source-quality regression case, never as
    # outcome-based sampling. The full cohort and training protocol stay fixed.
    smoke_ids = {split: list(ids) for split, ids in ctx['smoke_ids'].items()}
    regression_cve = 'CVE-2021-22822'
    require(regression_cve in ctx['full_ids']['train'], 'Regression CVE role changed')
    smoke_ids['train'] = sorted(set(smoke_ids['train']) | {regression_cve})
    for path in (Path(__file__), Path(coverage.__file__), SCRIPT, TEST, smoke/'cache_lock.json'):
        pins[str(path.resolve())] = sha256(path)
    base.check_pins(pins)
    return {**ctx, 'pins': pins, 'smoke_ids': smoke_ids, 'recovery': recovery}


def reuse_source(ctx, cve):
    if cve in ctx['recovery']:
        return Path(ctx['recovery'][cve])
    if cve in set(sum(ctx['prior']['ids'].values(), [])):
        return shared.DEFAULT_OUTPUT
    return None


def reuse_documents(ctx, cve):
    return (v1.docs_at(Path(ctx['recovery'][cve]), cve)
            if cve in ctx['recovery'] else shared.documents_for(cve))




class Builder:
    def __init__(self, prior, allow_download=False):
        self.encoder = shared.FrozenEncoder(allow_download)
        v1.check_encoder(self.encoder.identity, prior['encoder'])
        require(self.encoder.tokenizer.backend_tokenizer.normalizer.normalize_str('\ufffd') == '',
                'Frozen tokenizer no longer removes U+FFFD')
        self.pipeline = extension.ProductVersionSecurityPipeline()
        nlp = self.pipeline.frontend.nlp
        require(self.pipeline.frontend.model_name == 'en_core_web_sm' and
                all(nlp.has_pipe(p) for p in ('parser', 'tagger', 'ner')), 'Required spaCy components missing')
        original = read_json(base.CACHE_ROOT/'description_full_v1/manifest.json')
        self.nlp_hash = hashlib.sha256(nlp.to_bytes()).hexdigest()
        require(self.nlp_hash == original['resources']['spacy_pipeline_sha256'], 'spaCy resource changed')

    def build(self, cve, text):
        graph = self.pipeline.run(text, doc_id=cve)
        for node in graph.nodes():
            node.properties.extra['source_text_type'] = 'description'
        for edge in graph.edges():
            edge.properties.extra['source_text_type'] = 'description'
        shared.preview.semantic.frozen.audit.preview.validate_graph(graph, text)
        saved = json.loads(shared.GraphSONExporter().export_string(graph))
        documents, _ = shared.preview.variants(saved)
        bundle, vocab = coverage.build_bundle(documents, self.encoder)
        return documents, bundle, vocab


def entry_files(cve):
    require(Path(cve).name == cve and cve.startswith('CVE-'), 'Unsafe CVE ID')
    return [f'graphs/{cve}.pt', *[f'graphson/{mode}/{cve}.json' for mode in shared.MODES]]


def docs_at(directory, cve):
    return {mode: read_json(directory/'graphson'/mode/(cve+'.json')) for mode in shared.MODES}


def row_statistics(bundle, documents, vocab):
    stats = shared.row_statistics(bundle, vocab)
    count = len(coverage.missing_replacement_tokens(bundle, documents))
    # The frozen helper counted all unembedded tokens as whitespace; v2 makes
    # the newly permitted replacement-character category explicit.
    stats['unembedded_whitespace_tokens'] -= count
    stats['unembedded_replacement_tokens_count'] = count
    require(stats['unembedded_whitespace_tokens'] >= 0, 'Invalid missing-token accounting')
    return stats


def check_entry(directory, cve, ctx):
    receipt = read_json(directory/'receipts'/(cve+'.json'))
    require(receipt['cve'] == cve and set(receipt['files']) == set(entry_files(cve)), 'Invalid CVE receipt')
    for name, digest in receipt['files'].items():
        require(sha256(directory/name) == digest, f'Changed completed CVE artifact: {name}')
    docs = docs_at(directory, cve)
    for doc in docs.values():
        require(doc['doc_id'] == cve and doc['metadata']['source_text'] == ctx['descriptions'][cve],
                'Graph source wording differs')
        require(extension.VERSION in shared.import_graphson(doc).passes_applied, 'Missing v4 policy')
    bundle = torch.load(directory/'graphs'/(cve+'.pt'), weights_only=True, map_location='cpu')
    prior = ctx['prior']
    coverage.validate_bundle(bundle, docs, prior['type_vocabulary'], prior['embedding_dim'])
    reused = cve in set(sum(prior['ids'].values(), []))
    require(receipt['reused_dq16'] == reused, 'Reuse provenance differs')
    if reused:
        require(sha256(directory/'graphs'/(cve+'.pt')) ==
                sha256(shared.DEFAULT_OUTPUT/'graphs'/(cve+'.pt')), 'Reused DQ16 tensor changed')
        require(docs == shared.documents_for(cve), 'Reused DQ15 graphs changed')
    require(receipt['reused_dq17_v1'] == (cve in ctx['recovery']), 'Recovery provenance differs')
    if cve in ctx['recovery']:
        source = Path(ctx['recovery'][cve])
        require(sha256(directory/'graphs'/(cve+'.pt')) == sha256(source/'graphs'/(cve+'.pt'))
                and docs == v1.docs_at(source, cve), 'Recovered v1 entry changed')
    require(receipt['unembedded_replacement_tokens'] ==
            coverage.missing_replacement_tokens(bundle, docs), 'Missing-token audit differs')
    stats = row_statistics(bundle, docs, prior['type_vocabulary'])
    require(stats == receipt['statistics'], 'CVE statistics differ')
    return receipt


def identity(ctx, scale, smoke_lock=None, smoke_dir=None):
    return {'schema': SCHEMA, 'scale': scale, 'ids': ctx[scale+'_ids'],
            'source_pins': ctx['pins'], 'protocol': ctx['protocol'],
            'smoke_cache_lock_sha256': smoke_lock,
            'smoke_cache_directory': str(smoke_dir) if smoke_dir is not None else None,
            'predictor_fields': ['description'],
            'limitation': base.LIMITATION, 'training': False, 'test_graphs_built': False,
            'coverage_policy': coverage.POLICY, 'recovery': ctx['recovery']}


def report(manifest):
    rows = manifest['graphs']
    return '\n'.join([
        'DQ17: Matched development graph cache', '',
        f"Scale: {manifest['scale']}; CVEs by role: { {s: len(v) for s,v in manifest['ids'].items()} }.",
        f"Original DQ16 entries: {manifest['reused_dq16']}; verified v1 entries: {manifest['reused_dq17_v1']} (counts can overlap).",
        f"Reused entries total: {manifest['reused_total']}; newly parsed and embedded: {len(rows)-manifest['reused_total']}.",
        f"Unembedded U+FFFD-only tokens: {sum(len(v) for v in manifest['unembedded_replacement_tokens'].values())}.",
        'Configuration | Nodes | Edges',
        *[f"{mode} | {sum(r['nodes'] for r in rows)} | {sum(r['edges_by_mode'][mode] for r in rows)}"
          for mode in shared.MODES], '',
        'Every CVE has one shared node-feature tensor and three edge masks.',
        'no_coref removes all explicit COREF edges; repeated_only keeps repeated-text candidates;',
        'all_coref keeps all fixed v4 candidate edges, including the remaining pronoun links.',
        'Other edges, nodes, frozen embeddings and relation vocabulary are identical across modes.',
        'Mention nodes and other coreference-derived properties remain: this is an edge-only comparison.',
        'Only description text enters graph construction. EPSS histories and targets are joined later.',
        'No labels, structured CVSS, SMP, summaries or VulnCheck values are graph inputs.',
        'Original DQ16 tensors are copied byte-for-byte. New CVEs use fixed v4 and the same encoder.',
        'Ordinary text-token embeddings remain mandatory. Standalone U+FFFD replacement characters',
        'removed by the frozen tokenizer retain zero vectors and presence mask zero, with explicit',
        'node IDs and source spans in the per-CVE audit. No raw text, nodes or edges are removed.',
        'This is source encoding damage, not truncation or an invented semantic embedding.',
        'No record is silently skipped. Completed CVE receipts are checked before resuming.',
        'GraphSON stores all three unembedded variants for inspection; graphs/*.pt stores shared tensors.',
        'manifest.json and analysis.json include per-CVE counts and hashes; cache_lock.json locks all files.',
        'protocol.json fixes the future 3-seed, maximum-100-epoch development comparison.',
        'No training, validation predictions, cohort changes or test graphs are performed here.', '',
        base.LIMITATION,
        'Candidate relations are not verified facts. Validation has already informed development.', ''])


def verify(directory, ctx=None, expected_scale=None):
    directory = Path(directory).resolve()
    ctx = context() if ctx is None else ctx
    lock = read_json(directory/'cache_lock.json')
    require(lock['schema'] == SCHEMA, 'Wrong cache schema')
    manifest = read_json(directory/'manifest.json')
    scale = manifest['scale']
    require(scale in ('smoke', 'full') and (expected_scale is None or scale == expected_scale), 'Wrong cache scale')
    wanted = identity(ctx, scale, manifest['smoke_cache_lock_sha256'], manifest['smoke_cache_directory'])
    if scale == 'smoke':
        require(wanted['smoke_cache_lock_sha256'] is None and wanted['smoke_cache_directory'] is None,
                'Unexpected smoke dependency')
    else:
        require(isinstance(wanted['smoke_cache_lock_sha256'], str) and
                len(wanted['smoke_cache_lock_sha256']) == 64, 'Missing preview verification')
        require(isinstance(wanted['smoke_cache_directory'], str), 'Missing preview directory')
        preview_dir = Path(wanted['smoke_cache_directory']).resolve()
        verify(preview_dir, ctx, 'smoke')
        require(sha256(preview_dir/'cache_lock.json') == wanted['smoke_cache_lock_sha256'],
                'Preview lock changed')
    require(read_json(directory/'run_identity.json') == wanted and
            all(manifest[k] == v for k,v in wanted.items()), 'Source/cohort/protocol identity differs')
    require(read_json(directory/'run_status.json')['status'] == 'complete', 'Incomplete cache')
    ids = [c for s in ROLES for c in wanted['ids'][s]]
    expected = {'manifest.json', 'analysis.json', 'analysis.txt', 'protocol.json', 'run_identity.json', 'run_status.json'}
    for cve in ids:
        expected.update(entry_files(cve)); expected.add(f'receipts/{cve}.json')
    require(set(lock['files']) == expected, 'Unexpected/missing lock entries')
    require({str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()} ==
            expected | {'cache_lock.json'}, 'Unexpected/missing cache files')
    for name, digest in tqdm(lock['files'].items(), desc='Verify development artifact hashes', unit='file'):
        require(sha256(directory/name) == digest, f'Changed artifact: {name}')
    rows, reused, recovered, reused_total, missing = [], 0, 0, 0, {}
    for cve in tqdm(ids, desc='Verify three matched configurations', unit='CVE'):
        receipt = check_entry(directory, cve, ctx)
        rows.append(receipt['statistics']); reused += receipt['reused_dq16']
        recovered += receipt['reused_dq17_v1']
        reused_total += bool(receipt['reused_dq16'] or receipt['reused_dq17_v1'])
        missing[cve] = receipt['unembedded_replacement_tokens']
    require(manifest['graphs'] == rows and manifest['reused_dq16'] == reused
            and manifest['reused_dq17_v1'] == recovered and manifest['reused_total'] == reused_total
            and manifest['unembedded_replacement_tokens'] == missing, 'Coverage differs')
    require(manifest['encoder'] == ctx['prior']['encoder'] and
            manifest['embedding_dim'] == ctx['prior']['embedding_dim'] and
            manifest['type_vocabulary'] == ctx['prior']['type_vocabulary'], 'Feature identity differs')
    require(read_json(directory/'analysis.json') == manifest and read_json(directory/'protocol.json') == ctx['protocol'],
            'Report/protocol differs')
    require((directory/'analysis.txt').read_text() == report(manifest), 'Text report differs')
    base.check_pins(ctx['pins'])
    return manifest


def prepare(output, scale='smoke', check_only=False, verify_only=False, resume=False,
            allow_download=False, smoke_dir=None):
    output = Path(output).resolve()
    ctx = context()
    if verify_only:
        verify(output, ctx, scale)
        print(f'DQ17 v2 {scale} cache verified. No encoder rerun, training or test scoring.'); return
    require(ROOT.resolve() in output.parents, 'Use a separate folder inside coreference_development_v2')
    smoke_lock = None
    if scale == 'full':
        require(smoke_dir is not None, 'Verify the small DQ17 preview first; pass --smoke-dir')
        smoke_dir = Path(smoke_dir).resolve()
        require(output != smoke_dir and smoke_dir not in output.parents and output not in smoke_dir.parents,
                'Output overlaps preview')
        verify(smoke_dir, ctx, 'smoke')
        smoke_lock = sha256(smoke_dir/'cache_lock.json')
    else:
        require(smoke_dir is None, '--smoke-dir is only used for full preparation')
    ident = identity(ctx, scale, smoke_lock, smoke_dir)
    if resume:
        require(output.is_dir() and not (output/'cache_lock.json').exists(),
                'Resume needs an unfinished output; use --verify-only for completed caches')
        require(read_json(output/'run_identity.json') == ident, 'Resume source/settings changed')
        allowed = {'run_identity.json', 'run_status.json', 'manifest.json', 'analysis.json', 'analysis.txt', 'protocol.json'}
        for split in ROLES:
            for cve in ident['ids'][split]:
                allowed.update(entry_files(cve)); allowed.add(f'receipts/{cve}.json')
        actual = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}
        require(actual <= allowed | {p+'.tmp' for p in allowed}, 'Unexpected resume files')
    else:
        base.check_destination(output, [shared.DEFAULT_OUTPUT, shared.SOURCE, base.PREPARED])
    builder = Builder(ctx['prior'], allow_download)
    if check_only:
        # Exercise previously unseen descriptions, not only cached examples.
        checks = [next(c for c in ctx['smoke_ids'][s] if c not in ctx['prior']['ids'][s]) for s in ROLES]
        if 'CVE-2021-22822' in ctx['descriptions']:
            checks.append('CVE-2021-22822')
        for cve in tqdm(sorted(set(checks)), desc='Check new descriptions and failure regression', unit='CVE'):
            docs, bundle, vocab = builder.build(cve, ctx['descriptions'][cve])
            require(vocab == ctx['prior']['type_vocabulary'], 'Vocabulary changed')
            coverage.validate_bundle(bundle, docs, vocab, ctx['prior']['embedding_dim'])
        base.check_pins(ctx['pins'])
        print('DQ17 v2 preflight passed, including replacement-character coverage; no files or training.'); return
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/'run_identity.json', ident)
    write_json(output/'run_status.json', {'status': 'running'})
    try:
        rows, reused_count, recovered_count, reused_total, missing = [], 0, 0, 0, {}
        prior_ids = set(sum(ctx['prior']['ids'].values(), []))
        ids = [c for s in ROLES for c in ident['ids'][s]]
        for cve in tqdm(ids, desc='Prepare shared nodes and three edge configurations', unit='CVE'):
            receipt_path = output/'receipts'/(cve+'.json')
            if receipt_path.exists():
                receipt = check_entry(output, cve, ctx)
            else:
                reused = cve in prior_ids
                source = reuse_source(ctx, cve)
                if source is not None:
                    docs = reuse_documents(ctx, cve)
                    bundle = torch.load(source/'graphs'/(cve+'.pt'), weights_only=True, map_location='cpu')
                    vocab = ctx['prior']['type_vocabulary']
                else:
                    docs, bundle, vocab = builder.build(cve, ctx['descriptions'][cve])
                require(vocab == ctx['prior']['type_vocabulary'], 'Vocabulary changed')
                coverage.validate_bundle(bundle, docs, vocab, ctx['prior']['embedding_dim'])
                for mode, doc in docs.items():
                    write_json(output/'graphson'/mode/(cve+'.json'), doc)
                path = output/'graphs'/(cve+'.pt')
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix('.pt.tmp')
                if source is not None:
                    shutil.copyfile(source/'graphs'/(cve+'.pt'), temp)
                else:
                    torch.save(bundle, temp)
                temp.replace(path)
                receipt = {'cve': cve, 'reused_dq16': reused,
                           'reused_dq17_v1': cve in ctx['recovery'],
                           'unembedded_replacement_tokens': coverage.missing_replacement_tokens(bundle, docs),
                           'statistics': row_statistics(bundle, docs, vocab),
                           'files': {name: sha256(output/name) for name in entry_files(cve)}}
                write_json(receipt_path, receipt)
            rows.append(receipt['statistics']); reused_count += receipt['reused_dq16']
            recovered_count += receipt['reused_dq17_v1']
            reused_total += bool(receipt['reused_dq16'] or receipt['reused_dq17_v1'])
            missing[cve] = receipt['unembedded_replacement_tokens']
        manifest = {**ident, 'graphs': rows, 'reused_dq16': reused_count,
                    'reused_dq17_v1': recovered_count, 'reused_total': reused_total,
                    'unembedded_replacement_tokens': missing,
                    **{k: ctx['prior'][k] for k in ('encoder', 'embedding_dim', 'type_vocabulary')}}
        with tqdm(total=3, desc='Save reports and freeze cache', unit='stage') as bar:
            write_json(output/'manifest.json', manifest)
            write_json(output/'analysis.json', manifest)
            write_json(output/'protocol.json', ctx['protocol'])
            (output/'analysis.txt').write_text(report(manifest), encoding='utf-8'); bar.update()
            base.check_pins(ctx['pins']); bar.update()
            write_json(output/'run_status.json', {'status': 'complete'})
            write_json(output/'cache_lock.json', {'schema': SCHEMA, 'files': {
                str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}})
            bar.update()
        verify(output, ctx, scale)
    except Exception as exc:
        write_json(output/'run_status.json', {'status': 'failed', 'error': str(exc)})
        raise
    print(f'DQ17 v2 {scale} matched cache saved to {output}. No training or test scoring.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--smoke-dir', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--resume', action='store_true')
    parser.add_argument('--allow-model-download', action='store_true')
    args = parser.parse_args()
    if args.verify_only and args.allow_model_download:
        parser.error('--verify-only never loads an encoder')
    torch.set_num_threads(1)
    prepare(args.output_dir or ROOT/args.scale, args.scale, args.check_only, args.verify_only,
            args.resume, args.allow_model_download, args.smoke_dir)


if __name__ == '__main__':
    main()
