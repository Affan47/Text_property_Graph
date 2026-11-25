"""DQ16: shared frozen embeddings and three COREF edge masks, development only."""
import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import torch
from tqdm.auto import tqdm

from . import forecast_tpg_data as base
from tpg.embeddings import embed_graph
from tpg.exporters.exporters import GraphSONExporter, PyGExporter, import_graphson
from tpg.schema.types import NodeType

TPG = base.EPSS_ROOT/'01_tpg'
SCRIPTS = TPG/'02_examples/01_scripts'
sys.path[:0] = [str(SCRIPTS), str(TPG/'10_extensions')]
import preview_security_coreference_ablation as preview

require, read_json, write_json, sha256 = base.require, base.read_json, base.write_json, base.sha256
SCHEMA = 'dq16_shared_coreference_cache_v1'
MODEL = 'jackaduma/SecBERT'
SOURCE = preview.DEFAULT_OUTPUT
DEFAULT_OUTPUT = base.CACHE_ROOT/'coreference_v1/smoke'
MODES = preview.policy.MODES
EDGE_FIELDS = ('edge_index', 'edge_type', 'edge_features')


def selected_ids(rows, splits, cohorts):
    """Check only the fixed development IDs; no resampling or silent exclusion."""
    ids = {'train': [], 'validation': []}
    seen, groups = set(), {'train': set(), 'validation': set()}
    for row in rows:
        cve, split = row['cve'], row['split']
        require(split in ids and cve not in seen, 'Forbidden split or duplicate CVE')
        require(Path(cve).name == cve and cve.startswith('CVE-'), 'Unsafe CVE ID')
        require(splits[cve]['split'] == split and cve in cohorts['endpoint'][split],
                f'Changed role or endpoint eligibility: {cve}')
        ids[split].append(cve)
        groups[split].add(splits[cve]['group_id'])
        seen.add(cve)
    require(all(ids.values()) and groups['train'].isdisjoint(groups['validation']),
            'Empty role or related CVEs across splits')
    return ids


def context():
    rows, upstream_pins = preview.source_context()
    preview.verify_output(SOURCE, rows, upstream_pins)
    # The upstream lock already verifies the raw/prepared bundle. Read only roles
    # and cohort eligibility here; graph construction never receives labels or scores.
    splits = read_json(base.PREPARED/'split_manifest.json')
    cohorts = read_json(base.PREPARED/'task_cohorts.json')
    ids = selected_ids(rows, splits, cohorts)
    pins = dict(upstream_pins)
    for name, digest in read_json(SOURCE/'preview_lock.json')['files'].items():
        pins[str((SOURCE/name).resolve())] = digest
    for path in (SOURCE/'preview_lock.json', Path(__file__),
                 SCRIPTS/'prepare_security_coreference_cache.py',
                 TPG/'08_tests/test_security_coreference_cache.py',
                 base.PREPARED/'split_manifest.json', base.PREPARED/'task_cohorts.json'):
        key, digest = str(path.resolve()), sha256(path)
        require(key not in pins or pins[key] == digest, 'Source changed while collecting provenance')
        pins[key] = digest
    base.check_pins(pins)
    return rows, ids, pins


class FrozenEncoder:
    def __init__(self, allow_download=False):
        from transformers import AutoModel, AutoTokenizer
        with tqdm(total=3, desc='Load frozen SecBERT and record identity', unit='stage') as bar:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True,
                local_files_only=not allow_download, trust_remote_code=False)
            require(self.tokenizer.is_fast, 'Fast tokenizer with source offsets required')
            bar.update()
            # Only contextual token states are used. Disable the unused pooler,
            # which can otherwise be randomly initialized for an MLM checkpoint.
            self.model, info = AutoModel.from_pretrained(MODEL, add_pooling_layer=False,
                local_files_only=not allow_download, trust_remote_code=False, output_loading_info=True)
            require(not info.get('missing_keys') and not info.get('mismatched_keys') and
                    not info.get('error_msgs'), 'Incomplete encoder weights; no random fallback')
            self.model.cpu().eval().requires_grad_(False)
            self.dim = self.model.config.hidden_size
            bar.update()
            digest = hashlib.sha256()
            for name, value in self.model.state_dict().items():
                digest.update(name.encode('utf-8'))
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            self.identity = {'model_name': MODEL, 'revision': getattr(self.model.config, '_commit_hash', None),
                'weights_sha256': digest.hexdigest(),
                'tokenizer_sha256': base.text_digest(self.tokenizer.backend_tokenizer.to_str()),
                'config_sha256': base.text_digest(self.model.config.to_json_string()),
                'unused_checkpoint_keys': info.get('unexpected_keys', []), 'pooler': False,
                'frozen': True, 'device': 'cpu', 'embedding_dim': self.dim,
                'embedding_method': 'contextual_mean_nonoverlapping_windows',
                'packages': {n: importlib.metadata.version(n) for n in
                             ('torch', 'torch-geometric', 'transformers', 'tokenizers')}}
            bar.update()

    def embed(self, graph):
        require(not self.model.training and not any(p.requires_grad for p in self.model.parameters()),
                'Encoder is not frozen')
        with torch.inference_mode():
            embed_graph(graph, self.tokenizer, self.model, 'cpu')


def edge_tensors(payload):
    require(payload['y'] is None and payload['n_dropped_security_edges'] == 0, 'Labels/dropped security edges')
    return {'edge_index': torch.tensor(payload['edge_index'], dtype=torch.long).reshape(2, -1),
        'edge_type': torch.tensor(payload['edge_type'], dtype=torch.long),
        'edge_features': torch.tensor(payload['edge_features'], dtype=torch.float32).reshape(-1, payload['edge_feature_dim'])}


def materialize(bundle, mode):
    """Return the existing forecast graph-item contract; shared tensors are read-only."""
    require(mode in MODES, 'Unknown edge configuration')
    mask = bundle['edge_masks'][mode]
    return {**bundle['shared'], 'edge_index': bundle['edges']['edge_index'][:, mask],
            'edge_type': bundle['edges']['edge_type'][mask],
            'edge_features': bundle['edges']['edge_features'][mask]}


def build_bundle(documents, encoder):
    require(set(documents) == set(MODES), 'Missing/extra graph variant')
    graph = import_graphson(documents['all_coref'])
    require(not graph.validate(), 'Invalid source graph')
    require(all('embedding' not in n.properties.extra for n in graph.nodes()), 'Already embedded source')
    before = json.loads(GraphSONExporter().export_string(graph))
    require(before == documents['all_coref'], 'Source round trip differs')
    require(graph.metadata['coreference_ablation']['mode'] == 'all_coref', 'Wrong base graph')
    encoder.embed(graph)
    # Shared embedding code may add embedding properties/metadata, not change edges or wording.
    after = json.loads(GraphSONExporter().export_string(graph))
    require(after['edges'] == before['edges'], 'Embedding changed graph edges')
    for node, original in zip(graph.nodes(), import_graphson(before).nodes()):
        props = deepcopy(node.properties)
        embedding = props.extra.pop('embedding', None)
        if embedding is not None:
            require(len(embedding) == encoder.dim, 'Unexpected embedding width')
        require(node.id == original.id and node.node_type == original.node_type and props == original.properties,
                'Embedding changed source node properties')
    require(len(after['vertices']) == len(before['vertices']), 'Embedding changed node count')
    payload = PyGExporter().export(graph, embedding_dim=encoder.dim, use_security_edge_types=True)
    vocab = payload['type_vocabulary']
    x = torch.tensor(payload['x'], dtype=torch.float32)
    mask = torch.tensor(['embedding' in n.properties.extra for n in graph.nodes()], dtype=torch.float32)
    token = x[:, vocab['node_types']['TOKEN']] == 1
    lexical = torch.tensor([n.node_type == NodeType.TOKEN and bool(n.properties.text.strip())
                            for n in graph.nodes()], dtype=torch.bool)
    require(bool(lexical.any()) and bool((mask[lexical] == 1).all()),
            'Missing text-token embeddings; refuse silent truncation')
    pooled_tokens = token & mask.bool()
    shared = {'x': torch.cat((x, mask[:, None]), dim=1),
        'text_embedding': x[pooled_tokens, vocab['num_node_types']:].mean(0), 'cve': graph.doc_id,
        'description_sha256': base.text_digest(graph.metadata['source_text'])}
    bundle = {'shared': shared, 'edges': edge_tensors(payload), 'edge_masks': {
        mode: torch.tensor([preview.policy.retain(e, mode) for e in graph.edges()], dtype=torch.bool)
        for mode in MODES}}
    validate_bundle(bundle, documents, vocab, encoder.dim)
    return bundle, vocab


def validate_bundle(bundle, documents, vocab, dim):
    require(type(dim) is int and dim > 0, 'Invalid embedding dimension')
    require(set(bundle) == {'shared', 'edges', 'edge_masks'}, 'Unexpected bundle fields')
    require(set(bundle['shared']) == {'x', 'text_embedding', 'cve', 'description_sha256'} and
            set(bundle['edges']) == set(EDGE_FIELDS) and set(bundle['edge_masks']) == set(MODES),
            'Unexpected fields: targets and EPSS inputs are forbidden')
    base_graph = import_graphson(documents['all_coref'])
    require(bundle['shared']['cve'] == base_graph.doc_id and bundle['shared']['description_sha256'] ==
            base.text_digest(base_graph.metadata['source_text']), 'Source identity differs')
    base_payload = PyGExporter().export(base_graph, embedding_dim=0, use_security_edge_types=True)
    require(vocab == base_payload['type_vocabulary'], 'Vocabulary changed')
    for field, expected in edge_tensors(base_payload).items():
        require(bundle['edges'][field].dtype == expected.dtype and
                torch.equal(bundle['edges'][field], expected), f'Base {field} differs')
    for mode in MODES:
        mask = bundle['edge_masks'][mode]
        expected_mask = torch.tensor([preview.policy.retain(e, mode) for e in base_graph.edges()], dtype=torch.bool)
        require(mask.dtype == torch.bool and mask.shape == expected_mask.shape and torch.equal(mask, expected_mask),
                'Edge mask differs from frozen provenance policy')
        item = materialize(bundle, mode)
        base.validate_item(item, vocab, dim)
        require(item['x'].dtype == torch.float32 and item['text_embedding'].dtype == torch.float32,
                'Unexpected feature dtype')
        saved = documents[mode]
        require(saved['vertices'] == documents['all_coref']['vertices'], 'Variant nodes differ')
        require(saved['metadata']['coreference_ablation']['mode'] == mode, 'Variant identity differs')
        reference = PyGExporter().export(import_graphson(saved), embedding_dim=0, use_security_edge_types=True)
        require(reference['type_vocabulary'] == vocab, 'Variant vocabulary differs')
        for field, expected in edge_tensors(reference).items():
            require(torch.equal(item[field], expected), f'Variant {mode}/{field} differs')
        require(torch.equal(item['x'][:, :vocab['num_node_types']], torch.tensor(reference['x'], dtype=torch.float32)),
                'Node types differ from source')
    x = bundle['shared']['x']
    tokens = x[:, vocab['node_types']['TOKEN']] == 1
    lexical = torch.tensor([n.node_type == NodeType.TOKEN and bool(n.properties.text.strip())
                            for n in base_graph.nodes()], dtype=torch.bool)
    require(bool(lexical.any()) and bool((x[lexical, -1] == 1).all()), 'Missing text-token embeddings')
    pooled_tokens = tokens & x[:, -1].bool()
    require(torch.equal(bundle['shared']['text_embedding'], x[pooled_tokens, vocab['num_node_types']:-1].mean(0)),
            'Text pooling differs')


def documents_for(cve):
    return {m: read_json(SOURCE/'graphs'/m/(cve+'.json')) for m in MODES}


def row_statistics(bundle, vocab):
    x = bundle['shared']['x']
    embedded = x[:, -1] == 1
    return {'cve': bundle['shared']['cve'], 'description_sha256': bundle['shared']['description_sha256'],
        'nodes': len(x), 'embedded_nodes': int(embedded.sum()), 'unembedded_nodes': int((~embedded).sum()),
        'unembedded_whitespace_tokens': int(((x[:, vocab['node_types']['TOKEN']] == 1) & ~embedded).sum()),
        'embedded_by_node_type': {name: int(((x[:, i] == 1) & embedded).sum()) for name, i in vocab['node_types'].items()},
        'edges_by_mode': {m: int(bundle['edge_masks'][m].sum()) for m in MODES}}


def report(manifest):
    rows = manifest['graphs']
    return '\n'.join(['DQ16: Shared Node Embeddings for Three Edge Configurations', '',
        f"CVEs: {len(rows)}; training role: {len(manifest['ids']['train'])}; validation role: {len(manifest['ids']['validation'])}.",
        f"Embedding dimensions: {manifest['embedding_dim']}; encoder: {manifest['encoder']['model_name']}.",
        f"Nodes: {sum(r['nodes'] for r in rows)}; embedded: {sum(r['embedded_nodes'] for r in rows)}; "
        f"without embeddings: {sum(r['unembedded_nodes'] for r in rows)}.", '',
        'Configuration | Edges retained across the same CVEs',
        *[f"{m} | {sum(r['edges_by_mode'][m] for r in rows)}" for m in MODES], '',
        'Each CVE is embedded once. The file stores shared node features, pooled token text features,',
        'one full edge table and three boolean edge masks. A true mask value keeps that edge.',
        'The loader assembles the existing forecast graph-item format without changing relation IDs.',
        'Nodes contain node-type indicators, contextual SecBERT values and an embedding-present indicator.',
        'Nodes without a source-aligned embedding retain zero embedding values and indicator zero;',
        'they are not dropped. Tokens containing text must be embedded. Whitespace-only tokens such as',
        'line breaks may lack embeddings; they retain indicator zero and are excluded from text pooling.',
        f"Whitespace-only tokens without embeddings: {sum(r['unembedded_whitespace_tokens'] for r in rows)}.",
        'Counts by node type are in analysis.json.',
        'Sentence windows and span pooling use the shared core embedding code. No parser or extractor reruns.',
        'The unused BERT pooler is disabled. Contextual token weights are frozen, eval-only, on CPU.',
        'No EPSS scores, target values or VulnCheck fields are stored in graph tensors.',
        'Only development-role and endpoint eligibility metadata are read for the cohort check.',
        'Future training must separately join frozen EPSS inputs and targets by CVE, respecting saved roles.', '',
        'Limitations', base.LIMITATION,
        'This is an edge-only ablation: mention nodes and other coreference-derived structure remain.',
        'The 56-CVE sample includes reviewed cases and is for debugging, not an unbiased benchmark.',
        'A model-ready cache is not a trained model or evidence of forecasting improvement.',
        'No cohort expansion, parameter update, model prediction, validation metric or test scoring is performed.',
        'Old model-ready caches and trained weights remain separate. Use the new loader for this schema.', '',
        'Files: graphs/*.pt stores one tensor bundle per CVE; manifest.json records source/encoder identity;',
        'analysis.json and analysis.txt summarize coverage; cache_lock.json records checksums.', ''])


def verify(directory, current_context=None):
    directory = Path(directory).resolve()
    rows, ids, pins = current_context if current_context is not None else context()
    lock = read_json(directory/'cache_lock.json')
    require(lock['schema'] == SCHEMA, 'Wrong cache schema')
    expected_files = {'manifest.json', 'analysis.json', 'analysis.txt', 'run_status.json'} | {
        'graphs/'+r['cve']+'.pt' for r in rows}
    require(set(lock['files']) == expected_files, 'Unexpected/missing artifacts in lock')
    require({str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()} ==
            expected_files | {'cache_lock.json'}, 'Cache file set differs')
    for name, digest in tqdm(lock['files'].items(), desc='Verify shared-cache hashes', unit='file'):
        require(sha256(directory/name) == digest, f'Changed artifact: {name}')
    manifest = read_json(directory/'manifest.json')
    require(manifest['schema'] == SCHEMA and manifest['scale'] == 'smoke' and
            manifest['source_pins'] == pins and manifest['ids'] == ids,
            'Cache source/cohort identity differs')
    require(manifest['source_directory'] == str(SOURCE.resolve()) and manifest['predictor_fields'] == ['description']
            and manifest['limitation'] == base.LIMITATION and manifest['training'] is False
            and manifest['test_graphs_built'] is False, 'Cache scope differs')
    require(read_json(directory/'run_status.json')['status'] == 'complete', 'Incomplete cache')
    bundles, stats = {}, []
    for row in tqdm(rows, desc='Check tensors against three frozen graphs', unit='CVE'):
        cve = row['cve']
        bundle = torch.load(directory/'graphs'/(cve+'.pt'), map_location='cpu', weights_only=True)
        validate_bundle(bundle, documents_for(cve), manifest['type_vocabulary'], manifest['embedding_dim'])
        require(bundle['shared']['description_sha256'] == row['description_sha256'], 'Description checksum differs')
        stats.append(row_statistics(bundle, manifest['type_vocabulary']))
        bundles[cve] = bundle
    require(stats == manifest['graphs'], 'Coverage statistics differ')
    require(read_json(directory/'analysis.json') == {'schema': SCHEMA, 'ids': ids, 'graphs': stats}, 'Analysis differs')
    require((directory/'analysis.txt').read_text(encoding='utf-8') == report(manifest), 'Report differs')
    base.check_pins(pins)
    return manifest, bundles


def load_cache(directory, mode):
    """Verified adapter for future training; no labels loaded or training started."""
    require(mode in MODES, 'Unknown edge configuration')
    manifest, bundles = verify(directory)
    return manifest, {cve: materialize(bundle, mode) for cve, bundle in bundles.items()}


def prepare(output, check_only=False, verify_only=False, allow_download=False):
    output = Path(output).resolve()
    ctx = context()
    rows, ids, pins = ctx
    if verify_only:
        verify(output, ctx)
        print('DQ16 cache verified against fixed DQ15 graphs. No encoder rerun or training.'); return
    require(DEFAULT_OUTPUT.parent.resolve() in output.parents, 'Use a new folder inside forecast_graphs/coreference_v1')
    base.check_destination(output, [SOURCE, base.PREPARED, TPG/'01_core', TPG/'10_extensions'])
    encoder = FrozenEncoder(allow_download)
    if check_only:
        base.check_pins(pins)
        print(f'DQ16 preflight passed: {len(rows)} CVEs, frozen encoder loaded; no embeddings or cache created.'); return
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'run_status.json', {'status': 'running'})
    try:
        (output/'graphs').mkdir()
        stats, vocab = [], None
        for row in tqdm(rows, desc='Embed each CVE once; check three edge masks', unit='CVE'):
            bundle, current = build_bundle(documents_for(row['cve']), encoder)
            require(vocab is None or current == vocab, 'Vocabulary changed across CVEs')
            require(bundle['shared']['cve'] == row['cve'] and
                    bundle['shared']['description_sha256'] == row['description_sha256'], 'CVE/source mismatch')
            vocab = current
            stats.append(row_statistics(bundle, vocab))
            torch.save(bundle, output/'graphs'/(row['cve']+'.pt'))
        manifest = {'schema': SCHEMA, 'scale': 'smoke', 'ids': ids, 'source_directory': str(SOURCE.resolve()),
            'source_pins': pins, 'encoder': encoder.identity, 'embedding_dim': encoder.dim,
            'type_vocabulary': vocab, 'graphs': stats, 'predictor_fields': ['description'],
            'limitation': base.LIMITATION, 'training': False, 'test_graphs_built': False}
        with tqdm(total=3, desc='Save shared-cache reports and provenance', unit='stage') as bar:
            write_json(output/'manifest.json', manifest)
            write_json(output/'analysis.json', {'schema': SCHEMA, 'ids': ids, 'graphs': stats})
            (output/'analysis.txt').write_text(report(manifest), encoding='utf-8'); bar.update()
            base.check_pins(pins); bar.update()
            write_json(output/'run_status.json', {'status': 'complete'})
            write_json(output/'cache_lock.json', {'schema': SCHEMA, 'files': {
                str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}})
            bar.update()
        verify(output, ctx)
    except Exception as exc:
        write_json(output/'run_status.json', {'status': 'failed', 'error': str(exc)})
        raise
    print(f'DQ16 shared-embedding cache saved to {output}. No training, predictions or test graphs.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--check-only', action='store_true')
    action.add_argument('--verify-only', action='store_true')
    parser.add_argument('--allow-model-download', action='store_true', help='Opt in to fetching SecBERT; offline by default')
    args = parser.parse_args()
    if args.verify_only and args.allow_model_download:
        parser.error('--verify-only does not load/download an encoder')
    torch.set_num_threads(1)
    prepare(args.output_dir, args.check_only, args.verify_only, args.allow_model_download)


if __name__ == '__main__':
    main()
