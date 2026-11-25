"""Source-bound current-description TPG cache for retrospective EPSS experiments."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
from pathlib import Path
import sys

import torch
from tqdm.auto import tqdm

EPSS_ROOT = Path(__file__).resolve().parents[3]
CORE = EPSS_ROOT/'01_tpg/01_core'
sys.path.insert(0, str(CORE))
if __package__:
    from . import forecast_data as contract
else:
    import forecast_data as contract
read_json, write_json, require, sha256 = (
    contract.read_json, contract.write_json, contract.require, contract.sha256)

VERSION = 'epss_retrospective_description_tpg_v1'
PREPARED = EPSS_ROOT/'04_data/01_records_and_graphs/epss_day30_to_day60'
CACHE_ROOT = EPSS_ROOT/'01_tpg/05_workspace/forecast_graphs'
SCALES = {'smoke': {'train': 32, 'validation': 16}, 'full': {'train': None, 'validation': None}}
LIMITATION = ('Current stored description; historical availability at Day 30 is unverified. '
              'Retrospective association only, not evidence of deployable historical forecasting.')


def text_digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def description_only(record):
    text = record.get('description')
    require(isinstance(text, str) and bool(text.strip()), 'Missing description; no silent fallback or cohort drop')
    return text


def select_ids(data, scale):
    require(scale in SCALES, 'Unknown scale')
    selected = {}
    manifest = data['split_manifest.json']
    for split, limit in SCALES[scale].items():
        ids = data['task_cohorts.json']['endpoint'][split]
        require(ids == sorted(set(ids)) and bool(ids), 'Empty or invalid endpoint cohort')
        require(all(manifest[c]['split'] == split and data['labels.json'][c]['eligible']['endpoint'] for c in ids),
                'Cohort role/eligibility differs')
        # Fixed hash ordering avoids outcome-based selection and CVE-year prefix selection.
        selected[split] = sorted(sorted(ids, key=lambda c: text_digest('tpg-smoke-v1:'+c))[:limit])
    groups = [{manifest[c]['group_id'] for c in selected[s]} for s in ('train', 'validation')]
    require(groups[0].isdisjoint(groups[1]), 'Related groups cross train/validation')
    return selected


def code_pins():
    paths = list((CORE/'tpg').rglob('*.py')) + [Path(__file__), Path(contract.__file__)]
    return {str(p.resolve()): sha256(p) for p in sorted(paths)}


def check_pins(pins):
    for path, digest in pins.items():
        require(sha256(path) == digest, f'Changed source/code/artifact: {path}')


def check_destination(output, protected):
    output = Path(output).resolve()
    require(not output.exists(), 'Output already exists; preserve it and use a new --output-dir')
    for p in protected:
        p = Path(p).resolve()
        require(output != p and p not in output.parents and output not in p.parents,
                f'Output overlaps protected input: {p}')


def load_source(prepared):
    with tqdm(total=2, desc='Verify source and frozen splits', unit='stage') as bar:
        data = contract.load_prepared(prepared)
        bar.update()
        source = Path(data['preparation_summary.json']['source_path']).resolve()
        raw = read_json(source)
        by_id = {r['cve']: r for r in raw}
        require(len(by_id) == len(raw), 'Duplicate CVE records')
        bar.update()
    pins = code_pins()
    pins.update({str((prepared/name).resolve()): sha256(prepared/name)
                 for name in (*contract.PREPARED_FILES, 'preparation_lock.json')})
    pins[str(source)] = data['preparation_summary.json']['source_sha256']
    return data, by_id, source, pins


class GraphBuilder:
    """Use shared security parsing, enrichment, embeddings and exporter without fallback."""
    def __init__(self, model_name='jackaduma/SecBERT', allow_download=False):
        from tpg.pipeline import SecurityPipeline
        from transformers import AutoModel, AutoTokenizer
        with tqdm(total=3, desc='Load frozen TPG resources', unit='resource') as bar:
            self.pipeline = SecurityPipeline(include_security_relations=True)
            nlp = self.pipeline.frontend.nlp
            require(self.pipeline.frontend.model_name == 'en_core_web_sm'
                    and nlp.has_pipe('parser') and nlp.has_pipe('tagger') and nlp.has_pipe('ner'),
                    'en_core_web_sm parser/tagger/NER required; blank fallback is not permitted')
            bar.update()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True,
                local_files_only=not allow_download)
            require(self.tokenizer.is_fast, 'Fast tokenizer with source offsets required')
            bar.update()
            self.model = AutoModel.from_pretrained(model_name, local_files_only=not allow_download).cpu().eval()
            self.model.requires_grad_(False)
            bar.update()
        digest = hashlib.sha256()
        for name, weight in self.model.state_dict().items():
            digest.update(name.encode())
            digest.update(weight.detach().cpu().contiguous().numpy().tobytes())
        self.identity = {'model_name': model_name, 'model_revision': getattr(self.model.config, '_commit_hash', None),
            'model_weights_sha256': digest.hexdigest(),
            'tokenizer_sha256': text_digest(self.tokenizer.backend_tokenizer.to_str()),
            'spacy_pipeline_sha256': hashlib.sha256(nlp.to_bytes()).hexdigest(),
            'spacy_model_meta': nlp.meta, 'frozen': True, 'device': 'cpu',
            'packages': {n: importlib.metadata.version(n) for n in
                         ('torch', 'torch-geometric', 'spacy', 'transformers', 'tokenizers')}}

    def build(self, text, cve):
        from tpg.embeddings import embed_graph
        graph = self.pipeline.run(text, doc_id=cve)
        issues = graph.validate()
        require(not issues, f'{cve}: invalid TPG: {issues[:5]}')
        for node in graph.nodes():
            node.properties.extra['source_text_type'] = 'description'
        for edge in graph.edges():
            edge.properties.extra['source_text_type'] = 'description'
        embed_graph(graph, self.tokenizer, self.model, 'cpu')
        graph.mark_pass('embedding_pass')
        payload = self.pipeline.export_pyg(graph, embedding_dim=self.model.config.hidden_size,
                                            use_security_edge_types=True)
        require(payload['y'] is None and payload['n_dropped_security_edges'] == 0,
                'Targets or dropped security edges in graph export')
        mask = torch.tensor(['embedding' in n.properties.extra for n in graph.nodes()], dtype=torch.float32)
        x = torch.tensor(payload['x'], dtype=torch.float32)
        vocab = payload['type_vocabulary']
        token_mask = x[:, vocab['node_types']['TOKEN']] == 1
        require(bool((token_mask & mask.bool()).any()), 'No token embeddings')
        item = {'x': torch.cat([x, mask[:, None]], dim=1),
            'edge_index': torch.tensor(payload['edge_index'], dtype=torch.long).reshape(2, -1),
            'edge_type': torch.tensor(payload['edge_type'], dtype=torch.long),
            'edge_features': torch.tensor(payload['edge_features'], dtype=torch.float32).reshape(-1, payload['edge_feature_dim']),
            'text_embedding': x[token_mask & mask.bool(), payload['num_node_types']:].mean(0),
            'cve': cve, 'description_sha256': text_digest(text)}
        validate_item(item, vocab, self.model.config.hidden_size)
        info = {'cve': cve, 'description_sha256': text_digest(text), 'description_characters': len(text),
            'stats': graph.stats(), 'embedded_nodes': int(mask.sum()), 'unembedded_nodes': int((mask == 0).sum()),
            'embedded_tokens': int((token_mask & mask.bool()).sum()),
            'embedding_dim': self.model.config.hidden_size, 'source_text_type': 'description',
            'passes': graph.passes_applied, 'graph_version': payload['graph_version']}
        return item, info, vocab, graph


def validate_item(item, vocab, dim):
    require(set(item) == {'x', 'edge_index', 'edge_type', 'edge_features', 'text_embedding', 'cve', 'description_sha256'},
            'Unexpected graph-cache fields (targets are forbidden)')
    from tpg.contracts import EDGE_FEATURE_DIM
    for field in ('node_types', 'edge_types'):
        require(sorted(vocab[field].values()) == list(range(len(vocab[field]))), 'Non-contiguous type vocabulary')
    require(vocab['num_node_types'] == len(vocab['node_types'])
            and vocab['num_edge_types'] == len(vocab['edge_types']) and vocab['use_security_edge_types'] is True,
            'Vocabulary dimensions/security scope differ')
    x, ei, et, ef = (item[k] for k in ('x', 'edge_index', 'edge_type', 'edge_features'))
    require(x.ndim == 2 and x.shape[0] > 0 and x.shape[1] == len(vocab['node_types'])+dim+1, 'Invalid node features')
    require(ei.ndim == 2 and ei.shape[0] == 2 and ei.dtype == torch.long, 'Invalid edge index')
    require(et.shape == (ei.shape[1],) and et.dtype == torch.long and ef.shape == (ei.shape[1], EDGE_FEATURE_DIM),
            'Invalid edge fields')
    require(item['text_embedding'].shape == (dim,), 'Invalid text embedding')
    require(all(torch.isfinite(item[k]).all().item() for k in ('x', 'edge_features', 'text_embedding')), 'Non-finite graph input')
    if ei.numel():
        require(int(ei.min()) >= 0 and int(ei.max()) < len(x), 'Dangling graph edges')
        require(int(et.min()) >= 0 and int(et.max()) < len(vocab['edge_types']), 'Unknown edge type')
    types = x[:, :len(vocab['node_types'])]
    require(((types == 0) | (types == 1)).all().item() and (types.sum(1) == 1).all().item(), 'Invalid node-type encoding')
    require(((x[:, -1] == 0) | (x[:, -1] == 1)).all().item(), 'Invalid embedding mask')
    require((x[x[:, -1] == 0, len(vocab['node_types']):-1] == 0).all().item(),
            'Unembedded nodes must have zero vectors')


def prepare(prepared, output, scale='smoke', check_only=False, allow_download=False):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    data, raw, source, pins = load_source(prepared)
    check_destination(output, [prepared, source.parent])
    ids = select_ids(data, scale)
    descriptions = {c: description_only(raw[c]) for s in ids for c in ids[s]}
    builder = GraphBuilder(allow_download=allow_download)
    selected = [c for split in ('train', 'validation') for c in ids[split]]
    if check_only:
        selected = [ids[s][0] for s in ('train', 'validation')]
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/'run_status.json', {'status': 'running', 'scale': scale, 'limitation': LIMITATION})
    try:
        stats, files, vocab = [], {}, None
        for i, cve in enumerate(tqdm(selected, desc='Description -> TPG -> frozen embeddings', unit='CVE')):
            item, info, current_vocab, graph = builder.build(descriptions[cve], cve)
            require(vocab is None or vocab == current_vocab, 'Graph vocabulary changed between CVEs')
            vocab = current_vocab
            stats.append(info)
            if not check_only:
                filename = cve+'.pt'
                torch.save(item, output/filename)
                files[filename] = sha256(output/filename)
                if i == 0 or cve == ids['validation'][0]:
                    name = cve+'.graphson.json'
                    builder.pipeline.export_graphson(graph, str(output/name))
                    files[name] = sha256(output/name)
        check_pins(pins)
        if check_only:
            for row in stats:
                print(f"{row['cve']}: {row['stats']['total_nodes']} nodes, {row['stats']['total_edges']} edges, "
                      f"{row['embedded_nodes']} embedded nodes; typed security edges="
                      f"{sum(n for kind,n in row['stats']['edge_types'].items() if kind.startswith('SEC_'))}")
            print('Preflight passed: two real description graphs, no training or files written. '+LIMITATION)
            return
        edge_counts = Counter()
        for row in stats:
            edge_counts.update(row['stats']['edge_types'])
        manifest = {'schema': VERSION, 'scale': scale, 'ids': ids, 'source_pins': pins,
            'prepared_directory': str(prepared), 'resources': builder.identity, 'type_vocabulary': vocab,
            'embedding_dim': builder.model.config.hidden_size, 'selection': 'fixed SHA-256 order of eligible CVE IDs; no outcome sorting',
            'predictor_fields': ['description'], 'structured_labels_in_graph': False,
            'test_graphs_built': False, 'split_changes_performed': False, 'limitation': LIMITATION,
            'edge_counts': dict(edge_counts), 'graphs': stats}
        write_json(output/'manifest.json', manifest)
        files['manifest.json'] = sha256(output/'manifest.json')
        report = ('# Retrospective Description TPG Cache\n\n'+LIMITATION+'\n\n'
            f"Scale: {scale}. CVEs: { {s:len(v) for s,v in ids.items()} }.\n"
            'Only description text enters the shared security TPG. CVE IDs are metadata, not appended to text.\n'
            'No SMP, summary, structured CVSS, VulnCheck or EPSS fields enter graph construction.\n'
            'Existing groups and splits remain unchanged. Test graphs are not constructed.\n'
            'Node inputs: node-type one-hot, frozen contextual embedding, embedding-present mask.\n'
            'Structural nodes without text embeddings keep zero vectors and a zero mask.\n'
            'Edges keep saved type IDs and shared 64-dimensional relation-property features.\n'
            'Security relations are rule-extracted candidates, not verified facts or learned relations.\n'
            'Descriptions themselves can contain later disclosures; field allowlisting cannot remove that risk.\n'
            'SecBERT pretraining history is another unverified historical information source.\n'
            'Text baseline pools embedded TOKEN nodes from the same frozen graph embeddings.\n'
            'Two GraphSON examples permit node, edge, span and provenance inspection.\n'
            'manifest.json records every graph size, embedding coverage, vocabulary and provenance.\n')
        for name in ('README.md', 'summary.txt'):
            (output/name).write_text(report, encoding='utf-8')
            files[name] = sha256(output/name)
        write_json(output/'run_status.json', {'status': 'complete', 'scale': scale})
        files['run_status.json'] = sha256(output/'run_status.json')
        write_json(output/'cache_lock.json', {'schema': VERSION, 'files': files})
        print(f'TPG cache saved to {output}. No training or test graphs.')
    except Exception as exc:
        if not check_only:
            write_json(output/'run_status.json', {'status': 'failed', 'error': str(exc)})
        raise


def load_cache(directory):
    directory = Path(directory).resolve()
    lock = read_json(directory/'cache_lock.json')
    require(lock['schema'] == VERSION, 'Unknown graph cache')
    require(set(p.name for p in directory.iterdir()) == set(lock['files']) | {'cache_lock.json'}, 'Unexpected/missing cache artifacts')
    for name, digest in tqdm(lock['files'].items(), desc='Verify graph cache hashes', unit='file'):
        require(Path(name).name == name and sha256(directory/name) == digest, f'Changed/invalid cache artifact: {name}')
    manifest = read_json(directory/'manifest.json')
    require(manifest['schema'] == VERSION and read_json(directory/'run_status.json')['status'] == 'complete', 'Incomplete cache')
    require(manifest['predictor_fields'] == ['description'] and manifest['structured_labels_in_graph'] is False
            and manifest['test_graphs_built'] is False and manifest['limitation'] == LIMITATION, 'Changed predictor scope')
    require(all(manifest['source_pins'].get(p) == h for p,h in code_pins().items()), 'Cache code identity differs')
    with tqdm(total=2, desc='Recheck original source and splits', unit='stage') as bar:
        check_pins(manifest['source_pins'])
        bar.update()
        data = contract.load_prepared(Path(manifest['prepared_directory']))
        bar.update()
    require(select_ids(data, manifest['scale']) == manifest['ids'], 'Cache cohort differs')
    expected_ids = [c for split in ('train', 'validation') for c in manifest['ids'][split]]
    require([row['cve'] for row in manifest['graphs']] == expected_ids, 'Graph-statistics identities differ')
    stats = {row['cve']:row for row in manifest['graphs']}
    require({n for n in lock['files'] if n.endswith('.pt')} == {c+'.pt' for c in expected_ids},
            'Unexpected/missing cached CVEs')
    items = {}
    for split in ('train', 'validation'):
        for cve in tqdm(manifest['ids'][split], desc=f'Load {split} graphs', unit='CVE'):
            item = torch.load(directory/(cve+'.pt'), map_location='cpu', weights_only=True)
            require(item['cve'] == cve, 'Graph identity differs')
            validate_item(item, manifest['type_vocabulary'], manifest['embedding_dim'])
            require(item['description_sha256'] == stats[cve]['description_sha256']
                    and len(item['x']) == stats[cve]['stats']['total_nodes']
                    and len(item['edge_type']) == stats[cve]['stats']['total_edges'], 'Graph statistics/source identity differ')
            items[cve] = item
    return manifest, items, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, default=PREPARED)
    parser.add_argument('--scale', choices=SCALES, default='smoke')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--allow-model-download', action='store_true', help='Explicit opt-in; otherwise use cached weights only')
    args = parser.parse_args()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        prepare(args.prepared_dir, args.output_dir or CACHE_ROOT/('description_'+args.scale+'_v1'),
                args.scale, args.check_only, args.allow_model_download)
    finally:
        torch.set_num_threads(threads)


if __name__ == '__main__':
    main()
