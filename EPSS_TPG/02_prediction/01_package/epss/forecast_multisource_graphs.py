"""MS02: bounded Day-60 multi-source graphs with shared frozen document tensors."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from . import forecast_later_development as development
from . import forecast_coreference_coverage as coverage
from . import forecast_coreference_cache as shared
from .forecast_coreference_development import check_encoder
import tpg_security_product_v4 as extension

torch, tqdm = shared.torch, shared.tqdm
read, write, require, sha256 = development.read, development.write, development.require, development.sha256
ROOT = development.ROOT
SCHEMA = 'ms02_multisource_graph_preview_v1'
CONFIG = ROOT/'02_prediction/04_configs/multisource_graph_preview_v1.json'
OUTPUT = ROOT/'01_tpg/05_workspace/multisource_day60_v1/graph_preview'
SCRIPT = shared.SCRIPTS/'prepare_multisource_graph_preview.py'
TEST = shared.TPG/'08_tests/test_multisource_graph_preview.py'
SOURCES = development.h01.sources.SOURCES
PRIOR = shared.base.CACHE_ROOT/'coreference_development_v2/full'
DESCRIPTION = shared.base.CACHE_ROOT/'description_full_v1'
H02 = ROOT/'05_results/02_evaluation/epss_day60_to_day90/history_controls_v1/full'
NUMERIC_FIELDS = ('x', 'edge_index', 'edge_type', 'edge_features', 'text_embedding')


def collect_documents(ids, rows, splits):
    require(set(ids) == {'train','validation'} and all(v and v == sorted(set(v)) for v in ids.values()),
            'Only unique sorted development IDs are allowed')
    require(set(ids['train']).isdisjoint(ids['validation']), 'CVE crosses roles')
    groups = [{splits[c]['group_id'] for c in ids[s]} for s in ids]
    require(groups[0].isdisjoint(groups[1]), 'Related group crosses roles')
    rows_by_id = {r['cve']:r for r in rows}
    require(len(rows_by_id) == len(rows), 'Duplicate source rows')
    result, entries = [], {}
    for split, chosen in ids.items():
        for c in chosen:
            r = rows_by_id[c]
            require(splits[c]['split'] == r['split'] == split, 'Wrong or test role')
            require(set(r) == {'cve','split','documents','source_mask'}, 'Unexpected source-input fields')
            keys = []
            doc_ids = set()
            for d in r['documents']:
                require(set(d) == {'document_id','source_text_type','text'}, 'Unexpected document fields')
                source = d['source_text_type']
                require(source in SOURCES and isinstance(d['text'],str) and d['text'].strip(), 'Unknown/empty source')
                require(isinstance(d['document_id'],str) and d['document_id'].startswith(source+':') and
                        d['document_id'] not in doc_ids, 'Duplicate or invalid source document ID')
                doc_ids.add(d['document_id'])
                key = shared.base.text_digest(c+'\0'+d['document_id'])
                require(key not in entries, 'Document-key collision')
                entries[key] = {'key':key, 'graph_id':'MS02-'+key, 'cve':c, 'split':split,
                                **d, 'text_sha256':shared.base.text_digest(d['text'])}
                keys.append(key)
            mask = [int(any(entries[k]['source_text_type'] == s for k in keys)) for s in SOURCES]
            require(mask == r['source_mask'], 'Source mask differs from documents')
            result.append({'cve':c,'split':split,'document_keys':keys,'source_mask':mask})
    return result,entries


def checked_lock(directory, name, expected=None):
    lock = read(directory/name)
    if expected is not None:
        require(set(lock['files']) == expected, 'Wrong prerequisite file set')
    pins = {str(directory/name):sha256(directory/name)}
    for n,h in lock['files'].items():
        path = (directory/n).resolve()
        require(directory.resolve() in path.parents, 'Unsafe prerequisite path')
        require(sha256(path) == h, 'Changed prerequisite: '+str(path))
        pins[str(path)] = h
    return pins


def context():
    # Hash-check the complete bundle, but read no EPSS values or numerical labels.
    pins = checked_lock(development.OUTPUT,'preparation_lock.json',development.FILES)
    original = read(development.OUTPUT/'run_identity.json')
    development.h01.sources.check_pins(original['source_pins'])
    pins.update(original['source_pins'])
    pins.update(checked_lock(H02,'evaluation_lock.json'))
    h02_identity = read(H02/'run_identity.json')
    require(h02_identity['scale'] == 'full' and h02_identity['ids'] == original['ids']
            and h02_identity['test_scoring'] is False, 'Incomplete/wrong H02 handoff')
    p = read(CONFIG)
    require(p['schema'] == SCHEMA and p['anchor_day'] == 60 and p['source_order'] == list(SOURCES)
            and p['edge_configuration'] == 'no_coref' and p['extractor'] == extension.VERSION
            and p['view'] == 'retrospective_date_screened'
            and not any(p[k] for k in ('training','full_expansion','test_scoring')), 'Wrong preview policy')
    ids = original['smoke_ids']
    require(ids == read(development.h01.OUTPUT/'run_identity.json')['ids'] and
            {s:len(v) for s,v in ids.items()} == p['preview_counts'] and
            all(set(ids[s]) <= set(original['ids'][s]) for s in ids), 'Changed preview CVEs')
    payload = read(development.OUTPUT/'source_inputs.json')
    require(payload['anchor_day'] == 60 and payload['source_order'] == list(SOURCES) and
            payload['view'] == p['view'], 'Wrong source cutoff/view')
    splits = read(development.OUTPUT/'split_manifest.json')
    associations, entries = collect_documents(ids,payload['rows'],splits)
    audit = {r['cve']:r for r in read(development.OUTPUT/'source_audit.json')}
    payload_by_id = {r['cve']:r for r in payload['rows']}
    for row in associations:
        c = row['cve']
        require(audit[c]['anchor_date'] == splits[c]['input_end_date'], 'Audit anchor changed')
        require(payload_by_id[c] == {'cve':c,'split':row['split'],
                **development.h01.sources.model_input(audit[c],list(SOURCES))}, 'Day-60 mask does not reproduce')
    prior_lock = read(PRIOR/'cache_lock.json')
    require(sha256(PRIOR/'manifest.json') == prior_lock['files']['manifest.json'], 'Changed encoder reference')
    prior = read(PRIOR/'manifest.json')
    old_lock = read(DESCRIPTION/'cache_lock.json')
    require(sha256(DESCRIPTION/'manifest.json') == old_lock['files']['manifest.json'], 'Changed NLP reference')
    nlp_hash = read(DESCRIPTION/'manifest.json')['resources']['spacy_pipeline_sha256']
    code_pins = shared.base.code_pins()
    # Include imported local helper modules: verification must not depend on unpinned code.
    for module in list(sys.modules.values()):
        path = getattr(module,'__file__',None)
        if path:
            path = Path(path).resolve()
            if path.suffix == '.py' and (shared.TPG.resolve() in path.parents or
                                         (ROOT/'02_prediction/01_package').resolve() in path.parents):
                code_pins[str(path)] = sha256(path)
    for path,h in code_pins.items():
        if path in prior['source_pins']:
            require(prior['source_pins'][path] == h, 'Frozen graph implementation changed: '+path)
    pins.update(code_pins)
    for path in (CONFIG,SCRIPT,TEST,Path(__file__),PRIOR/'manifest.json',PRIOR/'cache_lock.json',
                 DESCRIPTION/'manifest.json',DESCRIPTION/'cache_lock.json'):
        pins[str(path.resolve())] = sha256(path)
    comparisons = read(development.h01.sources.CONFIG)['comparisons']
    for enabled in comparisons.values():
        require(len(enabled) == len(set(enabled)) and set(enabled) <= set(SOURCES), 'Invalid comparison')
    identity = {'schema':SCHEMA,'ids':ids,'protocol':p,'source_pins':pins,'comparisons':comparisons,
                'source_order':list(SOURCES),'associations':associations,
                'documents':[{k:v for k,v in r.items() if k != 'text'} for r in entries.values()],
                'encoder':prior['encoder'],'embedding_dim':prior['embedding_dim'],
                'type_vocabulary':prior['type_vocabulary'],'spacy_pipeline_sha256':nlp_hash,
                'training':False,'test_graphs':False}
    shared.base.check_pins(pins)
    return {'identity':identity,'entries':entries}


def stamp_graph(graph, entry):
    require(graph.doc_id == entry['graph_id'] and graph.metadata['source_text'] == entry['text'], 'Graph text/id differs')
    graph.metadata['source_text_type'] = entry['source_text_type']
    graph.metadata['source_document_id'] = entry['document_id']
    for obj in [*graph.nodes(),*graph.edges()]:
        obj.properties.extra['source_text_type'] = entry['source_text_type']
    return graph


def validate_documents(documents, entry):
    require(set(documents) == set(shared.MODES), 'Missing graph view')
    for mode,saved in documents.items():
        graph = shared.import_graphson(saved)
        require(not graph.validate() and graph.doc_id == entry['graph_id'] and
                graph.metadata['source_text'] == entry['text'] and extension.VERSION in graph.passes_applied,
                'Invalid graph/source/policy')
        require(graph.metadata['source_text_type'] == entry['source_text_type'] and
                graph.metadata['source_document_id'] == entry['document_id'], 'Source graph identity differs')
        require(all(o.properties.extra.get('source_text_type') == entry['source_text_type']
                    for o in [*graph.nodes(),*graph.edges()]), 'Missing/wrong node or edge source label')
        require(all('embedding' not in n.properties.extra for n in graph.nodes()), 'Expected unembedded GraphSON')
        # The fixed v4 extractor inherits the v3 span/evidence validation.
        shared.preview.semantic.frozen.audit.preview.validate_graph(graph,entry['text'])
        require(graph.metadata['coreference_ablation']['mode'] == mode,'Wrong coreference view')


class Builder:
    def __init__(self, identity, allow_download=False):
        self.encoder = shared.FrozenEncoder(allow_download)
        check_encoder(self.encoder.identity,identity['encoder'])
        self.pipeline = extension.ProductVersionSecurityPipeline()
        nlp = self.pipeline.frontend.nlp
        require(self.pipeline.frontend.model_name == 'en_core_web_sm' and
                all(nlp.has_pipe(p) for p in ('parser','tagger','ner')), 'Required spaCy components missing')
        require(hashlib.sha256(nlp.to_bytes()).hexdigest() == identity['spacy_pipeline_sha256'], 'spaCy resource changed')

    def build(self, entry):
        graph = stamp_graph(self.pipeline.run(entry['text'],doc_id=entry['graph_id']),entry)
        saved = json.loads(shared.GraphSONExporter().export_string(graph))
        documents,_ = shared.preview.variants(saved)
        validate_documents(documents,entry)
        bundle,vocab = coverage.build_bundle(documents,self.encoder)
        return documents,bundle,vocab


def entry_files(key):
    require(len(key) == 64 and set(key) <= set('0123456789abcdef'), 'Unsafe document key')
    return {f'graphs/{key}.pt', *[f'graphson/{mode}/{key}.json' for mode in shared.MODES]}


def statistics_for(entry, documents, bundle, vocab):
    stats = shared.row_statistics(bundle,vocab)
    missing = coverage.missing_replacement_tokens(bundle,documents)
    graph = shared.import_graphson(documents['no_coref'])
    counts = Counter(e['label'] for e in documents['no_coref']['edges'])
    return {**{k:entry[k] for k in ('key','cve','split','document_id','source_text_type','text_sha256')},
            'characters':len(entry['text']), 'nodes':stats['nodes'],
            'embedded_nodes':stats['embedded_nodes'],'unembedded_nodes':stats['unembedded_nodes'],
            'embedded_by_node_type':stats['embedded_by_node_type'],
            'unembedded_whitespace_tokens':stats['unembedded_whitespace_tokens']-len(missing),
            'unembedded_replacement_tokens':missing,
            'edges_no_coref':stats['edges_by_mode']['no_coref'],'edge_type_counts':dict(sorted(counts.items())),
            'security_edges':sum(n for label,n in counts.items() if label.startswith('SEC_')),
            'source_span_tokens':len(list(graph.nodes(shared.NodeType.TOKEN)))}


def check_entry(directory, entry, identity):
    key = entry['key']
    receipt = read(directory/'receipts'/(key+'.json'))
    require(receipt['key'] == key and set(receipt['files']) == entry_files(key), 'Wrong document receipt')
    for n,h in receipt['files'].items():
        require(sha256(directory/n) == h, 'Changed completed document: '+n)
    documents = {m:read(directory/'graphson'/m/(key+'.json')) for m in shared.MODES}
    validate_documents(documents,entry)
    bundle = torch.load(directory/'graphs'/(key+'.pt'),weights_only=True,map_location='cpu')
    coverage.validate_bundle(bundle,documents,identity['type_vocabulary'],identity['embedding_dim'])
    stats = statistics_for(entry,documents,bundle,identity['type_vocabulary'])
    require(stats == receipt['statistics'],'Document statistics differ')
    return bundle,stats


def materialize_view(identity, bundles, comparison):
    """Read-only numeric graphs grouped by source; empty sources have no dummy graph."""
    require(comparison in identity['comparisons'], 'Unknown source comparison')
    enabled = identity['comparisons'][comparison]
    entries = {e['key']:e for e in identity['documents']}
    output = {}
    for row in identity['associations']:
        by_source = {s:[] for s in SOURCES}
        for key in row['document_keys']:
            source = entries[key]['source_text_type']
            if source in enabled:
                item = shared.materialize(bundles[key], 'no_coref')
                by_source[source].append({k:item[k] for k in NUMERIC_FIELDS})
        output[row['cve']] = {'by_source':by_source,
                             'source_mask':torch.tensor([bool(by_source[s]) for s in SOURCES],dtype=torch.bool)}
    return output


def audit_views(identity,bundles):
    rows = []
    for name in identity['comparisons']:
        view = materialize_view(identity,bundles,name)
        require(set(view) == {c for ids in identity['ids'].values() for c in ids}, 'Source view drops CVEs')
        for split,ids in identity['ids'].items():
            row = {'comparison':name,'split':split,'cves':len(ids),'graphs':0,'nodes':0,'edges':0,
                   'cves_with_any_text':sum(bool(view[c]['source_mask'].any()) for c in ids)}
            for c in ids:
                for i,s in enumerate(SOURCES):
                    items = view[c]['by_source'][s]
                    require(bool(items) == bool(view[c]['source_mask'][i]), 'View mask differs')
                    for item in items:
                        require(set(item) == set(NUMERIC_FIELDS), 'Non-numeric audit/target fields in graph view')
                        row['graphs'] += 1
                        row['nodes'] += len(item['x'])
                        row['edges'] += item['edge_index'].shape[1]
            rows.append(row)
    return rows


def analysis(identity, stats, bundles):
    coverage_rows = []
    for split,ids in identity['ids'].items():
        for source in SOURCES:
            rows = [r for r in stats if r['split'] == split and r['source_text_type'] == source]
            coverage_rows.append({'split':split,'source':source,'cves':len(ids),
                'cves_with_graphs':len({r['cve'] for r in rows}), 'documents':len(rows),
                'nodes':sum(r['nodes'] for r in rows),'embedded_nodes':sum(r['embedded_nodes'] for r in rows),
                'unembedded_nodes':sum(r['unembedded_nodes'] for r in rows),
                'edges':sum(r['edges_no_coref'] for r in rows),'security_edges':sum(r['security_edges'] for r in rows)})
    return {'schema':SCHEMA,'ids':identity['ids'],'coverage':coverage_rows,'documents':stats,
            'source_views':audit_views(identity,bundles), 'training':False,'test_scoring':False,
            'limitations':identity['protocol']['limitations']}


def report(a):
    lines = ['MS02: small multi-source TPG graph/cache preview', '',
             f"CVEs: { {s:len(v) for s,v in a['ids'].items()} }. Fixed H01 IDs and Day-60 masks.",
             'No training, EPSS prediction or test graphs. No full-expansion mode.', '',
             'Split | Source | CVEs with graphs | Documents | Nodes | Embedded nodes | Unembedded nodes | Edges | Security edges']
    for r in a['coverage']:
        lines.append(' | '.join(str(r[k]) for k in ('split','source','cves_with_graphs','documents','nodes',
                                                   'embedded_nodes','unembedded_nodes','edges','security_edges')))
    lines += ['', 'Each document has an independent graph. Source text and local node spans are unchanged.',
              'A node is globally identified by (document graph ID, local node ID); edges never cross documents.',
              'source_text_type appears on every node and edge. The fixed four-source order routes graphs for future pooling.',
              'Every document is embedded once. The same node tensors are reused in all source-removal views.',
              'Absent sources have an empty graph list and mask zero; no CVE is dropped and no empty graph is invented.',
              'Primary edges use the fixed no-COREF view. Three edge masks are stored for compatibility/audit, not as a new experiment.',
              'Security-edge counts include typed SEC_* arrows only; generic ENTITY_REL mirrors remain in total edge counts.',
              'SecBERT is frozen. Source-aligned token embeddings are mandatory except the existing audited whitespace/U+FFFD rules.',
              'Structural nodes without embeddings retain zero vectors and an explicit presence mask.',
              'No text truncation or fallback to description is allowed. Missing ordinary-token embeddings stop the run.', '',
              'Source-removal views (no models fitted)',
              'Comparison | Split | CVEs retained | CVEs with text | Documents | Nodes | Edges']
    for r in a['source_views']:
        lines.append(' | '.join(str(r[k]) for k in ('comparison','split','cves','cves_with_any_text','graphs','nodes','edges')))
    lines += ['', 'Files: graphson/*/*.json retains inspectable unembedded graphs; graphs/*.pt stores shared node features and edge masks.',
              'receipts/*.json enables checked document-level resume. manifest.json maps CVEs to source documents.',
              'analysis.json/.txt report sizes and coverage; cache_lock.json locks the completed preview.',
              'Legacy bundle fields named cve and description_sha256 carry document graph ID and document text hash; they are stripped from numeric model views.',
              'EPSS histories and targets are not present in graph tensors. Source audits/dates are not numeric graph inputs.',
              'All text remains historically unverified. Generated text can itself contain later events or inaccuracies.',
              'Next: review coverage, graph sizes, security edges and missing embeddings before any expansion or fusion training.', '',
              'Limitations']+['- '+v for v in a['limitations']]
    return '\n'.join(lines)+'\n'


def expected_files(identity):
    files = {'manifest.json','run_identity.json','run_status.json','analysis.json','analysis.txt','protocol.json'}
    for e in identity['documents']:
        files.update(entry_files(e['key']))
        files.add(f"receipts/{e['key']}.json")
    return files


def verify(directory=OUTPUT, ctx=None):
    ctx = context() if ctx is None else ctx
    identity = ctx['identity']
    lock = read(directory/'cache_lock.json')
    require(lock['schema'] == SCHEMA and set(lock['files']) == expected_files(identity),'Wrong/incomplete cache lock')
    require({str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()} ==
            expected_files(identity)|{'cache_lock.json'}, 'Unexpected/missing cache files')
    for n,h in tqdm(lock['files'].items(),desc='Verify multi-source artifact hashes',unit='file'):
        require(sha256(directory/n) == h,'Changed cache artifact: '+n)
    require(read(directory/'run_identity.json') == identity and read(directory/'manifest.json') == identity and
            read(directory/'protocol.json') == identity['protocol'] and
            read(directory/'run_status.json')['status'] == 'complete','Changed/incomplete cache identity')
    bundles,stats = {},[]
    for key,e in tqdm(ctx['entries'].items(),desc='Verify source graphs and embeddings',unit='document'):
        bundle,row = check_entry(directory,e,identity)
        bundles[key] = bundle
        stats.append(row)
    a = analysis(identity,stats,bundles)
    require(read(directory/'analysis.json') == a and (directory/'analysis.txt').read_text() == report(a),'Analysis differs')
    shared.base.check_pins(identity['source_pins'])
    return identity,bundles


def load_cache(comparison, directory=OUTPUT):
    """Verified numeric source view for a later consumer; never starts an encoder."""
    identity,bundles = verify(directory)
    return identity,materialize_view(identity,bundles,comparison)


def prepare(check_only=False, verify_only=False, resume=False, allow_download=False):
    require(sum((check_only,verify_only,resume)) <= 1,'Choose one mode')
    ctx = context()
    identity = ctx['identity']
    if verify_only:
        require(not allow_download,'Verification does not load/download models')
        verify(OUTPUT,ctx)
        print('MS02 multi-source preview verified. No parser/encoder rerun, training or test scoring.')
        return
    if resume:
        require(OUTPUT.is_dir() and not (OUTPUT/'cache_lock.json').exists() and
                read(OUTPUT/'run_identity.json') == identity,'Wrong or complete resume directory')
        actual = {str(p.relative_to(OUTPUT)) for p in OUTPUT.rglob('*') if p.is_file()}
        allowed = expected_files(identity)
        require(actual <= allowed|{p+'.tmp' for p in allowed},'Unexpected resume files')
    elif not check_only:
        require(not OUTPUT.exists(),'Output exists; use --verify-only or --resume')
    builder = Builder(identity,allow_download)
    if check_only:
        probes = []
        for source in SOURCES:
            candidates = [e for e in ctx['entries'].values() if e['split'] == 'train' and e['source_text_type'] == source]
            if candidates:
                probes.append(max(candidates,key=lambda e:(len(e['text']),e['key'])))
        for e in tqdm(probes,desc='Check longest training document per source',unit='document'):
            _,_,vocab = builder.build(e)
            require(vocab == identity['type_vocabulary'],'Vocabulary changed')
        shared.base.check_pins(identity['source_pins'])
        print(f'MS02 preflight passed: {len(ctx["entries"])} selected documents; {len(probes)} temporary graph/embedding checks. No files or training.')
        return
    if not resume:
        OUTPUT.mkdir(parents=True,exist_ok=False)
        write(OUTPUT/'run_identity.json',identity)
    write(OUTPUT/'run_status.json',{'status':'running'})
    try:
        bundles,stats = {},[]
        for key,e in tqdm(ctx['entries'].items(),desc='Build independent source graphs and shared embeddings',unit='document'):
            receipt = OUTPUT/'receipts'/(key+'.json')
            if not receipt.exists():
                documents,bundle,vocab = builder.build(e)
                require(vocab == identity['type_vocabulary'],'Vocabulary changed')
                for mode,doc in documents.items():
                    write(OUTPUT/'graphson'/mode/(key+'.json'),doc)
                path = OUTPUT/'graphs'/(key+'.pt')
                path.parent.mkdir(parents=True,exist_ok=True)
                temp = Path(str(path)+'.tmp')
                torch.save(bundle,temp)
                temp.replace(path)
                write(receipt,{'key':key,'statistics':statistics_for(e,documents,bundle,vocab),
                               'files':{n:sha256(OUTPUT/n) for n in sorted(entry_files(key))}})
            bundle,row = check_entry(OUTPUT,e,identity)
            bundles[key] = bundle
            stats.append(row)
        a = analysis(identity,stats,bundles)
        with tqdm(total=3,desc='Freeze preview and source-removal checks',unit='stage') as bar:
            write(OUTPUT/'manifest.json',identity)
            write(OUTPUT/'protocol.json',identity['protocol'])
            write(OUTPUT/'analysis.json',a)
            (OUTPUT/'analysis.txt').write_text(report(a),encoding='utf-8')
            bar.update()
            shared.base.check_pins(identity['source_pins'])
            bar.update()
            write(OUTPUT/'run_status.json',{'status':'complete'})
            write(OUTPUT/'cache_lock.json',{'schema':SCHEMA,'files':{n:sha256(OUTPUT/n) for n in sorted(expected_files(identity))}})
            bar.update()
        verify(OUTPUT,ctx)
    except Exception as exc:
        if not (OUTPUT/'cache_lock.json').exists():
            write(OUTPUT/'run_status.json',{'status':'failed','error':str(exc)})
        raise
    print(f'MS02 multi-source TPG preview saved to {OUTPUT}. No training, expansion or test scoring.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument('--check-only',action='store_true')
    flags.add_argument('--verify-only',action='store_true')
    flags.add_argument('--resume',action='store_true')
    parser.add_argument('--allow-model-download',action='store_true',help='Opt in to encoder downloads; offline by default')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    prepare(args.check_only,args.verify_only,args.resume,args.allow_model_download)
