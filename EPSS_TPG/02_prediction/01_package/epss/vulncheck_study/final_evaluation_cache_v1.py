"""VK10 separate cache, exact existing-cache references and unchanged frozen builders."""
from collections import Counter
from pathlib import Path
from tqdm import tqdm

from . import final_evaluation_data_v1 as prep, graph_cache_v1 as descriptions
from .. import forecast_primary_cache_v2 as builder
from ..forecast_matched_multisource import writer_lock

data = prep.data


def context(scale, release=False):
    protocol, prepared, rows, _ = prep.load(scale, release)
    resources = data.read(prep.DEVELOPMENT_CACHE/'resources.json')
    lock = data.read(prep.DEVELOPMENT_CACHE/'artifact_lock.json')
    data.require(data.sha(prep.DEVELOPMENT_CACHE/'resources.json') == lock['files']['resources.json'], 'Changed resources')
    data.check_pins(resources['source_pins'])
    roots = [prep.DEVELOPMENT_CACHE, data.ROOT/'01_tpg/05_workspace/vulncheck_description_v1/full',
             data.ROOT/'01_tpg/05_workspace/primary_day60_v2/full',
             data.ROOT/'01_tpg/05_workspace/primary_day60_final_v1/full']
    sources = []
    for root in roots:
        if not (root/'artifact_lock.json').exists():
            continue
        old_lock = data.read(root/'artifact_lock.json')
        data.require(data.sha(root/'resources.json') == old_lock['files']['resources.json'], 'Changed reuse resources')
        data.require(data.read(root/'resources.json') == resources, 'Reuse processing identity differs')
        sources.append((root, old_lock))
    plan = prep.graph_plan(rows, prepared['role'], resources['protocol']['max_chunk_characters'])
    identity = {'schema': prep.SCHEMA, 'stage': 'cache', 'scale': scale, 'ids': prepared['ids'],
                'data_lock_sha256': data.sha(prep.DATA/scale/'artifact_lock.json'),
                'plan_sha256': data.digest(plan), 'resources_sha256': data.digest(resources),
                'pins': {**prepared['pins'], **{str(r/'artifact_lock.json'): data.sha(r/'artifact_lock.json') for r, _ in sources}},
                'training': False, 'test_graphs': scale == 'full'}
    return protocol, rows, resources, plan, identity, sources


def authenticate_references(references, plan, sources):
    allowed = {str(root.resolve()): (root, lock) for root, lock in sources}
    data.require(set(references) <= set(plan['entries']), 'Unknown referenced chunk')
    tensor_roots = {}
    for key, name in tqdm(references.items(), desc='Authenticate exact external chunks', unit='chunk'):
        data.require(name in allowed, 'Unpinned cache root')
        root, lock = allowed[name]; entry = plan['entries'][key]
        data.require(descriptions.reusable(root, lock, entry), 'Changed referenced entry')
        receipt = data.read(root/('receipts/'+key+'.json'))
        for relative, digest in receipt['files'].items():
            path = (root/relative).resolve()
            data.require(root.resolve() in path.parents and lock['files'].get(relative) == digest and
                         data.sha(path) == digest, 'Changed/unsafe referenced chunk file')
        tensor_roots['tensors/'+entry['stem']+'.pt'] = name
    return tensor_roots


def prepare(scale, release=False, workers=4, resume=False, verify=False):
    builder.torch.set_num_threads(1)
    data.require(1 <= workers <= 16, 'Workers must be 1..16')
    protocol, rows, resources, plan, identity, sources = context(scale, release)
    output = prep.CACHE/scale
    output.parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(output.parent/'.writer.lock'):
        if output.exists():
            data.require(verify or resume, 'Cache exists; use --resume or --verify-only')
            data.require(data.read(output/'run_identity.json') == identity, 'Cache identity changed')
            if verify or (output/'artifact_lock.json').exists():
                data.check_lock(output); verify = True
        else:
            data.require(not verify, 'No cache to verify')
            output.mkdir(); builder.atomic_json(output/'run_identity.json', identity)
            builder.atomic_json(output/'resources.json', resources)
        ref_path = output/'chunk_references.json'
        references = data.read(ref_path) if ref_path.exists() else {}
        provenance_path = output/'chunk_provenance.json'
        provenance = data.read(provenance_path) if provenance_path.exists() else {}
        tokenizer = builder.tokenizer_for(resources)
        if not verify:
            missing = []
            for key, entry in tqdm(plan['entries'].items(), desc='Resolve frozen evaluation chunk identities', unit='chunk'):
                if key in references:
                    continue
                if (output/('receipts/'+key+'.json')).exists():
                    builder.check_entry(output, entry, resources, tokenizer)
                    provenance.setdefault(key, 'verified_resumed_chunk')
                    continue
                candidate = next((root for root, lock in sources if descriptions.reusable(root, lock, entry)), None)
                if candidate is None:
                    missing.append(entry)
                else:
                    references[key] = str(candidate.resolve())
                    provenance[key] = 'external_exact_reuse:'+str(candidate.relative_to(data.ROOT))
            builder.atomic_json(ref_path, references); builder.atomic_json(provenance_path, provenance)
            print(f'[VK10 CACHE] {len(plan["entries"])} chunks; {len(references)} referenced; {len(missing)} to build.', flush=True)
            if missing:
                builder.bounded_build(output, missing, resources, workers, provenance)
            for path, document in plan['documents'].items():
                builder.atomic_json(output/path, document)
            builder.atomic_json(output/'cve_index.json', plan['index'])
        authenticate_references(references, plan, sources)
        audit = []
        for entry in tqdm(plan['entries'].values(), desc='Verify evaluation spans, embeddings and relations', unit='chunk'):
            root = Path(references[entry['key']]) if entry['key'] in references else output
            _, statistics = builder.check_entry(root, entry, resources, tokenizer)
            graph = data.read(root/('tpg_json/no_coref/'+entry['stem']+'.json'))
            edges = [e for e in graph['edges'] if e['label'].startswith('SEC_')]
            for edge in edges:
                extra = edge['properties'].get('EXTRA', {})
                data.require(extra.get('status') == 'candidate' and extra.get('source_text_type') == entry['source_text_type'], 'Security provenance changed')
            settings = protocol['training_settings']
            data.require(len(graph['vertices']) <= settings['graph_node_budget'] and len(graph['edges']) <= settings['graph_edge_budget'],
                         'Unprocessable graph; abort, never drop CVE: '+entry['cve'])
            audit.append({'cve': entry['cve'], 'key': entry['key'], 'source': entry['source_text_type'],
                          'statistics': statistics, 'security_edges': dict(Counter(e['label'] for e in edges))})
        data.require(set(provenance) == set(plan['entries']), 'Missing chunk provenance')
        summary = {'cves': len(rows), 'chunks': len(audit), 'referenced_chunks': len(references),
                   'chunks_by_source': dict(Counter(a['source'] for a in audit)),
                   'new_local_chunks': len(audit)-len(references), 'training': False, 'scale': scale}
        if verify:
            data.require(data.read(output/'cve_index.json') == plan['index'] and data.read(output/'resources.json') == resources, 'Cache metadata differs')
            for path, document in plan['documents'].items():
                data.require(data.read(output/path) == document, 'Document changed')
            data.require(data.read(output/'security_audit.json') == audit and data.read(output/'summary.json') == summary, 'Cache audit differs')
        else:
            builder.atomic_json(output/'security_audit.json', audit)
            builder.atomic_json(output/'summary.json', summary)
            builder.common.lock(output)
        data.check_pins(identity['pins'])
    print(f'VK10 {scale} evaluation cache '+('verified.' if verify else 'saved.')+' No fitting or scoring.')


def load(scale, release=False):
    protocol, rows, resources, plan, identity, sources = context(scale, release)
    output = prep.CACHE/scale; data.check_lock(output)
    data.require(data.read(output/'run_identity.json') == identity and data.read(output/'cve_index.json') == plan['index'], 'Cache identity/index changed')
    roots = authenticate_references(data.read(output/'chunk_references.json'), plan, sources)
    return protocol, rows, resources, plan['index'], roots, identity
