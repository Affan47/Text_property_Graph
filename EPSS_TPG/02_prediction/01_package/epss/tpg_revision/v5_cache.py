"""Exact-span v4 -> v5 cache adapter; no parser, encoder or target access in conversion."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile

import torch
from tqdm import tqdm

from ..vulncheck_study import graph_training_v1 as base
from . import subtypes_v5 as subtypes

data = base.data
ROOT = data.ROOT
OUTPUT = ROOT / '01_tpg/05_workspace/security_product_v5_subtypes_v1'
CLI = ROOT / '02_prediction/03_analysis/04_tpg_revision/02_run_v5_subtypes.py'


def publish_nested(directory, artifacts, verify=False):
    if verify:
        data.check_lock(directory)
        for name, value in artifacts.items():
            text = value if isinstance(value, str) else data.dump(value)
            data.require((directory / name).read_text() == text, 'Changed artifact: '+name)
        return
    data.require(not directory.exists(), 'Output exists; use --verify-only')
    directory.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.v5-publish-', dir=directory.parent))
    try:
        for name, value in artifacts.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value if isinstance(value, str) else data.dump(value))
        base.lock_tree(stage)
        data.require(not directory.exists(), 'Destination appeared')
        stage.rename(directory)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def convert(document, cached, resources):
    from .v5_audit import policy
    from tpg.exporters.exporters import PyGExporter, import_graphson
    old_nodes = document['vertices']
    node_row = {n['id']: i for i,n in enumerate(old_nodes)}
    vocab = resources['type_vocabulary']
    data.require(len(node_row) == cached.num_nodes, 'Node count mismatch')
    expected = PyGExporter().export(import_graphson(document), embedding_dim=0, use_security_edge_types=True)
    data.require(expected['type_vocabulary'] == vocab, 'Wrong saved vocabulary')
    data.require(torch.equal(cached.x[:, :vocab['num_node_types']], torch.tensor(expected['x'], dtype=torch.float32)), 'Original node types differ')
    for name in ('edge_index', 'edge_type', 'edge_features'):
        target = torch.tensor(expected[name], dtype=cached[name].dtype).reshape(cached[name].shape)
        data.require(torch.equal(target, cached[name]), 'Original graph/tensor mismatch: '+name)
    corrected = policy.correct(document)
    keep_rows = [node_row[n['id']] for n in corrected['vertices']]
    by_id = {n['id']: n for n in old_nodes}
    for n in corrected['vertices']:
        old = by_id[n['id']]
        data.require(n['label'] == old['label'] and policy.span(n) == policy.span(old)
                     and n['properties'].get('TEXT') == old['properties'].get('TEXT'), 'Changed span cannot reuse embedding')
    row_map = {n['id']: i for i,n in enumerate(corrected['vertices'])}
    edge_rows = {e['id']: i for i,e in enumerate(document['edges'])}
    data.require(len(edge_rows) == cached.num_edges, 'Duplicate/missing edge identity')
    keep_edges = [edge_rows[e['id']] for e in corrected['edges']]
    index = torch.tensor([[row_map[e['outV']] for e in corrected['edges']],
                          [row_map[e['inV']] for e in corrected['edges']]], dtype=torch.long).reshape(2,-1)
    item = {'x': cached.x[keep_rows].clone(), 'token_mask': cached.token_mask[keep_rows].clone(),
            'edge_index': index, 'edge_type': cached.edge_type[keep_edges].clone(),
            'edge_features': cached.edge_features[keep_edges].clone(), 'subtype': subtypes.encode_nodes(corrected['vertices'])}
    # No token is removed or re-embedded, including missing-token masks.
    data.require(torch.equal(item['x'][item['token_mask']], cached.x[cached.token_mask]), 'Token representations changed')
    data.require(torch.equal(item['x'], cached.x[keep_rows]), 'Embeddings or masks changed')
    data.require(item['subtype'].shape[0] == item['x'].shape[0] and torch.isfinite(item['x']).all(), 'Invalid subtype alignment')
    exported = PyGExporter().export(import_graphson(corrected), embedding_dim=0, use_security_edge_types=True)
    for name in ('edge_index', 'edge_type', 'edge_features'):
        check = torch.tensor(exported[name], dtype=item[name].dtype).reshape(item[name].shape)
        data.require(torch.equal(item[name], check), 'Corrected graph/tensor mismatch: '+name)
    return corrected, item


def context(scale):
    from . import v5_audit as audit
    data.require(scale in ('smoke', 'full'), 'Invalid scale')
    data.check_lock(audit.OUTPUT)
    data.check_pins(data.read(audit.OUTPUT / 'source_pins.json'))
    batches, targets, resources, dim, old = base.context('full')
    ids = deepcopy(old['ids'])
    if scale == 'smoke':
        old_smoke = data.read(ROOT / '01_tpg/05_workspace/vulncheck_description_v1/smoke/run_identity.json')['ids']
        reviewed = [r['cve'] for r in data.read(audit.OUTPUT / 'review.json')]
        ids = {'train': sorted(set(old_smoke['train']) | set(reviewed)), 'validation': old_smoke['validation']}
        data.require(set(ids['train']) <= set(old['ids']['train']) and set(ids['validation']) <= set(old['ids']['validation']), 'Wrong smoke roles')
    pins = {**old['pins'], str(audit.OUTPUT / 'artifact_lock.json'): data.sha(audit.OUTPUT / 'artifact_lock.json')}
    # Preserve the completed v4/TRV01 findings, not just the earlier backup snapshot.
    for directory in (ROOT / '05_results/06_tpg_revision/01_type_readout_v1/full',
                      ROOT / '05_results/05_vulncheck_study/07_description_graph_v1/full'):
        data.check_lock(directory)
        pins[str(directory / 'artifact_lock.json')] = data.sha(directory / 'artifact_lock.json')
    for path in (Path(__file__), Path(subtypes.__file__), Path(audit.policy.__file__), CLI):
        pins[str(path)] = data.sha(path)
    identity = {'schema': 'trv02_v5_subtypes_cache_v1', 'scale': scale, 'ids': ids,
                'vocab': subtypes.VOCAB, 'source_pins': pins, 'embedding_reuse': 'exact surviving node IDs and spans',
                'test_access': False, 'parser_rerun': False, 'encoder_rerun': False}
    print(f'[V5] {scale} cohort: { {k:len(v) for k,v in ids.items()} }; exact embedding reuse; no test inputs.', flush=True)
    return batches, targets, resources, dim, old, identity


def prepare(scale, verify=False):
    from ..forecast_matched_multisource import writer_lock
    batches, _, resources, _, _, identity = context(scale)
    output = OUTPUT / scale
    output.parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(output.parent / '.writer.lock'):
        if verify:
            data.check_lock(output)
            data.require(data.read(output / 'run_identity.json') == identity, 'Cache identity changed')
            stage = output
        else:
            data.require(not output.exists(), 'Cache exists; use --verify-only')
            stage = Path(tempfile.mkdtemp(prefix='.v5-cache-', dir=output.parent))
        try:
            index, events, subcounts = {}, Counter(), Counter()
            totals = Counter()
            for split, ids in identity['ids'].items():
                for cve in tqdm(ids, desc=f'Convert/verify {split} v5 TPGs', unit='CVE'):
                    index[cve] = deepcopy(batches.index[cve])
                    data.require(index[cve]['split'] == split, 'Wrong cached role')
                    for doc in index[cve]['documents']:
                        data.require(doc['source_text_type'] == 'description', 'Unexpected source')
                        for part in doc['chunks']:
                            if not part['has_graph']:
                                continue
                            graph = data.read(batches.store.directory / part['graph'])
                            original = batches.store.load(part['tensor'], 'no_coref')
                            corrected, item = convert(graph, original, resources)
                            graph_name, tensor_name = 'graphs/'+part['key']+'.json', 'tensors/'+part['key']+'.pt'
                            if verify:
                                data.require(data.read(stage / graph_name) == corrected, 'Saved corrected graph differs')
                                saved = torch.load(stage / tensor_name, map_location='cpu', weights_only=True)
                                data.require(set(saved) == set(item) and all(torch.equal(saved[k],v) for k,v in item.items()), 'Saved v5 tensor differs')
                            else:
                                for name in (graph_name, tensor_name):
                                    (stage / name).parent.mkdir(parents=True, exist_ok=True)
                                (stage / graph_name).write_text(data.dump(corrected))
                                torch.save(item, stage / tensor_name)
                            part.update(graph=graph_name, tensor=tensor_name)
                            events.update(r['rule'] for r in corrected['metadata']['v5_correction']['findings'] +
                                          corrected['metadata']['v5_correction']['edge_findings'])
                            subcounts.update(subtypes.NAMES[i] for i in item['subtype'].tolist())
                            totals.update(chunks=1, original_nodes=original.num_nodes, retained_nodes=len(item['x']),
                                          original_edges=original.num_edges, retained_edges=item['edge_index'].shape[1])
            summary = {'scale': scale, 'totals': dict(totals), 'correction_events': dict(events),
                       'subtypes': dict(subcounts), 'new_embeddings': 0, 'test_graphs': 0,
                       'interpretation': 'Narrow v5 corrections; residual extraction issues remain.'}
            artifacts = {'run_identity.json': identity, 'resources.json': resources, 'cve_index.json': index,
                         'summary.json': summary, 'summary.txt': 'V5 exact-span cache\n'+data.dump(summary)}
            for name, value in artifacts.items():
                text = value if isinstance(value,str) else data.dump(value)
                if verify:
                    data.require((stage/name).read_text() == text, 'Cache report differs')
                else:
                    (stage/name).write_text(text)
            data.check_pins(identity['source_pins'])
            if not verify:
                base.lock_tree(stage)
                data.require(not output.exists(), 'Destination appeared')
                stage.rename(output)
        finally:
            if not verify and stage.exists():
                shutil.rmtree(stage)
    print('V5 '+scale+' cache '+('verified.' if verify else f'saved to {output}.')+' No training or test scoring.')


class Store:
    def __init__(self, source, directory):
        self.directory, self.source = directory, source
        self.items, self.bytes = {}, 0

    def load(self, relative, view):
        from torch_geometric.data import Data
        data.require(view == 'no_coref', 'Only the frozen no-COREF view is prepared')
        if relative in self.items:
            return self.items[relative]
        path = (self.directory / relative).resolve()
        data.require(self.directory.resolve() in path.parents, 'Unsafe cache path')
        obj = Data(**torch.load(path, map_location='cpu', weights_only=True))
        size = sum(v.numel()*v.element_size() for _,v in obj if torch.is_tensor(v))
        if self.bytes + size > self.source.limit:
            self.items.clear()
            self.bytes = 0
        if size <= self.source.limit:
            self.items[relative] = obj
            self.bytes += size
        return obj
