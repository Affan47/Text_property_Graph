"""TRV05 immutable exact-reuse cache and matched description-only batches."""
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile

import torch
from tqdm import tqdm

from . import v6_audit as audit, v5_cache, subtypes_v5, subtypes_v6
from ..vulncheck_study import graph_training_v1 as base
from ..forecast_matched_multisource import writer_lock

data, ROOT = audit.data, audit.ROOT
CACHE = ROOT / '01_tpg/05_workspace/security_product_v6_subtypes_v1'
OUTPUT = ROOT / '05_results/06_tpg_revision/11_v6_comparison_v1'
CLI = ROOT / '02_prediction/03_analysis/04_tpg_revision/10_run_v6_comparison.py'
PARTITIONS = ('graph_repaired', 'severity_restored', 'unchanged')


def partition_ids(ids, differences):
    by_id = {r['cve']: r for r in differences}
    data.require(len(by_id) == len(differences), 'Duplicate audit CVE')
    result = {k: [] for k in PARTITIONS}
    for c in ids:
        r = by_id[c]
        graph = r['nodes_before'] != r['nodes_after'] or r['edges_before'] != r['edges_after']
        severity = r['severity_channels_restored'] > 0
        data.require(not (graph and severity), 'Approved disjoint partitions changed')
        result['graph_repaired' if graph else 'severity_restored' if severity else 'unchanged'].append(c)
    data.require(sum(map(len, result.values())) == len(ids), 'Partition coverage differs')
    return result


def contract(scale):
    data.require(scale in ('smoke', 'full'), 'Unknown scale')
    data.check_lock(audit.OUTPUT)
    spec = data.read(audit.OUTPUT / 'trv05_runner_spec.json')
    data.check_pins(spec['source_pins'])
    data.check_lock(audit.SOURCE)
    data.check_lock(audit.CONTROLS)
    differences = data.read(audit.OUTPUT / 'cve_diffs.json')
    ids = deepcopy(spec['ids'])
    parts = {s: partition_ids(values, differences) for s, values in ids.items()}
    data.require({k: len(v) for k, v in parts['validation'].items()} ==
                 {'graph_repaired': 16, 'severity_restored': 39, 'unchanged': 1049}, 'Validation groups changed')
    if scale == 'smoke':
        # Cover every repair group without inspecting scores or selecting model successes.
        original = data.read(v5_cache.OUTPUT / 'smoke/run_identity.json')['ids']
        ids = {s: sorted(set(original[s]) | {c for group in parts[s].values() for c in group[:2]}) for s in ids}
    settings = deepcopy(spec['settings'])
    if scale == 'smoke':
        settings.update(epochs=2, patience=2, batch_size=2, seeds=[17])
    pins = {**spec['source_pins'], str(audit.OUTPUT / 'artifact_lock.json'): data.sha(audit.OUTPUT / 'artifact_lock.json')}
    for path in (Path(__file__), CLI, v5_cache.OUTPUT / 'smoke/artifact_lock.json'):
        pins[str(path)] = data.sha(path)
    return {'schema': 'trv05_v6_cache_v1', 'scale': scale, 'ids': ids, 'settings': settings,
            'approved_spec_sha256': data.sha(audit.OUTPUT / 'trv05_runner_spec.json'),
            'partitions': {s: partition_ids(values, differences) for s, values in ids.items()},
            'source_pins': pins, 'subtype_vocabulary': subtypes_v6.VOCAB, 'test_access': False}


def prepare(scale, verify=False):
    identity = contract(scale)
    torch.set_num_threads(identity['settings']['threads'])
    index = data.read(audit.SOURCE / 'cve_index.json')
    resources = data.read(audit.SOURCE / 'resources.json')
    directory = CACHE / scale
    CACHE.mkdir(parents=True, exist_ok=True)
    with writer_lock(CACHE / '.writer.lock'):
        if verify:
            data.check_lock(directory)
            stage = directory
        else:
            data.require(not directory.exists(), 'Cache exists; use --verify-only')
            data.require(shutil.disk_usage(CACHE).free > 3 * 1024**3, 'Need at least 3 GiB free for cache publication')
            stage = Path(tempfile.mkdtemp(prefix='.incomplete-v6-cache-', dir=CACHE))
        try:
            selected, counts = {}, {'cves': 0, 'chunks': 0, 'nodes': 0, 'edges': 0}
            for split, ids in identity['ids'].items():
                for c in tqdm(ids, desc=f'Build/verify {split} exact v6 cache', unit='CVE'):
                    selected[c] = deepcopy(index[c])
                    data.require(selected[c]['split'] == split, 'Wrong cached split')
                    counts['cves'] += 1
                    for doc in selected[c]['documents']:
                        data.require(doc['source_text_type'] == 'description', 'D-only contract violated')
                        for part in doc['chunks']:
                            data.require(part['has_graph'], 'Missing graph')
                            graph = data.read(audit.source_path(part['graph']))
                            tensor = torch.load(audit.source_path(part['tensor']), map_location='cpu', weights_only=True)
                            corrected, item = audit.convert(graph, tensor, resources)
                            for path in (stage / part['graph'], stage / part['tensor']):
                                data.require(stage.resolve() in path.resolve().parents, 'Unsafe cache path')
                                if not verify:
                                    path.parent.mkdir(parents=True, exist_ok=True)
                            if verify:
                                data.require(data.read(stage / part['graph']) == corrected, 'Saved graph differs')
                                saved = torch.load(stage / part['tensor'], map_location='cpu', weights_only=True)
                                data.require(set(saved) == set(item) and all(torch.equal(saved[k], v) for k, v in item.items()),
                                             'Saved tensor differs')
                            else:
                                (stage / part['graph']).write_text(data.dump(corrected))
                                torch.save(item, stage / part['tensor'])
                            counts['chunks'] += 1
                            counts['nodes'] += len(item['x'])
                            counts['edges'] += item['edge_index'].shape[1]
            artifacts = {'run_identity.json': identity, 'resources.json': resources, 'cve_index.json': selected,
                         'summary.json': {'counts': counts, 'new_embeddings': 0, 'test_access': False}}
            for name, value in artifacts.items():
                if verify:
                    data.require((stage / name).read_text() == data.dump(value), 'Cache report differs: ' + name)
                else:
                    (stage / name).write_text(data.dump(value))
            data.check_pins(identity['source_pins'])
            if not verify:
                base.lock_tree(stage)
                data.require(not directory.exists(), 'Destination appeared')
                stage.rename(directory)
        finally:
            if not verify and stage.exists():
                shutil.rmtree(stage)
    print(f'TRV05 {scale} cache '+('verified.' if verify else f'saved to {directory}.')+' No new embeddings or test access.')


class OldSchemaStore:
    """Frozen-v5 graph intervention: new topology, old subtype meaning/width."""
    def __init__(self, source):
        self.source = source

    def load(self, path, view):
        graph = self.source.load(path, view).clone()
        graph.subtype[graph.subtype == subtypes_v6.VOCAB['SEVERITY']] = subtypes_v5.VOCAB['UNKNOWN']
        return graph


def load(scale):
    identity = contract(scale)
    directory = CACHE / scale
    data.check_lock(directory)
    data.require(data.read(directory / 'run_identity.json') == identity, 'Cache identity differs')
    source, targets, resources, dim, old_identity = base.context('full')
    data.require(old_identity['ids'] == data.read(audit.OUTPUT / 'trv05_runner_spec.json')['ids'], 'Development roles changed')
    data.require(data.read(directory / 'resources.json') == resources, 'Embedding/resources changed')
    index = data.read(directory / 'cve_index.json')
    rows = [source.rows[c] for split in identity['ids'].values() for c in split]
    new = base.network.Batches(directory, rows, index, resources, identity['settings'])
    new.store = v5_cache.Store(new.store, directory)
    old_index = data.read(audit.SOURCE / 'cve_index.json')
    old = base.network.Batches(audit.SOURCE, rows, {c: old_index[c] for c in index}, resources, identity['settings'])
    old.store = v5_cache.Store(old.store, audit.SOURCE)
    for c in index:
        data.require(new.rows[c] == old.rows[c], 'Predictor inputs differ beyond graphs/subtypes')
    pins = {**identity['source_pins'], **old_identity['pins'],
            str(directory / 'artifact_lock.json'): data.sha(directory / 'artifact_lock.json')}
    identity = {**identity, 'source_pins': pins}
    return new, old, targets, resources, dim, identity
