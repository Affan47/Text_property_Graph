"""Locate a real CVE/variant graph without running NLP, embeddings or training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / '01_tpg/10_extensions'))
sys.path.insert(0, str(ROOT / '02_prediction/01_package'))
from tpg_combined_v1.common import read, sha, require, check_files
from tpg_combined_datasets_v1.adapters import CACHES


def inspect(folder, cve, variant, export_json=None):
    folder = Path(folder)
    lock = read(folder / 'plan_lock.json')['files']
    require(sha(folder / 'cve_index.json') == lock['cve_index.json'], 'Index changed')
    index = read(folder / 'cve_index.json')
    require(cve in index and variant in index[cve]['views'], 'CVE/variant absent from this plan')
    view = index[cve]['views'][variant]
    result = {'cve': cve, 'variant': variant, 'split': index[cve]['split'], **view}
    if not view['object_key']:
        require(export_json is None, 'No accepted text: no graph to export')
        return {**result, 'reason': 'No accepted text for this variant; no fabricated graph.'}
    key = view['object_key']
    require(len(key) == 64 and all(x in '0123456789abcdef' for x in key), 'Invalid object key')
    obj = folder / 'objects' / key
    graph_path = obj / 'graph.json.gz'
    result.update(graph_path=str(graph_path), tensor_path=str(obj / 'tensor.pt'), graph_built=False)
    if (obj / 'graph_receipt.json').exists():
        receipt = read(obj / 'graph_receipt.json')
        expected = f'objects/{key}/graph.json.gz'
        require(set(receipt['files']) == {expected}, 'Unexpected graph receipt paths')
        check_files(folder, receipt['files'])
        result.update(statistics=receipt['statistics'], graph_built=True, graph_sha256=sha(graph_path))
    if export_json:
        require(result['graph_built'], 'Graph not built yet. Run the graphs stage before exporting.')
        target = Path(export_json).absolute()
        require(target.suffix == '.json' and not target.is_symlink(), 'Export must be a regular .json file')
        require(not target.resolve().is_relative_to(ROOT / '01_tpg/05_workspace') and
                not target.resolve().is_relative_to(folder.resolve()), 'Do not export into a cache directory')
        content = (json.dumps(read(graph_path), indent=2, ensure_ascii=True, allow_nan=False) + '\n').encode()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            require(target.read_bytes() == content, 'Different export already exists; choose another filename')
        else:
            with target.open('xb') as stream:
                stream.write(content)
        result.update(export_json=str(target), export_sha256=sha(target),
                      export_note='Readable inspection copy only; not a training cache or a new graph.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cve', required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--dataset', choices=CACHES, default='reddit')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--export-json', type=Path, help='Export verified GraphSON as indented plain JSON outside the cache')
    args = parser.parse_args()
    folder = args.cache_dir or CACHES[args.dataset] / args.scale
    result = inspect(folder, args.cve, args.variant, args.export_json)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
