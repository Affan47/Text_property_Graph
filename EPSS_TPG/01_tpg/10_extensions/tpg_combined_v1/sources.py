from collections import Counter
import gzip
import hashlib
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from .common import ROOT, MODEL, REVISION, SCHEMA, VARIANTS, ORDER
from .common import read, write, sha, digest, require, check_files, pins

INPUT_ROOT = ROOT / '04_data/03_reddit_collection/04_combined_text_inputs_v1'
BASE = INPUT_ROOT / '01_development'
OVERLAY = INPUT_ROOT / '02_summary_overlay'


def migrate_inputs():
    """Preserve verified inputs before the user-authorized removal of old runs."""
    if (INPUT_ROOT / 'migration_lock.json').exists():
        check_files(INPUT_ROOT, read(INPUT_ROOT / 'migration_lock.json')['files'])
        print('Preserved development inputs verified; no old results are required.')
        return
    pairs = [(ROOT / '05_results/07_reddit_studies/02_rd02_primary_v1/export/full', BASE),
             (ROOT / '05_results/07_reddit_studies/04_rd02_summary_extension_v1/preparation', OVERLAY)]
    manifest = {}
    for source, destination in pairs:
        check_files(source, read(source / 'artifact_lock.json')['files'])
        for path in tqdm(sorted(source.iterdir()), desc='Preserve verified predictor/label inputs', unit='file'):
            require(path.is_file() and not path.is_symlink(), 'Unexpected source input layout')
            target = destination / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                require(sha(target) == sha(path), 'Conflicting migrated input')
            else:
                shutil.copyfile(path, target)
            require(sha(target) == sha(path), 'Input copy mismatch')
            manifest[str(target.relative_to(INPUT_ROOT))] = sha(target)
    write(INPUT_ROOT / 'migration_lock.json', {'files': manifest,
          'purpose': 'Immutable development inputs, labels kept separate; original runs may be retired.'})
    print(f'Input migration verified: {INPUT_ROOT}')


def combine(documents, enabled, smp_policy):
    """Join original strings, recording exact offsets without inserting fake prose."""
    selected = [d for d in documents if d['source'] in enabled and
                (d['source'] != 'M' or smp_policy == 'all' or d['is_prose'])]
    selected.sort(key=lambda d: (ORDER.index(d['source']), d['document_id']))
    text, spans = '', []
    for d in selected:
        require(d['source'] in ORDER and isinstance(d['text'], str) and d['text'].strip(), 'Invalid source text')
        require(hashlib.sha256(d['text'].encode()).hexdigest() == d['text_sha256'], 'Source text hash mismatch')
        if spans:
            text += '\n\n'
        start = len(text)
        text += d['text']
        spans.append({'source': d['source'], 'document_id': d['document_id'], 'start': start,
                      'end': len(text), 'text_sha256': d['text_sha256']})
    return text, spans


def make_plan(scale, variants, smp_policy):
    require(scale in ('smoke', 'full') and smp_policy in ('all', 'prose'), 'Invalid plan options')
    require(variants and len(set(variants)) == len(variants) and set(variants) <= VARIANTS.keys(), 'Invalid variants')
    for folder in (BASE, OVERLAY):
        check_files(folder, read(folder / 'artifact_lock.json')['files'])
    rows = read(BASE / 'predictor_inputs.json.gz')
    identity = read(BASE / 'run_identity.json')
    with gzip.open(OVERLAY / 'summary_inputs.jsonl.gz', 'rt') as stream:
        summaries = [json.loads(line) for line in stream]
    overlay = {r['cve']: r for r in summaries}
    require(len(rows) == len({r['cve'] for r in rows}) == len(summaries) == len(overlay), 'Duplicate/missing CVEs')
    require({s: sorted(r['cve'] for r in rows if r['split'] == s) for s in ('train', 'validation')} ==
            {s: sorted(cs) for s, cs in identity['ids'].items()}, 'Development identities differ')
    require(Counter(r['split'] for r in rows) == {'train': 24213, 'validation': 972}, 'Unexpected full cohort')
    if scale == 'smoke':
        # Deterministic coverage sample, without reading outcomes.
        chosen = set()
        for split, count in (('train', 4), ('validation', 2)):
            pool = sorted((r for r in rows if r['split'] == split), key=lambda r: digest(r['cve']))
            for source in ('G', 'N') if split == 'train' else ():
                candidate = next((r for r in pool if r['cve'] not in chosen and
                                  any(d['source'] == source for d in overlay[r['cve']]['documents'])), None)
                if candidate:
                    chosen.add(candidate['cve'])
            for r in pool:
                if sum(x['cve'] in chosen for x in pool) >= count:
                    break
                chosen.add(r['cve'])
        rows = [r for r in rows if r['cve'] in chosen]
    processing = {'schema': SCHEMA, 'model': MODEL, 'revision': REVISION, 'window_tokens': 1024,
                  'stride': 0, 'separator': '\n\n', 'smp_policy': smp_policy, 'pins': pins(),
                  'graph_policy': 'v6 security; no COREF; one combined document, no graph chunking',
                  'embedding_policy': 'float32 mean of overlapping subwords per node span; full overflow coverage',
                  'max_characters': 500000, 'device_policy': 'one fixed device per cache, eager attention'}
    entries, index, coverage = {}, {}, {}
    for r in tqdm(sorted(rows, key=lambda x: x['cve']), desc='Plan combined-text variants', unit='CVE'):
        require(r['split'] in ('train', 'validation') and overlay[r['cve']]['split'] == r['split'], 'Test/role mismatch')
        documents = r['documents'] + overlay[r['cve']]['documents']
        require(set(d['source'] for d in r['documents']) <= {'D', 'M'} and
                set(d['source'] for d in overlay[r['cve']]['documents']) <= {'G', 'N'}, 'Unexpected source')
        views = {}
        for variant in variants:
            text, spans = combine(documents, VARIANTS[variant], smp_policy)
            mask = {s: any(p['source'] == s for p in spans) for s in ORDER}
            entry = {'cve': r['cve'], 'split': r['split'], 'text': text, 'spans': spans,
                     'processing_identity': digest(processing)}
            key = digest(entry) if spans else None
            if key:
                require(len(text) <= processing['max_characters'], f'{r["cve"]}: long input requires explicit review')
                entries[key] = entry
            views[variant] = {'object_key': key, 'source_mask': mask,
                              'all_requested_sources_present': all(mask[s] for s in VARIANTS[variant])}
            counts = coverage.setdefault(r['split'], {}).setdefault(variant, Counter())
            counts.update(cves=1, nonempty=int(bool(key)), **{s: int(mask[s]) for s in ORDER})
        index[r['cve']] = {'split': r['split'], 'views': views}
    paths = [BASE / x for x in ('artifact_lock.json', 'run_identity.json', 'predictor_inputs.json.gz',
                               'structured_vectors.json', 'structured_state.json')]
    paths += [OVERLAY / x for x in ('artifact_lock.json', 'summary_inputs.jsonl.gz')]
    return {'schema': SCHEMA, 'scale': scale, 'variants': variants, 'processing': processing,
            'input_hashes': {str(p): sha(p) for p in paths}, 'entries': entries, 'index': index,
            'coverage': coverage, 'structured_vectors_path': str(BASE / 'structured_vectors.json'),
            'labels_path_not_loaded': str(BASE / 'labels.json'), 'test_adapted': False,
            'summary_ablation_status': 'BLOCKED_NO_VALIDATION_SUMMARIES',
            'interpretation': 'Retrospective text representation; no historical availability claim.'}


def prepare(directory, scale, variants, smp_policy, resume):
    directory = Path(directory)
    plan = make_plan(scale, variants, smp_policy)
    if (directory / 'plan_lock.json').exists():
        require(resume, 'Existing plan: use --resume to check exact parity')
        check_files(directory, read(directory / 'plan_lock.json')['files'])
        require(read(directory / 'plan.json.gz') == plan, 'Plan differs; select a new cache directory')
    else:
        require(not directory.exists() or not any(directory.iterdir()), 'Incomplete/unrecognized plan directory')
        write(directory / 'plan.json.gz', plan)
        write(directory / 'cve_index.json', plan['index'])
        write(directory / 'coverage.json', plan['coverage'])
        files = ['plan.json.gz', 'cve_index.json', 'coverage.json']
        for number, variant in enumerate(variants, 1):
            name = f'variants/{number:02d}_{variant}/index.json'
            write(directory / name, {c: {'split': row['split'], **row['views'][variant]}
                                     for c, row in plan['index'].items()})
            files.append(name)
        write(directory / 'plan_lock.json', {'files': {p: sha(directory / p) for p in files}})
    print(f'Combined plan: {len(plan["index"])} CVEs, {len(plan["entries"])} distinct graphs. No test inputs.')
    print('URL-summary training comparison remains blocked: no accepted validation G/N summaries.')
