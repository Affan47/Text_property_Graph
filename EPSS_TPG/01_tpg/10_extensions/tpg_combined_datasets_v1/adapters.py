"""Separate cohorts and train-fitted features; one graph/embedding contract."""
from collections import Counter
import gzip
import hashlib
import math
from pathlib import Path
import re
import sys

from tqdm import tqdm
from tpg_combined_v1 import common as c, sources
from epss.reddit_primary import structured

ROOT = c.ROOT
sys.path.insert(0, str(ROOT / '02_prediction/03_analysis/06_reddit_dataset'))
from lock_rd01_sources import disclosure

OLD = ROOT / '04_data/01_records_and_graphs/primary_day60_v1/full'
OLD_TEST_IDENTITY = ROOT / '04_data/01_records_and_graphs/primary_day60_final_v1/full/run_identity.json'
RAW = ROOT.parent / 'SummTPGVul/Sec4AI4Sec-EPSS/Data_Files/epss_recollected_2026-09-30_smp.json'
RAW_SHA = '506ae7da7c9202ca132e4e7ff8511f1fbbea9ab534630374dde5370e0ecfd43c'
EXPORT = ROOT / '04_data/05_shared_architecture_inputs/02_previous_multisource_v1'
CACHES = {'reddit': c.DEFAULT,
          'previous': ROOT / '01_tpg/05_workspace/02_previous_multisource/01_combined_securebert2_v1'}
SOURCE_NAMES = {'description': 'D', 'smp': 'M', 'github_summary': 'G', 'non_github_summary': 'N'}
DEFAULT_VARIANTS = ['D', 'M', 'U', 'DM', 'DU', 'DMU']


def adapter_pins():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [Path(structured.__file__),
              ROOT / '02_prediction/01_package/epss/forecast_structured_features.py',
              ROOT / '02_prediction/03_analysis/06_reddit_dataset/lock_rd01_sources.py',
              ROOT / '01_tpg/02_examples/01_scripts/build_dataset_tpgs.py']
    return {str(p): c.sha(p) for p in sorted(paths)}


def publish(directory, payloads, resume):
    """A completed export is immutable; resume must reproduce every byte."""
    directory = Path(directory)
    encoded = {}
    for name, value in payloads.items():
        content = value.encode() if name.endswith('.txt') else c.encoded(value)
        encoded[name] = gzip.compress(content, mtime=0) if name.endswith('.gz') else content
    expected = {name: hashlib.sha256(value).hexdigest() for name, value in encoded.items()}
    lock = directory / 'artifact_lock.json'
    if lock.exists():
        c.require(resume, 'Existing export: use --resume to verify, not overwrite')
        c.require(c.read(lock)['files'] == expected, 'Export changed; use a versioned adapter/output')
        c.check_files(directory, expected)
        return
    # Interrupted writes can be completed, but never overwrite different bytes.
    for name, content in encoded.items():
        path = directory / name
        if path.exists():
            c.require(c.sha(path) == expected[name], f'Conflicting partial output: {path}')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + '.tmp')
            tmp.write_bytes(content)
            tmp.replace(path)
    c.write(lock, {'files': expected})


def canonical_document(document):
    source = SOURCE_NAMES[document['source_text_type']]
    text = document['text']
    c.require(isinstance(text, str) and text.strip(), 'Empty/non-text document')
    return {'document_id': document['document_id'], 'source': source, 'text': text,
            'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'is_prose': source != 'M' or re.fullmatch(r'CVE-\d{4}-\d+', text.strip(), re.I) is None}


def validate_roles(rows, expected, test_ids):
    ids = [r['cve'] for r in rows]
    c.require(len(ids) == len(set(ids)), 'Duplicate development CVE')
    c.require(set(r['split'] for r in rows) <= {'train', 'validation'}, 'Non-development input')
    actual = {s: sorted(r['cve'] for r in rows if r['split'] == s) for s in ('train', 'validation')}
    c.require(actual == {s: sorted(v) for s, v in expected.items()}, 'Frozen development roles changed')
    c.require(not (set(ids) & set(test_ids)), 'Test CVE in development export')


def validate_labels(labels, rows):
    expected = {r['cve']: r['split'] for r in rows}
    c.require(len(labels) == len(expected) == len({r['cve'] for r in labels}), 'Label count/identity mismatch')
    for label in labels:
        c.require(label['cve'] in expected and label['split'] == expected[label['cve']], 'Label role mismatch')
        value = label['value']
        c.require(label['observed'] is True and label['target_day'] == 60 and
                  type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                  'Missing/invalid Day-60 target; no target imputation permitted')


def source_coverage(rows):
    coverage = {}
    for split in ('train', 'validation'):
        group = [r for r in rows if r['split'] == split]
        coverage[split] = {'cves': len(group)}
        for source in ('D', 'M', 'G', 'N', 'U'):
            enabled = 'GN' if source == 'U' else source
            coverage[split][source] = sum(any(d['source'] in enabled and
                (d['source'] != 'M' or d['is_prose']) for d in r['documents']) for r in group)
    return coverage


def prepare_previous(resume=False):
    c.check_files(OLD, c.read(OLD / 'artifact_lock.json')['files'])
    c.require(c.sha(RAW) == RAW_SHA, 'Older raw dataset differs from the verified source')
    saved = c.read(OLD / 'inputs.json')['rows']
    identity = c.read(OLD / 'run_identity.json')
    test_ids = c.read(OLD_TEST_IDENTITY)['ids']  # IDs only, never test labels/text.
    validate_roles(saved, identity['ids'], test_ids)
    c.require(Counter(r['split'] for r in saved) == {'train': 2546, 'validation': 1104}, 'Older cohort changed')
    raw_records = c.read(RAW)
    raw = {r['cve']: r for r in raw_records}
    c.require(len(raw) == len(raw_records) == 5968, 'Unexpected older raw identities')
    rows, quarantine = [], []
    for r in tqdm(sorted(saved, key=lambda r: r['cve']), desc='Adapt older screened development texts', unit='CVE'):
        record = raw[r['cve']]
        documents = []
        for original in r['documents']:
            doc = canonical_document(original)
            flags = disclosure(doc['text'])
            if flags['quarantine']:
                quarantine.append({'cve': r['cve'], 'split': r['split'], 'document_id': doc['document_id'],
                                   'source': doc['source'], 'text_sha256': doc['text_sha256'],
                                   'reason': 'EPSS topic/answer disclosure screen', 'flags': flags})
            else:
                documents.append(doc)
        c.require(len({d['document_id'] for d in documents}) == len(documents), 'Duplicate document identity')
        fields = {key: record.get(key) for key in structured.FIELDS}
        for key in ('cvss_vector', 'cvss_vector_version', 'cvss_vector_score'):
            c.require(fields[key] == r['structured'][key], f'Raw/prepared CVSS mismatch: {r["cve"]}/{key}')
        rows.append({'cve': r['cve'], 'split': r['split'], 'structured': fields,
                     'documents': documents, 'source_mask': {s: any(d['source'] == s for d in documents) for s in c.ORDER}})
    labels = c.read(OLD / 'labels.json')
    validate_labels(labels, rows)
    state = structured.fit([r for r in rows if r['split'] == 'train'])
    vectors, issues = {}, {}
    for r in tqdm(rows, desc='Encode older structured fields with training-only statistics', unit='CVE'):
        vectors[r['cve']], flags = structured.transform(r['structured'], state)
        if flags:
            issues[r['cve']] = flags
    source_paths = [RAW, OLD_TEST_IDENTITY, OLD / 'artifact_lock.json', OLD / 'inputs.json',
                    OLD / 'labels.json', OLD / 'run_identity.json']
    provenance = {'source_hashes': {str(p): c.sha(p) for p in source_paths}, 'adapter_hashes': adapter_pins(),
                  'dataset': 'previous', 'raw_cves': len(raw), 'ids': identity['ids'], 'test_ids_count': len(test_ids),
                  'test_adapted': False, 'interpretation': 'Retrospective Day-60 EPSS estimation'}
    findings = {'source_coverage': source_coverage(rows), 'additional_quarantined_documents': len(quarantine),
                'quarantined_cves': len({r['cve'] for r in quarantine}), 'structured_dimensions': len(state['feature_names']),
                'structured_fields': list(structured.FIELDS), 'structured_issues': issues,
                'changes': ['Original source screening retained; current EPSS disclosure screen applied additionally.',
                            'Reference-count features aligned with Reddit: deduplicated all/GitHub/non-GitHub URLs.',
                            'Separate training-fitted preprocessing; no pooled statistics across datasets.']}
    publish(EXPORT, {'predictor_inputs.json.gz': rows, 'labels.json': labels,
                     'structured_vectors.json': vectors, 'structured_state.json': state,
                     'run_identity.json': provenance, 'audit.json': findings, 'quarantine.json': quarantine}, resume)
    print(f'Older dataset export verified: {len(rows)} development CVEs; {len(quarantine)} additionally screened documents.')


def previous_rows():
    c.check_files(EXPORT, c.read(EXPORT / 'artifact_lock.json')['files'])
    identity = c.read(EXPORT / 'run_identity.json')
    c.require(identity['adapter_hashes'] == adapter_pins(), 'Dataset adapter changed; version the export')
    for path, expected in identity['source_hashes'].items():
        c.require(c.sha(path) == expected, f'Source changed: {path}')
    return c.read(EXPORT / 'predictor_inputs.json.gz')


def select_smoke(rows):
    selected = set()
    for split, count in (('train', 4), ('validation', 2)):
        pool = sorted((r for r in rows if r['split'] == split), key=lambda r: c.digest(r['cve']))
        for source in ('G', 'N'):
            candidate = next((r for r in pool if r['cve'] not in selected and
                              any(d['source'] == source for d in r['documents'])), None)
            if candidate:
                selected.add(candidate['cve'])
        for r in pool:
            if sum(x['cve'] in selected for x in pool) >= count:
                break
            selected.add(r['cve'])
    return [r for r in rows if r['cve'] in selected]


def make_previous_plan(scale, variants, smp_policy):
    c.require(scale in ('smoke', 'full') and smp_policy in ('prose', 'all'), 'Invalid plan settings')
    c.require(variants and len(set(variants)) == len(variants) and set(variants) <= c.VARIANTS.keys(), 'Invalid variants')
    rows = previous_rows()
    if scale == 'smoke':
        rows = select_smoke(rows)
    # One frozen processing contract, inherited rather than reimplementing the graph/encoder settings.
    reference = c.DEFAULT / 'full'
    processing = dict(c.load_plan(reference)['processing'])
    processing['smp_policy'] = smp_policy
    entries, index, coverage = {}, {}, {}
    for r in tqdm(rows, desc='Plan older combined-text variants', unit='CVE'):
        views = {}
        for variant in variants:
            text, spans = sources.combine(r['documents'], c.VARIANTS[variant], smp_policy)
            mask = {s: any(p['source'] == s for p in spans) for s in c.ORDER}
            entry = {'cve': r['cve'], 'split': r['split'], 'text': text, 'spans': spans,
                     'processing_identity': c.digest(processing)}
            key = c.digest(entry) if spans else None
            if key:
                c.require(len(text) <= processing['max_characters'], f'{r["cve"]}: long text requires review')
                entries[key] = entry
            views[variant] = {'object_key': key, 'source_mask': mask,
                              'all_requested_sources_present': all(mask[s] for s in c.VARIANTS[variant])}
            counts = coverage.setdefault(r['split'], {}).setdefault(variant, Counter())
            counts.update(cves=1, nonempty=int(bool(key)), **{s: int(mask[s]) for s in c.ORDER})
        index[r['cve']] = {'split': r['split'], 'views': views}
    paths = [EXPORT / p for p in ('artifact_lock.json', 'run_identity.json', 'predictor_inputs.json.gz',
                                 'structured_vectors.json', 'structured_state.json')]
    paths += [reference / 'plan_lock.json']
    inputs = {str(p): c.sha(p) for p in paths}
    inputs.update(adapter_pins())
    return {'schema': c.SCHEMA, 'dataset': 'previous', 'scale': scale, 'variants': variants,
            'processing': processing, 'input_hashes': inputs, 'entries': entries, 'index': index,
            'coverage': coverage, 'structured_vectors_path': str(EXPORT / 'structured_vectors.json'),
            'labels_path_not_loaded': str(EXPORT / 'labels.json'), 'test_adapted': False,
            'summary_ablation_status': 'VALIDATION_SUMMARIES_PRESENT_QUALITY_NOT_GUARANTEED',
            'interpretation': 'Retrospective text representation; preserved older development roles; no historical availability claim.'}


def prepare_plan(dataset, scale, variants, smp_policy, resume=False, directory=None):
    c.require(dataset in CACHES, 'Unknown dataset')
    directory = Path(directory) if directory else CACHES[dataset] / scale
    if dataset == 'reddit':
        return sources.prepare(directory, scale, variants, smp_policy, resume)
    prepare_previous(resume=resume)
    plan = make_previous_plan(scale, variants, smp_policy)
    lock = directory / 'plan_lock.json'
    if lock.exists():
        c.require(resume, 'Existing plan: use --resume')
        c.check_files(directory, c.read(lock)['files'])
        c.require(c.read(directory / 'plan.json.gz') == plan, 'Plan differs; use another cache directory')
    else:
        c.require(not directory.exists() or not any(directory.iterdir()), 'Nonempty unrecognized cache')
        payloads = {'plan.json.gz': plan, 'cve_index.json': plan['index'], 'coverage.json': plan['coverage']}
        for n, variant in enumerate(variants, 1):
            payloads[f'variants/{n:02d}_{variant}/index.json'] = {
                key: {'split': r['split'], **r['views'][variant]} for key, r in plan['index'].items()}
        for name, value in payloads.items():
            c.write(directory / name, value)
        c.write(lock, {'files': {name: c.sha(directory / name) for name in payloads}})
    print(f'Older dataset plan: {len(plan["index"])} CVEs; {len(plan["entries"])} distinct graphs; test excluded.')
