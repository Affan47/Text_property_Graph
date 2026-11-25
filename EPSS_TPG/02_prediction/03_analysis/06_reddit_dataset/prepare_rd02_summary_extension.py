"""Development-only G/N summary overlay and collection queue; no fitting/network."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss.continuous_forecasting import data as core
from lock_rd01_sources import disclosure, normalize

ROOT = core.ROOT
REPO = ROOT.parent
CONFIG = ROOT / '02_prediction/04_configs/05_reddit_studies/RD02_summary_extension_v1.json'
OUTPUT_ROOT = ROOT / '05_results/07_reddit_studies/04_rd02_summary_extension_v1'
OUTPUT = OUTPUT_ROOT / 'preparation'
CVE = re.compile(r'\bCVE-\d{4}-\d{4,}\b', re.I)
EXPLICIT = re.compile(r'\bCVE\s+(?:identifier|ID)\s*(?:is\b|[:=])\s*[`*"\s]*(CVE-\d{4}-\d{4,})\b', re.I)
PLACEHOLDERS = {'n/a', 'none', 'null', 'nan', 'no summary', 'no summary available', 'error', '[]', '{}'}
INSUFFICIENT = re.compile(r'(?:no|not|without|insufficient).{0,55}(?:relevant|useful|information|content|details)|(?:failed to|unable to|could not).{0,35}(?:fetch|access|scrape)', re.I)


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def index_records(rows):
    core.require(isinstance(rows, list), 'Summary source must be a list of CVE records')
    result = {}
    for row in rows:
        core.require(isinstance(row, dict) and isinstance(row.get('cve'), str) and
                     CVE.fullmatch(row['cve']), 'Invalid summary source CVE')
        core.require(row['cve'] not in result, 'Duplicate summary CVE: ' + row['cve'])
        result[row['cve']] = row
    return result


def references(value):
    core.require(value is None or isinstance(value, list) and all(isinstance(x, str) for x in value),
                 'Reference field must be a list of strings or null')
    return sorted(set(value or []))


def reference_relation(previous, current):
    a, b = set(previous), set(current)
    if not a:
        return 'missing_original_references'
    if a == b:
        return 'exact_reference_set'
    if a <= b:
        return 'original_subset_of_current'
    return 'partial_reference_overlap' if a & b else 'no_reference_overlap'


def adapt_summary(row, old, source, fields, source_sha):
    """Only exact summary text enters predictors; all provenance stays in audit."""
    cve = row['cve']
    core.require(old is None or old['cve'] == cve, 'Cannot transfer summaries between CVEs')
    current = references(row['structured'][fields['references']])
    previous = references(old.get(fields['references'])) if old else []
    value = old.get(fields['text']) if old else None
    core.require(value is None or isinstance(value, str), f'{cve}: summary must be string/null')
    present = isinstance(value, str) and bool(value.strip())
    relation = reference_relation(previous, current)
    reasons, flags = [], []
    mentioned = []
    if old is None:
        reasons.append('no_matching_source_cve')
    elif not present:
        reasons.append('empty_summary')
    if present:
        if relation not in ('exact_reference_set', 'original_subset_of_current'):
            reasons.append('unmatched_reference_set')
        if normalize(value) in PLACEHOLDERS:
            reasons.append('placeholder_summary')
        cues = disclosure(value)
        if cues['quarantine']:
            reasons.append('epss_topic_disclosure')
        if cues['catalogue_cue']:
            flags.append('catalogue_text_mention_audit_only')
        normalized = value.translate(str.maketrans({chr(x): '-' for x in (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2212)}))
        mentioned = sorted(set(x.upper() for x in CVE.findall(normalized)))
        if cve not in mentioned:
            flags.append('target_id_not_literal_not_proof_off_target')
            if any(m.group(1).upper() != cve for m in EXPLICIT.finditer(normalized)):
                reasons.append('explicit_other_target_cve')
        if any(x != cve for x in mentioned):
            flags.append('mentions_other_cves_requires_context')
        if INSUFFICIENT.search(value):
            flags.append('insufficient_content_language_requires_review')
    audit = dict(cve=cve, split=row['split'], source=source, source_record_present=old is not None,
                 summary_present=present, accepted=present and not reasons,
                 reference_relation=relation, original_references=previous, current_references=current,
                 original_reference_count=len(previous), current_reference_count=len(current),
                 summary_field=fields['text'], source_file_sha256=source_sha,
                 text_sha256=text_hash(value) if present else None,
                 reasons=sorted(reasons), flags=sorted(flags), mentioned_cves=mentioned,
                 historical_availability_verified=False, factual_accuracy_verified=False)
    doc = None
    if audit['accepted']:
        doc = dict(document_id=source + '-' + audit['text_sha256'], source=source,
                   text=value, text_sha256=audit['text_sha256'], is_prose=False)
    return doc, audit


def overlay_rows(base, previous, fields, source_sha):
    overlays, audits = [], []
    for row in tqdm(sorted(base, key=lambda r: r['cve']), desc='Join and screen G/N summaries', unit='CVE'):
        core.require(row['split'] in ('train', 'validation'), 'Test adaptation is forbidden')
        docs = []
        for source in ('G', 'N'):
            doc, audit = adapt_summary(row, previous.get(row['cve']), source, fields[source], source_sha)
            audits.append(audit)
            if doc:
                docs.append(doc)
        overlays.append(dict(cve=row['cve'], split=row['split'], documents=docs,
                             mask_G=any(d['source'] == 'G' for d in docs),
                             mask_N=any(d['source'] == 'N' for d in docs)))
    core.require(len(overlays) == len({r['cve'] for r in base}), 'Repeated development CVE')
    return overlays, audits


def compose(base, overlay, enabled):
    """Future cache/model adapter: source metadata and target fields never pass through."""
    core.require(base['cve'] == overlay['cve'] and base['split'] == overlay['split'] and
                 base['split'] in ('train', 'validation'), 'Invalid summary join')
    core.require('D' in enabled and set(enabled) <= {'D', 'G', 'N', 'M_prose'}, 'Unknown view')
    docs = [d for d in base['documents'] if d['source'] == 'D' or
            d['source'] == 'M' and 'M_prose' in enabled and d['is_prose']]
    docs += [d for d in overlay['documents'] if d['source'] in enabled]
    core.require(all(set(d) == {'document_id', 'source', 'text', 'text_sha256', 'is_prose'} for d in docs),
                 'Unexpected predictor document fields')
    return dict(documents=docs, structured=base['structured'],
                source_mask=[any(d['source'] == s for d in docs) for s in ('D', 'G', 'N', 'M')])


def requests_for(base, audits):
    rows, invalid = [], []
    lookup = {r['cve']: r for r in base}
    for a in audits:
        if a['accepted']:
            continue
        urls = []
        for url in a['current_references']:
            try:
                parsed = urlsplit(url)
                valid = parsed.scheme in ('http', 'https') and bool(parsed.netloc) and not any(c.isspace() for c in url)
            except ValueError:
                valid = False
            if valid:
                urls.append(url)
            else:
                invalid.append(dict(cve=a['cve'], source=a['source'], url=url))
        if not urls:
            continue
        row = lookup[a['cve']]
        rows.append(dict(request_id=core.digest([a['cve'], a['source'], urls]), cve=a['cve'], split=a['split'],
                         source=a['source'], urls=urls,
                         target_description=next(d['text'] for d in row['documents'] if d['source'] == 'D'),
                         expected_summary_field=a['summary_field'], expected_reference_field=(
                             'github_only_urls' if a['source'] == 'G' else 'non_github_urls'),
                         reasons=a['reasons'], status='NOT_REQUESTED',
                         required_receipts=['URL', 'HTTP status', 'fetch time', 'content SHA256',
                                            'extracted text SHA256', 'model/version', 'prompt SHA256',
                                            'generation settings', 'summary SHA256']))
    return rows, invalid


def write_jsonl(path, rows):
    with path.open('wb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as f:
        for row in rows:
            f.write((json.dumps(row, sort_keys=True, allow_nan=False) + '\n').encode())


def csv_table(path, rows, columns):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def inventory(paths, roles):
    result = []
    for path in tqdm(sorted(set(paths)), desc='Inventory saved summary datasets', unit='file'):
        rows = index_records(core.read(path))
        fields = sorted({k for r in rows.values() for k in r if k.startswith('summ_')})
        counts = {}
        for split in ('train', 'validation'):
            subset = [r for c, r in rows.items() if roles.get(c) == split]
            counts[split] = dict(matching_cves=len(subset), nonempty_fields={
                k: sum(isinstance(r.get(k), str) and bool(r[k].strip()) for r in subset) for k in fields})
        result.append(dict(path=str(path.resolve()), sha256=core.sha(path), records=len(rows),
                           summary_fields=fields, development_coverage=counts,
                           automatically_merged=False))
    return result


def overlaps(base, overlays):
    groups = defaultdict(list)
    for b, o in zip(base, overlays):
        core.require(b['cve'] == o['cve'], 'Unordered join')
        for d in b['documents'] + o['documents']:
            if d['source'] == 'M' and not d['is_prose']:
                continue
            groups[text_hash(normalize(d['text']))].append(dict(cve=b['cve'], split=b['split'], source=d['source']))
    return [dict(normalized_sha256=k, associations=sorted(v, key=lambda r: (r['split'], r['cve'], r['source'])))
            for k, v in sorted(groups.items()) if {r['split'] for r in v} == {'train', 'validation'}
            and any(r['source'] in ('G', 'N') for r in v)]


def run(source, output, verify=False):
    cfg = core.read(CONFIG)
    core.require(cfg['schema'] == 'rd02_summary_extension_v1' and not cfg['training_enabled'] and
                 not cfg['test_enabled'] and not cfg['network_enabled'], 'Invalid preparation-only configuration')
    output, source = output.resolve(), source.resolve()
    core.require(OUTPUT_ROOT.resolve() in output.parents, 'Output must stay in the new summary-extension result directory')
    export = ROOT / cfg['base_export']
    manifest = ROOT / '05_results/03_dataset_analysis/02_reddit_dataset/08_rd01_source_lock_v1/rd01_frozen_manifest.json'
    approval = ROOT / '05_results/03_dataset_analysis/02_reddit_dataset/10_rd02_overlap_review_v1/source_approval.json'
    with tqdm(total=3, desc='Verify frozen development input identities', unit='stage') as bar:
        core.check_lock(export); identity = core.read(export / 'run_identity.json'); bar.update()
        task = core.read(manifest)['tasks']['RD02']; ids = identity['ids']
        core.require(ids == {s: task['ids'][s] for s in ('train', 'validation')} and
                     {s: len(cs) for s, cs in ids.items()} == {'train': 24213, 'validation': 972}, 'Changed frozen cohort')
        core.require(identity['approval_sha256'] == core.sha(approval), 'Changed source approval')
        with gzip.open(export / 'predictor_inputs.json.gz', 'rt') as f:
            base = sorted(json.load(f), key=lambda r: r['cve'])
        roles = {c: s for s, cs in ids.items() for c in cs}
        core.require(len(base) == len(roles) and all(roles.get(r['cve']) == r['split'] for r in base), 'Invalid base predictors')
        bar.update()
        source_sha = core.sha(source)
        if source == (REPO / cfg['default_summary_source']).resolve():
            core.require(source_sha == cfg['default_summary_sha256'], 'Default summary snapshot changed; use a separately versioned source')
        previous = index_records(core.read(source)); bar.update()
    paths = [source]
    data_dir = REPO / 'SummTPGVul/Sec4AI4Sec-EPSS/Data_Files'
    paths += list(data_dir.glob('*summaries*.json')) + list((data_dir / 'summ_samples').glob('*.json'))
    files = inventory(paths, roles)
    pins = {str(p.resolve()): core.sha(p) for p in [CONFIG, Path(__file__), Path(__file__).with_name('lock_rd01_sources.py'),
            Path(core.__file__), export / 'artifact_lock.json', manifest, approval,
            export / 'labels.json', export / 'structured_vectors.json', export / 'structured_state.json',
            ROOT / '02_prediction/04_configs/05_reddit_studies/RD02_spec.json']}
    pins.update({r['path']: r['sha256'] for r in files})
    core.require(pins[str(source)] == source_sha, 'Source changed during inventory')
    overlay, audits = overlay_rows(base, previous, cfg['fields'], source_sha)
    requests, invalid = requests_for(base, audits)
    duplicate_groups = overlaps(base, overlay)
    coverage = []
    for split, cs in ids.items():
        for s in ('G', 'N'):
            aa = [a for a in audits if a['split'] == split and a['source'] == s]
            coverage.append(dict(split=split, source=s, total_cves=len(cs),
                cves_with_reference_urls=sum(bool(a['current_references']) for a in aa),
                matching_old_cves=sum(a['source_record_present'] for a in aa),
                nonempty_saved_summaries=sum(a['summary_present'] for a in aa),
                candidate_summaries_after_screen=sum(a['accepted'] for a in aa),
                absent_or_quarantined=sum(not a['accepted'] for a in aa),
                collection_requests=sum(r['split'] == split and r['source'] == s for r in requests)))
    view_counts = {s: {v: Counter() for v in cfg['source_views']} for s in ids}
    for b, o in tqdm(zip(base, overlay), total=len(base), desc='Check matched source views and masks', unit='CVE'):
        for view, enabled in cfg['source_views'].items():
            adapted = compose(b, o, enabled)
            for name, active in zip(('D', 'G', 'N', 'M_prose'), adapted['source_mask']):
                view_counts[b['split']][view][name] += int(active)
        core.require(compose(b, o, ['D'])['documents'] == [d for d in b['documents'] if d['source'] == 'D'], 'D view changed')
    has_validation = {r['source']: r['candidate_summaries_after_screen'] for r in coverage if r['split'] == 'validation'}
    blocked = [s for s, n in has_validation.items() if n == 0]
    candidates = [a for a in audits if a['split'] == 'train' and a['summary_present']]
    candidates.sort(key=lambda a: (-bool(a['reasons']), -bool(a['flags']), a['cve'], a['source']))
    review = []
    for s in ('G', 'N'):
        for a in [a for a in candidates if a['source'] == s][:cfg['training_review_sample'] // 2]:
            review.append(dict(**a, text=previous[a['cve']][a['summary_field']], semantic_judgment='PENDING'))
    findings = dict(schema=cfg['schema'], stage='DEVELOPMENT_SUMMARY_JOIN_AND_COLLECTION_PREPARATION',
        cohort_counts={s: len(cs) for s, cs in ids.items()}, role_ids_sha256=task['role_ids_sha256'],
        summary_source=str(source), summary_source_sha256=source_sha, coverage=coverage,
        source_view_counts=view_counts, rejection_reasons=dict(Counter(x for a in audits for x in a['reasons'])),
        quality_flags=dict(Counter(x for a in audits for x in a['flags'])),
        reference_relations=dict(Counter(a['reference_relation'] for a in audits if a['summary_present'])),
        candidate_summary_cross_role_exact_groups=len(duplicate_groups),
        near_duplicate_review='PENDING after new summaries are collected; not established by exact matching',
        collection_requests=len(requests), invalid_reference_entries=len(invalid),
        readiness='BLOCKED_NO_VALIDATION_SUMMARIES' if blocked else 'SOURCE_REVIEW_CACHE_AND_SMOKE_REQUIRED',
        missing_validation_sources=blocked, training_enabled=False, network_requests=0,
        graphs_built=0, test_adapted=False, test_scored=False,
        limitations=cfg['limitations'])
    lines = ['RD02 WEBPAGE-SUMMARY EXTENSION: PREPARATION', '',
        'This is a new development overlay. Completed RD02 results and split IDs are unchanged.',
        'G means generated GitHub webpage summary; N means generated non-GitHub webpage summary.',
        'Reference URLs remain provenance/counts, not summary text. Shared URLs across CVEs are retained.',
        'No training, scraping, summary generation, graph construction or test evaluation was run.', '',
        'Coverage (CVEs, not individual scraped pages)',
        'Split | Source | CVEs | Existing summary | Candidate after screen | Collection requests']
    lines += [f"{r['split']} | {r['source']} | {r['total_cves']} | {r['nonempty_saved_summaries']} | {r['candidate_summaries_after_screen']} | {r['collection_requests']}" for r in coverage]
    lines += ['', 'Readiness: ' + findings['readiness'],
        'Zero validation coverage prevents measuring direct summary benefit. Do not move old CVEs into validation to solve it.',
        'Passed lexical/reference screening does not mean a summary is factually verified. The 16 training-only review cases remain pending.',
        'Six regenerated 50-record samples are inventoried separately; no LLM/temperature is selected from outcome performance.', '',
        'Proposed comparisons: pooled text and v6 TPG for D, DG, DN, DGN, DM_prose, DM_proseG, DM_proseN, DM_proseGN.',
        'Reuse the D/DM_prose reference fits only after exact model/input parity; six added views x two encoders x three seeds = 36 proposed new fits.',
        'All comparisons keep 24,213 train / 972 validation CVEs. Missing G/N means a false source mask, not CVE removal.',
        'Use identical structured vectors, labels, seeds and budgets. Keep source-level documents separate before attention fusion.', '',
        'Files',
        'summary_inputs.jsonl.gz: only development IDs, role, G/N document text and masks; combine with the frozen D/M export.',
        'summary_audit.jsonl.gz: source hashes, reference-set checks and quality flags; never feed this file to a model.',
        'collection_requests.jsonl.gz: missing G/N summaries with target CVE, D context and deduplicated same-CVE URLs; no scores or catalogue fields.',
        'training_review.json: 16 deterministically selected training-only summaries with pending human relevance judgments.',
        'source_inventory.json: full and sample-only saved sources, hashes and development coverage.',
        'cross_role_summary_overlap.json: normalized-exact train/validation overlaps involving G/N; no test text.',
        'base_artifact_references.json: pointers and hashes to unchanged labels and structured inputs; targets are separate.',
        'coverage.csv, findings.json and this report: counts and limitations.',
        'protocol.json and artifact_lock.json: proposed settings and immutable output hashes.', '',
        'Next: supply/generate summaries for development CVEs with URL references under one declared extraction/model policy.',
        'Preserve failed fetches, content hashes and prompts. Screen target-score disclosures and relevance, then review overlap before cache/smoke.',
        'No new temporal-text forecasting is authorized: historical availability is still unverified.']
    with core.transaction(output, verify) as stage:
        write_jsonl(stage / 'summary_inputs.jsonl.gz', overlay)
        write_jsonl(stage / 'summary_audit.jsonl.gz', audits)
        write_jsonl(stage / 'collection_requests.jsonl.gz', requests)
        for name, value in [('findings.json', findings), ('protocol.json', cfg), ('source_inventory.json', files),
                            ('training_review.json', review), ('cross_role_summary_overlap.json', duplicate_groups),
                            ('invalid_references.json', invalid), ('input_hashes.json', pins),
                            ('base_artifact_references.json', {str(export / f): core.sha(export / f) for f in (
                                'predictor_inputs.json.gz', 'labels.json', 'structured_vectors.json', 'structured_state.json', 'run_identity.json')})]:
            (stage / name).write_text(core.dump(value))
        csv_table(stage / 'coverage.csv', coverage, list(coverage[0]))
        (stage / 'REPORT.txt').write_text('\n'.join(lines) + '\n')
        core.check_pins(pins)
    print('RD02 summary preparation ' + ('reproduced' if verify else 'saved') + ': ' + str(output))
    print('Readiness: ' + findings['readiness'] + '. No training, network calls or test adaptation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-source', type=Path)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    source = args.summary_source or REPO / core.read(CONFIG)['default_summary_source']
    run(source, args.output_dir, args.verify_only)


if __name__ == '__main__':
    main()
