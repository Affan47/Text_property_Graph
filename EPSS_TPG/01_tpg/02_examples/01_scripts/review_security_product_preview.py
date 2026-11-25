"""Verify DQ10 smoke output and reproduce extraction without training or test data."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import tempfile

import audit_security_product_versions as audit


def canonical(row):
    """Baseline entity insertion order can vary; retain all values and evidence."""
    result = deepcopy(row)
    for side in ('before', 'after'):
        for name in ('software', 'versions', 'typed_security_relations'):
            result[side][name].sort(key=lambda value: audit.json.dumps(value, sort_keys=True))
    return result


def statistics(rows):
    audit.require(len({r['cve'] for r in rows}) == len(rows), 'Duplicate CVEs')
    audit.require(all(r['split'] in ('train', 'validation') for r in rows), 'Forbidden split')
    stats = {'cves': len(rows), 'by_split': dict(Counter(r['split'] for r in rows))}
    for field in ('old_version_links', 'new_version_links', 'unresolved_versions',
                  'typed_relation_endpoints_changed'):
        stats[field] = sum(r[field] for r in rows)
    for side in ('before', 'after'):
        stats[side] = {
            'recognized_versions': sum(len(r[side]['versions']) for r in rows),
            'cves_with_version_links': sum(any(e['relation'] == 'SEC_HAS_VERSION'
                for e in r[side]['typed_security_relations']) for r in rows),
            'relation_counts': dict(Counter(e['relation'] for r in rows
                for e in r[side]['typed_security_relations']))}
    stats['cves_with_unresolved_owners'] = sum(r['unresolved_versions'] > 0 for r in rows)
    stats['unresolved_reasons'] = dict(Counter(v['abstention'] for r in rows
        for v in r['after']['versions'] if v['abstention']))
    stats['version_operators'] = dict(Counter(e['evidence']['version_operator'] for r in rows
        for e in r['after']['typed_security_relations'] if e['relation'] == 'SEC_HAS_VERSION'))
    stats['non_version_relation_change_cves'] = [r['cve'] for r in rows
        if [e for e in audit.relation_keys(r['before']) if e[0] != 'SEC_HAS_VERSION'] !=
           [e for e in audit.relation_keys(r['after']) if e[0] != 'SEC_HAS_VERSION']]
    return stats


# Source-based review observations, not an exhaustive labelled benchmark.
OBSERVATIONS = [
    ('high', 'CVE-2022-24042', 'false_owner', 'Desigo DXR2 (All versions',
     'Four version links name All as the product. The actual products are Desigo DXR2, '
     'PXC3, PXC4 and PXC5. Capitalization and adjacency do not establish product identity.'),
    ('medium', 'CVE-2021-2163', 'incomplete_owner', 'Oracle GraalVM Enterprise Edition: 19.3.5',
     'The new owner is Enterprise Edition. Its saved sentence evidence starts at Enterprise; '
     'the parser boundary separates it from Oracle GraalVM. Same-sentence checks alone '
     'cannot ensure the complete product name.'),
    ('low', 'CVE-2024-48964', 'overbroad_owner', 'The package Snyk CLI before 1.1294.0',
     'The owner includes the generic word package. The product is Snyk CLI; this is a '
     'name-boundary issue, not a link to a different named product.'),
    ('medium', 'CVE-2021-33318', 'missing_bound', 'IpMatcher 1.0.4.1 and below',
     'The edge keeps mentioned_version but omits and below from its qualifier. '
     'It does not assert an incorrect bound; it fails to represent a stated bound.'),
    ('medium', 'CVE-2022-27790', 'missing_bound', '22.001.20085 (and earlier)',
     'The parenthesized upper bound is not captured. The later wildcard release is '
     'not recognized and the last numeric release has no owner.'),
    ('medium', 'CVE-2024-8190', 'incomplete_release', '4.6 Patch 518 and before',
     'The structured version is only 4.6. Patch 518 and the upper bound remain in '
     'the original description but not in the version/qualifier representation.'),
    ('medium', 'CVE-2020-28600', 'unsupported_release', 'openscad-2020.12-RC2',
     'No new version node is created for this product-prefixed release. The old graph '
     'had a version node but no owner link. Conservative rejection loses coverage here.'),
    ('medium', 'CVE-2022-30696', 'unsupported_release', 'before build 3640',
     'Integer-only build 3640 is not recognized. Zero unresolved owners does not mean '
     'that all releases in the description were extracted.'),
    ('medium', 'CVE-2023-1966', 'unsupported_release', 'Universal Copy Service v1.x',
     'Neither wildcard release is recognized. These missed mentions are not counted '
     'among recognized versions with unresolved owners.'),
]


def findings(rows):
    indexed = {r['cve']: r for r in rows}
    result = []
    for priority, cve, kind, excerpt, explanation in OBSERVATIONS:
        row = indexed[cve]
        audit.require(excerpt in row['description'], f'Review excerpt changed: {cve}')
        result.append({'priority': priority, 'cve': cve, 'kind': kind,
                       'source_excerpt': excerpt, 'explanation': explanation,
                       'after': row['after']})
    return result


def report(result):
    s = result['statistics']
    lines = ['# DQ10 Smoke Review', '',
        'The completed smoke output is reproducible, but the extractor needs another correction.',
        'Do not expand v1 or create a training cache yet. Integrity is not extraction accuracy.', '',
        '## Scope and Counts', '',
        f"Inspected {s['cves']} CVEs: {s['by_split']}. No test descriptions or model scoring.",
        'The fixed sample has 32 training and 16 validation CVEs. Eight further known validation '
        'cases were added; two known training cases already occur in the fixed sample.',
        'This is an engineering sample with known regression cases, not an independent accuracy benchmark.', '',
        '| Measure | Count |', '|---|---:|',
        f"| Old typed product/version links | {s['old_version_links']} |",
        f"| New typed product/version links | {s['new_version_links']} |",
        f"| CVEs whose typed relation endpoints changed | {s['typed_relation_endpoints_changed']} |",
        f"| Recognized versions without a supported owner | {s['unresolved_versions']} |",
        f"| CVEs containing those unresolved versions | {s['cves_with_unresolved_owners']} |", '',
        'Each link count counts the typed SEC_HAS_VERSION copy once, not its generic duplicate. '
        'More candidate links does not establish better accuracy. Unrecognized releases are not '
        'included in the unresolved-owner count.', '',
        '## Useful Corrections', '',
        'The earlier Pluck-CMS, full WordPress plugin/theme names, ESET product name and '
        'python-jose examples have the intended source-bound links. All ten previously '
        'reviewed cases are retained in analysis.json with their complete before/after evidence.', '',
        '## Problems Exposed by the Smoke Run', '']
    for f in result['findings']:
        lines += [f"### {f['cve']}: {f['kind']} ({f['priority']} priority)",
                  f"Source: `{f['source_excerpt']}`", f['explanation'], '']
    lines += ['## What Was Verified', '',
        'The exact artifact set, checksums, current source/code pins and DQ09 prerequisite were checked. '
        'A temporary replay rebuilt all 56 selected descriptions using the pinned code and NLP '
        'resource, checking against the frozen source cache without generating embeddings. '
        'All comparison rows agree after sorting entity/relation lists to ignore insertion order. '
        'All 56 new GraphSON objects agree exactly. No evidence fields or spans were discarded.',
        f"Non-version typed relation changes: {s['non_version_relation_change_cves']}.",
        'No training, parameter updates, new forecasts or test inspection. The original preview '
        'and its pinned files were checked again after reproduction and left unchanged.', '',
        '## Recommended Next Step', '',
        '1. Preserve v1 and add a versioned correction with focused negative controls for generic '
        'owners, product names split by the parser, and product qualifiers.',
        '2. Preserve complete release/build wording and supported bounds. Explicitly flag '
        'unsupported forms; do not force ownership or silently shorten a release.',
        '3. Check lists, ranges and parentheses with both correct-owner and wrong-owner examples. '
        'A lower link count with explicit abstention is preferable to fabricated ownership.',
        '4. Repeat the small preview before expanding to all development descriptions. Newly '
        'reviewed cases become engineering regression cases, not unseen evaluation examples.',
        '5. Only then specify a model-ready graph cache and how version metadata reaches the GNN. '
        'The current 64-value edge encoding does not encode the new version-bound metadata.', '']
    return '\n'.join(lines)


def run(source, output, check_only=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    summary = audit.read_json(source/'analysis.json')
    audit.verify_smoke(source, summary['pinned_files'])
    audit.cache.check_pins(summary['pinned_files'])
    audit.verify_review()
    source_lock = audit.sha256(source/'preview_lock.json')
    if check_only:
        saved = audit.read_json(output/'analysis.json')
        lock = audit.read_json(output/'review_lock.json')
        audit.require(set(lock) | {'review_lock.json'} ==
            {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}, 'Review file set changed')
        for name, digest in lock.items():
            audit.require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe review path')
            audit.require(audit.sha256(output/name) == digest, 'Review artifact changed')
        audit.require(saved['source_lock_sha256'] == source_lock and saved['reproduced'] is True,
                      'Review source identity changed')
        audit.require(saved['reviewer_sha256'] == audit.sha256(__file__), 'Reviewer changed')
        print('DQ10 saved review checks passed. No reproduction, training or test inspection.'); return
    audit.cache.check_destination(output, [source, audit.TPG/'01_core', audit.TPG/'10_extensions',
        audit.TPG/'08_tests', audit.cache.CACHE_ROOT, audit.cache.PREPARED,
        audit.REVIEW, audit.DIAGNOSTIC, audit.EPSS.parent/'SummTPGVul'])
    rows = audit.read_json(source/'comparisons.json')
    audit.require([r['cve'] for r in rows] == summary['inspected_ids'], 'Selection mismatch')
    with tempfile.TemporaryDirectory(prefix='dq10_review_') as temporary:
        replay = Path(temporary)/'replay'
        audit.run('smoke', replay)
        audit.require(audit.read_json(replay/'analysis.json') == summary, 'Replay selection or pins differ')
        repeated = audit.read_json(replay/'comparisons.json')
        audit.require([canonical(r) for r in rows] == [canonical(r) for r in repeated],
                      'Replayed extraction differs')
        for row in audit.tqdm(rows, desc='Verify saved GraphSON reproduction', unit='CVE'):
            name = f"graphs/{row['cve']}.json"
            audit.require(audit.read_json(source/name) == audit.read_json(replay/name),
                          f'GraphSON changed: {row["cve"]}')
    result = {'schema': 'dq10_smoke_review_v1', 'source': str(source),
        'source_lock_sha256': source_lock, 'reviewer_sha256': audit.sha256(__file__),
        'reproduced': True, 'training': False, 'test_inspection': False,
        'decision': 'hold_v1_expansion_pending_versioned_correction',
        'statistics': statistics(rows), 'findings': findings(rows),
        'previously_reviewed_examples': [r for r in rows if r['previously_reviewed']]}
    audit.verify_smoke(source, summary['pinned_files'])
    audit.cache.check_pins(summary['pinned_files'])
    audit.require(audit.sha256(source/'preview_lock.json') == source_lock, 'Source lock changed')
    output.mkdir(parents=True, exist_ok=False)
    audit.write_json(output/'analysis.json', result)
    for name in ('analysis.txt', 'README.md'):
        (output/name).write_text(report(result), encoding='utf-8')
    audit.write_json(output/'review_lock.json', {p.name: audit.sha256(p) for p in sorted(output.iterdir())})
    print(f'DQ10 review saved to {output}. Original preview unchanged; hold v1 expansion.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=audit.DEFAULT_ROOT/'smoke')
    parser.add_argument('--output-dir', type=Path, default=audit.DEFAULT_ROOT/'smoke_review')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    audit.torch.set_num_threads(1)
    run(args.source, args.output_dir, args.check_only)
