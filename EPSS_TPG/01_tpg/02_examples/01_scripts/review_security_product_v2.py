"""DQ11: inspect the saved small preview and prepare explicit source review.

No reparsing, model loading, extraction changes, expansion or training.
HAS_VERSION mentions are not converted into affected/fixed-version claims.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

import audit_security_product_versions_v2 as preview
from tpg.contracts import sentence_evidence
from tpg.exporters.exporters import import_graphson
from tpg.schema.types import NodeType, EdgeType, SecurityEdgeType

old = preview.old
SOURCE = preview.DEFAULT_OUTPUT
ROOT = old.TPG/'09_annotation/07_product_version_review'
SCHEMA = 'dq11_product_version_review_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def quote(text, span):
    old.require(isinstance(span, list) and len(span) == 2 and
                all(type(x) is int for x in span) and 0 <= span[0] < span[1] <= len(text), 'Invalid source span')
    return text[span[0]:span[1]]


def signature(edge):
    return {k: edge['evidence'][k] for k in ('source_span', 'target_span', 'version_operator', 'qualifier_text')}


def inspect_record(row, graph):
    text, cve = row['description'], row['cve']
    old.require(row['split'] in ('train', 'validation'), 'Test inspection forbidden')
    old.require(graph.doc_id == cve and graph.metadata['source_text'] == text, 'Graph/source identity differs')
    old.require(old.cache.text_digest(text) == row['description_sha256'], 'Description checksum differs')
    old.require(not graph.validate(), 'Invalid saved graph')
    old.require(preview.extraction(graph) == row['after'], 'Saved comparisons disagree with GraphSON')
    versions = [n for n in graph.nodes(NodeType.ENTITY) if n.properties.entity_type == 'VERSION']
    typed = graph.edges(SecurityEdgeType.HAS_VERSION)
    generic = [e for e in graph.edges(EdgeType.ENTITY_REL) if e.properties.entity_rel_type == 'HAS_VERSION']
    old.require(len(typed) == len(generic), 'Generic/typed version edge count differs')
    for edge in typed:
        matching = [e for e in generic if (e.source, e.target) == (edge.source, edge.target)]
        old.require(len(matching) == 1 and matching[0].properties.extra == edge.properties.extra,
                    'Generic/typed version evidence differs')
        source, target = graph.get_node(edge.source), graph.get_node(edge.target)
        old.require(source.properties.entity_type == 'SOFTWARE' and target in versions, 'Wrong endpoint types')
        evidence = edge.properties.extra
        expected = sentence_evidence(graph, source, target)
        old.require(expected is not None and all(evidence[k] == expected[k] for k in
            ('sentence_id', 'sent_idx', 'char_start', 'char_end', 'text', 'source_span', 'target_span')),
            'Sentence evidence disagrees with source nodes')
        old.require(quote(text, evidence['source_span']) == source.properties.text and
                    quote(text, evidence['target_span']) == target.properties.text and
                    quote(text, evidence['qualifier_span']) == evidence['qualifier_text'], 'Evidence quote differs')
        old.require(evidence['extraction_policy'] == preview.extension.VERSION and
                    evidence['status'] == 'candidate', 'Unexpected relation policy/status')
        old.require(evidence['qualifier_span'][0] == source.properties.char_end and
                    evidence['qualifier_span'][1] >= target.properties.char_end and
                    evidence['qualifier_span'][1] <= expected['char_end'], 'Qualifier outside sentence or version')
    items = []
    for node in versions:
        p = node.properties
        span = [p.char_start, p.char_end]
        old.require(quote(text, span) == p.text, 'Version text differs')
        old.require(quote(text, p.extra['version_literal_span']) == p.extra['version_literal'], 'Version literal differs')
        links = [e for e in typed if e.target == node.id]
        reason = p.extra.get('ownership_abstention')
        old.require((len(links) == 1 and reason is None) or
                    (len(links) == 0 and isinstance(reason, str) and bool(reason)),
                    'Version must have one owner or an explicit abstention')
        owner = p.extra.get('ownership_candidate')
        if links:
            evidence = links[0].properties.extra
            old.require(owner is not None and owner['owner_span'] == evidence['source_span'] and
                        owner['operator'] == evidence['version_operator'] and
                        owner['qualifier_span'] == evidence['qualifier_span'] and
                        owner['qualifier_text'] == evidence['qualifier_text'], 'Node/edge ownership differs')
        else:
            old.require(owner is None, 'Unresolved version has an owner candidate')
        new = next((e for e in row['after']['typed_security_relations']
                    if e['relation'] == 'SEC_HAS_VERSION' and e['target_span'] == span), None)
        # Same start aligns preserved numeric mentions whose Patch suffix grew.
        prior = [v for v in row['before']['versions'] if v['span'][0] == span[0]]
        old.require(len(prior) <= 1, 'Ambiguous v1 version alignment')
        prior_links = [e for e in row['before']['typed_security_relations']
                       if e['relation'] == 'SEC_HAS_VERSION' and e['target_span'][0] == span[0]]
        old.require(len(prior_links) <= 1, 'Ambiguous v1 link alignment')
        before = prior_links[0] if prior_links else None
        change = 'unchanged'
        if not prior:
            change = 'newly_recognized_mention'
        elif before and not new:
            change = 'owner_link_removed'
        elif new and not before:
            change = 'owner_link_added'
        elif new and before and signature(new) != signature(before):
            change = 'owner_release_or_qualifier_changed'
        elif prior[0]['text'] != p.text:
            change = 'release_text_extended_without_owner'
        item_id = digest([cve, row['description_sha256'], span])[:24]
        items.append({'item_id': item_id, 'version': p.text, 'version_span': span,
            'version_literal': p.extra['version_literal'], 'release_form': p.extra['release_form'],
            'state': 'linked' if new else 'unresolved', 'abstention_reason': reason,
            'change': change, 'v1_link': before, 'v2_link': new, 'judgment': 'pending', 'notes': ''})
    return {'cve': cve, 'split': row['split'], 'description': text,
            'description_sha256': row['description_sha256'], 'previously_reviewed': row['previously_reviewed'],
            'fixed_sample': row['fixed_sample'], 'items': items,
            'full_source_reviewed': False, 'reviewer': '', 'reviewed_at': None,
            'missed_mentions': [], 'notes': ''}


def summary(records):
    items = [i for r in records for i in r['items']]
    return {'cves': len(records), 'by_split': dict(Counter(r['split'] for r in records)),
            'version_mentions': len(items), 'states': dict(Counter(i['state'] for i in items)),
            'change_types': dict(Counter(i['change'] for i in items)),
            'unresolved_reasons': dict(Counter(i['abstention_reason'] for i in items if i['state'] == 'unresolved')),
            'documents_without_recognized_versions': [r['cve'] for r in records if not r['items']],
            'semantic_review_complete': False, 'extraction_accuracy': None,
            'full_expansion_authorized': False, 'training': False, 'test_inspection': False}


def instructions(packet, stats):
    lines = ['# DQ11: Product/Version Source Review', '',
        'Technical consistency is checked automatically. Meaning is not marked correct automatically.',
        'This packet covers only the same DQ10 development sample. It is not an independent benchmark.', '',
        '## Counts', '', json.dumps(stats, indent=2), '',
        '## How to Review', '',
        'Run the edit-copy command to create a separate decisions file. Never edit the locked packet.',
        'Read the complete description for each CVE, including records with no extracted versions.',
        'For linked items, judgment is supported_link, incorrect_link, or uncertain. Check the complete '
        'product and release wording and bound; use incorrect_link if any of those are wrong, explaining why.',
        'For unresolved items, judgment is appropriately_unresolved, resolvable_owner, or uncertain. '
        'Resolvable means the description supports a clear owner despite the extractor abstaining.',
        'Keep unreviewed items pending. Do not call an abstention a negative training label.',
        'For a finished CVE, set full_source_reviewed=true, reviewer to your name, and reviewed_at '
        'to an ISO timestamp with timezone, for example YYYY-MM-DDTHH:MM:SS+02:00. '
        'Every item must have a non-pending judgment. Explain wrong, uncertain, resolvable and empty cases.',
        'Use missed_mentions for releases absent from the extracted items. Each entry is '
        '{"span": [start, end], "text": "exact source words", "notes": "why this is a release"}. '
        'Offsets count Python characters: start is included, end is excluded. Do not change source text or evidence.',
        'Leave unfinished CVEs blank in the submission; keep working notes separately.', '',
        '## What the Results Can Establish', '',
        'Summarize reports review progress and decisions, not training labels or a gold benchmark. '
        'No precision/recall is computed: missed relationships need complete gold annotation, '
        'and judgments from these development examples are not independent evaluation.',
        'Version links are mention associations, not affected/fixed-version facts. '
        'The existing learned-claim review schema makes that stronger distinction; this packet '
        'does not coerce HAS_VERSION into those labels or change that schema.', '',
        '## Per-CVE Evidence', '']
    for record in packet['records']:
        lines += [f"### {record['cve']} ({record['split']})", record['description'], '']
        if not record['items']:
            lines.append('No recognized versions. Full-text review is still required for missed releases.')
        for item in record['items']:
            link = item['v2_link']
            decision = (f"{link['source']} -> {link['target']} ({link['evidence']['version_operator']})"
                        if link else f"No owner: {item['abstention_reason']}")
            lines += [f"- {item['item_id']}: {item['version']} at {item['version_span']}; {decision}; {item['change']}."]
        lines.append('')
    return '\n'.join(lines)


def validate_reviews(packet, decisions):
    old.require(set(decisions) == set(packet), 'Review envelope differs')
    old.require({k: v for k, v in decisions.items() if k != 'records'} ==
                {k: v for k, v in packet.items() if k != 'records'}, 'Review provenance differs')
    old.require(len(decisions['records']) == len(packet['records']), 'Missing review records')
    completed, judgments, missed = [], Counter(), 0
    editable = {'full_source_reviewed', 'reviewer', 'reviewed_at', 'missed_mentions', 'notes', 'items'}
    for original, record in zip(packet['records'], decisions['records']):
        old.require(set(record) == set(original), 'Review record fields differ')
        old.require({k: v for k, v in record.items() if k not in editable} ==
                    {k: v for k, v in original.items() if k not in editable}, 'Review source/order differs')
        old.require(type(record['full_source_reviewed']) is bool and isinstance(record['notes'], str), 'Invalid review fields')
        old.require(len(record['items']) == len(original['items']), 'Missing review items')
        for a, b in zip(original['items'], record['items']):
            old.require(set(a) == set(b) and {k: v for k, v in a.items() if k not in ('judgment', 'notes')} ==
                        {k: v for k, v in b.items() if k not in ('judgment', 'notes')}, 'Changed source evidence/item order')
            allowed = ({'supported_link', 'incorrect_link', 'uncertain'} if b['state'] == 'linked' else
                       {'appropriately_unresolved', 'resolvable_owner', 'uncertain'})
            old.require(b['judgment'] in allowed | {'pending'} and isinstance(b['notes'], str), 'Invalid judgment')
            if record['full_source_reviewed']:
                old.require(b['judgment'] != 'pending', 'Completed CVE has pending items')
                if b['judgment'] in ('incorrect_link', 'resolvable_owner', 'uncertain'):
                    old.require(bool(b['notes'].strip()), 'Explain the review concern')
        if not record['full_source_reviewed']:
            old.require(record == original, 'Incomplete reviews must remain blank in submissions')
            continue
        old.require(isinstance(record['reviewer'], str) and bool(record['reviewer'].strip()), 'Reviewer required')
        old.require(isinstance(record['reviewed_at'], str), 'Review timestamp required')
        stamp = datetime.fromisoformat(record['reviewed_at'].replace('Z', '+00:00'))
        old.require(stamp.utcoffset() is not None, 'Review timestamp requires timezone')
        old.require(isinstance(record['missed_mentions'], list), 'Missed mentions must be a list')
        spans = set()
        for mention in record['missed_mentions']:
            old.require(set(mention) == {'span', 'text', 'notes'} and
                        isinstance(mention['notes'], str) and bool(mention['notes'].strip()), 'Explain missed release')
            old.require(quote(record['description'], mention['span']) == mention['text'], 'Missed quote differs')
            span = tuple(mention['span'])
            old.require(span not in spans and not any(span == tuple(i['version_span']) for i in record['items']),
                        'Duplicate or already recognized missed mention')
            spans.add(span)
        if not record['items']:
            old.require(bool(record['notes'].strip()), 'Explain reviewed empty extraction')
        completed.append(record['cve'])
        judgments.update(i['judgment'] for i in record['items'])
        missed += len(record['missed_mentions'])
    return {'completed_cves': completed, 'pending_cves': len(packet['records']) - len(completed),
            'judgment_counts': dict(judgments), 'reported_missed_mentions': missed,
            'review_is_declared_not_authenticated': True, 'extraction_accuracy': None,
            'training': False, 'full_expansion_authorized': False}


def check_packet(directory):
    lock = old.read_json(directory/'review_lock.json')
    old.require({str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()} ==
                set(lock) | {'review_lock.json'}, 'Packet file set changed')
    for name, value in lock.items():
        old.require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe packet path')
        old.require(old.sha256(directory/name) == value, 'Packet artifact changed')
    packet = old.read_json(directory/'review_template.json')
    old.require(packet['schema'] == SCHEMA, 'Wrong review schema')
    old.cache.check_pins(packet['pinned_files'])
    preview.run(Path(packet['source']), verify_only=True)
    return packet


def prepare(output, check_only=False):
    preview.run(SOURCE, verify_only=True)
    summary_source = old.read_json(SOURCE/'analysis.json')
    old.require(summary_source['scale'] == 'smoke' and all(summary_source[k] is False for k in
        ('training', 'test_scoring', 'test_graphs', 'embeddings_generated', 'model_ready_cache')),
        'Only the unembedded development smoke preview is allowed')
    rows = old.read_json(SOURCE/'comparisons.json')
    old.require([r['cve'] for r in rows] == summary_source['inspected_ids'] and
                len(set(r['cve'] for r in rows)) == len(rows), 'Selection differs')
    old.require(preview.metrics(rows) == summary_source['statistics'], 'Aggregate counts differ')
    pins = dict(summary_source['pinned_files'])
    for path in (SOURCE/'preview_lock.json', Path(__file__), old.TPG/'08_tests/test_security_product_review_v2.py'):
        pins[str(path.resolve())] = old.sha256(path)
    old.cache.check_destination(output, [SOURCE.parent, old.DEFAULT_ROOT, old.TPG/'01_core',
        old.TPG/'10_extensions', old.TPG/'08_tests', old.cache.CACHE_ROOT,
        old.cache.PREPARED, old.EPSS.parent/'SummTPGVul'])
    records = [inspect_record(row, import_graphson(str(SOURCE/'graphs'/f"{row['cve']}.json")))
               for row in old.tqdm(rows, desc='Check saved links, spans and unresolved mentions', unit='CVE')]
    pins[str(Path(preview.__file__).resolve())] = old.sha256(preview.__file__)
    packet = {'schema': SCHEMA, 'source': str(SOURCE), 'pinned_files': pins, 'records': records}
    old.require(len({i['item_id'] for r in records for i in r['items']}) == sum(len(r['items']) for r in records),
                'Duplicate review item IDs')
    old.cache.check_pins(pins)
    preview.run(SOURCE, verify_only=True)
    if check_only:
        print(f'DQ11 preflight passed: {len(records)} saved descriptions; no output created.'); return
    output.mkdir(parents=True, exist_ok=False)
    with old.tqdm(total=3, desc='Save source review packet', unit='stage') as bar:
        stats = summary(records)
        old.write_json(output/'review_template.json', packet)
        old.write_json(output/'analysis.json', stats); bar.update()
        report = instructions(packet, stats)
        for name in ('analysis.txt', 'README.md'):
            (output/name).write_text(report, encoding='utf-8')
        bar.update(); old.cache.check_pins(pins)
        old.write_json(output/'review_lock.json', {p.name: old.sha256(p) for p in sorted(output.iterdir())})
        bar.update()
    print(f'DQ11 review packet saved to {output}. All semantic judgments remain pending; no training or expansion.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--output-dir', type=Path, default=ROOT/'packet_v1')
    p.add_argument('--check-only', action='store_true')
    for name in ('edit-copy', 'summarize'):
        p = commands.add_parser(name)
        p.add_argument('--packet-dir', type=Path, default=ROOT/'packet_v1')
        if name == 'edit-copy':
            p.add_argument('--decisions', type=Path, default=ROOT/'decisions_v1.json')
        else:
            p.add_argument('--decisions', type=Path, required=True)
            p.add_argument('--output-dir', type=Path, default=ROOT/'review_results_v1')
    args = parser.parse_args()
    old.torch.set_num_threads(1)
    if args.command == 'prepare':
        prepare(args.output_dir.resolve(), args.check_only); return
    packet_dir = args.packet_dir.resolve()
    packet = check_packet(packet_dir)
    if args.command == 'edit-copy':
        path = args.decisions.resolve()
        old.cache.check_destination(path, [packet_dir, SOURCE.parent, old.DEFAULT_ROOT,
            old.TPG/'01_core', old.TPG/'10_extensions', old.TPG/'08_tests', old.cache.CACHE_ROOT,
            old.cache.PREPARED, old.EPSS.parent/'SummTPGVul'])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as handle:
            json.dump(packet, handle, indent=2, ensure_ascii=True)
        print(f'Unreviewed edit copy saved to {path}'); return
    decisions_path = args.decisions.resolve()
    decisions_hash = old.sha256(decisions_path)
    result = validate_reviews(packet, old.read_json(decisions_path))
    result.update(packet_sha256=old.sha256(packet_dir/'review_template.json'), decisions_sha256=decisions_hash)
    output = args.output_dir.resolve()
    old.cache.check_destination(output, [packet_dir, SOURCE.parent, old.DEFAULT_ROOT, old.TPG/'01_core',
        old.TPG/'10_extensions', old.TPG/'08_tests', old.cache.CACHE_ROOT, old.cache.PREPARED,
        old.EPSS.parent/'SummTPGVul', decisions_path])
    check_packet(packet_dir)
    old.require(old.sha256(decisions_path) == decisions_hash, 'Decisions changed while reading')
    output.mkdir(parents=True, exist_ok=False)
    old.write_json(output/'analysis.json', result)
    (output/'analysis.txt').write_text('DQ11 declared review decisions (not model metrics)\n\n' +
                                     json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'DQ11 review progress saved to {output}. No model labels, expansion or training authorized.')


if __name__ == '__main__':
    main()
