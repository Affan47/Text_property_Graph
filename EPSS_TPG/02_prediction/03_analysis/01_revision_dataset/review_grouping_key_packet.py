"""Reproduce DQ08 setup checks and describe evidence without recording decisions."""
import argparse
from collections import Counter
from pathlib import Path
import re

import review_grouping_keys as workflow

read_json, require, sha256, write_json = (
    workflow.read_json, workflow.require, workflow.sha256, workflow.write_json)


def summarize(packet):
    contexts = packet['contexts']
    kinds = []
    for kind in ('url', 'post', 'description'):
        selected = [c for c in contexts if c['kind'] == kind]
        kinds.append({'kind': kind, 'shared_values': len(selected),
                      'distinct_cves': len({x for c in selected for x in c['cve_ids']})})
    roles = {}
    for c in contexts:
        for occurrence in c['source_occurrences']:
            cve, role = occurrence['cve'], occurrence['assigned_split']
            require(cve not in roles or roles[cve] == role, 'Conflicting CVE roles')
            roles[cve] = role
        if c['kind'] != 'url' and c['content_status'] == 'available':
            require(all(o['assigned_split'] == 'excluded' for o in c['source_occurrences']),
                    'Active-split text must not be analyzed')
    require(set(roles) == {x for c in contexts for x in c['cve_ids']}, 'Occurrence coverage differs')
    examples = []
    # Selection uses source metadata only, never scores, labels or model errors.
    selected = [c for c in contexts if c['homepage_review_hint']][:5]
    for kind, limit in (('post', 3), ('description', 1)):
        selected += sorted((c for c in contexts if c['kind'] == kind
                            and c['content_status'] == 'available'),
                           key=lambda c: (-len(c['cve_ids']), c['queue_rank']))[:limit]
    for c in selected:
        value = c['full_normalized_value']
        mentions = set(re.findall(r'\bcve-\d{4}-\d{4,}\b', value.lower()))
        examples.append({'queue_rank': c['queue_rank'], 'key_sha256': c['key_sha256'],
            'kind': c['kind'], 'directly_associated_cves': len(c['cve_ids']),
            'characters_in_normalized_value': len(value), 'excerpt': value[:500],
            'ending_excerpt': value[-220:], 'distinct_cve_id_strings': len(mentions),
            'associated_ids_present_in_text': sum(x.lower() in mentions for x in c['cve_ids']),
            'example_locations': c['source_occurrences'][:2]})
    return {'by_kind': kinds, 'distinct_cves': len(roles),
            'distinct_cves_by_assigned_role': dict(sorted(Counter(roles.values()).items())),
            'content_status_counts': dict(Counter(c['content_status'] for c in contexts)),
            'homepage_hint_keys': sum(c['homepage_review_hint'] for c in contexts),
            'preliminary_examples_not_decisions': examples}


def verify_saved(directory, packet, packet_sha):
    decisions = read_json(directory/'decisions.json')
    result = workflow.validate_decisions(packet, packet_sha, decisions)
    result.update(decisions_sha256=sha256(directory/'decisions.json'),
                  review_directory=str(directory), script_sha256=sha256(workflow.__file__))
    require(result == read_json(directory/'validation/analysis.json'),
            'Saved validation is stale or does not reproduce; run validate first')
    counts = result['decision_counts']
    expected = '\n'.join(['# DQ08 Decision Validation', '',
        f"All {result['total_keys']} exact keys are accounted for; the file structure is valid.",
        f"Retain: {counts['retain']}; proposed exclude: {counts['exclude']}; unresolved: {counts['unresolved']}.",
        f"Keys with review metadata: {result['keys_with_review_metadata']}; not yet reviewed: {result['keys_not_yet_reviewed']}.",
        f"Decisions SHA-256: {result['decisions_sha256']}", '',
        'This validates recorded evidence requirements, not the truth of the decisions.',
        'Unresolved keys remain unresolved. No key is automatically removed from grouping.',
        'No dataset/split changes, training, recovered-CVE estimates or test scoring.',
        'Even complete review does not authorize applying the proposed policy.',
        'Revalidate after edits; this report applies only to the checksum above.', ''])
    require((directory/'validation/analysis.txt').read_text(encoding='utf-8') == expected,
            'Saved validation text differs')
    return result, decisions == workflow.template(packet, packet_sha)


def explanation(r):
    v = r['validation']
    lines = ['# DQ08: Review Packet Findings', '', '## What Finished', '',
        'The packet reconstructs exactly from the stored source and DQ07 queue.',
        'Immutable file hashes and pinned source/code hashes pass. Saved JSON and',
        'TXT validation results reproduce. This is setup verification, not approval',
        'to remove shared evidence or change the splits.', '',
        f"Decisions: {v['decision_counts']}. Reviewed items: {v['keys_with_review_metadata']}.",
        f"Decisions still equal the original untouched template: {r['initial_template_unchanged']}.", '',
        '## What the Counts Mean', '',
        '| Shared evidence type | Different shared values | Different CVEs associated with these values |',
        '|---|---:|---:|']
    for row in r['by_kind']:
        lines.append(f"| {row['kind']} | {row['shared_values']} | {row['distinct_cves']} |")
    lines += ['', f"Across all types: {r['distinct_cves']:,} different CVEs.",
        'A shared value is one normalized URL or text, not one CVE or one post occurrence.',
        'A CVE can appear under several values/types; do not add the CVE column.',
        f"Current assigned roles: {r['distinct_cves_by_assigned_role']}.",
        f"Content availability: {r['content_status_counts']}.",
        'URL contexts can include reserved CVEs. All exposed post/description text',
        'belongs to currently excluded CVEs; active-split text is not analyzed.',
        f"Homepage-shaped URLs flagged for review: {r['homepage_hint_keys']}.", '',
        '## Real Examples and Their Limits', '']
    for e in r['preliminary_examples_not_decisions']:
        lines += [f"### Queue item {e['queue_rank']}: {e['kind']}", '',
            f"Associated with {e['directly_associated_cves']} CVEs; normalized length {e['characters_in_normalized_value']:,} characters.",
            f"Stored beginning: {e['excerpt']}",
            f"Stored ending: {e['ending_excerpt']}",
            f"Example source locations: {e['example_locations']}", '']
        if e['kind'] == 'post':
            lines += [f"The complete text contains {e['distinct_cve_id_strings']} distinct CVE-ID strings;",
                f"{e['associated_ids_present_in_text']} of its associated CVE IDs appear in that text.",
                'An ID match proves presence, not that the main article discusses that CVE.', '']
    lines += ['## Interpretation and Next Action', '',
        'Homepage URLs are broad connections, not proof of duplicate vulnerabilities.',
        'Long saved pages may mix an article with navigation, related headlines and',
        'cookie notices. The examples support examining article extraction and CVE',
        'relevance; they do not prove the whole page should be discarded.',
        'An identical short description can describe several different CVEs. It can',
        'also be a real duplicate-input risk if the same text is used in both splits.', '',
        'Review a bounded set of the homepage and long-page examples first. Record',
        'the exact source locations, decision, reason and reviewer. Any proposed',
        'exclusion needs a plan for removing/cleaning shared predictor content.',
        'Keep unresolved links protected. Do not merely drop duplicate protection',
        'while retaining identical text as a model input.', '',
        'Only a later approved, separately versioned preparation may apply changes.',
        'Removing one key need not separate a connected group: other links remain.',
        'These counts are not estimates of how many CVEs can return to training.',
        'Historical text availability is still a separate question for forecasting.',
        'No decisions, source records, splits, checkpoints or original reports changed.',
        'No training, new forecasts or test-outcome analysis was performed.', '']
    return '\n'.join(lines)


def review(directory):
    directory = Path(directory).resolve()
    before = {str(p): sha256(p) for p in directory.rglob('*') if p.is_file()}
    packet, packet_sha = workflow.load_packet(directory)
    validation, initial = verify_saved(directory, packet, packet_sha)
    result = {'schema': 'epss_grouping_key_packet_review_v1', 'checks_passed': True,
        'review_directory': str(directory), 'review_script_sha256': sha256(__file__),
        'reviewed_file_hashes': before, 'validation': validation,
        'initial_template_unchanged': initial, **summarize(packet),
        'original_results_modified': False, 'decisions_modified': False,
        'training_performed': False, 'split_changes_performed': False,
        'test_outcome_analysis_performed': False, 'policy_application_authorized': False}
    workflow.check_pins(packet['identity'])
    require(before == {str(p): sha256(p) for p in directory.rglob('*') if p.is_file()},
            'Review inputs changed during analysis')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-dir', type=Path, default=workflow.DEFAULT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    directory = args.review_dir.resolve()
    result = review(directory)
    if args.check_only:
        print('DQ08 packet review passed. No files written or decisions changed.')
        return
    output = (args.output_dir or directory.with_name(directory.name+'_review')).resolve()
    packet = read_json(directory/'review_packet.json')
    prepared = Path(packet['identity']['prepared_directory'])
    source = Path(read_json(prepared/'preparation_summary.json')['source_path'])
    workflow.audit.common.check_destination(output,
        [directory, prepared, source.parent, Path(packet['identity']['audit_directory'])])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'analysis.json', result)
    for name in ('analysis.txt', 'README.md'):
        (output/name).write_text(explanation(result), encoding='utf-8')
    print(f'DQ08 packet review saved to {output}. No decisions or splits changed.')


if __name__ == '__main__':
    main()
