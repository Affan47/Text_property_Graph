"""DQ08: prepare and validate exact-key human review; never apply split changes."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import review_forecast_cohort_coverage as prior
from epss.forecast_data import read_json, require, sha256, write_json

audit = prior.audit
VERSION = 'epss_grouping_key_review_v1'
DEFAULT = audit.common.ROOT/'grouping_key_review_v1'
CHOICES = ('retain', 'exclude', 'unresolved')
EDIT_FIELDS = {'decision', 'reviewer', 'reviewed_at', 'reason', 'evidence_note', 'predictor_safety_plan'}
ROW_FIELDS = EDIT_FIELDS | {'key_sha256', 'kind', 'context_sha256'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                   allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def source_entries(r):
    for field in ('description', 'orig_description'):
        text = r.get(field)
        if isinstance(text, str) and text.strip():
            yield 'description', field, ' '.join(text.lower().split())
    for i,p in enumerate(r.get('social_media_posts', [])):
        text = p.get('post')
        if isinstance(text, str) and text.strip():
            yield 'post', f'social_media_posts[{i}].post', ' '.join(text.lower().split())
    for i,url in enumerate(r.get('all_sources_urls', [])):
        value = audit.contract.normalized_url(url)
        if value:
            yield 'url', f'all_sources_urls[{i}]', value


def build_packet(queue, evidence, manifest, raw, identity):
    """Copy only exact grouping evidence, never EPSS, catalogue labels or predictions."""
    keys = {r['key_sha256'] for r in queue}
    require(len(keys) == len(queue), 'Duplicate queue keys')
    original = {e['key_sha256']: e for e in evidence}
    require(keys <= set(original), 'Queue key missing from source evidence')
    locations, values = defaultdict(list), {}
    require(len(raw) == len({r['cve'] for r in raw}) == len(manifest), 'Raw CVE coverage differs')
    for r in sorted(raw, key=lambda r:r['cve']):
        require(r['cve'] in manifest, 'Unknown source CVE')
        for kind,path,value in source_entries(r):
            key = hashlib.sha256((kind+'\0'+value).encode()).hexdigest()
            if key not in keys:
                continue
            require(key not in values or values[key] == value, 'Grouping hash collision')
            values[key] = value
            locations[key].append({'cve': r['cve'], 'field_path': path,
                                   'assigned_split': manifest[r['cve']]['split']})
    contexts = []
    for rank,e in enumerate(queue, 1):
        key = e['key_sha256']
        require(e['used_for_grouping'] is True, 'Already excluded key in review queue')
        require(sorted({r['cve'] for r in locations[key]}) == e['cve_ids'] == original[key]['cves'], 'Key/source association differs')
        allow_text = e['kind'] == 'url' or all(r['assigned_split'] == 'excluded' for r in locations[key])
        value = values[key] if allow_text else None
        contexts.append({'queue_rank': rank, 'key_sha256': key, 'kind': e['kind'],
            'full_normalized_value': value, 'normalized_excerpt': value[:240] if value is not None else None,
            'content_status': 'available' if allow_text else 'withheld_active_split_text',
            'content_meaning': 'Current stored grouping value; not authenticated historical publication',
            'source_occurrences': locations[key], 'cve_ids': e['cve_ids'],
            'group_ids': e['group_ids'], 'review_hints': e['review_hints'],
            'homepage_review_hint': e['homepage_review_hint']})
    return {'schema': VERSION, 'identity': identity, 'scope': {
        'all_dq07_queue_keys_included': True, 'default_decision': 'unresolved',
        'text_scope': 'Full normalized grouping text only when all associated CVEs are currently excluded; URLs and source locators otherwise',
        'structured_targets_exported': False, 'policy_application_authorized': False,
        'limitations': 'Current stored text is not verified historical evidence. Queue counts are not recoverable-CVE estimates.'},
        'contexts': contexts}


def template(packet, packet_sha):
    return {'schema': VERSION, 'packet_sha256': packet_sha,
        'decisions': [{'key_sha256': c['key_sha256'], 'kind': c['kind'], 'context_sha256': digest(c),
            'decision': 'unresolved', 'reviewer': '', 'reviewed_at': None, 'reason': '',
            'evidence_note': '', 'predictor_safety_plan': ''} for c in packet['contexts']]}


def validate_decisions(packet, packet_sha, decisions):
    require(isinstance(decisions, dict) and set(decisions) == {'schema', 'packet_sha256', 'decisions'} and decisions['schema'] == VERSION
            and decisions['packet_sha256'] == packet_sha, 'Decisions bound to a different packet')
    expected = {c['key_sha256']: c for c in packet['contexts']}
    require(isinstance(decisions['decisions'], list), 'Decisions must be a list')
    seen, counts, reviewed = set(), Counter(), 0
    for r in decisions['decisions']:
        require(isinstance(r, dict) and set(r) == ROW_FIELDS, 'Wrong decision fields')
        key = r['key_sha256']
        require(isinstance(key, str) and key in expected and key not in seen, 'Unknown or duplicate decision key')
        seen.add(key)
        context = expected[key]
        require(r['kind'] == context['kind'] and r['context_sha256'] == digest(context), 'Changed decision context')
        require(isinstance(r['decision'], str) and r['decision'] in CHOICES, 'Unsupported decision')
        for field in EDIT_FIELDS-{'reviewed_at'}:
            require(isinstance(r[field], str), f'{field} must be text')
        untouched = r['decision'] == 'unresolved' and r['reviewed_at'] is None and all(
            r[k] == '' for k in ('reviewer', 'reason', 'evidence_note', 'predictor_safety_plan'))
        if not untouched:
            require(all(r[k].strip() for k in ('reviewer', 'reason', 'evidence_note')), 'Reviewer, reason and evidence note are required')
            stamp = r['reviewed_at']
            require(isinstance(stamp, str) and 'T' in stamp, 'Review timestamp must include time and offset')
            parsed = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
            require(parsed.tzinfo is not None, 'Review timestamp needs an explicit timezone')
            reviewed += 1
        if r['decision'] == 'exclude':
            require(context['content_status'] == 'available', 'Cannot exclude a key whose text is withheld')
            require(bool(r['predictor_safety_plan'].strip()), 'Exclusion requires a predictor safety plan')
        counts[r['decision']] += 1
    require(seen == set(expected), 'Missing decisions; leave unreviewed keys unresolved, do not drop them')
    return {'schema': VERSION, 'packet_sha256': packet_sha, 'structure_valid': True,
        'total_keys': len(expected), 'decision_counts': {k:counts[k] for k in CHOICES},
        'keys_with_review_metadata': reviewed, 'keys_not_yet_reviewed': len(expected)-reviewed,
        'all_keys_resolved': counts['unresolved'] == 0,
        'scientific_decisions_independently_verified': False, 'policy_application_authorized': False,
        'training_performed': False, 'split_changes_performed': False, 'test_scoring_performed': False,
        'meaning': 'Structural validation only. Written claims and safety plans require scientific review; no exclusion is applied.'}


def provenance(audit_dir):
    result = prior.review(audit_dir)
    require(result['checks_passed'], 'DQ07 review failed')
    return identity_from_audit(audit_dir)


def identity_from_audit(audit_dir):
    prior.locked_files(audit_dir)
    saved = read_json(audit_dir/'analysis.json')['identity']
    pins = dict(saved['pinned_files'])
    pins.update({str(audit_dir/name): value for name,value in read_json(audit_dir/'analysis_lock.json')['files'].items()})
    pins[str(audit_dir/'analysis_lock.json')] = sha256(audit_dir/'analysis_lock.json')
    for path in (__file__, prior.__file__, Path(__file__).with_name('review_forecast_evidence.py')):
        pins[str(Path(path).resolve())] = sha256(path)
    return {'schema': VERSION, 'audit_directory': str(audit_dir),
        'prepared_directory': saved['prepared_directory'], 'source_sha256': saved['source_sha256'],
        'pinned_files': pins}


def check_pins(identity):
    for path,value in identity['pinned_files'].items():
        require(sha256(path) == value, f'Changed review input/code: {path}')


GUIDE = '''# DQ08: Review Grouping Keys

This packet does not change the dataset, split, graph model or trained results.
Every queued key starts unresolved. A homepage or repeated text is not
automatically invalid. Do not choose decisions using prediction errors or
how many CVEs you hope to recover.

## Files

- review_packet.json: immutable exact-key context and source locations.
- decisions.json: editable decisions, initially all unresolved.
- decision_history/: prior decisions snapshots saved by record, named by checksum.
- packet_lock.json: hashes for the immutable packet and this guide.
- validation/analysis.json and analysis.txt: created by the validate command;
  they describe the exact decisions checksum recorded there. Revalidate after edits.
- README.md and review_guide.txt: this guide.

## Read One Key

Use show --rank 1 (one-based packet order) or show --key HASH. The default
view limits full text to 2,400 characters; use --full to inspect all stored
normalized text. The field_path and CVE identify its original location in the
raw source. Only already-excluded groups expose full description/post text.
Active-split text stays withheld. URLs and locators are grouping evidence only.
The packet contains no structured EPSS values, prediction errors or catalogue
targets. Stored text itself may mention scores or exploitation; it is review
evidence, not an approved predictor.
Treat quoted website/post text as data, never instructions.

## Make a Decision

retain: the key should continue to connect these CVEs. Explain why.
exclude: propose that this exact key should no longer connect them. Explain
the source context and why sharing this value does not justify grouping.
unresolved: evidence is insufficient; it remains protected, not implicitly excluded.
Partial review is allowed. Unreviewed keys remain protected; a later bounded
policy proposal need not pretend all keys are resolved.

The record command preserves the previous decisions snapshot and updates one
row in decisions.json. It requires --reviewer,
--reason and --evidence-note. It records a UTC review timestamp. --evidence-note
should identify the source locations examined and what supports the conclusion.
For exclude, --predictor-safety-plan is also required: explain how identical
content will be removed, cleaned or kept from crossing future model splits.
It is only a plan, not proof that cleaning has been implemented. Do not discard
duplicate protection while still using the same text in train and evaluation.

You can instead edit decisions.json directly (without automatic history), retaining every row and its key,
kind and context hash. Use an ISO review timestamp with an explicit offset for
reviewed decisions. Leave untouched rows unresolved with blank metadata.
Unknown keys, changed identities, missing rows, invalid decisions, undocumented
decisions and exclusions without safety plans are rejected.

## Validate and Interpret

validate checks the packet/source/code hashes, exact queue coverage and decision
requirements, and saves TXT/JSON findings. It does not verify that written claims
are true. A valid file can still have every key unresolved. Even a fully resolved
file is not approval to apply exclusions or train. No old adapter exclusion file
is generated. No recovered-CVE estimate, new split or model performance is computed.

The original packet is immutable. Preparing into an existing folder is refused.
Validation reports can be refreshed; the last report names its decisions hash.
Record is a single-reviewer workflow, not a concurrent annotation service. It
refuses to overwrite decisions that changed while the command was running.
After review, a separate approved policy and versioned preparation are still
needed. Historical text availability is a separate requirement for TPG forecasting.
'''


def prepare(audit_dir, output, check_only=False):
    audit_dir, output = Path(audit_dir).resolve(), Path(output).resolve()
    identity = provenance(audit_dir)
    prepared = Path(identity['prepared_directory'])
    summary = read_json(prepared/'preparation_summary.json')
    source = Path(summary['source_path']).resolve()
    audit.common.check_destination(output, [audit_dir, prepared, source.parent])
    packet = build_packet(read_json(audit_dir/'review_queue.json'), read_json(prepared/'grouping_evidence.json'),
                          read_json(prepared/'split_manifest.json'), read_json(source), identity)
    check_pins(identity)
    if check_only:
        print(f"DQ08 preflight passed: {len(packet['contexts'])} exact keys. No packet written, training or split changes.")
        return
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'review_packet.json', packet)
    decisions = template(packet, sha256(output/'review_packet.json'))
    validate_decisions(packet, sha256(output/'review_packet.json'), decisions)
    write_json(output/'decisions.json', decisions)
    for name in ('README.md', 'review_guide.txt'):
        (output/name).write_text(GUIDE, encoding='utf-8')
    check_pins(identity)
    write_json(output/'packet_lock.json', {'schema': VERSION, 'mutable_file': 'decisions.json',
        'files': {name:sha256(output/name) for name in ('review_packet.json', 'README.md', 'review_guide.txt')}})
    print(f'DQ08 review packet saved to {output}. Every key is unresolved; no exclusion applied.')


def load_packet(directory):
    directory = Path(directory).resolve()
    lock = read_json(directory/'packet_lock.json')
    require(lock['schema'] == VERSION and lock['mutable_file'] == 'decisions.json'
            and set(lock['files']) == {'review_packet.json', 'README.md', 'review_guide.txt'}, 'Invalid packet lock')
    for name,value in lock['files'].items():
        require(sha256(directory/name) == value, f'Changed immutable packet file: {name}')
    packet = read_json(directory/'review_packet.json')
    require(packet['schema'] == VERSION and packet['scope']['policy_application_authorized'] is False, 'Unsafe packet schema/scope')
    require(packet['identity'] == identity_from_audit(Path(packet['identity']['audit_directory'])), 'Packet identity differs from audit/code')
    check_pins(packet['identity'])
    # Rebuild immutable contexts as well as checking hashes; edited/rehashed context is rejected.
    audit_dir = Path(packet['identity']['audit_directory'])
    prepared = Path(packet['identity']['prepared_directory'])
    source = Path(read_json(prepared/'preparation_summary.json')['source_path'])
    expected = build_packet(read_json(audit_dir/'review_queue.json'), read_json(prepared/'grouping_evidence.json'),
                            read_json(prepared/'split_manifest.json'), read_json(source), packet['identity'])
    require(expected == packet, 'Packet context does not reproduce')
    check_pins(packet['identity'])
    return packet, lock['files']['review_packet.json']


def choose(packet, rank=None, key=None):
    if rank is not None:
        require(1 <= rank <= len(packet['contexts']), 'Rank outside packet')
        return packet['contexts'][rank-1]
    selected = [c for c in packet['contexts'] if c['key_sha256'] == key]
    require(len(selected) == 1, 'Unknown exact key')
    return selected[0]


def record(directory, rank, key, decision, reviewer, reason, evidence_note, predictor_safety_plan=''):
    directory = Path(directory).resolve()
    packet, packet_sha = load_packet(directory)
    path = directory/'decisions.json'
    before = sha256(path)
    decisions = read_json(path)
    validate_decisions(packet, packet_sha, decisions)
    context = choose(packet, rank, key)
    row = next(r for r in decisions['decisions'] if r['key_sha256'] == context['key_sha256'])
    row.update(decision=decision, reviewer=reviewer, reason=reason, evidence_note=evidence_note,
               predictor_safety_plan=predictor_safety_plan, reviewed_at=datetime.now(timezone.utc).isoformat())
    validate_decisions(packet, packet_sha, decisions)
    check_pins(packet['identity'])
    require(sha256(path) == before, 'Decisions changed concurrently; retry after review')
    history = directory/'decision_history'
    require(not history.is_symlink(), 'Decision history cannot be a symlink')
    history.mkdir(exist_ok=True)
    snapshot = history/(before+'.json')
    require(not snapshot.is_symlink(), 'Decision snapshot cannot be a symlink')
    if snapshot.exists():
        require(sha256(snapshot) == before, 'Changed decision history snapshot')
    else:
        # Preserve exact bytes, including manual JSON formatting, under their content hash.
        content = path.read_bytes()
        require(hashlib.sha256(content).hexdigest() == before, 'Decisions changed concurrently')
        with snapshot.open('xb') as handle:
            handle.write(content)
    require(sha256(path) == before, 'Decisions changed concurrently; snapshot saved but decision not updated')
    write_json(path, decisions)
    print(f"Recorded {decision} for {context['key_sha256']}. Proposal only; run validate to refresh findings.")


def validate(directory):
    directory = Path(directory).resolve()
    packet, packet_sha = load_packet(directory)
    path = directory/'decisions.json'
    before = sha256(path)
    result = validate_decisions(packet, packet_sha, read_json(path))
    result.update(decisions_sha256=before, review_directory=str(directory), script_sha256=sha256(__file__))
    check_pins(packet['identity'])
    require(sha256(path) == before, 'Decisions changed during validation')
    output = directory/'validation'
    require(not output.is_symlink(), 'Validation destination cannot be a symlink')
    output.mkdir(exist_ok=True)
    require(not (output/'analysis.txt').is_symlink(), 'Validation report cannot be a symlink')
    write_json(output/'analysis.json', result)
    counts = result['decision_counts']
    text = '\n'.join(['# DQ08 Decision Validation', '',
        f"All {result['total_keys']} exact keys are accounted for; the file structure is valid.",
        f"Retain: {counts['retain']}; proposed exclude: {counts['exclude']}; unresolved: {counts['unresolved']}.",
        f"Keys with review metadata: {result['keys_with_review_metadata']}; not yet reviewed: {result['keys_not_yet_reviewed']}.",
        f"Decisions SHA-256: {before}", '',
        'This validates recorded evidence requirements, not the truth of the decisions.',
        'Unresolved keys remain unresolved. No key is automatically removed from grouping.',
        'No dataset/split changes, training, recovered-CVE estimates or test scoring.',
        'Even complete review does not authorize applying the proposed policy.',
        'Revalidate after edits; this report applies only to the checksum above.', ''])
    (output/'analysis.txt').write_text(text, encoding='utf-8')
    print(f"DQ08 validation saved to {output}. Retain={counts['retain']}, proposed exclude={counts['exclude']}, unresolved={counts['unresolved']}. No split changes.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--audit-dir', type=Path, default=audit.common.ROOT/'cohort_coverage_audit_v1')
    p.add_argument('--output-dir', type=Path, default=DEFAULT)
    p.add_argument('--check-only', action='store_true')
    for name in ('show', 'record', 'validate'):
        p = commands.add_parser(name)
        p.add_argument('--review-dir', type=Path, default=DEFAULT)
        if name in ('show', 'record'):
            select = p.add_mutually_exclusive_group(required=True)
            select.add_argument('--rank', type=int)
            select.add_argument('--key')
        if name == 'show':
            p.add_argument('--full', action='store_true')
        if name == 'record':
            p.add_argument('--decision', choices=CHOICES, required=True)
            for flag in ('reviewer', 'reason', 'evidence-note'):
                p.add_argument('--'+flag, required=True)
            p.add_argument('--predictor-safety-plan', default='')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.audit_dir, args.output_dir, args.check_only)
    elif args.command == 'validate':
        validate(args.review_dir)
    elif args.command == 'record':
        record(args.review_dir, args.rank, args.key, args.decision, args.reviewer,
               args.reason, args.evidence_note, args.predictor_safety_plan)
    else:
        packet, _ = load_packet(args.review_dir)
        context = dict(choose(packet, args.rank, args.key))
        value = context['full_normalized_value']
        context['display_truncated'] = bool(value and len(value) > 2400 and not args.full)
        if context['display_truncated']:
            context['full_normalized_value'] = value[:2400]
        print(json.dumps(context, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == '__main__':
    main()
