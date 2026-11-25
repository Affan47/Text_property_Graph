"""Audit saved labels, temporal ordering and text provenance without fitting."""
from collections import Counter
from datetime import datetime, timedelta, timezone
import csv
import gzip
import hashlib
import io
import json
import re
from pathlib import Path

from tqdm import tqdm
import audit_reddit as common
import repair_provenance as receipts
from ..vulncheck_study import policy_v2 as text_policy
from ..vulncheck_study import adapter_v2 as artifact_io

publish = artifact_io.publish

ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT/'02_prediction/04_configs/07_fresh_studies/10_catalogue_readiness_v1.json'
CLI = ROOT/'02_prediction/03_analysis/07_fresh_studies/19_audit_catalogue_readiness.py'
READ, SHA, REQUIRE = common.read, common.sha, common.require
DAY = re.compile(r'^\d{4}-\d{2}-\d{2}$')
ISO = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$')
EXPLOIT = re.compile(r'\b(?:actively exploited|exploited in the wild|exploitation observed|used in attacks)\b', re.I)
SOURCES = {'D': 'description', 'G': 'summ_github_only_urls_fast.gpt-oss:120b',
           'N': 'summ_non_github_urls_fast.gpt-oss:120b'}
POLICY_SOURCES = {'D': 'description', 'G': 'github_summary', 'N': 'non_github_summary', 'M': 'smp'}
EXCLUDED = ['in_vckev', 'vendor_project', 'vckev_cwes', 'vckev_date_added', 'cisa_date_added', 'due_date',
            'vckev_known_ransomware_campaign_use', 'vckev_reported_exploited_by_canaries',
            'vckev_reported_exploitation', 'vckev_xdb', 'vckev_updated_at']


def csv_text(rows, fields):
    stream = io.StringIO(newline=''); writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for row in sorted(rows, key=lambda r: common.dumps(r)):
        writer.writerow({k: common.dumps(v).strip() if isinstance(v, (list, dict)) else v for k, v in row.items()})
    return stream.getvalue()


def parse_time(value):
    if not isinstance(value, str) or not value:
        return None, 'missing'
    if not (DAY.fullmatch(value) or ISO.fullmatch(value)):
        return None, 'invalid'
    try:
        d = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None, 'invalid'
    kind = 'date_only' if DAY.fullmatch(value) else ('naive_assumed_UTC' if d.tzinfo is None else 'offset_timestamp')
    return (d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)), kind


def event_interval(value):
    stamp, kind = parse_time(value)
    if stamp is None:
        return None, None, kind
    if kind == 'date_only' or (stamp.hour, stamp.minute, stamp.second, stamp.microsecond) == (0, 0, 0, 0):
        return stamp, stamp+timedelta(days=1), 'day_resolution_or_midnight_unverified'
    return stamp, stamp, 'stored_timestamp_not_authenticated'


def interval_relation(lo, hi, anchor, horizon):
    """An interval's end is exclusive; a timestamp has lo == hi."""
    if lo is None or hi is None or anchor is None:
        return 'MISSING_OR_INVALID_TIME'
    deadline = anchor+timedelta(days=horizon)
    if hi <= anchor:
        return 'ALREADY_ADDED_AT_OR_BEFORE_ANCHOR'
    if lo > deadline:
        return 'AFTER_HORIZON'
    if lo > anchor and hi <= deadline:
        return 'WITHIN_HORIZON_STORED_TIME_SCENARIO'
    return 'BOUNDARY_OR_EVENT_ORDER_UNCERTAIN'


def timing(record):
    pub, pub_kind = parse_time(record.get('published_timestamp'))
    day, _ = parse_time(record.get('published_date'))
    added, add_kind = parse_time(record.get('vckev_date_added'))
    lo, hi, resolution = event_interval(record.get('vckev_date_added'))
    delta = (added-pub).total_seconds()/86400 if added is not None and pub is not None else None
    signed_day = (added.date()-pub.date()).days if delta is not None else None
    listed = record.get('in_vckev') is True
    if not listed:
        category = 'NO_SAVED_LISTING'
    elif delta is None:
        category = 'MISSING_OR_INVALID_TIME'
    elif delta < 0:
        category = 'EARLIER_TIME_SAME_DAY' if signed_day == 0 else 'EARLIER_DAY'
    elif delta == 0:
        category = 'SAME_STORED_TIMESTAMP'
    elif delta <= 60:
        category = 'AFTER_PUBLICATION_UP_TO_60_DAYS'
    else:
        category = 'AFTER_60_DAYS'
    return {'published_timestamp_kind': pub_kind, 'addition_timestamp_kind': add_kind,
            'addition_time_resolution': resolution, 'delta_days_exact_stored': delta,
            'day_difference_ignoring_time': signed_day, 'timing_category': category,
            'stored_nonnegative_within_60': listed and delta is not None and 0 <= delta <= 60,
            'date_only_within_60': listed and signed_day is not None and 0 <= signed_day <= 60,
            'publication_date_timestamp_disagree': day is not None and pub is not None and day.date() != pub.date()}, pub, lo, hi


def candidate_train_ids(groups, publications, intervals):
    selected, excluded = set(), set(); represented = set()
    for g in groups:
        members = set(g['current_cves'])
        REQUIRE(not represented & members, 'Repeated CVE across evidence groups'); represented.update(members)
        prior = set(g['prior_roles'])
        roles = {next((role for role, (lo, hi) in intervals.items() if lo <= publications[c] <= hi), 'outside') for c in members}
        eligible = roles == {'train'} and prior <= {'train'} and not g['connected_to_prior_test']
        (selected if eligible else excluded).update(members)
    REQUIRE(represented == set(publications), 'Incomplete component metadata')
    return selected, excluded


def select_scope(dataset, cfg, paths):
    epss = READ(paths['epss_protocol'])
    REQUIRE(all(r['outer_role'] == 'train' for r in epss['rows'].values()), 'EPSS reference contains non-training roles')
    epss_ids = set(epss['rows'])
    manifest = READ(paths['roles'])
    if dataset == 'reddit':
        REQUIRE(manifest['post_policy'] == 'retain_all' and manifest['exposure_policy'] == 'retain_matching_prior_roles'
                and not manifest['test_release_enabled'], 'Unexpected frozen split policy')
        publications = {}
        # Read IDs/publication metadata only; never inspect held-out text or scores.
        with gzip.open(paths['candidates'], 'rt') as stream:
            for line in stream:
                r = json.loads(line)
                REQUIRE(r['cve'] not in publications, 'Duplicate metadata CVE')
                publications[r['cve']] = r['published_date']
        ids, excluded = candidate_train_ids(READ(paths['components']), publications, manifest['publication_intervals'])
        blocked = {c for task in manifest['tasks'].values() for role in ('validation', 'test') for c in task['ids'][role]}
        REQUIRE(epss_ids == set(manifest['tasks']['RD02']['ids']['train']), 'RD02 reference role mismatch')
        basis = 'Frozen RD01 publication/evidence rules before EPSS availability and Day-60 maturity filters'
    else:
        ids = {c for c, r in manifest.items() if r['split'] == 'train'}
        blocked = set(manifest)-ids
        REQUIRE(ids == set(READ(paths['kev_identity'])['ids']['train']), 'Earlier classifier role IDs differ')
        basis = 'Existing classifier training roles; no new assignment or Day-60 availability filter'
    REQUIRE(len(ids) == cfg['expected_candidates'] and len(epss_ids) == cfg['expected_epss_train'], 'Cohort count changed')
    REQUIRE(epss_ids <= ids and not ids & blocked, 'Training role scope violation')
    return ids, epss_ids, basis


def train_records(path, ids):
    raw = READ(path)
    REQUIRE(isinstance(raw, list), 'Dataset must be a list')
    seen, records = set(), {}
    for r in tqdm(raw, desc='Select training IDs before inspecting source values', unit='CVE'):
        c = r['cve']; REQUIRE(c not in seen, 'Duplicate raw CVE'); seen.add(c)
        if c in ids:
            records[c] = r
    REQUIRE(set(records) == ids, 'Missing selected training CVEs')
    return records


def receipt_state(receipt, cve, directory):
    REQUIRE(receipt['cve'] == cve, 'Receipt CVE mismatch')
    receipts.verify_receipt(receipt, directory)
    started, sk = parse_time(receipt.get('request_started_at_utc'))
    checked, ck = parse_time(receipt.get('checked_at_utc'))
    REQUIRE(started is not None and checked is not None and sk == ck == 'offset_timestamp'
            and started <= checked, 'Receipt lacks consistent explicit-UTC timestamps')
    REQUIRE(receipt.get('label_date_basis') == 'fresh_lookup_not_original_snapshot', 'Unreviewed receipt semantics')
    value = receipt['verified_in_vckev']
    REQUIRE(value is None or type(value) is bool, 'Nonboolean receipt outcome')
    return ('LISTED' if value is True else 'NOT_LISTED' if value is False else 'UNKNOWN'), checked


def audit_receipts(folder, records, source_hash):
    pins, latest = {}, {}
    if folder is None or not folder.exists():
        return latest, pins, {'directory': str(folder) if folder else None, 'exists': False,
                              'scope': 'No receipts in the documented/explicit directory; not a filesystem-wide claim'}
    p = READ(folder/'protocol.json'); pins[str(folder/'protocol.json')] = SHA(folder/'protocol.json')
    REQUIRE(p['schema'] == 'reddit_kev_recheck_v1' and p['endpoint'] == receipts.ENDPOINT
            and p['source_sha256'] == source_hash and p['script_sha256'] == SHA(Path(receipts.__file__)),
            'Receipt protocol/source/collector identity requires review')
    REQUIRE(len(p['ids']) == len(set(p['ids'])), 'Duplicate receipt-plan IDs')
    for c in tqdm(sorted(set(records)&set(p['ids'])), desc='Verify training-only catalogue receipts', unit='CVE'):
        directory = folder/'receipts'/c; attempts = sorted(directory.glob('*.receipt.json'))
        last_time = None; last_attempt = 0
        for path in attempts:
            r = READ(path); state, checked = receipt_state(r, c, directory)
            REQUIRE(r['legacy_in_vckev'] == records[c].get('in_vckev'), 'Receipt legacy flag differs')
            REQUIRE(type(r['attempt']) is int and r['attempt'] > last_attempt
                    and (last_time is None or checked >= last_time), 'Unordered receipt attempts')
            last_time, last_attempt = checked, r['attempt']; pins[str(path)] = SHA(path)
            if r.get('response_file'):
                pins[str(directory/r['response_file'])] = r['response_sha256']
            latest[c] = {'verified_state_at_check': state, 'verified_target_at_check': r['verified_in_vckev'],
                         'checked_at_utc': r['checked_at_utc'], 'receipt_status': r['status']}
    return latest, pins, {'directory': str(folder), 'exists': True, 'training_cves_with_receipts': len(latest),
                          'not_original_snapshot': True, 'other_roles_receipts_opened': False}


def documents(record):
    for source, field in SOURCES.items():
        text = record.get(field)
        if isinstance(text, str) and text.strip():
            yield source, text, {}
    for item in record.get('social_media_posts') or []:
        if isinstance(item, dict) and isinstance(item.get('post'), str) and item['post'].strip():
            yield 'M', item['post'], item


def screen_documents(record, pub, epss_member):
    rows = []; seen = set()
    for source, text, meta in documents(record):
        digest = hashlib.sha256(text.encode()).hexdigest()
        identity = (source, digest, str(meta.get('date')), str(meta.get('platform')))
        if identity in seen:
            continue
        seen.add(identity)
        flags = text_policy.inspect(text, record['cve'], POLICY_SOURCES[source])
        stamp, kind = parse_time(meta.get('date'))
        relation = ('BEFORE_PUBLICATION' if stamp.date() < pub.date() else 'PUBLICATION_DAY'
                    if stamp.date() == pub.date() else 'AFTER_PUBLICATION') if stamp and pub else 'UNAVAILABLE'
        rows.append({'cve': record['cve'], 'source': source, 'text_sha256': digest, 'characters': len(text),
                     'in_epss_training_subset': epss_member, 'saved_listed_flag': record.get('in_vckev'),
                     'platform': str(meta.get('platform') or ''), 'stored_post_date': meta.get('date'),
                     'date_kind': kind, 'date_relative_to_publication': relation,
                     'date_screen_by_day30': bool(stamp and pub and stamp.date() <= (pub+timedelta(days=30)).date()),
                     'post_id_field_present': any(meta.get(k) not in (None, '') for k in ('id', 'post_id', 'permalink', 'url')),
                     'capture_field_present': any(meta.get(k) not in (None, '') for k in ('retrieved_at', 'captured_at', 'scraped_at')),
                     'history_field_present': any(k in meta for k in ('edited', 'edit_history', 'revisions')),
                     'historical_wording_verified': False,
                     'historical_wording_reason': 'No audited archive/content-version evidence linked to this exact text hash',
                     'bare_cve_only': bool(common.CVE.fullmatch(text.strip())),
                     'lexical_screen': flags['decision'], 'screen_reasons': flags['reasons'],
                     'exploitation_claim_candidate': bool(EXPLOIT.search(text)),
                     'target_cve_mentioned': flags['target_cve_mentioned'], 'other_cve_ids': flags['other_cve_ids']})
    return rows


def feature_policy():
    return {'catalogue_fields_excluded': EXCLUDED,
            'also_excluded': ['EPSS scores/metadata for this initial catalogue baseline',
                              'catalogue-derived missingness/counts/graph nodes/edges',
                              'source audit decisions, label check status, future event dates, CVE ID as numeric feature'],
            'candidates_not_approved_predictors': ['D/G/N/M text after independent source review',
                                                  'CVSS fields with value-observed flags',
                                                  'deduplicated reference counts from non-catalogue provenance'],
            'temporal_requirement': 'Every value and exact text version must be supported as available at or before anchor',
            'snapshot_requirement': 'Declare retrospective association, not future exploitation; review direct label disclosures',
            'lexical_pass_is_not_relevance_or_historical_verification': True}


def report(dataset, summary, support, timing_counts, provenance):
    lines = [f'CT08 {dataset}: training-only VulnCheck target readiness', '',
             '1. Scope', summary['cohort_basis'],
             f"Audit candidates: {summary['candidate_cves']:,}; existing EPSS training: {summary['epss_training_cves']:,}; "
             f"additional candidates without requiring a Day-60 score: {summary['additional_candidates']:,}.",
             'Candidates are not approved training examples. No official validation/test text or outcomes are analyzed.',
             'Reddit retains its existing publication-date envelope; this does not authorize expanding to older years.',
             'Older roles come from the earlier classifier; that study and its test evaluation are not erased.', '',
             '2. Saved flags versus verified checks',
             'Saved true/false flags are retained. UNKNOWN means no usable saved verification for this CVE at the relevant time.',
             'A failed request is never NOT_LISTED. A fresh check describes its own time, not the historical snapshot.',
             'The absence of a CVE in the catalogue does not establish absence of exploitation.',
             'Population | CVEs | Saved listed | Saved false | Verified listed at check | Verified not listed at check | Unverified']
    for r in support:
        lines.append(' | '.join(str(r[k]) for k in ('population', 'cves', 'saved_listed', 'saved_false',
                                                  'verified_listed', 'verified_not_listed', 'unknown')))
    lines += ['', '3. Publication-to-addition timing',
              'Addition minus NVD publication uses stored timestamps only. Neither event is first exploitation time.',
              'Negative differences remain visible. Earlier times on the same day may reflect missing time-of-day information.',
              'Midnight additions are conservatively treated as day intervals, not authenticated exact event times.',
              'Population | Timing category | CVEs']
    lines += [f"{r['population']} | {r['category']} | {r['cves']}" for r in timing_counts]
    lines += ['', 'The horizon table uses (anchor, anchor + horizon] in a stored-time sensitivity scenario only.',
              'Events already recorded at/before the anchor are not future positives. Ambiguous boundary cases remain unresolved.',
              'No future-inclusion training labels are released. Unknown cutoff/completeness cannot produce a zero label.', '',
              '4. Text provenance and leakage',
              'Source | Population | Saved documents | CVEs with text | Lexical-pass CVEs | Dated M by Day 30 | Historically verified text CVEs']
    lines += [f"{r['source']} | {r['population']} | {r['documents']} | {r['cves_with_text']} | {r['lexical_pass_cves']} | "
              f"{r['date_screen_day30_cves']} | {r['verified_historical_cves']}" for r in provenance]
    lines += ['D = description; G/N = generated GitHub/non-GitHub webpage summary; M = saved social posts.',
              'Deduplication here is only exact identical attachments within each CVE; shared documents across CVEs stay intact.',
              'These counts describe raw saved sources, not the earlier EPSS accepted-source masks.',
              'Joined summary overlays are outside this raw-source audit; missing raw summaries do not mean overlays never existed.',
              'Passing the reused lexical screen is not proof of correct target relevance or absence of leakage.',
              'A post date alone does not prove its saved wording existed then. An updated NVD date is not description history.',
              'URL reference-addition dates do not date generated summaries. Present CVSS values are not automatically historical.', '',
              '5. Target choices and decision',
              'Snapshot classification: predict listed/not-listed in one defined catalogue snapshot, not exploitation probability.',
              'This requires verified outcomes, an agreed collection-time policy, and a reviewed disclosure/source policy.',
              'Future inclusion: predict a new entry after an explicit anchor among CVEs not already listed at that anchor.',
              'This additionally requires authentic entry-time semantics, historical risk status, mature follow-up and cutoff-valid text/features.',
              'Neither task is ready for new training under this audit. There is no automatic promotion based on old true flags.',
              'Next: obtain/review bounded TRAINING-ONLY authenticated receipts and exact content lineage; do not query old test examples.',
              'Bounded receipts validate the collection path, not completeness of labels for the entire training cohort.',
              'A current complete snapshot cannot reconstruct earlier missing receipts. Do not overwrite old labels or checkpoints.',
              'No networks, models, embeddings, split changes, threshold tuning or official holdout scoring were run.']
    return '\n'.join(lines)+'\n'


def run(dataset, verify=False, receipt_dir=None, output_dir=None, check_only=False):
    spec = READ(CONFIG); cfg = spec['datasets'][dataset]
    REQUIRE(not any(spec[k] for k in ('training_enabled', 'network_enabled', 'test_access_enabled')), 'Forbidden execution scope')
    paths = {k: (ROOT/v).resolve() for k, v in cfg.items() if k+'_sha256' in cfg}
    pins = {str(p): cfg[k+'_sha256'] for k, p in paths.items()}
    code = [CONFIG, CLI, CLI.with_name('test_catalogue_readiness.py'), Path(__file__), Path(__file__).with_name('__init__.py'),
            Path(receipts.__file__), Path(common.__file__), Path(text_policy.__file__), Path(artifact_io.__file__)]
    pins.update({str(p): SHA(p) for p in code})
    for path, expected in tqdm(pins.items(), desc='Verify dataset and role identities', unit='file'):
        REQUIRE(SHA(Path(path)) == expected, 'Input changed: '+path)
    ids, epss_ids, basis = select_scope(dataset, cfg, paths)
    if check_only:
        print(f'CT08 {dataset} preflight: {len(ids)} train-role audit candidates; {len(epss_ids)} existing EPSS train. No labels or text audited.'); return
    records = train_records(paths['raw'], ids)
    folder = receipt_dir if receipt_dir is not None else (ROOT/cfg['default_receipts'] if cfg['default_receipts'] else None)
    verified, receipt_pins, receipt_inventory = audit_receipts(folder, records, cfg['raw_sha256']); pins.update(receipt_pins)
    rows, docs, horizons = [], [], []
    for c in tqdm(sorted(ids), desc='Audit training labels, event order and text provenance', unit='CVE'):
        raw = records[c]; event, pub, lo, hi = timing(raw)
        flag = raw.get('in_vckev'); saved = 'LISTED' if flag is True else 'NOT_LISTED_UNVERIFIED' if flag is False else 'UNKNOWN'
        receipt = verified.get(c, {'verified_state_at_check': 'UNKNOWN', 'verified_target_at_check': None,
                                  'checked_at_utc': None, 'receipt_status': 'NO_SAVED_RECEIPT'})
        pub_day, _ = parse_time(raw.get('published_date'))
        day60 = (pub_day+timedelta(days=60)).date().isoformat() if pub_day else None
        scores = raw.get('epss_scores')
        has_score = isinstance(scores, dict) and day60 in scores and scores[day60] is not None
        rows.append({'cve': c, 'in_epss_training_subset': c in epss_ids, 'day60_observed_not_an_eligibility_rule': has_score,
                     'saved_in_vckev': flag, 'saved_catalogue_state': saved, **receipt,
                     'original_snapshot_verified_state': 'UNKNOWN', 'future_horizon_target': None,
                     'snapshot_training_eligible': False, 'temporal_training_eligible': False,
                     'published_timestamp': raw.get('published_timestamp'), 'vckev_date_added': raw.get('vckev_date_added'),
                     **event, 'blocking_reasons': ['NO_REVIEWED_COMMON_SNAPSHOT_POLICY', 'NO_VERIFIED_HISTORICAL_LABEL_LINEAGE',
                                                  'NO_VERIFIED_CUTOFF_TEXT_LINEAGE', 'SOURCE_REVIEW_REQUIRED']
                     + (['NO_USABLE_SAVED_RECEIPT'] if receipt['verified_state_at_check'] == 'UNKNOWN' else [])})
        docs.extend(screen_documents(raw, pub, c in epss_ids))
        for anchor in spec['temporal_anchor_days']:
            for horizon in spec['horizons_days']:
                relation = interval_relation(lo, hi, pub+timedelta(days=anchor) if pub else None, horizon) if flag is True else 'NO_VERIFIED_EVENT_OR_ABSENCE'
                horizons.append({'cve': c, 'in_epss_training_subset': c in epss_ids, 'anchor_days_after_publication': anchor,
                                 'horizon_days_after_anchor': horizon, 'stored_time_scenario': relation, 'released_target': None})
    support, times, prov, hs = [], [], [], []
    for name in ('all_candidates', 'epss_training_subset', 'additional_candidates'):
        chosen = {r['cve'] for r in rows if name == 'all_candidates' or r['in_epss_training_subset'] == (name == 'epss_training_subset')}
        rs = [r for r in rows if r['cve'] in chosen]
        support.append({'population': name, 'cves': len(rs), 'saved_listed': sum(r['saved_in_vckev'] is True for r in rs),
                        'saved_false': sum(r['saved_in_vckev'] is False for r in rs),
                        'verified_listed': sum(r['verified_state_at_check'] == 'LISTED' for r in rs),
                        'verified_not_listed': sum(r['verified_state_at_check'] == 'NOT_LISTED' for r in rs),
                        'unknown': sum(r['verified_state_at_check'] == 'UNKNOWN' for r in rs),
                        'date_only_within60': sum(r['date_only_within_60'] for r in rs),
                        'stored_nonnegative_within60': sum(r['stored_nonnegative_within_60'] for r in rs),
                        'day60_observed': sum(r['day60_observed_not_an_eligibility_rule'] for r in rs)})
        times += [{'population': name, 'category': k, 'cves': v} for k, v in sorted(Counter(r['timing_category'] for r in rs).items())]
        for source in ('D', 'G', 'N', 'M'):
            ds = [d for d in docs if d['cve'] in chosen and d['source'] == source]
            prov.append({'population': name, 'source': source, 'documents': len(ds), 'cves_with_text': len({d['cve'] for d in ds}),
                         'lexical_pass_cves': len({d['cve'] for d in ds if d['lexical_screen'] == 'accept' and not d['bare_cve_only']}),
                         'catalogue_or_disclosure_screened_documents': sum(d['lexical_screen'] == 'quarantine' for d in ds),
                         'bare_id_documents': sum(d['bare_cve_only'] for d in ds),
                         'date_screen_day30_cves': len({d['cve'] for d in ds if d['date_screen_by_day30']}),
                         'verified_historical_cves': 0})
        counts = Counter((r['anchor_days_after_publication'], r['horizon_days_after_anchor'], r['stored_time_scenario']) for r in horizons if r['cve'] in chosen)
        hs += [{'population': name, 'anchor_day': a, 'horizon_days': h, 'scenario': status, 'cves': n} for (a, h, status), n in sorted(counts.items())]
    REQUIRE({r['cve'] for r in rows} == ids and {r['cve'] for r in docs} <= ids, 'Non-training artifact row')
    summary = {'schema': spec['schema'], 'dataset': dataset, 'cohort_basis': basis, 'candidate_cves': len(ids),
               'epss_training_cves': len(epss_ids), 'additional_candidates': len(ids-epss_ids),
               'receipt_inventory': receipt_inventory, 'readiness': 'BLOCKED_LABEL_TIME_AND_SOURCE_REVIEW',
               'snapshot_training_enabled': False, 'temporal_training_enabled': False,
               'support': support, 'timing_counts': times, 'source_coverage': prov, 'horizon_scenarios': hs,
               'official_validation_test_text_or_outcomes_analyzed': False, 'network_requests': 0, 'training': False}
    request_ids = []
    # Deterministic bounded machinery-check sample; no held-out ID may be sent to a collector.
    for value in (True, False):
        request_ids += sorted(c for c in ids if records[c].get('in_vckev') is value)[:10]
    selected_examples = []
    for category in ('EARLIER_TIME_SAME_DAY', 'EARLIER_DAY', 'AFTER_PUBLICATION_UP_TO_60_DAYS'):
        selected_examples += [r for r in rows if r['timing_category'] == category][:2]
    artifacts = {'identity.json': {'input_hashes': pins, 'spec': spec, 'receipt_inventory': receipt_inventory,
                                  'candidate_ids_sha256': hashlib.sha256(common.dumps(sorted(ids)).encode()).hexdigest()},
                 'findings.json': summary, 'training_scope_ids.json': sorted(ids), 'future_feature_policy.json': feature_policy(),
                 'receipt_request_plan.json': {'ids': sorted(request_ids), 'scope': 'Training-only bounded check; no collection performed',
                                               'source_sha256': cfg['raw_sha256'], 'full_cohort_receipts_ready': False,
                                               'not_a_reconstruction_of_old_snapshot': True},
                 'real_timing_examples.json': selected_examples,
                 'REPORT.txt': report(dataset, summary, support, times, prov)}
    for name, rs in [('cve_readiness', rows), ('source_documents_audit_only', docs), ('horizon_scenarios', hs),
                     ('cve_horizon_scenarios_audit_only', horizons),
                     ('cohort_support', support), ('timing_categories', times), ('source_coverage', prov)]:
        REQUIRE(bool(rs), 'Empty required audit table: '+name)
        artifacts[name+'.csv'] = csv_text(rs, list(rs[0]))
    destination = output_dir or ROOT/spec['output']/cfg['output_name']/'audit'
    publish(destination, artifacts, pins, verify)
    print(f"CT08 {dataset} readiness {'reproduced' if verify else 'saved'}: {destination}. No training, network calls or holdout scoring.")
