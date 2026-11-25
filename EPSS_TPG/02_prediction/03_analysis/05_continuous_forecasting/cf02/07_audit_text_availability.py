"""Audit saved historical-text evidence for CF01 development IDs only; CPU, no fitting."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, timedelta
import hashlib
from pathlib import Path
import sys

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / '01_package'))
from epss.continuous_forecasting import data
from epss import forecast_multisource as ms

OUTPUT = data.OUTPUT.parent / 'cf02_text_eligibility_v1'
AUDIT = data.ROOT / '05_results/03_dataset_analysis/01_revision_dataset/14_smp_date_verification'


def match_post(post, evidence):
    """A match transfers the earlier audit's evidence, not publication authentication."""
    data.require(set(post) <= {'post', 'platform', 'date'}, 'New post fields require provenance review')
    data.require(hashlib.sha256(post['post'].encode()).hexdigest() == evidence['post_sha256']
                 and post.get('platform') == evidence['platform']
                 and post.get('date') == evidence['candidate_timestamp'],
                 'Saved provenance audit does not match current post text/platform/date')
    data.require(not evidence['historical_text_verified']
                 and not evidence['post_publication_verified']
                 and evidence['verified_publication_timestamp'] is None
                 and 'HISTORICAL_TEXT_VERSION_UNVERIFIED' in evidence['publication_evidence_missing'],
                 'New publication/content evidence requires a fresh eligibility decision')


def write_csv(path, rows):
    data.require(bool(rows), 'No audit rows')
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r['split'], r['cve'], r['source'])))


def run(verify=False):
    series, identity = data.load('full')
    paths = [Path(p) for p in identity['source_pins'] if p.endswith('epss_recollected_2026-09-30_smp.json')]
    data.require(len(paths) == 1, 'Ambiguous source')
    source = paths[0]
    provenance_paths = [AUDIT / n for n in ('smp_dated_epss_analysis.json', 'smp_candidate_event_audit.json')]
    pins = {str(p): data.sha(p) for p in [source, *provenance_paths, Path(__file__), ms.CONFIG,
            Path(ms.__file__), data.OUTPUT / 'preflight/full/artifact_lock.json']}
    prior = data.read(provenance_paths[0])
    data.require(prior['source_sha256'] == pins[str(source)]
                 and prior['post_text_platform_and_order_unchanged'], 'Dated-source audit mismatch')
    wanted = {r['cve'] for rows in series.values() for r in rows}
    # Decode the container, then adapt only development records. Test fields are not audited.
    selected = [r for r in data.read(source) if r['cve'] in wanted]
    raw = {r['cve']: r for r in selected}
    data.require(len(raw) == len(selected) == len(wanted), 'Missing/repeated development record')
    prior_posts = [r for r in data.read(provenance_paths[1]) if r['cve'] in wanted]
    evidence = {(r['cve'], r['post_index']): r for r in prior_posts}
    data.require(len(evidence) == len(prior_posts), 'Repeated provenance key')
    fields = data.read(ms.CONFIG)['source_fields']
    coverage, rows, platforms, schemas = [], [], [], defaultdict(set)
    details, examples, matched = {}, [], set()
    for split, records in series.items():
        counts, platform_counts = Counter(), defaultdict(Counter)
        totals = {s: Counter() for s in ms.SOURCES}
        for record in tqdm(records, desc='Audit Day-30 text evidence: ' + split, unit='CVE'):
            cve = record['cve']
            r = raw[cve]
            anchor = (date.fromisoformat(record['publication_date']) + timedelta(days=30)).isoformat()
            primary = bool(data.eligible(record)[30, 29])
            counts['primary_eligible_cves'] += primary
            schemas['record_keys'].update(r)
            adapted = ms.adapt_record(r, anchor, fields)
            modified = ms.stored_date(r.get('last_modified_timestamp'), anchor)['relation']
            counts['nvd_modified_' + modified] += 1
            counts['description_equals_orig_description'] += r.get('description') == r.get('orig_description')
            for field in ('url_log_github_only_urls', 'url_log_non_github_urls'):
                value = r.get(field)
                schemas[field + '_types'].add(type(value).__name__)
                data.require(value is None or isinstance(value, str), 'New structured fetch-log evidence: review required')
            for ref in r.get('url_change_dates', []):
                schemas['reference_history_keys'].update(ref)
                counts['reference_added_' + ms.stored_date(ref.get('earliest_added_date'), anchor)['relation']] += 1
            for i, post in enumerate(r[fields['smp']]):
                key = (cve, i)
                data.require(key in evidence, 'Missing prior post evidence')
                e = evidence[key]
                match_post(post, e)
                matched.add(key)
                schemas['post_keys'].update(post)
                parsed = ms.stored_date(post.get('date'), anchor)
                pc = platform_counts[post.get('platform') or 'not_supplied']
                pc['post_associations'] += 1
                pc[parsed['relation']] += 1
                pc['explicit_timezone'] += parsed['timezone'] != 'not_supplied'
                pc['raw_platform_candidates'] += bool(e['raw_platform_candidates'])
                for gap in e['publication_evidence_missing']:
                    counts['post_gap:' + gap] += 1
            for s in ms.SOURCES:
                docs = [d for d in adapted['documents'] if d['source_text_type'] == s]
                candidates = sum(d['view_eligible']['retrospective_date_screened'] for d in docs) if s == 'smp' else 0
                conflicts = sum('same_platform_text_has_conflicting_dates' in d['quality_flags'] for d in docs)
                row = dict(split=split, cve=cve, source=s, day30=anchor, primary_eligible=primary,
                           documents=len(docs), date_screened_smp_documents=candidates,
                           conflicting_smp_documents=conflicts, verified_historical_documents=0,
                           eligibility='NOT_ESTABLISHED' if docs else 'NO_TEXT',
                           text_hashes=';'.join(sorted(d['text_sha256'] for d in docs)))
                rows.append(row)
                t = totals[s]
                t['cves_with_text'] += bool(docs)
                t['documents'] += len(docs)
                t['date_screened_smp_cves'] += bool(candidates)
                t['date_screened_smp_documents'] += candidates
                t['conflicting_smp_documents'] += conflicts
                t['primary_cves_with_text'] += primary and bool(docs)
                t['primary_date_screened_smp_cves'] += primary and bool(candidates)
            if cve == 'CVE-2022-46908':
                examples.append(dict(cve=cve, split=split, day30=anchor,
                    publication=r['published_timestamp'], nvd_modified=r['last_modified_timestamp'],
                    description=r['description'], references=r['url_change_dates'], posts=r['social_media_posts']))
        coverage.extend(dict(split=split, source=s, cohort_cves=len(records), **totals[s],
                             verified_historical_cves=0, verified_historical_documents=0) for s in ms.SOURCES)
        platforms.extend(dict(split=split, platform=p, **v) for p, v in sorted(platform_counts.items()))
        details[split] = dict(counts)
    data.require(matched == set(evidence), 'Not all selected prior post associations matched')
    findings = dict(schema='cf02_saved_text_eligibility_v1', source_pins=pins,
        cohort_sizes={s:len(v) for s,v in series.items()}, coverage=coverage, details=details,
        platform_counts=platforms, schemas={k:sorted(v) for k,v in schemas.items()}, examples=examples,
        matched_prior_post_associations=len(matched), decision='BLOCK_TEMPORAL_TEXT_FIT_CONSOLIDATE_REPORTS',
        evidence_scope='Pinned dataset and saved provenance audit; no new external authentication. '
                       'Historical eligibility not established is not proof all text was unavailable.',
        training=False, inference=False, test_adaptation=False,
        rationale={
          'D':'Current description and orig_description lack version-linked historical content evidence. '
              'An update after Day 30 does not prove the wording changed then.',
          'G_N':'Summary fields and string fetch logs do not provide authenticated earlier source-content versions '
                'and summary-to-content lineage. NVD link-addition dates date links, not page text.',
          'M':'All selected post text hashes, platforms and dates match the saved provenance audit. '
              'Raw candidates are not authenticated post/content versions; unresolved identity, timezone and '
              'historical wording gaps remain. Existing false eligibility flags are not independent evidence.',
          'all_anchors':'The missing content-version provenance is not repaired by moving the forecast anchor. '
                        'No strict temporal-text case is established by these saved artifacts.'})
    lines = ['CF02: can saved text support historical daily EPSS forecasts?', '',
             'Decision: do not run a temporal-text fit with the present evidence. Consolidate the completed studies.',
             'Scope: CF01 development CVEs only; no training, inference, external lookup or test adaptation.', '',
             'Text presence is not historical availability. Counts below are before relevance filtering.',
             'Each G/N summary is one document. M documents deduplicate identical platform/text within each CVE.',
             'Date-screened means every stored post date is strictly before Day 30, with no conflicting dates.',
             'This screening does NOT authenticate the post date or saved wording.', '',
             'Split | Source | CVEs with text | Documents | Date-screened M CVEs | Verified historical CVEs']
    for r in coverage:
        candidate = str(r['date_screened_smp_cves']) if r['source']=='smp' else 'not applicable'
        lines.append(f"{r['split']} | {r['source']} | {r['cves_with_text']} | {r['documents']} | {candidate} | 0")
    lines += ['', 'Primary Day-30 to Day-60 eligibility: ' + str({s:v['primary_eligible_cves'] for s,v in details.items()}),
              'Post associations matched to prior provenance: ' + str(len(matched)), '',
              *[k + ': ' + v for k,v in findings['rationale'].items()], '',
              'The zero verified count is a conservative evidence decision, not proof that every source contains future information.',
              'Date-screened posts can support explicitly conditional descriptive analysis, not a verified historical forecast.',
              'A summary generated later can be usable if reconstructed solely from authenticated pre-anchor content under a fixed policy.',
              'The current files do not supply that lineage. A hash computed today alone cannot prove earlier availability.', '',
              'Interpretation of CF01 Stage 3:',
              'Flat-history sensitivity is very small in these checkpoints; this is not a universal impossibility result.',
              'Last-only changes value availability, masks, age and known-history together. It does not isolate one GRU mechanism.',
              'The validation-derived constant offset is a descriptive decomposition, not a deployable training-only control.', '',
              'Next reporting task:',
              'Separate retrospective Day-60 estimation, snapshot VulnCheck classification and daily EPSS forecasting.',
              'Retain constants, persistence, all seeds, negative results and the distinction between development and held-out evaluation.',
              'Do not rerun a test, promote a subgroup winner or weaken provenance rules to obtain a text result.',
              'To reopen CF02, supply version-linked text, authoritative availability evidence, post/commit identities and timezone rules.',
              'Then audit eligibility before defining matched text/no-text cells.']
    with data.transaction(OUTPUT, verify) as stage:
        (stage/'findings.json').write_text(data.dump(findings))
        (stage/'findings.txt').write_text('\n'.join(lines)+'\n')
        write_csv(stage/'cve_source_evidence.csv', rows)
        data.check_pins(pins)
    print('\n'.join(lines[:21]))
    print('CF02 saved-text audit ' + ('reproduced.' if verify else 'saved.') + ' No training or test adaptation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    run(parser.parse_args().verify_only)
