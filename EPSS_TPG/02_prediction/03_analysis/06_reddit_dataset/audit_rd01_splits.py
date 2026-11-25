"""RD01 bridge, chronological qualification and deduplicated event coverage audit."""
import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
import csv
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile

import networkx as nx
from tqdm import tqdm

from audit_reddit import ROOT, OLD, OLD_ROLES, CVE, read, sha, dumps, require, fingerprint
from reddit_gates import RESULTS, SOURCE, UnionFind, verify_output, lock_output

CONFIG = ROOT/'02_prediction/04_configs/05_reddit_studies/RD01_diagnostic_v1.json'
SPEC = ROOT/'02_prediction/04_configs/05_reddit_studies/RD01_spec.json'
SAN = RESULTS/'stage1_sanitization'
OUT = RESULTS.parent/'06_rd01_diagnostic_v1'
COLLECTOR = SOURCE/'repository/Sec4AI4Sec-EPSS/Data_Files/fetch_epss_data.py'
ROLES = ('train','validation','test')


def jsonl(path):
    with gzip.open(path,'rt') as f:
        for line in f:
            yield json.loads(line)


def csv_file(path, rows, fields):
    opener = (lambda: io.TextIOWrapper(gzip.GzipFile(filename='', mode='wb', fileobj=path.open('wb'),mtime=0),encoding='utf-8',newline='')) if path.suffix=='.gz' else (lambda: path.open('w',newline=''))
    with opener() as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n'); w.writeheader()
        for row in rows:
            w.writerow(row)


def role_for(pub, protocol):
    return next((r for r in ROLES if protocol['publication_intervals'][r][0]<=pub<=protocol['publication_intervals'][r][1]), 'outside')


def post_group(text):
    mentions=set(m.upper() for m in CVE.findall(text))
    if CVE.fullmatch(text.strip()):
        return 'bare_id_only'
    require(bool(mentions), 'Post without a CVE mention requires a separate category')
    return 'multi_cve_text' if len(mentions)>1 else 'single_cve_descriptive'


def keep_post(degree, threshold):
    return threshold is None or degree<=threshold


def groups_for(records, posts, threshold=None, exact=False):
    uf=UnionFind(records)
    for p in posts.values():
        if keep_post(len(p['cves']),threshold):
            uf.join(p['cves'] if exact else p['endpoints'])
    groups=[]
    for members in uf.groups(set(uf.parent)).values():
        current=sorted(set(members)&records.keys())
        if current:
            groups.append({'current':current,'all':sorted(members)})
    return sorted(groups,key=lambda g:g['current'][0])


def daily_cells(observed_indices, max_target=90):
    """Each observed u permits anchors from first observation through u-1."""
    if not observed_indices:
        return 0
    first=min(observed_indices)
    return sum(u-first for u in observed_indices if first<u<=min(90,max_target))


def component_reason(members, records, exposure, mode, protocol):
    # Fixed priority makes exclusion counts mutually exclusive and reproducible.
    rs={role_for(records[c]['published_date'],protocol) for c in members}
    old=set().union(*(exposure[c] for c in members))
    if 'test' in old:
        return 'prior_test_connected', None
    if mode=='never_exposed' and old:
        return 'any_prior_exposure', None
    if len(rs)>1:
        return 'cross_publication_intervals', None
    role=next(iter(rs))
    if role=='outside':
        return 'outside_publication_intervals', None
    if role=='test' and old:
        return 'prior_exposure_barred_from_test', None
    if mode=='retain_matching_prior_roles' and old and old!={role}:
        return 'prior_role_conflict', None
    return None,role


def event_window(pub, postdate, observed_dates, statuses, width, releases):
    start,end=postdate-timedelta(days=width),postdate+timedelta(days=width)
    days=[(start+timedelta(days=i)).isoformat() for i in range(2*width+1)]
    outside=start<pub or end>pub+timedelta(days=90)
    missing=Counter(statuses.get(day,'OUTSIDE_STORED_WINDOW') for day in days if day not in observed_dates)
    return {'complete':not missing,'reason':'outside_stored_window' if outside else ('missing_daily_score' if missing else 'complete'),
            'missing_reasons':dict(sorted(missing.items())),
            'crosses_release':any(start<r<=end for r in releases)}


def load(protocol):
    for folder in [SAN,RESULTS/'stage0',RESULTS/'stage1_epss',RESULTS/'stage2_4_specifications_v2']:
        verify_output(folder)
    spec=read(SPEC)
    require(spec['proposed_publication_intervals']==protocol['publication_intervals'],'Cutoffs differ from saved proposal')
    records={}; posts={}; attachments=[]
    oldtexts=defaultdict(set)
    for r in read(OLD):
        for p in r.get('social_media_posts',[]):
            if p.get('platform')=='reddit' and isinstance(p.get('post'),str):
                oldtexts[p['post']].add(r['cve'])
    for r in tqdm(jsonl(SAN/'predictor_candidates.jsonl.gz'),total=44985,desc='Read dates and deduplicated Reddit evidence',unit='CVE'):
        c=r['cve']; records[c]={'published_date':r['published_date'],'pub':date.fromisoformat(r['published_date'])}
        for p in r['social_media_posts']:
            key=fingerprint(p)
            item=posts.setdefault(key,{'cves':set(),'endpoints':set(),'text':p['post'],'date':p['date'],'group':post_group(p['post'])})
            item['cves'].add(c)
            item['endpoints'].update({c}|{m.upper() for m in CVE.findall(p['post'])}|oldtexts.get(p['post'],set()))
            attachments.append((c,key,p['date']))
    require(len(attachments)==len(set(attachments))==74032,'Deduplicated attachment count differs')
    oldroles=read(OLD_ROLES); prior={c:r for r,ids in oldroles.items() for c in ids}
    baseline=read(SAN/'connected_components.json'); exposure={}
    for g in baseline:
        for c in g['current_cves']:
            exposure[c]=set(g['prior_roles'])
    actual=groups_for(records,posts)
    require({frozenset(g['current']) for g in actual}=={frozenset(g['current_cves']) for g in baseline},'Conservative graph not reproduced')
    require({frozenset(g['all']) for g in actual}=={frozenset(g['current_cves']+g['external_or_old_only_cves']) for g in baseline},'External exposure graph differs')
    exact=groups_for(records,posts,exact=True)
    require({frozenset(g['current']) for g in exact}=={frozenset(g) for g in read(SAN/'exact_shared_post_components.json')},'Exact graph not reproduced')
    for r in tqdm(jsonl(SAN/'outcomes_restricted.jsonl.gz'),total=len(records),desc='Read score availability, not score magnitudes',unit='CVE'):
        c=r['cve']; out=r['epss_outcomes']; scores=out['epss_scores']; statuses=out['epss_score_status']
        require(set(scores)==set(statuses) and len(scores)==91,'Unexpected history')
        observed={d for d,v in scores.items() if v is not None}
        require(all((d in observed)==(statuses[d]=='OBSERVED') for d in scores),'Score/status mismatch')
        indices=sorted((date.fromisoformat(d)-records[c]['pub']).days for d in observed)
        require(all(0<=i<=90 for i in indices),'Date outside 91-day contract')
        records[c].update(observed=observed,indices=indices,statuses=statuses,version_changed=out['epss_version_changed'])
    specmod=importlib.util.spec_from_file_location('rd01_pinned_collector',COLLECTOR)
    mod=importlib.util.module_from_spec(specmod); specmod.loader.exec_module(mod)
    release_map=mod.CONFIG['EPSS_VERSION_DATES']
    releases=sorted(date.fromisoformat(d) for d in release_map.values())[1:]
    return records,posts,attachments,exposure,prior,baseline,releases,release_map


def bridge_audit(records,posts,baseline,protocol,out):
    giant=max(baseline,key=lambda g:g['current_count']); members=set(giant['current_cves']); allids=members|set(giant['external_or_old_only_cves'])
    graph=nx.Graph(); graph.add_nodes_from(('c',c) for c in allids)
    selected={k:p for k,p in posts.items() if p['cves']&members}
    rows=[]
    for k,p in sorted(selected.items()):
        require(p['endpoints']<=allids,'Giant has an external edge')
        graph.add_edges_from((('p',k),('c',c)) for c in sorted(p['endpoints']))
        pubs=sorted(records[c]['published_date'] for c in p['cves'])
        years={d[:4] for d in pubs}
        rows.append({'fingerprint':k,'attached_current_cves':len(p['cves']),
            'conservative_cve_degree':len(p['endpoints']),'mentioned_ids':len(set(CVE.findall(p['text']))),
            'post_date':p['date'],'publication_min':pubs[0],'publication_max':pubs[-1],
            'publication_year_count':len(years),'cross_year_link':len(years)>1,
            'group':p['group'],'excerpt':p['text'][:400]})
    require(nx.is_connected(graph),'Giant bipartite graph is not connected')
    articulation={x[1] for x in nx.articulation_points(graph) if x[0]=='p'}
    bridges=[]
    for a,b in nx.bridges(graph):
        p,c=(a,b) if a[0]=='p' else (b,a)
        bridges.append({'fingerprint':p[1],'cve':c[1], 'current_cve':c[1] in records,
                        'meaning':'Removing this incidence edge disconnects the bipartite graph; may isolate only an external/leaf CVE'})
    for r in rows:
        r['articulation_post']=r['fingerprint'] in articulation
    top=sorted(rows,key=lambda r:(-r['attached_current_cves'],r['fingerprint']))[:protocol['hubs_to_detail']]
    cross=sorted([r for r in rows if r['cross_year_link']],key=lambda r:(-r['publication_year_count'],-r['attached_current_cves'],r['fingerprint']))[:protocol['hubs_to_detail']]
    art=sorted([r for r in rows if r['articulation_post']],key=lambda r:(-r['attached_current_cves'],r['fingerprint']))[:protocol['hubs_to_detail']]
    detail=[]
    for k in tqdm(sorted({r['fingerprint'] for r in top+cross+art}),desc='Measure bounded single-post bridge removals',unit='post'):
        view=nx.subgraph_view(graph,filter_node=lambda node,key=k: node!=('p',key))
        sizes=sorted([sum(n[0]=='c' and n[1] in members for n in component) for component in nx.connected_components(view)],reverse=True)
        sizes=[n for n in sizes if n]
        detail.append({'fingerprint':k,'remaining_current_components':len(sizes),'largest_current_component':sizes[0],
                       'current_cves_outside_largest':len(members)-sizes[0], 'component_sizes':sizes})
    csv_file(out/'giant_posts.csv',rows,list(rows[0]))
    csv_file(out/'giant_incidence_bridges.csv',sorted(bridges,key=lambda r:(r['fingerprint'],r['cve'])),['fingerprint','cve','current_cve','meaning'])
    for name,table in [('top_hub_posts',top),('top_cross_year_posts',cross)]:
        csv_file(out/(name+'.csv'),table,list(rows[0]))
    (out/'bounded_bridge_removals.json').write_text(dumps(detail))
    distribution=Counter(r['attached_current_cves'] for r in rows)
    csv_file(out/'giant_post_degree_distribution.csv',[{'attached_cves_per_fingerprint':k,'fingerprints':v} for k,v in sorted(distribution.items())],['attached_cves_per_fingerprint','fingerprints'])
    return {'current_cves':len(members),'post_launch_cves':sum(records[c]['published_date']>=protocol['publication_start'] for c in members),
            'post_fingerprints':len(rows),'external_or_old_only_ids':len(allids-members),
            'degree_distribution':{str(k):v for k,v in sorted(distribution.items())},
            'cross_year_fingerprints':sum(r['cross_year_link'] for r in rows),
            'articulation_post_nodes':len(articulation),'incidence_bridge_edges':len(bridges),
            'top_hubs':top,'top_cross_year':cross,'bounded_removals':detail,
            'interpretation':'A cross-year link is not necessarily a mathematical bridge. Fingerprints are not authenticated Reddit IDs.'},members


def qualify(groups,records,exposure,mode,protocol):
    allocations={}; exclusions={}
    for g in groups:
        reason,role=component_reason(g['current'],records,exposure,mode,protocol)
        for c in g['current']:
            if reason:
                exclusions[c]=reason
            else:
                allocations[c]=role
    require(len(allocations)+len(exclusions)==len(records),'Incomplete qualification')
    for g in groups:
        require(len({allocations[c] for c in g['current'] if c in allocations})<=1,'Component crosses roles')
    require(all(not exposure[c] for c,r in allocations.items() if r=='test'),'Exposed CVE in candidate test')
    require(all('test' not in exposure[c] for c in allocations),'Prior test exposure admitted')
    task={name:{r:set() for r in ROLES} for name in ('publication_candidate','RD02','RD04')}
    exclusions_by_task=defaultdict(Counter); cells=Counter()
    for c,role in sorted(allocations.items()):
        r=records[c]; task['publication_candidate'][role].add(c)
        end=min(date.fromisoformat(protocol['publication_intervals'][role][1]),date.fromisoformat(protocol['cutoff']))
        maximum=(end-r['pub']).days
        if 60 not in r['indices']:
            exclusions_by_task['RD02'][role+':missing_day60']+=1
        elif maximum<60:
            exclusions_by_task['RD02'][role+':target_after_role_end']+=1
        else:
            task['RD02'][role].add(c)
        eligible=daily_cells(r['indices'],maximum)
        if eligible:
            task['RD04'][role].add(c); cells[role]+=eligible
        else:
            exclusions_by_task['RD04'][role+':no_past_future_pair_within_role']+=1
    reasons=Counter(exclusions[c] for c in exclusions if records[c]['published_date']>=protocol['publication_start'])
    require(sum(len(s) for s in task['publication_candidate'].values())+sum(reasons.values())==35113,'Post-launch flow does not reconcile')
    return task, {'post_launch_exclusion_first_reason':dict(sorted(reasons.items())),
                 'task_exclusions':{k:dict(sorted(v.items())) for k,v in sorted(exclusions_by_task.items())},
                 'daily_cells':dict(cells)}


def split_audit(records,posts,attachments,exposure,giant,protocol,out):
    summaries=[]; rows=[]; flows={}; baselinegroups=None
    for policy,threshold in tqdm(protocol['post_policies'].items(),desc='Compare fixed text and exposure policies',unit='policy'):
        groups=groups_for(records,posts,threshold); exact=groups_for(records,posts,threshold,exact=True)
        excluded={k for k,p in posts.items() if not keep_post(len(p['cves']),threshold)}
        sizes=sorted([len(set(g['current'])&giant) for g in groups if set(g['current'])&giant],reverse=True)
        kept=Counter(c for c,k,_ in attachments if k not in excluded)
        summaries.append({'post_policy':policy,'quarantined_fingerprints':len(excluded),
            'quarantined_attachments':sum(len(posts[k]['cves']) for k in excluded),
            'retained_attachments':sum(kept.values()),'current_cves_without_remaining_M':sum(kept[c]==0 for c in records),
            'conservative_components':len(groups),'largest_conservative_component':max(len(g['current']) for g in groups),
            'exact_components':len(exact),'largest_exact_component':max(len(g['current']) for g in exact),
            'original_giant_parts':len(sizes),'original_giant_largest_remaining':sizes[0],
            'original_giant_singletons':sum(s==1 for s in sizes)})
        for mode in protocol['exposure_policies']:
            task,flow=qualify(groups,records,exposure,mode,protocol); flows[policy+'/'+mode]=flow
            lookup={(taskname,role):Counter() for taskname in task for role in ROLES}
            with_m={(taskname,role):set() for taskname in task for role in ROLES}
            for c,k,_ in attachments:
                if k in excluded: continue
                for name,roles in task.items():
                    for role,ids in roles.items():
                        if c in ids:
                            lookup[name,role][posts[k]['group']]+=1; with_m[name,role].add(c)
            for name,roles in task.items():
                for role,ids in roles.items():
                    pc=lookup[name,role]
                    rows.append({'post_policy':policy,'exposure_policy':mode,'task':name,'role':role,
                        'cves':len(ids),'cves_with_retained_M':len(with_m[name,role]),
                        'retained_attachments':sum(pc.values()),'bare_id_attachments':pc['bare_id_only'],
                        'single_cve_descriptive_attachments':pc['single_cve_descriptive'],
                        'multi_cve_attachments':pc['multi_cve_text'],
                        'eligible_anchor_target_cells':flow['daily_cells'].get(role,0) if name=='RD04' else ''})
    csv_file(out/'component_policy_comparison.csv',summaries,list(summaries[0]))
    csv_file(out/'candidate_split_counts.csv',rows,list(rows[0]))
    (out/'candidate_exclusion_flow.json').write_text(dumps(flows))
    return summaries,rows,flows


def window_audit(records,posts,attachments,releases,protocol,out):
    buckets={}; reasons=Counter(); totals=Counter()
    def event_rows():
        for c,k,dt in tqdm(sorted(attachments),desc='Recount deduplicated before/after windows',unit='attachment'):
            r=records[c]; p=posts[k]; posted=date.fromisoformat(dt)
            for width in protocol['windows_days_each_side']:
                result=event_window(r['pub'],posted,r['observed'],r['statuses'],width,releases)
                totals[width]+=1; reasons[width,result['reason']]+=1
                row={'cve':c,'fingerprint':k,'post_date':dt,'days_each_side':width,'text_group':p['group'],
                     'attached_current_cves':len(p['cves']),'complete_daily_window':result['complete'],
                     'exclusion_reason':result['reason'],'missing_reasons':json.dumps(result['missing_reasons'],sort_keys=True),
                     'cve_full_91day_version_changed':r['version_changed'],
                     'event_window_crosses_release':result['crosses_release'],
                     'quarantine_gt10':len(p['cves'])>10,'quarantine_gt25':len(p['cves'])>25}
                for policy,threshold in protocol['post_policies'].items():
                    if not keep_post(len(p['cves']),threshold): continue
                    for cohort in ['all_cves']+(['post_launch'] if r['published_date']>=protocol['publication_start'] else []):
                        for category in ('all_text',p['group']):
                            key=(policy,cohort,width,category)
                            b=buckets.setdefault(key,{'considered':0,'eligible':0,'cves':set(),'release_crossing':0,
                                'full_window_version_flag':0,'cves_release_crossing':set(),'reasons':Counter()})
                            b['considered']+=1; b['reasons'][result['reason']]+=1
                            if result['complete']:
                                b['eligible']+=1; b['cves'].add(c)
                                b['release_crossing']+=int(result['crosses_release'])
                                b['full_window_version_flag']+=int(r['version_changed'])
                                if result['crosses_release']: b['cves_release_crossing'].add(c)
                yield row
    fields=['cve','fingerprint','post_date','days_each_side','text_group','attached_current_cves','complete_daily_window',
            'exclusion_reason','missing_reasons','cve_full_91day_version_changed','event_window_crosses_release','quarantine_gt10','quarantine_gt25']
    csv_file(out/'post_window_coverage.csv.gz',event_rows(),fields)
    rows=[]
    for (policy,cohort,width,group),b in sorted(buckets.items()):
        rows.append({'post_policy':policy,'cohort':cohort,'days_each_side':width,'text_group':group,
            'considered_attachments':b['considered'],'complete_window_attachments':b['eligible'],
            'cves_with_complete_window':len(b['cves']), 'complete_windows_crossing_release':b['release_crossing'],
            'cves_with_crossing_window':len(b['cves_release_crossing']),
            'complete_windows_in_CVEs_with_global_version_flag':b['full_window_version_flag'],
            'outside_stored_window':b['reasons']['outside_stored_window'],'in_range_missing_scores':b['reasons']['missing_daily_score']})
    require(all(totals[w]==74032 for w in protocol['windows_days_each_side']),'Event denominators differ')
    for row in rows:
        require(row['considered_attachments']==row['complete_window_attachments']+row['outside_stored_window']+row['in_range_missing_scores'],'Window totals differ')
    csv_file(out/'post_window_summary.csv',rows,list(rows[0]))
    return rows


def text_report(f):
    g=f['giant']; c=f['cohort']; lines=['RD01: SHARED-EVIDENCE AND CHRONOLOGICAL QUALIFICATION','',
        'Diagnostic only. No split IDs frozen, predictors exported, model training or test scoring.',
        'Duplicate URL entries were removed within each CVE; shared references across CVEs remain preserved.',
        'This audit does not remove URLs. Post quarantine is a hypothetical source-view policy, not a raw-data edit.','',
        '1. Eligible score coverage is not a final split',dumps(c),
        'At least one observed score is not sufficient for daily forecasting. At least one earlier observation and later observed target are required.',
        'Day-60 candidates must also have publication+60 within their role end. Daily windows are bounded on target dates, not just publication.',
        'Only score presence/status and version metadata are used. Score magnitudes, class outcomes and test predictions do not select rules.','',
        '2. The giant component',
        f"Current CVEs: {g['current_cves']:,}; post-launch: {g['post_launch_cves']:,}; attached fingerprints: {g['post_fingerprints']:,}.",
        f"Cross-year fingerprints: {g['cross_year_fingerprints']:,}; articulation post nodes: {g['articulation_post_nodes']:,}; incidence bridges: {g['incidence_bridge_edges']:,}.",
        'A fingerprint combines exact text, date and platform; it is not an authenticated Reddit post ID.',
        'A hub attaches to many CVEs. A cross-year link spans publication years. A mathematical bridge disconnects the bipartite graph when removed; it may merely isolate a leaf CVE.',
        'Co-mentions and old exact text reuse are additional exposure links. None of these means two CVEs are the same vulnerability.',
        'Top hubs (fingerprint, attached CVEs, publication span, actual saved excerpt):']
    for r in g['top_hubs'][:10]:
        lines.append(f"{r['fingerprint']} | {r['attached_current_cves']} | {r['publication_min']} to {r['publication_max']} | {r['excerpt']}")
    lines+=['','3. Fixed bulk-post counterfactuals',
        'Strict >10 and >25 attached-CVE thresholds were declared before execution. They identify broadly shared text, not proven irrelevant posts.',
        'All attachments of each flagged fingerprint are quarantined from candidate M in that scenario. D and CVE identities remain. Bare-ID posts are tagged, not automatically removed.',
        'Original untrimmed prior exposure remains remembered: removing a post cannot make already inspected evidence unseen.',
        'Rows show structural effects before any exposure or chronology exclusion:',dumps(f['component_policies']),
        '4. Candidate chronological counts',
        'Fixed proposed publication roles: train 2021-04-14 through 2023-12-31; validation 2024; test 2025 through 2026-11-03.',
        'never_exposed uses only original components without any prior role. retain_matching_prior_roles also permits compatible prior train/validation exposure.',
        'The existing RD01 proposal is stricter than merely barring old-test evidence from new test: all original prior-test-connected CVEs are excluded from ALL candidate roles.',
        'Every prior-role-connected CVE is barred from the new test even after bulk-post quarantine. Matching-role conflicts quarantine the entire retained component.',
        'Cross-period components are quarantined before target filtering; pre-launch or outside-role members count as a different period.',
        'These stronger rules are reported explicitly, not silently relaxed to obtain larger cohorts.',
        'No IDs are published as a frozen manifest. Candidate counts are deterministic qualifications under each stated policy.',
        'policy | exposure | task | train | validation | test']
    rows=f['candidate_splits']
    for policy in f['protocol']['post_policies']:
        for mode in f['protocol']['exposure_policies']:
            for task in ('publication_candidate','RD02','RD04'):
                counts={r['role']:r['cves'] for r in rows if (r['post_policy'],r['exposure_policy'],r['task'])==(policy,mode,task)}
                lines.append(' | '.join([policy,mode,task,*[str(counts[r]) for r in ROLES]]))
    lines+=['','candidate_split_counts.csv also gives retained attachments, bare-ID, single-CVE and multi-CVE categories, CVEs with remaining M, and RD04 eligible cells.',
        'candidate_exclusion_flow.json accounts for all 35,113 post-launch CVEs with mutually exclusive first exclusion reasons. Different risks can overlap; the first reason is not their full overlap count.',
        'An unconnected current CVE can still carry prior exposure through external/old IDs. Zero exposure counts and disconnected components are not synonyms.','',
        '5. Deduplicated descriptive pre/post coverage',
        'Each +/-h window requires all 2h+1 daily scores, including the post date. Outside the stored publication-to-day90 range is separate from missing scores inside it.',
        'The three text groups are disjoint: bare-ID-only, multi-CVE mentions, and exactly one CVE mention with extra wording. Descriptive means syntax here, not verified relevance.',
        'The full-CVE epss_version_changed flag is not proof that this local event window crosses a release. Both flags are reported separately; local crossing uses start < release <= end.',
        'Actual release dates used: '+dumps(f['release_schedule']),
        'Retain-all, all-CVE rows: days each side | text group | complete attachments | CVEs | local release-crossing windows']
    for r in f['window_counts']:
        if r['post_policy']=='retain_all' and r['cohort']=='all_cves':
            lines.append(' | '.join(str(r[k]) for k in ('days_each_side','text_group','complete_window_attachments','cves_with_complete_window','complete_windows_crossing_release')))
    lines+=['','A CVE may have several posts and occur in multiple text categories. Attachment counts add across categories; unique-CVE counts need not.',
        'These are coverage tables, not EPSS changes, causal estimates or independent sample counts. No intraday ordering or historical wording availability is established.',
        'The event table includes all deduplicated attachments as requested, before split qualification. It exports availability flags, not numerical EPSS outcomes. Prospective study counts require joining the selected policy later.',
        '6. Decision before freezing',
        'Review the component counterfactual and exclusion counts, then explicitly select a source/exposure policy. No threshold is selected automatically by this audit.',
        'Retain the accepted missing-score policy: nulls/statuses unchanged, partial histories allowed, missing targets never imputed.',
        'RD03 remains gated on authenticated catalogue receipts. Strict temporal text fusion remains blocked on content lineage.',
        'Repeated/near-duplicate descriptions or posts and explicit target-answer wording still require the source-policy audit already required by RD01. This diagnostic alone does not establish every leakage check passed.',
        'The same CVE is never split across roles. A later-publication saved-snapshot holdout is not a prospectively collected dataset.',
        'Files: giant_posts.csv; top_hub_posts.csv; top_cross_year_posts.csv; giant_incidence_bridges.csv; giant_post_degree_distribution.csv; bounded_bridge_removals.json; component_policy_comparison.csv; candidate_split_counts.csv; candidate_exclusion_flow.json; post_window_summary.csv; post_window_coverage.csv.gz; findings.json; artifact_lock.json.',
        'Reproduce on CPU in CodeBERTFusion: python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/audit_rd01_splits.py --verify-only']
    return '\n'.join(lines)+'\n'


def run(output,verify=False):
    protocol=read(CONFIG)
    records,posts,attachments,exposure,prior,baseline,releases,schedule=load(protocol)
    inputs={str(p):sha(p) for p in [CONFIG,SPEC,COLLECTOR,OLD,OLD_ROLES,Path(__file__),
            Path(__file__).with_name('reddit_gates.py'),Path(__file__).with_name('audit_reddit.py'),SAN/'artifact_lock.json',
            RESULTS/'stage0/artifact_lock.json',RESULTS/'stage1_epss/artifact_lock.json',RESULTS/'stage2_4_specifications_v2/artifact_lock.json']}
    era={c:r for c,r in records.items() if r['published_date']>=protocol['publication_start']}
    cohort={'all_cves':len(records),'post_launch':len(era),'day60_observed_before_role_limits':sum(60 in r['indices'] for r in era.values()),
            'at_least_one_score_before_role_limits':sum(bool(r['indices']) for r in era.values()),
            'at_least_one_past_future_pair_before_role_limits':sum(bool(daily_cells(r['indices'])) for r in era.values()),
            'prior_test_connected_all_years':sum('test' in e for e in exposure.values()),
            'prior_test_connected_new_ids':sum('test' in e and c not in prior for c,e in exposure.items()),
            'post_launch_never_exposed':sum(not exposure[c] for c in era)}
    require((cohort['post_launch'],cohort['day60_observed_before_role_limits'],cohort['at_least_one_score_before_role_limits'],cohort['prior_test_connected_all_years'],cohort['prior_test_connected_new_ids'],cohort['post_launch_never_exposed'])==(35113,34921,34995,4987,4085,28872),'Baseline cohort counts differ')
    with tempfile.TemporaryDirectory(prefix='rd01-diagnostic-',dir='/tmp') as temp:
        stage=Path(temp)
        (stage/'protocol.json').write_text(dumps(protocol))
        giant,members=bridge_audit(records,posts,baseline,protocol,stage)
        policies,splits,flows=split_audit(records,posts,attachments,exposure,members,protocol,stage)
        windows=window_audit(records,posts,attachments,releases,protocol,stage)
        result={'schema':'rd01_diagnostic_v1','protocol':protocol,'cohort':cohort,'giant':giant,
                'component_policies':policies,'candidate_splits':splits,'exclusion_flows':flows,
                'window_counts':windows,'release_schedule':schedule,'split_frozen':False,'training':False,
                'numerical_epss_values_exported':False,'test_scoring':False,'source_changes':False,
                'software':{'networkx':nx.__version__}}
        (stage/'findings.json').write_text(dumps(result)); (stage/'REPORT.txt').write_text(text_report(result))
        for name,h in inputs.items():
            require(sha(Path(name))==h,'Input changed during audit: '+name)
        lock_output(stage,inputs)
        if verify:
            verify_output(output)
            require(read(stage/'artifact_lock.json')==read(output/'artifact_lock.json'),'Fresh RD01 replay differs')
        else:
            require(not output.exists(),'Output exists; use --verify-only')
            shutil.copytree(stage,output)
    print('RD01 diagnostic '+('reproduced' if verify else 'saved')+' at '+str(output)+'. No split freeze, training or test scoring.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=OUT)
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args(); run(a.output_dir,a.verify_only)
