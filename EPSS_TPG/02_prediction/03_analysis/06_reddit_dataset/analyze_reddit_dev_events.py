"""Descriptive EPSS event study on frozen RD04 development CVEs only; no training."""
import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
import shutil
import tempfile

import numpy as np
from tqdm import tqdm

from audit_reddit import read, sha, dumps, require, fingerprint
from reddit_gates import lock_output, verify_output
from audit_rd01_splits import SAN, OUT as DIAG, jsonl, csv_file, event_window, post_group
from lock_rd01_sources import OUT as LOCK, CONFIG, disclosure

OUT=DIAG.parent/'09_reddit_dev_events_v1'
METRICS=('change_before','change_after','after_minus_before','absolute_change_before',
         'absolute_change_after','positive_endpoint_before','positive_endpoint_after',
         'fraction_positive_daily_steps_before','fraction_positive_daily_steps_after',
         'slope_before','slope_after','slope_difference','minimum_after','maximum_after','days_to_maximum_after')


def event_metrics(values,width):
    require(len(values)==2*width+1,'Wrong event length')
    a=np.asarray(values,dtype=float)
    require(np.isfinite(a).all() and ((a>=0)&(a<=1)).all(),'Invalid EPSS score')
    before=float(a[width]-a[0]); after=float(a[-1]-a[width])
    # Slopes exclude the event date. One point at h=1 cannot define a slope.
    def slope(v):
        if len(v)<2: return None
        x=np.arange(len(v),dtype=float); x-=x.mean()
        return float(np.dot(x,v-v.mean())/np.dot(x,x))
    pre=slope(a[:width]); post=slope(a[width+1:])
    return {'change_before':before,'change_after':after,'after_minus_before':after-before,
        'absolute_change_before':abs(before),'absolute_change_after':abs(after),
        'positive_endpoint_before':int(before>0),'positive_endpoint_after':int(after>0),
        'fraction_positive_daily_steps_before':float(np.mean(np.diff(a[:width+1])>0)),
        'fraction_positive_daily_steps_after':float(np.mean(np.diff(a[width:])>0)),
        'slope_before':pre,'slope_after':post,'slope_difference':None if pre is None else post-pre,
        'minimum_after':float(np.min(a[width+1:])),'maximum_after':float(np.max(a[width+1:])),
        'days_to_maximum_after':int(np.argmax(a[width+1:]))+1,
        'direction':'both_rising' if before>0 and after>0 else ('before_only' if before>0 else ('after_only' if after>0 else 'neither_rising'))}


def statistic_rows(events,mode):
    results=[]; byc=defaultdict(list)
    for e in events: byc[e['cve']].append(e)
    for metric in METRICS:
        if mode=='equal_CVE':
            values=[float(np.mean(v)) for rows in byc.values() if (v:=[r[metric] for r in rows if r[metric] is not None])]
        else:
            values=[e[metric] for e in events if e[metric] is not None]
        results.append({'metric':metric,'mean':float(np.mean(values)) if values else None,
                        'median':float(np.median(values)) if values else None,'units':len(values)})
    return results


def summarize(events):
    first={}
    for e in sorted(events,key=lambda e:(e['post_date'],e['fingerprint'])):
        first.setdefault((e['role'],e['cve'],e['days_each_side']),e['event_key'])
    buckets=defaultdict(list)
    for e in events:
        for typ in ('all_text',e['text_group']):
            for version in ('all_versions','crossing_release' if e['local_release_crossing'] else 'single_version'):
                buckets[e['role'],e['days_each_side'],typ,version].append(e)
    summary=[]; counts=[]; curves=[]; directions=[]
    for (role,width,typ,version),rows in sorted(buckets.items()):
        common={'role':role,'days_each_side':width,'text_group':typ,'version_group':version}
        counts.append({**common,'events':len(rows),'cves':len({r['cve'] for r in rows}),
            'events_with_other_posts_in_window':sum(r['other_attachments_in_window']>0 for r in rows),
            'events_with_epss_topic':sum(r['epss_topic_screen'] for r in rows)})
        for mode in ('equal_CVE','event_weighted','first_event_per_CVE'):
            selected=rows if mode!='first_event_per_CVE' else [r for r in rows if first[r['role'],r['cve'],width]==r['event_key']]
            for stat in statistic_rows(selected,'equal_CVE' if mode=='equal_CVE' else 'events'):
                summary.append({**common,'weighting':mode,'events':len(selected),'cves':len({r['cve'] for r in selected}),**stat})
        grouped=defaultdict(list)
        for e in rows: grouped[e['cve']].append(e)
        for offset in range(-width,width+1):
            per_cve=[np.mean([r['trajectory'][offset+width] for r in rr]) for rr in grouped.values()]
            curves.append({**common,'relative_day':offset,'cves':len(grouped),
                'mean_equal_CVE':float(np.mean(per_cve)),'median_CVE_average':float(np.median(per_cve))})
        for pattern in ('both_rising','before_only','after_only','neither_rising'):
            per_cve=[sum(r['direction']==pattern for r in rr)/len(rr) for rr in grouped.values()]
            directions.append({**common,'direction':pattern,'events':sum(r['direction']==pattern for r in rows),
                'equal_CVE_fraction':float(np.mean(per_cve)),'cves':len(grouped)})
    return summary,counts,curves,directions


def execute(output,verify=False):
    verify_output(LOCK); verify_output(DIAG)
    manifest=read(LOCK/'rd01_frozen_manifest.json'); protocol=read(CONFIG)
    task=manifest['tasks']['RD04']; ids=task['ids']
    selected={c:role for role in ('train','validation') for c in ids[role]}
    require(not set(selected)&set(ids['test']),'Test ID entered development event study')
    release_dates=sorted(date.fromisoformat(d) for d in manifest['release_schedule'].values())[1:]
    posts={}; records={}
    for r in tqdm(jsonl(SAN/'predictor_candidates.jsonl.gz'),total=44985,desc='Read development post dates only',unit='CVE'):
        c=r['cve']
        if c not in selected: continue
        records[c]=date.fromisoformat(r['published_date'])
        posts[c]=[{'fingerprint':fingerprint(p),'post_date':p['date'],'group':post_group(p['post']),
                   'epss_topic_screen':disclosure(p['post'])['quarantine']} for p in r['social_media_posts']]
    inputs={str(p):sha(p) for p in [LOCK/'artifact_lock.json',LOCK/'rd01_frozen_manifest.json',SAN/'artifact_lock.json',
            DIAG/'artifact_lock.json',CONFIG,Path(__file__),Path(__file__).with_name('lock_rd01_sources.py'),
            Path(__file__).with_name('audit_rd01_splits.py')]}
    events=[]; coverage=defaultdict(Counter)
    for r in tqdm(jsonl(SAN/'outcomes_restricted.jsonl.gz'),total=44985,desc='Compute development events; skip held-out outcomes',unit='CVE'):
        c=r['cve']
        if c not in selected: continue
        role=selected[c]; scores=r['epss_outcomes']['epss_scores']; statuses=r['epss_outcomes']['epss_score_status']
        observed={d for d,v in scores.items() if v is not None}
        role_end=date.fromisoformat(manifest['publication_intervals'][role][1])
        for p in posts[c]:
            posted=date.fromisoformat(p['post_date'])
            for width in protocol['event_windows']:
                bucket=coverage[role,width]; bucket['considered_attachments']+=1
                check=event_window(records[c],posted,observed,statuses,width,release_dates)
                if not check['complete']:
                    bucket[check['reason']]+=1; continue
                bucket['complete_before_role_boundary']+=1
                if posted+timedelta(days=width)>role_end:
                    bucket['after_role_end']+=1; continue
                bucket['analyzed_events']+=1
                values=[scores[(posted+timedelta(days=d)).isoformat()] for d in range(-width,width+1)]
                other=sum(abs((date.fromisoformat(q['post_date'])-posted).days)<=width for q in posts[c])-1
                event={'cve':c,'role':role,'fingerprint':p['fingerprint'],'post_date':p['post_date'],'days_each_side':width,
                    'event_key':c+'|'+p['fingerprint']+'|'+str(width),'text_group':p['group'],
                    'local_release_crossing':check['crosses_release'],
                    'full_history_version_changed':r['epss_outcomes']['epss_version_changed'],
                    'other_attachments_in_window':other,'epss_topic_screen':p['epss_topic_screen'],
                    'trajectory':values,**event_metrics(values,width)}
                events.append(event)
    require(events and all(e['cve'] in selected and e['cve'] not in ids['test'] for e in events),'Event cohort violation')
    events.sort(key=lambda e:(e['role'],e['cve'],e['post_date'],e['fingerprint'],e['days_each_side']))
    summary,counts,curves,directions=summarize(events)
    coverage_rows=[]
    for (role,width),v in sorted(coverage.items()):
        require(v['considered_attachments']==v['outside_stored_window']+v['missing_daily_score']+v['after_role_end']+v['analyzed_events'],'Coverage does not reconcile')
        coverage_rows.append({'role':role,'days_each_side':width,**{k:v[k] for k in ('considered_attachments','outside_stored_window','missing_daily_score','complete_before_role_boundary','after_role_end','analyzed_events')}})
    report={'schema':'reddit_development_events_v1','manifest_sha256':sha(LOCK/'rd01_frozen_manifest.json'),
        'task':'Descriptive EPSS before/after, never causal','cohort':'Frozen RD04 train/validation only',
        'training':False,'test_analyzed':False,'test_cves_analyzed':0,
        'event_window_records':len(events),'unique_cves_any_window':len({e['cve'] for e in events}),
        'coverage':coverage_rows,'strata_counts':counts,'direction_counts':directions,
        'definition':protocol['event_interpretation'],'weighting':protocol['event_weighting'],
        'positive_jump':protocol['event_positive_jump'],
        'first_event_rule':'First eligible date/fingerprint per CVE and width BEFORE type/version stratification',
        'limits':['All events are dated saved attachments, not authenticated post IDs.',
            'Event selection uses observed complete windows; missingness can affect representativeness.',
            'Daily EPSS release and post timing within a day are unknown.',
            'No test or excluded-CVE changes analyzed; previous all-corpus coverage is not the new event denominator.',
            'EPSS-topic posts remain descriptive events and are tagged; they are excluded from RD02 encoding by its separate source policy.',
            'Overlapping discussions and shared posts are not independent interventions; no causal or significance claims.']}
    with tempfile.TemporaryDirectory(prefix='reddit-dev-events-',dir='/tmp') as td:
        stage=Path(td)
        for name,rows in [('event_summary',summary),('event_counts',counts),('event_trajectories',curves),('direction_patterns',directions),('coverage_flow',coverage_rows)]:
            csv_file(stage/(name+'.csv'),rows,list(rows[0]))
        export=[]
        for e in events:
            e=dict(e); e['trajectory']=dumps(e['trajectory']).strip().replace('\n',' '); export.append(e)
        csv_file(stage/'development_event_details_ANALYSIS_ONLY.csv.gz',export,list(export[0]))
        (stage/'findings.json').write_text(dumps(report))
        lines=['DESCRIPTIVE REDDIT / EPSS DEVELOPMENT EVENT STUDY','',
            'Only frozen RD04 training and validation records are analyzed. No training, test scores or causal estimates.',
            'All-corpus complete-window counts are reference coverage, not the denominator of this protected development-only study.',
            'Before change = EPSS on post date minus EPSS h days before. After change = EPSS h days after minus EPSS on post date.',
            'A positive jump means a strictly positive consecutive daily difference. The two fractions count h before and h after differences, respectively.',
            'Primary statistics average event quantities within CVE, then give every CVE equal weight. Medians refer to these per-CVE averages.',
            'Event-weighted means/medians and globally first-eligible-event sensitivity are secondary; no threshold is fitted.',
            'Slopes omit the post date and are undefined for h=1. Maximum/minimum use days 1..h; earliest maximum wins ties.',
            'Direction groups use endpoint changes: before only rising, after only rising, both rising, or neither. Stable/decreasing are not called negative exploitation evidence.',
            'Local version crossing uses start < release <= end. The full-91-day flag is separate.',
            '', 'Coverage and exclusions',dumps(coverage_rows),
            'Primary all-text results: role | window | version | CVEs | mean before | mean after | median before | median after']
        for row in counts:
            if row['text_group']!='all_text': continue
            def stat(metric,field):
                return next(s[field] for s in summary if s['role']==row['role'] and s['days_each_side']==row['days_each_side'] and s['version_group']==row['version_group'] and s['text_group']=='all_text' and s['weighting']=='equal_CVE' and s['metric']==metric)
            lines.append(' | '.join(map(str,[row['role'],row['days_each_side'],row['version_group'],row['cves'],stat('change_before','mean'),stat('change_after','mean'),stat('change_before','median'),stat('change_after','median')])))
        lines+=['','Complete type/version/weighting tables: event_summary.csv. Counts: event_counts.csv. Mean/median trajectories: event_trajectories.csv. Per-event output is analysis-only, not model input.',*report['limits']]
        (stage/'REPORT.txt').write_text('\n'.join(lines)+'\n')
        for p,h in inputs.items(): require(sha(Path(p))==h,'Input changed during event study')
        lock_output(stage,inputs)
        if verify:
            verify_output(output); require(read(stage/'artifact_lock.json')==read(output/'artifact_lock.json'),'Event replay differs')
        else:
            require(not output.exists(),'Use --verify-only'); shutil.copytree(stage,output)
    print('Development event study '+('reproduced' if verify else 'saved')+' at '+str(output)+'. No training or test analysis.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--verify-only',action='store_true')
    p.add_argument('--output-dir',type=Path,default=OUT); a=p.parse_args(); execute(a.output_dir,a.verify_only)
