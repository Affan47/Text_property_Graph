"""Offline review and single-gap support; no fits, network or official holdouts."""
from collections import Counter
import csv
from pathlib import Path

from tqdm import tqdm

from . import early_warning_support_v1 as prior
from .day7_availability_v1 import collector, plan as history

a = prior.a
ROOT = prior.old.BASE/'23_day7_followup_v1'
CLI = prior.CLI.with_name('28_review_day7_followup.py')
PASS = prior.old.BASE/'21_day7_tier_review_v1/01_reddit/assessment_pass1_20261116_v1'
COHORTS = ('reddit_v1', 'reddit_v2')
ROLES = ('optimization', 'assessment', 'immature_before_assessment', 'purged_to_assessment', 'future')
POLICY = {
    'schema': 'day7_fixed_epoch_support_v1', 'target_day': 90, 'anchor_day': 7,
    'dates': 'Inherit every original date-only 40/60/80 assessment block, unchanged',
    'maturity': 'Publication plus 90 days strictly before first assessment Day7 anchor',
    'groups': 'Inherited full original-training components; purge optimization groups touching assessment',
    'selection': 'None; no checkpoint or epoch selected on these assessment outcomes',
    'epochs': 'NOT_SELECTED_SUPPORT_AUDIT_ONLY',
    'scope': 'Exploratory design review after earlier support inspection; not independent confirmation',
    'weak_folds': 'Retain all; two-class support is not power or permission to discard a fold',
    'training': False, 'network': False, 'official_holdout_access': False,
}


def code_pins():
    return {str(p): a.sha(p) for p in (Path(__file__), CLI, CLI.with_name('test_day7_followup.py'),
                                      Path(prior.__file__), Path(a.__file__))}


def csv_rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def fixed_roles(rows, groups, start, end):
    result = {r: [] for r in ROLES}
    for c, row in sorted(rows.items()):
        if row['anchor'] >= end:
            result['future'].append(c)
        elif row['anchor'] >= start:
            result['assessment'].append(c)
        elif prior.maturity(row, 90) >= start:
            result['immature_before_assessment'].append(c)
        else:
            result['optimization'].append(c)
    blocked = {groups[c] for c in result['assessment']}
    result['purged_to_assessment'] = [c for c in result['optimization'] if groups[c] in blocked]
    result['optimization'] = [c for c in result['optimization'] if groups[c] not in blocked]
    ids = [c for r in ROLES for c in result[r]]
    a.require(len(ids) == len(set(ids)) and set(ids) == set(rows), 'Role accounting changed')
    a.require(not {groups[c] for c in result['optimization']} & blocked, 'Group crosses roles')
    a.require(all(prior.maturity(rows[c],90) < start for c in result['optimization']), 'Immature label')
    return {'assessment_start': start, 'assessment_end': end, **result}


def parent(cohort):
    a.require(cohort in COHORTS, 'Unsupported cohort')
    _, _, _, _, _ = prior.source_context(cohort)
    directory = prior.ROOT/cohort/'plan'; a.check_lock(directory)
    p = a.read(directory/'plan.json'); a.check_pins(p['pins'])
    pins = {**code_pins(), str(directory/'plan.json'):a.sha(directory/'plan.json'),
            str(directory/'artifact_lock.json'):a.sha(directory/'artifact_lock.json')}
    return p, pins


def make_plan(cohort, verify=False):
    p, pins = parent(cohort)
    folds = []
    for f in p['folds']:
        if f['target_day'] != 90:continue
        new = dict(fold=f['fold'], **fixed_roles(p['rows'],p['groups'],f['assessment_start'],f['assessment_end']))
        a.require(new['assessment'] == f['assessment'], 'Assessment IDs changed')
        expected = sorted(f['optimization']+f['selection']+f['immature_before_selection']+f['purged_to_selection'])
        a.require(new['optimization'] == expected, 'Unexpected change beyond removing selection')
        folds.append(new)
    result = {'policy':POLICY, 'cohort':cohort, 'rows':p['rows'], 'groups':p['groups'],
              'folds':folds, 'pins':pins, 'score_values_read':False}
    a.publish(ROOT/cohort/'plan', {'plan.json':result, 'REPORT.txt':
        'Single-gap support plan. Existing assessment dates/groups unchanged. No scores read for allocation.\n'
        'No epoch count or model chosen. Empty/one-class roles stay visible. No training or downloads.\n'}, pins, verify)
    print(cohort+': fixed-epoch support plan '+('reproduced.' if verify else 'saved.')+' No training.')


def load_labels(path):
    result = {name:{} for name in prior.CONFIG['label_policies']}
    for r in csv_rows(path):
        a.require(r['label'] in ('','0','1'), 'Unexpected saved crossing label')
        for field in ('assumed_negative','observed_high_score_before_day7'):
            a.require(r[field] in ('','True','False'), 'Invalid saved flag')
            r[field] = r[field] == 'True'
        r['label'] = None if r['label']=='' else int(r['label'])
        a.require(r['cve'] not in result[r['policy']], 'Duplicate label row')
        result[r['policy']][r['cve']] = r
    return result


def audit(cohort, verify=False):
    parent_plan, pins = parent(cohort)
    directory = ROOT/cohort/'plan'; a.check_lock(directory)
    p = a.read(directory/'plan.json')
    a.require(p['policy']==POLICY and p['pins']==pins, 'Frozen plan/code changed')
    source = prior.ROOT/cohort/'audit'; a.check_lock(source)
    for path in (directory/'plan.json',directory/'artifact_lock.json',source/'label_audit.csv',source/'artifact_lock.json'):
        pins[str(path)] = a.sha(path)
    labels = load_labels(source/'label_audit.csv')
    table, summaries = [], []
    for policy, lab in labels.items():
        a.require(set(lab)==set(p['rows']), 'Label population changed')
        for f in tqdm(p['folds'], desc=cohort+': audit fixed-epoch roles ('+policy+')',unit='fold'):
            expected = fixed_roles(p['rows'],p['groups'],f['assessment_start'],f['assessment_end'])
            a.require(all(f[k]==v for k,v in expected.items()), 'Plan does not reproduce')
            for role in ROLES:
                table.append({'policy':policy,'fold':f['fold'],'role':role,
                              **prior.support(f[role],lab,p['rows'],p['groups'])})
        usable = [f for f in p['folds'] if {lab[c]['label'] for c in f['optimization'] if lab[c]['label'] is not None}=={0,1}]
        for name, folds in [('all_fixed_blocks',p['folds']),('blocks_with_two_training_classes',usable)]:
            ids = [c for f in folds for c in f['assessment']]
            a.require(len(ids)==len(set(ids)), 'Repeated assessment record')
            summaries.append({'policy':policy,'scope':name,'folds_counted':[f['fold'] for f in folds],
                              **prior.support(ids,lab,p['rows'],p['groups'])})
    result = {'policy':POLICY,'cohort':cohort,'whole_version_cves':len(parent_plan['rows']),
              'role_support':table,'assessment_support':summaries,'source_pins':pins,
              'decision':'SUPPORT_ONLY_NO_TRAINING_AUTHORIZATION','historical_inputs_verified':False}
    report = ['Fixed-epoch / single-gap support: '+cohort,'',
        'Target Day90; no selection block. All original assessment dates, groups and label rules retained.',
        'Classes must be counted after missing-label exclusions, not just before them.',
        'All blocks remain reported. The two-class subset is a diagnostic, not an approved change of cohort.',
        'No epoch count selected; no fitting, new network requests or official holdout access.',
        'This does not repair historical input/encoder availability or establish statistical power.','',
        'Rule | Fold | Role | CVEs | Labelled | Risers | Non-risers']
    report += [f'{r["policy"]} | {r["fold"]} | {r["role"]} | {r["cves"]} | {r["labelled"]} | {r["risers"]} | {r["non_risers"]}'
               for r in table if r['role'] in ('optimization','assessment')]
    a.publish(ROOT/cohort/'audit',{'findings.json':result,'role_support.csv':prior.old.csv_text(table),
              'assessment_support.csv':prior.old.csv_text(summaries),'REPORT.txt':'\n'.join(report)+'\n'},pins,verify)
    print(cohort+': single-gap support '+('reproduced.' if verify else 'saved.')+' No training or downloads.')


def review(verify=False):
    p = history.load(); a.check_lock(PASS)
    snap = a.read(PASS/'snapshot.json'); findings = a.read(PASS/'findings.json')
    a.check_pins(findings['source_pins'])
    pins = {**code_pins(),**findings['source_pins']}
    for path in (PASS/'snapshot.json',PASS/'availability.csv',PASS/'field_reconstructions.json',PASS/'ranking_metrics.csv'):
        pins[str(path)] = a.sha(path)
    fields = a.read(PASS/'field_reconstructions.json')
    availability = csv_rows(PASS/'availability.csv')
    rates = []
    for cutoff in ('end_day6','end_day7'):
        at_cutoff = [r for r in availability if r['cutoff_name']==cutoff]
        for state in sorted({r['cvss_status'] for r in at_cutoff}):
            members = [r for r in at_cutoff if r['cvss_status']==state]
            labelled = [r for r in members if r['label']!='']
            npos = sum(r['label']=='1' for r in labelled)
            rates.append({'cutoff':cutoff,'cvss_state':state,'histories':len(members),'eligible':len(labelled),
                          'risers':npos,'rate':npos/len(labelled) if labelled else None})
    late, changed = [], []
    for c in tqdm(snap['received_ids'],desc='Check dated description events; no downloads',unit='CVE'):
        events, epins = collector.read_complete(c); a.require(events is not None,'Missing frozen history');pins.update(epins)
        meta = p['records'][c]
        after = [e['cveChangeId'] for e in events if history.timestamp(meta['cutoff']) < history.timestamp(e['created'])
                 <= history.timestamp(meta['snapshot_asof']) and any(d.get('type')=='Description' for d in e['details'])]
        if after:late.append({'cve':c,'event_ids':after})
        desc = fields[c]['end_day7']['description']
        if desc['supported'] and not desc['same_as_saved']:changed.append(c)
    risers = [r for r in availability if r['cutoff_name']=='end_day7' and r['label']=='1']
    lags = [int(r['crossing_minus_first_cvss_addition_days']) for r in risers if r['crossing_minus_first_cvss_addition_days']!='']
    matched = [r for r in risers if r['dated_or_assumed_description_cvss_match']=='True']
    same7 = [r for r in matched if r['first_cvss_addition_day']=='7']
    metrics = csv_rows(PASS/'ranking_metrics.csv')
    gains = []
    for fold in ('0','1','2','3'):
        part = [r for r in metrics if r['cutoff']=='end_day7' and r['tier']=='dated_or_assumed_description_cvss_match'
                and r['fold']==fold and not r.get('label_sensitivity')]
        base = next(r for r in part if r['method']=='past_epss')
        for r in part:
            if r['method'].startswith('frozen_D_plus_past_epss'):
                gains.append({'fold':fold,'method':r['method'],'risers':int(r['risers']),
                              'auroc_gain':float(r['auroc'])-float(base['auroc']) if r['auroc'] and base['auroc'] else None})
    result = {'snapshot':PASS.name,'received_cves':len(snap['received_ids']),
              'late_description_event_cves':len(late),'supported_different_description_cves':len(changed),
              'cvss_rates':rates,'risers':len(risers),'dated_cvss_risers':len(lags),
              'crossing_before':sum(n<0 for n in lags),'crossing_same_day':sum(n==0 for n in lags),
              'crossing_next_day':sum(n==1 for n in lags),'crossing_later':sum(n>1 for n in lags),
              'crossing_more_than_week_later':sum(n>7 for n in lags),
              'day7_matched_risers':len(matched),'day7_matched_first_cvss_on_day7':len(same7),
              'those_crossing_on_day8':sum(p['records'][r['cve']]['label_not_predictor']['first_observed_crossing_day']==8 for r in same7),
              'matched_fold_seed_gains':gains,'source_pins':pins,
              'control_review':{'external_logistic_numbers_independently_reproduced':False,
                  'reason':'No saved per-CVE control predictions/checkpoints supplied; script fits nonchronological leave-one-fold-out logistic models. Not rerun.',
                  'limits':['Assessment crossing labels fit the external control.',
                            'Temporal and evidence-group separation are not enforced by that leave-one-fold-out script.',
                            'Frozen D includes reference counts and nonlinear structured interactions; a simple CVSS control does not isolate description causality.',
                            'State-specific rates and broad matched-tier rates use different denominators.',
                            'Overlapping positive/inconclusive subgroup intervals do not establish equal or different effects.']},
              'training':False,'network_requests':0,'official_holdout_access':False}
    text = ['Review of completed-pass claims; no refitting.','',
            f'{len(late)} CVEs have a Description event after Day7 and through the saved snapshot; this is not proof every event changes wording.',
            f'Only {len(changed)} differing historical descriptions have direct dated support under the existing reconstruction.',
            f'Of {len(lags)} dated risers: {result["crossing_before"]} before addition, {result["crossing_same_day"]} same date, '
            f'{result["crossing_next_day"]} next date, {result["crossing_later"]} later; {result["crossing_more_than_week_later"]} more than a week later.',
            'The external fitted CVSS control is not rerun or promoted to official evidence.',
            'The master PDF already contains the 7,924-history update; its 4,895-history table is preserved historical context.',
            'Retain all missing histories and explicit input/model-time uncertainty. No training approved.']
    a.publish(ROOT/'review',{'findings.json':result,'late_description_events.json':late,
              'supported_different_descriptions.json':changed,'cvss_rates.csv':prior.old.csv_text(rates),
              'fold_seed_gains.csv':prior.old.csv_text(gains),'REPORT.txt':'\n'.join(text)+'\n'},pins,verify)
    print('Day-7 follow-up review '+('reproduced.' if verify else 'saved.')+' No training, requests or test scoring.')
