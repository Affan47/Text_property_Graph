"""Step 1: verify a locked preparation and report split suitability, without fitting models."""
from collections import Counter
from datetime import date
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'02_prediction/01_package'))
from epss.forecast_data import (load_prepared, read_json, sha256, split_manifest,
                               validated_split_policy, write_json, TASKS)

DERIVED = ROOT/'04_data/01_records_and_graphs/epss_day30_to_day60'
OUT = ROOT/'05_results/03_dataset_analysis/01_revision_dataset/13_epss_targets'


def describe_split(inputs, labels, manifest):
    result = {}
    for role in ('train','validation','test'):
        ids = sorted(c for c in inputs if manifest[c]['split'] == role and labels[c]['eligible']['endpoint'])
        anchors = sorted(inputs[c]['input_end_date'] for c in ids)
        target_dates = sorted(labels[c]['target_date'] for c in ids)
        group_sizes = Counter(manifest[c]['group_id'] for c in ids)
        result[role] = {
            'endpoint_cves': len(ids), 'distinct_groups': len(group_sizes),
            'largest_retained_group': max(group_sizes.values(),default=0),
            'anchor_min': min(anchors,default=None), 'anchor_max': max(anchors,default=None),
            'target_min': min(target_dates,default=None), 'target_max': max(target_dates,default=None),
            'anchor_month_counts': dict(sorted(Counter(a[:7] for a in anchors).items())),
            'complete_31_day_inputs':sum(all(inputs[c]['X']['observed_mask']) for c in ids),
            'partial_inputs':sum(not all(inputs[c]['X']['observed_mask']) for c in ids),
            'median_observed_input_days':None,
        }
        counts=sorted(sum(inputs[c]['X']['observed_mask']) for c in ids)
        if counts:
            n=len(counts)
            result[role]['median_observed_input_days']=(counts[(n-1)//2]+counts[n//2])/2
    result['excluded_cves_all_tasks']=sum(row['split']=='excluded' for row in manifest.values())
    result['excluded_endpoint_eligible_cves']=sum(row['split']=='excluded' and labels[c]['eligible']['endpoint'] for c,row in manifest.items())
    result['exclusion_reasons_all_cves']=dict(Counter(r['exclusion_reason'] for r in manifest.values() if r['split']=='excluded'))
    return result


def independent_chronology(inputs, labels, manifest, policy):
    """Check dates independently of candidate_split; target dates are score dates."""
    assigned_groups = {}
    for cve, row in manifest.items():
        anchor = date.fromisoformat(inputs[cve]['input_end_date'])
        target = date.fromisoformat(labels[cve]['target_date'])
        assert (target-anchor).days == 30
        role = row['split']
        if role == 'excluded':
            continue
        if policy['observation_end']:
            assert target <= date.fromisoformat(policy['observation_end'])
        if role == 'train':
            assert anchor < target < date.fromisoformat(policy['validation_start'])
        elif role == 'validation':
            assert date.fromisoformat(policy['validation_start']) <= anchor < target < date.fromisoformat(policy['test_start'])
        elif role == 'test':
            assert date.fromisoformat(policy['test_start']) <= anchor < date.fromisoformat(policy['test_end'])
        else:
            raise AssertionError(role)
        gid=row['group_id']
        assert assigned_groups.setdefault(gid,role)==role
    return {'anchor_and_target_dates_independently_checked':len(inputs),
            'active_groups_have_one_fold':True,'no_boundary_crossing_training_or_validation_targets':True}


def main():
    data = load_prepared(DERIVED,verify_source=True)
    inputs, labels, manifest = (data[k] for k in ('inputs.json','labels.json','split_manifest.json'))
    summary = data['preparation_summary.json']
    policy = data['split_policy.json']
    checks = independent_chronology(inputs,labels,manifest,policy)
    for role in ('train','validation','test','excluded'):
        for task in TASKS:
            assert summary['splits'][role][task] == sum(r['split']==role and labels[c]['eligible'][task] for c,r in manifest.items())
    chosen = describe_split(inputs,labels,manifest)
    assert all(chosen[role]['endpoint_cves'] > 0 for role in ('train','validation','test'))

    # Dates are explicit calendar alternatives, not a search over model scores.
    alternatives = [
        ('existing_calendar_split',policy),
        ('validation_starts_july_2024',dict(policy,validation_start='2024-07-01')),
        ('test_starts_april_2025',dict(policy,validation_start='2025-01-01',test_start='2025-04-01')),
    ]
    candidates=[]
    group_map={c:r['group_id'] for c,r in manifest.items()}
    for name, candidate in alternatives:
        validated_split_policy(candidate)
        proposed, conflicts=split_manifest(inputs,labels,group_map,candidate)
        candidates.append({'name':name,'policy':candidate,'cross_fold_groups':len(conflicts),
                           'counts':describe_split(inputs,labels,proposed),
                           'selected':name=='existing_calendar_split'})
    assert candidates[0]['counts']==chosen

    evidence=data['grouping_evidence.json']
    shared=sorted(evidence,key=lambda x:(-x['cve_count'],x['kind'],x['key_sha256']))
    generic=[r for r in shared if r['homepage_review_hint']]
    warnings=[
        'The selected dates remain the original conservative baseline design; no alternative was selected using prediction errors.',
        'The small validation cohort and large related groups limit strong neural-model conclusions; inspect grouping evidence before changing the research split.',
        'Exact shared text and URLs protect only declared groups; undetected semantic duplicates may remain.',
        'Shared current/later evidence is used only for conservative grouping, never as a predictor. This can select a restricted cohort.',
        'Day-30 scores have a date, not a verified release timestamp. Results would be retrospective daily-score forecasts, not proof of an intraday deployment.',
        'The observation cutoff is the documented recollection cutoff (2026-09-29), not a claim of complete EPSS coverage; missing targets remain masked.',
        'Do not reuse these CVE folds as an SMP-event experiment without a separately reviewed event-time contract.',
    ]
    if summary['direction_classes_by_split']['validation']['decrease']==0:
        warnings.append('No decrease examples in the existing direction validation cohort; this preparation is for endpoint regression, not approval of the three-class task.')
    for role in ('train','validation','test'):
        months=chosen[role]['anchor_month_counts']
        peak_month, count=max(months.items(),key=lambda item:(item[1],item[0]))
        chosen[role]['largest_anchor_month']={'month':peak_month,'cves':count,
                                             'fraction':count/chosen[role]['endpoint_cves']}
        if count/chosen[role]['endpoint_cves'] > .5:
            warnings.append(f"{role.title()}: {count:,} of {chosen[role]['endpoint_cves']:,} endpoint CVEs ({100*count/chosen[role]['endpoint_cves']:.1f}%) have anchors in {peak_month}; this concentration limits temporal representativeness.")
    tests=subprocess.run([sys.executable,'-m','unittest','test_forecast_data','test_training_feasibility',
                          'test_smp_dates','test_dated_smp_epss'],cwd=Path(__file__).parent,
                         capture_output=True,text=True)
    assert tests.returncode == 0, tests.stdout+tests.stderr
    report = {
        'step':1,'status':'DATA_PREPARATION_VERIFIED; no baseline evaluation or model fitting performed',
        'source_path':summary['source_path'],'source_sha256':summary['source_sha256'],
        'preparation_lock_sha256':sha256(DERIVED/'preparation_lock.json'),
        'split_policy':policy,'selected_split':chosen,'candidate_comparisons':candidates,
        'selection_reason':'Retain the previously declared chronological baseline; record alternatives for review only.',
        'grouping':summary['duplicate_groups'],'reviewed_url_exclusions':data['reviewed_url_exclusions.json'],
        'shared_homepage_review_hints':generic,'largest_shared_evidence':shared[:20],
        'readiness':{'prepared_for_next_persistence_implementation':True,
                     'prediction_test_set_evaluated':False,'trained_predictive_model':False,
                     'historical_text_or_smp_inputs_enabled':False,'vulncheck_targets_enabled':False,
                     'scientific_split_finalized_for_all_future_experiments':False},
        'checks':checks,'test_output':tests.stdout+tests.stderr,'warnings':warnings,
        'report_script_sha256':sha256(__file__),
    }
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'step1_readiness.json',report)
    rows='\n'.join(f"| {role.title()} | {chosen[role]['endpoint_cves']:,} | {chosen[role]['distinct_groups']:,} | {chosen[role]['anchor_min']} to {chosen[role]['anchor_max']} |" for role in ('train','validation','test'))
    alternatives_table='\n'.join(f"| {r['name']} | {r['policy']['validation_start']} | {r['policy']['test_start']} | {r['counts']['train']['endpoint_cves']:,} | {r['counts']['validation']['endpoint_cves']:,} | {r['counts']['test']['endpoint_cves']:,} |" for r in candidates)
    text=f'''# Step 1: Forecasting Data and Split Readiness

This step prepares and checks data. It does not train a model, calculate
baseline errors, or inspect test predictions.

## Input and Target

The current source is `{Path(summary['source_path']).name}`. Its SHA-256 is
`{summary['source_sha256']}`. Every input contains only EPSS scores dated
publication through Day 30, observation masks and past-derived statistics.
The endpoint target is EPSS on Day 60, saved separately. Missing values are
not replaced with zero labels. Text, SMP, CVSS and VulnCheck predictors remain
disabled. Statistical preprocessing is fitted on eligible training inputs only.

## Selected Baseline Split

The anchor is publication plus 30 days. Training anchors precede
{policy['validation_start']}; validation anchors run from that date up to,
but not including, {policy['test_start']}; test anchors run from that date up
to, but not including, {policy['test_end']}. Training and validation targets
must also precede the next split boundary. The score-date observation cutoff
is {policy['observation_end']}; the presence of a cutoff does not make missing
scores available. Dates refer to score days, not independently verified release times.

| Split | Endpoint-eligible CVEs | Distinct related groups | Actual retained anchor dates |
|---|---|---|---|
{rows}

These are CVE counts, not a random percentage allocation. The original
conservative dates are retained for the baseline implementation. They are
not asserted to be the final design for all later neural or SMP experiments.

The monthly counts are preserved in the JSON report. The largest training
month is {chosen['train']['largest_anchor_month']['month']}, with
{chosen['train']['largest_anchor_month']['cves']:,} CVEs
({100*chosen['train']['largest_anchor_month']['fraction']:.1f}% of training).
This is a sampling concentration, not evidence that these CVEs are duplicates.
It should be considered when interpreting later baseline and model results.

## Boundary Alternatives for Review

| Candidate | Validation starts | Test starts | Training CVEs | Validation CVEs | Test CVEs |
|---|---|---|---|---|---|
{alternatives_table}

All candidates keep the same end of the test period and grouping rules.
These counts use eligibility and dates, not predicted errors or test target
distributions. No alternative is automatically selected because it retains
more examples. The selection must be decided before evaluating predictions.

## Grouping Review

The largest group still contains {summary['duplicate_groups']['largest_group']:,} CVEs.
The prepared `grouping_evidence.json` names each shared key, its CVEs and
whether it was used. Homepage-like links are flagged for inspection but are
not silently removed. No reviewed URL exclusions are active in this run.
A later exclusion must identify an exact URL and give a written reason;
it does not cancel any identical-description or repeated-post link.
Groups spanning active chronological folds are quarantined. Future CVEs are
never moved backward into training.

## Checks and Lock

The source hash, raw inputs and labels, chronological boundaries, target
maturity, group separation, training-only scaling and task-specific CVE lists
were verified. A separate date check covered all {len(inputs):,} CVEs.
`preparation_lock.json` records hashes of every prepared artifact. A future
loader refuses modified or incomplete bundles and checks the source again.
This detects accidental changes; it is not a cryptographic signature or a
claim that all possible semantic duplicates have been discovered.

The focused test output is included below:

```text
{(tests.stdout+tests.stderr).strip()}
```

## Limitations

'''+'\n'.join('- '+w for w in warnings)+'''

## Next Step

Implement persistence and the evaluation harness, first as unit-tested code.
Do not use the final test set for repeated design decisions. Decide whether
to retain this conservative split or approve a revised grouping/boundary
policy before any final test evaluation. TPG-GNN and SMP are later steps.
'''
    for suffix in ('md','txt'):
        (OUT/f'STEP1_READINESS.{suffix}').write_text(text)
    print(json.dumps({'step':1,'split':chosen,'checks':checks,
                      'test_output':report['test_output'],'trained':False},indent=2))


if __name__ == '__main__':
    main()
