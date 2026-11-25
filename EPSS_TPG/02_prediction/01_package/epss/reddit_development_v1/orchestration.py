"""Release checks and consolidated reports; never train as a side effect."""
import shutil
from . import common as c
from . import cache_stage, forecast
from ..reddit_primary import data as primary_data, cache

OUTPUT=c.ROOT/'05_results/07_reddit_studies/03_full_development_release_v1'


def preflight(verify=False,check_only=False):
    hashes=c.gate()
    entries,index,res,cache_identity,counts=cache_stage.check_plan('full')
    rows,parent,labels=primary_data.load('full',True)
    export=c.read(primary_data.OUT/'export/full/summary.json')
    c.require(export['selected_cves']=={'train':24213,'validation':972} and export['structured_features']==53,'RD02 export differs')
    c.require(export['full_source_counts']=={
        'train':{'D':24213,'M_all':26578,'M_prose':3309,'cves_prose':2077},
        'validation':{'D':972,'M_all':1289,'M_prose':1283,'cves_prose':968}},'Frozen source counts differ')
    c.require(export['validation_full_partitions']=={
        'no_detected_overlap_with_train':943,'overlap_with_train':29},'Overlap counts differ')
    c.require(all(len(r['structured'])>0 for r in rows),'Missing structured predictors')
    c.require(set(labels)==set(index) and set(index)=={r['cve'] for r in rows},'RD02 cache/export CVEs differ')
    daily,daily_identity=forecast.context('full')
    coverage=c.read(forecast.data.OUTPUT/'preflight/full/summary.json')['full_preflight']
    estimated=counts['chunks']*700000
    free=shutil.disk_usage(cache.DEFAULT).free
    report=dict(schema='reddit_full_execution_preflight_v1',upstream_hashes=hashes,
        source_pins=c.pins(),test_scoring=False,training=False,
        rd02=dict(**counts,cohorts=export['selected_cves'],source_counts=export['full_source_counts'],
            export_lock_sha256=c.sha(primary_data.OUT/'export/full/artifact_lock.json'),
            graph_identity_hash=c.digest(cache_identity),chunk_characters=1200,node_packet_budget=4096,
            node_input_width=782,edge_types=23,subtype_channels=32,severity_index=31,
            structured_features=53,neural_families=6,seeds=list(c.SEEDS),total_predictors=21),
        rd04=dict(counts=daily_identity['counts'],training_release_coverage=coverage['train'],
            validation_release_coverage=coverage['validation'],predictors=5,
            source_gate='NOT_APPLICABLE_EPSS_ONLY',epoch_zero='Exact persistence, not selection eligible'),
        disk_policy=dict(initial_cache_estimate_bytes=estimated,operating_reserve_bytes=cache_stage.RESERVE,
                         note='Free space is checked live by cache execution; estimate is not a guaranteed upper bound.'),
        specification_clarifications=[
            'Frozen source chunk limit is 1200 characters. 4096 is the packet node budget; whitespace-boundary overshoot is unchanged.',
            'RD02 v6_tpg includes the 32-channel subtype projection, matching registered v2 and verified smoke.',
            'RD04 primary slice: 24179 training and 972 validation CVEs; all-cell cohort: 24548 and 1106.',
            'Stage-3 counterfactuals have their own immutable lock, outside the already locked RD04 full run.',
            'No probability/prediction averaging across seeds. Metrics and paired same-seed differences are averaged.',
            'Post-hoc constant-share percentages are signed diagnostic ratios, not proofs of causal mechanisms.'],
        execution_owner='User. Full cache build and training have not been launched by this preflight.')
    print(f'[DISK] free={free/2**30:.2f} GiB; estimated cache={estimated/2**30:.2f} GiB; reserve=8 GiB')
    print(f'[RD02] {counts}; [RD04] {daily_identity["counts"]}; test remains closed.')
    if check_only:
        return report
    with c.core.transaction(OUTPUT/'preflight',verify) as stage:
        c.put(stage,'findings.json',report)
        (stage/'REPORT.txt').write_text('Full Reddit development execution preflight\n'+c.dump(report))
    print('Full-development preflight '+('reproduced' if verify else 'saved')+'. No graph build, training or test scoring.')
    return report


def replay_gates():
    c.gate()
    from ..reddit_primary import runner as primary
    from ..reddit_forecasting import runner as daily
    cache_stage.prepare('smoke',verify=True)
    primary.run(True);daily.smoke(True)
    c.gate()
    print('Original RD02/RD04 smoke checkpoints reproduced; no retraining or test scoring.')


def report(scale,verify=False):
    c.gate()
    primary=primary_data.OUT/('full' if scale=='full' else 'development_smoke_v1')
    daily=forecast.destination(scale)
    diagnostic=forecast.data.OUTPUT/'counterfactuals_v1'/scale
    paths={'RD02':primary,'RD04':daily,'RD04_counterfactuals':diagnostic}
    hashes={}
    for name,directory in paths.items():
        c.core.check_lock(directory);hashes[name]=c.sha(directory/'artifact_lock.json')
    destination=OUTPUT/scale
    with c.core.transaction(destination,verify) as stage:
        findings=dict(schema='reddit_completed_development_report_v1',scale=scale,artifact_hashes=hashes,
                      upstream_hashes=c.gate(),source_pins=c.pins(),test_scoring=False,
                      studies={name:c.read(directory/'report/findings.json') for name,directory in paths.items()})
        for name,directory in paths.items():
            filename={'RD02':'rd02_full_metrics.csv','RD04':'rd04_full_metrics.csv',
                      'RD04_counterfactuals':'rd04_counterfactual_metrics.csv'}[name]
            shutil.copyfile(directory/'report'/filename,stage/filename)
        c.put(stage,'findings.json',findings)
        (stage/'REPORT.txt').write_text('Completed Reddit development studies\n'
            'Separate retrospective Day-60 estimation and daily EPSS forecasting; no held-out test scoring.\n'
            'All fixed seeds retained. Metrics averaged across seeds, never predictions.\n'
            'The counterfactual constant is validation-derived and post-hoc, not a deployable control.\n'+c.dump(findings))
    print(f'Combined development report '+('verified' if verify else 'saved')+f' at {destination}. No training or test scoring.')
