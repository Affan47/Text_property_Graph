"""Publish proposed RD01-RD04 contracts and a gate review; no splits or models."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import shutil
from collections import Counter

from audit_reddit import ROOT, read, sha, dumps, require
from repair_provenance import atomic
from reddit_gates import RESULTS, SOURCE, DERIVED, FINAL, CATALOGUE, verify_output, lock_output

OUTPUT = RESULTS/'stage2_4_specifications_v2'
CONFIGS = ROOT/'02_prediction/04_configs/05_reddit_studies'
CF_CONFIG = ROOT/'02_prediction/04_configs/02_continuous_forecasting/01_stage1.json'
TPG_SPEC = ROOT/'05_results/06_tpg_revision/10_v6_diff_audit_v1/trv05_runner_spec.json'
RESOURCES = ROOT/'01_tpg/05_workspace/vulncheck_description_v1/full/resources.json'


def specifications():
    shared={'state':'PROPOSED_NOT_FROZEN','training_enabled':False,'test_access_enabled':False,
        'seeds':[17,29,43],'bootstrap':{'repetitions':1000,'seed':6029,'primary_unit':'CVE',
            'additional_unit':'shared-post connected component','paired':True,'confidence':0.95},
        'gates':['Stage 0 hashes verified','Stage 1 limitations explicitly accepted or resolved',
                 'RD01 component/chronology review and freeze','source policy review',
                 'matched-batch smoke and deterministic checkpoint reload'],
        'catalogue_predictor_ban':['in_vckev',*CATALOGUE,'derived catalogue counts/masks/nodes/edges'],
        'reports':['all seeds and seed means','per-CVE predictions with actual value, masks and provenance',
            'fixed-baseline paired differences','missingness and source-availability strata',
            'component-aware uncertainty; no test-driven selection'],
        'new_evaluation_warning':'No untouched claim for previously inspected CVEs or connected evidence. This is a proposed later-publication holdout within the saved snapshot, not a prospectively collected cohort.'}
    split={**deepcopy(shared),'id':'RD01','stage':2,'task':'Chronological component-disjoint split proposal',
        'eligibility':'NVD published_date >= 2021-04-14; study-specific targets must be observed, never imputed',
        'anchor_for_role':'NVD published_date; rolling windows inherit CVE role and never split one CVE across roles',
        'proposed_publication_intervals':{'train':['2021-04-14','2023-12-31'],
            'validation':['2024-01-01','2024-12-31'],'test':['2025-01-01','2026-11-03']},
        'not_yet_assigned':True,'cutoff':'2026-11-03',
        'group_policy':{'graph':'conservative shared fingerprints + co-mentioned CVE IDs + exact old Reddit text reuse',
            'cross_interval_components':'Quarantine entire component; do not move late CVEs into earlier training',
            'prior_test':'Quarantine every current CVE connected to any prior test CVE from all new study cohorts',
            'prior_train_validation':'Preserve old role when it matches chronological interval; quarantine a whole component on role conflict',
            'future_test':'Reject entire component if any member has any previous-study role, including excluded',
            'text_identity':'Also audit repeated descriptions and normalized/near-duplicate posts before freeze; no automatic merging of distinct vulnerabilities',
            'shared_urls':'Report separately; shared URL alone is not proof of same vulnerability'},
        'target_maturity':{'RD02':'published+60 must be <= role period end and collection cutoff and have an observed score',
            'RD03':'Require explicit successful per-CVE receipt; unknown excluded; label is fresh catalogue inclusion, not a historical outcome',
            'RD04':'For each t in 0..89, u in t+1..90, date(pub+u) must remain inside the CVE role period and <= cutoff; require actual target score'},
        'distributions':{'before_freeze':'Counts by dates, component sizes, previous exposure and observed-target availability; no adaptive cutoff search on outcomes',
            'after_rules_frozen':'Per-role CVE counts, verified/unknown classes and fixed score bands; do not revise cuts to improve them',
            'score_bands':[0,.001,.01,.1,1],'band_boundaries':'[0,.001), [.001,.01), [.01,.1), [.1,1]',
            'untouched_test':'Restricted report only until final evaluation release; no per-CVE test labels/text in development exports'},
        'freeze_checks':['all CVEs assigned or given an exclusion reason','no connected component spans roles',
            'no old exposure component in test','no training target crosses validation boundary',
            'no vocabulary/scaler/threshold fit on validation/test','RD03 labels contain both classes in development or stop']}
    settings=deepcopy(read(TPG_SPEC)['settings'])
    settings.update(dropout=0.,device='cpu',readout='node_mean',fresh_initialization=True)
    settings.pop('threshold',None); settings.pop('selection',None); settings.pop('loss',None)
    structured={'categorical':'CVSS version + version-aware vector components, one-hot with explicit unknown/missing categories; vocabulary fit on train only',
        'numeric':['cvss_vector_score','log1p(distinct all_sources_urls)','log1p(distinct github_only_urls)','log1p(distinct non_github_urls)'],
        'binary':['GitHub URL present'],
        'numeric_preprocessing':'training-only median imputation, standardization and missing flags for allowed CVSS/count features only',
        'excluded':['n_references_nvd (retained for audit only)','CVE identity as feature','publication/update dates as features',
            'vuln_status','CVSS source identity','all EPSS fields','all catalogue fields and derived indicators'],
        'historical_status':'Current CVSS/reference counts are retrospective, not verified at publication or Day 60'}
    graph={'frontend':'SpacyFrontendV6 + versioned security product corrections through v6',
        'encoder_identity':read(RESOURCES)['encoder'],
        'subtype_vocabulary':read(TPG_SPEC)['subtype_vocabulary'],
        'gnn':'Two existing EdgeTypeGNNLayer layers, hidden=32, learned edge-type and edge-semantics channels, residual updates',
        'embedding':'Same frozen text encoder revision and token embeddings for matched pooled and graph controls; no fine-tuning',
        'reuse':'Only exact text hash, tokenizer/encoder identity, chunk offsets, parser/resources and all graph pass versions match',
        'required_new_input_gate':'32 training CVEs, then 12 train/6 validation integration smoke; audit predicates/subtypes/edge provenance before expansion',
        'subtype_channels':32,'subtype_initialization':'zero-initialized 32x32 projection; SEVERITY index 31; disabled arm zeros its contribution',
        'relations':'no COREF; use saved named edge vocabulary, not hardcoded relation indices',
        'source_pooling':'mean chunks within document; mean documents within source; masked source attention for D/M; structured late fusion',
        'source_absence':'D and DM retain identical CVE cohorts; absent M represented by mask, no source-complete-only selection',
        'source_policy':'Retain exact-deduplicated posts including tagged bare-ID/bulk candidates for first matched study; review explicit catalogue-answer text before freezing; no outcome-informed filtering',
        'budgets':'Node/edge/chunk budgets are microbatch limits, never truncation; deterministic VJP replay, no stochastic dropout',
        'normal_text_control':'pooled tokens is a non-graph control; not a general-text TPG',
        'new_model_families':'No GAT/RGCN/readout/weight-decay sweep in this bounded study'}
    regression={**deepcopy(shared),'id':'RD02','stage':3,'task':'Retrospective Day-60 EPSS estimation, not a historical forecast or exploitation classification',
        'target':'Stored observed EPSS on published_date + 60 days','inputs':'D = description; M = Reddit post text; no G/N summaries in this dataset',
        'structured':structured,'graph':graph,'models':[
            {'name':'training_mean','fits':1},{'name':'training_median','fits':1},
            {'name':'structured_ridge','fits':1,'alpha':1.0},
            *[{'name':f'{sources}_{model}','sources':sources,'structured':True,'fits':3,'subtypes':subtypes}
              for sources in ('D','DM') for model,subtypes in [('pooled_text',False),('v6_tpg',False),('v6_tpg_subtypes',True)]]],
        'total_fits':21,'neural_settings':{**settings,'loss':'equal-CVE raw MAE','selection':'earliest strict minimum clipped validation MAE','epochs':100,'patience':15},
        'outputs':'raw and [0,1]-clipped scalar per CVE, all seeds and constants',
        'metrics':['MAE (primary)','RMSE','R2','signed bias','Pearson','Spearman','clipped fraction','high-score error slices'],
        'not_metrics':['F1/accuracy are not native metrics of this regression task'],
        'version_clean':'Separate evaluation on no release transition in (publication, publication+60]; do not use the whole 91-day changed flag as a 60-day condition; no refit',
        'figures':['actual vs predicted','residual histogram','error by score band','training/validation curves','missingness/source coverage','seed comparison','version-clean vs cross-version'],
        'limitations':['Current D/M/CVSS may postdate target; label the study retrospective','No claim of improved future warning from retrospective text']}
    classification={**deepcopy(shared),'id':'RD03','stage':3,'task':'Fresh verified VulnCheck catalogue inclusion classification; not future exploitation',
        'target':'verified_in_vckev from per-CVE successful receipts; never use legacy false or unknown as zero',
        'collection_policy':'Bounded API check is a software gate only. Require broader verified labels and a declared <=7-day collection span for the cohort; report drift/times, do not claim a single historical snapshot',
        'structured':structured,'graph':{**graph,'subtypes_enabled':True},
        'models':[{'name':'structured_logistic','fits':1,'C':1.,'solver':'liblinear','max_iter':2000},
            {'name':'D_tfidf_structured_logistic','fits':1,'C':1.,'solver':'liblinear','max_iter':2000},
            {'name':'DM_tfidf_structured_logistic','fits':1,'C':1.,'solver':'liblinear','max_iter':2000},
            *[{'name':f'{source}_{model}','sources':source,'structured':True,'fits':3}
              for source in ('D','DM') for model in ('pooled_text','v6_tpg_subtypes')]],
        'tfidf':{'fit':'train only','max_features':20000,'ngram_range':[1,2],'min_df':2,'sublinear_tf':True,
                 'DM':'Separate D/M vocabularies; concatenate source vectors and allowed structured features'},
        'total_fits':15,'neural_settings':{**settings,'loss':'BCEWithLogits; training-only balanced class weights N/(2*Nclass)',
            'selection':'earliest strict maximum full-validation AP','epochs':100,'patience':15},
        'thresholds':{'reference':0.5,'additional':'One validation-only best-F1 cutoff per fit; candidate unique probabilities plus 0 and 1; highest cutoff wins exact ties; freeze before test',
            'no_tuning':'No subgroup-specific or test-selected cutoffs'},
        'metrics':['AP (primary)','ROC-AUC','Brier','log loss','precision','recall','F1','accuracy','balanced accuracy','MCC','TP/FP/TN/FN'],
        'undefined':'null with reason if single-class or denominator undefined',
        'figures':['PR','ROC','calibration','confusion matrices at both frozen cutoffs','probability distributions','source coverage by class','per-seed training diagnostics'],
        'blocking_gate':'No verified negatives/positives available yet; cannot train RD03 from the current receipt status'}
    cf=read(CF_CONFIG)
    forecasting={**deepcopy(shared),'id':'RD04','stage':4,'task':'Daily updated future EPSS trajectories; separate descriptive Reddit event analysis',
        'models':['persistence','training_only_lead_median_correction','temporal_GRU_seed17','temporal_GRU_seed29','temporal_GRU_seed43'],
        'temporal_settings':cf,'fresh_training':True,
        'inputs':'At anchor t=0..89: only scores observed <=t, observation flags, age and known-history channels; no text or future scores/statuses',
        'targets':'Actual observed scores at u=t+1..90 within role target-date bounds; keep missing-target masks; lazy windows',
        'weighting':'Equal observed leads within anchor, then equal eligible anchors within CVE, then equal CVEs; never weight CVEs by post count',
        'controls':'Persistence from last observed score; lead correction fitted from train residual weighted medians; report raw and clipped versions',
        'slices':['all eligible cells','Day30 to Day60 primary','lead7','lead14','lead30','single-version','cross-version'],
        'version_policy':'Assign each stored date from the pinned release schedule; test actual anchor/target and input-history spans. Cross-version label for history spanning a release or (anchor,target] crossing a release; do not use only the global 91-day flag',
        'metrics':['equal-CVE MAE','equal-CVE RMSE','signed bias','clipped fraction','counts of CVEs/anchors/cells'],
        'diagnostics':['Epoch0 exact persistence','per-epoch selection metric and clipping/bias','flat_history','last_obs_only','validation-derived constant shift (post-hoc diagnostic, not fitted predictor)'],
        'event_study':{'training':False,'causal_claim':False,'source':'Deduplicated post-date entries; historical wording not authenticated',
            'windows':[7,14],'eligibility':'Every daily EPSS score from -h through +h observed; recompute counts after recovery/dedup, do not assume old counts unchanged',
            'original_cve_counts':{'7':2756,'14':1729},
            'summaries':['EPSS at -h,-1,0,+1,+h','OLS pre/post slopes excluding event day from slopes','post-minus-pre change',
                         'max/min in days 1..h','first day of maximum','whether already rising before post'],
            'multiple_posts':'Report all events with equal-CVE aggregation and first eligible post sensitivity; count all other post entries in window; overlap flag, no claims of independent events',
            'timing':'Date-level association only; daily EPSS release vs post time unknown',
            'no_future_text':'Do not add TPG/text to forecaster without content-version/capture availability evidence',
            'test':'Event-level outcomes for prospective test-role CVEs withheld until final release'},
        'figures':['error by anchor and lead heatmap','true/predicted trajectories','version-crossing comparison','clipping/bias trajectory','event-centered score curves with denominators']}
    return {x['id']:x for x in (split,regression,classification,forecasting)}


def size_distribution(groups):
    return {str(k):v for k,v in sorted(Counter(g['current_count'] for g in groups).items())}


def publish(verify=False):
    for folder in [RESULTS/'stage0',RESULTS/'stage1_epss',RESULTS/'stage1_sanitization',RESULTS/'official_archive/stage1_epss']:
        verify_output(folder)
    sanitized=read(RESULTS/'stage1_sanitization/report.json')
    initial=read(RESULTS/'stage1_epss/report.json'); official=read(RESULTS/'official_archive/stage1_epss/report.json')
    groups=read(RESULTS/'stage1_sanitization/connected_components.json')
    with gzip.open(RESULTS/'stage1_sanitization/shared_post_graph.jsonl.gz','rt') as f:
        posts=[json.loads(line) for line in f]
    proposal=specifications()
    review={'schema':'reddit_premodeling_gate_review_v1','storage':read(RESULTS/'stage0/report.json'),
        'sanitization':sanitized,'epss_direct_retry':initial['counts'],'epss_official_retry':official['counts'],
        'official_archive_commit':read(RESULTS/'official_archive/stage1_epss/official_repository.json')['commit'],
        'bulk_candidate_fingerprints':sum(p['bulk_advisory_candidate'] for p in posts),
        'bare_id_unique_fingerprints':sum(p['bare_cve_id'] for p in posts),
        'largest_components':[{k:v for k,v in g.items() if k not in ('current_cves','external_or_old_only_cves')}
                              for g in sorted(groups,key=lambda g:(-g['current_count'],g['component_id']))[:10]],
        'component_size_distribution':size_distribution(groups),
        'gate_decisions':{'stage0':'PASSED','stage1_dedup_and_quarantine':'PASSED',
            'stage1_epss':'BLOCKED: all nine direct URLs returned 403; none of the nine date files is in the pinned official tree; 17,112 scores remain missing',
            'stage1_vulncheck':'BLOCKED: no authenticated receipts/API key provided; 44,985 outcomes unknown under the new verification contract',
            'stage1_historical_reddit':'UNRESOLVED: no raw post IDs, edit history or content capture evidence recovered',
            'stage2':'SPECIFICATION_ONLY: no split manifest frozen','stages3_4':'DISABLED: no model training or test scoring'},
        'recommendation':'Review giant component impact and accept or resolve remaining missing-score policy before RD01. RD03 requires verified labels. Do not claim strict historical leakage freedom for retrospective D/M/CVSS.',
        'prior_test_predictions_read':False,'models_trained':0,'split_assigned':False}
    summary='''RD00 Stage 0 and Stage 1 review

The complete downloaded bundle is now copied into durable project storage.
Every source byte hash matches the original manifest. The /tmp source remains unchanged.

EPSS recovery has NOT succeeded. Direct archive retries returned HTTP 403.
The official repository linked by FIRST does not contain the nine requested
date paths at the pinned commit. No missing value was filled or changed to zero.
Both versioned retry outputs are currently identical in bytes to v1; they are
separate attempt records, not improved score datasets. The one exploratory
FIRST API query also returned 422/no results; that single query does not prove
all CVEs are absent on that date. No bulk API scraping was attempted.

All ten catalogue-only fields, their absent/null patterns and in_vckev are kept
outside predictor_candidates.jsonl.gz. EPSS histories also remain on the
restricted outcome side until a timestamp-aware loader selects permitted past
inputs. CVE IDs and dates in candidate files are join/control metadata, not
automatically model features. Current CVSS and text are retrospective snapshots.

Exact within-CVE duplicate post entries are removed only in derived candidates.
Raw reference counts are retained beside exact-string distinct counts. Bare-ID
and multi-CVE/bulk posts are tagged in a separate audit, not silently dropped.
Connected CVEs remain distinct vulnerabilities. Components identify shared
evidence and possible split contamination, not duplicate CVE identities.

No new splits, graphs, embeddings, training or test scoring were performed.
RD01-RD04 JSON files are exact proposed contracts, not executable training jobs.
The 2024/2025 chronology boundaries are a proposal pending component-size and
coverage review, not a frozen allocation. Outcomes must not select new cutoffs.
'''
    summary+='\nVerified counts\n'+dumps(sanitized['counts'])
    summary+='\nGate decisions\n'+dumps(review['gate_decisions'])
    inputs={str(CF_CONFIG):sha(CF_CONFIG),str(TPG_SPEC):sha(TPG_SPEC),str(RESOURCES):sha(RESOURCES),str(Path(__file__)):sha(Path(__file__))}
    for name in ['stage0','stage1_epss','stage1_sanitization','official_archive/stage1_epss']:
        path=RESULTS/name/'artifact_lock.json'; inputs[str(path)]=sha(path)
    if verify:
        verify_output(OUTPUT)
        require(read(OUTPUT/'review.json')==review,'Review differs')
        require((OUTPUT/'review.txt').read_text()==summary,'Plain-text review differs')
        for name,spec in proposal.items():
            require(read(OUTPUT/(name+'_spec.json'))==spec,'Specification differs')
            require(sha(CONFIGS/(name+'_spec.json'))==sha(OUTPUT/(name+'_spec.json')),'Published config differs')
        print('Stage 0/1 review and RD01-RD04 specifications reproduced. No splits or training.'); return
    require(not OUTPUT.exists(),'Use --verify-only for existing specifications')
    OUTPUT.mkdir(parents=True); CONFIGS.mkdir(parents=True,exist_ok=True)
    atomic(OUTPUT/'review.json',review); (OUTPUT/'review.txt').write_text(summary)
    for name,spec in proposal.items():
        atomic(OUTPUT/(name+'_spec.json'),spec)
        destination=CONFIGS/(name+'_spec.json')
        if destination.exists():
            require(sha(destination)==sha(OUTPUT/(name+'_spec.json')),'Refuse to replace different config')
        else:
            shutil.copyfile(OUTPUT/(name+'_spec.json'),destination)
    lock_output(OUTPUT,inputs)
    print(summary); print('Proposed specifications saved: '+str(CONFIGS))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--verify-only',action='store_true')
    publish(parser.parse_args().verify_only)
