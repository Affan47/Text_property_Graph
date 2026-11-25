"""CS01: exhaustive named source ablations and explicit stage/feature contracts."""
from itertools import combinations
from pathlib import Path

from .. import forecast_day60_multisource as parent

ROOT = parent.ROOT
CONFIG = ROOT/'02_prediction/04_configs/01_consolidated/00_protocol.json'
SCHEMA = 'cs01_consolidated_design_v1'
SOURCES = ('description', 'github_summary', 'non_github_summary', 'smp')
CODES = dict(zip(SOURCES, ('D','G','N','M')))
read, write, require, sha256, tqdm = parent.read, parent.write, parent.require, parent.sha256, parent.tqdm


def load():
    p = read(CONFIG)
    require(p['schema'] == SCHEMA and p['primary_target_day'] == 60
            and p['source_order'] == list(SOURCES) and p['continuous_anchors'] == {'first':0,'last':89}
            and p['continuous_min_observed_input'] == 1 and not p['training_enabled']
            and not p['test_export_enabled'], 'Wrong consolidated design phase')
    return p


def experiment_matrix():
    rows = []
    for n in range(1,5):
        for sources in combinations(SOURCES,n):
            code = ''.join(CODES[s] for s in sources)
            for representation in ('pooled_text','security_tpg'):
                for structured in (False,True):
                    rows.append({'id':f'P01_{code}_{representation}_S{int(structured)}',
                        'study':'01_primary_tpg_day60', 'sources':list(sources),
                        'representation':representation, 'structured_blocks':['cvss','reference_count'] if structured else [],
                        'edge_view':'no_coref' if representation == 'security_tpg' else None,
                        'matched_cohort':'all primary-eligible CVEs, including missing-source cases',
                        'status':'registered_not_trained'})
    rows += [{'id':f'P01_NONE_{name}', 'study':'01_primary_tpg_day60', 'sources':[],
              'representation':name,'structured_blocks':['cvss','reference_count'] if name=='structured_ridge' else [],
              'edge_view':None,'matched_cohort':'all primary-eligible CVEs','status':'registered_not_trained'}
             for name in ('training_mean','training_median','structured_ridge')]
    for mode in ('repeated_only','all_coref','security_edges_removed'):
        rows.append({'id':f'P02_DGNM_{mode}_S1','study':'01_primary_tpg_day60','sources':list(SOURCES),
            'representation':'security_tpg','structured_blocks':['cvss','reference_count'],
            'edge_view':mode,'matched_cohort':'same full-source primary cohort',
            'interpretation':'Edge intervention with existing nodes; not a newly extracted normal-text graph.',
            'status':'registered_not_trained'})
    rows.append({'id':'P02_DGNM_normal_text_tpg_S1','study':'01_primary_tpg_day60','sources':list(SOURCES),
        'representation':'normal_text_tpg','structured_blocks':['cvss','reference_count'],
        'edge_view':'normal_frontend_new_extraction', 'matched_cohort':'same full-source primary cohort',
        'interpretation':'Use the existing general-text frontend; requires a separate cache, not SEC edge filtering.',
        'status':'registered_requires_normal_frontend_cache'})
    for removed in ('cvss','reference_count'):
        rows.append({'id':f'P03_DGNM_without_{removed}','study':'01_primary_tpg_day60','sources':list(SOURCES),
            'representation':'security_tpg','structured_blocks':[s for s in ('cvss','reference_count') if s != removed],
            'edge_view':'no_coref','matched_cohort':'same full-source primary cohort','status':'registered_not_trained'})
    for name in ('persistence','training_only_lead_correction','daily_temporal_model','daily_temporal_plus_tpg'):
        rows.append({'id':'T01_'+name,'study':'02_continuous_epss','representation':name,
            'sources':list(SOURCES) if name.endswith('plus_tpg') else [],'structured_blocks':[],
            'status':'blocked_pending_historical_text_verification' if name.endswith('plus_tpg') else 'registered_not_trained',
            'anchors':'every eligible day 0-89','leads':'every eligible day 1..90-anchor',
            'matched_cohort':'same CVE/anchor/lead targets for every comparator'})
    require(len(rows) == len({r['id'] for r in rows}), 'Repeated experiment ID')
    return rows


def feature_schema():
    from .. import forecast_structured_features as cvss
    return {'primary':{
        'cvss':{'fields':list(cvss.CVSS_FIELDS),'encoding':'Fixed per-metric CVSS v3 one-hot groups; base score /10; explicit observed masks',
                'feature_names':cvss.FEATURES,'historical_availability':'unverified; retrospective only'},
        'reference_count':{'fields':['n_references_nvd'],'encoding':'log1p count, train-only standardization, missing mask; no row dropping',
                           'historical_availability':'unverified; retrospective only'},
        'source_presence':{'encoding':'Four binary masks; disabled sources have mask zero and cannot contribute embeddings or source-derived counts'},
        'multi_label_candidates':{'encoding':'Multi-hot with train-only vocabulary, missing flag and unseen-category flag',
            'status':'utility implemented, not enabled as a predictor block',
            'reason':'No independently sourced NVD CWE list is registered. VulnCheck-derived vckev_cwes is excluded. SMP platform features require a separate source-coupled ablation.'}},
        'excluded':{**{k:v for k,v in cvss.EXCLUDED.items() if k!='n_references_nvd'},'epss_scores':'Targets only in primary study; never primary inputs.',
            'cve':'Join/group identifier, not predictor.','URL_strings':'Provenance only; content is represented by the separately named summaries.'},
        'edge_types':'Graph message-passing relations from the saved vocabulary; never tabular CVSS columns.',
        'continuous':'Past EPSS values, missingness and observation ages only. Current CVSS and current text blocked as historical inputs.'}


def evaluation_contract():
    return {'regression':['MAE (primary)','RMSE','median absolute error','signed bias','R2 (undefined for constant truth)',
        'Spearman correlation (undefined for constant input)','90th/95th percentile absolute error','error by score range',
        'paired differences from each baseline','CVE-cluster bootstrap intervals with fixed resampling seed'],
        'continuous':['metrics by forecast lead','anchor-by-lead error and skill matrices','same-anchor matched comparisons',
            'CVE-balanced aggregate, not flat repeated-window average','missingness and model-version strata',
            'fixed example trajectories chosen without selecting the best prediction'],
        'threshold_diagnostics':{'thresholds':[.01,.1],'metrics':['confusion matrix','accuracy','balanced accuracy','precision','recall','F1','ROC-AUC','average precision'],
            'meaning':'EPSS threshold categories, NOT exploitation or KEV. Undefined denominators/classes return null with a reason.'},
        'figures':['actual-versus-predicted hexbin','residual histogram','residual versus actual score',
            'training and validation curves','source-ablation heatmap','paired effect plot with CVE-level intervals',
            'structured-block and edge-view ablation comparisons','lead-wise MAE and persistence skill',
            'anchor-by-lead coverage/error heatmaps','actual/predicted daily trajectories','threshold confusion matrices and ROC/PR curves'],
        'not_claimed':['exploitation probability calibration from EPSS targets','causal impact of SMP','feature importance from attention alone',
            'forecast uncertainty bands without a trained/validated interval method','F1 or accuracy as primary regression metrics'],
        'test':'Freeze protocol, variants, preprocessing, checkpoints and thresholds after validation; score held-out test once, report all frozen variants without retuning.'}
