"""Publish a readable, hash-linked RD02/RD04 gate review without fitting."""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.reddit_primary import data,cache
from epss.reddit_forecasting import data as daily


def run(verify=False):
    root=data.OUT/'gate_review_v1'
    bundles=[data.OUT/'export/full',data.OUT/'export/smoke',data.OUT/'smoke',
             cache.DEFAULT/'gate',cache.DEFAULT/'smoke',daily.OUTPUT/'preflight/full',daily.OUTPUT/'smoke']
    for b in bundles:daily.core.check_lock(b)
    primary=data.read(data.OUT/'smoke/summary.json');forecast=data.read(daily.OUTPUT/'smoke/summary.json')
    export=data.read(data.OUT/'export/full/summary.json');daily_counts=data.read(daily.OUTPUT/'preflight/full/summary.json')
    graph=data.read(cache.DEFAULT/'smoke/summary.json');gate=data.read(cache.DEFAULT/'gate/summary.json')
    parity=data.read(data.OUT/'smoke/wrapper_parity.json')
    data.require(all(v['max_absolute_difference']==0 for r in parity.values() for v in r.values()),'Wrapper parity failed')
    entries,index,_,_=cache.plan('full')
    hashes={str(b/'artifact_lock.json'):data.sha(b/'artifact_lock.json') for b in bundles}
    hashes.update({str(p):data.sha(p) for p in (Path(__file__),data.rd.LOCK/'rd02_role_ids.json',
                    data.rd.LOCK/'rd04_role_ids.json',data.REVIEW/'source_approval.json')})
    comparison=[]
    for name,r in primary['models'].items():
        comparison.append({'model':name,**r['all'],
                           'no_overlap_n':r['no_training_overlap']['n'],'no_overlap_mae':r['no_training_overlap']['mae'],
                           'overlap_n':r['training_overlap']['n'],'overlap_mae':r['training_overlap']['mae'],
                           'local_single_version_n':r['local_single_version']['n'],
                           'local_crossing_n':r['local_crossing_release']['n']})
    findings={'schema':'reddit_development_gate_review_v1','artifact_hashes':hashes,
        'rd02_full_export':export,'rd02_graph_gate':gate,'rd02_smoke_cache':graph,
        'rd02_smoke_metrics':comparison,'rd02_wrapper_parity':parity,
        'rd02_full_graph_plan':{'chunks':len(entries),'documents':sum(len(r['documents']) for r in index.values()),
            'cves':len(index),'built':False,'status':'WAITING_FOR_USER_DURABLE_STORAGE',
            'disk_guard_GiB':len(entries)*700000/1024**3},
        'rd04_preflight':daily_counts,'rd04_smoke_metrics':forecast['metrics'],
        'test_scoring':False,'full_training':False,
        'preservation':'Initial untrained exports with the column-order bug are preserved under export_before_column_order_fix; do not use them.',
        'next_gate':'Provide durable cache storage, verify full RD02 cache, then approve separate full-training runners. No full training is enabled here.'}
    lines=['RD02 and RD04 development gates','',
        'What is complete','The full RD02 predictor/label export and train-fitted structured cache are complete for 24,213 training and 972 validation CVEs.',
        'The 32-training-CVE graph gate and 12-train/6-validation RD02 smoke test are complete. All 21 smoke predictors have saved outputs.',
        'RD04 full preflight and its three-seed smoke outputs have been replayed without retraining.',
        'No test text was cached and no test scores were analyzed. Full model training remains disabled.','',
        'What is not complete',f'The full RD02 graph cache needs {len(entries):,} chunks for {len(index):,} CVEs and is waiting for durable storage.',
        'The user selected smoke-only cache construction rather than placing the full cache in temporary storage.','',
        'Structured inputs','There are 53 columns in the full training-fitted encoding: validated CVSS v3 base categories/version, score, three distinct reference counts, and observed-value flags.',
        'Categories are fitted on training only, with missing/unknown channels. Numeric columns use training medians and scales.',
        'A score with an invalid or missing vector is treated as unusable under this conservative v3 encoding. Optional CVSS metrics are not used.',
        'No EPSS value, catalogue field, catalogue missingness, CVE ID or date is a structured predictor. Current CVSS/text are retrospective, not verified Day-60 inputs.','',
        'How text enters the models','Each description and each accepted Reddit attachment remains its own document. Chunks are pooled within a document, then documents within D or M.',
        'D, DM_all and DM_prose are views over shared chunks; prose filtering changes masks/document selection, not the CVE cohort.',
        'Both controls use the same frozen SecBERT node embeddings. Pooled text averages projected embedded token states without message passing.',
        'The v6 graph model projects typed node features plus 32 subtype channels, then applies two EdgeTypeGNNLayer layers (hidden width 32), with no COREF or dropout.',
        'The graph mean feeds masked source attention, followed by a structured branch and scalar sigmoid regression output. This is regression, not KEV classification.',
        'Security edges are text-dependent rule-based candidates, not a learned relation extractor or independently verified facts.',
        'Subtype SEVERITY is index 31. Source spans, candidate provenance, saved relation vocabulary and tensor alignment are checked.','',
        'RD02 smoke results: six validation CVEs only',
        'Each row is a control or one fixed-seed model. MAE is mean absolute EPSS score error. Smaller is better. These are integration values, not full-cohort estimates.',
        'Model | MAE | RMSE | Validation CVEs']
    for r in comparison:lines.append(f'{r["model"]} | {r["mae"]:.8f} | {r["rmse"]:.8f} | {r["n"]}')
    lines += ['', 'Partition checks',
        'The frozen full validation split is 943 without detected training-description overlap and 29 with exact training-description overlap.',
        'The smoke subset contains four without overlap and two with overlap. Both subsets are reported for every predictor; they do not replace the full 943/29 evaluation.',
        'All six smoke validation CVEs are local single-version cases. Empty crossing/version-change slices have null metrics, not zero error.',
        'The whole 91-day changed flag is reported separately from a release strictly after publication and through Day 60. Neither flag enters the model.',
        'Same-weight D versus masked-M DM outputs match exactly for all three seeds and both representations. Shared weights after separate training need not match.',
        '', 'RD04 checks',
        'Training: 24,548 CVEs and 96,289,074 eligible pairs. Validation: 1,106 CVEs and 3,692,438 pairs.',
        'Training targets stop at 2023-12-31; validation anchors and targets stay in 2024. Targets are never imputed.',
        '5,448,048 training pairs cross a release between anchor and target; 10,004,810 cross between first observed input and target. These spans overlap and must not be added.',
        'Validation has no release crossings under the frozen schedule. Missing crossing-slice metrics are therefore expected.',
        'RD04 smoke has only three primary-slice validation CVEs. Persistence is exact on those three; the selected neural smoke checkpoints clip predictions to zero.',
        'This checks the pipeline, not forecasting superiority. No hyperparameters were retuned.',
        '', 'Implementation correction',
        'A JSON reload changed category iteration order in the first untrained RD02 exports. Explicit sorted column order and a round-trip test corrected it before any model fit.',
        'Those initial exports remain preserved but must not be used. No completed historical study was changed.',
        '', 'Artifact identities',data.dump(hashes),'',findings['next_gate']]
    with daily.core.transaction(root,verify) as stage:
        (stage/'findings.json').write_text(data.dump(findings));(stage/'REPORT.txt').write_text('\n'.join(lines)+'\n')
        with (stage/'rd02_smoke_metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(comparison[0]),lineterminator='\n');writer.writeheader();writer.writerows(comparison)
        daily.core.check_pins(hashes)
    print('Reddit development gate report '+('reproduced' if verify else 'saved')+'. No fitting or test scoring.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-only',action='store_true')
    run(parser.parse_args().verify_only)
