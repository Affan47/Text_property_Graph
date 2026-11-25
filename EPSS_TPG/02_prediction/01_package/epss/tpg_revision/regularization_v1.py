"""Global-decay-only comparison. Original fitter, inputs and controls stay frozen."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import multiprocessing
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
from tqdm import tqdm

from . import v5_training as original

data, ROOT = original.data, original.ROOT
CONFIG = ROOT/'02_prediction/04_configs/03_tpg_revision/02_regularization_v1.json'
CLI = ROOT/'02_prediction/03_analysis/04_tpg_revision/04_run_regularization.py'
_WORKER = None
DIAGNOSTIC_PREDECESSOR = 'ad9221ab2456880f6f5d53873dd18f4d58810e617e62d93988e381b73b7ef973'


def condition(family, decay):
    return family+'__wd_'+format(decay, '.4f').replace('.', 'p')


def comparison_identity(control, config):
    data.require(config['families']==['v5_node_mean','v5_subtypes_node_mean','v5_subtypes_type_balanced']
                 and config['seeds']==[17,29,43] and config['control_decay']==.0001
                 and config['new_decays']==[.001,.01], 'Unexpected comparison grid')
    data.require(control['settings']['weight_decay']==config['control_decay']
                 and control['settings']['seeds']==config['seeds'], 'Control settings differ')
    data.require({s:len(ids) for s,ids in control['ids'].items()}==config['expected_counts']
                 and not control['test_scoring'], 'Control cohort/role mismatch')
    pins = {**control['source_pins'], str(ROOT/config['control_root']/'artifact_lock.json'):
            data.sha(ROOT/config['control_root']/'artifact_lock.json')}
    for path in (CONFIG, CLI, Path(__file__)):
        pins[str(path)] = data.sha(path)
    return {'schema':config['schema'], 'config':config, 'control_identity':data.digest(control),
            'ids':control['ids'], 'source_pins':pins, 'test_scoring':False}


def fit_protocol(control, identity, family, decay):
    cfg = identity['config']
    data.require(family in cfg['families'] and decay in cfg['new_decays'], 'Unfrozen fit request')
    result = deepcopy(control)
    # JSON object keys are strings; the original loss indexes integer labels.
    result['class_weights'] = {int(k):v for k,v in control['class_weights'].items()}
    data.require(set(result['class_weights'])=={0,1}, 'Invalid class-weight keys')
    result['settings']['weight_decay'] = decay
    result.update(schema=cfg['schema'], regularization_changed=True, source_pins=identity['source_pins'],
                  regularization_condition={'family':family,'decay':decay,'comparison':data.digest(identity)})
    unchanged = dict(result['settings']); unchanged['weight_decay'] = control['settings']['weight_decay']
    data.require(unchanged==control['settings'], 'Only decay may change in fitter settings')
    return result


def context():
    ctx = original.context('full')
    config = data.read(CONFIG)
    root = ROOT/config['control_root']
    data.check_lock(root)
    saved = data.read(root/'run_identity.json')
    data.require(data.dump(saved)==data.dump(ctx[-1]), 'Original control identity changed')
    identity = comparison_identity(saved, config)
    data.check_pins(identity['source_pins'])
    return (*ctx[:-1], saved, identity)


def check_diagnostic_recovery(previous, current):
    source = str(Path(__file__))
    data.require(previous['source_pins'].get(source)==DIAGNOSTIC_PREDECESSOR,
                 'Unrecognized diagnostic predecessor')
    adjusted = deepcopy(previous)
    adjusted['source_pins'][source] = current['source_pins'][source]
    data.require(adjusted==current, 'Recovery would change settings, data or other code')


def joined_outputs(batches):
    # Preserve predict()'s sigmoid batch shape; concatenating first changes float32 rounding.
    return torch.cat(batches), np.array([v for batch in batches for v in batch.sigmoid().tolist()])


def diagnostics(directory, family, seed, ctx, protocol, rows):
    batches, targets, resources, dim = ctx[:4]
    p = protocol['settings']
    model = original.make(resources,dim,p,seed,family).eval()
    checkpoint = torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    cfg = ctx[-1]['config']
    ids = protocol['ids']['validation']
    logits = []
    with torch.no_grad():
        for start in range(0,len(ids),p['batch_size']):
            plan = batches.plan(ids[start:start+p['batch_size']])
            out = model.fuse(original.network.representations(model,batches,plan,'security_tpg'),plan)
            logits.append(out)
    logits, scores = joined_outputs(logits)
    data.require(bool(torch.isfinite(logits).all()) and np.allclose(scores,[r['predicted'] for r in rows],atol=1e-7,rtol=0),
                 'Diagnostic logits do not reproduce saved predictions')
    y = np.array([targets[c] for c in ids])
    params = {name:parameter.detach().double() for name,parameter in model.named_parameters()}
    subtype = params['encoder.subtype_project.weight']
    norm = float(subtype.norm())
    history = data.read(directory/'history.json')
    best = next(h for h in history if h['epoch']==checkpoint['best_epoch'])
    return {'checkpoint_sha256':data.sha(directory/'checkpoint.pt'), 'training_protocol_sha256':data.digest(protocol),
            'selected_epoch':checkpoint['best_epoch'],
            'epochs':len(history), 'selected_train_bce':best['train_weighted_bce'],
            'last_train_bce':history[-1]['train_weighted_bce'], 'subtype_frobenius_norm':norm,
            'subtype_rms_weight':float((subtype.square().mean()).sqrt()), 'subtype_max_abs_weight':float(subtype.abs().max()),
            'subtype_column_norms':dict(zip(protocol['subtype_schema'],subtype.norm(dim=0).tolist())),
            'all_parameter_norm':float(sum(p.square().sum() for p in params.values()).sqrt()),
            'logit_min':float(logits.min()), 'logit_max':float(logits.max()),
            'absolute_logit_above_10':int((logits.abs()>cfg['large_absolute_logit']).sum()),
            'scores_exactly_zero':int((scores==0).sum()), 'scores_exactly_one':int((scores==1).sum()),
            'scores_at_or_below_0p0001':int((scores<=cfg['saturation_probability_low']).sum()),
            'scores_at_or_above_0p9999':int((scores>=cfg['saturation_probability_high']).sum()),
            'confident_false_negatives_at_or_below_0p0001':int(((y==1)&(scores<=cfg['saturation_probability_low'])).sum()),
            'confident_false_positives_at_or_above_0p9999':int(((y==0)&(scores>=cfg['saturation_probability_high'])).sum())}


def worker_init():
    global _WORKER
    _WORKER = context()


def worker_fit(task):
    family, decay, seed, root, expected_identity = task
    ctx = _WORKER
    batches,targets,resources,dim,control,identity = ctx
    data.require(data.digest(identity)==expected_identity, 'Worker protocol mismatch')
    protocol = fit_protocol(control,identity,family,decay)
    key = condition(family,decay)+f'__seed_{seed}'
    root = Path(root); directory = root/'fits'/key
    data.require(not directory.exists(), 'Worker refuses existing fit')
    stage = Path(tempfile.mkdtemp(prefix='.incomplete-'+key+'-',dir=root/'fits'))
    try:
        with (root/'logs'/(key+'.log')).open('w', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
            rows, metrics = original.fit(stage,family,seed,batches,targets,resources,dim,protocol)
            diag = diagnostics(stage,family,seed,ctx,protocol,rows)
        data.require(not directory.exists(),'Fit appeared during execution')
        stage.rename(directory)
        return key, rows, metrics, diag
    except BaseException:
        if stage.exists():
            failed = root/'failed_fits';failed.mkdir(exist_ok=True)
            stage.rename(failed/stage.name)
        raise


def summarize(identity, metrics, diagnostic):
    cfg = identity['config']
    variants = [condition(f,d) for f in cfg['families'] for d in [cfg['control_decay'],*cfg['new_decays']]]
    means, coverage = original.aggregate_metrics(metrics, variants, cfg['seeds'])
    diagnostics_means = {c:{k:float(np.mean([diagnostic[c+f'__seed_{s}'][k] for s in cfg['seeds']]))
                           for k in ('subtype_frobenius_norm','subtype_rms_weight','absolute_logit_above_10',
                                     'confident_false_negatives_at_or_below_0p0001','confident_false_positives_at_or_above_0p9999')}
                         for c in variants}
    paired = {}
    for f in cfg['families']:
        for d in cfg['new_decays']:
            new, old = condition(f,d), condition(f,cfg['control_decay'])
            paired[new] = {str(s):{k:(None if metrics[new+f'__seed_{s}'][k] is None or metrics[old+f'__seed_{s}'][k] is None
                                        else metrics[new+f'__seed_{s}'][k]-metrics[old+f'__seed_{s}'][k])
                                   for k in original.METRIC_NAMES} for s in cfg['seeds']}
    return {'schema':cfg['schema'], 'means':means, 'defined_metric_seeds':coverage,
            'diagnostic_means':diagnostics_means, 'paired_new_minus_control_by_seed':paired,
            'reused_controls':9, 'new_fits':18, 'test_scoring':False,
            'interpretation':'Global weight decay comparison, not subtype-specific causal attribution; no automatic promotion.',
            'selection':'All checkpoints selected by unchanged best validation AP; diagnostics never select epochs or thresholds.'}


def report_text(identity, summary, metrics, diagnostic):
    cfg = identity['config']
    lines = ['# TRV03: fixed global AdamW regularization comparison','',
             'Same 2,870 training / 1,104 validation CVEs; seeds 17, 29, 43. Nine locked controls reused; eighteen new fits.',
             'No test scoring, graph/schema changes, threshold tuning or Brier-based checkpoint selection.',
             'Only optimizer weight_decay differs. All parameters retain the original AdamW grouping.', '',
             '## Three-seed means','',
             '| Family and decay | AP | ROC-AUC | F1 at 0.5 | Precision | Recall | Brier | Log loss | Subtype norm |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x):return 'undefined' if x is None else f'{x:.6f}'
    for c,m in summary['means'].items():
        lines.append('| '+c+' | '+' | '.join(fmt(m[k]) for k in ('average_precision','roc_auc','f1','precision','recall','brier','log_loss'))
                     +' | '+fmt(summary['diagnostic_means'][c]['subtype_frobenius_norm'])+' |')
    lines += ['', 'AP/ROC-AUC/F1/precision/recall are higher-is-better. Brier/log loss are lower-is-better.',
              'Norm is descriptive: smaller is not automatically better. Means are across seeds, not ensemble scores.',
              'Global decay affects every trained branch; this experiment cannot isolate a subtype-only regularization effect.',
              '', '## Every seed', '', '| Condition | Seed | AP | F1 | TP | FP | Brier | Log loss | Subtype norm | Selected epoch |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key,m in metrics.items():
        c,s = key.rsplit('__seed_',1);cm=m['confusion_matrix_true_rows_predicted_columns'];d=diagnostic[key]
        lines.append(f'| {c} | {s} | {fmt(m["average_precision"])} | {fmt(m["f1"])} | {cm[1][1]} | {cm[0][1]} | '
                     f'{fmt(m["brier"])} | {fmt(m["log_loss"])} | {fmt(d["subtype_frobenius_norm"])} | {d["selected_epoch"]} |')
    lines += ['', '## Fixed comparisons against 0.0001', '', '| Condition | AP improves in seeds | Log loss improves in seeds | Subtype norm decreases in seeds |', '|---|---|---|---|']
    for f in cfg['families']:
        for d in cfg['new_decays']:
            c=condition(f,d);base=condition(f,cfg['control_decay'])
            ap=[s for s in cfg['seeds'] if metrics[c+f'__seed_{s}']['average_precision']>metrics[base+f'__seed_{s}']['average_precision']]
            ll=[s for s in cfg['seeds'] if metrics[c+f'__seed_{s}']['log_loss']<metrics[base+f'__seed_{s}']['log_loss']]
            norms=[s for s in cfg['seeds'] if diagnostic[c+f'__seed_{s}']['subtype_frobenius_norm']<diagnostic[base+f'__seed_{s}']['subtype_frobenius_norm']]
            lines.append(f'| {c} | {ap} | {ll} | {norms} |')
    lines += ['', 'These are descriptive paired development comparisons, not significance tests or independent-test claims.',
              'Validation has already informed development. An observed best cell is not automatically a deployable winner.',
              'Per-CVE scores, all metrics, raw-logit saturation counts, subtype RMS/max weights and per-subtype column norms are saved alongside this report.',
              'Norm diagnostics use selected checkpoints; original controls do not contain per-epoch weight snapshots.',
              'No claim that a Frobenius norm near 5.9 alone demonstrates explosion. Its RMS over 992 entries is about 0.19.',
              'Source fixes and DG (description plus GitHub only) remain separate future experiments.']
    return '\n'.join(lines)+'\n'


def run(workers=6, check_only=False, verify=False, resume=False):
    from ..forecast_matched_multisource import writer_lock
    ctx = context()
    batches,targets,resources,dim,control,identity = ctx
    cfg = identity['config'];output=ROOT/cfg['output_root']
    controls = [(f,cfg['control_decay'],s) for f in cfg['families'] for s in cfg['seeds']]
    new = [(f,d,s) for f in cfg['families'] for d in cfg['new_decays'] for s in cfg['seeds']]
    print('[TRV03] 9 immutable controls + 18 new fits; only global AdamW decay changes; test excluded.',flush=True)
    metrics, prediction_rows, diagnostic, references = {}, [], {}, {}
    for family,decay,seed in tqdm(controls,desc='Replay nine locked 0.0001 controls',unit='fit'):
        directory=ROOT/cfg['control_root']/f'{family}__seed_{seed}'
        rows,m=original.verify_fit(directory,family,seed,batches,targets,resources,dim,control)
        key=condition(family,decay)+f'__seed_{seed}'
        metrics[key]=m;diagnostic[key]=diagnostics(directory,family,seed,ctx,control,rows)
        references[key]={'path':str(directory.relative_to(ROOT)),'artifact_lock_sha256':data.sha(directory/'artifact_lock.json')}
        prediction_rows.extend({**r,'condition':condition(family,decay),'weight_decay':decay,'reused_control':True} for r in rows)
    if check_only:
        for family,decay,seed in new:
            fit_protocol(control,identity,family,decay)
        print('TRV03 preflight passed. Nine controls reproduced; no training or test scoring.')
        return
    output.parent.mkdir(parents=True,exist_ok=True)
    with writer_lock(output.parent/'.regularization-writer.lock'):
        legacy = None
        if output.exists():
            data.require(verify or resume,'Output exists; use --verify-only or --resume')
            previous = data.read(output/'run_identity.json')
            if previous != identity:
                data.require(resume and not (output/'artifact_lock.json').exists(),
                             'Use explicit resume for the known incomplete diagnostic repair')
                check_diagnostic_recovery(previous,identity)
                (output/'pre_diagnostic_repair_identity.json').write_text(data.dump(previous))
                (output/'run_identity.json').write_text(data.dump(identity))
            if (output/'pre_diagnostic_repair_identity.json').exists():
                legacy = data.read(output/'pre_diagnostic_repair_identity.json')
                check_diagnostic_recovery(legacy,identity)
                data.require(data.sha(output.parent/'regularization_before_diagnostic_fix.py')==DIAGNOSTIC_PREDECESSOR,
                             'Archived diagnostic implementation changed')
            if verify or (output/'artifact_lock.json').exists():data.check_lock(output)
        else:
            data.require(not verify,'No comparison to verify')
            output.mkdir();(output/'fits').mkdir();(output/'logs').mkdir()
            (output/'run_identity.json').write_text(data.dump(identity))
            (output/'execution.json').write_text(data.dump({'workers':workers,'torch_threads_per_worker':control['settings']['threads']}))
        pending=[]
        for family,decay,seed in tqdm(new,desc='Verify completed new fits or queue missing fits',unit='fit'):
            key=condition(family,decay)+f'__seed_{seed}';directory=output/'fits'/key
            if legacy and not directory.exists() and resume:
                copies = list((output.parent/'post_fit_recovery_copies').glob('.incomplete-'+key+'-*'))
                data.require(len(copies)<=1, 'Ambiguous preserved worker copies')
                if copies:
                    old_protocol=fit_protocol(control,legacy,family,decay)
                    original.verify_fit(copies[0],family,seed,batches,targets,resources,dim,old_protocol)
                    shutil.copytree(copies[0],directory)
            if directory.exists():
                protocol=fit_protocol(control,identity,family,decay)
                if legacy:
                    saved=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)
                    old_protocol=fit_protocol(control,legacy,family,decay)
                    if saved['identity']==data.digest(old_protocol):protocol=old_protocol
                rows,m=original.verify_fit(directory,family,seed,batches,targets,resources,dim,protocol)
                metrics[key]=m;diagnostic[key]=diagnostics(directory,family,seed,ctx,protocol,rows)
                prediction_rows.extend({**r,'condition':condition(family,decay),'weight_decay':decay,'reused_control':False} for r in rows)
            else:
                data.require(not verify,'Missing fit in verification mode')
                log=output/'logs'/(key+'.log')
                if legacy and log.exists():
                    archive=output/'initial_attempt_logs';archive.mkdir(exist_ok=True)
                    data.require(not (archive/log.name).exists(),'Initial log already archived')
                    log.rename(archive/log.name)
                pending.append((family,decay,seed,str(output),data.digest(identity)))
        if pending:
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=worker_init) as pool:
                futures={pool.submit(worker_fit,task):task for task in pending}
                for future in tqdm(as_completed(futures),total=len(futures),desc='Train new decay fits',unit='fit'):
                    key,rows,m,diag=future.result();family,decay,seed,*_=futures[future]
                    metrics[key]=m;diagnostic[key]=diag
                    prediction_rows.extend({**r,'condition':condition(family,decay),'weight_decay':decay,'reused_control':False} for r in rows)
                    print(f'[TRV03 DONE] {key}: AP={m["average_precision"]:.6f}, log_loss={m["log_loss"]:.6f}, subtype_norm={diag["subtype_frobenius_norm"]:.6f}',flush=True)
        metrics=dict(sorted(metrics.items()));diagnostic=dict(sorted(diagnostic.items()))
        prediction_rows.sort(key=lambda r:(r['condition'],r['seed'],r['cve']))
        data.require(len(metrics)==27 and len(prediction_rows)==27*len(control['ids']['validation']), 'Incomplete comparison')
        summary=summarize(identity,metrics,diagnostic);text=report_text(identity,summary,metrics,diagnostic)
        artifacts={'summary.json':data.dump(summary),'summary.txt':text,'REPORT.md':text,
                   'metrics.json':data.dump(metrics),'diagnostics.json':data.dump(diagnostic),'references.json':data.dump(references),
                   'predictions.json':data.dump(prediction_rows),'predictions.csv':original.base.development.csv_text(prediction_rows),
                   'model_means.csv':original.base.development.csv_text([{'condition':c,**m,**summary['diagnostic_means'][c]} for c,m in summary['means'].items()])}
        locked=verify or (output/'artifact_lock.json').exists()
        for name,value in artifacts.items():
            if locked:data.require((output/name).read_text()==value,'Report differs: '+name)
            else:(output/name).write_text(value)
        if not locked:original.base.lock_tree(output)
        data.check_lock(ROOT/cfg['control_root']);data.check_pins(identity['source_pins'])
    print('TRV03 full '+('verified.' if verify else f'saved to {output}.')+' Nine original controls unchanged; no test scoring.')
