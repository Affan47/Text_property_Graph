"""Bounded RD04 smoke fits and replay; full-development training is not enabled."""
from copy import deepcopy
import shutil
import numpy as np
import torch
from tqdm import tqdm
from . import data, evaluation
from ..continuous_forecasting import controls, runner as base


def fit(seed, rows, identity):
    p = identity['settings']
    model = base.model_for(seed, p)
    epoch0 = evaluation.evaluate(rows['validation'], p['anchor_packet'], model=model)
    persistence = evaluation.evaluate(rows['validation'], p['anchor_packet'])
    data.require(epoch0 == persistence, 'Initialization is not exact persistence')
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    history = []; best = float('inf'); stale = 0; state = None; best_epoch = 0
    for epoch in tqdm(range(1,p['smoke']['epochs']+1), desc=f'RD04 smoke seed {seed}', unit='epoch'):
        loss, grad = base.train_epoch(model, rows['train'], optimizer, seed, epoch, p)
        metrics = evaluation.evaluate(rows['validation'], p['anchor_packet'], model=model)['summary']
        score = metrics['day30_to_day60']['mae']
        data.require(score is not None and np.isfinite(score), 'Smoke primary slice is empty/nonfinite')
        if score < best:
            best, state, best_epoch, stale = score, deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        history.append({'epoch': epoch, 'train_raw_mae': loss, 'gradient_norm_sum': grad,
                        'validation': metrics, 'best_epoch': best_epoch, 'patience_used': stale})
        tqdm.write(f'[RD04] seed={seed} epoch={epoch} train_MAE={loss:.9f} primary_MAE={score:.9f} '
                   f'clipped_fraction={metrics["all"]["clipping_rate"]:.6f} bias={metrics["all"]["bias"]:.9f}')
        if stale >= p['smoke']['patience']:
            break
    return {'state_dict': state, 'seed': seed, 'identity': data.digest(identity), 'best_epoch': best_epoch}, history, epoch0


def smoke(verify=False):
    rows, identity = data.load('smoke'); p = identity['settings']; output = data.OUTPUT/'smoke'
    data.require(all(len(rows[s]) == p['smoke'][s+'_cves'] for s in rows), 'Smoke size mismatch')
    if verify:
        data.core.check_lock(output)
        data.require(data.read(output/'run_identity.json') == identity, 'Saved identity differs')
    correction = controls.fit(rows['train'])
    with data.core.transaction(output, verify) as stage:
        (stage/'run_identity.json').write_text(data.dump(identity))
        (stage/'lead_correction.json').write_text(data.dump(correction))
        results = {}
        for name, control in [('persistence',None),('lead_correction',correction)]:
            results[name] = evaluation.evaluate(rows['validation'],p['anchor_packet'],correction=control,
                                                prediction_file=stage/f'{name}_predictions.csv')
        for seed in tqdm(p['seeds'], desc='RD04 fixed smoke checkpoint gates', unit='seed'):
            name = f'seed_{seed}'
            initial = base.model_for(seed,p)
            epoch0 = evaluation.evaluate(rows['validation'],p['anchor_packet'],model=initial)
            data.require(epoch0 == results['persistence'], 'Epoch zero differs from persistence')
            if verify:
                shutil.copyfile(output/f'{name}.pt',stage/f'{name}.pt')
                history = data.read(output/f'{name}_history.json')
            else:
                checkpoint, history, checked0 = fit(seed,rows,identity)
                data.require(checked0 == epoch0,'Initialization replay differs')
                torch.save(checkpoint,stage/f'{name}.pt')
            checkpoint = torch.load(stage/f'{name}.pt',map_location='cpu',weights_only=True)
            data.require(checkpoint['identity']==data.digest(identity) and checkpoint['seed']==seed,'Checkpoint identity differs')
            best = min(history,key=lambda r:r['validation']['day30_to_day60']['mae'])
            data.require(best['epoch']==checkpoint['best_epoch'],'Checkpoint selection differs')
            model = base.model_for(seed,p); model.load_state_dict(checkpoint['state_dict'],strict=True)
            results[name] = evaluation.evaluate(rows['validation'],p['anchor_packet'],model=model,
                                                prediction_file=stage/f'{name}_predictions.csv')
            data.require(results[name]['summary']==best['validation'],'Checkpoint predictions do not reproduce')
            (stage/f'{name}_history.json').write_text(data.dump(history))
            (stage/f'{name}_epoch0.json').write_text(data.dump(epoch0))
        reference = results['persistence']
        for name,result in results.items():
            for s in evaluation.SLICES:
                data.require(result['summary'][s]['cells']==reference['summary'][s]['cells'],'Unmatched cells')
            (stage/f'{name}_evaluation.json').write_text(data.dump(result))
        (stage/'intervals.json').write_text(data.dump(evaluation.intervals(results,p['bootstrap_repetitions'],p['bootstrap_seed'])))
        summary = {'integration_only':True,'test_scoring':False,'full_training':False,
                   'cves':{s:len(rs) for s,rs in rows.items()},
                   'metrics':{k:v['summary'] for k,v in results.items()},
                   'release_note':'Release dates are diagnostic labels, never predictors. Empty crossing slices are not zero error.',
                   'weighting':p['weighting'],'selection':p['selection']}
        (stage/'summary.json').write_text(data.dump(summary))
        (stage/'REPORT.txt').write_text('RD04 smoke integration, not a performance claim\n'+data.dump(summary)+
            '\nAll three checkpoints reloaded. Raw and clipped predictions retained; canonical CVE/anchor/lead CSV order.\n'
            'Validation is in 2024; zero release-crossing validation pairs are expected under the frozen schedule.\n'
            'Training release coverage is in the full preflight; no test CVE is adapted.\n')
        data.core.check_pins(identity['source_pins'])
    print('RD04 smoke '+('reproduced; no retraining.' if verify else 'saved; bounded fits only.')+' No full training or test scoring.')
