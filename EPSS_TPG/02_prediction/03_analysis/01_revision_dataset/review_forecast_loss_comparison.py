"""Interpret locked R06 results; no fitting or new model predictions."""
import argparse
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_loss_comparison as run

SOURCE = run.OUTPUT
OUTPUT = SOURCE.with_name(SOURCE.name + '_review')


def analyze():
    lock = run.read(SOURCE/'evaluation_lock.json')
    run.require(set(lock['files']) == run.expected_files(run.protocol()), 'Incomplete R06 output')
    hashes = {str(SOURCE/n): h for n,h in lock['files'].items()}
    run.r05.verify_hashes(hashes, 'Verify R06 artifacts')
    identity = run.read(SOURCE/'run_identity.json')
    run.r05.verify_hashes(identity['pinned_files'], 'Verify frozen sources')
    s = run.read(SOURCE/'summary.json')
    choice = run.decision(s['fits'],identity['protocol'])
    run.require(choice == run.read(SOURCE/'full_training_decision.json') ==
                s['full_training_decision'], 'Selection decision differs')
    selected = identity['ids']['validation']
    labels = run.read(run.prior.cache.base.PREPARED/'labels.json')
    actual = [labels[c]['y']['endpoint'] for c in selected]
    base = s['baselines']['persistence']['predictions']
    base_errors = [abs(y-p) for y,p in zip(actual,base)]
    baseline = statistics.mean(base_errors)
    run.require(abs(baseline-s['baselines']['persistence']['metrics']['mae']) < 1e-14,
                'Persistence MAE differs')
    top_index = max(range(len(selected)),key=base_errors.__getitem__)
    fits = {}
    for name,m in sorted(s['fits'].items()):
        directory = SOURCE/'fits'/name
        pred = run.read(directory/'predictions.json')
        history = run.read(directory/'history.json')
        gradients = run.read(directory/'gradients.json')
        errors = [abs(y-p) for y,p in zip(actual,pred['prediction'])]
        run.require(len(errors) == len(selected) and
                    abs(statistics.mean(errors)-m['metrics']['mae']) < 1e-14, 'Saved MAE differs')
        run.require(len(history) == 30 and
                    m['selected_epoch'] == min(history,key=lambda h:h['validation_mae'])['epoch'],
                    'Fixed budget/checkpoint differs')
        initial_norms, final_norms = m['initial_parameter_norms'],m['selected_parameter_norms']
        fits[name] = {
            'metrics': m, 'mae_excess_over_persistence': m['metrics']['mae']-baseline,
            'improved_vs_persistence': sum(a < b for a,b in zip(errors,base_errors)),
            'worsened_vs_persistence': sum(a > b for a,b in zip(errors,base_errors)),
            'equal_error_vs_persistence': sum(a == b for a,b in zip(errors,base_errors)),
            'selected_to_initial_norm_ratios': {k:final_norms[k]/v if v else None for k,v in initial_norms.items()},
            'clipped_training_evaluations': sum(h['training_clipped_below_zero']+
                                                h['training_clipped_above_one'] for h in history),
            'history': history, 'gradients': gradients,
            'validation_cases': [{'cve':c,'actual_day60':y,'persistence':b,'prediction':p,
                                  'correction':d,'absolute_error':e}
                                 for c,y,b,p,d,e in zip(selected,actual,base,pred['prediction'],pred['correction'],errors)],
        }
    mae_rows = [v for k,v in fits.items() if '__mae__' in k]
    means = choice['mean_validation_mae_across_models_and_seeds']
    result = {
        'schema': 'r06_loss_comparison_review_v1',
        'source_lock_sha256': run.sha256(SOURCE/'evaluation_lock.json'),
        'review_script_sha256': run.sha256(Path(__file__)),
        'cohort_sizes': {k:len(v) for k,v in identity['ids'].items()},
        'training_performed': False, 'new_predictions': False, 'test_scoring': False,
        'fits':fits,'paired':s['paired'],'aggregates':s['aggregates'],
        'baselines':s['baselines'],'decision':choice,
        'mae_improved_pairs':sum(r['mae_loss_minus_smooth_l1_mae']<0 for r in s['paired']),
        'relative_macro_mae_reduction':1-means['mae']/means['smooth_l1'],
        'all_learned_fits_worse_than_persistence_mae':all(v['mae_excess_over_persistence']>0 for v in fits.values()),
        'mae_selected_epochs':{k:v['metrics']['selected_epoch'] for k,v in fits.items() if '__mae__' in k},
        'mae_mean_absolute_correction_range':[
            min(v['metrics']['clipping']['mean_absolute_correction'] for v in mae_rows),
            max(v['metrics']['clipping']['mean_absolute_correction'] for v in mae_rows)],
        'mae_clipped_training_evaluations':sum(v['clipped_training_evaluations'] for v in mae_rows),
        'unchanged_validation_endpoints':sum(y==p for y,p in zip(actual,base)),
        'largest_persistence_error_case':{
            'cve':selected[top_index], 'persistence':base[top_index], 'actual_day60':actual[top_index],
            'fraction_of_total_absolute_error':base_errors[top_index]/sum(base_errors),
            'mae_model_predictions':{k:v['validation_cases'][top_index]['prediction']
                                     for k,v in fits.items() if '__mae__' in k}},
        'limitations':s['limitations']+[
            'MAE and Smooth L1 checkpoints were selected at different epochs; norm differences are not a matched-step causal comparison.',
            'Very small correction differences do not establish useful TPG contribution.',
            'Only selected checkpoint weights were saved; final-epoch norms are not recoverable.',
        ],
    }
    run.r05.verify_hashes(hashes, 'Confirm original R06 outputs unchanged')
    return result


def report(a):
    lines = [
        'R06: Final bounded comparison and full-training handoff', '',
        '1. Decision',
        f"The preregistered rule selects {a['decision']['selected_loss']} loss for both model types.",
        f"MAE loss improves validation MAE in {a['mae_improved_pairs']}/6 model/seed pairs.",
        f"Mean validation MAE across both models and all seeds decreases by {100*a['relative_macro_mae_reduction']:.3f}%.",
        'The next stage is full eligible development training, not another bounded tuning round.',
        'No full training was started by this review.', '',
        '2. Verification',
        'The R06 verifier independently reproduced all 12 checkpoints and predictions, all 80 artifact hashes,',
        'and exact R05 original-decay parity for all six Smooth L1 controls.',
        'This review recomputes the decision, case errors and history summaries from saved outputs.',
        'No retraining, new forecasts, new embeddings, split changes or test scoring.', '',
        '3. Validation results',
        'Model | Training loss | Mean validation MAE across three seeds | Sample SD',
    ]
    for r in a['aggregates']:
        lines.append(f"{r['model']} | {r['loss']} | {r['mean_mae']:.11f} | {r['sample_std_mae']:.11f}")
    lines += [
        f"Persistence and the zero-offset scalar reference: MAE={a['baselines']['persistence']['metrics']['mae']:.11f}.",
        f"All 12 learned fits still have higher MAE than persistence: {a['all_learned_fits_worse_than_persistence_mae']}.",
        'Selecting MAE is a choice between training objectives, not evidence that a learned model beats persistence.',
        'The tiny difference between graph and EPSS-only results is not a demonstrated useful TPG contribution.', '',
        '4. What actually improved',
        'Smooth L1 models generally made small positive corrections that hurt unchanged targets.',
        'MAE training reduces those corrections nearly to zero: predictions are now very close to persistence.',
        f"The mean absolute correction ranges from {a['mae_mean_absolute_correction_range'][0]:.3g} "
        f"to {a['mae_mean_absolute_correction_range'][1]:.3g} across the six MAE models.",
        f"{a['unchanged_validation_endpoints']}/64 validation endpoints are unchanged from the last observed input score.",
        'Reducing an unnecessary correction helps those cases, but does not show forecasting of large increases.', '',
        'Model/seed | Selected MAE epoch | Validation MAE | Excess MAE over persistence',
    ]
    for k,r in a['fits'].items():
        if '__mae__' in k:
            lines.append(f"{k} | {r['metrics']['selected_epoch']} | {r['metrics']['metrics']['mae']:.11f} | "
                         f"{r['mae_excess_over_persistence']:.5g}")
    top = a['largest_persistence_error_case']
    lines += ['', '5. Large-change example',
              f"{top['cve']}: persistence={top['persistence']:.5f}, actual Day-60 score={top['actual_day60']:.5f}.",
              f"This one CVE accounts for {100*top['fraction_of_total_absolute_error']:.2f}% of baseline absolute error.",
              'The MAE models remain near 0.00210 and do not recover this large increase. The case is retained.']
    for k,v in top['mae_model_predictions'].items():
        lines.append(f"{k}: {v:.12f}")
    lines += ['', '6. Training and encoder behavior',
              'All fits ran 30 epochs, or 480 optimizer steps. MAE checkpoints occur later than Smooth L1 checkpoints.',
              f"Recorded MAE-training output-clipping evaluations: {a['mae_clipped_training_evaluations']}.",
              'MAE-selected validation predictions also have no output clipping.',
              'Training MAE remains near the training persistence value of 0.00639.',
              'Encoder gradients are nonzero in 478 or 479 of 480 steps; branches are not entirely disconnected.',
              'Original weight decay continues to shrink encoder weights. At the selected MAE graph checkpoints,',
              'the input projection retains roughly 0.011-0.834% of its initial norm.',
              'These checkpoints occur later, so this is not a same-step causal comparison with Smooth L1.',
              'Small learned corrections and preserved gradient paths do not establish useful predictive signal.', '',
              '7. Full development training next',
              'Use MAE for both EPSS-only and no-COREF TPG+EPSS models.',
              'Use the unchanged 1023 training / 197 validation CVEs, seeds 17/29/43, fresh initialization,',
              'Adam weight decay 0.0001, gain 0.00885, up to 100 epochs and patience 15.',
              'Do not warm-start from R05/R06 or open the held-out test set.',
              'Retain persistence and a training-fitted scalar reference in the evaluation.',
              'The full-training runner is the next implementation stage; the user will run it. No further bounded tuning is planned.',
              'The full experiment may or may not beat persistence. Report that result without changing the split.',
              '', '8. Limitations']+['- '+v for v in a['limitations']]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    args = parser.parse_args()
    run.require(args.verify_only or not OUTPUT.exists(),'Review exists; use --verify-only')
    result = analyze()
    text = report(result)
    if args.verify_only:
        run.require(run.read(OUTPUT/'analysis.json')==result and (OUTPUT/'analysis.txt').read_text()==text,
                    'Review reproduction differs')
        print('R06 review and full-training decision reproduced. No training or test scoring.')
        return
    OUTPUT.mkdir(parents=True,exist_ok=False)
    run.write(OUTPUT/'analysis.json',result)
    (OUTPUT/'analysis.txt').write_text(text,encoding='utf-8')
    print(f'R06 review saved to {OUTPUT}. MAE selected for the full-development handoff; no training or test scoring.')


if __name__ == '__main__':
    main()
