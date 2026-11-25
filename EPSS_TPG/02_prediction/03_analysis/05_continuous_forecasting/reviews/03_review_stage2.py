"""Review frozen CF01 Stage 2 metrics; no training, predictions or test access."""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / '01_package'))
from epss.continuous_forecasting import data, evaluation

SOURCE = data.OUTPUT.parent / 'stage2_v1/full'
OUTPUT = data.OUTPUT.parent / 'stage2_review_v1'


def run(verify=False):
    for name in ('controls', 'seed_17', 'seed_29', 'seed_43', 'report'):
        data.check_lock(SOURCE / name)
    identity = data.read(SOURCE / 'report/run_identity.json')
    data.check_pins(identity['source_pins'])
    baseline = data.read(SOURCE / 'controls/persistence_evaluation.json')
    intervals = data.read(SOURCE / 'report/intervals.json')
    results, fits = {}, {}
    for seed in (17, 29, 43):
        result = data.read(SOURCE / f'seed_{seed}/evaluation.json')
        history = data.read(SOURCE / f'seed_{seed}/history.json')
        best = min(history[1:], key=lambda h: h['validation']['day30_to_day60']['mae'])
        data.require(history[0]['epoch'] == 0 and history[0]['validation'] == baseline['summary'], 'Epoch-zero parity differs')
        data.require(best['validation'] == result['summary'], 'Selected history differs')
        data.require([r['cve'] for r in result['per_cve']] == [r['cve'] for r in baseline['per_cve']], 'Unmatched CVEs')
        results[seed] = result
        fits[str(seed)] = dict(stopped_epoch=history[-1]['epoch'], selected_epoch=best['epoch'],
                              updates=history[-1]['optimizer_updates'],
                              maximum_epoch_clipped_fraction=max(h['validation']['all']['clipping_rate'] for h in history),
                              selected_clipped_fraction=result['summary']['all']['clipping_rate'],
                              slices={})
        for group in evaluation.SLICES:
            deltas = np.array([r['metrics'][group]['mae']-b['metrics'][group]['mae']
                               for r, b in zip(result['per_cve'], baseline['per_cve'])])
            corrections = np.array([r['metrics'][group]['bias']-b['metrics'][group]['bias']
                                    for r, b in zip(result['per_cve'], baseline['per_cve'])])
            fits[str(seed)]['slices'][group] = dict(
                improved_cves=int((deltas < 0).sum()), worsened_cves=int((deltas > 0).sum()), tied_cves=int((deltas == 0).sum()),
                mean_prediction_change=float(corrections.mean()), minimum_prediction_change=float(corrections.min()),
                maximum_prediction_change=float(corrections.max()), std_prediction_change=float(corrections.std()),
                interpretation='Exact per-CVE prediction change for primary slice; hierarchical within-CVE mean change for other slices')
    slices = {}
    for group in evaluation.SLICES:
        b = baseline['summary'][group]
        mean = float(np.mean([r['summary'][group]['mae'] for r in results.values()]))
        interval = next(r for r in intervals if r['comparison'] == 'neural_mean_minus_persistence' and r['slice'] == group)
        slices[group] = dict(cves=b['cves'], cells=b['cells'], persistence_mae=b['mae'], mean_neural_mae=mean,
                             neural_minus_persistence=mean-b['mae'], relative_error_change_percent=100*(mean/b['mae']-1),
                             paired_interval=interval['metrics']['mae'], rmse_difference_interval=interval['metrics']['rmse'])
    correction = data.read(SOURCE / 'controls/lead_correction.json')
    nonzero = [{'lead': i+1, 'correction': v, 'validation_cells': int(sum(row[i] for row in baseline['matrix']['count']))}
               for i, v in enumerate(correction['corrections']) if v != 0]
    review = dict(schema='cf01_stage2_review_v1', counts=identity['counts'], fits=fits, slices=slices,
                  nonzero_training_lead_corrections=nonzero,
                  checkpoint_hashes=data.read(SOURCE / 'report/summary.json')['checkpoints'],
                  original_results_modified=False, training=False, test_scoring=False,
                  source_locks={name: data.sha(SOURCE / name / 'artifact_lock.json')
                                for name in ('controls', 'seed_17', 'seed_29', 'seed_43', 'report')},
                  review_script_sha256=data.sha(Path(__file__)))
    lines = ['CF01 Stage 2: completed daily-updated EPSS comparison', '',
             '1. Scope and verification',
             '2,563 training CVEs; 1,104 validation CVEs; 9,576,458 training and 3,067,429 validation cells.',
             'Each cell is one CVE, a forecast day, and a later observed target day. Missing targets were not filled.',
             'All five predictors use the same cells. This is EPSS-only regression, not TPG or KEV classification.',
             'Original output hashes, checkpoint hashes and selected-epoch metrics were checked. No results were changed.',
             '', '2. Main results',
             'Slice | Persistence MAE | Three-seed mean MAE | Change in error (%) | Paired 95% interval for MAE difference']
    for group, r in slices.items():
        ci = r['paired_interval']
        lines.append(f'{group} | {r["persistence_mae"]:.9f} | {r["mean_neural_mae"]:.9f} | '
                     f'{r["relative_error_change_percent"]:+.4f}% | [{ci["lower_2_5"]:+.9f}, {ci["upper_97_5"]:+.9f}]')
    lines += ['', 'Negative error change means improvement. MAE is absolute error on the 0-to-1 EPSS scale.',
              'all includes all eligible anchors/leads; day30_to_day60 is one forecast per validation CVE.',
              'lead7/14/30 include every eligible anchor forecasting that many days ahead, not just Day 30.',
              'The neural mean averages seed metrics, not predictions. Intervals resample 1,104 whole CVEs 1,000 times.',
              '', 'All three seeds improve the declared Day-30 to Day-60 slice slightly, but all three worsen complete-matrix MAE.',
              'Seven- and fourteen-day lead MAE worsens; thirty-day lead MAE improves slightly.',
              'RMSE improves slightly in all slices. Thus conclusions depend on metric and horizon, not just one winning table.',
              'The saved intervals exclude zero in these directions, conditional on the selected checkpoints.',
              'They do not account for checkpoint selection on this validation set or its repeated use in earlier studies.',
              'This is not an independent generalization result or a broad daily-forecasting improvement.',
              '', '3. Training and clipping', 'Seed | Stopped epoch | Selected epoch | Updates | Maximum epoch clipping | Selected clipping']
    for seed, r in fits.items():
        lines.append(f'{seed} | {r["stopped_epoch"]} | {r["selected_epoch"]} | {r["updates"]} | '
                     f'{r["maximum_epoch_clipped_fraction"]:.4%} | {r["selected_clipped_fraction"]:.4%}')
    lines += ['', 'Epoch 0 exactly matches persistence and is excluded from trained-epoch selection.',
              'All fits stopped after 15 non-improving epochs. None reached the 100-epoch cap.',
              'Selected predictions need no clipping. The all-zero smoke result did not persist in this full run.',
              '', '4. What was learned?',
              'The lead-median control has one nonzero value: lead 90, +0.000020. That lead has no validation cells.',
              'Therefore its validation forecasts equal persistence. This is the fitted control result, not a failed fit.',
              'At Day 30 to Day 60, every neural prediction is slightly above persistence.',
              'Seed | Mean upward correction | Standard deviation across CVEs | CVEs with smaller/larger primary error']
    for seed, r in fits.items():
        x = r['slices']['day30_to_day60']
        lines.append(f'{seed} | {x["mean_prediction_change"]:.9f} | {x["std_prediction_change"]:.9f} | '
                     f'{x["improved_cves"]}/{x["worsened_cves"]}')
    lines += ['', 'These small, relatively similar corrections are consistent with a modest upward adjustment.',
              'They do not prove that the model learned rich CVE-specific trajectories, nor that every correction is constant.',
              'The training-only median control optimizes all anchors for each lead; the neural checkpoint is selected on Day30->60.',
              'That objective difference matters when interpreting the primary-slice gain over the control.',
              '', '5. Decision',
              'Keep this as the completed, fixed EPSS-only comparison. Do not promote a seed or retune from these results.',
              'Do not describe it as a TPG result: no text or graph input was used.',
              'Before further training or text fusion, a bounded frozen-checkpoint history-reliance check could test',
              'whether past trajectory information changes predictions beyond the latest score, age and forecast lead.',
              'Such interventions need a separately declared design; none was run in this review.',
              'Test records remain untouched by this study. The previously exposed test cohort is not a fresh holdout.']
    with data.transaction(OUTPUT, verify) as stage:
        (stage / 'review.json').write_text(data.dump(review))
        (stage / 'review.txt').write_text('\n'.join(lines)+'\n')
    print('CF01 Stage 2 review ' + ('reproduced.' if verify else 'saved to ' + str(OUTPUT)) + ' No training or scoring.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    run(parser.parse_args().verify_only)
