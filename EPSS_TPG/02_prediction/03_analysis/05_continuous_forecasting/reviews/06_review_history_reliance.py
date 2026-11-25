"""Interpret saved Stage 3 diagnostics; no new predictions, training or test access."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'01_package'))
from epss.continuous_forecasting.stage3 import diagnostic as d

OUTPUT = d.data.OUTPUT.parent/'stage3_review_v1'


def run(verify=False):
    source = d.OUTPUT/'full'
    for name in ('report', *(f'seed_{s}' for s in d.SEEDS)):
        d.data.check_lock(source/name)
    summary = d.data.read(source/'report/summary.json')
    d.data.check_pins(summary['identity']['source_pins'])
    intervals = d.data.read(source/'report/paired_intervals.json')
    mean_metrics, shares = {}, {}
    for group in d.ev.SLICES:
        mean_metrics[group] = {v: {k: float(np.mean([summary['metrics'][str(s)][v][group][k] for s in d.SEEDS]))
                                   for k in ('mae','rmse')} for v in d.VARIANTS}
        shares[group] = {}
        for metric in ('mae','rmse'):
            gain = mean_metrics[group]['persistence'][metric]-mean_metrics[group]['intact'][metric]
            constant_gain = mean_metrics[group]['persistence'][metric]-mean_metrics[group]['constant_shift_control'][metric]
            shares[group][metric] = dict(intact_gain=gain, constant_gain=constant_gain,
                constant_share_percent=100*constant_gain/gain if gain>0 else None,
                note='Ratio of mean gains, not causal attribution; omitted when intact has no gain')
    counts = {g: Counter() for g in ('lead7','day30_to_day60')}
    with (d.SOURCE/'controls/persistence_predictions.csv').open(newline='') as stream:
        for r in csv.DictReader(stream):
            groups = []
            if r['lead_days']=='7':groups.append('lead7')
            if r['anchor_day']=='30' and r['target_day']=='60':groups.append('day30_to_day60')
            if groups:
                change = float(r['actual_epss'])-float(r['raw_prediction'])
                for group in groups:counts[group]['higher' if change>0 else 'lower' if change<0 else 'unchanged']+=1
    findings = dict(schema='cf01_history_reliance_review_v1', mean_metrics=mean_metrics, gain_decomposition=shares,
                    observed_changes={k:dict(v) for k,v in counts.items()},
                    mean_seed_paired_intervals=[r for r in intervals if r['seed']=='seed_mean'],
                    source_report_lock=d.data.sha(source/'report/artifact_lock.json'),
                    stage2_control_lock=d.data.sha(d.SOURCE/'controls/artifact_lock.json'),
                    review_script_sha256=d.data.sha(Path(__file__)),
                    prediction_differences=summary['prediction_differences'],
                    training=False, new_predictions=False, test_scoring=False)
    lines = ['CF01 Stage 3: findings from the frozen history diagnostic', '',
             '1. Main finding',
             'These frozen GRUs barely respond to replacing earlier observed scores with the latest score.',
             'Their small primary gain is almost entirely reproduced by a fixed upward adjustment.',
             'This is evidence about these checkpoints and observed validation inputs, not a proof that EPSS is unpredictable.',
             '', '2. Primary Day-30 to Day-60 prediction sensitivity',
             'Seed | Flat history: mean absolute change | Flat history: largest change | Last-only: mean absolute change | Last-only: largest change']
    for seed in d.SEEDS:
        a=summary['prediction_differences'][str(seed)]['flat_history']['day30_to_day60']
        b=summary['prediction_differences'][str(seed)]['last_obs_only']['day30_to_day60']
        lines.append(f'{seed} | {a["mean_abs_raw_diff"]:.12g} | {a["max_abs_raw_diff"]:.12g} | '
                     f'{b["mean_abs_raw_diff"]:.12g} | {b["max_abs_raw_diff"]:.12g}')
    lines += ['', 'Differences compare raw intervention predictions with the intact predictions on the same CVE and target day.',
              'A large count of changed cells can still consist of numerically tiny changes; magnitude matters.',
              '', '3. Paired error changes, averaged across the three seeds',
              'Slice | Intervention minus intact | MAE difference [95% interval] | RMSE difference [95% interval]']
    for r in findings['mean_seed_paired_intervals']:
        if r['reference']!='intact':continue
        a,b=r['metrics']['mae'],r['metrics']['rmse']
        lines.append(f'{r["slice"]} | {r["intervention"]} | {a["difference"]:+.12g} [{a["lower"]:+.12g}, {a["upper"]:+.12g}] | '
                     f'{b["difference"]:+.12g} [{b["lower"]:+.12g}, {b["upper"]:+.12g}]')
    lines += ['', 'Negative error changes favor the intervention; positive changes favor intact.',
              'Flat-history intervals include zero on both primary and complete-matrix MAE/RMSE.',
              'Last-only changes predictions more: its mean MAE is worse, while its mean RMSE improves.',
              'Removing older observations changes value availability, masks, age and known-history, not sequence length.',
              'Therefore its effect cannot be attributed to past score fluctuations alone.',
              '', '4. Fixed-offset decomposition',
              f'Fixed offsets reproduce {shares["day30_to_day60"]["mae"]["constant_share_percent"]:.4f}% of the primary mean MAE gain,',
              f'{shares["day30_to_day60"]["rmse"]["constant_share_percent"]:.4f}% of the primary RMSE gain, and',
              f'{shares["all"]["rmse"]["constant_share_percent"]:.4f}% of the complete-matrix RMSE gain.',
              'Intact retains a very small average advantage over the fixed offset; it is not exactly a constant model.',
              'These offsets are derived from validation predictions, not fitted on training. They are descriptive controls only.',
              'A bootstrap holds the offsets/checkpoints fixed; it does not remove validation-selection bias or establish causality.',
              '', '5. Actual score directions',
              f'Seven-day cells: {dict(counts["lead7"])}.',
              f'Day-30 to Day-60 CVEs: {dict(counts["day30_to_day60"])}.',
              'These are unweighted observed-cell counts. They are not the hierarchical training/evaluation weights.',
              'The upward-adjustment interpretation is consistent with these directions, but they do not establish why scores changed.',
              '', '6. Decision',
              'Retain the fixed Stage 2 baseline. Do not promote an intervention or run another GRU tuning sweep from these results.',
              'There is little evidence here that pre-anchor score movements drive useful predictions.',
              'The stronger response to last-only concerns history availability/state as well as values.',
              'Any subsequent TPG comparison should test additional, timestamp-supported evidence on the same eligible cells,',
              'against the unchanged EPSS-only model and persistence. These diagnostics do not establish that text fusion will help.',
              'No test scores, parameter updates or new training were produced. All original results remain unchanged.']
    with d.data.transaction(OUTPUT,verify) as stage:
        (stage/'findings.json').write_text(d.data.dump(findings))
        (stage/'findings.txt').write_text('\n'.join(lines)+'\n')
    print('CF01 Stage 3 findings ' + ('reproduced.' if verify else 'saved.') + ' No new predictions, training or test scoring.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-only',action='store_true')
    run(p.parse_args().verify_only)
