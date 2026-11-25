"""Matched history interventions; no optimizer, fit, test inputs or source mutations."""
from contextlib import contextmanager
import csv
from datetime import date, timedelta
import fcntl
import gzip
import io
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .. import data, evaluation as ev, runner

SOURCE = data.OUTPUT.parent / 'stage2_v1/full'
OUTPUT = data.OUTPUT.parent / 'stage3_v1'
CONFIG = data.ROOT / '02_prediction/04_configs/02_continuous_forecasting/02_stage3.json'
CLI = data.SCRIPTS / 'stage3'
SEEDS = (17, 29, 43)
VARIANTS = ('persistence', 'intact', 'flat_history', 'last_obs_only', 'constant_shift_control')
PAIRS = (('intact', 'persistence'), ('flat_history', 'intact'), ('last_obs_only', 'intact'),
         ('constant_shift_control', 'intact'), ('constant_shift_control', 'persistence'))


def transform(inputs, mode):
    data.require(mode in ('intact', 'flat_history', 'last_obs_only'), 'Unknown intervention')
    data.require(set(inputs) == {'past', 'lengths', 'leads', 'latest'}, 'Unexpected model inputs')
    out = {k: v.clone() for k, v in inputs.items()}
    for i, length in enumerate(inputs['lengths'].tolist()):
        observed = torch.nonzero(inputs['past'][i, :length, 1] == 1).flatten()
        data.require(len(observed) > 0, 'No observed history')
        last = int(observed[-1])
        if mode == 'flat_history':
            prior = observed[observed < length-1]
            out['past'][i, prior, 0] = inputs['latest'][i, 0].to(out['past'].dtype)
        elif mode == 'last_obs_only':
            out['past'][i, :last, :4] = 0
    return out


class OrderedWriter:
    """Assert canonical streaming order without holding millions of CSV rows."""
    def __init__(self, stream, header):
        self.writer = csv.writer(stream, lineterminator='\n')
        self.writer.writerow(header)
        self.previous = None

    def write(self, key, values):
        data.require(self.previous is None or key > self.previous, 'CSV keys duplicated or out of canonical order')
        self.writer.writerow(values)
        self.previous = key


def small_csv(path, header, rows):
    with path.open('w', newline='') as stream:
        writer = OrderedWriter(stream, header)
        for key, values in sorted(rows, key=lambda r: r[0]):
            writer.write(key, values)


@contextmanager
def compressed_csv(path, header):
    with path.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=1) as zipped:
            with io.TextIOWrapper(zipped, encoding='utf-8', newline='') as text:
                yield OrderedWriter(text, header)


def context(scale):
    rows, parent = data.load('full')
    settings = data.read(CONFIG)
    data.require(settings['seeds'] == list(SEEDS) and not settings['training_enabled']
                 and not settings['test_access_enabled'], 'Wrong diagnostic contract')
    for name in ('controls', 'report', *(f'seed_{s}' for s in SEEDS)):
        data.check_lock(SOURCE / name)
    saved_identity = data.read(SOURCE / 'report/run_identity.json')
    data.require(saved_identity['parent_identity'] == data.digest(parent), 'Full preparation identity changed')
    data.check_pins(saved_identity['source_pins'])
    baseline = data.read(SOURCE / 'controls/persistence_evaluation.json')
    ids = [r['cve'] for r in rows['validation']]
    data.require(len(ids) == 1104 and baseline['summary']['all']['cells'] == 3067429, 'Wrong full validation cohort')
    references, constants = {}, {}
    for seed in SEEDS:
        ref = data.read(SOURCE / f'seed_{seed}/evaluation.json')
        data.require([r['cve'] for r in ref['per_cve']] == ids == [r['cve'] for r in baseline['per_cve']], 'Unmatched reference CVEs')
        # Bias differences cancel the target exactly algebraically. The constant
        # will also be checked directly against replayed prediction residuals.
        constants[str(seed)] = float(np.mean([a['metrics']['day30_to_day60']['raw_bias'] -
                                              b['metrics']['day30_to_day60']['raw_bias']
                                              for a, b in zip(ref['per_cve'], baseline['per_cve'])]))
        references[seed] = ref
    selected = data.choose(ids, settings['smoke_validation_cves']) if scale == 'smoke' else ids
    selected_rows = sorted([r for r in rows['validation'] if r['cve'] in set(selected)], key=lambda r: r['cve'])
    del rows
    pins = dict(saved_identity['source_pins'])
    paths = [CONFIG, *Path(__file__).parent.glob('*.py'), *CLI.glob('*.py')]
    paths += [SOURCE / n / 'artifact_lock.json' for n in ('controls', 'report', *(f'seed_{s}' for s in SEEDS))]
    for path in paths:
        pins[str(path)] = data.sha(path)
    identity = dict(schema=settings['schema'], scale=scale, settings=settings,
                    model_settings=parent['settings'], source_identity=data.digest(saved_identity),
                    source_pins=pins, ids=selected, constants=constants,
                    constant_estimation_cves=1104,
                    cells=sum(int(data.eligible(r).sum()) for r in selected_rows),
                    checkpoint_hashes={str(s): data.sha(SOURCE / f'seed_{s}/checkpoint.pt') for s in SEEDS})
    return selected_rows, identity, references, baseline


def predict(row, model, packet_size, constant):
    mask = data.eligible(row)
    truth = np.full((90, 90), np.nan)
    predictions = {v: np.full((90, 90), np.nan) for v in VARIANTS}
    for items in data.packets(data.requests(row), packet_size):
        inputs, _, _, _ = data.batch(items)
        arrays = {mode: data.forward(model, transform(inputs, mode))[0].numpy()
                  for mode in ('intact', 'flat_history', 'last_obs_only')}
        for i, (_, t, leads, _) in enumerate(items):
            ix = leads-1
            truth[t, ix] = [row['values'][t+int(h)] for h in leads]
            last = float(inputs['latest'][i, 0])
            predictions['persistence'][t, ix] = last
            predictions['constant_shift_control'][t, ix] = last + constant
            for mode in arrays:
                predictions[mode][t, ix] = arrays[mode][i, :len(leads)]
    for values in predictions.values():
        data.require(np.isfinite(values[mask]).all() and np.isnan(values[~mask]).all(), 'Forecast cells changed')
    return truth, predictions, mask


def differences(raw, intact, mask):
    out = {}
    for group in ev.SLICES:
        selected = ev.slice_mask(mask, group)
        anchors = np.flatnonzero(selected.any(1))
        if not len(anchors):
            out[group] = None
            continue
        rd, cd = raw-intact, np.clip(raw, 0, 1)-np.clip(intact, 0, 1)
        out[group] = dict(cells=int(selected.sum()),
                          raw_changed_cells=int(np.count_nonzero(rd[selected])),
                          clipped_changed_cells=int(np.count_nonzero(cd[selected])),
                          mean_raw_diff=float(np.mean([rd[t, selected[t]].mean() for t in anchors])),
                          mean_abs_raw_diff=float(np.mean([np.abs(rd[t, selected[t]]).mean() for t in anchors])),
                          max_abs_raw_diff=float(np.abs(rd[selected]).max()),
                          mean_clipped_diff=float(np.mean([cd[t, selected[t]].mean() for t in anchors])),
                          mean_abs_clipped_diff=float(np.mean([np.abs(cd[t, selected[t]]).mean() for t in anchors])),
                          max_abs_clipped_diff=float(np.abs(cd[selected]).max()))
    return out


def summarize_differences(rows):
    out = {}
    for group in ev.SLICES:
        selected = [r[group] for r in rows if r[group] is not None]
        if not selected:
            out[group] = None
            continue
        out[group] = {k: (sum(r[k] for r in selected) if k.endswith('cells') else
                          max(r[k] for r in selected) if k.startswith('max_') else
                          float(np.mean([r[k] for r in selected]))) for k in selected[0]}
        out[group]['cves'] = len(selected)
    return out


@torch.inference_mode()
def run_seed(output, rows, identity, references, baseline, seed, verify=False, resume=False):
    destination = output / f'seed_{seed}'
    if destination.exists():
        data.check_lock(destination)
        data.require(data.read(destination / 'identity.json') == identity, 'Existing diagnostic identity changed')
        if resume and not verify:
            return data.read(destination / 'results.json')
    model = runner.model_for(seed, identity['model_settings'])
    checkpoint = torch.load(SOURCE / f'seed_{seed}/checkpoint.pt', map_location='cpu', weights_only=True)
    data.require(checkpoint['identity'] == identity['source_identity'] and checkpoint['seed'] == seed, 'Checkpoint identity differs')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval().requires_grad_(False)
    weights_before = {k: v.clone() for k, v in model.state_dict().items()}
    per_variant = {v: [] for v in VARIANTS}
    diffs = {v: [] for v in VARIANTS}
    ref_lookup = {r['cve']: r['metrics'] for r in references[seed]['per_cve']}
    base_lookup = {r['cve']: r['metrics'] for r in baseline['per_cve']}
    selected_ids = set(identity['ids'])
    constant = identity['constants'][str(seed)]
    primary_residuals = []
    header = ['cve', 'anchor_day', 'target_day', 'lead_days', 'anchor_date', 'target_date', 'actual_epss']
    for v in VARIANTS:
        header += [v+'_raw', v+'_clipped']
    for v in VARIANTS[2:]:
        header += [v+'_raw_minus_intact', v+'_clipped_minus_intact']
    with data.transaction(destination, verify) as stage:
        (stage / 'identity.json').write_text(data.dump(identity))
        with (SOURCE / f'seed_{seed}/predictions.csv').open(newline='') as saved:
            expected = (r for r in csv.DictReader(saved) if r['cve'] in selected_ids)
            with compressed_csv(stage / 'predictions_and_differences.csv.gz', header) as writer:
                for row in tqdm(rows, desc=f'Frozen seed {seed}: matched history interventions', unit='CVE'):
                    truth, preds, mask = predict(row, model, identity['model_settings']['anchor_packet'], constant)
                    for v in VARIANTS:
                        metrics = ev.cve_metrics(truth, preds[v], mask)
                        per_variant[v].append(dict(cve=row['cve'], metrics=metrics))
                        diffs[v].append(dict(cve=row['cve'], metrics=differences(preds[v], preds['intact'], mask)))
                    data.require(per_variant['intact'][-1]['metrics'] == ref_lookup[row['cve']], 'Intact per-CVE metrics differ')
                    data.require(per_variant['persistence'][-1]['metrics'] == base_lookup[row['cve']], 'Persistence metrics differ')
                    if mask[30, 29]:
                        primary_residuals.append(float(preds['intact'][30, 29]-preds['persistence'][30, 29]))
                    pub = date.fromisoformat(row['publication_date'])
                    clipped = {v: np.clip(p, 0, 1) for v, p in preds.items()}
                    for t, j in np.argwhere(mask):
                        t, j = int(t), int(j)
                        u = t+j+1
                        previous = next(expected, None)
                        anchor_date = (pub+timedelta(days=t)).isoformat()
                        target_date = (pub+timedelta(days=u)).isoformat()
                        data.require(previous is not None and previous['cve'] == row['cve'] and previous['split'] == 'validation'
                                     and int(previous['anchor_day']) == t and int(previous['target_day']) == u
                                     and int(previous['lead_days']) == j+1 and previous['anchor_date'] == anchor_date
                                     and previous['target_date'] == target_date
                                     and float(previous['actual_epss']) == truth[t, j]
                                     and float(previous['raw_prediction']) == preds['intact'][t, j]
                                     and float(previous['clipped_prediction']) == clipped['intact'][t, j],
                                     'Original prediction cell differs on intact replay')
                        values = [row['cve'], t, u, j+1, anchor_date, target_date, float(truth[t, j])]
                        for v in VARIANTS:
                            values += [float(preds[v][t, j]), float(clipped[v][t, j])]
                        for v in VARIANTS[2:]:
                            values += [float(preds[v][t, j]-preds['intact'][t, j]), float(clipped[v][t, j]-clipped['intact'][t, j])]
                        writer.write((row['cve'], t, u), values)
                data.require(next(expected, None) is None, 'Saved prediction cells left unmatched')
        if identity['scale'] == 'full':
            data.require(len(primary_residuals) == 1104 and abs(float(np.mean(primary_residuals))-constant) <= 1e-15,
                         'Constant does not match direct intact-minus-persistence residuals')
        data.require(all(torch.equal(v, model.state_dict()[k]) for k, v in weights_before.items()), 'Frozen parameters changed')
        result = {v: dict(summary=ev.summarize(per_variant[v]), per_cve=per_variant[v],
                          prediction_difference=summarize_differences([r['metrics'] for r in diffs[v]]),
                          per_cve_prediction_difference=diffs[v]) for v in VARIANTS}
        (stage / 'results.json').write_text(data.dump(result))
        data.check_pins(identity['source_pins'])
    return result


def paired(results, settings):
    report = []
    scopes = [(str(s), (s,)) for s in SEEDS]+[('seed_mean', SEEDS)]
    for group in tqdm(ev.SLICES, desc='Paired CVE intervals for predeclared slices', unit='slice'):
        ids = [r['cve'] for r in results[17]['intact']['per_cve'] if r['metrics'][group] is not None]
        data.require(bool(ids), 'Required slice empty')
        draws = np.random.default_rng(settings['bootstrap_seed']).integers(len(ids), size=(settings['bootstrap_repetitions'], len(ids)))
        packed = {}
        for seed in SEEDS:
            for v in VARIANTS:
                records = [r for r in results[seed][v]['per_cve'] if r['metrics'][group] is not None]
                data.require([r['cve'] for r in records] == ids, 'Bootstrap pairing changed')
                a = np.array([[r['metrics'][group][k] for k in ('mae', 'mse', 'raw_mae', 'raw_mse')] for r in records])
                point = a.mean(0)
                samples = a[draws].mean(1)
                point[[1, 3]] = np.sqrt(point[[1, 3]])
                samples[:, [1, 3]] = np.sqrt(samples[:, [1, 3]])
                packed[seed, v] = point, samples
        for scope, seeds in scopes:
            for a, b in PAIRS:
                point = np.mean([packed[s, a][0]-packed[s, b][0] for s in seeds], axis=0)
                samples = np.mean([packed[s, a][1]-packed[s, b][1] for s in seeds], axis=0)
                report.append(dict(seed=scope, slice=group, intervention=a, reference=b, cves=len(ids),
                                   metrics={k: dict(difference=float(point[i]), lower=float(np.percentile(samples[:, i], 2.5)),
                                                    upper=float(np.percentile(samples[:, i], 97.5)))
                                            for i, k in enumerate(('mae', 'rmse', 'raw_mae', 'raw_rmse'))}))
    return report


def publish(output, identity, results, verify):
    intervals = paired(results, identity['settings'])
    decomposition = {}
    for seed in SEEDS:
        decomposition[str(seed)] = {}
        for group in ev.SLICES:
            decomposition[str(seed)][group] = {}
            for metric in ('mae', 'rmse'):
                b, n, c = [results[seed][v]['summary'][group][metric] for v in ('persistence', 'intact', 'constant_shift_control')]
                gain = b-n
                decomposition[str(seed)][group][metric] = dict(intact_gain=gain, constant_gain=b-c,
                    variation_gain=c-n, constant_fraction_of_intact_gain=(b-c)/gain if abs(gain)>1e-15 else None)
    summary = dict(identity=identity, metrics={str(s): {v: results[s][v]['summary'] for v in VARIANTS} for s in SEEDS},
                   prediction_differences={str(s): {v: results[s][v]['prediction_difference'] for v in VARIANTS} for s in SEEDS},
                   decomposition=decomposition, training=False, test_scoring=False)
    with data.transaction(output / 'report', verify) as stage:
        (stage / 'summary.json').write_text(data.dump(summary))
        (stage / 'paired_intervals.json').write_text(data.dump(intervals))
        table = []
        for seed in SEEDS:
            for v in VARIANTS:
                for group in ev.SLICES:
                    m = results[seed][v]['summary'][group]
                    d = results[seed][v]['prediction_difference'][group]
                    table.append(((seed, v, group), [seed, v, group, m['cves'], m['cells'], m['mae'], m['rmse'], m['raw_mae'], m['raw_rmse'],
                                  m['clipping_rate'], d['raw_changed_cells'], d['mean_abs_raw_diff'], d['max_abs_raw_diff']]))
        small_csv(stage/'metrics.csv', ['seed', 'variant', 'slice', 'CVEs', 'cells', 'MAE', 'RMSE', 'raw_MAE', 'raw_RMSE',
                                      'clipped_fraction', 'raw_cells_changed_vs_intact', 'mean_abs_raw_change_vs_intact', 'max_abs_raw_change_vs_intact'], table)
        interval_rows = []
        for r in intervals:
            for metric, values in r['metrics'].items():
                key = (r['seed'], r['slice'], r['intervention'], r['reference'], metric)
                interval_rows.append((key, [*key, values['difference'], values['lower'], values['upper']]))
        small_csv(stage/'paired_changes.csv', ['seed', 'slice', 'intervention', 'reference', 'metric', 'difference', 'lower_95', 'upper_95'], interval_rows)
        lines = ['CF01 Stage 3: frozen-checkpoint history-reliance diagnostic', '',
                 f'Scale: {identity["scale"]}; validation CVEs: {len(identity["ids"])}; matched cells per seed: {identity["cells"]}.',
                 'CPU only. No parameter updates, threshold tuning, test inputs or changes to Stage 2.',
                 'Every intact prediction is checked exactly against the original prediction CSV.',
                 '', 'Interventions',
                 '- flat_history replaces prior observed values with the latest observed value; masks and ages stay fixed.',
                 '- last_obs_only erases observations before the last one, resetting earlier age/known-history channels.',
                 '  Prefix length, day indices, anchor, requested lead and latest score stay fixed. This is not sequence shortening.',
                 '- constant_shift_control uses each seed\'s FULL-validation Day30->60 mean raw residual, unchanged for every slice.',
                 '  This post-hoc decomposition is not a training-only control or a deployable forecasting estimate.',
                 '  It uses predictions rather than target values to choose the offset; full replay checks the direct residual.',
                 '', 'MAE and RMSE on each predeclared slice (three-seed means, not an ensemble)',
                 'Slice | Variant | MAE | RMSE | Mean absolute raw prediction change vs intact | Largest absolute raw change']
        for group in ev.SLICES:
            for v in VARIANTS:
                m = [results[s][v]['summary'][group] for s in SEEDS]
                d = [results[s][v]['prediction_difference'][group] for s in SEEDS]
                lines.append(f'{group} | {v} | {np.mean([r["mae"] for r in m]):.12f} | '
                             f'{np.mean([r["rmse"] for r in m]):.12f} | {np.mean([r["mean_abs_raw_diff"] for r in d]):.12g} | '
                             f'{max(r["max_abs_raw_diff"] for r in d):.12g}')
        lines += ['', 'Reading the outputs',
                  'A cell is one CVE, anchor day and future target day. Missing targets stay excluded for every intervention.',
                  'Errors average leads within an anchor, anchors within a CVE, then CVEs. Raw and clipped values are both saved.',
                  'Prediction differences are intervention minus intact. Changed-cell counts use exact inequality, not a tolerance.',
                  'Paired MAE/RMSE changes and 95% intervals are in paired_changes.csv and paired_intervals.json.',
                  'Negative error changes mean improvement over the named reference; positive means worse.',
                  'Bootstrap: 1,000 whole-CVE paired resamples, seed 6029, same draw for all seeds/variants.',
                  'Constants and selected checkpoints stay fixed in the bootstrap; intervals are conditional and unadjusted for multiple diagnostics.',
                  '', 'Decomposition',
                  'intact_gain = persistence error minus intact error; constant_gain = persistence error minus constant-shift error.',
                  'variation_gain = constant-shift error minus intact error. Positive variation_gain favors the intact CVE-specific corrections.',
                  'constant_fraction_of_intact_gain may exceed 1 or be negative; it is a descriptive ratio, not a causal explained fraction.',
                  '', 'Limitations',
                  'These are input interventions on frozen models, potentially outside their training distribution.',
                  'They test sensitivity, not a causal effect or the performance of retrained reduced-input models.',
                  'Validation has already been repeatedly inspected; no independent test conclusion is supported.',
                  'Small prediction changes do not by themselves establish useful temporal information.',
                  'No text or TPG is included. Do not reinterpret this as an exploitation or graph result.',
                  'CSV rows use explicit canonical keys; gzip timestamps and filenames are fixed for byte-identical replay.']
        (stage/'report.txt').write_text('\n'.join(lines)+'\n')
        data.check_pins(identity['source_pins'])


def execute(scale='smoke', verify=False, resume=False, check_only=False):
    data.require(not (verify and resume), 'Choose resume or verification')
    rows, identity, references, baseline = context(scale)
    print(f'[CF01 Stage3] {scale}; {len(rows)} validation CVEs; {identity["cells"]} cells; frozen FULL checkpoints; CPU only.', flush=True)
    print('[CONSTANTS] Full-validation primary residual means:', identity['constants'], flush=True)
    if check_only:
        print('Stage 3 preflight passed. No predictions or output written.')
        return
    output = OUTPUT / scale
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        results = {s: run_seed(output, rows, identity, references, baseline, s, verify, resume)
                   for s in tqdm(SEEDS, desc='Three frozen checkpoints, zero training', unit='seed')}
        if resume and (output/'report').exists():
            data.check_lock(output/'report')
            data.require(data.read(output/'report/summary.json')['identity'] == identity, 'Report identity differs')
        else:
            publish(output, identity, results, verify)
    print(f'CF01 Stage 3 {scale} ' + ('predictions, interventions, canonical CSVs and paired reports reproduced.' if verify else f'saved to {output}.') + ' No training or test scoring.')
