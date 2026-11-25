"""TR01: separate daily targets for Days 31-60, retaining the frozen CVE roles.

Versioned from MH01 so its pinned inputs, scripts and checkpoints stay intact.
"""
import argparse
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

import torch

from . import forecast_day60_multisource as parent
from . import forecast_matched_multisource as helpers

read, write, require, sha256, tqdm = parent.read, parent.write, parent.require, parent.sha256, parent.tqdm
SCHEMA = 'tr01_day30_daily_trajectory_v1'
ROOT = parent.ROOT
CONFIG = ROOT/'02_prediction/04_configs/trajectory_v1.json'
OUTPUT = ROOT/'04_data/01_records_and_graphs/day30_trajectory_v1'
SCRIPT = helpers.SCRIPTS/'prepare_forecast_trajectory.py'
TEST = helpers.SCRIPTS/'test_forecast_trajectory.py'
FILES = {'protocol.json', 'run_identity.json', 'inputs.json', 'labels.json', 'split_manifest.json',
         'coverage.json', 'summary.txt'}
LENGTHS, HORIZONS = (10, 20, 31), tuple(range(1, 31))


def protocol():
    p = read(CONFIG)
    require(p['schema'] == SCHEMA and p['anchor_day'] == 30 and p['history_lengths'] == list(LENGTHS)
            and p['horizons'] == list(HORIZONS) and p['target_days'] == list(range(31, 61))
            and p['gain'] == 1 and p['large_change_threshold'] == .01
            and not p['test_scoring'], 'Wrong TR01 protocol')
    return p


def input_view(row, length):
    require(length in LENGTHS, 'Unknown history length')
    h = parent.history_view(row)
    out = parent.temporal.history(h['dates'], h['values'], 31-length, 30)
    require(any(out['observed_mask']), 'No observed score in requested window; refuse outside-window fallback')
    return out


def input_tensor(rows, length):
    require(bool(rows), 'Empty input batch')
    histories = [input_view(r, length) for r in rows]
    return torch.tensor([[[0. if v is None else v, h['observed_mask'][j],
                          (h['age_since_observation_days'][j] or 0)/60, h['age_known_mask'][j],
                          h['day_offsets'][j]/60] for j, v in enumerate(h['values'])] for h in histories], dtype=torch.float32)


def derive_labels(raw, frozen_input, frozen_label):
    dates, values = parent.contract.score_days(raw)
    require(raw['cve'] == frozen_input['cve'] == frozen_label['cve'], 'Wrong source identity')
    require(raw['published_date'] == frozen_input['publication_date'] and
            values[:31] == parent.history_view(frozen_input)['values'] and
            dates[60] == frozen_label['target_date'] and values[60] == frozen_label['y']['endpoint'],
            'Raw source differs from frozen Day-60 preparation')
    selected = [values[30+h] for h in HORIZONS]
    return {'cve': raw['cve'], 'anchor_date': dates[30], 'horizons': list(HORIZONS),
            'target_dates': [dates[30+h] for h in HORIZONS], 'values': selected,
            'observed_mask': [int(v is not None) for v in selected]}


def context(scale):
    require(scale in ('smoke', 'full'), 'Unknown scale')
    p = protocol()
    full = parent.OUTPUT/'full'
    pins = parent.check_lock(full, 'preparation_lock.json')
    identity = read(full/'run_identity.json')
    require(identity['schema'] == parent.SCHEMA and identity['scale'] == 'full'
            and {s: len(v) for s, v in identity['ids'].items()} == p['cohort'], 'Wrong parent cohort')
    frozen_inputs, frozen_labels = read(full/'inputs.json'), read(full/'labels.json')
    if scale == 'smoke':
        small = parent.OUTPUT/'smoke'
        pins.update(parent.check_lock(small, 'preparation_lock.json'))
        ids = read(small/'run_identity.json')['ids']
    else:
        ids = identity['ids']
    require(set(ids) == {'train', 'validation'} and all(set(v) <= set(identity['ids'][s]) for s, v in ids.items()), 'Changed roles')
    locator = parent.sources.PREPARED/'preparation_summary.json'
    require(sha256(locator) == identity['source_pins'][str(locator)], 'Changed source locator')
    source = Path(read(locator)['source_path'])
    require(sha256(source) == identity['source_pins'][str(source)], 'Changed raw dataset')
    pins[str(locator)], pins[str(source)] = sha256(locator), sha256(source)
    for path in (CONFIG, SCRIPT, TEST, Path(__file__).resolve(), Path(parent.__file__).resolve(),
                 Path(parent.temporal.__file__).resolve(), Path(parent.contract.__file__).resolve()):
        if str(path) in identity['source_pins']:
            require(sha256(path) == identity['source_pins'][str(path)], 'Frozen helper changed')
        pins[str(path)] = sha256(path)
    selected = set(ids['train']+ids['validation'])
    manifest = read(full/'split_manifest.json')
    require(set(frozen_inputs) == set(frozen_labels) == set(identity['ids']['train']+identity['ids']['validation']), 'Wrong parent input/target IDs')
    # Load the file for integrity, but adapt only the chosen development records.
    records = read(source)
    require(len(records) == len({r['cve'] for r in records}), 'Duplicate source CVE')
    raw = {r['cve']: r for r in records if r['cve'] in selected}
    require(set(raw) == selected, 'Missing selected source CVE')
    inputs, labels = {}, {}
    for c in tqdm(sorted(selected), desc='Prepare development histories and masked horizons', unit='CVE'):
        inputs[c] = frozen_inputs[c]
        for length in LENGTHS:
            input_view(inputs[c], length)
        labels[c] = derive_labels(raw[c], inputs[c], frozen_labels[c])
    policy = read(full/'protocol.json')['split_policy']
    a = {'protocol.json': p, 'run_identity.json': {'schema': SCHEMA, 'scale': scale, 'ids': ids,
         'parent_counts': p['cohort'], 'source_pins': pins, 'split_policy': policy, 'test_scoring': False},
         'inputs.json': inputs, 'labels.json': labels,
         'split_manifest.json': {c: manifest[c] for c in sorted(selected)}}
    validate(a)
    a['coverage.json'] = coverage(a)
    a['summary.txt'] = report(a)
    return a


def validate(a):
    i, x, y, manifest = (a[n] for n in ('run_identity.json', 'inputs.json', 'labels.json', 'split_manifest.json'))
    ids = i['ids']
    require(i['schema'] == SCHEMA and set(ids) == {'train', 'validation'} and not i['test_scoring'], 'Wrong roles/schema')
    require(all(v and v == sorted(set(v)) for v in ids.values()) and not set(ids['train']) & set(ids['validation']), 'Repeated/cross-role IDs')
    selected = set(ids['train']+ids['validation'])
    require(set(x) == set(y) == set(manifest) == selected, 'Unmatched exported IDs or test export')
    for split, chosen in ids.items():
        for c in chosen:
            row, label = x[c], y[c]
            for length in LENGTHS:
                input_view(row, length)
            require(row['cve'] == label['cve'] == c and set(label) ==
                    {'cve', 'anchor_date', 'horizons', 'target_dates', 'values', 'observed_mask'}, 'Wrong target schema')
            require(label['horizons'] == list(HORIZONS) and label['anchor_date'] == row['input_end_date']
                    and len(label['values']) == len(label['observed_mask']) == len(HORIZONS), 'Wrong horizons/mask')
            expected = [(date.fromisoformat(row['input_end_date'])+timedelta(days=h)).isoformat() for h in HORIZONS]
            require(label['target_dates'] == expected and label['observed_mask'] == [int(v is not None) for v in label['values']]
                    and label['observed_mask'][-1] == 1, 'Inconsistent target masks or missing parent Day-60 target')
            for value in label['values']:
                parent.contract.score(value)
            require(manifest[c]['split'] == split and manifest[c]['input_end_date'] == row['input_end_date']
                    and manifest[c]['target_date'] == expected[-1], 'Changed parent role/date')
            require(all(parent.contract.candidate_split(row['input_end_date'], d, i['split_policy']) == (split, '')
                        for d in expected), 'Immature target or changed chronological role')


def coverage(a):
    result = {}
    for split, ids in a['run_identity.json']['ids'].items():
        result[split] = {'n': len(ids), 'horizons': {str(h): {'observed': sum(a['labels.json'][c]['observed_mask'][j] for c in ids),
            'missing': sum(not a['labels.json'][c]['observed_mask'][j] for c in ids)} for j, h in enumerate(HORIZONS)},
            'all_horizons_observed': sum(all(a['labels.json'][c]['observed_mask']) for c in ids),
            'histories': {str(length): {'complete': sum(all(input_view(a['inputs.json'][c], length)['observed_mask']) for c in ids),
                'partial': sum(not all(input_view(a['inputs.json'][c], length)['observed_mask']) for c in ids)} for length in LENGTHS}}
    return result


def report(a):
    lines = ['TR01 matched Day-30 histories and all 30 daily future targets',
        'Latest 10 days = Days 21-30; 20 days = Days 11-30; 31 days = Days 0-30.',
        'All forecasts are issued at Day 30. Leads +1 through +30 mean target Days 31-60, inclusive.',
        'Historical masks and observation age are recomputed inside each window; no outside-window fallback.',
        'inputs.json contains only past EPSS. labels.json holds separate targets and observed-target masks.',
        'Missing targets stay null. No test histories, labels or text are exported.',
        f'Coverage: {a["coverage.json"]}',
        'Comparisons use identical CVEs and observed target subsets across history lengths.',
        'The existing Day-60 eligible cohort is retained; this is not all possible short-horizon CVEs.',
        *a['protocol.json']['limitations']]
    return '\n'.join(lines)+'\n'


def prepare(scale, check_only=False, verify_only=False):
    a = context(scale)
    print('[COHORT] '+str({s: r['n'] for s,r in a['coverage.json'].items()})+'; unchanged parent roles.', flush=True)
    for split, row in a['coverage.json'].items():
        print(f'[TARGETS] {split}: {row["all_horizons_observed"]}/{row["n"]} CVEs have all 30 future scores; '
              f'{sum(h["missing"] for h in row["horizons"].values())} missing CVE/day targets remain masked.', flush=True)
    if check_only:
        print(f'TR01 {scale} preparation preflight passed. No files written or training.')
        return
    directory = OUTPUT/scale
    if verify_only:
        helpers.check_files(directory, 'preparation_lock.json', SCHEMA, FILES)
        for name, value in a.items():
            actual = (directory/name).read_text() if name.endswith('.txt') else read(directory/name)
            require(actual == value, 'Preparation reproduction differs: '+name)
        parent.sources.check_pins(a['run_identity.json']['source_pins'])
        print(f'TR01 {scale} preparation reproduced. No training or test scoring.')
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with helpers.writer_lock(OUTPUT/'.writer.lock'):
        require(not directory.exists(), 'Preparation exists; use --verify-only')
        directory.mkdir()
        for name, value in a.items():
            if name.endswith('.txt'):
                (directory/name).write_text(value, encoding='utf-8')
            else:
                write(directory/name, value)
        parent.sources.check_pins(a['run_identity.json']['source_pins'])
        write(directory/'preparation_lock.json', {'schema': SCHEMA, 'files': {n: sha256(directory/n) for n in sorted(FILES)}})
    print(f'TR01 {scale} preparation saved to {directory}. No training or test scoring.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    prepare(args.scale, args.check_only, args.verify_only)


if __name__ == '__main__':
    main()
