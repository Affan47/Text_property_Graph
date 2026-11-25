"""Train only DM/DGN S1; reference completed D/DGNM and baseline fits read-only."""
import argparse
import csv
from pathlib import Path

from ..primary_study import common
from ..primary_training import runner as original, reporting as original_reports
from ..consolidated.evaluation import score_predictions
from ..forecast_matched_multisource import writer_lock
from . import reporting

read, write, require, tqdm = common.read, common.write, common.require, common.tqdm
CONFIG = common.ROOT / '02_prediction/04_configs/01_consolidated/04_primary_source_comparison.json'
SCRIPT = common.SCRIPTS / '10_run_primary_source_comparison.py'
TEST = common.SCRIPTS / 'test_primary_source_comparison.py'
OUTPUT = common.RESULTS / 'source_comparison_v1'


def protocol():
    p = read(CONFIG)
    require(p['schema'] == 'ps04_bounded_sources_v1' and p['test_scoring'] is False, 'Wrong protocol')
    require(p['source_codes'] == ['D', 'DM', 'DGN', 'DGNM']
            and p['reused_codes'] == ['D', 'DGNM'] and p['new_codes'] == ['DM', 'DGN'], 'Source plan changed')
    require(p['representations'] == ['pooled_text', 'security_tpg']
            and p['structured_blocks'] == ['cvss', 'reference_count']
            and p['seeds'] == [17, 29, 43], 'Fixed comparison changed')
    return p


def plan(scale):
    require(scale in ('smoke', 'full'), 'Unknown scale')
    p = protocol()
    registered = {v['id']: v for v in original.variants_for('all-supported')}
    settings = original.settings_for(scale)
    require(original.settings_for('full')['seeds'] == p['seeds'], 'Seed protocol differs')
    rows = []
    for name, variant in registered.items():
        if name.startswith('P01_NONE_'):
            rows.append({'variant': variant, 'seed': None, 'key': original.fit_key(variant, None), 'origin': 'ps02_reused'})
    for code in p['source_codes']:
        for rep in p['representations']:
            variant = registered[f'P01_{code}_{rep}_S1']
            require(variant['structured_blocks'] == p['structured_blocks'], 'Structured comparison differs')
            for seed in settings['seeds']:
                rows.append({'variant': variant, 'seed': seed, 'key': original.fit_key(variant, seed),
                             'origin': 'ps02_reused' if code in p['reused_codes'] else 'ps04_new'})
    require(len(rows) == (27 if scale == 'full' else 11), 'Unexpected fit count')
    require(sum(r['origin'] == 'ps04_new' for r in rows) == (12 if scale == 'full' else 4), 'Unexpected new fits')
    return rows


def fit_directory(directory, source, item):
    return (source if item['origin'] == 'ps02_reused' else directory) / 'fits' / item['key']


def validate_source_identity(saved, current):
    require(saved == current, 'Original PS02 identity differs; do not mix cohorts, settings, code or packages')
    require(set(saved['ids']) == {'train', 'validation'} and saved['test_scoring'] is False, 'Invalid development roles')
    require(not set(saved['ids']['train']) & set(saved['ids']['validation']), 'Overlapping cohorts')


def context(scale):
    source = original.OUTPUT / scale
    require((source / 'run_identity.json').is_file(), 'Completed PS02 run required for this scale')
    saved = read(source / 'run_identity.json')
    batches, targets, base = original.context(scale, saved['threads'], saved['cache_mib'])
    validate_source_identity(saved, base)
    rows = plan(scale)
    if scale == 'full':
        require({s: len(v) for s, v in base['ids'].items()} == protocol()['expected_full_cohort'], 'Full cohort changed')
    pins = {str(p): common.sha256(p) for p in
            [CONFIG, SCRIPT, TEST, *sorted(Path(__file__).parent.glob('*.py')), source / 'run_identity.json']}
    for item in tqdm([r for r in rows if r['origin'] == 'ps02_reused'], desc='Check read-only completed fits', unit='fit'):
        out = source / 'fits' / item['key']
        common.verify_lock(out)
        require(read(out / 'fit_identity.json') == {'run_digest': common.digest(base),
                'variant': item['variant'], 'seed': item['seed']}, 'Reused fit identity mismatch')
        pins[str(out / 'artifact_lock.json')] = common.sha256(out / 'artifact_lock.json')
    identity = {**base, 'schema': 'ps04_bounded_source_run_v1', 'protocol': protocol(),
                'parent_run_digest': common.digest(base), 'comparison_plan': rows,
                'source_pins': {**base['source_pins'], **pins}}
    print(f"[PS04] {sum(r['origin']=='ps04_new' for r in rows)} new fits; "
          f"{sum(r['origin']=='ps02_reused' for r in rows)} read-only reused fits (including 3 baselines).", flush=True)
    print('[FIXED] S1 CVSS + reference count; original cohorts, seeds, optimizer, pooling and early stopping. Test closed.', flush=True)
    return source, batches, targets, base, identity, rows


def finish_new_fit(out, pending, item, batches, targets, identity):
    variant, seed = item['variant'], item['seed']
    require(item['origin'] == 'ps04_new' and seed is not None, 'Refuse training a reused fit')
    out.mkdir(parents=True, exist_ok=True)
    write(out / 'fit_identity.json', {'run_digest': common.digest(identity), 'variant': variant, 'seed': seed})
    predictions, diagnostic = original.train_fit(out, pending, variant, seed, batches, targets, identity)
    write(out / 'predictions.json', predictions)
    write(out / 'diagnostics.json', diagnostic)
    write(out / 'metrics.json', score_predictions([r['actual'] for r in predictions], [r['predicted'] for r in predictions]))
    with (out / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['cve', 'split', 'target_date', 'actual', 'predicted'])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in predictions)
    original_reports.fit_report(out)
    original.verify_fit(out, variant, seed, batches, targets, identity, locked=False)
    common.lock(out)


def execute(args):
    rows = plan(args.scale)
    if args.list:
        for item in rows:
            print(f"{item['origin']}: {item['key']}")
        print(f"{len(rows)} total: {sum(r['origin']=='ps04_new' for r in rows)} new; test disabled.")
        return
    source, batches, targets, base, identity, rows = context(args.scale)
    directory = OUTPUT / args.scale
    if args.check_only:
        print('PS04 preflight passed. No files written, training or test scoring.')
        return
    require(not args.verify_only or directory.is_dir(), 'No PS04 output to verify')
    require(not directory.exists() or args.resume or args.verify_only, 'Output exists; use --resume or --verify-only')
    directory.mkdir(parents=True, exist_ok=True)
    with writer_lock(directory / '.writer.lock'):
        path = directory / 'run_identity.json'
        if path.exists():
            require(read(path) == identity, 'PS04 identity changed; refuse mixed runs')
        else:
            require(not args.verify_only, 'Missing PS04 identity')
            write(path, identity)
        for item in tqdm(rows, desc='Bounded source comparisons', unit='fit'):
            out = fit_directory(directory, source, item)
            reused = item['origin'] == 'ps02_reused'
            fit_identity = base if reused else identity
            if reused or args.verify_only:
                original.verify_fit(out, item['variant'], item['seed'], batches, targets, fit_identity)
                print(('[REUSED unchanged] ' if reused else '[VERIFIED] ') + item['key'], flush=True)
            elif (out / 'artifact_lock.json').exists():
                original.verify_fit(out, item['variant'], item['seed'], batches, targets, identity)
                print('[SKIP completed and verified] ' + item['key'], flush=True)
            else:
                pending = directory / 'pending'
                pending.mkdir(exist_ok=True)
                print('[TRAIN new] ' + item['key'], flush=True)
                finish_new_fit(out, pending / (item['key'] + '.pt'), item, batches, targets, identity)
                print('[DONE and reload verified] ' + item['key'], flush=True)
        common.check_pins(identity['source_pins'])
        locations = {r['key']: fit_directory(directory, source, r) for r in rows}
        report = reporting.aggregate(locations, rows, identity)
        out = directory / 'reports'
        if args.verify_only:
            common.verify_lock(out)
            require(read(out / 'analysis.json') == report, 'PS04 report differs')
        else:
            reporting.save(out, report, identity)
    print(f'PS04 {args.scale} ' + ('reused/new checkpoints and matched reports verified.' if args.verify_only else f'saved to {directory}.') +
          ' No test scoring; original PS02 fits unchanged.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    mode = parser.add_mutually_exclusive_group()
    for name in ('list', 'check-only', 'resume', 'verify-only'):
        mode.add_argument('--' + name, action='store_true')
    execute(parser.parse_args())


if __name__ == '__main__':
    main()
