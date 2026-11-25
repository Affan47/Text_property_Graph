"""Compact CVE storage and lazy, past-only batches for the frozen CS01 roles."""
from contextlib import contextmanager
from datetime import date, timedelta
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
from tqdm import tqdm

from ..consolidated import contracts, models, protocol

ROOT = protocol.ROOT
DESIGN = ROOT / '05_results/04_consolidated_studies/00_design_and_preflight/full'
OUTPUT = ROOT / '05_results/04_consolidated_studies/02_continuous_epss/stage1_v1'
CONFIG = ROOT / '02_prediction/04_configs/02_continuous_forecasting/01_stage1.json'
SCRIPTS = ROOT / '02_prediction/03_analysis/05_continuous_forecasting'
require = protocol.require


def read(path):
    return json.loads(Path(path).read_text())


def dump(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(dump(value).encode()).hexdigest()


def check_pins(pins):
    for path, expected in tqdm(pins.items(), desc='Verify frozen source identities', unit='file', leave=False):
        require(sha(path) == expected, 'Changed prerequisite: ' + path)


def check_lock(directory, name='artifact_lock.json'):
    directory = Path(directory)
    files = read(directory / name)['files']
    for relative, expected in files.items():
        path = directory / relative
        require(directory.resolve() in path.resolve().parents and sha(path) == expected,
                'Changed/unsafe artifact: ' + str(path))
    return files


def lock(directory):
    files = {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}
    (directory / 'artifact_lock.json').write_text(dump({'files': files}))


@contextmanager
def transaction(destination, verify=False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / '.writer.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(destination.exists() if verify else not destination.exists(),
                'Existing output needs --verify-only; missing output cannot be verified')
        stage = Path(tempfile.mkdtemp(prefix='.cf01-', dir=destination.parent))
        try:
            yield stage
            lock(stage)
            if verify:
                check_lock(destination)
                require(read(stage / 'artifact_lock.json') == read(destination / 'artifact_lock.json'),
                        'Reproduced artifacts differ')
                shutil.rmtree(stage)
            else:
                stage.rename(destination)
        except BaseException:
            if stage.exists():
                failed = destination.parent / 'failed_attempts'
                failed.mkdir(exist_ok=True)
                stage.rename(failed / stage.name)
            raise


def settings():
    p = read(CONFIG)
    require(p['schema'] == 'cf01_daily_stage1_v1' and p['seeds'] == [17, 29, 43]
            and not p['full_training_enabled'] and not p['test_access_enabled'], 'Wrong stage-one contract')
    return p


def eligible(row):
    """One small per-CVE matrix, never a global expanded-window index."""
    observed = np.array([v is not None for v in row['values']])
    t = np.arange(90)[:, None]
    u = t + np.arange(1, 91)[None, :]
    day_ok = np.asarray(row['date_allowed'], dtype=bool)
    return ((u <= 90) & day_ok[t] & day_ok[np.minimum(u, 90)]
            & (np.cumsum(observed)[:90, None] > 0) & observed[np.minimum(u, 90)])


def sanitize(record, split, policy):
    out = contracts.temporal_coverage(record, split, policy)
    pub = date.fromisoformat(record['published_date'])
    dates = [(pub + timedelta(days=i)).isoformat() for i in range(91)]
    upper = policy['validation_start'] if split == 'train' else policy['test_start']
    lower = None if split == 'train' else policy['validation_start']
    allowed = [d < upper and (lower is None or d >= lower) and
               (policy['observation_end'] is None or d <= policy['observation_end']) for d in dates]
    row = {'cve': record['cve'], 'split': split, 'publication_date': record['published_date'],
           'values': out['values'], 'date_allowed': allowed}
    require(np.array_equal(eligible(row), out['eligible']), 'Lazy mask differs from frozen CS01 coverage')
    return row, out


def choose(ids, count):
    """ID-only bounded sample spanning the list, without inspecting outcomes."""
    return sorted(set(ids[:count // 2] + ids[-(count - count // 2):]))


def prepare(scale, verify=False):
    require(scale in ('smoke', 'full'), 'Unknown preflight scale')
    p = settings()
    check_lock(DESIGN, 'design_lock.json')
    identity = read(DESIGN / 'run_identity.json')
    check_pins(identity['source_pins'])
    expected = read(DESIGN / 'summary.json')['continuous']
    sources = [Path(k) for k in identity['source_pins'] if k.endswith('epss_recollected_2026-09-30_smp.json')]
    require(len(sources) == 1, 'Ambiguous frozen source')
    roles = {c: split for split, ids in identity['ids'].items() for c in ids}
    require(set(identity['ids']) == {'train', 'validation'} and len(roles) == sum(map(len, identity['ids'].values())),
            'Overlapping or forbidden roles')
    # Decode the source container once; only assigned development records are
    # retained/adapted. No test score, date, text or task-eligibility inspection.
    raw = read(sources[0])
    chosen = [r for r in raw if r['cve'] in roles]
    del raw
    require(len(chosen) == len(roles) and len({r['cve'] for r in chosen}) == len(roles), 'Missing/repeated development CVE')
    rows, counts, matrices = {}, {}, {}
    for split in ('train', 'validation'):
        rows[split] = []
        mats = {k: np.zeros((90, 90), dtype=np.int64) for k in ('eligible', 'date_admissible', 'missing_target', 'no_history')}
        for r in tqdm(chosen, desc='Verify ' + split + ' daily dates and masks', unit='CVE'):
            if roles[r['cve']] != split:
                continue
            row, coverage = sanitize(r, split, identity['split_policy'])
            for k in mats:
                mats[k] += coverage[k]
            if coverage['pair_count']:
                rows[split].append(row)
        rows[split].sort(key=lambda r: r['cve'])
        counts[split] = {'cves_with_eligible_pairs': len(rows[split]), **{k: int(v.sum()) for k, v in mats.items()}}
        require(counts[split] == expected[split], 'Coverage differs from approved CS01: ' + split)
        matrices[split] = {k: v.tolist() for k, v in mats.items()}
    del chosen
    full_counts = counts
    if scale == 'smoke':
        for split in rows:
            ids = choose([r['cve'] for r in rows[split]], p['smoke'][split + '_cves'])
            rows[split] = [r for r in rows[split] if r['cve'] in ids]
    selected_counts = {s: {'cves': len(v), 'cells': sum(int(eligible(r).sum()) for r in v),
                            'primary_cves': sum(bool(eligible(r)[30, 29]) for r in v)} for s, v in rows.items()}
    pins = dict(identity['source_pins'])
    for path in [CONFIG, DESIGN / 'design_lock.json', *Path(__file__).parent.glob('*.py'),
                 *SCRIPTS.glob('*.py'), Path(models.__file__), Path(contracts.__file__)]:
        pins[str(path)] = sha(path)
    run = {'schema': p['schema'], 'scale': scale, 'settings': p, 'source_pins': pins,
           'split_policy': identity['split_policy'], 'ids': {s: [r['cve'] for r in v] for s, v in rows.items()}}
    summary = {'full_preflight': full_counts, 'selected': selected_counts, 'training': False, 'test_adapted': False,
               'storage': 'One sanitized 91-day series per eligible development CVE; no expanded windows',
               'source_reading': 'Whole source JSON decoded for selection; only development records adapted; no test outputs'}
    text = 'CF01 daily-window preflight\n' + dump(summary)
    with transaction(OUTPUT / 'preflight' / scale, verify) as stage:
        for name, value in {'run_identity.json': run, 'series.json': rows, 'summary.json': summary,
                            'coverage.json': matrices}.items():
            (stage / name).write_text(dump(value))
        (stage / 'summary.txt').write_text(text)
        check_pins(pins)
    print(text + '\nCF01 ' + scale + ' preflight ' + ('reproduced.' if verify else 'saved.') + ' No training or test adaptation.')


def load(scale):
    directory = OUTPUT / 'preflight' / scale
    check_lock(directory)
    identity = read(directory / 'run_identity.json')
    check_pins(identity['source_pins'])
    require(identity['settings'] == settings() and identity['scale'] == scale, 'Preparation identity differs')
    rows = read(directory / 'series.json')
    for split, values in rows.items():
        require(split in ('train', 'validation') and [r['cve'] for r in values] == identity['ids'][split]
                and all(r['split'] == split for r in values), 'Forbidden role or identity mismatch')
    return rows, identity


def prefix(row, anchor):
    require(row['split'] in ('train', 'validation'), 'Test row forbidden')
    channels = contracts.past_channels(row['values'], anchor)
    last = max(j for j in range(anchor + 1) if row['values'][j] is not None)
    return channels, row['values'][last], anchor - last


def requests(row):
    mask = eligible(row)
    anchors = np.flatnonzero(mask.any(1))
    for t in anchors:
        leads = np.flatnonzero(mask[t]) + 1
        yield row, int(t), leads, 1. / len(anchors)


def packets(requests_, size):
    batch = []
    for item in requests_:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def batch(items):
    n, width = len(items), max(len(x[2]) for x in items)
    past = torch.zeros(n, max(x[1] for x in items) + 1, 5)
    lengths = torch.tensor([x[1] + 1 for x in items], dtype=torch.long)
    leads = torch.ones(n, width)
    latest = torch.zeros(n, 1, dtype=torch.float64)
    labels = torch.full((n, width), float('nan'), dtype=torch.float64)
    weights = torch.zeros(n, width, dtype=torch.float64)
    valid = torch.zeros(n, width, dtype=torch.bool)
    for i, (row, t, h, anchor_weight) in enumerate(items):
        mask = eligible(row)
        require(len(h) and len(set(map(int, h))) == len(h) and all(1 <= x <= 90-t and mask[t, x-1] for x in h),
                'Missing or date-ineligible target requested')
        channels, value, _ = prefix(row, t)
        past[i, :t+1] = torch.tensor(channels)
        latest[i, 0] = value
        leads[i, :len(h)] = torch.as_tensor(h)
        labels[i, :len(h)] = torch.tensor([row['values'][t+int(x)] for x in h], dtype=torch.float64)
        valid[i, :len(h)] = True
        weights[i, :len(h)] = anchor_weight / len(h)
    inputs = {'past': past, 'lengths': lengths, 'leads': leads, 'latest': latest}
    return inputs, labels, valid, weights


def forward(model, inputs):
    require(set(inputs) == {'past', 'lengths', 'leads', 'latest'}, 'Unexpected model input, possibly labels')
    out = model(inputs['past'], inputs['lengths'], inputs['leads'])
    # Preserve source-score precision in the persistence path while the encoder
    # uses float32. Zero correction must mean exactly persistence, not rounding.
    raw = inputs['latest'] + out['correction'].double()
    return raw, raw.clamp(0, 1)
