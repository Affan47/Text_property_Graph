"""Day-index EPSS forecasting data contract. No training or text/graph ingestion.

Scores are dated observations, not proof of their exact historical release time.
The existing graph-classification dataset must not consume this format.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

VERSION = "epss_endpoint_v1"
INPUT_LAST_DAY = 30
TARGET_LAST_DAY = 60
EPSILON = Decimal("0.01")
STAT_FEATURES = ("last_score", "mean", "population_std", "slope_per_day",
                 "observed_days", "last_observation_age_days", "slope_defined")
TASKS = ("endpoint", "delta", "direction", "minmax", "trajectory")
PREPARED_FILES = ("inputs.json", "labels.json", "split_manifest.json", "duplicate_groups.json",
                  "training_preprocessor.json", "preparation_summary.json", "split_policy.json",
                  "reviewed_url_exclusions.json", "grouping_evidence.json", "task_cohorts.json")
DEFAULT_SPLIT_POLICY = {"validation_start": "2024-01-01", "test_start": "2025-01-01",
                        "test_end": "2026-01-01", "observation_end": None}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    with Path(path).open('rb') as handle:
        require(not handle.read(100).startswith(b'version https://git-lfs'),
                f"{path}: Git LFS pointer, not a downloaded dataset")
    def reject(value):
        raise ValueError(f"Invalid JSON number {value}")
    def unique_pairs(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, f"Duplicate JSON key: {key}")
            out[key] = value
        return out
    return json.loads(Path(path).read_text(), parse_constant=reject, object_pairs_hook=unique_pairs)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def timestamp(value):
    require(isinstance(value, str) and "T" in value, "Expected publication timestamp with time")
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def score(value):
    require(value is None or type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1,
            f"EPSS must be a finite number in [0,1] or null, got {value!r}")
    return value


def score_days(record):
    cve = record.get("cve")
    require(isinstance(cve, str) and re.fullmatch(r"CVE-\d{4}-\d{4,}", cve), "Invalid canonical CVE ID")
    publication = timestamp(record["published_timestamp"])
    require(record["published_date"] == publication.date().isoformat(), f"{cve}: publication fields disagree")
    dates = [(publication.date() + timedelta(days=i)).isoformat() for i in range(91)]
    values = record.get("epss_scores")
    require(isinstance(values, dict) and set(values) == set(dates), f"{cve}: require exactly Day 0-90 date keys")
    return dates, [score(values[d]) for d in dates]


def history_statistics(values):
    observed = [(i, v) for i, v in enumerate(values) if v is not None]
    if not observed:
        return None
    xs, ys = zip(*observed)
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    denominator = sum((x - mean_x)**2 for x in xs)
    slope = sum((x - mean_x)*(y - mean_y) for x,y in observed) / denominator if denominator else 0.0
    return dict(zip(STAT_FEATURES, (ys[-1], mean_y, statistics.pstdev(ys), slope,
                                    len(ys), INPUT_LAST_DAY - xs[-1], int(len(xs) >= 2))))


def make_record(record):
    dates, values = score_days(record)
    past = values[:INPUT_LAST_DAY + 1]
    future = values[INPUT_LAST_DAY + 1:TARGET_LAST_DAY + 1]
    age, last_day = [], None
    for day, value in enumerate(past):
        if value is not None:
            last_day = day
        age.append(None if last_day is None else day - last_day)
    inputs = {
        "cve": record["cve"],
        "publication_date": record["published_date"],
        "input_end_date": dates[INPUT_LAST_DAY],
        "time_policy": "score_date_only_release_time_unverified",
        "X": {"dates": dates[:INPUT_LAST_DAY + 1], "day_offsets": list(range(INPUT_LAST_DAY + 1)),
              "values": past, "observed_mask": [int(v is not None) for v in past],
              "age_since_observation_days": age, "age_known_mask": [int(v is not None) for v in age],
              "statistics": history_statistics(past)},
    }
    endpoint = values[TARGET_LAST_DAY]
    delta = float(Decimal(str(endpoint)) - Decimal(str(past[-1]))) if endpoint is not None and past[-1] is not None else None
    change = Decimal(str(delta)) if delta is not None else None
    direction = None if change is None else "increase" if change > EPSILON else "decrease" if change < -EPSILON else "stable"
    complete = all(v is not None for v in future)
    target_values = {"endpoint": endpoint, "delta": delta, "direction": direction,
                     "minmax": [min(future), max(future)] if complete else None,
                     "trajectory": future}
    label_mask = {"endpoint": endpoint is not None, "delta": delta is not None,
                  "direction": direction is not None, "minmax": complete,
                  "trajectory": any(v is not None for v in future)}
    has_input = any(v is not None for v in past)
    labels = {"cve": record["cve"], "target_date": dates[TARGET_LAST_DAY],
              "future_dates": dates[INPUT_LAST_DAY+1:TARGET_LAST_DAY+1],
              "y": target_values, "label_mask": label_mask,
              "trajectory_observed_mask": [int(v is not None) for v in future],
              "epss_input_available": has_input,
              "eligible": {task: has_input and mask for task,mask in label_mask.items()},
              "epsilon": str(EPSILON)}
    return inputs, labels


def predictor_view(record, modalities=("epss",)):
    """Fail closed: future TPG/text integration needs a separately reviewed contract."""
    require(tuple(modalities) == ("epss",), "Only EPSS inputs are enabled; historical text/TPG/CVSS is not verified")
    x = record["X"]
    allowed = {"dates", "day_offsets", "values", "observed_mask", "age_since_observation_days", "age_known_mask", "statistics"}
    require(set(x) == allowed, "Unexpected predictor fields")
    require(len(x['values']) == 31 and x['day_offsets'] == list(range(31)), "Invalid input window")
    expected_dates = [(date.fromisoformat(record['publication_date']) + timedelta(days=i)).isoformat() for i in range(31)]
    require(x['dates'] == expected_dates and record['input_end_date'] == expected_dates[-1], "Input dates extend outside Day 0-30")
    require(x['observed_mask'] == [int(v is not None) for v in x['values']], "Score/mask mismatch")
    ages, previous = [], None
    for day,value in enumerate(x['values']):
        if value is not None:
            previous = day
        ages.append(None if previous is None else day - previous)
    require(x['age_since_observation_days'] == ages and x['age_known_mask'] == [int(a is not None) for a in ages], "Age/mask mismatch")
    require(x['statistics'] == history_statistics(x['values']), "Statistics are not derived from past scores")
    for value in x['values']:
        score(value)
    return {key: x[key] for key in sorted(allowed)}


def sequence_arrays(record):
    """Numeric storage only: zero padding is distinguishable from observed zero."""
    x = predictor_view(record)
    return {"values": [0.0 if v is None else v for v in x['values']],
            "observed_mask": list(x['observed_mask']),
            "age_days": [0 if v is None else v for v in x['age_since_observation_days']],
            "age_known_mask": list(x['age_known_mask']), "day_offsets": list(x['day_offsets'])}


def normalized_url(url):
    if not isinstance(url, str) or not url.strip():
        return None
    parsed = urlsplit(url.strip())
    if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
        return None
    # Fragments are not different documents; preserve paths and query identifiers.
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", parsed.query, ""))


def validated_split_policy(policy=None):
    if policy is None:
        policy = DEFAULT_SPLIT_POLICY
    require(isinstance(policy, dict) and set(policy) == set(DEFAULT_SPLIT_POLICY),
            "Split policy must explicitly contain validation_start, test_start, test_end and observation_end")
    out = dict(policy)
    for key, value in out.items():
        if key == 'observation_end' and value is None:
            continue
        require(isinstance(value, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}', value), f"Invalid split date: {key}")
        date.fromisoformat(value)
    require(out['validation_start'] < out['test_start'] < out['test_end'], "Split boundaries must be strictly increasing")
    require(out['observation_end'] is None or out['observation_end'] >= out['test_start'],
            "Observation cutoff precedes the test period")
    return out


def validate_url_exclusions(exclusions=None):
    exclusions = {} if exclusions is None else exclusions
    require(isinstance(exclusions, dict), "Reviewed URL exclusions must map exact URLs to written reasons")
    for url, reason in exclusions.items():
        require(isinstance(url, str) and normalized_url(url) == url,
                "URL exclusions must use exact normalized http(s) URLs, not hosts or patterns")
        require(isinstance(reason, str) and bool(reason.strip()), "Every URL exclusion needs a review reason")
    return dict(sorted(exclusions.items()))


def grouping_keys(record):
    text_keys = [("description", record.get(key)) for key in ("description", "orig_description")]
    text_keys += [("post", p.get('post')) for p in record.get('social_media_posts', [])]
    keys = {(kind, " ".join(text.lower().split())) for kind,text in text_keys if isinstance(text, str) and text.strip()}
    keys |= {("url", norm) for u in record.get('all_sources_urls', []) if (norm := normalized_url(u))}
    return keys


def duplicate_groups(records, url_exclusions=None, evidence=None):
    """Transitive, split-only links: descriptions, source URLs and exact post text."""
    parent = {r['cve']: r['cve'] for r in records}
    ignored = validate_url_exclusions(url_exclusions)
    evidence_ids = defaultdict(list)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            low, high = sorted((a, b))
            parent[high] = low
    first = {}
    reasons = Counter()
    for r in sorted(records, key=lambda r:r['cve']):
        for key in sorted(grouping_keys(r)):
            evidence_ids[key].append(r['cve'])
            if key[0] == 'url' and key[1] in ignored:
                continue
            if key in first and first[key] != r['cve']:
                union(r['cve'], first[key])
                reasons[key[0]] += 1
            else:
                first[key] = r['cve']
    groups = defaultdict(list)
    for cve in sorted(parent):
        groups[find(cve)].append(cve)
    by_cve, members = {}, {}
    for ids in groups.values():
        gid = "group_" + hashlib.sha256("\n".join(ids).encode()).hexdigest()[:20]
        members[gid] = ids
        for cve in ids:
            by_cve[cve] = gid
    require(set(ignored) <= {key[1] for key in evidence_ids if key[0] == 'url'},
            "Reviewed URL exclusion not found in this snapshot; check spelling and provenance")
    if evidence is not None:
        for (kind, value), ids in sorted(evidence_ids.items()):
            if len(ids) < 2:
                continue
            is_ignored = kind == 'url' and value in ignored
            parsed = urlsplit(value) if kind == 'url' else None
            evidence.append({'kind': kind, 'key_sha256': hashlib.sha256((kind+'\0'+value).encode()).hexdigest(),
                             'value_or_excerpt': value if kind == 'url' else value[:240],
                             'cves': ids, 'cve_count': len(ids), 'used_for_grouping': not is_ignored,
                             'review_reason': ignored.get(value) if is_ignored else None,
                             'homepage_review_hint': bool(parsed and parsed.path in ('','/') and not parsed.query)})
    return by_cve, members, dict(reasons)


def candidate_split(input_end, target_end, policy=None):
    policy = validated_split_policy(policy)
    anchor, end = date.fromisoformat(input_end), date.fromisoformat(target_end)
    require(anchor < end, "Target must occur after prediction anchor")
    if policy['observation_end'] and end > date.fromisoformat(policy['observation_end']):
        return "excluded", "target_after_observation_end"
    if anchor < date.fromisoformat(policy['validation_start']):
        return ("train", "") if end < date.fromisoformat(policy['validation_start']) else ("excluded", "label_crosses_train_boundary")
    if anchor < date.fromisoformat(policy['test_start']):
        return ("validation", "") if end < date.fromisoformat(policy['test_start']) else ("excluded", "label_crosses_validation_boundary")
    if anchor < date.fromisoformat(policy['test_end']):
        return "test", ""
    return "excluded", "later_anchor_not_primary_test"


def split_manifest(inputs, labels, groups, policy=None):
    policy = validated_split_policy(policy)
    manifest = {}
    roles = defaultdict(set)
    for cve in sorted(inputs):
        role, reason = candidate_split(inputs[cve]['input_end_date'], labels[cve]['target_date'], policy)
        manifest[cve] = {"cve": cve, "group_id": groups[cve], "candidate_split": role,
                         "split": role, "exclusion_reason": reason,
                         "input_end_date": inputs[cve]['input_end_date'], "target_date": labels[cve]['target_date']}
        if role != "excluded":
            roles[groups[cve]].add(role)
    conflicts = {gid for gid, folds in roles.items() if len(folds) > 1}
    for row in manifest.values():
        if row['group_id'] in conflicts and row['split'] != "excluded":
            row['split'] = "excluded"
            row['exclusion_reason'] = "related_group_crosses_time_folds"
    return manifest, sorted(conflicts)


def validate_prepared(inputs, labels, manifest, policy=None, groups=None):
    policy = validated_split_policy(policy)
    require(set(inputs) == set(labels) == set(manifest), "CVE alignment mismatch")
    folds = defaultdict(set)
    for cve in inputs:
        predictor_view(inputs[cve])
        lab, row = labels[cve], manifest[cve]
        require(lab['cve'] == row['cve'] == inputs[cve]['cve'] == cve, "CVE identity mismatch")
        require(isinstance(row['group_id'], str) and row['group_id'], "Missing group identity")
        if groups is not None:
            require(groups.get(cve) == row['group_id'], "Manifest group differs from reconstructed evidence")
        require(set(lab['y']) == set(TASKS) and set(lab['label_mask']) == set(TASKS) and set(lab['eligible']) == set(TASKS), "Unknown or missing task")
        require(lab['epsilon'] == str(EPSILON), "Unexpected direction tolerance")
        require(lab['epss_input_available'] == any(inputs[cve]['X']['observed_mask']), "Input availability mismatch")
        require(all(lab['eligible'][t] == (lab['epss_input_available'] and lab['label_mask'][t]) for t in TASKS), "Eligibility mismatch")
        require(lab['label_mask']['endpoint'] == (lab['y']['endpoint'] is not None), "Missing endpoint became a target")
        require(lab['label_mask']['minmax'] == all(lab['trajectory_observed_mask']), "Incomplete extrema target")
        expected_dates = [(date.fromisoformat(inputs[cve]['publication_date']) + timedelta(days=i)).isoformat() for i in range(31,61)]
        require(lab['future_dates'] == expected_dates and lab['target_date'] == expected_dates[-1], "Target dates are not Days 31-60")
        require(len(lab['y']['trajectory']) == 30 and lab['y']['endpoint'] == lab['y']['trajectory'][-1], "Endpoint/trajectory mismatch")
        require(lab['trajectory_observed_mask'] == [int(v is not None) for v in lab['y']['trajectory']], "Future mask mismatch")
        for value in lab['y']['trajectory']:
            score(value)
        require(lab['label_mask']['trajectory'] == any(lab['trajectory_observed_mask']), "Trajectory eligibility mismatch")
        anchor = inputs[cve]['X']['values'][-1]
        endpoint = lab['y']['endpoint']
        delta = None if anchor is None or endpoint is None else float(Decimal(str(endpoint))-Decimal(str(anchor)))
        direction = None if delta is None else 'increase' if Decimal(str(delta)) > EPSILON else 'decrease' if Decimal(str(delta)) < -EPSILON else 'stable'
        require(lab['y']['delta'] == delta and lab['y']['direction'] == direction, "Delta/direction mismatch")
        require(lab['label_mask']['delta'] == (delta is not None) and lab['label_mask']['direction'] == (direction is not None), "Delta/direction mask mismatch")
        if lab['label_mask']['minmax']:
            require(lab['y']['minmax'] == [min(lab['y']['trajectory']), max(lab['y']['trajectory'])], "Extrema mismatch")
        else:
            require(lab['y']['minmax'] is None, "Partial future must not become exact extrema")
        require(lab['target_date'] > inputs[cve]['input_end_date'], "Nonfuture target")
        if 'input_end_date' in row:
            require(row['input_end_date'] == inputs[cve]['input_end_date'] and row['target_date'] == lab['target_date'],
                    "Manifest dates differ from inputs/labels")
        expected, reason = candidate_split(inputs[cve]['input_end_date'], lab['target_date'], policy)
        require(row['candidate_split'] == expected, "Forged candidate split")
        require(row['split'] in ('train','validation','test','excluded'), "Unknown split role")
        if row['split'] != 'excluded':
            require(expected == row['split'] and row['exclusion_reason'] == '', "Chronological boundary violation")
            folds[row['group_id']].add(row['split'])
        else:
            require(row['exclusion_reason'] == (reason if expected == 'excluded' else 'related_group_crosses_time_folds'),
                    "Incorrect exclusion reason")
    require(all(len(roles) == 1 for roles in folds.values()), "Related group occurs in multiple folds")
    expected_manifest, _ = split_manifest(inputs, labels, {c:r['group_id'] for c,r in manifest.items()}, policy)
    for cve, row in manifest.items():
        require(all(row[k] == expected_manifest[cve][k] for k in ('candidate_split','split','exclusion_reason')),
                "Split manifest does not match deterministic policy")


def fit_training_preprocessor(inputs, labels, manifest, task="endpoint", policy=None):
    require(task in TASKS, "Unknown task")
    validate_prepared(inputs, labels, manifest, policy)
    ids = sorted(cve for cve,row in manifest.items() if row['split'] == 'train' and labels[cve]['eligible'][task])
    require(bool(ids), "No eligible training rows for preprocessing")
    columns = [[inputs[cve]['X']['statistics'][f] for cve in ids] for f in STAT_FEATURES]
    return {"task": task, "fit_split": "train", "fit_cve_ids": ids, "feature_names": list(STAT_FEATURES),
            "means": [statistics.mean(col) for col in columns],
            "scales": [statistics.pstdev(col) or 1.0 for col in columns],
            "method": "Training population mean/SD; constant feature scale is 1. No target values used."}


def transform_statistics(record, preprocessor):
    x = predictor_view(record)
    require(x['statistics'] is not None, "No observed inputs")
    require(preprocessor['fit_split'] == 'train' and preprocessor['feature_names'] == list(STAT_FEATURES), "Invalid preprocessor contract")
    require(len(preprocessor['means']) == len(preprocessor['scales']) == len(STAT_FEATURES), "Wrong preprocessor dimensions")
    require(all(type(v) in (int,float) and math.isfinite(v) for v in preprocessor['means']) and
            all(type(v) in (int,float) and math.isfinite(v) and v > 0 for v in preprocessor['scales']),
            "Invalid preprocessor means or scales")
    return [(x['statistics'][f] - mean) / scale for f,mean,scale in
            zip(STAT_FEATURES, preprocessor['means'], preprocessor['scales'])]


def prepare(records, policy=None, url_exclusions=None, evidence=None):
    policy = validated_split_policy(policy)
    url_exclusions = validate_url_exclusions(url_exclusions)
    require(isinstance(records, list) and records, "Source must be a nonempty CVE list")
    ids = [r['cve'] for r in records]
    require(len(ids) == len(set(ids)), "Duplicate CVE IDs: refuse to drop or merge rows")
    inputs, labels = {}, {}
    for record in sorted(records, key=lambda r:r['cve']):
        inputs[record['cve']], labels[record['cve']] = make_record(record)
    groups, members, links = duplicate_groups(records, url_exclusions, evidence)
    manifest, conflicts = split_manifest(inputs, labels, groups, policy)
    validate_prepared(inputs, labels, manifest, policy, groups)
    counts = Counter({task: 0 for task in TASKS})
    folds = {role: Counter({"all_cves": 0, **{t:0 for t in TASKS}}) for role in ('train', 'validation', 'test', 'excluded')}
    for cve, lab in labels.items():
        counts.update({task:int(yes) for task,yes in lab['eligible'].items()})
        folds[manifest[cve]['split']].update(all_cves=1, **{task:int(yes) for task,yes in lab['eligible'].items()})
    summary = {"contract_version": VERSION, "total_cves": len(records), "eligible_before_splits": dict(counts),
               "split_policy": policy, "reviewed_url_exclusions": url_exclusions,
               "splits": {k:dict(v) for k,v in folds.items()},
               "split_exclusion_reasons": dict(Counter(r['exclusion_reason'] for r in manifest.values() if r['split']=='excluded')),
               "duplicate_groups": {"groups": len(members), "non_singleton_groups": sum(len(v)>1 for v in members.values()),
                                    "largest_group": max(map(len, members.values())), "cross_fold_groups_excluded": len(conflicts),
                                    "shared_key_links": links,
                                    "policy": "Transitive links from normalized current/original descriptions, exact normalized post text, or same normalized reference URL, except explicitly reviewed exact URL exclusions recorded separately. Quarantine all active-fold rows of cross-fold groups; do not move future records into training.",
                                    "limitation": "Conservative known-group protection, not a proof that all semantic near-duplicates are detected. Generic links may over-group."},
               "direction_epsilon": str(EPSILON),
               "direction_classes_before_splits": dict(Counter(l['y']['direction'] for l in labels.values() if l['eligible']['direction'])),
               "direction_classes_by_split": {role: {name: sum(manifest[c]['split'] == role and l['eligible']['direction'] and l['y']['direction'] == name for c,l in labels.items())
                                                    for name in ('increase', 'stable', 'decrease')} for role in ('train','validation','test','excluded')},
               "input_missing_endpoint_usable": sum(l['eligible']['endpoint'] and 0 in inputs[cve]['X']['observed_mask'] for cve,l in labels.items()),
               "input_present_endpoint_missing": sum(l['epss_input_available'] and not l['label_mask']['endpoint'] for l in labels.values()),
               "training_performed": False, "text_inputs_enabled": False, "vckev_targets_enabled": False,
               "ready_for_historical_deployment_claim": False,
               "limitations": ["Day-index eligibility only: daily release times and retained numerical scores require provenance checks.",
                               "Default split is the previously proposed chronological candidate, not a performance-selected split.",
                               "Text and reference metadata are used only for conservative grouping where stated, never predictors.",
                               "Exact exploitation or future catalogue-entry labels are not constructed."]}
    return inputs, labels, manifest, members, summary


def task_cohorts(labels, manifest):
    return {task: {role: sorted(c for c in labels if labels[c]['eligible'][task] and manifest[c]['split'] == role)
                   for role in ('train','validation','test')} for task in TASKS}


def save_prepared(directory, artifacts, source_hash):
    """The lock is the last write: a partial replacement fails hash verification."""
    directory = Path(directory)
    require(set(artifacts) == set(PREPARED_FILES), "Incomplete prepared bundle")
    for filename, value in artifacts.items():
        write_json(directory / filename, value)
    lock = {'schema': 'epss_preparation_lock_v1', 'source_sha256': source_hash,
            'contract_version': VERSION, 'baseline_evaluated': False, 'model_trained': False,
            'files': {name: sha256(directory/name) for name in PREPARED_FILES}}
    write_json(directory/'preparation_lock.json', lock)


def load_prepared(directory, verify_source=True):
    """Validate the bundle before a future trainer or baseline consumes it."""
    directory = Path(directory)
    lock = read_json(directory/'preparation_lock.json')
    require(lock['schema'] == 'epss_preparation_lock_v1' and lock['contract_version'] == VERSION,
            "Unsupported preparation lock")
    require(set(lock['files']) == set(PREPARED_FILES), "Incomplete artifact lock")
    for filename, checksum in lock['files'].items():
        require(sha256(directory/filename) == checksum, f"Prepared file changed: {filename}")
    data = {name:read_json(directory/name) for name in PREPARED_FILES}
    x, y, manifest = (data[name] for name in ('inputs.json','labels.json','split_manifest.json'))
    summary, policy = data['preparation_summary.json'], data['split_policy.json']
    require(summary['source_sha256'] == lock['source_sha256'], "Source hash mismatch")
    require(summary['split_policy'] == policy, "Split policy mismatch")
    require(summary['reviewed_url_exclusions'] == data['reviewed_url_exclusions.json'], "Grouping review mismatch")
    group_map = {}
    for gid, members in data['duplicate_groups.json'].items():
        require(isinstance(members,list) and members == sorted(set(members)) and members, "Invalid group members")
        require(gid == 'group_'+hashlib.sha256('\n'.join(members).encode()).hexdigest()[:20], "Group identity changed")
        for cve in members:
            require(cve not in group_map, "CVE assigned to multiple groups")
            group_map[cve] = gid
    require(set(group_map) == set(x), "Grouping coverage mismatch")
    validate_prepared(x,y,manifest,policy,group_map)
    require(data['task_cohorts.json'] == task_cohorts(y,manifest), "Task cohort mismatch")
    require(data['training_preprocessor.json'] == fit_training_preprocessor(x,y,manifest,policy=policy),
            "Preprocessor differs from training-only fit")
    if verify_source:
        require(sha256(summary['source_path']) == lock['source_sha256'], "Raw source changed")
        raw = read_json(summary['source_path'])
        rebuilt, members, _ = duplicate_groups(raw, data['reviewed_url_exclusions.json'])
        require(rebuilt == group_map and members == data['duplicate_groups.json'], "Grouping does not match raw evidence")
        require(len(raw) == len(x) == len({r['cve'] for r in raw}), "Raw CVE coverage mismatch")
        for record in raw:
            expected_x, expected_y = make_record(record)
            require(expected_x == x[record['cve']] and expected_y == y[record['cve']], "Inputs or labels differ from raw scores")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--split-policy', type=Path, help='Explicit chronological boundaries and optional score-date observation cutoff')
    parser.add_argument('--reviewed-url-exclusions', type=Path, help='Exact normalized URLs mapped to written review reasons; no automatic filtering')
    parser.add_argument('--overwrite', action='store_true', help='Explicitly replace an existing derived preparation, never raw data')
    args = parser.parse_args()
    require(args.output_dir.resolve() != args.source.parent.resolve(), "Keep derived outputs outside the raw source directory")
    require(args.overwrite or not args.output_dir.exists() or not any(args.output_dir.iterdir()),
            "Output directory is not empty; use --overwrite after reviewing the existing artifacts")
    policy = validated_split_policy(read_json(args.split_policy) if args.split_policy else None)
    exclusions = validate_url_exclusions(read_json(args.reviewed_url_exclusions) if args.reviewed_url_exclusions else None)
    raw_hash = sha256(args.source)
    evidence = []
    records = read_json(args.source)
    inputs, labels, manifest, groups, summary = prepare(records, policy, exclusions, evidence)
    preprocessor = fit_training_preprocessor(inputs, labels, manifest, policy=policy)
    summary['source_sha256'] = raw_hash
    summary['source_path'] = str(args.source.resolve())
    summary['adapter_sha256'] = sha256(__file__)
    require(sha256(args.source) == raw_hash, "Raw input changed during processing")
    artifacts = {"inputs.json": inputs, "labels.json": labels, "split_manifest.json": manifest,
                 "duplicate_groups.json": groups, "training_preprocessor.json": preprocessor, "preparation_summary.json": summary,
                 "split_policy.json": policy, "reviewed_url_exclusions.json": exclusions,
                 "grouping_evidence.json": evidence, "task_cohorts.json": task_cohorts(labels,manifest)}
    save_prepared(args.output_dir, artifacts, raw_hash)
    load_prepared(args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
