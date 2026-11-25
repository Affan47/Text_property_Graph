"""Only verified CC02 predictors; labels stay in separate role-specific files."""
from copy import deepcopy
import numpy as np
from tqdm import tqdm

from ..combined_training_v1 import data as legacy
from ..final_comparison_v1.store import ResidentStore
from . import common as c
from . import cache

EPSS_NAMES = ['epss_score_or_fitting_median', 'epss_observed']


class Store(ResidentStore):
    def preload(self, ids, phase):
        super().preload(ids, phase)
        if self.representation == 'pooled':
            for cid, row in self.rows.items():
                key = self.ctx['index'][cid]['views'][self.view]['object_key']
                if key:
                    row['graph']['pooled_observed'] = self.ctx['observed'][key]
        return self


def prepare(cfg, dataset, scale, verify=False):
    base, _ = c.input_scope(cfg)
    prepared = base/'prepared'
    c.require(prepared.exists(), 'CC02 prepared inputs are not ready. No stale EPSS substitution. '
              'Complete the dated EPSS collection and 03_prepare_classification_inputs.py --stage prepare first.')
    c.frozen.check_lock(prepared)
    c.frozen.check_pins(c.read(prepared/'identity.json')['source_pins'])
    c.require(c.read(prepared/'findings.json')['gate'] == 'READY_MATCHED_INPUTS', 'CC02 input gate is closed')
    root = c.cache_location(cfg, dataset, scale)
    plan = cache.load_plan(root)
    c.frozen.check_lock(root/'verified')
    c.require(c.read(root/'verified/identity.json')['plan_sha256'] == c.sha(root/'plan/artifact_lock.json'),
              'Cache verification belongs to another plan')
    predictors = c.read(prepared/(dataset+'_predictors.json'))
    state = c.read(prepared/(dataset+'_preprocessing.json'))
    roles = c.read(prepared/'roles.json')['datasets'][dataset]
    c.require(state['fit_ids'] == roles['fitting'] == state['epss']['fitting_ids'], 'Preprocessing role leak')
    c.require(state['epss']['feature_names'] == EPSS_NAMES, 'Unexpected EPSS predictor contract')
    first = next(iter(plan['entries']))
    contract = legacy.contract(legacy.header(root/'objects'/first/'structure.npz'), plan['encoder'], state['feature_names'])
    contract['structured_names'] += EPSS_NAMES
    contract['classification_target'] = 'verified snapshot catalogue membership'
    all_ids = sorted(plan['index'])
    screened = c.read(base/'freeze'/(dataset+'_screened.json'))
    vectors, epss = {}, {}
    for cid in tqdm(all_ids, desc='Match screened text and date-aligned numerical predictors', unit='CVE'):
        row = predictors[cid]
        c.require(set(row) == {'structured', 'epss', 'epss_features', 'documents', 'mask_D', 'mask_M'},
                  'Unapproved predictor field: '+cid)
        c.require(row['documents'] == screened[cid]['documents'], 'Predictor/cache text mismatch')
        for view in cfg['views']:
            item = plan['index'][cid]['views'][view]
            wanted = [d for d in row['documents'] if d['source'] in view]
            key = item['object_key']
            c.require(cache.matching_documents(wanted, plan['entries'][key]) if key else not wanted,
                      'Screened text and graph mismatch: '+cid)
        vector = row['structured'] + row['epss_features']
        c.require(len(vector) == len(contract['structured_names']) and np.isfinite(vector).all(), 'Invalid numeric vector')
        raw = row['epss']
        c.require(raw['observed'] in (0, 1) and row['epss_features'][1] == raw['observed'], 'EPSS observation mismatch')
        c.require(row['epss_features'][0] == (raw['score'] if raw['observed'] else state['epss']['median']),
                  'EPSS imputation differs from fitting-only policy')
        vectors[cid], epss[cid] = vector, raw
    labels = c.labels_at(base, dataset)
    objects = {'predictors.json': vectors, 'epss.json': epss, 'contract.json': contract,
               'roles.json': plan['roles']}
    groups = c.read(base/'freeze/roles.json')['by_cve']
    objects['groups.json'] = {cid: groups[cid]['group'] for cid in all_ids}
    for role, members in plan['roles'].items():
        objects[role+'_labels.json'] = {cid: labels[cid] for cid in members}
    pins = c.code_pins()
    for path in (prepared/'artifact_lock.json', root/'plan/artifact_lock.json', root/'verified/artifact_lock.json'):
        pins[str(path)] = c.sha(path)
    objects['identity.json'] = {'source_pins': pins, 'dataset': dataset, 'scale': scale,
                                 'config': cfg, 'official_holdouts': False,
                                 'epss_date': c.read(prepared/'findings.json')['epss_date'],
                                 'cache_root': str(root)}
    c.publish(c.location(cfg, dataset, scale)/'inputs', objects, verify=verify)
    print(f'CC03 {dataset} {scale} classification adapter verified; labels separate; no training.')


def context(cfg, dataset, scale, assessment=False):
    output = c.location(cfg, dataset, scale)
    directory = output/'inputs'
    c.require(directory.exists(), 'Run --stage prepare first; real training needs verified CC02 inputs')
    c.frozen.check_lock(directory)
    identity = c.read(directory/'identity.json')
    c.require(identity['config'] == cfg and not identity['official_holdouts'], 'Input protocol changed')
    c.frozen.check_pins(identity['source_pins'])
    root = c.cache_location(cfg, dataset, scale)
    plan = cache.load_plan(root)
    c.frozen.check_lock(root/'verified')
    means = c.read(root/'verified/means.json')
    roles = c.read(directory/'roles.json')
    allowed_roles = ('assessment',) if assessment else ('fitting', 'selection')
    c.require(all(role in roles for role in allowed_roles), 'No assessment in smoke')
    labels = {}
    for role in allowed_roles:
        rows = c.read(directory/(role+'_labels.json'))
        c.require(set(rows) == set(roles[role]) and set(rows.values()) == {0, 1}, 'Role/label mismatch')
        labels.update(rows)
    ctx = {'directory': root, 'contract': c.read(directory/'contract.json'), 'index': plan['index'],
           'vectors': c.read(directory/'predictors.json'), 'epss': c.read(directory/'epss.json'),
           'groups': c.read(directory/'groups.json'), 'roles': roles, 'labels': labels,
           'receipts': c.read(root/'verified/receipts.json'), 'identity': identity,
           'statistics': c.read(root/'verified/statistics.json'),
           'observed': {key: row['observed'] for key, row in means.items()}, 'pooled': {}}
    for view in cfg['views']:
        ctx['pooled'][view] = {cid: means[row['views'][view]['object_key']]['vector']
                               for cid, row in ctx['index'].items() if row['views'][view]['object_key']}
    return ctx


def settings(cfg, scale):
    value = deepcopy(cfg['settings'])
    if scale == 'smoke':
        value.update({key: cfg['smoke'][key] for key in ('epochs', 'patience')})
    return value
