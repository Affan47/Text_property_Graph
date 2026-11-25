"""Fixed final-comparison models; no new graph arithmetic or encoder training."""
from copy import deepcopy
import random

import numpy as np
import torch

from ..combined_training_v1.data import require
from ..combined_training_v2.pooled import Regressor


MODELS = {
    'structured': ('structured', 'S'),
    'pooled_D': ('pooled', 'D'),
    'tpg_D': ('tpg', 'D'),
    'pooled_DM': ('pooled', 'DM'),
    'tpg_DM': ('tpg', 'DM'),
}
SEEDS = (17, 29, 43, 59, 71)
SETTINGS = {
    'hidden': 32, 'layers': 2, 'readout': 'mean', 'edge_chunk_size': 8192,
    'device': 'cpu', 'threads': 4, 'objective': 'mse',
    'learning_rate': 0.001, 'weight_decay': 0.0001,
    'batch_size': 8, 'epochs': 100, 'patience': 15,
}


def settings(identity, epochs_override=None):
    """Extra protocol fields are retained by the caller, not used as model knobs."""
    declared = identity.get('settings', {})
    for key, expected in SETTINGS.items():
        require(key not in declared or declared[key] == expected,
                'Final comparison setting is fixed: ' + key)
    if 'seeds' in declared:
        require(list(declared['seeds']) == list(SEEDS), 'Final comparison seeds changed')
    require(epochs_override is None or type(epochs_override) is int and epochs_override == 1,
            'Only the declared one-epoch integration override is supported')
    result = deepcopy(SETTINGS)
    if epochs_override is not None:
        result['epochs'] = epochs_override
    return result


def initialize(contract, model, seed, parameters=None):
    require(model in MODELS, 'Unknown final-comparison model: ' + str(model))
    require(type(seed) is int and seed in SEEDS, 'Unregistered final-comparison seed')
    parameters = deepcopy(parameters if parameters is not None else SETTINGS)
    for key in ('hidden', 'layers', 'readout', 'edge_chunk_size', 'device', 'threads'):
        require(parameters[key] == SETTINGS[key], 'Architecture/execution setting changed: ' + key)
    require(contract['encoder_identity'].get('trainable') is False,
            'Final comparison requires frozen embeddings')
    require(contract['edge_feature_dim'] == 88 and contract['embedding_dim'] == 768,
            'Unexpected frozen graph feature dimensions')
    require(contract['source_order'] == 'DMGN', 'Source-mask order changed')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    # Reuses typed edges, all 88 edge properties, node types and entity subtypes.
    # The pooled path uses the identical frozen TOKEN mean supplied by the adapter.
    return Regressor(contract, parameters, MODELS[model][0])
