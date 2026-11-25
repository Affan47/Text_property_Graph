"""Matched pooled predictors; no target or identifier enters forward()."""
import numpy as np
import torch
from torch import nn


class Regressor(nn.Module):
    def __init__(self, structured_dim, hidden=32):
        super().__init__()
        self.text = nn.Sequential(nn.Linear(768, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.structured = nn.Sequential(nn.Linear(structured_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2*hidden+4, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        if set(x) != {'text', 'structured', 'source_mask'}:
            raise ValueError('Unexpected predictor keys; targets must remain separate')
        return torch.sigmoid(self.head(torch.cat((self.text(x['text']), self.structured(x['structured']),
                                                 x['source_mask']), dim=1))).squeeze(1)


def inputs(arrays, role, arm):
    return {k: torch.from_numpy(np.asarray(arrays[role+suffix], dtype=np.float32)) for k, suffix in
            [('text', '_'+arm), ('structured', '_structured'), ('source_mask', '_'+arm+'_mask')]}


def predict(model, x, batch_size=256):
    model.eval()
    with torch.no_grad():
        return torch.cat([model({k:v[i:i+batch_size] for k,v in x.items()})
                          for i in range(0, len(x['text']), batch_size)]).numpy().astype(float)
