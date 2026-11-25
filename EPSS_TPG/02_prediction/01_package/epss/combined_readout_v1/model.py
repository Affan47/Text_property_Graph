"""Matched 32-value readouts; no new graph or encoder parameters."""
import torch
from torch import nn
from .data import require


class Regressor(nn.Module):
    def __init__(self, structured_dim, arm):
        super().__init__()
        require(arm in ('merged', 'source'), 'Unknown readout arm')
        self.arm = arm
        self.structured = nn.Sequential(nn.Linear(structured_dim, 32), nn.LayerNorm(32), nn.GELU())
        self.head = nn.Sequential(nn.Linear(68, 32), nn.GELU(), nn.Linear(32, 1))
        widths = [32] if arm == 'merged' else [16, 16]
        self.text = nn.ModuleList(nn.Sequential(nn.Linear(768, w), nn.LayerNorm(w), nn.GELU()) for w in widths)

    def forward(self, inputs):
        require(set(inputs) == {'merged', 'source', 'observed', 'mask', 'structured'}, 'Unexpected predictor; labels must remain separate')
        require(inputs['merged'].shape == (768,) and inputs['source'].shape == (2, 768) and
                inputs['observed'].shape == (2,) and inputs['mask'].shape == (4,), 'Wrong readout shape')
        if self.arm == 'merged':
            text = self.text[0](inputs['merged'])*inputs['observed'].any().to(inputs['merged'])
        else:
            text = torch.cat([project(inputs['source'][i])*inputs['observed'][i] for i, project in enumerate(self.text)])
        return torch.sigmoid(self.head(torch.cat([text, self.structured(inputs['structured']), inputs['mask']]))).squeeze(-1)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
