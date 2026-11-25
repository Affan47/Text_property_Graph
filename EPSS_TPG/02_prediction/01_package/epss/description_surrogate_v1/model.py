"""No structured branch, identifiers, source flags or EPSS in forward()."""
import numpy as np
import torch
from torch import nn


class DescriptionRegressor(nn.Module):
    def __init__(self,hidden=32,input_dim=768):
        super().__init__()
        self.input_dim=input_dim
        self.text=nn.Sequential(nn.Linear(input_dim,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.head=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,1))

    def forward(self,x):
        if set(x)!={'text'} or x['text'].ndim!=2 or x['text'].shape[1]!=self.input_dim:
            raise ValueError('Model accepts only the description vector')
        return torch.sigmoid(self.head(self.text(x['text']))).squeeze(1)


def predict(net,x,batch_size=256):
    net.eval()
    with torch.no_grad():
        return torch.cat([net({'text':x[i:i+batch_size]}) for i in range(0,len(x),batch_size)]).numpy().astype(np.float64)
