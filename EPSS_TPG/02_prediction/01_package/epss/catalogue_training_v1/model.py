"""Existing encoder, message passing and fusion; expose the binary logit."""
import torch

from ..combined_training_v2.pooled import Regressor
from .common import require


class Classifier(Regressor):
    def logits(self, inputs):
        require(set(inputs) == {'graph', 'structured', 'source_mask'}, 'Targets are not model inputs')
        graph, numeric, mask = inputs['graph'], inputs['structured'], inputs['source_mask']
        require(numeric.shape == (len(self.contract['structured_names']),)
                and mask.shape == (len(self.contract['source_order']),), 'Input dimension mismatch')
        require((graph is not None) == bool(mask.any()), 'Source mask/graph mismatch')
        text = self.encode(graph) if graph is not None else numeric.new_zeros(self.hidden)
        return self.head(torch.cat((text, self.structured(numeric), mask))).squeeze(-1)

    def forward(self, inputs):
        return torch.sigmoid(self.logits(inputs))
