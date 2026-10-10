"""Linear bit projections; no combination lookup embeddings."""
import torch
from torch import nn
from maimai_dataset.modifiers import ROLES

class ModifierProjection(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.projection = nn.Linear(len(ROLES), width, bias=False)

    def forward(self, ids):
        bits = torch.stack([ids["modifier_" + n] for n in ROLES], -1)
        return self.projection(bits.to(self.projection.weight.dtype))
