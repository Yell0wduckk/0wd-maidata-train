"""Seven semantic radar-axis tokens; their coordinates are not chart times."""
from dataclasses import dataclass

import torch
from torch import nn

from maimai_baseline.radar_contract import DIMENSION, normalize


@dataclass(frozen=True)
class RadarContext:
    tokens: torch.Tensor
    valid: torch.Tensor
    values: torch.Tensor
    enabled: bool = True
    rotary: bool = False

    def __post_init__(self):
        if (self.tokens.shape != (*self.valid.shape, 128)
                or self.valid.shape[1:] != (DIMENSION,) or self.valid.dtype != torch.bool
                or self.values.shape != self.valid.shape):
            raise ValueError('radar context requires seven semantic 128-wide axis tokens')

    @property
    def beats(self):
        # The compatibility property supports cache bookkeeping. Radar attention
        # never rotates queries or keys by these zero coordinates.
        return self.tokens.new_zeros(self.valid.shape, dtype=torch.float32)

    def signature(self):
        return (self.enabled, self.rotary, *((id(v), v._version, v.device, v.dtype, tuple(v.shape))
                for v in (self.tokens, self.valid, self.values)))


class RadarEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.axis = nn.Embedding(DIMENSION, 128)
        self.value = nn.Linear(1, 128)

    def forward(self, values, valid=None):
        if values.ndim != 2 or values.shape[-1] != DIMENSION:
            raise ValueError('radar values require [window,7]')
        if valid is None:
            valid = torch.ones(len(values), dtype=torch.bool, device=values.device)
        if valid.shape != values.shape[:1] or valid.dtype != torch.bool:
            raise ValueError('radar window validity mismatch')
        axes = torch.arange(DIMENSION, device=values.device)
        tokens = self.axis(axes)[None] + self.value(normalize(values).unsqueeze(-1))
        mask = valid[:, None].expand_as(values)
        return RadarContext(tokens.masked_fill(~mask[..., None], 0.), mask,
                            values.masked_fill(~mask, 0.))
