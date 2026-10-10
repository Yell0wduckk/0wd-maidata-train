"""Dense consumer-owned projections, optional beat RoPE, safe empty masks."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from maidata.nn.position import apply_rope


def positive_ints(**values):
    if any(type(v) is not int or v < 1 for v in values.values()):
        raise ValueError(f'positive integer dimensions required: {values}')


def numerical_config(dropout, norm_eps, base, scale):
    if not 0 <= dropout <= 1 or not math.isfinite(norm_eps) or norm_eps <= 0:
        raise ValueError('invalid dropout or LayerNorm epsilon')
    if not math.isfinite(base) or base <= 0 or not math.isfinite(scale):
        raise ValueError('invalid position base/scale')


class Attention(nn.Module):
    def __init__(self, query_dim, memory_dim, width, heads, dropout, *, rotary, base, scale):
        super().__init__()
        positive_ints(query_dim=query_dim, memory_dim=memory_dim, width=width, heads=heads)
        if width % heads or (rotary and (width // heads) % 2):
            raise ValueError('attention width/heads must be integral; RoPE head width must be even')
        self.width, self.heads, self.dropout = width, heads, dropout
        self.rotary, self.base, self.scale = rotary, base, scale
        self.query = nn.Linear(query_dim, width)
        self.key = nn.Linear(memory_dim, width)
        self.value = nn.Linear(memory_dim, width)
        self.output = nn.Linear(width, query_dim)

    def split(self, x):
        return x.reshape(*x.shape[:2], self.heads, self.width // self.heads).transpose(1, 2)

    def project_memory(self, values, beats, valid):
        values = values.masked_fill(~valid[..., None], 0)
        k, v = self.split(self.key(values)), self.split(self.value(values))
        if self.rotary:
            k = apply_rope(k, beats.masked_fill(~valid, 0), self.base, self.scale)
        return k, v

    def forward(self, query, beats, valid, kv, memory_valid):
        q = self.split(self.query(query.masked_fill(~valid[..., None], 0)))
        if self.rotary:
            q = apply_rope(q, beats.masked_fill(~valid, 0), self.base, self.scale)
        allowed = valid[:, None, :, None] & memory_valid[:, None, None, :]
        h = F.scaled_dot_product_attention(q, *kv, attn_mask=allowed,
                                          dropout_p=self.dropout if self.training else 0.)
        h = h.transpose(1, 2).reshape(*query.shape[:2], self.width)
        active = valid & memory_valid.any(-1, keepdim=True)
        return self.output(h).masked_fill(~active[..., None], 0)
