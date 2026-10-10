"""Consumer-owned projections and immutable K/V; no producer model imports."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from maidata.nn.position import apply_rope
from .types import KV, Memory, validate_memory


class Attention(nn.Module):
    def __init__(self, dim, memory_dim, heads, dropout, base, scale):
        super().__init__()
        self.dim, self.heads, self.dropout = dim, heads, dropout
        self.base, self.scale = base, scale
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(memory_dim, dim)
        self.value = nn.Linear(memory_dim, dim)
        self.output = nn.Linear(dim, dim)

    def split(self, x):
        return x.reshape(*x.shape[:2], self.heads, self.dim // self.heads).transpose(1, 2)

    def project(self, memory, *, plan_index=None):
        x = memory.values.masked_fill(~memory.valid[..., None], 0)
        raw = self.split(self.key(x))
        k = apply_rope(raw, memory.beats.masked_fill(~memory.valid, 0), self.base, self.scale)
        if plan_index is not None:
            k = torch.cat((k[:, :, :plan_index], raw[:, :, plan_index:]), 2)
        return KV(k, self.split(self.value(x)), memory.valid)

    def forward(self, x, beats, valid, kv, *, plan_index=None, causal_offset=None):
        raw = self.split(self.query(x.masked_fill(~valid[..., None], 0)))
        q = apply_rope(raw, beats.masked_fill(~valid, 0), self.base, self.scale)
        allowed = valid[:, None, :, None] & kv.valid[:, None, None, :]
        if causal_offset is not None:
            causal = (torch.arange(kv.key.shape[2], device=x.device)[None, :]
                      <= causal_offset + torch.arange(x.shape[1], device=x.device)[:, None])
            allowed = allowed & causal[None, None]
        if plan_index is None:
            y = F.scaled_dot_product_attention(q, kv.key, kv.value, attn_mask=allowed,
                                               dropout_p=self.dropout if self.training else 0.)
        else:
            # Unrotated q AND k for plan. A single softmax joins the two sources.
            scores = torch.cat((q.float() @ kv.key[:, :, :plan_index].float().transpose(-1, -2),
                                raw.float() @ kv.key[:, :, plan_index:].float().transpose(-1, -2)), -1)
            scores = scores / math.sqrt(q.shape[-1])
            active = allowed.any(-1, keepdim=True)
            scores = scores.masked_fill(~allowed, -torch.inf).masked_fill(~active, 0)
            weights = scores.softmax(-1).masked_fill(~allowed, 0).to(kv.value.dtype)
            y = F.dropout(weights, self.dropout, self.training) @ kv.value
        y = y.transpose(1, 2).reshape(*x.shape[:2], self.dim)
        return self.output(y).masked_fill(~(valid & kv.valid.any(-1, keepdim=True))[..., None], 0)

    def self_attention(self, x, beats, valid, previous):
        current = self.project(Memory(x, beats, valid))
        offset = 0 if previous is None else previous.valid.shape[1]
        kv = current if previous is None else KV(torch.cat((previous.key, current.key), 2),
                    torch.cat((previous.value, current.value), 2), torch.cat((previous.valid, valid), 1))
        return self(x, beats, valid, kv, causal_offset=offset), kv


class ConditionAdapter(nn.Module):
    def __init__(self, latent_dim, dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.z = nn.Linear(latent_dim, dim)
        self.plan = nn.Sequential(nn.Linear(12, dim), nn.GELU(), nn.Linear(dim, dim))
        self.source = nn.Embedding(2, dim)

    def forward(self, z, plan, plan_valid):
        validate_memory(z, self.latent_dim)
        if (plan.shape != (len(z.values), 12) or plan_valid.shape != plan.shape[:1]
                or plan_valid.dtype != torch.bool or not torch.isfinite(plan[plan_valid]).all()):
            raise ValueError('plan requires [B,12] and independent validity [B]')
        a = self.z(z.values.masked_fill(~z.valid[..., None], 0)) + self.source.weight[0]
        b = self.plan(plan.masked_fill(~plan_valid[:, None], 0)) + self.source.weight[1]
        return Memory(torch.cat((a, b[:, None]), 1),
                      torch.cat((z.beats, z.beats.new_zeros((len(plan), 1))), 1),
                      torch.cat((z.valid, plan_valid[:, None]), 1))
