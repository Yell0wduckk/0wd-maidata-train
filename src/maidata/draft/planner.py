"""Shared target-only attention pools and independent 8/4 readouts."""
import torch
from torch import nn

from .attention import Attention, positive_ints, numerical_config
from .types import PlannerOutput


class PlannerHead(nn.Module):
    def __init__(self, *, latent_dim, dim, heads, mlp_dim, dropout, norm_eps,
                 rope_base, rope_position_scale, span_beats=16):
        super().__init__()
        positive_ints(latent_dim=latent_dim, dim=dim, heads=heads, mlp_dim=mlp_dim)
        numerical_config(dropout, norm_eps, rope_base, rope_position_scale)
        if type(span_beats) is not int or span_beats != 16:
            raise ValueError('Planner span must be 16 beats')
        self.latent_dim = latent_dim
        self.span_beats = span_beats
        self.query = nn.Parameter(torch.zeros(dim))
        self.pool = Attention(dim, latent_dim, dim, heads, dropout,
                              rotary=True, base=rope_base, scale=rope_position_scale)
        self.mlp = nn.Sequential(nn.LayerNorm(dim, eps=norm_eps), nn.Linear(dim, mlp_dim),
                                 nn.GELU(), nn.Dropout(dropout))
        self.continuous = nn.Linear(mlp_dim, 8)
        self.category = nn.Linear(mlp_dim, 4)

    def forward(self, windows):
        memory = windows.memory
        starts = windows.target_start
        # Enforce target-only pooling even if the caller supplies a larger context.
        valid = (memory.valid & (memory.beats >= starts[:, None])
                 & (memory.beats < starts[:, None] + windows.target_beats)
                 & windows.window_valid[:, None])
        active = valid.any(-1) & windows.window_valid
        query = self.query[None, None].expand(len(starts), 1, -1)
        kv = self.pool.project_memory(memory.values, memory.beats, valid)
        pooled = self.pool(query, starts[:, None], active[:, None], kv, valid)[:, 0]
        h = self.mlp(pooled)
        continuous = self.continuous(h).masked_fill(~active[:, None], 0)
        category = self.category(h).masked_fill(~active[:, None], 0)
        plan = torch.cat((continuous.float().sigmoid(), category.float().softmax(-1)), -1)
        return PlannerOutput(continuous, category, plan.masked_fill(~active[:, None], 0), active)
