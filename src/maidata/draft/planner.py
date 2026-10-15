"""Shared target-only attention pools and independent 8/4 readouts."""
import torch
from torch import nn
from torch.nn import functional as F

from maidata.nn.position import apply_rope

from .attention import Attention, positive_ints, numerical_config
from .types import PlannerOutput


def select_planner_windows(z, groups, starts, valid, config, layout=None):
    from .windows import select_windows, gather_centers
    from .types import WindowLatents
    if config.get('planner_layout', 'legacy') == 'centered-overlap':
        from .spacing import integer_offsets
        centers = starts[:, None] + torch.tensor(integer_offsets(-8, 24, config['draft_z_stride_beats']), device=starts.device)
        return WindowLatents(gather_centers(z, groups, centers, valid, layout=layout),
                             starts, config['target_beats'], valid)
    if config.get('planner_layout', 'legacy') == 'window-single':
        from .spacing import integer_offsets
        centers = starts[:, None] + torch.tensor(integer_offsets(0, 16, config['draft_z_stride_beats']), device=starts.device)
        return WindowLatents(gather_centers(z, groups, centers, valid, layout=layout),
                             starts, config['target_beats'], valid)
    return select_windows(z, groups, starts, valid, target_beats=config['target_beats'],
        before_beats=0, after_beats=0, z_stride_beats=config['draft_z_stride_beats'], layout=layout)


class PlannerHead(nn.Module):
    def __init__(self, *, latent_dim, dim, heads, mlp_dim, dropout, norm_eps,
                 rope_base, rope_position_scale, span_beats=16, layout='legacy', stride_beats=4,
                 qk_normalization=False):
        super().__init__()
        positive_ints(latent_dim=latent_dim, dim=dim, heads=heads, mlp_dim=mlp_dim)
        numerical_config(dropout, norm_eps, rope_base, rope_position_scale)
        if type(span_beats) is not int or span_beats not in (4, 16):
            raise ValueError('Planner span must be 4 or 16 beats')
        self.latent_dim = latent_dim
        self.span_beats = span_beats
        if layout not in ('legacy', 'centered-overlap', 'window-single'):
            raise ValueError('unsupported Planner layout')
        if layout == 'centered-overlap' and (span_beats != 16 or stride_beats != 4):
            raise ValueError('centered-overlap Planner requires span16/stride4')
        if layout == 'window-single' and span_beats != 16:
            raise ValueError('window-single Planner requires span16')
        self.layout, self.stride_beats = layout, stride_beats
        self.query = nn.Parameter(torch.zeros(dim))
        self.pool = Attention(dim, latent_dim, dim, heads, dropout,
                              rotary=True, base=rope_base, scale=rope_position_scale,
                              qk_normalization=qk_normalization)
        self.mlp = nn.Sequential(nn.LayerNorm(dim, eps=norm_eps), nn.Linear(dim, mlp_dim),
                                 nn.GELU(), nn.Dropout(dropout))
        self.continuous = nn.Linear(mlp_dim, 8)
        self.category = nn.Linear(mlp_dim, 4)

    def forward(self, windows):
        if self.layout == 'centered-overlap':
            return self._centered(windows)
        if self.layout == 'window-single':
            return self._window_single(windows)
        if self.span_beats == 4:
            return self._four_beat(windows)
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

    def _window_single(self, windows):
        """One window statistic reads its integer anchor grid, including endpoints."""
        if windows.target_beats != 16:
            raise ValueError('window-single Planner requires sixteen-beat parent windows')
        memory, starts = windows.memory, windows.target_start
        valid = (memory.valid & windows.window_valid[:, None]
                 & (memory.beats >= starts[:, None]) & (memory.beats <= starts[:, None]+16))
        active = valid.any(-1) & windows.window_valid
        query = self.query[None, None].expand(len(starts), 1, -1)
        kv = self.pool.project_memory(memory.values, memory.beats, valid)
        # P retains its internal absolute-beat RoPE. Only M's Plan key is unpositioned.
        pooled = self.pool(query, (starts+8)[:, None], active[:, None], kv, valid)[:, 0]
        h = self.mlp(pooled)
        continuous = self.continuous(h).masked_fill(~active[:, None], 0)
        category = self.category(h).masked_fill(~active[:, None], 0)
        plan = torch.cat((continuous.float().sigmoid(), category.float().softmax(-1)), -1)
        return PlannerOutput(continuous, category, plan.masked_fill(~active[:, None], 0), active)

    def _four_beat(self, windows):
        """Project the shared Z once; each center reads only its own 4 beats."""
        if type(windows.target_beats) is not int or windows.target_beats < 4 or windows.target_beats % 4:
            raise ValueError('Four-beat Planner target length must be a positive multiple of four')
        memory, starts = windows.memory, windows.target_start
        segments = windows.target_beats // 4
        segment_starts = starts[:, None] + torch.arange(segments, device=starts.device) * 4
        centers = segment_starts + 2
        memory_valid = (memory.valid & (memory.beats >= starts[:, None])
                        & (memory.beats < starts[:, None] + windows.target_beats)
                        & windows.window_valid[:, None])
        allowed = (memory_valid[:, None, :] & (memory.beats[:, None, :] >= segment_starts[:, :, None])
                   & (memory.beats[:, None, :] < segment_starts[:, :, None] + 4))
        active = allowed.any(-1) & windows.window_valid[:, None]
        query = self.query[None, None].expand(len(starts), segments, -1)
        kv = self.pool.project_memory(memory.values, memory.beats, memory_valid)
        q = self.pool.project_query(query.masked_fill(~active[..., None], 0))
        q = apply_rope(q, centers.masked_fill(~active, 0), self.pool.base, self.pool.scale)
        pooled = F.scaled_dot_product_attention(q, *kv, attn_mask=allowed[:, None],
            dropout_p=self.pool.dropout if self.training else 0.)
        pooled = pooled.transpose(1, 2).reshape(len(starts), segments, self.pool.width)
        pooled = self.pool.output(pooled).masked_fill(~active[..., None], 0)
        h = self.mlp(pooled)
        continuous = self.continuous(h).masked_fill(~active[..., None], 0)
        category = self.category(h).masked_fill(~active[..., None], 0)
        plan = torch.cat((continuous.float().sigmoid(), category.float().softmax(-1)), -1)
        return PlannerOutput(continuous, category, plan.masked_fill(~active[..., None], 0), active)

    def _centered(self, windows):
        memory, starts = windows.memory, windows.target_start
        if windows.target_beats != 16:
            raise ValueError('centered-overlap Planner requires sixteen-beat parent windows')
        centers = starts[:, None] + torch.arange(5, device=starts.device)*4
        # Anchors must exist; available neighboring keys do not create a query
        # beyond full-curve coverage. Both read endpoints are included.
        anchor_exists = ((memory.beats[:, None] == centers[..., None])
                         & memory.valid[:, None]).any(-1)
        allowed = (memory.valid[:, None] & windows.window_valid[:, None, None]
                   & (memory.beats[:, None] >= centers[..., None]-8)
                   & (memory.beats[:, None] <= centers[..., None]+8))
        active = anchor_exists & windows.window_valid[:, None]
        query = self.query[None, None].expand(len(starts), 5, -1)
        kv = self.pool.project_memory(memory.values, memory.beats, memory.valid)
        q = self.pool.project_query(query.masked_fill(~active[..., None], 0))
        q = apply_rope(q, centers.masked_fill(~active, 0), self.pool.base, self.pool.scale)
        pooled = F.scaled_dot_product_attention(q, *kv, attn_mask=allowed[:, None] & active[:, None, :, None],
            dropout_p=self.pool.dropout if self.training else 0.)
        pooled = pooled.transpose(1, 2).reshape(len(starts), 5, self.pool.width)
        pooled = self.pool.output(pooled).masked_fill(~active[..., None], 0)
        h = self.mlp(pooled)
        continuous = self.continuous(h).masked_fill(~active[..., None], 0)
        category = self.category(h).masked_fill(~active[..., None], 0)
        plan = torch.cat((continuous.float().sigmoid(), category.float().softmax(-1)), -1)
        return PlannerOutput(continuous, category, plan.masked_fill(~active[..., None], 0), active,
                             plan_beats=centers)
