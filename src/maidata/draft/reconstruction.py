"""Independent local field queries and per-beat counts reading Z only."""
import torch
from torch import nn
from torch.nn import functional as F

from maidata.nn.position import sinusoidal, apply_rope
from maidata.time.discrete import TIME_CLASSES
from .attention import Attention, positive_ints, numerical_config
from .schema import FIELD_SIZES, TIME_FIELDS, COUNT_CLASSES, QUERY_ROLES, KINDS
from .types import ReconstructionOutput


class ReconstructionLayer(nn.Module):
    def __init__(self, dim, heads, ffn_dim, dropout, eps, base, scale):
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(dim, eps=eps) for _ in range(2))
        self.cross_attention = Attention(dim, dim, dim, heads, dropout,
                                         rotary=True, base=base, scale=scale)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn_dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, beats, active, memory, memory_beats, memory_valid, indices, local_valid):
        attention = self.cross_attention
        # Project each window's Z once per layer, then select each query's slots.
        k, v = attention.project_memory(memory, memory_beats, memory_valid)
        n, q, slots = indices.shape
        batch = torch.arange(n, device=query.device)[:, None, None]
        k = k.transpose(1, 2)[batch, indices].reshape(n*q, slots, attention.heads, -1).transpose(1, 2)
        v = v.transpose(1, 2)[batch, indices].reshape(n*q, slots, attention.heads, -1).transpose(1, 2)
        x = self.norms[0](query).reshape(n*q, 1, -1)
        h = attention(x, beats.reshape(n*q, 1), active.reshape(n*q, 1),
                      (k, v), local_valid.reshape(n*q, slots)).reshape(n, q, -1)
        query = query + self.dropout(h)
        return (query + self.dropout(self.ffn(self.norms[1](query)))).masked_fill(~active[..., None], 0)


class ReconstructionHead(nn.Module):
    def __init__(self, *, latent_dim, layers, dim, heads, ffn_dim, dropout, norm_eps,
                 target_beats, position_base, position_scale):
        super().__init__()
        positive_ints(latent_dim=latent_dim, layers=layers, dim=dim, heads=heads,
                      ffn_dim=ffn_dim, target_beats=target_beats)
        numerical_config(dropout, norm_eps, position_base, position_scale)
        self.latent_dim, self.dim, self.target_beats = latent_dim, dim, target_beats
        self.base, self.scale = position_base, position_scale
        self.input = nn.Linear(latent_dim, dim) if latent_dim != dim else nn.Identity()
        self.role = nn.Embedding(len(QUERY_ROLES)+1, dim)
        self.note_kind = nn.Embedding(len(KINDS)+1, dim)
        self.route = nn.Embedding(71, dim)
        self.prefix = nn.Linear(8, dim)
        self.layers = nn.ModuleList(ReconstructionLayer(dim, heads, ffn_dim, dropout,
            norm_eps, position_base, position_scale) for _ in range(layers))
        self.norm = nn.LayerNorm(dim, eps=norm_eps)
        self.field_heads = nn.ModuleDict({'field_' + name: nn.Linear(dim, n) for name, n in FIELD_SIZES.items()})
        self.time_heads = nn.ModuleDict({name: nn.Linear(dim, TIME_CLASSES) for name in TIME_FIELDS})
        self.count_head = nn.Linear(dim, COUNT_CLASSES)

    def forward(self, windows, queries):
        m, q = windows.memory, queries
        active = windows.window_valid & m.valid.any(-1)
        valid = m.valid & active[:, None]
        memory = self.input(m.values.masked_fill(~valid[..., None], 0)).masked_fill(~valid[..., None], 0)
        scalars = torch.stack((q.origins-windows.target_start[:, None], q.elapsed,
            q.ordinals.float(), q.tracks.float(), q.route_valid.sum(-1).float(), q.history_gaps,
            q.boundary.float(), q.last_real_valid.float()), -1)
        scalars = scalars.sign()*scalars.abs().log1p()
        query = self.role(q.roles) + self.note_kind(q.note_kinds+1) + self.prefix(scalars.to(memory))
        route_positions = torch.arange(q.route_ids.shape[-1], device=memory.device).float()
        ordered_routes = self.route(q.route_ids) * (1+sinusoidal(route_positions, self.dim, self.base, self.scale).to(memory))
        query = query + ordered_routes.masked_fill(~q.route_valid[..., None], 0).sum(-2)
        query = query + sinusoidal(q.beats-windows.target_start[:, None], self.dim, self.base, self.scale).to(memory)
        count_beats = windows.target_start[:, None] + torch.arange(self.target_beats, device=memory.device).float()+.5
        count_query = self.role.weight[-1][None, None] + sinusoidal(count_beats-windows.target_start[:, None], self.dim, self.base, self.scale).to(memory)
        query = torch.cat((query, count_query), 1)
        beats = torch.cat((q.beats, count_beats), 1)
        qvalid = torch.cat((q.valid & active[:, None], active[:, None].expand(-1, self.target_beats)), 1)
        moment = torch.cat((q.roles == QUERY_ROLES.index('MOMENT'), torch.zeros_like(count_beats, dtype=torch.bool)), 1)
        offsets = torch.arange(5, device=memory.device)
        desired = beats.floor()[..., None] + torch.where(moment[..., None], offsets, offsets-1) + .5
        index = torch.searchsorted(m.beats.contiguous(), desired.flatten(1).contiguous()).reshape_as(desired)
        safe = index.clamp_max(m.beats.shape[1]-1)
        owner = torch.arange(len(memory), device=memory.device)[:, None, None]
        local_valid = (index < m.beats.shape[1]) & valid[owner, safe] & (m.beats[owner, safe] == desired)
        local_valid &= moment[..., None] | (offsets < 3)
        local_valid &= qvalid[..., None]
        for layer in self.layers:
            query = layer(query, beats, qvalid, memory, m.beats, valid, safe, local_valid)
        hidden = self.norm(query).masked_fill(~qvalid[..., None], 0)
        rows = hidden[:, :q.valid.shape[1]]
        return ReconstructionOutput(
            {name.removeprefix('field_'): head(rows).masked_fill(~qvalid[:, :rows.shape[1], None], 0)
             for name, head in self.field_heads.items()},
            {name: head(rows).masked_fill(~qvalid[:, :rows.shape[1], None], 0) for name, head in self.time_heads.items()},
            self.count_head(hidden[:, rows.shape[1]:]).masked_fill(~active[:, None, None], 0), active)
