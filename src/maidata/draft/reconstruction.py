"""Fixed rows/count queries reading Z only. There is no row self-attention."""
import torch
from torch import nn

from maidata.nn.position import sinusoidal
from .attention import Attention, positive_ints, numerical_config
from .schema import FIELD_SIZES, TIME_FIELDS, COUNT_CLASSES
from .types import ReconstructionOutput, validate_memory


class ReconstructionLayer(nn.Module):
    def __init__(self, dim, heads, ffn_dim, dropout, eps, base, scale):
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(dim, eps=eps) for _ in range(2))
        self.cross_attention = Attention(dim, dim, dim, heads, dropout,
                                         rotary=False, base=base, scale=scale)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn_dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, memory, valid, active):
        qv = active[:, None].expand(query.shape[:2])
        kv = self.cross_attention.project_memory(memory, None, valid)
        query = query + self.dropout(self.cross_attention(self.norms[0](query), None, qv, kv, valid))
        return (query + self.dropout(self.ffn(self.norms[1](query)))).masked_fill(~qv[..., None], 0)


class ReconstructionHead(nn.Module):
    def __init__(self, *, latent_dim, layers, dim, heads, ffn_dim, dropout, norm_eps,
                 query_rows, target_beats, position_base, position_scale):
        super().__init__()
        positive_ints(latent_dim=latent_dim, layers=layers, dim=dim, heads=heads,
                      ffn_dim=ffn_dim, query_rows=query_rows, target_beats=target_beats)
        numerical_config(dropout, norm_eps, position_base, position_scale)
        self.latent_dim, self.dim, self.query_rows = latent_dim, dim, query_rows
        self.target_beats = target_beats
        self.base, self.scale = position_base, position_scale
        self.input = nn.Linear(latent_dim, dim) if latent_dim != dim else nn.Identity()
        self.layers = nn.ModuleList(ReconstructionLayer(dim, heads, ffn_dim, dropout,
            norm_eps, position_base, position_scale) for _ in range(layers))
        self.norm = nn.LayerNorm(dim, eps=norm_eps)
        self.field_heads = nn.ModuleDict({'field_' + name: nn.Linear(dim, n) for name, n in FIELD_SIZES.items()})
        self.time_heads = nn.ModuleDict({name: nn.Linear(dim, 1) for name in TIME_FIELDS})
        self.count_head = nn.Linear(dim, COUNT_CLASSES)

    def forward(self, windows):
        m = windows.memory
        validate_memory(m, self.latent_dim)
        if windows.target_beats != self.target_beats:
            raise ValueError('R target length differs from its count query capacity')
        active = windows.window_valid & m.valid.any(-1)
        valid = m.valid & active[:, None]
        relative = (m.beats - windows.target_start[:, None]).masked_fill(~valid, 0)
        memory = (self.input(m.values.masked_fill(~valid[..., None], 0))
                  + sinusoidal(relative, self.dim, self.base, self.scale).to(m.values))
        memory = memory.masked_fill(~valid[..., None], 0)
        positions = torch.cat((torch.arange(self.query_rows, device=m.values.device).float(),
                               torch.arange(self.target_beats, device=m.values.device).float() + .5))
        query = sinusoidal(positions, self.dim, self.base, self.scale).to(memory)
        query = query[None].expand(len(memory), -1, -1)
        for layer in self.layers:
            query = layer(query, memory, valid, active)
        hidden = self.norm(query)
        rows = hidden[:, :self.query_rows]
        return ReconstructionOutput(
            {name.removeprefix('field_'): head(rows).masked_fill(~active[:, None, None], 0)
             for name, head in self.field_heads.items()},
            {name: head(rows).squeeze(-1).masked_fill(~active[:, None], 0) for name, head in self.time_heads.items()},
            self.count_head(hidden[:, self.query_rows:]).masked_fill(~active[:, None, None], 0), active)
