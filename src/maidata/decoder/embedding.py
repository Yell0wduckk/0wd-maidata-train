"""Field input embeddings; NA vectors have no parameters."""
import torch
from torch import nn
from maidata.chart.types import APPLICABLE, FLAGS, KINDS
from maidata.nn.position import sinusoidal
from .checks import require


class TokenEmbedding(nn.Module):
    def __init__(self, dim, note_dims, route_dims, time_base, time_scale):
        super().__init__()
        self.validate_inputs = True
        if (len(note_dims) != 7 or len(route_dims) != 2
                or any(type(v) is not int or v <= 0 for v in (*note_dims, *route_dims))
                or sum(note_dims) != dim or sum(route_dims) != dim):
            raise ValueError('NOTE/ROUTE embedding quotas must each sum to decoder_dim')
        self.dim, self.base, self.scale = dim, time_base, time_scale
        self.kind = nn.Embedding(5, note_dims[0])
        self.button = nn.Embedding(8, note_dims[1])
        self.touch = nn.Embedding(33, note_dims[1])
        self.flags = nn.ModuleList(nn.Embedding(2, d) for d in note_dims[2:])
        self.route = nn.Embedding(71, route_dims[0])
        self.track_break = nn.Embedding(2, route_dims[1])
        self.input_type = nn.Embedding(8, dim)
        self.register_buffer('applicable', torch.tensor(
            [[f in APPLICABLE[k] for f in FLAGS] for k in KINDS]), persistent=False)

    def forward(self, tokens):
        t, valid = tokens.types, tokens.valid
        if (self.validate_inputs and (t.ndim != 2 or valid.shape != t.shape or valid.dtype != torch.bool
                or tokens.beats.shape != t.shape or tokens.time.shape != t.shape
                or tokens.flags.shape != (*t.shape, 5))):
            raise ValueError('invalid token tensor shapes')
        for x in (t, tokens.kind, tokens.pad, tokens.route, tokens.track_break):
            if self.validate_inputs and (x.shape != t.shape or x.dtype != torch.long or x.device != t.device):
                raise ValueError('token IDs require aligned int64 tensors')
        if self.validate_inputs and (tokens.flags.dtype != torch.long):
            raise ValueError('invalid token type/flag dtype')
        if self.validate_inputs:
            require(((t >= 0) & (t <= 7) | ~valid).all(), 'invalid token type')
        if self.validate_inputs:
            require(((torch.isfinite(tokens.beats) & torch.isfinite(tokens.time)) | ~valid).all(),
                    'valid token coordinates must be finite')
        note = valid & (t == 1)
        k, p, f = tokens.kind, tokens.pad, tokens.flags
        if self.validate_inputs:
            require(((k >= 0) & (k < 5) | ~note).all(), 'invalid NOTE kind')
        k = k.clamp(0, 4)
        touch = (k == 2) | (k == 3)
        if self.validate_inputs:
            require(((p >= 0) & (p < torch.where(touch, 33, 8)) | ~note).all(), 'invalid NOTE pad')
        applicable = self.applicable[k]
        if self.validate_inputs:
            require((torch.where(applicable, (f >= 0) & (f <= 1), f == -1) | ~note[..., None]).all(),
                    'NOTE flags disagree with kind applicability')
        pad = torch.where(touch[..., None], self.touch(p.clamp(0, 32)), self.button(p.clamp(0, 7)))
        parts = [self.kind(k), pad]
        for i, table in enumerate(self.flags):
            parts.append(table(f[..., i].clamp(0, 1)).masked_fill(~applicable[..., i, None], 0))
        notes = torch.cat(parts, -1)
        route = valid & (t == 2)
        r, b = tokens.route, tokens.track_break
        if self.validate_inputs:
            require(((r >= 0) & (r <= 70) & (b >= -1) & (b <= 1) | ~route).all(), 'invalid ROUTE value')
        routes = torch.cat((self.route(r.clamp(0, 70)),
            self.track_break(b.clamp(0, 1)).masked_fill(b[..., None] < 0, 0)), -1)
        times = sinusoidal(tokens.time.masked_fill(~valid, 0), self.dim, self.base, self.scale).to(notes.dtype)
        out = torch.where(note[..., None], notes, torch.where(route[..., None], routes, times))
        return (out + self.input_type(t.clamp(0, 7))).masked_fill(~valid[..., None], 0)
