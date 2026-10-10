"""Field input embeddings; NA vectors have no parameters."""
import torch
from torch import nn
from maidata.chart.types import APPLICABLE, FLAGS, KINDS
from maidata.nn.position import sinusoidal


class TokenEmbedding(nn.Module):
    def __init__(self, dim, note_dims, route_dims, time_base, time_scale):
        super().__init__()
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
        self.input_type = nn.Embedding(5, dim)
        self.register_buffer('applicable', torch.tensor(
            [[f in APPLICABLE[k] for f in FLAGS] for k in KINDS]), persistent=False)

    def forward(self, tokens):
        t, valid = tokens.types, tokens.valid
        if (t.ndim != 2 or valid.shape != t.shape or valid.dtype != torch.bool
                or tokens.beats.shape != t.shape or tokens.time.shape != t.shape
                or tokens.flags.shape != (*t.shape, 5)):
            raise ValueError('invalid token tensor shapes')
        for x in (t, tokens.kind, tokens.pad, tokens.route, tokens.track_break):
            if x.shape != t.shape or x.dtype != torch.long or x.device != t.device:
                raise ValueError('token IDs require aligned int64 tensors')
        if tokens.flags.dtype != torch.long or ((t[valid] < 0) | (t[valid] > 4)).any():
            raise ValueError('invalid token type/flag dtype')
        if not torch.isfinite(tokens.beats[valid]).all() or not torch.isfinite(tokens.time[valid]).all():
            raise ValueError('valid token coordinates must be finite')
        out = self.kind.weight.new_zeros((*t.shape, self.dim))
        note = valid & (t == 1)
        k, p, f = tokens.kind[note], tokens.pad[note], tokens.flags[note]
        if ((k < 0) | (k >= 5)).any():
            raise ValueError('invalid NOTE kind')
        touch = (k == 2) | (k == 3)
        if ((p < 0) | (p >= torch.where(touch, 33, 8))).any():
            raise ValueError('invalid NOTE pad')
        applicable = self.applicable[k]
        if ((f[applicable] < 0) | (f[applicable] > 1)).any() or (f[~applicable] != -1).any():
            raise ValueError('NOTE flags disagree with kind applicability')
        pad = self.button.weight.new_zeros((len(k), self.button.embedding_dim))
        pad[~touch], pad[touch] = self.button(p[~touch]), self.touch(p[touch])
        parts = [self.kind(k), pad]
        for i, table in enumerate(self.flags):
            parts.append(table(f[:, i].clamp_min(0)).masked_fill(~applicable[:, i, None], 0))
        out[note] = torch.cat(parts, -1)
        route = valid & (t == 2)
        r, b = tokens.route[route], tokens.track_break[route]
        if ((r < 0) | (r > 70) | (b < -1) | (b > 1)).any():
            raise ValueError('invalid ROUTE value')
        out[route] = torch.cat((self.route(r), self.track_break(b.clamp_min(0)).masked_fill(b[:, None] < 0, 0)), -1)
        time = valid & ((t == 0) | (t == 3) | (t == 4))
        out[time] = sinusoidal(tokens.time[time], self.dim, self.base, self.scale).to(out.dtype)
        return (out + self.input_type(t.masked_fill(~valid, 0))).masked_fill(~valid[..., None], 0)
