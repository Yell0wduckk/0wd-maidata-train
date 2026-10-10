"""Content readouts; no masks, rules, resources or sequence state are owned here."""
import torch
from torch import nn


class ConditionalReadout(nn.Module):
    def __init__(self, dim, middle, eps):
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(dim, eps=eps), nn.Linear(dim, middle), nn.GELU(), nn.Linear(middle, dim))

    def forward(self, h, condition):
        x = h + condition
        return x + self.mlp(x)


class ContentHeads(nn.Module):
    def __init__(self, dim, middle, norm_eps, components):
        super().__init__()
        self.kind = nn.Linear(dim, 5)
        self.route = nn.Linear(dim, 71)
        self.button = nn.Linear(dim, 8)
        self.touch = nn.Linear(dim, 33)
        self.flags = nn.ModuleList(nn.Linear(dim, 2) for _ in range(5))
        self.track_break = nn.Linear(dim, 2)
        self.kind_condition = nn.Embedding(5, dim)
        self.button_condition = nn.Embedding(8, dim)
        self.touch_condition = nn.Embedding(33, dim)
        self.route_condition = nn.Embedding(71, dim)
        self.pad_adapter = ConditionalReadout(dim, middle, norm_eps)
        self.flags_adapter = ConditionalReadout(dim, middle, norm_eps)
        self.break_adapter = ConditionalReadout(dim, middle, norm_eps)
        self.times = nn.ModuleDict({name: nn.Linear(dim, 3 * components) for name in
                                   ('MOMENT', 'HOLD_DURATION', 'SLIDE_DURATION')})
        self.components = components

    def pad_logits(self, h, kind):
        shape = h.shape[:-1]
        h, kind = h.reshape(-1, h.shape[-1]), kind.reshape(-1)
        x = self.pad_adapter(h, self.kind_condition(kind))
        touch = (kind == 2) | (kind == 3)
        logits = x.new_full((*x.shape[:-1], 33), -torch.inf)
        logits[~touch, :8] = self.button(x[~touch]).to(logits.dtype)
        logits[touch] = self.touch(x[touch]).to(logits.dtype)
        return logits.reshape(*shape, 33)

    def flag_logits(self, h, kind, pad):
        shape = h.shape[:-1]
        h, kind, pad = h.reshape(-1, h.shape[-1]), kind.reshape(-1), pad.reshape(-1)
        touch = (kind == 2) | (kind == 3)
        c = torch.zeros_like(h)
        c[~touch] = self.button_condition(pad[~touch])
        c[touch] = self.touch_condition(pad[touch])
        x = self.flags_adapter(h, self.kind_condition(kind) + c)
        return torch.stack([head(x) for head in self.flags], -2).reshape(*shape, 5, 2)

    def break_logits(self, h, route):
        return self.track_break(self.break_adapter(h, self.route_condition(route)))

    def time_parameters(self, h, kind):
        return self.times[kind](h).reshape(*h.shape[:-1], self.components, 3)
