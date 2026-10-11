"""Content readouts; no masks, rules, resources or sequence state are owned here."""
import torch
from torch import nn
from torch.nn import functional as F
from maidata.time.discrete import TIME_CLASSES


class ConditionalReadout(nn.Module):
    def __init__(self, dim, middle, eps):
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(dim, eps=eps), nn.Linear(dim, middle), nn.GELU(), nn.Linear(middle, dim))

    def forward(self, h, condition):
        x = h + condition
        return x + self.mlp(x)


class ContentHeads(nn.Module):
    def __init__(self, dim, middle, norm_eps):
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
        self.times = nn.ModuleDict({name: nn.Linear(dim, TIME_CLASSES) for name in
                                   ('MOMENT', 'HOLD_DURATION', 'SLIDE_DURATION')})

    def pad_logits(self, h, kind):
        shape = h.shape[:-1]
        h, kind = h.reshape(-1, h.shape[-1]), kind.reshape(-1)
        x = self.pad_adapter(h, self.kind_condition(kind))
        touch = (kind == 2) | (kind == 3)
        logits = torch.where(touch[:, None], self.touch(x), F.pad(self.button(x), (0, 25), value=-torch.inf)).to(x.dtype)
        return logits.reshape(*shape, 33)

    def flag_logits(self, h, kind, pad):
        shape = h.shape[:-1]
        h, kind, pad = h.reshape(-1, h.shape[-1]), kind.reshape(-1), pad.reshape(-1)
        touch = (kind == 2) | (kind == 3)
        c = torch.where(touch[:, None], self.touch_condition(pad), self.button_condition(pad.clamp_max(7)))
        x = self.flags_adapter(h, self.kind_condition(kind) + c)
        return torch.stack([head(x) for head in self.flags], -2).reshape(*shape, 5, 2)

    def break_logits(self, h, route):
        return self.track_break(self.break_adapter(h, self.route_condition(route)))

    def time_parameters(self, h, kind):
        head = self.times[kind]
        # Match the original time readout: keep linear output precise under AMP.
        with torch.autocast(device_type=h.device.type, enabled=False):
            value = F.linear(h.float(), head.weight.float(), head.bias.float())
        return value

    time_logits = time_parameters
