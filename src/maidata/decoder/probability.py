"""Stable Logistic mixtures on explicit legal intervals, including censored tails."""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F
from .checks import require


@dataclass(frozen=True)
class TimeDistribution:
    reference: float
    minimum_scale: float
    initial_centers: tuple
    initial_scales: tuple

    def validate(self, components):
        if (not math.isfinite(self.reference) or self.reference <= 0
                or not math.isfinite(self.minimum_scale) or self.minimum_scale <= 0
                or len(self.initial_centers) != components or len(self.initial_scales) != components
                or not all(math.isfinite(x) for x in (*self.initial_centers, *self.initial_scales))
                or any(s <= self.minimum_scale for s in self.initial_scales)):
            raise ValueError('invalid time distribution configuration/initialization')

    def initialize(self, head):
        self.validate(head.out_features // 3)
        with torch.no_grad():
            bias = head.bias.reshape(-1, 3)
            centers = torch.tensor(self.initial_centers, dtype=torch.float64, device=bias.device)
            scales = torch.tensor(self.initial_scales, dtype=torch.float64, device=bias.device)
            bias[:, 0].zero_()
            bias[:, 1].copy_(centers / self.reference)
            x = (scales - self.minimum_scale) / self.reference
            bias[:, 2].copy_(x + torch.log(-torch.expm1(-x)))


def interval_log_mass(raw, lower, upper, spec, *, validate=True):
    """raw [...,K,3], bounds [...,C]; upper=+inf denotes an actual tail."""
    x, a, b = raw.double(), lower.double(), upper.double()
    if validate:
        if x.shape[-1] != 3 or x.shape[:-2] != a.shape[:-1] or a.shape != b.shape:
            raise ValueError('invalid probability parameters or interval bounds')
    if validate:
        require(torch.isfinite(x).all() & torch.isfinite(a).all() & ~torch.isnan(b).any()
                & (a >= 0).all() & (b > a).all(), 'invalid probability parameters or interval bounds')
    centers = x[..., 1] * spec.reference
    scale = F.softplus(x[..., 2]) * spec.reference + spec.minimum_scale
    tail = torch.isposinf(b)
    # Keep the unselected finite-CDF branch finite as well: no inf * 0 gradients.
    finite_upper = torch.where(tail, a + 1, b)
    lo = (a[..., None] - centers.unsqueeze(-2)) / scale.unsqueeze(-2)
    width = (finite_upper - a)[..., None] / scale.unsqueeze(-2)
    finite = F.logsigmoid(lo + width) + F.logsigmoid(-lo) + torch.log(-torch.expm1(-width))
    component = torch.where(tail[..., None], F.logsigmoid(-lo), finite)
    return torch.logsumexp(x[..., 0].log_softmax(-1).unsqueeze(-2) + component, -1)


def interval_log_probabilities(raw, lower, upper, valid, spec, *, validate=True):
    """Normalize the joint finite cells + tail over BASIC support, before Physical."""
    if validate:
        if valid.shape != lower.shape or valid.dtype != torch.bool:
            raise ValueError('empty/misaligned time support')
    if validate:
        require(valid.any(-1).all(), 'empty/misaligned time support')
    a = lower.masked_fill(~valid, 0)
    b = upper.masked_fill(~valid, 1)
    mass = interval_log_mass(raw, a, b, spec, validate=validate).masked_fill(~valid, -torch.inf)
    return (mass - torch.logsumexp(mass, -1, keepdim=True)).float()


def categorical_log_probabilities(logits, allowed, *, validate=True):
    if validate:
        if allowed.shape != logits.shape or allowed.dtype != torch.bool:
            raise ValueError('empty/misaligned categorical support')
    if validate:
        require(allowed.any(-1).all(), 'empty/misaligned categorical support')
    return logits.float().masked_fill(~allowed, -torch.inf).log_softmax(-1)


def sample(log_probabilities, *, generator=None):
    """Sampling is explicit; caller decides grammar/Physical support and decoding policy."""
    return torch.multinomial(log_probabilities.exp(), 1, generator=generator).squeeze(-1)
