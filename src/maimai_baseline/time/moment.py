"""MOMENT distribution and shared audio-conditioned candidate probabilities."""

from __future__ import annotations
from maimai_baseline.time.grid import AUDIO_CENTERS_PER_BEAT, AUDIO_GRID_TICKS, INTEGRATION
from torch.nn import functional as F
from typing import Mapping
import math
import torch


LINEAR_MS = INTEGRATION["schema"]


LINEAR_CONTRACT = dict(INTEGRATION, history_input="unchanged-linear-ms",
    initial_centers_ms=[0., 100., 200., 400., 800.],
    initial_scales_ms=[50., 50., 200/3, 100., 200.])


def interval_log_probabilities(raw: torch.Tensor, lower_ms: torch.Tensor,
                               upper_ms: torch.Tensor, valid: torch.Tensor | None = None, *, distribution=LINEAR_MS):
    """K5 CDF differences in milliseconds, normalized across legal cells.

    Algebraic logistic differences avoid subtracting nearly identical CDFs.
    Double precision keeps both extreme tails and very narrow cells usable.
    """
    distribution_contract(distribution)
    value = raw.to(torch.float64)
    low, high = lower_ms.to(value), upper_ms.to(value)
    if low.shape != high.shape or value.shape[:-2] != low.shape[:-1] or value.shape[-2:] != (5, 3):
        raise ValueError("BPM probability tensor shape mismatch")
    valid = torch.ones_like(low, dtype=torch.bool) if valid is None else valid.bool()
    if valid.shape != low.shape:
        raise ValueError("BPM probability support shape mismatch")
    if bool((((high <= low) | (low < 0) | ~torch.isfinite(low) | ~torch.isfinite(high)) & valid).any()) or bool((~valid.any(-1)).any()):
        raise ValueError("invalid/empty BPM probability support")
    low = torch.where(valid, low, torch.zeros_like(low))
    high = torch.where(valid, high, torch.ones_like(high))
    center = value[..., 1] * 1000.0
    scale = F.softplus(value[..., 2]) * 1000.0 + 2.5
    width = torch.where(valid, high - low, torch.ones_like(low)).unsqueeze(-1) / scale.unsqueeze(-2)
    lo = (low.unsqueeze(-1) - center.unsqueeze(-2)) / scale.unsqueeze(-2)
    hi = lo + width
    component = F.logsigmoid(hi) + F.logsigmoid(-lo) + torch.log(-torch.expm1(-width))
    mass = torch.logsumexp(torch.log_softmax(value[..., 0], -1).unsqueeze(-2) + component, -1)
    mass = mass.masked_fill(~valid, -torch.inf)
    return (mass - torch.logsumexp(mass, -1, keepdim=True)).to(torch.float32)


def candidate_probabilities(raw, table, candidate, batch_indices, *, distribution=LINEAR_MS):
    support = table["valid"]
    base = interval_log_probabilities(raw, table["lower_ms"], table["upper_ms"], support,
                                      distribution=distribution)
    if candidate is None:
        return base, {"base_log_probability": base}
    keys = candidate["keys"].index_select(0, batch_indices).float()
    query = candidate["query"].float()
    source_score = torch.bmm(query[:, None], keys.transpose(1, 2)).squeeze(1) / math.sqrt(query.shape[-1])
    beats = candidate["beats"].index_select(0, batch_indices)
    # The encoder still has 24 centers per beat. Query all 96 time candidates
    # at their actual beat positions without rounding four candidates together.
    origin_k = torch.round(beats[:, :1].double() * AUDIO_CENTERS_PER_BEAT)
    positions = table["ticks"].double() / AUDIO_GRID_TICKS - origin_k
    in_range = (positions >= 0) & (positions <= keys.shape[1] - 1)
    left = positions.floor().long()
    alpha = (positions - left).float()
    padding = candidate["padding_mask"].index_select(0, batch_indices)
    def neighbor(indices):
        valid = (indices >= 0) & (indices < keys.shape[1])
        index = indices.clamp(0, keys.shape[1] - 1)
        return source_score.gather(1, index).masked_fill(~valid | padding.gather(1, index), 0)
    score = ((1 - alpha) * neighbor(left) + alpha * neighbor(left + 1)).masked_fill(~support | ~in_range, 0)
    gate = candidate["gate"].float()
    attenuation = ((table["delta_ms"].float() - 25) / 15).clamp(0, 1)
    first = table["first"][:, None]
    attenuation = torch.where(first, torch.ones_like(attenuation), attenuation).masked_fill(~support, 0)
    protected = support & attenuation.eq(0)
    remaining = support & ~protected
    all_protected = ~remaining.any(1, keepdim=True)
    mass = (1 - base.exp().masked_fill(~protected, 0).sum(1, keepdim=True)).clamp_min(torch.finfo(base.dtype).tiny)
    def adjust(g):
        full = torch.log_softmax((base + g * score).masked_fill(~support, -torch.inf), 1)
        difference = full.masked_fill(~support, 0) - base.masked_fill(~support, 0)
        remaining_logits = (base + attenuation * difference).masked_fill(~remaining, -torch.inf)
        remaining_logits = torch.where(all_protected, torch.zeros_like(remaining_logits), remaining_logits)
        shape = torch.log_softmax(remaining_logits, 1)
        mixed = torch.where(protected, base, shape + mass.log()).masked_fill(~support, -torch.inf)
        return torch.where(first, full, torch.where(all_protected, base, mixed))
    regular = adjust(gate)
    effective = gate + (torch.ones_like(gate) - gate).detach()
    enhanced = adjust(effective).masked_fill(~support, 0)
    straight = base.detach() + (enhanced - enhanced.detach())
    final = torch.where(gate == 0, straight, regular).masked_fill(~support, -torch.inf)
    detail = {"base_log_probability": base, "candidate_score": score,
              "candidate_attenuation": attenuation, "gate": gate,
              "score_variance": score.var(1, unbiased=False),
              "k5_final_kl": (final.exp() * (final - base).masked_fill(~support, 0)).sum(1).clamp_min(0)}
    return final, detail


def distribution_contract(name=LINEAR_MS):
    if name != LINEAR_MS:
        raise ValueError(f"unsupported MOMENT distribution: {name!r}; v9 requires linear milliseconds")
    return dict(INTEGRATION)


def generation_clock(name=LINEAR_MS):
    distribution_contract(name)
    return "absolute-chart-tick1-centers-linear-ms-cdf"


def validate_checkpoint_distribution(payload: Mapping):
    records=(payload, payload.get("baseline_manifest", {}), payload.get("baseline_contract", {}))
    if any(record.get("moment_distribution") != LINEAR_CONTRACT for record in records):
        raise ValueError("MOMENT distribution identity mismatch")
