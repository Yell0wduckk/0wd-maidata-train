"""Hold and Slide distributions, supports, and initialization formulas."""

from __future__ import annotations
from maimai_baseline.time.coordinates import seconds_at_ticks
from torch.nn import functional as F
from typing import Mapping
import numpy as np
import torch


HOLD = "hold_duration_tick"


SLIDE = "branch_duration_tick"


FIELDS = (HOLD, SLIDE)


REFERENCES = {HOLD: 1000., SLIDE: 96.}


LINEAR_DURATION = "maimai.duration_bucket4_written_bpm.v3"


LINEAR_CENTERS = {HOLD: (100., 200., 400., 800., 1600.), SLIDE: (0., 24., 48., 96., 192.)}


LINEAR_SCALES = {HOLD: (100., 100., 400/3, 200., 400.), SLIDE: (12., 12., 16., 24., 48.)}


MINIMUM_SCALES = {HOLD: 2.5, SLIDE: .25}


LINEAR_CONTRACT = {'schema': LINEAR_DURATION,
 'components': 5,
 'scale': 'softplus(raw_scale) * reference + minimum_scale',
 'fields': {'hold_duration_tick': {'coordinate': 'duration_ms / 1000',
                                   'reference': 1000.0,
                                   'initial_centers': [100.0, 200.0, 400.0, 800.0, 1600.0],
                                   'initial_scales': [100.0,
                                                      100.0,
                                                      133.33333333333334,
                                                      200.0,
                                                      400.0],
                                   'minimum_scale': 2.5,
                                   'minimum_tick': 2},
            'branch_duration_tick': {'coordinate': 'duration_tick / 96',
                                     'reference': 96.0,
                                     'initial_centers': [0.0, 24.0, 48.0, 96.0, 192.0],
                                     'initial_scales': [12.0, 12.0, 16.0, 24.0, 48.0],
                                     'minimum_scale': 0.25,
                                     'minimum_tick': 1}},
 'cells': '4tick-centers;half-up-target;first-last-half-tick-support-edges',
 'bucket_ticks': 4,
 'upper_source': 'audio-end-seconds-minus-global-wait;written-bpm-duration',
 'hold_timing': 'duration-ticks-times-written-bpm-period',
 'upper_boundary': 'last-legal-duration-tick-plus-half',
 'probability': 'mixture-cdf-difference-divided-by-legal-support-mass',
 'zero_lower_bound': 'finite-zero-cdf',
 'boundary_dtype': 'float64',
 'probability_dtype': 'float32',
 'history_input': 'unchanged-linear-tick-divided-by-96',
 'audio_rescoring': False}


def distribution_contract(name=LINEAR_DURATION):
    if name != LINEAR_DURATION:
        raise ValueError("unsupported duration distribution; v10 requires finite audio-end bucket4 time")
    return dict(LINEAR_CONTRACT)


def distribution_name(contract):
    if not isinstance(contract, Mapping) or contract != distribution_contract(contract.get("schema")):
        raise ValueError("invalid duration distribution identity")
    return contract["schema"]


def linear_interval_mass(raw, lower, upper, *, field):
    value = raw.double()
    a, b = lower.to(value), upper.to(value)
    if value.shape[-2:] != (5, 3) or value.shape[:-2] != a.shape or a.shape != b.shape:
        raise ValueError("duration probability tensor shape mismatch")
    if bool((~torch.isfinite(a) | ~torch.isfinite(b) | (a < 0) | (b <= a)).any() | (~torch.isfinite(value)).any()):
        raise ValueError("invalid duration probability bounds or parameters")
    center = value[..., 1] * REFERENCES[field]
    scale = F.softplus(value[..., 2]) * REFERENCES[field] + MINIMUM_SCALES[field]
    low = (a[..., None] - center) / scale
    width = (b - a)[..., None] / scale
    high = low + width
    component = F.logsigmoid(high) + F.logsigmoid(-low) + torch.log(-torch.expm1(-width))
    return torch.logsumexp(torch.log_softmax(value[..., 0], -1) + component, -1)


def validate_identity(payload: Mapping) -> None:
    records = (payload, payload.get("baseline_manifest", {}), payload.get("baseline_contract", {}))
    if any(record.get("duration_distribution") != LINEAR_CONTRACT for record in records):
        raise ValueError("duration distribution identity mismatch")


def initialize_heads(model) -> dict:
    distribution_contract(model.duration_distribution)
    with torch.no_grad():
        for field in FIELDS:
            bias = model.time_heads[field].bias.reshape(5, 3)
            # Preserve the legacy Slide initializer's float32 arithmetic.
            dtype = bias.dtype if field == SLIDE else torch.float64
            centers = torch.tensor(LINEAR_CENTERS[field], dtype=dtype, device=bias.device)
            scales = torch.tensor(LINEAR_SCALES[field], dtype=dtype, device=bias.device)
            bias[:, 1].copy_(centers / REFERENCES[field])
            bias[:, 2].copy_(torch.log(torch.expm1((scales - MINIMUM_SCALES[field]) / REFERENCES[field])))
    return dict(LINEAR_CONTRACT)


def interval_log_probability(raw, bounds, *, field, distribution=LINEAR_DURATION):
    """Bounds [...,4] contain query low/high, then support low/high."""
    if field not in FIELDS or bounds.shape != raw.shape[:-2] + (4,):
        raise ValueError("duration interval shape or field mismatch")
    if bool(((bounds[..., 0] < bounds[..., 2]) | (bounds[..., 1] > bounds[..., 3])).any()):
        raise ValueError("duration interval outside legal support")
    distribution_contract(distribution)
    expanded = raw.unsqueeze(-3).expand(*raw.shape[:-2], 2, 5, 3)
    both = linear_interval_mass(expanded, bounds[..., (0, 2)], bounds[..., (1, 3)], field=field)
    return (both[..., 0] - both[..., 1]).clamp_max(0).float()


def tick_bounds(field, anchor, query_lower, query_upper, support_lower, support_upper, *, seconds=None, tick_seconds=None):
    """Inclusive integer ranges -> half-tick edges -> ms for Hold only.

    seconds must accept a tensor of absolute ticks and preserve its shape.
    CPU uses the exact coordinate LRU; Physical supplies its resident timing map.
    """
    first = torch.div(support_lower.long() + 2, 4, rounding_mode='floor')
    last = torch.div(support_upper.long() + 2, 4, rounding_mode='floor')
    qfirst = torch.div(query_lower.long() + 2, 4, rounding_mode='floor')
    qlast = torch.div(query_upper.long() + 2, 4, rounding_mode='floor')
    lo = torch.where(qfirst.eq(first), support_lower.double() - .5, qfirst.double() * 4 - 2)
    hi = torch.where(qlast.eq(last), support_upper.double() + .5, qlast.double() * 4 + 2)
    edges = torch.stack((lo, hi, support_lower.double() - .5, support_upper.double() + .5), -1).clamp_min(0)
    if field == HOLD:
        if tick_seconds is None:
            raise ValueError("Hold duration requires written-BPM tick_seconds")
        edges = edges * tick_seconds.to(edges)[..., None] * 1000.
    elif field != SLIDE:
        raise ValueError("unknown duration field")
    return edges


def cpu_seconds(timing, ticks):
    values = seconds_at_ticks(timing, ticks.detach().cpu().reshape(-1).tolist())
    return torch.from_numpy(np.asarray(values)).reshape(ticks.shape).to(device=ticks.device, dtype=torch.float64)


def cpu_tick_seconds(timing, ticks):
    from maimai_dataset.event_time import written_tick_seconds
    values = [written_tick_seconds(timing, a) for a in ticks.detach().cpu().reshape(-1).tolist()]
    return torch.tensor(values, device=ticks.device, dtype=torch.float64).reshape(ticks.shape)


def candidate_ticks(*, field, lower, upper, device):
    if upper < lower or lower < minimum_tick(field):
        raise ValueError("empty/invalid duration candidate support")
    buckets = torch.arange((lower + 2) // 4, (upper + 2) // 4 + 1, dtype=torch.long, device=device)
    return buckets, (buckets * 4).clamp(lower, upper)


def first_representative(lower, upper):
    return min(max(4 * ((lower + 2) // 4), lower), upper)


def candidate_probabilities(raw, *, field, lower, upper, anchor_tick, timing, distribution=LINEAR_DURATION):
    buckets, ticks = candidate_ticks(field=field, lower=lower, upper=upper, device=raw.device)
    # Use an integer belonging to the bucket even when its representative clips.
    queries = (buckets * 4 - 2).clamp(lower, upper)
    bounds = tick_bounds(field, torch.full_like(ticks, anchor_tick), queries, queries,
                         torch.full_like(ticks, lower), torch.full_like(ticks, upper),
                         tick_seconds=cpu_tick_seconds(timing, torch.full_like(ticks, anchor_tick)))
    probability = interval_log_probability(raw.expand(len(ticks), -1, -1), bounds, field=field, distribution=distribution)
    return ticks, probability


def minimum_tick(field):
    if field not in FIELDS:
        raise ValueError('unknown duration field')
    return 2 if field == HOLD else 1


def first_tail_tick(threshold, lower, upper):
    """First bucket whose emitted representative is >= threshold; upper+1 if empty."""
    first = torch.div(lower + 2, 4, rounding_mode='floor')
    last = torch.div(upper + 2, 4, rounding_mode='floor')
    bucket = torch.div(threshold + 3, 4, rounding_mode='floor').maximum(first)
    # Clipped last/first representatives must be compared as emitted values.
    bucket = torch.where(threshold.le(lower), first, bucket)
    last_rep = (last * 4).minimum(upper).maximum(lower)
    bucket = torch.where((bucket > last) & (last_rep >= threshold), last, bucket)
    query = (bucket * 4 - 2).maximum(lower).minimum(upper)
    return torch.where((threshold > upper) | (bucket > last), upper + 1, query)
