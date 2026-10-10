"""Absolute-beat geometry with True meaning valid audio."""

import torch
from torch import Tensor
from torch.nn import functional as F

from .types import ScaleMemory


def mask_values(values: Tensor, valid: Tensor) -> Tensor:
    """Clear padding, including arbitrary non-finite padding payloads."""
    return values.masked_fill(~valid.unsqueeze(-1), 0)


def pool_stride(values: Tensor, beats: Tensor, valid: Tensor) -> ScaleMemory:
    """Mask-weighted k=3/s=2/p=1 pooling, keeping valid sampling centers."""
    if values.shape[1] == 0:
        return ScaleMemory(values, beats.float(), valid)
    weights = valid.float().unsqueeze(1)
    clean = mask_values(values, valid).float().transpose(1, 2)
    clean_beats = beats.float().masked_fill(~valid, 0).unsqueeze(1)

    def pooled(value):
        return F.avg_pool1d(value, 3, stride=2, padding=1, count_include_pad=False)

    mass = pooled(weights)
    pooled_values = (pooled(clean) / mass.clamp_min(1e-12)).transpose(1, 2)
    pooled_beats = (pooled(clean_beats) / mass.clamp_min(1e-12)).squeeze(1)
    pooled_valid = mass.squeeze(1).gt(0) & valid[:, ::2]
    return ScaleMemory(
        mask_values(pooled_values.to(values.dtype), pooled_valid),
        pooled_beats.masked_fill(~pooled_valid, 0),
        pooled_valid,
    )


def gather_slots(
    g0: ScaleMemory,
    target_start_beats: Tensor,
    window_to_song: Tensor,
    audio_before_beats: int,
    context_beats: int,
) -> ScaleMemory:
    """Gather each window's integer-beat centers without treating index 0 as beat 0."""
    targets = (
        target_start_beats.float().unsqueeze(1) - audio_before_beats
        + torch.arange(context_beats, device=g0.values.device, dtype=torch.float32) + 0.5
    )
    clean = mask_values(g0.values, g0.valid)
    rows, masks = [], []
    for window, song in enumerate(window_to_song.unbind()):
        song_values, song_beats, song_valid = clean[song], g0.beats[song], g0.valid[song]
        source_indices = torch.nonzero(song_valid, as_tuple=False).flatten()
        if source_indices.numel() == 0:
            rows.append(song_values.sum(0, keepdim=True).expand(context_beats, -1) * 0)
            masks.append(torch.zeros(context_beats, device=g0.values.device, dtype=torch.bool))
            continue
        source_beats = song_beats[source_indices].float().contiguous()
        positions = torch.searchsorted(source_beats, targets[window].contiguous())
        chosen = positions.clamp(max=source_beats.numel() - 1)
        valid = positions.lt(source_beats.numel()) & source_beats[chosen].eq(targets[window])
        rows.append(mask_values(song_values[source_indices[chosen]], valid))
        masks.append(valid)
    if not rows:
        return ScaleMemory(
            g0.values.new_zeros((0, context_beats, g0.values.shape[-1])),
            targets,
            torch.zeros_like(targets, dtype=torch.bool),
        )
    return ScaleMemory(torch.stack(rows), targets, torch.stack(masks))


def local_neighbors(
    source_beats: Tensor, source_valid: Tensor, query_beats: Tensor, radius: float
) -> tuple[Tensor, Tensor]:
    """Return only valid source indices inside each inclusive absolute-beat radius.

    The source's valid coordinates must be ordered. No query-by-source matrix
    is constructed; searchsorted determines the compact neighborhood width.
    """
    if radius < 0:
        raise ValueError("local attention radius must be non-negative")
    source_indices = torch.nonzero(source_valid, as_tuple=False).flatten()
    coordinates = source_beats[source_indices].float().contiguous()
    shape = (query_beats.numel(), 0)
    if coordinates.numel() == 0 or query_beats.numel() == 0:
        return (
            torch.empty(shape, device=source_beats.device, dtype=torch.long),
            torch.empty(shape, device=source_beats.device, dtype=torch.bool),
        )
    queries = query_beats.float()
    # Search bounds are conservative around FP32 cancellation; the actual
    # subtraction below decides edges, including points exactly at the radius.
    roundoff = torch.finfo(queries.dtype).eps * (queries.abs() + abs(radius))
    begin = torch.searchsorted(coordinates, (queries - radius - roundoff).contiguous())
    end = torch.searchsorted(coordinates, (queries + radius + roundoff).contiguous(), right=True)
    width = int((end - begin).max().item())
    positions = begin.unsqueeze(1) + torch.arange(width, device=source_beats.device)
    valid = positions.lt(end.unsqueeze(1))
    indices = source_indices[positions.clamp(max=source_indices.numel() - 1)]
    valid = valid & (source_beats[indices].float() - queries.unsqueeze(1)).abs().le(radius)
    return indices.masked_fill(~valid, 0), valid


def _resample(source: ScaleMemory, targets: Tensor, *, valid_only: bool) -> ScaleMemory:
    """Share coordinate search while keeping C coverage and FPN edge rules distinct."""
    if targets.ndim != 2 or targets.shape[0] != source.values.shape[0]:
        raise ValueError("interpolation targets must align with source batch")
    targets = targets.float()
    clean = mask_values(source.values, source.valid)
    rows, masks = [], []
    with torch.autocast(device_type=source.values.device.type, enabled=False):
        for batch, target in enumerate(targets.unbind()):
            indices = (
                torch.nonzero(source.valid[batch], as_tuple=False).flatten()
                if valid_only else torch.arange(source.values.shape[1], device=source.values.device)
            )
            if indices.numel() == 0:
                rows.append(clean[batch].sum(0, keepdim=True).expand(target.numel(), -1) * 0)
                masks.append(torch.zeros_like(target, dtype=torch.bool))
                continue
            coordinates = source.beats[batch, indices].float().contiguous()
            if coordinates.numel() == 1:
                left = right = torch.zeros_like(target, dtype=torch.long)
                weight = torch.zeros_like(target)
            else:
                right = torch.searchsorted(coordinates, target.contiguous()).clamp(1, coordinates.numel() - 1)
                left = right - 1
                denominator = (coordinates[right] - coordinates[left]).clamp_min(1e-8)
                weight = ((target - coordinates[left]) / denominator).clamp(0, 1)
            left, right = indices[left], indices[right]
            left_weight = (1 - weight) * source.valid[batch, left].float()
            right_weight = weight * source.valid[batch, right].float()
            mass = left_weight + right_weight
            valid = mass.gt(0)
            if valid_only:
                valid = valid & target.ge(coordinates[0]) & target.le(coordinates[-1])
            value = (
                clean[batch, left].float() * left_weight.unsqueeze(-1)
                + clean[batch, right].float() * right_weight.unsqueeze(-1)
            ) / mass.clamp_min(1e-12).unsqueeze(-1)
            rows.append(mask_values(value.to(source.values.dtype), valid))
            masks.append(valid)
    if not rows:
        return ScaleMemory(
            source.values.new_zeros((*targets.shape, source.values.shape[-1])),
            targets,
            torch.zeros_like(targets, dtype=torch.bool),
        )
    return ScaleMemory(torch.stack(rows), targets, torch.stack(masks))


def interpolate(source: ScaleMemory, target_beats: Tensor) -> ScaleMemory:
    """C-to-M interpolation over valid source coordinates, within their coverage.

    A single source covers only its own coordinate. At duplicate coordinates,
    left-sided search follows the first matching source, matching the reference.
    """
    return _resample(source, target_beats, valid_only=True)


def masked_upsample(source: ScaleMemory, target_beats: Tensor) -> ScaleMemory:
    """FPN interpolation with valid endpoint weights renormalized in FP32.

    At grid boundaries, clamp to the nearest endpoint instead of extrapolating
    values. The caller retains its finer M validity mask after fusion.
    """
    return _resample(source, target_beats, valid_only=False)
