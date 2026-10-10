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
    context_beats: int, layout=None,
) -> ScaleMemory:
    """Gather each window's integer-beat centers without treating index 0 as beat 0."""
    from .layout import prepare_slots, to_device
    if layout is None:
        layout = to_device(prepare_slots(g0.beats.detach().cpu(), g0.valid.cpu(),
            target_start_beats.detach().cpu(), window_to_song.cpu(), audio_before_beats, context_beats),
            g0.values.device)
    clean = mask_values(g0.values, g0.valid)
    values = clean.flatten(0, 1)[layout['indices']]
    return ScaleMemory(mask_values(values, layout['valid']), layout['beats'], layout['valid'])



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


def _resample(source: ScaleMemory, targets: Tensor, *, valid_only: bool, layout=None) -> ScaleMemory:
    """Apply prepared endpoint geometry in one batched gather; keep feature gradients."""
    from .layout import prepare_resample, to_device
    if targets.ndim != 2 or targets.shape[0] != source.values.shape[0]:
        raise ValueError("interpolation targets must align with source batch")
    if not len(targets):
        return ScaleMemory(source.values.new_zeros((*targets.shape, source.values.shape[-1])),
                           targets.float(), torch.zeros_like(targets, dtype=torch.bool))
    if layout is None:
        layout = to_device(prepare_resample(source.beats.detach().cpu(), source.valid.cpu(),
            targets.detach().cpu(), valid_only=valid_only), source.values.device)
    clean = mask_values(source.values, source.valid)
    def gather(key):
        return clean.gather(1, layout[key][..., None].expand(-1, -1, clean.shape[-1])).float()
    with torch.autocast(device_type=clean.device.type, enabled=False):
        a, b = layout['left_weight'], layout['right_weight']
        values = (gather('left')*a[..., None] + gather('right')*b[..., None]) / (a+b).clamp_min(1e-12)[..., None]
    return ScaleMemory(mask_values(values.to(clean.dtype), layout['valid']), targets.float(), layout['valid'])



def interpolate(source: ScaleMemory, target_beats: Tensor, layout=None) -> ScaleMemory:
    """C-to-M interpolation over valid source coordinates, within their coverage.

    A single source covers only its own coordinate. At duplicate coordinates,
    left-sided search follows the first matching source, matching the reference.
    """
    return _resample(source, target_beats, valid_only=True, layout=layout)


def masked_upsample(source: ScaleMemory, target_beats: Tensor, layout=None) -> ScaleMemory:
    """FPN interpolation with valid endpoint weights renormalized in FP32.

    At grid boundaries, clamp to the nearest endpoint instead of extrapolating
    values. The caller retains its finer M validity mask after fusion.
    """
    return _resample(source, target_beats, valid_only=False, layout=layout)
