"""BPM96 local attention: rectangular query blocks with overlapping K/V."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.profiler import record_function

from maimai_dataset.window import AUDIO_CENTERS_PER_BEAT
from maimai_baseline.model.rope import rotate, heads as split_heads


QUERY_BLOCKS = (32, 32, 16)


def index_halo(radius: float, stage: int) -> int:
    if stage not in (0, 1, 2) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("local attention requires an E0-E2 stage and positive radius")
    stride = 2 ** stage
    # A valid pooled center moves at most (stride-1)/24 beats from its
    # nominal center. Bound both endpoints, then apply the exact beat test.
    return math.ceil(radius * AUDIO_CENTERS_PER_BEAT / stride + 2 * (1 - 1 / stride))


def local_indices(beats: torch.Tensor, padding: torch.Tensor, *, radius: float, stage: int):
    if beats.ndim != 2 or beats.shape != padding.shape or beats.shape[1] == 0:
        raise ValueError("local attention expects nonempty aligned beat/padding matrices")
    length = beats.shape[1]
    halo = index_halo(radius, stage)
    width = min(QUERY_BLOCKS[stage], length)
    key_width = min(length, width + 2 * halo)
    blocks = (length + width - 1) // width
    starts = torch.arange(blocks, device=beats.device) * width
    queries = starts[:, None] + torch.arange(width, device=beats.device)
    keys = (starts - halo).clamp(0, length - key_width)[:, None] + torch.arange(key_width, device=beats.device)
    safe_queries = queries.clamp_max(length - 1)
    query_valid = (queries < length)[None] & ~padding[:, safe_queries]
    return safe_queries, keys, query_valid


def local_layout(beats: torch.Tensor, padding: torch.Tensor, *, radius: float, stage: int):
    """Reference rectangular neighborhoods; never form pairwise NxN."""
    safe_queries, keys, query_valid = local_indices(beats, padding, radius=radius, stage=stage)
    allowed = (beats[:, safe_queries][..., None] - beats[:, keys][:, :, None, :]).abs() <= radius
    allowed &= ~padding[:, keys][:, :, None, :]
    allowed &= query_valid[..., None]
    # Padding queries get a finite dummy row. Zeroing after output projection
    # removes their output and all gradient contribution to Q/K/V and biases.
    allowed[..., 0] |= ~query_valid
    return safe_queries, keys, allowed, query_valid


def prepare_local_layout(beats, padding, *, radius, stage):
    if beats.device.type != 'cpu' or padding.device.type != 'cpu':
        raise ValueError('local layout must be prepared on CPU')
    qi, ki, valid = local_indices(beats, padding, radius=radius, stage=stage)
    audio, block = valid.any(-1).nonzero(as_tuple=True)
    length = beats.shape[1]
    query = audio[:, None] * length + qi[block]
    key = audio[:, None] * length + ki[block]
    selected = valid[audio, block].flatten().nonzero().flatten()
    return dict(query=query, key=key, valid=valid[audio, block],
                selected=selected, output=query.flatten()[selected])


def with_local_mask(layout, beats, padding, radius):
    """Use actual device pooled coordinates; build once per stage, reuse its blocks."""
    q, k, valid = layout['query'], layout['key'], layout['valid']
    flat = beats.flatten()
    allowed = (flat[q][..., None] - flat[k][:, None, :]).abs() <= radius
    allowed &= ~padding.flatten()[k][:, None, :] & valid[..., None]
    allowed[..., 0] |= ~valid
    return dict(layout, allowed=allowed)


def prepare_pyramid_layouts(beats, padding, radii=(1., 2., 4.)):
    from maimai_baseline.model.layers import _masked_pool_scalar
    layouts = []
    for stage, radius in enumerate(radii):
        layouts.append(prepare_local_layout(beats, padding, radius=radius, stage=stage))
        if stage < 2:
            beats, empty = _masked_pool_scalar(beats, padding)
            padding = empty | padding[:, ::2]
            # Match transition's scalar geometry, including masked centers.
    return tuple(layouts)


def local_attention(attention: nn.MultiheadAttention, value: torch.Tensor,
                    padding: torch.Tensor, beats: torch.Tensor, *, radius: float, stage: int,
                    positions: torch.Tensor, layout=None) -> torch.Tensor:
    if value.ndim != 3 or value.shape[:2] != beats.shape:
        raise ValueError('local attention values must align with beat coordinates')
    if layout is None:  # Standalone callers; production prepares this before H2D.
        from maimai_baseline.data.attention_layout import to_device
        layout = to_device(prepare_local_layout(beats.detach().cpu(), padding.cpu(),
                           radius=radius, stage=stage), value.device)
    if 'allowed' not in layout:
        layout = with_local_mask(layout, beats, padding, radius)
    batch, length, dim = value.shape
    heads = attention.num_heads
    depth = dim // heads
    with record_function('audio.local_attention.project_and_gather'):
        q, k, v = F.linear(value, attention.in_proj_weight, attention.in_proj_bias).chunk(3, dim=-1)
        q = rotate(split_heads(q, heads), positions).transpose(1, 2).flatten(-2)
        k = rotate(split_heads(k, heads), positions).transpose(1, 2).flatten(-2)
        def gather(tensor, index):
            return tensor.reshape(batch * length, dim)[index].reshape(
                index.shape[0], index.shape[1], heads, depth).transpose(1, 2)
    merged = q.new_zeros(batch * length, dim)
    if layout['query'].shape[0]:
        with record_function('audio.local_attention.sdpa'):
            attended = F.scaled_dot_product_attention(
                gather(q, layout['query']), gather(k, layout['key']), gather(v, layout['key']),
                attn_mask=layout['allowed'][:, None],
                dropout_p=attention.dropout if attention.training else 0.)
        selected = attended.transpose(1, 2).reshape(-1, dim).index_select(0, layout['selected'])
        merged = merged.index_copy(0, layout['output'], selected)
    else:
        merged = (q + k + v).reshape(batch * length, dim) * 0.
    output = F.linear(merged.reshape(batch, length, dim), attention.out_proj.weight, attention.out_proj.bias)
    return output.masked_fill(padding[..., None], 0.)
