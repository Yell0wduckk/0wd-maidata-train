"""Shared FP32 rotary Q/K and attention primitives for full and cached paths."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def rotate(value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Rotate [B,H,T,32] using [B,T] coordinates, without rounding coordinates."""
    if value.shape[-1] != 32 or positions.shape != (value.shape[0], value.shape[2]):
        raise ValueError("RoPE requires aligned coordinates and 32 dimensions per head")
    with torch.autocast(device_type=value.device.type, enabled=False):
        omega = 10000.0 ** (-torch.arange(0, 32, 2, device=value.device, dtype=torch.float32) / 32)
        phase = positions.float()[:, None, :, None] * omega
        cosine, sine = phase.cos(), phase.sin()
        even, odd = value.float()[..., 0::2], value.float()[..., 1::2]
        output = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1)
    return output.flatten(-2).to(value.dtype)


def heads(value, count):
    return value.reshape(value.shape[0], value.shape[1], count, -1).transpose(1, 2)


def project_kv(attention, memory):
    d, bias = attention.embed_dim, attention.in_proj_bias
    if attention._qkv_same_embed_dim:
        k, v = F.linear(memory, attention.in_proj_weight[d:], None if bias is None else bias[d:]).chunk(2, -1)
    else:
        k = F.linear(memory, attention.k_proj_weight, None if bias is None else bias[d:2*d])
        v = F.linear(memory, attention.v_proj_weight, None if bias is None else bias[2*d:])
    return heads(k, attention.num_heads), heads(v, attention.num_heads)


def attend(attention, query, kv, *, allowed=None, positions=None):
    d = attention.embed_dim
    weight = attention.in_proj_weight[:d] if attention._qkv_same_embed_dim else attention.q_proj_weight
    q = heads(F.linear(query, weight, None if attention.in_proj_bias is None else attention.in_proj_bias[:d]), attention.num_heads)
    if positions is not None:
        q = rotate(q, positions)
    value = F.scaled_dot_product_attention(q, *kv, attn_mask=allowed,
        dropout_p=attention.dropout if attention.training else 0.)
    value = value.transpose(1, 2).flatten(-2)
    return F.linear(value, attention.out_proj.weight, attention.out_proj.bias)


def mapped_attention(query, kv, positions, padding, mapping, *, dropout, training, layout=None):
    """One SDPA per unique audio, with only valid queries and trimmed K/V."""
    if layout is None:  # Compatibility API; real loaders/cache prepare once.
        from maimai_baseline.data.attention_layout import cross_layout, to_device
        layout = to_device(cross_layout(torch.zeros(query.shape[:2], dtype=torch.bool),
            padding.cpu(), mapping.cpu()), query.device)
    flat = query.flatten(0, 1)
    output = flat * 0. + (kv[0].reshape(-1)[:1].sum() + kv[1].reshape(-1)[:1].sum()) * 0.
    for audio, (selected, (lo, hi)) in enumerate(zip(layout['indices'], layout['spans'])):
        if selected.numel() == 0 or hi == lo:
            continue
        q = flat.index_select(0, selected)[None]
        pos = positions.flatten().index_select(0, selected)[None]
        rotated = rotate(heads(q, kv[0].shape[1]), pos)
        value = F.scaled_dot_product_attention(rotated,
            kv[0][audio:audio+1, :, lo:hi], kv[1][audio:audio+1, :, lo:hi],
            attn_mask=(~padding[audio:audio+1, lo:hi])[:, None, None],
            dropout_p=dropout if training else 0.).transpose(1, 2).flatten(-2)[0]
        output = output.to(value.dtype).index_copy(0, selected, value)
    return output.reshape(query.shape)


def self_attention(attention, value, positions, padding, *, past=None, past_padding=None, causal=True):
    """Append already rotated K/raw V; cached prefixes keep their computation graph."""
    k, v = project_kv(attention, value)
    k = rotate(k, positions)
    previous = 0 if past is None else past[0].shape[2]
    if past is not None:
        k, v = (torch.cat((old, new), 2) for old, new in zip(past, (k, v)))
    keys_padding = padding if past_padding is None else torch.cat((past_padding, padding), 1)
    allowed = (~keys_padding)[:, None, None, :].expand(-1, 1, value.shape[1], -1).clone()
    if causal:
        qindex = previous + torch.arange(value.shape[1], device=value.device)
        allowed &= torch.arange(k.shape[2], device=value.device)[None, :] <= qindex[:, None]
    allowed &= ~padding[:, None, :, None]
    # A dummy first key prevents all-masked kernels; ignored query outputs are zero.
    allowed[..., 0] |= ~allowed.any(-1)
    output = attend(attention, value, (k, v), allowed=allowed, positions=positions)
    return output.masked_fill(padding.unsqueeze(-1), 0.), (k, v)


def decoder_positions(config, anchors, start_tick, padding):
    if config.decoder_rope_coordinate != "window-beat":
        raise ValueError("decoder requires window-beat coordinates")
    with torch.autocast(device_type=anchors.device.type, enabled=False):
        position = (anchors.float() - start_tick.float()[:, None]) / 96.
    return position.masked_fill(padding, 0.)
