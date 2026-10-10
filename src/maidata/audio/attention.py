"""Absolute-beat RoPE and attention over the two allowed joint key sets."""

import torch
from torch import Tensor
from torch.nn import functional as F

from .geometry import local_neighbors
from maidata.nn.position import apply_rope


def local_joint_attention(
    q_l: Tensor, k_l: Tensor, v_l: Tensor, k_s: Tensor, v_s: Tensor,
    l_beats: Tensor, l_valid: Tensor, s_valid: Tensor,
    radius: float, dropout_p: float,
) -> Tensor:
    """Each L query reads only its beat-local L neighbors and all S keys.

    Query chunks bound temporary gathers. No full L-by-L score or mask is built.
    Projections and RoPE are supplied by the shared block.
    """
    windows = []
    for batch in range(q_l.shape[0]):
        chunks = []
        for start in range(0, q_l.shape[2], 64):
            stop = min(start + 64, q_l.shape[2])
            indices, neighbors_valid = local_neighbors(
                l_beats[batch], l_valid[batch], l_beats[batch, start:stop], radius,
            )
            count = stop - start
            keys = torch.cat((
                k_l[batch, :, indices, :],
                k_s[batch, :, None, :, :].expand(-1, count, -1, -1),
            ), dim=-2)
            values = torch.cat((
                v_l[batch, :, indices, :],
                v_s[batch, :, None, :, :].expand(-1, count, -1, -1),
            ), dim=-2)
            allowed = torch.cat((neighbors_valid, s_valid[batch].expand(count, -1)), dim=-1)
            allowed = allowed & l_valid[batch, start:stop, None]
            attended = F.scaled_dot_product_attention(
                q_l[batch, :, start:stop, None, :], keys, values,
                attn_mask=allowed[None, :, None, :], dropout_p=dropout_p,
            ).squeeze(-2)
            chunks.append(attended.masked_fill(~allowed.any(-1)[None, :, None], 0))
        windows.append(torch.cat(chunks, dim=1))
    return torch.stack(windows)


def global_joint_attention(
    q_s: Tensor, k_g: Tensor, v_g: Tensor, k_l: Tensor, v_l: Tensor,
    s_valid: Tensor, g_valid: Tensor, l_valid: Tensor, dropout_p: float,
) -> Tensor:
    """S queries read G0 and L in one softmax; there are no S-to-S edges."""
    keys = torch.cat((k_g, k_l), dim=2)
    values = torch.cat((v_g, v_l), dim=2)
    key_valid = torch.cat((g_valid, l_valid), dim=1)
    allowed = s_valid[:, None, :, None] & key_valid[:, None, None, :]
    output = F.scaled_dot_product_attention(
        q_s, keys, values, attn_mask=allowed, dropout_p=dropout_p,
    )
    return output.masked_fill(~allowed.any(-1, keepdim=True), 0)
