"""Absolute-beat RoPE and attention over the two allowed joint key sets."""

import torch
from torch import Tensor
from torch.nn import functional as F

from maidata.nn.position import apply_rope


def local_joint_attention(
    q_l: Tensor, k_l: Tensor, v_l: Tensor, k_s: Tensor, v_s: Tensor,
    l_beats: Tensor, l_valid: Tensor, s_valid: Tensor,
    radius: float, dropout_p: float, layout=None,
) -> Tensor:
    """Each L query reads only its beat-local L neighbors and all S keys.

    Rectangular query blocks share neighborhood K/V across all windows.
    No full L-by-L score or mask is built.
    Projections and RoPE are supplied by the shared block.
    """
    from .layout import prepare_local, to_device, with_local_mask
    if layout is None:
        layout = to_device(prepare_local(l_beats.detach().cpu(), l_valid.cpu(), radius), q_l.device)
    if 'allowed' not in layout:
        layout = with_local_mask(layout, l_beats, radius)
    batch, heads, length, depth = q_l.shape
    def gather(tensor, indices):
        flat = tensor.transpose(1, 2).reshape(batch*length, heads, depth)
        return flat[indices].transpose(1, 2)
    if not layout['query'].shape[0]:
        return q_l * 0 + (k_l.sum() + v_l.sum() + k_s.sum() + v_s.sum()) * 0
    owners = layout['batch']
    keys = torch.cat((gather(k_l, layout['key']), k_s.index_select(0, owners)), dim=2)
    values = torch.cat((gather(v_l, layout['key']), v_s.index_select(0, owners)), dim=2)
    slot_allowed = layout['valid'][..., None] & s_valid[owners, None, :]
    allowed = torch.cat((layout['allowed'], slot_allowed), dim=-1)
    attended = F.scaled_dot_product_attention(gather(q_l, layout['query']), keys, values,
        attn_mask=allowed[:, None], dropout_p=dropout_p)
    selected = attended.transpose(1, 2).reshape(-1, heads, depth).index_select(0, layout['selected'])
    merged = q_l.new_zeros(batch*length, heads, depth).index_copy(0, layout['output'], selected)
    return merged.reshape(batch, length, heads, depth).transpose(1, 2)



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
