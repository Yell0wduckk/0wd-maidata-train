"""Condition attention with exact zero/one-key semantics."""
import torch
from torch.nn import functional as F


def single_value(value, valid, query_length, dropout_p):
    # Attention dropout samples one scalar per query/head, not per value feature.
    value = value.masked_fill(~valid[:, None, :, None], 0)
    weights = value.new_ones((len(value), value.shape[1], query_length, 1))
    return F.dropout(weights, p=dropout_p, training=dropout_p > 0) * value


def attend(query, key, value, valid, *, scale, dropout_p, all_multiple=False):
    """Only rows with multiple valid keys enter Q/K softmax.

    Query validity is applied by the caller after its output projection. The
    common plan-only bank has width one and needs no device-to-host decisions.
    """
    length = query.shape[2]
    if valid.shape[1] == 1:
        return single_value(value, valid, length, dropout_p)
    if all_multiple:
        # Worker buckets with key width > 1 contain only 2+ valid keys.
        return F.scaled_dot_product_attention(query, key, value,
            attn_mask=valid[:, None, None, :], scale=scale, dropout_p=dropout_p)
    counts = valid.sum(-1)
    single = (counts == 1).nonzero().flatten()
    multiple = (counts > 1).nonzero().flatten()
    if len(multiple) == len(valid):
        return F.scaled_dot_product_attention(query, key, value,
            attn_mask=valid[:, None, None, :], scale=scale, dropout_p=dropout_p)
    # Empty slice keeps the zero-key result differentiable only with respect to V.
    result = value[:, :, :0].sum(2, keepdim=True).expand(-1, -1, length, -1)
    if len(single):
        mask = valid.index_select(0, single)
        index = mask.long().argmax(-1)[:, None, None, None].expand(-1, value.shape[1], 1, value.shape[-1])
        selected = value.index_select(0, single).gather(2, index)
        result = result.index_copy(0, single, single_value(selected, mask.new_ones((len(single), 1)), length, dropout_p))
    if len(multiple):
        q, k, v = [t.index_select(0, multiple) for t in (query, key, value)]
        selected = F.scaled_dot_product_attention(q, k, v,
            attn_mask=valid.index_select(0, multiple)[:, None, None, :],
            scale=scale, dropout_p=dropout_p)
        result = result.index_copy(0, multiple, selected)
    return result
